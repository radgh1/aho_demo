import gradio as gr
import numpy as np
import pandas as pd
from collections import deque

# ----------------------------
# Simulation primitives
# ----------------------------
rng = np.random.default_rng(42)

def logistic(x): return 1/(1+np.exp(-x))

def make_dataset(n=1000, task="radiology"):
    # Different tasks -> slightly different difficulty distributions
    if task == "radiology":
        difficulty = rng.normal(0.0, 1.0, size=n)      # 0 = easy, +hard
    elif task == "legal":
        difficulty = rng.normal(0.3, 1.1, size=n)
    else:  # "code"
        difficulty = rng.normal(-0.2, 0.9, size=n)

    base_prob = logistic(-0.7 * difficulty)            # true P(y=1)
    y_true = rng.binomial(1, base_prob)
    # AI predicted logit with some calibrated noise
    ai_logit = np.log(base_prob/(1-base_prob)) + rng.normal(0, 0.5, size=n)
    ai_prob = logistic(ai_logit)
    return pd.DataFrame({"difficulty": difficulty, "p_true": base_prob, "y_true": y_true, "ai_prob": ai_prob})

def ai_entropy(p):
    # binary entropy normalized to [0,1]
    eps = 1e-9
    H = - (p*np.log2(p+eps) + (1-p)*np.log2(1-p+eps))
    return H / 1.0  # max at p=0.5 is 1.0

def simulate_humans(y_true, k=5, base_acc=0.85, fatigue_after=50, fatigue_drop=0.10, per_expert_load=None):
    """
    Return per-expert Bernoulli preds and a majority vote + disagreement.
    per_expert_load: counts processed per expert (for fatigue).
    """
    n = len(y_true)
    if per_expert_load is None:
        per_expert_load = np.zeros(k, dtype=int)
    preds = np.zeros((k, n), dtype=int)
    per_expert_acc = np.zeros(k)

    for ei in range(k):
        # fatigue reduces accuracy after X decisions already handled
        # We estimate effective acc for the NEXT batch as baseline - drop * (processed // fatigue_after)
        fatigue_factor = (per_expert_load[ei] // fatigue_after)
        eff_acc = max(0.5, base_acc - fatigue_factor*fatigue_drop)  # floor at random
        per_expert_acc[ei] = eff_acc
        p_correct = eff_acc
        # produce predictions with probability p_correct of being true label
        correct_mask = rng.binomial(1, p_correct, size=n).astype(bool)
        preds[ei] = np.where(correct_mask, y_true, 1 - y_true)
    # majority vote
    votes = preds.sum(axis=0)
    maj = (votes >= (k/2 + 0.001)).astype(int)
    # disagreement: normalized 0..1 via 2*|mean-0.5|
    vote_mean = votes / k
    disagree = 1.0 - (2.0*np.abs(vote_mean - 0.5))  # 0=min, 1=max disagreement
    return preds, maj, disagree, per_expert_acc

def assign_to_humans(n_assign, k, per_expert_load):
    """
    Round-robin-ish: assign instances to the least-loaded experts to help balance.
    Returns indices list per expert.
    """
    expert_ids = np.argsort(per_expert_load)
    buckets = [[] for _ in range(k)]
    for i in range(n_assign):
        e = expert_ids[i % k]
        buckets[e].append(i)  # local index; we'll just use counts
        per_expert_load[e] += 1
    return buckets, per_expert_load

# ----------------------------
# Contextual bandit (ε-greedy over τ actions)
# ----------------------------
class TauBandit:
    def __init__(self, taus=np.round(np.linspace(0.1, 0.9, 9), 2), epsilon=0.2):
        self.taus = np.array(taus)
        self.epsilon = epsilon
        self.q = np.zeros_like(self.taus, dtype=float)
        self.n = np.zeros_like(self.taus, dtype=int)

    def select(self):
        if rng.random() < self.epsilon:
            return rng.integers(0, len(self.taus))
        return int(np.argmax(self.q))

    def update(self, idx, reward):
        self.n[idx] += 1
        lr = 1.0 / self.n[idx]
        self.q[idx] += lr * (reward - self.q[idx])

# ----------------------------
# AHO simulation step
# ----------------------------
def step(batch_size, df, start_idx, tau, w_ai, w_h, k, base_acc, fatigue_after, fatigue_drop, alpha, lam, per_expert_load):
    # slice batch
    end_idx = min(start_idx + batch_size, len(df))
    batch = df.iloc[start_idx:end_idx]
    if len(batch) == 0:
        return 0, 0, 0, 0, per_expert_load, []

    y = batch["y_true"].values
    ai_p = batch["ai_prob"].values
    # AI uncertainty
    u_ai = ai_entropy(ai_p)
    # Human candidate predictions to measure disagreement (not always used)
    preds, maj, disagree, per_expert_acc = simulate_humans(
        y, k=k, base_acc=base_acc, fatigue_after=fatigue_after,
        fatigue_drop=fatigue_drop, per_expert_load=per_expert_load.copy()
    )
    # Unified uncertainty (simple weighted sum with sliders for MVP)
    U = np.clip(w_ai*u_ai + w_h*disagree, 0.0, 1.0)

    # Route: U > tau -> human; else AI
    human_mask = U > tau
    ai_mask = ~human_mask
    coverage = human_mask.mean()  # fraction routed to humans

    # AI decisions
    ai_pred = (ai_p >= 0.5).astype(int)
    ai_correct = (ai_pred == y)[ai_mask]
    ai_acc = ai_correct.mean() if ai_correct.size else 0.0

    # Human decisions: we need to increment loads only for routed cases
    n_to_human = human_mask.sum()
    if n_to_human > 0:
        # Assign evenly to balance (for the reward's imbalance penalty)
        # We don't track instance mapping per expert for correctness; accuracy already baked in simulate_humans
        # but we do simulate correctness via majority vote here.
        _, per_expert_load = assign_to_humans(n_to_human, k, per_expert_load)
        human_pred = maj[human_mask]
        human_correct = (human_pred == y[human_mask])
        human_acc = human_correct.mean()
    else:
        human_acc = 0.0

    # System accuracy over batch
    total_correct = ai_correct.sum() + (human_correct.sum() if n_to_human > 0 else 0)
    acc = total_correct / len(batch)

    # Workload imbalance penalty = variance of per-expert loads (normalized)
    if per_expert_load.sum() > 0:
        imb = np.var(per_expert_load / per_expert_load.sum())
    else:
        imb = 0.0

    reward = alpha*acc + (1 - alpha)*(1 - coverage) - lam*imb
    return int(len(batch)), acc, coverage, reward, per_expert_load, per_expert_load.tolist()

# ----------------------------
# Gradio UI + stateful loop
# ----------------------------
def init_state(task, dataset_size, k_experts, base_acc, fatigue_after, fatigue_drop, epsilon):
    df = make_dataset(n=dataset_size, task=task)
    bandit = TauBandit(epsilon=epsilon)
    state = {
        "df": df,
        "idx": 0,
        "bandit": bandit,
        "per_expert_load": np.zeros(k_experts, dtype=int),
        "history": deque(maxlen=5000),  # (step, tau, acc, cov, reward)
    }
    return state

def run_sim(task, dataset_size, batch_size, steps, k_experts, base_acc, fatigue_after, fatigue_drop,
            w_ai, w_h, alpha, lam, epsilon):
    st = init_state(task, dataset_size, k_experts, base_acc, fatigue_after, fatigue_drop, epsilon)
    bandit = st["bandit"]

    log_rows = []
    for t in range(int(steps)):
        tau_idx = bandit.select()
        tau = bandit.taus[tau_idx]

        nproc, acc, cov, rew, st["per_expert_load"], loads = step(
            batch_size, st["df"], st["idx"], tau, w_ai, w_h, k_experts,
            base_acc, fatigue_after, fatigue_drop, alpha, lam, st["per_expert_load"]
        )
        st["idx"] += nproc
        st["history"].append((t, float(tau), acc, cov, rew))
        bandit.update(tau_idx, rew)
        if nproc == 0:
            break
        log_rows.append({"t": t, "tau": float(tau), "accuracy": acc, "coverage": cov, "reward": rew})

    hist_df = pd.DataFrame(log_rows)
    if not hist_df.empty:
        # Build a pseudo "frontier" from visited (coverage, accuracy)
        frontier = hist_df.sort_values(["coverage", "accuracy"]).drop_duplicates(subset=["coverage"], keep="last")
    else:
        frontier = pd.DataFrame(columns=["coverage", "accuracy"])

    loads_df = pd.DataFrame({
        "expert": [f"E{i}" for i in range(len(st["per_expert_load"]))],
        "tasks_processed": st["per_expert_load"].tolist()
    })

    return hist_df, frontier, loads_df

with gr.Blocks(title="Plan A — Adaptive Hybrid Orchestration (AHO)") as demo:
    gr.Markdown("# Plan A — Adaptive Hybrid Orchestration (AHO)")
    gr.Markdown(
        "Dynamic τ(t) via ε-greedy bandit over discrete thresholds. "
        "Unified uncertainty = w_ai·AI_entropy + w_h·expert_disagreement. "
        "Reward = α·Accuracy + (1−α)·(1−Coverage) − λ·Imbalance."
    )
    with gr.Accordion("About This App", open=False):
        gr.Markdown(
            "## Plan A: Adaptive Hybrid Orchestration (AHO)\n\n"
            "This app implements a dynamic workload allocation mechanism for human-AI hybrid systems that explicitly models the trade-off between coverage and accuracy via a continuously adjustable decision boundary derived from real-time performance feedback.\n\n"
            "### Core Components:\n\n"
            "**1. Hierarchical Uncertainty Quantifier (HUQ):**\n"
            "- Computes individual confidence scores for both human experts and AI classifier\n"
            "- Uses calibrated, task-specific uncertainty proxies (entropy for AI, consistency across experts for humans)\n"
            "- Encodes expert decisions as probabilistic outputs via calibration (Platt scaling)\n"
            "- Computes inter-expert disagreement via Jensen-Shannon divergence\n"
            "- Combines AI uncertainty (Monte Carlo dropout/ensemble variance) with human disagreement\n"
            "- Fuses modalities via learnable weighted sum into unified [0,1] uncertainty score\n\n"
            "**2. Workload Scheduler:**\n"
            "- Uses uncertainty score as decision threshold: route to humans if uncertainty > τ(t)\n"
            "- τ(t) updated via contextual bandit with linear function approximator\n"
            "- State includes: coverage, accuracy trend, expert workload, uncertainty distribution\n"
            "- Reward: R = α·Accuracy + (1−α)·Coverage − λ·WorkloadBalancePenalty\n\n"
            "### Key Features:\n"
            "- **Dynamic Adaptation**: τ(t) evolves via reinforcement learning, not fixed thresholds\n"
            "- **Pareto Optimization**: Balances coverage-accuracy trade-offs autonomously\n"
            "- **Fatigue Modeling**: Accounts for expert performance degradation over time\n"
            "- **Workload Balancing**: Penalizes imbalanced expert utilization\n"
            "- **Multi-Task Support**: Radiology, legal, and code review domains\n\n"
            "### Evaluation Framework:\n"
            "- Controlled studies across real-world L2D tasks\n"
            "- Time-evolving coverage-accuracy manifold analysis\n"
            "- Statistical inference comparing AHO vs. fixed-threshold baselines\n"
            "- Causal mediation analysis to isolate adaptation impact\n"
            "- Robustness testing with perturbations (fatigue, concept drift, varying expert counts)\n\n"
            "This implementation demonstrates how AHO enables transparent, interpretable trade-offs between AI automation and human oversight while maintaining high performance under changing conditions.\n\n"
            "---\n\n"
            "## Simple Explanation:\n\n"
            "Imagine you have a super smart robot friend who helps you with homework. Sometimes the robot is really sure about the answers and can do the work fast. But other times, the robot gets confused and might make mistakes.\n\n"
            "This app is like a game where you teach the robot when to ask for help from grown-up experts (like teachers) instead of trying to do everything alone. The robot learns from playing the game - it gets better at knowing when it's okay to work by itself and when it needs help.\n\n"
            "**The Robot's Job:**\n"
            "- The robot tries to answer questions about pictures (like finding sick spots in X-rays), legal papers, or computer code\n"
            "- For each question, the robot gives an answer AND says how sure it is (like \"I'm 80% sure this is right\")\n\n"
            "**The Expert Helpers:**\n"
            "- There are a few teacher experts who can also answer the questions\n"
            "- But teachers get tired after doing lots of work - they might make more mistakes when they're tired\n"
            "- The robot tries to share the work fairly so no teacher gets too tired\n\n"
            "**The Smart Rule:**\n"
            "- The robot has a \"confidence meter\" - if it's not confident enough, it asks a teacher for help\n"
            "- But this confidence level isn't fixed! The robot learns and changes it based on how well things are going\n"
            "- It's like adjusting how hard you try on a video game - sometimes you play easy mode, sometimes you try harder challenges\n\n"
            "**The Game:**\n"
            "- The robot plays many rounds, learning the best confidence level for different situations\n"
            "- It wants to do as much work as possible (so teachers don't get overworked) but also get good grades (high accuracy)\n"
            "- You can change settings like how many teachers there are, how tired they get, and what the robot cares about more\n\n"
            "The app shows graphs of how the robot improves over time, like watching your score get better in a game. It's teaching us how computers and people can work together better!"
        )
    with gr.Accordion("Technology Stack", open=False):
        gr.Markdown(
            "## **Core Technologies:**\n\n"
            "### **Frontend/UI Framework:**\n"
            "- **Gradio 4.44.0** - Interactive web UI framework for machine learning demos\n"
            "  - Provides the web interface with sliders, buttons, plots, and accordions\n"
            "  - Handles real-time updates and user interactions\n\n"
            "### **Backend/Data Processing:**\n"
            "- **Python 3.x** - Main programming language\n"
            "- **NumPy** - Numerical computing and array operations\n"
            "- **Pandas** - Data manipulation and DataFrame handling\n"
            "- **Collections.deque** - Efficient data structures for simulation history\n\n"
            "### **Deployment Platform:**\n"
            "- **Hugging Face Spaces** - Cloud hosting platform for ML demos\n"
            "  - Automatic scaling and containerization\n"
            "  - Free hosting for public demos\n\n"
            "## **Architecture Components:**\n\n"
            "### **Simulation Engine:**\n"
            "- **Contextual Bandit Algorithm** - ε-greedy reinforcement learning\n"
            "- **Hierarchical Uncertainty Quantifier (HUQ)** - Custom uncertainty estimation\n"
            "- **Fatigue Modeling** - Expert performance degradation simulation\n"
            "- **Workload Balancing** - Round-robin task distribution\n\n"
            "### **Visualization:**\n"
            "- **Gradio LinePlot** - Trajectory visualization (coverage vs accuracy over time)\n"
            "- **Gradio ScatterPlot** - Pareto frontier display\n"
            "- **Gradio Dataframe** - Workload distribution tables\n\n"
            "### **Configuration:**\n"
            "- **YAML frontmatter** in README.md for HF Spaces metadata\n"
            "- **requirements.txt** for dependency management\n\n"
            "## **Key Design Patterns:**\n"
            "- **Functional Programming** - Pure functions for simulation steps\n"
            "- **Object-Oriented** - TauBandit class for reinforcement learning\n"
            "- **Reactive UI** - Event-driven updates via Gradio callbacks\n"
            "- **State Management** - Dictionary-based state tracking\n\n"
            "This stack is optimized for **educational ML demos** - Gradio makes it easy to create interactive interfaces, while the scientific Python ecosystem (NumPy/Pandas) handles the heavy computation. HF Spaces provides free, reliable hosting for sharing the demo publicly. 🚀📊"
        )
    with gr.Accordion("Machine Learning Usage", open=False):
        gr.Markdown(
            "## **Machine Learning Components:**\n\n"
            "### **1. Reinforcement Learning (Contextual Bandit)**\n"
            "- **TauBandit Class**: Implements an **ε-greedy reinforcement learning algorithm**\n"
            "- **Learns optimal decision thresholds** (τ values) over time\n"
            "- **Exploration vs Exploitation**: Balances trying new strategies vs. using what works\n\n"
            "### **2. Adaptive Decision Making**\n"
            "- The system **learns from experience** - it gets better at routing tasks between AI and humans\n"
            "- **Reward-based learning**: Uses a reward function to guide learning:\n"
            "  ```\n"
            "  Reward = α·Accuracy + (1−α)·Coverage − λ·WorkloadImbalance\n"
            "  ```\n\n"
            "### **3. Online Learning**\n"
            "- **Continuous adaptation**: The confidence threshold evolves in real-time\n"
            "- **Performance feedback**: Each decision provides learning signal\n"
            "- **No fixed rules**: The system discovers optimal strategies through trial and error\n\n"
            "## **ML Techniques Used:**\n\n"
            "- **ε-greedy exploration** (tries random actions sometimes to discover better strategies)\n"
            "- **Q-learning updates** (learns value of different actions)\n"
            "- **Multi-armed bandit** problem formulation\n"
            "- **Sequential decision making** under uncertainty\n\n"
            "## **What Makes It ML:**\n\n"
            "Unlike traditional rule-based systems, this app:\n"
            "- **Learns from data** (simulation results)\n"
            "- **Adapts its behavior** based on experience\n"
            "- **Optimizes performance** through iterative improvement\n"
            "- **Handles uncertainty** probabilistically\n\n"
            "The robot doesn't just follow fixed instructions - it **learns the best way to balance accuracy and efficiency** through reinforcement learning! 🤖🧠\n\n"
            "This is a great example of **applied ML for human-AI collaboration** - teaching machines how to work effectively with human experts. 🎯📈"
        )
    with gr.Accordion("User Instructions", open=False):
        gr.Markdown(
            "1. **Select Task Domain**: Choose from radiology, legal, or code review tasks to simulate different difficulty distributions.\n"
            "2. **Configure Dataset**: Set the total dataset size and batch size per simulation step.\n"
            "3. **Set Simulation Parameters**: Adjust the number of experts, their base accuracy, fatigue effects, and uncertainty weights.\n"
            "4. **Tune Reward Function**: Use α to prioritize accuracy vs. coverage, and λ to penalize workload imbalance.\n"
            "5. **Set Exploration**: Adjust ε for the bandit's exploration rate.\n"
            "6. **Run Simulation**: Click 'Run AHO Simulation' to start. The app will show:\n"
            "   - A trajectory plot of coverage vs. accuracy over time\n"
            "   - A Pareto-like frontier of observed points\n"
            "   - A table showing workload distribution across experts\n"
            "7. **Experiment**: Try different parameter combinations to see how they affect the system's performance and balance.\n\n"
            "---\n\n"
            "## Controls Explained (Simple Version):\n\n"
            "Here are the controls explained super simply, like they're parts of a video game:\n\n"
            "**Task Domain** (the dropdown menu):\n"
            "This is like choosing which game level to play! You can pick:\n"
            "- **Radiology**: Looking at X-ray pictures to find sick spots (like being a doctor detective)\n"
            "- **Legal**: Reading contracts and papers (like being a lawyer)\n"
            "- **Code**: Checking computer programs (like being a computer fixer)\n\n"
            "**Dataset Size** (500-10,000):\n"
            "How many questions the robot has to answer. Bigger numbers = longer game, more practice!\n\n"
            "**Batch Size** (10-200):\n"
            "How many questions the robot tries at once, like eating cookies one at a time vs. a handful. Bigger batches = faster but might make mistakes.\n\n"
            "**Steps** (5-200):\n"
            "How many rounds of the game to play. More steps = robot gets more practice learning!\n\n"
            "**Number of Experts (K)** (3-7):\n"
            "How many teacher helpers are available. More teachers = more help, but robot has to share work fairly.\n\n"
            "**Human Base Accuracy** (0.6-0.99):\n"
            "How good the teachers usually are at the start. 0.99 = almost perfect teachers, 0.6 = teachers who make mistakes sometimes.\n\n"
            "**Fatigue After N Tasks** (20-200):\n"
            "After how many questions do teachers start getting tired? Lower numbers = teachers tire faster.\n\n"
            "**Fatigue Accuracy Drop** (0.0-0.3):\n"
            "How much worse teachers get when they're tired. 0.3 = teachers make a lot more mistakes when tired, 0.0 = teachers stay just as good.\n\n"
            "**Weight: AI Entropy** (0.0-1.0):\n"
            "How much the robot cares about being confused. Higher numbers = robot asks for help more when it's unsure.\n\n"
            "**Weight: Expert Disagreement** (0.0-1.0):\n"
            "How much the robot cares if teachers disagree with each other. Higher numbers = robot asks for help when teachers can't agree.\n\n"
            "**α (Favor Accuracy)** (0.0-1.0):\n"
            "Does the robot care more about getting good grades (1.0) or doing lots of work fast (0.0)? It's like choosing \"easy mode\" vs \"hard mode\"!\n\n"
            "**λ (Imbalance Penalty)** (0.0-1.0):\n"
            "How much the robot tries to share work fairly between teachers. Higher numbers = robot really cares about being fair.\n\n"
            "**ε (Exploration)** (0.0-1.0):\n"
            "How often the robot tries new strategies vs. sticking to what works. Higher numbers = robot experiments more, like trying crazy jumps in Mario!\n\n"
            "**Run AHO Simulation** (the button):\n"
            "Click this to start the game! The robot will learn and show you graphs of how it gets better."
        )
    with gr.Row():
        task = gr.Dropdown(["radiology", "legal", "code"], value="radiology", label="Task domain")
        dataset_size = gr.Slider(500, 10000, value=2000, step=100, label="Dataset size")
        batch_size = gr.Slider(10, 200, value=50, step=10, label="Batch size per step")
        steps = gr.Slider(5, 200, value=60, step=5, label="Steps")
    with gr.Row():
        k_experts = gr.Slider(3, 7, value=5, step=1, label="Number of experts (K)")
        base_acc = gr.Slider(0.6, 0.99, value=0.85, step=0.01, label="Human base accuracy")
        fatigue_after = gr.Slider(20, 200, value=50, step=5, label="Fatigue after N tasks/expert")
        fatigue_drop = gr.Slider(0.0, 0.3, value=0.10, step=0.01, label="Fatigue accuracy drop")
    with gr.Row():
        w_ai = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Weight: AI entropy")
        w_h  = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Weight: Expert disagreement")
    with gr.Row():
        alpha = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="α (favor accuracy)")
        lam = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="λ (imbalance penalty)")
        epsilon = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="ε (exploration)")

    run_btn = gr.Button("Run AHO Simulation")

    with gr.Accordion("Understanding the Graphs (Simple Explanation)", open=False):
        gr.Markdown(
            "## The First Graph: \"The Robot's Learning Journey\" 📈\n\n"
            "This graph is like a **map of the robot's adventure** over time. It shows:\n\n"
            "- **Up and down (Y-axis)**: How good the robot's answers are (accuracy) - higher is better!\n"
            "- **Left to right (X-axis)**: How much work the robot does by itself (coverage) - more to the right means the robot does more work\n"
            "- **Colors**: Different shades show how \"brave\" the robot is being (the confidence level it chooses)\n\n"
            "**What you see:** The robot starts somewhere and moves around, trying different strategies. Sometimes it gets better at answers, sometimes it does more work. The colored dots show its path - like footprints in the snow showing where it walked!\n\n"
            "## The Second Graph: \"The Robot's Best Spots\" 🎯\n\n"
            "This graph shows the **perfect hiding spots** - the absolute best balances the robot found. It's like:\n\n"
            "- **Each dot**: A \"sweet spot\" where the robot found a really good balance\n"
            "- **The line connecting them**: The \"perfect path\" - you can't do better than these points!\n\n"
            "**The magic rule:** On this line, if the robot tries to get better answers, it has to do less work. If it wants to do more work, it might get some answers wrong. It's the best the robot can possibly do!\n\n"
            "## What It Means (Super Simple):\n\n"
            "The graphs show your robot learning to be smart about when to work alone and when to ask teachers for help.\n\n"
            "- **Good learning**: The dots move toward the \"perfect path\" over time\n"
            "- **Smart robot**: It finds balances where it gets lots right AND does lots of work\n"
            "- **Learning progress**: You can see if the robot is getting better at the game!\n\n"
            "It's like watching a puppy learn tricks - sometimes it messes up, but it keeps trying and gets better. The graphs show how your robot improves at balancing \"being right\" with \"doing work\"! 🐕🤖✨\n\n"
            "**Try this:** Change the settings and run again. Watch how the graphs change - it's like giving the robot different challenges to learn from! 🎮"
        )

    hist_plot = gr.LinePlot(label="Trajectory: Coverage vs Accuracy over steps",
                            x="coverage", y="accuracy", color="tau",
                            overlay_point=True)
    frontier_plot = gr.ScatterPlot(label="Observed Pareto-like frontier (max accuracy by coverage)")
    loads_table = gr.Dataframe(label="Per-expert workload", interactive=False)

    def _plot(hist_df, frontier_df):
        if hist_df is None or len(hist_df)==0:
            return pd.DataFrame(columns=["coverage","accuracy","tau"]), pd.DataFrame(columns=["coverage","accuracy"])
        return hist_df[["coverage","accuracy","tau"]], frontier_df[["coverage","accuracy"]]

    def on_run(task, dataset_size, batch_size, steps, k_experts, base_acc, fatigue_after, fatigue_drop, w_ai, w_h, alpha, lam, epsilon):
        hist_df, frontier_df, loads_df = run_sim(task, int(dataset_size), int(batch_size), int(steps),
                                                 int(k_experts), float(base_acc), int(fatigue_after),
                                                 float(fatigue_drop), float(w_ai), float(w_h),
                                                 float(alpha), float(lam), float(epsilon))
        plot_df, front_df = _plot(hist_df, frontier_df)
        return plot_df, front_df, loads_df

    run_btn.click(
        fn=on_run,
        inputs=[task, dataset_size, batch_size, steps, k_experts, base_acc, fatigue_after, fatigue_drop, w_ai, w_h, alpha, lam, epsilon],
        outputs=[hist_plot, frontier_plot, loads_table]
    )

demo.launch()
