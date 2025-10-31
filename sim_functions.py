"""
Extract simulation functions for testing without launching the Gradio app
"""

import numpy as np
import pandas as pd
from collections import deque

# ----------------------------
# Simulation primitives (copied from app.py)
# ----------------------------
rng = np.random.default_rng(42)

def logistic(x): 
    return 1/(1+np.exp(-x))

def make_dataset(n=1000, task="radiology"):
    if task == "radiology":
        difficulty = rng.normal(0.0, 1.0, size=n)
    elif task == "legal":
        difficulty = rng.normal(0.3, 1.1, size=n)
    else:  # "code"
        difficulty = rng.normal(-0.2, 0.9, size=n)

    base_prob = logistic(-0.7 * difficulty)
    y_true = rng.binomial(1, base_prob)
    ai_logit = np.log(base_prob/(1-base_prob)) + rng.normal(0, 0.5, size=n)
    ai_prob = logistic(ai_logit)
    return pd.DataFrame({"difficulty": difficulty, "p_true": base_prob, "y_true": y_true, "ai_prob": ai_prob})

def ai_entropy(p):
    eps = 1e-9
    H = - (p*np.log2(p+eps) + (1-p)*np.log2(1-p+eps))
    return H / 1.0

def simulate_humans(y_true, k=5, base_acc=0.85, fatigue_after=50, fatigue_drop=0.10, per_expert_load=None):
    n = len(y_true)
    if per_expert_load is None:
        per_expert_load = np.zeros(k, dtype=int)
    preds = np.zeros((k, n), dtype=int)
    per_expert_acc = np.zeros(k)

    for ei in range(k):
        fatigue_factor = (per_expert_load[ei] // fatigue_after)
        eff_acc = max(0.5, base_acc - fatigue_factor*fatigue_drop)
        per_expert_acc[ei] = eff_acc
        p_correct = eff_acc
        correct_mask = rng.binomial(1, p_correct, size=n).astype(bool)
        preds[ei] = np.where(correct_mask, y_true, 1 - y_true)
    
    votes = preds.sum(axis=0)
    maj = (votes >= (k/2 + 0.001)).astype(int)
    vote_mean = votes / k
    disagree = 1.0 - (2.0*np.abs(vote_mean - 0.5))
    return preds, maj, disagree, per_expert_acc

def assign_to_humans(n_assign, k, per_expert_load):
    expert_ids = np.argsort(per_expert_load)
    buckets = [[] for _ in range(k)]
    for i in range(n_assign):
        e = expert_ids[i % k]
        buckets[e].append(i)
        per_expert_load[e] += 1
    return buckets, per_expert_load

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

def step(batch_size, df, start_idx, tau, w_ai, w_h, k, base_acc, fatigue_after, fatigue_drop, alpha, lam, per_expert_load):
    end_idx = min(start_idx + batch_size, len(df))
    batch = df.iloc[start_idx:end_idx]
    if len(batch) == 0:
        return 0, 0, 0, 0, per_expert_load, []

    y = batch["y_true"].values
    ai_p = batch["ai_prob"].values
    u_ai = np.array([ai_entropy(p) for p in ai_p])
    preds, maj, disagree, per_expert_acc = simulate_humans(
        y, k=k, base_acc=base_acc, fatigue_after=fatigue_after,
        fatigue_drop=fatigue_drop, per_expert_load=per_expert_load.copy()
    )
    U = np.clip(w_ai*u_ai + w_h*disagree, 0.0, 1.0)

    human_mask = U > tau
    ai_mask = ~human_mask
    coverage = human_mask.mean()

    ai_pred = (ai_p >= 0.5).astype(int)
    ai_correct = (ai_pred == y)[ai_mask]
    ai_acc = ai_correct.mean() if ai_correct.size else 0.0

    n_to_human = human_mask.sum()
    if n_to_human > 0:
        _, per_expert_load = assign_to_humans(n_to_human, k, per_expert_load)
        human_pred = maj[human_mask]
        human_correct = (human_pred == y[human_mask])
        human_acc = human_correct.mean()
    else:
        human_acc = 0.0

    total_correct = ai_correct.sum() + (human_correct.sum() if n_to_human > 0 else 0)
    acc = total_correct / len(batch)

    if per_expert_load.sum() > 0:
        imb = np.var(per_expert_load / per_expert_load.sum())
    else:
        imb = 0.0

    reward = alpha*acc + (1 - alpha)*(1 - coverage) - lam*imb
    return int(len(batch)), acc, coverage, reward, per_expert_load, per_expert_load.tolist()

def init_state(task, dataset_size, k_experts, base_acc, fatigue_after, fatigue_drop, epsilon):
    df = make_dataset(n=dataset_size, task=task)
    bandit = TauBandit(epsilon=epsilon)
    state = {
        "df": df,
        "idx": 0,
        "bandit": bandit,
        "per_expert_load": np.zeros(k_experts, dtype=int),
        "history": deque(maxlen=5000),
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
        frontier = hist_df.sort_values("coverage").copy()
        frontier["max_acc"] = frontier["accuracy"].cummax()
        frontier = frontier[frontier["accuracy"] == frontier["max_acc"]].drop(columns=["max_acc"])
        
        if len(hist_df) > 0:
            final_tau = hist_df["tau"].iloc[-1]
            frontier["tau_diff"] = (frontier["tau"] - final_tau).abs()
            best_spot_idx = frontier["tau_diff"].idxmin()
            frontier["is_best"] = False
            frontier.loc[best_spot_idx, "is_best"] = True
            frontier = frontier.drop(columns=["tau_diff"])
    else:
        frontier = pd.DataFrame(columns=["coverage", "accuracy", "is_best"])

    loads_df = pd.DataFrame({
        "expert": [f"E{i}" for i in range(len(st["per_expert_load"]))],
        "tasks_processed": st["per_expert_load"].tolist()
    })

    return hist_df, frontier, loads_df
