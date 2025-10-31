---
title: AHO Demo - Adaptive Hybrid Orchestration
emoji: "🤖"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
short_description: Interactive AHO simulation for human-AI decision making
---

# 🤖 AHO Demo: Adaptive Hybrid Orchestration

**Human-AI Collaboration** - An interactive simulation that teaches you how robots and humans can work together perfectly!

[![Live Demo](https://img.shields.io/badge/🚀-Live_Demo-00ADD8)](https://radgh1-aho-demo.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717)](https://github.com/radgh1/aho_demo)

## 🎯 What This App Does

Imagine you have a super smart robot friend who helps you with homework. Sometimes the robot is really sure about the answers and can do the work fast. But other times, the robot gets confused and might make mistakes.

**This app is like a game that teaches the robot when to ask for help from grown-up experts (like teachers) instead of trying to do everything alone.** The robot learns from playing the game - it gets better at knowing when it's okay to work by itself and when it needs help.

### 🌟 The Big Idea

Instead of asking **"Can AI replace humans?"**, we ask **"How can AI and humans work together most effectively?"**

This app shows you how to create the perfect team where:
- 🤖 **AI handles** routine tasks quickly and accurately
- 👥 **Humans focus** on complex cases that need judgment and expertise
- 🧠 **The system learns** the best way to divide work between them

## 🎮 What You Can Do

### **Run Interactive Simulations**
- Choose different types of tasks (radiology, legal, code review)
- Adjust how many human experts are available
- Set how tired the experts get after working
- Control how much the robot trusts itself vs. asking for help
- Watch in real-time as the system learns and improves!

### **See Amazing Visualizations**
- **Learning Journey Graph**: Watch the robot's confidence evolve over time
- **Optimal Team Balance**: See the perfect human-robot ratios discovered
- **Workload Distribution**: Check how fairly work is shared between experts

### **Get Smart Analysis**
- **Human-Robot Insights**: Understand work distribution and collaboration patterns
- **Learning Progress**: See how the robot improves at teamwork
- **Efficiency Metrics**: Learn about the value of human-AI collaboration

## 🧠 How It Works (Simple Version)

### **The Robot's Job**
The robot tries to answer questions about:
- 🏥 **X-ray pictures** (finding sick spots)
- ⚖️ **Legal documents** (reading contracts)
- 💻 **Computer code** (checking programs)

For each question, the robot gives an answer AND says **"I'm X% sure this is right"**

### **The Human Experts**
- A team of teachers who can also answer questions
- They get tired after doing lots of work (just like real people!)
- They disagree sometimes (showing they're thinking carefully)

### **The Smart System**
- **Confidence Meter**: Routes work based on how sure the robot is
- **Learning Algorithm**: Gets better at knowing when to ask humans
- **Fair Work Sharing**: Makes sure no expert gets too tired
- **Quality Control**: Balances speed vs. accuracy

## 🔬 Technical Details (For Experts)

### **Core Algorithm: Contextual Bandit**
- Uses **ε-greedy reinforcement learning** to adapt decision thresholds
- Learns optimal τ (confidence threshold) values over time
- Balances exploration (trying new strategies) vs. exploitation (using what works)

### **Hierarchical Uncertainty Quantifier (HUQ)**
- **AI Uncertainty**: Binary entropy of model predictions
- **Human Uncertainty**: Disagreement across expert majority votes
- **Unified Score**: Weighted combination: `U = w_ai × u_ai + w_h × u_human`

### **Reward Function**
```
Reward = α × Accuracy + (1-α) × Coverage - λ × WorkloadImbalance
```
- **α**: Prioritizes accuracy vs. efficiency
- **Coverage**: Fraction of work handled (higher = more efficient)
- **Workload Balance**: Penalizes unfair distribution across experts

### **Fatigue Modeling**
- Expert accuracy degrades after N tasks: `eff_acc = max(0.5, base_acc - fatigue_factor × drop)`
- Simulates realistic human performance limitations

## 📊 What You'll Learn

### **Key Insights**
1. **Perfect Balance Exists**: There's an optimal way to divide work between AI and humans
2. **Context Matters**: Different tasks need different human-AI ratios
3. **Learning Pays Off**: Systems get dramatically better with experience
4. **Humans Add Value**: Even great AI benefits from human expertise on edge cases

### **Real-World Applications**
- 🏥 **Healthcare**: AI pre-screens X-rays, doctors review uncertain cases
- ⚖️ **Legal**: AI flags contract issues, lawyers review complex provisions
- 💻 **Software**: AI checks code quality, developers focus on architecture
- 🏦 **Finance**: AI detects fraud patterns, investigators handle suspicious cases

## 🚀 Getting Started

### **Online Demo**
Visit the [live demo](https://radgh1-aho-demo.hf.space) to try it right now!

### **Local Installation**
```bash
# Clone the repository
git clone https://github.com/radgh1/aho_demo.git
cd aho_demo

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### **Quick Start Guide**
1. **Pick a Task**: Start with "radiology" to see medical imaging simulation
2. **Set Experts**: Try 3-5 human experts for balanced workload
3. **Run Simulation**: Click "Run AHO Simulation" and watch the graphs!
4. **Experiment**: Change settings and see how they affect collaboration
5. **Interpret Results**: Use the "🔍 Interpret" buttons for insights

## 🎛️ Controls Guide

| Control | What It Does | Recommended Setting |
|---------|-------------|-------------------|
| **Task Domain** | Type of work (affects difficulty) | radiology |
| **Dataset Size** | How many tasks to simulate | 1000-2000 |
| **Batch Size** | Tasks processed at once | 25-50 |
| **Steps** | Learning rounds | 40-80 |
| **Expert Count** | Number of human helpers | 3-5 |
| **Base Accuracy** | How good experts are | 0.8-0.9 |
| **Fatigue Settings** | When experts tire | 30-60 tasks |
| **AI Weight** | Trust in robot confidence | 0.4-0.6 |
| **Human Weight** | Trust in expert agreement | 0.4-0.6 |
| **Accuracy Priority** | Speed vs. perfection | 0.5-0.7 |
| **Fairness Penalty** | Work sharing importance | 0.1-0.3 |
| **Exploration Rate** | How much to experiment | 0.1-0.3 |

## 📈 Understanding the Results

### **Trajectory Graph (First Plot)**
- **X-axis**: Coverage (what fraction goes to humans)
- **Y-axis**: Accuracy (what fraction are correct)
- **Colors**: Decision confidence levels
- **Shows**: How the robot learns optimal strategies

### **Pareto Frontier (Second Plot)**
- **Red dot**: Algorithm's final learned optimal point
- **Connected dots**: All discovered optimal balances
- **Shows**: Multiple ways humans and robots can collaborate effectively

### **Workload Table**
- Shows how many tasks each expert handled
- Helps understand workload distribution fairness

## 🧪 Testing & Validation

The app includes comprehensive tests validating:
- ✅ Mathematical accuracy of all calculations
- ✅ Proper fatigue modeling
- ✅ Correct uncertainty quantification
- ✅ Valid Pareto frontier computation
- ✅ Realistic human-AI interaction patterns

Run tests with: `python test_aho.py`

## 🤝 Contributing

Found a bug or have an idea? Open an issue or submit a pull request!

## 📄 License

This project demonstrates educational concepts for human-AI collaboration research.

## 🙏 Acknowledgments

Built with:
- **Gradio** for the interactive interface
- **NumPy & Pandas** for numerical computing
- **Reinforcement Learning** principles
- **Human-AI collaboration** research insights

---

**Ready to see AI and humans work together perfectly?** 🚀🤖👥

[Try the Live Demo Now!](https://radgh1-aho-demo.hf.space)
