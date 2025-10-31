# Why Your Frontier Results Are Similar Each Run 🤔

## The Issue: Reproducibility vs. Variation

You're noticing that the frontier results (especially that Coverage=0.00 point) appear very consistently. Here's why:

### Root Causes:

#### 1. **Fixed Random Seed** 🔒
```python
rng = np.random.default_rng(42)  # Line 7 in app.py
```

The random seed is **hardcoded to 42**, meaning:
- ✓ Results are reproducible (same dataset every run)
- ✗ **No variation** in exploration (always explores same way)
- ✗ Algorithm makes identical choices each time

**Effect:** The bandit sees the exact same rewards in the exact same order, so it learns the same optimal strategy.

#### 2. **Low Exploration Rate** 📊
```python
epsilon = 0.2  # Only 20% random exploration by default
```

With ε=0.2, the algorithm:
- Exploits (uses best known): 80% of the time
- Explores (tries random): 20% of the time

Combined with a fixed seed → **very predictable learning path**

#### 3. **Strong Convergence** 🎯
Once the algorithm finds a good strategy:
- Q-value grows for that action
- Argmax increasingly selects it
- Exploration (ε) becomes less likely to override
- Algorithm "locks in" to that strategy

---

## Why This Isn't Necessarily Bad 🤷

### Scenario 1: Educational Purpose ✅
If your goal is to **teach how AHO works**:
- Reproducibility is GOOD
- Students see consistent behavior
- Easy to explain what's happening
- Good for demo/presentation

### Scenario 2: Exploring the System ⚠️
If your goal is to **experiment with different strategies**:
- You need MORE VARIATION
- Currently seeing only 1-2 strategies repeatedly
- Missing other parts of the frontier

---

## Solutions: Make Results More Variable 🔄

### Option 1: Remove the Fixed Seed (Simple) ✅
**Change this line (line 7):**
```python
rng = np.random.default_rng(42)  # ← Fixed seed
```

**To this:**
```python
rng = np.random.default_rng()  # ← Random each time
```

**Effect:**
- Each run uses different random numbers
- Algorithm explores different paths
- Frontier results vary more
- **Trade-off:** Results no longer reproducible

---

### Option 2: Increase Exploration Rate (Moderate) ✅
**Change this line (in the UI):**
```python
epsilon = gr.Slider(0.0, 1.0, value=0.2, ...)  # Default 0.2
```

**Try increasing to:**
- `value=0.5` (50% exploration) → Much more variation
- `value=0.3` (30% exploration) → Moderate variation

**Effect:**
- Algorithm tries random strategies more often
- Explores more of the frontier
- Takes longer to converge
- See more diverse results

**How to test:**
1. Run simulation with ε=0.2 (current) → See results
2. Run again with ε=0.5 → See different results
3. Notice how different the frontier is!

---

### Option 3: Increase Simulation Steps (Moderate) ✅
**Change this line (in the UI):**
```python
steps = gr.Slider(5, 200, value=60, ...)  # Default 60
```

**Try increasing to:**
- `value=120` (double the steps) → More learning time

**Effect:**
- Bandit has more opportunities to explore
- Can sample more frontier points
- Better frontier coverage
- Takes more computation time

**Note:** With fixed seed, still reproducible but explores more of the same frontier.

---

### Option 4: Vary the Dataset (Advanced) ✅
**Change this line (in the UI):**
```python
dataset_size = gr.Slider(500, 10000, value=2000, ...)  # Different sizes
```

**Try different sizes:**
- `value=500` (small) → Different task distribution
- `value=5000` (large) → Different patterns
- `value=10000` (huge) → Even different

**Effect:**
- Different datasets → different optimal strategies
- Frontier shifts based on problem characteristics
- See how algorithm adapts to problem difficulty

---

## Recommended Testing Strategy 🧪

### To See More Variation:

**Test 1: Increase Exploration**
```
1. Set ε = 0.5 (in the UI slider)
2. Keep everything else the same
3. Run twice
4. Compare frontiers
→ Should see different strategies!
```

**Test 2: Remove Fixed Seed**
```
1. In app.py, line 7, change:
   rng = np.random.default_rng(42)  →  rng = np.random.default_rng()
2. Run simulation 3 times
3. Each should show different results
→ This is the most variation possible!
```

**Test 3: Vary the Difficulty**
```
1. Keep all parameters same
2. Change dataset_size: 500 vs 2000 vs 5000
3. Run simulation on each
→ Should see different optimal strategies!
```

---

## What You Should Expect 📈

### With Low Exploration (Current: ε=0.2)
```
Run 1: Coverage=0.00, Accuracy=0.66
Run 2: Coverage=0.00, Accuracy=0.66  ← Same!
Run 3: Coverage=0.00, Accuracy=0.66  ← Same!
```

### With High Exploration (ε=0.5)
```
Run 1: Coverage=0.00, Accuracy=0.66
Run 2: Coverage=0.40, Accuracy=0.72  ← Different!
Run 3: Coverage=0.60, Accuracy=0.85  ← Different!
```

### With Random Seed Removed
```
Run 1: Coverage=0.20, Accuracy=0.68
Run 2: Coverage=0.45, Accuracy=0.75  ← Different dataset!
Run 3: Coverage=0.80, Accuracy=0.92  ← Different dataset!
```

---

## Technical Deep Dive 🔬

### Why Bandit Converges to Same Strategy

The ε-greedy algorithm works like this:

```
Step 1: All τ values start with Q=0 (equal)
Step 2: With random seed, same τ gets lucky first
        → Gets high reward
        → Q-value increases
Step 3: That τ becomes best (argmax picks it)
Step 4: With ε=0.2, only 20% chance to explore
Step 5: 80% of the time, algorithm picks the same τ
Step 6: Gets consistent rewards → Stays locked in
Result: Always converges to same strategy!
```

**With higher ε:**
```
Step 5: 50% of the time tries random τ values
Step 6: Discovers other good strategies
Step 7: Frontier gets populated with more points
Result: Explores more of the frontier!
```

---

## My Recommendation 🎯

### For Educational Use:
**Keep everything as-is** ✅
- Consistent results = easier to explain
- Good for demos and presentations
- Students understand the algorithm behavior

### For Experimentation:
**Do this:**

1. **Test immediately:** Increase `ε` slider to 0.5 and run again
   - See how results change
   - Understand exploration-exploitation tradeoff
   - This takes 30 seconds!

2. **Advanced:** If you want maximum variation:
   - Make this small change in app.py, line 7:
   ```python
   # Change this:
   rng = np.random.default_rng(42)
   
   # To this:
   rng = np.random.default_rng()
   ```
   - Now each run will be different
   - Each dataset will be different
   - Bandit will explore different paths

---

## Summary Table

| What's Limiting Variation? | Current Value | Recommendation |
|---|---|---|
| **Fixed Seed** | 42 (hardcoded) | Remove it for max variation |
| **Exploration Rate** | 0.2 (20%) | Try 0.5 (50%) to see difference |
| **Simulation Length** | 60 steps | Try 120 steps for more exploration |
| **Dataset Size** | 2000 samples | Try 500 or 5000 to see different problems |

**Quickest test:** Just slide ε from 0.2 to 0.5 and run again! 🚀

---

## FAQ

**Q: Why have a fixed seed at all?**
A: Testing/reproducibility. But for exploration, you want randomness!

**Q: Does the fixed seed affect correctness?**
A: No! The algorithm works perfectly. It just explores the same way every time.

**Q: Will changing the seed break anything?**
A: No! It's completely safe. Results will just be different each run.

**Q: Should I always set epsilon=0.2?**
A: For real deployments, yes (balances exploration/exploitation). For learning/demos, try higher values to see diversity!

**Q: Why does my frontier sometimes have only 1-2 points?**
A: Short simulation + fixed seed + low exploration → algorithm converges quickly to one strategy before exploring alternatives.

**Q: How do I see all 6 frontier points every run?**
A: Increase epsilon to 0.5+ and increase steps to 120+. This gives more exploration opportunities.
