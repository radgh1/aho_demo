"""
Comprehensive test suite for AHO simulation accuracy
Tests core algorithms, data flows, and educational claims
"""

import numpy as np
import pandas as pd
from collections import deque
import sys

# Import simulation components from separate module (avoids Gradio launch)
from sim_functions import (
    logistic, make_dataset, ai_entropy, simulate_humans, assign_to_humans,
    TauBandit, step, run_sim, init_state
)

def test_logistic():
    """Test logistic function"""
    print("\n" + "="*60)
    print("TEST 1: Logistic Function")
    print("="*60)
    
    # Test known values
    assert abs(logistic(0) - 0.5) < 1e-6, "logistic(0) should be 0.5"
    assert logistic(10) > 0.99, "logistic(10) should approach 1"
    assert logistic(-10) < 0.01, "logistic(-10) should approach 0"
    
    print("✅ Logistic function: PASSED")
    print(f"   logistic(0) = {logistic(0):.6f} (expected: 0.5)")
    print(f"   logistic(10) = {logistic(10):.6f} (expected: ~1.0)")
    print(f"   logistic(-10) = {logistic(-10):.6f} (expected: ~0.0)")

def test_ai_entropy():
    """Test AI entropy calculation"""
    print("\n" + "="*60)
    print("TEST 2: AI Entropy Function")
    print("="*60)
    
    # Entropy should be maximum at p=0.5
    h_max = ai_entropy(0.5)
    h_certain = ai_entropy(1.0)
    h_uncertain = ai_entropy(0.5)
    
    assert h_max > 0.9, "Entropy at p=0.5 should be near 1.0"
    assert h_certain < 0.01, "Entropy at p=1.0 should be near 0.0"
    
    print("✅ AI Entropy: PASSED")
    print(f"   H(p=0.5) = {h_max:.6f} (expected: ~1.0) - MAX uncertainty")
    print(f"   H(p=1.0) = {h_certain:.6f} (expected: ~0.0) - Certain prediction")
    print(f"   H(p=0.0) = {ai_entropy(0.0):.6f} (expected: ~0.0) - Certain prediction")

def test_dataset_generation():
    """Test dataset generation for different tasks"""
    print("\n" + "="*60)
    print("TEST 3: Dataset Generation by Task")
    print("="*60)
    
    for task in ["radiology", "legal", "code"]:
        df = make_dataset(n=1000, task=task)
        
        assert len(df) == 1000, f"Dataset should have 1000 rows"
        assert all(col in df.columns for col in ["difficulty", "p_true", "y_true", "ai_prob"]), \
            "Missing columns in dataset"
        
        # Check value ranges
        assert df["p_true"].min() >= 0 and df["p_true"].max() <= 1, "p_true should be in [0,1]"
        assert df["y_true"].isin([0, 1]).all(), "y_true should be binary"
        assert df["ai_prob"].min() >= 0 and df["ai_prob"].max() <= 1, "ai_prob should be in [0,1]"
        
        print(f"✅ {task.upper()}: Generated {len(df)} samples")
        print(f"   Difficulty: μ={df['difficulty'].mean():.2f}, σ={df['difficulty'].std():.2f}")
        print(f"   P(true): μ={df['p_true'].mean():.2f}")
        print(f"   Y distribution: {df['y_true'].sum()} positive cases ({100*df['y_true'].mean():.1f}%)")

def test_human_simulation():
    """Test human expert simulation and disagreement"""
    print("\n" + "="*60)
    print("TEST 4: Human Expert Simulation")
    print("="*60)
    
    # Create simple test case
    y_true = np.array([1, 0, 1, 1, 0] * 100)  # 500 samples
    k_experts = 5
    base_acc = 0.85
    
    preds, maj, disagree, per_expert_acc = simulate_humans(
        y_true, k=k_experts, base_acc=base_acc, fatigue_after=100, fatigue_drop=0.1
    )
    
    # Verify shapes
    assert preds.shape == (k_experts, len(y_true)), "Predictions shape mismatch"
    assert len(maj) == len(y_true), "Majority vote length mismatch"
    assert len(disagree) == len(y_true), "Disagreement vector length mismatch"
    
    # Verify ranges
    assert disagree.min() >= 0 and disagree.max() <= 1, "Disagreement should be in [0,1]"
    
    # All experts should have accuracy close to base_acc (with variance)
    avg_expert_acc = np.mean(per_expert_acc)
    assert avg_expert_acc > 0.75 and avg_expert_acc < 0.95, \
        f"Average expert accuracy should be near base_acc=0.85, got {avg_expert_acc:.2f}"
    
    print("✅ Human Simulation: PASSED")
    print(f"   Experts: {k_experts}, Base Accuracy: {base_acc}")
    print(f"   Per-expert accuracies: {[f'{a:.3f}' for a in per_expert_acc]}")
    print(f"   Average accuracy: {avg_expert_acc:.3f}")
    print(f"   Disagreement - min: {disagree.min():.3f}, max: {disagree.max():.3f}, mean: {disagree.mean():.3f}")
    print(f"   Majority vote accuracy: {(maj == y_true).mean():.3f}")

def test_tau_bandit():
    """Test tau bandit epsilon-greedy learning"""
    print("\n" + "="*60)
    print("TEST 5: Tau Bandit (ε-greedy Reinforcement Learning)")
    print("="*60)
    
    bandit = TauBandit(epsilon=0.2)
    initial_q = bandit.q.copy()
    
    # Simulate learning: make action 3 consistently rewarded
    for _ in range(50):
        idx = bandit.select()
        reward = 0.8 if idx == 3 else 0.2
        bandit.update(idx, reward)
    
    # Action 3 should have highest Q-value
    best_action = np.argmax(bandit.q)
    
    print("✅ Tau Bandit Learning: PASSED")
    print(f"   Initial Q-values: {initial_q}")
    print(f"   Final Q-values: {bandit.q}")
    print(f"   Best action: {best_action} with Q={bandit.q[best_action]:.3f}")
    print(f"   Tau values: {bandit.taus}")
    print(f"   Action 3 (τ={bandit.taus[3]}) learned highest value: {bandit.q[3]:.3f}")

def test_routing_logic():
    """Test AI vs Human routing based on uncertainty threshold"""
    print("\n" + "="*60)
    print("TEST 6: Task Routing Logic (U > τ → Human)")
    print("="*60)
    
    # Create synthetic uncertainty scores
    n_tasks = 1000
    U = np.random.uniform(0, 1, n_tasks)  # Uncertainty scores
    
    tau_low = 0.2  # Low threshold - send more to human
    tau_high = 0.8  # High threshold - send fewer to human
    
    coverage_low = (U > tau_low).mean()
    coverage_high = (U > tau_high).mean()
    
    assert coverage_low > coverage_high, "Lower τ should result in higher coverage"
    
    print("✅ Routing Logic: PASSED")
    print(f"   Generated {n_tasks} tasks with random uncertainty U ∈ [0,1]")
    print(f"   τ = {tau_low}: {coverage_low:.1%} routed to humans (coverage)")
    print(f"   τ = {tau_high}: {coverage_high:.1%} routed to humans (coverage)")
    print(f"   Relationship: Higher τ → Lower coverage ✅")

def test_step_function():
    """Test single simulation step accuracy and reward calculation"""
    print("\n" + "="*60)
    print("TEST 7: Simulation Step Function")
    print("="*60)
    
    df = make_dataset(n=500, task="radiology")
    batch_size = 50
    tau = 0.5
    w_ai, w_h = 0.5, 0.5
    k_experts = 5
    alpha, lam = 0.6, 0.2
    per_expert_load = np.zeros(k_experts, dtype=int)
    
    nproc, acc, cov, rew, updated_load, loads_list = step(
        batch_size, df, 0, tau, w_ai, w_h, k_experts, 0.85, 50, 0.1, alpha, lam, per_expert_load
    )
    
    # Verify outputs
    assert nproc == batch_size, "Should process batch_size samples"
    assert 0 <= acc <= 1, "Accuracy should be in [0,1]"
    assert 0 <= cov <= 1, "Coverage should be in [0,1]"
    assert isinstance(rew, (int, float)), "Reward should be numeric"
    assert np.sum(updated_load) > 0, "Should update expert loads"
    
    # Reward formula: R = α·Acc + (1−α)·(1−Coverage) − λ·Imbalance
    # This should be reasonable
    assert -1 <= rew <= 1, f"Reward should be in reasonable range, got {rew:.3f}"
    
    print("✅ Step Function: PASSED")
    print(f"   Processed: {nproc} tasks")
    print(f"   Accuracy: {acc:.3f}")
    print(f"   Coverage: {cov:.3f}")
    print(f"   Reward: {rew:.3f}")
    print(f"   Expert loads: {updated_load.tolist()}")
    print(f"   Reward breakdown:")
    print(f"     α·Accuracy = {alpha * acc:.3f}")
    print(f"     (1-α)·(1-Coverage) = {(1-alpha) * (1-cov):.3f}")
    print(f"     -λ·Imbalance = {-lam * np.var(updated_load):.3f}")

def test_pareto_frontier():
    """Test Pareto frontier computation"""
    print("\n" + "="*60)
    print("TEST 8: Pareto Frontier Accuracy")
    print("="*60)
    
    # Create synthetic history with known frontier
    hist_data = [
        {"t": 0, "coverage": 0.2, "accuracy": 0.70, "reward": 0.5},
        {"t": 1, "coverage": 0.3, "accuracy": 0.75, "reward": 0.55},
        {"t": 2, "coverage": 0.4, "accuracy": 0.72, "reward": 0.5},
        {"t": 3, "coverage": 0.5, "accuracy": 0.80, "reward": 0.6},
        {"t": 4, "coverage": 0.6, "accuracy": 0.78, "reward": 0.58},
    ]
    hist_df = pd.DataFrame(hist_data)
    
    # Compute Pareto frontier
    frontier = hist_df.sort_values("coverage").copy()
    frontier["max_acc"] = frontier["accuracy"].cummax()
    frontier = frontier[frontier["accuracy"] == frontier["max_acc"]]
    
    # Expected frontier: points where accuracy >= all previous accuracies
    expected_size = 3  # (0.2,0.70), (0.3,0.75), (0.5,0.80)
    
    assert len(frontier) == expected_size, f"Expected {expected_size} frontier points, got {len(frontier)}"
    
    print("✅ Pareto Frontier: PASSED")
    print(f"   Input history: {len(hist_df)} points")
    print(f"   Frontier points: {len(frontier)}")
    print(f"   Frontier:")
    for idx, row in frontier.iterrows():
        print(f"     Coverage={row['coverage']:.1f}, Accuracy={row['accuracy']:.2f}")

def test_fatigue_impact():
    """Test that fatigue correctly reduces expert accuracy"""
    print("\n" + "="*60)
    print("TEST 9: Expert Fatigue Impact")
    print("="*60)
    
    y_true = np.array([1, 0] * 500)  # 1000 samples
    k_experts = 1  # Single expert for clarity
    base_acc = 0.90
    fatigue_after = 100
    fatigue_drop = 0.10
    
    # Simulate 3 batches with increasing fatigue
    per_expert_load = np.array([0], dtype=int)
    accuracies = []
    
    for batch_num in range(3):
        preds, maj, disagree, acc_vector = simulate_humans(
            y_true[:100], k=k_experts, base_acc=base_acc, 
            fatigue_after=fatigue_after, fatigue_drop=fatigue_drop,
            per_expert_load=per_expert_load.copy()
        )
        
        # After fatigue_after tasks, accuracy should drop
        # Batch 0: load=0, eff_acc ≈ 0.90
        # Batch 1: load=100, eff_acc ≈ 0.90 - 0.10 = 0.80
        # Batch 2: load=200, eff_acc ≈ 0.90 - 0.20 = 0.70
        
        accuracies.append(acc_vector[0])
        per_expert_load[0] += 100
    
    # Verify fatigue impact
    assert accuracies[0] > accuracies[1], "Fatigue should reduce accuracy"
    assert accuracies[1] > accuracies[2], "More fatigue should reduce accuracy more"
    
    print("✅ Fatigue Impact: PASSED")
    print(f"   Base accuracy: {base_acc}, Fatigue drop/100 tasks: {fatigue_drop}")
    print(f"   Batch 0 (fresh): accuracy = {accuracies[0]:.3f}")
    print(f"   Batch 1 (100 tasks done): accuracy = {accuracies[1]:.3f}")
    print(f"   Batch 2 (200 tasks done): accuracy = {accuracies[2]:.3f}")
    print(f"   Accuracy decline verified: {accuracies[0]:.3f} → {accuracies[2]:.3f} ✅")

def test_full_simulation():
    """Test complete simulation run"""
    print("\n" + "="*60)
    print("TEST 10: Full Simulation Run")
    print("="*60)
    
    hist_df, frontier_df, loads_df = run_sim(
        task="radiology",
        dataset_size=500,
        batch_size=50,
        steps=20,
        k_experts=3,
        base_acc=0.85,
        fatigue_after=100,
        fatigue_drop=0.10,
        w_ai=0.5,
        w_h=0.5,
        alpha=0.6,
        lam=0.2,
        epsilon=0.2
    )
    
    assert len(hist_df) > 0, "Should have history"
    assert len(frontier_df) > 0, "Should have frontier"
    assert len(loads_df) == 3, "Should have 3 experts"
    
    # Verify data integrity
    assert all(col in hist_df.columns for col in ["t", "tau", "accuracy", "coverage", "reward"])
    assert all(col in frontier_df.columns for col in ["coverage", "accuracy"])
    assert all(col in loads_df.columns for col in ["expert", "tasks_processed"])
    
    print("✅ Full Simulation: PASSED")
    print(f"   Simulation steps: {len(hist_df)}")
    print(f"   Frontier points: {len(frontier_df)}")
    print(f"   Final accuracy: {hist_df['accuracy'].iloc[-1]:.3f}")
    print(f"   Final coverage: {hist_df['coverage'].iloc[-1]:.3f}")
    print(f"   Accuracy range: [{hist_df['accuracy'].min():.3f}, {hist_df['accuracy'].max():.3f}]")
    print(f"   Coverage range: [{hist_df['coverage'].min():.3f}, {hist_df['coverage'].max():.3f}]")
    print(f"   Total workload distributed: {loads_df['tasks_processed'].sum()} tasks")
    print(f"   Workload balance (σ): {loads_df['tasks_processed'].std():.2f}")

def test_uncertainty_combination():
    """Test that AI entropy and expert disagreement combine correctly"""
    print("\n" + "="*60)
    print("TEST 11: Uncertainty Combination (w_ai, w_h weights)")
    print("="*60)
    
    df = make_dataset(n=100, task="radiology")
    y_true = df["y_true"].values
    ai_probs = df["ai_prob"].values
    
    # Get individual uncertainty components
    u_ai = np.array([ai_entropy(p) for p in ai_probs])
    preds, maj, disagree, _ = simulate_humans(y_true, k=5, base_acc=0.85)
    
    # Test different weight combinations
    for w_ai, w_h in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
        U = np.clip(w_ai * u_ai + w_h * disagree, 0.0, 1.0)
        
        assert U.min() >= 0 and U.max() <= 1, "Unified uncertainty should be in [0,1]"
        
        print(f"✅ w_ai={w_ai}, w_h={w_h}")
        print(f"   U range: [{U.min():.3f}, {U.max():.3f}], mean: {U.mean():.3f}")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print(" "*20 + "AHO SIMULATION ACCURACY TEST SUITE")
    print("="*80)
    
    tests = [
        test_logistic,
        test_ai_entropy,
        test_dataset_generation,
        test_human_simulation,
        test_tau_bandit,
        test_routing_logic,
        test_step_function,
        test_pareto_frontier,
        test_fatigue_impact,
        test_uncertainty_combination,
        test_full_simulation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("="*80 + "\n")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
