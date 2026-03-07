"""
tests for evaluation.metrics -- verifies accuracy, mean chunks, cost-adjusted
utility, precision@K, bootstrap CI, and compute_metrics (EDD Metrics Calculator)

run with:
    python -m tests.test_metrics
"""

from __future__ import annotations

import numpy as np

from evaluation.metrics import (
    accuracy,
    mean_chunks,
    cost_adjusted_utility,
    precision_at_k,
    bootstrap_ci,
    compute_metrics,
)


def _run_tests() -> None:
    """verify the evaluation metrics"""

    print("=" * 70)
    print("  evaluation.metrics -- Tests")
    print("=" * 70)

    # --- 1. Accuracy ---
    print("\n[Test 1] accuracy: num_correct / total_episodes")
    assert accuracy([True, True, False]) == 2 / 3, (
        f"Expected 2/3, got {accuracy([True, True, False])}"
    )
    assert accuracy([True]) == 1.0, f"Expected 1.0, got {accuracy([True])}"
    assert accuracy([]) == 0.0, f"Expected 0.0 for empty, got {accuracy([])}"
    print(f"  accuracy([T,T,F]) = {accuracy([True, True, False]):.4f}")
    print("  PASSED")

    # --- 2. Mean chunks ---
    print("\n[Test 2] mean_chunks: mean(num_chunks_retrieved)")
    assert mean_chunks([3, 5, 4]) == 4.0, (
        f"Expected 4.0, got {mean_chunks([3, 5, 4])}"
    )
    assert mean_chunks([]) == 0.0, f"Expected 0.0 for empty, got {mean_chunks([])}"
    print(f"  mean_chunks([3,5,4]) = {mean_chunks([3, 5, 4])}")
    print("  PASSED")

    # --- 3. Cost-adjusted utility ---
    print("\n[Test 3] cost_adjusted_utility: accuracy - alpha * avg_chunks")
    correct = [True, True, False]
    chunks = [2, 4, 6]
    u = cost_adjusted_utility(correct, chunks, alpha=0.1)
    expected_u = 2 / 3 - 0.4
    assert abs(u - expected_u) < 1e-9, f"Expected {expected_u}, got {u}"
    print(f"  cost_adjusted_utility(..., alpha=0.1) = {u:.4f}")
    print("  PASSED")

    # --- 4. Precision@K ---
    print("\n[Test 4] precision_at_k: retrieval quality (relevant chunks)")
    retrieved = [[1, 2, 3], [1, 2]]
    relevant = [[1, 2], [1, 2, 3]]
    p = precision_at_k(retrieved, relevant)
    expected_p = (2 / 3 + 1) / 2
    assert abs(p - expected_p) < 1e-9, f"Expected {expected_p}, got {p}"
    print(f"  precision_at_k(retrieved, relevant) = {p:.4f}")
    print("  PASSED")

    # --- 5. Bootstrap CI ---
    print("\n[Test 5] bootstrap_ci: 95% CI with 1000 samples")
    values = np.ones(100) * 0.5
    low, high = bootstrap_ci(values, n_bootstrap=500)
    assert low <= 0.5 <= high, f"Expected 0.5 in [{low}, {high}]"
    print(f"  bootstrap_ci(constant 0.5, n=100): [{low:.4f}, {high:.4f}]")
    print("  PASSED")

    # --- 6. compute_metrics ---
    print("\n[Test 6] compute_metrics: all metrics + CIs")
    correct = [True] * 80 + [False] * 20
    chunks = [3] * 50 + [5] * 50
    out = compute_metrics(correct, chunks, alpha=0.1, n_bootstrap=100)
    assert out["accuracy"] == 0.8, f"Expected 0.8, got {out['accuracy']}"
    assert out["mean_chunks"] == 4.0, f"Expected 4.0, got {out['mean_chunks']}"
    assert out["cost_adjusted_utility"] == 0.8 - 0.1 * 4.0, (
        f"Expected 0.4, got {out['cost_adjusted_utility']}"
    )
    assert out["n_episodes"] == 100, f"Expected 100, got {out['n_episodes']}"
    assert isinstance(out["accuracy_ci"], tuple), "accuracy_ci should be tuple"
    assert len(out["accuracy_ci"]) == 2, "accuracy_ci should be (lower, upper)"
    print(f"  accuracy = {out['accuracy']:.2f}, ci = {out['accuracy_ci']}")
    print(f"  mean_chunks = {out['mean_chunks']}, cost_adjusted_utility = {out['cost_adjusted_utility']:.2f}")
    print("  PASSED")

    # --- 7. compute_metrics with returns_per_episode ---
    print("\n[Test 7] compute_metrics: returns_per_episode -> mean_return + CI")
    correct_r = [True] * 5
    chunks_r = [2] * 5
    returns_r = [10.0, 8.0, 12.0, 9.0, 11.0]
    out_r = compute_metrics(
        correct_r, chunks_r, alpha=0.1, n_bootstrap=100,
        returns_per_episode=returns_r,
    )
    expected_mean_return = 10.0
    assert abs(out_r["mean_return"] - expected_mean_return) < 1e-9, (
        f"Expected mean_return={expected_mean_return}, got {out_r['mean_return']}"
    )
    assert isinstance(out_r["mean_return_ci"], tuple), "mean_return_ci should be tuple"
    assert len(out_r["mean_return_ci"]) == 2, "mean_return_ci should be (lower, upper)"
    assert out_r["mean_return_ci"][0] <= expected_mean_return <= out_r["mean_return_ci"][1], (
        f"mean_return should be within CI: {out_r['mean_return_ci']}"
    )
    assert out_r["returns_per_episode"] == returns_r, (
        "returns_per_episode should be echoed back in output"
    )
    print(f"  mean_return = {out_r['mean_return']:.2f}, ci = {out_r['mean_return_ci']}")
    print("  PASSED")

    # --- 8. compute_metrics without returns_per_episode (backward compat) ---
    print("\n[Test 8] compute_metrics: no returns_per_episode -> no mean_return key")
    out_no_ret = compute_metrics(
        [True, False], [3, 4], alpha=0.1, n_bootstrap=50,
    )
    assert "mean_return" not in out_no_ret, (
        "mean_return should not be present when returns_per_episode is None"
    )
    print("  No returns_per_episode -> mean_return absent: correct")
    print("  PASSED")

    # --- 9. compute_metrics: returns_per_episode length mismatch raises ---
    print("\n[Test 9] compute_metrics: returns_per_episode length mismatch raises")
    try:
        compute_metrics(
            [True, False], [3, 4], returns_per_episode=[1.0],  # wrong length
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ValueError raised on length mismatch: correct")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
