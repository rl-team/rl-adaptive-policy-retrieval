"""
Statistical significance tests for CQL vs each baseline.

Loads per-episode results from data/eval_results_final.json and computes:
  - Paired t-test on per-episode `correct` arrays (scipy.stats.ttest_rel)
  - Bootstrap 10000-sample 95% CI on accuracy difference

Usage:
    python -m scripts.compute_significance

Output:
    data/significance.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats


INPUT_PATH = "data/eval_results_final.json"
OUTPUT_PATH = "data/significance.json"
N_BOOTSTRAP = 10000
SEED = 42


def bootstrap_ci(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = SEED,
) -> dict:
    """Bootstrap 95% CI on difference in means (arr_a - arr_b)."""
    rng = np.random.default_rng(seed)
    n = len(arr_a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff = arr_a[idx].mean() - arr_b[idx].mean()
        diffs.append(diff)
    diffs = np.array(diffs)
    alpha = 1 - ci
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": lo,
        "ci_upper": hi,
    }


def main() -> None:
    print("=" * 70)
    print("  Statistical Significance Tests (R41)")
    print("=" * 70)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    # CQL is the primary policy
    cql_key = "cql"
    if cql_key not in data:
        print(f"  Error: '{cql_key}' not found in {INPUT_PATH}")
        return

    cql_correct = np.array(
        [ep["correct"] for ep in data[cql_key]["episodes"]], dtype=float
    )
    cql_acc = float(cql_correct.mean())
    print(f"\n  CQL accuracy: {cql_acc:.1%} (n={len(cql_correct)})")

    # Compare CQL against each other policy
    baseline_keys = [k for k in data.keys() if k != cql_key]
    results = {}

    print(f"\n  {'Comparison':<35} {'t-stat':>8} {'p-value':>10} {'Diff':>8} {'95% CI':>20}")
    print("  " + "-" * 85)

    for bk in baseline_keys:
        baseline_correct = np.array(
            [ep["correct"] for ep in data[bk]["episodes"]], dtype=float
        )
        baseline_acc = float(baseline_correct.mean())

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(cql_correct, baseline_correct)

        # Bootstrap CI on accuracy difference
        boot = bootstrap_ci(cql_correct, baseline_correct)

        comparison_name = f"CQL vs {data[bk]['name']}"
        results[bk] = {
            "comparison": comparison_name,
            "cql_accuracy": cql_acc,
            "baseline_accuracy": baseline_acc,
            "accuracy_diff": cql_acc - baseline_acc,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_005": bool(p_value < 0.05),
            "bootstrap_mean_diff": boot["mean_diff"],
            "bootstrap_ci_lower": boot["ci_lower"],
            "bootstrap_ci_upper": boot["ci_upper"],
        }

        sig_marker = "*" if p_value < 0.05 else " "
        diff = cql_acc - baseline_acc
        print(f"  {comparison_name:<35} {t_stat:>8.3f} {p_value:>10.4f} "
              f"{diff:>+7.1%} [{boot['ci_lower']:>+.3f}, {boot['ci_upper']:>+.3f}] {sig_marker}")

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
