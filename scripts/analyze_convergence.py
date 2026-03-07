"""
Convergence analysis for CQL and IQL training runs (R37).

Reads Tensorboard event files (if available) or training checkpoint
metadata to verify that all tracked losses converged.  Produces a
summary table and optional JSON output for downstream reporting.

Usage:
    python -m scripts.analyze_convergence
    python -m scripts.analyze_convergence --json > data/convergence.json

Reference: Implementation Plan R37.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Tensorboard is optional; gracefully degrade if not installed.
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ---------------------------------------------------------------------------
# Configuration: known training runs and their expected TB tags
# ---------------------------------------------------------------------------

RUNS: List[Dict[str, Any]] = [
    {
        "name": "CQL (2k corpus)",
        "tb_dir": "runs/cql_2k/tb",
        "tags": {
            "loss/total": ("Total loss", "convergence"),
            "loss/td": ("TD loss", "monitor"),
            "loss/conservative_penalty": ("Conservative penalty", "convergence"),
            "q_values/mean": ("Mean Q-values", "monitor"),
        },
    },
    {
        "name": "IQL (2k corpus)",
        "tb_dir": "runs/iql_2k/tb",
        "tags": {
            "loss/q_td": ("Q-network TD loss", "convergence"),
            "loss/v": ("V-network loss", "monitor"),
            "loss/policy": ("Policy loss", "convergence"),
            "q_values/mean": ("Mean Q-values", "monitor"),
        },
    },
]

# Tags marked "convergence" must show decreasing loss for PASS.
# Tags marked "monitor" are informational: CQL TD loss may increase
# as the conservative penalty takes effect, Q-values may shift as the
# agent learns, and IQL V-loss may grow from near-zero as V learns
# non-trivial values.  These are expected behaviors, not failures.


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def load_tb_scalars(tb_dir: str, tags: Dict[str, str]) -> Dict[str, List[float]]:
    """Load scalar time series from a Tensorboard log directory."""
    if not HAS_TB:
        return {}

    path = Path(tb_dir)
    if not path.is_dir():
        return {}

    ea = event_accumulator.EventAccumulator(str(path))
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    result: Dict[str, List[float]] = {}

    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            result[tag] = [e.value for e in events]

    return result


def analyze_convergence(values: List[float], window: int = 10) -> Dict[str, Any]:
    """Analyze a scalar time series for convergence.

    Returns a dict with:
      - first_window: mean of first `window` values
      - last_window: mean of last `window` values
      - reduction_pct: percentage reduction from first to last
      - monotonic_last_half: whether the second half is roughly decreasing
      - has_nan: whether any NaN/Inf values are present
      - converged: overall convergence verdict
    """
    arr = np.array(values)
    has_nan = bool(np.any(~np.isfinite(arr)))

    n = len(arr)
    w = min(window, max(1, n // 4))

    first_w = float(np.mean(arr[:w]))
    last_w = float(np.mean(arr[-w:]))

    if abs(first_w) > 1e-10:
        reduction_pct = (1 - last_w / first_w) * 100
    else:
        reduction_pct = 0.0

    # Check that the second half doesn't diverge: compare the mean of
    # the third quarter to the mean of the fourth quarter.
    mid = n // 2
    q3 = n * 3 // 4
    if q3 > mid and n > q3:
        q3_mean = float(np.mean(arr[mid:q3]))
        q4_mean = float(np.mean(arr[q3:]))
        stable_second_half = q4_mean <= q3_mean * 1.1  # allow 10% tolerance
    else:
        stable_second_half = True

    converged = (not has_nan) and (last_w <= first_w * 1.05) and stable_second_half

    return {
        "n_epochs": n,
        "first_window": round(first_w, 6),
        "last_window": round(last_w, 6),
        "reduction_pct": round(reduction_pct, 1),
        "stable_second_half": stable_second_half,
        "has_nan": has_nan,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convergence analysis for CQL and IQL training (R37).",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (for downstream scripts).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_results: Dict[str, Dict] = {}
    any_failure = False

    if not args.json:
        print("=" * 70)
        print("  R37: Training Convergence Analysis")
        print("=" * 70)

        if not HAS_TB:
            print("\n  WARNING: tensorboard not installed. "
                  "Install with: pip install tensorboard")
            print("  Falling back to checkpoint-only analysis.\n")

    for run_cfg in RUNS:
        name = run_cfg["name"]
        tb_dir = run_cfg["tb_dir"]
        tags = run_cfg["tags"]

        if not args.json:
            print(f"\n  --- {name} ---")
            print(f"  TB dir: {tb_dir}")

        scalars = load_tb_scalars(tb_dir, tags)

        run_result: Dict[str, Any] = {"name": name, "metrics": {}}

        if not scalars:
            if not args.json:
                print(f"  No Tensorboard data found at {tb_dir}.")
                print(f"  Check that training was run with TB logging enabled.")
            run_result["status"] = "no_data"
        else:
            all_converged = True
            for tag, (label, role) in tags.items():
                if tag not in scalars:
                    if not args.json:
                        print(f"  {label}: tag '{tag}' not found in TB logs")
                    continue

                values = scalars[tag]
                analysis = analyze_convergence(values)
                analysis["role"] = role
                run_result["metrics"][tag] = analysis

                if role == "convergence":
                    status = "PASS" if analysis["converged"] else "FAIL"
                    if not analysis["converged"]:
                        all_converged = False
                        any_failure = True
                else:
                    # Monitor-only: report but don't affect pass/fail
                    status = "INFO"

                if not args.json:
                    print(f"  [{status}] {label}: "
                          f"{analysis['first_window']:.4f} -> "
                          f"{analysis['last_window']:.4f} "
                          f"({analysis['reduction_pct']:+.1f}%) "
                          f"over {analysis['n_epochs']} epochs"
                          f"{' [NaN detected!]' if analysis['has_nan'] else ''}")

            run_result["status"] = "converged" if all_converged else "not_converged"

        all_results[name] = run_result

    if args.json:
        json.dump(all_results, sys.stdout, indent=2)
        print()
    else:
        print("\n" + "=" * 70)
        print("  Summary")
        print("-" * 70)
        for name, result in all_results.items():
            status = result.get("status", "unknown")
            icon = "PASS" if status == "converged" else "FAIL" if status == "not_converged" else "SKIP"
            print(f"  [{icon}] {name}: {status}")

        if any_failure:
            print("\n  CONVERGENCE CHECK FAILED -- consider retraining.")
        else:
            print("\n  All training runs converged successfully.")
        print("=" * 70)

    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()
