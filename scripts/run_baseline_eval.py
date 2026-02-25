"""
Run Fixed-K and Heuristic baselines (and optionally Conservative QL) on a test set;
report metrics with CIs and write results to JSON for plot scripts.

Uses the evaluation harness and real PA simulator. Test set = fixed seed so the
episode sequence is reproducible and the same for all policies (fair comparison).

Usage:
    python -m scripts.run_baseline_eval
    python -m scripts.run_baseline_eval --episodes 50 --seed 42
    python -m scripts.run_baseline_eval --checkpoint path/to/checkpoint.pt --output data/eval_results.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

with contextlib.redirect_stderr(io.StringIO()):
    from rl.env import PolicyRetrievalEnv

from simulator.pa_simulator import PASimulator
from rl.reward import RewardFunction
from baselines.fixed_k import FixedKPolicy
from baselines.heuristic import HeuristicPolicy
from scripts.evaluate_agent import run_baseline_episode, run_conservative_ql_episode
from evaluation.eval_harness import run_evaluation

# ---------------------------------------------------------------------------
# env factory and policy runners
# ---------------------------------------------------------------------------
def _env_factory(seed: int) -> PolicyRetrievalEnv:
    sim = PASimulator(seed=seed)
    return PolicyRetrievalEnv(
        simulator=sim,
        top_k=10,
        max_steps=20,
        reward_fn=RewardFunction(step_cost=0.1),
        query_encoder=sim.encode,
    )


def _make_policies(checkpoint_path: Optional[str] = None) -> List[Tuple[str, Callable]]:
    policies: List[Tuple[str, Callable]] = [
        ("FixedK(k=3)", lambda env, p=FixedKPolicy(k=3): run_baseline_episode(env, p)),
        ("FixedK(k=5)", lambda env, p=FixedKPolicy(k=5): run_baseline_episode(env, p)),
        ("Heuristic(0.8)", lambda env, p=HeuristicPolicy(confidence_threshold=0.8): run_baseline_episode(env, p)),
    ]
    if checkpoint_path:
        from rl.conservative_ql_agent import ConservativeQLAgent
        agent = ConservativeQLAgent.load(checkpoint_path)
        policies.append(
            ("Conservative QL (α=0.5)", lambda env, a=agent: run_conservative_ql_episode(env, a)),
        )
    return policies


def _format_ci(mean_val: float, ci: Tuple[float, float], pct: bool = False) -> str:
    low, high = ci
    if pct:
        return f"{mean_val:.1%} (95% CI: {low:.1%}-{high:.1%})"
    return f"{mean_val:.2f} (95% CI: {low:.2f}-{high:.2f})"


def _results_to_json_serializable(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert results dict so CI tuples become lists for JSON."""
    out: Dict[str, Any] = {}
    for name, m in results.items():
        row: Dict[str, Any] = {
            "accuracy": m["accuracy"],
            "accuracy_ci": list(m["accuracy_ci"]),
            "mean_chunks": m["mean_chunks"],
            "mean_chunks_ci": list(m["mean_chunks_ci"]),
            "cost_adjusted_utility": m["cost_adjusted_utility"],
            "cost_adjusted_utility_ci": list(m["cost_adjusted_utility_ci"]),
            "n_episodes": m["n_episodes"],
        }
        if "mean_return" in m and "mean_return_ci" in m:
            row["mean_return"] = m["mean_return"]
            row["mean_return_ci"] = list(m["mean_return_ci"])
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baselines (and optional CQL) on test set; write results to JSON.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="test set size (episodes per policy)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for test set (reproducible sequence, match milestone report)")
    parser.add_argument("--alpha", type=float, default=0.1, help="cost weight for cost-adjusted utility")
    parser.add_argument("--output", type=str, default="data/eval_results.json", help="write results to this JSON file")
    parser.add_argument("--checkpoint", type=str, default=None, help="if set, evaluate this CQL checkpoint and include in results")
    args = parser.parse_args()

    policies = _make_policies(args.checkpoint)
    results = run_evaluation(
        policies=policies,
        num_episodes=args.episodes,
        seed=args.seed,
        env_factory=_env_factory,
        alpha=args.alpha,
        n_bootstrap=1000,
    )

    print("=" * 76)
    print("  Baseline evaluation on test set (bootstrap 95% CI, 1000 samples)")
    print("=" * 76)
    print(f"  Episodes: {args.episodes}  Seed: {args.seed}  Alpha: {args.alpha}")
    print("-" * 76)
    print(f"  {'Policy':<24} {'Accuracy':>26} {'Steps':>26} {'Return':>26}")
    print("-" * 76)

    for name, m in results.items():
        acc_str = _format_ci(m["accuracy"], m["accuracy_ci"], pct=True)
        steps_str = _format_ci(m["mean_chunks"], m["mean_chunks_ci"], pct=False)
        ret_str = _format_ci(m["mean_return"], m["mean_return_ci"], pct=False) if "mean_return" in m else "N/A"
        print(f"  {name:<24} {acc_str:>26} {steps_str:>26} {ret_str:>26}")

    print("=" * 76)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"episodes": args.episodes, "seed": args.seed, "alpha": args.alpha},
        "results": _results_to_json_serializable(results),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results written to {out_path}")


if __name__ == "__main__":
    main()
