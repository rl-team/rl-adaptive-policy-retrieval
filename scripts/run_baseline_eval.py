"""
run Fixed-K and Heuristic baselines on 50 episodes (test set) and report metrics with CIs

uses the evaluation harness and real PA simulator. test set = fixed seed so the
episode sequence is reproducible and the same for all policies (fair comparison).

usage:
    python -m scripts.run_baseline_eval
    python -m scripts.run_baseline_eval --episodes 50 --seed 999
"""
from __future__ import annotations

import contextlib
import io
import argparse
from typing import Callable, List, Tuple

with contextlib.redirect_stderr(io.StringIO()):
    from rl.env import PolicyRetrievalEnv

from simulator.pa_simulator import PASimulator
from rl.reward import RewardFunction
from baselines.fixed_k import FixedKPolicy
from baselines.heuristic import HeuristicPolicy
from scripts.evaluate_agent import run_baseline_episode
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


def _make_policies() -> List[Tuple[str, Callable]]:
    return [
        ("FixedK(k=3)", lambda env, p=FixedKPolicy(k=3): run_baseline_episode(env, p)),
        ("FixedK(k=5)", lambda env, p=FixedKPolicy(k=5): run_baseline_episode(env, p)),
        ("Heuristic(0.8)", lambda env, p=HeuristicPolicy(confidence_threshold=0.8): run_baseline_episode(env, p)),
    ]


def _format_ci(mean_val: float, ci: Tuple[float, float], pct: bool = False) -> str:
    low, high = ci
    if pct:
        return f"{mean_val:.1%} (95% CI: {low:.1%}-{high:.1%})"
    return f"{mean_val:.2f} (95% CI: {low:.2f}-{high:.2f})"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Fixed-K and Heuristic baselines on test set (50 episodes).",
    )
    parser.add_argument("--episodes", type=int, default=50, help="test set size (episodes per policy)")
    parser.add_argument("--seed", type=int, default=999, help="random seed for test set (reproducible sequence)")
    parser.add_argument("--alpha", type=float, default=0.1, help="cost weight for cost-adjusted utility")
    args = parser.parse_args()

    policies = _make_policies()
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
    print(f"  {'Policy':<18} {'Accuracy':>26} {'Mean Chunks':>26} {'Cost-Adj Utility':>26}")
    print("-" * 76)

    for name, m in results.items():
        acc_str = _format_ci(m["accuracy"], m["accuracy_ci"], pct=True)
        ch_str = _format_ci(m["mean_chunks"], m["mean_chunks_ci"], pct=False)
        cau_str = _format_ci(m["cost_adjusted_utility"], m["cost_adjusted_utility_ci"], pct=False)
        print(f"  {name:<18} {acc_str:>26} {ch_str:>26} {cau_str:>26}")

    print("=" * 76)


if __name__ == "__main__":
    main()
