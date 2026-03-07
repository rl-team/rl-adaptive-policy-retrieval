"""
Full 6-policy evaluation: CQL, IQL, BC, FixedK(3), FixedK(5), Heuristic.

Runs each policy for 200 episodes with seed=42 on the 186-chunk corpus
environment, then writes per-episode results to data/eval_results_final.json.

Usage:
    python -m scripts.run_full_eval

Output:
    data/eval_results_final.json
"""

from __future__ import annotations

import json
import os
import time

from scripts.evaluate_agent import (
    evaluate_policy,
    run_trained_agent_episode,
    run_baseline_episode,
)

from rl.conservative_ql_agent import ConservativeQLAgent
from rl.iql_agent import IQLAgent
from baselines.bc import BehavioralCloningPolicy
from baselines.fixed_k import FixedKPolicy
from baselines.heuristic import HeuristicPolicy


NUM_EPISODES = 200
SEED = 42
OUTPUT_PATH = "data/eval_results_final.json"


def main() -> None:
    print("=" * 70)
    print("  Full 6-Policy Evaluation (R39)")
    print("=" * 70)

    start = time.time()

    # -- Load trained agents --
    cql_agent = ConservativeQLAgent.load("runs/cql_2k/checkpoint.pt")
    print(f"  CQL loaded (alpha={cql_agent.alpha}, gamma={cql_agent.gamma})")

    iql_agent = IQLAgent.load("runs/iql_2k/checkpoint.pt")
    print(f"  IQL loaded (tau={iql_agent.tau}, beta={iql_agent.beta})")

    bc_policy = BehavioralCloningPolicy.load("runs/bc_2k/checkpoint.pt")
    print("  BC loaded")

    # -- Define all 6 policies --
    policies = [
        ("CQL", "cql",
         lambda env, a=cql_agent: run_trained_agent_episode(env, a)),
        ("IQL", "iql",
         lambda env, a=iql_agent: run_trained_agent_episode(env, a)),
        ("BC", "bc",
         lambda env, p=bc_policy: run_baseline_episode(env, p)),
        ("FixedK(k=3)", "fixed_k_3",
         lambda env, p=FixedKPolicy(k=3): run_baseline_episode(env, p)),
        ("FixedK(k=5)", "fixed_k_5",
         lambda env, p=FixedKPolicy(k=5): run_baseline_episode(env, p)),
        ("Heuristic(0.8)", "heuristic",
         lambda env, p=HeuristicPolicy(confidence_threshold=0.8): run_baseline_episode(env, p)),
    ]

    # -- Evaluate each policy --
    results = {}

    print(f"\n  {'Policy':<23} {'Accuracy':>8}  {'Return':>12}  {'Steps':>5}")
    print("  " + "-" * 53)

    for display_name, key, runner in policies:
        metrics = evaluate_policy(
            display_name, runner, NUM_EPISODES, SEED,
        )
        results[key] = {
            "name": display_name,
            "accuracy": metrics["accuracy"],
            "mean_return": metrics["mean_return"],
            "std_return": metrics["std_return"],
            "mean_steps": metrics["mean_steps"],
            "episodes": metrics["episodes"],
        }
        print(f"  {display_name:<23} {metrics['accuracy']:>7.1%}  "
              f"{metrics['mean_return']:>6.2f} +/- {metrics['std_return']:<4.2f}  "
              f"{metrics['mean_steps']:>5.1f}")

    elapsed = time.time() - start
    print("  " + "-" * 53)
    print(f"  Evaluated in {elapsed:.1f}s")

    # -- Save results --
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
