"""
Environment sanity check for the expanded corpus (R32).

Runs a small number of episodes per baseline policy on the live environment
to verify:
1. No crashes or exceptions (env, oracle, retriever, baselines all compose).
2. All episodes terminate within max_steps.
3. Oracle produces valid decisions (approve/deny/pend).
4. At least one baseline achieves > 0% accuracy (corpus is solvable).

This is a fast pre-training gate -- run before committing to a multi-hour
CQL training job. Equivalent to a smoke test for the full pipeline.

Usage:
    python -m scripts.sanity_check
    python -m scripts.sanity_check --episodes 10 --seed 99

Reference: Implementation Plan R32.
"""

from __future__ import annotations

# Suppress gym's raw stderr deprecation notice.
import contextlib, io
with contextlib.redirect_stderr(io.StringIO()):
    from rl.env import PolicyRetrievalEnv

import argparse
import json
import sys
import time
from typing import Dict, List

import numpy as np

from simulator.pa_simulator import PASimulator
from rl.reward import RewardFunction

from baselines.fixed_k import FixedKPolicy
from baselines.heuristic import HeuristicPolicy


# ---------------------------------------------------------------------------
# Episode runner (same pattern as evaluate_agent.py)
# ---------------------------------------------------------------------------

def run_baseline_episode(env: PolicyRetrievalEnv, policy) -> Dict[str, object]:
    """Run one episode with a baseline policy. Returns episode metrics."""
    obs, _info = env.reset()
    policy.reset()
    episode_return = 0.0
    steps = 0

    while True:
        candidates = env.candidates

        if policy.should_stop(obs, env.retrieved_chunks):
            action = env.stop_action
        else:
            chosen = policy.select_action(obs, candidates)
            if chosen == -1:
                action = env.stop_action
            else:
                try:
                    action = candidates.index(chosen)
                except ValueError:
                    action = env.stop_action

        obs, reward, terminated, _truncated, step_info = env.step(action)
        episode_return += reward
        steps += 1

        if terminated:
            break

    return {
        "return": episode_return,
        "steps": steps,
        "correct": step_info.get("correct", False),
        "decision": step_info.get("decision", "N/A"),
        "ground_truth": step_info.get("ground_truth", "N/A"),
        "forced_stop": step_info.get("forced_stop", False),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Environment sanity check for expanded corpus (R32).",
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per policy (default: 5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  R32: Environment Sanity Check")
    print("=" * 70)
    print(f"\n  Episodes per policy: {args.episodes}")
    print(f"  Seed: {args.seed}")

    # Load corpus stats for context
    try:
        with open("data/corpus_stats.json") as f:
            stats = json.load(f)
        print(f"  Corpus: {stats['total_chunks']} chunks, "
              f"{stats['total_procedures']} procedures")
    except FileNotFoundError:
        print("  Warning: data/corpus_stats.json not found (run generate_corpus_stats.py)")

    # Define baseline policies
    policies = [
        ("FixedK(k=3)", FixedKPolicy(k=3)),
        ("FixedK(k=5)", FixedKPolicy(k=5)),
        ("FixedK(k=7)", FixedKPolicy(k=7)),
        ("Heuristic(0.8)", HeuristicPolicy(confidence_threshold=0.8)),
    ]

    start_time = time.time()
    all_results: Dict[str, List[Dict]] = {}
    any_failures = False

    for policy_name, policy in policies:
        # Fresh env per policy (same seed = same request sequence)
        sim = PASimulator(seed=args.seed)
        env = PolicyRetrievalEnv(
            simulator=sim,
            top_k=10,
            max_steps=20,
            reward_fn=RewardFunction(step_cost=0.1),
            query_encoder=sim.encode,
        )

        results = []
        print(f"\n  --- {policy_name} ---")

        for ep in range(args.episodes):
            result = run_baseline_episode(env, policy)
            results.append(result)
            status = "correct" if result["correct"] else "WRONG"
            forced = " [FORCED]" if result["forced_stop"] else ""
            print(f"    Episode {ep+1}: {status}  "
                  f"decision={result['decision']:>7s}  "
                  f"gt={result['ground_truth']:>7s}  "
                  f"steps={result['steps']:>2d}  "
                  f"return={result['return']:>6.2f}{forced}")

        accuracy = np.mean([r["correct"] for r in results])
        mean_steps = np.mean([r["steps"] for r in results])
        mean_return = np.mean([r["return"] for r in results])
        decisions = set(r["decision"] for r in results)

        print(f"    Summary: accuracy={accuracy:.0%}  "
              f"mean_steps={mean_steps:.1f}  "
              f"mean_return={mean_return:.2f}  "
              f"decisions={decisions}")

        all_results[policy_name] = results

    elapsed = time.time() - start_time

    # -- Sanity assertions --
    print("\n" + "=" * 70)
    print("  Sanity Checks")
    print("-" * 70)

    checks_passed = 0
    checks_total = 0

    # Check 1: all episodes terminated (no infinite loops)
    checks_total += 1
    all_terminated = all(
        r["steps"] <= 20
        for results in all_results.values()
        for r in results
    )
    if all_terminated:
        print("  [PASS] All episodes terminated within max_steps=20")
        checks_passed += 1
    else:
        print("  [FAIL] Some episodes exceeded max_steps")
        any_failures = True

    # Check 2: oracle produced valid decisions
    checks_total += 1
    all_decisions = set(
        r["decision"]
        for results in all_results.values()
        for r in results
    )
    valid_decisions = {"approve", "deny", "pend"}
    if all_decisions.issubset(valid_decisions):
        print(f"  [PASS] All decisions are valid: {all_decisions}")
        checks_passed += 1
    else:
        print(f"  [FAIL] Invalid decisions found: {all_decisions - valid_decisions}")
        any_failures = True

    # Check 3: at least one policy achieves > 0% accuracy
    checks_total += 1
    best_policy = max(all_results, key=lambda p: np.mean([r["correct"] for r in all_results[p]]))
    best_acc = np.mean([r["correct"] for r in all_results[best_policy]])
    if best_acc > 0:
        print(f"  [PASS] Best baseline: {best_policy} at {best_acc:.0%} accuracy")
        checks_passed += 1
    else:
        print(f"  [FAIL] All policies at 0% accuracy -- corpus may be unsolvable")
        any_failures = True

    # Check 4: ground truth includes multiple decision types
    checks_total += 1
    gt_decisions = set(
        r["ground_truth"]
        for results in all_results.values()
        for r in results
    )
    if len(gt_decisions) >= 2:
        print(f"  [PASS] Ground truth diversity: {gt_decisions}")
        checks_passed += 1
    else:
        print(f"  [WARN] Only one ground truth type seen: {gt_decisions} (may need more episodes)")

    # Check 5: no forced stops dominate (agent isn't timing out every episode)
    checks_total += 1
    total_episodes = sum(len(r) for r in all_results.values())
    forced_count = sum(
        r["forced_stop"]
        for results in all_results.values()
        for r in results
    )
    forced_pct = forced_count / total_episodes if total_episodes > 0 else 0
    if forced_pct < 0.8:
        print(f"  [PASS] Forced stops: {forced_count}/{total_episodes} ({forced_pct:.0%})")
        checks_passed += 1
    else:
        print(f"  [WARN] High forced stop rate: {forced_count}/{total_episodes} ({forced_pct:.0%})")

    print("-" * 70)
    print(f"  Result: {checks_passed}/{checks_total} checks passed in {elapsed:.1f}s")

    if any_failures:
        print("  SANITY CHECK FAILED -- do not proceed with training.")
        print("=" * 70)
        sys.exit(1)
    else:
        print("  SANITY CHECK PASSED -- safe to proceed with CQL training (R33).")
        print("=" * 70)


if __name__ == "__main__":
    main()
