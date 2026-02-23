"""
Evaluate a trained Conservative Q-Learning agent on-policy and compare
against baselines.

Loads a trained checkpoint, runs the greedy policy in the real environment,
and reports accuracy, mean steps, and mean return alongside baseline policies.
This is the primary convergence check after training.

Usage:
    python -m scripts.evaluate_agent --checkpoint runs/experiment_01/checkpoint.pt
    python -m scripts.evaluate_agent --checkpoint runs/experiment_01/checkpoint.pt \
        --episodes 50 --seed 99

Reference: EDD Use Case 4 (postconditions).
"""

from __future__ import annotations

# Suppress gym's raw stderr deprecation notice (see collect_offline_dataset.py).
import contextlib, io
with contextlib.redirect_stderr(io.StringIO()):
    from rl.env import PolicyRetrievalEnv

import argparse
import time
from typing import Dict, List

import numpy as np

from simulator.pa_simulator import PASimulator
from rl.conservative_ql_agent import ConservativeQLAgent
from rl.reward import RewardFunction

from baselines.fixed_k import FixedKPolicy
from baselines.heuristic import HeuristicPolicy


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_conservative_ql_episode(
    env: PolicyRetrievalEnv,
    agent: ConservativeQLAgent,
) -> Dict[str, object]:
    """Run one episode with the trained Conservative Q-Learning agent."""
    obs, info = env.reset()
    episode_return = 0.0
    steps = 0

    while True:
        candidates = env.candidates
        valid_actions = list(range(len(candidates))) + [env.stop_action]
        action = agent.select_action(obs, valid_actions=valid_actions)

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


def run_baseline_episode(
    env: PolicyRetrievalEnv,
    policy,
) -> Dict[str, object]:
    """Run one episode with a baseline policy."""
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
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_policy(
    name: str,
    runner,
    num_episodes: int,
    seed: int,
) -> Dict[str, float]:
    """Run a policy for N episodes and aggregate metrics.

    Creates a fresh environment with the given seed so that each policy
    sees the exact same sequence of PA requests (fair comparison).
    """
    sim = PASimulator(seed=seed)
    env = PolicyRetrievalEnv(
        simulator=sim,
        top_k=10,
        max_steps=20,
        reward_fn=RewardFunction(step_cost=0.1),
        query_encoder=sim.encode,
    )

    all_returns, all_steps, all_correct = [], [], []
    for _ep in range(num_episodes):
        result = runner(env)
        all_returns.append(result["return"])
        all_steps.append(result["steps"])
        all_correct.append(result["correct"])

    return {
        "name": name,
        "accuracy": float(np.mean(all_correct)),
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_steps": float(np.mean(all_steps)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Conservative Q-Learning agent vs baselines.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint (.pt file).",
    )
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per policy.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  On-Policy Evaluation: Conservative Q-Learning vs Baselines")
    print("=" * 70)

    # -- Load agent --
    print(f"\n  Checkpoint: {args.checkpoint}")
    agent = ConservativeQLAgent.load(args.checkpoint)
    print(f"  Agent loaded (alpha={agent.alpha}, gamma={agent.gamma})")
    print(f"  Episodes per policy: {args.episodes}")
    print(f"  Seed: {args.seed}")

    # -- Define all policies --
    # Each policy gets a fresh env with the same seed, ensuring identical
    # request sequences for fair apples-to-apples comparison.
    policies: List[tuple] = [
        ("ConservativeQL", lambda env: run_conservative_ql_episode(env, agent)),
        ("FixedK(k=3)", lambda env, p=FixedKPolicy(k=3): run_baseline_episode(env, p)),
        ("FixedK(k=5)", lambda env, p=FixedKPolicy(k=5): run_baseline_episode(env, p)),
        ("Heuristic(0.8)", lambda env, p=HeuristicPolicy(confidence_threshold=0.8): run_baseline_episode(env, p)),
    ]

    # -- Evaluate all policies --
    print(f"\n  {'Policy':<23} {'Accuracy':>8}  {'Return':>12}  {'Steps':>5}")
    print("  " + "-" * 53)

    start_time = time.time()
    all_metrics = []

    for name, runner in policies:
        metrics = evaluate_policy(name, runner, args.episodes, args.seed)
        all_metrics.append(metrics)
        print(f"  {metrics['name']:<23} {metrics['accuracy']:>7.1%}  "
              f"{metrics['mean_return']:>6.2f} +/- {metrics['std_return']:<4.2f}  "
              f"{metrics['mean_steps']:>5.1f}")

    elapsed = time.time() - start_time
    print("  " + "-" * 53)
    print(f"  Evaluated in {elapsed:.1f}s")

    # -- Convergence verdict --
    trained = all_metrics[0]
    baseline_accs = [m["accuracy"] for m in all_metrics[1:]]
    best_baseline_acc = max(baseline_accs)
    best_baseline_name = all_metrics[1 + baseline_accs.index(best_baseline_acc)]["name"]

    print("\n" + "=" * 70)
    print("  Convergence Analysis")
    print("-" * 70)

    if trained["accuracy"] > best_baseline_acc:
        print(f"  Trained agent accuracy ({trained['accuracy']:.1%}) > best baseline "
              f"({best_baseline_name}: {best_baseline_acc:.1%})")
        print("  Verdict: Training converged. Agent outperforms baselines.")
    elif trained["accuracy"] == best_baseline_acc:
        print(f"  Trained agent accuracy ({trained['accuracy']:.1%}) = best baseline "
              f"({best_baseline_name}: {best_baseline_acc:.1%})")
        # Check if the trained agent achieves a higher return (same accuracy
        # with fewer steps means better efficiency)
        best_idx = 1 + baseline_accs.index(best_baseline_acc)
        if trained["mean_return"] > all_metrics[best_idx]["mean_return"]:
            print(f"  Trained return ({trained['mean_return']:.2f}) > baseline "
                  f"({all_metrics[best_idx]['mean_return']:.2f})")
            print("  Verdict: Same accuracy, higher return. Training converged.")
        elif abs(trained["mean_return"] - all_metrics[best_idx]["mean_return"]) < 0.05:
            print(f"  Trained return ({trained['mean_return']:.2f}) ~ baseline "
                  f"({all_metrics[best_idx]['mean_return']:.2f})")
            print("  Verdict: Agent matches best baseline. Training converged,")
            print("           but learned the same strategy as FixedK.")
        else:
            print("  Verdict: Same accuracy but lower efficiency. Consider")
            print("           retraining with different alpha or more epochs.")
    else:
        print(f"  Trained agent accuracy ({trained['accuracy']:.1%}) < best baseline "
              f"({best_baseline_name}: {best_baseline_acc:.1%})")
        print("  Verdict: Agent underperforms. Retrain needed.")
        print("  Suggestions (per EDD Use Case 4, Alt flows):")
        print("    1. Reduce alpha (too conservative): --alpha 0.1")
        print("    2. Increase epochs: --epochs 500")
        print("    3. Increase learning rate: --lr 1e-3")

    print("=" * 70)


if __name__ == "__main__":
    main()
