"""
Collect offline dataset using behavior policies.

Generates an offline dataset of prior authorization retrieval episodes
using a mix of behavior policies (Fixed-K, Heuristic, and Epsilon-Greedy).
Saves the transitions to a pickled ReplayBuffer for offline RL training.

Usage:
    python -m scripts.collect_offline_dataset --episodes 200

Reference: EDD Decision 5 (Multi-Policy Dataset Generation).
"""

from __future__ import annotations

# Gym prints a raw deprecation notice to stderr on import (not via the
# warnings module), so we must redirect stderr during import to silence it.
# This project intentionally uses the older ``gym`` package for
# compatibility with standard offline RL libraries.
import contextlib, io
with contextlib.redirect_stderr(io.StringIO()):
    from rl.env import PolicyRetrievalEnv

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from simulator.pa_simulator import PASimulator
from rl.reward import RewardFunction
from rl.conservative_ql_agent import ReplayBuffer

from baselines.base import BaselinePolicy
from baselines.fixed_k import FixedKPolicy
from baselines.epsilon_greedy import EpsilonGreedyPolicy
from baselines.heuristic import HeuristicPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_baseline_episode(
    env: PolicyRetrievalEnv,
    policy: BaselinePolicy,
) -> Tuple[List[np.ndarray], List[int], List[float], List[np.ndarray], List[bool]]:
    """Run one episode using a baseline policy and return transitions.

    Baseline policies operate on **absolute** corpus indices (as returned by
    ``env.candidates``), while ``env.step()`` expects a **relative** index
    into that same candidate list (0..K-1) or the stop action.  This helper
    performs the mapping so callers don't have to.

    Returns
    -------
    Tuple of (states, actions, rewards, next_states, dones).
    """
    obs, _info = env.reset()
    policy.reset()

    states, actions, rewards, next_states, dones = [], [], [], [], []

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
                    # chosen index not in current candidates — stop safely
                    action = env.stop_action

        next_obs, reward, terminated, _truncated, _info = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_obs)
        dones.append(terminated)

        obs = next_obs
        if terminated:
            break

    return states, actions, rewards, next_states, dones


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline dataset.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to collect")
    parser.add_argument("--output", type=str, default="data/offline_buffer.pkl", help="Output pickle path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--step-cost", type=float, default=0.1,
                        help="Reward step cost (lambda). Default 0.1; "
                             "ablation values: {0.05, 0.1, 0.2}.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_dataset() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Offline Dataset Collection")
    print("=" * 70)

    start_time = time.time()
    sim = PASimulator(seed=args.seed)

    env = PolicyRetrievalEnv(
        simulator=sim,
        top_k=10,
        max_steps=20,
        reward_fn=RewardFunction(step_cost=args.step_cost),
        query_encoder=sim.encode,
    )

    buffer = ReplayBuffer()

    # EDD Decision 5: mix Fixed-K (conservative), Heuristic (aggressive
    # early stopping), and ε-greedy (random exploration) to ensure broad
    # state-action coverage and reduce distributional shift.
    policies: List[Tuple[str, BaselinePolicy]] = [
        ("FixedK(k=3)", FixedKPolicy(k=3)),
        ("FixedK(k=5)", FixedKPolicy(k=5)),
        ("Heuristic(thresh=0.8)", HeuristicPolicy(confidence_threshold=0.8)),
        ("EpsilonGreedy(e=0.3, FixedK=2)", EpsilonGreedyPolicy(FixedKPolicy(k=2), epsilon=0.3, stop_prob=0.3, seed=args.seed)),
        ("EpsilonGreedy(e=0.3, FixedK=4)", EpsilonGreedyPolicy(FixedKPolicy(k=4), epsilon=0.3, stop_prob=0.3, seed=args.seed + 1)),
    ]

    episodes_per_policy = args.episodes // len(policies)

    all_returns: List[float] = []
    all_lengths: List[int] = []

    print(f"\n  Collecting {args.episodes} episodes using {len(policies)} policies...")
    print(f"  Step cost (lambda): {args.step_cost}")

    total_episodes_done = 0
    for policy_name, policy in policies:
        n_eps = episodes_per_policy
        if policy_name == policies[-1][0]:
            n_eps += args.episodes % len(policies)

        print(f"    - {policy_name:30s}: {n_eps} episodes")

        for _ep in range(n_eps):
            ep_s, ep_a, ep_r, ep_ns, ep_d = run_baseline_episode(env, policy)

            buffer.add_episode(ep_s, ep_a, ep_r, ep_ns, ep_d)
            all_returns.append(sum(ep_r))
            all_lengths.append(len(ep_a))
            total_episodes_done += 1

            if total_episodes_done % 50 == 0:
                print(f"      [{total_episodes_done}/{args.episodes}] transitions: {len(buffer)}")

    elapsed = time.time() - start_time

    # Save buffer
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(buffer, f)

    print("\n" + "-" * 70)
    print("  Collection Summary")
    print("-" * 70)
    print(f"  Time elapsed:       {elapsed:.1f}s")
    print(f"  Total episodes:     {args.episodes}")
    print(f"  Total transitions:  {len(buffer)}")
    print(f"  Mean episode len:   {np.mean(all_lengths):.2f}")
    print(f"  Mean return:        {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}")
    print(f"  Buffer saved to:    {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    collect_dataset()
