"""
Validation script for the real PASimulator integration.

Runs a random policy for 5 episodes using the real CMS corpus and
sentence-transformer retrieval. This verifies that the environment
works end-to-end with real data.

Run with:
    python -m scripts.validate_real_env
"""

from __future__ import annotations

# Gym prints a raw deprecation notice to stderr on import (not via the
# warnings module), so we must redirect stderr during import to silence it.
# See collect_offline_dataset.py for details.
import contextlib, io
with contextlib.redirect_stderr(io.StringIO()):
    from rl.env import PolicyRetrievalEnv

import time
import numpy as np

from simulator.pa_simulator import PASimulator
from rl.reward import RewardFunction


def run_validation(num_episodes: int = 5, seed: int = 42) -> None:
    print("=" * 78)
    print("  Validating Real Simulator Integration")
    print("=" * 78)
    print("\n  Loading real PASimulator (may take a moment to load model/corpus)...")

    start_time = time.time()
    sim = PASimulator(seed=seed)
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.2f}s")
    print(f"  Corpus size: {len(sim.get_corpus())} chunks")
    print(f"  Embedding dim: {sim.embedding_dim}")

    env = PolicyRetrievalEnv(
        simulator=sim,
        top_k=10,
        max_steps=20,
        reward_fn=RewardFunction(step_cost=0.1),
        query_encoder=sim.encode,
    )

    rng = np.random.default_rng(seed)

    print("\n" + "=" * 78)
    print(f"  Running {num_episodes} random episodes")
    print("=" * 78)
    print(f"\n{'Ep':>3}  {'Steps':>5}  {'Return':>7}  "
          f"{'Decision':>8}  {'Truth':>8}  {'Correct':>7}  {'Forced':>6}")
    print("-" * 78)

    all_returns = []
    all_steps = []
    all_correct = []

    for ep in range(num_episodes):
        obs, info = env.reset()

        # Verify observation dimensionality against the env's own spec
        expected_shape = env.observation_space.shape
        if obs.shape != expected_shape:
            raise ValueError(
                f"Expected obs shape {expected_shape}, got {obs.shape}"
            )

        episode_return = 0.0
        steps = 0

        while True:
            n_candidates = len(env.candidates)
            valid_actions = list(range(n_candidates)) + [env.stop_action]
            action = valid_actions[int(rng.integers(0, len(valid_actions)))]

            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_return += reward
            steps += 1

            if terminated:
                break

        decision = step_info.get("decision", "N/A")
        ground_truth = step_info.get("ground_truth", "N/A")
        correct = step_info.get("correct", False)
        forced = step_info.get("forced_stop", False)

        all_returns.append(episode_return)
        all_steps.append(steps)
        all_correct.append(correct)

        print(f"{ep + 1:>3}  {steps:>5}  {episode_return:>7.2f}  "
              f"{decision:>8}  {ground_truth:>8}  "
              f"{'yes' if correct else 'no':>7}  "
              f"{'yes' if forced else 'no':>6}")

    print("-" * 78)
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 36)
    print(f"{'Mean return':<25} {np.mean(all_returns):>10.3f}")
    print(f"{'Mean steps':<25} {np.mean(all_steps):>10.1f}")
    print(f"{'Accuracy':<25} {np.mean(all_correct):>10.1%}")

    print("\n" + "=" * 78)
    print("  Done")
    print("=" * 78)


if __name__ == "__main__":
    run_validation()
