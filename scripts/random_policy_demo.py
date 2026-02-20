"""
Random policy demo -- run a uniform-random agent on PolicyRetrievalEnv.

Exercises the full environment loop (reset -> step -> terminate) for 10
episodes and prints per-episode and aggregate statistics. This is the R5
deliverable: verifying the env works end-to-end with mock data.

Run with:
    python -m scripts.random_policy_demo

Reference: EDD 7.2 (R5: Test env with mock data).
"""

from __future__ import annotations

import numpy as np

from tests.mock_simulator import MockPASimulator
from rl.env import PolicyRetrievalEnv
from rl.reward import RewardFunction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_EPISODES = 10
TOP_K = 10
MAX_STEPS = 20
STEP_COST = 0.1
SEED = 42


def run_random_policy(
    num_episodes: int = NUM_EPISODES,
    seed: int = SEED,
) -> None:
    """Run a uniform-random policy and report statistics."""

    sim = MockPASimulator(num_chunks=20, seed=seed)
    reward_fn = RewardFunction(step_cost=STEP_COST)
    env = PolicyRetrievalEnv(
        simulator=sim,
        top_k=TOP_K,
        max_steps=MAX_STEPS,
        reward_fn=reward_fn,
    )

    rng = np.random.default_rng(seed)

    # -- Header --
    print("=" * 78)
    print("  Random Policy Demo")
    print(f"  {num_episodes} episodes | top_k={TOP_K} | "
          f"step_cost={STEP_COST} | max_steps={MAX_STEPS}")
    print("=" * 78)

    # -- Column headers --
    print(f"\n{'Ep':>3}  {'Steps':>5}  {'Return':>7}  "
          f"{'Decision':>8}  {'Truth':>8}  {'Correct':>7}  {'Forced':>6}")
    print("-" * 78)

    # -- Per-episode tracking --
    all_returns = []
    all_steps = []
    all_correct = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        steps = 0

        while True:
            # Valid actions: 0..len(candidates)-1 (retrieve) or K (stop).
            # The candidate list shrinks as chunks are retrieved, so we
            # must check the current count to avoid out-of-range errors.
            n_candidates = len(env._candidates)
            valid_actions = list(range(n_candidates)) + [TOP_K]
            action = valid_actions[int(rng.integers(0, len(valid_actions)))]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps += 1

            if terminated:
                break

        decision = info.get("decision", "N/A")
        ground_truth = info.get("ground_truth", "N/A")
        correct = info.get("correct", False)
        forced = info.get("forced_stop", False)

        all_returns.append(episode_return)
        all_steps.append(steps)
        all_correct.append(correct)

        print(f"{ep + 1:>3}  {steps:>5}  {episode_return:>7.2f}  "
              f"{decision:>8}  {ground_truth:>8}  "
              f"{'yes' if correct else 'no':>7}  "
              f"{'yes' if forced else 'no':>6}")

    # -- Aggregate statistics --
    print("-" * 78)
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 36)
    print(f"{'Episodes':<25} {num_episodes:>10}")
    print(f"{'Mean return':<25} {np.mean(all_returns):>10.3f}")
    print(f"{'Std return':<25} {np.std(all_returns):>10.3f}")
    print(f"{'Mean steps':<25} {np.mean(all_steps):>10.1f}")
    print(f"{'Accuracy':<25} {np.mean(all_correct):>10.1%}")
    print(f"{'Correct episodes':<25} {sum(all_correct):>10}")

    print("\n" + "=" * 78)
    print("  Done")
    print("=" * 78)


if __name__ == "__main__":
    run_random_policy()
