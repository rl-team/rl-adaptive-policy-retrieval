"""
End-to-end test: train Conservative Q-Learning on mock data.

Verifies the full pipeline:
    1. Collect 10 episodes using a random behavior policy
    2. Fill a ReplayBuffer with collected transitions
    3. Train a ConservativeQLAgent for 20 epochs
    4. Evaluate the trained agent vs. the random baseline

This is the R10 deliverable: proving the entire training loop works
end-to-end with mock data before connecting the real simulator.

Run with:
    python -m scripts.train_conservative_ql_mock
"""

from __future__ import annotations

import numpy as np

from tests.mock_simulator import MockPASimulator
from rl.env import PolicyRetrievalEnv
from rl.reward import RewardFunction
from rl.conservative_ql_agent import ConservativeQLAgent, ReplayBuffer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_COLLECT_EPISODES = 10   # episodes for offline dataset
NUM_EVAL_EPISODES = 10      # episodes for evaluation
NUM_TRAIN_EPOCHS = 20
BATCH_SIZE = 32             # small batch for tiny dataset
TOP_K = 10
MAX_STEPS = 20
STEP_COST = 0.1
SEED = 42


def collect_episodes(
    env: PolicyRetrievalEnv,
    num_episodes: int,
    seed: int,
) -> ReplayBuffer:
    """Collect episodes using a uniform random behavior policy.

    Returns a ReplayBuffer filled with (s, a, r, s', done) transitions
    from random agent rollouts. This simulates loading an offline dataset
    collected by behavior policies (EDD Use Case 4, steps 2-3).
    """
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer()
    total_transitions = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []

        while True:
            # Random behavior policy (same as scripts/random_policy_demo.py)
            n_candidates = len(env._candidates)
            valid_actions = list(range(n_candidates)) + [TOP_K]
            action = valid_actions[int(rng.integers(0, len(valid_actions)))]

            next_obs, reward, terminated, truncated, info = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_obs)
            dones.append(terminated)

            obs = next_obs

            if terminated:
                break

        buf.add_episode(states, actions, rewards, next_states, dones)
        total_transitions += len(states)

    return buf, total_transitions


def evaluate_agent(
    env: PolicyRetrievalEnv,
    agent: ConservativeQLAgent,
    num_episodes: int,
    label: str,
) -> dict:
    """Run the agent greedily for evaluation.

    Returns dict with mean_return, accuracy, mean_steps.
    """
    returns, correct_list, steps_list = [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        steps = 0

        while True:
            n_candidates = len(env._candidates)
            valid_actions = list(range(n_candidates)) + [TOP_K]
            action = agent.select_action(obs, valid_actions=valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps += 1

            if terminated:
                break

        returns.append(episode_return)
        correct_list.append(info.get("correct", False))
        steps_list.append(steps)

    results = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "accuracy": float(np.mean(correct_list)),
        "mean_steps": float(np.mean(steps_list)),
    }

    print(f"\n  {label}:")
    print(f"    Mean return: {results['mean_return']:.3f} "
          f"(std {results['std_return']:.3f})")
    print(f"    Accuracy:    {results['accuracy']:.1%}")
    print(f"    Mean steps:  {results['mean_steps']:.1f}")

    return results


def evaluate_random(
    env: PolicyRetrievalEnv,
    num_episodes: int,
    seed: int,
) -> dict:
    """Run a random policy for baseline comparison."""
    rng = np.random.default_rng(seed)
    returns, correct_list, steps_list = [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        steps = 0

        while True:
            n_candidates = len(env._candidates)
            valid_actions = list(range(n_candidates)) + [TOP_K]
            action = valid_actions[int(rng.integers(0, len(valid_actions)))]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps += 1

            if terminated:
                break

        returns.append(episode_return)
        correct_list.append(info.get("correct", False))
        steps_list.append(steps)

    results = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "accuracy": float(np.mean(correct_list)),
        "mean_steps": float(np.mean(steps_list)),
    }

    print(f"\n  Random baseline:")
    print(f"    Mean return: {results['mean_return']:.3f} "
          f"(std {results['std_return']:.3f})")
    print(f"    Accuracy:    {results['accuracy']:.1%}")
    print(f"    Mean steps:  {results['mean_steps']:.1f}")

    return results


def main() -> None:
    """Full pipeline: collect, train, evaluate."""

    sim = MockPASimulator(num_chunks=20, seed=SEED)
    reward_fn = RewardFunction(step_cost=STEP_COST)
    env = PolicyRetrievalEnv(
        simulator=sim,
        top_k=TOP_K,
        max_steps=MAX_STEPS,
        reward_fn=reward_fn,
    )

    # -- Phase 1: Collect offline dataset --
    print("=" * 70)
    print("  Phase 1: Collecting offline dataset")
    print("=" * 70)

    buf, n_transitions = collect_episodes(env, NUM_COLLECT_EPISODES, SEED)
    print(f"  Episodes collected: {NUM_COLLECT_EPISODES}")
    print(f"  Total transitions:  {n_transitions}")
    print(f"  Buffer size:        {len(buf)}")

    # -- Phase 2: Train Conservative Q-Learning --
    print("\n" + "=" * 70)
    print("  Phase 2: Training Conservative Q-Learning")
    print(f"  {NUM_TRAIN_EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"alpha=1.0, lr=3e-4")
    print("=" * 70 + "\n")

    agent = ConservativeQLAgent(
        state_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        alpha=1.0,
        lr=3e-4,
    )

    metrics = agent.train(
        buf,
        num_epochs=NUM_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        log_every=5,
    )

    first_loss = np.mean(metrics["loss_history"][:3])
    last_loss = np.mean(metrics["loss_history"][-3:])
    print(f"\n  Loss: {first_loss:.4f} -> {last_loss:.4f} "
          f"({'decreased' if last_loss < first_loss else 'WARNING: did not decrease'})")

    # -- Phase 3: Evaluate --
    print("\n" + "=" * 70)
    print("  Phase 3: Evaluation")
    print("=" * 70)

    trained_results = evaluate_agent(
        env, agent, NUM_EVAL_EPISODES, "Trained Conservative Q-Learning",
    )
    random_results = evaluate_random(env, NUM_EVAL_EPISODES, seed=99)

    # -- Summary --
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"\n  {'Metric':<20} {'Conservative QL':>16} {'Random':>10}")
    print("  " + "-" * 48)
    print(f"  {'Mean return':<20} "
          f"{trained_results['mean_return']:>16.3f} "
          f"{random_results['mean_return']:>10.3f}")
    print(f"  {'Accuracy':<20} "
          f"{trained_results['accuracy']:>15.1%} "
          f"{random_results['accuracy']:>9.1%}")
    print(f"  {'Mean steps':<20} "
          f"{trained_results['mean_steps']:>16.1f} "
          f"{random_results['mean_steps']:>10.1f}")

    print("\n" + "=" * 70)
    print("  Pipeline verified: collect -> train -> evaluate")
    print("=" * 70)


if __name__ == "__main__":
    main()
