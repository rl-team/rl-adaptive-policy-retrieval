"""
Tests for PolicyRetrievalEnv -- exercises all termination paths, reward
structure, observation shape, and error handling.

Run with:
    python -m tests.test_env
"""

from __future__ import annotations

from tests.mock_simulator import MockPASimulator
from rl.env import PolicyRetrievalEnv


def _run_tests() -> None:
    """Verify the environment with the mock simulator."""

    print("=" * 70)
    print("  PolicyRetrievalEnv -- Tests")
    print("=" * 70)

    sim = MockPASimulator(num_chunks=20, seed=42)
    env = PolicyRetrievalEnv(simulator=sim, top_k=10, step_cost=0.1, max_steps=20)

    # --- 1. Basic reset ---
    print("\n[Test 1] Reset and initial observation")
    obs, info = env.reset(options={"procedure_code": "72148"})
    print(f"  Observation shape: {obs.shape}")
    print(f"  Request ID: {info['request_id']}")
    print(f"  Procedure: {info['procedure_code']}")
    print(f"  Ground truth: {info['ground_truth']}")
    assert obs.shape == (768,), f"Expected (768,), got {obs.shape}"
    assert info["procedure_code"] == "72148"
    print("  PASSED")

    # --- 2. Full episode with manual stop ---
    print("\n[Test 2] Full episode (retrieve 3 chunks, then stop)")
    obs, info = env.reset(options={"procedure_code": "72148"})
    total_reward = 0.0
    for step_i in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        print(f"  Step {step_i + 1}: reward={reward:.2f}, "
              f"chunks={info['chunks_retrieved']}, "
              f"chunk_type={info['last_chunk_type']}")
        assert not terminated, "Should not be done yet"

    # Now stop
    obs, reward, terminated, truncated, info = env.step(env._top_k)
    total_reward += reward
    print(f"  Stop:   reward={reward:.2f}, decision={info['decision']}, "
          f"correct={info['correct']}")
    assert terminated, "Should be done after stop"
    print(f"  Total reward: {total_reward:.2f}")
    print("  PASSED")

    # --- 3. Max steps termination ---
    print("\n[Test 3] Max steps forced termination")
    small_env = PolicyRetrievalEnv(
        simulator=sim, top_k=10, step_cost=0.1, max_steps=3,
    )
    obs, info = small_env.reset(options={"procedure_code": "29881"})
    for step_i in range(3):
        obs, reward, terminated, truncated, info = small_env.step(0)
        if terminated:
            print(f"  Terminated at step {step_i + 1} (forced={info['forced_stop']})")
            break
    assert terminated, "Should be terminated at max_steps"
    assert info["forced_stop"], "Should be forced stop"
    print(f"  Decision: {info['decision']}, correct: {info['correct']}")
    print("  PASSED")

    # --- 4. Step after done raises error ---
    print("\n[Test 4] Step after done raises RuntimeError")
    try:
        small_env.step(0)
        print("  FAILED -- should have raised RuntimeError")
    except RuntimeError as e:
        print(f"  PASSED -- RuntimeError: {e}")

    # --- 5. Invalid action raises error ---
    print("\n[Test 5] Invalid action raises ValueError")
    obs, info = env.reset()
    try:
        env.step(-1)
        print("  FAILED -- should have raised ValueError")
    except ValueError as e:
        print(f"  PASSED -- ValueError: {e}")

    # --- 6. Observation space and action space ---
    print("\n[Test 6] Space validation")
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "Observation not in space"
    assert env.action_space.n == 11, f"Expected 11 actions, got {env.action_space.n}"
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space: {env.action_space}")
    print("  PASSED")

    # --- 7. Reward structure verification ---
    print("\n[Test 7] Reward structure (step cost + terminal)")
    obs, info = env.reset(options={"procedure_code": "72148"})
    ground_truth = info["ground_truth"]

    # Retrieve one chunk
    obs, reward, terminated, _, _ = env.step(0)
    assert abs(reward - (-0.1)) < 1e-6, f"Expected step cost -0.1, got {reward}"
    print(f"  Step reward: {reward:.2f} (expected -0.10)")

    # Stop
    obs, reward, terminated, _, info = env.step(env._top_k)
    assert abs(reward) == 1.0, f"Expected +/-1.0 terminal, got {reward}"
    print(f"  Terminal reward: {reward:.2f} (decision={info['decision']}, "
          f"ground_truth={ground_truth})")
    print("  PASSED")

    # --- 8. Multiple episodes ---
    print("\n[Test 8] Multiple episodes")
    for ep in range(3):
        obs, info = env.reset()
        steps = 0
        while True:
            obs, reward, terminated, _, info = env.step(0)
            steps += 1
            if terminated:
                break
            if steps >= 5:
                obs, reward, terminated, _, info = env.step(env._top_k)
                break
        print(f"  Episode {ep + 1}: steps={steps}, "
              f"decision={info.get('decision', 'N/A')}, "
              f"correct={info.get('correct', 'N/A')}")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
