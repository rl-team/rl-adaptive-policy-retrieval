"""
Tests for RewardFunction -- verifies step cost, terminal correctness,
and custom parameterization.

Run with:
    python -m tests.test_reward
"""

from __future__ import annotations

from rl.reward import RewardFunction


def _run_tests() -> None:
    """Verify the reward function."""

    print("=" * 70)
    print("  RewardFunction -- Tests")
    print("=" * 70)

    rf = RewardFunction()  # default: step_cost=0.1

    # --- 1. Step reward ---
    print("\n[Test 1] Step reward (default step_cost=0.1)")
    sr = rf.step_reward()
    assert abs(sr - (-0.1)) < 1e-9, f"Expected -0.1, got {sr}"
    print(f"  step_reward() = {sr:.2f}")
    print("  PASSED")

    # --- 2. Terminal reward: correct decision ---
    print("\n[Test 2] Terminal reward: correct decision")
    tr = rf.terminal_reward("approve", "approve")
    assert tr == 1.0, f"Expected 1.0, got {tr}"
    print(f"  terminal_reward('approve', 'approve') = {tr:.1f}")
    tr2 = rf.terminal_reward("deny", "deny")
    assert tr2 == 1.0, f"Expected 1.0, got {tr2}"
    print(f"  terminal_reward('deny', 'deny') = {tr2:.1f}")
    tr3 = rf.terminal_reward("pend", "pend")
    assert tr3 == 1.0, f"Expected 1.0, got {tr3}"
    print(f"  terminal_reward('pend', 'pend') = {tr3:.1f}")
    print("  PASSED")

    # --- 3. Terminal reward: incorrect decision ---
    print("\n[Test 3] Terminal reward: incorrect decision")
    tr4 = rf.terminal_reward("approve", "deny")
    assert tr4 == -1.0, f"Expected -1.0, got {tr4}"
    print(f"  terminal_reward('approve', 'deny') = {tr4:.1f}")
    tr5 = rf.terminal_reward("pend", "approve")
    assert tr5 == -1.0, f"Expected -1.0, got {tr5}"
    print(f"  terminal_reward('pend', 'approve') = {tr5:.1f}")
    print("  PASSED")

    # --- 4. Custom step_cost ---
    print("\n[Test 4] Custom step_cost=0.05")
    rf2 = RewardFunction(step_cost=0.05)
    sr2 = rf2.step_reward()
    assert abs(sr2 - (-0.05)) < 1e-9, f"Expected -0.05, got {sr2}"
    assert rf2.step_cost == 0.05, f"Expected 0.05, got {rf2.step_cost}"
    print(f"  step_reward() = {sr2:.2f}")
    print(f"  step_cost = {rf2.step_cost}")
    print("  PASSED")

    # --- 5. Custom step_cost=0.2 ---
    print("\n[Test 5] Custom step_cost=0.2")
    rf3 = RewardFunction(step_cost=0.2)
    sr3 = rf3.step_reward()
    assert abs(sr3 - (-0.2)) < 1e-9, f"Expected -0.2, got {sr3}"
    print(f"  step_reward() = {sr3:.2f}")
    print("  PASSED")

    # --- 6. Integration: env uses RewardFunction ---
    print("\n[Test 6] Integration with PolicyRetrievalEnv")
    from tests.mock_simulator import MockPASimulator
    from rl.env import PolicyRetrievalEnv

    sim = MockPASimulator(num_chunks=20, seed=42)
    custom_rf = RewardFunction(step_cost=0.05)
    env = PolicyRetrievalEnv(
        simulator=sim, top_k=10, max_steps=20, reward_fn=custom_rf,
    )

    obs, info = env.reset(options={"procedure_code": "72148"})

    # Step reward should use the custom step_cost
    obs, reward, terminated, _, _ = env.step(0)
    assert abs(reward - (-0.05)) < 1e-9, f"Expected -0.05, got {reward}"
    print(f"  Step reward with custom RewardFunction: {reward:.2f}")

    # Terminal reward should still be +/-1.0
    obs, reward, terminated, _, info = env.step(env._top_k)
    assert abs(reward) == 1.0, f"Expected +/-1.0, got {reward}"
    print(f"  Terminal reward: {reward:.1f} (decision={info['decision']})")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
