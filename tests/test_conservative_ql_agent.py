"""
Tests for ConservativeQLAgent -- verifies replay buffer, Conservative Q-Learning loss,
training convergence, action selection, and checkpointing.

Run with:
    python -m tests.test_conservative_ql_agent
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from rl.conservative_ql_agent import ConservativeQLAgent, ReplayBuffer


def _make_synthetic_buffer(
    num_episodes: int = 20,
    steps_per_episode: int = 5,
    state_dim: int = 768,
    num_actions: int = 11,
    seed: int = 42,
) -> ReplayBuffer:
    """Create a replay buffer with simple synthetic transitions.

    Each episode: random states, random actions, small negative step
    rewards, and a +/-1.0 terminal reward. This is enough to verify
    that training mechanics work (loss computes, gradients flow,
    loss decreases).
    """
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer()

    for _ in range(num_episodes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in range(steps_per_episode):
            s = rng.standard_normal(state_dim).astype(np.float32)
            a = int(rng.integers(0, num_actions))
            ns = rng.standard_normal(state_dim).astype(np.float32)
            is_terminal = (t == steps_per_episode - 1)
            r = (1.0 if rng.random() > 0.5 else -1.0) if is_terminal else -0.1

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(is_terminal)

        buf.add_episode(states, actions, rewards, next_states, dones)

    return buf


def _run_tests() -> None:
    """Verify the Conservative Q-Learning agent."""

    print("=" * 70)
    print("  ConservativeQLAgent -- Tests")
    print("=" * 70)

    # --- 1. ReplayBuffer ---
    print("\n[Test 1] ReplayBuffer: add, sample, length")
    buf = ReplayBuffer(capacity=50)
    states = [np.zeros(4, dtype=np.float32) for _ in range(3)]
    actions = [0, 1, 2]
    rewards = [-0.1, -0.1, 1.0]
    next_states = [np.ones(4, dtype=np.float32) for _ in range(3)]
    dones = [False, False, True]

    buf.add_episode(states, actions, rewards, next_states, dones)
    assert len(buf) == 3, f"Expected 3, got {len(buf)}"

    batch = buf.sample(2)
    assert batch["states"].shape == (2, 4), f"Got {batch['states'].shape}"
    assert batch["actions"].shape == (2,), f"Got {batch['actions'].shape}"
    assert batch["rewards"].shape == (2,), f"Got {batch['rewards'].shape}"
    print(f"  Buffer length: {len(buf)}")
    print(f"  Batch shapes: states={batch['states'].shape}, "
          f"actions={batch['actions'].shape}")

    # Test capacity eviction
    for _ in range(20):
        buf.add_episode(states, actions, rewards, next_states, dones)
    assert len(buf) == 50, f"Expected 50 (capacity), got {len(buf)}"
    print(f"  After 63 transitions added (cap=50): len={len(buf)}")
    print("  PASSED")

    # --- 2. Conservative Q-Learning loss components ---
    print("\n[Test 2] Conservative Q-Learning loss: TD loss > 0, conservative penalty computed")
    agent = ConservativeQLAgent(
        state_dim=768, num_actions=11, lr=3e-4, alpha=1.0,
    )
    syn_buf = _make_synthetic_buffer(num_episodes=10, steps_per_episode=5)
    batch = syn_buf.sample(32)
    loss, td_loss, cql_penalty, q_mean = agent._train_step(batch)

    assert loss > 0, f"Expected positive loss, got {loss}"
    assert td_loss >= 0, f"TD loss should be non-negative, got {td_loss}"
    print(f"  Total loss: {loss:.4f}")
    print(f"  TD loss:    {td_loss:.4f}")
    print(f"  Conservative penalty: {cql_penalty:.4f}")
    print(f"  Q mean:     {q_mean:.4f}")
    print("  PASSED")

    # --- 3. Training convergence ---
    print("\n[Test 3] Training: loss decreases over 20 epochs")
    agent = ConservativeQLAgent(
        state_dim=768, num_actions=11, lr=1e-3, alpha=0.5,
    )
    syn_buf = _make_synthetic_buffer(num_episodes=50, steps_per_episode=5)
    metrics = agent.train(
        syn_buf, num_epochs=20, batch_size=64, log_every=5,
    )

    first_loss = np.mean(metrics["loss_history"][:3])
    last_loss = np.mean(metrics["loss_history"][-3:])
    assert last_loss < first_loss, (
        f"Loss should decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )
    print(f"  First 3 avg loss: {first_loss:.4f}")
    print(f"  Last 3 avg loss:  {last_loss:.4f}")
    print("  PASSED")

    # --- 4. Action selection ---
    print("\n[Test 4] Action selection: greedy + valid_actions mask")
    agent = ConservativeQLAgent(state_dim=8, num_actions=5, hidden_dim=16)
    state = np.random.randn(8).astype(np.float32)

    # Greedy (all actions valid)
    action = agent.select_action(state)
    assert 0 <= action < 5, f"Action out of range: {action}"
    print(f"  Greedy action (all valid): {action}")

    # With valid_actions mask
    action = agent.select_action(state, valid_actions=[1, 3])
    assert action in [1, 3], f"Expected 1 or 3, got {action}"
    print(f"  Masked action (valid=[1,3]): {action}")

    # Single valid action
    action = agent.select_action(state, valid_actions=[4])
    assert action == 4, f"Expected 4, got {action}"
    print(f"  Single valid action: {action}")
    print("  PASSED")

    # --- 5. Save / load checkpoint ---
    print("\n[Test 5] Save and load checkpoint")
    agent = ConservativeQLAgent(
        state_dim=32, num_actions=5, hidden_dim=16, alpha=2.0,
    )
    state = np.random.randn(32).astype(np.float32)
    q_before = agent.select_action(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "models", "test_checkpoint.pt")
        agent.save(path)
        assert os.path.exists(path), f"Checkpoint not found: {path}"
        print(f"  Saved to: {path}")

        loaded = ConservativeQLAgent.load(path)
        q_after = loaded.select_action(state)
        assert q_before == q_after, (
            f"Action mismatch: {q_before} vs {q_after}"
        )
        assert loaded.alpha == 2.0, f"Expected alpha=2.0, got {loaded.alpha}"
        print(f"  Loaded alpha: {loaded.alpha}")
        print(f"  Action before save: {q_before}, after load: {q_after}")
    print("  PASSED")

    # --- 6. Target network sync ---
    print("\n[Test 6] Target network syncs every N epochs")
    agent = ConservativeQLAgent(
        state_dim=32, num_actions=5, hidden_dim=16,
        target_update_freq=5,
    )
    syn_buf = _make_synthetic_buffer(
        num_episodes=20, steps_per_episode=3, state_dim=32, num_actions=5,
    )

    # Before training, main and target should match (from __init__)
    state_t = torch.randn(32)
    q_main = agent.q_network(state_t).detach()
    q_target = agent.target_network(state_t).detach()
    assert torch.allclose(q_main, q_target), "Should match before training"
    print(f"  Before training: main==target ✓")

    # Train 5 epochs (target should sync at epoch 5)
    agent.train(syn_buf, num_epochs=5, batch_size=16, log_every=5)

    # After sync at epoch 5, they should match again
    q_main = agent.q_network(state_t).detach()
    q_target = agent.target_network(state_t).detach()
    assert torch.allclose(q_main, q_target), (
        "Should match after sync at epoch 5"
    )
    print(f"  After epoch 5 (sync): main==target ✓")

    # Train 3 more epochs (no sync at 6, 7, 8)
    agent.train(syn_buf, num_epochs=3, batch_size=16, log_every=10)
    q_main = agent.q_network(state_t).detach()
    q_target = agent.target_network(state_t).detach()
    assert not torch.allclose(q_main, q_target), (
        "Should differ between syncs"
    )
    print(f"  After 3 more epochs (no sync): main!=target ✓")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
