"""
Tests for BehavioralCloningPolicy -- verifies training loss, action
selection, should_stop, action_prob, and checkpoint persistence.

Run with:
    python -m tests.test_bc
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from baselines.bc import BehavioralCloningPolicy
from rl.conservative_ql_agent import ReplayBuffer


def _make_synthetic_buffer(
    num_episodes: int = 20,
    steps_per_episode: int = 5,
    state_dim: int = 768,
    num_actions: int = 11,
    seed: int = 42,
) -> ReplayBuffer:
    """Create a replay buffer with simple synthetic transitions."""
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
    """Verify the Behavioral Cloning policy."""

    print("=" * 70)
    print("  BehavioralCloningPolicy -- Tests")
    print("=" * 70)

    # --- 1. Training: loss decreases ---
    print("\n[Test 1] Training: cross-entropy loss decreases over 30 epochs")
    bc = BehavioralCloningPolicy(
        state_dim=768, num_actions=11, lr=1e-3,
    )
    buf = _make_synthetic_buffer(num_episodes=50, steps_per_episode=5)
    metrics = bc.train(buf, num_epochs=30, batch_size=64, log_every=10)

    first_loss = np.mean(metrics["loss_history"][:3])
    last_loss = np.mean(metrics["loss_history"][-3:])
    assert last_loss < first_loss, (
        f"Loss should decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )
    print(f"  First 3 avg loss: {first_loss:.4f}")
    print(f"  Last 3 avg loss:  {last_loss:.4f}")
    print("  PASSED")

    # --- 2. Action selection + should_stop ---
    print("\n[Test 2] Action selection: valid_actions mask + should_stop")
    bc = BehavioralCloningPolicy(
        state_dim=8, num_actions=5, hidden_dim=16, stop_action=4,
    )
    state = np.random.randn(8).astype(np.float32)

    # select_action with candidates
    action = bc.select_action(state, candidates=[0, 1, 2])
    assert action in [0, 1, 2], f"Expected 0/1/2, got {action}"
    print(f"  Action from candidates [0,1,2]: {action}")

    # select_action with empty candidates
    action = bc.select_action(state, candidates=[])
    assert action == -1, f"Expected -1 for empty candidates, got {action}"
    print(f"  Action from empty candidates: {action}")

    # should_stop returns bool
    stops = bc.should_stop(state, history=[])
    assert isinstance(stops, bool), f"Expected bool, got {type(stops)}"
    print(f"  should_stop: {stops}")

    # action_prob returns valid probability
    prob = bc.action_prob(state, action=0, candidates=[0, 1, 2], history=[])
    assert 0.0 <= prob <= 1.0, f"Expected probability in [0,1], got {prob}"
    print(f"  action_prob(0, cands=[0,1,2]): {prob:.4f}")

    # action_prob for stop action
    stop_prob = bc.action_prob(state, action=-1, candidates=[0, 1, 2], history=[])
    assert 0.0 <= stop_prob <= 1.0, f"Stop prob out of range: {stop_prob}"
    print(f"  action_prob(stop): {stop_prob:.4f}")
    print("  PASSED")

    # --- 3. Save / load checkpoint ---
    print("\n[Test 3] Save and load checkpoint")
    bc = BehavioralCloningPolicy(
        state_dim=32, num_actions=5, hidden_dim=16, stop_action=4,
    )
    # Train a little so weights are non-random
    buf = _make_synthetic_buffer(
        num_episodes=20, steps_per_episode=3, state_dim=32, num_actions=5,
    )
    bc.train(buf, num_epochs=5, batch_size=16, log_every=5)

    state = np.random.randn(32).astype(np.float32)
    action_before = bc.select_action(state, candidates=[0, 1, 2, 3])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "models", "bc_test.pt")
        bc.save(path)
        assert os.path.exists(path), f"Checkpoint not found: {path}"
        print(f"  Saved to: {path}")

        loaded = BehavioralCloningPolicy.load(path)
        action_after = loaded.select_action(state, candidates=[0, 1, 2, 3])
        assert action_before == action_after, (
            f"Action mismatch: {action_before} vs {action_after}"
        )
        assert loaded._stop_action == 4, (
            f"Expected stop_action=4, got {loaded._stop_action}"
        )
        print(f"  Action before: {action_before}, after: {action_after}")
        print(f"  stop_action preserved: {loaded._stop_action}")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
