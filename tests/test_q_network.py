"""
Tests for QNetwork -- verifies output shape, gradient flow, target copy,
custom parameterization, and parameter count.

Run with:
    python -m tests.test_q_network
"""

from __future__ import annotations

import torch

from rl.q_network import QNetwork


def _run_tests() -> None:
    """Verify the Q-network."""

    print("=" * 70)
    print("  QNetwork -- Tests")
    print("=" * 70)

    # --- 1. Output shape: single state ---
    print("\n[Test 1] Single state forward pass")
    net = QNetwork(state_dim=768, num_actions=11)
    state = torch.randn(768)
    q_values = net(state)
    assert q_values.shape == (11,), f"Expected (11,), got {q_values.shape}"
    print(f"  Input:  {state.shape}")
    print(f"  Output: {q_values.shape}")
    print(f"  Q-values: {q_values.detach().numpy().round(3)}")
    print("  PASSED")

    # --- 2. Output shape: batch ---
    print("\n[Test 2] Batch forward pass")
    batch = torch.randn(32, 768)
    q_batch = net(batch)
    assert q_batch.shape == (32, 11), f"Expected (32, 11), got {q_batch.shape}"
    print(f"  Input:  {batch.shape}")
    print(f"  Output: {q_batch.shape}")
    print("  PASSED")

    # --- 3. Gradient flow ---
    print("\n[Test 3] Gradient flow through all layers")
    net.zero_grad()
    state = torch.randn(768)
    q_values = net(state)
    loss = q_values.sum()
    loss.backward()

    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
        print(f"  {name:<10s} grad norm: {param.grad.norm():.6f}")
    print("  PASSED")

    # --- 4. Target network copy ---
    print("\n[Test 4] Target network copy (independent weights)")
    main_net = QNetwork()
    target_net = main_net.copy()

    # Weights should be identical after copy
    state = torch.randn(768)
    q_main = main_net(state)
    q_target = target_net(state)
    assert torch.allclose(q_main, q_target), "Copy should produce identical outputs"
    print(f"  Main Q:   {q_main.detach().numpy().round(3)}")
    print(f"  Target Q: {q_target.detach().numpy().round(3)}")

    # Update main network — target should NOT change
    optimizer = torch.optim.Adam(main_net.parameters(), lr=0.1)
    loss = main_net(state).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    q_main_after = main_net(state)
    q_target_after = target_net(state)
    assert not torch.allclose(q_main_after, q_target_after), (
        "Target should be independent of main after update"
    )
    print(f"  After main update:")
    print(f"    Main Q:   {q_main_after.detach().numpy().round(3)}")
    print(f"    Target Q: {q_target_after.detach().numpy().round(3)} (unchanged)")
    print("  PASSED")

    # --- 5. Custom dimensions ---
    print("\n[Test 5] Custom state_dim and num_actions")
    custom_net = QNetwork(state_dim=128, num_actions=5, hidden_dim=64)
    state = torch.randn(128)
    q_values = custom_net(state)
    assert q_values.shape == (5,), f"Expected (5,), got {q_values.shape}"
    print(f"  state_dim=128, num_actions=5, hidden_dim=64")
    print(f"  Output: {q_values.shape}")
    print("  PASSED")

    # --- 6. Parameter count ---
    print("\n[Test 6] Parameter count")
    net = QNetwork(state_dim=768, num_actions=11, hidden_dim=256)
    # fc1: 768*256 + 256 = 196,864
    # fc2: 256*256 + 256 = 65,792
    # fc3: 256*11  + 11  = 2,827
    # Total: 265,483
    expected = (768 * 256 + 256) + (256 * 256 + 256) + (256 * 11 + 11)
    actual = sum(p.numel() for p in net.parameters())
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"  fc1: {768 * 256 + 256:,} params")
    print(f"  fc2: {256 * 256 + 256:,} params")
    print(f"  fc3: {256 * 11 + 11:,} params")
    print(f"  Total: {actual:,} (expected {expected:,})")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
