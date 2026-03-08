"""
Train Behavioral Cloning baseline on offline dataset.

Learns a supervised policy from (state, action) pairs in the replay
buffer, ignoring rewards.  Produces a checkpoint that can be loaded
by ``BehavioralCloningPolicy.load()`` and evaluated via
``evaluate_agent.py`` or ``run_baseline_episode()``.

Usage:
    python -m scripts.train_bc --dataset data/offline_buffer_2k.pkl
    python -m scripts.train_bc --dataset data/offline_buffer_2k.pkl \
        --epochs 200 --lr 1e-3 --output runs/bc_2k

Reference: Implementation Plan R36.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from baselines.bc import BehavioralCloningPolicy
from rl.conservative_ql_agent import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Behavioral Cloning baseline on offline dataset.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to pickled ReplayBuffer.",
    )
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs (default: 200).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3).")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="MLP hidden dimension (default: 256).")
    parser.add_argument("--state-dim", type=int, default=768,
                        help="State dimension (default: 768).")
    parser.add_argument("--num-actions", type=int, default=11,
                        help="Number of actions (default: 11 = top_k + stop).")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Print progress every N epochs.")
    parser.add_argument("--output", type=str, default="runs/bc_2k",
                        help="Output directory for checkpoint.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -- Set seeds for reproducibility --
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("  Behavioral Cloning Training")
    print("=" * 70)

    # Load buffer
    print(f"\n  Dataset: {args.dataset}")
    with open(args.dataset, "rb") as f:
        buf = pickle.load(f)
    if not isinstance(buf, ReplayBuffer):
        raise TypeError(f"Expected ReplayBuffer, got {type(buf).__name__}")
    print(f"  Buffer: {len(buf)} transitions")
    print(f"  Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}")

    # Create policy
    bc = BehavioralCloningPolicy(
        state_dim=args.state_dim,
        num_actions=args.num_actions,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )

    # Train
    start = time.time()
    metrics = bc.train(
        buf,
        num_epochs=args.epochs,
        batch_size=min(args.batch_size, len(buf)),
        log_every=args.log_every,
    )
    elapsed = time.time() - start

    # Summary
    loss_hist = metrics["loss_history"]
    first_loss = np.mean(loss_hist[:3])
    last_loss = np.mean(loss_hist[-3:])

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Loss: {first_loss:.4f} -> {last_loss:.4f} "
          f"({(1 - last_loss / first_loss) * 100:.0f}% reduction)")

    # Save checkpoint
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    bc.save(str(ckpt_path))
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
