"""
Training script for Conservative Q-Learning on offline datasets.

Loads a replay buffer (pickle), trains a ConservativeQLAgent, logs
metrics to Tensorboard, and saves a model checkpoint.

Usage:
    python -m scripts.train_conservative_ql --dataset data/offline_buffer.pkl
    python -m scripts.train_conservative_ql --dataset data/offline_buffer.pkl \
        --epochs 200 --alpha 0.5 --lr 1e-3 --output runs/experiment_01

Monitor training:
    tensorboard --logdir runs/

Reference: EDD Use Case 4 (train Conservative Q-Learning policy).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rl.conservative_ql_agent import ConservativeQLAgent, ReplayBuffer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Conservative Q-Learning on an offline dataset.",
    )

    # Required
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to pickled ReplayBuffer (e.g. data/offline_buffer.pkl).",
    )

    # Training hyperparameters (EDD defaults)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Conservative penalty coefficient.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor.")
    parser.add_argument("--target-update-freq", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")

    # Architecture
    parser.add_argument("--state-dim", type=int, default=768)
    parser.add_argument("--num-actions", type=int, default=11)
    parser.add_argument("--hidden-dim", type=int, default=256)

    # Output
    parser.add_argument(
        "--output", type=str, default="runs/conservative_ql",
        help="Output directory for Tensorboard logs and checkpoint.",
    )

    return parser.parse_args()


def load_buffer(path: str) -> ReplayBuffer:
    """Load a pickled ReplayBuffer from disk."""
    with open(path, "rb") as f:
        buf = pickle.load(f)
    if not isinstance(buf, ReplayBuffer):
        raise TypeError(
            f"Expected ReplayBuffer, got {type(buf).__name__}. "
            f"Ensure the dataset was saved with pickle.dump(buffer, f)."
        )
    return buf


def main() -> None:
    args = parse_args()

    # -- Set seeds for reproducibility --
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -- Load dataset --
    print("=" * 70)
    print("  Conservative Q-Learning Training")
    print("=" * 70)
    print(f"\n  Dataset:    {args.dataset}")
    print(f"  Output:     {args.output}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Alpha:      {args.alpha}")
    print(f"  LR:         {args.lr}")
    print(f"  Gamma:      {args.gamma}")
    print(f"  Seed:       {args.seed}")

    buf = load_buffer(args.dataset)
    print(f"\n  Buffer loaded: {len(buf)} transitions")

    if len(buf) < args.batch_size:
        print(f"  WARNING: buffer ({len(buf)}) smaller than batch_size "
              f"({args.batch_size}), reducing batch_size to {len(buf)}")
        args.batch_size = len(buf)

    # -- Initialize Tensorboard --
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        print(f"  Tensorboard: {output_dir / 'tb'}")
        print(f"    (run: tensorboard --logdir {args.output})")
    except ImportError:
        print("  Tensorboard: not available (pip install tensorboard)")
        print("    Training will proceed without TB logging.")

    # -- Initialize agent --
    agent = ConservativeQLAgent(
        state_dim=args.state_dim,
        num_actions=args.num_actions,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.alpha,
        target_update_freq=args.target_update_freq,
        grad_clip=args.grad_clip,
    )
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    print(f"  Q-Network:  {total_params:,} params")

    # -- Train --
    print("\n" + "=" * 70)
    print("  Training")
    print("=" * 70 + "\n")

    start_time = time.time()
    metrics = agent.train(
        buf,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=args.log_every,
        writer=writer,
    )
    elapsed = time.time() - start_time

    # -- Summary --
    first_loss = np.mean(metrics["loss_history"][:3])
    last_loss = np.mean(metrics["loss_history"][-3:])
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Loss: {first_loss:.4f} -> {last_loss:.4f}")

    # -- Save checkpoint --
    checkpoint_path = str(output_dir / "checkpoint.pt")
    agent.save(checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")

    # -- Log hyperparameters to Tensorboard --
    if writer is not None:
        writer.add_hparams(
            {
                "lr": args.lr,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "hidden_dim": args.hidden_dim,
                "grad_clip": args.grad_clip,
            },
            {
                "hparam/final_loss": last_loss,
                "hparam/final_td_loss": metrics["td_loss_history"][-1],
                "hparam/final_q_mean": metrics["q_values_mean_history"][-1],
            },
        )
        writer.close()
        print(f"  Tensorboard logs written to: {output_dir / 'tb'}")

    print("\n" + "=" * 70)
    print("  Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
