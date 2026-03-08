"""
Training script for Implicit Q-Learning on offline datasets.

Loads a replay buffer (pickle), trains an IQLAgent, logs metrics to
Tensorboard, and saves a model checkpoint.

Usage:
    python -m scripts.train_iql --dataset data/offline_buffer_2k.pkl
    python -m scripts.train_iql --dataset data/offline_buffer_2k.pkl \
        --epochs 200 --tau 0.7 --beta 3.0 --lr 3e-4 --output runs/iql_2k

Monitor training:
    tensorboard --logdir runs/

Reference: EDD Use Case 9 (train IQL policy); Kostrikov et al. 2021.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rl.iql_agent import IQLAgent
from rl.conservative_ql_agent import ReplayBuffer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Implicit Q-Learning on an offline dataset.",
    )

    # Required
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to pickled ReplayBuffer (e.g. data/offline_buffer_2k.pkl).",
    )

    # Training hyperparameters (v3 config, empirically tuned for PA retrieval env)
    # Kostrikov et al. defaults (tau=0.7, beta=3.0, epochs=200) underperform here;
    # v3 (tau=0.9, beta=10.0, lr=1e-3, 1000 epochs) achieves 78.5% vs 62% baseline.
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Expectile parameter for V-network (0.5 = mean, "
                             "higher = optimistic upper-tail). v3: 0.9.")
    parser.add_argument("--beta", type=float, default=10.0,
                        help="Inverse temperature for advantage-weighted "
                             "policy extraction. v3: 10.0.")
    parser.add_argument("--target-update-freq", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")

    # Architecture
    parser.add_argument("--state-dim", type=int, default=768)
    parser.add_argument("--num-actions", type=int, default=11)
    parser.add_argument("--hidden-dim", type=int, default=256)

    # Output
    parser.add_argument(
        "--output", type=str, default="runs/iql",
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
    print("  Implicit Q-Learning Training")
    print("=" * 70)
    print(f"\n  Dataset:    {args.dataset}")
    print(f"  Output:     {args.output}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Tau:        {args.tau}")
    print(f"  Beta:       {args.beta}")
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
    agent = IQLAgent(
        state_dim=args.state_dim,
        num_actions=args.num_actions,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
        target_update_freq=args.target_update_freq,
        grad_clip=args.grad_clip,
    )

    total_params = sum(
        p.numel()
        for net in [agent.q_network, agent.v_network, agent.policy_network]
        for p in net.parameters()
    )
    print(f"  Networks:   {total_params:,} params "
          f"(Q + V + Policy)")

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
    first_q = np.mean(metrics["q_loss_history"][:3])
    last_q = np.mean(metrics["q_loss_history"][-3:])
    first_v = np.mean(metrics["v_loss_history"][:3])
    last_v = np.mean(metrics["v_loss_history"][-3:])
    first_pi = np.mean(metrics["policy_loss_history"][:3])
    last_pi = np.mean(metrics["policy_loss_history"][-3:])

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Q-loss:      {first_q:.4f} -> {last_q:.4f}")
    print(f"  V-loss:      {first_v:.4f} -> {last_v:.4f}")
    print(f"  Policy-loss: {first_pi:.4f} -> {last_pi:.4f}")

    # -- Save checkpoint --
    checkpoint_path = str(output_dir / "checkpoint.pt")
    agent.save(checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")

    # -- Log hyperparameters to Tensorboard --
    if writer is not None:
        writer.add_hparams(
            {
                "lr": args.lr,
                "tau": args.tau,
                "beta": args.beta,
                "gamma": args.gamma,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "hidden_dim": args.hidden_dim,
                "grad_clip": args.grad_clip,
            },
            {
                "hparam/final_q_loss": last_q,
                "hparam/final_v_loss": last_v,
                "hparam/final_pi_loss": last_pi,
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
