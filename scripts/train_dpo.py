"""
Training script for Direct Preference Optimization on offline datasets.

Supports two modes:
  --mode transition  (default) — per-step DPO, analogous to token-level DPO
  --mode trajectory  — trajectory-level DPO with length normalisation

Usage:
    # Transition-level DPO (recommended)
    python -m scripts.train_dpo --dataset data/offline_buffer_2k.pkl

    # Trajectory-level DPO (for comparison)
    python -m scripts.train_dpo --dataset data/offline_buffer_2k.pkl \
        --mode trajectory

    # Sweep beta
    python -m scripts.train_dpo --dataset data/offline_buffer_2k.pkl \
        --beta 1.0 --output runs/dpo_2k_beta1

Monitor training:
    tensorboard --logdir runs/

Reference: R52 DPO stretch plan in main-doc.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from rl.dpo_agent import (
    DPOAgent,
    PreferenceDataset,
    TransitionPreferenceDataset,
)
from rl.conservative_ql_agent import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DPO agent on offline preference data.",
    )

    # Required
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to pickled ReplayBuffer.",
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, default="transition",
        choices=["transition", "trajectory"],
        help="DPO mode: 'transition' for per-step (recommended), "
             "'trajectory' for full-episode (default: transition).",
    )

    # BC warmup
    parser.add_argument("--bc-epochs", type=int, default=200,
                        help="Behavioral cloning warmup epochs.")

    # DPO training
    parser.add_argument("--epochs", type=int, default=500,
                        help="DPO training epochs.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Preference pairs per batch.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate.")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="DPO temperature (KL constraint strength).")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for noisy preferences.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Preference construction
    parser.add_argument("--preference-margin", type=float, default=0.5,
                        help="Min return gap for valid preference pair.")
    parser.add_argument("--max-pairs", type=int, default=100000,
                        help="Max preference pairs to sample.")

    # Architecture
    parser.add_argument("--state-dim", type=int, default=768)
    parser.add_argument("--num-actions", type=int, default=11)
    parser.add_argument("--hidden-dim", type=int, default=256)

    # Output
    parser.add_argument("--output", type=str, default="runs/dpo_2k")

    return parser.parse_args()


def load_buffer(path: str) -> ReplayBuffer:
    with open(path, "rb") as f:
        buf = pickle.load(f)
    if not isinstance(buf, ReplayBuffer):
        raise TypeError(
            f"Expected ReplayBuffer, got {type(buf).__name__}."
        )
    return buf


def main() -> None:
    args = parse_args()

    # -- Seeds --
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -- Load dataset --
    print("=" * 70)
    print(f"  Direct Preference Optimization Training ({args.mode}-level)")
    print("=" * 70)
    print(f"\n  Dataset:    {args.dataset}")
    print(f"  Mode:       {args.mode}")
    print(f"  Output:     {args.output}")
    print(f"  BC warmup:  {args.bc_epochs} epochs")
    print(f"  DPO epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Beta:       {args.beta}")
    print(f"  LR:         {args.lr}")
    print(f"  Pref margin:{args.preference_margin}")
    print(f"  Label smooth:{args.label_smoothing}")
    print(f"  Seed:       {args.seed}")

    buf = load_buffer(args.dataset)
    print(f"\n  Buffer loaded: {len(buf)} transitions")

    # -- Construct preference dataset --
    print("\n  Constructing preference pairs...")
    if args.mode == "transition":
        pref_dataset = TransitionPreferenceDataset.from_buffer(
            buf,
            margin=args.preference_margin,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
    else:
        pref_dataset = PreferenceDataset.from_buffer(
            buf,
            margin=args.preference_margin,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
    print(f"  Preference pairs: {len(pref_dataset)}")

    if len(pref_dataset) == 0:
        print("  ERROR: No preference pairs found. Check margin or dataset.")
        return

    # -- Tensorboard --
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        print(f"  Tensorboard: {output_dir / 'tb'}")
    except ImportError:
        print("  Tensorboard: not available")

    # -- Initialize agent --
    agent = DPOAgent(
        state_dim=args.state_dim,
        num_actions=args.num_actions,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        beta=args.beta,
        grad_clip=args.grad_clip,
        preference_margin=args.preference_margin,
        label_smoothing=args.label_smoothing,
    )

    total_params = sum(p.numel() for p in agent.policy_network.parameters())
    print(f"  Policy network: {total_params:,} params")
    print(f"  Reference network: {total_params:,} params (frozen after BC)")

    # -- BC warmup --
    print("\n" + "=" * 70)
    print("  Phase 1: Behavioral Cloning Warmup")
    print("=" * 70 + "\n")

    start_time = time.time()
    bc_metrics = agent.init_from_bc(
        buf,
        bc_epochs=args.bc_epochs,
        batch_size=min(256, len(buf)),
    )
    bc_elapsed = time.time() - start_time

    bc_first = np.mean(bc_metrics["bc_loss_history"][:3])
    bc_last = np.mean(bc_metrics["bc_loss_history"][-3:])
    print(f"\n  BC warmup complete in {bc_elapsed:.1f}s")
    print(f"  BC loss: {bc_first:.4f} -> {bc_last:.4f}")

    if writer is not None:
        for i, loss in enumerate(bc_metrics["bc_loss_history"], 1):
            writer.add_scalar("loss/bc_warmup", loss, i)

    # -- DPO training --
    print("\n" + "=" * 70)
    print(f"  Phase 2: DPO Training ({args.mode}-level)")
    print("=" * 70 + "\n")

    dpo_start = time.time()
    if args.mode == "transition":
        dpo_metrics = agent.train_transitions(
            pref_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            log_every=args.log_every,
            writer=writer,
        )
    else:
        dpo_metrics = agent.train(
            pref_dataset,
            num_epochs=args.epochs,
            batch_size=min(args.batch_size, 64),
            log_every=args.log_every,
            writer=writer,
        )
    dpo_elapsed = time.time() - dpo_start

    # -- Summary --
    dpo_first = np.mean(dpo_metrics["dpo_loss_history"][:3])
    dpo_last = np.mean(dpo_metrics["dpo_loss_history"][-3:])
    acc_first = np.mean(dpo_metrics["accuracy_history"][:3])
    acc_last = np.mean(dpo_metrics["accuracy_history"][-3:])

    print(f"\n  DPO training complete in {dpo_elapsed:.1f}s")
    print(f"  DPO loss:    {dpo_first:.4f} -> {dpo_last:.4f}")
    print(f"  Pref acc:    {acc_first:.2%} -> {acc_last:.2%}")

    # -- Save checkpoint --
    checkpoint_path = str(output_dir / "checkpoint.pt")
    agent.save(checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")

    # -- Log hyperparameters --
    if writer is not None:
        writer.add_hparams(
            {
                "mode": args.mode,
                "lr": args.lr,
                "beta": args.beta,
                "bc_epochs": args.bc_epochs,
                "dpo_epochs": args.epochs,
                "batch_size": args.batch_size,
                "preference_margin": args.preference_margin,
                "hidden_dim": args.hidden_dim,
                "grad_clip": args.grad_clip,
                "label_smoothing": args.label_smoothing,
            },
            {
                "hparam/final_dpo_loss": dpo_last,
                "hparam/final_pref_acc": acc_last,
            },
        )
        writer.close()

    total_elapsed = bc_elapsed + dpo_elapsed
    print(f"\n  Total training time: {total_elapsed:.1f}s")
    print("\n" + "=" * 70)
    print("  Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
