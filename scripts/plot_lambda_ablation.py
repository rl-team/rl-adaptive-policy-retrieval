"""
Lambda ablation figure: accuracy, steps, and return for each step_cost lambda.

For each lambda in {0.05, 0.1, 0.2}, evaluates the corresponding CQL
checkpoint with the MATCHING step_cost in the env's RewardFunction.
Plots three panels: accuracy, steps, and return.

Usage:
    python -m scripts.plot_lambda_ablation

Output:
    figures/lambda_ablation.png (300 DPI)
    figures/lambda_ablation.pdf (vector)
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluate_agent import (
    evaluate_policy,
    run_trained_agent_episode,
)
from rl.conservative_ql_agent import ConservativeQLAgent


OUTPUT_DIR = "figures"
NUM_EPISODES = 200
SEED = 42

# Lambda values and corresponding checkpoint paths
LAMBDA_CONFIGS = [
    (0.05, "runs/sweep_lambda_0.05/checkpoint.pt"),
    (0.1,  "runs/sweep_lambda_0.1/checkpoint.pt"),
    (0.2,  "runs/sweep_lambda_0.2/checkpoint.pt"),
]

BAR_COLORS = ["#3498db", "#2ecc71", "#e74c3c"]


def main() -> None:
    print("=" * 70)
    print("  Lambda Ablation (R44)")
    print("=" * 70)

    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lambdas = []
    accuracies = []
    mean_steps_list = []
    mean_returns = []

    for lam, ckpt_path in LAMBDA_CONFIGS:
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: {ckpt_path} not found, skipping lambda={lam}")
            continue

        agent = ConservativeQLAgent.load(ckpt_path)
        print(f"\n  lambda={lam}: loaded {ckpt_path} "
              f"(alpha={agent.alpha})")

        # IMPORTANT: use matching step_cost for this checkpoint
        metrics = evaluate_policy(
            name=f"CQL(lambda={lam})",
            runner=lambda env, a=agent: run_trained_agent_episode(env, a),
            num_episodes=NUM_EPISODES,
            seed=SEED,
            step_cost=lam,  # Match the training step_cost
        )

        lambdas.append(lam)
        accuracies.append(metrics["accuracy"] * 100)
        mean_steps_list.append(metrics["mean_steps"])
        mean_returns.append(metrics["mean_return"])

        print(f"    Accuracy: {metrics['accuracy']:.1%}, "
              f"Steps: {metrics['mean_steps']:.1f}, "
              f"Return: {metrics['mean_return']:.2f}")

    if not lambdas:
        print("  No checkpoints found. Exiting.")
        return

    elapsed = time.time() - start
    print(f"\n  Evaluation complete in {elapsed:.1f}s")

    # -- Plot: three subplots side by side --
    n = len(lambdas)
    x = np.arange(n)
    bar_width = 0.55
    colors = BAR_COLORS[:n]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.5))

    # --- Accuracy ---
    bars1 = ax1.bar(x, accuracies, bar_width, color=colors,
                    edgecolor="white", linewidth=0.8)
    ax1.set_xlabel(r"Step Cost ($\lambda$)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lambdas], fontsize=10)
    ax1.set_ylim(0, max(accuracies) * 1.18)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    # --- Mean Steps ---
    bars2 = ax2.bar(x, mean_steps_list, bar_width, color=colors,
                    edgecolor="white", linewidth=0.8)
    ax2.set_xlabel(r"Step Cost ($\lambda$)", fontsize=11)
    ax2.set_ylabel("Mean Steps", fontsize=11)
    ax2.set_title("Mean Retrieval Steps", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in lambdas], fontsize=10)
    ax2.set_ylim(0, max(mean_steps_list) * 1.18)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, mean_steps_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    # --- Mean Return ---
    bars3 = ax3.bar(x, mean_returns, bar_width, color=colors,
                    edgecolor="white", linewidth=0.8)
    ax3.set_xlabel(r"Step Cost ($\lambda$)", fontsize=11)
    ax3.set_ylabel("Mean Return", fontsize=11)
    ax3.set_title("Mean Episodic Return", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(l) for l in lambdas], fontsize=10)
    # Ensure y-axis includes zero reference
    y_min = min(min(mean_returns) * 1.3, -0.1)
    y_max = max(max(mean_returns) * 1.3, 0.1)
    ax3.set_ylim(y_min, y_max)
    ax3.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, mean_returns):
        offset = 0.03 if val >= 0 else -0.03
        va = "bottom" if val >= 0 else "top"
        ax3.text(bar.get_x() + bar.get_width() / 2, val + offset,
                 f"{val:.2f}", ha="center", va=va, fontsize=10,
                 fontweight="bold")

    fig.suptitle(
        r"CQL Performance Across Step Cost ($\lambda$) Values",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "lambda_ablation.png")
    pdf_path = os.path.join(OUTPUT_DIR, "lambda_ablation.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
