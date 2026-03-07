"""
Pareto frontier plot: accuracy vs mean retrieval steps.

Loads data/eval_results_final.json and creates a scatter plot with
the Pareto frontier connecting non-dominated points. Uses a broken
x-axis to handle the large gap between selective (3-6 steps) and
exhaustive (20 steps) policies.

Usage:
    python -m scripts.plot_pareto

Output:
    figures/pareto_frontier.png (300 DPI)
    figures/pareto_frontier.pdf (vector)
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = "data/eval_results_final.json"
OUTPUT_DIR = "figures"

# Display names and colors for each policy
POLICY_STYLES = {
    "cql":        {"label": "CQL",           "color": "#2ecc71", "marker": "s"},
    "iql":        {"label": "IQL",           "color": "#3498db", "marker": "D"},
    "bc":         {"label": "BC",            "color": "#9b59b6", "marker": "^"},
    "fixed_k_3":  {"label": "FixedK(3)",     "color": "#e74c3c", "marker": "o"},
    "fixed_k_5":  {"label": "FixedK(5)",     "color": "#f39c12", "marker": "v"},
    "heuristic":  {"label": "Heuristic(0.8)","color": "#1abc9c", "marker": "P"},
}

# Label offsets (dx, dy in points) per policy for readability
LABEL_OFFSETS = {
    "cql":       (-15, 10),
    "iql":       (10, -12),
    "bc":        (10, 10),
    "fixed_k_3": (-35, -15),
    "fixed_k_5": (10, 8),
    "heuristic": (10, -12),
}


def compute_pareto_frontier(points):
    """Find Pareto-optimal points (maximize accuracy, minimize steps).

    A point (steps, acc) dominates another if it has <= steps AND >= accuracy
    (with at least one strict inequality).

    Returns indices of Pareto-optimal points sorted by steps.
    """
    n = len(points)
    is_pareto = [True] * n
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (points[j][0] <= points[i][0] and
                points[j][1] >= points[i][1] and
                (points[j][0] < points[i][0] or points[j][1] > points[i][1])):
                is_pareto[i] = False
                break
    pareto_idx = [i for i in range(n) if is_pareto[i]]
    pareto_idx.sort(key=lambda i: points[i][0])
    return pareto_idx


def main() -> None:
    print("=" * 70)
    print("  Pareto Frontier Plot (R42)")
    print("=" * 70)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect data
    points = []
    point_keys = []
    for key, style in POLICY_STYLES.items():
        if key not in data:
            continue
        d = data[key]
        steps = d["mean_steps"]
        acc = d["accuracy"] * 100
        points.append((steps, acc))
        point_keys.append(key)

    if not points:
        print("  No data found. Exiting.")
        return

    # Determine axis break ranges
    all_steps = [p[0] for p in points]
    low_steps = [s for s in all_steps if s < 10]
    high_steps = [s for s in all_steps if s >= 10]

    use_break = bool(low_steps and high_steps)

    if use_break:
        # Broken x-axis: left panel [1, max_low+2], right panel [min_high-1, max_high+1]
        left_max = max(low_steps) + 2.0
        right_min = min(high_steps) - 1.0
        right_max = max(high_steps) + 1.0

        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, figsize=(8, 5.5),
            gridspec_kw={"width_ratios": [3, 1.5], "wspace": 0.08},
        )

        # Plot on both axes
        for idx, (steps, acc) in enumerate(points):
            key = point_keys[idx]
            style = POLICY_STYLES[key]
            for ax in (ax1, ax2):
                ax.scatter(
                    steps, acc,
                    c=style["color"],
                    marker=style["marker"],
                    s=160,
                    zorder=5,
                    edgecolors="white",
                    linewidth=1.0,
                )

            # Place label on the correct axis
            target_ax = ax1 if steps < 10 else ax2
            dx, dy = LABEL_OFFSETS.get(key, (10, 8))
            target_ax.annotate(
                style["label"],
                (steps, acc),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=9,
                fontweight="bold",
                color=style["color"],
            )

        # Pareto frontier
        pareto_idx = compute_pareto_frontier(points)
        pareto_steps = [points[i][0] for i in pareto_idx]
        pareto_acc = [points[i][1] for i in pareto_idx]
        for ax in (ax1, ax2):
            ax.plot(
                pareto_steps, pareto_acc,
                color="#2c3e50", linewidth=2.0, linestyle="--",
                alpha=0.7, zorder=3,
            )

        # Set axis limits
        ax1.set_xlim(max(0.5, min(low_steps) - 1), left_max)
        ax2.set_xlim(right_min, right_max)
        y_min = max(40, min(p[1] for p in points) - 8)
        ax1.set_ylim(y_min, 100)
        ax2.set_ylim(y_min, 100)

        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # Hide inner spines for break effect
        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(left=False)

        # Draw break marks
        d = 0.02
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.2)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        # Labels
        fig.supxlabel("Mean Retrieval Steps", fontsize=12, y=0.02)
        ax1.set_ylabel("Accuracy (%)", fontsize=12)
        ax1.tick_params(labelsize=10)
        ax2.tick_params(labelsize=10)

        # Build legend from both axes, deduplicate
        # Manual legend since scatter points are duplicated
        legend_handles = []
        for key in point_keys:
            style = POLICY_STYLES[key]
            h = plt.Line2D(
                [0], [0], marker=style["marker"], color="w",
                markerfacecolor=style["color"], markersize=10,
                markeredgecolor="white", markeredgewidth=0.5,
                label=style["label"],
            )
            legend_handles.append(h)
        legend_handles.append(
            plt.Line2D([0], [0], color="#2c3e50", linewidth=2.0,
                       linestyle="--", alpha=0.7, label="Pareto frontier")
        )
        ax1.legend(handles=legend_handles, loc="center left",
                   fontsize=9, framealpha=0.9)

        fig.suptitle(
            "Accuracy vs Retrieval Efficiency",
            fontsize=13, fontweight="bold", y=0.97,
        )
    else:
        # No break needed — simple plot
        fig, ax = plt.subplots(figsize=(7, 5.5))

        for idx, (steps, acc) in enumerate(points):
            key = point_keys[idx]
            style = POLICY_STYLES[key]
            ax.scatter(
                steps, acc, c=style["color"], marker=style["marker"],
                s=160, label=style["label"], zorder=5,
                edgecolors="white", linewidth=1.0,
            )
            dx, dy = LABEL_OFFSETS.get(key, (10, 8))
            ax.annotate(
                style["label"], (steps, acc),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=9, fontweight="bold", color=style["color"],
            )

        pareto_idx = compute_pareto_frontier(points)
        pareto_steps = [points[i][0] for i in pareto_idx]
        pareto_acc = [points[i][1] for i in pareto_idx]
        ax.plot(pareto_steps, pareto_acc, color="#2c3e50", linewidth=2.0,
                linestyle="--", alpha=0.7, label="Pareto frontier", zorder=3)

        ax.set_xlabel("Mean Retrieval Steps", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Accuracy vs Retrieval Efficiency",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=max(0, min(p[1] for p in points) - 10), top=105)
        ax.tick_params(labelsize=10)

    if not use_break:
        plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "pareto_frontier.png")
    pdf_path = os.path.join(OUTPUT_DIR, "pareto_frontier.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
