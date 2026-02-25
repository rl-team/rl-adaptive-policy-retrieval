"""
Partial Pareto plot: Accuracy vs mean chunks (publication-quality).

Shows the accuracy–efficiency tradeoff for baselines and Conservative QL.
Data from run_baseline_eval (50 episodes, seed 999) and milestone CQL eval.

Usage:
    python -m scripts.plot_pareto_scatter
    python -m scripts.plot_pareto_scatter --output figures/pareto_scatter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.plot_style import apply_publication_style, PUBLICATION_DPI


# Three points: (mean_chunks, accuracy_pct)
# FixedK(k=3): 100% accuracy, 4.00 chunks
# Heuristic(0.8): 98% accuracy, 3.64 chunks
# Conservative QL (α=0.5): 100% accuracy, 4.10 chunks (milestone eval)
CHUNKS = [4.00, 3.64, 4.10]
ACCURACY_PCT = [100.0, 98.0, 100.0]
LABELS = ["FixedK(k=3)", "Heuristic(0.8)", "Conservative QL (α=0.5)"]
COLORS = ["#2e86ab", "#e94f37", "#2ecc71"]  # blue, red, green
# Per-point (dx, dy) in points; ha: horizontal alignment. FixedK left, CQL right to avoid overlap.
ANNOT_OFFSETS = [(-10, 6), (8, 6), (8, 6)]   # FixedK, Heuristic, Conservative QL
ANNOT_HA = ["right", "left", "left"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Accuracy vs Mean Chunks scatter (partial Pareto)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/pareto_scatter.png",
        help="Output path (PNG). PDF saved with .pdf extension.",
    )
    parser.add_argument("--dpi", type=int, default=PUBLICATION_DPI, help="DPI for PNG.")
    args = parser.parse_args()

    apply_publication_style(dpi=args.dpi)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(CHUNKS, ACCURACY_PCT, s=140, zorder=3, color=COLORS, edgecolors="white", linewidths=0.8)
    for i, label in enumerate(LABELS):
        ax.annotate(
            label,
            (CHUNKS[i], ACCURACY_PCT[i]),
            xytext=ANNOT_OFFSETS[i],
            textcoords="offset points",
            fontsize=9,
            ha=ANNOT_HA[i],
        )
    ax.set_xlabel("Mean number of chunks retrieved", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Accuracy–efficiency tradeoff (partial Pareto)", fontsize=11, fontweight="bold")
    ax.set_xlim(3.4, 4.4)
    ax.set_ylim(96.5, 101)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(
        0.5, 0.02,
        "Figure 1. Accuracy–efficiency tradeoff (partial Pareto). Higher accuracy and fewer chunks are better; Pareto frontier is top-left. Test set: 50 episodes, seed 999.",
        ha="center", fontsize=8, style="italic",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    pdf_path = out_path.with_suffix(".pdf")
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.2))
    ax2.scatter(CHUNKS, ACCURACY_PCT, s=140, zorder=3, color=COLORS, edgecolors="white", linewidths=0.8)
    for i, label in enumerate(LABELS):
        ax2.annotate(
            label,
            (CHUNKS[i], ACCURACY_PCT[i]),
            xytext=ANNOT_OFFSETS[i],
            textcoords="offset points",
            fontsize=9,
            ha=ANNOT_HA[i],
        )
    ax2.set_xlabel("Mean number of chunks retrieved", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Accuracy–efficiency tradeoff (partial Pareto)", fontsize=11, fontweight="bold")
    ax2.set_xlim(3.4, 4.4)
    ax2.set_ylim(96.5, 101)
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    fig2.text(
        0.5, 0.02,
        "Figure 1. Accuracy–efficiency tradeoff (partial Pareto). Higher accuracy and fewer chunks are better; Pareto frontier is top-left. Test set: 50 episodes, seed 999.",
        ha="center", fontsize=8, style="italic",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
