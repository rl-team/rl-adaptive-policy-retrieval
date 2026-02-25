"""
Render the baseline comparison table as a figure

Usage:
    python -m scripts.plot_baseline_table
    python -m scripts.plot_baseline_table --output figures/my_table.png

Output: figures/baseline_comparison_table.png and .pdf
"""
import argparse

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

POLICY_NAMES = ["FixedK(k=3)", "FixedK(k=5)", "Heuristic(0.8)"]
ACCURACY = ["100.0% (100–100%)", "100.0% (100–100%)", "98.0% (92–100%)"]
MEAN_CHUNKS = ["4.00 (4.00–4.00)", "6.00 (6.00–6.00)", "3.64 (3.50–3.78)"]
COST_ADJ_UTILITY = ["0.60 (0.60–0.60)", "0.40 (0.40–0.40)", "0.62 (0.57–0.65)"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Render baseline comparison table as image.")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/baseline_comparison_table.png",
        help="Output path (PNG). PDF saved to same path with .pdf extension.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PNG.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Policy", "Accuracy (95% CI)", "Mean Chunks (95% CI)", "Cost-Adj Utility (95% CI)"]
    rows = [
        [POLICY_NAMES[i], ACCURACY[i], MEAN_CHUNKS[i], COST_ADJ_UTILITY[i]]
        for i in range(len(POLICY_NAMES))
    ]

    _, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    ax.set_title("Baseline Comparison: Fixed-K vs Heuristic", fontsize=12, fontweight="bold", pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e8e8e8"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    pdf_path = out_path.with_suffix(".pdf")
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    ax2.axis("off")
    ax2.set_title("Baseline Comparison: Fixed-K vs Heuristic", fontsize=12, fontweight="bold", pad=12)
    table2 = ax2.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e8e8e8"] * 4,
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 2.2)
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
