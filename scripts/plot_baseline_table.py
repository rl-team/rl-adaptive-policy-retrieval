"""
Render the baseline comparison table as a figure

Usage:
    python -m scripts.plot_baseline_table
    python -m scripts.plot_baseline_table --output figures/my_table.png

Output: figures/baseline_comparison_table.png and .pdf
"""
from __future__ import annotations
from pathlib import Path
from scripts.plot_style import apply_publication_style, PUBLICATION_DPI

import argparse
import matplotlib.pyplot as plt


POLICY_NAMES = ["FixedK(k=3)", "FixedK(k=5)", "Heuristic(0.8)", "Conservative QL (α=0.5)"]
ACCURACY = ["100.0% (100–100%)", "100.0% (100–100%)", "98.0% (92–100%)", "100.0% (100–100%)"]
MEAN_CHUNKS = ["4.00 (4.00–4.00)", "6.00 (6.00–6.00)", "3.64 (3.50–3.78)", "4.10 (4.10–4.10)"]
COST_ADJ_UTILITY = ["0.60 (0.60–0.60)", "0.40 (0.40–0.40)", "0.62 (0.57–0.65)", "0.59 (0.59–0.59)"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Render baseline comparison table as image.")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/baseline_comparison_table.png",
        help="Output path (PNG). PDF saved to same path with .pdf extension.",
    )
    parser.add_argument("--dpi", type=int, default=PUBLICATION_DPI, help="DPI for PNG.")
    args = parser.parse_args()

    apply_publication_style(dpi=args.dpi)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Policy", "Accuracy (95% CI)", "Mean chunks (95% CI)", "Cost-adj. utility (95% CI)"]
    rows = [
        [POLICY_NAMES[i], ACCURACY[i], MEAN_CHUNKS[i], COST_ADJ_UTILITY[i]]
        for i in range(len(POLICY_NAMES))
    ]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    ax.set_title("Policy comparison on test set", fontsize=12, fontweight="bold", pad=10)
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e8e8e8"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    fig.text(
        0.5, 0.02,
        "Table 1. Policy comparison on test set. 50 episodes, seed 999. Cost-adj. utility = accuracy − 0.1 × mean chunks; 95% CIs from bootstrap (1000 samples).",
        ha="center", fontsize=8, style="italic",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    pdf_path = out_path.with_suffix(".pdf")
    fig2, ax2 = plt.subplots(figsize=(10, 3.2))
    ax2.axis("off")
    ax2.set_title("Policy comparison on test set", fontsize=12, fontweight="bold", pad=10)
    table2 = ax2.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e8e8e8"] * 4,
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 2.0)
    fig2.text(
        0.5, 0.02,
        "Table 1. Policy comparison on test set. 50 episodes, seed 999. Cost-adj. utility = accuracy − 0.1 × mean chunks; 95% CIs from bootstrap (1000 samples).",
        ha="center", fontsize=8, style="italic",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
