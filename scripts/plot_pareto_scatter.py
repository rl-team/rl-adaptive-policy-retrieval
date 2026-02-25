"""
Load evaluation results and output Pareto (accuracy vs chunks) as LaTeX tabular;
optionally render scatter plot as PNG/PDF.

Data source: data/eval_results.json from scripts.run_baseline_eval.
Usage: python -m scripts.plot_pareto_scatter
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from scripts.plot_style import apply_publication_style, PUBLICATION_DPI

INPUT_JSON = "data/eval_results.json"
LATEX_OUT = "figures/pareto_table.tex"
OUTPUT_PNG_PDF = True
FIGURE_PATH = "figures/pareto_scatter.png"

POLICY_ORDER = [
    "FixedK(k=3)",
    "FixedK(k=5)",
    "Heuristic(0.8)",
    "Conservative QL (α=0.5)",
]
COLORS = ["#2e86ab", "#9b59b6", "#e94f37", "#2ecc71"]
ANNOT_OFFSETS = [(-10, 6), (-10, 6), (8, 6), (8, 6)]
ANNOT_HA = ["right", "right", "left", "left"]


def load_results(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("meta", {}), data.get("results", {})


def results_to_series(
    results: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[float], List[float]]:
    """Return (policy_names, mean_chunks, accuracy_pct) in POLICY_ORDER."""
    names, chunks, accs = [], [], []
    for name in POLICY_ORDER:
        if name not in results:
            continue
        m = results[name]
        names.append(name)
        chunks.append(m["mean_chunks"])
        accs.append(m["accuracy"] * 100.0)
    return names, chunks, accs


def emit_latex_tabular(
    names: List[str],
    chunks: List[float],
    accs: List[float],
    meta: Dict[str, Any],
    out_path: Optional[Path],
) -> str:
    """Emit LaTeX tabular: Policy | Average chunks retrieved | Decision accuracy (%)."""
    lines = [
        r"\begin{tabular}{lcc}",
        r"  \toprule",
        r"  Policy & Average chunks retrieved & Decision accuracy (\%) \\",
        r"  \midrule",
    ]
    for i, name in enumerate(names):
        policy_tex = name.replace("α=0.5", r"$\alpha=0.5$")
        lines.append(f"  {policy_tex} & {chunks[i]:.2f} & {accs[i]:.1f}\\% \\\\")
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    table = "\n".join(lines)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(table)
    return table


def render_figure(
    names: List[str],
    chunks: List[float],
    accs: List[float],
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    apply_publication_style(dpi=dpi)
    n = len(names)
    colors = COLORS[:n]
    offsets = ANNOT_OFFSETS[:n]
    has = ANNOT_HA[:n]
    ax.scatter(
        chunks, accs,
        s=140, zorder=3, color=colors, edgecolors="white", linewidths=0.8,
    )
    for i, label in enumerate(names):
        ax.annotate(
            label,
            (chunks[i], accs[i]),
            xytext=offsets[i],
            textcoords="offset points",
            fontsize=9,
            ha=has[i],
        )
    order = sorted(range(n), key=lambda i: chunks[i])
    x_line = [chunks[i] for i in order]
    y_line = [accs[i] for i in order]
    ax.step(x_line, y_line, where="post", color="gray", linestyle="--", alpha=0.7, zorder=1)
    ax.set_xlabel("Average chunks retrieved", fontsize=10)
    ax.set_ylabel("Decision accuracy (%)", fontsize=10)
    ax.set_xlim(min(chunks) - 0.3, max(chunks) + 0.3)
    ax.set_ylim(min(accs) - 2, 101)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.3)
    plt.close()


def main() -> None:
    path = Path(INPUT_JSON)
    if not path.exists():
        raise SystemExit(f"Input not found: {path}. Run: python -m scripts.run_baseline_eval [--checkpoint ...] --output {path}")

    meta, results = load_results(path)
    names, chunks, accs = results_to_series(results)
    if not names:
        raise SystemExit("No policy results found in JSON.")

    out = Path(LATEX_OUT)
    emit_latex_tabular(names, chunks, accs, meta, out)
    print(f"LaTeX written to {out}")

    if OUTPUT_PNG_PDF:
        out_path = Path(FIGURE_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_figure(names, chunks, accs, out_path, PUBLICATION_DPI)
        print(f"Figure saved: {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
