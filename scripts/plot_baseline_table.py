"""
Load evaluation results and output baseline comparison as LaTeX tabular;
optionally render the same data as a figure (PNG/PDF).

Data source: data/eval_results.json from scripts.run_baseline_eval.
Usage: python -m scripts.plot_baseline_table
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from scripts.plot_style import apply_publication_style, PUBLICATION_DPI

INPUT_JSON = "data/eval_results.json"
LATEX_OUT = "figures/baseline_table.tex"
OUTPUT_PNG_PDF = True
FIGURE_PATH = "figures/baseline_comparison_table.png"

POLICY_ORDER = [
    "FixedK(k=3)",
    "FixedK(k=5)",
    "Heuristic(0.8)",
    "Conservative QL (α=0.5)",
]


def _format_ci_str(mean: float, ci: List[float], pct: bool) -> str:
    low, high = ci[0], ci[1]
    if pct:
        return f"{mean:.1%} ({low:.1%}--{high:.1%})"
    return f"{mean:.2f} ({low:.2f}--{high:.2f})"


def load_results(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    with open(path) as f:
        data = json.load(f)
    meta = data.get("meta", {})
    results = data.get("results", {})
    return meta, results


def results_to_rows(meta: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """Build table rows: [Policy, Accuracy (95% CI), Mean chunks (95% CI), Cost-adj. utility (95% CI)]."""
    rows = []
    for name in POLICY_ORDER:
        if name not in results:
            continue
        m = results[name]
        acc_str = _format_ci_str(m["accuracy"], m["accuracy_ci"], pct=True)
        ch_str = _format_ci_str(m["mean_chunks"], m["mean_chunks_ci"], pct=False)
        cau_str = _format_ci_str(m["cost_adjusted_utility"], m["cost_adjusted_utility_ci"], pct=False)
        policy_display = name.replace("α=0.5", r"$\alpha=0.5$")
        rows.append([policy_display, acc_str, ch_str, cau_str])
    return rows


def emit_latex_tabular(rows: List[List[str]], meta: Dict[str, Any], out_path: Optional[Path]) -> str:
    """Produce LaTeX \\begin{tabular}...\\end{tabular}."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"  \toprule",
        r"  Policy & Accuracy (95\% CI) & Mean chunks (95\% CI) & Cost-adj.\ utility (95\% CI) \\",
        r"  \midrule",
    ]
    for row in rows:
        row_esc = [c.replace("%", r"\%").replace("&", r"\&") for c in row]
        lines.append("  " + " & ".join(row_esc) + r" \\")
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    table = "\n".join(lines)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(table)
    return table


def render_figure(rows: List[List[str]], out_path: Path, dpi: int) -> None:
    """Render table as PNG/PDF using matplotlib."""
    apply_publication_style(dpi=dpi)
    headers = ["Policy", "Accuracy (95% CI)", "Mean chunks (95% CI)", "Cost-adj. utility (95% CI)"]
    # Strip LaTeX for display
    display_rows = [[r.replace(r"$\alpha=0.5$", "α=0.5") for r in row] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    ax.set_title("Policy comparison on test set", fontsize=12, fontweight="bold", pad=10)
    table = ax.table(
        cellText=display_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e8e8e8"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.3)
    plt.close()


def main() -> None:
    path = Path(INPUT_JSON)
    if not path.exists():
        raise SystemExit(f"Input not found: {path}. Run: python -m scripts.run_baseline_eval [--checkpoint ...] --output {path}")

    meta, results = load_results(path)
    rows = results_to_rows(meta, results)
    if not rows:
        raise SystemExit("No policy results found in JSON.")

    out = Path(LATEX_OUT)
    emit_latex_tabular(rows, meta, out)
    print(f"LaTeX written to {out}")

    if OUTPUT_PNG_PDF:
        out_path = Path(FIGURE_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_figure(rows, out_path, PUBLICATION_DPI)
        print(f"Figure saved: {out_path} and {out_path.with_suffix('.pdf')}")

if __name__ == "__main__":
    main()
