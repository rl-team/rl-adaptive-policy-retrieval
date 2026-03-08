"""
Per-procedure accuracy breakdown: grouped bar chart for all 6 policies.

Loads data/eval_results_final.json and computes accuracy for each
(procedure, policy) pair. Produces a grouped bar chart showing how
each policy performs on each of the 10 CMS medical procedures.

Usage:
    python -m scripts.plot_per_procedure

Output:
    figures/per_procedure_breakdown.png (300 DPI)
    figures/per_procedure_breakdown.pdf (vector)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = "data/eval_results_final.json"
TEMPLATES_PATH = "data/templates.json"
OUTPUT_DIR = "figures"

# Policy display order and styles (consistent with plot_pareto.py)
POLICY_ORDER = ["cql", "iql", "bc", "fixed_k_3", "fixed_k_5", "heuristic"]
POLICY_STYLES = {
    "cql":       {"label": "CQL",           "color": "#2ecc71"},
    "iql":       {"label": "IQL",           "color": "#3498db"},
    "bc":        {"label": "BC",            "color": "#9b59b6"},
    "fixed_k_3": {"label": "FixedK(3)",     "color": "#e74c3c"},
    "fixed_k_5": {"label": "FixedK(5)",     "color": "#f39c12"},
    "heuristic": {"label": "Heuristic(0.8)","color": "#1abc9c"},
}


def load_procedure_names() -> dict[str, str]:
    """Load CPT code -> short procedure name mapping from templates."""
    if not os.path.exists(TEMPLATES_PATH):
        return {}
    with open(TEMPLATES_PATH) as f:
        templates = json.load(f)
    return {code: t.get("name", code) for code, t in templates.items()}


def compute_per_procedure_accuracy(
    data: dict,
) -> tuple[list[str], dict[str, list[float]], dict[str, int]]:
    """Compute accuracy per procedure per policy.

    Returns:
        procedure_codes: sorted list of unique procedure codes
        accuracies: {policy_key: [accuracy_per_procedure]}
        counts: {procedure_code: number_of_episodes}
    """
    # Gather all procedure codes across all policies
    all_codes: set[str] = set()
    for policy_key in POLICY_ORDER:
        if policy_key not in data:
            continue
        for ep in data[policy_key]["episodes"]:
            all_codes.add(ep["procedure_code"])

    procedure_codes = sorted(all_codes)

    # Count episodes per procedure (same across policies since seed is fixed)
    counts: dict[str, int] = defaultdict(int)
    first_policy = next(k for k in POLICY_ORDER if k in data)
    for ep in data[first_policy]["episodes"]:
        counts[ep["procedure_code"]] += 1

    # Compute accuracy
    accuracies: dict[str, list[float]] = {}
    for policy_key in POLICY_ORDER:
        if policy_key not in data:
            accuracies[policy_key] = [0.0] * len(procedure_codes)
            continue

        # Group episodes by procedure
        correct_by_proc: dict[str, int] = defaultdict(int)
        total_by_proc: dict[str, int] = defaultdict(int)
        for ep in data[policy_key]["episodes"]:
            pc = ep["procedure_code"]
            total_by_proc[pc] += 1
            if ep["correct"]:
                correct_by_proc[pc] += 1

        acc_list = []
        for pc in procedure_codes:
            if total_by_proc[pc] > 0:
                acc_list.append(correct_by_proc[pc] / total_by_proc[pc] * 100)
            else:
                acc_list.append(0.0)
        accuracies[policy_key] = acc_list

    return procedure_codes, accuracies, dict(counts)


def main() -> None:
    print("=" * 70)
    print("  Per-Procedure Accuracy Breakdown (M33/M36)")
    print("=" * 70)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    proc_names = load_procedure_names()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    procedure_codes, accuracies, counts = compute_per_procedure_accuracy(data)

    if not procedure_codes:
        print("  No procedure data found. Exiting.")
        return

    # Print table
    print(f"\n  {'Procedure':<35s}", end="")
    for pk in POLICY_ORDER:
        print(f"  {POLICY_STYLES[pk]['label']:>12s}", end="")
    print(f"  {'n':>4s}")
    print("  " + "-" * 115)
    for i, pc in enumerate(procedure_codes):
        name = proc_names.get(pc, pc)
        label = f"{name} ({pc})"
        print(f"  {label:<35s}", end="")
        for pk in POLICY_ORDER:
            print(f"  {accuracies[pk][i]:>11.1f}%", end="")
        print(f"  {counts[pc]:>4d}")

    # -- Grouped bar chart --
    n_procs = len(procedure_codes)
    n_policies = len(POLICY_ORDER)
    bar_width = 0.12
    group_width = n_policies * bar_width + 0.08  # small gap between groups

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x = np.arange(n_procs)

    for j, pk in enumerate(POLICY_ORDER):
        style = POLICY_STYLES[pk]
        offset = (j - n_policies / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            accuracies[pk],
            bar_width,
            label=style["label"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
        )

    # X-axis labels: procedure name (CPT code)\n(n=XX)
    x_labels = []
    for pc in procedure_codes:
        name = proc_names.get(pc, pc)
        # Shorten long names
        if len(name) > 20:
            name = name[:18] + "..."
        x_labels.append(f"{name}\n({pc}, n={counts[pc]})")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8, ha="center")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_title(
        "Per-Procedure Accuracy Breakdown (All Policies)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
        fontsize=9,
        framealpha=0.9,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "per_procedure_breakdown.png")
    pdf_path = os.path.join(OUTPUT_DIR, "per_procedure_breakdown.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
