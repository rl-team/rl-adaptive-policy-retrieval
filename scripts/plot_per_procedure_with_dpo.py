"""
Per-procedure accuracy breakdown: grouped bar chart for all 7 policies (with DPO).

Extends plot_per_procedure.py to include DPO as a 7th policy alongside
CQL, IQL, BC, FixedK(3), FixedK(5), and Heuristic. DPO per-procedure
data is loaded from the evaluation output at /tmp/dpo_per_proc.json
(produced by scripts.evaluate_agent with --agent-type dpo).

Usage:
    python -m scripts.plot_per_procedure_with_dpo

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

from scripts.plot_style import apply_publication_style

INPUT_PATH = "data/eval_results_final.json"
DPO_INPUT_PATH = "/tmp/dpo_per_proc.json"
TEMPLATES_PATH = "data/templates.json"
OUTPUT_DIR = "figures"

# Policy display order and styles -- DPO added as 7th policy
# Colors chosen to be colorblind-friendly (distinct hues)
POLICY_ORDER = ["cql", "iql", "bc", "fixed_k_3", "fixed_k_5", "heuristic", "dpo"]
POLICY_STYLES = {
    "cql":       {"label": "CQL",            "color": "#2ecc71"},
    "iql":       {"label": "IQL",            "color": "#3498db"},
    "bc":        {"label": "BC",             "color": "#9b59b6"},
    "fixed_k_3": {"label": "FixedK(3)",      "color": "#e74c3c"},
    "fixed_k_5": {"label": "FixedK(5)",      "color": "#f39c12"},
    "heuristic": {"label": "Heuristic(0.8)", "color": "#1abc9c"},
    "dpo":       {"label": "DPO",            "color": "#e67e22"},
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
    policy_order: list[str],
) -> tuple[list[str], dict[str, list[float]], dict[str, int]]:
    """Compute accuracy per procedure per policy.

    Returns:
        procedure_codes: sorted list of unique procedure codes
        accuracies: {policy_key: [accuracy_per_procedure]}
        counts: {procedure_code: number_of_episodes}
    """
    # Gather all procedure codes across all policies
    all_codes: set[str] = set()
    for policy_key in policy_order:
        if policy_key not in data:
            continue
        for ep in data[policy_key]["episodes"]:
            all_codes.add(ep["procedure_code"])

    procedure_codes = sorted(all_codes)

    # Count episodes per procedure (same across policies since seed is fixed)
    first_policy = next(k for k in policy_order if k in data)
    counts: dict[str, int] = defaultdict(int)
    for ep in data[first_policy]["episodes"]:
        counts[ep["procedure_code"]] += 1

    # Compute accuracy per procedure per policy
    accuracies: dict[str, list[float]] = {}
    for policy_key in policy_order:
        if policy_key not in data:
            accuracies[policy_key] = [0.0] * len(procedure_codes)
            continue

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
    print("  Per-Procedure Accuracy Breakdown with DPO")
    print("=" * 70)

    apply_publication_style()

    with open(INPUT_PATH) as f:
        data = json.load(f)

    # Load DPO per-procedure data
    if os.path.exists(DPO_INPUT_PATH):
        with open(DPO_INPUT_PATH) as f:
            dpo_data = json.load(f)
        # The DPO key in the evaluation output is "DPO" (uppercase).
        # Normalize to lowercase key for consistency with other policies.
        if "DPO" in dpo_data:
            data["dpo"] = dpo_data["DPO"]
            print(f"  Loaded DPO data from {DPO_INPUT_PATH}")
            print(f"    DPO episodes: {len(data['dpo']['episodes'])}")
            print(f"    DPO accuracy: {data['dpo']['accuracy']:.1%}")
        else:
            print(f"  Warning: 'DPO' key not found in {DPO_INPUT_PATH}")
    else:
        print(f"  Warning: DPO results not found at {DPO_INPUT_PATH}")
        print("  Run: python -m scripts.evaluate_agent --checkpoint "
              "runs/dpo_2k/checkpoint.pt --agent-type dpo --episodes 200 "
              "--seed 42 --output-json /tmp/dpo_per_proc.json")

    proc_names = load_procedure_names()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    procedure_codes, accuracies, counts = compute_per_procedure_accuracy(
        data, POLICY_ORDER
    )

    if not procedure_codes:
        print("  No procedure data found. Exiting.")
        return

    # Print summary table
    print(f"\n  {'Procedure':<35s}", end="")
    for pk in POLICY_ORDER:
        if pk in data:
            print(f"  {POLICY_STYLES[pk]['label']:>12s}", end="")
    print(f"  {'n':>4s}")
    print("  " + "-" * 130)
    for i, pc in enumerate(procedure_codes):
        name = proc_names.get(pc, pc)
        label = f"{name} ({pc})"
        print(f"  {label:<35s}", end="")
        for pk in POLICY_ORDER:
            if pk in data:
                print(f"  {accuracies[pk][i]:>11.1f}%", end="")
        print(f"  {counts[pc]:>4d}")

    # ---- Grouped bar chart ----
    n_procs = len(procedure_codes)
    active_policies = [pk for pk in POLICY_ORDER if pk in data]
    n_policies = len(active_policies)
    bar_width = 0.11
    group_gap = 0.08

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x = np.arange(n_procs)

    for j, pk in enumerate(active_policies):
        style = POLICY_STYLES[pk]
        offset = (j - n_policies / 2 + 0.5) * bar_width
        edgecolor = "white"
        linewidth = 0.5
        # Make DPO bars visually distinct with a hatching pattern
        hatch = "///" if pk == "dpo" else None
        bars = ax.bar(
            x + offset,
            accuracies[pk],
            bar_width,
            label=style["label"],
            color=style["color"],
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
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
        ncol=7,
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
