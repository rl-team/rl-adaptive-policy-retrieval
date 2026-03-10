"""
Retrieval heatmap (compact) with DPO added as 4th column.

Loads:
  - data/eval_results_final.json  (CQL, IQL, BC under lowercase keys)
  - data/dpo_eval_seed42.json     (DPO under uppercase "DPO" key)

Produces a compact heatmap aggregated by procedure code with columns
ordered: CQL, DPO, IQL, BC.

Usage:
    python -m scripts.plot_retrieval_heatmap_with_dpo

Output:
    figures/retrieval_heatmap_compact.png  (300 DPI)
    figures/retrieval_heatmap_compact.pdf  (vector)
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

# --- Paths ---
EVAL_RESULTS_PATH = "data/eval_results_final.json"
DPO_RESULTS_PATH = "data/dpo_eval_seed42.json"
OUTPUT_DIR = "figures"

# Column order: CQL, DPO, IQL, BC
POLICY_ORDER = ["cql", "dpo", "iql", "bc"]
POLICY_DISPLAY = {
    "cql": "CQL",
    "dpo": "DPO",
    "iql": "IQL",
    "bc": "BC",
}


def load_data():
    """Load evaluation results from both files and merge into a single dict."""
    with open(EVAL_RESULTS_PATH) as f:
        data = json.load(f)

    with open(DPO_RESULTS_PATH) as f:
        dpo_raw = json.load(f)

    # DPO data is stored under uppercase "DPO" key; normalise to lowercase
    data["dpo"] = dpo_raw["DPO"]

    # Collect all chunk IDs and build chunk -> procedure mapping
    all_chunk_ids: set[str] = set()
    chunk_to_procedure: dict[str, str] = {}
    policy_keys: list[str] = []

    for key in POLICY_ORDER:
        if key not in data:
            print(f"  WARNING: policy '{key}' not found in data, skipping.")
            continue
        policy_keys.append(key)
        for ep in data[key]["episodes"]:
            proc = ep.get("procedure_code", "unknown")
            for cid in ep.get("chunk_ids", []):
                all_chunk_ids.add(cid)
                if cid not in chunk_to_procedure:
                    chunk_to_procedure[cid] = proc

    return data, all_chunk_ids, chunk_to_procedure, policy_keys


def plot_compact_heatmap(data, all_chunk_ids, chunk_to_procedure, policy_keys):
    """Compact heatmap aggregated by procedure code."""
    n_policies = len(policy_keys)

    # Collect all procedures and sort
    proc_set = set(chunk_to_procedure.values())
    sorted_procs = sorted(proc_set)
    n_procs = len(sorted_procs)
    proc_idx = {p: i for i, p in enumerate(sorted_procs)}

    # Matrix: (n_procs x n_policies) = mean chunks retrieved per episode
    freq_matrix = np.zeros((n_procs, n_policies))

    for j, key in enumerate(policy_keys):
        for ep in data[key]["episodes"]:
            for cid in ep.get("chunk_ids", []):
                proc = chunk_to_procedure.get(cid)
                if proc and proc in proc_idx:
                    freq_matrix[proc_idx[proc], j] += 1

    # Normalise by number of episodes (per policy, since DPO may differ)
    for j, key in enumerate(policy_keys):
        n_episodes = len(data[key]["episodes"])
        if n_episodes > 0:
            freq_matrix[:, j] /= n_episodes

    # Count chunks per procedure for labels
    proc_chunk_counts: dict[str, int] = defaultdict(int)
    for cid, proc in chunk_to_procedure.items():
        proc_chunk_counts[proc] += 1

    # Build row labels
    row_labels = [f"{p} ({proc_chunk_counts[p]} chunks)" for p in sorted_procs]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, max(3.5, n_procs * 0.5)))

    im = ax.imshow(freq_matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax,
                        label="Mean chunks retrieved per episode",
                        shrink=0.8)

    # Add text annotations in each cell
    for i in range(n_procs):
        for j in range(n_policies):
            val = freq_matrix[i, j]
            text_color = "white" if val > freq_matrix.max() * 0.65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    policy_labels = [POLICY_DISPLAY[k] for k in policy_keys]
    ax.set_xticks(range(n_policies))
    ax.set_xticklabels(policy_labels, fontsize=10)
    ax.set_yticks(range(n_procs))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Policy", fontsize=11)
    ax.set_ylabel("Procedure Code", fontsize=11)
    ax.set_title("Procedure-level retrieval frequency\n"
                 "(mean chunks retrieved per episode)",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"retrieval_heatmap_compact.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)

    return n_procs


def main() -> None:
    print("=" * 70)
    print("  Retrieval Heatmap (compact, with DPO)")
    print("=" * 70)

    apply_publication_style()

    data, all_chunk_ids, chunk_to_procedure, policy_keys = load_data()

    if not all_chunk_ids:
        print("  No chunk data found. Skipping heatmap.")
        return

    print(f"\n  Policies: {[POLICY_DISPLAY[k] for k in policy_keys]}")
    n_procs = plot_compact_heatmap(
        data, all_chunk_ids, chunk_to_procedure, policy_keys)
    print(f"  Procedures: {n_procs}, Policies: {len(policy_keys)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
