"""
Retrieval heatmap: chunk retrieval frequency per policy.

Loads data/eval_results_final.json and counts how often each chunk was
retrieved by each policy across all evaluation episodes.

Produces TWO versions:
  1. Full heatmap: top 50 individual chunks (detailed)
  2. Compact heatmap: aggregated by procedure code (paper-friendly)

Usage:
    python -m scripts.plot_retrieval_heatmap

Output:
    figures/retrieval_heatmap.png/pdf      (full version)
    figures/retrieval_heatmap_compact.png/pdf (compact version)
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
OUTPUT_DIR = "figures"

POLICY_DISPLAY = {
    "cql": "CQL",
    "iql": "IQL",
    "bc": "BC",
    "fixed_k_3": "FixedK(3)",
    "fixed_k_5": "FixedK(5)",
    "heuristic": "Heuristic",
}


def load_data():
    """Load evaluation results and compute retrieval data."""
    with open(INPUT_PATH) as f:
        data = json.load(f)

    all_chunk_ids = set()
    chunk_to_procedure = {}
    policy_keys = []

    for key in POLICY_DISPLAY:
        if key not in data:
            continue
        policy_keys.append(key)
        for ep in data[key]["episodes"]:
            proc = ep.get("procedure_code", "unknown")
            for cid in ep.get("chunk_ids", []):
                all_chunk_ids.add(cid)
                if cid not in chunk_to_procedure:
                    chunk_to_procedure[cid] = proc

    return data, all_chunk_ids, chunk_to_procedure, policy_keys


def plot_full_heatmap(data, all_chunk_ids, chunk_to_procedure, policy_keys):
    """Full heatmap with individual chunk rows (top 50)."""
    proc_groups = defaultdict(list)
    for cid in all_chunk_ids:
        proc = chunk_to_procedure.get(cid, "unknown")
        proc_groups[proc].append(cid)

    sorted_chunks = []
    for proc in sorted(proc_groups.keys()):
        for cid in sorted(proc_groups[proc]):
            sorted_chunks.append(cid)

    n_chunks = len(sorted_chunks)
    n_policies = len(policy_keys)
    chunk_idx = {cid: i for i, cid in enumerate(sorted_chunks)}

    freq_matrix = np.zeros((n_chunks, n_policies))
    for j, key in enumerate(policy_keys):
        for ep in data[key]["episodes"]:
            for cid in ep.get("chunk_ids", []):
                if cid in chunk_idx:
                    freq_matrix[chunk_idx[cid], j] += 1

    n_episodes = len(data[policy_keys[0]]["episodes"]) if policy_keys else 1
    freq_matrix /= max(n_episodes, 1)

    max_chunks_display = min(n_chunks, 50)
    if n_chunks > max_chunks_display:
        total_freq = freq_matrix.sum(axis=1)
        top_idx = np.argsort(total_freq)[-max_chunks_display:]
        top_idx = np.sort(top_idx)
        freq_matrix = freq_matrix[top_idx]
        sorted_chunks = [sorted_chunks[i] for i in top_idx]
        n_chunks = max_chunks_display

    display_chunks = []
    for cid in sorted_chunks:
        if len(cid) > 20:
            display_chunks.append(cid[:18] + "..")
        else:
            display_chunks.append(cid)

    fig_height = max(6, n_chunks * 0.25)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    im = ax.imshow(freq_matrix, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Retrieval Frequency (per episode)")

    policy_labels = [POLICY_DISPLAY[k] for k in policy_keys]
    ax.set_xticks(range(n_policies))
    ax.set_xticklabels(policy_labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(n_chunks))
    ax.set_yticklabels(display_chunks, fontsize=7)
    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("Chunk ID", fontsize=12)
    ax.set_title("Chunk Retrieval Frequency by Policy",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"retrieval_heatmap.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)

    return n_chunks, n_policies


def plot_compact_heatmap(data, all_chunk_ids, chunk_to_procedure, policy_keys):
    """Compact heatmap aggregated by procedure code."""
    n_policies = len(policy_keys)

    # Count retrieval frequency per procedure per policy
    proc_set = set(chunk_to_procedure.values())
    sorted_procs = sorted(proc_set)
    n_procs = len(sorted_procs)
    proc_idx = {p: i for i, p in enumerate(sorted_procs)}

    # Matrix: (n_procs x n_policies) = total retrievals from that procedure
    freq_matrix = np.zeros((n_procs, n_policies))
    # Also count unique chunks retrieved per procedure per policy
    unique_matrix = np.zeros((n_procs, n_policies))

    for j, key in enumerate(policy_keys):
        proc_chunks = defaultdict(set)
        for ep in data[key]["episodes"]:
            for cid in ep.get("chunk_ids", []):
                proc = chunk_to_procedure.get(cid)
                if proc and proc in proc_idx:
                    freq_matrix[proc_idx[proc], j] += 1
                    proc_chunks[proc].add(cid)
        for proc, chunks in proc_chunks.items():
            unique_matrix[proc_idx[proc], j] = len(chunks)

    n_episodes = len(data[policy_keys[0]]["episodes"]) if policy_keys else 1
    freq_matrix /= max(n_episodes, 1)

    # Count chunks per procedure for labels
    proc_chunk_counts = defaultdict(int)
    for cid, proc in chunk_to_procedure.items():
        proc_chunk_counts[proc] += 1

    # Build row labels
    row_labels = [f"{p} ({proc_chunk_counts[p]} chunks)" for p in sorted_procs]

    fig, ax = plt.subplots(figsize=(7, max(3.5, n_procs * 0.5)))

    im = ax.imshow(freq_matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, label="Mean Retrievals (per episode)", shrink=0.8)

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
    ax.set_title("Retrieval Frequency by Procedure and Policy",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"retrieval_heatmap_compact.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)

    return n_procs


def main() -> None:
    print("=" * 70)
    print("  Retrieval Heatmap (R43)")
    print("=" * 70)

    data, all_chunk_ids, chunk_to_procedure, policy_keys = load_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not all_chunk_ids:
        print("  No chunk data found. Skipping heatmap.")
        return

    print("\n  --- Full heatmap ---")
    n_chunks, n_policies = plot_full_heatmap(
        data, all_chunk_ids, chunk_to_procedure, policy_keys)
    print(f"  Unique chunks shown: {n_chunks}, Policies: {n_policies}")

    print("\n  --- Compact heatmap ---")
    n_procs = plot_compact_heatmap(
        data, all_chunk_ids, chunk_to_procedure, policy_keys)
    print(f"  Procedures: {n_procs}, Policies: {n_policies}")

    print("=" * 70)


if __name__ == "__main__":
    main()
