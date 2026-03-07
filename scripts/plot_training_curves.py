"""
Generate publication-quality training curves from sweep data.

By default, reads pre-extracted scalar data from data/sweep_results.json
(committed to the repo for reproducibility). Use --from-tensorboard to
regenerate the JSON from raw Tensorboard event files in runs/.

Produces TWO separate figures:
  1. CQL hyperparameter sweep (alpha/lr) + primary CQL/IQL runs
  2. Lambda (step_cost) sweep

Usage:
    python -m scripts.plot_training_curves
    python -m scripts.plot_training_curves --from-tensorboard

Output:
    figures/training_curves_cql_sweep.png/pdf
    figures/training_curves_lambda_sweep.png/pdf
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# All Tensorboard log directories to scan
SWEEP_RUNS = {
    # Alpha/lr sweep (original)
    "sweep_alpha1.0_lr3e-4": "runs/sweep_alpha1.0_lr3e-4/tb",
    "sweep_alpha0.5_lr1e-3": "runs/sweep_alpha0.5_lr1e-3/tb",
    "sweep_alpha0.1_lr3e-4": "runs/sweep_alpha0.1_lr3e-4/tb",
    # Lambda sweep
    "sweep_lambda_0.05": "runs/sweep_lambda_0.05/tb",
    "sweep_lambda_0.1": "runs/sweep_lambda_0.1/tb",
    "sweep_lambda_0.2": "runs/sweep_lambda_0.2/tb",
    # Primary 2k runs
    "cql_2k": "runs/cql_2k/tb",
    "iql_2k": "runs/iql_2k/tb",
}

# Display labels
DISPLAY_LABELS = {
    "sweep_alpha0.5_lr1e-3": r"CQL $\alpha=0.5$, lr=1e-3 (best)",
    "sweep_alpha1.0_lr3e-4": r"CQL $\alpha=1.0$, lr=3e-4",
    "sweep_alpha0.1_lr3e-4": r"CQL $\alpha=0.1$, lr=3e-4",
    "sweep_lambda_0.05": r"CQL $\lambda=0.05$",
    "sweep_lambda_0.1": r"CQL $\lambda=0.1$",
    "sweep_lambda_0.2": r"CQL $\lambda=0.2$",
    "cql_2k": "CQL (2k corpus)",
    "iql_2k": "IQL (2k corpus)",
}

COLORS = {
    "sweep_alpha0.5_lr1e-3": "#2ecc71",
    "sweep_alpha1.0_lr3e-4": "#3498db",
    "sweep_alpha0.1_lr3e-4": "#e74c3c",
    "sweep_lambda_0.05": "#3498db",
    "sweep_lambda_0.1": "#2ecc71",
    "sweep_lambda_0.2": "#e74c3c",
    "cql_2k": "#2c3e50",
    "iql_2k": "#e67e22",
}

CQL_METRICS = [
    ("loss/total", "Total Loss"),
    ("loss/td", "TD Loss (Bellman Error)"),
    ("loss/conservative_penalty", "Conservative Penalty"),
    ("q_values/mean", "Mean Q-Value"),
]

# CQL sweep group: alpha/lr sweep + primary runs
CQL_SWEEP_RUNS = [
    "sweep_alpha0.5_lr1e-3",
    "sweep_alpha1.0_lr3e-4",
    "sweep_alpha0.1_lr3e-4",
    "cql_2k",
    "iql_2k",
]

# Lambda sweep group
LAMBDA_SWEEP_RUNS = [
    "sweep_lambda_0.05",
    "sweep_lambda_0.1",
    "sweep_lambda_0.2",
]

JSON_PATH = "data/sweep_results.json"
OUTPUT_DIR = "figures"
EMA_WEIGHT = 0.9


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_json(path: str) -> dict:
    """Load pre-extracted scalar data from JSON.

    Returns dict keyed by run_name (not display label) with
    {tag: (steps, values)} entries.
    """
    with open(path) as f:
        raw = json.load(f)

    result = {}
    for run_name in raw:
        result[run_name] = {}
        for tag, points in raw[run_name].items():
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            result[run_name][tag] = (steps, values)
    return result


def load_from_tensorboard() -> dict:
    """Load scalar data directly from Tensorboard event files."""
    from tensorboard.backend.event_processing import event_accumulator

    raw = {}
    result = {}

    for run_name, log_dir in SWEEP_RUNS.items():
        if not os.path.isdir(log_dir):
            print(f"  Skipped {run_name}: {log_dir} not found")
            continue

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        raw[run_name] = {}
        result[run_name] = {}

        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            raw[run_name][tag] = list(zip(steps, values))
            result[run_name][tag] = (steps, values)

        label = DISPLAY_LABELS.get(run_name, run_name)
        print(f"  Loaded {label}: {len(result[run_name])} tags")

    # Write JSON for reproducibility
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  Updated {JSON_PATH}")

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def smooth(values: list, weight: float = EMA_WEIGHT) -> list:
    """Exponential moving average for smoothing noisy curves."""
    smoothed = []
    val = values[0]
    for v in values:
        val = weight * val + (1 - weight) * v
        smoothed.append(val)
    return smoothed


def plot_group(
    all_data: dict,
    run_names: list,
    metrics: list,
    title: str,
    filename_base: str,
) -> None:
    """Create a 2x2 panel figure for a group of runs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for (tag, panel_title), ax in zip(metrics, flat_axes):
        for run_name in run_names:
            if run_name not in all_data:
                continue
            run_data = all_data[run_name]
            if tag not in run_data:
                continue
            color = COLORS.get(run_name, "#666666")
            label = DISPLAY_LABELS.get(run_name, run_name)
            steps, values = run_data[tag]

            ax.plot(steps, values, color=color, alpha=0.15, linewidth=0.5)
            ax.plot(steps, smooth(values), color=color,
                    linewidth=1.8, label=label)

        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Shared legend below the figure
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center",
            ncol=min(3, len(handles)), fontsize=9,
            frameon=True, bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training curves figures.",
    )
    parser.add_argument(
        "--from-tensorboard", action="store_true",
        help="Read from Tensorboard event files instead of JSON "
             "(also updates data/sweep_results.json).",
    )
    args = parser.parse_args()

    if args.from_tensorboard:
        all_data = load_from_tensorboard()
    else:
        if not os.path.isfile(JSON_PATH):
            print(f"  {JSON_PATH} not found. Run with --from-tensorboard first.")
            return
        all_data = load_from_json(JSON_PATH)
        print(f"  Loaded from {JSON_PATH}")

    if not all_data:
        print("No data found.")
        return

    print("\n  --- CQL Hyperparameter Sweep ---")
    plot_group(
        all_data,
        run_names=CQL_SWEEP_RUNS,
        metrics=CQL_METRICS,
        title=r"Training Curves: CQL Hyperparameter Sweep ($\alpha$, lr)",
        filename_base="training_curves_cql_sweep",
    )

    print("\n  --- Lambda Sweep ---")
    plot_group(
        all_data,
        run_names=LAMBDA_SWEEP_RUNS,
        metrics=CQL_METRICS,
        title=r"Training Curves: Step Cost ($\lambda$) Sweep",
        filename_base="training_curves_lambda_sweep",
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
