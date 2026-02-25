"""
Generate publication-quality training curves from sweep data.

By default, reads pre-extracted scalar data from data/sweep_results.json
(committed to the repo for reproducibility). Use --from-tensorboard to
regenerate the JSON from raw Tensorboard event files in runs/.

Usage:
    python -m scripts.plot_training_curves
    python -m scripts.plot_training_curves --from-tensorboard

Output:
    figures/training_curves.png (300 DPI)
    figures/training_curves.pdf (vector)
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

# Mapping from display labels to Tensorboard log directories (used only
# with --from-tensorboard).
SWEEP_RUNS = {
    "sweep_alpha1.0_lr3e-4": "runs/sweep_alpha1.0_lr3e-4/tb",
    "sweep_alpha0.5_lr1e-3": "runs/sweep_alpha0.5_lr1e-3/tb",
    "sweep_alpha0.1_lr3e-4": "runs/sweep_alpha0.1_lr3e-4/tb",
}

# Display labels keyed by run directory name (used for legend).
# Ordered best-to-worst so colors are intuitive:
#   green = best policy, blue = middle, red = worst.
DISPLAY_LABELS = {
    "sweep_alpha0.5_lr1e-3": r"$\alpha=0.5$, lr=1e-3 (best)",
    "sweep_alpha1.0_lr3e-4": r"$\alpha=1.0$, lr=3e-4",
    "sweep_alpha0.1_lr3e-4": r"$\alpha=0.1$, lr=3e-4",
}

# Semantic colors: green = best (highest accuracy + efficiency),
# blue = middle (accurate but over-retrieves),
# red = worst (inaccurate, stops too early).
COLORS = ["#2ecc71", "#3498db", "#e74c3c"]

METRICS = [
    ("loss/total", "Total Loss"),
    ("loss/td", "TD Loss (Bellman Error)"),
    ("loss/conservative_penalty", "Conservative Penalty"),
    ("q_values/mean", "Mean Q-Value"),
]

JSON_PATH = "data/sweep_results.json"
OUTPUT_DIR = "figures"
EMA_WEIGHT = 0.9  # Exponential moving average smoothing


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_json(path: str) -> dict:
    """Load pre-extracted scalar data from JSON.

    Returns dict keyed by display label, each containing tag -> (steps, values).
    Ordering follows DISPLAY_LABELS (best-to-worst) for consistent legend colors.
    """
    with open(path) as f:
        raw = json.load(f)

    result = {}
    for run_name, label in DISPLAY_LABELS.items():
        if run_name not in raw:
            continue
        result[label] = {}
        for tag, points in raw[run_name].items():
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            result[label][tag] = (steps, values)
    return result


def load_from_tensorboard() -> dict:
    """Load scalar data directly from Tensorboard event files.

    Also updates data/sweep_results.json for other teammates.
    """
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
        label = DISPLAY_LABELS[run_name]
        result[label] = {}

        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            raw[run_name][tag] = list(zip(steps, values))
            result[label][tag] = (steps, values)

        print(f"  Loaded {label}: {len(result[label])} tags")

    # Write JSON for reproducibility
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  Updated {JSON_PATH}")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(values: list, weight: float = EMA_WEIGHT) -> list:
    """Exponential moving average for smoothing noisy curves."""
    smoothed = []
    val = values[0]
    for v in values:
        val = weight * val + (1 - weight) * v
        smoothed.append(val)
    return smoothed


def plot(all_data: dict) -> None:
    """Create 4-panel training curves figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    fig.suptitle(
        "Conservative Q-Learning training curves (hyperparameter sweep)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for (tag, title), ax in zip(METRICS, flat_axes):
        for (label, run_data), color in zip(all_data.items(), COLORS):
            if tag not in run_data:
                continue
            steps, values = run_data[tag]
            ax.plot(steps, values, color=color, alpha=0.3, linewidth=0.8)
            ax.plot(
                steps, smooth(values), color=color,
                linewidth=2.0, label=label,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training epoch", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    # Shared legend
    handles, labels_leg = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg, loc="lower center", ncol=3, fontsize=9,
        frameon=True, bbox_to_anchor=(0.5, 0.02),
    )
    fig.text(
        0.5, 0.005,
        "Figure 2. Conservative Q-Learning training curves (hyperparameter sweep). Solid lines: exponential moving average (α=0.9). Best config (α=0.5, lr=1e-3) balances accuracy and efficiency.",
        ha="center", fontsize=8, style="italic",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    png_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    pdf_path = os.path.join(OUTPUT_DIR, "training_curves.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training curves figure.",
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

    plot(all_data)


if __name__ == "__main__":
    main()
