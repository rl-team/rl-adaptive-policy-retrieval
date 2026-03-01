"""
Generate a publication-quality workflow diagram for the CS234 poster.

Shows the PA authorization flow:
    Provider submits request → RL Agent retrieval loop
    (observe state → select chunk/stop → environment step) → Oracle decision

The diagram uses matplotlib patches and straight arrows.
Outputs suggested LaTeX caption to console.

Usage:
    python -m scripts.plot_poster_workflow

Output:
    figures/poster_workflow.png  (300 DPI)
    figures/poster_workflow.pdf  (vector)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------------------------------------------------------
# Palette & Fonts
# ---------------------------------------------------------------------------

COLORS = {
    "provider":   "#3498db",   # blue
    "request":    "#2980b9",   # darker blue
    "environment":"#27ae60",   # green
    "agent":      "#e74c3c",   # red
    "oracle":     "#f39c12",   # amber
    "outcome":    "#8e44ad",   # purple
    "arrow":      "#2c3e50",   # dark slate
    "loop_bg":    "#f8f9fa",   # very light grey
    "label":      "#2c3e50",   # dark descriptive text (contrast)
}

FONT = {"family": "sans-serif", "weight": "bold", "size": 11}
FONT_LABEL = {"family": "sans-serif", "size": 9, "color": COLORS["label"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rounded_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    text_color: str = "white",
    fontdict: dict | None = None,
) -> FancyBboxPatch:
    """Draw a rounded rectangle with centred text. x,y is BOTTOM LEFT."""
    fd = fontdict or FONT
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2, y + height / 2, text,
        fontdict=fd,
        ha="center", va="center",
        color=text_color,
        zorder=5,
    )
    return box


def _straight_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    color: str = COLORS["arrow"],
    shift_y: float = 0.0,
    bbox: bool = True,
) -> None:
    """Draw a straight arrow with an optional label shifted above the midpoint."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->,head_length=6,head_width=4",
        color=color,
        linewidth=1.8,
        zorder=4,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + shift_y
        kwargs = {
            "fontdict": FONT_LABEL,
            "ha": "center",
            "va": "center" if shift_y == 0.0 else "bottom",
            "zorder": 5,
        }
        if bbox:
            kwargs["bbox"] = dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=1.0)
        ax.text(mx, my, label, **kwargs)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_workflow() -> None:
    """Create the poster workflow diagram."""
    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(-0.2, 14.5)
    ax.set_ylim(-0.2, 5.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    nodes = {}
    def _add_node(name, xy, width, height, text, color, fontdict=None):
        _rounded_box(ax, xy, width, height, text, color, fontdict=fontdict)
        nodes[name] = {"x": xy[0], "y": xy[1], "w": width, "h": height}

    def _conn(name, side, ratio=0.5, pad=0.15):
        n = nodes[name]
        l, r = n["x"] - pad, n["x"] + n["w"] + pad
        b, t = n["y"] - pad, n["y"] + n["h"] + pad
        if side == 'left': return (l, b + ratio * (t - b))
        if side == 'right': return (r, b + ratio * (t - b))
        if side == 'bottom': return (l + ratio * (r - l), b)
        if side == 'top': return (l + ratio * (r - l), t)

    # ── Step 1: Provider ────────────────────────────────────────────
    _add_node("provider", (0.1, 2.1), 1.8, 0.8, "Provider", COLORS["provider"])
    ax.text(1.0, 1.9, "submits PA request", fontdict=FONT_LABEL, ha="center", va="top")

    # ── Step 2: PA Request ──────────────────────────────────────────
    _add_node("request", (2.7, 2.1), 1.8, 0.8, "PA Request", COLORS["request"])
    ax.text(3.6, 1.9, "procedure, diagnosis,\npatient context", fontdict=FONT_LABEL, ha="center", va="top")

    # ── RL Retrieval Loop (Background) ──────────────────────────────
    loop_bg = FancyBboxPatch(
        (5.1, 0.5), 4.7, 3.8,
        boxstyle="round,pad=0.3",
        facecolor=COLORS["loop_bg"],
        edgecolor="#bdc3c7",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(loop_bg)
    ax.text(7.45, 4.45, "RL Retrieval Loop", fontdict={**FONT, "size": 12, "color": COLORS["agent"]}, ha="center", va="center")

    # -- State observation (Top Middle) --
    _add_node("state", (6.7, 3.05), 1.6, 0.8, r"State  $s_t$", COLORS["environment"])
    ax.text(7.5, 4.1, "request emb + chunk history emb", fontdict=FONT_LABEL, ha="center", va="center")

    # -- Agent selects action (Bottom Left) --
    _add_node("agent", (5.4, 1.05), 1.6, 0.8, r"Agent  $\pi(a|s)$", COLORS["agent"])
    ax.text(6.2, 0.85, r"$a \in \{chunk_0, \dots, chunk_9, \mathrm{STOP}\}$", fontdict=FONT_LABEL, ha="center", va="top")

    # -- Environment step (Bottom Right) --
    _add_node("env", (8.1, 1.05), 1.4, 0.8, "Env step", COLORS["environment"])
    ax.text(8.8, 0.85, r"$r_t = -\lambda$" + "\n" + r"$s_{t+1}$", fontdict=FONT_LABEL, ha="center", va="top")

    # ── Step 3: Oracle ──────────────────────────────────────────────
    _add_node("oracle", (10.6, 2.1), 1.4, 0.8, "Oracle", COLORS["oracle"])
    ax.text(11.3, 1.9, "deterministic\nrule-based", fontdict=FONT_LABEL, ha="center", va="top")

    # ── Step 4: Outcomes ────────────────────────────────────────────
    _add_node("approve", (12.9, 3.25), 0.9, 0.5, "Approve", COLORS["outcome"], fontdict={**FONT, "size": 10})
    _add_node("deny", (12.9, 2.25), 0.9, 0.5, "Deny", COLORS["outcome"], fontdict={**FONT, "size": 10})
    _add_node("pend", (12.9, 1.25), 0.9, 0.5, "Pend", COLORS["outcome"], fontdict={**FONT, "size": 10})

    # ── Arrows ──────────────────────────────────────────────────────
    _straight_arrow(ax, _conn("provider", "right", 0.5), _conn("request", "left", 0.5), "generate", shift_y=0.1, bbox=False)
    _straight_arrow(ax, _conn("request", "right", 0.5), _conn("state", "left", 0.5))

    # RL Loop Triangle
    _straight_arrow(ax, _conn("state", "bottom", 0.2), _conn("agent", "top", 0.5), "observe", bbox=True)
    _straight_arrow(ax, _conn("agent", "right", 0.5), _conn("env", "left", 0.5), "act", bbox=True)
    _straight_arrow(ax, _conn("env", "top", 0.5), _conn("state", "bottom", 0.8), "transition", bbox=True)

    # STOP branch
    _straight_arrow(ax, _conn("env", "right", 0.5), _conn("oracle", "left", 0.5), "STOP action", bbox=True)

    # Oracle -> Outcomes
    _straight_arrow(ax, _conn("oracle", "right", 0.8), _conn("approve", "left", 0.5))
    _straight_arrow(ax, _conn("oracle", "right", 0.5), _conn("deny", "left", 0.5))
    _straight_arrow(ax, _conn("oracle", "right", 0.2), _conn("pend", "left", 0.5))

    # ── Save ────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.5)

    png_path = os.path.join("figures", "poster_workflow.png")
    pdf_path = os.path.join("figures", "poster_workflow.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

latex_caption = r"""
% ---------------------------------------------------------
% Paste this caption block into your Overleaf LaTeX document
% ---------------------------------------------------------
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/poster_workflow.pdf}
    \caption{\textbf{Adaptive Policy Retrieval for Prior Authorization --- System Flow.}
    A PA request is generated by the provider. The RL agent observes the state
    (concatenation of request and chunk history embeddings) and iteratively 
    retrieves policy chunks. A step cost $r_t = -\lambda$ (where $\lambda \in \{0.05, 0.1, 0.2\}$) 
    penalises long retrievals. The episode ends when the agent takes the STOP action, 
    at which point an Oracle gives a deterministic rule-based outcome. A terminal reward of 
    $r_T = +1$ (correct) or $-1$ (incorrect) is provided.}
    \label{fig:poster_workflow}
\end{figure}
% ---------------------------------------------------------
"""

def main() -> None:
    print("=" * 60)
    print("  Generating poster workflow diagram")
    print("=" * 60)
    plot_workflow()
    print("\n" + latex_caption)
    print("  Done.")


if __name__ == "__main__":
    main()
