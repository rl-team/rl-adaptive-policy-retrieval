"""
Publication-quality matplotlib style for figures.

Use apply_publication_style() at the start of a plotting script so all
figures have consistent fonts, sizes, and DPI. Suitable for papers and reports.
"""
from __future__ import annotations
import matplotlib.pyplot as plt

# Default DPI for PNG exports (publication: 300–600)
PUBLICATION_DPI = 300

def apply_publication_style(dpi: int = PUBLICATION_DPI) -> None:
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
