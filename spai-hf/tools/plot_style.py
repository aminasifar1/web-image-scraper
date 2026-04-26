#!/usr/bin/env python3
"""Shared plotting style for all generated figures in this workspace."""

from __future__ import annotations

from matplotlib import pyplot as plt

# Global figure export config
STANDARD_DPI = 180
STANDARD_BBOX = "tight"

# Semantic colors
COLOR_REAL = "#2f9e44"
COLOR_AI = "#d9480f"
COLOR_GRID = "#d9d9d9"
COLOR_TEXT = "#1f2937"
COLOR_ACCENT = "#5f3dc4"
COLOR_THRESHOLD = "#c92a2a"

# Category palette (fixed for consistency across all plots)
CATEGORY_COLORS = {
    "news": "#1f77b4",
    "social_media": "#ff7f0e",
    "arts_illustration": "#2ca02c",
    "education_institution": "#d62728",
    "corporate": "#9467bd",
}


def apply_plot_style() -> None:
    """Apply standard style to every generated plot."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": COLOR_TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "grid.color": COLOR_GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "font.size": 10,
            "savefig.dpi": STANDARD_DPI,
            "savefig.bbox": STANDARD_BBOX,
        }
    )
