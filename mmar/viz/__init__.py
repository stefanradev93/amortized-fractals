"""Visualization and pushforward helpers for the MMAR Bayesian workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREDICTION_PURPLE = "#6F4AA8"
PREDICTION_PURPLE_DARK = "#4D2F7A"
PREDICTION_PURPLE_LIGHT = "#D8C9F0"
CALIBRATION_BAND_COLOR = "#8E6BB8"
OBSERVED_COLOR = "#202124"
SECONDARY_COLOR = "#2F7F78"
RETURN_SCALE = 100.0
RETURN_UNIT = "%"


def configure_plot_style() -> None:
    """Apply the shared presentation style used by all workflow figures."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 13,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
        }
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
    figures_dir: str | Path = "figures",
    dpi: int = 180,
) -> Path:
    """Save a figure with consistent output settings and return its path."""

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return output_path


def plot_recovery(samples, targets, labels, grid=(2, 2), interval=0.68):
    """Original-parameter recovery with posterior medians and central intervals."""
    import bayesflow as bf

    fig = bf.diagnostics.recovery(
        estimates=samples,
        targets=targets,
        variable_names=labels,
        point_agg=np.median,
        uncertainty_agg_kwargs={"prob": interval},
        color=PREDICTION_PURPLE,
        num_row=grid[0],
        num_col=grid[1],
        figsize=(4 * grid[1] + 3, 3.5 * grid[0] + 2),
    )
    fig.suptitle("Parameter Recovery", fontsize=21, y=1.01)
    return fig


def plot_calibration(samples, targets, labels, grid=(2, 2), alpha=0.1):
    """SBC rank ECDF differences with a visible simultaneous reference band."""
    import bayesflow as bf

    fig = bf.diagnostics.calibration_ecdf(
        estimates=samples,
        targets=targets,
        variable_names=labels,
        difference=True,
        stacked=False,
        rank_ecdf_color=PREDICTION_PURPLE,
        fill_color=CALIBRATION_BAND_COLOR,
        num_row=grid[0],
        num_col=grid[1],
        figsize=(4 * grid[1] + 3, 3 * grid[0] + 2),
    )
    for axis in fig.axes:
        for band in axis.collections:
            band.set_alpha(alpha)
    fig.suptitle("Simulation-Based Calibration", fontsize=21, y=1.01)
    return fig
