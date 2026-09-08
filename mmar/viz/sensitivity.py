"""Plots for the volatility-cluster sensitivity notebook."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import OBSERVED_COLOR, PREDICTION_PURPLE, PREDICTION_PURPLE_DARK

CLUSTER_COLORS = ("#4D2F7A", "#5D3B8C", "#6F4AA8", "#805BB8", "#926DC4", "#A17BCB")


def plot_recovery_grid(
    recovery: Mapping[int, tuple[np.ndarray, np.ndarray]],
    labels: Sequence[str],
    interval=0.68,
):
    """BayesFlow-style recovery diagnostics arranged as d by parameter."""
    d_values = tuple(recovery)
    fig, axes = plt.subplots(
        len(d_values), len(labels), figsize=(17, 3.25 * len(d_values)), squeeze=False
    )
    tail = (1 - interval) / 2
    for row, (d, color) in enumerate(zip(d_values, CLUSTER_COLORS, strict=True)):
        estimates, targets = recovery[d]
        medians = np.median(estimates, axis=1)
        lower, upper = np.quantile(estimates, (tail, 1 - tail), axis=1)
        for column, label in enumerate(labels):
            ax = axes[row, column]
            x, y = targets[:, column], medians[:, column]
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lower[:, column], upper[:, column] - y)),
                fmt="o",
                color=color,
                alpha=0.5,
                markersize=3.5,
                elinewidth=0.8,
            )
            limits = np.array([np.min(np.r_[x, y]), np.max(np.r_[x, y])])
            padding = 0.04 * max(limits[1] - limits[0], 1e-12)
            limits += (-padding, padding)
            ax.plot(limits, limits, color=OBSERVED_COLOR, linestyle="--", linewidth=1)
            ax.set(xlim=limits, ylim=limits)
            correlation = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.04,
                0.92,
                rf"$r={correlation:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
            if row == 0:
                ax.set_title(label, fontsize=15)
            if row < len(d_values) - 1:
                ax.tick_params(labelbottom=False)
            if column == 0:
                ax.set_ylabel(f"d = {d}\nEstimate", color=color, fontsize=12)
            ax.tick_params(labelsize=9)
    fig.suptitle(
        f"Parameter recovery by volatility resolution ({int(100 * interval)}% intervals)",
        fontsize=20,
        y=0.998,
    )
    fig.supxlabel("Ground truth", fontsize=14)
    fig.tight_layout(rect=(0.015, 0.02, 1, 0.985))
    return fig


def plot_wealth_grid(
    predictive_wealth,
    observed_returns,
    tickers,
    d_values,
    figure_title="Posterior-predictive wealth sensitivity: VOO versus SPMO",
):
    """Plot posterior-predictive wealth under every fixed d for each asset."""
    wealth = np.asarray(predictive_wealth)
    observed = np.asarray(observed_returns)
    d_values = tuple(d_values)
    if wealth.shape[:2] != (len(tickers), len(d_values)):
        raise ValueError("Predictive wealth axes must match tickers and d values.")
    observed_wealth = np.column_stack(
        (np.ones(len(observed)), np.cumprod(1 + observed, axis=1))
    )
    days = np.arange(wealth.shape[-1])
    fig, axes = plt.subplots(
        len(tickers),
        len(d_values),
        figsize=(21, 7.8),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for row, ticker in enumerate(tickers):
        for column, d in enumerate(d_values):
            ax = axes[row, column]
            q04, q16, median, q84, q96 = np.quantile(
                wealth[row, column], (0.04, 0.16, 0.5, 0.84, 0.96), axis=0
            )
            ax.fill_between(days, q04, q96, color=PREDICTION_PURPLE, alpha=0.16, label="92%")
            ax.fill_between(days, q16, q84, color=PREDICTION_PURPLE, alpha=0.30, label="68%")
            ax.plot(days, median, color=PREDICTION_PURPLE_DARK, linewidth=1.8, label="Median")
            ax.plot(
                days,
                observed_wealth[row],
                color=OBSERVED_COLOR,
                linestyle="--",
                linewidth=1.7,
                label="Observed",
            )
            if row == 0:
                ax.set_title(f"d = {d}\n{256 // d}-day blocks", fontsize=14)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Day", fontsize=11)
            if column == 0:
                ax.set_ylabel(f"{ticker}\nWealth from $1", fontsize=13)
            ax.set_xlim(0, 256)
            ax.set_xticks((0, 128, 256))
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle(figure_title, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
