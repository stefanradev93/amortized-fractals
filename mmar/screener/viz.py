"""Presentation plots; return each figure once and save it in figures/screener."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .. import univariate as model
from .screen import rank_stocks

PURPLE, TEAL, GREY = "#6841a5", "#238681", "#abb0b9"
STYLE = {
    "axes.unicode_minus": False,
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def batch_plot(series, dates, path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
        values = 100 * series[..., 0]
        limit = np.quantile(np.abs(values), 0.98)
        im = ax.imshow(
            values,
            aspect="auto",
            cmap="PuOr",
            vmin=-limit,
            vmax=limit,
            extent=[mdates.date2num(dates[0]), mdates.date2num(dates[-1]), len(values), 0],
        )
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set(
            xlabel="Trading date",
            ylabel="Stock (alphabetical order)",
            title=f"One batch of {len(values)} stocks: 256 days x 1 channel",
        )
        fig.colorbar(
            im, ax=ax, label="Daily return (%); color clipped at 98th percentile", extend="both"
        )
        return _save(fig, path)


def opportunity_plot(table, path):
    table = rank_stocks(table)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
        groups = [
            (~table.coverage_ok, GREY, "Model mismatch warning"),
            (table.coverage_ok, PURPLE, "Within model ranges"),
        ]
        for mask, color, label in groups:
            subset = table.loc[mask]
            axes[0].scatter(
                subset.predictive_es95_pct,
                100 * subset.predictive_gain_probability,
                c=color,
                label=f"{label} ({len(subset)})",
                alpha=0.7,
                s=32,
            )
        axes[0].set(
            xlabel="20-day expected shortfall (%)",
            ylabel="Predictive probability of net gain (%)",
            title="Upside versus downside",
        )
        axes[0].legend()
        top = table.head(12).iloc[::-1]
        y = np.arange(len(top))
        axes[1].hlines(
            y,
            top.gain_to_es_posterior_q05,
            top.gain_to_es_posterior_q95,
            color=PURPLE,
            lw=3,
            alpha=0.65,
            label="90% posterior interval",
        )
        axes[1].scatter(
            top.gain_to_es,
            y,
            color=np.where(top.coverage_ok, TEAL, GREY),
            s=65,
            zorder=3,
        )
        axes[1].set(yticks=y, yticklabels=top.ticker)
        axes[1].set_xlim(left=0)
        axes[1].set(
            xlabel="Gain probability (%) / expected shortfall (%)",
            title="Top 12: gain probability per unit of risk",
        )
        for ax in axes:
            ax.grid(alpha=0.15)
        return _save(fig, path)


def parameter_plot(table, path):
    top = table.head(12).iloc[::-1]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True, layout="constrained")
        for ax, name, label, scale in zip(
            axes,
            model.PARAMETER_NAMES,
            ("Daily drift (%)", "Scale (%)", "Cascade q", "Tail nu"),
            (100, 100, 1, 1),
        ):
            y = np.arange(len(top))
            ax.hlines(
                y,
                scale * top[f"{name}_q05"],
                scale * top[f"{name}_q95"],
                color=PURPLE,
                alpha=0.6,
                lw=3,
            )
            ax.scatter(scale * top[f"{name}_median"], y, color=PURPLE)
            ax.set(xlabel=label, yticks=y, yticklabels=top.ticker)
            ax.grid(alpha=0.15)
        fig.suptitle("Top-ranked stocks: posterior medians and 90% intervals", fontsize=18)
        return _save(fig, path)


def fit_plot(posterior, series, metadata, table, dates, path, seed=None, figsize=(12, 6)):
    """One replicated 256-day path per posterior draw for the highest-ranked stock."""
    ticker = table.ticker.iloc[0]
    index = metadata.index[metadata.ticker == ticker][0]
    rng = np.random.default_rng(seed)
    draws = posterior[index]
    paths = model.simulate_from_parameters(draws, rng)["returns"]
    observed = series[index, :, 0]
    wealth = np.column_stack((np.ones(len(paths)), np.cumprod(1 + paths, axis=1)))
    observed_wealth = np.r_[1, np.cumprod(1 + observed)]
    drawdowns = 100 * (1 - np.min(wealth / np.maximum.accumulate(wealth, axis=1), axis=1))
    observed_drawdown = 100 * (1 - np.min(observed_wealth / np.maximum.accumulate(observed_wealth)))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
        days = np.arange(model.WINDOW + 1)
        for lo, hi, alpha, label in [
            (0.04, 0.96, 0.15, "92% predictive interval"),
            (0.16, 0.84, 0.3, "68% predictive interval"),
        ]:
            bands = np.quantile(wealth, [lo, hi], axis=0)
            axes[0].fill_between(days, *bands, color=PURPLE, alpha=alpha, label=label)
        axes[0].plot(days, np.median(wealth, axis=0), color=PURPLE, label="Predictive median", lw=2)
        axes[0].plot(days, observed_wealth, color="black", ls="--", label="Observed", lw=2)
        axes[0].set(
            xlabel="Day within observed window",
            ylabel="Wealth from $1",
            title=f"{ticker}: cumulative wealth",
        )
        axes[0].axhline(1)
        axes[0].legend()
        limits = np.quantile(np.r_[paths.ravel(), observed], [0.001, 0.999]) * 100
        bins = np.linspace(*limits, 60)
        # Weights, rather than density=True, preserve mass outside the displayed range.
        width = bins[1] - bins[0]
        axes[1].hist(
            100 * paths.ravel(),
            bins=bins,
            weights=np.full(paths.size, 1 / paths.size / width),
            color=PURPLE,
            alpha=0.35,
            label="Predictive",
        )

        axes[1].hist(
            100 * observed,
            bins=bins,
            weights=np.full(len(observed), 1 / len(observed) / width),
            histtype="step",
            color="black",
            linestyle="--",
            lw=2,
            label="Observed",
        )
        axes[1].set(
            xlabel="Daily return (%); central 99.8% display",
            ylabel="Density",
            title="Marginal return check",
        )
        axes[1].axvline(100 * np.median(paths), color=PURPLE, lw=2, label="Predictive median")
        axes[1].legend()
        axes[2].hist(
            drawdowns, bins=35, color=PURPLE, alpha=0.35, label="One path per posterior draw"
        )
        axes[2].axvline(np.median(drawdowns), color=PURPLE, lw=2, label="Predictive median")
        axes[2].axvline(observed_drawdown, color="black", ls="--", lw=2, label="Observed")
        axes[2].set(
            xlabel="Maximum drawdown (% loss)",
            ylabel="Number of draws",
            title="Maximum drawdown over 256 days",
        )
        axes[2].legend()
        fig.suptitle(
            f"Latest 256-day window ending {dates[-1]:%Y-%m-%d}",
            fontsize=17,
        )
        return _save(fig, path)
