"""Univariate prior checks, posterior predictive checks and rolling risk."""
from typing import Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import (
    PREDICTION_PURPLE,
    PREDICTION_PURPLE_DARK,
    PREDICTION_PURPLE_LIGHT,
    OBSERVED_COLOR,
    RETURN_SCALE,
    RETURN_UNIT,
)


def plot_q_turbulence(paths_by_q: Mapping[float, np.ndarray]) -> plt.Figure:
    """Plot matched return paths across values of ``q``."""

    n_panels = len(paths_by_q)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for ax, (q, paths) in zip(axes.ravel(), paths_by_q.items()):
        for path in paths:
            ax.plot(path, color=PREDICTION_PURPLE, linewidth=1.15, alpha=0.68)
        ax.axhline(0.0, color=OBSERVED_COLOR, linewidth=0.8, alpha=0.55)
        ax.set_title(f"q = {q:.2f}", fontsize=15)

    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)

    fig.suptitle("How q Changes Simulations", fontsize=18, y=0.99)
    fig.supxlabel("Trading day", fontsize=14)
    fig.supylabel(f"Daily return ({RETURN_UNIT})", fontsize=14)
    fig.tight_layout()
    return fig


def _common_return_bins(
    prior_returns_pct: np.ndarray,
    observed_returns_pct: pd.DataFrame,
    n_bins: int,
) -> np.ndarray:
    combined = np.concatenate(
        [prior_returns_pct.reshape(-1), observed_returns_pct.to_numpy().reshape(-1)]
    )
    lo, hi = np.nanquantile(combined, [0.001, 0.999])
    bound = max(abs(lo), abs(hi))
    return np.linspace(-bound, bound, n_bins + 1)


def _prior_histogram_envelope(
    prior_returns_pct: np.ndarray,
    bins: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    histograms = np.empty((prior_returns_pct.shape[0], len(bins) - 1), dtype="float64")
    for i, path in enumerate(prior_returns_pct):
        histograms[i] = np.histogram(path, bins=bins, density=True)[0]
    return np.quantile(histograms, quantiles, axis=0)


def plot_marginal_return_histograms(
    prior_returns_pct: np.ndarray,
    observed_returns_pct: pd.DataFrame,
    tickers: Sequence[str],
    n_bins: int = 90,
    envelope_quantiles: Sequence[float] = (0.05, 0.50, 0.95),
) -> plt.Figure:
    """Compare observed return histograms with the prior predictive envelope."""

    bins = _common_return_bins(prior_returns_pct, observed_returns_pct, n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    q_low, q_mid, q_high = _prior_histogram_envelope(prior_returns_pct, bins, envelope_quantiles)
    date_range_label = (
        f"{observed_returns_pct.index.min():%b %d, %Y}–"
        f"{observed_returns_pct.index.max():%b %d, %Y}"
    )

    fig, axes = plt.subplots(1, len(tickers), figsize=(6.2 * len(tickers), 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, ticker in zip(axes, tickers):
        observed_density = np.histogram(
            observed_returns_pct[ticker].dropna(), bins=bins, density=True
        )[0]
        ax.fill_between(
            centers,
            q_low,
            q_high,
            color=PREDICTION_PURPLE,
            alpha=0.24,
            label="Prior 90% envelope",
        )
        ax.plot(
            centers,
            q_mid,
            color=PREDICTION_PURPLE_DARK,
            linewidth=2.2,
            label="Prior median",
        )
        ax.step(
            centers,
            observed_density,
            where="mid",
            color=OBSERVED_COLOR,
            linewidth=2.0,
            label=f"{ticker} observed",
        )
        ax.axvline(0.0, color="#636363", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{ticker}: Marginal Daily Returns\n({date_range_label})", fontsize=15)
        ax.set_xlabel(f"Daily return ({RETURN_UNIT})", fontsize=13)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=11)

    fig.tight_layout()
    return fig


def plot_max_drawdown_checks(
    prior_stats: pd.DataFrame,
    observed_stats: pd.DataFrame,
    tickers: Sequence[str],
    observed_date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> plt.Figure:
    """Compare observed maximum drawdowns with the prior pushforward."""

    prior_drawdowns = prior_stats["max_drawdown_pct"].dropna().to_numpy()
    observed_drawdowns = observed_stats.loc[list(tickers), "max_drawdown_pct"]
    prior_q05, prior_q95 = np.quantile(prior_drawdowns, [0.05, 0.95])
    display_low, display_high = np.quantile(prior_drawdowns, [0.001, 0.999])
    display_low = min(display_low, float(observed_drawdowns.min()))
    display_high = max(display_high, float(observed_drawdowns.max()))
    padding = 0.04 * max(display_high - display_low, 1e-6)
    bins = np.linspace(display_low - padding, min(0.0, display_high + padding), 46)

    date_label = ""
    if observed_date_range is not None:
        date_label = f"{observed_date_range[0]:%b %d, %Y}–" f"{observed_date_range[1]:%b %d, %Y}"

    fig, axes = plt.subplots(1, len(tickers), figsize=(6.2 * len(tickers), 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, ticker in zip(axes, tickers):
        observed_value = float(observed_drawdowns.loc[ticker])
        ax.axvspan(
            prior_q05,
            prior_q95,
            color=PREDICTION_PURPLE_LIGHT,
            alpha=0.55,
            label="Prior 90% interval",
            zorder=0,
        )
        ax.hist(
            prior_drawdowns,
            bins=bins,
            density=True,
            color=PREDICTION_PURPLE,
            alpha=0.65,
            edgecolor=PREDICTION_PURPLE_DARK,
            linewidth=0.5,
            label="Prior predictive",
            zorder=1,
        )
        ax.axvline(
            observed_value,
            color=OBSERVED_COLOR,
            linewidth=2.5,
            label=f"Observed: {observed_value:.1f}%",
            zorder=2,
        )
        title = ticker if not date_label else f"{ticker}\n({date_label})"
        ax.set_title(title, fontsize=15)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False, axis="y")
        ax.legend(frameon=False, fontsize=11)

    fig.suptitle("Maximum Drawdown", fontsize=18, y=0.99)
    fig.supxlabel(f"Maximum drawdown ({RETURN_UNIT})", fontsize=14, y=0.01)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    return fig


def _predictive_bands(paths: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q04": np.quantile(paths, 0.04, axis=0),
        "q16": np.quantile(paths, 0.16, axis=0),
        "median": np.quantile(paths, 0.50, axis=0),
        "q84": np.quantile(paths, 0.84, axis=0),
        "q96": np.quantile(paths, 0.96, axis=0),
    }


def plot_posterior_predictive_checks(
    posterior_paths: np.ndarray,
    observed_windows: np.ndarray,
    dates: Sequence[pd.DatetimeIndex],
    tickers: Sequence[str],
    n_resimulations: int = 12,
) -> plt.Figure:
    """Plot return, wealth, and marginal-distribution PPCs by asset."""

    paths = np.asarray(posterior_paths, dtype="float64")
    observed = np.asarray(observed_windows, dtype="float64")
    if paths.ndim != 3 or observed.ndim != 2:
        raise ValueError("Expected paths (asset, draw, time) and observations (asset, time).")
    if paths.shape[0] != len(tickers) or observed.shape[0] != len(tickers):
        raise ValueError("Ticker count must match the first array dimension.")

    fig, axes = plt.subplots(
        len(tickers),
        3,
        figsize=(19, 4.6 * len(tickers)),
        squeeze=False,
    )

    for asset_id, (ticker, ticker_dates) in enumerate(zip(tickers, dates)):
        return_paths = RETURN_SCALE * paths[asset_id]
        observed_returns = RETURN_SCALE * observed[asset_id]
        return_bands = _predictive_bands(return_paths)

        wealth_paths = np.cumprod(1.0 + paths[asset_id], axis=1)
        observed_wealth = np.cumprod(1.0 + observed[asset_id])
        wealth_bands = _predictive_bands(wealth_paths)

        for col, (simulated, observed_series, bands, ylabel, title) in enumerate(
            [
                (
                    return_paths,
                    observed_returns,
                    return_bands,
                    f"Daily return ({RETURN_UNIT})",
                    "Daily returns",
                ),
                (
                    wealth_paths,
                    observed_wealth,
                    wealth_bands,
                    "Growth of $1",
                    "Cumulative wealth",
                ),
            ]
        ):
            ax = axes[asset_id, col]
            ax.fill_between(
                ticker_dates,
                bands["q04"],
                bands["q96"],
                color=PREDICTION_PURPLE_LIGHT,
                alpha=0.50,
                label="92% predictive interval",
            )
            ax.fill_between(
                ticker_dates,
                bands["q16"],
                bands["q84"],
                color=PREDICTION_PURPLE,
                alpha=0.30,
                label="68% predictive interval",
            )
            for resimulation in simulated[:n_resimulations]:
                ax.plot(
                    ticker_dates,
                    resimulation,
                    color=PREDICTION_PURPLE,
                    linewidth=0.65,
                    alpha=0.13,
                )
            ax.plot(
                ticker_dates,
                bands["median"],
                color=PREDICTION_PURPLE_DARK,
                linewidth=1.5,
                label="Predictive median",
            )
            ax.plot(
                ticker_dates,
                observed_series,
                color=OBSERVED_COLOR,
                linewidth=1.5,
                label="Observed",
            )
            ax.set_title(f"{ticker}: {title}", fontsize=15)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_xlabel("Date", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)

        histogram_ax = axes[asset_id, 2]
        flattened_predictive = return_paths.reshape(-1)
        combined = np.concatenate((flattened_predictive, observed_returns))
        low, high = np.quantile(combined, [0.001, 0.999])
        bound = max(abs(low), abs(high))
        bins = np.linspace(-bound, bound, 61)
        histogram_ax.hist(
            flattened_predictive,
            bins=bins,
            density=True,
            color=PREDICTION_PURPLE,
            alpha=0.48,
            label="Posterior predictive",
        )
        histogram_ax.hist(
            observed_returns,
            bins=bins,
            density=True,
            histtype="step",
            color=OBSERVED_COLOR,
            linewidth=2.4,
            label="Observed",
        )
        histogram_ax.axvline(0.0, color="#636363", linewidth=0.8, alpha=0.7)
        histogram_ax.set_title(f"{ticker}: Marginal returns", fontsize=15)
        histogram_ax.set_xlabel(f"Daily return ({RETURN_UNIT})", fontsize=14)
        histogram_ax.set_ylabel("Density", fontsize=14)
        histogram_ax.tick_params(axis="both", labelsize=12)
        histogram_ax.legend(frameon=False, fontsize=13)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=13,
    )
    fig.suptitle("Posterior Predictive Checks on the Latest 256-Day Window", fontsize=20)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    return fig


def plot_rolling_risk(
    decisions: pd.DataFrame,
    tickers: Sequence[str],
) -> plt.Figure:
    """Plot VaR and ES with posterior uncertainty and a historical benchmark."""

    metric_specs = (
        (
            "var_95_pct",
            "var_95_posterior_q05_pct",
            "var_95_posterior_q95_pct",
            "historical_var_95_pct",
            "95% Value at Risk (VaR)",
        ),
        (
            "expected_shortfall_95_pct",
            "es_95_posterior_q05_pct",
            "es_95_posterior_q95_pct",
            "historical_es_95_pct",
            "95% Expected Shortfall (ES)",
        ),
    )
    fig, axes = plt.subplots(
        len(tickers),
        2,
        figsize=(16, 5.2 * len(tickers)),
        sharex=True,
        squeeze=False,
    )

    for row, ticker in enumerate(tickers):
        frame = decisions.loc[decisions["ticker"] == ticker].sort_values("as_of")
        for col, (metric, lower, upper, benchmark, title) in enumerate(metric_specs):
            ax = axes[row, col]
            ax.fill_between(
                frame["as_of"],
                frame[lower],
                frame[upper],
                color=PREDICTION_PURPLE,
                alpha=0.24,
                label="90% posterior parameter uncertainty",
            )
            ax.plot(
                frame["as_of"],
                frame[metric],
                color=PREDICTION_PURPLE_DARK,
                linewidth=2.4,
                label="MMAR posterior predictive",
            )
            ax.plot(
                frame["as_of"],
                frame[benchmark],
                color=OBSERVED_COLOR,
                linewidth=2.0,
                linestyle="--",
                alpha=0.82,
                label="Historical simulation benchmark",
            )
            ax.set_title(f"{ticker}: {title}", fontsize=17)
            ax.set_ylabel("20-day loss (% of capital)", fontsize=15)
            if row == len(tickers) - 1:
                ax.set_xlabel("Window end date", fontsize=15)
            ax.tick_params(axis="both", labelsize=12)
            ax.legend(frameon=False, fontsize=13, loc="upper right")

    fig.suptitle(
        "Rolling 20-Day Downside Risk (256-Day Information Window)",
        fontsize=21,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig
