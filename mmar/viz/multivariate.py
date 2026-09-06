"""Visualization helpers for the static multivariate MMAR workflow."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from ..multivariate import ASSET_LABELS
from ..statistics import (
    ASSET_NAMES,
    dependence_statistics,
    equal_weight_portfolio_returns,
    max_drawdown_batch,
    return_statistics_batch,
)
from . import (
    OBSERVED_COLOR,
    PREDICTION_PURPLE,
    PREDICTION_PURPLE_DARK,
    SECONDARY_COLOR,
)


ASSET_COLORS = {
    "VOO": PREDICTION_PURPLE_DARK,
    "GLD": "#B8860B",
    "TLT": SECONDARY_COLOR,
}


def _histogram_envelope(
    paths: np.ndarray,
    observed: np.ndarray,
    num_bins: int = 65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.concatenate((paths.reshape(-1), observed.reshape(-1)))
    lower, upper = np.quantile(combined, (0.001, 0.999))
    bound = max(abs(lower), abs(upper))
    bins = np.linspace(-bound, bound, num_bins + 1)
    histograms = np.stack([np.histogram(path, bins=bins, density=True)[0] for path in paths])
    return bins, 0.5 * (bins[:-1] + bins[1:]), np.quantile(histograms, (0.05, 0.50, 0.95), axis=0)


def plot_example_paths(
    draw: dict[str, np.ndarray],
    assets: Sequence[str] = ASSET_NAMES,
    draw_index: int = 0,
) -> plt.Figure:
    """Show one joint return realization, its trading clocks, and cumulative wealth."""

    returns = np.asarray(draw["returns"])[draw_index]
    trading_time = np.asarray(draw["trading_time_increments"])[draw_index]
    q = np.asarray(draw["q"])[draw_index]
    days = np.arange(returns.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    for asset_index, asset in enumerate(assets):
        color = ASSET_COLORS[asset]
        axes[0].plot(
            days,
            100.0 * returns[:, asset_index],
            color=color,
            linewidth=1.05,
            alpha=0.82,
            label=asset,
        )
        axes[1].plot(
            days,
            trading_time[:, asset_index],
            color=color,
            linewidth=1.25,
            alpha=0.86,
            label=f"{asset}: q={q[asset_index]:.3f}",
        )
        axes[2].plot(
            days,
            np.cumprod(1.0 + returns[:, asset_index]),
            color=color,
            linewidth=2.0,
            label=asset,
        )

    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_ylabel("Daily return (%)", fontsize=13)
    axes[0].set_title("Joint returns", fontsize=15)
    axes[1].set_ylabel("Trading-time increment", fontsize=13)
    axes[1].set_title("Asset-specific intensity on one shared cascade tree", fontsize=15)
    axes[2].axhline(1.0, color="0.35", linewidth=0.8)
    axes[2].set_ylabel("Wealth from $1", fontsize=13)
    axes[2].set_xlabel("Trading day", fontsize=13)
    axes[2].set_title("Cumulative wealth", fontsize=15)
    for ax in axes:
        ax.legend(frameon=False, ncol=3, fontsize=11)
        ax.tick_params(labelsize=11)
    fig.suptitle("One Static Multivariate-MMAR Prior Draw", fontsize=19, y=1.01)
    fig.tight_layout()
    return fig


def plot_marginal_return_envelopes(
    prior_returns: np.ndarray,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = ASSET_NAMES,
) -> plt.Figure:
    """Plot per-asset histogram envelopes without pooling the three marginals."""

    assets = tuple(assets)
    fig, axes = plt.subplots(1, len(assets), figsize=(6.0 * len(assets), 4.8), sharey=False)
    axes = np.atleast_1d(axes)
    date_label = (
        f"{observed_returns.index.min():%b %d, %Y}–" f"{observed_returns.index.max():%b %d, %Y}"
    )
    for asset_index, (asset, ax) in enumerate(zip(assets, axes)):
        paths_pct = 100.0 * prior_returns[:, :, asset_index]
        observed_pct = 100.0 * observed_returns[asset].to_numpy()
        bins, centers, envelope = _histogram_envelope(paths_pct, observed_pct)
        observed_density = np.histogram(observed_pct, bins=bins, density=True)[0]
        ax.fill_between(
            centers,
            envelope[0],
            envelope[2],
            color=PREDICTION_PURPLE,
            alpha=0.24,
            linewidth=0,
            label="Prior 90% envelope",
        )
        ax.plot(
            centers,
            envelope[1],
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
            label="Observed",
        )
        ax.axvline(0.0, color="0.4", linewidth=0.8)
        ax.set_title(
            f"{asset}: {ASSET_LABELS[asset]}\n{date_label}",
            fontsize=14,
        )
        ax.set_xlabel("Daily return (%)", fontsize=13)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle("Marginal Prior-Predictive Return Distributions", fontsize=19, y=1.02)
    fig.tight_layout()
    return fig


def plot_marginal_drawdowns(
    prior_returns: np.ndarray,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = ASSET_NAMES,
) -> plt.Figure:
    """Plot per-asset maximum-drawdown pushforwards."""

    assets = tuple(assets)
    prior_drawdown = 100.0 * max_drawdown_batch(prior_returns)
    observed_drawdown = (
        100.0 * max_drawdown_batch(observed_returns.loc[:, assets].to_numpy()[None, ...])[0]
    )
    fig, axes = plt.subplots(1, len(assets), figsize=(6.0 * len(assets), 4.7), sharey=True)
    axes = np.atleast_1d(axes)
    for asset_index, (asset, ax) in enumerate(zip(assets, axes)):
        lower, upper = np.quantile(prior_drawdown[:, asset_index], (0.05, 0.95))
        ax.axvspan(lower, upper, color=PREDICTION_PURPLE, alpha=0.15, label="Prior 90% interval")
        ax.hist(
            prior_drawdown[:, asset_index],
            bins=34,
            color=PREDICTION_PURPLE,
            edgecolor=PREDICTION_PURPLE_DARK,
            alpha=0.68,
            density=True,
            label="Prior predictive",
        )
        ax.axvline(
            observed_drawdown[asset_index],
            color=OBSERVED_COLOR,
            linewidth=3.0,
            label=f"Observed: {observed_drawdown[asset_index]:.1f}%",
        )
        ax.set_title(f"{asset}: Maximum Drawdown", fontsize=15)
        ax.set_xlabel("Maximum drawdown (%)", fontsize=13)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle("Marginal Drawdown Pushforwards", fontsize=19, y=1.02)
    fig.tight_layout()
    return fig


def plot_dependence_pushforwards(
    prior_returns: np.ndarray,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = ASSET_NAMES,
    predictive_label: str = "Prior",
    figure_title: str = "Cross-Asset Dependence Pushforwards",
) -> plt.Figure:
    """Compare raw and absolute-return correlation across all asset pairs."""

    prior = dependence_statistics(prior_returns, assets=assets)
    observed = dependence_statistics(
        observed_returns.loc[:, list(assets)].to_numpy(), assets=assets
    )
    pairs = tuple(prior["pair"].drop_duplicates())
    statistics = (
        ("return_correlation", "Return correlation"),
        ("absolute_return_correlation", "Absolute-return correlation"),
    )
    fig, axes = plt.subplots(2, len(pairs), figsize=(5.7 * len(pairs), 8.0), sharex=False)
    for row, (statistic, label) in enumerate(statistics):
        for column, pair in enumerate(pairs):
            ax = axes[row, column]
            values = prior.loc[prior["pair"] == pair, statistic].to_numpy()
            observed_value = float(observed.loc[observed["pair"] == pair, statistic].iloc[0])
            lower, upper = np.quantile(values, (0.05, 0.95))
            ax.axvspan(lower, upper, color=PREDICTION_PURPLE, alpha=0.15)
            ax.hist(
                values,
                bins=36,
                color=PREDICTION_PURPLE,
                edgecolor=PREDICTION_PURPLE_DARK,
                alpha=0.68,
                density=True,
            )
            ax.axvline(observed_value, color=OBSERVED_COLOR, linewidth=3.0)
            ax.set_title(f"{pair}\n{label}", fontsize=14)
            ax.set_xlabel("Correlation", fontsize=12)
            ax.set_yticks([])
            ax.tick_params(labelsize=11)
            if row == 0 and column == 0:
                ax.plot(
                    [],
                    [],
                    color=PREDICTION_PURPLE,
                    linewidth=8,
                    alpha=0.5,
                    label=predictive_label,
                )
                ax.plot([], [], color=OBSERVED_COLOR, linewidth=3, label="Observed")
                ax.legend(frameon=False, fontsize=10)
    fig.suptitle(figure_title, fontsize=19, y=1.01)
    fig.tight_layout()
    return fig


def plot_equal_weight_portfolio_checks(
    prior_returns: np.ndarray,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = ASSET_NAMES,
) -> plt.Figure:
    """Plot volatility, expected shortfall, and drawdown for a daily-rebalanced portfolio."""

    prior_portfolio = equal_weight_portfolio_returns(prior_returns)[:, :, None]
    observed_portfolio = equal_weight_portfolio_returns(
        observed_returns.loc[:, list(assets)].to_numpy()
    )[None, :, None]
    prior_stats = return_statistics_batch(prior_portfolio)
    observed_stats = return_statistics_batch(observed_portfolio)
    panels = (
        ("volatility_pct", "Daily volatility", "%"),
        ("expected_shortfall_5_pct", "5% expected shortfall", "%"),
        ("max_drawdown_pct", "Maximum drawdown", "%"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for ax, (statistic, title, unit) in zip(axes, panels):
        values = prior_stats[statistic][:, 0]
        observed = float(observed_stats[statistic][0, 0])
        lower, upper = np.quantile(values, (0.05, 0.95))
        ax.axvspan(lower, upper, color=PREDICTION_PURPLE, alpha=0.15)
        ax.hist(
            values,
            bins=34,
            color=PREDICTION_PURPLE,
            edgecolor=PREDICTION_PURPLE_DARK,
            alpha=0.68,
            density=True,
        )
        ax.axvline(observed, color=OBSERVED_COLOR, linewidth=3.0)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(unit, fontsize=13)
        ax.set_yticks([])
    axes[0].plot([], [], color=PREDICTION_PURPLE, linewidth=8, alpha=0.5, label="Prior")
    axes[0].plot([], [], color=OBSERVED_COLOR, linewidth=3, label="Observed")
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle("Equal-Weight Portfolio Pushforwards", fontsize=19, y=1.02)
    fig.tight_layout()
    return fig


def _predictive_bands(paths: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q04": np.quantile(paths, 0.04, axis=0),
        "q16": np.quantile(paths, 0.16, axis=0),
        "median": np.quantile(paths, 0.50, axis=0),
        "q84": np.quantile(paths, 0.84, axis=0),
        "q96": np.quantile(paths, 0.96, axis=0),
    }


def plot_multivariate_posterior_predictive_checks(
    posterior_paths: np.ndarray,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = ASSET_NAMES,
    n_resimulations: int = 10,
) -> plt.Figure:
    """Plot marginal and equal-weight PPCs from the same joint path draws."""

    assets = tuple(assets)
    paths = np.asarray(posterior_paths, dtype=np.float64)
    observed = observed_returns.loc[:, list(assets)].to_numpy(dtype=np.float64)
    if paths.ndim != 3 or paths.shape[1:] != observed.shape:
        raise ValueError(
            "Expected posterior_paths (draw, time, asset) matching observed returns; "
            f"received {paths.shape} and {observed.shape}."
        )

    paths = np.concatenate((paths, paths.mean(axis=-1, keepdims=True)), axis=-1)
    observed = np.column_stack((observed, observed.mean(axis=1)))
    targets = (*assets, "EW")
    dates = observed_returns.index
    fig, axes = plt.subplots(
        len(targets),
        3,
        figsize=(18, 4.25 * len(targets)),
        squeeze=False,
    )

    for target_id, target in enumerate(targets):
        return_paths = 100.0 * paths[:, :, target_id]
        observed_target = 100.0 * observed[:, target_id]
        return_bands = _predictive_bands(return_paths)
        wealth_paths = np.cumprod(1.0 + paths[:, :, target_id], axis=1)
        observed_wealth = np.cumprod(1.0 + observed[:, target_id])
        wealth_bands = _predictive_bands(wealth_paths)

        for column, (simulated, actual, bands, ylabel, title) in enumerate(
            (
                (
                    return_paths,
                    observed_target,
                    return_bands,
                    "Daily return (%)",
                    "Daily returns",
                ),
                (
                    wealth_paths,
                    observed_wealth,
                    wealth_bands,
                    "Growth of $1",
                    "Cumulative wealth",
                ),
            )
        ):
            ax = axes[target_id, column]
            ax.fill_between(
                dates,
                bands["q04"],
                bands["q96"],
                color=PREDICTION_PURPLE,
                alpha=0.16,
                label="92% predictive interval",
            )
            ax.fill_between(
                dates,
                bands["q16"],
                bands["q84"],
                color=PREDICTION_PURPLE,
                alpha=0.28,
                label="68% predictive interval",
            )
            for resimulation in simulated[:n_resimulations]:
                ax.plot(
                    dates,
                    resimulation,
                    color=PREDICTION_PURPLE,
                    linewidth=0.6,
                    alpha=0.12,
                )
            ax.plot(
                dates,
                bands["median"],
                color=PREDICTION_PURPLE_DARK,
                linewidth=1.5,
                label="Predictive median",
            )
            ax.plot(
                dates,
                actual,
                color=OBSERVED_COLOR,
                linewidth=1.35,
                label="Observed",
            )
            ax.set_title(f"{target}: {title}", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.tick_params(labelsize=10)

        histogram_ax = axes[target_id, 2]
        flattened = return_paths.reshape(-1)
        low, high = np.quantile(
            np.concatenate((flattened, observed_target)),
            (0.001, 0.999),
        )
        bound = max(abs(low), abs(high))
        bins = np.linspace(-bound, bound, 61)
        histogram_ax.hist(
            flattened,
            bins=bins,
            density=True,
            color=PREDICTION_PURPLE,
            alpha=0.48,
            label="Posterior predictive",
        )
        histogram_ax.hist(
            observed_target,
            bins=bins,
            density=True,
            histtype="step",
            color=OBSERVED_COLOR,
            linewidth=2.2,
            label="Observed",
        )
        histogram_ax.axvline(0.0, color="0.4", linewidth=0.8)
        histogram_ax.set_title(f"{target}: Marginal returns", fontsize=14)
        histogram_ax.set_xlabel("Daily return (%)", fontsize=12)
        histogram_ax.set_yticks([])
        histogram_ax.legend(frameon=False, fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=11,
    )
    fig.suptitle(
        "Joint Posterior Predictive Checks: Assets and Equal-Weight Portfolio",
        fontsize=19,
        y=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.985))
    return fig


def plot_multivariate_rolling_risk(
    decisions: pd.DataFrame,
    targets: Sequence[str] = (*ASSET_NAMES, "EW"),
) -> plt.Figure:
    """Plot rolling posterior VaR/ES with uncertainty and historical benchmarks."""

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
    targets = tuple(targets)
    fig, axes = plt.subplots(
        len(targets),
        2,
        figsize=(16, 4.6 * len(targets)),
        sharex=True,
        squeeze=False,
    )
    for row, target in enumerate(targets):
        frame = decisions.loc[decisions["target"] == target].sort_values("as_of")
        for column, (metric, lower, upper, benchmark, title) in enumerate(metric_specs):
            ax = axes[row, column]
            ax.fill_between(
                frame["as_of"],
                frame[lower],
                frame[upper],
                color=PREDICTION_PURPLE,
                alpha=0.24,
                label="90% conditional-estimate band",
            )
            ax.plot(
                frame["as_of"],
                frame[metric],
                color=PREDICTION_PURPLE_DARK,
                linewidth=2.2,
                label="MMAR posterior predictive",
            )
            ax.plot(
                frame["as_of"],
                frame[benchmark],
                color=OBSERVED_COLOR,
                linewidth=1.8,
                linestyle="--",
                alpha=0.82,
                label="Historical simulation",
            )
            ax.set_title(f"{target}: {title}", fontsize=15)
            ax.set_ylabel("20-day loss (% of capital)", fontsize=12)
            ax.tick_params(labelsize=10)
            ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle(
        "Rolling Joint-MMAR Downside Risk (256-Day Information Window)",
        fontsize=20,
        y=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    return fig


def plot_model_allocation_backtest(
    backtest: pd.DataFrame,
    fee_levels: Sequence[float] = (0.0, 5.0, 20.0),
    selected_fee: float = 5.0,
    assets: Sequence[str] = ASSET_NAMES,
    initial_wealth: float = 256.0,
    figsize: tuple = (12, 8.5)
) -> plt.Figure:
    """Plot net wealth by fee plus model weights and cumulative turnover."""

    assets = tuple(assets)
    fees = tuple(float(fee) for fee in fee_levels)

    strategy_colors = {
        "MMAR tail utility (daily)": PREDICTION_PURPLE_DARK,
        "MMAR tail utility (21d)": SECONDARY_COLOR,
        "All Weather proxy (buy & hold)": OBSERVED_COLOR,
        "VOO (100%, buy & hold)": "#4C78A8",
    }
    strategy_styles = {
        "MMAR tail utility (daily)": "-",
        "MMAR tail utility (21d)": "-.",
        "All Weather proxy (buy & hold)": "--",
        "VOO (100%, buy & hold)": ":",
    }

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.08, 0.92), hspace=0.34, wspace=0.24)
    top_axes = [fig.add_subplot(grid[0, 0])]
    top_axes.extend(fig.add_subplot(grid[0, column], sharey=top_axes[0]) for column in range(1, 3))

    for column, fee in enumerate(fees):
        ax = top_axes[column]
        panel = backtest.loc[np.isclose(backtest["fee_bps"], fee)]
        for strategy, frame in panel.groupby("strategy", sort=False):
            frame = frame.sort_values("date")
            if frame.empty:
                continue
            dates = np.concatenate(
                (
                    [np.datetime64(pd.Timestamp(frame["signal_date"].iloc[0]))],
                    frame["date"].to_numpy(),
                )
            )
            wealth = np.concatenate(([initial_wealth], frame["wealth"].to_numpy(dtype=float)))
            ax.plot(
                dates,
                wealth,
                color=strategy_colors.get(strategy, "0.35"),
                linestyle=strategy_styles.get(strategy, "-"),
                linewidth=2.0,
                label=strategy,
            )
        ax.axhline(initial_wealth, color="0.55", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{fee:g} bp per unit of one-way turnover", fontsize=14)
        ax.set_ylabel("Portfolio value ($)", fontsize=12)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.grid(axis="y", alpha=0.18)
        if column == 0:
            ax.legend(frameon=False, fontsize=9, loc="best")
    for column in range(len(fees), 3):
        top_axes[column].set_visible(False)

    available_fees = np.asarray(sorted(backtest["fee_bps"].unique()), dtype=float)
    selected = float(available_fees[np.argmin(np.abs(available_fees - selected_fee))])
    daily = backtest.loc[
        np.isclose(backtest["fee_bps"], selected)
        & (backtest["strategy"] == "MMAR tail utility (daily)")
    ].sort_values("date")
    weights_ax = fig.add_subplot(grid[1, :2])
    if not daily.empty:
        weights_ax.stackplot(
            daily["date"],
            *(daily[f"weight_{asset}"] for asset in assets),
            colors=[ASSET_COLORS[asset] for asset in assets],
            labels=assets,
            alpha=0.82,
        )
    weights_ax.set_ylim(0.0, 1.0)
    weights_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    weights_ax.set_title(
        f"Weights actually held: daily model rule ({selected:g} bp case)", fontsize=14
    )
    weights_ax.set_ylabel("Portfolio weight", fontsize=12)
    weights_ax.legend(frameon=False, ncol=len(assets), loc="upper left")
    weights_ax.tick_params(axis="x", rotation=20, labelsize=9)

    turnover_ax = fig.add_subplot(grid[1, 2])
    selected_panel = backtest.loc[np.isclose(backtest["fee_bps"], selected)]
    for strategy, frame in selected_panel.groupby("strategy", sort=False):
        if not strategy.startswith("MMAR"):
            continue
        frame = frame.sort_values("date")
        turnover_ax.plot(
            frame["date"],
            frame["turnover"].cumsum(),
            color=strategy_colors.get(strategy, "0.35"),
            linestyle=strategy_styles.get(strategy, "-"),
            linewidth=2.0,
            label=strategy.replace("MMAR tail utility ", ""),
        )
    turnover_ax.set_title(f"Cumulative turnover ({selected:g} bp case)", fontsize=14)
    turnover_ax.set_ylabel("One-way portfolio turnover", fontsize=12)
    turnover_ax.legend(frameon=False, fontsize=9)
    turnover_ax.tick_params(axis="x", rotation=25, labelsize=9)
    turnover_ax.grid(axis="y", alpha=0.18)

    fig.suptitle(
        "Walk-Forward MMAR Allocation vs. Buy-and-Hold Benchmarks",
        fontsize=20,
        y=0.99,
    )
    fig.subplots_adjust(top=0.93, bottom=0.08)
    return fig
