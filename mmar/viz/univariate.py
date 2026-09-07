"""Univariate prior checks, posterior predictive checks and rolling risk."""
from typing import Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from scipy import stats as scipy_stats
from ..statistics import max_drawdown_batch
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
    """Plot maximum drawdown, wealth, and marginal-return PPCs by asset."""

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

        wealth_paths = np.cumprod(1.0 + paths[asset_id], axis=1)
        observed_wealth = np.cumprod(1.0 + observed[asset_id])
        wealth_bands = _predictive_bands(wealth_paths)

        drawdown_ax = axes[asset_id, 0]
        predictive_drawdowns = RETURN_SCALE * max_drawdown_batch(paths[asset_id])
        observed_drawdown = float(
            RETURN_SCALE * max_drawdown_batch(observed[asset_id][None, :])[0]
        )
        display_low, display_high = np.quantile(predictive_drawdowns, (0.001, 0.999))
        display_low = min(float(display_low), observed_drawdown)
        display_high = max(float(display_high), observed_drawdown)
        padding = 0.04 * max(display_high - display_low, 1e-6)
        drawdown_bins = np.linspace(
            display_low - padding,
            min(0.0, display_high + padding),
            46,
        )
        interval_low, interval_high = np.quantile(predictive_drawdowns, (0.04, 0.96))
        drawdown_ax.axvspan(
            interval_low,
            interval_high,
            color=PREDICTION_PURPLE_LIGHT,
            alpha=0.50,
        )
        drawdown_ax.hist(
            predictive_drawdowns,
            bins=drawdown_bins,
            density=True,
            color=PREDICTION_PURPLE,
            edgecolor=PREDICTION_PURPLE_DARK,
            linewidth=0.5,
            alpha=0.62,
        )
        drawdown_ax.axvline(observed_drawdown, color=OBSERVED_COLOR, linewidth=2.6)
        drawdown_ax.set_title(f"{ticker}: Predicted maximum drawdown", fontsize=17)
        drawdown_ax.set_xlabel(f"Maximum drawdown ({RETURN_UNIT})", fontsize=15)
        drawdown_ax.set_ylabel("Density", fontsize=15)
        drawdown_ax.set_yticks([])
        drawdown_ax.tick_params(axis="x", labelsize=13)
        drawdown_ax.grid(False, axis="y")

        wealth_ax = axes[asset_id, 1]
        wealth_ax.fill_between(
            ticker_dates,
            wealth_bands["q04"],
            wealth_bands["q96"],
            color=PREDICTION_PURPLE_LIGHT,
            alpha=0.50,
        )
        wealth_ax.fill_between(
            ticker_dates,
            wealth_bands["q16"],
            wealth_bands["q84"],
            color=PREDICTION_PURPLE,
            alpha=0.30,
        )
        for resimulation in wealth_paths[:n_resimulations]:
            wealth_ax.plot(
                ticker_dates,
                resimulation,
                color=PREDICTION_PURPLE,
                linewidth=0.75,
                alpha=0.14,
            )
        wealth_ax.plot(
            ticker_dates,
            wealth_bands["median"],
            color=PREDICTION_PURPLE_DARK,
            linewidth=1.8,
        )
        wealth_ax.plot(
            ticker_dates,
            observed_wealth,
            color=OBSERVED_COLOR,
            linewidth=1.8,
        )
        wealth_ax.set_title(f"{ticker}: Cumulative wealth", fontsize=17)
        wealth_ax.set_ylabel("Growth of $1", fontsize=15)
        wealth_ax.set_xlabel("Date", fontsize=15)
        wealth_ax.tick_params(axis="both", labelsize=13)

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
        histogram_ax.set_title(f"{ticker}: Marginal returns", fontsize=17)
        histogram_ax.set_xlabel(f"Daily return ({RETURN_UNIT})", fontsize=15)
        histogram_ax.set_ylabel("Density", fontsize=15)
        histogram_ax.tick_params(axis="both", labelsize=13)
        if asset_id == len(tickers) - 1:
            histogram_ax.legend(frameon=False, fontsize=14, loc="upper left")

    fig.suptitle("Posterior Predictive Checks on the Latest 256-Day Window", fontsize=22)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _magnitude_density(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Histogram density of finite, nonzero magnitudes."""

    magnitudes = np.abs(np.asarray(values, dtype="float64").reshape(-1))
    magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
    if not len(magnitudes):
        raise ValueError("At least one nonzero return is required.")
    counts = np.histogram(magnitudes, bins=bins)[0]
    return counts / (len(magnitudes) * np.diff(bins))


def _pointwise_hdi(draws: np.ndarray, probability: float) -> tuple[np.ndarray, np.ndarray]:
    """Narrowest sample interval at each plotted density bin."""

    values = np.asarray(draws, dtype="float64")
    if values.ndim != 2 or not 0.0 < probability < 1.0:
        raise ValueError("Expected 2D draws and a probability strictly between zero and one.")
    interval = max(1, min(len(values) - 1, int(np.floor(probability * len(values)))))
    ordered = np.sort(values, axis=0)
    widths = ordered[interval:] - ordered[:-interval]
    starts = np.argmin(widths, axis=0)
    columns = np.arange(values.shape[1])
    return ordered[starts, columns], ordered[starts + interval, columns]


def _smooth_density_draws(
    densities: np.ndarray,
    bins: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Smooth probability mass on an equally spaced log-bin grid."""

    values = np.asarray(densities, dtype="float64")
    if sigma == 0.0:
        return values
    bin_widths = np.diff(bins)
    masses = values * bin_widths
    smoothed_masses = gaussian_filter1d(
        masses,
        sigma=sigma,
        axis=1,
        mode="constant",
        cval=0.0,
    )
    original_mass = masses.sum(axis=1, keepdims=True)
    smoothed_mass = smoothed_masses.sum(axis=1, keepdims=True)
    smoothed_masses *= np.divide(
        original_mass,
        smoothed_mass,
        out=np.ones_like(original_mass),
        where=smoothed_mass > 0.0,
    )
    return smoothed_masses / bin_widths


def plot_gaussian_mmar_fit_comparison(
    observed_returns: pd.Series | np.ndarray,
    posterior_paths: np.ndarray,
    ticker: str = "VOO",
    n_bins: int = 50,
    x_label_fontsize: float = 19,
    y_label_fontsize: float = 19,
    legend_fontsize: float | None = None,
    uncertainty_style: str = "hdi",
    uncertainty_sd: float = 2.0,
    uncertainty_probability: float = 0.68,
    density_smoothing: float = 1.5,
    density_clip: float = 1e-3,
    minimum_magnitude: float = 1e-3,
) -> plt.Figure:
    """Compare absolute returns with folded-Gaussian and MMAR predictive densities."""

    date_label = None
    if isinstance(observed_returns, pd.Series) and isinstance(
        observed_returns.index, pd.DatetimeIndex
    ):
        date_label = (
            f"{observed_returns.index.min():%b %d, %Y}–"
            f"{observed_returns.index.max():%b %d, %Y}"
        )

    observed = np.asarray(observed_returns).reshape(-1)
    observed = observed[np.isfinite(observed)]
    predictive = np.asarray(posterior_paths, dtype="float64")
    if predictive.ndim != 2:
        raise ValueError("Expected posterior_paths with shape (draw, time).")
    predictive = predictive[np.isfinite(predictive)]
    if np.any(observed <= -1.0) or np.any(predictive <= -1.0):
        raise ValueError("Simple returns must be greater than -1 before taking log1p.")

    observed_log = np.log1p(observed)
    predictive_log = np.log1p(predictive)
    observed_magnitude = np.abs(observed_log)
    observed_magnitude = observed_magnitude[observed_magnitude > 0.0]
    predictive_magnitude = np.abs(predictive_log)
    predictive_magnitude = predictive_magnitude[predictive_magnitude > 0.0]
    if len(observed_magnitude) < 2 or len(predictive_magnitude) < 2:
        raise ValueError("Observed and posterior-predictive samples need nonzero returns.")
    if uncertainty_style not in {"band", "curves", "hdi", "hdi_band"}:
        raise ValueError(
            "uncertainty_style must be 'band', 'curves', 'hdi', or 'hdi_band'."
        )
    if (
        uncertainty_sd <= 0.0
        or density_smoothing < 0.0
        or density_clip <= 0.0
        or minimum_magnitude <= 0.0
    ):
        raise ValueError(
            "uncertainty_sd and density_clip must be positive; "
            "density_smoothing must be nonnegative; minimum_magnitude must be positive."
        )
    if not 0.0 < uncertainty_probability < 1.0:
        raise ValueError("uncertainty_probability must be strictly between zero and one.")

    lower = max(
        float(np.quantile(observed_magnitude, 0.005)),
        minimum_magnitude,
    )
    upper = 1.05 * float(observed_magnitude.max())
    if not lower < upper:
        raise ValueError("Absolute returns do not span a usable plotting range.")

    bins = np.geomspace(lower, upper, n_bins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    observed_density = _magnitude_density(observed_log, bins)
    realization_densities = np.stack(
        [_magnitude_density(np.log1p(path), bins) for path in posterior_paths]
    )
    realization_densities = _smooth_density_draws(
        realization_densities,
        bins,
        density_smoothing,
    )
    predictive_mean_density = realization_densities.mean(axis=0)
    predictive_std_density = realization_densities.std(axis=0, ddof=1)
    if uncertainty_style in {"hdi", "hdi_band"}:
        predictive_lower_density, predictive_upper_density = _pointwise_hdi(
            realization_densities,
            uncertainty_probability,
        )
        uncertainty_label = f"{100 * uncertainty_probability:g}% HDI"
    else:
        predictive_lower_density = (
            predictive_mean_density - uncertainty_sd * predictive_std_density
        )
        predictive_upper_density = (
            predictive_mean_density + uncertainty_sd * predictive_std_density
        )
        uncertainty_label = rf"$\pm$ {uncertainty_sd:g} SD"

    normal_loc, normal_scale = scipy_stats.norm.fit(observed_log)
    x_grid = np.geomspace(lower, upper, 500)
    normal_density = scipy_stats.norm.pdf(x_grid, normal_loc, normal_scale)
    normal_density += scipy_stats.norm.pdf(-x_grid, normal_loc, normal_scale)
    occupied_bins = observed_density > 0.0
    gaussian_99 = scipy_stats.foldnorm.ppf(
        0.99,
        abs(normal_loc) / normal_scale,
        scale=normal_scale,
    )
    gaussian_bin_probability = (
        scipy_stats.norm.cdf(bins[1:], normal_loc, normal_scale)
        - scipy_stats.norm.cdf(bins[:-1], normal_loc, normal_scale)
        + scipy_stats.norm.cdf(-bins[:-1], normal_loc, normal_scale)
        - scipy_stats.norm.cdf(-bins[1:], normal_loc, normal_scale)
    )
    gaussian_expected_count = len(observed_log) * gaussian_bin_probability
    missed_bins = occupied_bins & (centers > gaussian_99) & (gaussian_expected_count < 0.5)
    regular_bins = occupied_bins & ~missed_bins

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharex=True, sharey=True)
    model_specs = (
        ("Gaussian fit", x_grid, normal_density, "Gaussian MLE", "#285C4D"),
        (
            "Univariate MMAR fit",
            centers,
            predictive_mean_density,
            f"MMAR mean; {uncertainty_label}",
            PREDICTION_PURPLE_DARK,
        ),
    )
    model_lines = []
    display_floor = 0.25 * observed_density[observed_density > 0.0].min()
    for panel, (ax, (title, model_x, model_density, model_label, model_color)) in enumerate(
        zip(axes, model_specs)
    ):
        regular_bars = ax.bar(
            bins[:-1][regular_bins],
            observed_density[regular_bins],
            width=np.diff(bins)[regular_bins],
            align="edge",
            color="#C9CDD4",
            edgecolor="#737A86",
            linewidth=0.55,
            alpha=0.82,
            label=f"{ticker} observed",
            zorder=1,
        )
        missed_bars = ax.bar(
            bins[:-1][missed_bins],
            observed_density[missed_bins],
            width=np.diff(bins)[missed_bins],
            align="edge",
            color="#D1495B",
            edgecolor="#8E2433",
            linewidth=0.75,
            alpha=0.88,
            label="Beyond Gaussian 99% range",
            zorder=2,
        )
        if panel == 1 and uncertainty_style in {"band", "hdi_band"}:
            ax.fill_between(
                centers,
                np.maximum(predictive_lower_density, density_clip),
                predictive_upper_density,
                where=predictive_upper_density > density_clip,
                color=PREDICTION_PURPLE,
                alpha=0.24,
                linewidth=0.0,
                zorder=2.5,
            )
        elif panel == 1:
            for density in (predictive_lower_density, predictive_upper_density):
                ax.plot(
                    centers,
                    np.where(density > density_clip, density, np.nan),
                    color=PREDICTION_PURPLE,
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.82,
                    zorder=2.5,
                )
        model_line, = ax.plot(
            model_x,
            np.where(model_density > density_clip, model_density, np.nan),
            color=model_color,
            linewidth=3.0,
            label=model_label,
            zorder=3,
        )
        model_lines.append(model_line)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=17)
        ax.set_xlabel(
            r"Absolute daily log-return",
            fontsize=x_label_fontsize,
        )
        ax.grid(True, which="major", alpha=0.18)
        ax.grid(False, which="minor")

    axes[0].set_ylim(
        display_floor,
        1.8 * max(
            observed_density.max(),
            normal_density.max(),
            predictive_mean_density.max(),
            predictive_upper_density.max(),
        ),
    )
    axes[0].set_ylabel("Probability density", fontsize=y_label_fontsize)
    window_label = date_label or f"{len(observed_log):,} trading days"
    fig.suptitle(
        f"Absolute {ticker} Returns: Gaussian vs. MMAR ({window_label})",
        fontsize=20,
        y=0.96,
    )
    observed_handle = Patch(facecolor="#C9CDD4", edgecolor="#737A86")
    if legend_fontsize is None:
        legend_fontsize = max(x_label_fontsize, y_label_fontsize)
    fig.legend(
        [model_lines[0], model_lines[1], observed_handle],
        ["Gaussian MLE", model_specs[1][3], f"{ticker} observed; Gaussian misses red"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        frameon=False,
        fontsize=legend_fontsize,
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    return fig


def plot_rolling_risk(
    risk_estimates: pd.DataFrame,
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
        frame = risk_estimates.loc[risk_estimates["ticker"] == ticker].sort_values("as_of")
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
                label="Historical simulation (past returns only)",
            )
            ax.set_title(f"{ticker}: {title}", fontsize=17)
            ax.set_ylabel("20-day loss (% of capital)", fontsize=15)
            if row == len(tickers) - 1:
                ax.set_xlabel("Window end date", fontsize=15)
            ax.tick_params(axis="both", labelsize=12)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=13, loc="upper left")

    fig.suptitle(
        "Rolling 20-Day Downside Risk (256-Day Information Window)",
        fontsize=21,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig
