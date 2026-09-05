"""Return, drawdown and dependence summaries for prior and posterior checks."""
from itertools import combinations
from typing import Sequence
import numpy as np
import pandas as pd

from .multivariate import ASSET_NAMES

RETURN_STAT_NAMES = (
    "mean_pct",
    "avg_abs_return_pct",
    "volatility_pct",
    "q01_pct",
    "q05_pct",
    "expected_shortfall_5_pct",
    "max_drawdown_pct",
)


def statistics(x):
    x = np.atleast_2d(np.asarray(x, dtype="float64"))
    quantiles = np.quantile(x, [0.01, 0.05, 0.95, 0.99], axis=1)
    tail = x <= quantiles[1, :, None]
    wealth = np.concatenate((np.ones((len(x), 1)), np.cumprod(1 + x, axis=1)), axis=1)
    return pd.DataFrame(
        {
            "mean_pct": 100 * x.mean(1),
            "volatility_pct": 100 * x.std(1, ddof=1),
            "avg_abs_return_pct": 100 * np.abs(x).mean(1),
            "q01_pct": 100 * quantiles[0],
            "q05_pct": 100 * quantiles[1],
            "q95_pct": 100 * quantiles[2],
            "q99_pct": 100 * quantiles[3],
            "es05_pct": 100 * np.sum(x * tail, axis=1) / tail.sum(1),
            "max_drawdown_pct": 100
            * np.min(wealth / np.maximum.accumulate(wealth, axis=1) - 1, axis=1),
            "terminal_wealth": wealth[:, -1],
        }
    )


def pushforward_envelope_table(
    prior_stats: pd.DataFrame,
    observed_stats: pd.DataFrame,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:
    """Compare observed summaries with a central prior pushforward envelope."""

    lower_name = f"prior_q{int(round(100 * lower)):02d}_pct"
    upper_name = f"prior_q{int(round(100 * upper)):02d}_pct"
    bands = prior_stats.quantile([lower, upper]).T.rename(
        columns={lower: lower_name, upper: upper_name}
    )
    rows: list[dict[str, object]] = []

    for ticker, observed_row in observed_stats.iterrows():
        for stat, observed_value in observed_row.items():
            prior_low = float(bands.loc[stat, lower_name])
            prior_high = float(bands.loc[stat, upper_name])
            rows.append(
                {
                    "ticker": ticker,
                    "stat": stat,
                    "observed_pct": float(observed_value),
                    lower_name: prior_low,
                    upper_name: prior_high,
                    "inside_prior_90pct_band": prior_low <= observed_value <= prior_high,
                }
            )
    return pd.DataFrame(rows)


def max_drawdown_batch(returns: np.ndarray) -> np.ndarray:
    """Vectorized maximum drawdown over time axis 1."""

    values = np.asarray(returns, dtype=np.float64)
    wealth = np.concatenate((np.ones_like(values[:, :1]), np.cumprod(1.0 + values, axis=1)), axis=1)
    running_peak = np.maximum.accumulate(wealth, axis=1)
    return np.min(wealth / running_peak - 1.0, axis=1)


def return_statistics_batch(returns: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Return per-path, per-asset percent-scale pushforward statistics."""

    values = np.asarray(returns, dtype=np.float64)
    threshold = np.quantile(values, alpha, axis=1, keepdims=True)
    tail_mask = values <= threshold
    expected_shortfall = np.sum(values * tail_mask, axis=1) / np.sum(tail_mask, axis=1)
    return {
        "mean_pct": 100.0 * np.mean(values, axis=1),
        "avg_abs_return_pct": 100.0 * np.mean(np.abs(values), axis=1),
        "volatility_pct": 100.0 * np.std(values, axis=1, ddof=1),
        "q01_pct": 100.0 * np.quantile(values, 0.01, axis=1),
        "q05_pct": 100.0 * np.quantile(values, 0.05, axis=1),
        "expected_shortfall_5_pct": 100.0 * expected_shortfall,
        "max_drawdown_pct": 100.0 * max_drawdown_batch(values),
    }


def marginal_pushforward_table(
    prior_returns: np.ndarray,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = ASSET_NAMES,
    interval: float = 0.90,
) -> pd.DataFrame:
    """Compare observed marginal statistics to central prior intervals."""

    assets = tuple(assets)
    prior_stats = return_statistics_batch(prior_returns)
    observed_array = observed_returns.loc[:, assets].to_numpy(dtype=np.float64)[None, ...]
    observed_stats = return_statistics_batch(observed_array)
    tail = 0.5 * (1.0 - interval)
    lower_label = int(round(100 * tail))
    upper_label = int(round(100 * (1.0 - tail)))
    rows = []
    for asset_index, asset in enumerate(assets):
        for statistic in RETURN_STAT_NAMES:
            lower, upper = np.quantile(prior_stats[statistic][:, asset_index], [tail, 1.0 - tail])
            observed = float(observed_stats[statistic][0, asset_index])
            rows.append(
                {
                    "asset": asset,
                    "stat": statistic,
                    "observed_pct": observed,
                    f"prior_q{lower_label:02d}_pct": float(lower),
                    f"prior_q{upper_label:02d}_pct": float(upper),
                    "inside_prior_interval": bool(lower <= observed <= upper),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_correlations(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    covariance = np.einsum("bti,btj->bij", centered, centered)
    scale = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    return covariance / (scale[:, :, None] * scale[:, None, :])


def dependence_statistics(
    returns: np.ndarray,
    assets: Sequence[str] = ASSET_NAMES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute raw, absolute, and joint-downside dependence for every path."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    raw_correlation = _pairwise_correlations(values)
    absolute_correlation = _pairwise_correlations(np.abs(values))
    thresholds = np.quantile(values, alpha, axis=1, keepdims=True)
    downside = values <= thresholds
    rows = []
    for left, right in combinations(range(len(assets)), 2):
        joint_downside = np.mean(downside[:, :, left] & downside[:, :, right], axis=1)
        pair = f"{assets[left]}–{assets[right]}"
        for draw_index in range(values.shape[0]):
            rows.append(
                {
                    "draw": draw_index,
                    "pair": pair,
                    "return_correlation": raw_correlation[draw_index, left, right],
                    "absolute_return_correlation": absolute_correlation[draw_index, left, right],
                    "joint_downside_probability": joint_downside[draw_index],
                }
            )
    return pd.DataFrame(rows)


def dependence_pushforward_table(
    prior_returns: np.ndarray,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = ASSET_NAMES,
    interval: float = 0.90,
) -> pd.DataFrame:
    """Compare observed dependence statistics to prior intervals."""

    prior = dependence_statistics(prior_returns, assets=assets)
    observed = dependence_statistics(
        observed_returns.loc[:, list(assets)].to_numpy(), assets=assets
    )
    statistics = (
        "return_correlation",
        "absolute_return_correlation",
        "joint_downside_probability",
    )
    tail = 0.5 * (1.0 - interval)
    rows = []
    for pair in prior["pair"].drop_duplicates():
        prior_pair = prior.loc[prior["pair"] == pair]
        observed_pair = observed.loc[observed["pair"] == pair].iloc[0]
        for statistic in statistics:
            lower, upper = prior_pair[statistic].quantile([tail, 1.0 - tail])
            observed_value = float(observed_pair[statistic])
            scale = 100.0 if statistic == "joint_downside_probability" else 1.0
            rows.append(
                {
                    "pair": pair,
                    "stat": statistic,
                    "observed": scale * observed_value,
                    "prior_q05": scale * float(lower),
                    "prior_q95": scale * float(upper),
                    "unit": "%" if scale == 100.0 else "correlation",
                    "inside_prior_90pct_interval": bool(lower <= observed_value <= upper),
                }
            )
    return pd.DataFrame(rows)


def equal_weight_portfolio_returns(returns: np.ndarray) -> np.ndarray:
    """Compute daily rebalanced equal-weight portfolio returns."""

    return np.mean(np.asarray(returns, dtype=np.float64), axis=-1)
