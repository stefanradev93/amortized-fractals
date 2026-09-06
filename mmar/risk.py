"""Rolling posterior-predictive risk with a matched historical benchmark."""
from typing import Sequence
import numpy as np
import pandas as pd
from . import univariate, multivariate


def rolling_return_windows(
    returns: pd.DataFrame,
    tickers: Sequence[str],
    window_size: int = 256,
    step: int = 21,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create fixed-length, ticker-major windows and their metadata."""

    windows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        series = returns[ticker].dropna()
        endpoints = list(range(window_size, len(series) + 1, step))
        if endpoints[-1] != len(series):
            endpoints.append(len(series))
        for end in endpoints:
            window = series.iloc[end - window_size : end]
            windows.append(window.to_numpy(dtype="float32"))
            rows.append(
                {
                    "ticker": ticker,
                    "as_of": window.index[-1],
                    "window_start": window.index[0],
                    "series_end_position": end,
                }
            )

    return np.stack(windows), pd.DataFrame(rows)


def _historical_horizon_returns(window: np.ndarray, horizon: int) -> np.ndarray:
    """Return overlapping, compounded horizon returns from an observed window."""

    values = np.asarray(window, dtype="float64")
    rolling = np.lib.stride_tricks.sliding_window_view(values, horizon, axis=0)
    return np.prod(1.0 + rolling, axis=-1) - 1.0


def _loss_var_and_es(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """Compute positive-loss VaR and expected shortfall in percentage points."""

    values = np.asarray(returns, dtype="float64")
    threshold = float(np.quantile(values, alpha))
    tail = values[values <= threshold]
    return -100.0 * threshold, -100.0 * float(np.mean(tail))


def posterior_risk_table(
    posterior_samples: np.ndarray,
    metadata: pd.DataFrame,
    observed_returns: pd.DataFrame,
    horizon: int = 20,
    n_parameter_draws: int = 160,
    n_replications: int = 192,
    tail_alpha: float = 0.05,
    seed: int = 20260903,
) -> pd.DataFrame:
    """Turn rolling posterior draws into risk and parameter-uncertainty metrics.

    For every rolling window, repeated paths are simulated conditionally on
    multiple posterior parameter draws. Their pooled distribution determines
    posterior-predictive VaR/ES. The distribution of conditional VaR/ES values
    across parameter draws provides a 90% posterior uncertainty interval.

    A model-free historical-simulation estimate from all overlapping horizon
    returns in the same 256-day information window is included as a benchmark.
    It uses only observations available as of the window end. The realized
    forward return is retrospective evaluation context only and is never used
    to estimate VaR or ES.
    """

    posterior = np.asarray(posterior_samples)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for batch_id, meta in metadata.reset_index(drop=True).iterrows():
        ticker = str(meta["ticker"])
        available_draws = posterior.shape[1]
        draw_ids = rng.choice(
            available_draws,
            size=min(n_parameter_draws, available_draws),
            replace=False,
        )
        conditional_returns = univariate.conditional_horizon_returns(
            posterior[batch_id, draw_ids],
            horizon=horizon,
            n_replications=n_replications,
            rng=rng,
        )
        conditional_returns = np.asarray(conditional_returns, dtype="float64")
        predictive = conditional_returns.reshape(-1)
        predictive_var, predictive_es = _loss_var_and_es(predictive, tail_alpha)
        conditional_var = np.empty(len(draw_ids), dtype="float64")
        conditional_es = np.empty(len(draw_ids), dtype="float64")
        for draw_id, draw_returns in enumerate(conditional_returns):
            conditional_var[draw_id], conditional_es[draw_id] = _loss_var_and_es(
                draw_returns, tail_alpha
            )

        series = observed_returns[ticker].dropna()
        as_of = pd.Timestamp(meta["as_of"])
        window_start = pd.Timestamp(meta["window_start"])
        historical_window = series.loc[window_start:as_of].iloc[-256:].to_numpy()
        historical_returns = _historical_horizon_returns(historical_window, horizon)
        historical_var, historical_es = _loss_var_and_es(historical_returns, tail_alpha)
        future = series.loc[series.index > as_of].iloc[:horizon]
        realized = (
            100.0 * (float(np.prod(1.0 + future)) - 1.0) if len(future) == horizon else np.nan
        )

        rows.append(
            {
                "ticker": ticker,
                "window_start": window_start,
                "as_of": as_of,
                "horizon_days": horizon,
                "var_95_pct": predictive_var,
                "var_95_posterior_q05_pct": float(np.quantile(conditional_var, 0.05)),
                "var_95_posterior_q95_pct": float(np.quantile(conditional_var, 0.95)),
                "historical_var_95_pct": historical_var,
                "expected_shortfall_95_pct": predictive_es,
                "es_95_posterior_q05_pct": float(np.quantile(conditional_es, 0.05)),
                "es_95_posterior_q95_pct": float(np.quantile(conditional_es, 0.95)),
                "historical_es_95_pct": historical_es,
                "predictive_return_median_pct": 100.0 * float(np.median(predictive)),
                "predictive_return_q04_pct": 100.0 * float(np.quantile(predictive, 0.04)),
                "predictive_return_q96_pct": 100.0 * float(np.quantile(predictive, 0.96)),
                "realized_forward_return_pct": realized,
            }
        )

    return pd.DataFrame(rows)


def joint_rolling_return_windows(
    returns: pd.DataFrame,
    assets: Sequence[str] = multivariate.ASSET_NAMES,
    window_size: int = 256,
    step: int = 21,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create synchronized rolling windows with shape ``(window, time, asset)``."""

    assets = tuple(assets)
    common = returns.loc[:, list(assets)].dropna()
    endpoints = list(range(window_size, len(common) + 1, step))
    if endpoints[-1] != len(common):
        endpoints.append(len(common))

    windows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for end in endpoints:
        window = common.iloc[end - window_size : end]
        windows.append(window.to_numpy(dtype=np.float32))
        rows.append(
            {
                "window_start": window.index[0],
                "as_of": window.index[-1],
                "series_end_position": end,
            }
        )
    return np.stack(windows), pd.DataFrame(rows)


def multivariate_posterior_risk_table(
    posterior_samples: dict[str, np.ndarray] | np.ndarray,
    metadata: pd.DataFrame,
    observed_returns: pd.DataFrame,
    assets: Sequence[str] = multivariate.ASSET_NAMES,
    window_size: int = 256,
    horizon: int = 20,
    n_parameter_draws: int = 160,
    n_replications: int = 192,
    tail_alpha: float = 0.05,
    seed: int = 20260905,
) -> pd.DataFrame:
    """Compute joint posterior-predictive VaR/ES for assets and EW portfolio."""

    posterior = multivariate.stack_parameters(posterior_samples)

    assets = tuple(assets)
    targets = (*assets, "EW")
    common = observed_returns.loc[:, list(assets)].dropna()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for batch_id, meta in metadata.reset_index(drop=True).iterrows():
        available_draws = posterior.shape[1]
        draw_ids = rng.choice(
            available_draws,
            size=min(n_parameter_draws, available_draws),
            replace=False,
        )
        selected = posterior[batch_id, draw_ids]
        repeated = np.repeat(selected, n_replications, axis=0)
        joint_paths = multivariate.forward_paths(
            repeated,
            horizon=horizon,
            rng=rng,
        ).reshape(len(draw_ids), n_replications, horizon, len(assets))
        daily_targets = np.concatenate(
            (joint_paths, joint_paths.mean(axis=-1, keepdims=True)),
            axis=-1,
        )
        terminal_returns = np.prod(1.0 + daily_targets, axis=2) - 1.0

        as_of = pd.Timestamp(meta["as_of"])
        window_start = pd.Timestamp(meta["window_start"])
        historical_window = common.loc[window_start:as_of].iloc[-window_size:]
        historical_daily = historical_window.to_numpy(dtype=np.float64)
        historical_daily = np.column_stack((historical_daily, historical_daily.mean(axis=1)))
        historical_terminal = _historical_horizon_returns(
            historical_daily,
            horizon,
        )

        future = common.loc[common.index > as_of].iloc[:horizon]
        if len(future) == horizon:
            future_daily = future.to_numpy(dtype=np.float64)
            future_daily = np.column_stack((future_daily, future_daily.mean(axis=1)))
            realized = 100.0 * (np.prod(1.0 + future_daily, axis=0) - 1.0)
        else:
            realized = np.full(len(targets), np.nan)

        for target_id, target in enumerate(targets):
            conditional = terminal_returns[:, :, target_id]
            predictive_var, predictive_es = _loss_var_and_es(conditional.reshape(-1), tail_alpha)
            conditional_risk = np.asarray(
                [_loss_var_and_es(draw, tail_alpha) for draw in conditional]
            )
            historical_var, historical_es = _loss_var_and_es(
                historical_terminal[:, target_id], tail_alpha
            )
            rows.append(
                {
                    "target": target,
                    "window_start": window_start,
                    "as_of": as_of,
                    "horizon_days": horizon,
                    "var_95_pct": predictive_var,
                    "var_95_posterior_q05_pct": float(np.quantile(conditional_risk[:, 0], 0.05)),
                    "var_95_posterior_q95_pct": float(np.quantile(conditional_risk[:, 0], 0.95)),
                    "historical_var_95_pct": historical_var,
                    "expected_shortfall_95_pct": predictive_es,
                    "es_95_posterior_q05_pct": float(np.quantile(conditional_risk[:, 1], 0.05)),
                    "es_95_posterior_q95_pct": float(np.quantile(conditional_risk[:, 1], 0.95)),
                    "historical_es_95_pct": historical_es,
                    "predictive_return_median_pct": 100.0 * float(np.median(conditional)),
                    "realized_forward_return_pct": float(realized[target_id]),
                }
            )
    return pd.DataFrame(rows)
