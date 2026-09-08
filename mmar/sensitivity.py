"""Univariate MMAR sensitivity to the number of volatility clusters."""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from . import univariate as base

WINDOW = base.WINDOW
PRIOR = base.PRIOR
PARAMETER_NAMES = base.PARAMETER_NAMES
PARAMETER_LABELS = base.PARAMETER_LABELS
CLUSTER_COUNTS = (128, 64, 32, 16, 8, 4)


def _cluster_counts(n: int, d, rng: np.random.Generator) -> np.ndarray:
    if d is None:
        values = rng.choice(CLUSTER_COUNTS, size=n)
    elif np.ndim(d) == 0:
        values = np.full(n, d)
    else:
        values = np.asarray(d)
        if values.shape != (n,):
            raise ValueError(f"Expected one d per simulation, got shape {values.shape}.")
    counts = np.asarray(values, dtype=int)
    if not np.array_equal(counts, np.asarray(values)):
        raise ValueError("Every d must be an integer cluster count.")
    if not np.isin(counts, CLUSTER_COUNTS).all():
        raise ValueError(f"d must be one of {CLUSTER_COUNTS}.")
    return counts


def cascade(q: np.ndarray, d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw cascades with one terminal cluster count per simulation."""
    weights = np.empty((len(q), WINDOW))
    for count in np.unique(d):
        indices = np.flatnonzero(d == count)
        cluster_weights = np.ones((len(indices), 1))
        high, low = 2 * q[indices, None], 2 * (1 - q[indices, None])
        for _ in range(int(np.log2(count))):
            left = rng.random(cluster_weights.shape) < 0.5
            cluster_weights = np.stack(
                (
                    cluster_weights * np.where(left, high, low),
                    cluster_weights * np.where(left, low, high),
                ),
                axis=-1,
            ).reshape(len(indices), -1)
        cluster_weights /= cluster_weights.mean(axis=1, keepdims=True)
        daily = np.repeat(cluster_weights, WINDOW // count, axis=1)
        offsets = rng.integers(0, WINDOW, size=(len(indices), 1))
        weights[indices] = np.take_along_axis(
            daily, (np.arange(WINDOW)[None, :] + offsets) % WINDOW, axis=1
        )
    return weights


def simulate_from_parameters(parameters, d, rng: np.random.Generator):
    """Simulate one valid 256-day return path per parameter vector and d."""
    parameters = np.asarray(parameters)
    if parameters.ndim != 2 or parameters.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("Expected parameters with shape (simulations, 4).")
    n = len(parameters)
    counts = _cluster_counts(n, d, rng)
    mu, sigma, q, nu = parameters.T
    returns = np.empty((n, WINDOW))
    weights = np.empty_like(returns)
    pending = np.arange(n)
    for _ in range(100):
        weights[pending] = cascade(q[pending], counts[pending], rng)
        df = nu[pending, None]
        innovations = rng.standard_t(df, size=(len(pending), WINDOW)) * np.sqrt(
            (df - 2) / df
        )
        returns[pending] = (
            mu[pending, None]
            + sigma[pending, None] * np.sqrt(weights[pending]) * innovations
        )
        invalid = (~np.isfinite(returns[pending]).all(1)) | (returns[pending] <= -1).any(1)
        pending = pending[invalid]
        if not len(pending):
            break
    else:
        raise RuntimeError("Could not draw valid simple-return paths in 100 attempts.")

    return {
        "mu": mu[:, None],
        "sigma_bar": sigma[:, None],
        "q": q[:, None],
        "nu": nu[:, None],
        "d": counts[:, None],
        "log_d": np.log(counts)[:, None],
        "returns": returns,
        "trading_time_increments": weights,
    }


def simulate(n, d=None, seed=None, prior=PRIOR):
    """Draw parameters and an independent d for every simulation."""
    rng = np.random.default_rng(seed)
    parameters = base.draw_parameters(n, rng, prior)
    return simulate_from_parameters(parameters, _cluster_counts(n, d, rng), rng)


def stack_parameters(data):
    return np.concatenate([np.asarray(data[name]) for name in PARAMETER_NAMES], axis=-1)


def recovery_samples(
    workflow,
    d_values: Iterable[int] = CLUSTER_COUNTS,
    n_test=80,
    num_samples=256,
    batch_size=40,
):
    """Generate fixed-d test sets and their amortized posterior draws."""
    recovery = {}
    for d in d_values:
        test = simulate(n_test, d=d)
        draws = workflow.sample(
            conditions=test,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        recovery[int(d)] = (stack_parameters(draws), stack_parameters(test))
    return recovery


def posterior_predictive_wealth(
    workflow,
    observed_returns,
    d_values: Iterable[int] = CLUSTER_COUNTS,
    num_samples=400,
    seed=200,
):
    """Infer and resimulate observed windows under each fixed cluster count."""
    return posterior_predictive_analysis(
        workflow,
        observed_returns,
        d_values=d_values,
        num_samples=num_samples,
        seed=seed,
    )["wealth"]


def posterior_predictive_analysis(
    workflow,
    observed_returns,
    d_values: Iterable[int] = CLUSTER_COUNTS,
    num_samples=400,
    seed=200,
):
    """Return posterior parameters and predictive wealth for every fixed d."""
    observed = np.asarray(observed_returns, dtype="float32")
    if observed.ndim != 2 or observed.shape[1] != WINDOW:
        raise ValueError("Expected observed returns with shape (assets, 256).")
    d_values = tuple(d_values)
    posterior_parameters = np.empty(
        (len(observed), len(d_values), num_samples, len(PARAMETER_NAMES))
    )
    wealth = np.empty((len(observed), len(d_values), num_samples, WINDOW + 1))
    rng = np.random.default_rng(seed)
    for column, d in enumerate(d_values):
        conditions = {
            "returns": observed[..., None],
            "log_d": np.full((len(observed), 1), np.log(d), dtype="float32"),
        }
        draws = workflow.sample(conditions=conditions, num_samples=num_samples, seed=seed + column)
        posterior = stack_parameters(draws)
        posterior_parameters[:, column] = posterior
        for asset in range(len(observed)):
            paths = simulate_from_parameters(posterior[asset], d, rng)["returns"]
            wealth[asset, column, :, 0] = 1.0
            wealth[asset, column, :, 1:] = np.cumprod(1 + paths, axis=1)
    return {"parameters": posterior_parameters, "wealth": wealth}


def _median_spread_and_ratio(values: np.ndarray, interval: float) -> tuple[float, float]:
    """Across-d median range and its size relative to a typical interval width."""
    tail = 0.5 * (1.0 - interval)
    medians = np.median(values, axis=-1)
    lower, upper = np.quantile(values, (tail, 1.0 - tail), axis=-1)
    spread = float(np.ptp(medians))
    typical_width = float(np.median(upper - lower))
    ratio = spread / typical_width if typical_width > 0 else np.nan
    return spread, ratio


def resolution_sensitivity_summary(
    posterior_parameters,
    predictive_wealth,
    tickers: Sequence[str],
    d_values: Iterable[int] = CLUSTER_COUNTS,
    interval=0.92,
) -> pd.DataFrame:
    """Quantify how strongly q and predictive outcomes move across d.

    Each sensitivity ratio is the range of the per-d medians divided by the
    median within-d posterior or predictive interval width. Ratios near zero
    indicate that changing d matters little relative to uncertainty; ratios
    above one indicate a material resolution-dependent shift.
    """
    posterior = np.asarray(posterior_parameters, dtype="float64")
    wealth = np.asarray(predictive_wealth, dtype="float64")
    d_values = np.asarray(tuple(d_values), dtype=int)
    tickers = tuple(tickers)
    expected_posterior_shape = (len(tickers), len(d_values))
    if posterior.ndim != 4 or posterior.shape[:2] != expected_posterior_shape:
        raise ValueError("Posterior axes must match tickers and d values.")
    if posterior.shape[-1] != len(PARAMETER_NAMES):
        raise ValueError(f"Expected {len(PARAMETER_NAMES)} posterior parameters.")
    if wealth.ndim != 4 or wealth.shape[:2] != expected_posterior_shape:
        raise ValueError("Predictive-wealth axes must match tickers and d values.")
    if posterior.shape[2] != wealth.shape[2]:
        raise ValueError("Posterior and predictive sample counts must match.")
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must lie strictly between zero and one.")

    finest = int(np.max(d_values))
    coarsest = int(np.min(d_values))
    finest_index = int(np.flatnonzero(d_values == finest)[0])
    coarsest_index = int(np.flatnonzero(d_values == coarsest)[0])
    q_index = PARAMETER_NAMES.index("q")
    rows = []
    for asset, ticker in enumerate(tickers):
        q_draws = posterior[asset, :, :, q_index]
        q_medians = np.median(q_draws, axis=-1)
        q_spread, q_ratio = _median_spread_and_ratio(q_draws, interval)

        terminal_returns = 100.0 * (wealth[asset, :, :, -1] - 1.0)
        terminal_spread, terminal_ratio = _median_spread_and_ratio(
            terminal_returns, interval
        )

        running_peak = np.maximum.accumulate(wealth[asset], axis=-1)
        maximum_drawdowns = -100.0 * np.min(
            wealth[asset] / running_peak - 1.0, axis=-1
        )
        drawdown_spread, drawdown_ratio = _median_spread_and_ratio(
            maximum_drawdowns, interval
        )

        rows.append(
            {
                f"$q$ median, $d={finest}$": q_medians[finest_index],
                f"$q$ median, $d={coarsest}$": q_medians[coarsest_index],
                r"$\Delta_d q$ (coarse $-$ fine)": (
                    q_medians[coarsest_index] - q_medians[finest_index]
                ),
                r"$S_q$": q_ratio,
                r"$\Delta_d$ terminal return (pp)": terminal_spread,
                r"$S_{R_T}$": terminal_ratio,
                r"$\Delta_d$ max drawdown (pp)": drawdown_spread,
                r"$S_{\mathrm{MDD}}$": drawdown_ratio,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(tickers, name="ETF"))
