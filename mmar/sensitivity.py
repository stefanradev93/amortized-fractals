"""Univariate MMAR sensitivity to the number of volatility clusters."""

from collections.abc import Iterable

import numpy as np

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
        shocks = rng.standard_t(df, size=(len(pending), WINDOW)) * np.sqrt((df - 2) / df)
        returns[pending] = (
            mu[pending, None] + sigma[pending, None] * np.sqrt(weights[pending]) * shocks
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
        "cascade": weights,
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
    observed = np.asarray(observed_returns, dtype="float32")
    if observed.ndim != 2 or observed.shape[1] != WINDOW:
        raise ValueError("Expected observed returns with shape (assets, 256).")
    d_values = tuple(d_values)
    wealth = np.empty((len(observed), len(d_values), num_samples, WINDOW + 1))
    rng = np.random.default_rng(seed)
    for column, d in enumerate(d_values):
        conditions = {
            "returns": observed[..., None],
            "log_d": np.full((len(observed), 1), np.log(d), dtype="float32"),
        }
        draws = workflow.sample(conditions=conditions, num_samples=num_samples, seed=seed + column)
        posterior = stack_parameters(draws)
        for asset in range(len(observed)):
            paths = simulate_from_parameters(posterior[asset], d, rng)["returns"]
            wealth[asset, column, :, 0] = 1.0
            wealth[asset, column, :, 1:] = np.cumprod(1 + paths, axis=1)
    return wealth
