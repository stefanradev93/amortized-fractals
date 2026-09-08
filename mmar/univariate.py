"""Univariate MMAR: a randomized block cascade and Student-t innovations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = 256
PARAMETER_NAMES = ("mu", "sigma_bar", "q", "nu")
PARAMETER_LABELS = (r"$\mu$", r"$\bar\sigma$", r"$q$", r"$\nu$")


@dataclass(frozen=True)
class Prior:
    mu_loc: float = 0.0
    mu_sd: float = 0.0015
    sigma_low: float = 0.005
    sigma_high: float = 0.025
    q_low: float = 0.52
    q_high: float = 0.90
    nu_low: float = 2.05
    nu_high: float = 12.0


PRIOR = Prior()
CASCADE_BLOCK = 32


def prior_table(prior=PRIOR):
    return pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "prior": [
                f"Normal({prior.mu_loc:g}, {prior.mu_sd:g}²)",
                f"Uniform({prior.sigma_low:g}, {prior.sigma_high:g})",
                f"Uniform({prior.q_low:g}, {prior.q_high:g})",
                f"2 + exp(Uniform(log({prior.nu_low - 2:.4g}), log({prior.nu_high - 2:g})))",
            ],
            "meaning": [
                "Daily drift",
                "Baseline RMS return scale",
                "Cascade contrast and volatility intermittency",
                "Student-t degrees of freedom and tail thickness",
            ],
        }
    ).set_index("parameter")


def draw_parameters(n, rng, prior=PRIOR):
    sigma = rng.uniform(prior.sigma_low, prior.sigma_high, n)
    q = rng.uniform(prior.q_low, prior.q_high, n)
    nu = 2 + np.exp(rng.uniform(np.log(prior.nu_low - 2), np.log(prior.nu_high - 2), n))
    mu = rng.normal(prior.mu_loc, prior.mu_sd, n)
    return np.column_stack((mu, sigma, q, nu))


def stack_samples(samples):
    return np.concatenate([samples[key] for key in samples], axis=-1)


def cascade(q, rng):
    """Eight 32-day volatility blocks, with a random circular phase."""
    weights = np.ones((len(q), 1))
    high, low = 2 * q[:, None], 2 * (1 - q[:, None])
    for _ in range(int(np.log2(WINDOW // CASCADE_BLOCK))):
        left = rng.random(weights.shape) < 0.5
        weights = np.stack(
            (weights * np.where(left, high, low), weights * np.where(left, low, high)), axis=-1
        ).reshape(len(q), -1)
    weights /= weights.mean(axis=1, keepdims=True)
    weights = np.repeat(weights, CASCADE_BLOCK, axis=1)
    offsets = rng.integers(0, WINDOW, size=(len(q), 1))
    return np.take_along_axis(weights, (np.arange(WINDOW)[None, :] + offsets) % WINDOW, axis=1)


def simulate_from_parameters(parameters, rng):
    """One 256-day path per parameter vector; redraw paths crossing -100%."""
    mu, sigma, q, nu = parameters.T
    n = len(parameters)
    returns = np.empty((n, WINDOW))
    weights = np.empty_like(returns)
    pending = np.arange(n)
    for _ in range(100):
        weights[pending] = cascade(q[pending], rng)
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
        "returns": returns, 
        "trading_time_increments": weights,
    }


def simulate(n, seed=None, prior=PRIOR):
    rng = np.random.default_rng(seed)
    return simulate_from_parameters(draw_parameters(n, rng, prior), rng)


def forward_paths(parameters, horizon, rng):
    """Fresh segments of the training model, not a shorter-depth cascade."""
    paths = np.empty((len(parameters), horizon), dtype="float32")
    for start in range(0, len(parameters), 512):
        paths[start : start + 512] = simulate_from_parameters(parameters[start : start + 512], rng)[
            "returns"
        ][:, :horizon]
    return paths


def posterior_resimulations(posterior, horizon=WINDOW, seed=None):
    rng = np.random.default_rng(seed)
    return np.stack([forward_paths(draws, horizon, rng) for draws in posterior])


def conditional_horizon_returns(parameter_draws, horizon, n_replications, rng):
    paths = forward_paths(np.repeat(parameter_draws, n_replications, axis=0), horizon, rng)
    return (np.prod(1 + paths.astype("float64"), axis=1) - 1).reshape(
        len(parameter_draws), n_replications
    )


def q_scenarios(q_values, n_paths=4, seed=20260906):
    """Matched orientations, phase and shocks; only q changes. Returns in %."""
    return {
        float(q): 100
        * forward_paths(
            np.tile([0, 0.015, q, 4.0], (n_paths, 1)), WINDOW, np.random.default_rng(seed)
        )
        for q in q_values
    }
