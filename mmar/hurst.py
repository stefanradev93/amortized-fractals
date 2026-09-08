"""Multivariate MMAR with asset-specific fractional-Gaussian driver memory.

Notation used in the model description:

``g`` is the IID standard-Gaussian seed, ``z^(H)`` is the unit-variance
fractional-Gaussian driver, and ``epsilon`` is the final Student-t innovation
after applying the shared radial scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import multivariate as base

ASSET_NAMES = base.ASSET_NAMES
ASSET_LABELS = base.ASSET_LABELS
WINDOW = base.WINDOW
BASE_PARAMETER_COUNT = 13
INFERENCE_VARIABLES = (*base.INFERENCE_VARIABLES, "hurst")


@dataclass(frozen=True)
class HurstPrior:
    base: base.MultivariateMMARPrior = base.PRIOR
    hurst_low: float = 0.35
    hurst_high: float = 0.75


PRIOR = HurstPrior()


def draw_prior(n, prior=PRIOR, rng=None):
    """Draw the baseline 13 parameters and one Hurst coefficient per asset."""
    rng = np.random.default_rng(rng)
    parameters = base.draw_multivariate_prior(n, prior.base, rng)
    parameters["hurst"] = rng.uniform(
        prior.hurst_low, prior.hurst_high, size=(n, len(ASSET_NAMES))
    )
    return parameters


def stack_parameters(parameters):
    """Stack the baseline coordinates followed by the three Hurst coefficients."""
    if isinstance(parameters, dict):
        return np.concatenate([parameters[name] for name in INFERENCE_VARIABLES], axis=-1)
    values = np.asarray(parameters, dtype="float64")
    if values.shape[-1] != BASE_PARAMETER_COUNT + len(ASSET_NAMES):
        raise ValueError("Expected 16 parameters: 13 baseline coordinates and three H values.")
    return values


def q_hurst_posterior_draws(posterior):
    """Extract the asset-specific cascade and Hurst draws from a posterior array."""
    values = stack_parameters(posterior)
    dimension = len(ASSET_NAMES)
    q = values[..., 2 * dimension : 3 * dimension]
    hurst = values[..., BASE_PARAMETER_COUNT : BASE_PARAMETER_COUNT + dimension]
    return q, hurst


def q_hurst_posterior_summary(posterior, interval_probability=0.92, index_name="ETF"):
    """Summarize cascade strength and fractional-Gaussian driver memory."""
    if not 0 < interval_probability < 1:
        raise ValueError("interval_probability must lie strictly between zero and one.")
    q, hurst = q_hurst_posterior_draws(posterior)
    if q.ndim != 2:
        raise ValueError("Expected posterior draws for one observed data set.")

    dimension = len(ASSET_NAMES)
    tail_probability = (1 - interval_probability) / 2
    quantiles = (tail_probability, 1 - tail_probability)
    q_interval = np.quantile(q, quantiles, axis=0)
    hurst_interval = np.quantile(hurst, quantiles, axis=0)
    interval_label = f"{100 * interval_probability:g}% ETI"

    return pd.DataFrame(
        {
            "$q$ median": np.median(q, axis=0),
            f"$q$ {interval_label}": [
                f"[{low:.3f}, {high:.3f}]" for low, high in q_interval.T
            ],
            "$H$ median": np.median(hurst, axis=0),
            f"$H$ {interval_label}": [
                f"[{low:.3f}, {high:.3f}]" for low, high in hurst_interval.T
            ],
            "$P(H>0.5)$": np.mean(hurst > 0.5, axis=0),
            r"$\operatorname{corr}(q,H)$": [
                np.corrcoef(q[:, i], hurst[:, i])[0, 1] for i in range(dimension)
            ],
        },
        index=pd.Index(ASSET_NAMES, name=index_name),
    )


def style_q_hurst_posterior_summary(posterior, interval_probability=0.92, index_name="ETF"):
    """Return a consistently formatted q/H posterior summary for notebook display."""
    table = q_hurst_posterior_summary(posterior, interval_probability, index_name)
    return table.style.format(
        {
            "$q$ median": "{:.3f}",
            "$H$ median": "{:.3f}",
            "$P(H>0.5)$": "{:.1%}",
            r"$\operatorname{corr}(q,H)$": "{:+.2f}",
        }
    )


def q_hurst_posterior_correlations(posterior):
    """Return all posterior correlations between q and H coordinates."""
    q, hurst = q_hurst_posterior_draws(posterior)
    if q.ndim != 2:
        raise ValueError("Expected posterior draws for one observed data set.")
    dimension = len(ASSET_NAMES)
    correlations = np.corrcoef(np.column_stack((q, hurst)), rowvar=False)[
        :dimension, dimension:
    ]
    return pd.DataFrame(
        correlations,
        index=[rf"$q_{{\mathrm{{{asset}}}}}$" for asset in ASSET_NAMES],
        columns=[rf"$H_{{\mathrm{{{asset}}}}}$" for asset in ASSET_NAMES],
    )


def style_q_hurst_posterior_correlations(posterior):
    """Return the q/H posterior-correlation table formatted for display."""
    return (
        q_hurst_posterior_correlations(posterior)
        .style.format("{:+.2f}")
        .set_caption("Posterior correlations between cascade and Hurst coordinates")
    )


def fractional_gaussian_autocovariance(hurst, num_observations=WINDOW):
    """Return the unit-variance ``z^(H)`` autocovariance for lags 0,...,T-1."""
    hurst = np.asarray(hurst)
    lags = np.arange(num_observations)
    exponent = 2 * hurst[..., None]
    return 0.5 * (
        (lags + 1) ** exponent
        - 2 * lags**exponent
        + np.abs(lags - 1) ** exponent
    )


def standardized_fractional_multivariate_t(
    hurst,
    nu,
    correlation,
    num_observations,
    rng,
):
    """Draw final Student-t innovations from a fractional-Gaussian driver.

    An IID Gaussian seed ``g`` is first mixed across assets according to
    ``correlation`` and then filtered into the unit-variance driver ``z^(H)``.
    A shared, independently drawn chi-square radial scale converts each daily
    Gaussian cross-section into the final innovation ``epsilon`` with
    Student-t marginals. The requested fGn autocovariance is therefore exact
    for ``z^(H)``; the radial scales attenuate nonzero-lag linear correlations
    in ``epsilon``. At H=0.5 this reduces to the baseline standardized
    multivariate Student-t innovation.
    """
    hurst = np.asarray(hurst, dtype="float64")
    correlation = np.asarray(correlation, dtype="float64")
    n, dimension = hurst.shape
    autocovariance = fractional_gaussian_autocovariance(hurst, num_observations)
    circulant_row = np.concatenate(
        (
            autocovariance,
            np.zeros((n, dimension, 1)),
            autocovariance[..., 1:][..., ::-1],
        ),
        axis=-1,
    )
    eigenvalues = np.fft.rfft(circulant_row, axis=-1).real
    tolerance = 1e-9 * np.maximum(1.0, np.max(eigenvalues, axis=-1, keepdims=True))
    if np.any(eigenvalues < -tolerance):
        raise RuntimeError("The Davies-Harte circulant embedding is not positive semidefinite.")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    embedding_size = 2 * num_observations
    gaussian_seed = rng.normal(size=(n, embedding_size, dimension))
    cholesky = np.linalg.cholesky(correlation)
    correlated_seed = np.einsum("bti,bji->btj", gaussian_seed, cholesky)
    frequencies = np.fft.rfft(correlated_seed, axis=1)
    fractional_driver = np.fft.irfft(
        frequencies * np.sqrt(np.moveaxis(eigenvalues, 1, 2)),
        n=embedding_size,
        axis=1,
    )[:, :num_observations]

    degrees = np.asarray(nu).reshape(n, 1, 1)
    radial_chi_square = rng.chisquare(df=degrees, size=(n, num_observations, 1))
    return fractional_driver * np.sqrt((degrees - 2.0) / radial_chi_square)


def simulate_from_parameters(parameters, rng, batch_size=512):
    """Simulate one valid 256-day joint path per 16-parameter vector."""
    parameters = stack_parameters(parameters)
    returns = np.empty((len(parameters), WINDOW, len(ASSET_NAMES)))
    trading_time = np.empty_like(returns)
    for start in range(0, len(parameters), batch_size):
        stop = min(start + batch_size, len(parameters))
        batch = parameters[start:stop]
        mu, sigma, q = batch[:, :3], batch[:, 3:6], batch[:, 6:9]
        nu = batch[:, 9:10]
        correlation = base.correlation_partials_to_matrix(batch[:, 10:13])
        hurst = batch[:, 13:16]
        pending = np.arange(len(batch))
        for _ in range(100):
            weights = base.shared_orientation_cascade_weights(q[pending], WINDOW, rng)
            innovations = standardized_fractional_multivariate_t(
                hurst[pending], nu[pending], correlation[pending], WINDOW, rng
            )
            candidates = (
                mu[pending, None]
                + sigma[pending, None] * np.sqrt(weights) * innovations
            )
            invalid = (~np.isfinite(candidates).all(axis=(1, 2))) | (
                candidates <= -1
            ).any(axis=(1, 2))
            accepted = ~invalid
            target = start + pending[accepted]
            returns[target] = candidates[accepted]
            trading_time[target] = weights[accepted]
            pending = pending[invalid]
            if not len(pending):
                break
        else:
            raise RuntimeError("Could not draw valid joint simple-return paths in 100 attempts.")
    return {"returns": returns, "trading_time_increments": trading_time}


def simulate(n, seed=None, prior=PRIOR):
    rng = np.random.default_rng(seed)
    parameters = draw_prior(n, prior, rng)
    return {
        **{key: value for key, value in parameters.items()},
        **simulate_from_parameters(stack_parameters(parameters), rng),
    }


def posterior_resimulations(posterior, seed=None):
    rng = np.random.default_rng(seed)
    return simulate_from_parameters(stack_parameters(posterior), rng)["returns"]


def prior_table(prior=PRIOR):
    table = base.prior_table(prior.base).copy()
    table.loc["correlation", "meaning"] = (
        "Pre-filter Gaussian coherence R; inferred through 3 vine coordinates"
    )
    additions = pd.DataFrame(
        {
            "prior": [f"Uniform({prior.hurst_low:g}, {prior.hurst_high:g})"]
            * len(ASSET_NAMES),
            "meaning": [
                f"{asset} fractional-Gaussian driver memory" for asset in ASSET_NAMES
            ],
        },
        index=[f"H[{asset}]" for asset in ASSET_NAMES],
    )
    additions.index.name = "parameter"
    return pd.concat((table, additions))


PARAMETER_LABELS = (
    *base.PARAMETER_LABELS,
    *(rf"$H_{{\mathrm{{{asset}}}}}$" for asset in ASSET_NAMES),
)
