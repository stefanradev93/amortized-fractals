"""Static multivariate MMAR simulation, posterior prediction, and risk helpers.

The model uses the univariate prior ranges and 32-day block resolution, with a
shared cascade orientation tree, a shared circular shift, and standardized
multivariate Student-t innovations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .univariate import CASCADE_BLOCK, WINDOW, PRIOR as UNIVARIATE_PRIOR

ASSET_NAMES = ("VOO", "GLD", "TLT")
ASSET_LABELS = {
    "VOO": "US stocks",
    "GLD": "Gold",
    "TLT": "Long Treasuries",
}
INFERENCE_VARIABLES = (
    "mu",
    "sigma_bar",
    "q",
    "nu",
    "correlation_partials",
)


@dataclass(frozen=True)
class MultivariateMMARPrior:
    """Hyperparameters for a static, three-asset multivariate MMAR prior."""

    assets: tuple[str, ...] = ASSET_NAMES
    mu_loc: tuple[float, ...] = (UNIVARIATE_PRIOR.mu_loc,) * 3
    mu_scale: tuple[float, ...] = (UNIVARIATE_PRIOR.mu_sd,) * 3
    sigma_bar_low: tuple[float, ...] = (UNIVARIATE_PRIOR.sigma_low,) * 3
    sigma_bar_high: tuple[float, ...] = (UNIVARIATE_PRIOR.sigma_high,) * 3
    q_low: tuple[float, ...] = (UNIVARIATE_PRIOR.q_low,) * 3
    q_high: tuple[float, ...] = (UNIVARIATE_PRIOR.q_high,) * 3
    nu_low: float = UNIVARIATE_PRIOR.nu_low
    nu_high: float = UNIVARIATE_PRIOR.nu_high
    correlation_df: int = 7

    @property
    def dimension(self) -> int:
        return len(self.assets)


PRIOR = MultivariateMMARPrior()


def draw_correlation_matrices(
    num_draws: int,
    dimension: int,
    degrees_of_freedom: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw positive-definite zero-centered correlations via normalized Wishart matrices."""

    gaussian = rng.normal(size=(num_draws, degrees_of_freedom, dimension))
    scatter = np.einsum("bki,bkj->bij", gaussian, gaussian)
    standard_deviations = np.sqrt(np.diagonal(scatter, axis1=1, axis2=2))
    correlation = scatter / (standard_deviations[:, :, None] * standard_deviations[:, None, :])
    indices = np.arange(dimension)
    correlation[:, indices, indices] = 1.0
    return correlation


def correlation_matrix_to_partials(
    correlation: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Map a 3x3 correlation matrix to its three vine partial correlations.

    The coordinates are ``(rho_01, rho_02, rho_12_given_0)``.  Unlike the
    three raw off-diagonal entries, any coordinate vector strictly inside
    ``(-1, 1)^3`` reconstructs a positive-definite correlation matrix.
    """

    matrix = np.asarray(correlation, dtype=np.float64)
    rho_01 = matrix[..., 0, 1]
    rho_02 = matrix[..., 0, 2]
    rho_12 = matrix[..., 1, 2]
    denominator = np.sqrt(
        np.maximum(
            (1.0 - rho_01**2) * (1.0 - rho_02**2),
            np.finfo(np.float64).tiny,
        )
    )
    rho_12_given_0 = (rho_12 - rho_01 * rho_02) / denominator
    return np.stack(
        (
            np.clip(rho_01, -1.0 + epsilon, 1.0 - epsilon),
            np.clip(rho_02, -1.0 + epsilon, 1.0 - epsilon),
            np.clip(rho_12_given_0, -1.0 + epsilon, 1.0 - epsilon),
        ),
        axis=-1,
    )


def correlation_partials_to_matrix(
    correlation_partials: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Reconstruct a positive-definite 3x3 correlation matrix."""

    partials = np.asarray(correlation_partials, dtype=np.float64)
    rho_01, rho_02, rho_12_given_0 = (
        np.clip(partials[..., index], -1.0 + epsilon, 1.0 - epsilon) for index in range(3)
    )
    rho_12 = rho_01 * rho_02 + rho_12_given_0 * np.sqrt((1.0 - rho_01**2) * (1.0 - rho_02**2))

    matrix = np.zeros(partials.shape[:-1] + (3, 3), dtype=np.float64)
    diagonal = np.arange(3)
    matrix[..., diagonal, diagonal] = 1.0
    matrix[..., 0, 1] = matrix[..., 1, 0] = rho_01
    matrix[..., 0, 2] = matrix[..., 2, 0] = rho_02
    matrix[..., 1, 2] = matrix[..., 2, 1] = rho_12
    return matrix


def unique_correlation_components(correlation: np.ndarray) -> np.ndarray:
    """Return the unique upper-triangular off-diagonal entries of ``R``."""

    matrix = np.asarray(correlation)
    return matrix[..., (0, 0, 1), (1, 2, 2)]


def draw_multivariate_prior(
    num_draws: int,
    prior: MultivariateMMARPrior = PRIOR,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Draw all fixed parameters, including the innovation correlation matrix."""

    rng = np.random.default_rng(rng)
    mu_loc = np.asarray(prior.mu_loc)
    mu_scale = np.asarray(prior.mu_scale)
    sigma_low = np.asarray(prior.sigma_bar_low)
    sigma_high = np.asarray(prior.sigma_bar_high)
    q_low = np.asarray(prior.q_low)
    q_high = np.asarray(prior.q_high)

    mu = rng.normal(mu_loc, mu_scale, size=(num_draws, prior.dimension))
    sigma_bar = rng.uniform(sigma_low, sigma_high, size=(num_draws, prior.dimension))
    q = rng.uniform(q_low, q_high, size=(num_draws, prior.dimension))
    nu = 2 + np.exp(
        rng.uniform(np.log(prior.nu_low - 2), np.log(prior.nu_high - 2), size=(num_draws, 1))
    )
    correlation = draw_correlation_matrices(
        num_draws,
        prior.dimension,
        prior.correlation_df,
        rng,
    )
    correlation_partials = correlation_matrix_to_partials(correlation)
    correlation_unique = unique_correlation_components(correlation)
    baseline_covariance = sigma_bar[:, :, None] * correlation * sigma_bar[:, None, :]
    return {
        "mu": mu,
        "sigma_bar": sigma_bar,
        "q": q,
        "nu": nu,
        "correlation_partials": correlation_partials,
        "correlation_unique": correlation_unique,
        "correlation": correlation,
        "baseline_covariance": baseline_covariance,
    }


def shared_orientation_cascade_weights(
    q: np.ndarray,
    num_observations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shared tree and phase; asset-specific intensities on 32-day blocks."""

    q = np.asarray(q, dtype=np.float64)
    num_draws, num_assets = q.shape
    weights = np.ones((num_draws, 1, num_assets), dtype=np.float64)
    high = 2.0 * q[:, None, :]
    low = 2.0 * (1.0 - q[:, None, :])

    for _ in range(int(np.log2(num_observations // CASCADE_BLOCK))):
        # A common orientation synchronizes volatility regimes without forcing
        # each asset to have the same cascade strength q.
        high_on_left = rng.random((num_draws, weights.shape[1], 1)) < 0.5
        left = weights * np.where(high_on_left, high, low)
        right = weights * np.where(high_on_left, low, high)
        weights = np.stack((left, right), axis=2).reshape(num_draws, -1, num_assets)
    weights /= weights.mean(axis=1, keepdims=True)
    weights = np.repeat(weights, CASCADE_BLOCK, axis=1)
    offsets = rng.integers(0, num_observations, size=(num_draws, 1, 1))
    indices = (np.arange(num_observations)[None, :, None] + offsets) % num_observations
    return np.take_along_axis(weights, indices, axis=1)


def standardized_multivariate_t(
    nu: np.ndarray,
    correlation: np.ndarray,
    num_observations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw elliptical Student-t innovations standardized to covariance ``correlation``."""

    num_draws, dimension, _ = correlation.shape
    cholesky = np.linalg.cholesky(correlation)
    independent_normal = rng.normal(size=(num_draws, num_observations, dimension))
    correlated_normal = np.einsum("bti,bji->btj", independent_normal, cholesky)
    degrees = np.asarray(nu, dtype=np.float64).reshape(num_draws, 1, 1)
    radial_chi_square = rng.chisquare(
        df=degrees,
        size=(num_draws, num_observations, 1),
    )
    return correlated_normal * np.sqrt((degrees - 2.0) / radial_chi_square)


def stack_parameters(parameters):
    """Stack the 13 original-unit coordinates, never the duplicated full R."""
    if isinstance(parameters, dict):
        return np.concatenate([parameters[name] for name in INFERENCE_VARIABLES], axis=-1)
    return np.asarray(parameters, dtype="float64")


def simulate_from_parameters(parameters, rng):
    """One full joint 256-day path per vector; redraw invalid simple-return paths."""
    parameters = stack_parameters(parameters)
    mu, sigma, q = parameters[:, :3], parameters[:, 3:6], parameters[:, 6:9]
    nu = parameters[:, 9:10]
    correlation = correlation_partials_to_matrix(parameters[:, 10:13])
    returns = np.empty((len(parameters), WINDOW, 3))
    weights = np.empty_like(returns)
    pending = np.arange(len(parameters))
    for _ in range(100):
        weights[pending] = shared_orientation_cascade_weights(q[pending], WINDOW, rng)
        shocks = standardized_multivariate_t(nu[pending], correlation[pending], WINDOW, rng)
        returns[pending] = (
            mu[pending, None] + sigma[pending, None] * np.sqrt(weights[pending]) * shocks
        )
        invalid = (~np.isfinite(returns[pending]).all(axis=(1, 2))) | (returns[pending] <= -1).any(
            axis=(1, 2)
        )
        pending = pending[invalid]
        if not len(pending):
            break
    else:
        raise RuntimeError("Could not draw valid joint simple-return paths in 100 attempts.")
    return {
        "returns": returns.astype("float32"),
        "trading_time_increments": weights.astype("float32"),
    }


def simulate(n, seed=None, prior=PRIOR):
    rng = np.random.default_rng(seed)
    parameters = draw_multivariate_prior(n, prior, rng)
    return {
        **{key: value.astype("float32") for key, value in parameters.items()},
        **simulate_from_parameters(stack_parameters(parameters), rng),
    }


def training_data(n, seed=None, prior=PRIOR):
    data = simulate(n, seed, prior)
    return {name: data[name] for name in (*INFERENCE_VARIABLES, "returns")}


def forward_paths(parameters, horizon, rng):
    """Start a fresh 256-day cascade, then take the requested horizon."""
    paths = np.empty((len(parameters), horizon, 3), dtype="float32")
    for start in range(0, len(parameters), 512):
        paths[start : start + 512] = simulate_from_parameters(parameters[start : start + 512], rng)[
            "returns"
        ][:, :horizon]
    return paths


def posterior_resimulations(posterior, horizon=WINDOW, seed=None):
    rng = np.random.default_rng(seed)
    return np.stack([forward_paths(draws, horizon, rng) for draws in stack_parameters(posterior)])


def prior_table(prior=PRIOR):
    rows = []
    for i, asset in enumerate(prior.assets):
        rows.extend(
            [
                (f"mu[{asset}]", f"Normal({prior.mu_loc[i]:g}, {prior.mu_scale[i]:g}²)"),
                (
                    f"sigma_bar[{asset}]",
                    f"Uniform({prior.sigma_bar_low[i]:g}, {prior.sigma_bar_high[i]:g})",
                ),
                (f"q[{asset}]", f"Uniform({prior.q_low[i]:g}, {prior.q_high[i]:g})"),
            ]
        )
    rows.extend(
        [
            (
                "nu (shared)",
                f"2 + exp(Uniform(log({prior.nu_low - 2:.4g}), log({prior.nu_high - 2:g})))",
            ),
            (
                "correlation",
                f"Normalized Wishart(df={prior.correlation_df}, scale=I); 3 partial coordinates",
            ),
        ]
    )
    return pd.DataFrame(rows, columns=["parameter", "prior"]).set_index("parameter")


PARAMETER_LABELS = (
    *(rf"$\mu_{{\mathrm{{{a}}}}}$" for a in ASSET_NAMES),
    *(rf"$\bar\sigma_{{\mathrm{{{a}}}}}$" for a in ASSET_NAMES),
    *(rf"$q_{{\mathrm{{{a}}}}}$" for a in ASSET_NAMES),
    r"$\nu$",
    r"$\rho_{\mathrm{VOO,GLD}}$",
    r"$\rho_{\mathrm{VOO,TLT}}$",
    r"$\rho_{\mathrm{GLD,TLT}\mid\mathrm{VOO}}$",
)
