"""Rank stocks by 20-day gain probability divided by expected shortfall."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .. import univariate as model
from ..statistics import statistics

CHECKS = ("mean_pct", "volatility_pct", "q01_pct", "q99_pct", "max_drawdown_pct")


@dataclass(frozen=True)
class Rule:
    horizon: int = 20
    round_trip_cost: float = 0.002
    tail_alpha: float = 0.05


def infer_parameters(approximator, series, num_samples=512, batch_size=32, seed=42):
    """One batched inference call; no normalization, fitting, or added context."""
    series = np.asarray(series, dtype="float32")
    if series.ndim != 3 or series.shape[1:] != (model.WINDOW, 1):
        raise ValueError("Expected raw decimal returns with shape (stocks, 256, 1).")
    start = perf_counter()
    draws = approximator.sample(
        conditions={"series": series},
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
    )
    return model.stack_parameters(draws), perf_counter() - start


def conditional_metrics(terminal_returns, rule):
    """Rows are parameter draws; columns are independent conditional replications."""
    tail_count = max(1, int(np.ceil(rule.tail_alpha * terminal_returns.shape[1])))
    tail = np.partition(terminal_returns, tail_count - 1, axis=1)[:, :tail_count]
    es = np.maximum(-tail.mean(axis=1), 0)
    gain_probability = np.mean(terminal_returns > rule.round_trip_cost, axis=1)
    return gain_probability, es


def rank_stocks(table):
    """Gain probability (%) / ES (%); zero-ES ratios are undefined and sorted last."""
    table = table.copy()
    es = table.predictive_es95_pct.where(table.predictive_es95_pct > 0)
    table["gain_to_es"] = 100 * table.predictive_gain_probability / es
    table = table.sort_values(
        ["gain_to_es", "predictive_gain_probability", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    )
    table["rank"] = np.arange(1, len(table) + 1)
    return table.reset_index(drop=True)


def _check_envelope(observed, simulated, ticker, kind):
    # Heuristic marginal 99% bands, not a calibrated joint test or coverage proof.
    low, high = simulated.loc[:, list(CHECKS)].quantile([0.005, 0.995]).to_numpy()
    values = observed.loc[list(CHECKS)].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "check": kind,
            "statistic": CHECKS,
            "observed": values,
            "lower": low,
            "upper": high,
            "inside": (values >= low) & (values <= high),
        }
    )


def screen_stocks(
    posterior,
    series,
    metadata,
    rule=Rule(),
    n_parameter_draws=64,
    n_replications=512,
    n_ppc=512,
    seed=43,
    progress=True,
):
    """Rank all stocks by gain probability / ES; attach model-fit flags.

    Gain means a positive compounded horizon return after round-trip costs.
    ES measures gross return losses. Conditional probabilities and ES across
    parameter draws supply posterior intervals with finite simulation error.
    The simulator draws fresh cascade phases; its priors must match the checkpoint.
    """
    if not 1 <= rule.horizon <= model.WINDOW:
        raise ValueError("The forecast horizon must be between 1 and 256 days.")
    if len(metadata) != len(series) or len(posterior) != len(series):
        raise ValueError("Metadata, series and posterior stock axes must agree.")
    rng = np.random.default_rng(seed)
    prior_stats = statistics(model.simulate(4096, seed=seed)["returns"])
    observed_stats = statistics(np.asarray(series)[..., 0])
    checks, rows = [], []
    start = perf_counter()
    for i, meta in metadata.reset_index(drop=True).iterrows():
        draws = posterior[i]
        selected = draws[rng.choice(len(draws), min(n_parameter_draws, len(draws)), replace=False)]
        terminal = model.conditional_horizon_returns(selected, rule.horizon, n_replications, rng)
        gain_probability, es = conditional_metrics(terminal, rule)
        ratios = gain_probability[es > 0] / es[es > 0]
        ratio_interval = np.quantile(ratios, [0.05, 0.95]) if len(ratios) else [np.nan, np.nan]
        predictive = terminal.ravel()
        pooled_tail = max(1, int(np.ceil(rule.tail_alpha * len(predictive))))
        predictive_es = max(0, -np.partition(predictive, pooled_tail - 1)[:pooled_tail].mean())

        ppc_parameters = draws[rng.choice(len(draws), n_ppc, replace=True)]
        replicated = model.simulate_from_parameters(ppc_parameters, rng)["returns"]
        stock_checks = pd.concat(
            [
                _check_envelope(observed_stats.iloc[i], prior_stats, meta.ticker, "prior"),
                _check_envelope(
                    observed_stats.iloc[i], statistics(replicated), meta.ticker, "posterior"
                ),
            ],
            ignore_index=True,
        )
        checks.append(stock_checks)
        sigma = draws[:, 1]
        margin = 0.02 * (model.PRIOR.sigma_high - model.PRIOR.sigma_low)
        edge_mass = np.mean(
            (sigma < model.PRIOR.sigma_low + margin) | (sigma > model.PRIOR.sigma_high - margin)
        )
        covered = bool(stock_checks.inside.all() and edge_mass < 0.5)
        failed = stock_checks.loc[~stock_checks.inside]
        reasons = [f"{r.check}:{r.statistic}" for r in failed.itertuples()]
        if edge_mass >= 0.5:
            reasons.append("sigma posterior at training boundary")
        row = {
            **meta.to_dict(),
            "predictive_gain_probability": np.mean(predictive > rule.round_trip_cost),
            "gain_probability_posterior_q05": np.quantile(gain_probability, 0.05),
            "gain_probability_posterior_q95": np.quantile(gain_probability, 0.95),
            "predictive_median_net_return_pct": 100
            * (np.median(predictive) - rule.round_trip_cost),
            "predictive_es95_pct": 100 * predictive_es,
            "gain_to_es_posterior_q05": ratio_interval[0],
            "gain_to_es_posterior_q95": ratio_interval[1],
            "es95_posterior_q05_pct": 100 * np.quantile(es, 0.05),
            "es95_posterior_q95_pct": 100 * np.quantile(es, 0.95),
            "coverage_ok": covered,
            "coverage_reason": "; ".join(reasons),
            "sigma_boundary_mass": edge_mass,
            "horizon_days": rule.horizon,
            "round_trip_cost_pct": 100 * rule.round_trip_cost,
            "n_parameter_draws": len(selected),
            "n_replications": n_replications,
        }
        for j, name in enumerate(model.PARAMETER_NAMES):
            for label, value in zip(
                ("q05", "median", "q95"), np.quantile(draws[:, j], [0.05, 0.5, 0.95])
            ):
                row[f"{name}_{label}"] = value
        rows.append(row)
        if progress and ((i + 1) % 25 == 0 or i + 1 == len(metadata)):
            print(
                f"Risk + PPC: {i + 1}/{len(metadata)} stocks ({perf_counter() - start:.1f}s)",
                flush=True,
            )
    return rank_stocks(pd.DataFrame(rows)), pd.concat(checks, ignore_index=True)
