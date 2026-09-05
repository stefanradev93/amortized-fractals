"""Posterior-driven allocation: choose at t, apply only the observed return at t+1."""
from typing import Sequence
import numpy as np
import pandas as pd
from .multivariate import ASSET_NAMES, stack_parameters, forward_paths

# Three-ETF proxy, not the actual five-sleeve All Weather portfolio.
ALL_WEATHER_PROXY_WEIGHTS = np.array((0.30, 0.15, 0.55))
VOO_BUY_AND_HOLD_WEIGHTS = np.array((1.0, 0.0, 0.0))


def posterior_terminal_asset_scenarios(
    posterior, horizon=20, n_parameter_draws=64, n_replications=8, seed=None
):
    """Joint buy-and-hold horizon returns for each posterior information window."""
    posterior = stack_parameters(posterior)
    rng = np.random.default_rng(seed)
    count = min(n_parameter_draws, posterior.shape[1])
    terminal = np.empty((len(posterior), count * n_replications, 3), dtype="float32")
    for i, draws in enumerate(posterior):
        chosen = draws[rng.choice(len(draws), count, replace=False)]
        paths = forward_paths(np.repeat(chosen, n_replications, axis=0), horizon, rng)
        terminal[i] = np.prod(1.0 + paths.astype("float64"), axis=1) - 1.0
    return terminal


def simplex_weight_grid(
    step: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 0.80,
) -> np.ndarray:
    """Create a deterministic long-only weight grid whose rows sum to one."""

    units = int(round(1.0 / step))
    minimum_units = int(np.ceil(min_weight / step - 1e-12))
    maximum_units = int(np.floor(max_weight / step + 1e-12))
    rows = []
    for first in range(minimum_units, maximum_units + 1):
        for second in range(minimum_units, maximum_units + 1):
            third = units - first - second
            if minimum_units <= third <= maximum_units:
                rows.append((first, second, third))
    return np.asarray(rows, dtype=np.float64) / units


def tail_utility_scores(
    terminal_asset_scenarios: np.ndarray,
    weight_grid: np.ndarray,
    tail_alpha: float = 0.05,
    risk_aversion: float = 0.75,
) -> np.ndarray:
    """Score allocations as median 20-day return minus an ES loss penalty."""

    scenarios = np.asarray(terminal_asset_scenarios, dtype=np.float64)
    weights = np.asarray(weight_grid, dtype=np.float64)
    scores = np.full((scenarios.shape[0], weights.shape[0]), np.nan, dtype=np.float64)
    tail_count = max(1, int(np.ceil(tail_alpha * scenarios.shape[1])))
    for condition in range(len(scenarios)):
        asset_returns = scenarios[condition]
        portfolio_returns = asset_returns @ weights.T
        median_return = np.median(portfolio_returns, axis=0)
        left_tail = np.partition(portfolio_returns, tail_count - 1, axis=0)[:tail_count]
        expected_shortfall_loss = np.maximum(-left_tail.mean(axis=0), 0.0)
        scores[condition] = median_return - risk_aversion * expected_shortfall_loss
    return scores


def _tail_utility_for_one_weight(
    terminal_asset_scenarios: np.ndarray,
    weights: np.ndarray,
    tail_alpha: float,
    risk_aversion: float,
) -> float:
    portfolio_returns = terminal_asset_scenarios @ weights
    tail_count = max(1, int(np.ceil(tail_alpha * len(portfolio_returns))))
    left_tail = np.partition(portfolio_returns, tail_count - 1)[:tail_count]
    expected_shortfall_loss = max(-float(left_tail.mean()), 0.0)
    return float(np.median(portfolio_returns)) - risk_aversion * expected_shortfall_loss


def run_posterior_allocation_backtest(
    observed_returns: pd.DataFrame,
    metadata: pd.DataFrame,
    terminal_asset_scenarios: np.ndarray,
    weight_grid: np.ndarray,
    candidate_scores: np.ndarray | None = None,
    assets: Sequence[str] = ASSET_NAMES,
    benchmark_weights: Sequence[float] = ALL_WEATHER_PROXY_WEIGHTS,
    voo_benchmark_weights: Sequence[float] = VOO_BUY_AND_HOLD_WEIGHTS,
    rebalance_every: Sequence[int] = (1, 21),
    fee_bps: Sequence[float] = (0.0, 5.0, 20.0),
    initial_wealth: float = 256.0,
    tail_alpha: float = 0.05,
    risk_aversion: float = 0.75,
    turnover_penalty_bps: float = 10.0,
    no_trade_threshold_bps: float = 1.0,
) -> pd.DataFrame:
    """Run a strictly walk-forward, fee-aware allocation backtest.

    The initial allocation is established without a fee for every strategy.
    Later one-way turnover is ``0.5 * sum(abs(target - pretrade))`` and costs
    ``fee_bps / 10_000 * turnover`` of current wealth.
    """

    assets = tuple(assets)
    returns = observed_returns.loc[:, list(assets)].dropna().sort_index()
    meta = metadata.reset_index(drop=True).copy()
    scenarios = np.asarray(terminal_asset_scenarios, dtype=np.float64)
    weights = np.asarray(weight_grid, dtype=np.float64)
    signal_dates = pd.to_datetime(meta["as_of"])
    if not signal_dates.is_monotonic_increasing or signal_dates.duplicated().any():
        raise ValueError("metadata as_of dates must be unique and strictly increasing.")
    if candidate_scores is None:
        candidate_scores = tail_utility_scores(
            scenarios,
            weights,
            tail_alpha=tail_alpha,
            risk_aversion=risk_aversion,
        )
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    benchmark_specs = (
        ("All Weather proxy (buy & hold)", benchmark_weights),
        ("VOO (100%, buy & hold)", voo_benchmark_weights),
    )
    normalized_benchmarks: list[tuple[str, np.ndarray]] = []
    for label, benchmark_values in benchmark_specs:
        benchmark = np.asarray(benchmark_values, dtype=np.float64)
        normalized_benchmarks.append((label, benchmark / benchmark.sum()))

    # Locate the first strictly subsequent observed return for each signal date.
    return_index = returns.index
    decisions: list[tuple[int, pd.Timestamp, pd.Timestamp, np.ndarray]] = []
    for condition, row in meta.iterrows():
        signal_date = pd.Timestamp(row["as_of"])
        next_position = int(return_index.searchsorted(signal_date, side="right"))
        if next_position >= len(return_index):
            continue
        next_date = pd.Timestamp(return_index[next_position])
        decisions.append(
            (
                condition,
                signal_date,
                next_date,
                returns.iloc[next_position].to_numpy(dtype=np.float64),
            )
        )
    if not decisions:
        raise ValueError("No signal date has a subsequent observed return.")

    records: list[dict[str, object]] = []
    for fee in map(float, fee_bps):
        for cadence_value in rebalance_every:
            cadence = int(cadence_value)
            strategy = (
                "MMAR tail utility (daily)" if cadence == 1 else f"MMAR tail utility ({cadence}d)"
            )
            current_weights: np.ndarray | None = None
            wealth = float(initial_wealth)
            for decision_number, (condition, signal_date, next_date, asset_return) in enumerate(
                decisions
            ):
                cadence_due = current_weights is None or decision_number % cadence == 0
                scheduled = cadence_due
                turnover = 0.0
                traded = False
                if current_weights is None:
                    current_weights = weights[int(np.argmax(candidate_scores[condition]))].copy()
                    traded = True
                elif scheduled:
                    candidate_turnover = 0.5 * np.abs(weights - current_weights).sum(axis=1)
                    hurdle_rate = (turnover_penalty_bps + fee) / 10_000.0
                    net_scores = candidate_scores[condition] - hurdle_rate * candidate_turnover
                    best_id = int(np.argmax(net_scores))
                    hold_score = _tail_utility_for_one_weight(
                        scenarios[condition],
                        current_weights,
                        tail_alpha,
                        risk_aversion,
                    )
                    if net_scores[best_id] > hold_score + no_trade_threshold_bps / 10_000.0:
                        target = weights[best_id]
                        turnover = float(candidate_turnover[best_id])
                        current_weights = target.copy()
                        traded = turnover > 1e-12

                pre_return_weights = current_weights.copy()
                transaction_cost = wealth * (fee / 10_000.0) * turnover
                wealth_after_cost = wealth - transaction_cost
                gross_return = float(pre_return_weights @ asset_return)
                next_wealth = wealth_after_cost * (1.0 + gross_return)
                net_return = next_wealth / wealth - 1.0
                end_values = pre_return_weights * (1.0 + asset_return)
                current_weights = end_values / end_values.sum()
                wealth = next_wealth
                records.append(
                    _backtest_record(
                        signal_date,
                        next_date,
                        strategy,
                        fee,
                        cadence,
                        gross_return,
                        net_return,
                        wealth,
                        turnover,
                        transaction_cost,
                        scheduled,
                        traded,
                        pre_return_weights,
                        assets,
                    )
                )

        # "Simply holding" means fixed shares after the free initial purchase;
        # weights drift with market returns and no subsequent fee is charged.
        for benchmark_label, initial_weights in normalized_benchmarks:
            current_weights = initial_weights.copy()
            wealth = float(initial_wealth)
            for _, signal_date, next_date, asset_return in decisions:
                pre_return_weights = current_weights.copy()
                gross_return = float(pre_return_weights @ asset_return)
                wealth *= 1.0 + gross_return
                end_values = pre_return_weights * (1.0 + asset_return)
                current_weights = end_values / end_values.sum()
                records.append(
                    _backtest_record(
                        signal_date,
                        next_date,
                        benchmark_label,
                        fee,
                        0,
                        gross_return,
                        gross_return,
                        wealth,
                        0.0,
                        0.0,
                        False,
                        False,
                        pre_return_weights,
                        assets,
                    )
                )

    result = (
        pd.DataFrame(records).sort_values(["fee_bps", "strategy", "date"]).reset_index(drop=True)
    )
    result.attrs["initial_wealth"] = float(initial_wealth)
    return result


def _backtest_record(
    signal_date: pd.Timestamp,
    return_date: pd.Timestamp,
    strategy: str,
    fee_bps: float,
    cadence: int,
    gross_return: float,
    net_return: float,
    wealth: float,
    turnover: float,
    transaction_cost: float,
    scheduled: bool,
    traded: bool,
    weights: np.ndarray,
    assets: Sequence[str],
) -> dict[str, object]:
    record: dict[str, object] = {
        "signal_date": signal_date,
        "date": return_date,
        "strategy": strategy,
        "fee_bps": fee_bps,
        "rebalance_every": cadence,
        "gross_return": gross_return,
        "net_return": net_return,
        "wealth": wealth,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "scheduled_rebalance": scheduled,
        "traded": traded,
    }
    record.update({f"weight_{asset}": float(weight) for asset, weight in zip(assets, weights)})
    return record


def allocation_backtest_summary(
    backtest: pd.DataFrame,
    initial_wealth: float = 256.0,
    annualization: int = 252,
) -> pd.DataFrame:
    """Summarize net-of-fee wealth, risk, turnover, and trading costs."""

    rows: list[dict[str, object]] = []
    for (fee, strategy), frame in backtest.groupby(["fee_bps", "strategy"], sort=True):
        frame = frame.sort_values("date")
        daily = frame["net_return"].to_numpy(dtype=np.float64)
        wealth = np.concatenate(([initial_wealth], frame["wealth"].to_numpy(dtype=np.float64)))
        running_peak = np.maximum.accumulate(wealth)
        drawdown = wealth / running_peak - 1.0
        terminal_wealth = float(wealth[-1])
        years = len(daily) / annualization
        daily_volatility = float(daily.std(ddof=1)) if len(daily) > 1 else np.nan
        rows.append(
            {
                "fee_bps": float(fee),
                "strategy": strategy,
                "observations": len(daily),
                "final_wealth": terminal_wealth,
                "total_return_pct": 100.0 * (terminal_wealth / initial_wealth - 1.0),
                "cagr_pct": 100.0 * ((terminal_wealth / initial_wealth) ** (1.0 / years) - 1.0),
                "annual_volatility_pct": 100.0 * np.sqrt(annualization) * daily_volatility,
                "sharpe_zero_rf": (
                    np.sqrt(annualization) * float(daily.mean()) / daily_volatility
                    if daily_volatility > 0.0
                    else np.nan
                ),
                "max_drawdown_pct": 100.0 * float(drawdown.min()),
                "total_turnover": float(frame["turnover"].sum()),
                "trades": int(frame["traded"].sum()),
                "fees_paid_dollars": float(frame["transaction_cost"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["fee_bps", "strategy"]).reset_index(drop=True)
