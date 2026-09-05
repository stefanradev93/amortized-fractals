"""Adjusted-close simple returns, cached locally for repeatable demo runs."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_market_returns(tickers, start="2020-01-01", end="2026-09-01"):
    cache = DATA_DIR / f"{'_'.join(tickers)}_{start}_{end}.csv"
    if cache.exists():
        returns = pd.read_csv(cache, index_col=0, parse_dates=True)
        returns.index.name = "date"
        return returns

    import yfinance as yf

    prices = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)[
        "Close"
    ]
    returns = prices.loc[:, list(tickers)].ffill().pct_change(fill_method=None).dropna()
    if returns.empty:
        raise RuntimeError(
            "No market returns downloaded; check the connection or restore the data cache."
        )
    returns.index.name = "date"
    DATA_DIR.mkdir(exist_ok=True)
    returns.to_csv(cache)
    return returns
