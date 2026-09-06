"""Current constituents and adjusted prices; never fill missing stock returns."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..univariate import WINDOW

CACHE = Path(__file__).resolve().parents[2] / "data" / "screener"
UNIVERSE_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)


def load_universe(snapshot=None, cache_dir=CACHE):
    """Cache today's membership, NOT historical point-in-time constituents.

    A previous snapshot can be replayed only if its dated cache already exists.
    Source: datasets/s-and-p-500-companies (Wikipedia-derived membership).
    """
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    snapshot = today if snapshot is None else str(pd.Timestamp(snapshot).date())
    cache = Path(cache_dir) / f"constituents_{snapshot}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    if snapshot != today:
        raise ValueError("Historical membership requires an existing dated snapshot.")
    members = pd.read_csv(UNIVERSE_URL).rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
    )[["ticker", "name", "sector"]]
    members["ticker"] = members.ticker.str.replace(".", "-", regex=False)
    members = members.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    members["membership_snapshot"] = snapshot
    cache.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(cache, index=False)
    return members


def load_prices(members, as_of, cache_dir=CACHE):
    """Two years of adjusted closes in batches; SPY supplies the trading calendar.

    as_of must be a completed day. Missing downloads remain visible in the audit.
    Delete the dated price cache explicitly to retry a partial download.
    """
    import yfinance as yf

    as_of = pd.Timestamp(as_of).normalize()
    if as_of.date() >= pd.Timestamp.now(tz="UTC").date():
        raise ValueError("Use yesterday or earlier to exclude incomplete trading sessions.")
    snapshot = str(members.membership_snapshot.iloc[0])
    cache = Path(cache_dir) / f"prices_{snapshot}_through_{as_of.date()}.csv"
    tickers = list(dict.fromkeys([*members.ticker, "SPY"]))
    if cache.exists():
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        chunks = []
        for start in range(0, len(tickers), 50):
            batch = tickers[start : start + 50]
            downloaded = yf.download(
                batch,
                start=str((as_of - pd.Timedelta(days=730)).date()),
                end=str((as_of + pd.Timedelta(days=1)).date()),
                auto_adjust=True,
                progress=False,
                threads=4,
            )
            if not downloaded.empty:
                chunks.append(downloaded["Close"].reindex(columns=batch))
            print(f"Prices: {min(start + 50, len(tickers))}/{len(tickers)} symbols", flush=True)
        if not chunks:
            raise RuntimeError("No prices downloaded. Check network access and retry.")
        prices = pd.concat(chunks, axis=1).sort_index()
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    prices = prices.reindex(columns=tickers).loc[:as_of]
    if prices["SPY"].dropna().empty:
        raise RuntimeError("SPY calendar is missing; cannot align stock windows safely.")
    return prices


def stock_windows(prices, members, as_of, n_stocks=500):
    """Return (stock, 256, 1), aligned metadata, dates, and a full exclusion audit.

    Select up to 500 complete histories alphabetically, not by performance.
    The index may contain multiple share classes; report the actual batch size.
    """
    as_of = pd.Timestamp(as_of).normalize()
    calendar = prices.loc[:as_of, "SPY"].dropna().index
    if len(calendar) < WINDOW + 1 or (as_of - calendar[-1]).days > 4:
        raise ValueError("Benchmark calendar is too short or stale for this as-of date.")
    dates = calendar[-(WINDOW + 1) :]
    aligned = prices.reindex(index=dates, columns=members.ticker)
    returns = aligned.pct_change(fill_method=None).iloc[1:]
    complete = np.isfinite(aligned).all(axis=0) & (aligned > 0).all(axis=0)
    valid = complete & np.isfinite(returns).all(axis=0) & (returns > -1).all(axis=0)
    chosen = sorted(valid.index[valid])[:n_stocks]
    audit = members.copy()
    audit["status"] = np.where(
        audit.ticker.isin(valid.index[valid]),
        "not selected (batch cap)",
        "missing/invalid price history",
    )
    audit.loc[audit.ticker.isin(chosen), "status"] = "screened"
    if not chosen:
        raise ValueError("No stocks have a complete, valid 256-day return window.")
    metadata = members.set_index("ticker").loc[chosen].reset_index()
    metadata["as_of"] = dates[-1]
    metadata["window_start"] = dates[1]
    series = returns.loc[:, chosen].to_numpy(dtype="float32").T[..., None]
    return series, metadata, dates[1:], audit
