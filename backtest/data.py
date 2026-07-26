"""
OHLCV history for backtesting, paginated and cached to CSV.

Delta's ``fetch_ohlcv`` caps out around 2k bars per call, which is a few days on
5m — not enough to say anything about a strategy. This paginates with ``since``
and caches under ``backtest/data/`` (gitignored) so a grid run is offline after
the first fetch.

Source note: the default venue is Binance BTC/USDT perp, not Delta BTCUSD. It is
a different instrument, and the substitution is deliberate — Binance has years
of deep history where Delta India gives months. The two are highly correlated,
so it is a reasonable proxy for asking "does this rule have edge", but it is NOT
the right source for anything fee- or microstructure-specific to Delta.
"""

import os
import time

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def load_cached(timeframe, venue="binance"):
    """Return the cached frame for a timeframe, or None."""
    path = os.path.join(CACHE_DIR, f"{venue}_btc_{timeframe}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def fetch(timeframe, years=5, symbol="BTC/USDT:USDT", venue="binance", refresh=False):
    """
    Fetch (or load from cache) ``years`` of history for ``timeframe``.

    Returns a DataFrame indexed by UTC datetime with open/high/low/close/volume.
    """
    if timeframe not in TF_MS:
        raise ValueError(f"unsupported timeframe {timeframe!r}; expected one of {sorted(TF_MS)}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{venue}_btc_{timeframe}.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import ccxt
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})

    cursor = ex.milliseconds() - int(years * 365 * 86400 * 1000)
    step = TF_MS[timeframe]
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1500)
        if not batch:
            break
        rows += batch
        nxt = batch[-1][0] + step
        # Guard against a venue that returns the same page forever.
        if nxt <= cursor or len(batch) < 2:
            break
        cursor = nxt
        if cursor > ex.milliseconds():
            break
        time.sleep(0.15)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df.to_csv(path)
    return df


def describe(df):
    """One-line span summary for logging."""
    days = (df.index[-1] - df.index[0]).days
    return f"{len(df):,} bars, {days} days ({df.index[0].date()} → {df.index[-1].date()})"
