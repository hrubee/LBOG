"""
Higher-timeframe (HTF) trend filter utility.

Determines trend direction on a higher timeframe using a 50 EMA.
Used as a pre-filter: only pass BUY when HTF is bullish, SELL when bearish.
"""

import numpy as np


# Default LTF → HTF mapping
_HTF_MAP = {
    "1m": "15m",
    "5m": "1h",
    "15m": "1h",
    "30m": "4h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1w",
    "1w": "1M",
}


def get_default_htf(timeframe: str) -> str:
    """Map a lower timeframe to its default higher timeframe."""
    return _HTF_MAP.get(timeframe, "4h")


def htf_trend_filter(symbol, timeframe, fetch_fn, htf=None, ema_period=50):
    """
    Fetch HTF data, compute EMA, return trend direction.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT" or "BTC")
        timeframe: The strategy's (lower) timeframe
        fetch_fn: Callable(symbol, timeframe, limit) → DataFrame with 'close' column
        htf: Override HTF timeframe (default: auto from LTF)
        ema_period: EMA lookback period (default 50)

    Returns:
        dict with keys:
            htf_timeframe: str — the HTF used
            htf_trend: int — 1 (bull), -1 (bear), 0 (neutral/error)
            htf_ema: float — current HTF EMA value
            htf_close: float — current HTF close price
    """
    htf = htf or get_default_htf(timeframe)
    result = {"htf_timeframe": htf, "htf_trend": 0, "htf_ema": 0.0, "htf_close": 0.0}

    try:
        df = fetch_fn(symbol, htf, ema_period + 10)
        if df is None or len(df) < ema_period:
            return result

        closes = df["close"].astype(float).values
        ema = _compute_ema(closes, ema_period)

        current_close = float(closes[-1])
        current_ema = float(ema[-1])

        result["htf_close"] = round(current_close, 6)
        result["htf_ema"] = round(current_ema, 6)

        if current_close > current_ema:
            result["htf_trend"] = 1
        elif current_close < current_ema:
            result["htf_trend"] = -1
        else:
            result["htf_trend"] = 0

    except Exception:
        # On any error, return neutral (don't block signals)
        pass

    return result


def apply_htf_filter(signal, htf_trend):
    """
    Filter a strategy signal based on HTF trend.

    Rules:
        BUY  + HTF bull  → BUY   (aligned)
        BUY  + HTF bear  → HOLD  (counter-trend, filtered)
        SELL + HTF bear  → SELL  (aligned)
        SELL + HTF bull  → HOLD  (counter-trend, filtered)
        HTF neutral      → pass signal through
        signal == 0      → HOLD  (no signal to filter)

    Returns:
        Filtered signal (1, -1, or 0)
    """
    if signal == 0 or htf_trend == 0:
        return signal

    if signal == 1 and htf_trend == 1:
        return 1
    if signal == -1 and htf_trend == -1:
        return -1

    # Counter-trend: filter to HOLD
    return 0


def _compute_ema(values, period):
    """Compute EMA over a numpy array."""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(values, dtype=float)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema
