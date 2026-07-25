"""
LBOG (Line Break Original) Strategy — trend-following strategy with ratcheting previous-candle stop loss.

Generates Long/Short signals using a canonical N-Line-Break (N-LB) charting system
to identify trend flips, combined with an active trailing stop loss locked at the
previous candle's Low (for Longs) or High (for Shorts).
"""

import numpy as np
import pandas as pd


def linebreak(close: np.ndarray, n: int = 3) -> list[dict]:
    """
    Canonical N-line break calculation on the close series.
    Returns a list of dict(dir, top, bot, idx).
    """
    lines = []
    for i in range(len(close)):
        p = float(close[i])
        if not lines:
            lines.append(dict(dir=0, top=p, bot=p, idx=i))
            continue
        last = lines[-1]
        pc = last["top"] if last["dir"] >= 0 else last["bot"]  # prior line's close level
        if last["dir"] >= 0:                                   # currently up (or flat)
            if p > pc:                                         # up continuation: block [pc, p]
                lines.append(dict(dir=1, top=p, bot=pc, idx=i))
            elif p < min(x["bot"] for x in lines[-n:]):        # reversal down: block [p, pc]
                lines.append(dict(dir=-1, top=pc, bot=p, idx=i))
        else:                                                  # currently down
            if p < pc:                                         # down continuation: block [p, pc]
                lines.append(dict(dir=-1, top=pc, bot=p, idx=i))
            elif p > max(x["top"] for x in lines[-n:]):        # reversal up: block [pc, p]
                lines.append(dict(dir=1, top=p, bot=pc, idx=i))
    return lines


def lbog_core(
    df: pd.DataFrame,
    n: int = 3,
) -> pd.DataFrame:
    """
    Generate LBOG (Line Break Original) trend-following signals with ratcheting stop loss.

    Parameters
    ----------
    df : DataFrame with open, high, low, close columns
    n : lookback depth for the line break chart reversal (default 3)

    Returns
    -------
    DataFrame with columns: signal, position, sl_level, lb_dir
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    num_bars = len(df)

    if num_bars == 0:
        result["signal"] = 0
        result["position"] = 0
        result["sl_level"] = 0.0
        result["lb_dir"] = 0
        return result

    # 1. Compute Line Break bricks
    lb_lines = linebreak(close, n=n)

    # Map line break direction to each bar index
    brick_dir = np.zeros(num_bars, dtype=int)
    for line in lb_lines:
        idx = line["idx"]
        if idx < num_bars:
            brick_dir[idx] = line["dir"]

    # Forward fill brick direction so every bar knows the active line break trend
    for i in range(1, num_bars):
        if brick_dir[i] == 0:
            brick_dir[i] = brick_dir[i - 1]

    # 2. Iterate bar-by-bar to simulate state, ratcheting SL, and signals
    position = np.zeros(num_bars, dtype=int)
    sl = np.zeros(num_bars, dtype=float)

    for i in range(1, num_bars):
        prev_pos = position[i - 1]
        prev_sl = sl[i - 1]
        bd = brick_dir[i]

        if prev_pos == 0:
            # Flat: enter Long if brick_dir == 1, Short if brick_dir == -1
            if bd == 1:
                position[i] = 1
                sl[i] = low[i - 1]
            elif bd == -1:
                position[i] = -1
                sl[i] = high[i - 1]
            else:
                position[i] = 0
                sl[i] = 0.0

        elif prev_pos == 1:
            # Long: evaluate SL hit or reversal signal
            if low[i] <= prev_sl:
                # Stop loss hit -> flatten position
                position[i] = 0
                sl[i] = 0.0
            elif bd == -1:
                # Opposite line break printed -> flip to Short
                position[i] = -1
                sl[i] = high[i - 1]
            else:
                # Continue Long -> ratchet SL up to max(prev_sl, low[i-1])
                position[i] = 1
                sl[i] = max(prev_sl, low[i - 1])

        elif prev_pos == -1:
            # Short: evaluate SL hit or reversal signal
            if high[i] >= prev_sl:
                # Stop loss hit -> flatten position
                position[i] = 0
                sl[i] = 0.0
            elif bd == 1:
                # Opposite line break printed -> flip to Long
                position[i] = 1
                sl[i] = low[i - 1]
            else:
                # Continue Short -> ratchet SL down to min(prev_sl, high[i-1])
                position[i] = -1
                sl[i] = min(prev_sl, high[i - 1])

    result["position"] = position
    result["sl_level"] = sl
    result["lb_dir"] = brick_dir

    # Generate signals (-1, 0, 1) based on position changes
    pos_diff = result["position"].diff().fillna(0).astype(int)

    # Buy signals (enter long or exit short)
    result["signal"] = np.where((result["position"] == 1) & (pos_diff > 0), 1, 0)
    # Sell signals (enter short or exit long)
    result["signal"] = np.where((result["position"] == -1) & (pos_diff < 0), -1, result["signal"])
    # Exits to flat (position = 0)
    exit_mask = (result["position"] == 0) & (pos_diff != 0)
    prev_pos = result["position"].shift(1).fillna(0).astype(int)
    result["signal"] = np.where(exit_mask, -np.sign(prev_pos), result["signal"])

    # Convert signal column to integer
    result["signal"] = result["signal"].astype(int)

    return result


def lbog_strategy(df: pd.DataFrame, n: int = 3, **kwargs) -> pd.DataFrame:
    """Strategy wrapper entry-point for open strategy registry."""
    return lbog_core(df, n=n)
