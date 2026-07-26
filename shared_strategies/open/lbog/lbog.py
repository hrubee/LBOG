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
    Generate LBOG (Line Break Original) trend-following signals with 3-Line-Break (3LB).

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

    # 2. Iterate bar-by-bar to track active 3LB bricks, ratcheting brick SL, and signals
    position = np.zeros(num_bars, dtype=int)
    sl = np.zeros(num_bars, dtype=float)
    brick_dir = np.zeros(num_bars, dtype=int)

    line_idx = 0
    active_lines = []
    pos = 0
    curr_sl = 0.0

    for i in range(1, num_bars):
        # Ingest line break bricks formed up to bar index i
        while line_idx < len(lb_lines) and lb_lines[line_idx]["idx"] <= i:
            active_lines.append(lb_lines[line_idx])
            line_idx += 1

        if not active_lines:
            continue

        last_brick = active_lines[-1]
        bd = last_brick["dir"]
        brick_dir[i] = bd
        
        # "Previous candle" relative to bar i — the stop level in force while
        # bar i trades. low[i - 1] / high[i - 1] are already closed and known
        # before bar i opens, so there is no look-ahead here.
        prev_low = float(low[i - 1])
        prev_high = float(high[i - 1])

        # A brick only counts as an entry trigger on the bar that printed it.
        # Line break bricks never repaint, so this is the "permanently painted"
        # event — a stale brick from 10 bars ago must not re-enter.
        brick_printed_now = (last_brick["idx"] == i)

        if pos == 0:
            # Fresh entry. SL starts at the candle before the signal candle.
            # No same-bar stop test: live, the order is placed at the signal
            # candle's close, so that bar's range is already history.
            if brick_printed_now and bd == 1:
                pos = 1
                curr_sl = prev_low
            elif brick_printed_now and bd == -1:
                pos = -1
                curr_sl = prev_high

        elif pos == 1:
            # Ratchet SL up to the previous candle's low (never loosens), then
            # test whether this bar's low took it out. The resting exchange stop
            # fires intrabar, so it is checked before the close-based flip.
            curr_sl = max(curr_sl, prev_low) if curr_sl > 0.0 else prev_low
            if low[i] <= curr_sl:
                pos = 0
                curr_sl = 0.0
            if brick_printed_now and bd == -1:
                # Red brick printed -> short, whether or not the stop just hit
                pos = -1
                curr_sl = prev_high

        elif pos == -1:
            # Ratchet SL down to the previous candle's high (never loosens).
            curr_sl = min(curr_sl, prev_high) if curr_sl > 0.0 else prev_high
            if high[i] >= curr_sl:
                pos = 0
                curr_sl = 0.0
            if brick_printed_now and bd == 1:
                # Green brick printed -> long, whether or not the stop just hit
                pos = 1
                curr_sl = prev_low

        position[i] = pos
        sl[i] = curr_sl

    result["position"] = position
    result["sl_level"] = sl
    result["lb_dir"] = brick_dir

    # Generate signals (-1, 0, 1) based on position changes
    pos_diff = result["position"].diff().fillna(0).astype(int)

    # Buy signal (enter long from flat or short)
    result["signal"] = np.where((result["position"] == 1) & (pos_diff > 0), 1, 0)
    # Sell signal (enter short from flat or long)
    result["signal"] = np.where((result["position"] == -1) & (pos_diff < 0), -1, result["signal"])

    # Convert signal column to integer
    result["signal"] = result["signal"].astype(int)

    return result


def lbog_strategy(df: pd.DataFrame, n: int = 3, **kwargs) -> pd.DataFrame:
    """Strategy wrapper entry-point for open strategy registry."""
    return lbog_core(df, n=n)
