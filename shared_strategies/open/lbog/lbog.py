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


STOP_MODES = ("prev_candle", "brick", "none")


def stop_breached(side: str, stop: float, price: float) -> bool:
    """
    True when `price` has already traded through `stop` for a position on `side`.

    The live mirror of the intrabar test in lbog_core (`low[i] <= curr_sl` for a
    long). It exists because a breached stop CANNOT be placed as a resting stop
    order: an exchange reads a sell order below market as a stop but a sell above
    market as a take-profit, so submitting one silently converts the protective
    order into a profit target and leaves the position naked. When this returns
    True the only correct action is to close the position — the exit condition is
    already satisfied.
    """
    if stop <= 0 or price <= 0:
        return False
    return price <= stop if side == "long" else price >= stop


def stop_levels(active_lines: list[dict], low, high, i: int, n: int, stop_mode: str,
                stop_lookback: int = 2) -> tuple[float, float]:
    """
    Return (long_stop, short_stop) — the stop levels in force while bar `i` trades.

    prev_candle : a prior candle's low / high, `stop_lookback` bars back from the
                  bar currently trading. lookback=1 is the immediately preceding
                  candle (tightest); lookback=2 is the candle before that, which
                  lags price by an extra bar and so ratchets more slowly. Only
                  this mode uses stop_lookback.
    brick       : the 3LB structural reversal level (lowest bottom / highest top
                  of the last `n` bricks) — i.e. the price at which the line
                  break chart would flip. Wider, holds trends far longer.

    Both read only closed data (bar i-1 and earlier), so neither look-aheads.
    """
    if stop_mode == "prev_candle":
        j = max(i - stop_lookback, 0)   # clamp at the series start
        return float(low[j]), float(high[j])
    if stop_mode == "brick":
        last_n = active_lines[-n:] if len(active_lines) >= n else active_lines
        return (
            float(min(x["bot"] for x in last_n)),
            float(max(x["top"] for x in last_n)),
        )
    if stop_mode == "none":
        # No protective stop at all. A position is held until an opposite brick
        # prints — the "enter on colour change, hold to the opposite colour"
        # rule. 0.0 signals "no stop", and callers must treat it as such rather
        # than as a price (see the curr_sl > 0 guards in lbog_core).
        return 0.0, 0.0
    raise ValueError(f"unknown stop_mode {stop_mode!r}; expected one of {STOP_MODES}")


def lbog_core(
    df: pd.DataFrame,
    n: int = 3,
    stop_mode: str = "prev_candle",
    stop_lookback: int = 2,
) -> pd.DataFrame:
    """
    Generate LBOG (Line Break Original) trend-following signals with 3-Line-Break (3LB).

    Entry is close-confirmed: a position opens only on the bar whose close
    permanently paints a new line break brick.

    Parameters
    ----------
    df : DataFrame with open, high, low, close columns
    n : lookback depth for the line break chart reversal (default 3)
    stop_mode : "prev_candle" (default) trails a prior candle's low/high;
                "brick" trails the 3LB structural reversal level. See stop_levels.
    stop_lookback : how many bars back the prev_candle stop reads (default 2 —
                the candle before the previous one). Ignored by "brick".

    Returns
    -------
    DataFrame with columns: signal, position, sl_level, lb_dir
    """
    if stop_mode not in STOP_MODES:
        raise ValueError(f"unknown stop_mode {stop_mode!r}; expected one of {STOP_MODES}")
    if stop_lookback < 1:
        raise ValueError(f"stop_lookback must be >= 1, got {stop_lookback}")
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
        
        # Stop levels in force while bar i trades, per the selected stop_mode.
        long_stop, short_stop = stop_levels(active_lines, low, high, i, n, stop_mode, stop_lookback)

        # A brick only counts as an entry trigger on the bar that printed it.
        # Line break bricks never repaint, so this is the "permanently painted"
        # event — a stale brick from 10 bars ago must not re-enter.
        brick_printed_now = (last_brick["idx"] == i)

        if pos == 0:
            # Fresh entry. No same-bar stop test: live, the order is placed at
            # the signal candle's close, so that bar's range is already history.
            if brick_printed_now and bd == 1:
                pos = 1
                curr_sl = long_stop
            elif brick_printed_now and bd == -1:
                pos = -1
                curr_sl = short_stop

        elif pos == 1:
            # Ratchet SL up (never loosens), then test whether this bar's low
            # took it out. The resting exchange stop fires intrabar, so it is
            # checked before the close-based brick flip.
            curr_sl = max(curr_sl, long_stop) if curr_sl > 0.0 else long_stop
            # curr_sl == 0 means "no stop" (stop_mode='none'), not "stop at zero".
            if curr_sl > 0.0 and low[i] <= curr_sl:
                pos = 0
                curr_sl = 0.0
            if brick_printed_now and bd == -1:
                # Red brick printed -> short, whether or not the stop just hit
                pos = -1
                curr_sl = short_stop

        elif pos == -1:
            # Ratchet SL down (never loosens).
            curr_sl = min(curr_sl, short_stop) if curr_sl > 0.0 else short_stop
            # Without this guard a zero stop reads as "high >= 0", which is always
            # true — every short would exit on the bar after entry.
            if curr_sl > 0.0 and high[i] >= curr_sl:
                pos = 0
                curr_sl = 0.0
            if brick_printed_now and bd == 1:
                # Green brick printed -> long, whether or not the stop just hit
                pos = 1
                curr_sl = long_stop

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


def lbog_strategy(df: pd.DataFrame, n: int = 3, stop_mode: str = "prev_candle",
                  stop_lookback: int = 2, **kwargs) -> pd.DataFrame:
    """Strategy wrapper entry-point for open strategy registry."""
    return lbog_core(df, n=n, stop_mode=stop_mode, stop_lookback=stop_lookback)
