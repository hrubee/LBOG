"""
Instrumented LBOG replay — records per-trade entry/exit prices so costs can be
applied honestly.

Why this exists separately from ``lbog_core``: the strategy core returns a
position series, which is enough to trade but not enough to measure. To compute
a real return you need the price each trade actually got, and the two exits have
different fills:

  * stop-out — fills AT the stop level, intrabar
  * brick flip — fills at the signal candle's close

Crediting a stop-out at the candle's close instead of the stop level is the
classic way to get this wrong. It does not merely add noise, it changes the
measured payoff ratio: the stop CAPS the loss, whereas the candle frequently
closes further past it. On 7y of 4h BTC that single detail moved the measured
avgWin/avgLoss from 1.51 to 2.10.

This replay must stay behaviourally identical to ``lbog_core`` or its numbers
describe a strategy nobody is running. ``tests/test_engine.py`` asserts the two
produce the same position series bar-for-bar; if you change one, run the tests.
"""

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_BASE, "shared_strategies", "open", "lbog"),
           os.path.join(_BASE, "shared_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lbog import linebreak          # noqa: E402
from atr import standard_atr        # noqa: E402

# Delta taker fee is ~0.05% per side; 1bp of slippage per side is optimistic for
# a market order and deliberately so — see README on which way the biases run.
DEFAULT_FEE_BPS = 5.0
DEFAULT_SLIP_BPS = 1.0

STOP_MODES = ("prev_candle", "brick", "atr", "none")


def _ema(values, period):
    return pd.Series(values).ewm(span=period, adjust=False).mean().values


def replay(
    df,
    n=3,
    stop_mode="prev_candle",
    stop_lookback=2,
    atr_mult=3.0,
    atr_period=14,
    min_brick_atr=0.0,
    ema_period=0,
    confirm_bricks=1,
    fee_bps=DEFAULT_FEE_BPS,
    slip_bps=DEFAULT_SLIP_BPS,
    record=True,
):
    """
    Replay LBOG over ``df`` and return ``(trades, position_series)``.

    Entries are close-confirmed: a position opens only on the bar whose close
    permanently paints a new line break brick.

    Stop rules
    ----------
    prev_candle : a prior candle's low / high, ``stop_lookback`` bars back (the
                  live default is 2 — the candle before the previous one)
    brick       : lowest bottom / highest top of the last ``n`` bricks
    atr         : chandelier — extreme since entry -/+ ``atr_mult`` * ATR
    none        : no trailing stop; exit only when an opposite brick prints

    Entry filters (all off by default)
    ---------------------------------
    min_brick_atr  : entry brick height must be >= this multiple of ATR
    ema_period     : close must be on the correct side of this EMA (0 = off)
    confirm_bricks : require this many consecutive same-direction bricks

    ``record=False`` skips trade bookkeeping when only positions are needed.
    """
    if stop_mode not in STOP_MODES:
        raise ValueError(f"unknown stop_mode {stop_mode!r}; expected one of {STOP_MODES}")

    close, high, low = df["close"].values, df["high"].values, df["low"].values
    num_bars = len(df)
    need_atr = stop_mode == "atr" or min_brick_atr > 0
    atr = standard_atr(df, period=atr_period).values if need_atr else np.zeros(num_bars)
    ema_v = _ema(close, ema_period) if ema_period > 0 else None

    lb_lines = linebreak(close, n=n)
    positions = np.zeros(num_bars, dtype=int)
    trades = []

    line_idx, active = 0, []
    pos, curr_sl, entry_px, entry_i, extreme = 0, 0.0, 0.0, 0, 0.0
    # One-way cost. Applied twice per trade (entry + exit) in net.
    cost = (fee_bps + slip_bps) / 10_000.0

    def close_trade(i, exit_px, reason):
        nonlocal pos, curr_sl, entry_px, extreme
        gross = ((exit_px - entry_px) if pos == 1 else (entry_px - exit_px)) / entry_px
        if record:
            trades.append(dict(
                entry_i=entry_i, exit_i=i, bars=i - entry_i, side=pos,
                entry=entry_px, exit=exit_px, reason=reason,
                gross=gross, net=gross - 2 * cost,
            ))
        pos, curr_sl, entry_px, extreme = 0, 0.0, 0.0, 0.0

    def open_trade(i, side):
        nonlocal pos, entry_px, entry_i, extreme
        pos, entry_i, entry_px = side, i, float(close[i])
        extreme = high[i] if side == 1 else low[i]

    def entry_allowed(i, direction, brick):
        if ema_v is not None:
            if direction == 1 and not close[i] > ema_v[i]:
                return False
            if direction == -1 and not close[i] < ema_v[i]:
                return False
        if min_brick_atr > 0:
            if atr[i] <= 0 or (brick["top"] - brick["bot"]) < min_brick_atr * atr[i]:
                return False
        if confirm_bricks > 1:
            real = [b for b in active if b["dir"] != 0]
            if len(real) < confirm_bricks:
                return False
            if any(b["dir"] != direction for b in real[-confirm_bricks:]):
                return False
        return True

    for i in range(1, num_bars):
        while line_idx < len(lb_lines) and lb_lines[line_idx]["idx"] <= i:
            active.append(lb_lines[line_idx])
            line_idx += 1
        if not active:
            continue

        brick = active[-1]
        bd = brick["dir"]
        printed = brick["idx"] == i

        # Stop candidates in force while bar i trades. All read closed data only.
        if stop_mode == "prev_candle":
            j = max(i - stop_lookback, 0)
            long_stop, short_stop = float(low[j]), float(high[j])
        elif stop_mode == "brick":
            last_n = active[-n:] if len(active) >= n else active
            long_stop = float(min(x["bot"] for x in last_n))
            short_stop = float(max(x["top"] for x in last_n))
        elif stop_mode == "atr":
            a = atr[i - 1] if atr[i - 1] > 0 else 0.0
            long_stop = float(extreme - atr_mult * a) if pos == 1 and a > 0 else 0.0
            short_stop = float(extreme + atr_mult * a) if pos == -1 and a > 0 else 0.0
        else:  # none
            long_stop = short_stop = 0.0

        if pos == 0:
            if printed and bd != 0 and entry_allowed(i, bd, brick):
                open_trade(i, bd)
                curr_sl = long_stop if bd == 1 else short_stop

        elif pos == 1:
            extreme = max(extreme, high[i])
            if stop_mode == "atr":
                a = atr[i] if atr[i] > 0 else 0.0
                long_stop = float(extreme - atr_mult * a) if a > 0 else curr_sl
            if long_stop > 0:
                curr_sl = max(curr_sl, long_stop) if curr_sl > 0 else long_stop
            # The stop is a resting exchange order, so it fires intrabar —
            # before any close-based flip can be evaluated.
            if curr_sl > 0 and low[i] <= curr_sl:
                close_trade(i, curr_sl, "stop")
            if printed and bd == -1:
                if pos == 1:
                    close_trade(i, float(close[i]), "flip")
                if entry_allowed(i, -1, brick):
                    open_trade(i, -1)
                    curr_sl = short_stop

        elif pos == -1:
            extreme = min(extreme, low[i])
            if stop_mode == "atr":
                a = atr[i] if atr[i] > 0 else 0.0
                short_stop = float(extreme + atr_mult * a) if a > 0 else curr_sl
            if short_stop > 0:
                curr_sl = min(curr_sl, short_stop) if curr_sl > 0 else short_stop
            if curr_sl > 0 and high[i] >= curr_sl:
                close_trade(i, curr_sl, "stop")
            if printed and bd == 1:
                if pos == -1:
                    close_trade(i, float(close[i]), "flip")
                if entry_allowed(i, 1, brick):
                    open_trade(i, 1)
                    curr_sl = long_stop

        positions[i] = pos

    return trades, positions
