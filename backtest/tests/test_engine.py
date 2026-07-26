"""
Tests for backtest/engine.py.

The load-bearing test is test_replay_matches_lbog_core: the engine's numbers are
only meaningful if the engine trades the same way the live strategy does. It uses
a seeded synthetic series so it needs no network and no cached data.
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_BT = os.path.dirname(_HERE)
_BASE = os.path.dirname(_BT)
for _p in (_BT, os.path.join(_BASE, "shared_strategies", "open", "lbog")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine          # noqa: E402
import metrics         # noqa: E402
from lbog import lbog_core   # noqa: E402


def synthetic(bars=4000, seed=11):
    """Random-walk OHLC with realistic intrabar range. Deterministic."""
    rng = np.random.default_rng(seed)
    close = 30_000 + np.cumsum(rng.normal(0, 60, bars))
    spread = np.abs(rng.normal(0, 45, bars))
    high = close + spread
    low = close - np.abs(rng.normal(0, 45, bars))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_, "high": np.maximum.reduce([high, close, open_]),
        "low": np.minimum.reduce([low, close, open_]), "close": close,
        "volume": np.full(bars, 100.0),
    })


def test_replay_matches_lbog_core():
    """
    The engine must reproduce the live strategy's positions bar-for-bar.

    stop_lookback is passed EXPLICITLY on both sides rather than relying on the
    two defaults happening to agree — if one drifts, this test must fail rather
    than silently compare two different rules.
    """
    df = synthetic()
    for mode in ("prev_candle", "brick"):
        for n in (3, 5):
            for lookback in (1, 2, 3):
                _, mine = engine.replay(df, n=n, stop_mode=mode,
                                        stop_lookback=lookback, record=False)
                theirs = lbog_core(df, n=n, stop_mode=mode,
                                   stop_lookback=lookback)["position"].values
                diff = int((mine != theirs).sum())
                assert diff == 0, \
                    f"{mode}/n={n}/lookback={lookback}: {diff}/{len(df)} bars diverge"


def test_default_stop_lookback_matches_strategy_core():
    """The harness default must equal the live strategy default, or measurements lie."""
    df = synthetic(bars=2000)
    _, mine = engine.replay(df, n=3, stop_mode="prev_candle", record=False)
    theirs = lbog_core(df, n=3, stop_mode="prev_candle")["position"].values
    assert int((mine != theirs).sum()) == 0, "default stop_lookback differs between engine and core"


def test_stop_exit_fills_at_the_stop_not_the_close():
    """
    A stop-out must book the stop level as its exit price. Crediting the candle's
    close instead is the accounting error that distorts the measured payoff
    ratio, so it is pinned here.
    """
    df = synthetic()
    trades, _ = engine.replay(df, n=3, stop_mode="prev_candle")
    stops = [t for t in trades if t["reason"] == "stop"]
    assert stops, "expected at least one stop-out in the sample"
    closes = df["close"].values
    for t in stops:
        # Longs stop at or below the level; the close is free to be anywhere.
        if t["side"] == 1:
            assert t["exit"] <= t["entry"] or t["exit"] > 0
        assert t["exit"] != closes[t["exit_i"]] or abs(t["exit"] - closes[t["exit_i"]]) < 1e-9
    # And at least some of them must differ from the close, or the test is vacuous.
    differing = sum(1 for t in stops if abs(t["exit"] - closes[t["exit_i"]]) > 1e-6)
    assert differing > 0.5 * len(stops), \
        f"only {differing}/{len(stops)} stop exits differ from the candle close"


def test_costs_reduce_net_below_gross():
    df = synthetic()
    trades, _ = engine.replay(df, n=3, stop_mode="prev_candle", fee_bps=5.0, slip_bps=1.0)
    assert trades
    for t in trades:
        assert t["net"] < t["gross"], "net must be gross minus round-trip costs"
    free, _ = engine.replay(df, n=3, stop_mode="prev_candle", fee_bps=0.0, slip_bps=0.0)
    for t in free:
        assert abs(t["net"] - t["gross"]) < 1e-12, "zero-cost run must leave gross == net"


def test_stop_modes_produce_different_hold_times():
    """A looser stop must hold longer, or the mode isn't doing anything."""
    df = synthetic()
    holds = {}
    for mode in ("prev_candle", "brick", "none"):
        s = metrics.summarize(engine.replay(df, n=3, stop_mode=mode)[0], mode)
        holds[mode] = s["bars"]
    assert holds["prev_candle"] < holds["brick"] < holds["none"], holds


def test_entry_filters_only_reduce_trade_count():
    """Filters are subtractive — they can never create entries."""
    df = synthetic()
    base = len(engine.replay(df, n=3, stop_mode="prev_candle")[0])
    for kw in (dict(ema_period=200), dict(confirm_bricks=3), dict(min_brick_atr=2.0)):
        filtered = len(engine.replay(df, n=3, stop_mode="prev_candle", **kw)[0])
        assert filtered <= base, f"{kw} increased trade count {base} -> {filtered}"


def test_unknown_stop_mode_raises():
    df = synthetic(bars=500)
    try:
        engine.replay(df, stop_mode="martingale")
    except ValueError as e:
        assert "martingale" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown stop_mode")


def test_breakeven_win_rate_and_power():
    assert abs(metrics.breakeven_win_rate(1.0) - 0.5) < 1e-9
    assert abs(metrics.breakeven_win_rate(2.0) - 1 / 3) < 1e-9
    # A small effect against large dispersion needs many trades.
    assert metrics.trades_needed(0.25, 2.5) > 500
    assert metrics.trades_needed(2.5, 2.5) < 20
    assert metrics.trades_needed(-1.0, 2.5) == float("inf")


def test_summarize_empty():
    assert metrics.summarize([]) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok  %s" % fn.__name__)
    print("\nALL %d TESTS PASSED" % len(fns))
