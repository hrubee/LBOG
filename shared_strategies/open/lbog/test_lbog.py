"""Tests for lbog/lbog.py — LBOG (Line Break Original) strategy."""

import os
import sys

import numpy as np
import pandas as pd

# Add parent directory so package imports resolve cleanly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from lbog.lbog import linebreak, lbog_core, stop_breached, STOP_MODES


# ─── Helpers ────────────────────────────────

def make_ohlcv(closes, highs=None, lows=None, opens=None):
    """Build an OHLCV DataFrame from price arrays."""
    closes = np.array(closes, dtype=float)
    n = len(closes)
    if highs is None:
        highs = closes + 1.0
    else:
        highs = np.array(highs, dtype=float)
    if lows is None:
        lows = closes - 1.0
    else:
        lows = np.array(lows, dtype=float)
    if opens is None:
        opens = closes

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 100.0),
        }
    )
    return df


# ─── Unit Tests ──────────────────────────────

def test_linebreak_bricks_up_trend():
    """Test standard Line Break up-brick generation."""
    closes = [100.0, 105.0, 110.0, 115.0]
    lines = linebreak(closes, n=3)
    assert len(lines) == 4
    assert lines[0]["dir"] == 0
    assert lines[1]["dir"] == 1
    assert lines[2]["dir"] == 1
    assert lines[3]["dir"] == 1
    assert lines[3]["top"] == 115.0
    assert lines[3]["bot"] == 110.0


def test_linebreak_reversal():
    """Test 3-line break reversal logic (down brick printed only after breaking lowest of last 3)."""
    closes = [10.0, 11.0, 12.0, 13.0, 10.5]  # lowest bot of last 3 is 10.0
    # closes[0]: seed
    # closes[1]: up brick [10, 11]
    # closes[2]: up brick [11, 12]
    # closes[3]: up brick [12, 13]
    # closes[4]: 10.5 is NOT < 10.0 (the min bot of last 3), so NO reversal brick printed yet!
    lines = linebreak(closes, n=3)
    assert len(lines) == 4

    # Drop to 9.5 (< 10.0) -> Reversal brick printed!
    closes2 = [10.0, 11.0, 12.0, 13.0, 9.5]
    lines2 = linebreak(closes2, n=3)
    assert len(lines2) == 5
    assert lines2[-1]["dir"] == -1
    assert lines2[-1]["top"] == 13.0
    assert lines2[-1]["bot"] == 9.5


def test_lbog_long_entry():
    """Test that LBOG enters Long on the first up brick with SL at the previous candle's low."""
    closes = [10.0, 11.0, 12.0]
    lows = [9.0, 10.0, 11.0]
    df = make_ohlcv(closes, lows=lows)
    result = lbog_core(df, n=3)

    # Bar 0: seed -> Flat (0)
    # Bar 1: up brick [10, 11] prints -> enters Long, SL = previous candle low = low[0] = 9.0
    assert result["position"].iloc[0] == 0
    assert result["position"].iloc[1] == 1
    assert result["signal"].iloc[1] == 1
    assert result["sl_level"].iloc[1] == 9.0


def test_lbog_trailing_sl_long():
    """Default stop_lookback=2: a Long SL trails the candle BEFORE the previous one."""
    closes = [10.0, 11.0, 12.0, 13.0]
    df = make_ohlcv(closes, lows=[9.0, 10.0, 11.5, 12.2])
    result = lbog_core(df, n=3)

    # Bar 1: up brick prints -> Long. lookback=2 clamps to low[0] = 9.0
    assert result["position"].iloc[1] == 1
    assert result["sl_level"].iloc[1] == 9.0

    # Bar 2: still clamped to low[0] = 9.0
    assert result["position"].iloc[2] == 1
    assert result["sl_level"].iloc[2] == 9.0

    # Bar 3: SL trails to low[1] = 10.0 — one bar further back than lookback=1
    assert result["position"].iloc[3] == 1
    assert result["sl_level"].iloc[3] == 10.0


def test_stop_lookback_1_is_tighter_than_2():
    """lookback=1 must ratchet strictly faster than lookback=2 on a rising series."""
    closes = [10.0, 11.0, 12.0, 13.0]
    df = make_ohlcv(closes, lows=[9.0, 10.0, 11.5, 12.2])
    tight = lbog_core(df, n=3, stop_lookback=1)
    slow = lbog_core(df, n=3, stop_lookback=2)

    assert tight["sl_level"].iloc[3] == 11.5   # low[2]
    assert slow["sl_level"].iloc[3] == 10.0    # low[1]
    assert tight["sl_level"].iloc[3] > slow["sl_level"].iloc[3]


def test_stop_lookback_rejects_zero():
    """lookback must be >= 1; 0 would read the bar currently trading (look-ahead)."""
    df = make_ohlcv([10.0, 11.0, 12.0])
    try:
        lbog_core(df, n=3, stop_lookback=0)
    except ValueError as e:
        assert "stop_lookback" in str(e)
    else:
        raise AssertionError("expected ValueError for stop_lookback=0")


def test_lbog_trailing_sl_short():
    """Test that a Short SL ratchets DOWN to the previous candle's high on every new candle."""
    closes = [10.0, 9.0, 8.0, 7.0]
    highs = [12.0, 11.0, 10.0, 9.0]
    lows = [9.0, 8.0, 7.0, 6.0]
    df = make_ohlcv(closes, highs=highs, lows=lows)
    result = lbog_core(df, n=1)

    # Bar 1: down brick prints -> Short. lookback=2 clamps to high[0] = 12.0
    assert result["position"].iloc[1] == -1
    assert result["signal"].iloc[1] == -1
    assert result["sl_level"].iloc[1] == 12.0

    # Bar 2: still clamped to high[0] = 12.0
    assert result["position"].iloc[2] == -1
    assert result["sl_level"].iloc[2] == 12.0

    # Bar 3: SL trails DOWN to high[1] = 11.0 (high[3]=9.0 does not breach it)
    assert result["position"].iloc[3] == -1
    assert result["sl_level"].iloc[3] == 11.0


def test_lbog_no_reentry_without_fresh_brick():
    """A stale brick must not re-enter: after a stop-out, entry waits for a NEW brick."""
    # Bar 1 prints the only up brick and goes Long with SL = low[0] = 9.0.
    # Bar 2 gaps down through the stop -> flat. Bars 3-4 print no new brick,
    # so the position must stay flat even though the last brick is still green.
    closes = [10.0, 11.0, 10.5, 10.6, 10.7]
    lows = [9.0, 10.0, 8.0, 10.2, 10.3]
    highs = [11.0, 12.0, 11.0, 11.0, 11.0]
    df = make_ohlcv(closes, highs=highs, lows=lows)
    result = lbog_core(df, n=3)

    assert result["position"].iloc[1] == 1
    assert result["position"].iloc[2] == 0   # low[2]=8.0 breached SL=low[1]=10.0
    assert result["position"].iloc[3] == 0   # no fresh brick -> no re-entry
    assert result["position"].iloc[4] == 0


def test_lbog_sl_hit():
    """Test that SL hit triggers position exit (to flat/0) and outputs exit signal."""
    closes = [10.0, 11.0, 12.0, 12.0]
    # Bar 1: SL = 10.0. On Bar 3, low[3]=9.5 <= SL=10.0 -> SL hit!
    df = make_ohlcv(closes, lows=[9.0, 10.0, 11.5, 9.5])
    result = lbog_core(df, n=3)

    assert result["position"].iloc[2] == 1
    assert result["position"].iloc[3] == 0
    assert result["sl_level"].iloc[3] == 0.0


def test_lbog_opposite_brick_flip():
    """Test that opposite line break brick prints flips position immediately without hitting SL."""
    # N=1 to make reversals fast
    closes = [10.0, 11.0, 9.0]
    # highs/lows:
    # bar 0: low=5.0, high=11.0 -> SL = 5.0
    # bar 1: low=5.0, high=12.0
    # bar 2: low=6.0, high=10.0 -> does not hit SL (6.0 > 5.0)
    df = make_ohlcv(closes, highs=[11.0, 12.0, 10.0], lows=[5.0, 5.0, 6.0])
    result = lbog_core(df, n=1)

    # Bar 0: seed
    # Bar 1: Long (up brick, dir=1), SL = low[0] = 5.0
    # Bar 2: Short (down brick, dir=-1 since close 9.0 < last brick bot 10.0).
    #        lookback=2 clamps to high[0] = 11.0, not high[1] = 12.0.
    assert result["position"].iloc[1] == 1
    assert result["position"].iloc[2] == -1
    assert result["signal"].iloc[2] == -1  # Flip signal (sell/short entry)
    assert result["sl_level"].iloc[2] == 11.0


def test_stop_mode_brick_uses_structural_level():
    """stop_mode='brick' must trail the 3LB reversal level, not the candle low."""
    closes = [10.0, 11.0, 12.0, 13.0]
    lows = [9.0, 10.8, 11.5, 12.2]
    df = make_ohlcv(closes, lows=lows)

    brick = lbog_core(df, n=3, stop_mode="brick")
    prev = lbog_core(df, n=3, stop_mode="prev_candle")

    # Bricks are [10,11], [11,12], [12,13]; the n=3 structural floor stays at
    # the lowest brick bottom, so the brick stop is looser than the candle stop.
    assert brick["position"].iloc[3] == 1
    assert brick["sl_level"].iloc[3] == 10.0
    assert prev["sl_level"].iloc[3] == 10.8          # low[1], per lookback=2
    assert brick["sl_level"].iloc[3] < prev["sl_level"].iloc[3]


def test_stop_mode_brick_holds_trade_that_prev_candle_stops_out():
    """The two modes must actually diverge in outcome, not just in level."""
    # A pullback that dips under the candle stop but stays above the 3LB
    # structural floor: prev_candle exits, brick stays in.
    closes = [10.0, 11.0, 12.0, 12.5]
    lows = [9.0, 10.9, 11.5, 10.5]   # low[3]=10.5 < prev stop 10.9 (=low[1]), > brick stop 10.0
    df = make_ohlcv(closes, lows=lows)

    assert lbog_core(df, n=3, stop_mode="prev_candle")["position"].iloc[3] == 0
    assert lbog_core(df, n=3, stop_mode="brick")["position"].iloc[3] == 1


def test_stop_mode_short_ratchets_in_both_modes():
    """
    Neither mode may leave a short stop pinned at its widest level — this is the
    regression that 03806db introduced by using max() on falling brick tops.
    Highs stay strictly under both stop levels so the short survives to bar 3.
    """
    closes = [10.0, 9.0, 8.0, 7.0]
    highs = [12.0, 9.5, 8.8, 7.5]
    lows = [9.0, 8.0, 7.0, 6.0]
    df = make_ohlcv(closes, highs=highs, lows=lows)

    # 'none' is excluded by construction: it publishes no stop, so there is
    # nothing to ratchet. Covered separately by the stop_mode_none tests.
    for mode in [m for m in STOP_MODES if m != "none"]:
        r = lbog_core(df, n=1, stop_mode=mode)
        seg = [r["sl_level"].iloc[i] for i in (1, 2, 3)]
        assert all(r["position"].iloc[i] == -1 for i in (1, 2, 3)), \
            f"{mode}: short did not survive to bar 3, positions={list(r['position'])}"
        # Monotone non-increasing, and strictly tighter by the end. Not strictly
        # decreasing every bar: with lookback=2 the stop holds for a bar before
        # stepping down, which is the whole point of the slower ratchet.
        assert seg[0] >= seg[1] >= seg[2], f"{mode}: short stop loosened {seg}"
        assert seg[2] < seg[0], f"{mode}: short stop never ratcheted down {seg}"


def test_stop_mode_rejects_unknown_value():
    """An unrecognized stop_mode must fail loudly, not silently pick a default."""
    df = make_ohlcv([10.0, 11.0, 12.0])
    try:
        lbog_core(df, n=3, stop_mode="trailing_atr")
    except ValueError as e:
        assert "trailing_atr" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown stop_mode")


def test_stop_breached_detects_unplaceable_stops():
    """
    A stop price has already traded through is unplaceable as a resting stop —
    the exchange would classify it as a take-profit and leave the position naked.
    """
    # Long: price at or below the stop means it is already hit.
    assert stop_breached("long", 100.0, 99.0) is True
    assert stop_breached("long", 100.0, 100.0) is True
    assert stop_breached("long", 100.0, 101.0) is False
    # Short: price at or above the stop means it is already hit.
    assert stop_breached("short", 100.0, 101.0) is True
    assert stop_breached("short", 100.0, 100.0) is True
    assert stop_breached("short", 100.0, 99.0) is False
    # No stop set, or no price: nothing to breach.
    assert stop_breached("long", 0.0, 99.0) is False
    assert stop_breached("short", 100.0, 0.0) is False


def test_stop_breached_matches_core_exit_condition():
    """The live predicate must agree with the intrabar test lbog_core uses."""
    closes = [10.0, 11.0, 12.0, 12.0]
    lows = [9.0, 10.0, 11.5, 9.5]
    df = make_ohlcv(closes, lows=lows)
    r = lbog_core(df, n=3, stop_lookback=1)

    # Bar 3 stops out in the core: SL trailed to low[2]=11.5 and low[3]=9.5 broke it.
    assert r["position"].iloc[2] == 1
    assert r["position"].iloc[3] == 0
    # The live predicate reaches the same verdict from price vs level.
    assert stop_breached("long", 11.5, 9.5) is True


def test_stop_mode_none_holds_until_opposite_brick():
    """stop_mode='none' must never stop out — only an opposite brick exits."""
    # A deep dip that would breach any candle-based stop, but no opposite brick.
    closes = [10.0, 11.0, 12.0, 12.5]
    lows = [9.0, 10.0, 5.0, 5.0]      # low[2]=5.0 breaks every prev-candle stop
    df = make_ohlcv(closes, lows=lows)

    # Bar 2: low 5.0 breaks the prev-candle stop (low[0]=9.0) -> flat.
    # In none mode there is no stop, so the position rides through.
    assert lbog_core(df, n=3, stop_mode="prev_candle")["position"].iloc[2] == 0
    assert lbog_core(df, n=3, stop_mode="none")["position"].iloc[2] == 1
    # and no stop level is ever published
    assert (lbog_core(df, n=3, stop_mode="none")["sl_level"] == 0.0).all()


def test_stop_mode_none_shorts_do_not_instantly_exit():
    """
    Regression: a zero stop compared as `high >= curr_sl` is always true, which
    exited every short on the bar after entry. Shorts must survive in none mode.
    """
    closes = [10.0, 9.0, 8.0, 7.0]
    highs = [12.0, 11.0, 10.0, 9.0]
    lows = [9.0, 8.0, 7.0, 6.0]
    df = make_ohlcv(closes, highs=highs, lows=lows)
    r = lbog_core(df, n=1, stop_mode="none")
    assert r["position"].iloc[1] == -1
    assert r["position"].iloc[2] == -1, "short exited despite no stop"
    assert r["position"].iloc[3] == -1, "short exited despite no stop"


def test_stop_mode_none_still_flips_on_opposite_brick():
    """The only exit in none mode is the colour flip — verify it still fires."""
    closes = [10.0, 11.0, 9.0]
    df = make_ohlcv(closes, highs=[11.0, 12.0, 10.0], lows=[5.0, 5.0, 6.0])
    r = lbog_core(df, n=1, stop_mode="none")
    assert r["position"].iloc[1] == 1
    assert r["position"].iloc[2] == -1, "did not flip on the opposite brick"


def test_static_sl_gives_none_mode_a_hard_floor():
    """stop_mode='none' has no stop; static_sl_pct must supply one."""
    closes = [100.0, 101.0, 102.0, 90.0]
    df = make_ohlcv(closes, highs=[101.0, 102.0, 103.0, 103.0],
                    lows=[99.0, 100.0, 101.0, 90.0])
    bare = lbog_core(df, n=3, stop_mode="none")
    with_sl = lbog_core(df, n=3, stop_mode="none", static_sl_pct=0.02)

    assert (bare["sl_level"] == 0.0).all(), "none mode should publish no stop"
    # Long entered at close 101 -> floor at 101 * 0.98
    assert abs(with_sl["sl_level"].iloc[1] - 98.98) < 1e-9
    # Short entered at close 90 -> floor at 90 * 1.02
    assert abs(with_sl["sl_level"].iloc[3] - 91.80) < 1e-9


def test_static_sl_only_ever_tightens_a_trailing_stop():
    """It is a floor, never a loosening: the tighter of the two always wins."""
    closes = [10.0, 11.0, 12.0, 13.0]
    df = make_ohlcv(closes, lows=[9.0, 10.0, 11.5, 12.2])
    trail = lbog_core(df, n=3, stop_mode="prev_candle")
    both = lbog_core(df, n=3, stop_mode="prev_candle", static_sl_pct=0.50)

    for i in (1, 2, 3):
        t, b = trail["sl_level"].iloc[i], both["sl_level"].iloc[i]
        if t > 0 and b > 0:
            assert b >= t, f"bar {i}: static loosened the long stop {t} -> {b}"


def test_static_sl_rejects_negative():
    df = make_ohlcv([10.0, 11.0, 12.0])
    try:
        lbog_core(df, n=3, static_sl_pct=-0.01)
    except ValueError as e:
        assert "static_sl_pct" in str(e)
    else:
        raise AssertionError("expected ValueError for negative static_sl_pct")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok  %s" % fn.__name__)
    print("\nALL %d TESTS PASSED" % len(fns))
