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

from lbog.lbog import linebreak, lbog_core


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
    """Test that LBOG strategy enters Long on first up brick and sets initial SL to min brick bot."""
    closes = [10.0, 11.0, 12.0]
    lows = [9.0, 10.0, 11.0]
    df = make_ohlcv(closes, lows=lows)
    result = lbog_core(df, n=3)

    # Bar 0: seed -> Flat (0)
    # Bar 1: up brick [10, 11] -> enters Long (1), signal=1, SL = min brick bot = 10.0
    assert result["position"].iloc[0] == 0
    assert result["position"].iloc[1] == 1
    assert result["signal"].iloc[1] == 1
    assert result["sl_level"].iloc[1] == 10.0


def test_lbog_trailing_sl_long():
    """Test that SL ratchets up as new 3LB bricks form."""
    closes = [10.0, 11.0, 12.0, 13.0]
    # Bar 0: seed 10.0
    # Bar 1: up brick [10, 11] -> SL = 10.0
    # Bar 2: up brick [11, 12] -> SL = min(10, 11) = 10.0
    # Bar 3: up brick [12, 13] -> SL = min(10, 11, 12) = 10.0
    df = make_ohlcv(closes, lows=[9.0, 10.0, 11.5, 11.0])
    result = lbog_core(df, n=3)

    assert result["position"].iloc[1] == 1
    assert result["sl_level"].iloc[1] == 10.0

    assert result["position"].iloc[2] == 1
    assert result["sl_level"].iloc[2] == 10.0

    assert result["position"].iloc[3] == 1
    assert result["sl_level"].iloc[3] == 10.0


def test_lbog_sl_hit():
    """Test that SL hit triggers position exit (to flat/0) and outputs exit signal."""
    closes = [10.0, 11.0, 12.0, 12.0]
    # Bar 1: SL = 10.0. On Bar 3, low[3]=9.5 <= SL=10.0 -> SL hit!
    df = make_ohlcv(closes, lows=[9.0, 10.0, 11.5, 9.5])
    result = lbog_core(df, n=3)

    assert result["position"].iloc[2] == 1
    assert result["position"].iloc[3] == 0
    assert result["signal"].iloc[3] == -1
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
    # Bar 2: Short (down brick, dir=-1 since close 9.0 < last brick bot 10.0), SL = high[1] = 12.0
    assert result["position"].iloc[1] == 1
    assert result["position"].iloc[2] == -1
    assert result["signal"].iloc[2] == -1  # Flip signal (sell/short entry)
    assert result["sl_level"].iloc[2] == 11.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok  %s" % fn.__name__)
    print("\nALL %d TESTS PASSED" % len(fns))
