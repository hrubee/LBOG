#!/usr/bin/env python3
"""
Backtest CLI for LBOG.

  # one config
  python3 backtest/run.py --tf 4h --stop-mode brick --n 8

  # the full variant grid, with an honest out-of-sample split
  python3 backtest/run.py --tf 4h --grid --split

  # does the signal have edge at all, before costs?
  python3 backtest/run.py --tf 4h --grid --zero-cost

  # fetch/refresh history first (needs network)
  python3 backtest/run.py --tf 4h --fetch --years 7

Read the `t` column before the return column. With ~20 variants on one asset,
several will look profitable by luck; `--split` is what tells them apart.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data
import engine
import metrics

# Variant families, matching the three hypotheses about why LBOG loses:
#   A — the exit truncates winners      (wider / looser stops)
#   B — trades too often for the fee    (fewer, bigger swings)
#   C — entries are not selective       (trend and confirmation filters)
GRID = [
    ("BASELINE prev_candle",        dict(n=3, stop_mode="prev_candle")),
    ("A brick (3LB structural)",    dict(n=3, stop_mode="brick")),
    ("A atr chandelier x2",         dict(n=3, stop_mode="atr", atr_mult=2.0)),
    ("A atr chandelier x3",         dict(n=3, stop_mode="atr", atr_mult=3.0)),
    ("A atr chandelier x4",         dict(n=3, stop_mode="atr", atr_mult=4.0)),
    ("A atr chandelier x6",         dict(n=3, stop_mode="atr", atr_mult=6.0)),
    ("A no stop (flip only)",       dict(n=3, stop_mode="none")),
    ("B 5LB depth",                 dict(n=5, stop_mode="prev_candle")),
    ("B 8LB depth",                 dict(n=8, stop_mode="prev_candle")),
    ("B 5LB + brick stop",          dict(n=5, stop_mode="brick")),
    ("B 8LB + brick stop",          dict(n=8, stop_mode="brick")),
    ("B min brick 1.0xATR",         dict(n=3, stop_mode="prev_candle", min_brick_atr=1.0)),
    ("B min brick 2.0xATR",         dict(n=3, stop_mode="prev_candle", min_brick_atr=2.0)),
    ("C EMA50 trend filter",        dict(n=3, stop_mode="prev_candle", ema_period=50)),
    ("C EMA200 trend filter",       dict(n=3, stop_mode="prev_candle", ema_period=200)),
    ("C 2-brick confirmation",      dict(n=3, stop_mode="prev_candle", confirm_bricks=2)),
    ("C 3-brick confirmation",      dict(n=3, stop_mode="prev_candle", confirm_bricks=3)),
    ("A+C brick + EMA200",          dict(n=3, stop_mode="brick", ema_period=200)),
    ("A+C atr4 + EMA200",           dict(n=3, stop_mode="atr", atr_mult=4.0, ema_period=200)),
    ("A+B+C 5LB brick EMA200",      dict(n=5, stop_mode="brick", ema_period=200)),
    ("A+B+C 8LB atr4 EMA200",       dict(n=8, stop_mode="atr", atr_mult=4.0, ema_period=200)),
]

MIN_TRADES = 30   # below this, per-trade stats are not worth printing


def run_one(df, label, kw, costs):
    trades, _ = engine.replay(df, **kw, **costs)
    return metrics.summarize(trades, label)


def main():
    p = argparse.ArgumentParser(description="LBOG backtest")
    p.add_argument("--tf", default="4h", help="timeframe (default 4h)")
    p.add_argument("--fetch", action="store_true", help="fetch/refresh history (network)")
    p.add_argument("--years", type=float, default=7.0, help="years of history to fetch")
    p.add_argument("--grid", action="store_true", help="run the full variant grid")
    p.add_argument("--split", action="store_true",
                   help="honest OOS: select on the train half, evaluate on the test half")
    p.add_argument("--zero-cost", action="store_true",
                   help="set fees and slippage to zero, to test for edge before costs")
    p.add_argument("--n", type=int, default=3, help="line break depth")
    p.add_argument("--stop-mode", default="prev_candle", choices=engine.STOP_MODES)
    p.add_argument("--atr-mult", type=float, default=3.0)
    p.add_argument("--ema-period", type=int, default=0)
    p.add_argument("--min-brick-atr", type=float, default=0.0)
    p.add_argument("--confirm-bricks", type=int, default=1)
    p.add_argument("--fee-bps", type=float, default=engine.DEFAULT_FEE_BPS)
    p.add_argument("--slip-bps", type=float, default=engine.DEFAULT_SLIP_BPS)
    args = p.parse_args()

    costs = {"fee_bps": 0.0, "slip_bps": 0.0} if args.zero_cost else \
            {"fee_bps": args.fee_bps, "slip_bps": args.slip_bps}

    if args.fetch:
        df = data.fetch(args.tf, years=args.years, refresh=True)
    else:
        df = data.load_cached(args.tf)
        if df is None:
            print(f"No cached history for {args.tf}. Run with --fetch first.", file=sys.stderr)
            return 1

    tag = " [ZERO COST]" if args.zero_cost else ""
    print(f"{args.tf}  —  {data.describe(df)}{tag}")
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"buy & hold over the same window: {bh:+.0f}%\n")

    configs = GRID if args.grid else [("custom", dict(
        n=args.n, stop_mode=args.stop_mode, atr_mult=args.atr_mult,
        ema_period=args.ema_period, min_brick_atr=args.min_brick_atr,
        confirm_bricks=args.confirm_bricks))]

    print(metrics.HEADER)
    results = {}
    for label, kw in configs:
        s = run_one(df, label, kw, costs)
        results[label] = (s, kw)
        print(metrics.row(s))

    usable = [(l, s) for l, (s, _) in results.items() if s and s["n"] >= MIN_TRADES]
    if usable:
        best_t = max(s["t"] for _, s in usable)
        n_sig = sum(1 for _, s in usable if s["t"] > 1.65)
        n_neg = sum(1 for _, s in usable if s["t"] < -1.65)
        print(f"\n  {len(usable)} configs with >={MIN_TRADES} trades | t>1.65: {n_sig} "
              f"(chance alone predicts ~{0.05*len(usable):.1f}) | t<-1.65: {n_neg} | best t: {best_t:+.2f}")

    if args.split:
        half = len(df) // 2
        train, test = df.iloc[:half], df.iloc[half:]
        # Selection must see ONLY the train half, or the "out-of-sample" result
        # is contaminated by the choice that produced it.
        best, best_s = None, None
        for label, kw in configs:
            s = run_one(train, label, kw, costs)
            if s and s["n"] >= MIN_TRADES and (best_s is None or s["exp"] > best_s["exp"]):
                best, best_s = (label, kw), s
        if best is None:
            print("\n  split: no config had enough trades in the train half")
            return 0
        st = run_one(test, best[0], best[1], costs)
        print(f"\n  honest out-of-sample")
        print(f"    train {train.index[0].date()} → {train.index[-1].date()}"
              f" | test {test.index[0].date()} → {test.index[-1].date()}")
        print(f"    chosen on train : {best[0]}  (train net/tr {100*best_s['exp']:+.3f}%, n={best_s['n']})")
        print(f"    on unseen data  : net/tr {100*st['exp']:+.3f}%, n={st['n']}, t={st['t']:+.2f}")
        need = metrics.trades_needed(abs(100 * st["exp"]) or 0.1, 100 * st["sd"])
        print(f"    trades needed to confirm an effect that size: {need:,} "
              f"(this test had {st['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
