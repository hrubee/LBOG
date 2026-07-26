# LBOG backtest harness

Measures whether an LBOG configuration has edge, and — more importantly — whether
an apparent edge is distinguishable from luck.

```bash
# one config
python3 backtest/run.py --tf 4h --stop-mode brick --n 8

# the variant grid + honest out-of-sample split
python3 backtest/run.py --tf 4h --grid --split

# is there edge at all, before costs?
python3 backtest/run.py --tf 4h --grid --zero-cost

# fetch history (first run only; cached under backtest/data/, gitignored)
python3 backtest/run.py --tf 4h --fetch --years 7
```

## Layout

| file | role |
|---|---|
| `engine.py` | the replay — records per-trade entry/exit prices and exit reasons |
| `metrics.py` | payoff ratio, breakeven win rate, expectancy, t-stat, power |
| `data.py` | paginated OHLCV fetch + CSV cache |
| `run.py` | CLI: single config, grid, train/test split, zero-cost mode |
| `tests/test_engine.py` | fidelity to `lbog_core` + accounting invariants |

## The one invariant that matters

`engine.replay()` must trade exactly like `lbog_core()`, or its numbers describe
a strategy nobody is running. `tests/test_engine.py::test_replay_matches_lbog_core`
asserts the two produce identical position series bar-for-bar on a seeded
synthetic series (no network needed). **If you change either side, run the tests.**

## Reading the output

Read `t` before you read the return columns.

- `R` — avgWin / avgLoss.
- `BE%` — the win rate that `R` implies you need just to break even.
- `edge` — actual win rate minus `BE%`. Signed. Negative means the strategy is on
  the wrong side of its own geometry, regardless of what the return column says.
- `gross%` vs `net/tr%` — run `--zero-cost` to separate "no edge" from "edge that
  fees eat". These are very different diagnoses with very different fixes.
- `t` — t-stat on per-trade net expectancy. With ~20 variants on one asset,
  roughly one will show `t > 1.65` by chance alone; the footer prints that
  expected count next to the observed one so the comparison is unavoidable.

`--split` selects the best config on the train half **only**, then reports it on
data never used for selection. Picking the winner on the full sample and then
quoting its second-half number is contaminated, and was a mistake made during
the original study.

## Known biases, and which way they run

Every unmodelled cost here makes real results **worse** than the harness reports,
so treat its output as an optimistic bound:

- no perpetual funding (a 15-bar 4h hold spans several funding payments)
- stops assumed to fill exactly at the stop level; in fast moves they slip
- entry at the signal candle's close, whereas `run_live.py` enters seconds later
- when a stop and a brick flip land on the same bar, the stop is assumed to fire
  first — defensible (it is a resting order) but not verifiable from OHLC alone
- default venue is Binance BTC/USDT perp, not Delta BTCUSD: correlated proxy
  chosen for history depth, wrong source for Delta-specific microstructure

## What this harness has already established

Run on 2–7 years of BTC across 15m/1h/4h (63 configurations):

- **Baseline gross expectancy with fees and slippage set to zero is −0.009% (4h),
  −0.007% (1h), −0.001% (15m).** Zero to three decimals. The 3LB brick print
  carries no directional information — there is no edge for costs to eat.
- 0 of 63 configurations reached `t > 1.65`; chance alone predicts ~3.2. 45 were
  significantly *negative*.
- Wider stops (`brick`, ATR chandelier ×2–6, no stop) were **worse** on every
  timeframe, and *reduced* `R` from 2.10 to 1.2–1.9.
- Entry filters (EMA50/200, 2- and 3-brick confirmation) were neutral to harmful.
- Fewer/bigger trades (higher line-break depth, min brick size) helped
  directionally but never significantly, and flipped sign across timeframes.

Before proposing a fix, check it is not on that list.
