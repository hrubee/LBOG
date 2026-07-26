"""
Trade-list metrics, oriented around the question "is this distinguishable from
noise?" rather than "what was the return?".

The two columns that matter most:

  R        avgWin / avgLoss. A trend-follower lives or dies here.
  be_win   the win rate that R implies you need just to break even. Comparing
           the ACTUAL win rate against it gives `edge` — a signed number that
           says whether the strategy is on the right side of its own geometry.

`t` is the t-statistic on per-trade net expectancy. It is the guard against
reading a 40-trade fluke as a discovery, and it should be consulted before the
return column, not after.
"""

import numpy as np


def breakeven_win_rate(R):
    """Win rate needed for zero expectancy at payoff ratio R."""
    return 1.0 / (1.0 + R) if R != float("inf") else 0.0


def summarize(trades, label=""):
    """Reduce a trade list to a metrics dict, or None if there are no trades."""
    if not trades:
        return None
    net = np.array([t["net"] for t in trades], dtype=float)
    gross = np.array([t["gross"] for t in trades], dtype=float)
    wins = net[net > 0]
    losses = -net[net <= 0]
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    R = avg_w / avg_l if avg_l > 0 else float("inf")
    exp_ = float(net.mean())
    sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    t = exp_ / (sd / np.sqrt(len(net))) if sd > 0 else 0.0
    be = breakeven_win_rate(R)
    win = len(wins) / len(net)
    return dict(
        label=label, n=len(trades), win=win, avgW=avg_w, avgL=avg_l, R=R,
        be_win=be, edge=win - be,
        exp=exp_, gross_exp=float(gross.mean()), total=float(net.sum()),
        sd=sd, t=t,
        bars=float(np.mean([x["bars"] for x in trades])),
        maxbars=max(x["bars"] for x in trades),
        stop_frac=sum(1 for x in trades if x["reason"] == "stop") / len(trades),
    )


def trades_needed(effect_pct, sd_pct, power=0.80, alpha=0.05):
    """
    Trades required to detect ``effect_pct`` per-trade expectancy at the given
    power, one-sided.

    Use this BEFORE deciding to forward-test something. A rule that trades 17
    times a year and needs 600 trades to prove itself cannot be validated by
    running it — that is a fact about the plan, not about the market.
    """
    from math import ceil
    z_a = 1.6449 if abs(alpha - 0.05) < 1e-9 else 2.3263 if abs(alpha - 0.01) < 1e-9 else 1.6449
    z_b = 0.8416 if abs(power - 0.80) < 1e-9 else 1.2816 if abs(power - 0.90) < 1e-9 else 0.8416
    if effect_pct <= 0:
        return float("inf")
    return int(ceil(((z_a + z_b) ** 2) * (sd_pct / effect_pct) ** 2))


HEADER = (f"{'variant':<32} {'n':>5} {'win%':>5} {'BE%':>4} {'edge':>6} {'R':>5} "
          f"{'hold':>6} {'stop%':>5} {'gross%':>8} {'net/tr%':>8} {'sum%':>7} {'t':>6}")


def row(s):
    """Format one summarize() dict as a fixed-width table row."""
    if s is None:
        return "  (no trades)"
    return (f"{s['label']:<32} {s['n']:>5} {100*s['win']:>4.0f}% {100*s['be_win']:>3.0f}% "
            f"{100*s['edge']:>+5.1f}% {s['R']:>5.2f} {s['bars']:>5.1f}b "
            f"{100*s['stop_frac']:>4.0f}% {100*s['gross_exp']:>+7.3f}% "
            f"{100*s['exp']:>+7.3f}% {100*s['total']:>+6.0f}% {s['t']:>+6.2f}")
