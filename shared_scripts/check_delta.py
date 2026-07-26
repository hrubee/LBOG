#!/usr/bin/env python3
"""
Delta Exchange spot/futures strategy check script.
Fetches OHLCV from Delta via CCXT, runs strategy, outputs JSON to stdout, exits.

Signal check mode (paper or live):
    check_delta.py <strategy> <symbol> <timeframe> [--mode=paper|live] [--inst-type=spot|futures]

Execution mode (live only, called by Go as phase 2):
    check_delta.py --execute --symbol=BTC --side=buy|sell --size=0.01 [--mode=live] [--inst-type=spot|futures]
"""

import sys
import os
import json
import math
import traceback
from datetime import datetime, timezone

# Add paths: platforms/delta/ for adapter, shared_tools/ for utilities.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'platforms', 'delta')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared_tools')))

from atr import ensure_atr_indicator, latest_atr
from regime import latest_regime

# Use futures registry for futures, spot registry for spot.
_inst_type = "spot"
for _arg in sys.argv:
    if _arg.startswith("--inst-type="):
        _inst_type = _arg.split("=", 1)[1]
        break
if _inst_type == "spot":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_strategies', 'open', 'spot'))
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_strategies', 'open', 'futures'))


def _make_dataframe(candles):
    import pandas as pd
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df.sort_index(inplace=True)
    return df


def _extract_fee(response):
    if not isinstance(response, dict):
        return None
    fee_info = response.get("fee")
    if isinstance(fee_info, dict):
        cost = fee_info.get("cost")
        return float(cost) if cost is not None else None
    return float(fee_info) if fee_info is not None else None


def run_signal_check(strategy_name, symbol, timeframe, mode, inst_type="spot",
                     strategy_params_override=None, open_strategy=None,
                     close_strategies=None, position_side="", position_ctx=None,
                     regime_enabled=False, regime_period=14, regime_adx_threshold=20.0,
                     close_params_by_name=None):
    try:
        from adapter import DeltaExchangeAdapter
        from strategies import apply_strategy, get_strategy, list_strategies
        from close_registry_loader import evaluate as close_evaluate
        from strategy_composition import (
            evaluate_open_close,
            finalize_decision,
            normalize_signal,
        )

        adapter = DeltaExchangeAdapter()

        if not symbol or not timeframe:
            print(json.dumps({"error": f"Missing symbol ({symbol}) or timeframe ({timeframe})", "platform": "delta"}))
            sys.exit(1)

        print(f"Fetching {symbol} {timeframe} from Delta ({mode}, {inst_type})...", file=sys.stderr)
        candles = adapter.get_ohlcv(symbol, interval=timeframe, limit=200, inst_type=inst_type)

        if not candles or len(candles) < 30:
            print(json.dumps({"error": f"Insufficient data: {len(candles)} candles", "platform": "delta"}))
            sys.exit(1)

        df = _make_dataframe(candles)
        
        # Determine regime if enabled
        regime = "ranging"
        if regime_enabled:
            from regime import compute_regime_label
            regime = compute_regime_label(df, period=regime_period, adx_threshold=regime_adx_threshold)

        # Full composition evaluation
        open_name = open_strategy or strategy_name
        evaluation = evaluate_open_close(
            apply_strategy,
            get_strategy,
            df,
            strategy_name,
            open_strategy,
            close_strategies,
            position_side,
            params=strategy_params_override,
            position_ctx=position_ctx,
            close_evaluate=close_evaluate,
            close_params_by_name=close_params_by_name,
        )
        
        decision = finalize_decision(evaluation, position_side)
        signal = decision["signal"]
        indicators = {}

        signal = normalize_signal(signal)
        price = float(df.iloc[-1]["close"])

        # Update price with live last
        try:
            if inst_type == "futures":
                live_price = adapter.get_perp_price(symbol)
            else:
                live_price = adapter.get_spot_price(symbol)
            if live_price > 0:
                price = live_price
        except:
            pass

        print(json.dumps({
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "price": round(price, 4),
            "indicators": indicators,
            "regime": regime,
            "mode": mode,
            "platform": "delta",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e), "platform": "delta"}))
        sys.exit(1)


def run_execute(symbol, side, size, mode, inst_type="spot", reduce_only=False):
    if mode != "live":
        print(json.dumps({"error": "--execute requires --mode=live"}))
        sys.exit(1)

    try:
        from adapter import DeltaExchangeAdapter
        adapter = DeltaExchangeAdapter()
        
        # AI LAYER: Check sentiment before entry
        try:
            from ai_market_gate import ai_analyze_trade
            allowed, reason = ai_analyze_trade(symbol, side, "New Strategy Signal")
            if not allowed:
                print(f"AI VETO: {reason}", file=sys.stderr)
                sys.exit(0)
            else:
                print(f"AI APPROVED: {reason}", file=sys.stderr)
        except Exception as e:
            print(f"AI Layer Error (Skipping filter): {e}", file=sys.stderr)

        side_norm = side.lower()
        if side_norm == "long":
            side_norm = "buy"
        elif side_norm == "short":
            side_norm = "sell"
        
        is_buy = side_norm == "buy"
        result = adapter.market_open(symbol, is_buy, size, inst_type=inst_type, reduce_only=reduce_only)

        filled_sz = float(result.get("filled", 0) or 0)
        fill = {
            "avg_px": float(result.get("average", 0) or 0),
            "total_sz": filled_sz,
            "fee": _extract_fee(result),
            "oid": str(result.get("id", ""))
        }

        # Post-close protective cleanup (mirrors Binance): a fully-filled
        # reduce-only close means the position is flat — cancel orphaned SL/TP
        # conditional orders. Gated on full fill so partial closes keep protection.
        if reduce_only and inst_type == "futures" and filled_sz >= size * 0.99:
            try:
                stops = adapter.list_open_stop_orders(symbol, inst_type)
                oids = [o["id"] for o in stops if o.get("id")]
                if oids:
                    print(f"[INFO] post-close cleanup: cancelling {len(oids)} orphaned protective order(s) for {symbol}", file=sys.stderr)
                    adapter.cancel_orders(symbol, inst_type, oids)
            except Exception as e:
                print(f"[WARN] post-close protective cleanup failed for {symbol}: {e}", file=sys.stderr)

        print(json.dumps({
            "execution": {
                "action": "buy" if is_buy else "sell",
                "symbol": symbol,
                "size": size,
                "fill": fill,
            },
            "platform": "delta",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e), "platform": "delta"}))
        sys.exit(1)


def run_update_stop_loss(symbol, side, size, stop_loss_price, cancel_oid=None, inst_type="spot"):
    try:
        from adapter import DeltaExchangeAdapter
        adapter = DeltaExchangeAdapter()

        side_norm = side.lower()
        is_buy = side_norm in ("short", "sell")

        result = adapter.market_stop_loss(symbol, is_buy, size, stop_loss_price, inst_type=inst_type)

        print(json.dumps({
            "stop_loss_oid": str(result.get("id", "")),
            "stop_loss_trigger_px": stop_loss_price,
            "platform": "delta",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e), "platform": "delta"}))
        sys.exit(1)


def _normalize_tp_tiers(tp_tiers=None):
    if tp_tiers is None:
        return []

    print(f"DEBUG: tp_tiers={tp_tiers}", file=sys.stderr)
    tiers = []
    for tier in tp_tiers:
        if isinstance(tier, dict):
            multiple = tier.get("atr_multiple", tier.get("multiple", tier.get("Multiple")))
            fraction = tier.get("close_fraction", tier.get("fraction", tier.get("Fraction")))
        else:
            try:
                multiple, fraction = tier
            except (TypeError, ValueError):
                continue
        try:
            multiple = float(multiple)
            fraction = min(max(float(fraction), 0.0), 1.0)
        except (TypeError, ValueError):
            continue
        if multiple > 0 and fraction > 0:
            tiers.append((multiple, fraction))
    tiers.sort(key=lambda item: item[0])

    prev_fraction = 0.0
    for _multiple, fraction in tiers:
        if fraction <= prev_fraction:
            return []
        prev_fraction = fraction
    if not tiers:
        return []

    tiers[-1] = (tiers[-1][0], 1.0)
    return tiers


def compute_tp_tier_sizes(size, tiers, floor_size_fn):
    if not tiers or size <= 0:
        return [0.0] * len(tiers)
    floored_total = floor_size_fn(size)
    sizes = []
    placed = 0.0
    prev_fraction = 0.0
    for idx, (_atr_mult, cumulative_fraction) in enumerate(tiers):
        is_final = idx == len(tiers) - 1
        if is_final:
            tier_size = max(floored_total - placed, 0.0)
        else:
            raw = size * max(cumulative_fraction - prev_fraction, 0.0)
            tier_size = floor_size_fn(raw)
            placed += tier_size
        prev_fraction = cumulative_fraction
        sizes.append(tier_size)
    return sizes


def run_sync_protection(symbol, side, size, avg_cost, entry_atr, stop_loss_atr_mult, tiers_json, stop_loss_oid, tp_oids_json, tp_armed_tiers_json, inst_type="spot", stop_loss_price=None):
    try:
        from adapter import DeltaExchangeAdapter
        adapter = DeltaExchangeAdapter()
        
        side = side.lower()
        tp_tiers_raw = json.loads(tiers_json) if tiers_json else None
        tp_oids = json.loads(tp_oids_json) if tp_oids_json else []
        tp_armed_tiers = json.loads(tp_armed_tiers_json) if tp_armed_tiers_json else []
        
        close_is_buy = side in ("short", "sell")
        close_side = "buy" if close_is_buy else "sell"
        
        out = {
            "platform": "delta",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stop_loss_oid": stop_loss_oid,
            "tp_oids": tp_oids,
        }
        
        # 1. Update/Place Stop Loss
        if (stop_loss_price and stop_loss_price > 0) or stop_loss_atr_mult > 0:
            if stop_loss_price and stop_loss_price > 0:
                sl_px = stop_loss_price
            else:
                sl_distance = stop_loss_atr_mult * entry_atr
                min_sl_distance = 0.015 * avg_cost
                if sl_distance < min_sl_distance:
                    sl_distance = min_sl_distance
                if side in ("long", "buy"):
                    sl_px = avg_cost - sl_distance
                else:
                    sl_px = avg_cost + sl_distance
            
            out["stop_loss_trigger_px"] = sl_px

            # Idempotent SL reconcile — mirrors the Binance fix. The old
            # fetch_order(oid) verify could not see trigger/stop orders and
            # re-placed a new stop every cycle, stacking duplicates. We now
            # enumerate the real open stop orders, keep exactly ONE on the
            # SL side at the target price, cancel duplicates/stale, and only
            # place a fresh one when none match.
            tol = max(1e-4, abs(sl_px) * 1e-4)
            existing = adapter.list_open_stop_orders(symbol, inst_type)

            # Cancel any leftover stop orders from previous opposite-side positions
            opposite_side_orders = [o for o in existing if o.get("side") != close_side and o.get("id")]
            if opposite_side_orders:
                opp_ids = [o["id"] for o in opposite_side_orders]
                print(f"[INFO] SL reconcile {symbol}: cancelling {len(opp_ids)} leftover opposite-side stop(s).", file=sys.stderr)
                adapter.cancel_orders(symbol, inst_type, opp_ids)
                existing = [o for o in existing if o["id"] not in opp_ids]

            sl_side_orders = [o for o in existing if o.get("side") == close_side and o.get("trigger_price") is not None]

            at_target = [o for o in sl_side_orders if o.get("trigger_price") is not None and abs(o["trigger_price"] - sl_px) <= tol]
            stale = [o for o in sl_side_orders if o not in at_target]

            if at_target:
                keep = at_target[0]
                dupes = [o["id"] for o in at_target[1:] if o.get("id")] + [o["id"] for o in stale if o.get("id")]
                if dupes:
                    print(f"[INFO] SL reconcile {symbol}: keeping {keep['id']} @ {keep['trigger_price']}, cancelling {len(dupes)} duplicate/stale stop(s).", file=sys.stderr)
                    adapter.cancel_orders(symbol, inst_type, dupes)
                out["stop_loss_oid"] = str(keep["id"])
                out["stop_loss_trigger_px"] = keep.get("trigger_price") or sl_px
            else:
                stale_ids = [o["id"] for o in sl_side_orders if o.get("id")]
                if stale_ids:
                    print(f"[INFO] SL reconcile {symbol}: cancelling {len(stale_ids)} stale stop(s) before placing fresh SL @ {sl_px}.", file=sys.stderr)
                    adapter.cancel_orders(symbol, inst_type, stale_ids)
                try:
                    result = adapter.market_stop_loss(symbol, close_is_buy, size, sl_px, inst_type=inst_type)
                    out["stop_loss_oid"] = str(result.get("id", ""))
                    out["stop_loss_trigger_px"] = sl_px
                except Exception as e:
                    out["stop_loss_error"] = str(e)
        
        # 2. Update/Place TPs — idempotent reconcile (mirrors the Binance TP fix).
        tiers_tuples = _normalize_tp_tiers(tp_tiers_raw)
        if tiers_tuples:
            if len(tp_oids) < len(tiers_tuples):
                tp_oids.extend(["0"] * (len(tiers_tuples) - len(tp_oids)))
            if len(tp_armed_tiers) < len(tiers_tuples):
                tp_armed_tiers.extend([False] * (len(tiers_tuples) - len(tp_armed_tiers)))

            tier_sizes = compute_tp_tier_sizes(size, tiers_tuples, lambda sz: adapter.floor_size(symbol, sz))
            tp_pxs = []
            tp_errors = [""] * len(tiers_tuples)

            existing = adapter.list_open_stop_orders(symbol, inst_type)
            tp_side = []
            for o in existing:
                if o.get("side") != close_side:
                    continue
                tp = o.get("trigger_price")
                if tp is None:
                    continue
                is_tp_side = (tp >= avg_cost) if side in ("long", "buy") else (tp <= avg_cost)
                if is_tp_side:
                    tp_side.append(o)

            matched_ids = set()
            for idx, ((atr_mult, frac), tier_size) in enumerate(zip(tiers_tuples, tier_sizes)):
                if entry_atr > 0:
                    raw_px = avg_cost + (atr_mult * entry_atr if side in ("long", "buy") else -atr_mult * entry_atr)
                else:
                    raw_px = atr_mult  # Explicit tp_price passed directly
                tp_pxs.append(raw_px)
                tol = max(1e-4, abs(raw_px) * 5e-4)

                matches = [o for o in tp_side
                           if o.get("id") and o["id"] not in matched_ids
                           and o.get("trigger_price") is not None
                           and abs(o["trigger_price"] - raw_px) <= tol]
                if matches:
                    keep = matches[0]
                    matched_ids.add(keep["id"])
                    tp_oids[idx] = str(keep["id"])
                    tp_armed_tiers[idx] = True
                    dupes = [d["id"] for d in matches[1:]]
                    for d in matches[1:]:
                        matched_ids.add(d["id"])
                    if dupes:
                        print(f"[INFO] TP reconcile {symbol} tier{idx}: keeping {keep['id']} @ {keep['trigger_price']}, cancelling {len(dupes)} duplicate(s).", file=sys.stderr)
                        adapter.cancel_orders(symbol, inst_type, dupes)
                    continue

                if tp_armed_tiers[idx]:
                    tp_oids[idx] = "0"  # armed but no longer resting → FILLED, do not re-place
                    continue
                if tier_size <= 0:
                    continue
                try:
                    result = adapter.market_stop_loss(symbol, close_is_buy, tier_size, raw_px, inst_type=inst_type)
                    new_oid = str(result.get("id", ""))
                    tp_oids[idx] = new_oid
                    if new_oid and new_oid != "0":
                        tp_armed_tiers[idx] = True
                        print(f"[INFO] TP reconcile {symbol} tier{idx}: placed TP @ {raw_px} (oid={new_oid})", file=sys.stderr)
                except Exception as e:
                    tp_errors[idx] = str(e)

            orphans = [o["id"] for o in tp_side if o.get("id") and o["id"] not in matched_ids]
            if orphans:
                print(f"[INFO] TP reconcile {symbol}: cancelling {len(orphans)} orphan TP-side order(s).", file=sys.stderr)
                adapter.cancel_orders(symbol, inst_type, orphans)

            out["tp_oids"] = tp_oids
            out["tp_pxs"] = tp_pxs
            out["tp_armed_tiers"] = tp_armed_tiers
            if any(tp_errors):
                out["tp_errors"] = tp_errors

        print(json.dumps(out))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e), "platform": "delta"}))
        sys.exit(1)


def run_fetch_atr(symbol, timeframe, period=14):
    try:
        from adapter import DeltaExchangeAdapter
        adapter = DeltaExchangeAdapter()
        candles = adapter.get_ohlcv(symbol, interval=timeframe, limit=max(100, period * 3), inst_type="futures")
        if not candles:
            print(json.dumps({"error": "No candles for ATR"}))
            sys.exit(1)
        
        df = _make_dataframe(candles)
        ensure_atr_indicator(df, period=period)
        atr = latest_atr(df)
        
        print(json.dumps({
            "atr": round(float(atr), 6),
            "candles": len(candles),
            "platform": "delta"
        }))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e), "platform": "delta"}))
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reduce-only", action="store_true", default=False)
    parser.add_argument("--update-stop-loss", action="store_true")
    parser.add_argument("--sync-protection", action="store_true")
    parser.add_argument("--fetch-atr", action="store_true")
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("strategy", nargs="?")
    parser.add_argument("symbol", nargs="?")
    parser.add_argument("timeframe", nargs="?")
    parser.add_argument("--symbol", dest="exec_symbol")
    parser.add_argument("--timeframe", dest="exec_timeframe")
    parser.add_argument("--side", choices=["buy", "sell", "long", "short"])
    parser.add_argument("--size", type=float)
    parser.add_argument("--avg-cost", type=float)
    parser.add_argument("--entry-atr", type=float)
    parser.add_argument("--stop-loss-atr-mult", type=float)
    parser.add_argument("--tiers-json")
    parser.add_argument("--stop-loss-oid")
    parser.add_argument("--tp-oids-json")
    parser.add_argument("--tp-armed-tiers-json")
    parser.add_argument("--stop-loss-price", type=float)
    parser.add_argument("--cancel-stop-loss-oid")
    parser.add_argument("--mode", default="paper")
    parser.add_argument("--inst-type", default="spot", choices=["spot", "futures"])
    parser.add_argument("--params", default=None)
    parser.add_argument("--open-strategy", default=None)
    parser.add_argument("--close-strategies", default=None)
    parser.add_argument("--strategy-refs", default=None)
    parser.add_argument("--position-side", default="")
    parser.add_argument("--position-avg-cost", type=float, default=None)
    parser.add_argument("--position-qty", type=float, default=None)
    parser.add_argument("--position-initial-qty", type=float, default=None)
    parser.add_argument("--position-entry-atr", type=float, default=None)
    parser.add_argument("--position-regime", default="")
    parser.add_argument("--regime-enabled", action="store_true", default=False)
    parser.add_argument("--regime-period", type=int, default=14)
    parser.add_argument("--regime-adx-threshold", type=float, default=20.0)
    parser.add_argument("--probe-only", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    if args.probe_only:
        sys.exit(0)
        
    if args.execute:
        run_execute(args.exec_symbol, args.side, args.size, args.mode, args.inst_type, args.reduce_only)
    elif args.update_stop_loss:
        run_update_stop_loss(args.exec_symbol, args.side, args.size, args.stop_loss_price, args.cancel_stop_loss_oid, args.inst_type)
    elif args.sync_protection:
        run_sync_protection(
            args.exec_symbol, args.side, args.size, args.avg_cost, args.entry_atr,
            args.stop_loss_atr_mult, args.tiers_json, args.stop_loss_oid,
            args.tp_oids_json, args.tp_armed_tiers_json, args.inst_type,
            stop_loss_price=args.stop_loss_price
        )
    elif args.fetch_atr:
        run_fetch_atr(args.exec_symbol or args.symbol, args.exec_timeframe or args.timeframe, args.period)
    else:
        from strategy_composition import parse_strategy_refs_arg
        refs = parse_strategy_refs_arg(args.strategy_refs)
        open_strategy_name = refs["open_name"] if refs else args.open_strategy
        close_strategies_arg = refs["close_csv"] if refs else args.close_strategies
        params_override = refs["open_params"] if refs else (json.loads(args.params) if args.params else None)
        close_params_by_name = refs["close_params_by_name"] if refs else None
        
        # position ctx extraction
        position_ctx = {}
        for attr, key in (
            ("position_avg_cost", "avg_cost"),
            ("position_qty", "qty"),
            ("position_initial_qty", "initial_qty"),
            ("position_entry_atr", "entry_atr"),
        ):
            val = getattr(args, attr, None)
            if val is not None:
                position_ctx[key] = val
        regime = (getattr(args, "position_regime", "") or "").strip()
        if regime:
            position_ctx["regime"] = regime
        
        run_signal_check(
            args.strategy, args.symbol, args.timeframe, args.mode, args.inst_type,
            params_override, open_strategy_name, close_strategies_arg,
            args.position_side, position_ctx,
            regime_enabled=args.regime_enabled,
            regime_period=args.regime_period,
            regime_adx_threshold=args.regime_adx_threshold,
            close_params_by_name=close_params_by_name
        )


if __name__ == "__main__":
    main()
