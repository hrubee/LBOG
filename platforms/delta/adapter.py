"""
Delta Exchange Adapter — supports perpetual swap (futures) and spot trading.
Uses CCXT for all API interactions.

Supports paper (public API only, no credentials) and
live (real orders on Delta, API credentials required) modes.

Demo/Testnet mode can be toggled via DELTA_SANDBOX=1.
"""

import os
import sys
import math
import time
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared_tools'))

import ccxt
import pandas as pd


class DeltaExchangeAdapter:
    """
    Exchange adapter for Delta Exchange — spot and perpetual futures.

    Paper mode:  no credentials needed; uses live Delta prices for simulation.
    Live mode:   requires DELTA_LIVE_API_KEY, DELTA_LIVE_API_SECRET (or DELTA_DEMO_*).
    """

    def __init__(self):
        sandbox = os.environ.get("DELTA_SANDBOX", "") == "1"
        
        if sandbox:
            api_key = os.environ.get("DELTA_DEMO_API_KEY") or os.environ.get("DELTA_API_KEY", "")
            api_secret = os.environ.get("DELTA_DEMO_API_SECRET") or os.environ.get("DELTA_API_SECRET", "")
        else:
            api_key = os.environ.get("DELTA_LIVE_API_KEY") or os.environ.get("DELTA_API_KEY", "")
            api_secret = os.environ.get("DELTA_LIVE_API_SECRET") or os.environ.get("DELTA_API_SECRET", "")

        config = {
            "enableRateLimit": True,
            "options": {
                "fetchCurrencies": False,
                "builderFee": False,  # Disable CCXT builder fee
            }
        }
        
        # Determine if we are in live mode based on presence of credentials
        self._is_live = bool(api_key and api_secret)
        
        if self._is_live:
            config["apiKey"] = api_key
            config["secret"] = api_secret

        self._exchange = ccxt.delta(config)
        
        # Public instance for market data from Binance to receive highly liquid, accurate price data
        self._public_exchange = ccxt.binance({"enableRateLimit": True})
        
        india_mainnet_url = "https://api.india.delta.exchange"
        india_testnet_url = "https://cdn-ind.testnet.deltaex.org"
        
        if sandbox:
            self._exchange.set_sandbox_mode(True)
            self._exchange.urls["api"] = {"public": india_testnet_url, "private": india_testnet_url}
        else:
            self._exchange.urls["api"] = {"public": india_mainnet_url, "private": india_mainnet_url}


        self._markets_loaded = False

    @property
    def is_live(self) -> bool:
        """True if API credentials are provided (live mode)."""
        return self._is_live

    @property
    def mode(self) -> str:
        """'live' or 'paper'."""
        return "live" if self.is_live else "paper"

    @property
    def name(self) -> str:
        return "delta"

    def get_wallet_balance(self, quote: str = "USDT") -> float:
        """Fetch total wallet collateral balance in USDT or USD."""
        if not self._is_live:
            return 1000.0
        try:
            bal = self._exchange.fetch_balance()
            total_dict = bal.get("total", {})
            if quote in total_dict and total_dict[quote]:
                return float(total_dict[quote])
            if "USD" in total_dict and total_dict["USD"]:
                return float(total_dict["USD"])
            if "BTC" in total_dict and total_dict["BTC"]:
                # Convert BTC to USDT
                btc_price = self.get_spot_price("BTC")
                return float(total_dict["BTC"]) * btc_price
            free_dict = bal.get("free", {})
            for k in ("USDT", "USD", "DETO"):
                if k in free_dict and free_dict[k]:
                    return float(free_dict[k])
        except Exception as e:
            print(f"Error fetching wallet balance: {e}", file=sys.stderr)
        return 1000.0

    # ─────────────────────────────────────────────
    # Market data
    # ─────────────────────────────────────────────

    def _load_markets(self):
        """Load and cache markets from exchanges."""
        if not self._markets_loaded:
            try:
                self._exchange.load_markets()
            except Exception:
                pass
            try:
                self._public_exchange.load_markets()
            except Exception:
                pass
            self._markets_loaded = True

    def _format_symbol(self, symbol: str, inst_type: str = "spot") -> str:
        """Format symbol for Delta CCXT dynamically by matching loaded markets (e.g. BTC/USDT -> BTC/USDT:USDT or BTC/USD:USD)."""
        if not symbol:
            return ""
        
        # Extract base currency cleanly (e.g., BTC from BTC/USDT, BTC/USD, or raw BTC)
        base = symbol.split("/")[0] if "/" in symbol else symbol
        
        # Ensure markets are loaded to perform dynamic scanning
        try:
            self._load_markets()
        except Exception:
            pass
            
        if inst_type == "futures":
            # Search active swap/perpetual markets on Delta for this base currency
            if self._exchange.markets:
                for sym, market in self._exchange.markets.items():
                    if market.get("swap") and market.get("base") == base:
                        return sym
            # Fallback standard convention if not found or markets failed
            return f"{base}/USD:USD"
        else:
            # Search spot market
            if self._exchange.markets:
                for sym, market in self._exchange.markets.items():
                    if market.get("spot") and market.get("base") == base:
                        return sym
            # Fallback standard spot convention
            return f"{base}/USDT"


    def _format_binance_symbol(self, symbol: str, inst_type: str = "spot") -> str:
        """Format symbol for Binance CCXT (e.g. BTC -> BTC/USDT or BTC/USDT:USDT)."""
        if not symbol:
            return ""
        if "/" in symbol:
            return symbol
        if inst_type == "futures":
            return f"{symbol}/USDT:USDT"
        else:
            return f"{symbol}/USDT"

    def get_spot_price(self, symbol: str) -> float:
        """Get current spot price for a coin (e.g. 'BTC') from Binance."""
        if not symbol:
            return 0.0
        pair = self._format_binance_symbol(symbol, "spot")
        try:
            ticker = self._public_exchange.fetch_ticker(pair)
            price = ticker.get("last") or 0
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        return 0.0

    def get_perp_price(self, symbol: str) -> float:
        """Get current last price for a perpetual swap (e.g. 'BTC') directly from Delta Exchange."""
        if not symbol:
            return 0.0
        pair = self._format_symbol(symbol, "futures")
        try:
            self._load_markets()
            ticker = self._exchange.fetch_ticker(pair)
            price = ticker.get("last") or 0
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        return 0.0

    def get_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 200, inst_type: str = "spot") -> list:
        """
        Fetch OHLCV candles directly from Delta Exchange live market feed.
        Uses MARK price candles for futures to match TradingView MARK:BTCUSD 100%.
        """
        if not symbol:
            return []
        
        # Primary: Fetch directly from Delta Exchange (MARK price for futures)
        pair = self._format_symbol(symbol, inst_type)
        try:
            self._load_markets()
            params = {}
            if inst_type == "futures":
                base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                params = {"symbol": f"MARK:{base_coin}USD"}

            candles = self._exchange.fetch_ohlcv(pair, timeframe=interval, limit=limit, params=params)
            if candles and len(candles) >= 10:
                return candles
        except Exception:
            pass

        # Fallback to Binance
        binance_pair = self._format_binance_symbol(symbol, inst_type)
        try:
            candles = self._public_exchange.fetch_ohlcv(binance_pair, interval, limit=limit)
            return candles
        except Exception:
            return []

    def floor_size(self, symbol: str, sz: float) -> float:
        """Floor order size to Delta precision."""
        if sz <= 0:
            return 0.0
        try:
            self._load_markets()
            pair = self._format_symbol(symbol, "spot")
            if pair not in self._exchange.markets:
                pair = self._format_symbol(symbol, "futures")
            
            market = self._exchange.markets[pair]
            if market.get("type") == "swap" or market.get("type") == "future":
                contract_size = market.get("contractSize", 1.0)
                contracts = sz / contract_size
                floored_contracts = float(self._exchange.amount_to_precision(pair, contracts))
                return floored_contracts * contract_size
            else:
                return float(self._exchange.amount_to_precision(pair, sz))
        except Exception:
            return sz

    # ─────────────────────────────────────────────
    # Order execution (live mode only)
    # ─────────────────────────────────────────────

    def fetch_open_positions(self) -> list:
        """Return every open perpetual swap position on the account."""
        if not self._is_live:
            raise RuntimeError("fetch_open_positions requires live mode")
        
        return self._exchange.fetch_positions(params={"type": "future"}) or []

    def fetch_my_trades(self, symbol: str, inst_type: str = "futures", limit: int = 50) -> list:
        """Fetch actual executed wallet trades/fills directly from Delta Exchange, normalized to coin units."""
        if not self._is_live:
            return []
        try:
            self._load_markets()
            pair = self._format_symbol(symbol, inst_type)
            params = {}
            contract_size = 1.0
            if inst_type == "futures":
                params["type"] = "future"
                market = self._exchange.market(pair)
                contract_size = market.get("contractSize", 1.0) or 1.0

            trades = self._exchange.fetch_my_trades(pair, limit=limit, params=params)
            normalized = []
            for t in (trades or []):
                t_norm = dict(t)
                if contract_size and contract_size != 1.0:
                    t_norm["amount"] = float(t.get("amount", 0) or 0) * contract_size
                normalized.append(t_norm)
            return normalized
        except Exception as e:
            print(f"Error fetching wallet trades/fills: {e}", file=sys.stderr)
            return []

    def market_open(self, symbol: str, is_buy: bool, size: float, inst_type: str = "spot", reduce_only: bool = False, stop_loss_price: float = None) -> dict:
        """Place a market order. Returns CCXT order dict with `filled`/`amount`
        rewritten to coin units (not contracts), so callers always see size in
        the same units they passed in. Without this conversion, Go records
        contract counts as coin quantities and inflates positions by 1/contract_size."""
        if not self._is_live:
            raise RuntimeError("market_open requires live mode")

        self._load_markets()
        side = "buy" if is_buy else "sell"
        pair = self._format_symbol(symbol, inst_type)

        amount = size
        contract_size = 1.0
        if inst_type == "futures":
            market = self._exchange.market(pair)
            contract_size = market.get("contractSize", 1.0) or 1.0
            amount = size / contract_size

        amount = float(self._exchange.amount_to_precision(pair, amount))

        params = {}
        if inst_type == "futures":
            params["type"] = "future"
            if reduce_only:
                params["reduceOnly"] = True
            if stop_loss_price and stop_loss_price > 0:
                sl_prec = self._exchange.price_to_precision(pair, stop_loss_price)
                params["stop_loss_price"] = str(sl_prec)
                params["stop_loss_order_type"] = "market"

        result = self._exchange.create_market_order(pair, side, amount, params=params)
        return self._normalize_fill_units(result, contract_size)

    def limit_open(self, symbol: str, is_buy: bool, size: float, price: float, inst_type: str = "futures", reduce_only: bool = False) -> dict:
        """Place a resting LIMIT order (maker fill on the pullback). `size` is in COIN units and is
        converted to contracts internally (same as market_open); the returned CCXT dict is normalized
        back to coin units. `reduce_only=True` for a reduce-only TP limit. Used by the mother-son retest
        entry, which rests a limit AT the broken level and fills passively — the supertrend executor only
        ever needed market_open, so this is the one method the retest strategy adds."""
        if not self._is_live:
            raise RuntimeError("limit_open requires live mode")

        self._load_markets()
        side = "buy" if is_buy else "sell"
        pair = self._format_symbol(symbol, inst_type)

        amount = size
        contract_size = 1.0
        if inst_type == "futures":
            market = self._exchange.market(pair)
            contract_size = market.get("contractSize", 1.0) or 1.0
            amount = size / contract_size

        amount = float(self._exchange.amount_to_precision(pair, amount))
        px = float(self._exchange.price_to_precision(pair, price))

        params = {}
        if inst_type == "futures":
            params["type"] = "future"
            if reduce_only:
                params["reduce_only"] = True   # Delta REST body uses snake_case (verified, see market_stop_loss)

        result = self._exchange.create_order(pair, "limit", side, amount, price=px, params=params)
        return self._normalize_fill_units(result, contract_size)

    def market_stop_loss(self, symbol: str, is_buy: bool, size: float, stop_price: float, inst_type: str = "spot") -> dict:
        """Place a stop-loss or take-profit order."""
        if not self._is_live:
            raise RuntimeError("market_stop_loss requires live mode")
        
        self._load_markets()
        side = "buy" if is_buy else "sell"
        pair = self._format_symbol(symbol, inst_type)
        
        amount = size
        if inst_type == "futures":
            market = self._exchange.market(pair)
            contract_size = market.get("contractSize", 1.0)
            amount = size / contract_size
            
        amount = float(self._exchange.amount_to_precision(pair, amount))
        px = float(self._exchange.price_to_precision(pair, stop_price))
        
        current_price = self.get_spot_price(symbol) if inst_type == "spot" else self.get_perp_price(symbol)
        
        # Determine if TP or SL
        is_tp = False
        if is_buy:
            if px < current_price:
                is_tp = True
        else:
            if px > current_price:
                is_tp = True

        if inst_type == "futures":
            # Delta's stop_order_type enum is stop_loss_order / take_profit_order
            # (NOT *_market — that is rejected with bad_schema); reduce_only is
            # snake_case in Delta's REST body. Verified on testnet 2026-05-30.
            params = {
                "type": "future",
                "stop_price": px,
                "stop_order_type": "take_profit_order" if is_tp else "stop_loss_order",
                "reduce_only": True,
            }
            order = self._exchange.create_order(pair, "market", side, amount, price=None, params=params)
            return self._normalize_fill_units(order, contract_size)
        else:
            # Spot limit stop-loss
            limit_px = px * 0.99 if side == "sell" else px * 1.01
            limit_px = float(self._exchange.price_to_precision(pair, limit_px))
            params = {
                "stop_price": px,
                "stop_order_type": "take_profit_limit" if is_tp else "stop_loss_limit",
            }
            return self._exchange.create_order(pair, "limit", side, amount, price=limit_px, params=params)

    def market_close(self, symbol: str, sz: float | None = None) -> dict:
        """Close an open perpetual swap position for a symbol (reduce-only)."""
        if not self._is_live:
            raise RuntimeError("market_close requires live mode")
            
        pair = self._format_symbol(symbol, "futures")
        self._load_markets()
        market = self._exchange.market(pair)
        contract_size = market.get("contractSize", 1.0)

        positions = self._exchange.fetch_positions([pair], params={"type": "future"})
        results = []
        for pos in positions:
            amount = abs(float(pos.get("contracts", 0) or pos.get("amount", 0) or 0))
            if amount > 0:
                pos_side = pos.get("side", "")
                close_side = "sell" if pos_side in ("long", "buy") else "buy"
                
                close_sz = amount
                if sz is not None:
                    close_sz = min(float(sz) / contract_size, amount)
                
                close_sz = float(self._exchange.amount_to_precision(pair, close_sz))
                if close_sz <= 0:
                    continue
                    
                raw = self._exchange.create_market_order(
                    pair, close_side, close_sz,
                    params={"type": "future", "reduceOnly": True}
                )
                results.append(self._normalize_fill_units(raw, contract_size))
        return results[0] if results else {}

    def _normalize_fill_units(self, order: dict, contract_size: float) -> dict:
        """Rewrite CCXT order dict so `filled`, `amount`, and `cost` are in
        coin units, not contracts. Without this, downstream Go code records
        contract counts as quantity and inflates positions by 1/contract_size
        (e.g. 100x for Delta India ETH where contractSize=0.01)."""
        if not order or not isinstance(order, dict) or not contract_size or contract_size == 1.0:
            return order
        for key in ("filled", "amount", "remaining"):
            v = order.get(key)
            if v is None:
                continue
            try:
                order[key] = float(v) * float(contract_size)
            except (TypeError, ValueError):
                pass
        return order

    def tiered_stop_loss(self, symbol: str, inst_type: str, side: str, prices: list[float], size: float) -> tuple[list[str], list[str]]:
        """Place multiple reduce-only stop/TP orders."""
        oids = []
        errors = []
        is_buy_protection = (side.lower() in ("sell", "short"))
        tier_size = size / len(prices) if prices else 0
        
        for px in prices:
            try:
                order = self.market_stop_loss(symbol, is_buy_protection, tier_size, px, inst_type)
                oids.append(str(order.get("id", "")))
                errors.append("")
            except Exception as e:
                oids.append("")
                errors.append(str(e))
                
        return oids, errors

    def cancel_orders(self, symbol: str, inst_type: str, oids: list[str]):
        """Cancel multiple orders by ID. Tries a plain cancel first, then a
        trigger-aware cancel — Delta stop orders may live on the trigger/stop
        order list (mirrors the Binance conditional-order handling)."""
        pair = self._format_symbol(symbol, inst_type)
        base = {"type": "future"} if inst_type == "futures" else {}
        for oid in oids:
            if not oid or str(oid) == "0":
                continue
            ok = False
            for extra in ({}, {"trigger": True}, {"stop": True}):
                try:
                    self._exchange.cancel_order(str(oid), pair, params={**base, **extra})
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                print(f"WARNING: Failed to cancel order {oid} (plain + trigger + stop)", file=sys.stderr)

    def list_open_stop_orders(self, symbol: str, inst_type: str) -> list:
        """Return open protective (stop/take-profit) orders for a symbol as
        [{id, side, trigger_price, reduce_only, qty}]. Delta's untriggered stop
        orders may appear on the plain or the trigger order list depending on
        type, so query both and merge by id (mirrors the Binance fix)."""
        pair = self._format_symbol(symbol, inst_type)
        base = {"type": "future"} if inst_type == "futures" else {}
        seen, out = set(), []
        for extra in ({}, {"trigger": True}, {"stop": True}):
            try:
                for o in self._exchange.fetch_open_orders(pair, params={**base, **extra}) or []:
                    info = o.get("info") or {}
                    oid = str(o.get("id") or info.get("id") or "")
                    trig = o.get("stopPrice") or o.get("triggerPrice") or info.get("stop_price") or info.get("triggerPrice")
                    try:
                        trig = float(trig) if trig is not None else None
                    except (TypeError, ValueError):
                        trig = None
                    # Only treat orders that actually carry a trigger price as protective.
                    if trig is None or not oid or oid in seen:
                        continue
                    seen.add(oid)
                    ro = o.get("reduceOnly")
                    if ro is None:
                        ro = str(info.get("reduce_only")).lower() == "true"
                    out.append({
                        "id": oid,
                        "side": (o.get("side") or info.get("side") or "").lower(),
                        "trigger_price": trig,
                        "reduce_only": bool(ro),
                        "qty": o.get("amount") or info.get("size"),
                    })
            except Exception:
                continue
        return out

    def get_account_balance(self, inst_type: str = "spot") -> float:
        """Return total USDT-denominated account value."""
        if not self._is_live:
            raise RuntimeError("get_account_balance requires live mode")
            
        params = {"type": "future"} if inst_type == "futures" else {}
        bal = self._exchange.fetch_balance(params=params)
        total = bal.get("total") or {}
        try:
            return float(total.get("USDT") or total.get("USD") or 0.0)
        except (TypeError, ValueError):
            return 0.0
