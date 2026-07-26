#!/usr/bin/env python3
"""
Live Execution Runner for LBOG (Line Break Original) Strategy on Delta Exchange.
Runs continuously on your Mac, fetching market updates, evaluating 3LB signals,
maintaining ratcheting stop-losses, and enforcing strict single-position risk rules.
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timezone

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'platforms', 'delta'))
sys.path.insert(0, os.path.join(BASE_DIR, 'shared_scripts'))
sys.path.insert(0, os.path.join(BASE_DIR, 'shared_tools'))
sys.path.insert(0, os.path.join(BASE_DIR, 'shared_strategies', 'open', 'lbog'))

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

from adapter import DeltaExchangeAdapter
from lbog import lbog_strategy, linebreak
from check_delta import _make_dataframe, run_sync_protection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, 'live_trader.log'))
    ]
)

import argparse

# Parse CLI parameters
parser = argparse.ArgumentParser(description="LBOG Live Strategy Runner")
parser.add_argument("--symbols", type=str, default="BTC,ETH", help="Comma-separated trading asset symbols (default: BTC,ETH)")
parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe (default: 5m)")
parser.add_argument("--size", type=float, default=0.001, help="Default fallback position size in coin units (default: 0.001)")
parser.add_argument("--interval", type=int, default=10, help="Loop sleep interval in seconds (default: 10)")
args = parser.parse_args()

SYMBOLS = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
TIMEFRAME = args.timeframe
INST_TYPE = "futures"
POSITION_SIZE = args.size
LOOP_INTERVAL = args.interval

FITS_LOG_FILE = os.path.join(BASE_DIR, 'wallet_trades.log')
FITS_JSON_FILE = os.path.join(BASE_DIR, 'wallet_fills.json')

import urllib.request
import urllib.parse

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8932106364:AAEZXH_UCyvYmJi3W_oFl9piAvsNQ07k8pA")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7871236037")


CHARTS_DIR = os.path.join(BASE_DIR, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def send_telegram_notification(msg: str):
    """Send text trade notification via Telegram Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
ENTRY_MSGS_FILE = os.path.join(os.getcwd(), "entry_msgs.json")
def load_entry_msg_ids() -> dict:
    if os.path.exists(ENTRY_MSGS_FILE):
        try:
            with open(ENTRY_MSGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_entry_msg_ids(msg_dict: dict):
    try:
        with open(ENTRY_MSGS_FILE, "w") as f:
            json.dump(msg_dict, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving entry_msgs.json: {e}")

ENTRY_MSG_IDS = load_entry_msg_ids()


def send_telegram_notification(msg: str, reply_to_message_id: int = None) -> int:
    """Send text notification via Telegram Bot API. Returns message_id."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
    if not token or not chat_id:
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg}
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                return data.get("result", {}).get("message_id")
    except Exception as e:
        logging.error(f"Error sending Telegram notification: {e}")
    return None


def send_telegram_photo(image_path: str, caption: str, reply_to_message_id: int = None) -> int:
    """Send matplotlib chart image via Telegram Bot sendPhoto API. Returns message_id."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
    if not token or not chat_id or not os.path.exists(image_path):
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            if reply_to_message_id:
                data["reply_to_message_id"] = reply_to_message_id
            resp = requests.post(url, data=data, files=files, timeout=10)
            if resp.status_code == 200:
                res_data = resp.json()
                if res_data.get("ok"):
                    return res_data.get("result", {}).get("message_id")
    except Exception as e:
        logging.error(f"Error sending Telegram photo: {e}")
    return None


def render_entry_chart(df, symbol: str, timeframe: str, side: str, entry_price: float, sl_price: float, fill_id: str) -> str:
    """
    Renders an authentic dark-mode Line Break chart matching Delta Exchange / TradingView using stocktrends.
    Returns path to rendered PNG image.
    """
    try:
        from stocktrends import LineBreak
        import matplotlib.patches as patches

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')

        df_st = df.reset_index()
        df_st.rename(columns={'datetime': 'date'}, inplace=True)

        lb = LineBreak(df_st)
        lb.line_number = 3  # 3-Line Break (3LB)
        data = lb.get_ohlc_data()

        recent_data = data.tail(35).reset_index(drop=True)
        n_bricks = len(recent_data)

        # Draw TradingView Box Bricks using Rectangle patches (Zero Gap)
        for idx, row in recent_data.iterrows():
            op = row['open']
            cl = row['close']
            bot = min(op, cl)
            top = max(op, cl)
            h = max(top - bot, 1.0)
            color = '#22C55E' if cl >= op else '#EF4444'
            
            rect = patches.Rectangle((idx - 0.5, bot), 1.0, h, facecolor=color, edgecolor='#0E1117', linewidth=1, alpha=0.9)
            ax.add_patch(rect)

        last_idx = n_bricks - 1 if n_bricks > 0 else 0
        marker_color = '#22C55E' if side.upper() in ('BUY', 'LONG') else '#EF4444'
        marker_symbol = '^' if side.upper() in ('BUY', 'LONG') else 'v'
        
        ax.scatter([last_idx], [entry_price], color=marker_color, s=180, marker=marker_symbol, zorder=6, label=f'{side.upper()} Fill (${entry_price:,.2f})')

        if sl_price > 0:
            ax.axhline(sl_price, color='#FACC15', linestyle='--', linewidth=1.8, zorder=5, label=f'Stop Loss (${sl_price:,.2f})')

        all_y = list(recent_data['open'].values) + list(recent_data['close'].values) + [entry_price]
        if sl_price > 0:
            all_y.append(sl_price)
        
        y_min = min(all_y)
        y_max = max(all_y)
        y_pad = max((y_max - y_min) * 0.08, 15.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlim(-1, n_bricks)

        ax.set_title(f'{symbol}USD — {timeframe} — Delta — Line Break [3]', fontsize=13, fontweight='bold', color='white', pad=12)
        ax.set_xlabel('Line Break Brick Index', color='#94A3B8')
        ax.set_ylabel('Price (USD)', color='#94A3B8')
        ax.grid(True, color='#1E293B', linestyle=':', alpha=0.6)
        
        ax.legend(loc='upper left', facecolor='#18181f', edgecolor='#333333')

        chart_path = os.path.join(CHARTS_DIR, f'fill_{fill_id[:8]}.png')
        plt.savefig(chart_path, bbox_inches='tight', facecolor='#0E1117')
        plt.close(fig)
        return chart_path
    except Exception as e:
        logging.error(f"Error rendering stocktrends 1LB brick chart: {e}")
        return ""


def seed_historical_fill_ids(adapter: DeltaExchangeAdapter, symbols: list):
    """Seed existing historical trade IDs into wallet_fills.json on startup to avoid replaying past fills."""
    known_fill_ids = set()
    if os.path.exists(FITS_JSON_FILE):
        try:
            with open(FITS_JSON_FILE, 'r') as f:
                known_fill_ids = set(json.load(f))
        except Exception:
            pass

    for sym in symbols:
        try:
            trades = adapter.fetch_my_trades(sym, inst_type=INST_TYPE, limit=100)
            if trades:
                for trade in trades:
                    trade_id = str(trade.get("id") or trade.get("info", {}).get("id") or "")
                    if trade_id:
                        known_fill_ids.add(trade_id)
        except Exception as e:
            logging.error(f"Error seeding historical fills for {sym}: {e}")

    try:
        with open(FITS_JSON_FILE, 'w') as f:
            json.dump(list(known_fill_ids), f, indent=2)
        logging.info(f"Seeded {len(known_fill_ids)} historical wallet trade fills into memory/disk.")
    except Exception as e:
        logging.error(f"Error saving seeded fills to wallet_fills.json: {e}")


def sync_real_wallet_fills(adapter: DeltaExchangeAdapter, symbol: str, df=None, sl_level: float = 0.0):
    """
    Queries actual executed trade fills directly from Delta Exchange account/wallet.
    Logs each new real fill to wallet_trades.log, stdout, and sends instant Telegram text + matplotlib chart photo!
    """
    known_fill_ids = set()
    if os.path.exists(FITS_JSON_FILE):
        try:
            with open(FITS_JSON_FILE, 'r') as f:
                known_fill_ids = set(json.load(f))
        except Exception:
            pass

    trades = adapter.fetch_my_trades(symbol, inst_type=INST_TYPE, limit=50)
    if not trades:
        return

    new_fills = []
    for trade in trades:
        trade_id = str(trade.get("id") or trade.get("info", {}).get("id") or "")
        if not trade_id or trade_id in known_fill_ids:
            continue

        timestamp = trade.get("datetime") or trade.get("timestamp")
        side = (trade.get("side") or "").upper()
        price = float(trade.get("price") or 0.0)
        amount = float(trade.get("amount") or 0.0)
        cost = float(trade.get("cost") or 0.0)
        fee_info = trade.get("fee")
        fee_cost = fee_info.get("cost") if isinstance(fee_info, dict) else fee_info

        info = trade.get("info", {})
        meta = info.get("meta_data", {})
        new_pos = meta.get("new_position", {})
        new_pos_size = new_pos.get("size", None)
        r_pnl = new_pos.get("realized_pnl") or info.get("pnl") or info.get("realized_pnl") or "0"

        # Determine if this fill is an EXIT (closing out position to size 0)
        is_exit_fill = (new_pos_size is not None and float(new_pos_size) == 0.0)

        fill_record = (
            f"[REAL WALLET FILL] Trade ID: {trade_id} | Time: {timestamp} | Symbol: {symbol} | "
            f"Side: {side} | Price: ${price} | Amount: {amount} | Cost: ${cost} | Realized PnL: ${r_pnl} | Fee: ${fee_cost}"
        )
        
        # Log to console and local file
        logging.info(fill_record)
        with open(FITS_LOG_FILE, 'a') as lf:
            lf.write(fill_record + "\n")

        if is_exit_fill:
            # EXIT FILL: Reply 'exit' directly to the corresponding entry alert message!
            reply_id = ENTRY_MSG_IDS.get(symbol)
            pnl_val = float(r_pnl) if r_pnl else 0.0
            pnl_str = f"+${pnl_val:.2f}" if pnl_val >= 0 else f"-${abs(pnl_val):.2f}"
            exit_text = f"exit (PnL: {pnl_str})" if pnl_val != 0.0 else "exit"
            
            send_telegram_notification(exit_text, reply_to_message_id=reply_id)
            logging.info(f"Sent EXIT reply to Telegram entry message ID {reply_id}: {exit_text}")
        else:
            # ENTRY FILL: Send rendered 3LB chart photo with full fill details as its caption!
            emoji = "🟢 LONG FILL" if side == "BUY" else "🔴 SHORT FILL"
            caption = (
                f"⚡ {emoji}\n\n"
                f"• Asset: {symbol} Perpetual\n"
                f"• Action: {side}\n"
                f"• Fill Price: ${price:,.2f}\n"
                f"• Size: {amount} {symbol}\n"
                f"• Notional Value: ${cost:,.2f}\n"
                f"• Fee: ${fee_cost}\n"
                f"• Fill ID: {trade_id[:12]}...\n"
                f"• Time: {timestamp}"
            )
            
            msg_id = None
            if df is not None:
                chart_path = render_entry_chart(df, symbol, TIMEFRAME, side, price, sl_level, trade_id)
                if chart_path and os.path.exists(chart_path):
                    msg_id = send_telegram_photo(chart_path, caption=caption)

            # Fallback to text notification if photo send fails
            if not msg_id:
                msg_id = send_telegram_notification(caption)

            if msg_id:
                ENTRY_MSG_IDS[symbol] = msg_id
                save_entry_msg_ids(ENTRY_MSG_IDS)

        known_fill_ids.add(trade_id)
        new_fills.append(trade_id)

    if new_fills:
        try:
            with open(FITS_JSON_FILE, 'w') as f:
                json.dump(list(known_fill_ids), f, indent=2)
        except Exception as e:
            logging.error(f"Error updating wallet_fills.json: {e}")


def check_existing_position(adapter: DeltaExchangeAdapter, symbol: str, current_price: float = 0.0) -> dict:
    """
    Fetch open positions from Delta Exchange to check if a trade is currently active.
    Returns dict: {"active": bool, "side": "long"|"short"|"flat", "size": float, "entry_price": float, "unrealized_pnl": float, "pnl_pct": float, "raw": dict}
    """
    try:
        positions = adapter.fetch_open_positions()
        target_pair = adapter._format_symbol(symbol, INST_TYPE)
        for pos in positions:
            sym = pos.get("symbol") or pos.get("info", {}).get("product_symbol")
            if sym == target_pair or (sym and symbol in sym):
                contracts = float(pos.get("contracts", 0) or pos.get("info", {}).get("size", 0) or 0)
                contract_size = float(pos.get("contractSize") or pos.get("info", {}).get("product", {}).get("contract_value") or 1.0)
                
                # Convert contract counts to coin units (e.g. 12 contracts * 0.001 = 0.012 BTC)
                pos_size_coin = contracts * contract_size
                if abs(pos_size_coin) > 1e-6:
                    side = "long" if pos.get("side") in ("long", "buy") or contracts > 0 else "short"
                    entry_px = float(pos.get("entryPrice") or pos.get("info", {}).get("entry_price") or 0.0)
                    
                    unrealized_pnl = float(pos.get("unrealizedPnl") or pos.get("info", {}).get("unrealized_pnl") or 0.0)
                    if unrealized_pnl == 0.0 and current_price > 0 and entry_px > 0:
                        if side == "long":
                            unrealized_pnl = (current_price - entry_px) * abs(pos_size_coin)
                        else:
                            unrealized_pnl = (entry_px - current_price) * abs(pos_size_coin)
                    
                    pnl_pct = 0.0
                    if entry_px > 0 and abs(pos_size_coin) > 0:
                        pos_cost = entry_px * abs(pos_size_coin)
                        pnl_pct = (unrealized_pnl / pos_cost) * 100.0 if pos_cost > 0 else 0.0

                    return {
                        "active": True,
                        "side": side,
                        "size": abs(pos_size_coin),
                        "entry_price": entry_px,
                        "unrealized_pnl": unrealized_pnl,
                        "pnl_pct": pnl_pct,
                        "raw": pos
                    }
    except Exception as e:
        logging.error(f"Error fetching active positions from Delta Exchange for {symbol}: {e}")
    return {"active": False, "side": "flat", "size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0, "pnl_pct": 0.0, "raw": None}


def calculate_3pct_risk_size(adapter: DeltaExchangeAdapter, symbol: str, entry_price: float, sl_price: float, max_leverage: float = 20.0) -> float:
    """
    Calculate position size strictly based on 3% wallet risk per trade using full leverage capacity (20x max leverage cap).
    """
    wallet_balance = adapter.get_wallet_balance()
    risk_amount = wallet_balance * 0.03  # 3% risk of wallet size
    risk_distance = abs(entry_price - sl_price)

    min_size = get_minimum_trade_size(symbol)

    if risk_distance <= 0 or entry_price <= 0:
        return min_size

    # Fetch available free margin to prevent 'insufficient_margin' when multiple assets trade simultaneously
    avail_margin = wallet_balance
    try:
        bal_info = adapter._exchange.fetch_balance()
        free_bal = float(bal_info.get("free", {}).get("USD") or bal_info.get("USD", {}).get("free") or 0.0)
        if free_bal > 0:
            avail_margin = free_bal
    except Exception:
        pass

    raw_size_coin = risk_amount / risk_distance
    # Cap notional size to 98% of available free margin * max_leverage (20.0x)
    max_notional = (avail_margin * 0.98) * max_leverage
    max_size_cap = max_notional / entry_price if entry_price > 0 else 0.0

    final_size_coin = min(raw_size_coin, max_size_cap)
    floored_size = adapter.floor_size(symbol, final_size_coin)

    logging.info(
        f"Lot Sizing [{symbol} 3% Risk + {max_leverage:.0f}x Leverage Cap]: Total Bal = ${wallet_balance:.2f} | Avail Margin = ${avail_margin:.2f} | "
        f"3% Risk = ${risk_amount:.2f} | Risk Distance = ${risk_distance:.2f} | Raw Size = {raw_size_coin:.4f} {symbol} | Capped Size = {final_size_coin:.4f} {symbol}"
    )

    return floored_size if floored_size >= min_size else min_size


def get_minimum_trade_size(symbol: str) -> float:
    """Return exchange-compliant minimum trade size in coin units."""
    return 0.01 if symbol.upper() == "ETH" else 0.001


LAST_PROCESSED_CANDLE_TIME = {}
IS_ORDER_IN_FLIGHT = {}


def execute_trade_cycle(adapter: DeltaExchangeAdapter, symbol: str):
    logging.info(f"--- Cycle check: {symbol} ({TIMEFRAME}, {INST_TYPE}) ---")
    
    if IS_ORDER_IN_FLIGHT.get(symbol, False):
        logging.info(f"Order currently in-flight for {symbol}. Skipping new entry check...")
        return
    
    # 1. Fetch OHLCV candles
    candles = adapter.get_ohlcv(symbol, interval=TIMEFRAME, limit=200, inst_type=INST_TYPE)
    # Ensure we only process closed candles (drop in-progress live candle)
    tf_ms = 5 * 60 * 1000
    if TIMEFRAME.endswith("m"):
        tf_ms = int(TIMEFRAME.replace("m", "")) * 60 * 1000
    elif TIMEFRAME.endswith("h"):
        tf_ms = int(TIMEFRAME.replace("h", "")) * 3600 * 1000

    now_ms = time.time() * 1000
    if candles and (now_ms - candles[-1][0]) < tf_ms:
        candles = candles[:-1]

    if not candles:
        return

    latest_candle_time = candles[-1][0]
    is_new_candle = (LAST_PROCESSED_CANDLE_TIME.get(symbol) != latest_candle_time)

    df = _make_dataframe(candles)
    
    # 2. Run LBOG Strategy Core
    result = lbog_strategy(df)
    latest_bar = result.iloc[-1]
    sig = int(latest_bar["signal"])
    sl_level = float(latest_bar["sl_level"])
    lb_dir = int(latest_bar["lb_dir"])
    latest_close = float(df.iloc[-1]["close"])

    # 3. Sync and log real wallet fills + send matplotlib chart photo to Telegram
    sync_real_wallet_fills(adapter, symbol, df=df, sl_level=sl_level)

    logging.info(f"Market Close ({symbol}): ${latest_close:.2f} | 3LB Trend: {lb_dir} | Signal: {sig} | Calculated SL: ${sl_level:.2f}")

    # 4. Check account's active position on Delta Exchange
    pos_info = check_existing_position(adapter, symbol, current_price=latest_close)
    if pos_info["active"]:
        pnl_str = f"+${pos_info['unrealized_pnl']:.2f}" if pos_info['unrealized_pnl'] >= 0 else f"-${abs(pos_info['unrealized_pnl']):.2f}"
        pct_str = f"+{pos_info['pnl_pct']:.2f}%" if pos_info['pnl_pct'] >= 0 else f"{pos_info['pnl_pct']:.2f}%"
        logging.info(
            f"Account Position State ({symbol}): active=True | Side={pos_info['side'].upper()} | Size={pos_info['size']} {symbol} | "
            f"Entry Price=${pos_info['entry_price']:.2f} | Unrealized PnL={pnl_str} ({pct_str})"
        )
    else:
        logging.info(f"Account Position State ({symbol}): active=False (Flat)")

    # Calculate 3% risk-based position size if entering a new trade (20x leverage capacity)
    trade_size = get_minimum_trade_size(symbol)
    if sl_level > 0:
        trade_size = calculate_3pct_risk_size(adapter, symbol, latest_close, sl_level, max_leverage=20.0)
    
    # 1:2 Risk/Reward Take Profit (TP) Calculation
    risk_dist = abs(latest_close - sl_level) if sl_level > 0 else 30.0
    tp_level = 0.0
    if pos_info["active"]:
        entry_px = pos_info["entry_price"] if pos_info["entry_price"] > 0 else latest_close
        if pos_info["side"] == "long":
            tp_level = entry_px + (2.0 * risk_dist)
        else:
            tp_level = max(0.0, entry_px - (2.0 * risk_dist))
    elif sig == 1:
        tp_level = latest_close + (2.0 * risk_dist)
    elif sig == -1:
        tp_level = max(0.0, latest_close - (2.0 * risk_dist))

    tp_tiers_json = json.dumps([{"tp_price": tp_level, "pct": 1.0}]) if tp_level > 0 else ""

    # Trade Entry logic (Strict 4-Rule Architecture + 1 Entry Per Candle Guard):
    # Rule 1 & 2: Enter ONLY when a fresh 3LB brick breakout signal fires (sig == 1 or sig == -1) AND candle is NEW.
    # Rule 3: Zero stacking when position is active. Zero duplicate entries within the same 5m bar.
    should_enter_long = (sig == 1) and is_new_candle
    should_enter_short = (sig == -1) and is_new_candle

    if should_enter_long:
        LAST_PROCESSED_CANDLE_TIME[symbol] = latest_candle_time
        if pos_info["active"] and pos_info["side"] == "long":
            logging.info(f"Position is already LONG on {symbol}. Syncing SL @ ${sl_level:.2f} | 1:2 TP @ ${tp_level:.2f}...")
            try:
                run_sync_protection(
                    symbol=symbol, side=pos_info["side"], size=pos_info["size"], avg_cost=latest_close,
                    entry_atr=0.0, stop_loss_atr_mult=0.0, tiers_json=tp_tiers_json, stop_loss_oid="", tp_oids_json="",
                    tp_armed_tiers_json="", inst_type=INST_TYPE, stop_loss_price=sl_level
                )
            except Exception as e:
                logging.error(f"Error syncing protection for LONG position: {e}")
        else:
            IS_ORDER_IN_FLIGHT[symbol] = True
            try:
                if pos_info["active"] and pos_info["side"] == "short":
                    logging.info(f"Reversal signal: Closing active SHORT position on {symbol} first...")
                    adapter.market_open(symbol, is_buy=True, size=pos_info["size"], inst_type=INST_TYPE, reduce_only=True)
                    time.sleep(1)

                logging.info(f"EXECUTIVE ENTRY: Opening LONG position of size {trade_size} {symbol} with Bracket SL @ ${sl_level:.2f} & 1:2 TP @ ${tp_level:.2f}...")
                order_res = adapter.market_open(symbol, is_buy=True, size=trade_size, inst_type=INST_TYPE, stop_loss_price=sl_level if sl_level > 0 else None)
                logging.info(f"Atomic Order & SL filled: {order_res}")
                time.sleep(1)
                if sl_level > 0:
                    try:
                        run_sync_protection(
                            symbol=symbol, side="long", size=trade_size, avg_cost=latest_close, entry_atr=0.0,
                            stop_loss_atr_mult=0.0, tiers_json=tp_tiers_json, stop_loss_oid="", tp_oids_json="",
                            tp_armed_tiers_json="", inst_type=INST_TYPE, stop_loss_price=sl_level
                        )
                    except Exception as e:
                        logging.error(f"Error placing initial SL for LONG: {e}")
            finally:
                IS_ORDER_IN_FLIGHT[symbol] = False

    elif should_enter_short:
        LAST_PROCESSED_CANDLE_TIME[symbol] = latest_candle_time
        if pos_info["active"] and pos_info["side"] == "short":
            logging.info(f"Position is already SHORT on {symbol}. Syncing SL @ ${sl_level:.2f} | 1:2 TP @ ${tp_level:.2f}...")
            try:
                run_sync_protection(
                    symbol=symbol, side=pos_info["side"], size=pos_info["size"], avg_cost=latest_close,
                    entry_atr=0.0, stop_loss_atr_mult=0.0, tiers_json=tp_tiers_json, stop_loss_oid="", tp_oids_json="",
                    tp_armed_tiers_json="", inst_type=INST_TYPE, stop_loss_price=sl_level
                )
            except Exception as e:
                logging.error(f"Error syncing protection for SHORT position: {e}")
        else:
            IS_ORDER_IN_FLIGHT[symbol] = True
            try:
                if pos_info["active"] and pos_info["side"] == "long":
                    logging.info(f"Reversal signal: Closing active LONG position on {symbol} first...")
                    adapter.market_open(symbol, is_buy=False, size=pos_info["size"], inst_type=INST_TYPE, reduce_only=True)
                    time.sleep(1)

                logging.info(f"EXECUTIVE ENTRY: Opening SHORT position of size {trade_size} {symbol} with Bracket SL @ ${sl_level:.2f} & 1:2 TP @ ${tp_level:.2f}...")
                order_res = adapter.market_open(symbol, is_buy=False, size=trade_size, inst_type=INST_TYPE, stop_loss_price=sl_level if sl_level > 0 else None)
                logging.info(f"Atomic Order & SL filled: {order_res}")
                time.sleep(1)
                if sl_level > 0:
                    try:
                        run_sync_protection(
                            symbol=symbol, side="short", size=trade_size, avg_cost=latest_close, entry_atr=0.0,
                            stop_loss_atr_mult=0.0, tiers_json=tp_tiers_json, stop_loss_oid="", tp_oids_json="",
                            tp_armed_tiers_json="", inst_type=INST_TYPE, stop_loss_price=sl_level
                        )
                    except Exception as e:
                        logging.error(f"Error placing initial SL for SHORT: {e}")
            finally:
                IS_ORDER_IN_FLIGHT[symbol] = False

    else: # sig == 0
        if pos_info["active"] and sl_level > 0:
            logging.info(f"Maintaining active {pos_info['side'].upper()} trade on {symbol}. Syncing SL @ ${sl_level:.2f} | 1:2 TP @ ${tp_level:.2f}...")
            close_is_buy = (pos_info["side"] == "short")
            try:
                run_sync_protection(
                    symbol=symbol,
                    side=pos_info["side"],
                    size=pos_info["size"],
                    avg_cost=latest_close,
                    entry_atr=0.0,
                    stop_loss_atr_mult=0.0,
                    tiers_json=tp_tiers_json,
                    stop_loss_oid="",
                    tp_oids_json="",
                    tp_armed_tiers_json="",
                    inst_type=INST_TYPE,
                    stop_loss_price=sl_level
                )
            except Exception as e:
                logging.error(f"Error updating stop loss for {symbol}: {e}")


def main():
    is_real = os.environ.get("DELTA_SANDBOX", "1") == "0"
    mode_name = "REAL Production API" if is_real else "Demo Account API"
    logging.info("=" * 65)
    logging.info(f"Starting LBOG Live Execution Daemon on Mac ({mode_name})")
    logging.info("=" * 65)
    
    adapter = DeltaExchangeAdapter()
    if not adapter.is_live:
        logging.error("Delta API Key / Secret missing in .env! Cannot start live runner.")
        sys.exit(1)

    logging.info(f"Successfully connected to Delta Exchange {mode_name}.")
    for sym in SYMBOLS:
        try:
            pair = adapter._format_symbol(sym, INST_TYPE)
            adapter._exchange.set_leverage(20, pair)
            logging.info(f"Set Exchange Leverage to 20x for {sym} ({pair}).")
        except Exception as lev_err:
            logging.warning(f"Could not set leverage to 20x on Delta for {sym}: {lev_err}")

    seed_historical_fill_ids(adapter, SYMBOLS)
    logging.info(f"Starting continuous multi-asset loop for {', '.join(SYMBOLS)} on {TIMEFRAME} chart (polling every {LOOP_INTERVAL}s)... Press Ctrl+C to stop.")

    try:
        while True:
            for symbol in SYMBOLS:
                try:
                    execute_trade_cycle(adapter, symbol)
                except Exception as e:
                    logging.error(f"Unexpected error in execution cycle for {symbol}: {e}")
            
            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        logging.info("Shutting down LBOG Live Execution Daemon cleanly. Bye!")


if __name__ == "__main__":
    main()
