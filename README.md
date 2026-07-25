# LBOG — Line Break Original Trading Strategy & Delta Exchange Integration

LBOG is a trend-following trading strategy powered by a canonical Three-Line-Break (3LB) charting system and a ratcheting previous-candle stop loss.

## Strategy Overview

- **Entry Signals:** 
  - **Long:** Triggered on the first Up line-break brick. Stop loss is initialized at the previous candle's Low.
  - **Short:** Triggered on the first Down line-break brick. Stop loss is initialized at the previous candle's High.
- **Ratcheting Stop Loss:** As price moves favorably in the trade direction, the Stop Loss triggers ratchet higher (for Longs) or lower (for Shorts) to lock in profits and never loosen.
- **Reversals & Exits:** Position exits immediately on a Stop Loss hit or flips direction when an opposite line-break brick prints.

## Setup & Running

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Delta Exchange Demo / Live credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
DELTA_SANDBOX=1
DELTA_DEMO_API_KEY=your_demo_api_key
DELTA_DEMO_API_SECRET=your_demo_api_secret
```

### 3. Run Strategy Tests
```bash
python3 shared_strategies/open/lbog/test_lbog.py
```

### 4. Run Strategy on Delta Exchange
```bash
python3 shared_scripts/check_delta.py lbog BTC 15m --mode=live --inst-type=futures
```
