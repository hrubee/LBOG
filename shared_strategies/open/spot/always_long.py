"""
Always long strategy: returns 1 (buy) on every call.
"""
def solve(df, metadata):
    # df: DataFrame with OHLCV columns
    # metadata: dict with symbol, timeframe, etc.
    return 1
