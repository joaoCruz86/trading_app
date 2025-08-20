# core/fundamentals.py

"""
core/fundamentals.py

This module fetches company-level fundamental data using the Yahoo Finance API (via yfinance).

Purpose:
- Provides essential financial metrics (growth, valuation, liquidity, profitability) for a given stock ticker.
- Powers the fundamental screening logic in Layer 1 (rules_engine.py).

Metrics Retrieved:
- EPS Growth, Revenue Growth, Debt/Equity, P/E Ratio, Market Cap, Avg Volume
- Plus additional metrics: PEG, P/B, ROE, Dividend Yield, Beta, Current/Quick Ratios, Profit Margin

Notes:
- Values are cleaned and normalized for consistency.
- Some values are scaled (e.g. % or millions).
- Missing or non-numeric values are safely defaulted to 0.

Used directly by: `rules_engine.evaluate_stock()`
"""


import yfinance as yf

def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    def safe_get(key, default=0):
        value = info.get(key)
        return value if isinstance(value, (int, float)) else default

    return {
        # --- Used fundamentals ---
        "EPS_Growth": safe_get("earningsQuarterlyGrowth") * 100,  # %
        "Revenue_Growth": safe_get("revenueGrowth") * 100,        # %
        "Debt/Equity": safe_get("debtToEquity"),                  # ratio
        "P/E": safe_get("trailingPE"),                            # ratio
        "Market_Cap": round(safe_get("marketCap") / 1e6, 2),     # million USD
        "Average_Volume": safe_get("averageVolume"),
        "PEG": safe_get("pegRatio"),
        "P/B": safe_get("priceToBook"),
        "ROE": safe_get("returnOnEquity") * 100,                  # %
        "Dividend_Yield": safe_get("dividendYield") * 100,       # %
        "Beta": safe_get("beta"),
        "Current_Ratio": safe_get("currentRatio"),
        "Quick_Ratio": safe_get("quickRatio"),
        "Profit_Margin": safe_get("profitMargins") * 100         # %
    }
