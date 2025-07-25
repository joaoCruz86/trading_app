# core/fundamentals.py

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
