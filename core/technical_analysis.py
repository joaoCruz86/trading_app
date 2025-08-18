# core/technical_analysis.py

import yfinance as yf
import pandas_ta as ta


# --- Data Fetching ---

def fetch_price_data(ticker, period=None, interval="1d", start_date=None, end_date=None):
    """
    Fetch historical price data from Yahoo Finance.
    Either specify `period` (e.g. '3y') or `start_date` and `end_date`.
    """
    stock = yf.Ticker(ticker)
    
    if start_date and end_date:
        df = stock.history(start=start_date, end=end_date, interval=interval)
    else:
        df = stock.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data fetched for ticker: {ticker}")
    return df



# --- Indicator Calculations ---

def compute_technical_indicators(df):
    """
    Compute technical indicators on price data.
    """
    df = df.copy()

    print(f"⚙️ Computing indicators, rows={len(df)}, cols={list(df.columns)}")

    # RSI
    try:
        df["RSI"] = ta.rsi(df["Close"], length=14)
    except Exception as e:
        print("❌ RSI failed:", e)

    # MACD
    try:
        macd = ta.macd(df["Close"])
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
    except Exception as e:
        print("❌ MACD failed:", e)

    # EMA
    try:
        df["EMA_20"] = ta.ema(df["Close"], length=20)
        df["EMA_50"] = ta.ema(df["Close"], length=50)
    except Exception as e:
        print("❌ EMA failed:", e)

    # SMA
    try:
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["SMA_200"] = ta.sma(df["Close"], length=200)
    except Exception as e:
        print("❌ SMA failed:", e)

    # Bollinger Bands
    try:
        bbands = ta.bbands(df["Close"])
        upper_col = next((col for col in bbands.columns if col.startswith("BBU")), None)
        lower_col = next((col for col in bbands.columns if col.startswith("BBL")), None)

        if upper_col and lower_col:
            df["BB_upper"] = bbands[upper_col]
            df["BB_lower"] = bbands[lower_col]
        else:
            raise ValueError("Missing Bollinger Band columns")
    except Exception as e:
        print("❌ Bollinger Bands failed:", e)

    print(f"✅ Done indicators, now cols={list(df.columns)}")
    return df




# --- Signal Interpretation ---

def interpret_rsi(rsi):
    if rsi is None or rsi != rsi:  # Check for NaN
        return "N/A"
    if rsi > 70:
        return "Overbought"
    elif rsi < 30:
        return "Oversold"
    else:
        return "Neutral"


def interpret_macd(macd, signal):
    if macd is None or signal is None or macd != macd or signal != signal:
        return "N/A"
    if macd > signal:
        return "Bullish"
    elif macd < signal:
        return "Bearish"
    else:
        return "Neutral"



# --- Main Interface ---

def get_technical_summary(ticker):
    """
    Fetch price data, compute indicators, and return latest summary.
    """
    try:
        df = fetch_price_data(ticker, period="6mo")  # ensure enough history for indicators
        df = compute_technical_indicators(df)
        latest = df.iloc[-1]

        summary = {
            "RSI": latest.get("RSI"),
            "RSI_signal": interpret_rsi(latest.get("RSI")),
            "MACD": latest.get("MACD"),
            "MACD_signal": latest.get("MACD_signal"),
            "MACD_trend": interpret_macd(latest.get("MACD"), latest.get("MACD_signal")),
            "EMA_20": latest.get("EMA_20"),
            "EMA_50": latest.get("EMA_50"),
            "SMA_20": latest.get("SMA_20"),
            "SMA_200": latest.get("SMA_200"),
            "BB_upper": latest.get("BB_upper"),
            "BB_lower": latest.get("BB_lower"),
            "Close": latest.get("Close"),
            "Date": latest.name.strftime("%Y-%m-%d")
        }
        return summary

    except Exception as e:
        return {
            "Error": str(e)
        }



# --- Example usage ---

if __name__ == "__main__":
    ticker = "AAPL"
    summary = get_technical_summary(ticker)
    print(f"Technical summary for {ticker}:")
    for key, value in summary.items():
        print(f"{key}: {value}")
