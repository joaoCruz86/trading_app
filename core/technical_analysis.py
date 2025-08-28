# core/technical_analysis.py

"""
core/technical_analysis.py

This module provides functions to fetch historical price data and compute a range of technical indicators (e.g., RSI, MACD, EMAs, SMAs, Bollinger Bands) using yfinance and pandas-ta.

Purpose:
- Used during price data backfilling to enrich the MongoDB dataset with daily technical indicators.
- Can be called independently to retrieve the latest technical summary for any ticker.

Note:
- This module does NOT currently feed into Layer 1 (rules_engine.py).
- It is not used in rule-based screening or confidence scoring.
- It may support future enhancements or Layer 2 (AI signal generation).

"""


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

def compute_technical_indicators(df, use_candle_features=True, volume_filter=True):
    """
    Compute technical indicators on price data, including candlestick features.
    """
    df = df.copy()
    print(f"⚙️ Computing indicators, rows={len(df)}, cols={list(df.columns)}")

    # --- Standard Indicators ---
    try:
        df["RSI"] = ta.rsi(df["Close"], length=14)
    except Exception as e:
        print("❌ RSI failed:", e)

    try:
        macd = ta.macd(df["Close"])
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
    except Exception as e:
        print("❌ MACD failed:", e)

    try:
        df["EMA_20"] = ta.ema(df["Close"], length=20)
        df["EMA_50"] = ta.ema(df["Close"], length=50)
    except Exception as e:
        print("❌ EMA failed:", e)

    try:
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["SMA_200"] = ta.sma(df["Close"], length=200)
    except Exception as e:
        print("❌ SMA failed:", e)

    try:
        bbands = ta.bbands(df["Close"])
        upper_col = next((col for col in bbands.columns if col.startswith("BBU")), None)
        lower_col = next((col for col in bbands.columns if col.startswith("BBL")), None)
        if upper_col and lower_col:
            df["BB_upper"] = bbands[upper_col]
            df["BB_lower"] = bbands[lower_col]
    except Exception as e:
        print("❌ Bollinger Bands failed:", e)

    # --- Candlestick Features ---
    if use_candle_features:
        try:
            df["body_size"] = (df["Close"] - df["Open"]).abs()
            df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
            df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
            df["is_bullish"] = (df["Close"] > df["Open"]).astype(int)

            # Normalize wick and body ratios
            df["body_to_range"] = df["body_size"] / (df["High"] - df["Low"] + 1e-6)
            df["wick_to_body"] = (df["upper_wick"] + df["lower_wick"]) / (df["body_size"] + 1e-6)

            # Optional: filter for meaningful candles only (body must be a decent part of range)
            if volume_filter:
                rolling_volume = df["Volume"].rolling(5).mean()
                df["high_volume"] = (df["Volume"] > rolling_volume).astype(int)
                df = df[df["high_volume"] == 1]
                df = df[df["body_to_range"] > 0.2]  # remove doji/noise
        except Exception as e:
            print("❌ Candlestick features failed:", e)

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
