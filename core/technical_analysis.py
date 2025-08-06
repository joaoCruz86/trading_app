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

    # RSI
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # MACD and signal line
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]

    # Exponential moving averages
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["EMA_50"] = ta.ema(df["Close"], length=50)

    # Simple moving averages
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_200"] = ta.sma(df["Close"], length=200)

    # Bollinger Bands with dynamic column detection
    bbands = ta.bbands(df["Close"])

    upper_col = next((col for col in bbands.columns if col.startswith("BBU")), None)
    lower_col = next((col for col in bbands.columns if col.startswith("BBL")), None)

    if upper_col is None or lower_col is None:
        raise ValueError("Could not find Bollinger Bands columns in pandas_ta output")

    df["BB_upper"] = bbands[upper_col]
    df["BB_lower"] = bbands[lower_col]

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
        df = fetch_price_data(ticker)
        df = compute_technical_indicators(df)
        latest = df.iloc[-1]

        summary = {
            "RSI": latest["RSI"],
            "RSI_signal": interpret_rsi(latest["RSI"]),
            "MACD": latest["MACD"],
            "MACD_signal": latest["MACD_signal"],
            "MACD_trend": interpret_macd(latest["MACD"], latest["MACD_signal"]),
            "EMA_20": latest["EMA_20"],
            "EMA_50": latest["EMA_50"],
            "SMA_20": latest["SMA_20"],
            "SMA_200": latest["SMA_200"],
            "BB_upper": latest["BB_upper"],
            "BB_lower": latest["BB_lower"],
            "Close": latest["Close"],
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
