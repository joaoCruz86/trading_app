import pandas as pd
from datetime import datetime, timezone
import yfinance as yf
import pandas_ta as ta
from pymongo import UpdateOne
from core.db import db

TICKERS_CSV = "tickers.csv"
YEARS_BACK = "3y"  # 3 years of data

# ------------------------
# Compute Technical Indicators
# ------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Momentum
    df["RSI14"] = ta.rsi(df["Close"], length=14)

    # Simple Moving Averages
    df["SMA20"] = ta.sma(df["Close"], length=20)
    df["SMA50"] = ta.sma(df["Close"], length=50)

    # Exponential Moving Averages
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)

    # MACD
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"] = macd["MACDh_12_26_9"]

    # ATR
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Bollinger Bands
    bbands = ta.bbands(df["Close"], length=20, std=2)
    if bbands is not None:
        df["BB_upper"] = bbands["BBU_20_2.0"]
        df["BB_middle"] = bbands["BBM_20_2.0"]
        df["BB_lower"] = bbands["BBL_20_2.0"]

    return df

# ------------------------
# MongoDB Upsert
# ------------------------
def upsert_prices(ticker: str, df: pd.DataFrame):
    ops = []
    for dt, row in df.iterrows():
        date_utc = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

        # Scalar extraction helper
        def scalar(val):
            return val.item() if hasattr(val, "item") else val

        open_val = scalar(row["Open"])
        high_val = scalar(row["High"])
        low_val = scalar(row["Low"])
        close_val = scalar(row["Close"])
        volume_val = scalar(row["Volume"])

        # Indicators
        indicators_dict = {}
        for k in [
            "RSI14", "SMA20", "SMA50", "EMA20", "EMA50",
            "MACD", "MACD_signal", "MACD_hist",
            "ATR14", "BB_upper", "BB_middle", "BB_lower"
        ]:
            if k in row:
                val = scalar(row[k])
                indicators_dict[k] = float(val) if pd.notna(val) else None

        doc = {
            "ticker": ticker,
            "date": date_utc,
            "open": float(open_val) if pd.notna(open_val) else None,
            "high": float(high_val) if pd.notna(high_val) else None,
            "low": float(low_val) if pd.notna(low_val) else None,
            "close": float(close_val) if pd.notna(close_val) else None,
            "volume": int(volume_val) if pd.notna(volume_val) else None,
            "indicators": indicators_dict,
        }

        ops.append(
            UpdateOne(
                {"ticker": ticker, "date": date_utc},
                {"$set": doc},
                upsert=True
            )
        )

    if ops:
        res = db.prices.bulk_write(ops, ordered=False)
        print(f"[{ticker}] upserted={res.upserted_count}, modified={res.modified_count}")

# ------------------------
# Main
# ------------------------
def main():
    tickers = pd.read_csv(TICKERS_CSV)["Ticker"].tolist()

    for ticker in tickers:
        print(f"Fetching {YEARS_BACK} of data for {ticker}...")
        df = yf.download(ticker, period=YEARS_BACK, interval="1d", auto_adjust=False)
        if df.empty:
            print(f"No data found for {ticker}")
            continue

        df = compute_indicators(df)
        upsert_prices(ticker, df)

    print("✅ Backfill complete.")

if __name__ == "__main__":
    main()
