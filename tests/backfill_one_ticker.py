import pandas as pd
from datetime import datetime, timezone
import yfinance as yf
import pandas_ta as ta
from pymongo import UpdateOne
from core.db import db

TICKER = "AAPL"       # you can change this
DAYS_BACK = "90d"     # 3 months of data

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["SMA20"] = ta.sma(df["Close"], length=20)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    return df

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def upsert_prices(ticker: str, df: pd.DataFrame):
    ops = []
    for dt, row in df.iterrows():
        date_utc = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        doc = {
            "ticker": ticker,
            "date": date_utc,
            "open": safe_float(row.get("Open")),
            "high": safe_float(row.get("High")),
            "low": safe_float(row.get("Low")),
            "close": safe_float(row.get("Close")),
            "volume": safe_int(row.get("Volume")),
            "indicators": {
                "RSI14": safe_float(row.get("RSI14")),
                "SMA20": safe_float(row.get("SMA20")),
                "SMA50": safe_float(row.get("SMA50")),
            },
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

def main():
    print(f"Fetching {DAYS_BACK} of data for {TICKER}...")
    df = yf.download(TICKER, period=DAYS_BACK, interval="1d", auto_adjust=False)
    if df.empty:
        print("No data found!")
        return
    df = compute_indicators(df)
    upsert_prices(TICKER, df)
    print("Done ✅")

if __name__ == "__main__":
    main()
