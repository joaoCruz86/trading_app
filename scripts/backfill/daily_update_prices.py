import pandas as pd
from datetime import datetime, timedelta, timezone
import yfinance as yf
from pymongo import UpdateOne
from core.db import db
from scripts.backfill.backfill_prices import compute_indicators  # reuse

TICKERS_CSV = "tickers.csv"

def scalar(v): return v.item() if hasattr(v, "item") else v
def f(x): x = scalar(x); return float(x) if pd.notna(x) else None
def i(x): x = scalar(x); return int(x) if pd.notna(x) else None

def update_one(ticker: str):
    last = db.prices.find({"ticker": ticker}, {"date": 1}).sort("date", -1).limit(1)
    last_date = next(iter(last), {}).get("date")
    if not last_date:
        print(f"[{ticker}] No history found. Run backfill first.")
        return
    start = (last_date + timedelta(days=1)).date().isoformat()
    print(f"[{ticker}] Updating from {start}...")
    df = yf.download(ticker, start=start, interval="1d", auto_adjust=False)
    if df.empty:
        print(f"[{ticker}] No new rows.")
        return
    df = compute_indicators(df)
    ops = []
    for dt, row in df.iterrows():
        date_utc = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        inds = {}
        for k in ["RSI14","SMA20","SMA50","EMA20","EMA50","MACD","MACD_signal","MACD_hist",
                  "ATR14","BB_upper","BB_middle","BB_lower"]:
            if k in row: inds[k] = f(row[k])
        doc = {
            "ticker": ticker, "date": date_utc,
            "open": f(row["Open"]), "high": f(row["High"]),
            "low": f(row["Low"]), "close": f(row["Close"]),
            "volume": i(row["Volume"]), "indicators": inds,
        }
        ops.append(UpdateOne({"ticker": ticker, "date": date_utc}, {"$set": doc}, upsert=True))
    if ops:
        res = db.prices.bulk_write(ops, ordered=False)
        print(f"[{ticker}] upserted={res.upserted_count}, modified={res.modified_count}")

def main():
    tickers = pd.read_csv(TICKERS_CSV)["Ticker"].dropna().tolist()
    for t in tickers:
        try: update_one(t)
        except Exception as e: print(f"❌ {t}: {e}")
    print("✅ Daily price update complete.")

if __name__ == "__main__":
    main()
