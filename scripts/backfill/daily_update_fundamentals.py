import pandas as pd
import yfinance as yf
from core.db import db
from scripts.backfill.backfill_fundamentals import build_quarter_docs  # reuse

TICKERS_CSV = "tickers.csv"

def update_one(ticker: str):
    tkr = yf.Ticker(ticker)
    docs = build_quarter_docs(ticker, tkr)
    if not docs:
        print(f"[{ticker}] No fundamentals docs.")
        return
    existing = set(d["period"] for d in db.fundamentals.find(
        {"ticker": ticker}, {"_id":0,"period":1}
    ))
    new_docs = [d for d in docs if d["period"] not in existing]
    if not new_docs:
        print(f"[{ticker}] No new quarters.")
        return
    from pymongo import UpdateOne
    ops = [UpdateOne({"ticker": d["ticker"], "period": d["period"]}, {"$set": d}, upsert=True)
           for d in new_docs]
    res = db.fundamentals.bulk_write(ops, ordered=False)
    print(f"[{ticker}] fundamentals upserted={res.upserted_count}, modified={res.modified_count}")

def main():
    tickers = pd.read_csv(TICKERS_CSV)["Ticker"].dropna().tolist()
    for t in tickers:
        try: update_one(t)
        except Exception as e: print(f"❌ {t}: {e}")
    print("✅ Daily fundamentals update complete.")

if __name__ == "__main__":
    main()
