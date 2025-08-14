import pandas as pd
from core.db import db

TICKERS_CSV = "tickers.csv"

def rng(coll, ticker, date_field):
    cur = list(db[coll].find({"ticker": ticker}, {"_id":0, date_field:1}).sort(date_field, 1).limit(1))
    lo = cur[0][date_field] if cur else None
    cur = list(db[coll].find({"ticker": ticker}, {"_id":0, date_field:1}).sort(date_field, -1).limit(1))
    hi = cur[0][date_field] if cur else None
    return lo, hi

def main():
    tickers = pd.read_csv(TICKERS_CSV)["Ticker"].dropna().tolist()
    for t in tickers:
        pc = db.prices.count_documents({"ticker": t})
        fc = db.fundamentals.count_documents({"ticker": t})
        p_lo, p_hi = rng("prices", t, "date")
        f_lo, f_hi = rng("fundamentals", t, "period")
        print(f"{t}: prices={pc} [{p_lo} → {p_hi}]  fundamentals={fc} [{f_lo} → {f_hi}]")

if __name__ == "__main__":
    main()
