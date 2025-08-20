"""
Build training dataset for entry signal prediction.

This script:
- Loads historical price data and computes technical indicators.
- Optionally merges company fundamentals and macroeconomic indicators.
- Creates a binary target label based on future returns.
- Saves the dataset to both CSV and MongoDB ('training' collection).

Run this script before training any ML models.
"""

import os
import pandas as pd
from core.db import db
from core.technical_analysis import compute_technical_indicators
from core.macro_filter import is_macro_environment_favorable, get_macro_for_country

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def build_dataset(ticker: str, horizon: int = 20, target_return: float = 0.05):
    """
    Build dataset for one ticker.
    
    Parameters:
        - horizon: days ahead to calculate future return
        - target_return: threshold for labeling a 'buy' signal (e.g. 0.05 = +5%)
    """

    # --- 1. Load prices ---
    cursor = db["prices"].find({"ticker": ticker}).sort("date", 1)
    prices = pd.DataFrame(list(cursor))
    if prices.empty:
        raise ValueError(f"No price data found for {ticker}")

    prices.rename(columns={col: col.capitalize() for col in prices.columns}, inplace=True)
    prices.drop(columns=["_id"], errors="ignore", inplace=True)
    prices["date"] = pd.to_datetime(prices["Date"])
    prices.set_index("date", inplace=True)
    prices.drop(columns=["Date", "Indicators"], errors="ignore", inplace=True)
    prices.sort_index(inplace=True)

    # 🔎 Debug pre-indicators
    print(f"\n🔎 Pre-indicator debug for {ticker}")
    print("Index monotonic?", prices.index.is_monotonic_increasing)
    print("Any duplicates?", prices.index.duplicated().sum())

    # --- 2. Compute technical indicators ---
    prices = prices.asfreq("D").ffill()
    prices = compute_technical_indicators(prices)
    print(f"Post-indicators index monotonic? {prices.index.is_monotonic_increasing}")

    # --- 3. Merge fundamentals (optional) ---
    fundamentals = pd.DataFrame(list(db["fundamentals"].find({"ticker": ticker})))
    if not fundamentals.empty and "period" in fundamentals.columns:
        fundamentals["period"] = pd.to_datetime(fundamentals["period"], errors="coerce")
        fundamentals.dropna(subset=["period"], inplace=True)
        fundamentals.set_index("period", inplace=True)
        fundamentals = fundamentals[~fundamentals.index.duplicated(keep="last")].sort_index()
        fundamentals.drop(columns=["_id"], errors="ignore", inplace=True)

        try:
            fundamentals = fundamentals.asfreq("D").ffill()
            prices = prices.join(fundamentals, how="left").sort_index()
        except Exception as e:
            print(f"⚠️ Fundamentals merge failed for {ticker}: {e}")
    else:
        print(f"ℹ️ No 'period' field in fundamentals for {ticker} — skipping fundamentals merge.")

    # --- 4. Add macro filter ---
    macro = get_macro_for_country("USA")
    prices["Macro_OK"] = is_macro_environment_favorable(macro)

    # --- 5. Create target label ---
    prices["future_close"] = prices["Close"].shift(-horizon)
    prices["future_return"] = (prices["future_close"] - prices["Close"]) / prices["Close"]
    prices["target"] = (prices["future_return"] >= target_return).astype(int)

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Target creation broke index monotonicity!")

    # --- 6. Drop unnecessary columns ---
    prices.drop(columns=[
        "fundamentals", "financials", "ratios", "balanceSheet",
        "cashflow", "earnings", "source"
    ], errors="ignore", inplace=True)

    # --- 7. Final clean ---
    print("📉 Nulls before dropna():")
    print(prices.isna().sum().sort_values(ascending=False).head(10))
    dataset = prices.dropna(subset=["future_close", "future_return", "target"]).reset_index()  # ✅ keeps 'date'
    dataset["ticker"] = ticker

    print(f"📊 Built dataset shape for {ticker}: {dataset.shape}")
    return dataset


if __name__ == "__main__":
    tickers = db["prices"].distinct("ticker")
    print(f"🔍 Found tickers: {tickers}")
    all_data = []

    for ticker in tickers:
        try:
            ds = build_dataset(ticker)
            if not ds.empty:
                all_data.append(ds)
                print(f"✅ Added {ticker}, rows={len(ds)}")
            else:
                print(f"⚠️ {ticker} produced empty dataset.")
        except Exception as e:
            print(f"⚠️ Skipped {ticker}: {e}")

    if all_data:
        final = pd.concat(all_data, ignore_index=True)
        print(f"\n📂 Final dataset rows: {len(final)}")

        # Save to CSV
        out_path = os.path.join(DATA_DIR, "training_dataset.csv")
        final.to_csv(out_path, index=False)
        print(f"📁 Saved to {out_path}")

        # Save to MongoDB
        final = final.fillna(value=pd.NA)
        final = final.astype(object).where(pd.notnull(final), None)
        records = final.to_dict("records")

        training = db["training"]
        training.drop()
        training.insert_many(records)
        print(f"📊 Inserted {len(records)} rows into MongoDB collection 'training'.")
        print("📁 Collections now in DB:", db.list_collection_names())
        print("📊 'training' document count:", training.count_documents({}))

    else:
        print("❌ No datasets were built.")

    from core.db import _client
    _client.close()


