import os
import pandas as pd
from core.db import db  # ✅ Correctly import the MongoDB database object
from core.technical_analysis import compute_technical_indicators
from core.macro_filter import is_macro_environment_favorable, get_macro_for_country

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def build_dataset(ticker: str, horizon: int = 20, target_return: float = 0.05):
    """
    Build dataset for one ticker:
      horizon: number of days ahead for target label
      target_return: percentage move for positive label (e.g. +5%)
    """

    # --- 1. Load prices ---
    cursor = db["prices"].find({"ticker": ticker}).sort("date", 1)
    prices = pd.DataFrame(list(cursor))

    if prices.empty:
        raise ValueError(f"No price data found for {ticker}")

    # Normalize column names
    prices.rename(columns={col: col.capitalize() for col in prices.columns}, inplace=True)

    # ✅ Fix datetime index and clean up
    prices.drop(columns=["_id"], errors="ignore", inplace=True)
    prices["date"] = pd.to_datetime(prices["Date"])
    prices.set_index("date", inplace=True)
    prices.drop(columns=["Date", "Indicators"], errors="ignore", inplace=True)

    prices.sort_index(inplace=True)

    # 🔎 Debug before indicators
    print(f"\n🔎 Pre-indicator debug for {ticker}")
    print("Index monotonic?", prices.index.is_monotonic_increasing)
    print("Any duplicates?", prices.index.duplicated().sum())
    print(f"Rows: {len(prices)} Cols: {list(prices.columns)}")

    # --- 2. Compute technical indicators ---
    prices = prices.asfreq("D").ffill()  # enforce daily frequency and fill gaps
    prices = compute_technical_indicators(prices)

    # 🔎 Debug after indicators
    print(f"Post-indicators index monotonic? {prices.index.is_monotonic_increasing}")

    # --- 3. Merge fundamentals ---
    fundamentals = pd.DataFrame(list(db["fundamentals"].find({"ticker": ticker})))
    if not fundamentals.empty:
        if "period" not in fundamentals.columns:
            print(f"ℹ️ No 'period' in fundamentals for {ticker} — skipping fundamentals merge.")
        else:
            fundamentals["period"] = pd.to_datetime(fundamentals["period"], errors="coerce")
            fundamentals = fundamentals.dropna(subset=["period"])
            fundamentals.set_index("period", inplace=True)
            fundamentals = fundamentals[~fundamentals.index.duplicated(keep="last")]
            fundamentals.sort_index(inplace=True)

            fundamentals.drop(columns=["_id"], errors="ignore", inplace=True)

            print(
                f"Fundamentals pre-resample — rows={len(fundamentals)}, "
                f"monotonic={fundamentals.index.is_monotonic_increasing}, "
                f"dups={fundamentals.index.duplicated().sum()}"
            )

            try:
                fundamentals = fundamentals.asfreq("D").ffill()
                prices = prices.join(fundamentals, how="left")
                prices = prices.sort_index()
            except Exception as e:
                print(f"⚠️ Fundamentals merge failed for {ticker}: {e}")
                print("↪️ Proceeding without fundamentals for this ticker.")

    # --- 4. Add macro features ---
    macro = get_macro_for_country("USA")  # 🧪 Using mock data
    prices["Macro_OK"] = is_macro_environment_favorable(macro)

    # --- 5. Create target label ---
    prices["future_close"] = prices["Close"].shift(-horizon)
    prices["future_return"] = (prices["future_close"] - prices["Close"]) / prices["Close"]
    prices["target"] = (prices["future_return"] >= target_return).astype(int)

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Target creation broke monotonicity!")

    # --- 6. Drop unnecessary or problematic columns ---
    prices.drop(columns=[
        "fundamentals", "financials", "ratios", "balanceSheet",
        "cashflow", "earnings", "source"
    ], errors="ignore", inplace=True)

    # --- 7. Final cleaning: only drop rows with missing target values ---
    columns_to_check = ["future_close", "future_return", "target"]
    print("📉 Null value counts before dropna():")
    print(prices.isna().sum().sort_values(ascending=False).head(10))

    print("\n📋 Last few rows before dropna():")
    print(prices.tail(5))

    dataset = prices.dropna(subset=columns_to_check).reset_index(drop=True)
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
            if isinstance(ds, pd.DataFrame) and not ds.empty:
                all_data.append(ds)
                print(f"✅ Added {ticker}, rows={len(ds)}")
            else:
                print(f"⚠️ {ticker} produced empty dataset after cleaning.")
        except Exception as e:
            print(f"⚠️ Skipped {ticker}: {e}")

    if all_data:
        final = pd.concat(all_data, ignore_index=True)
        print(f"\n📂 Final dataset rows: {len(final)}")

        # --- Save to CSV ---
        out_path = os.path.join(DATA_DIR, "training_dataset.csv")
        final.to_csv(out_path, index=False)
        print(f"📂 Saved training dataset to {out_path}")

        # --- Save to MongoDB ---
        final = final.fillna(value=pd.NA)  # clean regular NaNs
        final = final.astype(object).where(pd.notnull(final), None)  # clean NaT and others
        records = final.to_dict("records")
        if records:
            training = db["training"]
            training.drop()
            training.insert_many(records)
            print(f"📊 Inserted {len(records)} rows into MongoDB collection 'training'.")
            print("📁 Collections now in DB:", db.list_collection_names())
            print("📊 'training' document count:", training.count_documents({}))
        else:
            print("❌ Final dataset is empty. Nothing inserted into MongoDB.")
    else:
        print("❌ No datasets were built.")
