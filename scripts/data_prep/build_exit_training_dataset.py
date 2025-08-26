"""
Build a training dataset to detect weakening signals (exit points) based on technical indicators.
This script pulls historical stock price data, applies technical indicators, and labels data points
where the price drops by a specified percentage over a short horizon.

The labels (1 = exit signal, 0 = hold) will be used to train the Exit Signal Model.
It generates:
- A MongoDB collection for sequence models (with 'date')
"""

import os
import pandas as pd
from core.db import db
from core.technical_analysis import compute_technical_indicators
from core.macro_filter import is_macro_environment_favorable, get_macro_for_country

def build_exit_dataset(ticker: str, horizon: int = 5, drop_threshold: float = -0.02):
    """
    Build exit signal dataset for a single ticker.
      horizon: number of days to look ahead for drop
      drop_threshold: % drop to label an exit signal (e.g., -2%)
    """
    cursor = db["prices"].find({"ticker": ticker}).sort("date", 1)
    prices = pd.DataFrame(list(cursor))

    if prices.empty:
        raise ValueError(f"No price data found for {ticker}")

    # Normalize and clean columns
    prices.rename(columns={col: col.capitalize() for col in prices.columns}, inplace=True)
    prices.drop(columns=["_id"], errors="ignore", inplace=True)
    prices["date"] = pd.to_datetime(prices["Date"])
    prices.set_index("date", inplace=True)
    prices.drop(columns=["Date", "Indicators"], errors="ignore", inplace=True)
    prices.sort_index(inplace=True)

    # Apply indicators
    prices = prices.asfreq("D").ffill()
    prices = compute_technical_indicators(prices)

    # Merge fundamentals (optional)
    fundamentals = pd.DataFrame(list(db["fundamentals"].find({"ticker": ticker})))
    if not fundamentals.empty and "period" in fundamentals.columns:
        fundamentals["period"] = pd.to_datetime(fundamentals["period"], errors="coerce")
        fundamentals.dropna(subset=["period"], inplace=True)
        fundamentals.set_index("period", inplace=True)
        fundamentals = fundamentals[~fundamentals.index.duplicated(keep="last")].sort_index()
        fundamentals = fundamentals.asfreq("D").ffill()
        prices = prices.join(fundamentals, how="left").sort_index()

    # Add macro environment (optional)
    macro = get_macro_for_country("USA")
    prices["Macro_OK"] = is_macro_environment_favorable(macro)

    # Create exit label
    prices["future_close"] = prices["Close"].shift(-horizon)
    prices["future_return"] = (prices["future_close"] - prices["Close"]) / prices["Close"]
    prices["exit_target"] = (prices["future_return"] <= drop_threshold).astype(int)

    # Drop unused
    prices.drop(columns=[
        "fundamentals", "financials", "ratios", "balanceSheet",
        "cashflow", "earnings", "source"
    ], errors="ignore", inplace=True)

    # Keep 'date' column for sequence models
    prices["date"] = prices.index

    # Clean rows
    columns_to_check = ["future_close", "future_return", "exit_target"]
    dataset = prices.dropna(subset=columns_to_check).reset_index(drop=True)
    dataset["ticker"] = ticker

    print(f"📉 Built exit dataset for {ticker} — rows: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    tickers = db["prices"].distinct("ticker")
    print(f"🔍 Found tickers: {tickers}")
    all_data = []

    for ticker in tickers:
        try:
            ds = build_exit_dataset(ticker)
            if not ds.empty:
                all_data.append(ds)
        except Exception as e:
            print(f"⚠️ Skipped {ticker}: {e}")

    if all_data:
        final = pd.concat(all_data, ignore_index=True)

        # Save to MongoDB for sequence models — keep 'date'
        final_mongo = final.fillna(value=pd.NA).astype(object).where(pd.notnull(final), None)
        records = final_mongo.to_dict("records")
        for record in records:
            record.pop("_id", None)

        db_exit = db["exit_training"]
        db_exit.drop()  # Clear old data
        db_exit.insert_many(records)

        print(f"✅ Inserted {len(final_mongo)} rows into MongoDB collection 'exit_training'")
    else:
        print("❌ No exit datasets were built.")