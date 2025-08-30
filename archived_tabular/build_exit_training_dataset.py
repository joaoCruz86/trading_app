# scripts/build_exit_training_dataset.py

import os
import pandas as pd
import numpy as np
from core.db import db
from core.technical_analysis import compute_technical_indicators
from core.macro_filter import is_macro_environment_favorable, get_macro_for_country

db_training = db["training"]
db_training.drop()  # ⚠️ Clears existing data (safe if it's regenerated fresh)



def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = abs(df["Close"] - df["Open"])
    df["range"] = df["High"] - df["Low"]
    df["body_to_range"] = df["body"] / df["range"].replace(0, 1e-6)

    def classify_candle(row):
        if row["body_to_range"] >= 0.6:
            return "strong"
        elif row["body_to_range"] >= 0.2:
            return "medium"
        else:
            return "weak"

    df["candle_type"] = df.apply(classify_candle, axis=1)
    return df


def build_exit_dataset(ticker: str, horizon: int = 5, drop_threshold: float = -0.02):
    cursor = db["prices"].find({"ticker": ticker})
    prices = pd.DataFrame(list(cursor)).sort_values("date")

    if prices.empty:
        raise ValueError(f"No price data found for {ticker}")

    # --- Normalize and clean ---
    prices.rename(columns={col: col.capitalize() for col in prices.columns}, inplace=True)
    prices.drop(columns=["_id"], errors="ignore", inplace=True)
    prices["date"] = pd.to_datetime(prices["Date"])
    prices.set_index("date", inplace=True)
    prices.drop(columns=["Date", "Indicators"], errors="ignore", inplace=True)
    prices.sort_index(inplace=True)

    prices = prices.asfreq("D").ffill()
    prices = compute_technical_indicators(prices)
    prices = add_candlestick_features(prices)

    # --- Fundamentals (optional) ---
    fundamentals = pd.DataFrame(list(db["fundamentals"].find({"ticker": ticker})))
    if not fundamentals.empty and "period" in fundamentals.columns:
        fundamentals["period"] = pd.to_datetime(fundamentals["period"], errors="coerce")
        fundamentals.dropna(subset=["period"], inplace=True)
        fundamentals.set_index("period", inplace=True)
        fundamentals = fundamentals[~fundamentals.index.duplicated(keep="last")].sort_index()
        fundamentals = fundamentals.asfreq("D").ffill()
        prices = prices.join(fundamentals, how="left").sort_index()

    # --- Macro (optional) ---
    macro = get_macro_for_country("USA")
    prices["Macro_OK"] = is_macro_environment_favorable(macro)

    # --- Exit label: price drop over N days ---
    prices["future_close"] = prices["Close"].shift(-horizon)
    prices["future_return"] = (prices["future_close"] - prices["Close"]) / prices["Close"]
    prices["exit_target"] = (prices["future_return"] <= drop_threshold).astype(int)

    # --- Drop unused ---
    prices.drop(columns=[
        "fundamentals", "financials", "ratios", "balanceSheet",
        "cashflow", "earnings", "source"
    ], errors="ignore", inplace=True)

    prices["date"] = prices.index

    # --- Clean rows ---
    required_cols = ["future_close", "future_return", "exit_target"]
    dataset = prices.dropna(subset=required_cols).reset_index(drop=True)
    dataset["ticker"] = ticker

    # --- Replace NaNs/Infs ---
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_cols = dataset.columns[dataset.isna().any()].tolist()
    if nan_cols:
        print(f"⚠️ Ticker '{ticker}' has NaNs in: {nan_cols}")
    dataset.fillna(0, inplace=True)

    # --- Validate exit target ---
    if not set(dataset["exit_target"].unique()).issubset({0, 1}):
        raise ValueError(f"❌ Invalid exit_target values for {ticker}: {dataset['exit_target'].unique()}")

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
        db_exit.drop()
        db_exit.insert_many(records)

        print(f"✅ Inserted {len(final_mongo)} rows into MongoDB collection 'exit_training'")
    else:
        print("❌ No exit datasets were built.")
