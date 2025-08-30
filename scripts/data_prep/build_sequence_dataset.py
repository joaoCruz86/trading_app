# scripts/data_prep/build_sequence_dataset.py

"""
📦 build_sequence_dataset.py

Builds the final training dataset (`sequence_dataset.npz`) used for both entry and exit sequence models.

🔹 Features:
- Auto-generates the `training` collection (entry data) if it's missing:
    • Loads raw prices from MongoDB
    • Computes technical indicators + candlestick features
    • Labels entry points where future max return ≥ +6% within 15 days
    • Saves processed rows to MongoDB as `training`
- Loads exit sequences from `exit_training`
- Applies filters, normalization, and sliding window logic
- Saves everything into `data/sequence_dataset.npz`

⚙️ Output:
- X_entry, y_entry, tickers_entry
- X_exit, y_exit, tickers_exit
- metadata (timestamp, filters, window config)

"""

import os
import numpy as np
import pandas as pd
from core.db import db
from core.technical_analysis import compute_technical_indicators

# --- Config ---
WINDOW_LEN = 40
HORIZON = 15  # looking further ahead for long trend entries
FEATURE_COLS = None
OUTPUT_PATH = "data/sequence_dataset.npz"
USE_CANDLE_FILTER = True
MIN_BODY_TO_RANGE = 0.2
USE_VOLUME_FILTER = True
ENTRY_THRESHOLD = 0.06  # +6% future return → entry (long trend)

def compute_candle_strength(df: pd.DataFrame) -> pd.DataFrame:
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["range"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body_to_range"] = df["body"] / df["range"]
    df.drop(columns=["body", "range"], inplace=True, errors="ignore")
    return df

def generate_entry_dataset_from_prices():
    print("🛠️  Generating entry dataset from 'prices' collection...")
    tickers = db["prices"].distinct("ticker")
    all_data = []

    for ticker in tickers:
        raw = pd.DataFrame(list(db["prices"].find({"ticker": ticker})))
        if raw.empty:
            continue

        raw.rename(columns={col: col.capitalize() for col in raw.columns}, inplace=True)
        raw.drop(columns=["_id"], errors="ignore", inplace=True)
        raw["date"] = pd.to_datetime(raw["Date"])
        raw.set_index("date", inplace=True)
        raw.sort_index(inplace=True)
        raw.drop(columns=["Date", "Indicators"], errors="ignore", inplace=True)

        raw = compute_technical_indicators(raw)
        raw = compute_candle_strength(raw)

        # --- Label entry target using MAX future close within horizon ---
        raw["future_max"] = raw["Close"].shift(-1).rolling(window=HORIZON).max()
        raw["future_return"] = (raw["future_max"] - raw["Close"]) / raw["Close"]
        raw["target"] = (raw["future_return"] >= ENTRY_THRESHOLD).astype(int)

        raw["ticker"] = ticker
        raw["date"] = raw.index

        # Drop unused and invalid rows
        raw.drop(columns=["future_max", "future_return"], errors="ignore", inplace=True)
        raw = raw.dropna(subset=["target"])

        if not set(raw["target"].unique()).issubset({0, 1}):
            print(f"❌ Invalid targets for {ticker}")
            continue

        raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        raw.fillna(0, inplace=True)

        all_data.append(raw.reset_index(drop=True))

    if not all_data:
        print("❌ No entry dataset could be generated.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_records = final_df.to_dict("records")
    for r in final_records:
        r.pop("_id", None)

    db["training"].drop()
    db["training"].insert_many(final_records)
    print(f"✅ Inserted {len(final_df)} rows into 'training' collection.")

def build_sequences(df: pd.DataFrame, ticker_col="ticker", date_col="date", target_col="target"):
    X, y, tickers_out = [], [], []

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col])
    tickers = df[ticker_col].unique()

    for ticker in tickers:
        df_t = df[df[ticker_col] == ticker].copy()

        # --- Feature filtering ---
        if USE_CANDLE_FILTER and "body_to_range" in df_t.columns:
            df_t = df_t[df_t["body_to_range"] >= MIN_BODY_TO_RANGE]

        if USE_VOLUME_FILTER and "Volume" in df_t.columns:
            df_t["vol_ma"] = df_t["Volume"].rolling(5).mean()
            df_t = df_t[df_t["Volume"] > df_t["vol_ma"]]

        if len(df_t) < WINDOW_LEN + HORIZON:
            print(f"⚠️ Skipping {ticker} — not enough data after filters.")
            continue

        features = df_t.drop(columns=[ticker_col, date_col, target_col], errors="ignore")
        features = features.select_dtypes(include=[np.number])
        if FEATURE_COLS:
            features = features[[col for col in FEATURE_COLS if col in features.columns]]

        features = (features - features.min()) / (features.max() - features.min())
        features.replace([np.inf, -np.inf], np.nan, inplace=True)

        nan_cols = features.columns[features.isna().any()].tolist()
        if nan_cols:
            print(f"⚠️ Ticker '{ticker}' has NaNs in columns: {nan_cols}")
        features.fillna(0, inplace=True)

        values = features.to_numpy(dtype=np.float32)
        targets = df_t[target_col].to_numpy()

        for i in range(len(df_t) - WINDOW_LEN - HORIZON):
            X_seq = values[i:i + WINDOW_LEN]
            label_index = i + WINDOW_LEN + HORIZON - 1
            if label_index >= len(targets):
                continue

            if np.isnan(X_seq).any():
                continue

            if np.all(X_seq == 0):
                continue

            X.append(X_seq)
            y.append(targets[label_index])
            tickers_out.append(ticker)

    return np.array(X, dtype=np.float32), np.array(y), np.array(tickers_out)

def main():
    print("⚡ Building sequence dataset...")

    # --- ENTRY: Auto-generate if needed ---
    df_entry = pd.DataFrame(list(db["training"].find()))
    if df_entry.empty:
        generate_entry_dataset_from_prices()
        df_entry = pd.DataFrame(list(db["training"].find()))

    df_entry.drop(columns=["_id"], errors="ignore", inplace=True)
    X_entry, y_entry, tickers_entry = build_sequences(df_entry, target_col="target")

    # --- EXIT ---
    df_exit = pd.DataFrame(list(db["exit_training"].find()))
    df_exit.drop(columns=["_id"], errors="ignore", inplace=True)
    X_exit, y_exit, tickers_exit = build_sequences(df_exit, target_col="exit_target")

    # --- SAVE ---
    os.makedirs("data", exist_ok=True)
    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "entry_count": len(X_entry),
        "exit_count": len(X_exit),
        "window_len": WINDOW_LEN,
        "horizon": HORIZON,
        "filters": {
            "candle_filter": USE_CANDLE_FILTER,
            "min_body_to_range": MIN_BODY_TO_RANGE,
            "volume_filter": USE_VOLUME_FILTER,
        },
    }

    np.savez_compressed(
        OUTPUT_PATH,
        X_entry=X_entry, y_entry=y_entry,
        X_exit=X_exit, y_exit=y_exit,
        tickers_entry=tickers_entry,
        tickers_exit=tickers_exit,
        metadata=metadata
    )

    print(f"\n✅ Saved to {OUTPUT_PATH}")
    print(f"   Entry → X: {X_entry.shape}, y: {y_entry.shape}")
    print(f"   Exit  → X: {X_exit.shape}, y: {y_exit.shape}")
    print("   Metadata:", metadata)


if __name__ == "__main__":
    main()
