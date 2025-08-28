# scripts/build_training_dataset.py

import os
import numpy as np
import pandas as pd
from core.db import db

# --- Config ---
WINDOW_LEN = 40          # number of past days used as input
HORIZON = 5              # prediction horizon (used only for label positioning)
FEATURE_COLS = None      # Set to a list of feature names to use a subset
OUTPUT_PATH = "data/sequence_dataset.npz"
USE_CANDLE_FILTER = True
MIN_BODY_TO_RANGE = 0.2
USE_VOLUME_FILTER = True


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

        # --- Select numeric features ---
        features = df_t.drop(columns=[ticker_col, date_col, target_col], errors="ignore")
        features = features.select_dtypes(include=[np.number])
        if FEATURE_COLS:
            features = features[[col for col in FEATURE_COLS if col in features.columns]]

        # --- Normalize ---
        features = (features - features.min()) / (features.max() - features.min())
        features.replace([np.inf, -np.inf], np.nan, inplace=True)

        # --- Handle NaNs ---
        nan_cols = features.columns[features.isna().any()].tolist()
        if nan_cols:
            print(f"⚠️ Ticker '{ticker}' has NaNs in columns: {nan_cols}")
        features.fillna(0, inplace=True)

        # --- DEBUG ---
        print(f"\n📊 Ticker: {ticker}")
        print("✅ Feature columns:", features.columns.tolist())
        print("📈 Sample row:", features.head(1).to_dict())

        values = features.to_numpy(dtype=np.float32)
        targets = df_t[target_col].to_numpy()

        for i in range(len(df_t) - WINDOW_LEN - HORIZON):
            X_seq = values[i:i + WINDOW_LEN]
            label_index = i + WINDOW_LEN + HORIZON - 1
            if label_index >= len(targets):
                continue

            if np.isnan(X_seq).any():
                print(f"❌ Skipping sequence with NaNs for {ticker} at index {i}")
                continue

            if np.all(X_seq == 0):
                print(f"⚠️ Skipping all-zero sequence for {ticker} at index {i}")
                continue

            X.append(X_seq)
            y.append(targets[label_index])
            tickers_out.append(ticker)

    return np.array(X, dtype=np.float32), np.array(y), np.array(tickers_out)


def main():
    print("⚡ Building sequence dataset...")

    # --- Entry ---
    df_entry = pd.DataFrame(list(db["training"].find()))
    df_entry.drop(columns=["_id"], errors="ignore", inplace=True)

    if not {"ticker", "date", "target"}.issubset(df_entry.columns):
        raise ValueError("Missing required fields in 'training' collection.")
    assert set(df_entry["target"].unique()).issubset({0, 1}), "Entry target must be binary."

    X_entry, y_entry, tickers_entry = build_sequences(df_entry, target_col="target")

    # --- Exit ---
    df_exit = pd.DataFrame(list(db["exit_training"].find()))
    df_exit.drop(columns=["_id"], errors="ignore", inplace=True)

    if not {"ticker", "date", "exit_target"}.issubset(df_exit.columns):
        raise ValueError("Missing required fields in 'exit_training' collection.")
    assert set(df_exit["exit_target"].unique()).issubset({0, 1}), "Exit target must be binary."

    X_exit, y_exit, tickers_exit = build_sequences(df_exit, target_col="exit_target")

    # --- Save All ---
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(OUTPUT_PATH,
                        X_entry=X_entry, y_entry=y_entry,
                        X_exit=X_exit, y_exit=y_exit,
                        tickers_entry=tickers_entry,
                        tickers_exit=tickers_exit)

    print(f"\n✅ Saved to {OUTPUT_PATH}")
    print(f"   Entry → X: {X_entry.shape}, y: {y_entry.shape}")
    print(f"   Exit  → X: {X_exit.shape}, y: {y_exit.shape}")


if __name__ == "__main__":
    main()
