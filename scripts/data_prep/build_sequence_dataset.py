# scripts/data_prep/build_sequence_dataset.py
"""
Build Sequence Dataset for Time-Aware Models
--------------------------------------------
This script transforms your tabular daily data into rolling windows
so that GRU/LSTM/Transformers can learn temporal patterns.

Each sample = last N days of features for a stock.
Label = forward return (e.g. price change over next horizon days).

Output: sequence_dataset.npz
- X_entry, y_entry  → for entry model
- X_exit, y_exit    → for exit model
"""

import os
import numpy as np
import pandas as pd
from core.db import db

# --- Config ---
WINDOW_LEN = 60   # number of past days used as input
HORIZON = 5       # predict next 5 days return
FEATURE_COLS = None  # if None, will auto-select numeric features
OUTPUT_PATH = "data/sequence_dataset.npz"


def build_sequences(df: pd.DataFrame, ticker_col="ticker", date_col="date", target_col="target"):
    """
    Convert daily tabular data into rolling windows for sequence models.
    """
    X, y = [], []

    # ensure sorting
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col])

    tickers = df[ticker_col].unique()

    for ticker in tickers:
        df_t = df[df[ticker_col] == ticker].copy()

        # auto-select features if not set
        features = df_t.drop(columns=[ticker_col, date_col, target_col], errors="ignore")
        if FEATURE_COLS:
            features = features[FEATURE_COLS]

        values = features.to_numpy()
        targets = df_t[target_col].to_numpy()

        for i in range(len(df_t) - WINDOW_LEN - HORIZON):
            X.append(values[i:i+WINDOW_LEN])   # last N days
            # label: did the stock go up after horizon days?
            future_return = targets[i+WINDOW_LEN+HORIZON-1]
            y.append(future_return)

    return np.array(X), np.array(y)


def main():
    print("⚡ Building sequence dataset...")

    # Load from DB
    df = pd.DataFrame(list(db["training"].find()))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    # Ensure target is binary
    assert set(df["target"].unique()).issubset({0, 1}), "Target must be binary!"

    # Build sequences
    X, y = build_sequences(df)

    # For simplicity, use same labels for entry & exit now
    X_entry, y_entry = X, y
    X_exit, y_exit = X, y

    # Save dataset
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, 
                        X_entry=X_entry, y_entry=y_entry,
                        X_exit=X_exit, y_exit=y_exit)

    print(f"✅ Sequence dataset saved to {OUTPUT_PATH}")
    print(f"   Shapes → X: {X.shape}, y: {y.shape}")


if __name__ == "__main__":
    main()
