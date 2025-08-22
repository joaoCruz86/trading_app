"""
Build Sequence Dataset for AI Trading Models
============================================

This script prepares the time-series (sequence) dataset for Layer 2 of the trading system.

It pulls labeled tabular data from MongoDB:
- Entry data from the 'training' collection (produced by the tabular model prep pipeline)
- Exit data from the 'exit_training' collection (produced by the exit signal labeling pipeline)

Each sample is a rolling window of N days of stock features (e.g., technical indicators).
The label is the entry or exit signal N days into the future, used to train LSTM/GRU/Transformer models.

Key Features:
-------------
- Uses a fixed WINDOW_LEN for sequence length
- Uses a forward HORIZON to define the prediction offset
- Auto-handles multiple tickers and merges them into single input arrays
- Saves compressed `.npz` file with:
    - X_entry, y_entry       → entry signal sequences
    - X_exit, y_exit         → exit signal sequences
    - tickers_entry, tickers_exit

Output:
-------
- data/sequence_dataset.npz

Usage:
------
$ PYTHONPATH="." python scripts/data_prep/build_sequence_dataset.py
"""

import os
import numpy as np
import pandas as pd
from core.db import db

# --- Config ---
WINDOW_LEN = 60    # number of past days used as input
HORIZON = 5        # prediction horizon (used only for label positioning)
FEATURE_COLS = None
OUTPUT_PATH = "data/sequence_dataset.npz"

def build_sequences(df: pd.DataFrame, ticker_col="ticker", date_col="date", target_col="target"):
    X, y, tickers_out = [], [], []

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col])

    tickers = df[ticker_col].unique()

    for ticker in tickers:
        df_t = df[df[ticker_col] == ticker].copy()

        # Select features
        features = df_t.drop(columns=[ticker_col, date_col, target_col], errors="ignore")
        if FEATURE_COLS:
            features = features[FEATURE_COLS]

        values = features.to_numpy()
        targets = df_t[target_col].to_numpy()

        for i in range(len(df_t) - WINDOW_LEN - HORIZON):
            X.append(values[i:i + WINDOW_LEN])
            label_index = i + WINDOW_LEN + HORIZON - 1
            if label_index >= len(targets):
                continue
            y.append(targets[label_index])
            tickers_out.append(ticker)

    return np.array(X), np.array(y), np.array(tickers_out)

def main():
    print("⚡ Building sequence dataset...")

    # --- Entry Model Sequences ---
    df_entry = pd.DataFrame(list(db["training"].find()))
    df_entry.drop(columns=["_id"], errors="ignore", inplace=True)

    if not {"ticker", "date", "target"}.issubset(df_entry.columns):
        raise ValueError("Missing required fields in 'training' collection.")

    assert set(df_entry["target"].unique()).issubset({0, 1}), "Entry target must be binary."

    X_entry, y_entry, tickers_entry = build_sequences(df_entry, target_col="target")

    # --- Exit Model Sequences ---
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

    print(f"✅ Saved to {OUTPUT_PATH}")
    print(f"   Entry → X: {X_entry.shape}, y: {y_entry.shape}")
    print(f"   Exit  → X: {X_exit.shape}, y: {y_exit.shape}")

if __name__ == "__main__":
    main()
