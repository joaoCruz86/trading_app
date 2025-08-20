"""
Runs sequence-based predictions for entry and exit signals.

This script loads the trained GRU sequence models and runs predictions on the
latest sequence dataset. Results are printed for each ticker.

Expected:
- Trained models in models/sequence/
- Sequence dataset in data/sequence_dataset.npz
"""

import numpy as np
from tensorflow.keras.models import load_model
import os

# --- Paths ---
ENTRY_MODEL_PATH = "models/sequence/entry_gru.h5"
EXIT_MODEL_PATH = "models/sequence/exit_gru.h5"
DATASET_PATH = "data/sequence_dataset.npz"


def run_sequence_prediction():
    # --- Load sequence dataset ---
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return

    data = np.load(DATASET_PATH)
    X = data["X"]
    tickers = data["tickers"]

    # --- Load models ---
    if not os.path.exists(ENTRY_MODEL_PATH) or not os.path.exists(EXIT_MODEL_PATH):
        print("❌ One or both GRU models are missing!")
        return

    entry_model = load_model(ENTRY_MODEL_PATH)
    exit_model = load_model(EXIT_MODEL_PATH)

    # --- Predict ---
    entry_preds = entry_model.predict(X)
    exit_preds = exit_model.predict(X)

    # --- Display results ---
    print("📈 Sequence-based predictions (Entry / Exit):\n")
    for ticker, entry, exit_ in zip(tickers, entry_preds, exit_preds):
        entry_signal = "✅ BUY" if entry > 0.5 else "❌ WAIT"
        exit_signal = "🚪 EXIT" if exit > 0.5 else "🔒 HOLD"
        print(f"{ticker}:  Entry → {entry_signal} ({entry[0]:.2f})   Exit → {exit_signal} ({exit_[0]:.2f})")


if __name__ == "__main__":
    run_sequence_prediction()
