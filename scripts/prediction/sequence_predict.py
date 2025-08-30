# scripts/model_training/sequence/sequence_predict.py

"""
Runs sequence-based predictions for entry and exit signals.

This script loads the trained GRU sequence models and runs predictions on the
latest sequence dataset. Results are printed for each ticker.

Expected:
- Trained models in models/sequence/
- Sequence dataset in data/sequence_dataset.npz

Strategy:
- Entry model is trained on strong uptrends (≥ +6% in 15 days).
- At prediction time, we apply a softer confirmation rule (≥ +3% price move)
  before signaling a BUY.
- Exit model is trained on sharp drops, but we also apply a stop-loss rule:
  if price fell ≥ –3% within the sequence, we flag EXIT even if the model is unsure.
"""

import numpy as np
from tensorflow.keras.models import load_model
import os

# =========================
# 🔧 CONFIG
# =========================
CONFIG = {
    # Paths
    "ENTRY_MODEL_PATH": "models/sequence/entry_gru.h5",
    "EXIT_MODEL_PATH": "models/sequence/exit_gru.h5",
    "DATASET_PATH": "data/sequence_dataset.npz",

    # Thresholds
    "ENTRY_PROB_THRESHOLD": 0.5,   # model confidence for entry
    "EXIT_PROB_THRESHOLD": 0.5,    # model confidence for exit
    "CONFIRMATION_ENTRY": 0.03,    # +3% confirmation for entry
    "CONFIRMATION_EXIT": -0.03,    # –3% safety stop for exit
}


def run_sequence_prediction():
    # --- Load sequence dataset ---
    if not os.path.exists(CONFIG["DATASET_PATH"]):
        print(f"❌ Dataset not found at {CONFIG['DATASET_PATH']}")
        return

    data = np.load(CONFIG["DATASET_PATH"], allow_pickle=True)
    X_entry = data["X_entry"].astype("float32")
    X_exit = data["X_exit"].astype("float32")
    tickers_entry = [str(t) for t in data["tickers_entry"]]
    tickers_exit = [str(t) for t in data["tickers_exit"]]

    # --- Load models ---
    if not os.path.exists(CONFIG["ENTRY_MODEL_PATH"]) or not os.path.exists(CONFIG["EXIT_MODEL_PATH"]):
        print("❌ One or both GRU models are missing!")
        return

    entry_model = load_model(CONFIG["ENTRY_MODEL_PATH"])
    exit_model = load_model(CONFIG["EXIT_MODEL_PATH"])

    # --- Predict ---
    entry_preds = entry_model.predict(X_entry)
    exit_preds = exit_model.predict(X_exit)

    # --- Display results ---
    print("📈 Sequence-based predictions (Entry / Exit):\n")
    for i, (ticker, entry, exit_) in enumerate(zip(tickers_entry, entry_preds, exit_preds)):
        entry_prob = entry[0]
        exit_prob = exit_[0]

        # --- Extract price info from sequence ---
        seq_prices_entry = X_entry[i][:, 0]  # assuming feature[0] = normalized Close
        seq_prices_exit = X_exit[i][:, 0]

        # Relative changes
        rel_change_entry = (seq_prices_entry[-1] - seq_prices_entry[0]) / (seq_prices_entry[0] + 1e-6)
        rel_change_exit = (seq_prices_exit[-1] - seq_prices_exit[0]) / (seq_prices_exit[0] + 1e-6)

        # --- Entry rule ---
        if entry_prob > CONFIG["ENTRY_PROB_THRESHOLD"] and rel_change_entry >= CONFIG["CONFIRMATION_ENTRY"]:
            entry_signal = f"✅ BUY ({entry_prob:.2f}, +{rel_change_entry*100:.1f}%)"
        else:
            entry_signal = f"❌ WAIT ({entry_prob:.2f}, +{rel_change_entry*100:.1f}%)"

        # --- Exit rule ---
        if exit_prob > CONFIG["EXIT_PROB_THRESHOLD"] or rel_change_exit <= CONFIG["CONFIRMATION_EXIT"]:
            exit_signal = f"🚪 EXIT ({exit_prob:.2f}, {rel_change_exit*100:.1f}%)"
        else:
            exit_signal = f"🔒 HOLD ({exit_prob:.2f}, {rel_change_exit*100:.1f}%)"

        print(f"{ticker}: Entry → {entry_signal}   Exit → {exit_signal}")


if __name__ == "__main__":
    run_sequence_prediction()
