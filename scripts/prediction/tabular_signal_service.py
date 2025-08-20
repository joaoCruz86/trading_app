# scripts/prediction/tabular_signal_service.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

from core.db import db
from scripts.model_training.tabular.train_model import load_tabular_models
from scripts.model_training.sequence.sequence_model_utils import (
    load_sequence_models,
    predict_with_sequence
)

# --- Config ---
TABULAR_OUTPUT = "outputs/tabular_shortlist.csv"
FINAL_OUTPUT = "outputs/final_signals.csv"
CONFIDENCE_THRESHOLD = 0.7


def get_last_n_days(ticker: str, n: int = 60) -> np.ndarray:
    """
    Temporary stub for building sequences.
    Replace with DB query for production.
    """
    num_features = 16
    return np.random.rand(n, num_features)


def run_signal_service():
    # 1. Get latest data
    df = pd.DataFrame(list(db["latest"].find()))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    features = df.drop(columns=["target"], errors="ignore")

    # 2. Load tabular models (Layer 1)
    model_lgb, model_rf = load_tabular_models()

    # 3. Predict and average scores
    proba_lgb = model_lgb.predict_proba(features)[:, 1]
    proba_rf = model_rf.predict_proba(features)[:, 1]
    tabular_scores = (proba_lgb + proba_rf) / 2

    df["tabular_score"] = tabular_scores

    # 4. Shortlist by threshold
    shortlist = df[df["tabular_score"] > 0.5].copy()
    os.makedirs("outputs", exist_ok=True)
    shortlist.to_csv(TABULAR_OUTPUT, index=False)
    print(f"📊 Tabular shortlist saved to {TABULAR_OUTPUT}")

    # 5. Load sequence models (Layer 2)
    entry_model, exit_model = load_sequence_models()

    final_signals = []
    for _, row in shortlist.iterrows():
        ticker = row["ticker"]
        seq = get_last_n_days(ticker, n=60)

        # Predict entry/exit signals using sequence model
        seq_result = predict_with_sequence(entry_model, exit_model, seq)

        final_signals.append({
            "date": datetime.today().strftime("%Y-%m-%d"),
            "ticker": ticker,
            "tabular_score": row["tabular_score"],
            "entry_confidence": seq_result["entry_confidence"],
            "exit_confidence": seq_result["exit_confidence"],
            "decision": decide_signal(row["tabular_score"], seq_result)
        })

    final_df = pd.DataFrame(final_signals)
    final_df.to_csv(FINAL_OUTPUT, index=False)
    print(f"✅ Final signals saved to {FINAL_OUTPUT}")

    # Optional: Save to DB
    db["signals"].insert_many(final_df.to_dict("records"))
    print("🧠 Final signals also saved to MongoDB → signals")


def decide_signal(tabular_score, seq_result):
    """
    Decide final action: BUY, EXIT, or HOLD.
    """
    entry = seq_result["entry_confidence"]
    exit_ = seq_result["exit_confidence"]

    if entry >= CONFIDENCE_THRESHOLD:
        return "BUY"
    elif exit_ >= CONFIDENCE_THRESHOLD:
        return "EXIT"
    else:
        return "HOLD"


if __name__ == "__main__":
    run_signal_service()
