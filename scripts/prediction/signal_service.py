# scripts/prediction/signal_service.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

from core.db import db
from scripts.model_training.train_lightgbm import train_lightgbm
from scripts.model_training.train_sequence_model import load_sequence_models, predict_with_sequence

# --- Config ---
TABULAR_OUTPUT = "outputs/tabular_shortlist.csv"
FINAL_OUTPUT = "outputs/final_signals.csv"
CONFIDENCE_THRESHOLD = 0.7   # adjust to control how many trades per month


def get_last_n_days(ticker: str, n: int = 60) -> np.ndarray:
    """
    Temporary stub for building sequences.
    In the future, this should query your DB or feature store
    to return the last `n` days of features for the given ticker.

    For now: returns a random sequence [n, num_features].
    """
    num_features = 16  # adjust when you know exact feature count
    return np.random.rand(n, num_features)


def run_signal_service():
    # 1. Get latest data from DB
    df = pd.DataFrame(list(db["latest"].find()))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    # 2. Run tabular screener (Layer 1)
    tabular_model, _, _, _ = train_lightgbm(df)  # or load from disk
    tabular_scores = tabular_model.predict_proba(df.drop(columns=["target"]))[:, 1]
    df["tabular_score"] = tabular_scores

    # shortlist candidates
    shortlist = df[df["tabular_score"] > 0.5].copy()  # loose threshold
    os.makedirs("outputs", exist_ok=True)
    shortlist.to_csv(TABULAR_OUTPUT, index=False)
    print(f"📊 Tabular shortlist saved to {TABULAR_OUTPUT}")

    # 3. Load time-aware models (Layer 2)
    entry_model, exit_model = load_sequence_models()

    final_signals = []
    for _, row in shortlist.iterrows():
        ticker = row["ticker"]

        # get last n days (stub for now)
        seq = get_last_n_days(ticker, n=60)

        # run sequence models
        entry_prob, exit_prob = predict_with_sequence(entry_model, exit_model, seq)

        # make decision
        if entry_prob >= CONFIDENCE_THRESHOLD:
            decision = "BUY"
        elif exit_prob >= CONFIDENCE_THRESHOLD:
            decision = "EXIT"
        else:
            decision = "HOLD"

        final_signals.append({
            "date": datetime.today().strftime("%Y-%m-%d"),
            "ticker": ticker,
            "tabular_score": row["tabular_score"],
            "entry_prob": entry_prob,
            "exit_prob": exit_prob,
            "decision": decision
        })

    final_df = pd.DataFrame(final_signals)
    final_df.to_csv(FINAL_OUTPUT, index=False)
    print(f"✅ Final signals saved to {FINAL_OUTPUT}")


if __name__ == "__main__":
    run_signal_service()
