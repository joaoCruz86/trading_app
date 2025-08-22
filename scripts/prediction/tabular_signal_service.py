# scripts/prediction/tabular_signal_service.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from core.db import db

from scripts.model_training.tabular.train_model import load_tabular_models

USE_SEQUENCE_MODEL = False
if USE_SEQUENCE_MODEL:
    from scripts.model_training.sequence.sequence_model_utils import (
        load_sequence_models,
        predict_with_sequence
    )

SOURCE_COLLECTION = "training"
TABULAR_OUTPUT = "outputs/tabular_shortlist.csv"
FINAL_OUTPUT = "outputs/final_signals.csv"
CONFIDENCE_THRESHOLD = 0.7

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek

    drop_cols = [c for c in df.columns if c.lower() in {"_id", "ticker", "symbol", "date", "timestamp", "ds"}]
    features = df.drop(columns=drop_cols + ["target"], errors="ignore")

    features = features.select_dtypes(include=["number", "bool"]).copy()
    features = features.ffill().bfill().fillna(0).astype(float)

    return features


def get_last_n_days(ticker: str, n: int = 60) -> np.ndarray:
    return np.random.rand(n, 16)


def decide_signal(tabular_score, seq_result):
    entry = seq_result["entry_confidence"]
    exit_ = seq_result["exit_confidence"]

    if entry >= CONFIDENCE_THRESHOLD:
        return "BUY"
    elif exit_ >= CONFIDENCE_THRESHOLD:
        return "EXIT"
    else:
        return "HOLD"


def run_tabular_prediction():
    print(f"📥 Loading data from: {SOURCE_COLLECTION}")
    df = pd.DataFrame(list(db[SOURCE_COLLECTION].find()))
    print(f"📊 Number of samples: {len(df)}")
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    features = preprocess_features(df)

    model_lgb, model_rf = load_tabular_models()

    if features.shape[1] != model_lgb.n_features_in_:
        raise ValueError(
            f"🚫 Feature mismatch: Expected {model_lgb.n_features_in_}, got {features.shape[1]}"
        )

    proba_lgb = model_lgb.predict_proba(features)[:, 1]
    proba_rf = model_rf.predict_proba(features)[:, 1]
    df["tabular_score"] = (proba_lgb + proba_rf) / 2

    shortlist = df[df["tabular_score"] > 0.5].copy()
    os.makedirs("outputs", exist_ok=True)
    shortlist.to_csv(TABULAR_OUTPUT, index=False)
    print(f"✅ Tabular shortlist saved → {TABULAR_OUTPUT}")

    if USE_SEQUENCE_MODEL:
        entry_model, exit_model = load_sequence_models()
        final_signals = []

        for _, row in shortlist.iterrows():
            ticker = row["ticker"]
            seq = get_last_n_days(ticker)
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
        db["signals"].insert_many(final_df.to_dict("records"))
        print(f"🧠 Final signals saved → {FINAL_OUTPUT} and MongoDB: signals")


if __name__ == "__main__":
    run_tabular_prediction()
