import os
import pandas as pd
import numpy as np
from datetime import datetime
from core.db import db

from scripts.model_training.tabular.train_model import load_tabular_models

# --- Optional: Sequence model (if enabled) ---
USE_SEQUENCE_MODEL = False  # 🔧 Toggle sequence second layer
if USE_SEQUENCE_MODEL:
    from scripts.model_training.sequence.sequence_model_utils import (
        load_sequence_models,
        predict_with_sequence
    )

# --- Config ---
SOURCE_COLLECTION = "training"   # "latest" | "training" | "exit_training" ...
TABULAR_OUTPUT = "outputs/tabular_shortlist.csv"
FINAL_OUTPUT   = "outputs/final_signals.csv"
CONFIDENCE_THRESHOLD = 0.7


def _engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day/month/day_of_week if 'date' exists."""
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        df = df.assign(
            day=d.dt.day,
            month=d.dt.month,
            day_of_week=d.dt.dayofweek
        )
    return df


def _build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numeric feature frame:
    - add calendar features
    - drop ids/non-features
    - keep numeric/bool only
    - fillna
    """
    df = _engineer_time_features(df.copy())

    drop_cols = [
        c for c in df.columns
        if c.lower() in {"_id", "ticker", "symbol", "date", "timestamp", "ds", "ticker ", "Ticker"}
        or c == "target"
    ]
    base = df.drop(columns=drop_cols, errors="ignore")

    # keep numeric or bool only
    base = base.select_dtypes(include=["number", "bool"]).copy()
    base = base.ffill().bfill().fillna(0)

    # ensure float dtype for safety across libs
    for col in base.columns:
        try:
            base[col] = base[col].astype(float)
        except Exception:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0).astype(float)

    return base


def _align_for_model(base: pd.DataFrame, expected: list[str], model_name: str) -> pd.DataFrame:
    """
    Align a base numeric feature frame to a given model's expected feature list.
    - add any missing columns as 0.0
    - select and order columns exactly as expected
    """
    missing = [c for c in expected if c not in base.columns]
    extra   = [c for c in base.columns if c not in expected]

    if missing:
        print(f"⚠️ [{model_name}] Adding missing features as zeros: {missing}")
        for c in missing:
            base[c] = 0.0

    if extra:
        print(f"ℹ️ [{model_name}] Ignoring extra features: {extra}")

    aligned = base[expected].copy().astype(float)
    return aligned


def get_last_n_days(ticker: str, n: int = 60) -> np.ndarray:
    """
    Dummy placeholder for sequence data lookup.
    Replace with actual historical sequence retrieval.
    """
    num_features = 16
    return np.random.rand(n, num_features)


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
    # 1) Load data
    print(f"📥 Loading data from: {SOURCE_COLLECTION}")
    df = pd.DataFrame(list(db[SOURCE_COLLECTION].find()))
    print(f"📊 Number of samples: {len(df)}")
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    if df.empty:
        print("❌ No rows found. Aborting.")
        return

    # 2) Load models first (so we know their exact expected features)
    model_lgb, model_rf = load_tabular_models()
    expected_lgb = list(getattr(model_lgb, "feature_names_in_", []))
    expected_rf  = list(getattr(model_rf,  "feature_names_in_", []))

    if not expected_lgb or not expected_rf:
        print("❌ Could not read feature_names_in_ from one or both models. Retrain with sklearn>=1.0.")
        return

    # 3) Build base features once
    base = _build_base_features(df)

    # 4) Align per-model
    X_lgb = _align_for_model(base.copy(), expected_lgb, "LightGBM")
    X_rf  = _align_for_model(base.copy(), expected_rf,  "RandomForest")

    # Guard: shapes must match what the models expect
    if X_lgb.shape[1] != getattr(model_lgb, "n_features_in_", X_lgb.shape[1]):
        raise ValueError(f"🚫 LightGBM feature mismatch: expected {model_lgb.n_features_in_}, got {X_lgb.shape[1]}")

    if X_rf.shape[1] != getattr(model_rf, "n_features_in_", X_rf.shape[1]):
        raise ValueError(f"🚫 RandomForest feature mismatch: expected {model_rf.n_features_in_}, got {X_rf.shape[1]}")

    # 5) Predict
    proba_lgb = model_lgb.predict_proba(X_lgb)[:, 1]
    proba_rf  = model_rf.predict_proba(X_rf)[:, 1]
    df["tabular_score"] = (proba_lgb + proba_rf) / 2

    # 6) Save shortlist
    shortlist = df[df["tabular_score"] > 0.5].copy()
    os.makedirs("outputs", exist_ok=True)
    shortlist.to_csv(TABULAR_OUTPUT, index=False)
    print(f"✅ Tabular shortlist saved → {TABULAR_OUTPUT}")

    # 7) Optional sequence second layer
    if USE_SEQUENCE_MODEL and not shortlist.empty:
        entry_model, exit_model = load_sequence_models()
        final_signals = []
        for _, row in shortlist.iterrows():
            ticker = row.get("ticker") or row.get("Ticker")
            seq = get_last_n_days(str(ticker))
            seq_result = predict_with_sequence(entry_model, exit_model, seq)

            final_signals.append({
                "date": datetime.today().strftime("%Y-%m-%d"),
                "ticker": ticker,
                "tabular_score": float(row["tabular_score"]),
                "entry_confidence": float(seq_result["entry_confidence"]),
                "exit_confidence": float(seq_result["exit_confidence"]),
                "decision": decide_signal(row["tabular_score"], seq_result)
            })

        final_df = pd.DataFrame(final_signals)
        final_df.to_csv(FINAL_OUTPUT, index=False)
        db["signals"].insert_many(final_df.to_dict("records"))
        print(f"🧠 Final signals saved → {FINAL_OUTPUT} and MongoDB: signals")


if __name__ == "__main__":
    run_tabular_prediction()
