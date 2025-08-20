# scripts/model_training/train_lightgbm.py
import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from core.db import db


def _load_training_df() -> pd.DataFrame:
    df = pd.DataFrame(list(db["training"].find()))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def _prep_features(X: pd.DataFrame) -> pd.DataFrame:
    # Drop common non-feature columns if they slipped in
    drop_cols = [c for c in X.columns if c.lower() in {"_id", "ticker", "symbol", "date", "timestamp", "ds"}]
    X = X.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric/bool for LightGBM, then fill NAs
    X = X.select_dtypes(include=["number", "bool"]).copy()
    if X.empty:
        raise ValueError("After preprocessing there are no numeric features left for training.")
    X = X.ffill().bfill().fillna(0)
    return X.astype(float)


def train_lightgbm(
    df: pd.DataFrame | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model: bool = True,
    model_path: str = "models/trading_model.pkl"  # <<< test expects this
):
    """
    Train a LightGBM classifier on the 'training' collection (or provided df).
    Saves the model to `model_path` by default and returns (model, metrics, report, cm, saved_path_or_None).
    """
    if df is None:
        df = _load_training_df()

    assert "target" in df.columns, "DataFrame must contain a 'target' column"
    y = df["target"].astype(int)

    X_raw = df.drop(columns=["target"], errors="ignore")
    X = _prep_features(X_raw)

    stratify = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print("⚡ Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_features": int(X.shape[1]),
    }
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    saved_path = None
    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        saved_path = model_path

    return model, metrics, report, cm, saved_path


if __name__ == "__main__":
    # CLI convenience: prints and saves (plot omitted to keep it simple)
    model, metrics, report, cm, saved = train_lightgbm()
    print("\n📊 Classification Report:\n", report)
    print("✅ Metrics:", metrics)
    print("✅ Confusion Matrix:\n", cm)


def load_lightgbm(model_path: str = "models/trading_model.pkl"):
    """
    Load the saved LightGBM model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ LightGBM model not found at {model_path}")
    return joblib.load(model_path)


