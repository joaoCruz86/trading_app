import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from core.db import db


def _load_training_df() -> pd.DataFrame:
    df = pd.DataFrame(list(db["training"].find()))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def _prep_features(X: pd.DataFrame) -> pd.DataFrame:
    # 🧠 Optional: Extract time-based features if 'date' column exists
    if "date" in X.columns and pd.api.types.is_datetime64_any_dtype(X["date"]):
        X["day_of_week"] = X["date"].dt.dayofweek
        X["month"] = X["date"].dt.month
        X["day"] = X["date"].dt.day
        X = X.drop(columns=["date"])

    # 🧹 Drop non-numeric and datetime columns
    non_numeric_cols = X.select_dtypes(include=["object", "datetime64"]).columns
    print("🧹 Dropping non-numeric columns:", list(non_numeric_cols))
    X = X.drop(columns=non_numeric_cols, errors="ignore")

    return X.ffill().bfill().fillna(0)


def train_random_forest(
    df: pd.DataFrame | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model: bool = True,
    model_path: str = "models/entry_model.pkl"
):
    """
    Train a Random Forest classifier and save it.
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

    print("\n🧪 Train Label Distribution:\n", y_train.value_counts(normalize=True))
    print("\n🧪 Test Label Distribution:\n", y_test.value_counts(normalize=True))

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "n_features": int(X.shape[1])
    }
    print("✅ Metrics:", metrics)
    print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    saved_path = None
    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        saved_path = model_path

    return model, metrics, saved_path


def load_random_forest(model_path: str = "models/entry_model.pkl"):
    """
    Load a trained Random Forest model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Random Forest model not found at {model_path}")
    return joblib.load(model_path)


if __name__ == "__main__":
    train_random_forest()
