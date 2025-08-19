import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from core.db import db

def train_random_forest():
    # --- Load data from MongoDB ---
    df = pd.DataFrame(list(db["training"].find()))
    df.drop(columns=["_id"], inplace=True)

    # --- Drop unnecessary columns ---
    non_numeric_cols = df.select_dtypes(include=["object"]).columns
    print("🧹 Dropping non-numeric columns:", list(non_numeric_cols))
    df.drop(columns=non_numeric_cols, inplace=True)

    # --- Separate features and target ---
    X = df.drop(columns=["target"])
    y = df["target"]

    assert set(y.unique()).issubset({0, 1}), "Target column is not binary!"

    # --- Handle missing values ---
    X = X.ffill().bfill().fillna(0)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🧪 Label Distribution (Train):")
    print(y_train.value_counts(normalize=True).rename(lambda x: f"Signal {x}"))

    print("\n🧪 Label Distribution (Test):")
    print(y_test.value_counts(normalize=True).rename(lambda x: f"Signal {x}"))

    # --- Train Random Forest ---
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("✅ Precision:", precision_score(y_test, y_pred))
    print("✅ Recall:", recall_score(y_test, y_pred))
    print("✅ Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Save model ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/entry_model.pkl")
    print("✅ Model saved to models/entry_model.pkl")

# Optional: allow script to be run directly
if __name__ == "__main__":
    train_random_forest()
