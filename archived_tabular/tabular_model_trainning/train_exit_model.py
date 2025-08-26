# scripts/train_exit_model.py
"""
Train a machine learning model to detect weakening positions (exit signals).
This script uses technical indicators as input features and learns to predict
whether a position should be exited based on future performance.

Output: models/exit_model.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Paths ---
DATA_PATH = "data/exit_training_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "exit_model.pkl")

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded exit training dataset: {df.shape}")
print(df.head(3))

# --- Clean up ---
df = df.dropna()
df = df.drop(columns=["ticker", "date", "future_close", "future_return"], errors="ignore")

# --- Features and target ---
TARGET = "exit_target"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Train model ---
print("⚙️ Training Exit Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")

# --- Save model ---
joblib.dump(model, MODEL_PATH)
print(f"\n💾 Exit model saved to {MODEL_PATH}")
