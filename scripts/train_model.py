# scripts/train_model.py
"""
Train an AI model on technical indicators to predict stock signals.
- Uses dataset prepared by scripts/training_dataset.py
- Model: LightGBM (classification: BUY / HOLD / SELL)
- Output: models/trading_model.pkl
"""

import os
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Config ---
DATA_PATH = "data/training_dataset.csv"   # dataset built previously
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "trading_model.pkl")

# Ensure models dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load dataset ---
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Drop rows with missing values
df = df.dropna()

# Features (X) and target (y)
FEATURES = [
    "RSI", "MACD", "MACD_signal", "EMA_20", "EMA_50",
    "SMA_20", "SMA_200", "BB_upper", "BB_lower", "Close"
]
TARGET = "Signal"   # from training_dataset.py (e.g. BUY / SELL / HOLD)

X = df[FEATURES]
y = df[TARGET]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train ---
print("⚡ Training LightGBM model...")
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n✅ Model trained!")
print("Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Save ---
joblib.dump(model, MODEL_PATH)
print(f"\n💾 Model saved to {MODEL_PATH}")
