# scripts/model_training/sequence/train_sequence_exit.py
"""
Train a GRU model to detect when a position should be exited (Exit Signal).
Reads labeled sequence data from MongoDB collection: sequence_exit.

Saves model to: models/sequence/exit_gru.h5
"""

import os
import numpy as np
from core.db import db
from scripts.model_training.sequence.sequence_model_utils import build_gru_model, DEFAULT_WINDOW_LEN
from tensorflow.keras.callbacks import EarlyStopping

# --- Config ---
MODEL_PATH = "models/sequence/exit_gru.h5"
WINDOW_LEN = DEFAULT_WINDOW_LEN
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5  # EarlyStopping patience


def load_exit_sequence_data():
    """Load preprocessed exit sequence data from MongoDB."""
    data = list(db["sequence_exit"].find())
    X = np.array([d["sequence"] for d in data])
    y = np.array([d["label"] for d in data])
    return X, y


def train_exit_sequence_model():
    """Train and save the GRU-based exit signal model."""
    print("⚙️ Loading exit sequence data...")
    X, y = load_exit_sequence_data()
    print(f"✅ Loaded: {X.shape[0]} samples | Shape: {X.shape}")

    n_features = X.shape[2]
    input_shape = (WINDOW_LEN, n_features)

    print("🚀 Training GRU model for exit signals...")
    model = build_gru_model(input_shape=input_shape)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"💾 Exit GRU model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_exit_sequence_model()
