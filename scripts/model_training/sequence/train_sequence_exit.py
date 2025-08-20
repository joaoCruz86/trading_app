# scripts/model_training/sequence/train_sequence_exit.py
"""
Train a GRU model to detect when a position should be exited (Exit Signal).
Reads labeled sequence data from MongoDB collection: sequence_exit.

Saves model to: models/sequence/exit_gru.h5
"""

import os
import numpy as np
from core.db import db
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Config ---
MODEL_PATH = "models/sequence/exit_gru.h5"
WINDOW_LEN = 60
N_FEATURES = 16
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001


def load_exit_sequence_data():
    """Load preprocessed exit sequence data from MongoDB."""
    data = list(db["sequence_exit"].find())
    X = np.array([d["sequence"] for d in data])
    y = np.array([d["label"] for d in data])
    return X, y


def build_model(input_shape, hidden_units=64, dropout=0.2, lr=LR):
    """Build and compile a GRU model."""
    model = Sequential()
    model.add(GRU(hidden_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_exit_sequence_model():
    """Train and save the GRU-based exit signal model."""
    print("⚙️ Loading exit sequence data...")
    X, y = load_exit_sequence_data()
    print(f"✅ Loaded: {X.shape[0]} samples")

    print("🚀 Training GRU model for exit signals...")
    model = build_model((WINDOW_LEN, N_FEATURES))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"💾 Exit GRU model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_exit_sequence_model()
