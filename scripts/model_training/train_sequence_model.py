# scripts/model_training/train_sequence_model.py
"""
Train Sequence Models (GRU/LSTM) for Entry/Exit signals.
--------------------------------------------------------
This script prepares rolling windows from the sequence dataset,
trains GRU models to capture temporal patterns, and saves them.

- Entry Model: predicts if a stock should be bought today (horizon = +5 days).
- Exit Model: predicts if an open position should be exited soon.

Models are saved under: models/sequence/
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Config ---
ENTRY_MODEL_PATH = "models/sequence/entry_gru.h5"
EXIT_MODEL_PATH = "models/sequence/exit_gru.h5"
WINDOW_LEN = 60   # number of past days
N_FEATURES = 16   # adjust when you know feature count
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001


def build_model(input_shape, hidden_units=64, dropout=0.2, lr=LR):
    """
    Build a GRU binary classifier.

    Args:
        input_shape (tuple): Shape of the sequence (WINDOW_LEN, N_FEATURES).
        hidden_units (int): Number of GRU units.
        dropout (float): Dropout rate.
        lr (float): Learning rate.

    Returns:
        keras.Model: Compiled GRU model.
    """
    model = Sequential()
    model.add(GRU(hidden_units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_entry_exit_models(X_entry, y_entry, X_exit, y_exit):
    """
    Train both entry and exit GRU models and save them.

    Args:
        X_entry (np.ndarray): Sequence features for entry model (samples, window, features).
        y_entry (np.ndarray): Binary labels for entry model.
        X_exit (np.ndarray): Sequence features for exit model.
        y_exit (np.ndarray): Binary labels for exit model.
    """
    os.makedirs("models/sequence", exist_ok=True)

    # Entry model
    entry_model = build_model((WINDOW_LEN, N_FEATURES))
    entry_model.fit(X_entry, y_entry, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    entry_model.save(ENTRY_MODEL_PATH)
    print(f"✅ Entry GRU model saved to {ENTRY_MODEL_PATH}")

    # Exit model
    exit_model = build_model((WINDOW_LEN, N_FEATURES))
    exit_model.fit(X_exit, y_exit, epochs=EPOCHS,
                   batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    exit_model.save(EXIT_MODEL_PATH)
    print(f"✅ Exit GRU model saved to {EXIT_MODEL_PATH}")


def load_sequence_models():
    """
    Load trained entry and exit GRU models from disk.

    Returns:
        (keras.Model, keras.Model): entry_model, exit_model
    """
    entry_model = load_model(ENTRY_MODEL_PATH)
    exit_model = load_model(EXIT_MODEL_PATH)
    return entry_model, exit_model


def predict_with_sequence(entry_model, exit_model, seq: np.ndarray):
    """
    Predict entry/exit probabilities for a given sequence.

    Args:
        entry_model (keras.Model): Trained entry GRU model.
        exit_model (keras.Model): Trained exit GRU model.
        seq (np.ndarray): Shape (WINDOW_LEN, N_FEATURES).

    Returns:
        tuple: (entry_prob, exit_prob)
    """
    seq = np.expand_dims(seq, axis=0)  # add batch dimension
    entry_prob = float(entry_model.predict(seq, verbose=0)[0][0])
    exit_prob = float(exit_model.predict(seq, verbose=0)[0][0])
    return entry_prob, exit_prob


def main():
    """Standalone training with dummy data (for dev only)."""
    # Dummy dataset: 1000 samples, 60-day window, 16 features
    X_dummy = np.random.rand(1000, WINDOW_LEN, N_FEATURES)
    y_entry_dummy = np.random.randint(0, 2, 1000)
    y_exit_dummy = np.random.randint(0, 2, 1000)

    train_entry_exit_models(X_dummy, y_entry_dummy,
                            X_dummy, y_exit_dummy)


if __name__ == "__main__":
    main()
