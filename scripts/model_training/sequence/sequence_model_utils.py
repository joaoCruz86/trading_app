# scripts/model_training/sequence_model_utils.py
"""
Utility module for building sequence models (GRU).
Shared across entry and exit training scripts.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


DEFAULT_WINDOW_LEN = 60
DEFAULT_N_FEATURES = 16
DEFAULT_LR = 0.001


def build_gru_model(input_shape, hidden_units=64, dropout=0.2, lr=DEFAULT_LR):
    """
    Build and compile a GRU model.

    Args:
        input_shape (tuple): Shape of the input sequence.
        hidden_units (int): GRU units.
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
