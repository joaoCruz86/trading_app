# scripts/model_training/train_sequence_entry.py
"""
Train GRU Entry Model on Sequential Data
----------------------------------------
This script trains a GRU model on the pre-built sequence dataset
for entry signal prediction (i.e., buy signal).
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from scripts.model_training.sequence.sequence_model_utils import (
    build_gru_model, get_early_stopping, DEFAULT_WINDOW_LEN, DEFAULT_N_FEATURES
)


# --- Load Data ---
data = np.load("data/sequence_dataset.npz")
X = data["X_entry"]
y = data["y_entry"]

print(f"📊 Loaded Entry Dataset: X={X.shape}, y={y.shape}")

# --- Split Data ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Build Model ---
input_shape = (DEFAULT_WINDOW_LEN, DEFAULT_N_FEATURES)
model = build_gru_model(input_shape=input_shape)
model.summary()

# --- Train Model ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[get_early_stopping()],
    verbose=1
)

# --- Save Model ---
model.save("models/entry_sequence_gru.h5")
print("✅ Entry GRU model saved to models/entry_sequence_gru.h5")
