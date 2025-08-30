# scripts/model_training/train_sequence_entry.py
"""
Train GRU Entry Model on Sequential Data
----------------------------------------
This script trains a GRU model on the pre-built sequence dataset
for entry signal prediction (i.e., buy signal).
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from scripts.model_training.sequence.sequence_model_utils import (
    build_gru_model, get_early_stopping, DEFAULT_WINDOW_LEN, DEFAULT_N_FEATURES
)

import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from core.sequence_loader import load_sequence_data_entry

# --- Load Data ---
X_train, y_train = load_sequence_data_entry()
print(f"📊 Loaded Entry Dataset: X={X_train.shape}, y={y_train.shape}")

# --- Model Builder ---
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# --- Dynamic Input Shape ---
n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]
model = build_gru_model(input_shape=(n_timesteps, n_features))
print("📊 GRU Model Summary:")
model.summary()

# --- Callbacks ---
checkpoint_cb = ModelCheckpoint("models/gru_entry_model.keras", save_best_only=True)
earlystop_cb = EarlyStopping(patience=5, restore_best_weights=True)

# --- Train ---
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)


# Plot training & validation loss values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# --- Save Model ---
model.save("models/entry_sequence_gru.h5")
print("✅ Entry GRU model saved to models/entry_sequence_gru.h5")
