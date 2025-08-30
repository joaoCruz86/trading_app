# core/sequence_loader.py

import numpy as np

def load_sequence_data_entry(npz_path="data/sequence_dataset.npz"):
    data = np.load(npz_path)
    X_entry = data["X_entry"]
    y_entry = data["y_entry"]
    return X_entry, y_entry

def load_sequence_data_exit(npz_path="data/sequence_dataset.npz"):
    data = np.load(npz_path)
    X_exit = data["X_exit"]
    y_exit = data["y_exit"]
    return X_exit, y_exit
