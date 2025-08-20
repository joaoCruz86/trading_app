# tests/test_sequence_model.py
# ---------------------------------------------------
# Unit tests for sequence model training, saving, loading, and predicting.
# These tests validate the core sequence modeling pipeline built on top of Keras.
# ---------------------------------------------------

import os
import warnings
import numpy as np
import pytest

# Quieter logs for test runs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TF INFO/WARN
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from scripts.model_training.sequence import sequence_model_utils as seq


@pytest.fixture(autouse=True)
def speed_up_training(monkeypatch):
    """Run fewer epochs in tests to keep them fast."""
    monkeypatch.setattr(seq, "EPOCHS", 1)
    yield


def test_build_model():
    """Test that the model builds successfully with given input shape."""
    model = seq.build_model((60, 16))
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_train_and_save_models():
    """Test training and saving entry/exit models using default paths."""
    # Synthetic dataset
    X = np.random.rand(50, 60, 16).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.int32)

    # Train and save
    seq.train_entry_exit_models(X, y, X, y)

    # Check default file paths
    assert os.path.exists(seq.ENTRY_MODEL_PATH)
    assert os.path.exists(seq.EXIT_MODEL_PATH)


def test_load_and_predict():
    """Test loading saved models and running prediction."""
    # Synthetic dataset
    X = np.random.rand(20, 60, 16).astype(np.float32)
    y = np.random.randint(0, 2, 20).astype(np.int32)

    # Train & save using default paths
    seq.train_entry_exit_models(X, y, X, y)

    # Load from default paths
    entry_model, exit_model = seq.load_sequence_models()

    # Predict on a single sequence
    test_seq = X[0]  # shape (60, 16)
    entry_prob, exit_prob = seq.predict_with_sequence(entry_model, exit_model, test_seq)

    assert 0.0 <= entry_prob <= 1.0
    assert 0.0 <= exit_prob <= 1.0
