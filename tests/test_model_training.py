# tests/test_model_training.py
"""
Unit test for model training functions.
This test ensures:
- Models can be trained without errors (LightGBM, Random Forest)
- Output model files are created successfully in the 'models/' folder
"""

import os
import unittest
from scripts.model_training.tabular.train_random_forest import train_random_forest
from scripts.model_training.tabular.train_lightgbm import train_lightgbm


class TestModelTraining(unittest.TestCase):

    def test_train_random_forest_creates_model(self):
        train_random_forest()
        self.assertTrue(os.path.exists("models/entry_model.pkl"))

    def test_train_lightgbm_creates_model(self):
        train_lightgbm()
        self.assertTrue(os.path.exists("models/trading_model.pkl"))


if __name__ == "__main__":
    unittest.main()
