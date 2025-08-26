# tests/test_data_prep.py

"""
Unit test for data preparation logic (build_training_dataset.py).
This test ensures:
- The dataset builds correctly for a given ticker from MongoDB
- Expected features and target columns are present
- Both target classes (0 and 1) exist for classification balance
"""

import unittest
import pandas as pd
from archived_tabular.build_training_dataset import build_dataset
# from core.db import _client  # Don't close it here!

class TestDataPrep(unittest.TestCase):

    def setUp(self):
        self.df = build_dataset("AAPL")

    def test_structure(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertIn("target", self.df.columns)
        self.assertIn("RSI", self.df.columns)
        self.assertIn("ticker", self.df.columns)
        self.assertEqual(self.df["ticker"].nunique(), 1)

    def test_target_balance(self):
        unique_targets = self.df["target"].unique().tolist()
        self.assertIn(0, unique_targets)
        self.assertIn(1, unique_targets)

if __name__ == "__main__":
    unittest.main()
