"""
Unit test for exit signal dataset building (build_exit_dataset.py).

This test ensures:
- Exit dataset builds correctly from MongoDB prices
- Output contains 'target_exit' and indicator columns
- Both 0 and 1 classes are present in the target_exit label
"""

import unittest
import pandas as pd
from scripts.data_prep.build_exit_training_dataset import build_exit_dataset


class TestExitDataset(unittest.TestCase):
    def setUp(self):
        self.df = build_exit_dataset("AAPL")

    def test_dataset_structure(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertIn("exit_target", self.df.columns)
        self.assertIn("RSI", self.df.columns)
        self.assertIn("ticker", self.df.columns)
        self.assertEqual(self.df["ticker"].nunique(), 1)

    def test_target_exit_balance(self):
        unique_targets = self.df["exit_target"].unique().tolist()
        self.assertIn(0, unique_targets)
        self.assertIn(1, unique_targets)

if __name__ == "__main__":
    unittest.main()
    
from core.db import _client
_client.close()

