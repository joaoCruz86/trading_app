# tests/test_sequence_dataset.py
"""
Unit tests for build_sequence_dataset.py
----------------------------------------
Checks that the sequence dataset builder:
- Loads data correctly
- Produces rolling windows with expected shape
- Saves to .npz file
"""

import os
import numpy as np
import pandas as pd
import unittest
from scripts.data_prep.build_sequence_dataset import build_sequences, main, OUTPUT_PATH, WINDOW_LEN, HORIZON


class TestSequenceDataset(unittest.TestCase):

    def setUp(self):
        # create a tiny dummy dataset
        dates = pd.date_range("2020-01-01", periods=100)
        self.df = pd.DataFrame({
            "ticker": ["AAPL"] * 100,
            "date": dates,
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50   # alternating labels
        })

    def test_build_sequences_shapes(self):
        X, y = build_sequences(self.df)
        # check sequence dimensions
        self.assertEqual(X.shape[1], WINDOW_LEN)
        self.assertEqual(X.shape[2], 2)  # 2 features
        self.assertEqual(len(X), len(y))
        self.assertTrue(set(y).issubset({0, 1}))

    def test_output_file_created(self):
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
        main()  # run builder
        self.assertTrue(os.path.exists(OUTPUT_PATH))
        data = np.load(OUTPUT_PATH)
        self.assertIn("X_entry", data)
        self.assertIn("y_entry", data)


if __name__ == "__main__":
    unittest.main()
