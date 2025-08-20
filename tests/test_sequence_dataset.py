import os
import numpy as np
import pandas as pd
import unittest
from scripts.data_prep.build_sequence_dataset import build_sequences, main, OUTPUT_PATH, WINDOW_LEN, HORIZON
from core.db import db


class TestSequenceDataset(unittest.TestCase):

    def setUp(self):
        # Dummy data
        dates = pd.date_range("2020-01-01", periods=100)
        self.df = pd.DataFrame({
            "ticker": ["AAPL"] * 100,
            "date": dates,
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50   # alternating labels
        })

        # Clear test collections before running
        db["training"].delete_many({})
        db["sequence_entry"].delete_many({})
        db["sequence_exit"].delete_many({})

        # Insert dummy data into 'training' collection for main()
        db["training"].insert_many(self.df.to_dict("records"))

    def test_build_sequences_shapes(self):
        X, y = build_sequences(self.df)
        self.assertEqual(X.shape[1], WINDOW_LEN)
        self.assertEqual(X.shape[2], 2)  # two features
        self.assertEqual(len(X), len(y))
        self.assertTrue(set(y).issubset({0, 1}))

    def test_output_file_created_and_mongo_saved(self):
        # Remove old .npz if exists
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)

        main()  # Rebuild

        # Check file saved
        self.assertTrue(os.path.exists(OUTPUT_PATH))
        data = np.load(OUTPUT_PATH)
        self.assertIn("X_entry", data)
        self.assertIn("y_entry", data)

        # Check MongoDB
        self.assertGreater(db["sequence_entry"].count_documents({}), 0)
        self.assertGreater(db["sequence_exit"].count_documents({}), 0)


if __name__ == "__main__":
    unittest.main()
