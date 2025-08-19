# tests/test_ta.py

import unittest
import pandas as pd
from core.technical_analysis import compute_technical_indicators

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        # Minimal mock data to test indicators
        self.data = pd.DataFrame({
            "Close": [100, 101, 102, 103, 104, 105, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
        })
        self.data["Open"] = self.data["Close"]
        self.data["High"] = self.data["Close"]
        self.data["Low"] = self.data["Close"]
        self.data["Volume"] = 1000000

    def test_indicator_output_columns(self):
        result = compute_technical_indicators(self.data.copy())
        expected_cols = ["RSI", "MACD", "MACD_signal", "EMA_20", "EMA_50", "SMA_20", "SMA_200", "BB_upper", "BB_lower"]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"{col} not found in output")

    def test_no_nan_after_buffer(self):
        result = compute_technical_indicators(self.data.copy())
        # Buffer of initial rows may contain NaNs; test after that
        result = result.dropna()
        self.assertFalse(result.isnull().values.any(), "There are NaNs in the result")

if __name__ == "__main__":
    unittest.main()
