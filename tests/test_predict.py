# tests/test_predict.py
"""
Integration test for the prediction script.
This test ensures:
- The prediction function runs without error for at least one ticker
- The output DataFrame is not empty and includes expected columns
"""

from scripts.prediction.predict import run_prediction

def test_run_prediction_outputs_results():
    results = run_prediction()
    assert results is not None, "No results returned"
    assert not results.empty, "Prediction results are empty"
    assert "ticker" in results.columns, "Missing 'ticker' column in results"
    assert "prediction" in results.columns, "Missing 'prediction' column in results"
