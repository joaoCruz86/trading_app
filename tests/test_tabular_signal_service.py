# tests/test_tabular_signal_service.py
"""
Integration test for tabular signal service pipeline.
Ensures:
- Tabular signal service runs without error
- Final output DataFrame is non-empty
- Required columns exist in the results
"""

import os
import pandas as pd
from scripts.prediction.tabular_signal_service import run_signal_service, FINAL_OUTPUT

def test_tabular_signal_service_outputs_results():
    # Run the signal service
    run_signal_service()

    # Load the output file
    assert os.path.exists(FINAL_OUTPUT), "Final signals CSV file was not created"
    df = pd.read_csv(FINAL_OUTPUT)

    # Basic assertions
    assert not df.empty, "Final signals output is empty"
    for col in ["ticker", "tabular_score", "entry_confidence", "exit_confidence", "decision"]:
        assert col in df.columns, f"Missing column: {col}"
