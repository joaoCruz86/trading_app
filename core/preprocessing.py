# core/preprocessing.py

import pandas as pd

def prep_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a feature DataFrame for prediction by:
    - Removing non-numeric columns
    - Filling missing values
    - Converting to float
    """
    drop_cols = [c for c in X.columns if c.lower() in {"_id", "ticker", "symbol", "date", "timestamp", "ds"}]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=["number", "bool"]).copy()
    X = X.ffill().bfill().fillna(0)
    return X.astype(float)
