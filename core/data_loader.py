# core/data_loader.py

import pandas as pd

def load_tickers_from_csv(file):
    """
    Expects a CSV with a 'Ticker' column.
    Returns a cleaned list of unique tickers.
    """
    try:
        df = pd.read_csv(file)
        tickers = df["Ticker"].dropna().unique().tolist()
        return tickers
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []
