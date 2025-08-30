# scripts/sequence/build_exit_sequence_dataset.py
"""
Build EXIT sequences for the sequence model (no daily padding; trading days only).

For each ticker:
1) Load prices from MongoDB.
2) Compute technical indicators + candlestick features + macro flag.
3) Create the EXIT label using future N-day return <= drop_threshold.
4) Slide a fixed-length window (window_size) with stride over the time series.
5) Save sequences into MongoDB 'exit_sequences':
   {
     ticker, start_date, end_date,
     X: list[list[float]],            # shape = [window_size, n_features]
     y: int,                          # 0/1 exit label
     dates: list[str],                # dates for each step in the window
     feature_names: list[str]
   }
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from core.db import db
from core.technical_analysis import compute_technical_indicators
from core.macro_filter import is_macro_environment_favorable, get_macro_for_country


# -----------------------------
# Config
# -----------------------------
WINDOW_SIZE = 30     # sequence length
STRIDE = 1           # step between sequences
HORIZON = 5          # lookahead days for exit label
DROP_THRESHOLD = -0.02  # <= -2% over HORIZON -> exit_target = 1

COLL_PRICES = "prices"
COLL_FUND = "fundamentals"   # optional join (ffill on available trading days only)
COLL_OUT = "exit_sequences"  # NEW collection to store sequences


# -----------------------------
# Feature engineering
# -----------------------------
def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["range"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body_to_range"] = df["body"] / df["range"]

    # Map string candle into numeric strength for sequence models
    # strong=2, medium=1, weak=0
    def classify_strength(x: float) -> int:
        if pd.isna(x):
            return 0
        if x >= 0.6:
            return 2
        if x >= 0.2:
            return 1
        return 0

    df["candle_strength"] = df["body_to_range"].apply(classify_strength)

    # Clean helpers
    df.drop(columns=["body", "range"], errors="ignore", inplace=True)
    return df


def prepare_base_frame(ticker: str) -> pd.DataFrame:
    # --- Load prices (keep trading days only; DO NOT pad to calendar days)
    prices = pd.DataFrame(list(db[COLL_PRICES].find({"ticker": ticker})))
    if prices.empty:
        raise ValueError(f"No price data found for {ticker}")

    # Normalize and clean
    prices.rename(columns={c: c.capitalize() for c in prices.columns}, inplace=True)
    prices.drop(columns=["_id"], errors="ignore", inplace=True)
    if "Date" not in prices.columns:
        raise ValueError(f"Expected 'Date' column for {ticker}")
    prices["date"] = pd.to_datetime(prices["Date"])
    prices.sort_values("date", inplace=True)
    prices.set_index("date", inplace=True)
    prices.drop(columns=["Date", "Indicators"], errors="ignore", inplace=True)

    # Technicals + Candles
    prices = compute_technical_indicators(prices)
    prices = add_candlestick_features(prices)

    # Optional fundamentals (join on existing trading days only, forward-fill along present index)
    fundamentals = pd.DataFrame(list(db[COLL_FUND].find({"ticker": ticker})))
    if not fundamentals.empty and "period" in fundamentals.columns:
        fundamentals["period"] = pd.to_datetime(fundamentals["period"], errors="coerce")
        fundamentals = fundamentals.dropna(subset=["period"])
        fundamentals = fundamentals.set_index("period") \
                                   .sort_index() \
                                   .loc[~fundamentals.index.duplicated(keep="last")]
        # Align to trading-day index only; ffill available fundamentals
        fundamentals = fundamentals.reindex(prices.index).ffill()
        prices = prices.join(fundamentals, how="left")

    # Macro flag (simple, static across rows for now)
    macro = get_macro_for_country("USA")
    prices["Macro_OK"] = int(bool(is_macro_environment_favorable(macro)))

    return prices


def label_exit(prices: pd.DataFrame, horizon: int, drop_threshold: float) -> pd.DataFrame:
    df = prices.copy()
    df["future_close"] = df["Close"].shift(-horizon)
    df["future_return"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["exit_target"] = (df["future_return"] <= drop_threshold).astype(int)
    return df


def select_feature_columns(df: pd.DataFrame) -> list:
    # Exclude non-features and known label/aux columns
    exclude = {
        "_id", "ticker", "Ticker",
        "future_close", "future_return", "exit_target",
        # raw fundamentals blobs if any
        "fundamentals", "financials", "ratios", "balanceSheet",
        "cashflow", "earnings", "source",
    }
    # Keep only numeric features for sequence models
    feats = [c for c in df.columns
             if (c not in exclude) and (pd.api.types.is_numeric_dtype(df[c]))]
    return feats


def to_sequences(df: pd.DataFrame,
                 feature_cols: list,
                 window_size: int,
                 stride: int) -> list[dict]:
    """
    Returns a list of documents to insert into MongoDB.
    Each document:
      { ticker, start_date, end_date, X, y, dates, feature_names }
    """
    docs = []
    values = df[feature_cols].values
    labels = df["exit_target"].values
    dates = df.index.to_pydatetime()

    # We require label to be defined at the END of each window
    max_start = len(df) - window_size
    for start in range(0, max_start + 1, stride):
        end = start + window_size  # exclusive
        # label aligned to last timestep in the window
        y = labels[end - 1]
        # Only keep windows where the label is defined (i.e., not created from NaN future)
        if np.isnan(y):
            continue
        # Build doc
        X = values[start:end].tolist()
        d_slice = dates[start:end]
        doc = {
            "ticker": df["ticker"].iloc[0] if "ticker" in df.columns else None,
            "start_date": d_slice[0].isoformat(),
            "end_date": d_slice[-1].isoformat(),
            "X": X,
            "y": int(y),
            "dates": [d.isoformat() for d in d_slice],
            "feature_names": feature_cols,
            "window_size": window_size,
            "horizon": HORIZON,
            "drop_threshold": DROP_THRESHOLD,
        }
        docs.append(doc)
    return docs


# -----------------------------
# Main builder
# -----------------------------
def build_exit_sequences_for_ticker(ticker: str,
                                    window_size: int = WINDOW_SIZE,
                                    stride: int = STRIDE,
                                    horizon: int = HORIZON,
                                    drop_threshold: float = DROP_THRESHOLD) -> int:
    prices = prepare_base_frame(ticker)
    prices["ticker"] = ticker  # keep metadata

    # Label for exit (computed relative to the last day of window)
    prices = label_exit(prices, horizon=horizon, drop_threshold=drop_threshold)

    # Clean NaNs/Infs in features ONLY (labels handled by window filter)
    prices.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Choose features
    feature_cols = select_feature_columns(prices)

    # Drop rows where features are all NaN
    if not feature_cols:
        raise ValueError(f"No numeric features available for {ticker}")
    feat_frame = prices[feature_cols].dropna(how="all")
    prices = prices.loc[feat_frame.index]

    # Also need label rows present
    prices = prices.dropna(subset=["exit_target"])

    if len(prices) < window_size:
        print(f"⚠️ {ticker}: not enough rows for window_size={window_size}")
        return 0

    # Build sequences
    seq_docs = to_sequences(prices, feature_cols, window_size, stride)
    if not seq_docs:
        print(f"⚠️ {ticker}: 0 sequences generated.")
        return 0

    # Write to Mongo
    coll = db[COLL_OUT]
    coll.create_index([("ticker", 1), ("start_date", 1)], name="ticker_start_idx", unique=False)
    coll.insert_many(seq_docs)
    print(f"✅ {ticker}: inserted {len(seq_docs)} sequences into '{COLL_OUT}'")
    return len(seq_docs)


if __name__ == "__main__":
    tickers = db[COLL_PRICES].distinct("ticker")
    print(f"🔍 Found tickers: {tickers}")

    # Fresh rebuild of the sequences collection
    db[COLL_OUT].drop()

    total = 0
    for tk in tickers:
        try:
            total += build_exit_sequences_for_ticker(tk)
        except Exception as e:
            print(f"⚠️ Skipped {tk}: {e}")

    print(f"🏁 Done. Total sequences inserted: {total}")
