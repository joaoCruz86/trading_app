# scripts/prediction/today_tabular_signals.py
"""
Score today's tabular BUY/HOLD signals using trained tabular models.

What it does
------------
- For each ticker in MongoDB 'prices':
  - Pull last N years of history (default ~3y via date filter)
  - Recompute technical indicators across that window
  - Build *today's* (latest row) feature vector
- Load trained LightGBM + RandomForest tabular models
- Align today's features to each model's expected columns (no mismatch)
- Predict probabilities and ensemble (mean of LGBM & RF)
- Decide BUY/HOLD using a threshold
- Save results to outputs/today_tabular_signals.csv and (optional) MongoDB 'signals_tabular'

Notes
-----
- This script **does not** call the sequence layer.
- If your models were trained with a specific feature set, we auto-align
  to `model.feature_names_in_` to ensure the same columns/order.
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.db import db
from core.technical_analysis import compute_technical_indicators
from core.macro_filter import is_macro_environment_favorable, get_macro_for_country
from scripts.model_training.tabular.train_model import load_tabular_models


# ------------ Config ------------
YEARS_BACK = 3                 # how many years of history to use for indicators
CONFIDENCE_THRESHOLD = 0.60    # BUY threshold on ensemble probability
OUTPUT_CSV = "outputs/today_tabular_signals.csv"
SAVE_TO_DB = True              # also store to MongoDB 'signals_tabular'
COUNTRY_FOR_MACRO = "USA"      # pass to your macro filter


def _load_prices_from_db(ticker: str, years_back: int = YEARS_BACK) -> pd.DataFrame:
    """Load last N years of daily OHLCV for a ticker from MongoDB 'prices'."""
    since = datetime.now(timezone.utc) - timedelta(days=365 * years_back)
    cursor = db["prices"].find(
        {"ticker": ticker, "date": {"$gte": since}},
        projection={"_id": 0, "ticker": 1, "date": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ).sort("date", 1)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df

    # Normalize columns to match your training prep
    df.rename(
        columns={
            "ticker": "Ticker",
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()
    return df


def _build_today_row(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Given a history DataFrame (index=Date, cols include OHLCV),
    compute indicators across the window and return a single-row DataFrame
    for the latest date with engineered features (incl. Macro_OK, optional date parts).
    """
    if df_hist.empty:
        return df_hist

    # Daily grid & forward-fill to avoid gaps, then compute indicators (your helper)
    df = df_hist.asfreq("D").ffill()
    df = compute_technical_indicators(df)

    # Add macro flag (same as training)
    macro = get_macro_for_country(COUNTRY_FOR_MACRO)
    df["Macro_OK"] = is_macro_environment_favorable(macro)

    # Latest available row (might be yesterday if today is not a trading day)
    latest = df.iloc[[-1]].copy()  # keep as DataFrame

    # Add friendly date column + date-derived numerics in case models expect them
    latest["date"] = latest.index.tz_convert("UTC").tz_localize(None) if latest.index.tz else latest.index
    latest["day"] = latest["date"].dt.day.astype(float)
    latest["month"] = latest["date"].dt.month.astype(float)
    latest["day_of_week"] = latest["date"].dt.dayofweek.astype(float)
    latest["Ticker"] = df_hist["Ticker"].iloc[-1] if "Ticker" in df_hist.columns else None

    # Keep only numeric/bool + a few IDs we may want to output
    # (We'll align to model columns later.)
    return latest.reset_index(drop=True)


def _prep_for_model(today_df: pd.DataFrame, feature_names_in: np.ndarray) -> pd.DataFrame:
    """
    Align today's single-row features to a model's expected columns.
    - Keep only numeric/bool
    - Add any missing expected columns as 0.0
    - Ensure exact order matches `feature_names_in`
    """
    X = today_df.select_dtypes(include=["number", "bool"]).copy()

    # Make sure all expected columns exist
    for col in feature_names_in:
        if col not in X.columns:
            X[col] = 0.0

    # Now reduce to exactly expected columns / order
    X = X.reindex(columns=feature_names_in).astype(float)
    return X


def main():
    os.makedirs("outputs", exist_ok=True)

    # Get tickers we actually have in 'prices'
    tickers = db["prices"].distinct("ticker")
    tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
    if not tickers:
        print("❌ No tickers found in 'prices'.")
        return

    # Load tabular models
    try:
        model_lgb, model_rf = load_tabular_models()
    except Exception as e:
        print(f"❌ Could not load tabular models: {e}")
        return

    rows_out = []
    for ticker in tickers:
        try:
            hist = _load_prices_from_db(ticker, YEARS_BACK)
            if hist.empty:
                print(f"⚠️ No price history for {ticker} — skipping.")
                continue

            today_row = _build_today_row(hist)
            if today_row.empty:
                print(f"⚠️ Could not build features for {ticker} — skipping.")
                continue

            # Align to each model's expected features
            X_lgb = _prep_for_model(today_row, getattr(model_lgb, "feature_names_in_", np.array([])))
            X_rf = _prep_for_model(today_row, getattr(model_rf, "feature_names_in_", np.array([])))

            # Predict probabilities (class 1) from each model
            proba_lgb = float(model_lgb.predict_proba(X_lgb)[:, 1][0]) if X_lgb.shape[1] else np.nan
            proba_rf = float(model_rf.predict_proba(X_rf)[:, 1][0]) if X_rf.shape[1] else np.nan

            # Ensemble (mean of available probs)
            probs = [p for p in [proba_lgb, proba_rf] if not np.isnan(p)]
            ensemble = float(np.mean(probs)) if probs else np.nan

            # Decision
            decision = "BUY" if (not np.isnan(ensemble) and ensemble >= CONFIDENCE_THRESHOLD) else "HOLD"

            # Latest date used
            latest_date = today_row["date"].iloc[0] if "date" in today_row.columns else hist.index[-1]
            latest_date_str = pd.to_datetime(latest_date).strftime("%Y-%m-%d")

            rows_out.append(
                {
                    "date": latest_date_str,
                    "ticker": ticker,
                    "proba_lightgbm": proba_lgb,
                    "proba_random_forest": proba_rf,
                    "tabular_score": ensemble,
                    "decision": decision,
                }
            )
        except Exception as e:
            print(f"⚠️ {ticker}: error while scoring — {e}")

    if not rows_out:
        print("❌ No results to write.")
        return

    out_df = pd.DataFrame(rows_out).sort_values(["decision", "tabular_score"], ascending=[False, False])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {len(out_df)} rows → {OUTPUT_CSV}")

    if SAVE_TO_DB:
        # upsert by (date, ticker)
        ops = []
        from pymongo import UpdateOne

        for r in rows_out:
            key = {"date": r["date"], "ticker": r["ticker"]}
            ops.append(UpdateOne(key, {"$set": r}, upsert=True))
        if ops:
            res = db["signals_tabular"].bulk_write(ops, ordered=False)
            print(f"🧠 Upserted into MongoDB 'signals_tabular' "
                  f"(upserted={res.upserted_count}, modified={res.modified_count})")


if __name__ == "__main__":
    main()
