# scripts/predict.py
"""
Load trained model and make predictions for a ticker.
- Uses technical indicators as features
- Requires models/trading_model.pkl from train_model.py
"""

import joblib
import os
import pandas as pd
from core.technical_analysis import fetch_price_data, compute_technical_indicators

MODEL_PATH = "models/trading_model.pkl"

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

# --- Predict for a ticker ---
def predict_signal(ticker: str):
    """
    Get the AI prediction (BUY/HOLD/SELL) for a given ticker.
    """
    # Fetch last 90 days of prices
    df = fetch_price_data(ticker, period="3mo")
    df = compute_technical_indicators(df)
    latest = df.iloc[-1]

    features = {
        "RSI": latest["RSI"],
        "MACD": latest["MACD"],
        "MACD_signal": latest["MACD_signal"],
        "EMA_20": latest["EMA_20"],
        "EMA_50": latest["EMA_50"],
        "SMA_20": latest["SMA_20"],
        "SMA_200": latest["SMA_200"],
        "BB_upper": latest["BB_upper"],
        "BB_lower": latest["BB_lower"],
        "Close": latest["Close"],
    }

    X = pd.DataFrame([features])

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "ticker": ticker,
        "prediction": prediction,
        "probabilities": dict(zip(model.classes_, [round(p, 3) for p in proba])),
        "features": features
    }


# --- Example Usage ---
if __name__ == "__main__":
    ticker = "AAPL"
    result = predict_signal(ticker)
    print(f"\n📊 AI Prediction for {ticker}: {result['prediction']}")
    print("Probabilities:", result["probabilities"])
