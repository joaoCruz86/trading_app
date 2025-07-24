# core/rules_engine.py

import random
import matplotlib.pyplot as plt
import yfinance as yf
from core.config import METRIC_WEIGHTS
from core.config import THRESHOLDS
from core.macro_filter import get_mock_macro_data, is_macro_environment_favorable


def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "P/E": info.get("trailingPE", 0),
        "EPS_Growth": info.get("earningsQuarterlyGrowth", 0) * 100,
        "Debt/Equity": info.get("debtToEquity", 0),
        "Revenue_Growth": info.get("revenueGrowth", 0) * 100,
    }


def evaluate_stock(ticker):
    data = get_fundamentals(ticker)
    signal = "HOLD"

    # Macro conditions
    macro = get_mock_macro_data()
    macro_ok = is_macro_environment_favorable(macro)

    # 🚫 SELL conditions (strong negative fundamentals)
    if (
        data["EPS_Growth"] < -5
        or data["Revenue_Growth"] < 0
        or data["Debt/Equity"] > 150
    ):
        signal = "SELL 🚫"

    # ✅ BUY conditions (only if macro is favorable)
    elif (
        data["EPS_Growth"] > 10
        and data["Revenue_Growth"] > 5
        and data["Debt/Equity"] < 100
    ):
        signal = "BUY ✅" if macro_ok else "HOLD 🧯"

    # 🟡 Otherwise HOLD (default)
    else:
        signal = "HOLD"

    score = compute_confidence_score(data)

    return {"signal": signal, "score": score, "details": data}


def compute_confidence_score(data):
    """
    Takes in stock fundamentals and computes a weighted score (0–100).
    Higher means the stock aligns well with ideal fundamentals.
    """
    total_score = 0

    for metric, weight in METRIC_WEIGHTS.items():
        value = data.get(metric, 0)
        ideal = THRESHOLDS.get(metric)

        if metric == "Debt/Equity":
            # Lower is better
            score = (
                100 if value <= ideal else max(0, 100 - ((value - ideal) / ideal) * 100)
            )
        elif metric in ["P/E"]:
            # Lower is better (up to a point)
            score = (
                100 if value <= ideal else max(0, 100 - ((value - ideal) / ideal) * 50)
            )
        else:
            # Higher is better
            score = 100 if value >= ideal else max(0, (value / ideal) * 100)

        total_score += score * (weight / 100)

    return round(total_score, 1)
