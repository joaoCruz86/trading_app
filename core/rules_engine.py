# core/rules_engine.py

from core.config import THRESHOLDS, METRIC_WEIGHTS
from core.fundamentals import get_fundamentals
from core.macro_filter import is_macro_environment_favorable

def evaluate_stock(ticker, macro):
    data = get_fundamentals(ticker)
    score = 0
    total_weight = sum(METRIC_WEIGHTS.values())

    # Liquidity Filter: Market Cap (in millions) and Avg Volume
    if data.get("Market_Cap", 0) < 500 or data.get("Avg_Volume", 0) < 500_000:
        return {
            "signal": "❌ TOO SMALL",
            "confidence": "N/A",
            "details": data
        }

    # Apply thresholds and weight-based scoring
    for metric, value in data.items():
        threshold = THRESHOLDS.get(metric)
        weight = METRIC_WEIGHTS.get(metric, 1)

        if threshold is not None:
            if metric in ["Debt/Equity", "P/E"]:  # Lower is better
                if value <= threshold:
                    score += weight
            else:  # Higher is better
                if value >= threshold:
                    score += weight

    confidence = round((score / total_weight) * 100)

    # Macro suppression
    if not is_macro_environment_favorable(macro):
        signal = "HOLD 🧯"
    else:
        signal = (
            "BUY ✅" if confidence >= 70 else
            "SELL ❌" if confidence < 40 else
            "HOLD 🟡"
        )

    return {
        "signal": signal,
        "confidence": f"{confidence}%",
        "details": data
    }
