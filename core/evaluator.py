# core/evaluator.py

from core.rules_engine import evaluate_stock

def evaluate_multiple(tickers, macro):
    """
    Evaluates a list of stock tickers and returns a list of results.
    Each result includes the ticker, signal, confidence, and metric values.
    """
    results = []
    for t in tickers:
        result = evaluate_stock(t, macro)
        row = {
            "Ticker": t,
            "Signal": result['signal'],
            "Confidence": result['confidence'],
            **result['details']
        }
        results.append(row)
    return results
