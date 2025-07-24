# core/evaluator.py

from core.rules_engine import evaluate_stock

def evaluate_multiple(tickers):
    """
    Evaluates a list of stock tickers and returns a list of results.
    Each result includes the ticker, signal, and metric values.
    """
    results = []
    for t in tickers:
        result = evaluate_stock(t)
        row = {
            "Ticker": t,
            "Signal": result['signal'],
            "Score (%)": result['score'],
            **result['details']
        }
        results.append(row)
    return results
