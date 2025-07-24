# core/config.py

# Thresholds for rule-based evaluation
THRESHOLDS = {
    "EPS_Growth": 10,         # % YoY
    "Revenue_Growth": 5,      # % YoY
    "Debt/Equity": 100,       # %
    "P/E": 25,                # Acceptable if <= 25
    "Market_Cap": 500,        # Minimum $500 million
    "Avg_Volume": 500_000     # Minimum 500k daily shares
}

# Relative importance of each metric (out of 100 total)
METRIC_WEIGHTS = {
    "EPS_Growth": 30,
    "Revenue_Growth": 25,
    "Debt/Equity": 15,
    "P/E": 15,
    "Market_Cap": 10,
    "Avg_Volume": 5
}
