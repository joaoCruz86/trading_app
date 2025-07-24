# core/config.py

# Ideal thresholds for each metric
THRESHOLDS = {
    "P/E": 25,
    "EPS_Growth": 10,
    "Debt/Equity": 50,
    "Revenue_Growth": 5
}

# Importance weight for each metric (must sum to ~100)
METRIC_WEIGHTS = {
    "P/E": 15,
    "EPS_Growth": 35,
    "Debt/Equity": 20,
    "Revenue_Growth": 30
}
