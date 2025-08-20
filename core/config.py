"""
core/config.py

Contains thresholds and weights used in fundamental filtering and scoring logic.

These are used by the rules engine and evaluator modules to:
- Filter companies based on financial health
- Score stocks for prioritization before model prediction

All values are relative to percentage or absolute market metrics.
"""

# Thresholds for rule-based evaluation
THRESHOLDS = {
    "EPS_Growth": 10,         # % YoY
    "Revenue_Growth": 5,      # % YoY
    "Debt/Equity": 100,       # % Max
    "P/E": 25,                # Acceptable if <= 25
    "Market_Cap": 500,        # Minimum $500 million
    "Avg_Volume": 500_000     # Minimum 500k daily shares
}

# Relative importance of each metric (total should add to 100)
METRIC_WEIGHTS = {
    "EPS_Growth": 30,
    "Revenue_Growth": 25,
    "Debt/Equity": 15,
    "P/E": 15,
    "Market_Cap": 10,
    "Avg_Volume": 5
}
