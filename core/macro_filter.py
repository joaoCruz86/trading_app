# core/macro_filter.py
import pandas as pd

mock_macro_data = [
    {"Country": "USA", "GDP_Growth": 2.3, "Inflation": 3.1, "Interest_Rate": 4.5},
    {"Country": "Germany", "GDP_Growth": 0.5, "Inflation": 5.2, "Interest_Rate": 3.0},
    {"Country": "Japan", "GDP_Growth": 1.1, "Inflation": 2.0, "Interest_Rate": 0.1},
]

def get_macro_for_country(country_name):
    for row in mock_macro_data:
        if row["Country"] == country_name:
            return row
    return {"GDP_Growth": 0, "Inflation": 0, "Interest_Rate": 0}  # fallback


def is_macro_environment_favorable(macro):
    """
    Basic logic: economy is considered favorable if:
    - GDP growth > 1.5%
    - Inflation < 5%
    - Interest rate < 5%
    """
    return (
        macro["GDP_Growth"] > 1.5 and
        macro["Inflation"] < 5.0 and
        macro["Interest_Rate"] < 5.0
    )
def load_macro_data():
    """
    Returns the list of mock macroeconomic data as a DataFrame.
    """
    
    return pd.DataFrame(mock_macro_data)