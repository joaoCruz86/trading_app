# core/macro_filter.py

def get_mock_macro_data():
    """
    Returns mock macroeconomic data.
    Replace later with real data from APIs (e.g. FRED, World Bank).
    """
    return {
        "GDP_Growth": 2.1,        # in percent
        "Inflation": 4.2,         # in percent
        "Interest_Rate": 3.5      # in percent
    }

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
