# app.py

import streamlit as st
import pandas as pd
from core.technical_analysis import get_technical_summary
from core.data_loader import load_tickers_from_csv
from core.evaluator import evaluate_multiple
from core.macro_filter import get_macro_for_country

from core.macro_filter import load_macro_data

# --- Page setup ---
st.set_page_config(page_title="Trading Signal App", layout="centered")
st.title("📈 AI Trading Signal Generator (Fundamentals v0.1)")

# --- Macroeconomic selection ---
macro_df = load_macro_data()
country = st.selectbox("🌍 Select macro environment", macro_df["Country"].unique())
macro = get_macro_for_country(country)

with st.expander("📊 View Macroeconomic Data Being Applied"):
    st.write(f"**Country:** {country}")
    st.write(f"- GDP Growth: {macro['GDP_Growth']}%")
    st.write(f"- Inflation: {macro['Inflation']}%")
    st.write(f"- Interest Rate: {macro['Interest_Rate']}%")

# --- File upload or manual selection ---
st.subheader("📎 Upload Ticker List (CSV format)")
uploaded_file = st.file_uploader("Upload a CSV file with one column: Ticker", type=["csv"])

tickers = []

if uploaded_file:
    tickers = load_tickers_from_csv(uploaded_file)
else:
    tickers = st.multiselect(
        "Or select tickers manually:",
        options=["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "AMZN", "META", "NFLX", "JPM", "V"],
        default=["AAPL", "MSFT"]
    )

# --- Run evaluation ---
if tickers:
    results = evaluate_multiple(tickers, macro)
    df = pd.DataFrame(results)

    st.subheader("📋 Evaluation Results")
    st.dataframe(df)
    st.caption("Showing results for selected tickers based on current thresholds.")

    # --- Technical Analysis ---
    st.subheader("📉 Technical Analysis (TradingView)")

    for ticker in tickers:
        tech = get_technical_summary(ticker)

        st.markdown(f"### {ticker}")
        st.write(f"**Summary Recommendation:** {tech.get('Summary')}")

        with st.expander("View detailed indicators"):
            st.write({
                "RSI": tech.get("RSI"),
                "MACD": tech.get("MACD"),
                "ATR": tech.get("ATR"),
                "ADX": tech.get("ADX"),
                "Stochastic %K": tech.get("Stochastic_K"),
                "EMA 20": tech.get("EMA_20"),
                "EMA 50": tech.get("EMA_50"),
                "SMA 20": tech.get("SMA_20"),
                "SMA 200": tech.get("SMA_200"),
            })
