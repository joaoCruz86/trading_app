# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from core.technical_analysis import get_technical_summary, fetch_price_data, compute_technical_indicators
from core.data_loader import load_tickers_from_csv
from core.evaluator import evaluate_multiple
from core.macro_filter import get_macro_for_country, load_macro_data
from scripts.prediction.tabular_signal_service import run_signal_service
from scripts.prediction.sequence_predict import run_sequence_prediction


# --- Cache wrappers for expensive ops ---
@st.cache_data(show_spinner=True)
def cached_fetch_price_data(ticker, start_date, end_date):
    return fetch_price_data(ticker, period=None, interval="1d", start_date=start_date, end_date=end_date)

@st.cache_data(show_spinner=True)
def cached_compute_technical_indicators(df):
    return compute_technical_indicators(df)

# --- Page setup ---
st.set_page_config(page_title="AI Trading Signal App", layout="wide")
st.title("📈 AI Trading Signal App (Multi-Layered)")
st.caption("Pre-Screening + Layer 1 (Tabular) + Layer 2 (Sequence) + Technical Indicators")

# --- Global Inputs ---
macro_df = load_macro_data()
country = st.selectbox("🌍 Select macro environment", macro_df["Country"].unique())
macro = get_macro_for_country(country)

st.subheader("📎 Upload or Select Tickers")
uploaded_file = st.file_uploader("Upload a CSV with one column: Ticker", type=["csv"])

if uploaded_file:
    tickers = load_tickers_from_csv(uploaded_file)
else:
    tickers = st.multiselect(
        "Or select tickers manually:",
        options=["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "AMZN", "META", "NFLX", "JPM", "V"],
        default=["AAPL", "MSFT"]
    )

st.subheader("📅 Select Date Range for Charts")
default_end = datetime.today()
default_start = default_end - timedelta(days=365 * 3)

start_date = st.date_input("Start Date", default_start)
end_date = st.date_input("End Date", default_end)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# --- Tabs ---
tab_screen, tab_layer1, tab_layer2, tab_tech, tab_macro = st.tabs([
    "Pre-Screening (Fundamentals)",
    "Layer 1: Tabular AI",
    "Layer 2: Sequence AI",
    "Technical Indicators",
    "Macroeconomics"
])

# --- Pre-Screening Tab ---
with tab_screen:
    st.subheader("🔎 Rule-Based Pre-Screening")
    if tickers:
        results = evaluate_multiple(tickers, macro)
        df = pd.DataFrame(results)
        st.dataframe(df)
    else:
        st.info("Please upload or select tickers.")

# --- Layer 1 Tabular AI ---
with tab_layer1:
    st.subheader("🤖 AI Layer 1: Tabular Model")
    if tickers:
        df_tabular = predict_tabular_signals(tickers)
        st.dataframe(df_tabular)
    else:
        st.info("Please upload or select tickers.")

# --- Layer 2 Sequence AI ---
with tab_layer2:
    st.subheader("🔁 AI Layer 2: Sequence Model")
    if tickers:
        df_seq = predict_sequence_signals(tickers)
        st.dataframe(df_seq)
    else:
        st.info("Please upload or select tickers.")

# --- Technical Analysis Tgit piusab ---
with tab_tech:
    st.subheader("📉 Technical Analysis & Charts")
    if tickers:
        for ticker in tickers:
            tech = get_technical_summary(ticker)
            if "Error" in tech:
                st.warning(f"{ticker}: {tech['Error']}")
                continue

            st.markdown(f"### {ticker}")
            st.write(f"**Date:** {tech.get('Date', 'N/A')}")
            st.write(f"**RSI:** {tech.get('RSI')} ({tech.get('RSI_signal')})")
            st.write(f"**MACD:** {tech.get('MACD')} ({tech.get('MACD_signal')}, {tech.get('MACD_trend')})")

            df_prices = cached_fetch_price_data(ticker, start_date, end_date)
            df_ind = cached_compute_technical_indicators(df_prices)

            fig = go.Figure(data=[go.Candlestick(
                x=df_ind.index,
                open=df_ind['Open'], high=df_ind['High'],
                low=df_ind['Low'], close=df_ind['Close'], name='Candlestick')])
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['EMA_20'], name='EMA 20'))
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['SMA_200'], name='SMA 200'))
            fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload or select tickers.")

# --- Macroeconomic Tab ---
with tab_macro:
    st.subheader("🌍 Macro Filters in Use")
    st.write(f"**Country:** {country}")
    st.write(f"- GDP Growth: {macro['GDP_Growth']}%")
    st.write(f"- Inflation: {macro['Inflation']}%")
    st.write(f"- Interest Rate: {macro['Interest_Rate']}%")
