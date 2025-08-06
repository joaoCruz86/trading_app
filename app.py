# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from core.technical_analysis import get_technical_summary, fetch_price_data, compute_technical_indicators
from core.data_loader import load_tickers_from_csv
from core.evaluator import evaluate_multiple
from core.macro_filter import get_macro_for_country, load_macro_data

# --- Cache wrappers for expensive ops ---
@st.cache_data(show_spinner=True)
def cached_fetch_price_data(ticker, start_date, end_date):
    stock = fetch_price_data(ticker, period=None, interval="1d", start_date=start_date, end_date=end_date)
    return stock

@st.cache_data(show_spinner=True)
def cached_compute_technical_indicators(df):
    return compute_technical_indicators(df)

# --- Page setup ---
st.set_page_config(page_title="Trading Signal App", layout="centered")
st.title("📈 AI Trading Signal Generator (Fundamentals v0.1)")

# --- Global Inputs ---
macro_df = load_macro_data()
country = st.selectbox("🌍 Select macro environment", macro_df["Country"].unique())
macro = get_macro_for_country(country)

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

st.subheader("📅 Select Date Range for Charts")
default_end = datetime.today()
default_start = default_end - timedelta(days=365*3)  # 3 years default

start_date = st.date_input("Start Date", default_start)
end_date = st.date_input("End Date", default_end)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# --- Tabs ---
tab_eval, tab_tech, tab_macro = st.tabs(["Evaluation", "Technical Analysis", "Macroeconomics"])

# --- Evaluation Tab ---
with tab_eval:
    if tickers:
        results = evaluate_multiple(tickers, macro)
        df = pd.DataFrame(results)

        st.subheader("📋 Evaluation Results")
        st.dataframe(df)
        st.caption("Showing results for selected tickers based on current thresholds.")
    else:
        st.info("Select or upload tickers to see evaluation results.")

# --- Technical Analysis Tab ---
with tab_tech:
    if tickers:
        st.subheader("📉 Technical Analysis & Visualization")

        for ticker in tickers:
            tech = get_technical_summary(ticker)

            if "Error" in tech:
                st.warning(f"Error fetching technical data for {ticker}: {tech['Error']}")
                continue

            st.markdown(f"### {ticker}")
            st.write(f"**Date:** {tech.get('Date', 'N/A')}")
            st.write(f"**RSI:** {tech.get('RSI')} ({tech.get('RSI_signal')})")
            st.write(f"**MACD:** {tech.get('MACD')} (Signal: {tech.get('MACD_signal')}, Trend: {tech.get('MACD_trend')})")
            st.write(f"**EMA 20:** {tech.get('EMA_20')}")
            st.write(f"**EMA 50:** {tech.get('EMA_50')}")
            st.write(f"**SMA 20:** {tech.get('SMA_20')}")
            st.write(f"**SMA 200:** {tech.get('SMA_200')}")
            st.write(f"**Bollinger Bands Upper:** {tech.get('BB_upper')}")
            st.write(f"**Bollinger Bands Lower:** {tech.get('BB_lower')}")

            df_prices = cached_fetch_price_data(ticker, start_date=start_date, end_date=end_date)
            df_indicators = cached_compute_technical_indicators(df_prices)

            # Candlestick chart with overlays
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_indicators.index,
                open=df_indicators['Open'],
                high=df_indicators['High'],
                low=df_indicators['Low'],
                close=df_indicators['Close'],
                name='Candlestick'
            )])

            fig_candle.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['EMA_20'], mode='lines', name='EMA 20'))
            fig_candle.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['EMA_50'], mode='lines', name='EMA 50'))
            fig_candle.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['SMA_200'], mode='lines', name='SMA 200'))

            fig_candle.add_trace(go.Scatter(
                x=df_indicators.index, y=df_indicators['BB_upper'], line=dict(color='rgba(0,0,255,0.2)'),
                name='BB Upper', fill=None))
            fig_candle.add_trace(go.Scatter(
                x=df_indicators.index, y=df_indicators['BB_lower'], line=dict(color='rgba(0,0,255,0.2)'),
                name='BB Lower', fill='tonexty'))

            fig_candle.update_layout(title=f"{ticker} Price Candlestick with Indicators", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_candle, use_container_width=True)

            # RSI chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['RSI'], mode='lines', name='RSI'))
            fig_rsi.add_hline(y=70, line_dash='dash', line_color='red')
            fig_rsi.add_hline(y=30, line_dash='dash', line_color='green')
            fig_rsi.update_layout(title=f"{ticker} RSI", xaxis_title="Date", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD'], mode='lines', name='MACD'))
            fig_macd.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD_signal'], mode='lines', name='Signal'))
            fig_macd.update_layout(title=f"{ticker} MACD", xaxis_title="Date", yaxis_title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True)

    else:
        st.info("Select or upload tickers to see technical analysis.")

# --- Macroeconomics Tab ---
with tab_macro:
    st.subheader("🌍 Macroeconomic Data Being Applied")
    st.write(f"**Country:** {country}")
    st.write(f"- GDP Growth: {macro['GDP_Growth']}%")
    st.write(f"- Inflation: {macro['Inflation']}%")
    st.write(f"- Interest Rate: {macro['Interest_Rate']}%")
