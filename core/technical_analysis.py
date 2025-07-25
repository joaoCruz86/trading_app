# core/technical_analysis.py

from tradingview_ta import TA_Handler, Interval

def get_technical_summary(ticker, exchange="NASDAQ"):
    try:
        analysis = TA_Handler(
            symbol=ticker,
            screener="america",
            exchange=exchange,
            interval=Interval.INTERVAL_1_DAY
        ).get_analysis()

        indicators = analysis.indicators

        return {
            "Summary": analysis.summary.get("RECOMMENDATION", "N/A"),
            "RSI": indicators.get("RSI"),
            "MACD": indicators.get("MACD.macd"),
            "Stochastic_K": indicators.get("Stoch.K"),
            "ADX": indicators.get("ADX"),
            "ATR": indicators.get("ATR"),
            "EMA_20": indicators.get("EMA20"),
            "EMA_50": indicators.get("EMA50"),
            "SMA_20": indicators.get("SMA20"),
            "SMA_200": indicators.get("SMA200"),
            "BB_upper": indicators.get("BB.upper"),
            "BB_lower": indicators.get("BB.lower"),
        }

    except Exception as e:
        return {
            "Summary": "N/A",
            "Error": str(e)
        }
