import math
from datetime import timezone
import pandas as pd
import yfinance as yf
from pymongo import UpdateOne
from core.db import db

TICKERS_CSV = "tickers.csv"
MAX_QUARTERS = 12  # keep it light: ~3 years

def _to_utc(dt):
    # yfinance returns pandas Timestamp (tz-naive). Normalize to UTC midnight.
    return dt.to_pydatetime().replace(tzinfo=timezone.utc)

def _clean(x):
    # Convert NaN-like to None; keep ints/floats as Python scalars
    if x is None:
        return None
    try:
        if isinstance(x, (pd.Series,)):
            x = x.item()
    except Exception:
        pass
    if isinstance(x, (float, int)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    return None

def _row_value(df: pd.DataFrame, row_name: str, col) -> float | None:
    try:
        if df is None or df.empty:
            return None
        if row_name not in df.index:
            return None
        return _clean(df.loc[row_name, col])
    except Exception:
        return None

def build_quarter_docs(ticker: str, tkr: yf.Ticker):
    # Pull frames
    q_fin   = tkr.quarterly_financials   # rows by line item, columns by period end
    q_bs    = tkr.quarterly_balance_sheet
    q_cf    = tkr.quarterly_cashflow
    q_earn  = tkr.quarterly_earnings     # columns: Revenue, Earnings

    # Use the union of all period-end columns we can find, most recent first
    cols = []
    for df in [q_fin, q_bs, q_cf]:
        if df is not None and not df.empty:
            cols.extend(list(df.columns))
    if not cols and (q_earn is not None and not q_earn.empty):
        cols.extend(list(q_earn.index))  # earnings uses index as period ends
    # De-duplicate and sort desc
    cols = sorted(set(cols), reverse=True)
    if not cols:
        return []

    cols = cols[:MAX_QUARTERS]

    docs = []
    for col in cols:
        # Period end
        period_dt = _to_utc(col if hasattr(col, "to_pydatetime") else pd.Timestamp(col))

        # Financial statement values
        total_revenue   = _row_value(q_fin, "Total Revenue", col)
        gross_profit    = _row_value(q_fin, "Gross Profit", col)
        operating_income= _row_value(q_fin, "Operating Income", col)
        net_income      = _row_value(q_fin, "Net Income", col)

        total_assets    = _row_value(q_bs, "Total Assets", col)
        total_liab      = _row_value(q_bs, "Total Liab", col)
        equity          = _row_value(q_bs, "Total Stockholder Equity", col)
        total_debt      = _row_value(q_bs, "Total Debt", col) or _row_value(q_bs, "Short Long Term Debt", col)

        op_cash_flow    = _row_value(q_cf, "Total Cash From Operating Activities", col)
        capex           = _row_value(q_cf, "Capital Expenditures", col)
        free_cash_flow  = None
        if op_cash_flow is not None and capex is not None:
            free_cash_flow = op_cash_flow + capex  # capex is negative in Yahoo—this yields FCF

        # Earnings (Yahoo’s quarterly_earnings is simpler: Revenue & Earnings)
        earn_rev = None
        earn_net = None
        if q_earn is not None and not q_earn.empty and col in q_earn.index:
            earn_rev = _clean(q_earn.loc[col, "Revenue"]) if "Revenue" in q_earn.columns else None
            earn_net = _clean(q_earn.loc[col, "Earnings"]) if "Earnings" in q_earn.columns else None

        # Derived ratios (guard divide-by-zero)
        debt_to_equity = None
        if equity not in (None, 0):
            debt_to_equity = (total_debt or 0.0) / equity

        roe = None
        if equity not in (None, 0) and net_income is not None:
            roe = net_income / equity

        profit_margin = None
        base_rev = total_revenue if total_revenue is not None else earn_rev
        if base_rev not in (None, 0) and (net_income is not None or earn_net is not None):
            profit_margin = (net_income if net_income is not None else earn_net) / base_rev

        docs.append({
            "ticker": ticker,
            "period": period_dt,  # unique key with ticker
            "financials": {
                "totalRevenue": total_revenue,
                "grossProfit": gross_profit,
                "operatingIncome": operating_income,
                "netIncome": net_income,
            },
            "balanceSheet": {
                "totalAssets": total_assets,
                "totalLiabilities": total_liab,
                "totalEquity": equity,
                "totalDebt": total_debt,
            },
            "cashflow": {
                "operatingCashFlow": op_cash_flow,
                "capitalExpenditures": capex,
                "freeCashFlow": free_cash_flow,
            },
            "earnings": {
                "revenue": earn_rev,
                "earnings": earn_net,
            },
            "ratios": {
                "debtToEquity": debt_to_equity,
                "roe": roe,
                "profitMargin": profit_margin,
            },
            "source": "yfinance",
        })
    return docs

def upsert_quarters(ticker: str, docs: list[dict]):
    if not docs:
        print(f"[{ticker}] No quarterly data found.")
        return
    ops = [
        UpdateOne(
            {"ticker": d["ticker"], "period": d["period"]},
            {"$set": d},
            upsert=True
        )
        for d in docs
    ]
    res = db.fundamentals.bulk_write(ops, ordered=False)
    print(f"[{ticker}] fundamentals upserted={res.upserted_count}, modified={res.modified_count}")

def main():
    tickers = pd.read_csv(TICKERS_CSV)["Ticker"].dropna().tolist()
    for ticker in tickers:
        try:
            print(f"\nFetching ~{MAX_QUARTERS} quarters for {ticker}...")
            tkr = yf.Ticker(ticker)
            docs = build_quarter_docs(ticker, tkr)
            upsert_quarters(ticker, docs)
        except Exception as e:
            print(f"❌ {ticker}: {e}")
    print("\n✅ Fundamentals backfill complete.")

if __name__ == "__main__":
    main()
