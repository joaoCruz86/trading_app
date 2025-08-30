## pipeline.md

### 📈 End-to-End Model Pipeline

This project generates buy/sell/hold signals for stocks using a two-layer ML system:

---
### 🧩 Core/Foundation Layer
| Filename                | Purpose                                                                    |
| ----------------------- | -------------------------------------------------------------------------- |
| `__init__.py`           | No logic, just boilerplate.         |
| `config.py`             |  contains **global configs** (DB connections, paths, etc).      |
| `data_loader.py`        | Utility for **fetching or preprocessing data** (e.g. price, fundamentals). |
| `db.py`                 | MongoDB connection logic (`db = client[...]; db.collection[...]`).         |
| `evaluator.py`          | Likely holds evaluation metrics, scoring functions, or backtest helpers.   |
| `fundamentals.py`       | Handles parsing/cleaning of **company financial data**.                    |
| `macro_filter.py`       | Logic for filtering based on **macroeconomic indicators**.                 |
| `rules_engine.py`       | If present, for **manual rule-based signals** (pre-AI).           |
| `technical_analysis.py` | Contains all the **indicator calculations** (e.g. RSI, SMA, MACD).         |


🧪 Step 0 – Initial Screening (Rule-Based Filter)

Before running any AI models, each stock is evaluated using a rule-based engine that screens for fundamental quality and macroeconomic compatibility.

Script Location: core/rules_engine.py

✅ What This Step Does:

Pulls fundamental data (EPS growth, Revenue growth, P/E, Debt/Equity, Market Cap, Avg Volume)

Applies static thresholds and weighted scoring defined in core/config.py

Enforces liquidity filters (e.g., Market Cap ≥ $500M, Avg Volume ≥ 500K)

Modifies recommendation based on macro filters (is_macro_environment_favorable)

Outputs a discrete signal:

BUY ✅ (confidence ≥ 70%)

HOLD 🟡 (confidence between 40–69%)

SELL ❌ (confidence < 40%)

TOO SMALL ❌ (fails liquidity filter)




### 🧩 Layer 1: Tabular AI Models - ARCHIVED!!!

* **Input:** Raw historical price data + technical indicators + fundamentals + macro filters.
* **Target:** Binary classification (buy signal if +5% return in 20 days).
* **Output:** Screener scores for each ticker.

**Steps:**

1. `build_training_dataset.py`: Generates labeled dataset from historical data.
2. `train_model.py`: Trains both Random Forest and LightGBM models.
3. `tabular_signal_service.py`: Applies trained models on latest data to shortlist potential buys.

---

### 🧠 Layer 2: Sequence AI Models

* **Input:** 60-day historical sequence per shortlisted ticker.
* **Output:** Entry and Exit confidence scores (0–1).

**Steps:**

1. `build_sequence_dataset.py`: Builds rolling sequences from historical training data.
2. `train_sequence_models.py`: Trains entry and exit LSTM models.
3. `run_signal_service.py`: Combines tabular and sequence models to generate final signals.

---

### 🪙 Final Output

* `final_signals.csv` and MongoDB collection `signals`
* Contains:

  * Ticker
  * Tabular Score
  * Entry Confidence
  * Exit Confidence
  * Final Decision: BUY, HOLD, EXIT

---

## database.md

### 📦 MongoDB Collections

| Collection       | Purpose                                                          |
| ---------------- | ---------------------------------------------------------------- |
| `prices`         | Raw historical OHLCV + indicators per ticker                     |
| `fundamentals`   | Quarterly financial data for tickers                             |
| `macro`          | Macroeconomic conditions (e.g. interest rates, CPI)              |
| `training`       | Tabular training dataset (output of `build_training_dataset.py`) |
| `sequence_entry` | Sequence dataset for entry model                                 |
| `sequence_exit`  | Sequence dataset for exit model                                  |
| `latest`         | Most recent feature snapshot for all tickers                     |
| `signals`        | Final signal results (BUY, HOLD, EXIT)                           |

**Notes:**

* Data is inserted with upserts to avoid duplicates.
* Most scripts use `core.db` for MongoDB interaction.

---

## setup.md

### 🛠️ Local Setup Instructions

#### 1. Clone the Repository

```bash
git clone <repo-url>
cd trading_app
```

#### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
./venv/Scripts/Activate.ps1  # PowerShell on Windows
```

#### 3. Install Requirements

```bash
pip install -r requirements.txt
```

#### 4. Start MongoDB

* Ensure MongoDB service is running locally on default port (27017)
* Data will be written to the default database set in `core/db.py`

#### 5. Backfill Data (Once)

```bash
python scripts/data_fetch/backfill_prices.py
python scripts/data_fetch/backfill_fundamentals.py
python scripts/data_fetch/backfill_macro.py
```

#### 6. Build & Train

```bash
python scripts/data_prep/build_training_dataset.py
python scripts/model_training/train_model.py --model both
python scripts/model_training/sequence/train_sequence_models.py
```

#### 7. Run Signal Generator

```bash
python scripts/prediction/run_signal_service.py
```