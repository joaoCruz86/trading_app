# Project Overview

This project is a two-layer AI-powered stock trading assistant. It uses fundamental and technical data to identify strong entry/exit signals based on a combination of tabular and sequence modeling.

---

## 🧠 System Architecture

**Layer 1: Tabular Screening (Random Forest + LightGBM)**  - ARCHIVED!!!
- Consumes structured features from price, fundamentals, and macroeconomic indicators.
- Predicts potential "buy" signals using two parallel models:  
  - Random Forest  
  - LightGBM  
- Models trained using historical price/fundamental data.

**Layer 2: Sequence-Based Confidence Validation (LSTM)**  
- Consumes recent time series sequences (e.g. last 60 days of indicators).
- Two LSTM models:  
  - Entry confidence model  
  - Exit confidence model  
- Used to validate tabular predictions.

---

## ⚙️ Execution Flow

```text
1. Run data ingestion: backfill_prices.py
2. Build tabular training dataset: build_training_dataset.py
3. Train tabular models: train_model.py --model both
4. Build sequence dataset: build_sequence_dataset.py
5. Train LSTM models: train_sequence_models.py
6. Run signal generation: tabular_signal_service.py
