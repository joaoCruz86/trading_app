# scripts/model_training/train_model.py
"""
Train and load both Random Forest and LightGBM models for tabular signals.
"""

import argparse
import pandas as pd
from datetime import datetime
from scripts.model_training.tabular.train_random_forest import train_random_forest, load_random_forest
from scripts.model_training.tabular.train_lightgbm import train_lightgbm, load_lightgbm
from core.db import db

def _timestamped_model_path(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"models/{base_name}_{timestamp}.pkl"

def train_tabular_models():
    df = pd.DataFrame(list(db["training"].find()))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    lgb_path = _timestamped_model_path("lightgbm_model")
    model_lgb, metrics_lgb, report_lgb, cm_lgb, saved_lgb = train_lightgbm(df, model_path=lgb_path)
    print("\n📊 LightGBM Classification Report:\n", report_lgb)
    print("✅ LightGBM Metrics:", metrics_lgb)
    print("✅ LightGBM Confusion Matrix:\n", cm_lgb)
    print("✅ LightGBM model saved to:", saved_lgb)

    rf_path = _timestamped_model_path("random_forest_model")
    model_rf, metrics_rf, saved_rf = train_random_forest(df, model_path=rf_path)
    print("\n✅ Random Forest Metrics:", metrics_rf)
    print("✅ Random Forest model saved to:", saved_rf)

def load_tabular_models():
    model_lgb = load_lightgbm()
    model_rf = load_random_forest()
    return model_lgb, model_rf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", help="Choose: random_forest | lightgbm | both")
    args = parser.parse_args()

    if args.model == "random_forest":
        rf_path = _timestamped_model_path("random_forest_model")
        model_rf, metrics_rf, saved_rf = train_random_forest(model_path=rf_path)
        print("\n✅ Random Forest Metrics:", metrics_rf)
        print("✅ Random Forest model saved to:", saved_rf)

    elif args.model == "lightgbm":
        lgb_path = _timestamped_model_path("lightgbm_model")
        model_lgb, metrics_lgb, report_lgb, cm_lgb, saved_lgb = train_lightgbm(model_path=lgb_path)
        print("\n📊 LightGBM Classification Report:\n", report_lgb)
        print("✅ LightGBM Metrics:", metrics_lgb)
        print("✅ LightGBM Confusion Matrix:\n", cm_lgb)
        print("✅ LightGBM model saved to:", saved_lgb)

    elif args.model == "both":
        train_tabular_models()
    else:
        raise ValueError("❌ Unsupported model. Use 'random_forest', 'lightgbm', or 'both'")

if __name__ == "__main__":
    main()
