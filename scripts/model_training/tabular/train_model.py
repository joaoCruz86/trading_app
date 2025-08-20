# scripts/model_training/train_model.py
"""
Train and load both Random Forest and LightGBM models for tabular signals.
Used as a unified interface for the tabular prediction layer.
"""

import argparse
import pandas as pd

from scripts.model_training.tabular.train_random_forest import train_random_forest, load_random_forest
from scripts.model_training.tabular.train_lightgbm import train_lightgbm, load_lightgbm
from core.db import db


def train_tabular_models():
    """
    Train both LightGBM and Random Forest models using latest data from MongoDB.
    """
    df = pd.DataFrame(list(db["latest"].find()))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    
    train_lightgbm(df)
    train_random_forest(df)


def load_tabular_models():
    """
    Load trained LightGBM and Random Forest models.

    Returns:
        tuple: (lightgbm_model, random_forest_model)
    """
    model_lgb = load_lightgbm()
    model_rf = load_random_forest()
    return model_lgb, model_rf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", help="Choose: random_forest | lightgbm | both")
    args = parser.parse_args()

    if args.model == "random_forest":
        train_random_forest()
    elif args.model == "lightgbm":
        train_lightgbm()
    elif args.model == "both":
        train_tabular_models()
    else:
        raise ValueError("❌ Unsupported model. Use 'random_forest', 'lightgbm', or 'both'")


if __name__ == "__main__":
    main()
