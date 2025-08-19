# scripts/train_model.py
"""
Dispatcher to train different ML models for trading signals.
Usage:
  python train_model.py --model random_forest
  python train_model.py --model lightgbm
"""

import argparse
from scripts.model_training.train_random_forest import train_random_forest
from scripts.model_training.train_lightgbm import train_lightgbm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest")
    args = parser.parse_args()

    if args.model == "random_forest":
        train_random_forest()
    elif args.model == "lightgbm":
        train_lightgbm()
    else:
        raise ValueError("❌ Unsupported model. Use 'random_forest' or 'lightgbm'")
