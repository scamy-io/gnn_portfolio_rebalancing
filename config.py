"""
Configuration and constants for LSTM-GAT Portfolio Model v4 (28-Stock Universe).
Paper: "From Headlines to Holdings: Deep Learning for Smarter Portfolio Decisions"
       (Lin, Lou, Zhang, July 2025 / arXiv:2509.24144v2)
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FNSPID_PARQUET_PATH = RAW_DIR / "fnspid_news_filtered.parquet"

# --- 28-Stock Universe Across 11 GICS Sectors ---
TICKERS = {
    "AAPL": "Information Technology",
    "NVDA": "Information Technology",
    "MSFT": "Information Technology",
    "JNJ":  "Health Care",
    "TMO":  "Health Care",
    "UNH":  "Health Care",
    "JPM":  "Financials",
    "BAC":  "Financials",
    "GS":   "Financials",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD":   "Consumer Discretionary",
    "GOOGL": "Communication Services",
    "NFLX":  "Communication Services",
    "BA":   "Industrials",
    "CAT":  "Industrials",
    "GE":   "Industrials",
    "COST": "Consumer Staples",
    "PG":   "Consumer Staples",
    "KO":   "Consumer Staples",
    "VLO":  "Energy",
    "XOM":  "Energy",
    "CVX":  "Energy",
    "APD":  "Materials",
    "NEE":  "Utilities",
    "DUK":  "Utilities",
    "PLD":  "Real Estate",
    "AMT":  "Real Estate",
}
TICKER_LIST = list(TICKERS.keys())
NUM_ASSETS = len(TICKERS)

DATE_START = "2018-12-01"
DATE_END   = "2023-12-15"
LOOKBACK_R = 30
WARMUP_DAYS = LOOKBACK_R

TRAIN_START = "2019-01-15"
TRAIN_END   = "2022-01-15"
VAL_START   = "2022-01-15"
VAL_END     = "2022-07-15"
TEST_START  = "2022-07-15"
TEST_END    = "2023-12-15"

STRESS_WINDOW_START = "2023-03-08"
STRESS_WINDOW_END   = "2023-03-24"
STRESS_WINDOW_LABEL = "Mar 2023 US Regional-Banking Stress"

DYNAMIC_GRAPH_REFRESH_DAYS = 5
SECTOR_CORR_THRESHOLD = 0.5
NUM_FEATURES = 10

SEED = 42
LEARNING_RATE      = 3.98e-3
LSTM_WEIGHT_DECAY  = 3.33e-3
GAT_WEIGHT_DECAY   = 2.48e-4
FINAL_WEIGHT_DECAY = 2.69e-4
BATCH_SIZE         = 64
MAX_EPOCHS         = 40

LSTM_HIDDEN_SIZE   = 80
LSTM_NUM_LAYERS    = 1
LSTM_DROPOUT       = 0.27
LSTM_BIDIRECTIONAL = False

GAT_HIDDEN_SIZE    = 80
GAT_LAYERS         = 2
GAT_HEADS          = 1
GAT_DROPOUT        = 0.20
LEAKY_RELU_ALPHA   = 0.15

FINAL_DROPOUT      = 0.29
WEIGHT_NORM_EPS    = 1e-8


def set_seed(seed: int = SEED) -> None:
    """Set global determinism across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def filter_split(dates: pd.DatetimeIndex | pd.Series | list, split: str):
    """
    Helper function to filter dates or boolean masks according to explicit
    half-open intervals:
      train: [TRAIN_START, TRAIN_END)
      val:   [VAL_START, VAL_END)
      test:  [TEST_START, TEST_END]
    """
    ts_dates = pd.to_datetime(dates)
    train_s = pd.Timestamp(TRAIN_START)
    train_e = pd.Timestamp(TRAIN_END)
    val_s = pd.Timestamp(VAL_START)
    val_e = pd.Timestamp(VAL_END)
    test_s = pd.Timestamp(TEST_START)
    test_e = pd.Timestamp(TEST_END)

    if split == "train":
        return (ts_dates >= train_s) & (ts_dates < train_e)
    elif split == "val":
        return (ts_dates >= val_s) & (ts_dates < val_e)
    elif split == "train_val":
        return (ts_dates >= train_s) & (ts_dates < val_e)
    elif split == "test":
        return (ts_dates >= test_s) & (ts_dates <= test_e)
    else:
        raise ValueError(f"Unknown split: {split}. Choose from 'train', 'val', 'train_val', 'test'.")
