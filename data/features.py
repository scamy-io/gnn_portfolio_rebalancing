"""
Feature engineering and rolling feature calculations for portfolio model inputs.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "close",
    "volume",
    "log_return",
    "ann_return_1w",
    "ann_return_2w",
    "ann_return_1m",
    "rolling_vol_5d",
    "macd_1w_1m",
    "sentiment_variance",
    "weighted_sentiment",
]


def compute_ticker_features(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the 10 unscaled features for a single ticker's daily time series.
    """
    df = df_ticker.copy().sort_index()
    p = df["adjusted_close"]
    vol = df["volume"]

    df["close"] = p
    df["volume"] = vol
    df["log_return"] = np.log(p / p.shift(1)).fillna(0.0)

    r_1w = (p / p.shift(5) - 1.0).fillna(0.0)
    df["ann_return_1w"] = r_1w * (252.0 / 5.0)

    r_2w = (p / p.shift(10) - 1.0).fillna(0.0)
    df["ann_return_2w"] = r_2w * (252.0 / 10.0)

    r_1m = (p / p.shift(21) - 1.0).fillna(0.0)
    df["ann_return_1m"] = r_1m * (252.0 / 21.0)

    df["rolling_vol_5d"] = df["log_return"].rolling(window=5, min_periods=1).std(ddof=0).fillna(0.0)

    ema_5 = p.ewm(span=5, adjust=False).mean()
    ema_21 = p.ewm(span=21, adjust=False).mean()
    df["macd_1w_1m"] = (ema_5 - ema_21).fillna(0.0)

    if "sentiment_variance" not in df.columns:
        df["sentiment_variance"] = 0.0
    else:
        df["sentiment_variance"] = df["sentiment_variance"].fillna(0.0)

    if "weighted_sentiment" not in df.columns:
        df["weighted_sentiment"] = 0.0
    else:
        df["weighted_sentiment"] = df["weighted_sentiment"].fillna(0.0)

    return df[FEATURE_NAMES]


def compute_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the 10 raw features for all tickers across the panel.
    Returns DataFrame indexed by (date, ticker) with columns FEATURE_NAMES.
    """
    records = []
    tickers = panel_df.index.get_level_values("ticker").unique()

    for ticker in tickers:
        ticker_slice = panel_df.xs(ticker, level="ticker")
        feats = compute_ticker_features(ticker_slice)
        feats["ticker"] = ticker
        records.append(feats.reset_index())

    combined = pd.concat(records, ignore_index=True)
    return combined.set_index(["date", "ticker"]).sort_index()


def fit_transform_scalers(
    features_df: pd.DataFrame,
    train_start: str = config.TRAIN_START,
    train_end: str = config.TRAIN_END
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """
    Fits a StandardScaler per ticker exclusively on training samples [train_start, train_end),
    then transforms the full panel features to avoid lookahead leakage.
    """
    tickers = features_df.index.get_level_values("ticker").unique()
    scaled_records = []
    scalers = {}

    t_start = pd.Timestamp(train_start)
    t_end = pd.Timestamp(train_end)

    for ticker in tickers:
        t_df = features_df.xs(ticker, level="ticker")
        scaler = StandardScaler()

        # Training slice only: [train_start, train_end)
        train_mask = (t_df.index >= t_start) & (t_df.index < t_end)
        train_data = t_df.loc[train_mask, FEATURE_NAMES]

        scaler.fit(train_data)
        scalers[ticker] = scaler

        # Transform entire time series
        scaled_vals = scaler.transform(t_df[FEATURE_NAMES])
        scaled_ticker_df = pd.DataFrame(
            scaled_vals,
            index=t_df.index,
            columns=FEATURE_NAMES
        )
        scaled_ticker_df["ticker"] = ticker
        scaled_records.append(scaled_ticker_df.reset_index())

    scaled_all = pd.concat(scaled_records, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    return scaled_all, scalers


def build_daily_feature_tensor(
    scaled_df: pd.DataFrame,
    tickers: List[str] = config.TICKER_LIST
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Converts scaled panel DataFrame to 3D tensor of shape (N_days, 9, 10).
    """
    dates = scaled_df.index.get_level_values("date").unique().sort_values()
    n_days = len(dates)
    n_assets = len(tickers)
    n_feats = len(FEATURE_NAMES)

    tensor = np.zeros((n_days, n_assets, n_feats), dtype=np.float32)

    for i, t_name in enumerate(tickers):
        t_data = scaled_df.xs(t_name, level="ticker").reindex(dates)[FEATURE_NAMES].values
        tensor[:, i, :] = t_data

    return tensor, dates


def create_lookback_dataset(
    daily_tensor: np.ndarray,
    dates: pd.DatetimeIndex,
    raw_panel_df: pd.DataFrame,
    lookback: int = config.LOOKBACK_R,
    tickers: List[str] = config.TICKER_LIST
) -> dict:
    """
    Creates sequence samples for training and evaluation.
    For each decision date t >= lookback:
      - X_t in R^{9 x lookback x 10}
      - realized next-period return r_{t+1} in R^9
      - realized 30-day covariance Sigma_t in R^{9 x 9}
    """
    n_days, n_assets, n_feats = daily_tensor.shape
    
    # Compute daily asset simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    prices_matrix = np.zeros((n_days, n_assets), dtype=np.float64)
    for i, t_name in enumerate(tickers):
        p_series = raw_panel_df.xs(t_name, level="ticker").reindex(dates)["adjusted_close"].values
        prices_matrix[:, i] = p_series

    # Daily simple returns: (P_t - P_{t-1}) / P_{t-1}
    returns_matrix = np.zeros_like(prices_matrix)
    returns_matrix[1:] = (prices_matrix[1:] - prices_matrix[:-1]) / np.maximum(prices_matrix[:-1], 1e-8)

    sample_dates = []
    X_list = []
    next_returns_list = []
    cov_list = []

    # Usable dates: index idx from lookback - 1 to n_days - 2 (so next return idx+1 exists)
    for idx in range(lookback - 1, n_days - 1):
        d = dates[idx]
        sample_dates.append(d)

        # X_t: shape (9, lookback, 10)
        # Slicing daily_tensor[idx - lookback + 1 : idx + 1] -> (lookback, 9, 10)
        window_feats = daily_tensor[idx - lookback + 1 : idx + 1]
        # Transpose to (9, lookback, 10)
        x_t = np.transpose(window_feats, (1, 0, 2))
        X_list.append(x_t)

        # Realized next-period return r_{t+1} in R^9
        r_next = returns_matrix[idx + 1]
        next_returns_list.append(r_next)

        # Realized 30-day covariance matrix Sigma_t in R^{9 x 9}
        # Trailing 30 days of returns: returns_matrix[idx - lookback + 1 : idx + 1]
        ret_window = returns_matrix[idx - lookback + 1 : idx + 1]  # (30, 9)
        sigma_t = np.cov(ret_window, rowvar=False)  # (9, 9)
        # Guard for positive semi-definiteness / regularize diagonal if needed
        sigma_t = np.nan_to_num(sigma_t, nan=1e-4) + np.eye(n_assets) * 1e-6
        cov_list.append(sigma_t)

    dataset = {
        "dates": pd.DatetimeIndex(sample_dates),
        "X": np.array(X_list, dtype=np.float32),              # (N, 9, 30, 10)
        "next_returns": np.array(next_returns_list, dtype=np.float32),  # (N, 9)
        "covariances": np.array(cov_list, dtype=np.float32),        # (N, 9, 9)
    }
    return dataset


if __name__ == "__main__":
    from data.cleaning import clean_and_merge_panel
    panel = clean_and_merge_panel()
    raw_feats = compute_features(panel)
    print(f"Raw features shape: {raw_feats.shape}")
    print(f"Feature columns (10): {raw_feats.columns.tolist()}")

    # Training mask
    dates_idx = raw_feats.index.get_level_values("date")
    train_mask = pd.Series(config.filter_split(dates_idx, "train"), index=dates_idx)
    scaled_feats, scalers = fit_transform_scalers(raw_feats, train_mask)
    print(f"Scaled features shape: {scaled_feats.shape}")

    daily_tensor, dates = build_daily_feature_tensor(scaled_feats)
    print(f"Daily Feature Tensor shape: {daily_tensor.shape}")  # (n_days, 9, 10)

    dataset = create_lookback_dataset(daily_tensor, dates, panel)
    print(f"Dataset X shape: {dataset['X'].shape}")              # (N, 9, 30, 10)
    print(f"Dataset next_returns shape: {dataset['next_returns'].shape}")  # (N, 9)
    print(f"Dataset covariances shape: {dataset['covariances'].shape}")    # (N, 9, 9)
