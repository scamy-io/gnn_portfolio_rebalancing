"""
Dynamic graph construction based on sector hierarchy and rolling correlations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import config

logger = logging.getLogger(__name__)


def build_static_sector_matrix(tickers: List[str] = config.TICKER_LIST) -> np.ndarray:
    """
    Builds the static binary sector adjacency matrix S where S_ij = 1 if tickers i and j share the same GICS sector.
    """
    n = len(tickers)
    S = np.zeros((n, n), dtype=np.float32)
    sectors = [config.TICKERS[t] for t in tickers]

    for i in range(n):
        for j in range(n):
            if sectors[i] == sectors[j]:
                S[i, j] = 1.0
    return S


def compute_5day_correlation_edges(
    series_window: np.ndarray,
    threshold: float = config.SECTOR_CORR_THRESHOLD
) -> np.ndarray:
    """
    Computes correlation adjacency matrix from a (5, 9) matrix of 5-day series
    (returns or sentiment).
    Returns binary matrix where edge (i,j) = 1 if |corr(i,j)| > threshold.
    """
    n_days, n_assets = series_window.shape
    adj = np.zeros((n_assets, n_assets), dtype=np.float32)

    # Compute correlation matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        std = np.std(series_window, axis=0)
        # Only compute corr if both series have non-zero variance
        valid_mask = std > 1e-8
        corr = np.corrcoef(series_window, rowvar=False)  # (N, N)
        corr = np.nan_to_num(corr, nan=0.0)

    valid_pair = np.outer(valid_mask, valid_mask)
    adj = ((np.abs(corr) > threshold) & valid_pair).astype(np.float32)
    return adj


def build_dynamic_adjacency_matrices(
    panel_df: pd.DataFrame,
    refresh_days: int = config.DYNAMIC_GRAPH_REFRESH_DAYS,
    corr_threshold: float = config.SECTOR_CORR_THRESHOLD,
    tickers: List[str] = config.TICKER_LIST
) -> Tuple[Dict[pd.Timestamp, np.ndarray], np.ndarray]:
    """
    Builds weekly dynamic binary adjacency matrices A_t for all trading days.
    Every refresh_days (5 days), computes A_t from the trailing 5-day window
    of returns and sentiment, and holds it constant for the following 5 days.
    
    Returns:
      date_to_adj: dict mapping each trading date to its (9, 9) adjacency matrix.
      sector_matrix: static (9, 9) sector matrix.
    """
    dates = panel_df.index.get_level_values("date").unique().sort_values()
    n_days = len(dates)
    n_assets = len(tickers)

    # Extract aligned (n_days, 9) matrices for log returns and weighted sentiment
    returns_matrix = np.zeros((n_days, n_assets), dtype=np.float64)
    sentiment_matrix = np.zeros((n_days, n_assets), dtype=np.float64)

    for i, t_name in enumerate(tickers):
        t_data = panel_df.xs(t_name, level="ticker").reindex(dates)
        p = t_data["adjusted_close"]
        r = np.log(p / p.shift(1)).fillna(0.0).values
        s = t_data["weighted_sentiment"].fillna(0.0).values
        returns_matrix[:, i] = r
        sentiment_matrix[:, i] = s

    # 1. Sector matrix (static baseline)
    S = build_static_sector_matrix(tickers)

    # 2. Dynamic graph construction
    date_to_adj = {}
    current_A = None

    for idx in range(n_days):
        d = dates[idx]

        # Refresh trigger: every refresh_days starting at idx == 4 (first complete 5-day window)
        if idx >= 4 and (idx % refresh_days == 4 or current_A is None):
            ret_window = returns_matrix[idx - 4 : idx + 1]       # (5, 9)
            sent_window = sentiment_matrix[idx - 4 : idx + 1]   # (5, 9)

            ret_edges = compute_5day_correlation_edges(ret_window, corr_threshold)
            sent_edges = compute_5day_correlation_edges(sent_window, corr_threshold)

            # Combined adjacency with OR logic
            combined = np.clip(S + ret_edges + sent_edges, 0.0, 1.0)
            
            # Ensure self-loops (diagonal = 1.0)
            np.fill_diagonal(combined, 1.0)

            # Ensure symmetric
            combined = np.maximum(combined, combined.T)
            current_A = combined.astype(np.float32)

        # In warmup days before idx == 4, use sector matrix with self-loops
        if current_A is None:
            init_A = S.copy()
            np.fill_diagonal(init_A, 1.0)
            current_A = init_A.astype(np.float32)

        date_to_adj[d] = current_A.copy()

    return date_to_adj, S


def dense_adj_to_edge_index(adj: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a dense (N, N) binary adjacency matrix to PyTorch Geometric format:
    edge_index of shape (2, num_edges) and edge_weight of shape (num_edges,).
    """
    src, dst = np.where(adj > 0.5)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


if __name__ == "__main__":
    from data.cleaning import clean_and_merge_panel
    panel = clean_and_merge_panel()
    date_to_adj, S = build_dynamic_adjacency_matrices(panel)
    sample_date = list(date_to_adj.keys())[50]
    A_sample = date_to_adj[sample_date]

    print(f"Total dates mapped to graphs: {len(date_to_adj)}")
    print(f"Sample Adjacency matrix for {sample_date.strftime('%Y-%m-%d')}:\n{A_sample}")
    print(f"Is symmetric: {np.array_equal(A_sample, A_sample.T)}")
    print(f"Diagonal all 1s: {np.all(np.diag(A_sample) == 1.0)}")
    print(f"Values in {{0, 1}}: {set(np.unique(A_sample)).issubset({0.0, 1.0})}")
