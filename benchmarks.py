"""
Benchmark portfolio allocation strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import config


def compute_equal_weight_benchmark(
    next_returns: np.ndarray,
    dates: pd.DatetimeIndex
) -> pd.Series:
    """
    Computes daily returns of the 1/N Equal-Weight benchmark portfolio:
      r_EW,t = (1/N) * sum_i r_i,t
    """
    n_assets = next_returns.shape[1]
    ew_weights = np.ones(n_assets) / n_assets
    ew_returns = np.dot(next_returns, ew_weights)
    return pd.Series(ew_returns, index=dates, name="Equal-Weight")
