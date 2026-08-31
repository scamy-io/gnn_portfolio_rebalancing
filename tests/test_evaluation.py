"""
Unit tests for evaluation metrics and equal-weight benchmark calculations.
"""

import numpy as np
import pandas as pd
import pytest
import config
from benchmarks import compute_equal_weight_benchmark
from evaluate import compute_metrics


def test_equal_weight_benchmark_calculation():
    """Test 1/N equal-weight portfolio returns (N = config.NUM_ASSETS)."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    next_returns = np.ones((5, config.NUM_ASSETS)) * 0.01  # 1% across all assets
    ew_series = compute_equal_weight_benchmark(next_returns, dates)

    assert len(ew_series) == 5
    np.testing.assert_allclose(ew_series.values, 0.01, atol=1e-6)


def test_metrics_formulas():
    """Test mathematical accuracy of evaluation metric formulas."""
    # Synthetic 252 daily returns with mean 0.1% / day, std 1% / day
    np.random.seed(42)
    daily_rets = pd.Series(np.full(252, 0.001), index=pd.date_range("2024-01-01", periods=252, freq="D"))
    rf_series = pd.Series(np.full(252, 0.04), index=daily_rets.index)  # 4% Rf

    metrics = compute_metrics(daily_rets, rf_series)

    # Expected cum return: (1.001)^252 - 1 = 0.2863 (28.63%)
    expected_cum = (1.001 ** 252) - 1.0
    np.testing.assert_allclose(metrics["Cumulative Return (%)"], expected_cum * 100.0, atol=1e-2)
    np.testing.assert_allclose(metrics["Annualized Return (%)"], expected_cum * 100.0, atol=1e-2)
    assert metrics["Sharpe Ratio"] > 0
    assert metrics["Max Drawdown (%)"] == 0.0  # Monotonically increasing
