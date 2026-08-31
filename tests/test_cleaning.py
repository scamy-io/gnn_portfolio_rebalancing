"""
Unit tests for data cleaning and panel alignment.
"""

import numpy as np
import pandas as pd
import pytest
import config
from data.cleaning import clean_and_merge_panel, shift_to_next_trading_day


def test_shift_to_next_trading_day():
    """Test shifting non-trading days to next trading day."""
    calendar = pd.date_range("2021-01-04", "2021-01-08", freq="D")  # Mon-Fri
    # Saturday 2021-01-02 -> Mon 2021-01-04
    sat = pd.Timestamp("2021-01-02")
    shifted = shift_to_next_trading_day(sat, calendar)
    assert shifted == pd.Timestamp("2021-01-04")

    # Tuesday 2021-01-05 (already trading day) -> 2021-01-05
    tue = pd.Timestamp("2021-01-05")
    shifted_tue = shift_to_next_trading_day(tue, calendar)
    assert shifted_tue == pd.Timestamp("2021-01-05")


def test_clean_and_merge_panel_acceptance():
    """Verify zero NaNs and complete panel structure."""
    panel = clean_and_merge_panel()
    assert panel is not None
    assert not panel.empty

    # Assert 0 NaNs
    assert panel.isna().sum().sum() == 0, f"Panel has NaNs:\n{panel.isna().sum()}"

    # Assert MultiIndex names
    assert panel.index.names == ["date", "ticker"]

    # Assert all configured tickers present (28-stock universe)
    tickers = panel.index.get_level_values("ticker").unique().tolist()
    assert sorted(tickers) == sorted(config.TICKER_LIST)

    # Assert total rows = n_dates * NUM_ASSETS
    n_dates = len(panel.index.get_level_values("date").unique())
    assert len(panel) == n_dates * config.NUM_ASSETS
    # ~5-year window (2018-12 to 2023-12) after warmup buffer; sanity floor well
    # below the actual trading-day count to tolerate calendar differences.
    assert n_dates >= 1000

    # Assert essential columns
    for col in ["adjusted_close", "volume", "article_count", "weighted_sentiment", "sentiment_variance", "rf_annualized"]:
        assert col in panel.columns
