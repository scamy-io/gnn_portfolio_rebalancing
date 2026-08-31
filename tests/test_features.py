"""
Unit tests for feature extraction and scaling.
"""

import numpy as np
import pandas as pd
import pytest
import config
from data.cleaning import clean_and_merge_panel
from data.features import (
    FEATURE_NAMES,
    build_daily_feature_tensor,
    compute_features,
    create_lookback_dataset,
    fit_transform_scalers,
)


@pytest.fixture(scope="module")
def sample_panel():
    return clean_and_merge_panel()


def test_feature_count_and_names(sample_panel):
    """Assert exactly 10 features matching Model v4 Table 1."""
    feats = compute_features(sample_panel)
    assert feats.shape[1] == 10
    assert list(feats.columns) == FEATURE_NAMES
    assert feats.isna().sum().sum() == 0, "Features contain unexpected NaNs"


def test_daily_feature_tensor_shape(sample_panel):
    """Acceptance check: feature tensor per ticker per day has shape (10,); full panel
    tensor shape (n_days, NUM_ASSETS, 10)."""
    feats = compute_features(sample_panel)
    scaled_feats, _ = fit_transform_scalers(feats)

    daily_tensor, dates = build_daily_feature_tensor(scaled_feats)
    n_days = len(dates)
    assert daily_tensor.shape == (n_days, config.NUM_ASSETS, 10)
    assert daily_tensor.shape[1] == config.NUM_ASSETS
    assert daily_tensor.shape[2] == 10
    assert not np.isnan(daily_tensor).any()


def test_scaler_fit_strictly_on_train_no_leakage(sample_panel):
    """Assert that perturbing test data does not alter the scaler fitted on train."""
    feats1 = compute_features(sample_panel)
    _, scalers1 = fit_transform_scalers(feats1)

    # Perturb test set data
    feats2 = feats1.copy()
    dates_idx = feats2.index.get_level_values("date")
    test_mask = (dates_idx >= pd.Timestamp(config.TEST_START)) & (dates_idx <= pd.Timestamp(config.TEST_END))
    feats2.loc[test_mask, "close"] *= 50.0  # Huge perturbation in test data

    _, scalers2 = fit_transform_scalers(feats2)

    for ticker in config.TICKER_LIST:
        np.testing.assert_allclose(
            scalers1[ticker].mean_,
            scalers2[ticker].mean_,
            err_msg=f"Scaler mean leaked for {ticker}"
        )
        np.testing.assert_allclose(
            scalers1[ticker].scale_,
            scalers2[ticker].scale_,
            err_msg=f"Scaler variance/scale leaked for {ticker}"
        )


def test_lookback_dataset_shapes(sample_panel):
    """Test 30-day lookback sequence tensor X_t in R^{NUM_ASSETS x 30 x 10} and targets."""
    feats = compute_features(sample_panel)
    scaled_feats, _ = fit_transform_scalers(feats)
    daily_tensor, dates = build_daily_feature_tensor(scaled_feats)

    dataset = create_lookback_dataset(daily_tensor, dates, sample_panel)
    N = len(dataset["dates"])
    A = config.NUM_ASSETS
    assert dataset["X"].shape == (N, A, 30, 10)
    assert dataset["next_returns"].shape == (N, A)
    assert dataset["covariances"].shape == (N, A, A)
    assert not np.isnan(dataset["X"]).any()
    assert not np.isnan(dataset["next_returns"]).any()
    assert not np.isnan(dataset["covariances"]).any()
