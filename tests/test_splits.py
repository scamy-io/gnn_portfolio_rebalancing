"""
Unit test asserting explicit half-open train/val/test date split intervals
and zero date overlap.
"""

import pandas as pd
import pytest
from config import (
    DATE_START,
    DATE_END,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    filter_split,
)


def test_zero_date_overlap_and_boundaries():
    """Assert zero date overlap and correct half-open interval coverage."""
    # Generate full daily calendar
    all_dates = pd.date_range(start=DATE_START, end=DATE_END, freq="D")
    
    train_mask = filter_split(all_dates, "train")
    val_mask = filter_split(all_dates, "val")
    test_mask = filter_split(all_dates, "test")

    train_dates = set(all_dates[train_mask])
    val_dates = set(all_dates[val_mask])
    test_dates = set(all_dates[test_mask])

    # Assert mutual exclusivity (zero date overlap)
    train_val_overlap = train_dates.intersection(val_dates)
    val_test_overlap = val_dates.intersection(test_dates)
    train_test_overlap = train_dates.intersection(test_dates)

    assert len(train_val_overlap) == 0, f"Overlap between train and val: {train_val_overlap}"
    assert len(val_test_overlap) == 0, f"Overlap between val and test: {val_test_overlap}"
    assert len(train_test_overlap) == 0, f"Overlap between train and test: {train_test_overlap}"

    # Assert exact interval boundaries
    assert min(train_dates) == pd.Timestamp(TRAIN_START)
    assert max(train_dates) < pd.Timestamp(TRAIN_END)
    assert max(train_dates) == pd.Timestamp("2022-01-14")  # Day before 2022-01-15

    assert min(val_dates) == pd.Timestamp(VAL_START)
    assert max(val_dates) < pd.Timestamp(VAL_END)
    assert max(val_dates) == pd.Timestamp("2022-07-14")  # Day before 2022-07-15

    assert min(test_dates) == pd.Timestamp(TEST_START)
    assert max(test_dates) == pd.Timestamp(TEST_END)


def test_train_val_combined_split():
    """Test that train_val combined split equals train union val exactly."""
    all_dates = pd.date_range(start=DATE_START, end=DATE_END, freq="D")
    train_mask = filter_split(all_dates, "train")
    val_mask = filter_split(all_dates, "val")
    train_val_mask = filter_split(all_dates, "train_val")

    train_val_set = set(all_dates[train_val_mask])
    expected_set = set(all_dates[train_mask]).union(set(all_dates[val_mask]))
    assert train_val_set == expected_set
