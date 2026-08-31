"""
Unit tests for data lookahead prevention, numerical constraints, and seed determinism.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import config
from data.cleaning import clean_and_merge_panel
from data.features import (
    build_daily_feature_tensor,
    compute_features,
    create_lookback_dataset,
    fit_transform_scalers,
)
from data.graph import build_dynamic_adjacency_matrices
from models.lstm_gat import LSTMGATModel
from train import build_full_dataset


@pytest.fixture(scope="module")
def prepared_data():
    dataset_dict, date_to_adj, panel = build_full_dataset()
    return dataset_dict, date_to_adj, panel


def test_no_feature_lookahead(prepared_data):
    """Verify that feature at time t does not change if future prices (t+1..T) are altered."""
    _, _, panel = prepared_data
    panel_copy = panel.copy()

    t_eval_idx = 100
    dates = panel.index.get_level_values("date").unique().sort_values()
    t_eval = dates[t_eval_idx]
    future_dates = dates[dates > t_eval]

    # Baseline features
    feats_base = compute_features(panel_copy)
    base_val = feats_base.xs("AAPL", level="ticker").loc[t_eval].values

    # Alter future prices
    panel_mutated = panel_copy.copy()
    for fd in future_dates:
        panel_mutated.loc[(fd, "AAPL"), "adjusted_close"] *= 10.0
        panel_mutated.loc[(fd, "AAPL"), "volume"] *= 10.0

    feats_mutated = compute_features(panel_mutated)
    mutated_val = feats_mutated.xs("AAPL", level="ticker").loc[t_eval].values

    np.testing.assert_allclose(
        base_val, mutated_val, atol=1e-7,
        err_msg="Lookahead detected: modifying future prices changed features at time t!"
    )


def test_no_graph_lookahead(prepared_data):
    """Verify that dynamic graph A_t does not change if future returns (t+1..T) are altered."""
    _, _, panel = prepared_data
    t_eval_idx = 120
    dates = panel.index.get_level_values("date").unique().sort_values()
    t_eval = dates[t_eval_idx]
    future_dates = dates[dates > t_eval]

    date_to_adj_base, _ = build_dynamic_adjacency_matrices(panel)
    base_adj = date_to_adj_base[t_eval]

    # Alter future returns
    panel_mutated = panel.copy()
    for fd in future_dates:
        panel_mutated.loc[(fd, "AAPL"), "adjusted_close"] *= 5.0

    date_to_adj_mut, _ = build_dynamic_adjacency_matrices(panel_mutated)
    mut_adj = date_to_adj_mut[t_eval]

    np.testing.assert_array_equal(
        base_adj, mut_adj,
        err_msg="Lookahead detected: modifying future prices changed graph adjacency A_t at time t!"
    )


def test_weights_sum_to_one_and_architecture_allows_negatives(prepared_data):
    """Verify predicted weights sum to 1.0 every day and architecture enables negative positions."""
    dataset_dict, _, _ = prepared_data
    all_dates = dataset_dict["dates"]
    test_mask = config.filter_split(all_dates, "test")
    test_indices = np.where(test_mask)[0]

    hp_path = config.CACHE_DIR / "best_hyperparameters.json"
    if hp_path.exists():
        import json
        with open(hp_path, "r") as f:
            hp = json.load(f)
        model = LSTMGATModel(
            num_assets=config.NUM_ASSETS,
            lstm_hidden=hp.get("hidden_size", config.LSTM_HIDDEN_SIZE),
            gat_hidden=hp.get("hidden_size", config.GAT_HIDDEN_SIZE),
            gat_heads=hp.get("gat_heads", config.GAT_HEADS),
            tilt_scale=hp.get("tilt_scale", None),
            top_k=hp.get("top_k", None)
        )
    else:
        model = LSTMGATModel(num_assets=config.NUM_ASSETS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    weights_path = config.CACHE_DIR / "final_retrained_model.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    batch_size = 64
    w_list = []
    with torch.no_grad():
        for i in range(0, len(test_indices), batch_size):
            bx = torch.tensor(dataset_dict["X"][test_indices[i:i+batch_size]], dtype=torch.float32, device=device)
            badj = torch.tensor(dataset_dict["adj"][test_indices[i:i+batch_size]], dtype=torch.float32, device=device)
            bw = model(bx, badj).cpu().numpy()
            w_list.append(bw)
    w = np.concatenate(w_list, axis=0)

    # Sum to 1.0 for every day on test set
    daily_sums = np.sum(w, axis=1)
    np.testing.assert_allclose(daily_sums, 1.0, atol=1e-5)

    assert not np.isnan(w).any(), "NaNs found in predicted test weights"
    assert not np.isinf(w).any(), "Infs found in predicted test weights"

    # Verify architecture allows negative (short) entries under strong differential signals
    raw_model = LSTMGATModel()
    raw_model.eval()
    n = config.NUM_ASSETS
    x_test = torch.randn(8, n, 30, 10) * 4.0
    adj_test = torch.eye(n).unsqueeze(0).repeat(8, 1, 1)
    with torch.no_grad():
        w_gen = raw_model(x_test, adj_test).numpy()
    assert (w_gen < 0).any(), "Model architecture should be capable of producing negative weights"


def test_determinism_seed_reproducibility():
    """Verify seed=42 produces identical predictions across independent runs in eval mode."""
    config.set_seed(100)
    n = config.NUM_ASSETS
    x = torch.randn(4, n, 30, 10)
    adj = torch.eye(n).unsqueeze(0).repeat(4, 1, 1)

    config.set_seed(42)
    m1 = LSTMGATModel()
    m1.eval()
    with torch.no_grad():
        w1 = m1(x, adj).numpy()

    config.set_seed(42)
    m2 = LSTMGATModel()
    m2.eval()
    with torch.no_grad():
        w2 = m2(x, adj).numpy()

    np.testing.assert_allclose(w1, w2, atol=1e-6, err_msg="Seed 42 is not deterministic across runs!")
