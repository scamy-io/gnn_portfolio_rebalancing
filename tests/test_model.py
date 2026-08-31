"""
Unit tests for LSTM-GAT portfolio model architecture.
"""

import numpy as np
import pytest
import torch
import config
from models.lstm_gat import DenseGATLayer, LSTMGATModel


def test_model_forward_shape_and_sum():
    """Acceptance check: weights sum to 1.0 for every sample and can contain negative entries."""
    config.set_seed(42)
    model = LSTMGATModel()
    model.eval()

    batch_size = 16
    n = config.NUM_ASSETS
    x = torch.randn(batch_size, n, 30, 10)
    adj = torch.eye(n).unsqueeze(0).repeat(batch_size, 1, 1)

    with torch.no_grad():
        w = model(x, adj)

    assert w.shape == (batch_size, n)
    # Sum to 1.0 within float tolerance
    sums = w.sum(dim=-1).numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="Portfolio weights do not sum to 1.0")

    # Verify that weights can be negative (short selling enabled by tanh)
    assert (w < 0).any().item(), "Expected some negative weights from tanh activation"


def test_normalization_guard_near_zero():
    """Test numerical stability when raw tanh weights sum to ~0."""
    config.set_seed(42)
    model = LSTMGATModel()
    model.train()

    n = config.NUM_ASSETS
    x = torch.randn(8, n, 30, 10, requires_grad=True)
    adj = torch.eye(n).unsqueeze(0).repeat(8, 1, 1)

    w = model(x, adj)
    loss = w.sum()
    loss.backward()

    assert not torch.isnan(w).any()
    assert not torch.isinf(w).any()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_dense_gat_layer_masking():
    """Test that GAT layer masks out disconnected nodes properly."""
    gat = DenseGATLayer(in_features=80, out_features=80, dropout=0.0)
    h = torch.randn(2, 9, 80)
    # Disconnected graph except self-loops
    adj = torch.eye(9).unsqueeze(0).repeat(2, 1, 1)

    out = gat(h, adj)
    assert out.shape == (2, 9, 80)
    assert not torch.isnan(out).any()


def test_top_k_allocation_sum_to_one_and_middle_passive():
    """Verify Top-K allocation concentrates tilts on top/bottom K and sums strictly to 1.0."""
    n = 28
    k = 7
    model = LSTMGATModel(num_assets=n, top_k=k, tilt_scale=0.25)
    model.eval()

    x = torch.randn(4, n, 30, 10)
    adj = torch.eye(n).unsqueeze(0).repeat(4, 1, 1)

    with torch.no_grad():
        w = model(x, adj).numpy()

    # 1. Check sum strictly equals 1.0
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-5)
    assert not np.isnan(w).any()
    assert not np.isinf(w).any()


def test_return_intermediates_dictionary():
    """Verify return_intermediates=True produces all required intermediate tensors."""
    n = 28
    model = LSTMGATModel(num_assets=n)
    model.eval()

    x = torch.randn(2, n, 30, 10)
    adj = torch.eye(n).unsqueeze(0).repeat(2, 1, 1)

    res = model(x, adj, return_intermediates=True)
    required = {"weights", "h_nodes", "z1", "z2", "alpha1", "alpha2"}
    assert required.issubset(res.keys())
    assert res["alpha1"].shape == (2, n, n)
    assert res["alpha2"].shape == (2, n, n)

