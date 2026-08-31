"""
Unit tests for portfolio graph risk radar and shock propagation engine.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk.concentration import build_weighted_graph, hidden_concentration
from risk.shock_engine import ShockSimulator


class _FakeGATLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(dim, 1) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(dim, 1) * 0.1)
        self.leaky_relu = nn.LeakyReLU(0.10)
        self.bias = nn.Parameter(torch.zeros(dim))


class _FakePortfolioModel(nn.Module):
    """Minimal stand-in matching the attribute interface extract_intermediates
    expects from models.lstm_gat.PortfolioModel, sized down for fast tests."""

    def __init__(self, n_features=10, hidden_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.0)
        self.gat1 = _FakeGATLayer(hidden_dim)
        self.gat2 = _FakeGATLayer(hidden_dim)
        self.final_dropout = nn.Dropout(0.0)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.tilt_scale = 0.22


TICKERS = ["AAA", "BBB", "CCC", "DDD"]
SECTORS = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy", "DDD": "Energy"}


def _make_inputs(n=4, lookback=30, feats=10):
    X_t = torch.randn(1, n, lookback, feats)
    A_t = torch.ones(1, n, n)
    return X_t, A_t


def test_build_weighted_graph_is_undirected():
    alpha_t = np.random.rand(4, 4)
    A_t = np.ones((4, 4))
    graph = build_weighted_graph(alpha_t, A_t)
    assert not graph.is_directed()


def test_hidden_concentration_equal_weights_finite_entropy():
    alpha_t = np.full((4, 4), 0.25)
    A_t = np.ones((4, 4))
    graph = build_weighted_graph(alpha_t, A_t)
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    result = hidden_concentration(weights, graph, alpha_t, A_t)
    assert np.isfinite(result["community_entropy"])


def test_analyze_live_returns_required_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _FakePortfolioModel()
    sim = ShockSimulator(model, TICKERS, SECTORS, scaler_stds=np.ones((4, 10)))
    X_t, A_t = _make_inputs(n=4)

    result = sim.analyze_live(X_t, A_t, date_str="2025-04-08")

    required_keys = {
        "date", "weights", "centrality_risk", "community_entropy",
        "num_communities", "hidden_2hop_exposure", "communities",
        "alerts", "stock_metrics",
    }
    assert required_keys.issubset(result.keys())


def test_inject_sector_shock_zero_magnitude_returns_baseline():
    model = _FakePortfolioModel()
    sim = ShockSimulator(model, TICKERS, SECTORS, scaler_stds=np.ones((4, 10)))
    X_t, A_t = _make_inputs(n=4)

    results = sim.inject_sector_shock(X_t, A_t, sector="Tech", magnitude=0.0, steps=3, save=False)

    baseline_weights = results[0]["weights"]
    for r in results:
        assert np.allclose(r["weights"], baseline_weights)


def test_weight_sum_after_shock_propagation_equals_one():
    model = _FakePortfolioModel()
    sim = ShockSimulator(model, TICKERS, SECTORS, scaler_stds=np.ones((4, 10)))
    X_t, A_t = _make_inputs(n=4)

    results = sim.inject_sector_shock(X_t, A_t, sector="Tech", magnitude=-0.15, steps=3, save=False)

    for r in results:
        assert np.isclose(r["weights"].sum(), 1.0, atol=1e-5)
