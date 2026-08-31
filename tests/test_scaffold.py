"""
Unit tests for project scaffolding, imports, universe configuration, and reproducibility.
"""

import numpy as np
import torch
import config


def test_imports():
    """Test clean imports of all scaffold modules."""
    import data.ingestion
    import data.cleaning
    import data.features
    import data.graph
    import models.lstm_gat
    assert config is not None


def test_universe_config():
    """Verify universe constants."""
    assert len(config.TICKERS) == config.NUM_ASSETS
    assert config.NUM_ASSETS == 28
    assert "AAPL" in config.TICKERS
    assert "NVDA" in config.TICKERS
    assert config.TICKERS["AAPL"] == "Information Technology"
    assert config.TICKERS["BA"] == "Industrials"
    assert "META" not in config.TICKERS, "META has 0 FNSPID articles and should be excluded"
    assert "LIN" not in config.TICKERS, "LIN has only 7 FNSPID articles and should be excluded"
    assert config.NUM_FEATURES == 10
    assert config.LOOKBACK_R == 30


def test_seed_determinism():
    """Test that set_seed ensures reproducible random numbers."""
    config.set_seed(42)
    a = torch.randn(10)
    b = np.random.randn(10)

    config.set_seed(42)
    a2 = torch.randn(10)
    b2 = np.random.randn(10)

    assert torch.equal(a, a2)
    assert np.array_equal(b, b2)
