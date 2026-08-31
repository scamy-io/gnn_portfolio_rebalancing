"""
Unit tests for training objectives, loss functions, and optimizer parameter groups.
"""

import numpy as np
import pytest
import torch
import config
from models.lstm_gat import LSTMGATModel
from train import NegativeSharpeLoss, create_optimizer


def test_negative_sharpe_loss_computation():
    """Test mathematical accuracy of Negative Sharpe loss."""
    loss_fn = NegativeSharpeLoss()

    # Create synthetic portfolio: 2 assets
    weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    next_returns = torch.tensor([[0.05, 0.02]], dtype=torch.float32)
    covariances = torch.tensor([[[0.04, 0.00], [0.00, 0.01]]], dtype=torch.float32)

    # Expected port return: 0.6*0.05 + 0.4*0.02 = 0.038
    # Expected port variance: 0.6^2*0.04 + 0.4^2*0.01 = 0.0144 + 0.0016 = 0.016
    # Expected std: sqrt(0.016) = 0.1264911
    # Expected loss: - 0.038 / 0.1264911 = -0.300416
    expected_loss = - (0.038 / np.sqrt(0.016))

    computed_loss = loss_fn(weights, next_returns, covariances).item()
    np.testing.assert_allclose(computed_loss, expected_loss, atol=1e-4)


def test_negative_sortino_loss_computation():
    """Test mathematical accuracy of Negative Sortino loss."""
    from train import NegativeSortinoLoss
    loss_fn = NegativeSortinoLoss(scale=1.0)

    weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    next_returns = torch.tensor([[-0.05, 0.02]], dtype=torch.float32)

    # port return = 0.6*(-0.05) + 0.4*(0.02) = -0.03 + 0.008 = -0.022
    # downside diff = -0.022 -> downside std = 0.022
    # Sortino = -0.022 / 0.022 = -1.0 -> Loss = -(-1.0) = 1.0
    computed_loss = loss_fn(weights, next_returns).item()
    np.testing.assert_allclose(computed_loss, 1.0, atol=1e-4)


def test_optimizer_parameter_groups():
    """Acceptance check: verify 3 distinct weight-decay groups from Table 8."""
    model = LSTMGATModel()
    optimizer = create_optimizer(model)

    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[0]["lr"] == config.LEARNING_RATE
    assert optimizer.param_groups[0]["weight_decay"] == config.LSTM_WEIGHT_DECAY  # 3.33e-3
    assert optimizer.param_groups[1]["weight_decay"] == config.GAT_WEIGHT_DECAY   # 2.48e-4
    assert optimizer.param_groups[2]["weight_decay"] == config.FINAL_WEIGHT_DECAY # 2.69e-4
