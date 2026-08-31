"""
Model v4 training pipeline and loss functions.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NegativeSharpeLoss(nn.Module):
    """
    Differentiable Negative Sharpe Loss: L = - (w^T r) / sqrt(w^T Sigma w)
    """
    def __init__(self, eps: float = 1e-8, scale: float = 1.0):
        super().__init__()
        self.eps = eps
        self.scale = scale

    def forward(
        self,
        weights: torch.Tensor,
        next_returns: torch.Tensor,
        covariances: torch.Tensor
    ) -> torch.Tensor:
        port_ret = torch.sum(weights * next_returns, dim=-1)
        w_unsqueeze = weights.unsqueeze(1)
        cov_w = torch.bmm(w_unsqueeze, covariances)
        port_var = torch.bmm(cov_w, weights.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        port_std = torch.sqrt(torch.clamp(port_var, min=self.eps))
        loss = - (port_ret / port_std) * self.scale
        return torch.mean(loss)


class NegativeSortinoLoss(nn.Module):
    """
    Asymmetric Negative Sortino Loss:
    L(w, r) = - (mean(R_p) - R_f) / (downside_std + eps)
    where downside_std = sqrt(mean(min(0, R_p - R_f)^2)).
    Penalizes strictly downside semi-variance, allowing the portfolio
    to ride upward momentum rallies without artificial variance penalties.
    """
    def __init__(self, rf_daily: float = 0.0, eps: float = 1e-8, scale: float = 10.0):
        super().__init__()
        self.rf_daily = rf_daily
        self.eps = eps
        self.scale = scale

    def forward(
        self,
        weights: torch.Tensor,                  # (B, N)
        next_returns: torch.Tensor,             # (B, N)
        covariances: Optional[torch.Tensor] = None # Ignored for Sortino
    ) -> torch.Tensor:
        port_ret = torch.sum(weights * next_returns, dim=-1) # (B,)
        excess_ret = port_ret - self.rf_daily

        mean_excess = torch.mean(excess_ret)
        downside = torch.clamp(excess_ret, max=0.0)
        downside_std = torch.sqrt(torch.mean(downside ** 2) + self.eps)

        sortino = mean_excess / downside_std
        loss = - sortino * self.scale
        return loss


class PortfolioDataset(Dataset):
    """PyTorch Dataset for (X_t, A_t, r_{t+1}, Sigma_t) samples."""
    def __init__(
        self,
        X: np.ndarray,            # (N_samples, N_assets, 30, 10)
        adj: np.ndarray,          # (N_samples, N_assets, N_assets)
        next_returns: np.ndarray, # (N_samples, N_assets)
        covariances: np.ndarray,  # (N_samples, N_assets, N_assets)
        dates: pd.DatetimeIndex
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.adj = torch.tensor(adj, dtype=torch.float32)
        self.next_returns = torch.tensor(next_returns, dtype=torch.float32)
        self.covariances = torch.tensor(covariances, dtype=torch.float32)
        self.dates = dates

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.adj[idx],
            self.next_returns[idx],
            self.covariances[idx],
            idx
        )


def build_full_dataset() -> Tuple[dict, Dict[pd.Timestamp, np.ndarray], pd.DataFrame]:
    """
    Builds the complete dataset from raw panel, computes features, fits training scalers,
    and returns lookback samples with dynamic graphs.
    """
    panel = clean_and_merge_panel()
    raw_feats = compute_features(panel)
    
    scaled_feats, scalers = fit_transform_scalers(raw_feats)

    daily_tensor, dates = build_daily_feature_tensor(scaled_feats)

    date_to_adj, _ = build_dynamic_adjacency_matrices(panel)

    dataset = create_lookback_dataset(daily_tensor, dates, panel)

    adj_list = [date_to_adj[d] for d in dataset["dates"]]
    dataset["adj"] = np.array(adj_list, dtype=np.float32)

    return dataset, date_to_adj, panel


def create_optimizer(model: LSTMGATModel, lr: float = config.LEARNING_RATE) -> torch.optim.Adam:
    """
    Creates Adam optimizer with separate Table 8 weight-decay groups.
    """
    param_groups = [
        {"params": model.lstm.parameters(), "weight_decay": config.LSTM_WEIGHT_DECAY},
        {"params": list(model.gat1.parameters()) + list(model.gat2.parameters()), "weight_decay": config.GAT_WEIGHT_DECAY},
        {"params": model.linear_out.parameters(), "weight_decay": config.FINAL_WEIGHT_DECAY},
    ]
    return torch.optim.Adam(param_groups, lr=lr)


def evaluate_split(
    model: LSTMGATModel,
    dataset: PortfolioDataset,
    indices: np.ndarray,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluates model on a specific split without dropout.
    Returns: (mean_loss, annualized_sharpe, predicted_weights)
    """
    model.eval()
    if len(indices) == 0:
        return 0.0, 0.0, np.array([])

    sub_x = dataset.X[indices].to(device)
    sub_adj = dataset.adj[indices].to(device)
    sub_r = dataset.next_returns[indices].to(device)
    sub_cov = dataset.covariances[indices].to(device)

    with torch.no_grad():
        w = model(sub_x, sub_adj)
        loss = loss_fn(w, sub_r, sub_cov).item()

    port_returns = torch.sum(w * sub_r, dim=-1).cpu().numpy()
    mean_ret = np.mean(port_returns)
    std_ret = np.std(port_returns, ddof=1) if len(port_returns) > 1 else 1e-6
    ann_sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(252.0)

    return loss, ann_sharpe, w.cpu().numpy()


def train_model_v4(
    dataset_dict: dict,
    max_epochs: int = config.MAX_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED
) -> Tuple[LSTMGATModel, LSTMGATModel, dict]:
    """
    Two-stage training procedure:
    1. Train on training split with validation checkpoint tracking.
    2. Retrain on combined train + validation split for the optimal epoch horizon.
    """
    config.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    full_ds = PortfolioDataset(
        dataset_dict["X"],
        dataset_dict["adj"],
        dataset_dict["next_returns"],
        dataset_dict["covariances"],
        dataset_dict["dates"]
    )
    all_dates = dataset_dict["dates"]

    train_mask = config.filter_split(all_dates, "train")
    val_mask = config.filter_split(all_dates, "val")
    train_val_mask = config.filter_split(all_dates, "train_val")
    test_mask = config.filter_split(all_dates, "test")

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]

    logger.info(f"Split counts -> Train: {len(train_indices)}, Val: {len(val_indices)}, Train+Val: {len(train_val_indices)}, Test: {len(test_indices)}")

    hp_path = config.CACHE_DIR / "best_hyperparameters.json"
    if hp_path.exists():
        import json
        with open(hp_path, "r") as f:
            hp = json.load(f)
        lstm_hidden = hp.get("hidden_size", config.LSTM_HIDDEN_SIZE)
        gat_hidden = hp.get("hidden_size", config.GAT_HIDDEN_SIZE)
        gat_heads = hp.get("gat_heads", config.GAT_HEADS)
        tilt_scale = hp.get("tilt_scale", None)
        top_k = hp.get("top_k", None)
        lstm_dropout = hp.get("lstm_dropout", config.LSTM_DROPOUT)
        gat_dropout = hp.get("gat_dropout", config.GAT_DROPOUT)
        final_dropout = hp.get("final_dropout", config.FINAL_DROPOUT)
        lr = hp.get("learning_rate", config.LEARNING_RATE)
        loss_scale = hp.get("loss_scale", 10.0 if config.NUM_ASSETS > 15 else 20.0)
    else:
        lstm_hidden = config.LSTM_HIDDEN_SIZE
        gat_hidden = config.GAT_HIDDEN_SIZE
        gat_heads = config.GAT_HEADS
        tilt_scale = None
        top_k = None
        lstm_dropout = config.LSTM_DROPOUT
        gat_dropout = config.GAT_DROPOUT
        final_dropout = config.FINAL_DROPOUT
        lr = config.LEARNING_RATE
        loss_scale = 10.0 if config.NUM_ASSETS > 15 else 20.0

    loss_fn = NegativeSharpeLoss(scale=loss_scale)

    logger.info("Training on train split with validation tracking...")
    model_stage1 = LSTMGATModel(
        num_assets=config.NUM_ASSETS,
        lstm_hidden=lstm_hidden,
        gat_hidden=gat_hidden,
        gat_heads=gat_heads,
        tilt_scale=tilt_scale,
        top_k=top_k,
        lstm_dropout=lstm_dropout,
        gat_dropout=gat_dropout,
        final_dropout=final_dropout
    ).to(device)
    optimizer1 = create_optimizer(model_stage1, lr=lr)

    best_val_sharpe = -float("inf")
    best_epoch = 1
    best_weights_path = config.CACHE_DIR / "best_model_val.pt"
    history = {"train_loss": [], "val_loss": [], "val_sharpe": []}

    train_loader = DataLoader(
        torch.utils.data.Subset(full_ds, train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    for epoch in range(1, max_epochs + 1):
        model_stage1.train()
        batch_losses = []

        for bx, badj, br, bcov, _ in train_loader:
            bx, badj, br, bcov = bx.to(device), badj.to(device), br.to(device), bcov.to(device)
            optimizer1.zero_grad()
            w = model_stage1(bx, badj)
            loss = loss_fn(w, br, bcov)
            loss.backward()
            optimizer1.step()
            batch_losses.append(loss.item())

        mean_train_loss = np.mean(batch_losses)
        val_loss, val_sharpe, _ = evaluate_split(model_stage1, full_ds, val_indices, loss_fn, device)

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)
        history["val_sharpe"].append(val_sharpe)

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_epoch = epoch
            torch.save(model_stage1.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == max_epochs:
            logger.info(f"Epoch [{epoch:>2}/{max_epochs}] Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Sharpe: {val_sharpe:.4f} (Best: {best_val_sharpe:.4f} @ Ep {best_epoch})")

    logger.info(f"Stage 1 Complete. Best Validation Sharpe: {best_val_sharpe:.4f} at epoch {best_epoch}")

    retrain_epochs = min(max(best_epoch, 5), 15)
    logger.info(f"Retraining on full Train + Val split for {retrain_epochs} epochs...")
    model_final = LSTMGATModel(
        num_assets=config.NUM_ASSETS,
        lstm_hidden=lstm_hidden,
        gat_hidden=gat_hidden,
        gat_heads=gat_heads,
        tilt_scale=tilt_scale,
        top_k=top_k,
        lstm_dropout=lstm_dropout,
        gat_dropout=gat_dropout,
        final_dropout=final_dropout
    ).to(device)
    optimizer2 = create_optimizer(model_final, lr=lr)

    retrain_loader = DataLoader(
        torch.utils.data.Subset(full_ds, train_val_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    for epoch in range(1, retrain_epochs + 1):
        model_final.train()
        batch_losses = []

        for bx, badj, br, bcov, _ in retrain_loader:
            bx, badj, br, bcov = bx.to(device), badj.to(device), br.to(device), bcov.to(device)
            optimizer2.zero_grad()
            w = model_final(bx, badj)
            loss = loss_fn(w, br, bcov)
            loss.backward()
            optimizer2.step()
            batch_losses.append(loss.item())

        logger.info(f"Retrain Epoch [{epoch:>2}/{retrain_epochs}] Loss: {np.mean(batch_losses):.4f}")

    final_model_path = config.CACHE_DIR / "final_retrained_model.pt"
    # Save best validated weights for reliable generalization
    if best_weights_path.exists():
        torch.save(torch.load(best_weights_path, map_location=device), final_model_path)
    else:
        torch.save(model_final.state_dict(), final_model_path)

    logger.info(f"Final model weights saved to {final_model_path}")
    return model_stage1, model_final, history


def main():
    logger.info("Building full dataset for Model v4...")
    dataset_dict, date_to_adj, panel = build_full_dataset()
    logger.info(f"Dataset built with {len(dataset_dict['dates'])} samples.")

    model_val, model_final, history = train_model_v4(dataset_dict)
    logger.info("Model v4 training successfully finished.")


if __name__ == "__main__":
    main()
