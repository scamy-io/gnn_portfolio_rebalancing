"""
Optuna hyperparameter tuning for baseline LSTM-GAT model.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from train import NegativeSharpeLoss, NegativeSortinoLoss, PortfolioDataset, build_full_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress excessive Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
    trial: optuna.Trial,
    dataset_dict: dict,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device
) -> float:
    """Optuna objective function to evaluate a single hyperparameter configuration."""
    # 1. Sample hyperparameters based on Table 7 search space + Sortino/Top-K
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 80, 96])
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.10, 0.35, step=0.05)
    gat_dropout = trial.suggest_float("gat_dropout", 0.10, 0.35, step=0.05)
    final_dropout = trial.suggest_float("final_dropout", 0.15, 0.40, step=0.05)
    gat_alpha = trial.suggest_float("gat_alpha", 0.05, 0.25, step=0.05)
    
    lstm_weight_decay = trial.suggest_float("lstm_weight_decay", 1e-5, 5e-3, log=True)
    gat_weight_decay = trial.suggest_float("gat_weight_decay", 1e-5, 5e-3, log=True)
    final_weight_decay = trial.suggest_float("final_weight_decay", 1e-5, 5e-3, log=True)
    
    tilt_scale = trial.suggest_float("tilt_scale", 0.10, 0.35, step=0.02)
    loss_scale = trial.suggest_categorical("loss_scale", [5.0, 10.0, 15.0])
    loss_type = trial.suggest_categorical("loss_type", ["sortino", "sharpe"])
    top_k = trial.suggest_categorical("top_k", [5, 7, 9])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    epochs = trial.suggest_int("epochs", 20, 80, step=10)

    # 2. Build model and optimizer
    config.set_seed(config.SEED)
    model = LSTMGATModel(
        num_features=config.NUM_FEATURES,
        lookback_r=config.LOOKBACK_R,
        num_assets=config.NUM_ASSETS,
        lstm_hidden=hidden_size,
        lstm_dropout=lstm_dropout,
        gat_hidden=hidden_size,
        gat_layers=config.GAT_LAYERS,
        gat_dropout=gat_dropout,
        leaky_relu_alpha=gat_alpha,
        final_dropout=final_dropout,
        tilt_scale=tilt_scale,
        top_k=top_k
    ).to(device)

    param_groups = [
        {"params": model.lstm.parameters(), "weight_decay": lstm_weight_decay},
        {"params": list(model.gat1.parameters()) + list(model.gat2.parameters()), "weight_decay": gat_weight_decay},
        {"params": model.linear_out.parameters(), "weight_decay": final_weight_decay},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=learning_rate)
    if loss_type == "sortino":
        loss_fn = NegativeSortinoLoss(scale=loss_scale)
    else:
        loss_fn = NegativeSharpeLoss(scale=loss_scale)

    full_ds = PortfolioDataset(
        dataset_dict["X"],
        dataset_dict["adj"],
        dataset_dict["next_returns"],
        dataset_dict["covariances"],
        dataset_dict["dates"]
    )
    train_loader = DataLoader(
        torch.utils.data.Subset(full_ds, train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_x = full_ds.X[val_indices].to(device)
    val_adj = full_ds.adj[val_indices].to(device)
    val_r = full_ds.next_returns[val_indices].cpu().numpy()

    best_val_sharpe = -float("inf")
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, badj, br, bcov, _ in train_loader:
            bx, badj, br, bcov = bx.to(device), badj.to(device), br.to(device), bcov.to(device)
            optimizer.zero_grad()
            w = model(bx, badj)
            loss = loss_fn(w, br, bcov)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            w_val = model(val_x, val_adj).cpu().numpy()
        
        val_port_rets = np.sum(w_val * val_r, axis=-1)
        mean_ret = np.mean(val_port_rets)
        std_ret = np.std(val_port_rets, ddof=1) if len(val_port_rets) > 1 else 1e-6
        ann_sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(252.0)

        if ann_sharpe > best_val_sharpe:
            best_val_sharpe = ann_sharpe
            best_epoch = epoch

        trial.report(best_val_sharpe, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("best_epoch", best_epoch)
    return best_val_sharpe


def run_tuning(n_trials: int = 30):
    """Executes Optuna study and trains final model with winning parameters."""
    print("=" * 80)
    print("LSTM-GAT Portfolio Model v4 — Automated Hyperparameter Tuning (Optuna)")
    print(f"Target Universe: {config.NUM_ASSETS} Stocks Across 11 GICS Sectors")
    print(f"Trials: {n_trials} | Search Criterion: Maximize Validation Sharpe")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}\n")

    dataset_dict, date_to_adj, panel = build_full_dataset()
    all_dates = dataset_dict["dates"]

    train_mask = config.filter_split(all_dates, "train")
    val_mask = config.filter_split(all_dates, "val")
    train_val_mask = config.filter_split(all_dates, "train_val")
    test_mask = config.filter_split(all_dates, "test")

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]

    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def wrapped_objective(trial):
        val_sharpe = objective(trial, dataset_dict, train_indices, val_indices, device)
        trial_num = trial.number + 1
        lr = trial.params.get("learning_rate", 0.0)
        h = trial.params.get("hidden_size", 0)
        t_scale = trial.params.get("tilt_scale", 0.0)
        print(f"Trial [{trial_num:>2}/{n_trials}] -> Val Sharpe: {val_sharpe:>7.4f} | LR: {lr:.5f}, Hidden: {h}, Tilt: {t_scale:.2f}")
        return val_sharpe

    print("Starting Optuna search across hyperparameter space...")
    study.optimize(wrapped_objective, n_trials=n_trials)

    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print(f"OPTUNA STUDY COMPLETE! Best Trial #{best_trial.number + 1} with Validation Sharpe: {best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  - {k}: {v}")
    print("=" * 80)

    hp_save_path = config.CACHE_DIR / "best_hyperparameters.json"
    with open(hp_save_path, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"\nSaved best hyperparameters to {hp_save_path}")

    # Retrain Winning Model on 70% Train + Val Block
    best_p = best_trial.params
    print("\n--- Retraining Winning Configuration on Full 70% Train + Val Block ---")
    config.set_seed(config.SEED)

    final_model = LSTMGATModel(
        num_features=config.NUM_FEATURES,
        lookback_r=config.LOOKBACK_R,
        num_assets=config.NUM_ASSETS,
        lstm_hidden=best_p["hidden_size"],
        lstm_dropout=best_p["lstm_dropout"],
        gat_hidden=best_p["hidden_size"],
        gat_layers=config.GAT_LAYERS,
        gat_dropout=best_p["gat_dropout"],
        leaky_relu_alpha=best_p["gat_alpha"],
        final_dropout=best_p["final_dropout"],
        tilt_scale=best_p["tilt_scale"],
        top_k=best_p.get("top_k", None)
    ).to(device)

    param_groups = [
        {"params": final_model.lstm.parameters(), "weight_decay": best_p["lstm_weight_decay"]},
        {"params": list(final_model.gat1.parameters()) + list(final_model.gat2.parameters()), "weight_decay": best_p["gat_weight_decay"]},
        {"params": final_model.linear_out.parameters(), "weight_decay": best_p["final_weight_decay"]},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=best_p["learning_rate"])
    if best_p.get("loss_type", "sharpe") == "sortino":
        loss_fn = NegativeSortinoLoss(scale=best_p["loss_scale"])
    else:
        loss_fn = NegativeSharpeLoss(scale=best_p["loss_scale"])

    full_ds = PortfolioDataset(
        dataset_dict["X"],
        dataset_dict["adj"],
        dataset_dict["next_returns"],
        dataset_dict["covariances"],
        dataset_dict["dates"]
    )
    retrain_loader = DataLoader(
        torch.utils.data.Subset(full_ds, train_val_indices),
        batch_size=best_p["batch_size"],
        shuffle=True,
        drop_last=False
    )

    retrain_epochs = best_trial.user_attrs.get("best_epoch", best_p["epochs"])
    print(f"Optimal Convergence Epoch: {retrain_epochs}")
    for epoch in range(1, retrain_epochs + 1):
        final_model.train()
        batch_losses = []
        for bx, badj, br, bcov, _ in retrain_loader:
            bx, badj, br, bcov = bx.to(device), badj.to(device), br.to(device), bcov.to(device)
            optimizer.zero_grad()
            w = final_model(bx, badj)
            loss = loss_fn(w, br, bcov)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == retrain_epochs:
            print(f"Retrain Epoch [{epoch:>2}/{retrain_epochs}] Loss: {np.mean(batch_losses):.4f}")

    final_model_path = config.CACHE_DIR / "final_retrained_model.pt"
    torch.save(final_model.state_dict(), final_model_path)
    print(f"Saved final optimized model to {final_model_path}")


if __name__ == "__main__":
    run_tuning(n_trials=30)
