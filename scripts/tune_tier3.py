"""
Optuna hyperparameter optimization for multi-head GAT and graph topology.
"""

import sys
import json
import logging
from pathlib import Path

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
    panel: pd.DataFrame,
    dataset_dict_base: dict,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
    graph_cache: dict
) -> float:
    """Optuna objective function exploring Tier 3 graph topology & multi-head architecture."""
    # 1. Sample Tier 3 Structural & Graph Parameters
    gat_heads = trial.suggest_categorical("gat_heads", [1, 2, 4])
    corr_threshold = trial.suggest_categorical("corr_threshold", [0.40, 0.50, 0.60])
    refresh_days = trial.suggest_categorical("refresh_days", [1, 3, 5, 10])
    top_k = trial.suggest_categorical("top_k", [5, 7])
    tilt_scale = trial.suggest_categorical("tilt_scale", [0.20, 0.28, 0.34])
    loss_type = trial.suggest_categorical("loss_type", ["sortino", "sharpe"])
    loss_scale = trial.suggest_categorical("loss_scale", [5.0, 10.0, 15.0])
    
    # Tier 1 & 2 defaults / fine search
    hidden_size = 64
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.15, 0.35, step=0.05)
    gat_dropout = trial.suggest_float("gat_dropout", 0.10, 0.25, step=0.05)
    final_dropout = trial.suggest_float("final_dropout", 0.25, 0.40, step=0.05)
    gat_alpha = 0.05
    lstm_weight_decay = 1e-4
    gat_weight_decay = 2e-4
    final_weight_decay = 1e-3
    batch_size = 32
    epochs = 40

    # 2. Retrieve or compute dynamic graph adjacency
    cache_key = (corr_threshold, refresh_days)
    if cache_key in graph_cache:
        adj_array = graph_cache[cache_key]
    else:
        date_to_adj, _ = build_dynamic_adjacency_matrices(
            panel,
            refresh_days=refresh_days,
            corr_threshold=corr_threshold
        )
        dates = dataset_dict_base["dates"]
        adj_list = [date_to_adj[d] for d in dates]
        adj_array = np.stack(adj_list, axis=0) # (T, N, N)
        graph_cache[cache_key] = adj_array

    # 3. Build Model with Multi-Head Attention
    config.set_seed(config.SEED)
    model = LSTMGATModel(
        num_features=config.NUM_FEATURES,
        lookback_r=config.LOOKBACK_R,
        num_assets=config.NUM_ASSETS,
        lstm_hidden=hidden_size,
        lstm_dropout=lstm_dropout,
        gat_hidden=hidden_size,
        gat_layers=config.GAT_LAYERS,
        gat_heads=gat_heads,
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
        dataset_dict_base["X"],
        adj_array,
        dataset_dict_base["next_returns"],
        dataset_dict_base["covariances"],
        dataset_dict_base["dates"]
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


def run_tier3_tuning(n_trials: int = 25):
    """Executes Optuna study exploring Tier 3 topology and structural hyperparameters."""
    print("=" * 80)
    print("LSTM-GAT Portfolio Model v4 — Tier 3 Graph & Topology Optimization (Optuna)")
    print(f"Target Universe: {config.NUM_ASSETS} Stocks Across 11 GICS Sectors")
    print(f"Exploration: Multi-Head GAT (1,2,4), Graph Threshold (0.4-0.6), Refresh (1-10d)")
    print(f"Trials: {n_trials} | Search Criterion: Maximize Validation Sharpe")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}\n")

    # Load dataset and merged panel
    dataset_dict_base, _, panel = build_full_dataset()
    all_dates = dataset_dict_base["dates"]

    train_mask = config.filter_split(all_dates, "train")
    val_mask = config.filter_split(all_dates, "val")
    train_val_mask = train_mask | val_mask

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    train_val_indices = np.where(train_val_mask)[0]

    graph_cache = {}

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    print("Starting Optuna Tier 3 search across graph topology & multi-head space...")
    study.optimize(
        lambda trial: objective(
            trial,
            panel,
            dataset_dict_base,
            train_indices,
            val_indices,
            device,
            graph_cache
        ),
        n_trials=n_trials,
        show_progress_bar=False
    )

    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print(f"TIER 3 STUDY COMPLETE! Best Trial #{best_trial.number + 1} with Validation Sharpe: {best_trial.value:.4f}")
    print("Best Tier 3 Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  - {k}: {v}")
    print("=" * 80)

    # Retrain Winning Model on 70% Train + Val Block
    best_p = best_trial.params
    print("\n--- Retraining Winning Tier 3 Configuration on Full 70% Train + Val Block ---")
    config.set_seed(config.SEED)

    # Build optimal dynamic graph
    cache_key = (best_p["corr_threshold"], best_p["refresh_days"])
    if cache_key in graph_cache:
        adj_array = graph_cache[cache_key]
    else:
        date_to_adj, _ = build_dynamic_adjacency_matrices(
            panel,
            refresh_days=best_p["refresh_days"],
            corr_threshold=best_p["corr_threshold"]
        )
        adj_list = [date_to_adj[d] for d in all_dates]
        adj_array = np.stack(adj_list, axis=0)

    final_model = LSTMGATModel(
        num_features=config.NUM_FEATURES,
        lookback_r=config.LOOKBACK_R,
        num_assets=config.NUM_ASSETS,
        lstm_hidden=64,
        lstm_dropout=best_p["lstm_dropout"],
        gat_hidden=64,
        gat_layers=config.GAT_LAYERS,
        gat_heads=best_p["gat_heads"],
        gat_dropout=best_p["gat_dropout"],
        leaky_relu_alpha=0.05,
        final_dropout=best_p["final_dropout"],
        tilt_scale=best_p["tilt_scale"],
        top_k=best_p["top_k"]
    ).to(device)

    param_groups = [
        {"params": final_model.lstm.parameters(), "weight_decay": 1e-4},
        {"params": list(final_model.gat1.parameters()) + list(final_model.gat2.parameters()), "weight_decay": 2e-4},
        {"params": final_model.linear_out.parameters(), "weight_decay": 1e-3},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=best_p["learning_rate"])
    if best_p["loss_type"] == "sortino":
        loss_fn = NegativeSortinoLoss(scale=best_p["loss_scale"])
    else:
        loss_fn = NegativeSharpeLoss(scale=best_p["loss_scale"])

    full_ds = PortfolioDataset(
        dataset_dict_base["X"],
        adj_array,
        dataset_dict_base["next_returns"],
        dataset_dict_base["covariances"],
        dataset_dict_base["dates"]
    )
    retrain_loader = DataLoader(
        torch.utils.data.Subset(full_ds, train_val_indices),
        batch_size=32,
        shuffle=True,
        drop_last=False
    )

    retrain_epochs = best_trial.user_attrs.get("best_epoch", 25)
    print(f"Optimal Retrain Convergence Epoch: {retrain_epochs}")
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

    # Save best hyperparameters and retrained model
    best_p_full = dict(best_trial.params)
    best_p_full["hidden_size"] = 64
    best_p_full["epochs"] = retrain_epochs

    hp_save_path = config.CACHE_DIR / "best_hyperparameters.json"
    with open(hp_save_path, "w") as f:
        json.dump(best_p_full, f, indent=2)
    print(f"\nSaved winning Tier 3 hyperparameters to {hp_save_path}")

    model_save_path = config.CACHE_DIR / "final_retrained_model.pt"
    torch.save(final_model.state_dict(), model_save_path)
    print(f"Saved final retrained Tier 3 model to {model_save_path}")


if __name__ == "__main__":
    run_tier3_tuning(n_trials=25)
