"""
Model diagnostics, attention dissection, and representation drift analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.cleaning import clean_and_merge_panel
from data.features import (
    build_daily_feature_tensor,
    compute_features,
    create_lookback_dataset,
    fit_transform_scalers,
)
from data.graph import build_dynamic_adjacency_matrices, build_static_sector_matrix
from models.lstm_gat import LSTMGATModel
from train import build_full_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_embeddings_and_attentions(model, sub_x, sub_adj, device, batch_size=64):
    """
    Extracts intermediate embeddings and GAT attention matrices in chunks.
    """
    model.eval()
    B = sub_x.shape[0]

    h_nodes_list, z1_list, z2_list = [], [], []
    alpha1_list, alpha2_list, weights_list = [], [], []

    for i in range(0, B, batch_size):
        bx = sub_x[i : i + batch_size].to(device)
        badj = sub_adj[i : i + batch_size].to(device)
        b_cur, N, R, F_in = bx.shape

        with torch.no_grad():
            x_flat = bx.view(b_cur * N, R, F_in)
            _, (h_n, _) = model.lstm(x_flat)
            h_lstm = model.lstm_dropout(h_n.squeeze(0))
            h_nodes = h_lstm.view(b_cur, N, -1)

            # GAT 1 Attention extraction
            Wh1 = model.gat1.W(h_nodes)
            f_src1 = torch.matmul(Wh1, model.gat1.a_src)
            f_dst1 = torch.matmul(Wh1, model.gat1.a_dst)
            e1 = model.gat1.leaky_relu(f_src1 + f_dst1.transpose(1, 2))
            mask1 = (badj > 0.5)
            e1 = e1.masked_fill(~mask1, -1e9)
            alpha1 = F.softmax(e1, dim=-1)
            z1 = torch.bmm(alpha1, Wh1) + model.gat1.bias
            z1 = F.elu(z1) + h_nodes

            # GAT 2 Attention extraction
            Wh2 = model.gat2.W(z1)
            f_src2 = torch.matmul(Wh2, model.gat2.a_src)
            f_dst2 = torch.matmul(Wh2, model.gat2.a_dst)
            e2 = model.gat2.leaky_relu(f_src2 + f_dst2.transpose(1, 2))
            mask2 = (badj > 0.5)
            e2 = e2.masked_fill(~mask2, -1e9)
            alpha2 = F.softmax(e2, dim=-1)
            z2 = torch.bmm(alpha2, Wh2) + model.gat2.bias
            z2 = F.elu(z2) + z1

            # Predicted weights
            z_drop = model.final_dropout(z2)
            raw_out = model.linear_out(z_drop).squeeze(-1)
            tilt = torch.tanh(raw_out)
            tilt_centered = tilt - tilt.mean(dim=-1, keepdim=True)
            weights = (1.0 / N) + model.tilt_scale * tilt_centered

        h_nodes_list.append(h_nodes.cpu().numpy())
        z1_list.append(z1.cpu().numpy())
        z2_list.append(z2.cpu().numpy())
        alpha1_list.append(alpha1.cpu().numpy())
        alpha2_list.append(alpha2.cpu().numpy())
        weights_list.append(weights.cpu().numpy())

    return {
        "h_lstm": np.concatenate(h_nodes_list, axis=0),
        "z1": np.concatenate(z1_list, axis=0),
        "z2": np.concatenate(z2_list, axis=0),
        "alpha1": np.concatenate(alpha1_list, axis=0),
        "alpha2": np.concatenate(alpha2_list, axis=0),
        "weights": np.concatenate(weights_list, axis=0),
    }


def run_all_diagnostics():
    print("=" * 80)
    print("STARTING TASK 1: EMBEDDING QUALITY & ATTENTION DIAGNOSTICS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Load Dataset & Model
    dataset_dict, date_to_adj, panel = build_full_dataset()
    all_dates = dataset_dict["dates"]

    train_mask = config.filter_split(all_dates, "train")
    val_mask = config.filter_split(all_dates, "val")
    test_mask = config.filter_split(all_dates, "test")

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    test_dates = all_dates[test_idx]

    hp_path = config.CACHE_DIR / "best_hyperparameters.json"
    with open(hp_path, "r") as f:
        hp = json.load(f)

    model = LSTMGATModel(
        num_features=config.NUM_FEATURES,
        lookback_r=config.LOOKBACK_R,
        num_assets=config.NUM_ASSETS,
        lstm_hidden=hp["hidden_size"],
        lstm_dropout=hp["lstm_dropout"],
        gat_hidden=hp["hidden_size"],
        gat_layers=config.GAT_LAYERS,
        gat_dropout=hp["gat_dropout"],
        leaky_relu_alpha=hp["gat_alpha"],
        final_dropout=hp["final_dropout"],
        tilt_scale=hp["tilt_scale"]
    ).to(device)

    model_weights_path = config.CACHE_DIR / "final_retrained_model.pt"
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    model.eval()
    logger.info("Model loaded successfully.")

    # 2. Extract Embeddings across Train, Val, Test
    x_all = torch.tensor(dataset_dict["X"], dtype=torch.float32, device=device)
    adj_all = torch.tensor(dataset_dict["adj"], dtype=torch.float32, device=device)
    outputs_all = extract_embeddings_and_attentions(model, x_all, adj_all, device)

    z_test = outputs_all["z2"][test_idx]  # (325, 30, 96)
    weights_test = outputs_all["weights"][test_idx]  # (325, 30)
    alpha1_test = outputs_all["alpha1"][test_idx]  # (325, 30, 30)
    alpha2_test = outputs_all["alpha2"][test_idx]  # (325, 30, 30)
    r_test = dataset_dict["next_returns"][test_idx]  # (325, 30)

    # -------------------------------------------------------------
    # 1. t-SNE / Cluster Analysis of GAT Embeddings
    # -------------------------------------------------------------
    print("\n--- Diagnostic 1: t-SNE & Embedding Clustering ---")
    # Flatten test embeddings: (n_test_days * n_assets, gat_hidden_dim)
    n_test_days, n_assets, d_dim = z_test.shape
    z_flat = z_test.reshape(n_test_days * n_assets, d_dim)
    
    # Metadata for coloring
    tickers_rep = np.tile(config.TICKER_LIST, n_test_days)
    sectors_rep = np.array([config.TICKERS[t] for t in tickers_rep])
    weights_flat = weights_test.flatten()
    
    # Time period classification, keyed off config.STRESS_WINDOW_* (the in-window
    # stress episode -- see config.py for why this replaces the original paper's
    # April-2025 tariff-shock window, which FNSPID does not cover).
    stress_start = pd.Timestamp(config.STRESS_WINDOW_START)
    stress_end = pd.Timestamp(config.STRESS_WINDOW_END)
    period_pre = f"Pre-Stress ({config.TEST_START} to {config.STRESS_WINDOW_START})"
    period_shock = f"Stress Window ({config.STRESS_WINDOW_LABEL})"
    period_post = f"Post-Stress ({config.STRESS_WINDOW_END} to {config.TEST_END})"

    dates_rep = np.repeat(test_dates, n_assets)
    periods_rep = np.array([
        period_pre if d < stress_start
        else (period_shock if d <= stress_end
        else period_post)
        for d in dates_rep
    ])

    # Sample a balanced subset for clear t-SNE rendering (e.g. every 5th day)
    sample_indices = np.arange(0, len(z_flat), 5)
    tsne = TSNE(n_components=2, perplexity=35, random_state=config.SEED, max_iter=1000)
    z_2d = tsne.fit_transform(z_flat[sample_indices])

    # Compute Silhouette Score for Sector Clustering
    sector_labels_sample = sectors_rep[sample_indices]
    unique_sectors = list(set(config.TICKERS.values()))
    sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
    sector_ids_sample = np.array([sector_to_id[s] for s in sector_labels_sample])
    sil_score = silhouette_score(z_flat[sample_indices], sector_ids_sample)
    print(f"Sector Embedding Silhouette Score: {sil_score:.4f} (Positive indicates structured sector clustering)")

    # Plot 3-Panel t-SNE
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # 1A. By Sector
    cmap_sectors = plt.cm.tab20(np.linspace(0, 1, len(unique_sectors)))
    for s_idx, sector in enumerate(unique_sectors):
        mask = (sector_labels_sample == sector)
        axes[0].scatter(z_2d[mask, 0], z_2d[mask, 1], label=sector, color=cmap_sectors[s_idx], alpha=0.7, s=20)
    axes[0].set_title("t-SNE by GICS Sector", fontweight="bold")
    axes[0].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.45, 1.0))
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # 1B. By Weight Magnitude
    sc = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=weights_flat[sample_indices], cmap="coolwarm", alpha=0.8, s=20)
    axes[1].set_title("t-SNE by Portfolio Weight Allocation", fontweight="bold")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="Portfolio Weight")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    # 1C. By Time Period (Shock Regime)
    period_names = [period_pre, period_shock, period_post]
    colors_p = ["#2ca02c", "#d62728", "#1f77b4"]
    for p_name, color in zip(period_names, colors_p):
        mask = (periods_rep[sample_indices] == p_name)
        axes[2].scatter(z_2d[mask, 0], z_2d[mask, 1], label=p_name, color=color, alpha=0.7, s=20)
    axes[2].set_title("t-SNE by Time Period / Market Shock", fontweight="bold")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    tsne_plot_path = config.CACHE_DIR / "tsne_diagnostics_plot.png"
    plt.savefig(tsne_plot_path)
    plt.close()
    print(f"Saved t-SNE plot to {tsne_plot_path.name}")

    # -------------------------------------------------------------
    # 2. Attention Weight Analysis
    # -------------------------------------------------------------
    print("\n--- Diagnostic 2: Attention Weight & Edge Type Analysis ---")
    mean_alpha1 = np.mean(alpha1_test, axis=0)  # (30, 30)
    mean_alpha2 = np.mean(alpha2_test, axis=0)  # (30, 30)
    combined_alpha = 0.5 * (mean_alpha1 + mean_alpha2)

    sector_mat = build_static_sector_matrix()
    adj_test_mean = np.mean(dataset_dict["adj"][test_idx], axis=0)

    # Dissect average attention by edge type
    self_loop_attn = np.mean([combined_alpha[i, i] for i in range(n_assets)])
    
    same_sector_edges = (sector_mat > 0.5) & (~np.eye(n_assets, dtype=bool))
    diff_sector_edges = (sector_mat <= 0.5) & (~np.eye(n_assets, dtype=bool)) & (adj_test_mean > 0.1)

    same_sector_attn = np.mean(combined_alpha[same_sector_edges]) if np.any(same_sector_edges) else 0.0
    dynamic_corr_attn = np.mean(combined_alpha[diff_sector_edges]) if np.any(diff_sector_edges) else 0.0

    print(f"Average Attention on Self-Loops:            {self_loop_attn:.4f}")
    print(f"Average Attention on Same-Sector Edges:     {same_sector_attn:.4f}")
    print(f"Average Attention on Dynamic Corr Edges:    {dynamic_corr_attn:.4f}")

    # Top-5 Attended Cross-Asset Pairs (excluding self-loops)
    pairs = []
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                pairs.append((
                    config.TICKER_LIST[i],
                    config.TICKER_LIST[j],
                    combined_alpha[i, j],
                    "Same Sector" if sector_mat[i, j] > 0.5 else "Dynamic Correlation"
                ))
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 5 Most Attended Inter-Stock Pairs:")
    for rank, (src, dst, score, etype) in enumerate(pairs[:5], 1):
        print(f"  {rank}. {src:>5} -> {dst:<5} | Attention Score: {score:.4f} | Type: {etype}")

    # -------------------------------------------------------------
    # 3. Embedding Drift Analysis (April 2025 Shock)
    # -------------------------------------------------------------
    print("\n--- Diagnostic 3: Temporal Embedding Drift (z_t vs z_{t+1}) ---")
    daily_drift = []
    for t in range(n_test_days - 1):
        sims = []
        for i in range(n_assets):
            # Cosine similarity between z_{i, t} and z_{i, t+1}
            v_t = z_test[t, i]
            v_next = z_test[t + 1, i]
            cos_sim = np.dot(v_t, v_next) / (np.linalg.norm(v_t) * np.linalg.norm(v_next) + 1e-8)
            sims.append(cos_sim)
        # Drift = 1.0 - Cosine Similarity
        daily_drift.append(1.0 - np.mean(sims))

    drift_dates = test_dates[:-1]
    df_drift = pd.Series(daily_drift, index=drift_dates)

    # In-window stress episode (see config.STRESS_WINDOW_* / config.py docstring)
    stress_start_ts = pd.Timestamp(config.STRESS_WINDOW_START)
    stress_end_ts = pd.Timestamp(config.STRESS_WINDOW_END)
    in_stress_window = (df_drift.index >= stress_start_ts) & (df_drift.index <= stress_end_ts)
    shock_window = df_drift.loc[in_stress_window] if in_stress_window.any() else df_drift.tail(20)
    baseline_drift = df_drift.loc[df_drift.index < stress_start_ts].mean()
    shock_drift_max = shock_window.max()
    shock_drift_mean = shock_window.mean()

    print(f"Baseline Mean Daily Embedding Drift (pre-stress): {baseline_drift:.6f}")
    print(f"{config.STRESS_WINDOW_LABEL} Mean Drift: {shock_drift_mean:.6f} (+{((shock_drift_mean/baseline_drift)-1)*100:.1f}% surge)")
    print(f"{config.STRESS_WINDOW_LABEL} Peak Drift: {shock_drift_max:.6f} (Spike Date: {shock_window.idxmax().strftime('%Y-%m-%d')})")

    # Plot Drift Over Time
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(drift_dates, df_drift.values * 1000.0, color="#d62728", lw=1.8, label="Daily Embedding Drift (1 - Cosine Sim x 10^-3)")
    ax.axvspan(stress_start_ts, stress_end_ts, color="orange", alpha=0.25, label=config.STRESS_WINDOW_LABEL)
    ax.set_title("Temporal Representation Drift of GAT Embeddings Over Time", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Representation Drift (x 10^-3)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    drift_plot_path = config.CACHE_DIR / "embedding_drift_plot.png"
    plt.savefig(drift_plot_path)
    plt.close()
    print(f"Saved Embedding Drift plot to {drift_plot_path.name}")

    # -------------------------------------------------------------
    # 4. Predictive Probing (Directional Return Prediction)
    # -------------------------------------------------------------
    print("\n--- Diagnostic 4: Linear / Logistic Predictive Probe ---")
    z_all = outputs_all["z2"]  # (1077, 30, 96)
    r_all = dataset_dict["next_returns"]  # (1077, 30)

    # Flatten train+val vs test
    train_val_idx = np.concatenate([train_idx, val_idx])
    
    X_probe_train = z_all[train_val_idx].reshape(-1, d_dim)
    y_probe_train = (r_all[train_val_idx].reshape(-1) > 0).astype(int)

    X_probe_test = z_all[test_idx].reshape(-1, d_dim)
    y_probe_test = (r_all[test_idx].reshape(-1) > 0).astype(int)

    probe_model = LogisticRegression(C=1.0, max_iter=500, random_state=config.SEED)
    probe_model.fit(X_probe_train, y_probe_train)

    y_pred_prob = probe_model.predict_proba(X_probe_test)[:, 1]
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_probe_test, y_pred_class)
    f1 = f1_score(y_probe_test, y_pred_class)
    auc = roc_auc_score(y_probe_test, y_pred_prob)

    print(f"Directional Prediction Accuracy: {acc*100:.2f}%")
    print(f"Directional Prediction F1-Score: {f1:.4f}")
    print(f"Directional Prediction ROC-AUC:  {auc:.4f}")

    # Cross-sectional Hit-Rate @ 1 and @ 3
    y_prob_matrix = y_pred_prob.reshape(n_test_days, n_assets)
    y_true_matrix = (r_test > 0).astype(int)

    hit_1 = []
    hit_3 = []
    for t in range(n_test_days):
        ranked_assets = np.argsort(y_prob_matrix[t])[::-1]  # Highest prob first
        # Hit @ 1: Did top-1 predicted asset have positive return?
        hit_1.append(y_true_matrix[t, ranked_assets[0]])
        # Hit @ 3: Proportion of top-3 predicted assets with positive returns
        hit_3.append(np.mean(y_true_matrix[t, ranked_assets[:3]]))

    hit_rate_1 = np.mean(hit_1) * 100.0
    hit_rate_3 = np.mean(hit_3) * 100.0
    print(f"Top-1 Stock Hit Rate (Hit@1):     {hit_rate_1:.2f}% (Chance level: ~52%)")
    print(f"Top-3 Stock Hit Rate (Hit@3):     {hit_rate_3:.2f}%")

    # -------------------------------------------------------------
    # 5. Concentration Risk Scan (Eigenvector Centrality)
    # -------------------------------------------------------------
    print("\n--- Diagnostic 5: Graph Eigenvector Centrality & Concentration Risk Scan ---")
    # For each test day, build graph from attention matrix, compute centrality
    centrality_records = []
    for t in range(n_test_days):
        A_attn_t = alpha2_test[t]  # (30, 30)
        G = nx.from_numpy_array(A_attn_t, create_using=nx.DiGraph)
        try:
            cent = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
            c_vec = np.array([cent[i] for i in range(n_assets)])
        except Exception:
            c_vec = np.ones(n_assets) / np.sqrt(n_assets)

        w_t = np.abs(weights_test[t])
        c_weighted_exposure = c_vec * w_t
        centrality_records.append(c_weighted_exposure)

    cent_array = np.array(centrality_records)  # (325, 30)
    
    # Flag threshold > 0.6
    flagged_mask = cent_array > 0.60
    n_flagged = np.sum(flagged_mask)
    print(f"Concentration Violations (> 0.60 Threshold): {n_flagged} dates")

    # Top 5 Hidden Concentrations
    mean_centrality_exposure = np.mean(cent_array, axis=0)
    sorted_risk_indices = np.argsort(mean_centrality_exposure)[::-1]

    print("\nTop 5 Systemic Concentration Risk Stocks (Centrality x Weight Exposure):")
    for rank, idx in enumerate(sorted_risk_indices[:5], 1):
        ticker = config.TICKER_LIST[idx]
        sector = config.TICKERS[ticker]
        exp = mean_centrality_exposure[idx]
        max_exp = np.max(cent_array[:, idx])
        print(f"  {rank}. {ticker:>5} ({sector:<22}) | Mean Centrality Exposure: {exp:.4f} | Max Peak: {max_exp:.4f}")

    print("\n" + "=" * 80)
    print("TASK 1 DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_diagnostics()
