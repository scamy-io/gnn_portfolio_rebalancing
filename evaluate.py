"""
Portfolio performance evaluation and backtest reporting.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from benchmarks import compute_equal_weight_benchmark
from data.graph import build_dynamic_adjacency_matrices
from models.lstm_gat import LSTMGATModel
from train import build_full_dataset, evaluate_split, PortfolioDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    daily_returns: pd.Series,
    rf_series: pd.Series
) -> Dict[str, float]:
    """
    Computes cumulative return, annualized return, volatility, Sharpe ratio, VaR, and max drawdown.
    """
    r = daily_returns.values
    n_days = len(r)

    cum_growth = np.cumprod(1.0 + r)
    cum_return = float(cum_growth[-1] - 1.0) if n_days > 0 else 0.0
    ann_return = float((1.0 + cum_return) ** (252.0 / max(n_days, 1)) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(252.0)) if n_days > 1 else 1e-6

    rf_aligned = rf_series.reindex(daily_returns.index).ffill().bfill()
    mean_rf_ann = float(rf_aligned.mean())
    ann_sharpe = float((ann_return - mean_rf_ann) / max(ann_vol, 1e-8))

    var_95 = float(np.percentile(r, 5.0))

    running_max = np.maximum.accumulate(cum_growth)
    drawdowns = (cum_growth / np.maximum(running_max, 1e-8)) - 1.0
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "Cumulative Return (%)": cum_return * 100.0,
        "Annualized Return (%)": ann_return * 100.0,
        "Annualized Volatility (%)": ann_vol * 100.0,
        "Sharpe Ratio": ann_sharpe,
        "VaR (95%) (%)": var_95 * 100.0,
        "Max Drawdown (%)": max_dd * 100.0,
        "Mean Annualized Rf (%)": mean_rf_ann * 100.0,
    }


def generate_evaluation_report(
    model_weights_path: Optional[Path] = None,
    output_dir: Path = config.CACHE_DIR
) -> pd.DataFrame:
    """
    Evaluates final Model v4 against Equal-Weight on the held-out test split
    [TEST_START, TEST_END] (config.py: 2022-07-15 to 2023-12-15).
    Produces metrics summary table and visualization plots.
    """
    if model_weights_path is None:
        model_weights_path = config.CACHE_DIR / "final_retrained_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading Model v4 weights from {model_weights_path}...")

    # Load dataset
    dataset_dict, date_to_adj, panel = build_full_dataset()
    all_dates = dataset_dict["dates"]

    test_mask = config.filter_split(all_dates, "test")
    test_indices = np.where(test_mask)[0]
    test_dates = all_dates[test_indices]
    logger.info(f"Evaluating over {len(test_indices)} test trading dates ({test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')})...")

    # Load best hyperparameters if available
    hp_path = config.CACHE_DIR / "best_hyperparameters.json"
    if hp_path.exists():
        with open(hp_path, "r") as f:
            hp = json.load(f)
        logger.info(f"Using tuned hyperparameters from {hp_path.name}")

        # If graph topology was tuned, rebuild adjacency with optimal settings
        if "corr_threshold" in hp or "refresh_days" in hp:
            corr_th = hp.get("corr_threshold", config.SECTOR_CORR_THRESHOLD)
            ref_days = hp.get("refresh_days", config.DYNAMIC_GRAPH_REFRESH_DAYS)
            date_to_adj, _ = build_dynamic_adjacency_matrices(panel, refresh_days=ref_days, corr_threshold=corr_th)
            adj_list = [date_to_adj[d] for d in all_dates]
            dataset_dict["adj"] = np.array(adj_list, dtype=np.float32)

        model = LSTMGATModel(
            num_features=config.NUM_FEATURES,
            lookback_r=config.LOOKBACK_R,
            num_assets=config.NUM_ASSETS,
            lstm_hidden=hp.get("hidden_size", config.LSTM_HIDDEN_SIZE),
            lstm_dropout=hp.get("lstm_dropout", config.LSTM_DROPOUT),
            gat_hidden=hp.get("hidden_size", config.GAT_HIDDEN_SIZE),
            gat_layers=config.GAT_LAYERS,
            gat_heads=hp.get("gat_heads", config.GAT_HEADS),
            gat_dropout=hp.get("gat_dropout", config.GAT_DROPOUT),
            leaky_relu_alpha=hp.get("gat_alpha", config.LEAKY_RELU_ALPHA),
            final_dropout=hp.get("final_dropout", config.FINAL_DROPOUT),
            tilt_scale=hp.get("tilt_scale", None),
            top_k=hp.get("top_k", None)
        ).to(device)
    else:
        model = LSTMGATModel(num_assets=config.NUM_ASSETS).to(device)

    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    model.eval()

    # Extract test tensors
    sub_x = torch.tensor(dataset_dict["X"][test_indices], dtype=torch.float32, device=device)
    sub_adj = torch.tensor(dataset_dict["adj"][test_indices], dtype=torch.float32, device=device)
    sub_r = dataset_dict["next_returns"][test_indices]

    with torch.no_grad():
        w_pred = model(sub_x, sub_adj).cpu().numpy()

    # Compute daily model returns: sum_i (w_i,t * r_i,t+1)
    model_daily_returns = np.sum(w_pred * sub_r, axis=-1)
    model_series = pd.Series(model_daily_returns, index=test_dates, name="Model v4 (LSTM-GAT)")

    # Compute equal-weight returns
    ew_series = compute_equal_weight_benchmark(sub_r, test_dates)

    # Risk-free rate series for test period
    rf_test_series = panel.xs(config.TICKER_LIST[0], level="ticker")["rf_annualized"].reindex(test_dates)

    # Calculate metrics
    model_metrics = compute_metrics(model_series, rf_test_series)
    ew_metrics = compute_metrics(ew_series, rf_test_series)

    # Paper reference benchmark (9-Stock universe, Jan 2021 - May 2025)
    paper_ref_metrics = {
        "Cumulative Return (%)": np.nan,
        "Annualized Return (%)": 28.10,
        "Annualized Volatility (%)": 26.60,
        "Sharpe Ratio": 1.06,
        "VaR (95%) (%)": -2.68,
        "Max Drawdown (%)": -21.70,
        "Mean Annualized Rf (%)": np.nan,
    }

    # Build Table 2 Comparison DataFrame
    df_metrics = pd.DataFrame({
        "Model v4 (Tuned Optuna)": model_metrics,
        "Equal-Weight Benchmark": ew_metrics,
        "Paper Model v4 (Original 9-Stock, 2021-2025, Reference Only)": paper_ref_metrics
    })

    # Save metrics table
    output_dir.mkdir(parents=True, exist_ok=True)
    table_csv_path = output_dir / "evaluation_metrics_table.csv"
    df_metrics.to_csv(table_csv_path)
    logger.info(f"Saved evaluation metrics to {table_csv_path}")

    # ==========================================
    # Plot 1: Cumulative Return Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    model_cum = (np.cumprod(1.0 + model_daily_returns) - 1.0) * 100.0
    ew_cum = (np.cumprod(1.0 + ew_series.values) - 1.0) * 100.0

    ax.plot(test_dates, model_cum, label=f"Model v4 (Cum: {model_metrics['Cumulative Return (%)']:.1f}%, Sharpe: {model_metrics['Sharpe Ratio']:.2f})", color="#1f77b4", lw=2.0)
    ax.plot(test_dates, ew_cum, label=f"Equal-Weight (Cum: {ew_metrics['Cumulative Return (%)']:.1f}%, Sharpe: {ew_metrics['Sharpe Ratio']:.2f})", color="#7f7f7f", lw=1.8, ls="--")

    ax.set_title("Cumulative Test Returns: Model v4 (Tuned LSTM-GAT) vs Equal-Weight Benchmark", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    cum_plot_path = output_dir / "cumulative_return_plot.png"
    plt.savefig(cum_plot_path)
    plt.close()
    logger.info(f"Saved cumulative return plot to {cum_plot_path}")

    # ==========================================
    # Plot 2: Predicted Weights Over Time
    # ==========================================
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    for i, ticker in enumerate(config.TICKER_LIST):
        sector_name = config.TICKERS.get(ticker, "Other")
        ax.plot(test_dates, w_pred[:, i], label=f"{ticker} ({sector_name})", lw=1.2)

    ax.axhline(0.0, color="black", linestyle="--", alpha=0.5, lw=1.0)
    ax.set_title(f"Model v4 Predicted Asset Allocations Over Time ({config.NUM_ASSETS}-Stock Universe)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Weight", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    weights_plot_path = output_dir / "predicted_weights_plot.png"
    plt.savefig(weights_plot_path)
    plt.close()
    logger.info(f"Saved predicted weights plot to {weights_plot_path}")

    # Copy plots to dedicated plots/ directory
    plots_dir = config.PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(cum_plot_path, plots_dir / "cumulative_return_plot.png")
    shutil.copy(weights_plot_path, plots_dir / "predicted_weights_plot.png")
    logger.info(f"Published visual plots to {plots_dir}")

    # Print Table to console
    print("\n" + "=" * 80)
    print(f"TABLE 2 REPRODUCTION & EVALUATION METRICS (TEST SET: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')})")
    print("=" * 80)
    print(df_metrics.round(2).to_string())
    print("=" * 80 + "\n")

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model v4 against benchmark")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights checkpoint")
    args = parser.parse_args()

    weights_p = Path(args.weights) if args.weights else None
    generate_evaluation_report(model_weights_path=weights_p)
