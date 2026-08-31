"""
Live Risk Radar and Sector Shock Contagion CLI.
"""

import argparse
import json
import logging

import numpy as np
import torch

import config
from data.cleaning import clean_and_merge_panel
from data.features import compute_features, fit_transform_scalers
from models.lstm_gat import LSTMGATModel
from risk.shock_engine import ShockSimulator
from train import build_full_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trained_model(device: torch.device) -> LSTMGATModel:
    hp_path = config.CACHE_DIR / "best_hyperparameters.json"
    weights_path = config.CACHE_DIR / "final_retrained_model.pt"
    if not hp_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {hp_path.name} and/or {weights_path.name} in {config.CACHE_DIR}. "
            f"Run `python tune.py` first to produce a trained checkpoint."
        )
    with open(hp_path, "r") as f:
        hp = json.load(f)

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
        top_k=hp.get("top_k", None),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_scaler_stds() -> np.ndarray:
    """Recomputes the per-ticker training-window StandardScaler std devs
    (shape (NUM_ASSETS, NUM_FEATURES)), in config.TICKER_LIST order, needed to
    de-standardize shock magnitudes back into feature-scale units.
    """
    panel = clean_and_merge_panel()
    raw_feats = compute_features(panel)
    _, scalers = fit_transform_scalers(raw_feats)
    return np.array([scalers[t].scale_ for t in config.TICKER_LIST])


def main():
    parser = argparse.ArgumentParser(description="Run live risk monitoring or a shock simulation.")
    parser.add_argument("--mode", choices=["live", "shock"], default="live")
    parser.add_argument("--sector", type=str, default=None, help="e.g. 'Information Technology'")
    parser.add_argument("--ticker", type=str, default=None, help="e.g. 'AAPL' (single-name shock)")
    parser.add_argument("--magnitude", type=float, default=-0.15)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(device)
    scaler_stds = load_scaler_stds()

    sim = ShockSimulator(model, config.TICKER_LIST, config.TICKERS, scaler_stds)

    dataset_dict, _, _ = build_full_dataset()
    x_last = torch.tensor(dataset_dict["X"][-1:], dtype=torch.float32, device=device)
    adj_last = torch.tensor(dataset_dict["adj"][-1:], dtype=torch.float32, device=device)
    date_str = str(dataset_dict["dates"][-1])[:10]

    if args.mode == "live":
        result = sim.analyze_live(x_last, adj_last, date_str=date_str)
        logger.info(f"Live risk snapshot for {date_str}: centrality_risk={result['centrality_risk']:.4f}, "
                    f"community_entropy={result['community_entropy']:.4f}, alerts={len(result['alerts'])}")
        for a in result["alerts"]:
            logger.info(f"  [ALERT] {a}")
    else:
        if args.ticker:
            results = sim.inject_node_shock(x_last, adj_last, args.ticker, magnitude=args.magnitude,
                                             steps=args.steps, date_str=date_str)
            target = args.ticker
        elif args.sector:
            results = sim.inject_sector_shock(x_last, adj_last, args.sector, magnitude=args.magnitude,
                                               steps=args.steps, date_str=date_str)
            target = args.sector
        else:
            parser.error("--mode shock requires either --sector or --ticker")
            return

        final = results[-1]
        metrics = sim._metrics_for(final)
        logger.info(f"Shock on '{target}' ({args.magnitude:+.0%}, {args.steps} steps) for {date_str}: "
                    f"centrality_risk={metrics['centrality_risk']:.4f}, "
                    f"community_entropy={metrics['community_entropy']:.4f}")
        for a in metrics["alerts"]:
            logger.info(f"  [ALERT] {a['action']}")


if __name__ == "__main__":
    main()
