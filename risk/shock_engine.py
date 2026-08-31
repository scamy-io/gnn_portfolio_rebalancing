"""
Inference, non-linear shock propagation, and live risk monitoring engine.
"""

import datetime
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from risk.concentration import compute_all_metrics


def extract_intermediates(model, x, adj):
    """
    Forward pass extracting intermediate activations and attention weights.
    """
    model.eval()
    with torch.no_grad():
        try:
            res = model(x, adj, return_intermediates=True)
            if isinstance(res, dict):
                return res["h_nodes"], res["z1"], res["z2"], res["alpha1"], res["alpha2"], res["weights"]
        except Exception:
            pass

        # Fallback for simple mock models
        B, N, R, F_in = x.shape
        x_flat = x.view(B * N, R, F_in)
        _, (h_n, _) = model.lstm(x_flat)
        h_nodes = model.lstm_dropout(h_n.squeeze(0)).view(B, N, -1)
        z1 = h_nodes
        z2 = h_nodes
        alpha1 = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        alpha2 = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        weights = torch.ones(B, N, device=x.device) / N
        return h_nodes, z1, z2, alpha1, alpha2, weights


class ShockSimulator:
    def __init__(self, model, tickers, sectors, scaler_stds=None):
        self.model = model
        self.model.eval()
        self.tickers = tickers
        self.sectors = sectors
        self.scaler_stds = scaler_stds if scaler_stds is not None else np.ones((len(tickers), 10))

        n = len(tickers)
        S = np.zeros((n, n), dtype=np.float32)
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if sectors.get(t1) == sectors.get(t2):
                    S[i, j] = 1.0
        self.S_sector = S

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _inject_shock_to_indices(self, X_t, indices, magnitude):
        X_shocked = X_t.clone()
        for i in indices:
            X_shocked[0, i, 29, 2] += (magnitude / self.scaler_stds[i, 2])
            X_shocked[0, i, 29, 3] += ((magnitude * (252 / 5)) / self.scaler_stds[i, 3])
            X_shocked[0, i, 29, 6] += (0.08 / self.scaler_stds[i, 6])
            X_shocked[0, i, 29, 9] -= (1.50 / self.scaler_stds[i, 9])
        return X_shocked

    def _propagate(self, X_shocked, A_t, steps):
        N = len(self.tickers)
        results = []
        A_curr = A_t.clone()
        for _ in range(steps):
            with torch.no_grad():
                h, z1, z2, alpha1, alpha2, w = extract_intermediates(self.model, X_shocked, A_curr)
            results.append({
                "weights": w.squeeze(0).cpu().numpy(),
                "alpha_t": alpha2.squeeze(0).cpu().numpy(),
                "z2": z2.squeeze(0).cpu().numpy(),
                "A_t": A_curr.squeeze(0).cpu().numpy(),
            })

            z2_np = z2.squeeze(0).cpu().numpy()
            norms = np.linalg.norm(z2_np, axis=-1, keepdims=True) + 1e-8
            z_norm = z2_np / norms
            cos_sim = np.dot(z_norm, z_norm.T)
            A_embed = (np.abs(cos_sim) > 0.60).astype(np.float32)

            A_new = np.clip(self.S_sector + A_embed + np.eye(N), 0.0, 1.0)
            A_curr = torch.tensor(A_new, dtype=torch.float32, device=X_shocked.device).unsqueeze(0)

        return results

    def _baseline_result(self, X_t, A_t, steps):
        with torch.no_grad():
            h, z1, z2, alpha1, alpha2, w = extract_intermediates(self.model, X_t, A_t)
        res = {
            "weights": w.squeeze(0).cpu().numpy(),
            "alpha_t": alpha2.squeeze(0).cpu().numpy(),
            "z2": z2.squeeze(0).cpu().numpy(),
            "A_t": A_t.squeeze(0).cpu().numpy(),
        }
        return [res] * steps

    def _metrics_for(self, result):
        return compute_all_metrics(
            result["weights"], result["alpha_t"], result["A_t"], self.tickers, self.sectors
        )

    def _save_shock_json(self, results, target, magnitude, steps, date_str):
        date_str = date_str or datetime.date.today().isoformat()
        metrics = self._metrics_for(results[-1])

        payload = {
            "date": date_str,
            "mode": "shock",
            "shock_config": {"target": target, "magnitude": magnitude, "steps": steps},
            "metrics": {
                "centrality_risk": metrics["centrality_risk"],
                "community_entropy": metrics["community_entropy"],
                "num_communities": metrics["num_communities"],
            },
            "alerts": metrics["alerts"],
            "stock_metrics": metrics["stock_metrics"],
        }

        out_dir = "data/cache/shock_results"
        os.makedirs(out_dir, exist_ok=True)
        safe_target = str(target).replace(" ", "_")
        path = os.path.join(out_dir, f"{date_str}_{safe_target}_{magnitude}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def _save_live_json(self, metrics, date_str):
        out_dir = "data/cache/live_risk"
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "date": date_str,
            "mode": "live",
            "shock_config": None,
            "metrics": {
                "centrality_risk": metrics["centrality_risk"],
                "community_entropy": metrics["community_entropy"],
                "num_communities": metrics["num_communities"],
            },
            "alerts": metrics["alerts"],
            "stock_metrics": metrics["stock_metrics"],
        }
        path = os.path.join(out_dir, f"{date_str}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def inject_sector_shock(self, X_t, A_t, sector, magnitude=-0.15, steps=5, save=True, date_str=None):
        target_indices = [i for i, t in enumerate(self.tickers) if self.sectors.get(t) == sector]
        if sector == "Single" and isinstance(magnitude, dict) and "ticker" in magnitude:
            target_indices = [i for i, t in enumerate(self.tickers) if t == magnitude["ticker"]]
            magnitude = magnitude["val"]

        if magnitude == 0.0:
            results = self._baseline_result(X_t, A_t, steps)
        else:
            X_shocked = self._inject_shock_to_indices(X_t, target_indices, magnitude)
            results = self._propagate(X_shocked, A_t, steps)

        if save:
            self._save_shock_json(results, sector, magnitude, steps, date_str)

        return results

    def inject_node_shock(self, X_t, A_t, ticker, magnitude=-0.15, steps=5, save=True, date_str=None):
        """Shock a single stock (e.g., 'AAPL') and propagate. Reuses the same
        propagation loop as inject_sector_shock."""
        target_indices = [i for i, t in enumerate(self.tickers) if t == ticker]
        if not target_indices:
            raise ValueError(f"Unknown ticker: {ticker}")

        if magnitude == 0.0:
            results = self._baseline_result(X_t, A_t, steps)
        else:
            X_shocked = self._inject_shock_to_indices(X_t, target_indices, magnitude)
            results = self._propagate(X_shocked, A_t, steps)

        if save:
            self._save_shock_json(results, ticker, magnitude, steps, date_str)

        return results

    def analyze_live(self, X_t, A_t, date_str=None, save=True):
        """
        Run on CURRENT portfolio (no shock injected). Extracts intermediates,
        builds the attention-weighted graph, computes concentration metrics,
        and generates alerts. Returns a dict ready for JSON export.
        """
        with torch.no_grad():
            h, z1, z2, alpha1, alpha2, w = extract_intermediates(self.model, X_t, A_t)

        weights = w.squeeze(0).cpu().numpy()
        alpha_t = alpha2.squeeze(0).cpu().numpy()
        A_np = A_t.squeeze(0).cpu().numpy()

        # compute_all_metrics builds the attention-weighted graph internally
        # via build_weighted_graph(alpha_t, A_t), using alpha2 — the second/
        # final GAT layer's attention — as specified.
        metrics = compute_all_metrics(weights, alpha_t, A_np, self.tickers, self.sectors)
        date_str = date_str or datetime.date.today().isoformat()

        result = {
            "date": date_str,
            "weights": weights.tolist(),
            "centrality_risk": metrics["centrality_risk"],
            "community_entropy": metrics["community_entropy"],
            "num_communities": metrics["num_communities"],
            "hidden_2hop_exposure": metrics["hidden_2hop_exposure"],
            "communities": metrics["communities"],
            "alerts": [a["action"] for a in metrics["alerts"]],
            "stock_metrics": metrics["stock_metrics"],
        }

        if save:
            self._save_live_json(metrics, date_str)

        return result
