import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (GRAPHS_DIR, LOOKBACK, N_BOTTOM, N_TOP, PROCESSED_DIR,
                        SAMPLE_END, SAMPLE_START, TICKERS, TRAIN_END, VAL_END,
                        ensure_dirs)
from src.data.build_features import CHANNELS
from src.data.labels import compute_forward_returns

PRICE_CHANNELS = [c for c in CHANNELS
                  if c not in ("sent_ema", "log_count", "sent_count")]


def assign_split(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    out = np.empty(len(d), dtype=object)
    out[d <= pd.Timestamp(TRAIN_END)] = "train"
    out[(d > pd.Timestamp(TRAIN_END)) & (d <= pd.Timestamp(VAL_END))] = "val"
    out[d > pd.Timestamp(VAL_END)] = "test"
    return out


def top_bottom_labels(y_row, n_top=N_TOP, n_bottom=N_BOTTOM):
    y = np.asarray(y_row, dtype=float)
    label = np.zeros(len(y), dtype=np.float32)
    lmask = np.zeros(len(y), dtype=bool)
    valid = np.where(~np.isnan(y))[0]
    if len(valid) < n_top + n_bottom:
        return label, lmask
    order = valid[np.argsort(y[valid])]
    top, bottom = order[-n_top:], order[:n_bottom]
    label[top] = 1.0
    lmask[top] = True
    lmask[bottom] = True
    return label, lmask


def load_feature_tensor(panel):
    assert set(panel["ticker"].unique()) == set(TICKERS)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    dims = []
    for ch in CHANNELS:
        w = (panel.pivot(index="date", columns="ticker", values=ch)
                  .reindex(index=dates, columns=TICKERS))
        dims.append(w.to_numpy(dtype=np.float32))
    return dates, np.stack(dims, axis=2)


def load_forward_returns(prices):
    close = (prices.pivot(index="date", columns="ticker", values="close")
                   .reindex(columns=TICKERS).sort_index())
    return close.index, compute_forward_returns(close).to_numpy(dtype=np.float32)


def _empty_edges():
    return (np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32))


def load_graph_index(graphs_df):
    idx = {t: i for i, t in enumerate(TICKERS)}
    out = {}
    for (d, rel), g in graphs_df.groupby(["date", "relation"]):
        assert g["src"].isin(idx).all() and g["dst"].isin(idx).all()
        s = g["src"].map(idx).to_numpy(np.int64)
        t_ = g["dst"].map(idx).to_numpy(np.int64)
        w = g["weight"].to_numpy(np.float32)
        ei = np.stack([np.concatenate([s, t_]), np.concatenate([t_, s])])
        out.setdefault(pd.Timestamp(d), {})[rel] = (ei, np.concatenate([w, w]))
    return out


def build_samples(X, y, cal, graphs, relations, feature_channels):
    ch_idx = [CHANNELS.index(c) for c in feature_channels]
    valid_day = ~np.isnan(X).any(axis=(1, 2))
    samples, sk_window, sk_label = [], 0, 0
    for t in range(LOOKBACK - 1, len(cal)):
        d = cal[t]
        if d < pd.Timestamp(SAMPLE_START) or d > pd.Timestamp(SAMPLE_END):
            continue
        if not valid_day[t]:
            sk_window += 1
            continue
        y_t = y[t]
        if (~np.isnan(y_t)).sum() < N_TOP + N_BOTTOM:
            sk_label += 1
            continue
        label, lmask = top_bottom_labels(y_t)
        g = graphs.get(d, {})
        s = {"date": d,
             "x": X[t - LOOKBACK + 1: t + 1][:, :, ch_idx].transpose(1, 0, 2),
             "y": np.nan_to_num(y_t, nan=0.0).astype(np.float32),
             "ymask": ~np.isnan(y_t),
             "label": label, "lmask": lmask}
        for rel in relations:
            s[f"ei_{rel}"], s[f"ew_{rel}"] = g.get(rel, _empty_edges())
        samples.append(s)
    return samples, sk_window, sk_label


class StockDayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        item = {"x": torch.from_numpy(s["x"]),
                "y": torch.from_numpy(s["y"]),
                "ymask": torch.from_numpy(s["ymask"]),
                "label": torch.from_numpy(s["label"]),
                "lmask": torch.from_numpy(s["lmask"])}
        for k, v in s.items():
            if k.startswith(("ei_", "ew_")):
                item[k] = torch.from_numpy(v)
        item["date"] = s["date"]
        return item


def _to_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype)
    t = torch.from_numpy(np.asarray(x))
    return t if dtype is None else t.to(dtype)


def collate_samples(batch, relations):
    n = batch[0]["x"].shape[0]
    out = {k: torch.stack([_to_tensor(b[k]) for b in batch])
           for k in ("x", "y", "ymask", "label", "lmask")}
    for rel in relations:
        eis, ews, off = [], [], 0
        for b in batch:
            ei = _to_tensor(b[f"ei_{rel}"], torch.int64)
            ew = _to_tensor(b[f"ew_{rel}"], torch.float32)
            eis.append(ei + off)
            ews.append(ew)
            off += n
        out[f"ei_{rel}"] = torch.cat(eis, dim=1)
        out[f"ew_{rel}"] = torch.cat(ews, dim=0)
    out["dates"] = [b["date"] for b in batch]
    return out


def build_datasets(feature_set="full", relations=("pos", "neg", "sent")):
    ensure_dirs()
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "features_panel.parquet")
    graphs_df = pd.read_parquet(GRAPHS_DIR / "graphs.parquet")
    channels = PRICE_CHANNELS if feature_set == "price" else list(CHANNELS)

    cal, y = load_forward_returns(prices)
    dates_f, X = load_feature_tensor(panel)
    assert (dates_f == cal).all(), "feature calendar != price calendar"
    graphs = load_graph_index(graphs_df)

    samples, sk_w, sk_l = build_samples(X, y, cal, graphs, relations, channels)
    n_empty = sum(1 for s in samples
                  if all(len(s[f"ei_{r}"][0]) == 0 for r in relations))
    splits = assign_split([s["date"] for s in samples])
    ds = {k: StockDayDataset([s for s, m in zip(samples, splits) if m == k])
          for k in ("train", "val", "test")}
    print(f"[dataset] train={len(ds['train'])} val={len(ds['val'])} "
          f"test={len(ds['test'])} | skipped window={sk_w} label={sk_l} | "
          f"all-empty-graph days={n_empty} | features={channels}")
    return ds, channels
