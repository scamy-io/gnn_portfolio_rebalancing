import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import RESULTS_DIR, VAL_END
from src.model.dataset import (StockDayDataset, assign_split, collate_samples,
                               top_bottom_labels)
from src.model.thgnn import THGNN
from src.model.train import daily_rankic, nw_t_stat, spearman

torch.manual_seed(0)
np.random.seed(0)

N, T, F_CH = 8, 20, 7
REL = ("pos", "sent")


def _mk_sample(rng, y):
    label, lmask = top_bottom_labels(y, n_top=2, n_bottom=2)
    return {"date": pd.Timestamp("2023-01-03"),
            "x": rng.normal(0, 1, (N, T, F_CH)).astype(np.float32),
            "y": y.astype(np.float32),
            "ymask": ~np.isnan(y),
            "label": label, "lmask": lmask,
            "ei_pos": np.array([[0, 1], [1, 0]], dtype=np.int64),
            "ew_pos": np.array([0.9, 0.9], dtype=np.float32),
            "ei_sent": np.zeros((2, 0), dtype=np.int64),
            "ew_sent": np.zeros((0,), dtype=np.float32)}


def test_label_protocol_with_nan():
    y = np.array([5, 1, 4, 2, 3, 9, 0, 8, np.nan, 7, 6, 10])
    lab, m = top_bottom_labels(y, n_top=3, n_bottom=2)
    assert m.sum() == 5 and not m[8]
    assert set(np.where(lab == 1)[0]) == {5, 7, 11}
    assert set(np.where((lab == 0) & m)[0]) == {1, 6}


def test_assign_split_boundaries():
    d = pd.to_datetime(["2020-12-31", "2021-01-01", "2021-12-31",
                        "2022-01-01"])
    assert list(assign_split(d)) == ["train", "val", "val", "test"]


def test_collate_union_disjoint():
    rng = np.random.default_rng(0)
    batch = collate_samples([_mk_sample(rng, np.arange(N, dtype=float)),
                             _mk_sample(rng, np.arange(N)[::-1].astype(float))], REL)
    ei = batch["ei_pos"]
    assert batch["x"].shape == (2, N, T, F_CH)
    assert ei.shape[1] == 4 and ei.max() < 2 * N
    assert set(map(tuple, ei.T.tolist())) == {(0, 1), (1, 0),
                                              (N, N + 1), (N + 1, N)}
    assert batch["ei_sent"].shape == (2, 0)


def test_forward_shapes_and_empty_edges():
    rng = np.random.default_rng(1)
    model = THGNN(d_feat=F_CH, T=T, d_in=16, d_enc=16, d_att=16,
                  n_enc_heads=2, n_gat_heads=2, d_ff=32, relations=REL)
    batch = collate_samples([_mk_sample(rng, np.arange(N, dtype=float)) for _ in range(2)], REL)
    edges = {r: (batch[f"ei_{r}"], batch[f"ew_{r}"]) for r in REL}
    scores, Z = model(batch["x"], edges)
    assert scores.shape == (2, N) and Z.shape == (2, N, 16)
    assert torch.isfinite(scores).all()


def _fit(samples, epochs=120):
    model = THGNN(d_feat=F_CH, T=T, d_in=16, d_enc=16, d_att=16,
                  n_enc_heads=2, n_gat_heads=2, d_ff=32, d_hga=8, relations=REL)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    dl = DataLoader(StockDayDataset(samples), batch_size=40, shuffle=False,
                    collate_fn=lambda b: collate_samples(b, REL))
    loss = None
    for _ in range(epochs):
        for batch in dl:
            edges = {r: (batch[f"ei_{r}"], batch[f"ew_{r}"]) for r in REL}
            scores, _ = model(batch["x"], edges)
            lm = batch["lmask"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scores[lm], batch["label"][lm])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, float(loss)


def test_overfit_tiny_and_shuffle_collapse():
    rng = np.random.default_rng(7)
    D = 60
    Fsig = rng.normal(0, 1, (D, N, T, F_CH)).astype(np.float32)
    sig = Fsig[:, :, :, 0].mean(axis=2)
    z = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
    y_true = 2.0 * z + 0.1 * rng.normal(0, 1, (D, N))
    samples = []
    for d in range(D):
        s = _mk_sample(rng, y_true[d])
        s["x"] = Fsig[d]
        samples.append(s)

    model, loss = _fit(samples)
    assert loss < 0.6, f"tiny overfit failed: loss={loss}"
    batch = collate_samples(samples, REL)
    edges = {r: (batch[f"ei_{r}"], batch[f"ew_{r}"]) for r in REL}
    with torch.no_grad():
        scores, _ = model(batch["x"], edges)
    ic = daily_rankic(scores.numpy(), batch["y"].numpy(),
                      batch["ymask"].numpy())
    assert ic.mean() > 0.4, f"planted signal not learned: IC={ic.mean()}"

    idx = rng.permutation(D * N)
    y_shuf = y_true.reshape(-1)[idx].reshape(D, N)
    samples_sh = []
    for d in range(D):
        s = _mk_sample(rng, y_shuf[d])
        s["x"] = Fsig[d]
        samples_sh.append(s)
    model2, _ = _fit(samples_sh)
    with torch.no_grad():
        scores2, _ = model2(batch["x"], edges)
    ic2 = daily_rankic(scores2.numpy(), batch["y"].numpy(),
                       batch["ymask"].numpy())
    assert abs(ic2.mean()) < 0.25, f"shuffled model still predicts: IC={ic2.mean()}"


def test_spearman_and_nw_t():
    assert np.isclose(spearman(np.arange(5.0), np.array([0, 10, 20, 30, 40])), 1.0)
    assert nw_t_stat(np.array([0.5, 0.5, 0.5])) == 0.0
    rng = np.random.default_rng(0)
    ic = rng.normal(0.05, 0.19, 500)
    t = nw_t_stat(ic, lag=5)
    assert 2.0 < t < 8.0


@pytest.mark.skipif(not (RESULTS_DIR / "preds_C_seed42.parquet").exists(),
                    reason="run: python -m src.model.train --model C --seed 42")
def test_training_outputs_integration():
    preds = pd.read_parquet(RESULTS_DIR / "preds_C_seed42.parquet")
    metrics = json.loads((RESULTS_DIR / "metrics_C_seed42.json").read_text())
    assert (pd.to_datetime(preds["date"]) > pd.Timestamp(VAL_END)).all()
    assert np.isfinite(preds["score"]).all()
    assert metrics["test_days"] > 200 and metrics["test_days"] < 520
