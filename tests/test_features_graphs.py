import numpy as np
import pandas as pd
import pytest

from src.config import GRAPHS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.build_features import (CHANNELS, build_features_panel,
                                     compute_price_features,
                                     cross_sectional_zscore)
from src.data.build_graphs import (build_from_wides, corr_matrix,
                                   edges_from_mask, mask_from_threshold,
                                   sparsify)

P = dict(lookback=20, corr_threshold=0.6, sent_threshold=0.5,
         min_overlap=10, top_k=5, max_degree=10)


def test_zscore_same_day_only_and_stats():
    idx = pd.bdate_range("2023-01-02", periods=5)
    idx.name = "date"
    df = pd.DataFrame({"A": [1., 2., 3., 4., 5.],
                       "B": [2., 3., 4., 5., 6.],
                       "C": [3., 4., 5., 6., 7.]}, index=idx)
    z1 = cross_sectional_zscore(df)
    df2 = df.copy()
    df2.iloc[-1, 0] = 100.0
    z2 = cross_sectional_zscore(df2)
    pd.testing.assert_frame_equal(z1.iloc[:-1], z2.iloc[:-1])
    assert np.allclose(z1.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(z1.std(axis=1, ddof=0), 1, atol=1e-9)


def test_zscore_constant_day_gives_zeros_not_inf():
    idx = pd.bdate_range("2023-01-02", periods=2)
    idx.name = "date"
    df = pd.DataFrame({"A": [1., 5.], "B": [1., 6.]}, index=idx)
    z = cross_sectional_zscore(df)
    assert (z.iloc[0] == 0).all() and np.isfinite(z.to_numpy()).all()


def test_no_future_leak_features():
    idx = pd.bdate_range("2023-01-02", periods=30)
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, (30, 3)), axis=0)
    w1 = {"close": pd.DataFrame(base, index=idx, columns=list("ABC")),
          "open": pd.DataFrame(base * (1 + 0.001), index=idx, columns=list("ABC")),
          "volume": pd.DataFrame(1e6 + rng.integers(0, 1000, (30, 3)),
                                 index=idx, columns=list("ABC"))}
    f1 = compute_price_features(w1)
    cut = 20
    w2 = {k: v.copy() for k, v in w1.items()}
    for k in w2:
        w2[k].iloc[cut + 1:] = rng.normal(100, 5, (30 - cut - 1, 3))
    f2 = compute_price_features(w2)
    for ch in f1:
        pd.testing.assert_frame_equal(f1[ch].iloc[: cut + 1],
                                      f2[ch].iloc[: cut + 1])


def test_features_panel_shape_and_zstats():
    n_d, n_t = 60, 4
    idx = pd.bdate_range("2023-01-02", periods=n_d); idx.name = "date"
    tk = list("ABCD")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({
        "date": np.repeat(idx, n_t), "ticker": tk * n_d,
        "open": 100 + rng.normal(0, 1, n_d * n_t),
        "close": 100 + rng.normal(0, 1, n_d * n_t),
        "volume": rng.integers(1e5, 2e5, n_d * n_t).astype(float)})
    sent = pd.DataFrame({"date": np.repeat(idx, n_t), "ticker": tk * n_d,
                         "sent": rng.normal(0, .3, n_d * n_t),
                         "count": rng.integers(0, 5, n_d * n_t).astype(float),
                         "log_count": 0., "sent_ema": rng.normal(0, .3, n_d * n_t)})
    panel = build_features_panel(prices, sent)
    assert len(panel) == n_d * n_t and list(panel.columns)[2:] == CHANNELS
    z = panel.groupby("date")["ret"]
    assert (z.mean().dropna().abs() < 1e-9).all()


@pytest.fixture
def three_stock_returns():
    idx = pd.bdate_range("2023-01-02", periods=40)
    a = np.where(np.arange(40) % 2 == 0, 0.01, -0.01)
    r = pd.DataFrame({"A": a, "B": a, "C": -a}, index=idx)
    return r, idx


def test_corr_edge_semantics(three_stock_returns):
    r, idx = three_stock_returns
    C = corr_matrix(r.to_numpy())
    M_pos = mask_from_threshold(C, 0.6, signed=True)
    M_neg = mask_from_threshold(C, 0.6, signed=True)
    pos = edges_from_mask(C, mask_from_threshold(C, 0.6, True), list(r.columns))
    neg = edges_from_mask(C, mask_from_threshold(C, -0.6, True) |
                          mask_from_threshold(C, 0.6, False) &
                          (C < -0.6), list(r.columns)) if False else \
          edges_from_mask(C, (C < -0.6) & ~np.eye(3, dtype=bool), list(r.columns))
    assert {("A", "B")} <= {(a, b) for a, b, _ in pos}
    assert ("A", "C") not in {(a, b) for a, b, _ in pos}
    assert {("A", "C"), ("B", "C")} <= {(a, b) for a, b, _ in neg}


def test_overlap_filter_binds(three_stock_returns):
    r, idx = three_stock_returns
    r = r[["A", "B"]].rename(columns={"B": "X"})
    n = len(idx)
    rng = np.random.default_rng(2)
    def mk(days):
        s = pd.DataFrame(0.0, index=idx, columns=list("AX"))
        c = pd.DataFrame(0.0, index=idx, columns=list("AX"))
        for d in days:
            s.iloc[d, :] = 1.0
            c.iloc[d, :] = 1.0
        return s, c

    s5, c5 = mk(range(20, 25))
    e5, _ = build_from_wides(r, s5, c5, start=idx[25], end=idx[-1], **P)
    assert e5[e5.relation == "sent"].empty

    s12, c12 = mk(range(20, 32))
    e12, _ = build_from_wides(r, s12, c12, start=idx[25], end=idx[-1], **P)
    sent12 = e12[e12.relation == "sent"]
    assert {("A", "X")} <= set(zip(sent12["src"], sent12["dst"]))


def test_no_future_leak_graphs(three_stock_returns):
    r, idx = three_stock_returns
    rng = np.random.default_rng(3)
    s = pd.DataFrame(rng.normal(0, .2, r.shape), index=r.index, columns=r.columns)
    c = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    cut = 25
    e1, _ = build_from_wides(r, s, c, start=idx[5], end=idx[-1], **P)
    r2 = r.copy(); r2.iloc[cut + 1:] = rng.normal(0, 1, (len(idx) - cut - 1, 3))
    e2, _ = build_from_wides(r2, s, c, start=idx[5], end=idx[-1], **P)
    e1_old = e1[e1.date <= idx[cut]]
    e2_old = e2[e2.date <= idx[cut]]
    pd.testing.assert_frame_equal(e1_old, e2_old)


def test_sparsify_caps_degree():
    C = np.array([[0., .9, .3, .1],
                  [.9, 0., .8, .2],
                  [.3, .8, 0., .7],
                  [.1, .2, .7, 0.]])
    M = np.ones((4, 4), dtype=bool); np.fill_diagonal(M, False)
    out = sparsify(C, M, k=2, max_degree=2)
    deg = out.sum(axis=1)
    assert deg.max() <= 2 and out.sum() > 0
    assert out[0, 1] and out[1, 2]


@pytest.mark.skipif(not (GRAPHS_DIR / "graphs.parquet").exists(),
                    reason="run build_graphs first")
def test_graphs_integration():
    g = pd.read_parquet(GRAPHS_DIR / "graphs.parquet")
    stats = pd.read_csv(RESULTS_DIR / "graph_stats.csv", parse_dates=["date"])
    assert set(g["relation"].unique()) == {"pos", "neg", "sent"}
    assert (g["src"] != g["dst"]).all()
    assert not g.duplicated(["date", "relation", "src", "dst"]).any()
    assert (g.loc[g.relation == "pos", "weight"] > 0.6).all()
    assert (g.loc[g.relation == "neg", "weight"] < -0.6).all()
    assert (g.loc[g.relation == "sent", "weight"].abs() > 0.5).all()
    pos19 = stats.query("date >= '2019-01-01' and date <= '2019-12-31'")["pos_edges"]
    pos_covid = stats.query("date >= '2020-03-01' and date <= '2020-03-31'")["pos_edges"]
    assert pos_covid.max() > 1.2 * max(pos19.median(), 1)
    assert stats["sent_vs_price_jaccard"].mean() < 0.8
    assert (g.relation == "sent").sum() > 0
