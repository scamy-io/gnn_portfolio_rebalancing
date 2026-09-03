import numpy as np
import pandas as pd
import torch
from scipy import stats as st
from torch.utils.data import DataLoader

from src.config import (CORR_THRESHOLD, EMA_HALFLIFE, LOOKBACK, MAX_DEGREE,
                        PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SAMPLE_END,
                        SAMPLE_START, SENT_CORR_THRESHOLD,
                        SENT_MIN_OVERLAP_DAYS, TICKERS, TOP_K_NEIGHBORS,
                        ensure_dirs)
from src.data.build_features import build_features_panel, load_price_wide
from src.data.build_graphs import build_from_wides, corr_matrix
from src.data.build_sentiment import (aggregate_daily, assign_trading_day,
                                      build_daily_panel)
from src.eval.backtest import perf, weekly_backtest
from src.model.dataset import (PRICE_CHANNELS, StockDayDataset, assign_split,
                               build_samples, collate_samples,
                               load_feature_tensor, load_forward_returns,
                               load_graph_index)
from src.model.thgnn import THGNN
from src.model.train import evaluate, nw_t_stat

C1 = ("2022-01-01", "2022-10-31")
SEEDS = [42, 0, 1]


def make_ds(panel, prices, edges_df, relations, channels):
    cal, y = load_forward_returns(prices)
    dates_f, X = load_feature_tensor(panel)
    assert (dates_f == cal).all()
    graphs = load_graph_index(edges_df)
    samples, sk_w, sk_l = build_samples(X, y, cal, graphs, relations, channels)
    splits = assign_split([s["date"] for s in samples])
    return {k: StockDayDataset([s for s, m in zip(samples, splits) if m == k])
            for k in ("train", "val", "test")}


def train_one(ds, relations, channels, seed, device,
              epochs=120, bs=32, lr=3e-4, patience=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = THGNN(d_feat=len(channels), T=LOOKBACK, relations=relations).to(device)
    dl = DataLoader(ds["train"], batch_size=bs, shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                    collate_fn=lambda b: collate_samples(b, relations))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_ic, best_state, bad = -np.inf, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in dl:
            edges = {r: (batch[f"ei_{r}"].to(device), batch[f"ew_{r}"].to(device))
                     for r in relations}
            scores, _ = model(batch["x"].to(device), edges)
            lm = batch["lmask"].to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scores[lm], batch["label"].to(device)[lm])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vic = evaluate(model, ds["val"], relations, device)["ic"].mean()
        if vic > best_ic + 1e-4:
            best_ic, bad = vic, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, evaluate(model, ds["test"], relations, device)


def scores_frame(ev):
    rows = [{"date": d, "ticker": t, "score": ev["scores"][b, i]}
            for b, d in enumerate(ev["dates"])
            for i, t in enumerate(TICKERS) if ev["ymask"][b, i]]
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    news = pd.read_parquet(RAW_DIR / "fnspid_news_filtered.parquet")
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))

    tr = pd.read_csv(PROCESSED_DIR / "timestamp_robustness.csv",
                     index_col=0)["corr_same_vs_next"].sort_values()
    print("worst 5 midnight-sensitive tickers:",
          {k: round(v, 3) for k, v in tr.head(5).items()})

    daily = aggregate_daily(assign_trading_day(news, cal, "next_day"))
    panel_long = build_daily_panel(daily, cal, sorted(TICKERS), EMA_HALFLIFE)
    panel = build_features_panel(prices, panel_long)
    wides = load_price_wide(prices)
    ret_w = wides["close"].pct_change()
    sent_w = (panel_long.pivot(index="date", columns="ticker", values="sent_ema")
              .reindex(index=ret_w.index, columns=ret_w.columns).fillna(0.0))
    cnt_w = (panel_long.pivot(index="date", columns="ticker", values="count")
             .reindex(index=ret_w.index, columns=ret_w.columns).fillna(0.0))
    edges_nd, _ = build_from_wides(
        ret_w, sent_w, cnt_w, lookback=LOOKBACK,
        corr_threshold=CORR_THRESHOLD, sent_threshold=SENT_CORR_THRESHOLD,
        min_overlap=SENT_MIN_OVERLAP_DAYS, top_k=TOP_K_NEIGHBORS,
        max_degree=MAX_DEGREE, start=SAMPLE_START, end=SAMPLE_END)
    print(f"[next_day] edges: pos={len(edges_nd[edges_nd.relation == 'pos'])} "
          f"neg={len(edges_nd[edges_nd.relation == 'neg'])} "
          f"sent={len(edges_nd[edges_nd.relation == 'sent'])}")

    R = ret_w.to_numpy()
    off = ~np.eye(R.shape[1], dtype=bool)
    rows = []
    for t in range(LOOKBACK - 1, len(ret_w)):
        C = corr_matrix(R[t - LOOKBACK + 1: t + 1])
        if not np.isnan(C).all():
            rows.append((ret_w.index[t], np.nanmean(C[off])))
    mpc = pd.Series(dict(rows))
    b19 = mpc.loc["2019-01-01":"2019-12-31"].median()
    r20 = mpc.loc["2020-03-01":"2020-03-31"].max() / max(b19, 1e-9)
    r22 = mpc.loc["2022-01-01":"2022-10-31"].mean() / max(b19, 1e-9)
    print(f"[pre-cap stress] mean-pairwise-corr ratios: Mar20={r20:.2f} "
          f"2022={r22:.2f} (vs 2019 med {b19:.3f}) -> "
          f"{'PASS' if max(r20, r22) > 1.5 else 'WARN'}")

    out = {}
    for name, rels, chans in (("C", ("pos", "neg", "sent"), None),
                              ("A", ("pos", "neg"), PRICE_CHANNELS)):
        chans = chans or None
        if chans is None:
            from src.data.build_features import CHANNELS
            chans = list(CHANNELS)
        ds = make_ds(panel, prices, edges_nd, rels, chans)
        ics, bts = [], []
        for s in SEEDS:
            model, ev = train_one(ds, rels, chans, s, device)
            ic = pd.Series(ev["ic"], index=pd.to_datetime(ev["dates"]))
            ics.append(ic)
            bt = weekly_backtest(scores_frame(ev), wides["close"])
            bts.append(bt["port_ret"])
            print(f"[{name} nd seed{s}] testIC {ic.mean():+.4f} "
                  f"C1 {ic.loc[C1[0]:C1[1]].mean():+.4f} "
                  f"(NW t {nw_t_stat(ic.to_numpy()):+.2f})")
        out[name] = {"ics": ics, "bt": pd.concat(bts, axis=1).mean(axis=1)}

    print("\n--- PRIMARY CONTRAST under next_day policy ---")
    for s_i, s in enumerate(SEEDS):
        d = out["C"]["ics"][s_i] - out["A"]["ics"][s_i].reindex(out["C"]["ics"][s_i].index)
        print(f"seed{s}: C1 delta {d.loc[C1[0]:C1[1]].mean():+.4f}  "
              f"full {d.mean():+.4f}")
    dc = out["C"]["bt"] - out["A"]["bt"].reindex(out["C"]["bt"].index)
    j = dc.dropna()
    t = j.mean() / (j.std(ddof=1) / np.sqrt(len(j)) + 1e-12)
    print(f"backtest delta C-A (next_day): {j.mean():+.4f}/wk  "
          f"t({len(j)-1})={t:+.2f} over {len(j)} weeks")


if __name__ == "__main__":
    main()
