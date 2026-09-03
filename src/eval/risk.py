import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (LOOKBACK, PROCESSED_DIR, RESULTS_DIR, SAMPLE_END,
                        SAMPLE_START, SECTORS, TICKERS, TRAIN_END, VAL_END,
                        ensure_dirs)
from src.model.dataset import build_datasets, collate_samples
from src.model.thgnn import THGNN
from src.model.train import PRESETS, spearman

EPS = 1e-8
DAMPEN = 0.1


def participation_ratio(Z: np.ndarray) -> float:
    Zc = Z - Z.mean(0, keepdims=True)
    sd = Zc.std(0, keepdims=True)
    Zs = Zc / np.maximum(sd, EPS)
    k = len(Zs)
    if k < 3:
        return np.nan
    C = Zs.T @ Zs / (k - 1)
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    return float(lam.sum() ** 2 / max((lam ** 2).sum(), EPS))


def concentration_series(Z, dates, tickers, preds=None):
    idx = {t: i for i, t in enumerate(tickers)}
    rows = []
    top5 = None
    if preds is not None:
        preds = preds.copy()
        preds["date"] = pd.to_datetime(preds["date"])
        top5 = preds.sort_values(["date", "score"], ascending=[True, False]) \
                    .groupby("date").head(5)
    for b, d in enumerate(dates):
        row = {"date": pd.Timestamp(d), "n_eff_universe":
               participation_ratio(Z[b])}
        if top5 is not None:
            day = top5[top5["date"] == pd.Timestamp(d)]
            if len(day) == 5:
                sel = [idx[t] for t in day["ticker"]]
                row["n_eff_top5"] = participation_ratio(Z[b][sel])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def cross_sector_links(Z, dates, tickers, top_k=5):
    idx = {t: i for i, t in enumerate(tickers)}
    sec = [SECTORS.get(t, "?") for t in tickers]
    acc = {}
    n_days = 0
    for b in range(len(dates)):
        Zn = Z[b] / np.maximum(np.linalg.norm(Z[b], axis=1, keepdims=True), EPS)
        S = Zn @ Zn.T
        np.fill_diagonal(S, -np.inf)
        n_days += 1
        for i in range(len(tickers)):
            for j in np.argsort(-S[i])[:top_k]:
                a, b_ = tickers[i], tickers[j]
                key = tuple(sorted((a, b_)))
                if sec[i] == sec[j]:
                    continue
                f, c = acc.get(key, (0, 0.0))
                acc[key] = (f + 1, c + float(S[i, j]))
    rows = [{"src": a, "dst": b_, "freq": f / n_days, "mean_cos": c / max(f, 1)}
            for (a, b_), (f, c) in acc.items()]
    df = pd.DataFrame(rows).sort_values(["freq", "mean_cos"], ascending=False)
    return df.reset_index(drop=True)


def _sector_mask(tickers, sector):
    return np.array([SECTORS.get(t) == sector for t in tickers])


@torch.no_grad()
def _embed(model, x, edges, relations, device, mask=None, return_beta=False):
    xt = torch.from_numpy(x[None].astype(np.float32)).to(device)
    if mask is not None:
        xt = xt.clone()
        xt[:, mask] = 0.0
    ed = {}
    for r in relations:
        ei, ew = edges[r]
        if mask is not None and len(ew):
            keep = ~(np.isin(ei[0], mask) | np.isin(ei[1], mask))
            ew2 = ew[keep] * DAMPEN
            ei2 = ei[:, keep]
        else:
            ei2, ew2 = ei, ew
        ed[r] = (torch.from_numpy(ei2).to(device),
                 torch.from_numpy(ew2.astype(np.float32)).to(device))
    out = model(xt, ed, return_beta=True) if return_beta else model(xt, ed)
    return tuple(o.cpu().numpy()[0] for o in out)


@torch.no_grad()
def shock_exposure(model, samples_by_date, relations, device, tickers,
                   dates, sector, shock_window):
    mask = np.where(_sector_mask(tickers, sector))[0]
    assert len(mask), f"no tickers in sector {sector}"
    disp = {}
    n_used = 0
    for d in dates:
        if not (pd.Timestamp(shock_window[0]) <= pd.Timestamp(d)
                <= pd.Timestamp(shock_window[1])):
            continue
        s = samples_by_date.get(pd.Timestamp(d))
        if s is None:
            continue
        edges = {r: (s[f"ei_{r}"], s[f"ew_{r}"]) for r in relations}
        _, Z0 = _embed(model, s["x"], edges, relations, device)
        _, Z1 = _embed(model, s["x"], edges, relations, device, mask=mask)
        keep = np.setdiff1d(np.arange(len(tickers)), mask)
        cos = (np.sum(Z0[keep] * Z1[keep], 1)
               / np.maximum(np.linalg.norm(Z0[keep], axis=1)
                            * np.linalg.norm(Z1[keep], axis=1), EPS))
        for t, c in zip(np.array(tickers)[keep], 1 - cos):
            disp.setdefault(t, []).append(float(c))
        n_used += 1
    assert n_used, "no shock-window days found in test embeddings"
    out = pd.Series({t: np.mean(v) for t, v in disp.items()}).sort_values(
        ascending=False)
    print(f"[shock:{sector}] days used={n_used}, masked={len(mask)}")
    return out


def validate_energy_shock(model, samples_by_date, relations, device, tickers):
    disp = shock_exposure(model, samples_by_date, relations, device, tickers,
                          sorted(samples_by_date), "energy",
                          ("2022-01-03", "2022-02-28"))
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    close = (prices.pivot(index="date", columns="ticker", values="close")
                   .sort_index())
    base = close.loc[: "2022-02-23"].iloc[-1]
    win = close.loc["2022-02-24": "2022-06-30"]
    dd = (win / base - 1).min()
    common = disp.index.intersection(dd.dropna().index)
    rho, p = (spearman(disp[common].to_numpy(), dd[common].to_numpy()), None)
    rng = np.random.default_rng(0)
    null = [spearman(disp[common].to_numpy(),
                     rng.permutation(dd[common].to_numpy()))
            for _ in range(20000)]
    p = float((np.abs(np.array(null)) >= abs(rho)).mean())
    print(f"VALIDATION energy-shock: Spearman(disp, realized_dd) = {rho:+.3f} "
          f"perm-p={p:.3f}  (want rho<0, p<0.05)  n={len(common)}")
    out = pd.DataFrame({"displacement": disp, "realized_dd": dd})
    out["sector"] = [SECTORS.get(t, "?") for t in out.index]
    out.to_csv(RESULTS_DIR / "shock_validation_energy.csv")
    print(out.round(3).to_string())
    return rho, p


def fetch_vix() -> pd.Series:
    import yfinance as yf
    cache = PROCESSED_DIR / "vix.parquet"
    if not cache.exists():
        v = yf.download("^VIX", start="2021-12-01", end=SAMPLE_END,
                        auto_adjust=True, progress=False)
        if isinstance(v.columns, pd.MultiIndex):
            lvl0 = set(v.columns.get_level_values(0))
            level = 0 if "Close" in lvl0 or "close" in lvl0 else 1
            v.columns = v.columns.get_level_values(level)
        v.columns = [c.capitalize() for c in v.columns]
        v[["Close"]].to_parquet(cache)
    v = pd.read_parquet(cache)
    close_col = "Close" if "Close" in v.columns else v.columns[0]
    s = v[close_col]
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s.rename("vix")


def beta_sent_series(model, samples_by_date, relations, device, tickers):
    if "sent" not in relations:
        print("[beta] model has no sent relation - skipping")
        return None
    si = relations.index("sent") + 1
    rows = []
    for d in sorted(samples_by_date):
        s = samples_by_date[d]
        edges = {r: (s[f"ei_{r}"], s[f"ew_{r}"]) for r in relations}
        _, _, beta = _embed(model, s["x"], edges, relations, device,
                            return_beta=True)
        rows.append({"date": d, "beta_sent": float(beta[:, si].mean())})
    return pd.DataFrame(rows).set_index("date")["beta_sent"]


def beta_vs_vix(beta: pd.Series, tag: str):
    vix = fetch_vix().reindex(beta.index).ffill()
    df = pd.DataFrame({"beta_sent": beta, "vix": vix}).dropna()
    print(f"beta_sent: mean={df['beta_sent'].mean():.3f} "
          f"p90={df['beta_sent'].quantile(0.9):.3f}")
    for lag in (0, 5, 10):
        v = df["vix"].shift(-lag)
        m = v.notna()
        print(f"  corr(beta_sent[t], VIX[t+{lag}]) = "
              f"{np.corrcoef(df['beta_sent'][m], v[m])[0, 1]:+.3f}")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, df["beta_sent"], color="tab:blue", label="beta_sent")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["vix"], color="tab:red", alpha=0.6, label="VIX")
    ax1.set_ylabel("beta_sent"); ax2.set_ylabel("VIX")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"fig_beta_sent_vix_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"saved fig_beta_sent_vix_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshot", default=None,
                    help="optional YYYY-MM-DD risk snapshot")
    a = ap.parse_args()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_set, relations = PRESETS[a.model]
    tag = f"{a.model}_seed{a.seed}"

    npz = np.load(RESULTS_DIR / f"embeddings_{tag}.npz", allow_pickle=False)
    Z, dates, tickers = npz["Z"], list(pd.to_datetime(npz["dates"])), list(npz["tickers"])
    assert tickers == list(TICKERS) and np.all(np.diff(pd.to_datetime(dates)) > pd.Timedelta(0))
    preds = pd.read_parquet(RESULTS_DIR / f"preds_{tag}.parquet")

    ds, channels = build_datasets(feature_set, relations)
    samples_by_date = {s["date"]: s for s in ds["test"].samples}
    model = THGNN(d_feat=len(channels), T=LOOKBACK, relations=relations).to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / f"ckpt_{tag}.pt",
                                     map_location=device))
    model.eval()

    conc = concentration_series(Z, dates, tickers, preds)
    print("\n--- concentration (N_eff) ---")
    print(f"universe: mean={conc['n_eff_universe'].mean():.2f} "
          f"min={conc['n_eff_universe'].min():.2f} "
          f"max={conc['n_eff_universe'].max():.2f}")
    if "n_eff_top5" in conc:
        print(f"top-5 portfolio: mean={conc['n_eff_top5'].mean():.2f} "
              f"min={conc['n_eff_top5'].min():.2f}")
    h1 = conc.loc["2022-06-01": "2022-10-31", "n_eff_universe"].mean()
    h2 = conc.loc["2023-05-01": "2023-09-30", "n_eff_universe"].mean()
    print(f"N_eff 2022-bear-half={h1:.2f} vs 2023-calm-half={h2:.2f} "
          f"({'lower in stress: consistent' if h1 < h2 else 'no stress drop'})")
    conc.to_csv(RESULTS_DIR / f"concentration_{tag}.csv")

    links = cross_sector_links(Z, dates, tickers)
    links.head(15).to_csv(RESULTS_DIR / f"hidden_links_{tag}.csv", index=False)
    print(f"\ntop persistent cross-sector links:\n{links.head(8).to_string(index=False)}")

    print("\n--- shock simulator ---")
    validate_energy_shock(model, samples_by_date, relations, device, tickers)

    beta = beta_sent_series(model, samples_by_date, relations, device, tickers)
    if beta is not None:
        beta_vs_vix(beta, tag)
        beta.to_frame().to_csv(RESULTS_DIR / f"beta_sent_{tag}.csv")

    if a.snapshot:
        d = pd.Timestamp(a.snapshot)
        print(f"\n=== RISK SNAPSHOT {d.date()} ({tag}) ===")
        row = conc.loc[:d].iloc[-1]
        print(f"N_eff universe={row['n_eff_universe']:.2f} "
              f"(percentile {(conc['n_eff_universe'] < row['n_eff_universe']).mean():.0%})")
        lk = links.head(5)
        print("hidden links:", "; ".join(
            f"{r.src}-{r.dst} ({r.freq:.0%}d)" for r in lk.itertuples()))
        disp = shock_exposure(model, samples_by_date, relations, device,
                              tickers, dates, "energy", (str(d.date()),) * 2)
        print("energy-shock most exposed:", ", ".join(
            f"{t}({v:.2f})" for t, v in disp.head(4).items()))
        if beta is not None:
            b = beta.loc[:d].iloc[-1]
            print(f"beta_sent={b:.3f} "
                  f"(pct {(beta < b).mean():.0%} since 2022)")


if __name__ == "__main__":
    main()
