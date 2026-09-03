import json

import numpy as np
import pandas as pd

from src.config import HORIZON, PROCESSED_DIR, RESULTS_DIR
from src.model.train import nw_t_stat, spearman

CRISIS = {"C1_bear2022": ("2022-01-01", "2022-10-31"),
          "C2_svb2023": ("2023-03-01", "2023-03-31")}
PAIRS = [("C", "A"), ("C", "B"), ("B", "A")]
EPS_STD = 1e-6


def load_runs():
    runs = []
    for p in sorted(RESULTS_DIR.glob("metrics_*.json")):
        tag = p.stem.replace("metrics_", "")
        ic_path = RESULTS_DIR / f"ic_series_{tag}.csv"
        if not ic_path.exists():
            print(f"[warn] {tag}: metrics without ic_series — rerun via "
                  f"run_matrix to backfill; skipped here")
            continue
        m = json.loads(p.read_text())
        ic = pd.read_csv(ic_path, parse_dates=["date"]).set_index("date")["ic"]
        runs.append({"model": m["model"], "seed": m["seed"], "tag": tag, "ic": ic, "val_ic": m["val_ic"]})
    return runs


def _win(s, w):
    return s if w is None else s.loc[w[0]:w[1]]


def table_summary(runs):
    rows = []
    for model in sorted({r["model"] for r in runs}):
        rs = [r for r in runs if r["model"] == model]
        ts = [nw_t_stat(r["ic"].to_numpy()) for r in rs]
        rows.append({"model": model, "n_seeds": len(rs),
                     "val_ic_mean": np.mean([r["val_ic"] for r in rs]),
                     "test_ic_mean": np.mean([r["ic"].mean() for r in rs]),
                     "test_ic_sd": (np.std([r["ic"].mean() for r in rs], ddof=1)
                                    if len(rs) > 1 else np.nan),
                     "nw_t_min": min(ts), "nw_t_max": max(ts)})
    return pd.DataFrame(rows).round(4)


def table_crisis(runs):
    rows = []
    for model in sorted({r["model"] for r in runs}):
        rs = [r for r in runs if r["model"] == model]
        for wname, w in {**CRISIS, "NC_rest": None}.items():
            per = []
            for r in rs:
                if w is None:
                    cr = pd.DatetimeIndex([])
                    for lo, hi in CRISIS.values():
                        cr = cr.union(r["ic"].loc[lo:hi].index)
                    per.append(float(r["ic"].drop(cr).mean()))
                else:
                    per.append(float(_win(r["ic"], w).mean()))
            rows.append({"model": model, "window": wname,
                         "ic_mean": np.mean(per),
                         "ic_sd": np.std(per, ddof=1) if len(per) > 1 else np.nan,
                         "n_seeds": len(per)})
    return pd.DataFrame(rows).round(4)


def table_paired(runs):
    rows = []
    for hi_m, lo_m in PAIRS:
        hi = {r["seed"]: r for r in runs if r["model"] == hi_m}
        lo = {r["seed"]: r for r in runs if r["model"] == lo_m}
        common = sorted(set(hi) & set(lo))
        if not common:
            continue
        for wname, w in {**CRISIS, "full_test": None}.items():
            ds, ts = [], []
            for s in common:
                a = hi[s]["ic"]
                b = lo[s]["ic"].reindex(a.index)
                assert b.notna().all(), f"date mismatch {hi_m}/{lo_m} seed{s}"
                d = _win(a - b, w)
                ds.append(float(d.mean()))
                ts.append(nw_t_stat(d.to_numpy()))
            rows.append({"pair": f"{hi_m}-{lo_m}", "window": wname,
                         "d_ic_mean": np.mean(ds),
                         "d_ic_sd": np.std(ds, ddof=1) if len(ds) > 1 else np.nan,
                         "nw_t_mean": np.mean(ts),
                         "nw_t_range": f"[{min(ts):+.2f},{max(ts):+.2f}]",
                         "n_seeds": len(common)})
    return pd.DataFrame(rows).round(4)


def table_baselines(test_dates):
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    feats = pd.read_parquet(PROCESSED_DIR / "features_panel.parquet")
    close = (prices.pivot(index="date", columns="ticker", values="close")
                   .sort_index())
    y = close.shift(-HORIZON) / close - 1
    mom20 = close.pct_change(20)
    mom5 = (feats.pivot(index="date", columns="ticker", values="mom5")
                  .reindex(index=close.index, columns=close.columns))
    rows = []
    for name, sig in (("mom5_feature", mom5), ("mom20_raw", mom20)):
        ics = {w: [] for w in list(CRISIS) + ["full_test", "NC_rest"]}
        for d in test_dates:
            if d not in sig.index:
                continue
            s, yy = sig.loc[d], y.loc[d]
            m = yy.notna() & s.notna()
            if m.sum() < 3:
                continue
            ic = spearman(s[m].to_numpy(), yy[m].to_numpy())
            ics["full_test"].append((d, ic))
            for wn, w in CRISIS.items():
                if pd.Timestamp(w[0]) <= d <= pd.Timestamp(w[1]):
                    ics[wn].append((d, ic))
        cr = pd.DatetimeIndex([])
        for lo, hi in CRISIS.values():
            cr = cr.union(sig.loc[lo:hi].index)
        for w, lst in ics.items():
            if w == "NC_rest":
                vals = [ic for d, ic in ics["full_test"] if d not in cr]
            else:
                vals = [ic for _, ic in lst]
            if vals:
                rows.append({"baseline": name, "window": w,
                             "ic_mean": float(np.mean(vals)),
                             "nw_t": nw_t_stat(np.array(vals)),
                             "n_days": len(vals)})
    return pd.DataFrame(rows).round(4)


def collapse_check(runs):
    rows = []
    for r in runs:
        pr = pd.read_parquet(RESULTS_DIR / f"preds_{r['tag']}.parquet")
        sd = pr.groupby("date")["score"].std()
        n_flat = int((sd < EPS_STD).sum())
        rec = {}
        for d, g in pr.groupby("date"):
            if len(g) >= 3 and g["score"].std() > EPS_STD:
                rec[d] = spearman(g["score"].to_numpy(), g["y"].to_numpy())
            else:
                rec[d] = 0.0
        rec = pd.Series(rec).sort_index()
        stored = r["ic"].reindex(rec.index)
        diff = float((rec - stored).abs().max())
        rows.append({"tag": r["tag"], "n_days": len(rec), "n_flat_days": n_flat,
                     "ic_recompute_max_diff": diff})
        if diff > 1e-4:
            print(f"[warn] {r['tag']}: preds-vs-ic_series mismatch {diff:.2e}")
    return pd.DataFrame(rows)


def table_per_year(runs, model="C"):
    rs = [r for r in runs if r["model"] == model]
    rows = []
    for year in sorted({d.year for r in rs for d in r["ic"].index}):
        per = [float(r["ic"][r["ic"].index.year == year].mean()) for r in rs]
        rows.append({"model": model, "year": year, "ic_mean": np.mean(per),
                     "ic_sd": np.std(per, ddof=1) if len(per) > 1 else np.nan})
    return pd.DataFrame(rows).round(4)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("no complete runs found — run src.model.run_matrix first")
    test_dates = sorted(set().union(*[set(r["ic"].index) for r in runs]))

    t1 = table_summary(runs)
    t2 = table_crisis(runs)
    t3 = table_paired(runs)
    t4 = table_baselines(pd.DatetimeIndex(test_dates))
    t5 = collapse_check(runs)
    t6 = table_per_year(runs)

    print("\n=== (1) summary (mean over seeds) ===\n", t1.to_string(index=False))
    print("\n=== (2) crisis windows (mean over seeds) ===\n", t2.to_string(index=False))
    print("\n=== (3) PAIRED IC deltas — THE THESIS TEST ===\n", t3.to_string(index=False))
    print("\n=== (4) baselines, same test days ===\n", t4.to_string(index=False))
    print("\n=== (5) collapse check ===\n", t5.to_string(index=False))
    print("\n=== (6) model C per-year ===\n", t6.to_string(index=False))

    for name, df in (("summary", t1), ("crisis_windows", t2), ("paired_deltas", t3),
                     ("baselines", t4), ("collapse_check", t5), ("per_year_C", t6)):
        df.to_csv(RESULTS_DIR / f"analysis_{name}.csv", index=False)
    print("\nsaved analysis_*.csv ->", RESULTS_DIR)


if __name__ == "__main__":
    main()
