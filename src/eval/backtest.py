import argparse

import numpy as np
import pandas as pd

from src.config import HORIZON, PROCESSED_DIR, RESULTS_DIR


def weekly_backtest(preds: pd.DataFrame, close: pd.DataFrame, top_k: int = 5):
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    reb = sorted(preds["date"].unique())[::HORIZON]
    fwd = close.shift(-HORIZON) / close - 1
    rows, prev = [], set()
    for d in reb:
        day = preds[preds["date"] == d]
        if len(day) < 29 or d not in fwd.index:
            continue
        hold = day.nlargest(top_k, "score")["ticker"].tolist()
        r = fwd.loc[d, hold]
        if r.isna().any():
            continue
        rows.append({"date": d,
                     "port_ret": float(r.mean()),
                     "bench_ret": float(fwd.loc[d].dropna().mean()),
                     "turnover": 1 - len(set(hold) & prev) / top_k})
        prev = set(hold)
    return pd.DataFrame(rows).set_index("date")


def perf(ret: pd.Series):
    ppy = 252 / HORIZON
    eq = (1 + ret).cumprod()
    ann = eq.iloc[-1] ** (ppy / len(ret)) - 1
    sharpe = ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(ppy)
    maxdd = float((eq / eq.cummax() - 1).min())
    return ann, sharpe, maxdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["C", "A"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    a = ap.parse_args()
    close = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    close = close.pivot(index="date", columns="ticker", values="close").sort_index()

    bt_frames, all_ret = {}, {}
    for m in a.models:
        outs = []
        for s in a.seeds:
            tag = f"{m}_seed{s}"
            p = RESULTS_DIR / f"preds_{tag}.parquet"
            if not p.exists():
                print(f"[skip] {tag} (no preds)")
                continue
            bt = weekly_backtest(pd.read_parquet(p), close)
            bt.to_csv(RESULTS_DIR / f"backtest_{tag}.csv")
            outs.append(bt)
        if not outs:
            continue
        bt = pd.concat(outs).groupby(level=0).mean()
        bt_frames[m] = bt
        all_ret[m] = bt["port_ret"]
        ann, sh, dd = perf(bt["port_ret"])
        bann, bsh, bdd = perf(bt["bench_ret"])
        print(f"\n[{m}] top-5: ARR={ann:+.2%} Sharpe={sh:+.2f} MaxDD={dd:+.2%} "
              f"turnover={bt['turnover'].mean():.0%}/wk | "
              f"EW-29: ARR={bann:+.2%} Sharpe={bsh:+.2f} MaxDD={bdd:+.2%}")

    if "C" in all_ret and "A" in all_ret:
        d = all_ret["C"] - all_ret["A"]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"\npaired weekly delta C-A: mean={d.mean():+.4f} "
              f"t({len(d)-1})={t:+.2f} over {len(d)} weeks")


if __name__ == "__main__":
    main()
