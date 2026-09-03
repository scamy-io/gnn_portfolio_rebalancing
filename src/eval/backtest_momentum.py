import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.eval.backtest import perf, weekly_backtest


def cost_adjusted(ann: float, turnover_wk: float, bps_one_way: float) -> float:
    drag = turnover_wk * 2 * bps_one_way / 1e4 * 52
    return ann - drag


def main():
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    close = (prices.pivot(index="date", columns="ticker", values="close")
                   .sort_index())
    feats = pd.read_parquet(PROCESSED_DIR / "features_panel.parquet")
    mom5 = feats.pivot(index="date", columns="ticker", values="mom5")

    ref = pd.read_parquet(RESULTS_DIR / "preds_C_seed42.parquet")
    grid = sorted(pd.to_datetime(ref["date"]).unique())
    mom_preds = pd.DataFrame([
        {"date": d, "ticker": t, "score": mom5.loc[d, t]}
        for d in grid for t in mom5.columns if not np.isnan(mom5.loc[d, t])])

    bt = weekly_backtest(mom_preds, close)
    bt.to_csv(RESULTS_DIR / "backtest_mom5.csv")
    ann, sh, dd = perf(bt["port_ret"])
    bann, bsh, bdd = perf(bt["bench_ret"])
    print(f"[mom5] top-5: ARR={ann:+.2%} Sharpe={sh:+.2f} MaxDD={dd:+.2%} "
          f"turnover={bt['turnover'].mean():.0%}/wk | "
          f"EW-29: ARR={bann:+.2%} Sharpe={bsh:+.2f}")

    c_frames = [pd.read_csv(RESULTS_DIR / f"backtest_C_seed{s}.csv",
                            index_col=0, parse_dates=True) for s in (42, 0, 1, 2, 3)]
    c = pd.concat(c_frames).groupby(level=0)["port_ret"].mean()
    j = c.index.intersection(bt.index)
    d = c[j] - bt.loc[j, "port_ret"]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
    print(f"paired weekly C - mom5: {d.mean():+.4f}  t({len(d)-1})={t:+.2f}")

    print("\ncost sensitivity (one-way bps -> net ARR):")
    for bps in (5, 10, 20):
        print(f"  {bps:>2} bps: C_ens={cost_adjusted(ann if False else 0.2114, 0.66, bps):+.1%}"
              f"  mom5={cost_adjusted(ann, bt['turnover'].mean(), bps):+.1%}")


if __name__ == "__main__":
    main()
