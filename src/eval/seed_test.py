import numpy as np
import pandas as pd
from scipy import stats as st

from src.config import RESULTS_DIR
from src.eval.analysis import CRISIS, load_runs


def main():
    runs = load_runs()
    for hi_m, lo_m in (("C", "A"), ("C", "B"), ("B", "A")):
        hi = {r["seed"]: r for r in runs if r["model"] == hi_m}
        lo = {r["seed"]: r for r in runs if r["model"] == lo_m}
        for wname, w in {**CRISIS, "full_test": None}.items():
            ds = []
            for s in sorted(set(hi) & set(lo)):
                d = hi[s]["ic"] - lo[s]["ic"].reindex(hi[s]["ic"].index)
                ds.append(float((d if w is None else d.loc[w[0]:w[1]]).mean()))
            arr = np.asarray(ds)
            t, p = st.ttest_1samp(arr, 0.0)
            n_pos = int((arr > 0).sum())
            print(f"{hi_m}-{lo_m:<3} {wname:<12} mean={arr.mean():+.4f} "
                  f"sd={arr.std(ddof=1):.4f} t({len(arr)-1})={t:+.2f} "
                  f"p={p:.3f} seeds_positive={n_pos}/{len(arr)}  deltas="
                  f"{np.round(arr, 4).tolist()}")


if __name__ == "__main__":
    main()
