import numpy as np
import pandas as pd

from src.config import (CORR_THRESHOLD, GRAPHS_DIR, LOOKBACK, MAX_DEGREE,
                        PROCESSED_DIR, RESULTS_DIR, SAMPLE_END, SAMPLE_START,
                        SENT_CORR_THRESHOLD, SENT_MIN_OVERLAP_DAYS,
                        TOP_K_NEIGHBORS, ensure_dirs)
from src.data.build_features import load_price_wide

EPS = 1e-12
KEY_DATES = ["2019-06-28", "2020-03-18", "2022-09-28", "2023-03-13"]


def corr_matrix(W: np.ndarray) -> np.ndarray:
    if not np.isfinite(W).all():
        return np.full((W.shape[1], W.shape[1]), np.nan)
    std = W.std(axis=0, ddof=1)
    Wc = W - W.mean(axis=0, keepdims=True)
    cov = (Wc.T @ Wc) / max(W.shape[0] - 1, 1)
    denom = np.outer(std, std)
    C = np.where(denom > EPS ** 2, cov / np.maximum(denom, EPS ** 2), np.nan)
    np.fill_diagonal(C, 1.0)
    return C


def mask_from_threshold(C: np.ndarray, threshold: float, signed) -> np.ndarray:
    if signed == "neg":
        M = C < -threshold
    elif signed == "pos" or signed is True:
        M = C > threshold
    else:
        M = np.abs(C) > threshold
    np.fill_diagonal(M, False)
    return M


def _ranks_topk(C: np.ndarray, k: int) -> np.ndarray:
    A = np.abs(np.nan_to_num(C, nan=0.0))
    np.fill_diagonal(A, -np.inf)
    order = np.argsort(-A, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(A.shape[0])[:, None]
    rank[rows, order] = np.arange(A.shape[1])[None, :]
    return rank


def cap_node_degrees(A: np.ndarray, mask: np.ndarray, max_degree: int) -> np.ndarray:
    mask = mask.copy()
    for i in range(A.shape[0]):
        while mask[i].sum() > max_degree:
            nbrs = np.where(mask[i])[0]
            weakest = nbrs[np.argmin(A[i, nbrs])]
            mask[i, weakest] = False
            mask[weakest, i] = False
    return mask


def sparsify(C: np.ndarray, M: np.ndarray, k: int, max_degree: int) -> np.ndarray:
    topk = _ranks_topk(C, k) < k
    mutual = topk & topk.T & M
    A = np.abs(np.nan_to_num(C, nan=0.0))
    np.fill_diagonal(A, 0.0)
    return cap_node_degrees(A, mutual, max_degree)


def edges_from_mask(C: np.ndarray, M: np.ndarray,
                    names: list) -> list:
    iu = np.triu_indices(len(names), k=1)
    sel = M[iu]
    return [(names[i], names[j], float(C[i, j]))
            for i, j in zip(iu[0][sel], iu[1][sel])]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def build_from_wides(ret_w, sent_w, cnt_w, *, lookback, corr_threshold,
                     sent_threshold, min_overlap, top_k, max_degree,
                     start, end):
    names = list(ret_w.columns)
    dates = ret_w.index
    R, S = ret_w.to_numpy(), sent_w.to_numpy()
    B = (cnt_w.to_numpy() > 0)

    edge_rows, stat_rows = [], []
    prev_sets = {"pos": None, "neg": None, "sent": None}
    skipped = 0

    for t in range(lookback - 1, len(dates)):
        d = dates[t]
        if d < pd.Timestamp(start) or d > pd.Timestamp(end):
            continue
        sl = slice(t - lookback + 1, t + 1)
        C = corr_matrix(R[sl])
        if np.isnan(C).all():
            skipped += 1
            continue
        Cs = corr_matrix(S[sl])
        Bw = B[sl].astype(np.int64)
        O = Bw.T @ Bw

        day_edges, day_sets, day_stats = {}, {}, {}
        for rel, signed in (("pos", "pos"), ("neg", "neg")):
            M = mask_from_threshold(C, corr_threshold, signed)
            sparsified = False
            if M.sum() and M.sum(axis=1).max() > max_degree:
                M = sparsify(C, M, top_k, max_degree)
                sparsified = True
            day_edges[rel] = edges_from_mask(C, M, names)
            day_sets[rel] = {(a, b) for a, b, _ in day_edges[rel]}
            day_stats[rel] = (len(day_edges[rel]), sparsified)
        M_s = mask_from_threshold(Cs, sent_threshold, signed=False) & (O >= min_overlap)
        sparsified_s = False
        if M_s.sum() and M_s.sum(axis=1).max() > max_degree:
            M_s = sparsify(Cs, M_s, top_k, max_degree)
            sparsified_s = True
        day_edges["sent"] = edges_from_mask(Cs, M_s, names)
        day_sets["sent"] = {(a, b) for a, b, _ in day_edges["sent"]}
        day_stats["sent"] = (len(day_edges["sent"]), sparsified_s)

        n = len(names)
        jac_sp = jaccard(day_sets["sent"], day_sets["pos"] | day_sets["neg"])
        stat_rows.append({
            "date": d,
            **{f"{rel}_edges": day_stats[rel][0] for rel in day_stats},
            **{f"{rel}_sparsified": day_stats[rel][1] for rel in day_stats},
            **{f"{rel}_max_deg": max(
                [sum(1 for a, b in day_sets[rel] if x in (a, b)) for x in names]
                + [0]) for rel in day_stats},
            "sent_vs_price_jaccard": jac_sp,
            **{f"jac_{rel}_prev": (
                np.nan if prev_sets[rel] is None
                else jaccard(day_sets[rel], prev_sets[rel]))
               for rel in day_stats},
        })
        for rel in day_stats:
            for a, b, w in day_edges[rel]:
                edge_rows.append({"date": d, "relation": rel,
                                  "src": a, "dst": b, "weight": w})
            prev_sets[rel] = day_sets[rel]

    edges = pd.DataFrame(edge_rows, columns=["date", "relation", "src", "dst", "weight"])
    stats = pd.DataFrame(stat_rows).sort_values("date").reset_index(drop=True)
    if skipped:
        print(f"[graphs] skipped {skipped} days (non-finite price window)")
    return edges, stats


def main() -> None:
    ensure_dirs()
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    sentiment = pd.read_parquet(PROCESSED_DIR / "sentiment_daily.parquet")

    wides = load_price_wide(prices)
    ret_w = wides["close"].pct_change()
    cal, tickers = wides["close"].index, wides["close"].columns
    sent_w = (sentiment.pivot(index="date", columns="ticker", values="sent_ema")
                       .reindex(index=cal, columns=tickers).fillna(0.0))
    cnt_w = (sentiment.pivot(index="date", columns="ticker", values="count")
                      .reindex(index=cal, columns=tickers).fillna(0.0))

    edges, stats = build_from_wides(
        ret_w, sent_w, cnt_w,
        lookback=LOOKBACK, corr_threshold=CORR_THRESHOLD,
        sent_threshold=SENT_CORR_THRESHOLD, min_overlap=SENT_MIN_OVERLAP_DAYS,
        top_k=TOP_K_NEIGHBORS, max_degree=MAX_DEGREE,
        start=SAMPLE_START, end=SAMPLE_END)

    print(f"\n--- graph gate diagnostics ({stats['date'].nunique()} days) ---")
    print(f"edges total: pos={len(edges[edges.relation == 'pos'])} "
          f"neg={len(edges[edges.relation == 'neg'])} "
          f"sent={len(edges[edges.relation == 'sent'])}")

    stats_i = stats.set_index("date")
    print("\nmean degree at key dates (pos / neg / sent):")
    for kd in KEY_DATES:
        d0 = stats_i.index.get_indexer([pd.Timestamp(kd)], method="nearest")[0]
        row = stats_i.iloc[d0]
        print(f"  {row.name.date()}: "
              f"{row['pos_edges'] * 2 / 29:.1f} / "
              f"{row['neg_edges'] * 2 / 29:.1f} / "
              f"{row['sent_edges'] * 2 / 29:.1f}")

    y19 = stats_i.loc["2019-01-01":"2019-12-31", "pos_edges"] * 2 / 29
    mar20 = stats_i.loc["2020-03-01":"2020-03-31", "pos_edges"] * 2 / 29
    ratio = mar20.max() / max(y19.median(), EPS)
    print(f"\nGATE 1 - COVID degree spike: max Mar-2020 pos degree "
          f"{mar20.max():.1f} vs 2019 median {y19.median():.1f} -> ratio {ratio:.2f} "
          f"({'PASS' if ratio > 1.5 else 'WARN: expected stress spike'})")

    mjp = stats_i["sent_vs_price_jaccard"].mean()
    print(f"GATE 2 - sent vs price-graph clone check: mean jaccard {mjp:.3f} "
          f"({'PASS: distinct graphs' if mjp < 0.8 else 'FAIL: near-clone — STOP & rethink'})")

    cand = (edges.relation == "sent")
    print(f"GATE 3 - sent edges all satisfy overlap >= {SENT_MIN_OVERLAP_DAYS}: "
          f"enforced by construction; sent graph non-empty: {cand.sum() > 0}")

    sp = (stats_i[[f"{r}_sparsified" for r in ("pos", "neg", "sent")]]
          .groupby(stats_i.index.year).sum())
    print(f"\nsparsification trigger days by year:\n{sp.to_string()}")
    jac = stats_i[["jac_pos_prev", "jac_neg_prev", "jac_sent_prev"]].mean()
    print(f"\nconsecutive-day edge-set jaccard (stability):\n{jac.to_string()}")

    edges.to_parquet(GRAPHS_DIR / "graphs.parquet", index=False)
    stats.to_csv(RESULTS_DIR / "graph_stats.csv", index=False)
    print(f"\nsaved -> {GRAPHS_DIR / 'graphs.parquet'} "
          f"({len(edges)} edges), {RESULTS_DIR / 'graph_stats.csv'}")


if __name__ == "__main__":
    main()
