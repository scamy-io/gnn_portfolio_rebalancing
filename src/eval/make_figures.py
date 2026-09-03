import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import SECTORS, TICKERS

ROOT = Path(__file__).resolve().parents[2]
R, P, G = ROOT / "results", ROOT / "data" / "processed", ROOT / "data" / "graphs"
OUT = ROOT / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C1 = ("2022-01-01", "2022-10-31")
C2 = ("2023-03-01", "2023-03-31")
SEEDS = [42, 0, 1, 2, 3]
ACC, FLA, GRN, RED = "#0F4C81", "#E4572E", "#2E933C", "#C1292E"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})


def shade(ax):
    for lo, hi, lab in ((*C1, "2022 bear"), (*C2, "SVB")):
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="0.88", zorder=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {name}")


def fig_network():
    import networkx as nx
    g = pd.read_parquet(G / "graphs.parquet")
    idx = {t: i for i, t in enumerate(TICKERS)}
    cmap = dict(tech=ACC, energy=RED, industrials="#7A6AA0", financials=GRN,
                consumer=FLA, healthcare="#2AA8A8", materials="#8B6F3B",
                utilities="#5B7F2A", reits="#B0567B")
    colors = [cmap.get(SECTORS.get(t, "?"), "gray") for t in TICKERS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (day, title) in zip(axes, [("2019-06-28", "calm: 2019-06-28"),
                                       ("2020-03-18", "stress: 2020-03-18")]):
        d = g[g["date"] == pd.Timestamp(day)]
        G_ = nx.Graph()
        G_.add_nodes_from(TICKERS)
        for rel, col in (("pos", GRN), ("neg", RED), ("sent", FLA)):
            for _, e in d[d.relation == rel].iterrows():
                G_.add_edge(e.src, e.dst, color=col, w=abs(e.weight))
        pos = nx.spring_layout(G_, seed=7, k=0.9)
        ec = [G_[u][v]["color"] for u, v in G_.edges()] if G_.edges else []
        ew = [2.2 * G_[u][v]["w"] for u, v in G_.edges()] if G_.edges else []
        nx.draw_networkx_edges(G_, pos, ax=ax, edge_color=ec, width=ew, alpha=0.6)
        nx.draw_networkx_nodes(G_, pos, ax=ax, node_color=colors,
                               node_size=260, edgecolors="white")
        nx.draw_networkx_labels(G_, pos, ax=ax, font_size=5.5)
        ax.set_title(f"{title}  ({G_.number_of_edges()} edges)")
        ax.axis("off")
    fig.suptitle("Daily relationship graphs: pos(green) neg(red) sent(orange), "
                 "node color = sector", y=1.02, fontsize=10)
    save(fig, "fig_network.png")


def fig_degree():
    s = pd.read_csv(R / "graph_stats.csv", parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(s.index, s["pos_edges"] * 2 / 29, color=GRN, label="pos mean degree")
    ax.plot(s.index, s["sent_edges"] * 2 / 29, color=FLA, label="sent mean degree")
    shade(ax)
    ax.set_ylabel("mean degree"); ax.legend(fontsize=8)
    ax.set_title("Graph connectivity over time (post degree-cap)")
    save(fig, "fig_degree.png")


def fig_overlap():
    s = pd.read_csv(R / "graph_stats.csv", parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.plot(s.index, s["sent_vs_price_jaccard"], color=ACC, lw=0.9)
    ax.axhline(s["sent_vs_price_jaccard"].mean(), color=FLA, ls="--",
               label=f"mean = {s['sent_vs_price_jaccard'].mean():.3f}")
    shade(ax); ax.set_ylabel("Jaccard(sent edges, price edges)")
    ax.legend(fontsize=8); ax.set_title("Sentiment graph vs price graph overlap")
    save(fig, "fig_overlap.png")


def _ic_stack(model):
    fs = sorted(R.glob(f"ic_series_{model}_seed*.csv"))
    if not fs:
        return None
    dfs = [pd.read_csv(f, parse_dates=["date"]).set_index("date")["ic"] for f in fs]
    return pd.concat(dfs, axis=1).mean(axis=1)


def fig_ic():
    c, a = _ic_stack("C"), _ic_stack("A")
    if c is None:
        return print("  [skip] fig_ic (no ic_series files)")
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(c.index, c.rolling(30).mean(), color=ACC, lw=1.8, label="C (thesis)")
    if a is not None:
        ax.plot(a.index, a.rolling(30).mean(), color="0.45", lw=1.5,
                label="A (price-only)")
    ax.axhline(0, color="k", lw=0.8)
    shade(ax); ax.set_ylabel("30-day rolling RankIC"); ax.legend(fontsize=8)
    ax.set_title("Daily ranking skill, test period 2022-2023")
    save(fig, "fig_ic.png")


def fig_yearly():
    c = _ic_stack("C")
    if c is None:
        return print("  [skip] fig_yearly")
    yr = c.groupby(c.index.year).mean()
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.bar(yr.index.astype(str), yr.values,
           color=[RED if v < 0 else GRN for v in yr.values], width=0.55)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Model C: mean RankIC by year"); ax.set_ylabel("IC")
    save(fig, "fig_yearly.png")


def fig_backtest():
    def load(model):
        fs = [R / f"backtest_{model}_seed{s}.csv" for s in SEEDS]
        fs = [f for f in fs if f.exists()]
        if not fs:
            return None
        return pd.concat([pd.read_csv(f, index_col=0, parse_dates=True)["port_ret"]
                          for f in fs], axis=1).mean(axis=1)
    c, a, m = load("C"), load("A"), load("mom5")
    ref = R / "backtest_C_seed42.csv"
    if not ref.exists():
        return print("  [skip] fig_backtest")
    bench = pd.read_csv(ref, index_col=0, parse_dates=True)["bench_ret"]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for s, lab, col in ((c, "C ensemble (top-5)", ACC), (a, "A price-only", "0.45"),
                        (m, "momentum top-5", FLA), (bench, "EW-29 benchmark", "k")):
        if s is None:
            continue
        eq = (1 + s.sort_index()).cumprod()
        ax.plot(eq.index, eq, label=lab, color=col, lw=1.8 if s is c else 1.3)
    shade(ax); ax.set_ylabel("cumulative growth of 1")
    ax.legend(fontsize=8); ax.set_title("Weekly top-5 equal-weight backtests")
    save(fig, "fig_backtest.png")


def fig_neff():
    f = R / "concentration_C_seed42.csv"
    if not f.exists():
        return print("  [skip] fig_neff")
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.plot(df.index, df["n_eff_universe"], color=ACC, lw=1.2,
            label="universe N_eff")
    if "n_eff_top5" in df:
        ax.plot(df.index, df["n_eff_top5"], color=FLA, lw=1.2,
                label="top-5 portfolio N_eff")
    shade(ax); ax.set_ylabel("effective number of bets"); ax.legend(fontsize=8)
    ax.set_title("Concentration monitor (model C)")
    save(fig, "fig_neff.png")


def fig_links():
    f = R / "hidden_links_C_seed42.csv"
    if not f.exists():
        return print("  [skip] fig_links")
    df = pd.read_csv(f).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.barh([f"{r.src}-{r.dst}" for r in df.itertuples()], df["freq"] * 100,
            color=ACC, alpha=0.85)
    ax.set_xlabel("% of test days connected")
    ax.set_title("Most persistent cross-sector links")
    ax.tick_params(axis="y", labelsize=6.5)
    save(fig, "fig_links.png")


def fig_shock():
    f = R / "shock_validation_energy.csv"
    if not f.exists():
        return print("  [skip] fig_shock")
    df = pd.read_csv(f, index_col=0).dropna(subset=["displacement", "realized_dd"])
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(df["displacement"], df["realized_dd"])
    except ImportError:
        ra = df["displacement"].rank(); rb = df["realized_dd"].rank()
        rho = np.corrcoef(ra, rb)[0, 1]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(df["displacement"], df["realized_dd"], s=28, color=ACC, alpha=0.8)
    for t, r in df.iterrows():
        ax.annotate(t, (r["displacement"], r["realized_dd"]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    z = np.polyfit(df["displacement"], df["realized_dd"], 1)
    xs = np.linspace(df["displacement"].min(), df["displacement"].max(), 10)
    ax.plot(xs, np.polyval(z, xs), color=FLA, ls="--",
            label=f"Spearman rho = {rho:+.2f}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("predicted exposure (embedding displacement)")
    ax.set_ylabel("realized Feb-Jun 2022 drawdown")
    ax.set_title("Energy-shock validation"); ax.legend(fontsize=8)
    save(fig, "fig_shock.png")


def fig_beta_vix():
    f = R / "beta_sent_C_seed42.csv"
    v = P / "vix.parquet"
    if not (f.exists() and v.exists()):
        return print("  [skip] fig_beta_vix (needs beta_sent + vix artifacts)")
    b = pd.read_csv(f, index_col=0, parse_dates=True)["beta_sent"]
    vx = pd.read_parquet(v)
    close_col = "Close" if "Close" in vx.columns else vx.columns[0]
    s = vx[close_col]
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    df = pd.DataFrame({"beta": b, "vix": s}).dropna()
    fig, ax1 = plt.subplots(figsize=(9.5, 3.4))
    ax1.plot(df.index, df["beta"], color=ACC, label="beta_sent")
    ax1.set_ylabel("beta_sent", color=ACC)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["vix"], color=RED, alpha=0.65, label="VIX (right)")
    ax2.set_ylabel("VIX", color=RED); ax2.grid(False)
    shade(ax1); ax1.set_title("How much the model leans on the sentiment relation")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    save(fig, "fig_beta_vix.png")


def fig_umap():
    f = R / "embeddings_C_seed42.npz"
    if not f.exists():
        return print("  [skip] fig_umap")
    Z = np.load(f)["Z"]
    emb = Z[-20:].mean(axis=0)
    try:
        import umap
        XY = umap.UMAP(n_neighbors=6, min_dist=0.25, random_state=0)\
                 .fit_transform(emb)
        method = "UMAP"
    except ImportError:
        Xc = emb - emb.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        XY = Xc @ Vt[:2].T
        method = "PCA (install umap-learn for UMAP)"
    cmap = dict(tech=ACC, energy=RED, industrials="#7A6AA0", financials=GRN,
                consumer=FLA, healthcare="#2AA8A8", materials="#8B6F3B",
                utilities="#5B7F2A", reits="#B0567B")
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    seen = set()
    for t, (x, y) in zip(TICKERS, XY):
        s = SECTORS.get(t, "?")
        ax.scatter(x, y, s=60, color=cmap.get(s, "gray"),
                   label=s if s not in seen else None, edgecolors="white")
        seen.add(s)
        ax.annotate(t, (x, y), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.legend(fontsize=6.5, loc="best", ncol=2)
    ax.set_title(f"Embedding space, final 20 days ({method})")
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig_umap.png")


if __name__ == "__main__":
    only = sys.argv[1:] or None
    FIGS = {"network": fig_network, "degree": fig_degree, "overlap": fig_overlap,
            "ic": fig_ic, "yearly": fig_yearly, "backtest": fig_backtest,
            "neff": fig_neff, "links": fig_links, "shock": fig_shock,
            "beta_vix": fig_beta_vix, "umap": fig_umap}
    for name, fn in FIGS.items():
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")