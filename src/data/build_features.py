import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ensure_dirs

CHANNELS = ["ret", "vol", "gap", "mom5", "sent_ema", "log_count", "sent_count"]
EPS = 1e-12
FFILL_LIMIT = 3


def load_price_wide(prices: pd.DataFrame) -> dict:
    wides, n_filled, n_remaining = {}, 0, 0
    for field in ("open", "close", "volume"):
        w = (prices.pivot(index="date", columns="ticker", values=field)
                   .sort_index())
        filled = w.ffill(limit=FFILL_LIMIT)
        n_filled += int(w.isna().sum().sum() - filled.isna().sum().sum())
        n_remaining += int(filled.isna().sum().sum())
        filled.index.name = "date"
        filled.columns.name = "ticker"
        wides[field] = filled
    if n_filled or n_remaining:
        print(f"[prices-wide] ffilled cells={n_filled}, "
              f"unfilled (NaN) cells={n_remaining}")
    return wides


def compute_price_features(wides: dict) -> dict:
    close, open_, vol = wides["close"], wides["open"], wides["volume"]
    return {
        "ret": close.pct_change(),
        "vol": np.log1p(vol),
        "gap": open_ / close.shift(1) - 1,
        "mom5": close.pct_change(5),
    }


def cross_sectional_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    v = wide.to_numpy(dtype=float)
    mask = np.isnan(v)
    safe = np.where(mask, 0.0, v)
    n = (~mask).sum(axis=1, keepdims=True)
    mean = safe.sum(axis=1, keepdims=True) / np.maximum(n, 1)
    dev2 = np.where(mask, 0.0, (v - mean) ** 2).sum(axis=1, keepdims=True)
    std = np.sqrt(dev2 / np.maximum(n, 1))
    ok = (std > EPS) & (n >= 2)
    z = np.where(ok, (v - mean) / np.where(ok, std, 1.0), 0.0)
    z[mask] = np.nan
    out = pd.DataFrame(z, index=wide.index, columns=wide.columns)
    out.index.name = "date"
    out.columns.name = "ticker"
    return out


def build_features_panel(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    wides = load_price_wide(prices)
    feats = compute_price_features(wides)

    cal, tickers = wides["close"].index, wides["close"].columns
    sent_w = (sentiment.pivot(index="date", columns="ticker", values="sent_ema")
                       .reindex(index=cal, columns=tickers).fillna(0.0))
    cnt_w = (sentiment.pivot(index="date", columns="ticker", values="count")
                      .reindex(index=cal, columns=tickers).fillna(0.0))
    sent_w.index.name = cnt_w.index.name = "date"
    sent_w.columns.name = cnt_w.columns.name = "ticker"

    feats["sent_ema"] = sent_w
    feats["log_count"] = np.log1p(cnt_w)
    feats["sent_count"] = sent_w * np.log1p(cnt_w)

    parts = []
    for name in CHANNELS:
        z = cross_sectional_zscore(feats[name])
        part = (z.reset_index()
                 .melt(id_vars="date", var_name="ticker", value_name=name)
                 .set_index(["date", "ticker"]))
        parts.append(part)
    panel = pd.concat(parts, axis=1).reset_index()
    return panel[["date", "ticker", *CHANNELS]]


def main() -> None:
    ensure_dirs()
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    sentiment = pd.read_parquet(PROCESSED_DIR / "sentiment_daily.parquet")

    panel = build_features_panel(prices, sentiment)

    print("--- feature diagnostics ---")
    print(f"rows={len(panel)} (expect {prices['date'].nunique() * 29}), "
          f"dates={panel['date'].nunique()}, tickers={panel['ticker'].nunique()}")
    for c in CHANNELS:
        col = panel[c]
        nan_n = int(col.isna().sum())
        finite = col.dropna()
        bad = int(np.isinf(finite).sum())
        print(f"  {c:<10} NaN={nan_n:5d}  inf={bad}  "
              f"z-mean={finite.mean():+.2e}  z-std={finite.std():.3f}")
        if bad:
            raise SystemExit(f"FATAL: inf values in channel {c}")

    out = PROCESSED_DIR / "features_panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"saved -> {out}")
    print("note: NaN rows are warmup (buffer period) only — dataset masks them")


if __name__ == "__main__":
    main()
