import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import PRICE_END, PRICE_START, PROCESSED_DIR, TICKERS, ensure_dirs

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        level = 0 if "Close" in lvl0 else 1
        df = df.copy()
        df.columns = df.columns.get_level_values(level)
    missing = [f for f in FIELDS if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"{ticker}: yfinance returned unexpected columns; "
            f"missing={missing}, got={list(df.columns)}"
        )
    return df[FIELDS].copy()


def download_one(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False, threads=False)
            if df.empty:
                raise RuntimeError("empty download")
            df = _flatten_columns(df, ticker)
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            df.index = idx.normalize()
            df.index.name = "date"
            return df
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed for {ticker} after {retries} tries: {last_err}")


def validate_prices(long_df: pd.DataFrame):
    issues = []
    close = long_df["close"].astype(float)
    n_bad = int(((close <= 0) | close.isna()).sum())
    if n_bad:
        issues.append(("FATAL", "NONPOSITIVE_OR_NAN_CLOSE", n_bad))

    n_hl = int((long_df["high"] < long_df["low"]).sum())
    if n_hl:
        issues.append(("WARN", "HIGH_LT_LOW", n_hl))

    ret = (long_df.sort_values(["ticker", "date"])
                  .groupby("ticker")["close"].pct_change())
    n_extreme = int((ret.abs() > 0.5).sum())
    if n_extreme:
        issues.append(("WARN", "EXTREME_RETURN_GT_50pct", n_extreme))

    union = pd.DatetimeIndex(np.sort(long_df["date"].unique()))
    gaps = {}
    for t, g in long_df.groupby("ticker"):
        gaps[t] = len(union) - pd.DatetimeIndex(g["date"]).nunique()
    worst = sorted(gaps.items(), key=lambda kv: -kv[1])[:5]
    return issues, worst


def main() -> None:
    ensure_dirs()
    out = PROCESSED_DIR / "prices.parquet"
    if out.exists() and "--force" not in sys.argv:
        print(f"[skip] {out} exists (use --force to re-download)")
        return

    frames = []
    for i, t in enumerate(TICKERS, 1):
        df = download_one(t, PRICE_START, PRICE_END)
        df["ticker"] = t
        frames.append(df.reset_index())
        print(f"[{i:2d}/{len(TICKERS)}] {t:<6} rows={len(df):5d}  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
        time.sleep(1.0)

    long_df = pd.concat(frames, ignore_index=True)
    long_df.columns = [c.lower() for c in long_df.columns]

    print("\n--- price validation ---")
    issues, worst = validate_prices(long_df)
    for lvl, code, n in issues:
        print(f"  [{lvl}] {code}: {n}")
    print(f"  missing days vs union calendar (top 5): {worst}")
    if any(lvl == "FATAL" for lvl, _, _ in issues):
        raise SystemExit("FATAL price issues — fix before continuing")

    long_df.to_parquet(out, index=False)
    print(f"saved -> {out}  "
          f"({long_df['ticker'].nunique()} tickers, {long_df['date'].nunique()} dates)")


if __name__ == "__main__":
    main()
