import numpy as np
import pandas as pd

from src.config import (AFTER_HOURS_HOUR, EMA_HALFLIFE, PROCESSED_DIR, RAW_DIR,
                        TICKERS, ensure_dirs)


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        return pd.to_datetime(s, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(s, errors="coerce")


def assign_trading_day(news: pd.DataFrame, trading_days: pd.DatetimeIndex,
                       midnight_policy: str = "same_day") -> pd.DataFrame:
    req = {"Date", "Stock_symbol", "sentiment_score"}
    missing = req - set(news.columns)
    if missing:
        raise ValueError(f"news frame missing columns: {missing}")

    pub = _parse_dates(news["Date"])
    if pub.dt.tz is not None:
        pub = pub.dt.tz_convert("US/Eastern").dt.tz_localize(None)

    hours, minutes = pub.dt.hour, pub.dt.minute
    after_hours = hours >= AFTER_HOURS_HOUR
    midnight = (hours == 0) & (minutes == 0)
    shift = (after_hours | (midnight & (midnight_policy == "next_day"))).astype(int)
    cand = pub.dt.normalize() + pd.to_timedelta(shift, unit="D")

    cal = pd.DatetimeIndex(trading_days)
    cand_vals = cand.to_numpy()

    pos = np.full(len(cand_vals), len(cal), dtype=np.int64)
    notna = ~pd.isna(cand_vals)
    pos[notna] = cal.searchsorted(cand_vals[notna], side="left")
    valid = pos < len(cal)

    td = np.full(len(cand_vals), np.datetime64("NaT"), dtype="datetime64[ns]")
    td[valid] = cal.values[pos[valid]]

    out = pd.DataFrame({
        "published_at": pub,
        "trading_day": td,
        "ticker": news["Stock_symbol"].astype(str).str.upper(),
        "sentiment_score": news["sentiment_score"].astype(float),
        "after_hours": after_hours.to_numpy(),
        "midnight": midnight.to_numpy(),
    }).reset_index(drop=True)
    return out


def aggregate_daily(shifted: pd.DataFrame) -> pd.DataFrame:
    ok = shifted.dropna(subset=["trading_day"])
    return (ok.groupby(["trading_day", "ticker"], as_index=False)
              .agg(sent=("sentiment_score", "mean"),
                   count=("sentiment_score", "size")))


def build_wide_panels(daily: pd.DataFrame, trading_days: pd.DatetimeIndex,
                      tickers=None, halflife: int = EMA_HALFLIFE) -> dict:
    cal = pd.DatetimeIndex(trading_days)
    if tickers is None:
        tickers = sorted(daily["ticker"].unique())

    if daily.empty:
        z = pd.DataFrame(0.0, index=cal, columns=tickers)
        return {"sent": z, "count": z.copy(),
                "log_count": z.copy(), "sent_ema": z.copy()}

    sent_w = (daily.pivot(index="trading_day", columns="ticker", values="sent")
                   .reindex(index=cal, columns=tickers).fillna(0.0))
    cnt_w = (daily.pivot(index="trading_day", columns="ticker", values="count")
                  .reindex(index=cal, columns=tickers).fillna(0.0))
    return {
        "sent": sent_w,
        "count": cnt_w,
        "log_count": np.log1p(cnt_w),
        "sent_ema": sent_w.ewm(halflife=halflife, adjust=True).mean(),
    }


def build_daily_panel(daily: pd.DataFrame, trading_days: pd.DatetimeIndex,
                      tickers=None, halflife: int = EMA_HALFLIFE) -> pd.DataFrame:
    wides = build_wide_panels(daily, trading_days, tickers, halflife)
    frames = []
    for name, w in wides.items():
        f = (w.rename_axis("date").reset_index()
              .melt(id_vars="date", var_name="ticker", value_name=name))
        frames.append(f)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["date", "ticker"], how="inner")
    return out[["date", "ticker", "sent", "count", "log_count", "sent_ema"]]


def main() -> None:
    ensure_dirs()
    news = pd.read_parquet(RAW_DIR / "fnspid_news_filtered.parquet")
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    cal = pd.DatetimeIndex(np.sort(prices["date"].unique()))

    mean, std = news["sentiment_score"].mean(), news["sentiment_score"].std()
    print(f"sentiment_score mean={mean:.3f} std={std:.3f}   (README: 0.040 / 0.488)")
    if abs(mean - 0.040) > 0.005 or abs(std - 0.488) > 0.005:
        print("  [WARN] distribution mismatch vs README — investigate ingest")

    unexpected = set(news["Stock_symbol"].astype(str).str.upper()) - set(TICKERS)
    if unexpected:
        print(f"  [WARN] unexpected tickers dropped: {sorted(unexpected)}")
        news = news[news["Stock_symbol"].astype(str).str.upper().isin(TICKERS)]

    print("publication-year counts (should match README table):")
    print(_parse_dates(news["Date"]).dt.year.value_counts().sort_index().to_string())

    shifted = assign_trading_day(news, cal)
    n_dropped = int(shifted["trading_day"].isna().sum())
    print(f"rows={len(shifted)}  after_hours_shifted={int(shifted['after_hours'].sum())}  "
          f"midnight_rows={int(shifted['midnight'].sum())}  dropped_beyond_calendar={n_dropped}")

    daily = aggregate_daily(shifted)
    tickers = sorted(TICKERS)
    wides = build_wide_panels(daily, cal, tickers)

    daily_nd = aggregate_daily(assign_trading_day(news, cal, midnight_policy="next_day"))
    wides_nd = build_wide_panels(daily_nd, cal, tickers)
    corr = wides["sent_ema"].corrwith(wides_nd["sent_ema"]).dropna()
    print(f"robustness (midnight same_day vs next_day) sent_ema corr: "
          f"min={corr.min():.3f} mean={corr.mean():.3f} "
          f"constant_tickers={len(tickers) - len(corr)}")
    if len(corr) and corr.min() < 0.9:
        print("  [WARN] min corr < 0.9 → investigate midnight rows before modelling")

    long_df = build_daily_panel(daily, cal, tickers)
    out = PROCESSED_DIR / "sentiment_daily.parquet"
    long_df.to_parquet(out, index=False)
    expected = len(cal) * len(tickers)
    print(f"saved -> {out}  rows={len(long_df)} (expect {expected})")


if __name__ == "__main__":
    main()
