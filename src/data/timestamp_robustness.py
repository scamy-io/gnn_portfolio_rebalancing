import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, TICKERS, ensure_dirs
from src.data.build_sentiment import (_parse_dates, aggregate_daily,
                                      assign_trading_day, build_wide_panels)

README_YEARS = {2018: 13415, 2019: 15333, 2020: 15608, 2021: 8110,
                2022: 22628, 2023: 40555}


def main() -> None:
    ensure_dirs()
    news = pd.read_parquet(RAW_DIR / "fnspid_news_filtered.parquet")
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))

    print("--- publication-year counts vs README ---")
    yc = _parse_dates(news["Date"]).dt.year.value_counts().sort_index()
    all_ok = True
    for y, exp in README_YEARS.items():
        got = int(yc.get(y, 0))
        ok = got == exp
        all_ok &= ok
        print(f"  {y}: got={got} expected={exp} {'OK' if ok else 'MISMATCH'}")

    print("\n--- midnight policy robustness (sent_ema per ticker) ---")
    s = aggregate_daily(assign_trading_day(news, cal, "same_day"))
    n = aggregate_daily(assign_trading_day(news, cal, "next_day"))
    ws = build_wide_panels(s, cal, sorted(TICKERS))["sent_ema"]
    wn = build_wide_panels(n, cal, sorted(TICKERS))["sent_ema"]
    corr = ws.corrwith(wn)
    n_const = int(corr.isna().sum())
    print(f"  min={corr.min():.4f}  mean={corr.mean():.4f}  "
          f"constant/NaN tickers={n_const}")
    verdict = (len(corr) - n_const) and corr.dropna().min() >= 0.9
    print(f"  VERDICT: {'PASS (min >= 0.9)' if verdict else 'FAIL - investigate before report'}")
    corr.rename("corr_same_vs_next").to_csv(
        PROCESSED_DIR / "timestamp_robustness.csv")
    print(f"  year_counts_all_match={all_ok}")


if __name__ == "__main__":
    main()
