"""
Data ingestion pipeline for prices, benchmarks, and news sentiment with local caching.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_date_coverage(
    df: pd.DataFrame,
    start_date: str = config.DATE_START,
    end_date: str = config.DATE_END,
    min_rows: int = 1000,
    date_col: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validates that a DataFrame covers the expected date range without invalid gaps.
    Allows for up to 5 days buffer at start/end for weekends and market holidays.
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None."

    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col]).sort_values()
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.sort_values()
    else:
        return False, "Could not identify datetime index or date column."

    min_d = dates.min()
    max_d = dates.max()
    target_s_buffered = pd.Timestamp(start_date) + pd.Timedelta(days=5)
    target_e_buffered = pd.Timestamp(end_date) - pd.Timedelta(days=5)

    if min_d > target_s_buffered:
        return False, f"Start date {min_d.strftime('%Y-%m-%d')} is after target window {start_date}."
    if max_d < target_e_buffered:
        return False, f"End date {max_d.strftime('%Y-%m-%d')} is before target window {end_date}."
    if len(dates) < min_rows:
        return False, f"Row count {len(dates)} is below expected minimum {min_rows}."

    return True, f"Valid coverage: {len(dates)} rows from {min_d.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}."


# ==========================================
# 1. Price Data Ingestion
# ==========================================
def _fetch_single_ticker_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str
) -> pd.DataFrame:
    """Fetch prices for a single ticker with AlphaVantage and robust yfinance retry fallback."""
    fetched_df = None

    if api_key:
        logger.info(f"[{ticker}] Attempting AlphaVantage pull...")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            if "Time Series (Daily)" in data:
                raw_ts = data["Time Series (Daily)"]
                records = []
                for d_str, vals in raw_ts.items():
                    records.append({
                        "date": pd.Timestamp(d_str),
                        "open": float(vals.get("1. open", 0.0)),
                        "high": float(vals.get("2. high", 0.0)),
                        "low": float(vals.get("3. low", 0.0)),
                        "close": float(vals.get("4. close", 0.0)),
                        "adjusted_close": float(vals.get("5. adjusted close", vals.get("4. close", 0.0))),
                        "volume": float(vals.get("6. volume", 0.0)),
                    })
                fetched_df = pd.DataFrame(records).set_index("date").sort_index()
            elif "Information" in data:
                logger.warning(f"[{ticker}] AlphaVantage endpoint notice: {data['Information'][:75]}...")
        except Exception as e:
            logger.warning(f"[{ticker}] AlphaVantage request failed: {e}")

    if fetched_df is None or len(fetched_df) < 100:
        logger.info(f"[{ticker}] Fetching via yfinance (with retries)...")
        end_dt = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        for attempt in range(1, 4):
            try:
                t = yf.Ticker(ticker)
                raw_hist = t.history(start=start_date, end=end_dt, auto_adjust=False)
                if raw_hist.empty:
                    raw_hist = yf.download(ticker, start=start_date, end=end_dt, progress=False, auto_adjust=False)
                    if isinstance(raw_hist.columns, pd.MultiIndex):
                        raw_hist.columns = [c[0] for c in raw_hist.columns]

                if not raw_hist.empty:
                    cols_map = {
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adjusted_close",
                        "Volume": "volume",
                    }
                    df_renamed = raw_hist.rename(columns=cols_map)
                    if "adjusted_close" not in df_renamed.columns and "close" in df_renamed.columns:
                        df_renamed["adjusted_close"] = df_renamed["close"]

                    df_clean = df_renamed[["open", "high", "low", "close", "adjusted_close", "volume"]].copy()
                    if df_clean.index.tz is not None:
                        df_clean.index = df_clean.index.tz_localize(None)
                    df_clean.index.name = "date"
                    fetched_df = df_clean
                    break
            except Exception as e:
                logger.warning(f"[{ticker}] yfinance attempt {attempt} failed: {e}")
                time.sleep(1.0 * attempt)

    if fetched_df is None or fetched_df.empty:
        raise RuntimeError(f"Failed to fetch price data for {ticker}")

    fetched_df = fetched_df[(fetched_df.index >= pd.Timestamp(start_date)) & (fetched_df.index <= pd.Timestamp(end_date))]
    return fetched_df


def fetch_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = config.DATE_START,
    end_date: str = config.DATE_END,
    force_repull: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily price data for tickers.
    Primary rule: Downstream features strictly use adjusted_close.
    """
    if tickers is None:
        tickers = config.TICKER_LIST

    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    price_dfs = {}

    for ticker in tickers:
        cache_file = config.CACHE_DIR / f"prices_{ticker}.parquet"

        if cache_file.exists() and not force_repull:
            try:
                cached_df = pd.read_parquet(cache_file)
                is_valid, msg = validate_date_coverage(cached_df, start_date, end_date)
                if is_valid and "adjusted_close" in cached_df.columns:
                    logger.info(f"[{ticker}] Price Cache valid: {msg}")
                    price_dfs[ticker] = cached_df
                    continue
            except Exception:
                pass

        fetched_df = _fetch_single_ticker_prices(ticker, start_date, end_date, api_key)
        fetched_df.to_parquet(cache_file)
        price_dfs[ticker] = fetched_df
        logger.info(f"[{ticker}] Saved {len(fetched_df)} price rows to {cache_file.name}")
        time.sleep(0.5)

    return price_dfs


# ==========================================
# 2. News Sentiment Ingestion (Local FNSPID Parquet)
# ==========================================
_FNSPID_CACHE: Optional[pd.DataFrame] = None


def _load_fnspid_raw(parquet_path: Path = config.FNSPID_PARQUET_PATH) -> pd.DataFrame:
    """
    Loads the full FNSPID news-sentiment parquet once per process and caches it
    in memory (it is read repeatedly, once per ticker, by fetch_news_sentiment).
    Expected columns: Date (datetime64), Article_title (str), Stock_symbol (str),
    sentiment_score (float).
    """
    global _FNSPID_CACHE
    if _FNSPID_CACHE is not None:
        return _FNSPID_CACHE

    if not Path(parquet_path).exists():
        raise FileNotFoundError(
            f"FNSPID news parquet not found at {parquet_path}. "
            f"Place the filtered FNSPID file there before running ingestion."
        )

    df = pd.read_parquet(parquet_path)
    required_cols = {"Date", "Article_title", "Stock_symbol", "sentiment_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"FNSPID parquet is missing expected columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    _FNSPID_CACHE = df
    return df


def fetch_news_sentiment(
    tickers: Optional[List[str]] = None,
    start_date: str = config.DATE_START,
    end_date: str = config.DATE_END,
    force_repull: bool = False,
    parquet_path: Path = config.FNSPID_PARQUET_PATH,
) -> Dict[str, pd.DataFrame]:
    """
    Loads per-ticker news/sentiment records from the local FNSPID parquet dump,
    filtered to [start_date, end_date], and reshapes them into the same schema the
    rest of the pipeline (data/cleaning.py::aggregate_daily_news) expects:
    columns ["uuid", "ticker", "published_at", "title", "snippet",
    "relevance_score", "sentiment_score"].

    FNSPID has no per-article relevance score, so relevance_score is set to 1.0 for
    every row (matching the fallback default previously used for the AlphaVantage
    path). Tickers with zero matching rows in the window (e.g. any ticker outside
    FNSPID's 2018-01-01..2023-12-16 coverage) return an empty DataFrame rather than
    raising -- data/cleaning.py::aggregate_daily_news() already fills those
    (date, ticker) cells with neutral zeros.

    Results are cached to disk per ticker exactly like the old API-based path, so
    downstream code and force_repull semantics are unchanged.
    """
    if tickers is None:
        tickers = config.TICKER_LIST

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # inclusive of end_date

    news_dfs: Dict[str, pd.DataFrame] = {}
    raw_df = None  # lazily loaded only if any ticker cache is missing/invalid

    for ticker in tickers:
        cache_file = config.CACHE_DIR / f"news_{ticker}.parquet"

        if cache_file.exists() and not force_repull:
            try:
                cached_news = pd.read_parquet(cache_file)
                news_dfs[ticker] = cached_news
                logger.info(f"[{ticker}] FNSPID news cache valid ({len(cached_news)} articles).")
                continue
            except Exception:
                pass

        if raw_df is None:
            raw_df = _load_fnspid_raw(parquet_path)

        ticker_rows = raw_df[
            (raw_df["Stock_symbol"] == ticker)
            & (raw_df["Date"] >= start_ts)
            & (raw_df["Date"] < end_ts)
        ].sort_values("Date")

        if ticker_rows.empty:
            logger.warning(
                f"[{ticker}] 0 FNSPID articles found in [{start_date}, {end_date}]. "
                f"Sentiment features for this ticker will be neutral (0.0) over this window."
            )
            df_news = pd.DataFrame(
                columns=["uuid", "ticker", "published_at", "title", "snippet", "relevance_score", "sentiment_score"]
            )
        else:
            df_news = pd.DataFrame({
                "uuid": [f"{ticker}_{i}" for i in range(len(ticker_rows))],
                "ticker": ticker,
                "published_at": ticker_rows["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "title": ticker_rows["Article_title"].fillna(""),
                "snippet": "",
                "relevance_score": 1.0,
                "sentiment_score": ticker_rows["sentiment_score"].astype(float).values,
            })

        df_news.to_parquet(cache_file)
        news_dfs[ticker] = df_news
        logger.info(f"[{ticker}] Cached {len(df_news)} FNSPID news rows to {cache_file.name}")

    return news_dfs


# ==========================================
# 3. Benchmark Ingestion (S&P 500 & TB3MS)
# ==========================================
def fetch_benchmarks(
    start_date: str = config.DATE_START,
    end_date: str = config.DATE_END,
    force_repull: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch S&P 500 index (^GSPC) and FRED 3-Month Treasury yield (TB3MS).
    TB3MS is reindexed and forward-filled to the daily NYSE trading calendar.
    """
    sp500_cache = config.CACHE_DIR / "benchmark_sp500.parquet"
    rf_cache = config.CACHE_DIR / "benchmark_rf.parquet"

    # S&P 500
    sp500_df = None
    if sp500_cache.exists() and not force_repull:
        try:
            cached_sp = pd.read_parquet(sp500_cache)
            is_valid, msg = validate_date_coverage(cached_sp, start_date, end_date)
            if is_valid:
                sp500_df = cached_sp
        except Exception:
            pass

    if sp500_df is None:
        logger.info("[S&P 500] Downloading ^GSPC...")
        end_dt = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        t = yf.Ticker("^GSPC")
        raw_sp = t.history(start=start_date, end=end_dt, auto_adjust=False)
        if raw_sp.empty:
            raw_sp = yf.download("^GSPC", start=start_date, end=end_dt, progress=False, auto_adjust=False)
            if isinstance(raw_sp.columns, pd.MultiIndex):
                raw_sp.columns = [c[0] for c in raw_sp.columns]

        sp500_df = raw_sp[["Close", "Adj Close", "Volume"]].rename(
            columns={"Close": "close", "Adj Close": "adjusted_close", "Volume": "volume"}
        ).copy()
        if sp500_df.index.tz is not None:
            sp500_df.index = sp500_df.index.tz_localize(None)
        sp500_df.index.name = "date"
        sp500_df = sp500_df[(sp500_df.index >= pd.Timestamp(start_date)) & (sp500_df.index <= pd.Timestamp(end_date))]
        sp500_df.to_parquet(sp500_cache)

    # FRED TB3MS
    rf_df = None
    if rf_cache.exists() and not force_repull:
        try:
            cached_rf = pd.read_parquet(rf_cache)
            is_valid, msg = validate_date_coverage(cached_rf, start_date, end_date)
            if is_valid:
                rf_df = cached_rf
        except Exception:
            pass

    if rf_df is None:
        logger.info("[FRED TB3MS] Fetching 3-Month Treasury yield...")
        fred_key = os.getenv("FRED_API_KEY", "")
        monthly_obs = []
        if fred_key:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&api_key={fred_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    for item in r.json().get("observations", []):
                        val_str = item.get("value", ".")
                        if val_str != ".":
                            monthly_obs.append({
                                "date": pd.Timestamp(item["date"]),
                                "tb3ms_pct": float(val_str)
                            })
            except Exception as e:
                logger.warning(f"FRED API request failed: {e}")

        if not monthly_obs:
            import pandas_datareader.data as web
            try:
                fred_raw = web.DataReader("TB3MS", "fred", start_date, end_date)
                for d, row in fred_raw.iterrows():
                    monthly_obs.append({"date": pd.Timestamp(d), "tb3ms_pct": float(row["TB3MS"])})
            except Exception as e:
                logger.error(f"FRED DataReader fallback failed: {e}")

        if monthly_obs:
            df_monthly = pd.DataFrame(monthly_obs).set_index("date").sort_index()
            df_monthly["rf_annualized"] = df_monthly["tb3ms_pct"] / 100.0

            trading_calendar = sp500_df.index
            combined_idx = df_monthly.index.union(trading_calendar).sort_values()
            df_aligned = df_monthly.reindex(combined_idx).ffill().bfill()
            
            rf_df = df_aligned.reindex(trading_calendar).copy()
            rf_df["rf_daily"] = (1.0 + rf_df["rf_annualized"]) ** (1.0 / 252.0) - 1.0
            rf_df.index.name = "date"
            rf_df.to_parquet(rf_cache)
        else:
            raise RuntimeError("Failed to fetch FRED TB3MS risk-free series.")

    return sp500_df, rf_df


# ==========================================
# 4. Ingestion Check / Diagnostics Runner
# ==========================================
def run_ingestion_check():
    """Validates all caches and prints diagnostic row counts and coverage status."""
    print("=" * 70)
    print("LSTM-GAT Portfolio Model v4 — Data Ingestion & Cache Check")
    print("=" * 70)

    print("\n--- Asset Price Data Check ---")
    all_prices_valid = True
    for ticker in config.TICKER_LIST:
        p_file = config.CACHE_DIR / f"prices_{ticker}.parquet"
        if p_file.exists():
            df = pd.read_parquet(p_file)
            is_valid, msg = validate_date_coverage(df)
            adj_check = "adjusted_close" in df.columns
            print(f"[{ticker:>4}] Rows: {len(df):>4} | Start: {df.index.min().strftime('%Y-%m-%d')} | End: {df.index.max().strftime('%Y-%m-%d')} | AdjClose: OK | Valid: {is_valid}")
            if not is_valid:
                all_prices_valid = False
        else:
            print(f"[{ticker:>4}] NOT CACHED")
            all_prices_valid = False

    print("\n--- News & Sentiment Data Check (FNSPID) ---")
    all_news_present = True
    total_articles_all = 0
    for ticker in config.TICKER_LIST:
        n_file = config.CACHE_DIR / f"news_{ticker}.parquet"
        if n_file.exists():
            df = pd.read_parquet(n_file)
            total_articles_all += len(df)
            if len(df) > 0:
                date_range = f"{df['published_at'].min()[:10]} to {df['published_at'].max()[:10]}"
            else:
                date_range = "NO ARTICLES IN CONFIGURED WINDOW (neutral/zero sentiment)"
            print(f"[{ticker:>4}] Articles: {len(df):>5} | Date Range: {date_range}")
        else:
            print(f"[{ticker:>4}] NOT CACHED")
            all_news_present = False
    print(f"Total News Articles Ingested: {total_articles_all}")

    print("\n--- Benchmark Data Check ---")
    sp_file = config.CACHE_DIR / "benchmark_sp500.parquet"
    rf_file = config.CACHE_DIR / "benchmark_rf.parquet"

    sp_ok = sp_file.exists()
    rf_ok = rf_file.exists()
    print(f"[S&P 500 (^GSPC)] Cached: {sp_ok}")
    print(f"[FRED TB3MS (Rf)] Cached: {rf_ok}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Ingestion CLI for Model v4")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (e.g. AAPL,NVDA)")
    parser.add_argument("--pull-prices", action="store_true", help="Fetch price series")
    parser.add_argument("--pull-news", action="store_true", help="Fetch news sentiment series")
    parser.add_argument("--pull-benchmarks", action="store_true", help="Fetch benchmark series")
    parser.add_argument("--check", action="store_true", help="Run date coverage validation check")
    parser.add_argument("--force", action="store_true", help="Force repull ignoring cache")
    parser.add_argument("--full", action="store_true", help="Run full ingestion pipeline")

    args = parser.parse_args()
    tickers = args.tickers.split(",") if args.tickers else config.TICKER_LIST

    if args.check:
        run_ingestion_check()
        return

    if args.pull_prices or args.full:
        fetch_prices(tickers=tickers, force_repull=args.force)

    if args.pull_benchmarks or args.full:
        fetch_benchmarks(force_repull=args.force)

    if args.pull_news or args.full:
        fetch_news_sentiment(tickers=tickers, force_repull=args.force)

    if not any([args.pull_prices, args.pull_news, args.pull_benchmarks, args.check, args.full]):
        run_ingestion_check()


if __name__ == "__main__":
    main()
