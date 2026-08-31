
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config
from data.ingestion import fetch_benchmarks, fetch_news_sentiment, fetch_prices

logger = logging.getLogger(__name__)


def get_canonical_trading_calendar(price_dfs: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    common_dates = None
    for ticker, df in price_dfs.items():
        dt_index = pd.to_datetime(df.index)
        if common_dates is None:
            common_dates = set(dt_index)
        else:
            common_dates = common_dates.intersection(set(dt_index))

    sorted_calendar = pd.DatetimeIndex(sorted(list(common_dates)))
    return sorted_calendar


def shift_to_next_trading_day(
    date_val: pd.Timestamp,
    trading_calendar: pd.DatetimeIndex
) -> Optional[pd.Timestamp]:
    date_only = pd.Timestamp(date_val.date())
    if date_only in trading_calendar:
        return date_only

    future_days = trading_calendar[trading_calendar >= date_only]
    if len(future_days) > 0:
        return future_days[0]
    return None


def aggregate_daily_news(
    news_dfs: dict[str, pd.DataFrame],
    trading_calendar: pd.DatetimeIndex,
    tickers: list[str] = config.TICKER_LIST
) -> pd.DataFrame:
    vec_records = []
    cal_arr = trading_calendar.values
    for ticker in tickers:
        df_n = news_dfs.get(ticker)
        if df_n is not None and not df_n.empty:
            pub_dates = pd.to_datetime(df_n["published_at"]).dt.normalize().values
            idx = np.searchsorted(cal_arr, pub_dates, side="left")
            valid = idx < len(cal_arr)
            if np.any(valid):
                sub_df = pd.DataFrame({
                    "date": cal_arr[idx[valid]],
                    "ticker": ticker,
                    "sentiment_score": df_n["sentiment_score"].values[valid].astype(float)
                })
                vec_records.append(sub_df)

    # Build dense grid of (date, ticker)
    grid = pd.MultiIndex.from_product([trading_calendar, tickers], names=["date", "ticker"]).to_frame().reset_index(drop=True)

    if not vec_records:
        grid["article_count"] = 0
        grid["avg_sentiment"] = 0.0
        grid["sentiment_variance"] = 0.0
        grid["news_frequency"] = 0.0
        grid["weighted_sentiment"] = 0.0
        return grid.set_index(["date", "ticker"]).sort_index()

    raw_news_df = pd.concat(vec_records, ignore_index=True)

    # Fast vectorized aggregation
    grouped = raw_news_df.groupby(["date", "ticker"])["sentiment_score"].agg(
        article_count="count",
        avg_sentiment="mean",
        sentiment_variance=lambda s: float(np.var(s.values, ddof=0)) if len(s) > 1 else 0.0
    ).reset_index()

    merged_news = pd.merge(grid, grouped, on=["date", "ticker"], how="left")
    merged_news["article_count"] = merged_news["article_count"].fillna(0).astype(int)
    merged_news["avg_sentiment"] = merged_news["avg_sentiment"].fillna(0.0).astype(float)
    merged_news["sentiment_variance"] = merged_news["sentiment_variance"].fillna(0.0).astype(float)

    # Compute daily total article count across all 9 tickers
    daily_totals = merged_news.groupby("date")["article_count"].transform("sum")
    
    # NewsFrequency_t = (stock's article count that day) / (total articles across all 9 tickers that day)
    # If daily_totals == 0, frequency = 0.0
    merged_news["news_frequency"] = np.where(
        daily_totals > 0,
        merged_news["article_count"] / daily_totals,
        0.0
    )
    # Weighted Sentiment = NewsFrequency_t * AvgSentiment_t
    merged_news["weighted_sentiment"] = merged_news["news_frequency"] * merged_news["avg_sentiment"]

    return merged_news.set_index(["date", "ticker"]).sort_index()


def clean_and_merge_panel(
    price_dfs: Optional[dict[str, pd.DataFrame]] = None,
    news_dfs: Optional[dict[str, pd.DataFrame]] = None,
    sp500_df: Optional[pd.DataFrame] = None,
    rf_df: Optional[pd.DataFrame] = None,
    save_cache: bool = True
) -> pd.DataFrame:
    if price_dfs is None:
        price_dfs = fetch_prices()
    if news_dfs is None:
        news_dfs = fetch_news_sentiment()
    if sp500_df is None or rf_df is None:
        sp500_df, rf_df = fetch_benchmarks()

    # 1. Establish common trading calendar
    trading_calendar = get_canonical_trading_calendar(price_dfs)
    logger.info(f"Canonical trading calendar established: {len(trading_calendar)} days ({trading_calendar.min().strftime('%Y-%m-%d')} to {trading_calendar.max().strftime('%Y-%m-%d')})")

    # 2. Align and clean price DataFrames
    price_records = []
    for ticker, df in price_dfs.items():
        # Reindex to common trading calendar and forward fill if minor gap
        aligned_df = df.reindex(trading_calendar)
        # Forward fill prices, back fill if start missing, fill 0 for missing volume
        for col in ["open", "high", "low", "close", "adjusted_close"]:
            if col in aligned_df.columns:
                aligned_df[col] = aligned_df[col].ffill().bfill()
        if "volume" in aligned_df.columns:
            aligned_df["volume"] = aligned_df["volume"].fillna(0.0)

        aligned_df["ticker"] = ticker
        aligned_df.index.name = "date"
        price_records.append(aligned_df.reset_index())

    all_prices_df = pd.concat(price_records, ignore_index=True).set_index(["date", "ticker"]).sort_index()

    # 3. Aggregate daily news per ticker
    news_panel = aggregate_daily_news(news_dfs, trading_calendar, config.TICKER_LIST)

    # 4. Align benchmarks
    aligned_sp500 = sp500_df.reindex(trading_calendar).ffill().bfill()
    aligned_rf = rf_df.reindex(trading_calendar).ffill().bfill()

    # 5. Merge all components
    merged_panel = all_prices_df.join(news_panel, how="left")

    # Fill any remaining NaNs in news features with 0.0
    for col in ["article_count", "avg_sentiment", "sentiment_variance", "news_frequency", "weighted_sentiment"]:
        merged_panel[col] = merged_panel[col].fillna(0.0)

    # Add benchmark columns to multi-index dataframe
    merged_panel["sp500_adj_close"] = aligned_sp500["adjusted_close"].reindex(merged_panel.index.get_level_values("date")).values
    merged_panel["rf_annualized"] = aligned_rf["rf_annualized"].reindex(merged_panel.index.get_level_values("date")).values
    merged_panel["rf_daily"] = aligned_rf["rf_daily"].reindex(merged_panel.index.get_level_values("date")).values

    # Check for NaNs
    nan_counts = merged_panel.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        logger.warning(f"NaNs detected in merged panel:\n{nan_counts[nan_counts > 0]}")
        raise ValueError(f"Merged panel contains {total_nans} unexpected NaNs before feature engineering.")
    else:
        logger.info(f"Merged panel cleanly created with shape {merged_panel.shape} and 0 NaNs.")

    if save_cache:
        cache_path = config.CACHE_DIR / "merged_panel.parquet"
        merged_panel.to_parquet(cache_path)
        logger.info(f"Saved merged panel to {cache_path}")

    return merged_panel


if __name__ == "__main__":
    panel = clean_and_merge_panel()
    print("--- Merged Panel Summary ---")
    print(f"Shape: {panel.shape}")
    print(f"Index levels: {panel.index.names}")
    print(f"Date range: {panel.index.get_level_values('date').min().strftime('%Y-%m-%d')} to {panel.index.get_level_values('date').max().strftime('%Y-%m-%d')}")
    print(f"Tickers: {panel.index.get_level_values('ticker').unique().tolist()}")
    print(f"Columns: {panel.columns.tolist()}")
    print(f"Total NaNs: {panel.isna().sum().sum()}")
