import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR, TICKERS
from src.data.build_sentiment import (aggregate_daily, assign_trading_day,
                                      build_daily_panel, build_wide_panels)
from src.data.labels import compute_forward_returns


@pytest.fixture
def calendar():
    return pd.bdate_range("2023-01-02", "2023-01-31")


def _mk_news(rows):
    return pd.DataFrame(rows, columns=["Date", "Stock_symbol", "sentiment_score"])


def test_after_hours_maps_to_next_trading_day(calendar):
    rows = [
        ("2023-01-03 18:30:00", "AAPL", -0.5),
        ("2023-01-03 15:00:00", "AAPL",  0.2),
        ("2023-01-06 20:00:00", "KO",   -0.1),
        ("2023-01-07 10:00:00", "KO",    0.3),
    ]
    out = assign_trading_day(_mk_news(rows), calendar)
    td = dict(zip(out["published_at"].astype(str), out["trading_day"]))
    assert td["2023-01-03 18:30:00"] == pd.Timestamp("2023-01-04")
    assert td["2023-01-03 15:00:00"] == pd.Timestamp("2023-01-03")
    assert td["2023-01-06 20:00:00"] == pd.Timestamp("2023-01-09")
    assert td["2023-01-07 10:00:00"] == pd.Timestamp("2023-01-09")


def test_no_assignment_before_publication(calendar):
    rows = [("2023-01-07 09:00:00", "XOM", 0.1),
            ("2023-01-31 23:00:00", "GS", -0.2),
            ("2023-01-03 12:00:00", "BA", 0.0)]
    out = assign_trading_day(_mk_news(rows), calendar)
    ok = out.dropna(subset=["trading_day"])
    assert (ok["trading_day"] >= ok["published_at"].dt.normalize()).all()
    assert len(ok) == 2


def test_midnight_policy_variants(calendar):
    rows = [("2023-01-03 00:00:00", "AAPL", 0.4)]
    same = assign_trading_day(_mk_news(rows), calendar, midnight_policy="same_day")
    nxt = assign_trading_day(_mk_news(rows), calendar, midnight_policy="next_day")
    assert same["trading_day"].iloc[0] == pd.Timestamp("2023-01-03")
    assert nxt["trading_day"].iloc[0] == pd.Timestamp("2023-01-04")


def test_aggregation_mean_and_count(calendar):
    rows = [("2023-01-03 10:00:00", "AAPL", 0.4),
            ("2023-01-03 14:00:00", "AAPL", -0.2),
            ("2023-01-03 11:00:00", "KO", 0.9)]
    daily = aggregate_daily(assign_trading_day(_mk_news(rows), calendar))
    aapl = daily[daily["ticker"] == "AAPL"].iloc[0]
    assert aapl["count"] == 2 and np.isclose(aapl["sent"], 0.1)


def test_panel_zero_fill_and_ema_decay(calendar):
    rows = [("2023-01-02 10:00:00", "AAPL", 1.0)]
    daily = aggregate_daily(assign_trading_day(_mk_news(rows), calendar))
    panel = build_daily_panel(daily, calendar, tickers=["AAPL", "KO"])

    assert panel.notna().all().all()
    sub = panel[panel["ticker"] == "AAPL"].sort_values("date").reset_index(drop=True)
    assert sub.loc[0, "sent"] == 1.0
    assert (sub.loc[1:, "sent"] == 0).all()
    assert (sub["count"].to_numpy()[1:] == 0).all()
    ema = sub["sent_ema"].to_numpy()
    assert np.isclose(ema[0], 1.0)
    assert 0 < ema[1] < ema[0]
    assert (np.diff(ema[1:]) < 0).all()
    ko = panel[panel["ticker"] == "KO"]
    assert (ko[["sent", "count", "log_count", "sent_ema"]] == 0).all().all()


def test_forward_return_alignment_and_nan_tail():
    idx = pd.bdate_range("2023-01-02", periods=10)
    close = pd.DataFrame({"A": np.arange(100, 110, dtype=float)}, index=idx)
    fr = compute_forward_returns(close, horizon=5)
    assert np.isclose(fr.iloc[0, 0], 105 / 100 - 1)
    assert fr.iloc[-5:, :].isna().all().all()
    assert fr.iloc[:-5, :].notna().all().all()


@pytest.mark.skipif(not (PROCESSED_DIR / "prices.parquet").exists(),
                    reason="run download_prices first")
def test_prices_integration():
    df = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    assert set(df["ticker"].unique()) == set(TICKERS)
    assert df["close"].notna().all()
    n_days = df.groupby("ticker")["date"].nunique()
    assert n_days.min() > 1200


@pytest.mark.skipif(not (PROCESSED_DIR / "sentiment_daily.parquet").exists(),
                    reason="run build_sentiment first")
def test_sentiment_integration():
    sent = pd.read_parquet(PROCESSED_DIR / "sentiment_daily.parquet")
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    assert set(sent["ticker"].unique()) == set(TICKERS)
    assert sent.notna().all().all()
    assert sent["sent"].abs().le(1.0).all()
    assert (sent["count"] >= 0).all()
    cal = set(pd.DatetimeIndex(prices["date"].unique()))
    assert set(pd.DatetimeIndex(sent["date"].unique())) <= cal
