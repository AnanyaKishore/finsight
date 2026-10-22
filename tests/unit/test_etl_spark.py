"""Tests for the Spark ETL transformations (local SparkSession, tiny data)."""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from finsight.data.export import write_parquet
from finsight.etl import process
from tests.factories import make_news, make_prices

pyspark = pytest.importorskip("pyspark")

ETL = {
    "spark_master": "local[2]",
    "driver_memory": "512m",
    "shuffle_partitions": 4,
    "spark_ui": False,
    "max_ffill_days": 2,
    "feature_windows": [5, 20],
    "processed_prices_path": "data/processed/prices",
    "processed_news_path": "data/processed/news",
    "news_weekly_path": "data/processed/news_weekly",
    "metrics_path": "reports/etl_metrics.json",
}
EXPORT = {"prices_path": "data/raw/prices.parquet", "news_path": "data/raw/news.parquet"}


@pytest.fixture(scope="session")
def spark():
    from finsight.etl.spark_session import build_spark

    session, _ = build_spark(ETL, app_name="finsight-tests")
    yield session
    session.stop()


def _aligned(spark, prices: pd.DataFrame, max_ffill=2) -> pd.DataFrame:
    sdf = spark.createDataFrame(prices)
    first, last = prices["date"].min(), prices["date"].max()
    cal = process.trading_calendar(spark, first, last)
    out = process.align_to_calendar(process.with_model_price(sdf), cal, max_ffill)
    return out.toPandas().sort_values(["ticker", "date"]).reset_index(drop=True)


def test_calendar_skips_weekends_and_holidays(spark):
    days = [
        r.date
        for r in process.trading_calendar(spark, date(2026, 11, 23), date(2026, 11, 30)).collect()
    ]
    assert date(2026, 11, 26) not in days  # Thanksgiving
    assert date(2026, 11, 28) not in days  # Saturday
    assert len(days) == 5


def test_single_gap_is_forward_filled_and_flagged(spark):
    gap = date(2026, 9, 2)
    out = _aligned(spark, make_prices(drop={"AAPL": [gap]}))
    row = out[(out["ticker"] == "AAPL") & (out["date"] == gap)].iloc[0]
    prev = out[(out["ticker"] == "AAPL") & (out["date"] == date(2026, 9, 1))].iloc[0]
    assert bool(row["is_filled"]) and row["close"] == prev["close"] and row["volume"] == 0
    assert row["open"] == row["high"] == row["low"] == row["close"]
    assert out["is_filled"].sum() == 1


def test_long_gap_is_only_partly_filled(spark):
    gap = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]
    out = _aligned(spark, make_prices(drop={"AAPL": gap}), max_ffill=2)
    aapl_days = set(out.loc[out["ticker"] == "AAPL", "date"])
    assert gap[0] in aapl_days and gap[1] in aapl_days and gap[2] not in aapl_days


def test_non_trading_rows_are_dropped(spark):
    prices = make_prices()
    saturday = prices.iloc[[3]].assign(date=date(2026, 10, 17))
    out = _aligned(spark, pd.concat([prices, saturday], ignore_index=True))
    assert date(2026, 10, 17) not in set(out["date"])
    assert len(out) == len(prices)


def test_model_price_uses_close_when_adj_incomplete(spark):
    prices = make_prices()
    prices.loc[prices["ticker"] == "MSFT", "adj_close"] = prices["close"] * 0.5
    prices.loc[(prices["ticker"] == "AAPL") & (prices.index == 0), "adj_close"] = np.nan
    out = _aligned(spark, prices)
    aapl, msft = out[out["ticker"] == "AAPL"], out[out["ticker"] == "MSFT"]
    assert np.allclose(aapl["price"], aapl["close"])
    assert np.allclose(msft["price"], msft["adj_close"])


def test_features_match_pandas_and_respect_warmup(spark):
    prices = make_prices(tickers=("AAPL",))
    sdf = process.with_model_price(spark.createDataFrame(prices)).withColumn(
        "is_filled", pyspark.sql.functions.lit(False)
    )
    out = process.add_features(sdf, [5]).toPandas().sort_values("date").reset_index(drop=True)
    expected = np.log(prices["adj_close"] / prices["adj_close"].shift(1))
    assert np.isnan(out.loc[0, "log_return"])
    assert np.allclose(out["log_return"][1:], expected[1:])
    assert out["ret_mean_5"][:5].isna().all() and not np.isnan(out.loc[5, "ret_mean_5"])
    assert np.isclose(out.loc[10, "ret_mean_5"], expected[6:11].mean())
    assert np.isclose(out.loc[10, "volatility_5"], expected[6:11].std(ddof=1))
    assert out["ma_5"][:4].isna().all() and np.isclose(
        out.loc[4, "ma_5"], prices["adj_close"][:5].mean()
    )


def test_clean_news_removes_blank_and_duplicates(spark):
    news = make_news(tickers=("AAPL",), weeks=1, per_week=4)
    dup_id = news.iloc[[0]]
    syndicated = news.iloc[[1]].assign(news_id="other", headline="  aapl HEADLINE   0-1 ")
    blank = news.iloc[[2]].assign(news_id="blank", headline="   ")
    sdf = spark.createDataFrame(pd.concat([news, dup_id, syndicated, blank], ignore_index=True))
    clean, removed = process.clean_news(sdf)
    assert removed == {"blank_headlines": 1, "duplicate_ids": 1, "duplicate_headlines": 1}
    rows = clean.toPandas()
    assert len(rows) == 4
    assert set(rows["week_start"]) == {date(2026, 6, 1)}  # Monday


def _digest(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def test_write_is_deterministic_and_clean(spark, tmp_path):
    frame = spark.createDataFrame(make_prices())
    process.write_deterministic(frame, tmp_path / "a", ["ticker", "date"], partition_by="ticker")
    process.write_deterministic(
        frame.orderBy(pyspark.sql.functions.rand(seed=3)),
        tmp_path / "b",
        ["ticker", "date"],
        partition_by="ticker",
    )
    first, second = _digest(tmp_path / "a"), _digest(tmp_path / "b")
    assert sorted(first) == ["ticker=AAPL/part-00000.parquet", "ticker=MSFT/part-00000.parquet"]
    assert first == second  # same data, shuffled input order -> identical bytes


def test_run_etl_end_to_end_and_replication(spark, tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    write_parquet(make_prices(drop={"MSFT": [date(2026, 9, 2)]}), raw / "prices.parquet")
    write_parquet(make_news(), raw / "news.parquet")

    metrics = process.run_etl(spark, ETL, EXPORT, tmp_path)
    assert metrics["quality"]["price_rows_forward_filled"] == 1
    assert [s["step"] for s in metrics["steps"]] == [
        "read_snapshot",
        "align_calendar",
        "engineer_features",
        "clean_aggregate_news",
        "write_parquet",
    ]
    prices = pd.read_parquet(tmp_path / "data/processed/prices")
    assert set(prices["ticker"].astype(str)) == {"AAPL", "MSFT"}
    assert {"log_return", "volatility_20", "price_to_ma_5", "is_filled"} <= set(prices.columns)
    weekly = pd.read_parquet(tmp_path / "data/processed/news_weekly")
    assert weekly["article_count"].sum() == metrics["outputs"]["news_rows"] == 160

    scaled = process.run_etl(
        spark, ETL, EXPORT, tmp_path, output_root=tmp_path / "bench", replicate=3
    )
    assert scaled["inputs"]["price_rows"] == 3 * metrics["inputs"]["price_rows"]
    assert scaled["outputs"]["tickers"] == 6
