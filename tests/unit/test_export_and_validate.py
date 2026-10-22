"""Tests for the export snapshot and the data-quality expectation suite."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from finsight.common.mongo import NEWS, PRICES
from finsight.data import validate
from finsight.data.export import export_snapshot, file_sha256
from finsight.ingestion.models import NewsArticle, PriceBar
from finsight.ingestion.repository import IngestionRepository
from tests.factories import make_news, make_prices

TICKERS = ["AAPL", "MSFT"]
NOW = lambda: datetime(2026, 10, 21, 23, 0, tzinfo=UTC)  # noqa: E731 -- Wed after close
PARAMS = {
    "min_history_rows": 114,
    "max_missing_days_warn": 3,
    "max_missing_days_error": 10,
    "max_staleness_days": 3,
    "return_outlier_z": 6.0,
    "max_abs_log_return": 0.5,
    "min_articles_per_week": 5,
}
EXPORT_PARAMS = {
    "prices_path": "data/raw/prices.parquet",
    "news_path": "data/raw/news.parquet",
    "manifest_path": "reports/export_manifest.json",
}


def _status(suite, name):
    return next(c for c in suite.checks if c.name == name)


# ----------------------------------------------------------------------- export
def _seed_db(db):
    repo = IngestionRepository(db)
    repo.upsert_prices(
        [
            PriceBar("MSFT", datetime(2026, 10, 19, tzinfo=UTC), 1, 2, 0.5, 1.5, 10, 1.4, "ms"),
            PriceBar("AAPL", datetime(2026, 10, 20, tzinfo=UTC), 1, 2, 0.5, 1.6, 10, None, "ms"),
            PriceBar("AAPL", datetime(2026, 10, 19, tzinfo=UTC), 1, 2, 0.5, 1.5, 10, None, "ms"),
        ]
    )
    repo.upsert_news(
        [
            NewsArticle(
                "AAPL",
                "1",
                "Apple up",
                "s",
                "Reuters",
                "u",
                datetime(2026, 10, 19, 9, tzinfo=UTC),
                "finnhub",
                {"category": "company"},
            )
        ]
    )


def test_export_writes_sorted_typed_snapshot_and_manifest(db, tmp_path):
    _seed_db(db)
    manifest = export_snapshot(db, TICKERS, EXPORT_PARAMS, tmp_path)
    prices = pd.read_parquet(tmp_path / "data/raw/prices.parquet")
    assert list(prices["ticker"]) == ["AAPL", "AAPL", "MSFT"]
    assert prices["date"].iloc[0] == date(2026, 10, 19)
    assert "ingested_at" not in prices.columns and "_id" not in prices.columns
    news = pd.read_parquet(tmp_path / "data/raw/news.parquet")
    assert news.loc[0, "category"] == "company"
    assert manifest["prices"]["rows"] == 3 and manifest["prices"]["tickers"] == 2
    assert manifest["prices"]["date_range"] == ["2026-10-19", "2026-10-20"]
    on_disk = json.loads((tmp_path / "reports/export_manifest.json").read_text())
    assert on_disk["prices"]["sha256"] == file_sha256(tmp_path / "data/raw/prices.parquet")


def test_export_is_byte_identical_when_data_unchanged(db, tmp_path):
    """DVC skips downstream stages only if an unchanged DB gives identical files."""
    _seed_db(db)
    first = export_snapshot(db, TICKERS, EXPORT_PARAMS, tmp_path / "a")
    db[PRICES].update_many({}, {"$set": {"ingested_at": datetime.now(UTC)}})  # bookkeeping only
    second = export_snapshot(db, TICKERS, EXPORT_PARAMS, tmp_path / "b")
    assert first["prices"]["sha256"] == second["prices"]["sha256"]
    assert first["news"]["sha256"] == second["news"]["sha256"]
    db[NEWS].update_one({}, {"$set": {"headline": "Apple down"}})
    third = export_snapshot(db, TICKERS, EXPORT_PARAMS, tmp_path / "c")
    assert third["news"]["sha256"] != first["news"]["sha256"]


def test_export_refuses_empty_database(db, tmp_path):
    from finsight.common.exceptions import FinSightError

    with pytest.raises(FinSightError):
        export_snapshot(db, TICKERS, EXPORT_PARAMS, tmp_path)


# --------------------------------------------------------------------- validate
def test_clean_dataset_passes():
    suite = validate.run_suite(make_prices(), make_news(), TICKERS, PARAMS, now=NOW)
    report = suite.to_dict()
    assert report["status"] == "pass", [c for c in report["checks"] if not c["success"]]
    assert report["per_ticker"]["AAPL"]["missing_trading_days"] == 0
    json.dumps(report)  # must be serialisable


def test_suite_fails_on_dataset_with_known_issues():
    """Module 2 checklist: the suite must FAIL on a dataset with known problems."""
    prices = make_prices()
    dup = prices.iloc[[5]]
    prices = pd.concat([prices, dup], ignore_index=True)  # duplicate key
    prices.loc[10, "close"] = -1.0  # impossible price
    suite = validate.run_suite(prices, make_news(), TICKERS, PARAMS, now=NOW)
    assert not suite.success
    assert not _status(suite, "prices.unique_ticker_date").success
    assert not _status(suite, "prices.positive_values").success


def test_missing_days_warn_then_block():
    few = {"AAPL": [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4)]}
    suite = validate.run_suite(make_prices(drop=few), make_news(), TICKERS, PARAMS, now=NOW)
    assert suite.success
    assert _status(suite, "prices.missing_days").observed == {"AAPL": 4}
    assert suite.per_ticker["AAPL"]["missing_dates"][0] == date(2026, 9, 1)

    many = {"AAPL": [d.date() for d in pd.bdate_range("2026-08-03", "2026-08-21")]}
    suite = validate.run_suite(make_prices(drop=many), make_news(), TICKERS, PARAMS, now=NOW)
    assert not suite.success
    assert "AAPL" in _status(suite, "prices.missing_days_blocking").observed


def test_unadjusted_split_is_blocking():
    prices = make_prices()
    mask = (prices["ticker"] == "AAPL") & (prices["date"] >= date(2026, 9, 1))
    prices.loc[mask, ["open", "high", "low", "close", "adj_close"]] /= 4  # 4:1 split
    suite = validate.run_suite(prices, make_news(), TICKERS, PARAMS, now=NOW)
    check = _status(suite, "prices.no_extreme_returns")
    assert not check.success and check.details[0][:2] == ["AAPL", "2026-09-01"]


def test_short_history_and_missing_ticker_block():
    prices = make_prices(tickers=("AAPL",), start=date(2026, 8, 3))
    suite = validate.run_suite(prices, make_news(), TICKERS, PARAMS, now=NOW)
    assert not _status(suite, "prices.min_history").success
    assert _status(suite, "prices.universe_complete").details["absent"] == ["MSFT"]


def test_stale_data_and_weekend_rows_only_warn():
    prices = make_prices(end=date(2026, 10, 9))
    weekend = prices.iloc[[0]].assign(date=date(2026, 10, 10))  # Saturday
    prices = pd.concat([prices, weekend], ignore_index=True)
    suite = validate.run_suite(prices, make_news(), TICKERS, PARAMS, now=NOW)
    assert suite.success
    assert not _status(suite, "prices.fresh").success
    assert not _status(suite, "prices.only_trading_days").success


def test_return_outlier_warns_but_does_not_block():
    prices = make_prices()
    idx = prices.index[(prices["ticker"] == "MSFT") & (prices["date"] == date(2026, 7, 1))][0]
    prices.loc[idx:, ["close", "adj_close"]] = prices.loc[idx:, ["close", "adj_close"]] * 1.25
    prices.loc[prices["ticker"] != "MSFT", ["close", "adj_close"]] = make_prices().loc[
        prices["ticker"] != "MSFT", ["close", "adj_close"]
    ]
    suite = validate.run_suite(prices, make_news(), TICKERS, PARAMS, now=NOW)
    assert suite.success
    assert not _status(suite, "prices.return_outliers").success


def test_model_price_prefers_complete_adj_close():
    prices = make_prices()
    prices.loc[prices["ticker"] == "MSFT", "adj_close"] = prices["close"] * 0.9
    prices.loc[(prices["ticker"] == "AAPL") & (prices.index % 7 == 0), "adj_close"] = np.nan
    price = validate.model_price(prices)
    aapl, msft = prices["ticker"] == "AAPL", prices["ticker"] == "MSFT"
    assert (price[aapl] == prices.loc[aapl, "close"]).all()  # incomplete -> close
    assert (price[msft] == prices.loc[msft, "adj_close"]).all()


def test_news_duplicates_block_and_thin_weeks_warn():
    news = make_news(per_week=3)
    suite = validate.run_suite(make_prices(), news, TICKERS, PARAMS, now=NOW)
    assert suite.success and not _status(suite, "news.weekly_coverage").success
    dup = pd.concat([news, news.iloc[[0]]], ignore_index=True)
    suite = validate.run_suite(make_prices(), dup, TICKERS, PARAMS, now=NOW)
    assert not _status(suite, "news.unique_ticker_id").success and not suite.success


def test_broken_schema_is_reported_once():
    prices = make_prices().drop(columns=["close"])
    suite = validate.run_suite(prices, make_news(), TICKERS, PARAMS, now=NOW)
    names = [c.name for c in suite.checks if c.name.startswith("prices.")]
    assert names == ["prices.schema"] and not suite.success
