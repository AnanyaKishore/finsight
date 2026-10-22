"""Regression tests for zero prices from a free feed (found on real data, Day 2).

The real snapshot contained 3 bars with a non-positive price: one isolated and
two on consecutive days, which produced 4 infinite log returns.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from finsight.common.mongo import PIPELINE_RUNS, PRICES
from finsight.data import validate
from finsight.ingestion.models import PriceBar, make_price_bar
from finsight.ingestion.providers.marketstack import parse_eod_row
from finsight.ingestion.repository import IngestionRepository
from finsight.ingestion.service import IngestionService
from tests.conftest import FIXED_NOW, FakePriceProvider
from tests.factories import make_news, make_prices
from tests.unit.test_export_and_validate import NOW, PARAMS, TICKERS

DAY = datetime(2026, 9, 1, tzinfo=UTC)


# ------------------------------------------------------------- boundary rules
def test_parser_rejects_zero_close_and_cleans_other_fields():
    assert parse_eod_row({"date": "2026-09-01", "close": 0, "open": 1}, "AAPL") is None
    assert parse_eod_row({"date": "2026-09-01", "close": "0.0"}, "AAPL") is None
    bar = parse_eod_row(
        {
            "date": "2026-09-01",
            "close": 10,
            "open": 0,
            "high": None,
            "low": -1,
            "adj_close": 0,
            "volume": -5,
        },
        "AAPL",
    )
    assert (bar.open, bar.high, bar.low, bar.adj_close, bar.volume) == (10, 10, 10, None, 0.0)


def test_make_price_bar_rejects_nan_and_inf():
    for bad in (float("nan"), float("inf"), None, "abc"):
        assert (
            make_price_bar(
                "A", DAY, open_=1, high=1, low=1, close=bad, volume=1, adj_close=1, source="t"
            )
            is None
        )


# ----------------------------------------------------------------- validation
def _with_zero_bars():
    prices = make_prices()
    aapl = prices.index[prices["ticker"] == "AAPL"]
    for i in (aapl[50], aapl[80], aapl[81]):  # one isolated, two consecutive
        prices.loc[i, ["close", "adj_close"]] = 0.0
    return prices


def test_zero_bars_are_listed_and_not_double_counted():
    suite = validate.run_suite(_with_zero_bars(), make_news(), TICKERS, PARAMS, now=NOW)
    positive = next(c for c in suite.checks if c.name == "prices.positive_values")
    extreme = next(c for c in suite.checks if c.name == "prices.no_extreme_returns")
    assert not suite.success and positive.observed == 3
    assert positive.details[0][0] == "AAPL" and "repair" in positive.details[-1]
    assert extreme.success  # zero prices no longer turn into infinite "returns"


# --------------------------------------------------------------------- repair
def _service(db, params, fallback):
    return IngestionService(
        IngestionRepository(db),
        params,
        ["AAPL", "MSFT"],
        price_provider=None,
        fallback_provider=fallback,
        news_provider=None,
        ledger=None,
        clock=lambda: FIXED_NOW,
    )


def _store_zero_bar(db):
    repo = IngestionRepository(db)
    repo.upsert_prices(
        [
            PriceBar("AAPL", DAY, 1, 1, 1, 0.0, 100, 0.0, "marketstack"),  # legacy bad row
            PriceBar(
                "AAPL", datetime(2026, 9, 2, tzinfo=UTC), 1, 2, 1, 1.5, 100, 1.4, "marketstack"
            ),
        ]
    )


def test_repair_dry_run_changes_nothing(db, ingestion_params):
    _store_zero_bar(db)
    summary = _service(db, ingestion_params, FakePriceProvider("yfinance")).repair(dry_run=True)
    assert summary["invalid_bars"] == 1 and summary["actions"][0]["action"] == "would_repair"
    assert db[PRICES].find_one({"ticker": "AAPL", "date": DAY})["close"] == 0.0
    assert db[PIPELINE_RUNS].count_documents({}) == 0


def test_repair_replaces_from_fallback_without_adj_close(db, ingestion_params):
    _store_zero_bar(db)
    summary = _service(db, ingestion_params, FakePriceProvider("yfinance")).repair()
    assert summary["replaced"] == 1 and summary["deleted"] == 0
    fixed = db[PRICES].find_one({"ticker": "AAPL", "date": DAY})
    assert fixed["close"] > 0 and fixed["adj_close"] is None and fixed["source"] == "yfinance"
    assert IngestionRepository(db).invalid_prices() == []
    run = db[PIPELINE_RUNS].find_one({"mode": "repair"})
    assert run["status"] == "success" and run["repair"]["replaced"] == 1


def test_repair_deletes_when_fallback_has_nothing(db, ingestion_params):
    _store_zero_bar(db)
    summary = _service(
        db, ingestion_params, FakePriceProvider("yfinance", fail_for={"AAPL"})
    ).repair()
    assert summary["deleted"] == 1 and "fallback_error" in summary["actions"][0]
    assert db[PRICES].count_documents({"ticker": "AAPL"}) == 1


def test_repaired_day_is_a_trading_day_in_fixture():
    assert date(2026, 9, 1).weekday() < 5
