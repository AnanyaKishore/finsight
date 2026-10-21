"""Unit tests for the ingestion orchestration (fake providers + mongomock)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from finsight.common.exceptions import ProviderError
from finsight.common.mongo import NEWS, PIPELINE_RUNS, PRICES
from finsight.ingestion.quota import QuotaLedger
from finsight.ingestion.repository import IngestionRepository
from finsight.ingestion.service import IngestionService
from tests.conftest import FIXED_NOW, FakeNewsProvider, FakePriceProvider

SYMBOLS = ["AAPL", "MSFT", "NVDA"]


def _service(db, params, tmp_path=None, **overrides):
    kwargs = {
        "price_provider": FakePriceProvider("marketstack", metered=True),
        "fallback_provider": FakePriceProvider("yfinance"),
        "news_provider": FakeNewsProvider(),
        "ledger": QuotaLedger(
            db,
            "marketstack",
            budget=params["marketstack_monthly_budget"],
            reserve=params["marketstack_reserve"],
            clock=lambda: FIXED_NOW,
        ),
        "report_dir": tmp_path,
        "clock": lambda: FIXED_NOW,
    }
    kwargs.update(overrides)
    return IngestionService(IngestionRepository(db), params, SYMBOLS, **kwargs)


def test_backfill_loads_prices_news_and_records_run(db, ingestion_params, tmp_path):
    service = _service(db, ingestion_params, tmp_path)
    report = service.run("backfill")

    assert report.status == "success"
    # 30 calendar days back to 2026-10-21 -> 23 weekdays per ticker (fake calendar).
    assert db[PRICES].count_documents({"ticker": "AAPL"}) == 23
    assert db[PRICES].count_documents({}) == 69
    assert report.api_calls["marketstack"] == 3
    assert report.provider_rows == {"marketstack": 69}
    # 15 days of news in 7-day windows -> 3 windows x 2 articles x 3 tickers.
    assert db[NEWS].count_documents({}) == 18
    run = db[PIPELINE_RUNS].find_one({"run_id": report.run_id})
    assert run["status"] == "success" and run["prices"]["fetched"] == 69
    saved = json.loads((tmp_path / "ingestion_last_run.json").read_text())
    assert saved["run_id"] == report.run_id and saved["throughput_rows_per_s"] >= 0


def test_rerun_is_idempotent(db, ingestion_params):
    _service(db, ingestion_params).run("backfill")
    second = _service(db, ingestion_params).run("backfill")
    assert db[PRICES].count_documents({}) == 69
    assert db[NEWS].count_documents({}) == 18
    assert second.prices["inserted"] == 0 and second.news["inserted"] == 0
    # Identical data must not count as a revision (caught by the real-Mongo test).
    assert second.prices["updated"] == 0 and second.news["updated"] == 0


def test_incremental_skips_up_to_date_tickers(db, ingestion_params):
    _service(db, ingestion_params).run("backfill", prices=True, news=False)
    primary = FakePriceProvider("marketstack", metered=True)
    service = _service(db, ingestion_params, price_provider=primary)
    tasks = service.plan("incremental", prices=True, news=False)
    assert tasks == []  # everything already has 2026-10-21
    report = service.run("incremental", news=False)
    assert primary.calls == [] and report.status == "success"


def test_incremental_gap_fills_from_last_date(db, ingestion_params):
    repo = IngestionRepository(db)
    from finsight.ingestion.models import PriceBar

    repo.upsert_prices([PriceBar("AAPL", datetime(2026, 10, 15, tzinfo=UTC), 1, 1, 1, 1, 1)])
    service = _service(db, ingestion_params)
    tasks = [t for t in service.plan("incremental", ["AAPL"], prices=True, news=False)]
    assert len(tasks) == 1
    assert (tasks[0].date_from, tasks[0].date_to) == (date(2026, 10, 16), date(2026, 10, 21))


def test_quota_exhaustion_switches_to_fallback(db, ingestion_params):
    ingestion_params = {
        **ingestion_params,
        "marketstack_monthly_budget": 2,
        "marketstack_reserve": 0,
    }
    primary = FakePriceProvider("marketstack", metered=True)
    fallback = FakePriceProvider("yfinance")
    service = _service(db, ingestion_params, price_provider=primary, fallback_provider=fallback)
    report = service.run("backfill", news=False)

    assert report.status == "success"
    assert len(primary.calls) == 2 and len(fallback.calls) == 1
    assert report.prices["per_ticker"]["NVDA"]["provider"] == "yfinance"
    assert db[PRICES].count_documents({"source": "yfinance"}) == 23


def test_provider_reported_quota_marks_ledger(db, ingestion_params):
    primary = FakePriceProvider("marketstack", metered=True, quota_after=1)
    service = _service(db, ingestion_params, price_provider=primary)
    report = service.run("backfill", news=False)
    assert report.status == "success"
    assert service.ledger.remaining() == 0  # synced to "exhausted"
    assert report.provider_rows["yfinance"] == 46


def test_primary_failure_refunds_quota_and_uses_fallback(db, ingestion_params):
    primary = FakePriceProvider("marketstack", metered=True, fail_for={"MSFT"})
    service = _service(db, ingestion_params, price_provider=primary)
    report = service.run("backfill", news=False)
    assert report.prices["per_ticker"]["MSFT"]["provider"] == "yfinance"
    assert service.ledger.used() == 2  # MSFT's failed call was refunded


def test_both_providers_failing_gives_partial_status(db, ingestion_params):
    primary = FakePriceProvider("marketstack", metered=True, fail_for={"MSFT"})
    fallback = FakePriceProvider("yfinance", fail_for={"MSFT"})
    service = _service(db, ingestion_params, price_provider=primary, fallback_provider=fallback)
    report = service.run("backfill", news=False)
    assert report.status == "partial"
    assert report.errors[0]["ticker"] == "MSFT"
    assert db[PRICES].count_documents({"ticker": "MSFT"}) == 0
    assert db[PRICES].count_documents({"ticker": "AAPL"}) == 23


def test_news_auth_failure_stops_news_but_keeps_prices(db, ingestion_params):
    news = FakeNewsProvider(error=ProviderError("401", provider="fakenews", status=401))
    service = _service(db, ingestion_params, news_provider=news)
    report = service.run("backfill")
    assert len(news.calls) == 1  # stopped after the first auth failure
    assert report.status == "partial"
    assert db[PRICES].count_documents({}) == 69


def test_no_providers_configured_fails_cleanly(db, ingestion_params):
    service = _service(
        db, ingestion_params, price_provider=None, fallback_provider=None, news_provider=None
    )
    report = service.run("backfill")
    assert report.status == "failed"
    assert db[PIPELINE_RUNS].find_one({"run_id": report.run_id})["status"] == "failed"


def test_unknown_ticker_rejected(db, ingestion_params):
    import pytest

    with pytest.raises(ValueError):
        _service(db, ingestion_params).plan("incremental", ["ZZZZ"], prices=True, news=True)


def test_estimate_cost(db, ingestion_params):
    service = _service(db, ingestion_params)
    tasks = service.plan("backfill", prices=True, news=True)
    # FakePriceProvider uses the default cost of 1 per ticker.
    assert service.estimate_cost(tasks) == {"marketstack": 3, "fakenews": 9}


class _EmptyProvider(FakePriceProvider):
    """Mimics yfinance when Yahoo is unreachable: no exception, no rows."""

    def fetch_eod(self, ticker, date_from, date_to):
        self.calls.append((ticker, date_from, date_to))
        return []


def test_empty_primary_response_falls_back(db, ingestion_params):
    primary = _EmptyProvider("marketstack", metered=True)
    service = _service(db, ingestion_params, price_provider=primary)
    report = service.run("backfill", ["AAPL"], news=False)
    assert report.status == "success"
    assert report.prices["per_ticker"]["AAPL"]["provider"] == "yfinance"


def test_empty_fallback_response_is_an_error_not_silent_success(db, ingestion_params):
    service = _service(
        db, ingestion_params, price_provider=None, fallback_provider=_EmptyProvider("yfinance")
    )
    report = service.run("backfill", ["AAPL"], news=False)
    assert report.status == "failed"
    assert "no bars returned" in report.errors[0]["error"]
