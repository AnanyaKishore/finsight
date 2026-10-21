"""Shared pytest fixtures.

Unit tests never touch the network or a real database:
* MongoDB is replaced by ``mongomock`` (in-memory, same pymongo API).
* HTTP is intercepted with ``responses`` or replaced by fake providers.
"""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime
from pathlib import Path

import mongomock
import mongomock.collection
import pytest

from finsight.common.mongo import ensure_indexes
from finsight.common.timeutils import to_utc_midnight
from finsight.ingestion.models import NewsArticle, PriceBar
from finsight.ingestion.providers.base import NewsProvider, PriceProvider

# --------------------------------------------------------------------------- #
# Test-only compatibility shim: pymongo >= 4.11 passes ``sort=`` to the bulk
# builder's add_update(), which mongomock 4.3 does not accept yet. Production
# code runs against real MongoDB and is unaffected.
# --------------------------------------------------------------------------- #
_original_add_update = mongomock.collection.BulkOperationBuilder.add_update
if "sort" not in inspect.signature(_original_add_update).parameters:

    def _add_update(self, *args, sort=None, **kwargs):  # noqa: ARG001
        return _original_add_update(self, *args, **kwargs)

    mongomock.collection.BulkOperationBuilder.add_update = _add_update

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A Wednesday after US close (22:30 UTC = 18:30 New York): EOD for 2026-10-21 exists.
FIXED_NOW = datetime(2026, 10, 21, 22, 30, tzinfo=UTC)


@pytest.fixture
def db():
    """Fresh in-memory Mongo database with production indexes."""
    client = mongomock.MongoClient(tz_aware=True)
    database = client["finsight_test"]
    ensure_indexes(database)
    yield database
    client.close()


@pytest.fixture
def ingestion_params():
    """Ingestion params mirroring params.yaml but with tiny windows."""
    return {
        "price_backfill_days": 30,
        "news_backfill_days": 14,
        "news_window_days": 7,
        "max_articles_per_window": 5,
        "marketstack_monthly_budget": 10,
        "marketstack_reserve": 1,
        "price_fallback_provider": "yfinance",
        "finnhub_min_interval_s": 0,
        "http_max_retries": 2,
        "http_backoff_base_s": 0,
        "http_backoff_max_s": 0,
        "http_timeout_s": 5,
    }


def trading_days(start: date, end: date) -> list[date]:
    """Weekdays between two dates inclusive (good enough for fakes)."""
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current = date.fromordinal(current.toordinal() + 1)
    return days


class FakePriceProvider(PriceProvider):
    """Deterministic price provider; can be told to fail or hit its quota."""

    def __init__(self, name="fake", metered=False, fail_for=(), quota_after=None, error=None):
        self.name = name
        self.metered = metered
        self.fail_for = set(fail_for)
        self.quota_after = quota_after
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    def fetch_eod(self, ticker, date_from, date_to):
        from finsight.common.exceptions import ProviderError, QuotaExceededError

        if self.quota_after is not None and len(self.calls) >= self.quota_after:
            raise QuotaExceededError("usage limit", provider=self.name)
        self.calls.append((ticker, date_from, date_to))
        if ticker in self.fail_for:
            raise self.error or ProviderError("boom", provider=self.name, retryable=False)
        return [
            PriceBar(ticker, to_utc_midnight(d), 10, 11, 9, 10.5 + i, 1000, 10.4 + i, self.name)
            for i, d in enumerate(trading_days(date_from, date_to))
        ]


class FakeNewsProvider(NewsProvider):
    """Deterministic news provider: two articles per window."""

    name = "fakenews"

    def __init__(self, error=None):
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    def fetch_news(self, ticker, date_from, date_to):
        self.calls.append((ticker, date_from, date_to))
        if self.error:
            raise self.error
        published = datetime(date_from.year, date_from.month, date_from.day, 12, tzinfo=UTC)
        return [
            NewsArticle(
                ticker,
                f"{ticker}-{date_from}-{i}",
                f"Headline {i}",
                "summary",
                "src",
                f"https://x/{ticker}/{date_from}/{i}",
                published,
                "fakenews",
            )
            for i in range(2)
        ]