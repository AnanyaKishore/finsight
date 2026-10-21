"""Integration tests against the real MongoDB container.

Run after ``docker compose up -d mongo``:

    pytest tests/integration -m integration -q

They use a throw-away database (``finsight_it``) so real data is never touched,
and are skipped automatically when MongoDB is not reachable.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from finsight.common.config import load_settings
from finsight.common.exceptions import FinSightError
from finsight.common.mongo import PRICES, ensure_indexes, get_client
from finsight.ingestion.models import PriceBar
from finsight.ingestion.quota import QuotaLedger
from finsight.ingestion.repository import IngestionRepository
from tests.conftest import PROJECT_ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_db():
    settings = replace(load_settings(PROJECT_ROOT), mongo_db="finsight_it")
    try:
        client = get_client(settings, timeout_ms=2000)
    except FinSightError as exc:
        pytest.skip(f"MongoDB not reachable: {exc}")
    client.drop_database("finsight_it")
    database = client["finsight_it"]
    ensure_indexes(database)
    yield database
    client.drop_database("finsight_it")
    client.close()


def _bar(day: int, close: float) -> PriceBar:
    return PriceBar("AAPL", datetime(2026, 10, day, tzinfo=UTC), 1, 2, 0.5, close, 100, None, "it")


def test_upsert_is_idempotent_and_updates_revisions(real_db):
    repo = IngestionRepository(real_db)
    first = repo.upsert_prices([_bar(19, 10.0), _bar(20, 11.0)])
    again = repo.upsert_prices([_bar(19, 10.0), _bar(20, 11.5)])  # provider revised the 20th
    assert (first.inserted, again.inserted, again.updated) == (2, 0, 1)
    assert real_db[PRICES].count_documents({"ticker": "AAPL"}) == 2
    assert repo.last_price_date("AAPL") == datetime(2026, 10, 20, tzinfo=UTC)


def test_unique_index_exists(real_db):
    indexes = real_db[PRICES].index_information()
    assert any(spec.get("unique") for spec in indexes.values())


def test_quota_ledger_atomic_on_real_mongo(real_db):
    ledger = QuotaLedger(real_db, "it-provider", budget=3, reserve=0)
    for _ in range(3):
        ledger.reserve_calls(1)
    assert ledger.remaining() == 0


def test_coverage_aggregation(real_db):
    coverage = IngestionRepository(real_db).coverage()
    assert coverage["AAPL"]["prices"]["rows"] == 2
