"""Unit tests for the quota ledger, trading-calendar helpers and settings loading."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from finsight.common.config import load_settings
from finsight.common.exceptions import ConfigError, QuotaExceededError
from finsight.common.timeutils import latest_expected_session, week_start
from finsight.ingestion.quota import QuotaLedger


# ------------------------------------------------------------------ quota ledger
def test_ledger_reserves_until_usable_budget(db):
    clock = lambda: datetime(2026, 10, 21, tzinfo=UTC)  # noqa: E731
    ledger = QuotaLedger(db, "marketstack", budget=5, reserve=1, clock=clock)
    for _ in range(4):
        ledger.reserve_calls(1)
    assert ledger.used() == 4 and ledger.remaining() == 0
    with pytest.raises(QuotaExceededError):
        ledger.reserve_calls(1)
    ledger.refund(1)
    assert ledger.remaining() == 1


def test_ledger_mark_exhausted_and_set_used(db):
    ledger = QuotaLedger(
        db, "marketstack", budget=100, reserve=5, clock=lambda: datetime(2026, 10, 21, tzinfo=UTC)
    )
    ledger.reserve_calls(3)
    ledger.mark_exhausted()
    assert ledger.used() == 100 and ledger.remaining() == 0
    ledger.set_used(12)
    assert ledger.used() == 12


def test_ledger_resets_each_month(db):
    now = {"t": datetime(2026, 10, 31, tzinfo=UTC)}
    ledger = QuotaLedger(db, "marketstack", budget=2, clock=lambda: now["t"])
    ledger.reserve_calls(2)
    assert ledger.remaining() == 0
    now["t"] = datetime(2026, 11, 1, tzinfo=UTC)
    assert ledger.remaining() == 2


# ------------------------------------------------------------- trading calendar
@pytest.mark.parametrize(
    ("now_utc", "expected"),
    [
        (datetime(2026, 10, 21, 23, 0, tzinfo=UTC), date(2026, 10, 21)),  # Wed after close
        (datetime(2026, 10, 21, 15, 0, tzinfo=UTC), date(2026, 10, 20)),  # Wed before close
        (datetime(2026, 10, 25, 12, 0, tzinfo=UTC), date(2026, 10, 23)),  # Sunday -> Friday
        (datetime(2026, 10, 26, 10, 0, tzinfo=UTC), date(2026, 10, 23)),  # Mon morning -> Fri
        (datetime(2026, 11, 27, 1, 0, tzinfo=UTC), date(2026, 11, 25)),  # after Thanksgiving
    ],
)
def test_latest_expected_session(now_utc, expected):
    assert latest_expected_session(now_utc) == expected


def test_week_start_is_monday_utc():
    assert week_start(datetime(2026, 10, 22, 15, tzinfo=UTC)) == datetime(2026, 10, 19, tzinfo=UTC)


# ---------------------------------------------------------------------- settings
def _write_project(tmp_path, tickers_yaml):
    (tmp_path / "configs").mkdir()
    (tmp_path / "params.yaml").write_text("ingestion:\n  price_backfill_days: 10\n")
    (tmp_path / "configs" / "tickers.yaml").write_text(tickers_yaml)
    return tmp_path


def test_load_settings_and_uri_escaping(tmp_path, monkeypatch):
    root = _write_project(tmp_path, "tickers:\n  - {symbol: aapl, name: Apple}\n")
    monkeypatch.setenv("MONGO_ROOT_USER", "user")
    monkeypatch.setenv("MONGO_ROOT_PASSWORD", "p@ss/word")
    monkeypatch.setenv("MARKETSTACK_API_KEY", "replace-me")
    settings = load_settings(root)
    assert settings.symbols == ["AAPL"]
    assert "p%40ss%2Fword" in settings.mongo_uri
    assert settings.section("ingestion")["price_backfill_days"] == 10
    with pytest.raises(ConfigError):
        settings.require("marketstack_api_key")  # placeholder is rejected
    with pytest.raises(ConfigError):
        settings.section("training")


def test_duplicate_tickers_rejected(tmp_path):
    root = _write_project(
        tmp_path, "tickers:\n  - {symbol: A, name: a}\n  - {symbol: a, name: b}\n"
    )
    with pytest.raises(ConfigError):
        load_settings(root)


def test_repository_params_file_is_valid():
    """The real params.yaml / tickers.yaml must always load (guards config typos)."""
    from tests.conftest import PROJECT_ROOT

    settings = load_settings(PROJECT_ROOT)
    assert len(settings.symbols) == 10
    assert settings.section("ingestion")["price_backfill_days"] >= 200
