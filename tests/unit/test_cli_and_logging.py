"""CLI wiring and structured-logging tests (Mongo replaced by mongomock)."""

from __future__ import annotations

import json
import logging

import mongomock
import pytest

from finsight.common import config as config_module
from finsight.common.logging import JsonFormatter, bind_run_id
from finsight.ingestion import cli


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Point the CLI at the real repo config, a fake Mongo and fake API keys."""
    from tests.conftest import PROJECT_ROOT

    monkeypatch.setenv("FINSIGHT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv("MARKETSTACK_API_KEY", "test-key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    config_module.get_settings.cache_clear()
    client = mongomock.MongoClient(tz_aware=True)
    monkeypatch.setattr(cli, "get_client", lambda settings: client)
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)
    yield client
    config_module.get_settings.cache_clear()


def test_dry_run_reports_cost_without_calls(cli_env, capsys):
    code = cli.main(["backfill", "--dry-run"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "10 price calls" in out
    assert "'marketstack': 10" in out
    assert cli_env["finsight"]["pipeline_runs"].count_documents({}) == 0


def test_status_and_quota_commands(cli_env, capsys):
    assert cli.main(["quota", "--set-used", "7"]) == cli.EXIT_OK
    assert cli.main(["status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "used 7" in out
    assert "AAPL" in out


def test_build_service_without_keys_uses_fallback(cli_env, monkeypatch):
    monkeypatch.setenv("MARKETSTACK_API_KEY", "replace-me")
    monkeypatch.delenv("FINNHUB_API_KEY")
    config_module.get_settings.cache_clear()
    service = cli.build_service(config_module.get_settings(), cli_env["finsight"])
    assert service.primary is None
    assert service.fallback is not None and service.fallback.name == "yfinance"
    assert service.news is None


def test_json_formatter_includes_context_and_run_id():
    record = logging.makeLogRecord(
        {"name": "t", "levelname": "INFO", "msg": "hello", "ticker": "AAPL"}
    )
    with bind_run_id("abc123"):
        payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello"
    assert payload["ticker"] == "AAPL"
    assert payload["run_id"] == "abc123"
