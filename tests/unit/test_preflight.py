"""The pre-demo check's pure helpers (scripts/preflight.py)."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts" / "preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_runs_needed_for_the_ac11_target():
    assert preflight.runs_needed(4, 5) == 5  # 9/10 = 90 %
    assert preflight.runs_needed(9, 10) == 0
    assert preflight.runs_needed(5, 5) == 0
    assert preflight.runs_needed(0, 0) == 1
    assert preflight.runs_needed(8, 10) == 10  # 18/20


def test_compose_ps_both_output_formats():
    rows = (
        '{"Service": "backend", "State": "running", "Health": "healthy"}\n'
        '{"Service": "mongo", "State": "running", "Health": ""}'
    )
    assert [r["Service"] for r in preflight.parse_compose_ps(rows)] == ["backend", "mongo"]
    array = '[{"Service": "backend", "State": "running", "Health": ""}]'
    assert preflight.parse_compose_ps(array)[0]["Service"] == "backend"
    assert preflight.parse_compose_ps("") == []


def test_container_problems_lists_missing_stopped_and_unhealthy():
    ps = [
        {"Service": "backend", "State": "running", "Health": "healthy"},
        {"Service": "frontend", "State": "running", "Health": ""},
        {"Service": "mongo", "State": "running", "Health": "unhealthy"},
        {"Service": "grafana", "State": "exited", "Health": ""},
    ]
    services = ["backend", "frontend", "mongo", "grafana", "mlflow"]
    problems = preflight.container_problems(services, ps)
    assert problems == ["mongo: unhealthy", "grafana: exited", "mlflow: not created"]


def test_any_fail_fails_the_preflight():
    assert preflight.overall([("a", ("PASS", "")), ("b", ("WARN", ""))]) == 0
    assert preflight.overall([("a", ("PASS", "")), ("b", ("FAIL", ""))]) == 1


def _fake_get(responses):
    def get(url, **params):
        for suffix, body in responses.items():
            if url.endswith(suffix):
                return body
        raise AssertionError(url)

    return get


def test_last_run_reports_ac11_progress(monkeypatch):
    now = preflight.datetime(2026, 11, 1, 12, tzinfo=preflight.UTC)
    runs = [
        {"run_id": "r6", "kind": "daily", "trigger": "console", "status": "success",
         "started_at": "2026-11-01T10:00:00Z"},
        *[{"run_id": f"r{i}", "kind": "daily", "trigger": "schedule", "status": "success",
           "started_at": f"2026-10-{27 + i}T01:00:00Z"} for i in range(1, 5)],
        {"run_id": "r0", "kind": "daily", "trigger": "schedule", "status": "failed",
         "started_at": "2026-10-27T13:21:00Z"},
    ]  # fmt: skip
    monkeypatch.setattr(preflight, "get", _fake_get({"/api/pipeline/runs": {"runs": runs}}))
    status, message = preflight.check_last_run(now)
    assert status == "PASS"
    assert "r6" in message and "AC-11 4/5" in message and "needs 5 more" in message


def test_an_active_drill_fails_and_info_alerts_are_ignored(monkeypatch):
    body = {"report": {"drill": {"until": "x"}}, "alerts": []}
    monkeypatch.setattr(preflight, "get", _fake_get({"/api/monitoring": body}))
    assert preflight.check_monitoring()[0] == "FAIL"
    body = {
        "report": {"drill": None, "freshness": {"max_lag": 0}},
        "alerts": [{"name": "SentimentShift", "severity": "info"}],
    }
    monkeypatch.setattr(preflight, "get", _fake_get({"/api/monitoring": body}))
    assert preflight.check_monitoring()[0] == "PASS"
    body["alerts"].append({"name": "DataStale", "severity": "warning"})
    assert preflight.check_monitoring() == ("WARN", "alerts firing: DataStale")


def test_scheduler_warns_when_the_daily_run_is_close(monkeypatch):
    now = preflight.datetime(2026, 11, 3, 0, 30, tzinfo=preflight.UTC)
    body = {"reachable": True, "busy": False, "triggers_enabled": True,
            "next_daily_run": "2026-11-03T01:00:00Z"}  # fmt: skip
    monkeypatch.setattr(preflight, "get", _fake_get({"/api/pipeline/status": body}))
    assert preflight.check_scheduler(now)[0] == "WARN"
    body["next_daily_run"] = "2026-11-04T01:00:00Z"
    assert preflight.check_scheduler(now)[0] == "PASS"


def test_next_run_is_shown_in_local_time(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    now = preflight.datetime(2026, 11, 1, 12, tzinfo=preflight.UTC)
    body = {"reachable": True, "busy": False, "triggers_enabled": True,
            "next_daily_run": "2026-11-03T06:30:00+05:30"}  # fmt: skip
    monkeypatch.setattr(preflight, "get", _fake_get({"/api/pipeline/status": body}))
    assert preflight.check_scheduler(now) == ("PASS", "idle; next daily run Tue 03 Nov 06:30 IST")
    monkeypatch.delenv("TZ")
    time.tzset()
