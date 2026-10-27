"""Tests for the Pipeline Console endpoints (/api/pipeline/*, Day 7)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import requests

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from finsight.api.app import create_app  # noqa: E402
from finsight.api.config import ApiConfig  # noqa: E402
from finsight.api.pipeline import SchedulerClient, SchedulerUnavailable  # noqa: E402
from finsight.common.config import get_settings  # noqa: E402
from finsight.common.mongo import PIPELINE_EXECUTIONS, PIPELINE_RUNS  # noqa: E402

NOW = datetime(2026, 10, 27, 3, 0, tzinfo=UTC)


class FakeScheduler:
    """Stands in for SchedulerClient."""

    def __init__(self, up=True, token="t" * 32, answer=(202, {"run_id": "new1", "kind": "daily"})):
        self.up, self.token, self.answer, self.triggered = up, token, answer, []

    def status(self):
        if not self.up:
            raise SchedulerUnavailable("connection refused")
        return {
            "busy": False,
            "running": None,
            "next_daily_run": "2026-10-28T06:30:00+05:30",
            "schedule": {"time": "06:30", "days": ["tue", "wed"], "timezone": "Asia/Kolkata"},
        }

    def trigger(self, kind, reason):
        if not self.up:
            raise SchedulerUnavailable("connection refused")
        self.triggered.append((kind, reason))
        return self.answer


def _step(name, status, **extra):
    return {
        "name": name,
        "label": name.title(),
        "status": status,
        "critical": name == "data",
        **extra,
    }


def _seed(db):
    runs = db[PIPELINE_EXECUTIONS]
    runs.insert_many(
        [
            {
                "run_id": "r1",
                "kind": "daily",
                "trigger": "schedule",
                "status": "success",
                "started_at": NOW - timedelta(days=2),
                "duration_s": 600.0,
                "steps": [_step("ingest", "success", log_tail=["ok"]), _step("data", "success")],
                "summary": {"etl_rows_per_s": 800.0, "scoring_articles_per_s": 2.5},
            },
            {
                "run_id": "r2",
                "kind": "daily",
                "trigger": "console",
                "status": "failed",
                "started_at": NOW - timedelta(days=1),
                "duration_s": 120.0,
                "steps": [
                    _step("ingest", "success"),
                    _step(
                        "data",
                        "failed",
                        error="exit code 1: validation failed",
                        log_tail=["a", "b"],
                    ),
                ],
                "summary": {},
            },
            {
                "run_id": "r3",
                "kind": "retrain",
                "trigger": "webhook",
                "reason": "alert: LstmMapeDrift",
                "status": "partial",
                "started_at": NOW - timedelta(hours=3),
                "duration_s": 2400.0,
                "steps": [_step("retrain", "success"), _step("forecasts", "partial")],
                "summary": {"etl_rows_per_s": 900.0},
            },
        ]
    )
    db[PIPELINE_RUNS].insert_many(
        [
            {
                "run_id": "s1",
                "pipeline": "ingestion",
                "pipeline_run_id": "r2",
                "status": "success",
                "started_at": NOW - timedelta(days=1),
                "prices": {"fetched": 10},
                "news": {"fetched": 30},
                "throughput_rows_per_s": 4.0,
            },
            {
                "run_id": "s2",
                "pipeline": "dvc",
                "stage": "validate",
                "pipeline_run_id": "r2",
                "status": "failed",
                "started_at": NOW - timedelta(days=1),
                "error": "3 checks failed",
                "metrics": {},
            },
            {
                "run_id": "s3",
                "pipeline": "dvc",
                "stage": "process",
                "status": "success",
                "started_at": NOW - timedelta(days=2),
                "duration_s": 95.0,
                "metrics": {"rows_in": 2500},
            },
        ]
    )
    db.pipeline_dag.insert_one(
        {
            "_id": "dvc",
            "nodes": ["validate", "process", "train_lstm"],
            "edges": [["validate", "process"], ["process", "train_lstm"]],
            "stale": ["train_lstm"],
            "generated_at": NOW,
        }
    )


@pytest.fixture
def client_for(db):
    def build(scheduler=None, seed=True):
        if seed:
            _seed(db)
        scheduler = scheduler or FakeScheduler()
        app = create_app(
            settings=get_settings(),
            config=ApiConfig(frontend_origins=("http://localhost:8080",)),
            db=db,
            registry=lambda name: None,
            clock=lambda: NOW,
            scheduler=scheduler,
        )
        return TestClient(app), scheduler

    return build


def test_runs_newest_first_with_filters_and_no_logs(client_for):
    client, _ = client_for()
    runs = client.get("/api/pipeline/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == ["r3", "r2", "r1"]
    assert runs[1]["failed_steps"] == ["data"] and "steps" not in runs[1]
    failed = client.get("/api/pipeline/runs", params={"status": "failed"}).json()["runs"]
    assert [r["run_id"] for r in failed] == ["r2"]
    retrains = client.get("/api/pipeline/runs", params={"kind": "retrain"}).json()["runs"]
    assert [r["run_id"] for r in retrains] == ["r3"]
    assert client.get("/api/pipeline/runs", params={"status": "nope"}).status_code == 422


def test_run_detail_has_steps_logs_and_stage_records(client_for):
    client, _ = client_for()
    detail = client.get("/api/pipeline/runs/r2").json()
    assert detail["steps"][1]["log_tail"] == ["a", "b"]
    assert detail["steps"][1]["error"].startswith("exit code 1")
    stages = {s["stage"]: s for s in detail["stages"]}
    assert stages["ingest"]["metrics"]["articles_added"] == 30
    assert stages["validate"]["error"] == "3 checks failed"
    missing = client.get("/api/pipeline/runs/zzz")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "unknown_run"


def test_dag_colours_stages_by_their_latest_record(client_for):
    client, _ = client_for()
    dag = client.get("/api/pipeline/dag").json()
    nodes = {n["name"]: n for n in dag["nodes"]}
    assert (
        nodes["validate"]["status"] == "failed" and nodes["validate"]["error"] == "3 checks failed"
    )
    assert nodes["process"]["status"] == "success" and nodes["process"]["duration_s"] == 95.0
    assert nodes["train_lstm"]["status"] == "never" and nodes["train_lstm"]["stale"] is True
    assert ["validate", "process"] in dag["edges"]


def test_dag_missing_is_a_clear_404(client_for):
    client, _ = client_for(seed=False)
    response = client.get("/api/pipeline/dag")
    assert response.status_code == 404 and response.json()["error"]["code"] == "no_dag"


def test_stats_success_rate_durations_and_throughput(client_for):
    client, _ = client_for()
    stats = client.get("/api/pipeline/stats", params={"days": 30}).json()
    assert stats["runs"] == 3 and stats["by_status"] == {"success": 1, "failed": 1, "partial": 1}
    assert stats["success_rate"] == pytest.approx(0.667, abs=1e-3)
    assert stats["median_duration_s"] == {"daily": 600.0, "retrain": 2400.0}
    # Newest value of each measure: r3 for ETL, r1 for scoring.
    assert stats["latest_throughput"] == {"etl_rows_per_s": 900.0, "scoring_articles_per_s": 2.5}
    # r2 started exactly 24 h ago (inclusive) and r3 three hours ago.
    assert client.get("/api/pipeline/stats", params={"days": 1}).json()["runs"] == 2


def test_status_proxies_the_scheduler_and_falls_back_to_the_database(client_for, db):
    client, _ = client_for()
    status = client.get("/api/pipeline/status").json()
    assert (
        status["reachable"] and status["triggers_enabled"] and status["schedule"]["time"] == "06:30"
    )

    db[PIPELINE_EXECUTIONS].insert_one(
        {
            "run_id": "live",
            "kind": "daily",
            "trigger": "schedule",
            "status": "running",
            "started_at": NOW,
            "current_step": "data",
            "steps": [],
        }
    )
    down, _ = client_for(FakeScheduler(up=False), seed=False)
    status = down.get("/api/pipeline/status").json()
    assert not status["reachable"] and status["busy"] and status["current_step"] == "data"
    assert not status["triggers_enabled"] and "unreachable" in status["detail"]


def test_trigger_forwards_and_maps_errors(client_for):
    client, scheduler = client_for()
    ok = client.post("/api/pipeline/trigger", json={"kind": "retrain"})
    assert ok.status_code == 202 and ok.json() == {"run_id": "new1", "kind": "daily"}
    assert scheduler.triggered == [("retrain", "started from the console")]

    busy, _ = client_for(FakeScheduler(answer=(409, {"error": {"run_id": "r9"}})), seed=False)
    response = busy.post("/api/pipeline/trigger", json={"kind": "daily"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "pipeline_busy"

    down, _ = client_for(FakeScheduler(up=False), seed=False)
    response = down.post("/api/pipeline/trigger", json={"kind": "daily"})
    assert (
        response.status_code == 503 and response.json()["error"]["code"] == "scheduler_unavailable"
    )
    assert client.post("/api/pipeline/trigger", json={"kind": "weekly"}).status_code == 422


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Http:
    def __init__(self):
        self.posts = []

    def get(self, url, timeout):
        raise requests.ConnectionError("refused")

    def post(self, url, json, headers, timeout):
        self.posts.append((url, json, headers))
        return _Resp(202, {"run_id": "x", "kind": json["kind"]})


def test_scheduler_client_sends_the_token_and_refuses_without_one():
    http = _Http()
    client = SchedulerClient("http://scheduler:8001/", "s" * 32, http=http)
    assert client.trigger("daily", "why") == (202, {"run_id": "x", "kind": "daily"})
    url, body, headers = http.posts[0]
    assert url == "http://scheduler:8001/run" and body["trigger"] == "console"
    assert headers == {"X-Finsight-Token": "s" * 32}
    with pytest.raises(SchedulerUnavailable):
        client.status()
    with pytest.raises(SchedulerUnavailable, match="SCHEDULER_TOKEN"):
        SchedulerClient("http://s", None, http=http).trigger("daily", "")
