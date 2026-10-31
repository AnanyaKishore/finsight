"""Tests for the scheduler (Day 7): schedule rules, steps, runner, service, API."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finsight.common.config import usable_secret
from finsight.common.mongo import PIPELINE_EXECUTIONS, PIPELINE_RUNS
from finsight.scheduler.app import create_app
from finsight.scheduler.dag import parse_dot, stale_stages
from finsight.scheduler.plan import PLANS, overall_status, plan_for
from finsight.scheduler.runner import StepResult, run_command
from finsight.scheduler.schedule import Schedule
from finsight.scheduler.service import Busy, SchedulerService
from finsight.scheduler.store import ExecutionStore
from finsight.scheduler.versioning import commit_pipeline_outputs

PARAMS = {
    "timezone": "Asia/Kolkata",
    "daily_time": "06:30",
    "run_days": ["tue", "wed", "thu", "fri", "sat"],
    "check_interval_s": 60,
    "log_tail_lines": 5,
    "git_commit": False,
    "dvc_push": False,
    "timeouts_s": {"ingest": 5},
}
IST = Schedule.from_params(PARAMS).tz


def ist(*args: int) -> datetime:
    return datetime(*args, tzinfo=IST)


TUESDAY_7AM = ist(2026, 10, 27, 7, 0)


# ------------------------------------------------------------------ schedule
def test_daily_run_is_due_after_the_slot_on_run_days_only():
    sched = Schedule.from_params(PARAMS)
    assert not sched.is_due(ist(2026, 10, 27, 6, 29), None)  # Tuesday, before 06:30
    assert sched.is_due(ist(2026, 10, 27, 6, 30), None)
    assert not sched.is_due(ist(2026, 10, 26, 9, 0), None)  # Monday: no US session to catch
    assert not sched.is_due(ist(2026, 11, 1, 9, 0), None)  # Sunday


def test_never_two_daily_runs_on_one_day_but_catch_up_after_sleep():
    sched = Schedule.from_params(PARAMS)
    ran_today = ist(2026, 10, 27, 6, 31)
    assert not sched.is_due(ist(2026, 10, 27, 18, 0), ran_today)
    # Laptop asleep all Wednesday morning: the run starts when it wakes at 11:05.
    wake = ist(2026, 10, 28, 11, 5)
    assert sched.is_due(wake, ran_today)
    assert sched.is_catch_up(wake) and not sched.is_catch_up(ist(2026, 10, 28, 6, 35))
    # Timestamps stored in UTC compare by the local (IST) calendar day.
    assert not sched.is_due(ist(2026, 10, 28, 6, 40), datetime(2026, 10, 28, 1, 5, tzinfo=UTC))


def test_next_run_skips_sunday_and_monday():
    sched = Schedule.from_params(PARAMS)
    saturday_done = ist(2026, 10, 31, 6, 30)
    assert sched.next_run(ist(2026, 10, 31, 12, 0), saturday_done) == ist(2026, 11, 3, 6, 30)
    assert sched.next_run(ist(2026, 10, 27, 5, 0), None) == ist(2026, 10, 27, 6, 30)


def test_schedule_rejects_empty_days():
    with pytest.raises(ValueError, match="at least one day"):
        Schedule.from_params({**PARAMS, "run_days": []})


# ---------------------------------------------------------------------- plan
def test_daily_plan_never_retrains_and_retrain_runs_everything():
    daily = [s.argv for s in plan_for("daily") if s.type == "command"]
    assert ("dvc", "repro") not in daily
    assert all("--single-item" in argv for argv in daily if "score_sentiment" in argv)
    assert ("dvc", "repro") in [s.argv for s in plan_for("retrain")]
    assert [s.name for s in PLANS["daily"]][-3:] == ["forecasts", "monitor", "version"]
    with pytest.raises(ValueError, match="unknown run kind"):
        plan_for("weekly")


def test_overall_status():
    assert overall_status(["success", "success"], False) == "success"
    assert overall_status(["success", "partial"], False) == "partial"
    assert overall_status(["success", "failed"], False) == "partial"
    assert overall_status(["failed"], True) == "failed"


# -------------------------------------------------------------------- runner
def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_runner_success_partial_failure_and_log(tmp_path):
    log = tmp_path / "run.log"
    kw = {"cwd": tmp_path, "env": {"PATH": "/usr/bin:/bin"}, "timeout_s": 30, "log_path": log}
    ok = run_command(_py("print('hello')"), **kw)
    assert ok.status == "success" and ok.exit_code == 0 and ok.log_tail == ["hello"]
    partial = run_command(_py("import sys; sys.exit(3)"), partial_exit_codes=frozenset({3}), **kw)
    assert partial.status == "partial"
    bad = run_command(_py("print('x'); print('boom'); raise SystemExit(2)"), tail_lines=1, **kw)
    assert bad.status == "failed" and "exit code 2: boom" in bad.error and bad.log_tail == ["boom"]
    missing = run_command(["no-such-binary-xyz"], **kw)
    assert missing.status == "failed" and "could not start" in missing.error
    assert "hello" in log.read_text() and "$ " in log.read_text()


def test_runner_timeout_kills_the_process_tree(tmp_path):
    result = run_command(
        _py("import time; print('started', flush=True); time.sleep(60)"),
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        timeout_s=1,
        log_path=tmp_path / "t.log",
    )
    assert result.status == "failed" and "time limit" in result.error
    assert result.duration_s < 20


# ----------------------------------------------------------------------- DAG
DOT = """strict digraph  {
"unit_tests";
"export";
"process";
"score_sentiment";
"unit_tests" -> "export";
"export" -> "process";
"process" -> "score_sentiment";
}
"""


def test_parse_dot_and_stale_stages():
    nodes, edges = parse_dot(DOT)
    assert nodes == ["unit_tests", "export", "process", "score_sentiment"]
    assert ["export", "process"] in edges and len(edges) == 3
    status = {"export": ["always changed"], "process": [{"changed deps": {"data/raw": "modified"}}]}
    assert stale_stages(status) == ["process"]


# ------------------------------------------------------------------- service
class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code, self._body = status_code, body or {}

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self, stale=()):
        self.calls, self.stale = [], set(stale)

    def get(self, url, timeout):
        self.calls.append(url)
        ticker = url.rsplit("/", 1)[-1]
        if ticker == "DOWN":
            return FakeResponse(503)
        return FakeResponse(200, {"stale": ticker in self.stale})


class FakeRunner:
    """Records commands; outcomes per step name (keyed by the argv's last word)."""

    def __init__(self, outcomes=None, gate=None):
        self.outcomes, self.calls, self.gate = outcomes or {}, [], gate

    def __call__(self, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs["env"].get("FINSIGHT_PIPELINE_RUN_ID")))
        if self.gate is not None:
            self.gate.wait(5)
        key = argv[-1]
        return self.outcomes.get(key, StepResult("success", 1.5, 0, log_tail=["ok"]))


@pytest.fixture
def make_service(db, tmp_path):
    def build(runner=None, http=None, now=TUESDAY_7AM, tickers=("AAPL", "MSFT"), **extra):
        clock = {"now": now}
        service = SchedulerService(
            root=tmp_path,
            params=PARAMS,
            store=ExecutionStore(db),
            tickers=list(tickers),
            backend_url="http://backend:8000",
            lstm_url="http://lstm-serve:5001",
            env={"PATH": "/usr/bin:/bin"},
            clock=lambda: clock["now"],
            command_runner=runner or FakeRunner(),
            dag_builder=lambda root, env: {
                "nodes": ["a", "b"],
                "edges": [["a", "b"]],
                "stale": [],
                "generated_at": clock["now"],
            },
            http=http or FakeHttp(),
            sleep=lambda s: None,
            **extra,
        )
        return service, clock

    return build


def test_daily_run_records_every_step_and_the_dag(make_service, db):
    runner = FakeRunner()
    service, _ = make_service(runner=runner)
    run_id = service.start("daily", "console", "test", wait=True)
    doc = db[PIPELINE_EXECUTIONS].find_one({"run_id": run_id})
    assert doc["status"] == "success" and doc["current_step"] is None
    # monitor is "skipped" here: this service has no monitoring section configured
    assert [s["status"] for s in doc["steps"]] == ["success"] * 5 + ["skipped", "success"]
    assert {rid for _, rid in runner.calls} == {run_id}  # stage records can be grouped
    assert doc["steps"][4]["metrics"]["fresh"] == 2
    assert db.pipeline_dag.find_one({"_id": "dvc"})["edges"] == [["a", "b"]]
    assert service.metrics.runs.labels("daily", "success")._value.get() == 1
    assert not service.busy


def test_critical_failure_skips_the_rest(make_service, db):
    failing = StepResult(
        "failed", 2.0, 1, error="exit code 1: validation failed", log_tail=["x"] * 5
    )
    service, _ = make_service(runner=FakeRunner({"process": failing}))
    run_id = service.start("daily", "console", wait=True)
    doc = db[PIPELINE_EXECUTIONS].find_one({"run_id": run_id})
    statuses = {s["name"]: s["status"] for s in doc["steps"]}
    assert doc["status"] == "failed" and statuses["data"] == "failed"
    assert statuses["score_sentiment"] == statuses["forecasts"] == "skipped"
    assert doc["steps"][1]["log_tail"] == ["x"] * 5  # full tail kept for failures


def test_non_critical_problems_make_a_partial_run(make_service, db):
    partial_ingest = StepResult("partial", 1.0, 3, detail="finished with some errors")
    http = FakeHttp(stale={"MSFT"})
    service, _ = make_service(runner=FakeRunner({"incremental": partial_ingest}), http=http)
    run_id = service.start("daily", "schedule", wait=True)
    doc = db[PIPELINE_EXECUTIONS].find_one({"run_id": run_id})
    assert doc["status"] == "partial"
    assert doc["steps"][4]["status"] == "partial" and "1 stale" in doc["steps"][4]["detail"]


def test_only_one_run_at_a_time(make_service):
    gate = threading.Event()
    service, _ = make_service(runner=FakeRunner(gate=gate))
    first = service.start("daily", "console")
    with pytest.raises(Busy) as info:
        service.start("retrain", "console")
    assert info.value.run_id == first
    assert service.tick() is None  # the loop also waits
    gate.set()
    for thread in service._threads:
        thread.join(5)
    assert not service.busy


def test_tick_starts_once_per_day_and_marks_catch_up(make_service, db):
    service, clock = make_service(now=ist(2026, 10, 27, 6, 0))
    assert service.tick() is None  # before 06:30
    clock["now"] = ist(2026, 10, 27, 9, 0)  # woke up late
    run_id = service.tick()
    service._threads[-1].join(5)
    assert db[PIPELINE_EXECUTIONS].find_one({"run_id": run_id})["trigger"] == "catch-up"
    clock["now"] = ist(2026, 10, 27, 15, 0)
    assert service.tick() is None  # already ran today
    # A console run does not count as the day's scheduled run.
    db[PIPELINE_EXECUTIONS].delete_many({})
    service.start("daily", "console", wait=True)
    assert service.tick() is not None
    service._threads[-1].join(5)


def test_recover_closes_runs_left_running(make_service, db):
    db[PIPELINE_EXECUTIONS].insert_one(
        {
            "run_id": "old",
            "kind": "daily",
            "status": "running",
            "started_at": ist(2026, 10, 26, 6, 30),
        }
    )
    service, _ = make_service()
    assert service.recover() == 1
    assert db[PIPELINE_EXECUTIONS].find_one({"run_id": "old"})["status"] == "interrupted"


def test_summary_collects_stage_metrics(make_service, db):
    class Recording(FakeRunner):
        def __call__(self, argv, **kwargs):
            rid = kwargs["env"]["FINSIGHT_PIPELINE_RUN_ID"]
            if argv[-1] == "process":
                db[PIPELINE_RUNS].insert_one(
                    {
                        "run_id": "s1",
                        "pipeline": "dvc",
                        "stage": "process",
                        "pipeline_run_id": rid,
                        "metrics": {"rows_in": 2500, "rows_per_s": 812.5},
                    }
                )
            if argv[-1] == "incremental":
                db[PIPELINE_RUNS].insert_one(
                    {
                        "run_id": "s2",
                        "pipeline": "ingestion",
                        "pipeline_run_id": rid,
                        "prices": {"fetched": 10},
                        "news": {"fetched": 42},
                        "throughput_rows_per_s": 3.1,
                    }
                )
            return super().__call__(argv, **kwargs)

    service, _ = make_service(runner=Recording())
    run_id = service.start("daily", "console", wait=True)
    summary = db[PIPELINE_EXECUTIONS].find_one({"run_id": run_id})["summary"]
    assert summary == {
        "stages_run": 1,
        "etl_rows": 2500,
        "etl_rows_per_s": 812.5,
        "prices_added": 10,
        "articles_added": 42,
        "ingest_rows_per_s": 3.1,
    }
    assert service.metrics.throughput.labels("etl_rows_per_s")._value.get() == 812.5


def test_forecasts_wait_for_the_model_server_after_a_promotion(make_service, tmp_path):
    promotion = tmp_path / "reports" / "promotion.json"
    promotion.parent.mkdir()
    promotion.write_text(json.dumps({"lstm": {"production": {"version": "3"}}}))

    class Promoting(FakeRunner):
        def __call__(self, argv, **kwargs):
            if argv == ("dvc", "repro"):
                promotion.write_text(json.dumps({"lstm": {"production": {"version": "4"}}}))
            return super().__call__(argv, **kwargs)

    class Http(FakeHttp):
        pings = iter([200, 503, 503, 200])

        def get(self, url, timeout):
            if url.endswith("/ping"):
                return FakeResponse(next(self.pings))
            return super().get(url, timeout)

    service, _ = make_service(runner=Promoting(), http=Http())
    run_id = service.start("retrain", "webhook", "alert: LstmDrift", wait=True)
    doc = service.store.get(run_id)
    assert "promoted v3 -> v4" in doc["steps"][2]["detail"]
    assert "reloaded" in doc["steps"][2]["detail"]
    assert doc["summary"]["lstm_promoted"] == "v3 -> v4"


def test_all_forecasts_failing_fails_the_step(make_service, db):
    service, _ = make_service(tickers=("DOWN",))
    run_id = service.start("daily", "console", wait=True)
    doc = db[PIPELINE_EXECUTIONS].find_one({"run_id": run_id})
    assert doc["steps"][4]["status"] == "failed" and doc["status"] == "partial"


# ----------------------------------------------------------------------- API
TOKEN = "t" * 32


@pytest.fixture
def scheduler_api(make_service):
    gate = threading.Event()
    service, _ = make_service(runner=FakeRunner(gate=gate))
    yield TestClient(create_app(service, TOKEN)), service, gate
    gate.set()
    for thread in service._threads:
        thread.join(5)


def test_api_requires_the_token_and_answers_202_then_409(scheduler_api):
    client, service, gate = scheduler_api
    assert client.get("/health").json() == {"status": "ok", "busy": False}
    assert client.post("/run", json={"kind": "daily"}).status_code == 401
    wrong = client.post("/run", json={"kind": "daily"}, headers={"X-Finsight-Token": "nope"})
    assert wrong.status_code == 401
    ok = client.post("/run", json={"kind": "daily"}, headers={"X-Finsight-Token": TOKEN})
    assert ok.status_code == 202 and ok.json()["kind"] == "daily"
    busy = client.post(
        "/run", json={"kind": "retrain"}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert busy.status_code == 409 and busy.json()["error"]["run_id"] == ok.json()["run_id"]
    status = client.get("/status").json()
    assert status["busy"] and status["running"]["run_id"] == ok.json()["run_id"]
    assert status["schedule"]["days"] == ["tue", "wed", "thu", "fri", "sat"]
    assert (
        client.post(
            "/run", json={"kind": "weekly"}, headers={"X-Finsight-Token": TOKEN}
        ).status_code
        == 422
    )
    assert b"finsight_pipeline_running 1.0" in client.get("/metrics").content


def test_retrain_webhook_uses_alert_names_and_ignores_resolved(scheduler_api):
    client, service, _ = scheduler_api
    auth = {"Authorization": f"Bearer {TOKEN}"}
    resolved = client.post("/retrain", json={"status": "resolved", "alerts": []}, headers=auth)
    assert resolved.status_code == 200 and not service.busy
    body = {"status": "firing", "alerts": [{"labels": {"alertname": "LstmMapeDrift"}}]}
    started = client.post("/retrain", json=body, headers=auth)
    assert started.status_code == 202
    doc = service.store.get(started.json()["run_id"])
    assert doc["kind"] == "retrain" and doc["trigger"] == "webhook"
    assert doc["reason"] == "alert: LstmMapeDrift"


def test_retrain_alert_during_a_daily_run_is_queued_not_lost(scheduler_api):
    client, service, gate = scheduler_api
    auth = {"Authorization": f"Bearer {TOKEN}"}
    daily = client.post("/run", json={"kind": "daily"}, headers=auth).json()["run_id"]
    body = {"status": "firing", "alerts": [{"labels": {"alertname": "ReturnsDriftDetected"}}]}
    queued = client.post("/retrain", json=body, headers=auth)
    assert queued.status_code == 202 and queued.json() == {
        "queued": True,
        "after_run_id": daily,
        "reason": "alert: ReturnsDriftDetected",
    }
    assert client.get("/status").json()["queued_retrain"] == "alert: ReturnsDriftDetected"
    assert service.tick() is None  # still busy: nothing starts
    gate.set()
    for thread in list(service._threads):
        thread.join(5)
    run_id = service.tick()  # the loop's next check starts the queued retrain
    doc = service.store.get(run_id)
    assert doc["kind"] == "retrain" and doc["trigger"] == "webhook"
    assert doc["reason"] == "alert: ReturnsDriftDetected" and service.queued_retrain is None


def test_retrain_alert_during_a_retrain_is_ignored(scheduler_api):
    client, service, _ = scheduler_api
    auth = {"Authorization": f"Bearer {TOKEN}"}
    running = client.post("/run", json={"kind": "retrain"}, headers=auth).json()["run_id"]
    again = client.post("/retrain", json={"status": "firing", "alerts": []}, headers=auth)
    assert again.status_code == 200 and again.json()["run_id"] == running
    assert service.queued_retrain is None


def test_api_without_token_disables_triggers(make_service):
    service, _ = make_service()
    client = TestClient(create_app(service, None))
    response = client.post("/run", json={"kind": "daily"}, headers={"X-Finsight-Token": ""})
    assert response.status_code == 503 and response.json()["error"]["code"] == "trigger_disabled"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("replace-me", None), ("short", None), ("x" * 16, "x" * 16)],
)
def test_usable_secret_rejects_placeholders(value, expected):
    assert usable_secret(value) == expected


# ---------------------------------------------------------------- versioning
@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_commit_only_touches_dvc_lock_and_reports(tmp_path):
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True, text=True
        ).stdout  # noqa: S603,S607

    git("init", "-q", "-b", "main")
    (tmp_path / "dvc.lock").write_text("v1\n")
    (tmp_path / "notes.txt").write_text("mine\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert commit_pipeline_outputs(tmp_path, env, "noop") == "no changes to commit"

    (tmp_path / "dvc.lock").write_text("v2\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "m.json").write_text("{}\n")
    (tmp_path / "notes.txt").write_text("work in progress\n")
    git("add", "notes.txt")  # the developer's own staged change must stay staged
    result = commit_pipeline_outputs(tmp_path, env, "chore(pipeline): run")
    assert result.startswith("committed") and result.endswith("on main")
    assert git("show", "--name-only", "--format=", "HEAD").split() == ["dvc.lock", "reports/m.json"]
    assert git("diff", "--cached", "--name-only").split() == ["notes.txt"]

    (tmp_path / ".git" / "MERGE_HEAD").write_text("x")
    assert commit_pipeline_outputs(tmp_path, env, "m") == "skipped: a merge is in progress"


def test_commit_skips_outside_a_repository(tmp_path):
    assert commit_pipeline_outputs(Path(tmp_path), {}, "m") == "skipped: not a Git repository"


def test_store_last_daily_start_ignores_console_runs(db):
    store = ExecutionStore(db)
    base = datetime(2026, 10, 27, 1, 0, tzinfo=UTC)
    store.create(
        {
            "run_id": "a",
            "kind": "daily",
            "trigger": "schedule",
            "status": "success",
            "started_at": base,
        }
    )
    store.create(
        {
            "run_id": "b",
            "kind": "daily",
            "trigger": "console",
            "status": "success",
            "started_at": base + timedelta(hours=5),
        }
    )
    assert store.last_daily_start() == base
    assert store.last_finished()["run_id"] == "b"
