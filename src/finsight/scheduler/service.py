"""The scheduler: one run at a time, a background loop, and Prometheus metrics.

* :meth:`SchedulerService.start` launches a run in a worker thread and returns
  its ``run_id`` at once (the HTTP API answers ``202 Accepted``). A second
  request while a run is in progress raises :class:`Busy` (HTTP ``409``).
* :meth:`SchedulerService.tick` is called every ``check_interval_s`` by the
  loop and starts the daily run when :meth:`Schedule.is_due` says so.
* Every step is written to MongoDB as it starts and finishes, so the console
  shows live progress; afterwards the DVC DAG snapshot is refreshed.
* Day 8: :meth:`SchedulerService.refresh_monitoring` computes the drift / live
  error report every ``monitoring.interval_s`` and after each run, stores it
  and exports it with the pipeline metrics; webhook retrains respect a cooldown.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from prometheus_client import CollectorRegistry, Counter, Gauge

from finsight.common.logging import get_logger
from finsight.common.mongo import MONITORING_REPORTS, PIPELINE_RUNS
from finsight.common.runs import git_commit
from finsight.monitoring.metrics import MonitorMetrics
from finsight.monitoring.monitor import build_report, summarise
from finsight.scheduler.dag import build_dag
from finsight.scheduler.plan import Step, overall_status, plan_for
from finsight.scheduler.runner import StepResult, run_command
from finsight.scheduler.schedule import Schedule
from finsight.scheduler.store import ExecutionStore
from finsight.scheduler.versioning import commit_pipeline_outputs, dvc_push

log = get_logger(__name__)

STATUS_CODE = {"success": 1, "partial": 0.5, "failed": 0, "interrupted": 0}


class Busy(Exception):  # noqa: N818 -- reads naturally: `except Busy`
    """A run is already in progress."""

    def __init__(self, run_id: str | None) -> None:
        """Remember which run is blocking."""
        super().__init__(f"run {run_id} is in progress")
        self.run_id = run_id


class SchedulerMetrics:
    """Prometheus metrics for pipeline health (Day 8 alerts on these)."""

    def __init__(self) -> None:
        """Create the metric families on a private registry."""
        self.registry = CollectorRegistry()
        self.runs = Counter(
            "finsight_pipeline_runs_total",
            "Finished pipeline runs by kind and status",
            ["kind", "status"],
            registry=self.registry,
        )
        self.last_status = Gauge(
            "finsight_pipeline_last_run_status",
            "Last run outcome: 1 success, 0.5 partial, 0 failed",
            ["kind"],
            registry=self.registry,
        )
        self.last_finished = Gauge(
            "finsight_pipeline_last_run_timestamp_seconds",
            "Unix time the last run of each kind finished",
            ["kind"],
            registry=self.registry,
        )
        self.duration = Gauge(
            "finsight_pipeline_last_run_duration_seconds",
            "Duration of the last run of each kind",
            ["kind"],
            registry=self.registry,
        )
        self.step_duration = Gauge(
            "finsight_pipeline_step_duration_seconds",
            "Duration of each step in the last run that included it",
            ["step"],
            registry=self.registry,
        )
        self.step_failures = Counter(
            "finsight_pipeline_step_failures_total",
            "Failed steps",
            ["step"],
            registry=self.registry,
        )
        self.running = Gauge(
            "finsight_pipeline_running", "1 while a run is in progress", registry=self.registry
        )
        self.throughput = Gauge(
            "finsight_pipeline_throughput",
            "Throughput of the last run (ETL rows/s, articles scored/s, ingested rows/s)",
            ["measure"],
            registry=self.registry,
        )
        self.retrains = Counter(
            "finsight_retrains_triggered_total",
            "Retrain runs started, by trigger (console, webhook, api)",
            ["trigger"],
            registry=self.registry,
        )
        self.retrains_skipped = Counter(
            "finsight_retrains_skipped_total",
            "Webhook retrains ignored because of the cooldown",
            registry=self.registry,
        )
        self.monitor = MonitorMetrics(self.registry)
        # Start every counter series at 0: Prometheus' increase() cannot count
        # a series' first sample, so a first retrain would otherwise show as 0.
        for kind in ("daily", "retrain"):
            for status in STATUS_CODE:
                self.runs.labels(kind, status)
        for trigger in ("console", "webhook", "api"):
            self.retrains.labels(trigger)


class SchedulerService:
    """Runs plans, records them, and decides when the daily run is due."""

    def __init__(  # noqa: PLR0913 -- every dependency injectable for tests
        self,
        *,
        root: Path,
        params: dict[str, Any],
        store: ExecutionStore,
        tickers: Sequence[str],
        backend_url: str,
        lstm_url: str,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        command_runner: Callable[..., StepResult] = run_command,
        dag_builder: Callable[..., dict[str, Any]] = build_dag,
        http: Any = requests,
        sleep: Callable[[float], None] = time.sleep,
        monitoring: dict[str, Any] | None = None,
        monitor_builder: Callable[..., dict[str, Any]] = build_report,
    ) -> None:
        """Create the service; nothing runs until :meth:`start` or :meth:`tick`."""
        self.root, self.params, self.store = root, params, store
        self.schedule = Schedule.from_params(params)
        self.tickers = list(tickers)
        self.backend_url, self.lstm_url = backend_url.rstrip("/"), lstm_url.rstrip("/")
        self.env = dict(env if env is not None else os.environ)
        self.clock, self.command_runner, self.dag_builder = clock, command_runner, dag_builder
        self.http, self.sleep = http, sleep
        self.monitoring = monitoring or {}
        self.monitor_builder = monitor_builder
        self.metrics = SchedulerMetrics()
        self._lock = threading.Lock()
        self._monitor_lock = threading.Lock()
        self._current: str | None = None
        self._queued_retrain: str | None = None  # reason of a webhook retrain waiting for a run
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------- control
    @property
    def busy(self) -> bool:
        """Whether a run is in progress."""
        return self._current is not None

    def recover(self) -> int:
        """At start-up, close runs a previous process left as ``running``.

        Also seeds the "last run" gauges from MongoDB, so a scheduler restart
        does not blank the pipeline alerts until the next run.
        """
        closed = self.store.mark_interrupted(self.clock())
        if closed:
            log.warning("closed interrupted runs", extra={"count": closed})
        for kind in ("daily", "retrain"):
            last = self.store.last_finished(kind)
            if last and last.get("finished_at"):
                finished = last["finished_at"]
                finished = finished if finished.tzinfo else finished.replace(tzinfo=UTC)
                self.metrics.last_status.labels(kind).set(STATUS_CODE.get(last["status"], 0))
                self.metrics.last_finished.labels(kind).set(finished.timestamp())
                if last.get("duration_s") is not None:
                    self.metrics.duration.labels(kind).set(last["duration_s"])
        return closed

    def retrain_cooldown_left(self) -> timedelta | None:
        """Time until a webhook may start another retrain (``None`` = allowed now)."""
        hours = float(self.monitoring.get("retrain_cooldown_h", 0) or 0)
        last = self.store.last_started("retrain")
        if not hours or last is None:
            return None
        left = last + timedelta(hours=hours) - self.clock()
        return left if left > timedelta(0) else None

    def start(self, kind: str, trigger: str, reason: str = "", wait: bool = False) -> str:
        """Start a run in a worker thread and return its ``run_id``.

        Raises:
            Busy: If a run is already in progress.
            ValueError: For an unknown run kind.
        """
        steps = plan_for(kind)
        if not self._lock.acquire(blocking=False):
            raise Busy(self._current)
        run_id = uuid.uuid4().hex[:12]
        self._current = run_id
        try:
            now = self.clock()
            self.store.create(
                {
                    "run_id": run_id,
                    "kind": kind,
                    "trigger": trigger,
                    "reason": reason[:200],
                    "status": "running",
                    "started_at": now,
                    "finished_at": None,
                    "duration_s": None,
                    "git_commit": git_commit(),
                    "current_step": steps[0].name,
                    "steps": [
                        {
                            "name": s.name,
                            "label": s.label,
                            "status": "pending",
                            "critical": s.critical,
                        }
                        for s in steps
                    ],
                    "summary": {},
                }
            )
        except Exception:
            self._current = None
            self._lock.release()
            raise
        self.metrics.running.set(1)
        if kind == "retrain":
            self.metrics.retrains.labels(trigger).inc()
        log.info("run started", extra={"run_id": run_id, "kind": kind, "trigger": trigger})
        thread = threading.Thread(
            target=self._execute, args=(run_id, kind, steps), name=f"run-{run_id}", daemon=True
        )
        self._threads.append(thread)
        thread.start()
        if wait:
            thread.join()
        return run_id

    def queue_retrain(self, reason: str) -> str | None:
        """Remember a webhook retrain that arrived during another run.

        Alertmanager does not retry a refused webhook and repeats a retrain
        alert only once a day, so a ``409`` would lose it. The loop starts the
        queued retrain as soon as the current run ends. Returns the ``run_id``
        of the run it waits for, or ``None`` when that run is itself a retrain
        (nothing is queued: the retrain it asks for is already happening).
        """
        running = self.store.get(self._current) if self._current else None
        if running and running.get("kind") == "retrain":
            return None
        self._queued_retrain = reason[:200] or "queued alert"
        log.info("retrain queued", extra={"after_run_id": self._current, "reason": reason})
        return self._current

    @property
    def queued_retrain(self) -> str | None:
        """Reason of the retrain waiting for the current run, if any."""
        return self._queued_retrain

    def tick(self) -> str | None:
        """Start a queued retrain, or the daily run if it is due; return its ``run_id``."""
        if self.busy:
            return None
        if self._queued_retrain is not None:
            reason, self._queued_retrain = self._queued_retrain, None
            try:
                return self.start("retrain", "webhook", reason)
            except Busy:
                self._queued_retrain = reason
                return None
        now = self.clock()
        if not self.schedule.is_due(now, self.store.last_daily_start()):
            return None
        trigger = "catch-up" if self.schedule.is_catch_up(now) else "schedule"
        try:
            return self.start("daily", trigger)
        except Busy:
            return None

    def loop(self, stop: threading.Event) -> None:
        """Check every ``check_interval_s`` whether the daily run is due."""
        interval = float(self.params.get("check_interval_s", 60))
        while not stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 -- the loop must survive a bad tick
                log.exception("scheduler tick failed")
            stop.wait(interval)

    def status(self) -> dict[str, Any]:
        """Current state for ``/status`` and the console header."""
        now = self.clock()
        running = self.store.get(self._current) if self._current else None
        return {
            "busy": self.busy,
            "running": running,
            "queued_retrain": self._queued_retrain,
            "next_daily_run": self.schedule.next_run(now, self.store.last_daily_start()),
            "last_run": self.store.last_finished(),
            "schedule": {
                "time": self.schedule.at.strftime("%H:%M"),
                "days": [
                    d
                    for i, d in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun"))
                    if i in self.schedule.days
                ],
                "timezone": str(self.schedule.tz),
            },
        }

    # ------------------------------------------------------------ execution
    def _execute(self, run_id: str, kind: str, steps: Sequence[Step]) -> None:
        started = time.monotonic()
        statuses: list[str] = []
        critical_failed = False
        context: dict[str, Any] = {"lstm_version_before": self._lstm_production_version()}
        try:
            for index, step in enumerate(steps):
                if critical_failed:
                    self.store.set_step(
                        run_id,
                        index,
                        self._step_doc(
                            step, StepResult("skipped", detail="an earlier step failed")
                        ),
                    )
                    continue
                begun = self.clock()
                self.store.set_step(
                    run_id,
                    index,
                    {
                        "name": step.name,
                        "label": step.label,
                        "status": "running",
                        "critical": step.critical,
                        "started_at": begun,
                    },
                )
                try:
                    result = self._run_step(run_id, step, context)
                except Exception as exc:  # noqa: BLE001 -- a step error never kills the scheduler
                    log.exception("step crashed", extra={"step": step.name})
                    result = StepResult("failed", error=f"{exc.__class__.__name__}: {exc}"[:500])
                doc = self._step_doc(step, result) | {
                    "started_at": begun,
                    "finished_at": self.clock(),
                }
                self.store.set_step(run_id, index, doc)
                statuses.append(result.status)
                self.metrics.step_duration.labels(step.name).set(result.duration_s)
                if result.status == "failed":
                    self.metrics.step_failures.labels(step.name).inc()
                    critical_failed = step.critical
                log.info(
                    "step finished",
                    extra={
                        "run_id": run_id,
                        "step": step.name,
                        "status": result.status,
                        "duration_s": result.duration_s,
                    },
                )
            status = overall_status(statuses, critical_failed)
        except Exception as exc:  # noqa: BLE001
            log.exception("run crashed")
            status = "failed"
            self.store.update(run_id, {"error": f"{exc.__class__.__name__}: {exc}"[:500]})
        finally:
            duration = round(time.monotonic() - started, 2)
            summary = self._summary(run_id, context)
            self.store.update(
                run_id,
                {
                    "status": status,
                    "finished_at": self.clock(),
                    "duration_s": duration,
                    "current_step": None,
                    "summary": summary,
                },
            )
            self._record_metrics(kind, status, duration, summary)
            self._refresh_dag()
            log.info(
                "run finished", extra={"run_id": run_id, "status": status, "duration_s": duration}
            )
            self._current = None
            self.metrics.running.set(0)
            self._lock.release()

    def _step_doc(self, step: Step, result: StepResult) -> dict[str, Any]:
        return {
            "name": step.name,
            "label": step.label,
            "critical": step.critical,
            "status": result.status,
            "duration_s": result.duration_s,
            "exit_code": result.exit_code,
            "detail": result.detail,
            "error": result.error,
            "log_tail": (
                result.log_tail if result.status in {"failed", "partial"} else result.log_tail[-5:]
            ),
            "metrics": result.metrics,
        }

    def _run_step(self, run_id: str, step: Step, context: dict[str, Any]) -> StepResult:
        timeout = float(self.params.get("timeouts_s", {}).get(step.name, 3600))
        if step.type == "command":
            env = self.env | {"FINSIGHT_PIPELINE_RUN_ID": run_id}
            return self.command_runner(
                step.argv,
                cwd=self.root,
                env=env,
                timeout_s=timeout,
                log_path=self.root / "logs" / "scheduler" / f"{run_id}.log",
                tail_lines=int(self.params.get("log_tail_lines", 40)),
                partial_exit_codes=step.partial_exit_codes,
            )
        if step.type == "forecasts":
            return self._forecasts(context)
        if step.type == "monitor":
            return self._monitor()
        if step.type == "version":
            return self._version(run_id)
        raise ValueError(f"unknown step type {step.type}")

    # ------------------------------------------------------------ forecasts
    def _lstm_production_version(self) -> str | None:
        path = self.root / "reports" / "promotion.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            return str(report["lstm"]["production"]["version"])
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def _wait_for_reload(self, timeout_s: float = 420) -> str:
        """After a promotion, wait until lstm-serve has restarted with the new model.

        The serving container's watcher restarts it within ~60 s of a change of
        ``@production``; we wait for it to go down and come back, then for the
        backend's 60 s registry cache to expire.
        """
        deadline = time.monotonic() + timeout_s
        went_down = False
        while time.monotonic() < deadline:
            try:
                up = self.http.get(f"{self.lstm_url}/ping", timeout=3).status_code == 200
            except requests.RequestException:
                up = False
            if not up:
                went_down = True
            elif went_down:
                self.sleep(65)
                return "lstm-serve reloaded the new production model"
            self.sleep(5)
        return "lstm-serve did not restart in time; forecasts may use the previous model"

    def _forecasts(self, context: dict[str, Any]) -> StepResult:
        started = time.monotonic()
        notes = []
        after = self._lstm_production_version()
        if context.get("lstm_version_before") and after and after != context["lstm_version_before"]:
            notes.append(f"LSTM promoted v{context['lstm_version_before']} -> v{after}")
            notes.append(self._wait_for_reload())
        counts = {"fresh": 0, "stored": 0, "stale": 0, "failed": 0}
        errors = []
        for ticker in self.tickers:
            try:
                response = self.http.get(f"{self.backend_url}/api/prediction/{ticker}", timeout=60)
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                body = response.json()
                counts["stale" if body.get("stale") else "fresh"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                errors.append(f"{ticker}: {exc}"[:120])
        done = counts["fresh"] + counts["stale"]
        if counts["failed"] == len(self.tickers):
            status = "failed"
        elif counts["failed"] or counts["stale"]:
            status = "partial"
        else:
            status = "success"
        notes.append(f"{counts['fresh']} fresh, {counts['stale']} stale, {counts['failed']} failed")
        return StepResult(
            status,
            round(time.monotonic() - started, 2),
            detail="; ".join(notes),
            error="; ".join(errors[:5]) or None,
            metrics={"forecasts": done, **counts},
        )

    # ----------------------------------------------------------- monitoring
    def refresh_monitoring(self) -> dict[str, Any] | None:
        """Compute, store and export the monitoring report (one at a time)."""
        if not self.monitoring:
            return None
        with self._monitor_lock:
            report = self.monitor_builder(
                self.store.db, self.root, self.monitoring, now=self.clock()
            )
            try:
                self.store.db[MONITORING_REPORTS].insert_one(dict(report))
            except Exception as exc:  # noqa: BLE001 -- metrics still update
                log.warning("could not store monitoring report", extra={"error": _short(exc)})
            self.metrics.monitor.update(report)
            return report

    def monitor_loop(self, stop: threading.Event) -> None:
        """Refresh the monitoring report every ``monitoring.interval_s``."""
        interval = float(self.monitoring.get("interval_s", 120))
        while not stop.is_set():
            try:
                self.refresh_monitoring()
            except Exception:  # noqa: BLE001 -- keep monitoring through bad data
                log.exception("monitoring refresh failed")
            stop.wait(interval)

    def _monitor(self) -> StepResult:
        started = time.monotonic()
        report = self.refresh_monitoring()
        if report is None:
            return StepResult("skipped", detail="monitoring not configured")
        failed = [
            k for k, v in report.items() if isinstance(v, dict) and v.get("status") == "error"
        ]
        alerts = [
            name
            for name, section in (("returns drift", "returns"), ("sentiment shift", "sentiment"),
                                  ("forecast error", "forecast"))  # fmt: skip
            if (report.get(section) or {}).get("detected")
        ]
        detail = summarise(report) + (f"; ALERT: {', '.join(alerts)}" if alerts else "")
        return StepResult(
            "partial" if failed else "success",
            round(time.monotonic() - started, 2),
            detail=detail[:500],
            error=f"sections failed: {', '.join(failed)}" if failed else None,
            metrics={
                "returns_drifted_share": (report.get("returns") or {}).get("drifted_share"),
                "live_mape": (report.get("forecast") or {}).get("mape"),
            },
        )

    # ------------------------------------------------------------- version
    def _version(self, run_id: str) -> StepResult:
        started = time.monotonic()
        notes, failed = [], []
        if self.params.get("dvc_push", True):
            try:
                notes.append("dvc push: " + dvc_push(self.root, self.env))
            except Exception as exc:  # noqa: BLE001
                failed.append(f"dvc push failed: {_short(exc)}")
        if self.params.get("git_commit", True):
            day = self.schedule.local(self.clock()).date().isoformat()
            message = f"chore(pipeline): scheduler run {day} [{run_id}]"
            try:
                notes.append("git: " + commit_pipeline_outputs(self.root, self.env, message))
            except Exception as exc:  # noqa: BLE001
                failed.append(f"git commit failed: {_short(exc)}")
        return StepResult(
            "partial" if failed else "success",
            round(time.monotonic() - started, 2),
            detail="; ".join(notes),
            error="; ".join(failed) or None,
        )

    # -------------------------------------------------------------- summary
    def _summary(self, run_id: str, context: dict[str, Any]) -> dict[str, Any]:
        """Headline numbers from the stage records this run produced."""
        summary: dict[str, Any] = {}
        try:
            records = list(
                self.store.db[PIPELINE_RUNS].find({"pipeline_run_id": run_id}, {"_id": 0})
            )
        except Exception:  # noqa: BLE001
            return summary
        summary["stages_run"] = sum(1 for r in records if r.get("pipeline") == "dvc")
        for record in records:
            metrics = record.get("metrics") or {}
            if record.get("pipeline") == "ingestion":
                summary["prices_added"] = (record.get("prices") or {}).get("fetched")
                summary["articles_added"] = (record.get("news") or {}).get("fetched")
                summary["ingest_rows_per_s"] = record.get("throughput_rows_per_s")
            elif record.get("stage") == "process":
                summary["etl_rows"] = metrics.get("rows_in")
                summary["etl_rows_per_s"] = metrics.get("rows_per_s")
            elif record.get("stage") == "score_sentiment":
                summary["articles_scored"] = metrics.get("scored_this_run")
                summary["scoring_articles_per_s"] = metrics.get("throughput_articles_per_s")
        after = self._lstm_production_version()
        if after and context.get("lstm_version_before") and after != context["lstm_version_before"]:
            summary["lstm_promoted"] = f"v{context['lstm_version_before']} -> v{after}"
        return {k: v for k, v in summary.items() if v is not None}

    def _record_metrics(
        self, kind: str, status: str, duration: float, summary: dict[str, Any]
    ) -> None:
        self.metrics.runs.labels(kind, status).inc()
        self.metrics.last_status.labels(kind).set(STATUS_CODE.get(status, 0))
        self.metrics.last_finished.labels(kind).set(self.clock().timestamp())
        self.metrics.duration.labels(kind).set(duration)
        for measure, key in (
            ("etl_rows_per_s", "etl_rows_per_s"),
            ("scoring_articles_per_s", "scoring_articles_per_s"),
            ("ingest_rows_per_s", "ingest_rows_per_s"),
        ):
            if summary.get(key) is not None:
                self.metrics.throughput.labels(measure).set(float(summary[key]))

    def _refresh_dag(self) -> None:
        try:
            self.store.save_dag(self.dag_builder(self.root, self.env))
        except Exception as exc:  # noqa: BLE001 -- the diagram is a nice-to-have
            log.warning("DAG refresh failed", extra={"error": _short(exc)})

    def refresh_dag_async(self) -> None:
        """Refresh the DAG snapshot in the background (at start-up)."""
        threading.Thread(target=self._refresh_dag, name="dag", daemon=True).start()


def _short(exc: BaseException) -> str:
    stderr = getattr(exc, "stderr", None)
    text = (stderr or str(exc)).strip().splitlines()
    return (text[-1] if text else exc.__class__.__name__)[:300]
