"""Pipeline Console endpoints (Day 7).

Read side: run history, run detail and the DAG come straight from MongoDB
(``pipeline_executions``, ``pipeline_runs``, ``pipeline_dag``), so the console
keeps working when the scheduler container is down. Write side:
``POST /api/pipeline/trigger`` forwards to the scheduler's ``/run`` with the
shared ``SCHEDULER_TOKEN``; the browser never sees the token.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pymongo import DESCENDING

from finsight.api import schemas as s
from finsight.common.mongo import PIPELINE_DAG, PIPELINE_EXECUTIONS, PIPELINE_RUNS

LIST_FIELDS = {"_id": 0, "steps.log_tail": 0}


class SchedulerUnavailable(Exception):  # noqa: N818 -- like ModelUnavailable
    """The scheduler could not be reached or refused the request."""


class SchedulerClient:
    """Talks to the scheduler service over the compose network."""

    def __init__(
        self, base_url: str, token: str | None, http: Any = requests, timeout_s: float = 5
    ) -> None:
        """``token`` is ``SCHEDULER_TOKEN``; without it triggers are disabled."""
        self.base_url, self.token, self.http, self.timeout_s = (
            base_url.rstrip("/"),
            token,
            http,
            timeout_s,
        )

    def status(self) -> dict[str, Any]:
        """The scheduler's ``/status`` document."""
        try:
            response = self.http.get(f"{self.base_url}/status", timeout=self.timeout_s)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise SchedulerUnavailable(str(exc)) from exc

    def trigger(self, kind: str, reason: str) -> tuple[int, dict[str, Any]]:
        """Ask for a run; returns the scheduler's status code and body."""
        if not self.token:
            raise SchedulerUnavailable("SCHEDULER_TOKEN is not configured for the backend")
        try:
            response = self.http.post(
                f"{self.base_url}/run",
                json={"kind": kind, "reason": reason, "trigger": "console"},
                headers={"X-Finsight-Token": self.token},
                timeout=self.timeout_s,
            )
            return response.status_code, response.json()
        except (requests.RequestException, ValueError) as exc:
            raise SchedulerUnavailable(str(exc)) from exc


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    doc = dict(doc)
    doc["failed_steps"] = [
        st["name"] for st in doc.get("steps", []) if st.get("status") == "failed"
    ]
    doc.pop("steps", None)
    return doc


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def build_router(
    db: Any,
    scheduler: SchedulerClient,
    error: Callable[[int, str, str], Exception],
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> APIRouter:
    """Routes under ``/api/pipeline``; ``error`` builds the app's ``ApiError``."""
    router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])
    executions, stage_runs = db[PIPELINE_EXECUTIONS], db[PIPELINE_RUNS]

    @router.get("/status", response_model=s.SchedulerStatus)
    def status() -> s.SchedulerStatus:
        """Scheduler state; falls back to the database when the scheduler is down."""
        try:
            st = scheduler.status()
        except SchedulerUnavailable as exc:
            running = executions.find_one(
                {"status": "running"}, {"_id": 0, "run_id": 1, "current_step": 1}
            )
            return s.SchedulerStatus(
                reachable=False,
                busy=bool(running),
                running_run_id=(running or {}).get("run_id"),
                current_step=(running or {}).get("current_step"),
                triggers_enabled=False,
                detail=f"scheduler unreachable: {str(exc)[:120]}",
            )
        run = st.get("running") or {}
        return s.SchedulerStatus(
            reachable=True,
            busy=bool(st.get("busy")),
            running_run_id=run.get("run_id"),
            current_step=run.get("current_step"),
            next_daily_run=st.get("next_daily_run"),
            schedule=st.get("schedule") or {},
            triggers_enabled=bool(scheduler.token),
        )

    @router.get("/runs", response_model=s.PipelineRunList)
    def runs(
        limit: int = Query(30, ge=1, le=200),
        status: s.RunStatus | None = None,
        kind: str | None = Query(None, pattern="^(daily|retrain)$"),
    ) -> s.PipelineRunList:
        """Newest runs first."""
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if kind:
            query["kind"] = kind
        docs = executions.find(query, LIST_FIELDS).sort("started_at", DESCENDING).limit(limit)
        return s.PipelineRunList(runs=[s.PipelineRunSummary(**_summary(d)) for d in docs])

    @router.get("/runs/{run_id}", response_model=s.PipelineRunDetail)
    def run_detail(run_id: str) -> s.PipelineRunDetail:
        """Steps (with log tails for failures) and the stage records of one run."""
        doc = executions.find_one({"run_id": run_id}, {"_id": 0})
        if doc is None:
            raise error(404, "unknown_run", f"no pipeline run {run_id!r}")
        stages = []
        for rec in stage_runs.find({"pipeline_run_id": run_id}, {"_id": 0}).sort("started_at", 1):
            is_ingest = rec.get("pipeline") == "ingestion"
            stages.append(
                s.StageRecord(
                    stage="ingest" if is_ingest else rec.get("stage", "?"),
                    status=rec.get("status", "?"),
                    started_at=rec.get("started_at"),
                    duration_s=rec.get("duration_s"),
                    metrics=(
                        {
                            "prices_added": (rec.get("prices") or {}).get("fetched"),
                            "articles_added": (rec.get("news") or {}).get("fetched"),
                            "rows_per_s": rec.get("throughput_rows_per_s"),
                        }
                        if is_ingest
                        else rec.get("metrics") or {}
                    ),
                    error=rec.get("error") if isinstance(rec.get("error"), str) else None,
                    git_commit=rec.get("git_commit"),
                )
            )
        summary = _summary(doc)
        return s.PipelineRunDetail(**summary, steps=doc.get("steps", []), stages=stages)

    @router.get("/dag", response_model=s.DagResponse)
    def dag() -> s.DagResponse:
        """DVC stages and edges, each stage with its latest recorded execution."""
        snapshot = db[PIPELINE_DAG].find_one({"_id": "dvc"})
        if not snapshot:
            raise error(
                404, "no_dag", "no DAG recorded yet: the scheduler stores it when it starts"
            )
        latest: dict[str, dict[str, Any]] = {}
        names = set(snapshot["nodes"])
        for rec in (
            stage_runs.find({"pipeline": "dvc", "stage": {"$in": list(names)}}, {"_id": 0})
            .sort("started_at", DESCENDING)
            .limit(2000)
        ):
            latest.setdefault(rec["stage"], rec)
        stale = set(snapshot.get("stale", []))
        nodes = []
        for name in snapshot["nodes"]:
            rec = latest.get(name)
            nodes.append(
                s.DagNode(
                    name=name,
                    status=(
                        (rec or {}).get("status", "never")
                        if (rec or {}).get("status") in {"success", "failed", "running"}
                        else "never"
                    ),
                    stale=name in stale,
                    last_run_at=(rec or {}).get("started_at"),
                    duration_s=(rec or {}).get("duration_s"),
                    error=(rec or {}).get("error"),
                    metrics=(rec or {}).get("metrics") or {},
                )
            )
        return s.DagResponse(
            nodes=nodes,
            edges=[(a, b) for a, b in snapshot["edges"]],
            generated_at=snapshot["generated_at"],
        )

    @router.get("/stats", response_model=s.PipelineStats)
    def stats(days: int = Query(30, ge=1, le=365)) -> s.PipelineStats:
        """Success rate, typical duration and latest throughput."""
        since = clock() - timedelta(days=days)
        docs = list(
            executions.find(
                {"started_at": {"$gte": since}, "status": {"$ne": "running"}}, LIST_FIELDS
            )
        )
        by_status: dict[str, int] = {}
        durations: dict[str, list[float]] = {}
        for d in docs:
            by_status[d["status"]] = by_status.get(d["status"], 0) + 1
            if d.get("duration_s") is not None and d["status"] in {"success", "partial"}:
                durations.setdefault(d["kind"], []).append(float(d["duration_s"]))
        good = by_status.get("success", 0) + by_status.get("partial", 0)
        latest_throughput: dict[str, float] = {}
        for d in sorted(docs, key=lambda x: _aware(x["started_at"]), reverse=True):
            for key in ("etl_rows_per_s", "scoring_articles_per_s", "ingest_rows_per_s"):
                value = (d.get("summary") or {}).get(key)
                if value is not None and key not in latest_throughput:
                    latest_throughput[key] = float(value)
        return s.PipelineStats(
            days=days,
            runs=len(docs),
            by_status=by_status,
            success_rate=round(good / len(docs), 3) if docs else None,
            median_duration_s={k: round(statistics.median(v), 1) for k, v in durations.items()},
            latest_throughput=latest_throughput,
        )

    @router.post(
        "/trigger",
        response_model=s.TriggerResponse,
        status_code=202,
        responses={409: {"model": s.ErrorResponse}, 503: {"model": s.ErrorResponse}},
    )
    def trigger(body: s.TriggerRequest) -> Any:
        """Start a daily run or a retrain now (forwarded to the scheduler)."""
        try:
            code, payload = scheduler.trigger(body.kind, body.reason)
        except SchedulerUnavailable as exc:
            raise error(503, "scheduler_unavailable", str(exc)[:200]) from exc
        if code == 409:
            raise error(
                409, "pipeline_busy", f"run {payload.get('error', {}).get('run_id')} is in progress"
            )
        if code != 202:
            message = (payload.get("error") or {}).get("message", f"scheduler answered HTTP {code}")
            raise error(503, "scheduler_unavailable", message)
        return JSONResponse(
            status_code=202, content={"run_id": payload["run_id"], "kind": payload["kind"]}
        )

    return router
