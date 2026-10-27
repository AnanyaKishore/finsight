"""HTTP API of the scheduler (internal: reached by the backend and Alertmanager).

=====================  ======================================================
``GET  /health``        liveness (compose healthcheck)
``GET  /status``        busy?, the run in progress, next daily run, last run
``POST /run``           start a run: ``{"kind": "daily"|"retrain", "reason"}``
``POST /retrain``       start a retrain; accepts an Alertmanager webhook body
``GET  /metrics``       Prometheus metrics (pipeline health, Day 8)
=====================  ======================================================

``POST`` requests need the shared token from ``SCHEDULER_TOKEN``, as
``X-Finsight-Token: <token>`` or ``Authorization: Bearer <token>`` (the form
Alertmanager sends). Without a configured token, starting runs over HTTP is
disabled. Answers: ``202`` started, ``409`` a run is in progress, ``401`` bad
token, ``422`` bad request. Alertmanager also posts when an alert *resolves*;
``/retrain`` answers those with ``200 ignored`` and starts nothing.
"""

from __future__ import annotations

import hmac
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from finsight.scheduler.service import Busy, SchedulerService


class RunRequest(BaseModel):
    """``POST /run`` body."""

    kind: Literal["daily", "retrain"] = "daily"
    reason: str = Field(default="", max_length=200)
    trigger: Literal["console", "api"] = "api"


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message, **extra}}
    )


def create_app(service: SchedulerService, token: str | None) -> FastAPI:
    """Build the API around a running :class:`SchedulerService`."""
    app = FastAPI(title="FinSight scheduler", docs_url="/docs")

    def authorised(request: Request) -> JSONResponse | None:
        if not token:
            return _error(
                503,
                "trigger_disabled",
                "SCHEDULER_TOKEN is not set; runs can only start on schedule",
            )
        given = request.headers.get("X-Finsight-Token") or ""
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            given = given or auth[7:]
        if not hmac.compare_digest(given.encode(), token.encode()):
            return _error(401, "unauthorised", "missing or wrong scheduler token")
        return None

    def start(kind: str, trigger: str, reason: str) -> JSONResponse:
        try:
            run_id = service.start(kind, trigger, reason)
        except Busy as exc:
            return _error(409, "busy", "a pipeline run is already in progress", run_id=exc.run_id)
        return JSONResponse(
            status_code=202, content={"run_id": run_id, "kind": kind, "trigger": trigger}
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "busy": service.busy}

    @app.get("/status")
    def status() -> Any:
        return jsonable_encoder(service.status())

    @app.post("/run", status_code=202)
    def run(body: RunRequest, request: Request) -> JSONResponse:
        if denied := authorised(request):
            return denied
        return start(body.kind, body.trigger, body.reason)

    @app.post("/retrain", status_code=202)
    async def retrain(request: Request) -> JSONResponse:
        if denied := authorised(request):
            return denied
        try:
            body = await request.json()
        except ValueError:
            body = {}
        if isinstance(body, dict) and body.get("status") == "resolved":
            return JSONResponse({"ignored": "resolved notification"})
        alerts = body.get("alerts", []) if isinstance(body, dict) else []
        names = sorted(
            {a.get("labels", {}).get("alertname", "?") for a in alerts if isinstance(a, dict)}
        )
        reason = (
            ("alert: " + ", ".join(names))
            if names
            else str(body.get("reason", "") if isinstance(body, dict) else "")
        )
        return start("retrain", "webhook", reason[:200])

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(service.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
