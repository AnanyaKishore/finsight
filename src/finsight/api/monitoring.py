"""``GET /api/monitoring`` (Day 8): latest drift / live-error report + active alerts.

The report is computed by the scheduler and stored in ``monitoring_reports``;
alerts come from Alertmanager's API. Either can be missing (not computed yet,
Alertmanager down) and the endpoint still answers, saying what is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests
from fastapi import APIRouter
from pymongo import DESCENDING

from finsight.api import schemas as s
from finsight.common.mongo import MONITORING_REPORTS


def fetch_alerts(base_url: str, http: Any = requests, timeout_s: float = 3) -> list[dict]:
    """Active, unsilenced alerts from Alertmanager (raises on failure)."""
    response = http.get(
        f"{base_url.rstrip('/')}/api/v2/alerts",
        params={"active": "true", "silenced": "false", "inhibited": "false"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    alerts = []
    for alert in response.json():
        labels, annotations = alert.get("labels", {}), alert.get("annotations", {})
        alerts.append(
            {
                "name": labels.get("alertname", "?"),
                "severity": labels.get("severity", "none"),
                "action": labels.get("action"),
                "summary": annotations.get("summary", ""),
                "starts_at": alert.get("startsAt"),
            }
        )
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(alerts, key=lambda a: (order.get(a["severity"], 3), a["name"]))


def build_router(
    db: Any,
    alertmanager_url: str,
    error: Callable[[int, str, str], Exception],
    alerts: Callable[[str], list[dict]] = fetch_alerts,
) -> APIRouter:
    """Routes under ``/api/monitoring``."""
    router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

    @router.get("", response_model=s.MonitoringResponse)
    def monitoring() -> s.MonitoringResponse:
        """The newest monitoring report and Alertmanager's active alerts."""
        report = db[MONITORING_REPORTS].find_one(
            {}, {"_id": 0}, sort=[("generated_at", DESCENDING)]
        )
        try:
            active, reachable, detail = alerts(alertmanager_url), True, ""
        except (requests.RequestException, ValueError) as exc:
            active, reachable, detail = [], False, f"alertmanager unreachable: {str(exc)[:120]}"
        if report is None and not reachable:
            raise error(404, "no_monitoring", "no monitoring report yet and Alertmanager is down")
        return s.MonitoringResponse(
            report=report, alerts=active, alertmanager_reachable=reachable, detail=detail
        )

    return router
