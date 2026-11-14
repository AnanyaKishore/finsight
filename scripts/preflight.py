"""Pre-demo check (Day 11): is the running stack ready to be shown?

    python scripts/preflight.py        # or: make preflight

The e2e tests prove the stack *works*; this answers the question you have five
minutes before a demo or viva: is everything up, is nothing left over from a
drill, is the data current, is a run about to start in the middle of the demo,
and is there enough free memory? Every check is independent and prints PASS,
WARN or FAIL with what to do. Exit code 1 if anything FAILs.

It also says how many more successful scheduled runs AC-11 needs.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("FINSIGHT_API_URL", "http://localhost:8000").rstrip("/")
WEB = os.environ.get("FINSIGHT_WEB_URL", "http://localhost:8080").rstrip("/")
MLFLOW = os.environ.get("FINSIGHT_MLFLOW_URL", "http://localhost:5000").rstrip("/")
PROMETHEUS = os.environ.get("FINSIGHT_PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
ALERTMANAGER = os.environ.get("FINSIGHT_ALERTMANAGER_URL", "http://localhost:9093").rstrip("/")
GRAFANA = os.environ.get("FINSIGHT_GRAFANA_URL", "http://localhost:3000").rstrip("/")
TIMEOUT = 10
SCHEDULED_TRIGGERS = {"schedule", "catch-up"}  # same as scripts/test_report.py (AC-11)
AC11_TARGET = 0.9
DEMO_WINDOW = timedelta(hours=1)  # warn if a scheduled run starts within this window
MIN_FREE_MEMORY_GB = 1.5  # a retrain during the demo needs about this much
MIN_FREE_DISK_GB = 5.0

DRILL_STOP = "docker compose exec scheduler python -m finsight.monitoring.drill stop"

Result = tuple[str, str]  # (PASS | WARN | FAIL, message)


def get(url: str, **params: Any) -> Any:
    response = requests.get(url, params=params or None, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json() if "json" in response.headers.get("content-type", "") else response


# ---------------------------------------------------------------- pure helpers
def started(run: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))


def local(when: datetime) -> datetime:
    """A timezone-aware time in this machine's time zone, for display."""
    return when.astimezone()


def runs_needed(ok: int, total: int, target: float = AC11_TARGET) -> int:
    """Further successful runs until ``ok / total`` reaches ``target`` (0 if it already has)."""
    if not total:
        return 1
    if ok / total >= target:
        return 0
    return max(0, math.ceil(round((target * total - ok) / (1 - target), 9)))


def parse_compose_ps(output: str) -> list[dict[str, Any]]:
    """``docker compose ps --format json``: a JSON array (older Compose) or one object per line."""
    output = output.strip()
    if not output:
        return []
    if output.startswith("["):
        return json.loads(output)
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def container_problems(services: list[str], ps: list[dict[str, Any]]) -> list[str]:
    """Services that are not running, or running but not healthy."""
    state = {row.get("Service"): row for row in ps}
    problems = []
    for name in services:
        row = state.get(name)
        if row is None:
            problems.append(f"{name}: not created")
        elif row.get("State") != "running":
            problems.append(f"{name}: {row.get('State')}")
        elif row.get("Health") not in ("", None, "healthy"):
            problems.append(f"{name}: {row.get('Health')}")
    return problems


def overall(results: list[tuple[str, Result]]) -> int:
    return 1 if any(status == "FAIL" for _, (status, _) in results) else 0


# ---------------------------------------------------------------------- checks
def sh(*cmd: str, check: bool = True) -> str:
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=check)  # noqa: S603
    return done.stdout.strip()


def check_git() -> Result:
    dirty = sh("git", "status", "--porcelain")
    head = sh("git", "describe", "--tags", "--always", "--dirty", check=False)
    if dirty:
        return "WARN", f"{head}: {len(dirty.splitlines())} uncommitted change(s) (git status)"
    return "PASS", f"clean working tree at {head}"


def check_containers() -> Result:
    services = sh("docker", "compose", "config", "--services").split()
    ps = parse_compose_ps(sh("docker", "compose", "ps", "--format", "json"))
    problems = container_problems(services, ps)
    if problems:
        return "FAIL", "; ".join(problems) + "  →  docker compose up -d"
    return "PASS", f"all {len(services)} services running (healthy where checked)"


def check_health() -> Result:
    body = get(f"{API}/api/health")
    down = [name for name, dep in body["dependencies"].items() if not dep["ok"]]
    if down:
        return "FAIL", f"degraded: {', '.join(down)} down"
    return "PASS", f"backend ok ({len(body['dependencies'])} dependencies up)"


def check_portal() -> Result:
    get(f"{WEB}/")
    get(f"{WEB}/config.js")
    return "PASS", f"portal at {WEB}"


def check_models() -> Result:
    body = get(f"{API}/api/model-info")
    if not body["registry_reachable"]:
        return "FAIL", "MLflow registry not reachable"
    missing = [m["name"] for m in body["models"] if not m.get("version")]
    if missing:
        return "FAIL", f"no @production version: {', '.join(missing)}"
    return "PASS", ", ".join(f"{m['name']} v{m['version']}" for m in body["models"])


def check_scheduler(now: datetime) -> Result:
    body = get(f"{API}/api/pipeline/status")
    if not body["reachable"]:
        return "FAIL", "scheduler not reachable  →  docker compose up -d scheduler"
    if body["busy"]:
        return "WARN", f"a run is in progress ({body.get('current_step')}); wait for it to finish"
    if not body.get("triggers_enabled"):
        return "WARN", "Run now / Retrain now are disabled (SCHEDULER_TOKEN not set)"
    nxt = body.get("next_daily_run")
    if nxt:
        when = datetime.fromisoformat(nxt.replace("Z", "+00:00"))
        if when - now < DEMO_WINDOW:
            return "WARN", f"the daily run starts at {local(when):%H:%M %Z}, within the hour"
        return "PASS", f"idle; next daily run {local(when):%a %d %b %H:%M %Z}"
    return "PASS", "idle"


def check_last_run(now: datetime) -> Result:
    runs = get(f"{API}/api/pipeline/runs", limit=200)["runs"]
    finished = [r for r in runs if r["status"] != "running"]
    if not finished:
        return "WARN", "no pipeline runs yet"
    last = finished[0]
    since = now - timedelta(days=30)
    scheduled = [
        r for r in finished if r.get("trigger") in SCHEDULED_TRIGGERS and started(r) >= since
    ]
    ok = sum(r["status"] in {"success", "partial"} for r in scheduled)
    need = runs_needed(ok, len(scheduled))
    ac11 = f"AC-11 {ok}/{len(scheduled)}" + (
        " (passes)" if need == 0 else f", needs {need} more successful scheduled run(s)"
    )
    message = f"last run {last['run_id']} ({last['kind']}, {last['trigger']}): {last['status']}"
    message += f"; {ac11}"
    return ("PASS" if last["status"] in {"success", "partial"} else "WARN"), message


def check_monitoring() -> Result:
    body = get(f"{API}/api/monitoring")
    report = body.get("report") or {}
    if report.get("drill"):
        return "FAIL", f"a drift drill is active  →  {DRILL_STOP}"
    firing = [a for a in body.get("alerts", []) if a.get("severity") != "info"]
    lag = (report.get("freshness") or {}).get("max_lag")
    stale = f"; prices {lag} trading day(s) behind" if lag else ""
    if firing:
        names = ", ".join(sorted({a["name"] for a in firing}))
        return "WARN", f"alerts firing: {names}{stale}"
    if lag:
        return "WARN", f"no alerts firing{stale}"
    return "PASS", "no alerts firing, prices up to date, no drill active"


def check_ops_uis() -> Result:
    get(f"{MLFLOW}/health")
    get(f"{PROMETHEUS}/-/ready")
    get(f"{ALERTMANAGER}/-/ready")
    get(f"{GRAFANA}/api/health")
    targets = get(f"{PROMETHEUS}/api/v1/targets")["data"]["activeTargets"]
    down = [t["labels"].get("job", "?") for t in targets if t.get("health") != "up"]
    if down:
        return "FAIL", f"Prometheus targets down: {', '.join(sorted(set(down)))}"
    return "PASS", f"MLflow, Prometheus ({len(targets)} targets up), Alertmanager, Grafana"


def check_resources() -> Result:
    available = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) / 1024**2  # kB → GB
    except OSError:
        pass
    disk = shutil.disk_usage(ROOT).free / 1024**3
    text = f"{disk:.0f} GB disk free"
    if available is not None:
        text = f"{available:.1f} GB memory available, {text}"
    if available is not None and available < MIN_FREE_MEMORY_GB:
        return "WARN", text + "  →  close the browser tabs and apps you don't need"
    if disk < MIN_FREE_DISK_GB:
        return "WARN", text + "  →  run docker builder prune -af and dvc gc -w -f"
    return "PASS", text


def main() -> int:
    now = datetime.now(UTC)
    checks: list[tuple[str, Callable[[], Result]]] = [
        ("git", check_git),
        ("containers", check_containers),
        ("API health", check_health),
        ("portal", check_portal),
        ("production models", check_models),
        ("scheduler", lambda: check_scheduler(now)),
        ("pipeline runs", lambda: check_last_run(now)),
        ("monitoring", check_monitoring),
        ("ops UIs", check_ops_uis),
        ("resources", check_resources),
    ]
    results: list[tuple[str, Result]] = []
    for name, check in checks:
        try:
            result = check()
        except requests.ConnectionError as exc:
            host = getattr(exc.request, "url", "") or ""
            result = ("FAIL", f"not reachable {host.split('?')[0]}  →  docker compose up -d")
        except (requests.RequestException, subprocess.CalledProcessError, OSError,
                KeyError, ValueError) as exc:  # fmt: skip
            result = ("FAIL", f"{exc.__class__.__name__}: {str(exc)[:160]}")
        results.append((name, result))
        print(f"{result[0]:5} {name:18} {result[1]}", flush=True)
    code = overall(results)
    print("\nready for the demo" if code == 0 else "\nfix the FAIL lines above, then run again")
    return code


if __name__ == "__main__":
    sys.exit(main())
