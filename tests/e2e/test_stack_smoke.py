"""End-to-end smoke tests against the running docker compose stack (Day 9).

Run after ``docker compose up -d`` (and ``make monitoring``):

    pytest tests/e2e -m e2e -q

Each test talks to the real services over HTTP, the way the browser, Prometheus
or an operator would. Tests are skipped (not failed) when the service they need
is not running, so the suite also works on a partial stack. Base URLs can be
changed with environment variables (``FINSIGHT_API_URL`` and friends).
"""

from __future__ import annotations

import os

import pytest
import requests

from finsight.common.config import load_settings
from tests.conftest import PROJECT_ROOT

pytestmark = pytest.mark.e2e

API = os.environ.get("FINSIGHT_API_URL", "http://localhost:8000").rstrip("/")
WEB = os.environ.get("FINSIGHT_WEB_URL", "http://localhost:8080").rstrip("/")
SCHEDULER = os.environ.get("FINSIGHT_SCHEDULER_URL", "http://localhost:8001").rstrip("/")
MLFLOW = os.environ.get("FINSIGHT_MLFLOW_URL", "http://localhost:5000").rstrip("/")
PROMETHEUS = os.environ.get("FINSIGHT_PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
ALERTMANAGER = os.environ.get("FINSIGHT_ALERTMANAGER_URL", "http://localhost:9093").rstrip("/")
GRAFANA = os.environ.get("FINSIGHT_GRAFANA_URL", "http://localhost:3000").rstrip("/")
TIMEOUT = 15


def get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, timeout=TIMEOUT, **kwargs)


def need(base: str, path: str = "/") -> None:
    """Skip the test when the service behind ``base`` is not running."""
    try:
        requests.get(base + path, timeout=3)
    except requests.RequestException:
        pytest.skip(f"{base} is not reachable (start the stack first)")


@pytest.fixture(scope="module")
def tickers() -> list[str]:
    need(API, "/api/health")
    stocks = get(f"{API}/api/stocks").json()["stocks"]
    return [s["symbol"] for s in stocks]


# ------------------------------------------------------------------- backend
def test_health_reports_every_dependency():
    need(API, "/api/health")
    body = get(f"{API}/api/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["dependencies"]) == {
        "mongodb",
        "lstm-serve",
        "sentiment-serve",
        "mlflow-registry",
    }
    assert body["dependencies"]["mongodb"]["ok"], "the backend cannot reach MongoDB"


def test_universe_and_search(tickers):
    assert len(tickers) == 10
    results = get(f"{API}/api/search", params={"q": "app"}).json()["results"]
    assert results and results[0]["symbol"] == "AAPL"


def test_every_stock_has_recent_prices(tickers):
    for ticker in tickers:
        body = get(f"{API}/api/stock/{ticker}", params={"period": "1M"}).json()
        assert body["history"], ticker
        assert body["overview"]["direction"] in {"up", "down", "flat"}, ticker


def test_forecasts_have_14_trading_days_after_the_last_price(tickers):
    for ticker in tickers:
        response = get(f"{API}/api/prediction/{ticker}")
        assert response.status_code == 200, (ticker, response.text[:200])
        body = response.json()
        assert len(body["forecast"]) == 14, ticker
        assert body["forecast"][0]["date"] > body["as_of"][:10], ticker
        assert all(point["price"] > 0 for point in body["forecast"]), ticker


def test_sentiment_history_and_live_scoring(tickers):
    weeks = get(f"{API}/api/sentiment/{tickers[0]}", params={"weeks": 12}).json()["weeks"]
    assert weeks, "no weekly sentiment stored yet"
    assert all(-1 <= w["mean_score"] <= 1 for w in weeks)
    live = requests.post(
        f"{API}/api/sentiment/analyze",
        json={
            "texts": [
                "Profit beat expectations and guidance was raised.",
                "Shares plunged after the recall.",
            ]
        },
        timeout=60,
    )
    assert live.status_code == 200, live.text[:200]
    good, bad = live.json()["results"]
    assert good["score"] > 0 > bad["score"], (good, bad)


def test_portfolio_weights_are_a_long_only_allocation():
    need(API, "/api/health")
    response = get(f"{API}/api/portfolio")
    assert response.status_code == 200, response.text  # 404 no_portfolio: run the pipeline once
    body = response.json()
    weights = body["allocation"]["weights"]
    assert abs(sum(weights.values()) - 1) < 1e-6
    assert all(w >= 0 for w in weights.values())
    assert "beats_equal_weight_out_of_sample" in body


def test_errors_use_the_documented_codes():
    need(API, "/api/health")
    unknown = get(f"{API}/api/stock/NOPE")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "unknown_ticker"
    bad = get(f"{API}/api/stock/AAPL", params={"period": "5Y"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"
    too_many = requests.post(f"{API}/api/sentiment/analyze", json={"texts": []}, timeout=TIMEOUT)
    assert too_many.status_code == 422


def test_cors_allows_the_portal_only():
    need(API, "/api/health")
    preflight = {"Access-Control-Request-Method": "GET"}
    ok = requests.options(
        f"{API}/api/stocks",
        headers={"Origin": "http://localhost:8080", **preflight},
        timeout=TIMEOUT,
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8080"
    evil = requests.options(
        f"{API}/api/stocks", headers={"Origin": "http://evil.example", **preflight}, timeout=TIMEOUT
    )
    assert "access-control-allow-origin" not in evil.headers


def test_backend_metrics_are_exposed():
    need(API, "/api/health")
    text = get(f"{API}/metrics").text
    for name in (
        "finsight_http_requests_total",
        "finsight_http_request_duration_seconds",
        "finsight_dependency_up",
    ):
        assert name in text


def test_pipeline_console_endpoints():
    need(API, "/api/health")
    status = get(f"{API}/api/pipeline/status").json()
    assert status["reachable"], "the backend cannot reach the scheduler"
    assert get(f"{API}/api/pipeline/runs", params={"limit": 5}).json()["runs"]
    assert get(f"{API}/api/pipeline/dag").json()["nodes"]
    monitoring = get(f"{API}/api/monitoring").json()
    assert monitoring["report"]["returns"]["per_ticker"]


# ------------------------------------------------------------------ frontend
def test_portal_serves_the_app_with_security_headers():
    need(WEB, "/healthz")
    page = get(f"{WEB}/")
    assert page.status_code == 200 and "FinSight" in page.text
    csp = page.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp and "script-src 'self'" in csp
    assert page.headers.get("x-content-type-options") == "nosniff"
    assert page.headers.get("x-frame-options") == "DENY"
    for asset in (
        "/assets/js/app.js",
        "/assets/js/pipeline.js",
        "/assets/css/app.css",
        "/help.html",
    ):
        assert get(WEB + asset).status_code == 200, asset


def test_portal_reaches_the_backend_only_through_its_configured_url():
    """Loose coupling: the only link between the two containers is the REST base URL."""
    need(WEB, "/healthz")
    config = get(f"{WEB}/config.js").text
    assert f'backendUrl: "{API}"' in config
    assert get(f"{WEB}/api/stocks").status_code == 404  # the portal does not proxy the API


# ------------------------------------------------------------- other services
def test_scheduler_and_mlflow_are_up():
    need(SCHEDULER, "/health")
    assert get(f"{SCHEDULER}/health").json()["status"] == "ok"
    assert (
        requests.post(f"{SCHEDULER}/run", json={"kind": "daily"}, timeout=TIMEOUT).status_code
        == 401
    )
    need(MLFLOW, "/health")
    assert get(f"{MLFLOW}/health").status_code == 200


def test_every_prometheus_target_is_up():
    need(PROMETHEUS, "/-/ready")
    targets = get(f"{PROMETHEUS}/api/v1/targets").json()["data"]["activeTargets"]
    down = {t["labels"]["job"]: t.get("lastError") for t in targets if t["health"] != "up"}
    assert len(targets) >= 9 and not down, down
    groups = get(f"{PROMETHEUS}/api/v1/rules").json()["data"]["groups"]
    assert sum(len(g["rules"]) for g in groups) >= 20


def test_alertmanager_and_grafana_are_up():
    need(ALERTMANAGER, "/-/ready")
    assert get(f"{ALERTMANAGER}/-/ready").status_code == 200
    need(GRAFANA, "/api/health")
    assert get(f"{GRAFANA}/api/health").json()["database"] == "ok"
    assert get(f"{GRAFANA}/api/search").status_code == 401  # dashboards need a login


def test_mongodb_refuses_connections_without_credentials():
    settings = load_settings(PROJECT_ROOT)
    from pymongo import MongoClient  # noqa: PLC0415
    from pymongo.errors import OperationFailure, PyMongoError  # noqa: PLC0415

    client = MongoClient(settings.mongo_host, settings.mongo_port, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")  # allowed without auth
    except PyMongoError:
        pytest.skip("MongoDB not reachable")
    with pytest.raises(OperationFailure):
        client[settings.mongo_db].list_collection_names()
    client.close()
