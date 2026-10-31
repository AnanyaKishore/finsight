"""Tests for monitoring (Day 8): drift statistics, the report, gauges, scheduler and API hooks."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from finsight.common.mongo import (
    FORECASTS,
    MONITORING_DRILL,
    MONITORING_REPORTS,
    NEWS_SENTIMENT,
    PRICES,
)
from finsight.monitoring import drill
from finsight.monitoring.drift import ks_statistic, psi
from finsight.monitoring.metrics import MonitorMetrics
from finsight.monitoring.monitor import (
    active_drill,
    build_report,
    data_quality,
    forecast_error,
    freshness,
    load_prices,
    returns_drift,
    sentiment_drift,
    summarise,
)
from finsight.scheduler.app import create_app as create_scheduler_app
from tests.unit.test_scheduler import TOKEN, make_service  # noqa: F401 -- fixture

PARAMS = {
    "interval_s": 120,
    "current_days": 60,
    "reference_days": 250,
    "psi_bins": 5,
    "psi_threshold": 0.25,
    "drift_ticker_share": 0.3,
    "sentiment_current_days": 14,
    "sentiment_reference_days": 90,
    "sentiment_psi_threshold": 0.25,
    "error_window_days": 30,
    "min_error_points": 20,
    "error_ratio_to_naive": 1.25,
    "error_mape_floor": 3.0,
    "retrain_cooldown_h": 24,
}
NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)  # Tuesday; latest expected session is Mon 26th


# ------------------------------------------------------------------ helpers
def business_days(end: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n, tz="UTC")


def price_path(n: int, seed: int, vol: float = 0.01, shock_last: int = 0, shock: float = 1.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, vol, n - 1)
    if shock_last:
        returns[-shock_last:] *= shock
    return 100 * np.exp(np.concatenate([[0], np.cumsum(returns)]))


def seed_prices(db, tickers=("AAPL", "MSFT", "NVDA"), n=300, end="2026-10-26", shocked=()):
    days = business_days(end, n)
    docs = []
    for i, ticker in enumerate(tickers):
        path = price_path(n, i, shock_last=20 if ticker in shocked else 0, shock=5.0)
        docs += [
            {"ticker": ticker, "date": d.to_pydatetime(), "close": float(p), "adj_close": float(p)}
            for d, p in zip(days, path, strict=True)
        ]
    db[PRICES].insert_many(docs)
    return days


# --------------------------------------------------------------------- drift
def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(0, 4, 500)) > 0.25  # volatility shock
    assert psi(ref, rng.normal(2, 1, 500)) > 1.0  # mean shift
    assert psi(ref, np.full(50, 100.0)) > 1.0  # beyond the reference range still counts


def test_psi_handles_a_constant_reference_and_rejects_tiny_samples():
    assert np.isfinite(psi(np.zeros(100), np.linspace(-1, 1, 30)))
    with pytest.raises(ValueError):
        psi(np.arange(5), np.arange(5), bins=10)


def test_ks_statistic_bounds():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 1000)
    assert ks_statistic(a, a) == 0
    assert ks_statistic(np.zeros(10), np.ones(10)) == 1
    assert 0 < ks_statistic(a, rng.normal(0.5, 1, 1000)) < 1
    with pytest.raises(ValueError):
        ks_statistic([], [1.0])


# -------------------------------------------------------------------- report
def test_load_prices_uses_adj_close_only_when_complete(db):
    days = business_days("2026-10-26", 3)
    db[PRICES].insert_many(
        [{"ticker": "A", "date": d.to_pydatetime(), "close": 10.0, "adj_close": 9.0} for d in days]
        + [{"ticker": "B", "date": days[0].to_pydatetime(), "close": 20.0, "adj_close": None}]
        + [{"ticker": "B", "date": days[1].to_pydatetime(), "close": 21.0, "adj_close": 20.5}]
    )
    prices = load_prices(db)
    assert set(prices.loc[prices.ticker == "A", "price"]) == {9.0}
    assert list(prices.loc[prices.ticker == "B", "price"]) == [20.0, 21.0]


def test_returns_drift_flags_shocked_tickers_and_the_share(db):
    seed_prices(db, tickers=("AAPL", "MSFT", "NVDA"), shocked=("NVDA",))
    out = returns_drift(load_prices(db), PARAMS)
    assert out["per_ticker"]["NVDA"]["drifted"] and out["per_ticker"]["NVDA"]["psi"] > 0.25
    assert not out["per_ticker"]["AAPL"]["drifted"]
    assert out["drifted_share"] == pytest.approx(1 / 3, abs=1e-3)
    assert out["detected"]  # 33% >= 30%
    assert (
        out["per_ticker"]["NVDA"]["current_volatility"]
        > out["per_ticker"]["NVDA"]["reference_volatility"]
    )


def test_returns_drift_without_shock_is_not_detected_but_a_drill_is(db):
    seed_prices(db)
    prices = load_prices(db)
    assert not returns_drift(prices, PARAMS)["detected"]
    drilled = returns_drift(prices, PARAMS, {"scale": 4.0})
    assert drilled["detected"] and drilled["drifted_share"] == 1.0


def test_short_history_is_reported_not_scored(db):
    seed_prices(db, tickers=("NEW",), n=40)
    out = returns_drift(load_prices(db), PARAMS)
    assert out["per_ticker"]["NEW"]["status"] == "insufficient_data"
    assert not out["detected"] and out["max_psi"] == 0


def test_active_drill_expires(db):
    db[MONITORING_DRILL].insert_one(
        {"_id": "drill", "until": NOW + timedelta(minutes=5), "scale": 4.0}
    )
    assert active_drill(db, NOW)["scale"] == 4.0
    assert active_drill(db, NOW + timedelta(minutes=6)) is None


def test_sentiment_drift_needs_enough_articles_and_sees_a_shift(db):
    rng = np.random.default_rng(3)

    ids = iter(range(10_000))

    def article(days_ago, score):
        return {
            "ticker": "AAPL",
            "news_id": f"n{next(ids)}",
            "published_at": NOW - timedelta(days=days_ago),
            "score": float(score),
        }

    db[NEWS_SENTIMENT].insert_many(
        [article(d % 90 + 15, s) for d, s in enumerate(rng.normal(0.1, 0.3, 200))]
    )
    assert sentiment_drift(db, PARAMS, NOW)["status"] == "insufficient_data"
    db[NEWS_SENTIMENT].insert_many(
        [article(d % 14, s) for d, s in enumerate(rng.normal(0.1, 0.3, 60))]
    )
    calm = sentiment_drift(db, PARAMS, NOW)
    assert calm["n_current"] == 60 and calm["n_reference"] == 200 and not calm["detected"]
    shifted = sentiment_drift(db, PARAMS, NOW, {"sentiment_shift": -0.5})
    assert shifted["detected"] and shifted["mean_current"] < calm["mean_current"]


def _seed_forecasts(db, days, error: float, n_docs=5):
    """Forecasts made on earlier days whose 5 target days have now traded."""
    prices = load_prices(db)
    actual = {(r.ticker, r.date): r.price for r in prices.itertuples(index=False)}
    for i in range(n_docs):
        as_of = days[-10 - i]
        targets = days[-9 - i : -4 - i]
        last = actual[("AAPL", as_of)]
        db[FORECASTS].insert_one(
            {
                "ticker": "AAPL",
                "as_of": as_of.to_pydatetime(),
                "last_price": last,
                "model_version": "3",
                "forecast": [
                    {"date": t.date().isoformat(), "price": actual[("AAPL", t)] * (1 + error)}
                    for t in targets
                ],
            }
        )


def test_forecast_error_compares_with_naive_on_the_same_points(db):
    days = seed_prices(db, tickers=("AAPL",))
    _seed_forecasts(db, days, error=0.0, n_docs=2)
    few = forecast_error(db, load_prices(db), PARAMS, NOW)
    assert few == {
        "points": 10,
        "model_versions": ["3"],
        "status": "insufficient_data",
        "detected": False,
    }
    db[FORECASTS].delete_many({})
    _seed_forecasts(db, days, error=0.08)
    bad = forecast_error(db, load_prices(db), PARAMS, NOW)
    assert bad["points"] == 25 and bad["mape"] == pytest.approx(8.0, abs=0.01)
    assert bad["ratio_to_naive"] == pytest.approx(bad["mape"] / bad["naive_mape"], abs=0.01)
    assert bad["detected"] and bad["per_ticker_mape"] == {"AAPL": bad["mape"]}
    db[FORECASTS].delete_many({})
    _seed_forecasts(db, days, error=0.0001)
    assert not forecast_error(db, load_prices(db), PARAMS, NOW)["detected"]


def test_freshness_counts_trading_days_behind(db):
    seed_prices(db, tickers=("AAPL",), end="2026-10-26", n=5)
    seed_prices(db, tickers=("OLD",), end="2026-10-22", n=5)
    out = freshness(load_prices(db), NOW)
    assert out["expected_session"] == "2026-10-26"
    assert out["lag_trading_days"] == {"AAPL": 0, "OLD": 2}
    assert out["max_lag"] == 2


def test_data_quality_counts_failed_checks(tmp_path):
    assert data_quality(tmp_path) == {"status": "no_report"}
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "data_quality.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "as_of_session": "2026-10-26",
                "checks": [
                    {"name": "prices.positive_values", "severity": "error", "success": False},
                    {"name": "news.recent", "severity": "warn", "success": False},
                    {"name": "prices.unique", "severity": "error", "success": True},
                ],
            }
        )
    )
    out = data_quality(tmp_path)
    assert out["errors"] == 1 and out["warnings"] == 1
    assert out["failed_checks"] == ["prices.positive_values", "news.recent"]


def test_build_report_isolates_a_broken_section(db, tmp_path):
    seed_prices(db)
    db[FORECASTS].insert_one(
        {"ticker": "AAPL", "as_of": NOW, "forecast": [{"date": "bad-date", "price": 1}]}
    )
    db[PRICES].update_many({}, {"$set": {"adj_close": 1.0}})  # keep load_prices valid
    report = build_report(db, tmp_path, PARAMS, now=NOW)
    assert report["forecast"]["status"] == "error" and "error" in report["forecast"]
    assert "per_ticker" in report["returns"] and report["quality"] == {"status": "no_report"}
    assert report["drill"] is None and report["generated_at"] == NOW
    assert "returns drift on 0/3" in summarise(report)


def test_summarise_mentions_the_drill():
    assert summarise({}) == "no data yet"
    text = summarise({"drill": {"scale": 4}, "forecast": {"points": 3}})
    assert "DRIFT DRILL ACTIVE" in text and "3 realised points" in text


# -------------------------------------------------------------------- gauges
def test_metrics_follow_the_report():
    registry = CollectorRegistry()
    metrics = MonitorMetrics(registry)
    metrics.update(
        {
            "generated_at": NOW,
            "drill": {"scale": 4},
            "returns": {
                "per_ticker": {
                    "AAPL": {"psi": 0.5, "ks": 0.4},
                    "NEW": {"status": "insufficient_data"},
                },
                "drifted_share": 1.0,
                "detected": True,
            },
            "sentiment": {"status": "insufficient_data"},
            "forecast": {
                "points": 25,
                "mape": 4.0,
                "naive_mape": 2.0,
                "ratio_to_naive": 2.0,
                "per_ticker_mape": {"AAPL": 4.0},
                "detected": True,
            },
            "freshness": {"lag_trading_days": {"AAPL": 1}},
            "quality": {"errors": 2, "warnings": 0},
            "models": {"lstm": "7", "sentiment": None},
        }
    )
    text = generate_latest(registry).decode()
    for line in (
        'finsight_drift_psi{feature="returns",ticker="AAPL"} 0.5',
        'finsight_drift_detected{feature="returns"} 1.0',
        'finsight_forecast_live_mape_percent{model="naive"} 2.0',
        "finsight_forecast_error_ratio_to_naive 2.0",
        "finsight_forecast_degraded 1.0",
        'finsight_data_lag_trading_days{ticker="AAPL"} 1.0',
        'finsight_data_quality_failed_checks{severity="error"} 2.0',
        'finsight_model_production_version{model="lstm"} 7.0',
        "finsight_drift_drill_active 1.0",
    ):
        assert line in text, line
    assert 'ticker="NEW"' not in text and 'feature="sentiment"' not in text


# ----------------------------------------------------------------- scheduler
def test_monitor_step_stores_the_report_and_updates_gauges(make_service, db):
    calls = []

    def builder(db_, root, params, now):
        calls.append(now)
        return {
            "generated_at": now,
            "drill": None,
            "returns": {"per_ticker": {}, "detected": True, "drifted_share": 0.5, "max_psi": 0.4},
            "forecast": {"status": "error", "error": "x"},
        }

    service, _ = make_service(monitoring=PARAMS, monitor_builder=builder)
    run_id = service.start("daily", "console", wait=True)
    doc = db.pipeline_executions.find_one({"run_id": run_id})
    monitor = next(s for s in doc["steps"] if s["name"] == "monitor")
    assert monitor["status"] == "partial" and doc["status"] == "partial"
    assert db[MONITORING_REPORTS].count_documents({}) == 1 and calls
    assert service.metrics.monitor.detected.labels("returns")._value.get() == 1


def test_retrain_webhook_has_a_cooldown(make_service):
    service, clock = make_service(monitoring=PARAMS, monitor_builder=lambda *a, **k: {})
    client = TestClient(create_scheduler_app(service, TOKEN))
    auth = {"Authorization": f"Bearer {TOKEN}"}
    body = {"status": "firing", "alerts": [{"labels": {"alertname": "ReturnsDriftDetected"}}]}
    first = client.post("/retrain", json=body, headers=auth)
    assert first.status_code == 202
    for thread in service._threads:
        thread.join(5)
    clock["now"] += timedelta(hours=2)
    again = client.post("/retrain", json=body, headers=auth)
    assert again.status_code == 200 and again.json()["ignored"] == "cooldown"
    assert 21 * 3600 < again.json()["retry_after_s"] <= 22 * 3600
    assert service.metrics.retrains_skipped._value.get() == 1
    assert service.metrics.retrains.labels("webhook")._value.get() == 1
    clock["now"] += timedelta(hours=23)
    assert client.post("/retrain", json=body, headers=auth).status_code == 202
    for thread in service._threads:
        thread.join(5)


def test_monitor_endpoint_and_disabled_monitoring(make_service):
    service, _ = make_service(
        monitoring=PARAMS,
        monitor_builder=lambda db, root, p, now: {"generated_at": now, "drill": None},
    )
    client = TestClient(create_scheduler_app(service, TOKEN))
    assert client.post("/monitor").status_code == 401
    ok = client.post("/monitor", headers={"X-Finsight-Token": TOKEN})
    assert ok.status_code == 200 and ok.json()["summary"] == "no data yet"
    plain, _ = make_service()
    off = TestClient(create_scheduler_app(plain, TOKEN)).post(
        "/monitor", headers={"X-Finsight-Token": TOKEN}
    )
    assert off.status_code == 503 and off.json()["error"]["code"] == "monitoring_disabled"


def test_recover_seeds_last_run_gauges(make_service, db):
    service, _ = make_service()
    service.start("daily", "console", wait=True)
    fresh, _ = make_service()
    fresh.recover()
    assert fresh.metrics.last_status.labels("daily")._value.get() == 1
    assert fresh.metrics.last_finished.labels("daily")._value.get() > 0


# ----------------------------------------------------------------------- API
class FakeAlertResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_fetch_alerts_sorts_by_severity():
    from finsight.api.monitoring import fetch_alerts

    class Http:
        def get(self, url, params, timeout):
            assert url == "http://am:9093/api/v2/alerts" and params["silenced"] == "false"
            return FakeAlertResponse(
                [
                    {
                        "labels": {"alertname": "SentimentShift", "severity": "info"},
                        "annotations": {},
                    },
                    {
                        "labels": {
                            "alertname": "ReturnsDriftDetected",
                            "severity": "warning",
                            "action": "retrain",
                        },
                        "annotations": {"summary": "drift"},
                    },
                ]
            )

    alerts = fetch_alerts("http://am:9093/", Http())
    assert [a["name"] for a in alerts] == ["ReturnsDriftDetected", "SentimentShift"]
    assert alerts[0]["action"] == "retrain" and alerts[0]["summary"] == "drift"


def _api(db, alerts):
    from finsight.api.app import create_app
    from finsight.api.config import ApiConfig
    from finsight.common.config import get_settings

    app = create_app(
        settings=get_settings(),
        config=ApiConfig(frontend_origins=("http://localhost:8080",)),
        db=db,
        registry=lambda name: None,
        clock=lambda: NOW,
        monitoring_kwargs={"alerts": alerts},
    )
    return TestClient(app)


def test_api_monitoring_returns_report_and_alerts(db):
    def down(url):
        raise requests.ConnectionError("refused")

    assert _api(db, down).get("/api/monitoring").json()["error"]["code"] == "no_monitoring"
    db[MONITORING_REPORTS].insert_many(
        [
            {"generated_at": NOW - timedelta(minutes=m), "drill": None, "returns": {"max_psi": m}}
            for m in (4, 2)
        ]
    )
    up = _api(
        db,
        lambda url: [
            {"name": "X", "severity": "warning", "action": None, "summary": "", "starts_at": None}
        ],
    ).get("/api/monitoring")
    body = up.json()
    assert up.status_code == 200 and body["alertmanager_reachable"]
    assert body["report"]["returns"]["max_psi"] == 2 and body["alerts"][0]["name"] == "X"
    partial = _api(db, down).get("/api/monitoring").json()
    assert not partial["alertmanager_reachable"] and "refused" in partial["detail"]


# --------------------------------------------------------------------- drill
def test_drill_cli_validates_and_writes(monkeypatch, db, capsys):
    import mongomock

    client = mongomock.MongoClient(tz_aware=True)
    monkeypatch.setattr(drill, "get_client", lambda settings: client)
    monkeypatch.setenv("MONGO_DB", "finsight_test")
    with pytest.raises(SystemExit):
        drill.main(["start", "--minutes", "999"])
    assert drill.main(["start", "--minutes", "10", "--scale", "3"]) == 0
    name = drill.get_settings().mongo_db
    doc = client[name][MONITORING_DRILL].find_one({"_id": "drill"})
    assert doc["scale"] == 3 and doc["until"] - doc["started_at"] == timedelta(minutes=10)
    drill.main(["stop"])
    assert client[name][MONITORING_DRILL].count_documents({}) == 0
    assert "stopped" in capsys.readouterr().out
