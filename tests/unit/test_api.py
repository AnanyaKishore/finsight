"""Tests for the FastAPI backend (in-memory MongoDB, stub model servers)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
import requests

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from finsight.api.app import create_app  # noqa: E402
from finsight.api.config import ApiConfig  # noqa: E402
from finsight.api.model_client import ModelClient, ModelUnavailable  # noqa: E402
from finsight.api.repository import next_trading_days  # noqa: E402
from finsight.common.config import get_settings  # noqa: E402
from finsight.common.mongo import PORTFOLIOS, PRICES, SENTIMENT_WEEKLY  # noqa: E402
from finsight.common.timeutils import is_trading_day  # noqa: E402
from tests.factories import make_prices  # noqa: E402


class StubModel:
    """Stands in for a ModelClient: records calls, can be switched off."""

    def __init__(self, fn, up=True):
        self.fn, self.up, self.calls = fn, up, 0

    def predict(self, columns, rows):
        self.calls += 1
        if not self.up:
            raise ModelUnavailable("down")
        return [self.fn(dict(zip(columns, row, strict=True))) for row in rows]

    def ping(self):
        return self.up, "HTTP 200" if self.up else "ConnectionError"


def _lstm_fn(row):
    last = row["p_099"]
    return {f"h_{i:02d}": last + i for i in range(1, 15)}


def _sent_fn(row):
    pos = "rose" in row["text"]
    return {
        "label": "positive" if pos else "negative",
        "confidence": 0.9,
        "score": 0.8 if pos else -0.8,
    }


def _seed(db):
    prices = make_prices(tickers=("AAPL", "MSFT"), start=date(2026, 3, 2), end=date(2026, 10, 20))
    docs = prices.to_dict("records")
    for d in docs:
        d["date"] = datetime(d["date"].year, d["date"].month, d["date"].day, tzinfo=UTC)
    db[PRICES].insert_many(docs)
    for i, (mean, n) in enumerate([(0.5, 10), (-0.25, 30)]):
        db[SENTIMENT_WEEKLY].insert_one(
            {
                "ticker": "AAPL",
                "week_start": datetime(2026, 10, 5 + 7 * i, tzinfo=UTC),
                "article_count": n,
                "mean_score": mean,
                "positive": 3,
                "negative": 2,
                "neutral": n - 5,
                "top_positive": [
                    {"headline": "Apple rose", "score": 0.99, "source": "X", "url": "u"}
                ],
                "top_negative": [],
            }
        )
    alloc = {
        "method": "max_sharpe",
        "weights": {"AAPL": 0.6, "MSFT": 0.4},
        "expected_return": 0.3,
        "volatility": 0.2,
        "sharpe": 1.3,
    }
    db[PORTFOLIOS].insert_one(
        {
            "as_of": datetime(2026, 10, 20, tzinfo=UTC),
            "window_start": "2026-09-08",
            "window_end": "2026-10-20",
            "window_days": 30,
            "risk_free_rate": 0.039,
            "grid_size": 92378,
            "fallback_used": False,
            "allocation": alloc,
            "in_sample": {
                "max_sharpe": alloc,
                "equal_weight": alloc | {"method": "equal_weight", "sharpe": 0.9},
                "min_volatility": alloc | {"method": "min_volatility", "sharpe": 1.0},
            },
            "walk_forward": {
                "performance": {"max_sharpe": {"sharpe": 0.1}, "equal_weight": {"sharpe": 0.6}},
                "beats_equal_weight_sharpe": False,
            },
        }
    )


@pytest.fixture
def api(db):
    _seed(db)
    lstm, sent = StubModel(_lstm_fn), StubModel(_sent_fn)
    registry = {"finsight-lstm": {"version": "3", "run_id": "r3", "tags": {"test_mape": "2.9"}}}
    app = create_app(
        settings=get_settings(),
        config=ApiConfig(frontend_origins=("http://localhost:8080",)),
        db=db,
        lstm=lstm,
        sentiment=sent,
        registry=lambda name: registry.get(name),
    )
    return TestClient(app), lstm, sent, db


# ------------------------------------------------------------------- search
def test_search_ranks_symbol_name_alias(api):
    client, *_ = api
    hits = client.get("/api/search", params={"q": "ms"}).json()["results"]
    assert hits[0] == {
        "symbol": "MSFT",
        "name": "Microsoft",
        "sector": "Technology",
        "matched": "symbol",
    }
    assert (
        client.get("/api/search", params={"q": "iphone"}).json()["results"][0]["symbol"] == "AAPL"
    )
    assert client.get("/api/search", params={"q": "zzz"}).json()["results"] == []
    bad = client.get("/api/search", params={"q": ""})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"


# -------------------------------------------------------------------- stock
def test_stock_overview_history_and_errors(api):
    client, *_ = api
    body = client.get("/api/stock/aapl", params={"period": "1M"}).json()
    ov = body["overview"]
    assert body["symbol"] == "AAPL" and ov["date"] == "2026-10-20"
    assert ov["change"] == pytest.approx(ov["close"] - ov["previous_close"], abs=1e-3)
    assert ov["direction"] in ("up", "down") and 18 <= len(body["history"]) <= 24
    assert len(client.get("/api/stock/AAPL", params={"period": "ALL"}).json()["history"]) > 150
    unknown = client.get("/api/stock/XYZ")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "unknown_ticker"
    assert client.get("/api/stock/TSLA").json()["error"]["code"] == "no_data"
    assert client.get("/api/stock/AAPL", params={"period": "2Y"}).status_code == 422


# --------------------------------------------------------------- prediction
def test_prediction_calls_model_once_per_day_and_stores_it(api):
    client, lstm, _, db = api
    first = client.get("/api/prediction/AAPL").json()
    assert lstm.calls == 1 and first["stale"] is False
    assert first["model"] == {"name": "finsight-lstm", "version": "3", "run_id": "r3"}
    assert len(first["forecast"]) == 14 and len(first["history"]) == 60
    days = [date.fromisoformat(p["date"]) for p in first["forecast"]]
    assert days == next_trading_days(date(2026, 10, 20), 14) and all(map(is_trading_day, days))
    last = first["history"][-1]["price"]
    assert first["forecast"][0]["price"] == pytest.approx(last + 1, abs=1e-3)
    second = client.get("/api/prediction/AAPL").json()
    assert lstm.calls == 1 and second["forecast"] == first["forecast"]  # served from Mongo
    assert db["forecasts"].count_documents({"ticker": "AAPL"}) == 1


def test_prediction_degrades_to_stale_then_503(api):
    client, lstm, _, db = api
    client.get("/api/prediction/AAPL")
    db["forecasts"].update_many({}, {"$set": {"model_version": "2"}})  # pretend: older version
    lstm.up = False
    stale = client.get("/api/prediction/AAPL").json()
    assert stale["stale"] is True and stale["model"]["version"] == "2"
    none = client.get("/api/prediction/MSFT")
    assert none.status_code == 503 and none.json()["error"]["code"] == "model_unavailable"
    metrics = client.get("/metrics").text
    assert 'finsight_forecasts_served_total{source="stale"} 1.0' in metrics


# ---------------------------------------------------------------- sentiment
def test_sentiment_weeks_and_weighted_mean(api):
    client, *_ = api
    body = client.get("/api/sentiment/AAPL", params={"weeks": 12}).json()
    assert [w["week_start"] for w in body["weeks"]] == ["2026-10-05", "2026-10-12"]
    assert body["total_articles"] == 40
    assert body["overall_mean_score"] == pytest.approx((0.5 * 10 - 0.25 * 30) / 40)
    assert body["weeks"][0]["top_positive"][0]["headline"] == "Apple rose"
    assert "Not investment advice" in body["disclaimer"]
    assert client.get("/api/sentiment/MSFT").json()["weeks"] == []
    assert client.get("/api/sentiment/AAPL", params={"weeks": 0}).status_code == 422


def test_analyze_calls_the_sentiment_server(api):
    client, _, sent, _ = api
    body = client.post(
        "/api/sentiment/analyze", json={"texts": ["Shares rose", "Shares fell"]}
    ).json()
    assert [r["label"] for r in body["results"]] == ["positive", "negative"]
    assert client.post("/api/sentiment/analyze", json={"texts": ["x" * 2001]}).status_code == 422
    assert client.post("/api/sentiment/analyze", json={"texts": []}).status_code == 422
    sent.up = False
    down = client.post("/api/sentiment/analyze", json={"texts": ["a"]})
    assert down.status_code == 503 and down.json()["error"]["code"] == "model_unavailable"


# ---------------------------------------------------------------- portfolio
def test_portfolio_includes_out_of_sample_verdict(api):
    client, _, _, db = api
    body = client.get("/api/portfolio").json()
    assert body["allocation"]["weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    assert (
        body["equal_weight"]["sharpe"] == 0.9 and body["beats_equal_weight_out_of_sample"] is False
    )
    assert body["as_of"] == "2026-10-20"
    db[PORTFOLIOS].delete_many({})
    assert client.get("/api/portfolio").json()["error"]["code"] == "no_portfolio"


# --------------------------------------------------------------- operations
def test_health_model_info_metrics_and_request_ids(api):
    client, lstm, *_ = api
    assert client.get("/api/health").json()["status"] == "ok"
    lstm.up = False
    health = client.get("/api/health").json()
    assert health["status"] == "degraded" and health["dependencies"]["lstm-serve"]["ok"] is False
    info = client.get("/api/model-info").json()
    assert info["models"][0]["version"] == "3" and info["models"][1]["version"] is None
    response = client.get("/api/stock/AAPL", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    text = client.get("/metrics").text
    assert 'route="/api/stock/{ticker}"' in text  # route template, not raw path
    assert 'finsight_dependency_up{dependency="lstm-serve"} 0.0' in text


def test_cors_allows_only_the_configured_frontend(api):
    client, *_ = api
    ok = client.get("/api/search?q=a", headers={"Origin": "http://localhost:8080"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8080"
    other = client.get("/api/search?q=a", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_registry_lookup_failure_degrades_gracefully(db):
    def broken(name):
        raise ConnectionError("mlflow down")

    app = create_app(
        settings=get_settings(),
        config=ApiConfig(),
        db=db,
        lstm=StubModel(_lstm_fn),
        sentiment=StubModel(_sent_fn),
        registry=broken,
    )
    client = TestClient(app)
    info = client.get("/api/model-info").json()
    assert info["registry_reachable"] is False and info["models"][0]["version"] is None
    assert client.get("/api/health").json()["dependencies"]["mlflow-registry"]["ok"] is False


# -------------------------------------------------------------- model client
class FakeResponse:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes, self.posts = list(outcomes), 0

    def post(self, url, json, timeout):
        self.posts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get(self, url, timeout):
        return FakeResponse(200)


def test_model_client_retries_transient_errors_only():
    seen = []
    ok = FakeResponse(200, {"predictions": [{"h_01": 1.0}]})
    session = FakeSession([requests.ConnectionError("x"), FakeResponse(503), ok])
    client = ModelClient(
        "lstm",
        "http://m",
        retries=2,
        session=session,
        sleep=lambda s: None,
        observe=lambda *a: seen.append(a[1]),
    )
    assert client.predict(["p"], [[1.0]]) == [{"h_01": 1.0}] and session.posts == 3
    assert seen == ["ok"]

    rejected = ModelClient(
        "lstm",
        "http://m",
        retries=3,
        session=FakeSession([FakeResponse(400, text="bad")]),
        sleep=lambda s: None,
    )
    with pytest.raises(ModelUnavailable, match="rejected"):
        rejected.predict(["p"], [[1.0]])

    down = FakeSession([requests.Timeout("t")] * 3)
    with pytest.raises(ModelUnavailable, match="unavailable"):
        ModelClient("lstm", "http://m", retries=2, session=down, sleep=lambda s: None).predict(
            ["p"], [[1]]
        )
    assert down.posts == 3


def test_model_client_accepts_column_oriented_predictions():
    body = {"predictions": {"label": ["positive", "negative"], "score": [0.5, -0.5]}}
    client = ModelClient(
        "s", "http://m", session=FakeSession([FakeResponse(200, body)]), sleep=lambda s: None
    )
    assert client.predict(["text"], [["a"], ["b"]])[1] == {"label": "negative", "score": -0.5}
