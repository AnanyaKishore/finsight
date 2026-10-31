"""FastAPI application factory and routes.

Endpoints (I/O contracts in :mod:`finsight.api.schemas`, live docs at ``/docs``):

=================================  ==============================================
``GET  /api/stocks``                the 10-stock universe (portal home page)
``GET  /api/search?q=``             type-ahead over the 10-stock universe
``GET  /api/stock/{ticker}``        overview + price history (``period`` filter)
``GET  /api/prediction/{ticker}``   14-day LSTM forecast (via ``lstm-serve``)
``GET  /api/sentiment/{ticker}``    weekly sentiment with explanatory headlines
``POST /api/sentiment/analyze``     score arbitrary texts (via ``sentiment-serve``)
``GET  /api/portfolio``             latest optimised allocation + out-of-sample check
``GET  /api/health``                dependency status (``ok`` / ``degraded``)
``GET  /api/model-info``            production versions from the MLflow registry
``GET  /api/pipeline/...``          Pipeline Console: status, runs, run detail, DAG, stats
``POST /api/pipeline/trigger``      start a daily run or a retrain (via the scheduler)
``GET  /api/monitoring``            latest drift / live-error report + active alerts (Day 8)
``GET  /metrics``                   Prometheus metrics
=================================  ==============================================

Graceful degradation: if ``lstm-serve`` is down, ``/api/prediction`` returns
the most recent stored forecast with ``stale: true``; if MongoDB or a model
server is down, the affected endpoints answer ``503`` with a JSON error while
the rest keep working, and ``/api/health`` reports ``degraded``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pymongo.errors import PyMongoError

import finsight
from finsight.api import schemas as s
from finsight.api.config import ApiConfig
from finsight.api.metrics import ApiMetrics
from finsight.api.model_client import ModelClient, ModelUnavailable
from finsight.api.monitoring import build_router as build_monitoring_router
from finsight.api.pipeline import SchedulerClient, build_router
from finsight.api.repository import MarketRepository, as_date, next_trading_days
from finsight.common.config import Settings, get_settings
from finsight.common.logging import get_logger

log = get_logger(__name__)

LOOKBACK, HORIZON = 100, 14


class ApiError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        """Create the error."""
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


# --------------------------------------------------------------------------- #
# Registry lookup (cached; never blocks the API for long)
# --------------------------------------------------------------------------- #
class RegistryLookup:
    """Production model versions from MLflow, cached for ``ttl_s`` seconds."""

    def __init__(self, ttl_s: float, fetch: Callable[[str], dict | None] | None = None) -> None:
        """``fetch(name)`` returns ``{version, run_id, tags}`` or ``None``."""
        self.ttl_s = ttl_s
        self.fetch = fetch or self._fetch_mlflow
        self._cache: dict[str, tuple[float, dict | None]] = {}
        self.reachable = True

    @staticmethod
    def _fetch_mlflow(name: str) -> dict | None:
        from mlflow import MlflowClient  # noqa: PLC0415

        version = MlflowClient().get_model_version_by_alias(name, "production")
        return {"version": str(version.version), "run_id": version.run_id, "tags": version.tags}

    def __call__(self, name: str) -> dict | None:
        """Cached production info for ``name`` (``None`` if unknown/unreachable)."""
        now = time.monotonic()
        hit = self._cache.get(name)
        if hit and now - hit[0] < self.ttl_s:
            return hit[1]
        try:
            value = self.fetch(name)
            self.reachable = True
        except Exception as exc:  # noqa: BLE001 -- registry down must not break the API
            log.warning("registry lookup failed", extra={"model": name, "error": str(exc)[:200]})
            value, self.reachable = None, False
        self._cache[name] = (now, value)
        return value


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_app(  # noqa: C901 -- one function keeps every route visible in one place
    settings: Settings | None = None,
    config: ApiConfig | None = None,
    db: Any = None,
    lstm: ModelClient | None = None,
    sentiment: ModelClient | None = None,
    registry: Callable[[str], dict | None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    scheduler: SchedulerClient | None = None,
    monitoring_kwargs: dict[str, Any] | None = None,
) -> FastAPI:
    """Build the app; every dependency is injectable for tests."""
    monitoring_kwargs = monitoring_kwargs or {}
    settings = settings or get_settings()
    config = config or ApiConfig.from_env()
    metrics = ApiMetrics()
    if db is None:
        from pymongo import MongoClient  # noqa: PLC0415

        # No ping here: the API starts even if MongoDB is still starting (degraded).
        client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=2000, tz_aware=True)
        db = client[settings.mongo_db]
    repo = MarketRepository(db)
    lstm = lstm or ModelClient(
        "lstm",
        config.lstm_url,
        timeout_s=config.model_timeout_s,
        retries=config.model_retries,
        observe=metrics.observe_model,
    )
    sentiment = sentiment or ModelClient(
        "sentiment",
        config.sentiment_url,
        timeout_s=config.model_timeout_s,
        retries=config.model_retries,
        observe=metrics.observe_model,
    )
    registry_lookup = (
        registry
        if isinstance(registry, RegistryLookup)
        else RegistryLookup(config.registry_cache_s, registry)
    )
    universe = {t.symbol: t for t in settings.tickers}

    app = FastAPI(
        title="FinSight API",
        version=finsight.__version__,
        description="Stock analytics: forecasts, news sentiment, portfolio. Not investment advice.",
        responses={
            404: {"model": s.ErrorResponse},
            422: {"model": s.ErrorResponse},
            503: {"model": s.ErrorResponse},
        },
    )
    app.state.metrics, app.state.repo, app.state.config = metrics, repo, config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.frontend_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def observe(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        template = getattr(route, "path", "unmatched")
        if template != "/metrics":
            metrics.requests.labels(request.method, template, str(response.status_code)).inc()
            metrics.latency.labels(request.method, template).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": template,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000, 1),
            },
        )
        return response

    # -------------------------------------------------------- error handlers
    def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
        body = s.ErrorResponse(
            error=s.ErrorBody(
                code=code, message=message, request_id=getattr(request.state, "request_id", None)
            )
        )
        return JSONResponse(status_code=status, content=body.model_dump())

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []))
        return _error(request, 422, "invalid_request", f"{where}: {first.get('msg', 'invalid')}")

    @app.exception_handler(PyMongoError)
    async def database_down(request: Request, exc: PyMongoError) -> JSONResponse:
        log.error("database error", extra={"error": str(exc)[:200]})
        return _error(request, 503, "database_unavailable", "data store temporarily unavailable")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error(request, 500, "internal_error", "unexpected server error")

    def known(ticker: str) -> Any:
        symbol = ticker.upper()
        if symbol not in universe:
            raise ApiError(404, "unknown_ticker", f"{ticker!r} is not in the FinSight universe")
        return universe[symbol]

    def model_ref(name: str) -> s.ModelRef:
        info = registry_lookup(name) or {}
        return s.ModelRef(name=name, version=info.get("version"), run_id=info.get("run_id"))

    # ----------------------------------------------------------------- routes
    @app.get("/api/stocks", response_model=s.StockListResponse, tags=["stocks"])
    def stocks() -> s.StockListResponse:
        """Every stock FinSight analyses, in configuration order."""
        return s.StockListResponse(
            stocks=[
                s.StockInfo(symbol=t.symbol, name=t.name, sector=t.sector)
                for t in universe.values()
            ]
        )

    @app.get("/api/search", response_model=s.SearchResponse, tags=["stocks"])
    def search(
        q: str = Query(min_length=1, max_length=40), limit: int = Query(10, ge=1, le=10)
    ) -> s.SearchResponse:
        """Type-ahead: symbol prefix, then name, then alias matches."""
        needle = q.strip().lower()
        ranked: list[tuple[int, s.SearchHit]] = []
        for t in universe.values():
            if t.symbol.lower().startswith(needle):
                ranked.append(
                    (
                        0,
                        s.SearchHit(
                            symbol=t.symbol, name=t.name, sector=t.sector, matched="symbol"
                        ),
                    )
                )
            elif needle in t.name.lower():
                ranked.append(
                    (1, s.SearchHit(symbol=t.symbol, name=t.name, sector=t.sector, matched="name"))
                )
            elif any(needle in a.lower() for a in t.aliases):
                ranked.append(
                    (2, s.SearchHit(symbol=t.symbol, name=t.name, sector=t.sector, matched="alias"))
                )
        ranked.sort(key=lambda r: (r[0], r[1].symbol))
        return s.SearchResponse(query=q, results=[hit for _, hit in ranked[:limit]])

    @app.get("/api/stock/{ticker}", response_model=s.StockResponse, tags=["stocks"])
    def stock(ticker: str, period: s.Period = "1M") -> s.StockResponse:
        """Latest price, daily change and history for the period."""
        meta = known(ticker)
        latest = repo.latest_bars(meta.symbol, 2)
        if not latest:
            raise ApiError(404, "no_data", f"no prices stored for {meta.symbol}")
        last = latest[-1]
        prev = latest[0] if len(latest) == 2 else None
        change = last["close"] - prev["close"] if prev else None
        pct = change / prev["close"] * 100 if prev and prev["close"] else None
        direction = "flat" if not change else ("up" if change > 0 else "down")
        bars = repo.history(meta.symbol, period)
        return s.StockResponse(
            symbol=meta.symbol,
            name=meta.name,
            sector=meta.sector,
            period=period,
            overview=s.Overview(
                date=as_date(last["date"]),
                close=round(last["close"], 4),
                previous_close=round(prev["close"], 4) if prev else None,
                change=round(change, 4) if change is not None else None,
                change_pct=round(pct, 4) if pct is not None else None,
                direction=direction,
            ),
            history=[
                s.PricePoint(
                    date=as_date(b["date"]),
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    volume=b.get("volume") or 0.0,
                )
                for b in bars
            ],
            source=str(last.get("source", "unknown")),
        )

    @app.get("/api/prediction/{ticker}", response_model=s.PredictionResponse, tags=["models"])
    def prediction(ticker: str) -> s.PredictionResponse:
        """14-day forecast: stored for today, else computed by ``lstm-serve``, else stale."""
        meta = known(ticker)
        dates, prices = repo.model_prices(meta.symbol, LOOKBACK)
        if len(prices) < LOOKBACK:
            raise ApiError(
                503, "insufficient_history", f"need {LOOKBACK} prices, have {len(prices)}"
            )
        as_of = dates[-1]
        ref = model_ref(config.lstm_name)
        version_key = ref.version or "unknown"
        history = [
            s.ForecastPoint(date=d, price=round(p, 4))
            for d, p in zip(
                dates[-config.history_days_in_forecast :],
                prices[-config.history_days_in_forecast :],
                strict=True,
            )
        ]
        doc, stale = repo.get_forecast(meta.symbol, as_of, version_key), False
        if doc is not None:
            metrics.forecasts.labels("cache").inc()
        else:
            try:
                columns = [f"p_{i:03d}" for i in range(LOOKBACK)]
                record = lstm.predict(columns, [prices])[0]
                values = [float(record[f"h_{i + 1:02d}"]) for i in range(HORIZON)]
            except (ModelUnavailable, KeyError, TypeError, ValueError) as exc:
                fallback = repo.latest_forecast(meta.symbol)
                if fallback is None:
                    raise ApiError(503, "model_unavailable", str(exc)) from exc
                doc, stale = fallback, True
                metrics.forecasts.labels("stale").inc()
            else:
                doc = {
                    "ticker": meta.symbol,
                    "as_of": datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC),
                    "model_version": version_key,
                    "model_name": config.lstm_name,
                    "run_id": ref.run_id,
                    "forecast": [
                        {"date": datetime(d.year, d.month, d.day, tzinfo=UTC), "price": round(v, 4)}
                        for d, v in zip(next_trading_days(as_of, HORIZON), values, strict=True)
                    ],
                    "last_price": prices[-1],
                    "generated_at": clock(),
                }
                repo.save_forecast(doc)
                metrics.forecasts.labels("model").inc()
        return s.PredictionResponse(
            symbol=meta.symbol,
            as_of=as_date(doc["as_of"]),
            horizon=HORIZON,
            history=history,
            forecast=[
                s.ForecastPoint(date=as_date(p["date"]), price=p["price"]) for p in doc["forecast"]
            ],
            model=s.ModelRef(
                name=config.lstm_name, version=doc["model_version"], run_id=doc.get("run_id")
            ),
            stale=stale,
            generated_at=doc["generated_at"],
        )

    @app.get("/api/sentiment/{ticker}", response_model=s.SentimentResponse, tags=["models"])
    def sentiment_weeks(ticker: str, weeks: int = Query(12, ge=1, le=52)) -> s.SentimentResponse:
        """Weekly mean score, label counts and the headlines behind each bar."""
        meta = known(ticker)
        rows = repo.sentiment_weeks(meta.symbol, weeks)
        total = sum(r["article_count"] for r in rows)
        overall = (
            round(sum(r["mean_score"] * r["article_count"] for r in rows) / total, 4)
            if total
            else None
        )
        return s.SentimentResponse(
            symbol=meta.symbol,
            weeks=[
                s.SentimentWeek(
                    week_start=as_date(r["week_start"]),
                    mean_score=r["mean_score"],
                    article_count=r["article_count"],
                    positive=r["positive"],
                    negative=r["negative"],
                    neutral=r["neutral"],
                    top_positive=r.get("top_positive", []),
                    top_negative=r.get("top_negative", []),
                )
                for r in rows
            ],
            overall_mean_score=overall,
            total_articles=total,
            model_version=repo.sentiment_model_version(meta.symbol),
        )

    @app.post("/api/sentiment/analyze", response_model=s.AnalyzeResponse, tags=["models"])
    def analyze(body: s.AnalyzeRequest) -> s.AnalyzeResponse:
        """Score up to 32 texts with the production sentiment model (live)."""
        too_long = [i for i, t in enumerate(body.texts) if len(t) > config.max_text_chars]
        if too_long:
            raise ApiError(
                422, "invalid_request", f"texts {too_long} exceed {config.max_text_chars} chars"
            )
        try:
            records = sentiment.predict(["text"], [[t] for t in body.texts])
        except ModelUnavailable as exc:
            raise ApiError(503, "model_unavailable", str(exc)) from exc
        return s.AnalyzeResponse(
            results=[
                s.AnalyzeResult(
                    text=t, label=r["label"], confidence=r["confidence"], score=r["score"]
                )
                for t, r in zip(body.texts, records, strict=True)
            ],
            model=model_ref(config.sentiment_name),
        )

    @app.get("/api/portfolio", response_model=s.PortfolioResponse, tags=["portfolio"])
    def portfolio() -> s.PortfolioResponse:
        """Latest allocation, its in-sample baselines and the walk-forward verdict."""
        doc = repo.latest_portfolio()
        if doc is None:
            raise ApiError(
                404, "no_portfolio", "no optimisation result yet (dvc repro optimize_portfolio)"
            )
        in_sample, wf = doc["in_sample"], doc["walk_forward"]
        return s.PortfolioResponse(
            as_of=as_date(doc["as_of"]),
            window_start=date.fromisoformat(doc["window_start"]),
            window_end=date.fromisoformat(doc["window_end"]),
            window_days=doc["window_days"],
            risk_free_rate=doc["risk_free_rate"],
            grid_size=doc["grid_size"],
            fallback_used=doc["fallback_used"],
            allocation=doc["allocation"],
            equal_weight=in_sample["equal_weight"],
            min_volatility=in_sample["min_volatility"],
            out_of_sample=wf["performance"],
            beats_equal_weight_out_of_sample=wf["beats_equal_weight_sharpe"],
        )

    @app.get("/api/health", response_model=s.HealthResponse, tags=["operations"])
    def health() -> s.HealthResponse:
        """Status of MongoDB, both model servers and the registry."""
        deps: dict[str, s.DependencyStatus] = {}
        try:
            repo.ping()
            deps["mongodb"] = s.DependencyStatus(ok=True)
        except Exception as exc:  # noqa: BLE001
            deps["mongodb"] = s.DependencyStatus(ok=False, detail=exc.__class__.__name__)
        for name, client in (("lstm-serve", lstm), ("sentiment-serve", sentiment)):
            ok, detail = client.ping()
            deps[name] = s.DependencyStatus(ok=ok, detail=detail)
        registry_lookup(config.lstm_name)
        deps["mlflow-registry"] = s.DependencyStatus(ok=registry_lookup.reachable)
        for name, status in deps.items():
            metrics.dependency_up.labels(name).set(1 if status.ok else 0)
        overall = "ok" if all(d.ok for d in deps.values()) else "degraded"
        return s.HealthResponse(status=overall, dependencies=deps, version=finsight.__version__)

    @app.get("/api/model-info", response_model=s.ModelInfoResponse, tags=["operations"])
    def model_info() -> s.ModelInfoResponse:
        """Production model versions and their evaluation tags."""
        models = []
        for name in (config.lstm_name, config.sentiment_name):
            info = registry_lookup(name) or {}
            models.append(
                s.ModelInfo(
                    name=name,
                    version=info.get("version"),
                    run_id=info.get("run_id"),
                    tags={k: str(v) for k, v in (info.get("tags") or {}).items()},
                )
            )
        return s.ModelInfoResponse(models=models, registry_reachable=registry_lookup.reachable)

    scheduler = scheduler or SchedulerClient(config.scheduler_url, config.scheduler_token)
    app.include_router(build_router(db, scheduler, ApiError, clock))
    app.include_router(
        build_monitoring_router(db, config.alertmanager_url, ApiError, **monitoring_kwargs)
    )
    last_check = {"at": 0.0}

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        # Refresh finsight_dependency_up on scrapes (at most every 10 s), so the
        # model-server alerts work even when nobody has the portal open.
        if time.monotonic() - last_check["at"] > 10:
            last_check["at"] = time.monotonic()
            health()
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
