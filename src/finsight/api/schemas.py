"""Request / response models: the backend's I/O contract (and the LLD's endpoint spec).

FastAPI turns these into the OpenAPI schema served at ``/openapi.json`` and the
interactive docs at ``/docs``. Every analytical response carries a
``disclaimer``: FinSight is analytical insight, not investment advice.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "For information and education only. Not investment advice; past performance "
    "and model forecasts do not guarantee future results."
)

Period = Literal["1W", "1M", "3M", "6M", "1Y", "ALL"]


class ErrorBody(BaseModel):
    """Uniform error payload."""

    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Envelope for every 4xx / 5xx response."""

    error: ErrorBody


# ------------------------------------------------------------------ search
class SearchHit(BaseModel):
    """One type-ahead suggestion."""

    symbol: str
    name: str
    sector: str
    matched: str = Field(description="what matched: symbol, name or alias")


class SearchResponse(BaseModel):
    """``GET /api/search``."""

    query: str
    results: list[SearchHit]


class StockInfo(BaseModel):
    """One stock of the universe (home-page list)."""

    symbol: str
    name: str
    sector: str


class StockListResponse(BaseModel):
    """``GET /api/stocks``: the whole universe, in configuration order."""

    stocks: list[StockInfo]


# ------------------------------------------------------------------- stock
class PricePoint(BaseModel):
    """One end-of-day bar."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class Overview(BaseModel):
    """Latest price with the day's change (green/red indicator in the UI)."""

    date: date
    close: float
    previous_close: float | None
    change: float | None
    change_pct: float | None
    direction: Literal["up", "down", "flat"]


class StockResponse(BaseModel):
    """``GET /api/stock/{ticker}``."""

    symbol: str
    name: str
    sector: str
    period: Period
    overview: Overview
    history: list[PricePoint]
    source: str = Field(description="price provider of the latest bar")


# -------------------------------------------------------------- prediction
class ForecastPoint(BaseModel):
    """Price on one date (history or forecast)."""

    date: date
    price: float


class ModelRef(BaseModel):
    """Which registered model produced a result."""

    name: str
    version: str | None
    run_id: str | None = None


class PredictionResponse(BaseModel):
    """``GET /api/prediction/{ticker}``: history + dashed 14-day forecast."""

    symbol: str
    as_of: date = Field(description="last observed trading day (forecast origin)")
    horizon: int
    history: list[ForecastPoint]
    forecast: list[ForecastPoint]
    model: ModelRef
    stale: bool = Field(description="true if served from an earlier forecast (model down)")
    generated_at: datetime
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------- sentiment
class Headline(BaseModel):
    """An explanatory headline for a weekly bar."""

    headline: str
    score: float
    source: str | None = None
    url: str | None = None
    published_at: str | None = None


class SentimentWeek(BaseModel):
    """One bar of the weekly sentiment chart."""

    week_start: date
    mean_score: float = Field(ge=-1, le=1)
    article_count: int
    positive: int
    negative: int
    neutral: int
    top_positive: list[Headline]
    top_negative: list[Headline]


class SentimentResponse(BaseModel):
    """``GET /api/sentiment/{ticker}``."""

    symbol: str
    weeks: list[SentimentWeek]
    overall_mean_score: float | None
    total_articles: int
    model_version: str | None
    disclaimer: str = DISCLAIMER


class AnalyzeRequest(BaseModel):
    """``POST /api/sentiment/analyze`` body."""

    texts: list[str] = Field(min_length=1, max_length=32)


class AnalyzeResult(BaseModel):
    """Model output for one text."""

    text: str
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    score: float


class AnalyzeResponse(BaseModel):
    """``POST /api/sentiment/analyze`` response."""

    results: list[AnalyzeResult]
    model: ModelRef


# --------------------------------------------------------------- portfolio
class Allocation(BaseModel):
    """Weights and annualised statistics of one portfolio."""

    method: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float


class PortfolioResponse(BaseModel):
    """``GET /api/portfolio``."""

    as_of: date
    window_start: date
    window_end: date
    window_days: int
    risk_free_rate: float
    grid_size: int
    fallback_used: bool
    allocation: Allocation
    equal_weight: Allocation
    min_volatility: Allocation
    out_of_sample: dict[str, dict[str, float]] = Field(
        description="walk-forward performance per strategy (annualised)"
    )
    beats_equal_weight_out_of_sample: bool
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------- operations
class DependencyStatus(BaseModel):
    """Health of one dependency."""

    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    """``GET /api/health``: ``ok`` if everything is up, ``degraded`` otherwise."""

    status: Literal["ok", "degraded"]
    dependencies: dict[str, DependencyStatus]
    version: str


class ModelInfo(BaseModel):
    """Production model as recorded in the MLflow registry."""

    name: str
    version: str | None
    run_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    """``GET /api/model-info``."""

    models: list[ModelInfo]
    registry_reachable: bool


# ---------------------------------------------------------------- pipeline (Day 7)
RunStatus = Literal["running", "success", "partial", "failed", "interrupted"]


class PipelineStep(BaseModel):
    """One step of a scheduler run."""

    name: str
    label: str
    status: Literal["pending", "running", "success", "partial", "failed", "skipped"]
    critical: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    exit_code: int | None = None
    detail: str | None = None
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)


class PipelineRunSummary(BaseModel):
    """A run in the history list (no log output)."""

    run_id: str
    kind: Literal["daily", "retrain"]
    trigger: str
    reason: str = ""
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_s: float | None = None
    current_step: str | None = None
    git_commit: str | None = None
    failed_steps: list[str] = Field(default_factory=list)
    summary: dict[str, float | int | str | None] = Field(default_factory=dict)


class PipelineRunList(BaseModel):
    """``GET /api/pipeline/runs``."""

    runs: list[PipelineRunSummary]


class StageRecord(BaseModel):
    """A DVC stage (or the ingestion job) executed by a run."""

    stage: str
    status: str
    started_at: datetime | None = None
    duration_s: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    git_commit: str | None = None


class PipelineRunDetail(PipelineRunSummary):
    """``GET /api/pipeline/runs/{run_id}``: steps with log tails + stage records."""

    steps: list[PipelineStep]
    stages: list[StageRecord]
    error: str | None = None


class DagNode(BaseModel):
    """A DVC stage with its most recent recorded execution."""

    name: str
    status: Literal["success", "failed", "running", "never"]
    stale: bool = Field(description="inputs changed since the stage last ran (dvc status)")
    last_run_at: datetime | None = None
    duration_s: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class DagResponse(BaseModel):
    """``GET /api/pipeline/dag``."""

    nodes: list[DagNode]
    edges: list[tuple[str, str]]
    generated_at: datetime


class PipelineStats(BaseModel):
    """``GET /api/pipeline/stats``: the last ``days`` days of runs."""

    days: int
    runs: int
    by_status: dict[str, int]
    success_rate: float | None
    median_duration_s: dict[str, float]
    latest_throughput: dict[str, float]


class SchedulerStatus(BaseModel):
    """``GET /api/pipeline/status``."""

    reachable: bool
    busy: bool
    running_run_id: str | None = None
    current_step: str | None = None
    next_daily_run: datetime | None = None
    schedule: dict[str, Any] = Field(default_factory=dict)
    triggers_enabled: bool = False
    detail: str = ""


class TriggerRequest(BaseModel):
    """``POST /api/pipeline/trigger`` body."""

    kind: Literal["daily", "retrain"]
    reason: str = Field(default="started from the console", max_length=200)


class TriggerResponse(BaseModel):
    """A run was accepted by the scheduler."""

    run_id: str
    kind: str


class ActiveAlert(BaseModel):
    """An alert that is firing in Alertmanager."""

    name: str
    severity: str
    action: str | None = None
    summary: str = ""
    starts_at: datetime | None = None


class MonitoringResponse(BaseModel):
    """``GET /api/monitoring``: report sections as computed by the scheduler."""

    report: dict[str, Any] | None = Field(
        description="generated_at, drill, returns, sentiment, forecast, freshness, quality, models"
    )
    alerts: list[ActiveAlert]
    alertmanager_reachable: bool
    detail: str = ""
