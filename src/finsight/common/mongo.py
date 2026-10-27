"""MongoDB access: client factory, collection names and indexes.

MongoDB is the operational store of FinSight (proposal, AIOps step 1):

=====================  =====================================================
collection             contents
=====================  =====================================================
``prices``             one document per (ticker, trading day): OHLCV + source
``news``               one document per (ticker, article)
``pipeline_runs``      one document per pipeline/ingestion run (console UI)
``api_usage``          request ledger per provider per month (quota guard)
=====================  =====================================================

Later days add ``predictions``, ``portfolio``, ``model_quality``.
Unique indexes make every write an idempotent *upsert*, so re-running a job
never duplicates data -- a prerequisite for safe automatic retries.
"""

from __future__ import annotations

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from finsight.common.config import Settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import get_logger

log = get_logger(__name__)

PRICES = "prices"
NEWS = "news"
PIPELINE_RUNS = "pipeline_runs"
API_USAGE = "api_usage"
# Day 4: analysis results published for the API (written by DVC stages).
NEWS_SENTIMENT = "news_sentiment"  # one document per scored article
SENTIMENT_WEEKLY = "sentiment_weekly"  # one document per (ticker, week)
PORTFOLIOS = "portfolios"  # one document per optimisation date
# Day 5: forecasts served by the API (one per ticker / origin / model version).
FORECASTS = "forecasts"
# Day 7: scheduler runs (one document per daily / retrain run) and the DVC DAG.
PIPELINE_EXECUTIONS = "pipeline_executions"
PIPELINE_DAG = "pipeline_dag"  # single document, refreshed after every run


def get_client(settings: Settings, timeout_ms: int = 5000) -> MongoClient:
    """Create a timezone-aware client and fail fast if the server is unreachable.

    Raises:
        FinSightError: If MongoDB cannot be reached within ``timeout_ms``.
    """
    client: MongoClient = MongoClient(
        settings.mongo_uri, serverSelectionTimeoutMS=timeout_ms, tz_aware=True
    )
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        raise FinSightError(
            f"MongoDB unreachable at {settings.mongo_host}:{settings.mongo_port} "
            f"({exc.__class__.__name__}). Is 'docker compose up -d mongo' running?"
        ) from exc
    return client


def ensure_indexes(db: Database) -> None:
    """Create the unique / lookup indexes (idempotent; safe to call on every start)."""
    db[PRICES].create_index([("ticker", ASCENDING), ("date", ASCENDING)], unique=True)
    db[PRICES].create_index([("date", DESCENDING)])
    db[NEWS].create_index([("ticker", ASCENDING), ("news_id", ASCENDING)], unique=True)
    db[NEWS].create_index([("ticker", ASCENDING), ("published_at", DESCENDING)])
    db[PIPELINE_RUNS].create_index([("run_id", ASCENDING)], unique=True)
    db[PIPELINE_RUNS].create_index([("pipeline", ASCENDING), ("started_at", DESCENDING)])
    db[API_USAGE].create_index([("provider", ASCENDING), ("period", ASCENDING)], unique=True)
    db[NEWS_SENTIMENT].create_index([("ticker", ASCENDING), ("news_id", ASCENDING)], unique=True)
    db[NEWS_SENTIMENT].create_index([("ticker", ASCENDING), ("published_at", DESCENDING)])
    db[SENTIMENT_WEEKLY].create_index(
        [("ticker", ASCENDING), ("week_start", ASCENDING)], unique=True
    )
    db[PORTFOLIOS].create_index([("as_of", DESCENDING)], unique=True)
    db[FORECASTS].create_index(
        [("ticker", ASCENDING), ("as_of", ASCENDING), ("model_version", ASCENDING)], unique=True
    )
    db[FORECASTS].create_index([("ticker", ASCENDING), ("as_of", DESCENDING)])
    db[PIPELINE_RUNS].create_index([("pipeline_run_id", ASCENDING)])
    db[PIPELINE_EXECUTIONS].create_index([("run_id", ASCENDING)], unique=True)
    db[PIPELINE_EXECUTIONS].create_index([("started_at", DESCENDING)])
    db[PIPELINE_EXECUTIONS].create_index([("kind", ASCENDING), ("started_at", DESCENDING)])
    log.debug("mongo indexes ensured", extra={"db": db.name})
