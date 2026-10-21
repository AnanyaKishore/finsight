"""MongoDB persistence for ingestion (Repository pattern).

All writes are bulk *upserts* keyed on the unique indexes from
:func:`finsight.common.mongo.ensure_indexes`, which makes every ingestion run
idempotent: re-running a day, retrying after a crash, or overlapping windows
never creates duplicates.

Only data fields go into ``$set`` (no per-write timestamp), so MongoDB's
``modified_count`` means "the provider actually revised a value" and an
identical re-upsert is a no-op. ``ingested_at`` records first sight (lineage).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pymongo import DESCENDING, UpdateOne
from pymongo.database import Database

from finsight.common.mongo import NEWS, PIPELINE_RUNS, PRICES
from finsight.ingestion.models import NewsArticle, PriceBar


@dataclass(frozen=True)
class UpsertResult:
    """Outcome of a bulk upsert."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def written(self) -> int:
        """Documents that were inserted or changed."""
        return self.inserted + self.updated


class IngestionRepository:
    """Read/write access to the ingestion collections."""

    def __init__(self, db: Database) -> None:
        """Wrap a Mongo database handle."""
        self.db = db

    # ------------------------------------------------------------------ prices
    def upsert_prices(self, bars: Iterable[PriceBar]) -> UpsertResult:
        """Insert new bars and refresh existing ones (provider revisions win)."""
        now = datetime.now(UTC)
        ops = []
        for bar in bars:
            doc = bar.to_document()
            key = {"ticker": doc.pop("ticker"), "date": doc.pop("date")}
            ops.append(
                UpdateOne(
                    key,
                    {"$set": doc, "$setOnInsert": {"ingested_at": now}},
                    upsert=True,
                )
            )
        return self._bulk(PRICES, ops)

    def last_price_date(self, ticker: str) -> datetime | None:
        """Most recent stored trading day for ``ticker`` (UTC midnight) or ``None``."""
        doc = self.db[PRICES].find_one({"ticker": ticker}, sort=[("date", DESCENDING)])
        return _aware(doc["date"]) if doc else None

    # -------------------------------------------------------------------- news
    def upsert_news(self, articles: Iterable[NewsArticle]) -> UpsertResult:
        """Insert unseen articles; never overwrite sentiment written later (Day 4)."""
        now = datetime.now(UTC)
        ops = []
        for article in articles:
            doc = article.to_document()
            key = {"ticker": doc.pop("ticker"), "news_id": doc.pop("news_id")}
            ops.append(
                UpdateOne(
                    key,
                    {"$set": doc, "$setOnInsert": {"ingested_at": now}},
                    upsert=True,
                )
            )
        return self._bulk(NEWS, ops)

    def last_news_date(self, ticker: str) -> datetime | None:
        """Publication time of the newest stored article for ``ticker``."""
        doc = self.db[NEWS].find_one({"ticker": ticker}, sort=[("published_at", DESCENDING)])
        return _aware(doc["published_at"]) if doc else None

    # -------------------------------------------------------------------- runs
    def start_run(self, record: dict[str, Any]) -> None:
        """Insert a run record in state ``running`` (visible in the console at once)."""
        self.db[PIPELINE_RUNS].insert_one({**record, "status": "running"})

    def finish_run(self, run_id: str, fields: dict[str, Any]) -> None:
        """Complete a run record with its final status and metrics."""
        self.db[PIPELINE_RUNS].update_one({"run_id": run_id}, {"$set": fields})

    def recent_runs(self, pipeline: str, limit: int = 5) -> list[dict[str, Any]]:
        """Newest run records for one pipeline (``_id`` stripped)."""
        cursor = (
            self.db[PIPELINE_RUNS]
            .find({"pipeline": pipeline}, {"_id": 0})
            .sort("started_at", DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    # ------------------------------------------------------------------ status
    def coverage(self) -> dict[str, dict[str, Any]]:
        """Per-ticker row counts and date ranges for prices and news."""
        out: dict[str, dict[str, Any]] = {}
        price_stats = self.db[PRICES].aggregate(
            [
                {
                    "$group": {
                        "_id": "$ticker",
                        "rows": {"$sum": 1},
                        "first": {"$min": "$date"},
                        "last": {"$max": "$date"},
                    }
                }
            ]
        )
        for row in price_stats:
            out.setdefault(row["_id"], {})["prices"] = {
                "rows": row["rows"],
                "first": row["first"],
                "last": row["last"],
            }
        news_stats = self.db[NEWS].aggregate(
            [
                {
                    "$group": {
                        "_id": "$ticker",
                        "rows": {"$sum": 1},
                        "first": {"$min": "$published_at"},
                        "last": {"$max": "$published_at"},
                    }
                }
            ]
        )
        for row in news_stats:
            out.setdefault(row["_id"], {})["news"] = {
                "rows": row["rows"],
                "first": row["first"],
                "last": row["last"],
            }
        return out

    # ---------------------------------------------------------------- helpers
    def _bulk(self, collection: str, ops: list[UpdateOne]) -> UpsertResult:
        if not ops:
            return UpsertResult()
        result = self.db[collection].bulk_write(ops, ordered=False)
        inserted = result.upserted_count
        updated = result.modified_count
        return UpsertResult(
            inserted=inserted, updated=updated, unchanged=len(ops) - inserted - updated
        )


def _aware(value: datetime) -> datetime:
    """Treat naive datetimes from Mongo as UTC (defensive; clients are tz_aware)."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)
