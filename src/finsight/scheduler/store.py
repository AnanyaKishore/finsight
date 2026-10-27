"""MongoDB persistence for scheduler runs (``pipeline_executions``) and the DAG.

One document per run::

    {run_id, kind: "daily"|"retrain", trigger: "schedule"|"catch-up"|"console"|
     "webhook"|"api", reason, status: "running"|"success"|"partial"|"failed"|
     "interrupted", started_at, finished_at, duration_s, git_commit,
     current_step, steps: [{name, label, status, started_at, finished_at,
     duration_s, exit_code, detail, error, log_tail, metrics}], summary: {...}}

Each DVC stage the run starts also writes its own record to ``pipeline_runs``
with ``pipeline_run_id`` = this ``run_id`` (see ``finsight.common.runs``), so
the console can drill down from a run to its stages.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pymongo import DESCENDING

from finsight.common.mongo import PIPELINE_DAG, PIPELINE_EXECUTIONS


class ExecutionStore:
    """Reads and writes run documents."""

    def __init__(self, db: Any) -> None:
        """Wrap a pymongo (or mongomock) database."""
        self.db = db
        self.runs = db[PIPELINE_EXECUTIONS]

    def create(self, doc: dict[str, Any]) -> None:
        """Insert a new run in state ``running``."""
        self.runs.insert_one({**doc})

    def update(self, run_id: str, fields: dict[str, Any]) -> None:
        """Set top-level fields of a run."""
        self.runs.update_one({"run_id": run_id}, {"$set": fields})

    def set_step(self, run_id: str, index: int, step: dict[str, Any]) -> None:
        """Replace step ``index`` of a run (steps are pre-filled as ``pending``)."""
        self.runs.update_one(
            {"run_id": run_id}, {"$set": {f"steps.{index}": step, "current_step": step["name"]}}
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        """One run without Mongo's ``_id``."""
        return self.runs.find_one({"run_id": run_id}, {"_id": 0})

    def running(self) -> dict[str, Any] | None:
        """The run in progress, if any."""
        return self.runs.find_one(
            {"status": "running"}, {"_id": 0}, sort=[("started_at", DESCENDING)]
        )

    def last_daily_start(self) -> datetime | None:
        """Start time of the newest daily run (any outcome): one attempt per day."""
        doc = self.runs.find_one(
            {"kind": "daily", "trigger": {"$in": ["schedule", "catch-up"]}},
            {"started_at": 1},
            sort=[("started_at", DESCENDING)],
        )
        return _aware(doc["started_at"]) if doc else None

    def last_finished(self, kind: str | None = None) -> dict[str, Any] | None:
        """The newest finished run (optionally of one kind)."""
        query: dict[str, Any] = {"status": {"$nin": ["running"]}}
        if kind:
            query["kind"] = kind
        return self.runs.find_one(query, {"_id": 0}, sort=[("started_at", DESCENDING)])

    def mark_interrupted(self, now: datetime, older_than: timedelta | None = None) -> int:
        """Close ``running`` records left behind by a crash or a restart.

        With ``older_than=None`` every running record is closed (used at
        start-up: nothing can be running before the scheduler itself starts).
        """
        query: dict[str, Any] = {"status": "running"}
        if older_than is not None:
            query["started_at"] = {"$lt": now - older_than}
        result = self.runs.update_many(
            query,
            {
                "$set": {
                    "status": "interrupted",
                    "finished_at": now,
                    "error": "the scheduler stopped before this run finished",
                }
            },
        )
        return int(result.modified_count)

    def save_dag(self, dag: dict[str, Any]) -> None:
        """Store the latest DAG snapshot (single document)."""
        self.db[PIPELINE_DAG].replace_one({"_id": "dvc"}, {"_id": "dvc", **dag}, upsert=True)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)
