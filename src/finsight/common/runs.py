"""Run records for DVC pipeline stages (feeds the Pipeline Console, Day 7).

Every stage entry point wraps its work in :func:`stage_run`::

    with stage_run("validate") as run:
        ...
        run.metrics["checks_failed"] = 0

which writes one document to ``pipeline_runs``::

    {run_id, pipeline: "dvc", stage: "validate", status: "success"|"failed",
     started_at, finished_at, duration_s, metrics: {...}, error, git_commit,
     pipeline_run_id}

``pipeline_run_id`` (Day 7) links the stage to the scheduler run that started
it (``FINSIGHT_PIPELINE_RUN_ID``, set by the scheduler); it is ``None`` when a
stage is run by hand.

Recording is *best effort* (graceful degradation): if MongoDB is unreachable,
or recording is disabled with ``FINSIGHT_RUN_RECORDING=0`` (unit tests), the
stage still runs and only a warning is logged. A stage must never fail because
its bookkeeping failed.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from finsight.common.config import Settings, get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import bind_run_id, get_logger

log = get_logger(__name__)


@dataclass
class StageRun:
    """Mutable record of one stage execution."""

    stage: str
    pipeline: str = "dvc"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    error: str | None = None
    duration_s: float = 0.0

    def document(self) -> dict[str, Any]:
        """Mongo representation."""
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC),
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "error": self.error,
            "git_commit": git_commit(),
            "pipeline_run_id": os.getenv("FINSIGHT_PIPELINE_RUN_ID") or None,
        }


def git_commit() -> str | None:
    """Short hash of HEAD (with ``-dirty`` if the tree has changes), or ``None``."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 -- fixed command
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def recording_enabled() -> bool:
    """Whether run records should be written (``FINSIGHT_RUN_RECORDING``, default on)."""
    return os.getenv("FINSIGHT_RUN_RECORDING", "1").lower() not in {"0", "false", "no"}


def _persist(run: StageRun, settings: Settings | None) -> None:
    if not recording_enabled():
        return
    try:
        from finsight.common.mongo import PIPELINE_RUNS, get_client  # noqa: PLC0415

        resolved = settings or get_settings()
        client = get_client(resolved, timeout_ms=1500)
        try:
            client[resolved.mongo_db][PIPELINE_RUNS].insert_one(run.document())
        finally:
            client.close()
    except (
        FinSightError,
        Exception,
    ) as exc:  # noqa: BLE001 -- bookkeeping must never break a stage
        log.warning("run record not saved", extra={"stage": run.stage, "error": str(exc)})


@contextlib.contextmanager
def stage_run(stage: str, settings: Settings | None = None) -> Iterator[StageRun]:
    """Time a stage, capture failures, and persist a run record.

    Exceptions are re-raised after recording, so DVC still sees the failure
    (non-zero exit) and stops the pipeline.
    """
    run = StageRun(stage=stage)
    started = time.monotonic()
    with bind_run_id(run.run_id):
        log.info("stage started", extra={"stage": stage})
        try:
            yield run
            run.status = "success"
        except BaseException as exc:
            run.status = "failed"
            run.error = f"{exc.__class__.__name__}: {exc}"
            raise
        finally:
            run.duration_s = round(time.monotonic() - started, 3)
            log.info(
                "stage finished",
                extra={"stage": stage, "status": run.status, "duration_s": run.duration_s},
            )
            _persist(run, settings)
