"""Structured (JSON-lines) logging shared by every FinSight process.

Why JSON: each log line is one machine-parseable object, so logs from all
containers can be grepped, filtered by ``run_id`` and later shipped to any log
store without regexes. A ``run_id`` set via :func:`bind_run_id` is attached to
every record emitted in that context, which is what lets the pipeline console
(Day 7) show all log lines for one pipeline run.

Usage::

    from finsight.common.logging import configure_logging, get_logger, bind_run_id

    configure_logging(service="ingestion")
    log = get_logger(__name__)
    with bind_run_id("a1b2c3"):
        log.info("fetched prices", extra={"ticker": "AAPL", "rows": 250})
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import logging.handlers
import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)
_service_name = "finsight"

# Attributes present on every LogRecord; anything else passed through `extra=`
# is treated as structured context and emitted as its own JSON key.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render a :class:`logging.LogRecord` as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record, its ``extra`` context and any exception info."""
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": _service_name,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        run_id = _run_id.get()
        if run_id:
            payload["run_id"] = run_id
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    service: str, level: str | None = None, log_dir: str | os.PathLike[str] | None = None
) -> None:
    """Configure the root logger once per process.

    Logs go to stdout (collected by ``docker logs``) and to a size-rotated file
    ``<log_dir>/<service>.log`` (5 x 5 MB) for post-mortem debugging.

    Args:
        service: Name stamped on every record (e.g. ``"ingestion"``).
        level: Log level name; defaults to ``$LOG_LEVEL`` or ``INFO``.
        log_dir: Directory for the rotating file; defaults to ``$LOG_DIR`` or
            ``logs``. File logging is skipped (with a warning) if the directory
            is not writable, so logging can never crash the application.
    """
    global _service_name
    _service_name = service
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level_name)
    formatter = JsonFormatter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    directory = Path(log_dir or os.getenv("LOG_DIR", "logs"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            directory / f"{service}.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # read-only FS, permissions... keep stdout logging
        root.warning("file logging disabled", extra={"log_dir": str(directory), "error": str(exc)})

    # Third-party libraries are noisy at INFO.
    for noisy in ("urllib3", "pymongo", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (thin wrapper kept for a single import site)."""
    return logging.getLogger(name)


@contextlib.contextmanager
def bind_run_id(run_id: str) -> Iterator[None]:
    """Attach ``run_id`` to every log record emitted inside the ``with`` block."""
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)
