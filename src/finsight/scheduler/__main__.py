"""Scheduler entry point: ``python -m finsight.scheduler``.

Starts the background loop (daily run + catch-up) and the HTTP API. Runs in
the ``scheduler`` compose service, with the project directory mounted at
``/app`` so that ``dvc repro`` works on the real repository.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import uvicorn

from finsight.common.config import get_settings, usable_secret
from finsight.common.logging import configure_logging, get_logger
from finsight.common.mongo import ensure_indexes, get_client
from finsight.scheduler.app import create_app
from finsight.scheduler.service import SchedulerService
from finsight.scheduler.store import ExecutionStore

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Start the loop and serve the API until stopped."""
    parser = argparse.ArgumentParser(description="FinSight scheduler")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 -- inside the container network
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-loop", action="store_true", help="API only: no scheduled runs")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("scheduler", settings.log_level, settings.root / "logs")
    client = get_client(settings, timeout_ms=10000)
    db = client[settings.mongo_db]
    ensure_indexes(db)
    service = SchedulerService(
        root=settings.root,
        params=settings.section("scheduler"),
        store=ExecutionStore(db),
        tickers=[t.symbol for t in settings.tickers],
        backend_url=os.getenv("BACKEND_INTERNAL_URL", "http://localhost:8000"),
        lstm_url=os.getenv("LSTM_SERVE_URL", "http://localhost:5001"),
    )
    service.recover()
    service.refresh_dag_async()
    stop = threading.Event()
    if not args.no_loop:
        threading.Thread(target=service.loop, args=(stop,), name="loop", daemon=True).start()
        log.info("scheduler loop started", extra=service.status()["schedule"])
    token = usable_secret(os.getenv("SCHEDULER_TOKEN"))
    if not token:
        log.warning(
            "SCHEDULER_TOKEN missing, a placeholder or shorter than 16 characters: "
            "runs start on schedule only, not from the console or the alert webhook"
        )
    try:
        uvicorn.run(create_app(service, token), host=args.host, port=args.port, log_level="warning")
    finally:
        stop.set()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
