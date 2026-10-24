"""Run the backend: ``python -m finsight.api [--host 0.0.0.0] [--port 8000]``."""

from __future__ import annotations

import argparse
import os

import uvicorn

from finsight.common.config import get_settings
from finsight.common.logging import configure_logging


def main() -> None:
    """Start uvicorn with the app factory (one worker: the laptop's memory budget)."""
    parser = argparse.ArgumentParser(description="FinSight API")
    parser.add_argument("--host", default=os.getenv("API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    args = parser.parse_args()
    settings = get_settings()
    configure_logging("backend", settings.log_level, os.getenv("LOG_DIR") or None)
    uvicorn.run(
        "finsight.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
        log_config=None,  # keep our JSON logging
        access_log=False,  # requests are logged (with request IDs) by our middleware
    )


if __name__ == "__main__":
    main()
