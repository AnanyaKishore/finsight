"""Drift drill: make the monitor *see* drift for a while, without touching data.

    python -m finsight.monitoring.drill start --minutes 30   # default scale 4x
    python -m finsight.monitoring.drill status
    python -m finsight.monitoring.drill stop

While a drill is active the monitor multiplies the recent window of daily
returns by ``--scale`` (a volatility shock) and shifts recent sentiment by
``--sentiment-shift`` before computing PSI/KS. Stored prices and news are
never modified; the report and the ``finsight_drift_drill_active`` gauge say a
drill is running, and it ends by itself at ``until``. This exercises the whole
loop: drift gauge -> Prometheus alert -> Alertmanager -> scheduler /retrain ->
retrain -> promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta

from finsight.common.config import get_settings
from finsight.common.mongo import MONITORING_DRILL, get_client


def main(argv: list[str] | None = None) -> int:
    """Start, stop or show the drill."""
    parser = argparse.ArgumentParser(description="FinSight drift drill")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="pretend the recent data drifted")
    start.add_argument("--minutes", type=int, default=30)
    start.add_argument("--scale", type=float, default=4.0, help="multiply recent returns")
    start.add_argument("--sentiment-shift", type=float, default=-0.5)
    sub.add_parser("stop", help="end the drill now")
    sub.add_parser("status", help="show the drill")
    args = parser.parse_args(argv)

    settings = get_settings()
    client = get_client(settings)
    collection = client[settings.mongo_db][MONITORING_DRILL]
    now = datetime.now(UTC)
    if args.command == "start":
        if not 1 <= args.minutes <= 240 or not 1 < args.scale <= 20:
            parser.error("--minutes must be 1-240 and --scale between 1 and 20")
        doc = {
            "_id": "drill",
            "started_at": now,
            "until": now + timedelta(minutes=args.minutes),
            "scale": args.scale,
            "sentiment_shift": args.sentiment_shift,
        }
        collection.replace_one({"_id": "drill"}, doc, upsert=True)
        print(f"drift drill active until {doc['until']:%H:%M} UTC (scale {args.scale}x); "
              "the monitor picks it up within monitoring.interval_s")  # fmt: skip
    elif args.command == "stop":
        collection.delete_one({"_id": "drill"})
        print("drift drill stopped")
    else:
        doc = collection.find_one({"_id": "drill"}, {"_id": 0})
        active = bool(doc) and doc["until"].replace(tzinfo=UTC) > now
        print(json.dumps({"active": active, **(doc or {})}, default=str, indent=2))
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
