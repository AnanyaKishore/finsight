"""ETL throughput benchmark: how does the Spark pipeline scale with data volume?

    python -m finsight.etl.benchmark --factors 1 10 50

Runs the full ``process`` transformations on the real snapshot replicated
``k`` times (synthetic tickers ``AAPL_1``, ``AAPL_2`` ...), writes to a temporary
directory (never touching DVC outputs) and records rows/s per factor in
``reports/etl_benchmark.json``. Not a DVC stage: run it on demand for the
report, the Grafana panel and the viva ("what is your pipeline throughput?").

Expect throughput to *rise* with volume: at 1x the fixed per-job overhead of
Spark (task scheduling, JVM warm-up) dominates; larger inputs amortise it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from finsight.common.config import get_settings
from finsight.common.logging import configure_logging, get_logger
from finsight.etl.process import run_etl
from finsight.etl.spark_session import build_spark

log = get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark for each replication factor."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 10, 50])
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("etl-benchmark", settings.log_level, settings.root / "logs")
    etl, export = settings.section("etl"), settings.section("export")

    spark, startup = build_spark(etl, app_name="finsight-etl-benchmark")
    results = []
    try:
        # Warm-up run so JIT compilation does not penalise the first factor.
        with tempfile.TemporaryDirectory() as tmp:
            run_etl(spark, etl, export, settings.root, output_root=Path(tmp), replicate=1)
        for factor in args.factors:
            with tempfile.TemporaryDirectory() as tmp:
                metrics = run_etl(
                    spark, etl, export, settings.root, output_root=Path(tmp), replicate=factor
                )
            row = {
                "factor": factor,
                "rows_in": metrics["totals"]["rows_in"],
                "processing_seconds": metrics["totals"]["processing_seconds"],
                "rows_per_s": metrics["totals"]["rows_per_s"],
                "tickers": metrics["outputs"]["tickers"],
                "steps": {s["step"]: s["seconds"] for s in metrics["steps"]},
            }
            results.append(row)
            print(
                f"x{factor:<4} rows_in={row['rows_in']:>9,}  "
                f"{row['processing_seconds']:>7.2f}s  {row['rows_per_s']:>10,.0f} rows/s"
            )
    finally:
        spark.stop()

    report = {
        "spark": {
            "master": etl["spark_master"],
            "driver_memory": etl["driver_memory"],
            "shuffle_partitions": int(etl["shuffle_partitions"]),
            "startup_seconds": startup,
        },
        "results": results,
    }
    path = settings.root / "reports" / "etl_benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
