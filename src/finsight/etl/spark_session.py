"""SparkSession factory tuned for an 8 GB / 12-core laptop.

Local mode runs the driver and executors in one JVM: ``local[4]`` uses four
cores (four tasks in parallel), and ``spark.driver.memory`` caps the JVM heap.
Shuffle partitions are cut from Spark's default 200 to 8 (Lecture 7: "tune for
YOUR cluster size") because our data fits in a handful of partitions and 200
near-empty tasks would cost more in scheduling than in work.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from finsight.common.logging import get_logger

log = get_logger(__name__)


def build_spark(params: dict[str, Any], app_name: str = "finsight-etl") -> tuple[Any, float]:
    """Create (or reuse) a SparkSession.

    Args:
        params: The ``etl`` section of ``params.yaml``.
        app_name: Name shown in the Spark UI.

    Returns:
        ``(spark, startup_seconds)`` -- JVM start-up time is reported separately
        from processing time so the throughput figures are honest.
    """
    from pyspark.sql import SparkSession  # noqa: PLC0415 -- heavy import, only when needed

    # Keep PySpark workers on the same interpreter as the driver (conda env).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    started = time.monotonic()
    spark = (
        SparkSession.builder.master(str(params["spark_master"]))
        .appName(app_name)
        .config("spark.driver.memory", str(params["driver_memory"]))
        .config("spark.sql.shuffle.partitions", str(params["shuffle_partitions"]))
        .config("spark.ui.enabled", str(bool(params.get("spark_ui", False))).lower())
        # Dates are calendar dates and timestamps are UTC instants: never let the
        # laptop's Asia/Kolkata time zone shift them.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # Small data: adaptive execution would re-plan partitions between runs.
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    startup = round(time.monotonic() - started, 3)
    log.info("spark session ready", extra={"master": params["spark_master"], "startup_s": startup})
    return spark, startup
