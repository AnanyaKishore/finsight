"""DVC stage ``process``: Spark ETL from the raw snapshot to model-ready data.

Input  (from ``export``):  ``data/raw/prices.parquet``, ``data/raw/news.parquet``
Output (DVC-tracked):      ``data/processed/prices/ticker=*/part-00000.parquet``
                           ``data/processed/news/ticker=*/part-00000.parquet``
                           ``data/processed/news_weekly/part-00000.parquet``
Metrics (Git-tracked):     ``reports/etl_metrics.json`` (rows, seconds, rows/s per step)

Transformations
---------------
1. **Model price** -- per ticker, ``adj_close`` if every bar has it, else ``close``
   (same rule as the validation stage; see ``finsight.data.validate.model_price``).
2. **Calendar alignment** -- a (ticker x NYSE trading day) grid is built with a
   *broadcast* join against the small calendar table (Lecture 7: ship the small
   side to every executor instead of shuffling the big one). Missing trading
   days are forward-filled for at most ``max_ffill_days`` consecutive days and
   flagged ``is_filled``; rows on non-trading days are dropped.
3. **Features** -- window functions per ticker: daily log return, rolling mean
   return, volatility, moving average and price-to-MA for each window, plus
   rolling volume. Values are null until a full window is available (no
   partial-window leakage of "warm-up" noise into models or drift statistics).
4. **News** -- drop blank headlines, de-duplicate by id and by normalised
   headline (syndicated copies), add the Monday ``week_start`` key, and
   aggregate weekly article counts per ticker.

Every output is written deterministically (one sorted file per partition with a
fixed name), so an unchanged input reproduces byte-identical outputs and DVC
can skip the downstream training stages.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from finsight.common.config import get_settings
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.common.timeutils import is_trading_day
from finsight.etl.spark_session import build_spark

log = get_logger(__name__)

PRICE_OUTPUT_ORDER = ["ticker", "date"]
NEWS_OUTPUT_ORDER = ["ticker", "published_at", "news_id"]


# --------------------------------------------------------------------------- #
# Throughput bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class StepTimer:
    """Collects per-step timings and row counts for ``etl_metrics.json``."""

    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, started: float, rows_in: int, rows_out: int) -> None:
        """Store one step (``started`` is a ``time.monotonic()`` value)."""
        seconds = round(time.monotonic() - started, 3)
        self.steps.append(
            {
                "step": name,
                "seconds": seconds,
                "rows_in": int(rows_in),
                "rows_out": int(rows_out),
                "rows_per_s": round(rows_in / seconds, 1) if seconds > 0 else None,
            }
        )
        log.info("etl step", extra=self.steps[-1])


# --------------------------------------------------------------------------- #
# Pure transformations (unit-tested with a local SparkSession)
# --------------------------------------------------------------------------- #
def trading_calendar(spark: Any, start: date, end: date) -> Any:
    """One-column DataFrame of NYSE trading days in ``[start, end]``."""
    days, current = [], start
    while current <= end:
        if is_trading_day(current):
            days.append((current,))
        current += timedelta(days=1)
    return spark.createDataFrame(days, "date DATE")


def with_model_price(prices: Any) -> Any:
    """Add ``price``: ``adj_close`` when complete for the ticker, else ``close``."""
    from pyspark.sql import Window  # noqa: PLC0415
    from pyspark.sql import functions as F  # noqa: PLC0415

    per_ticker = Window.partitionBy("ticker")
    missing_adj = F.sum(F.when(F.col("adj_close").isNull(), 1).otherwise(0)).over(per_ticker)
    return prices.withColumn(
        "price", F.when(missing_adj == 0, F.col("adj_close")).otherwise(F.col("close"))
    )


def align_to_calendar(prices: Any, calendar: Any, max_ffill_days: int) -> Any:
    """Re-index each ticker onto the trading calendar with bounded forward-fill.

    Each ticker's grid starts at its first observed day and ends at the latest
    day in the snapshot. Gaps of up to ``max_ffill_days`` consecutive trading
    days are filled from the last observation (``is_filled = True``, volume 0);
    longer gaps -- and any leading rows -- are dropped.
    """
    from pyspark.sql import Window  # noqa: PLC0415
    from pyspark.sql import functions as F  # noqa: PLC0415

    bounds = prices.groupBy("ticker").agg(F.min("date").alias("first_date"))
    last_day = prices.agg(F.max("date")).first()[0]
    cal = calendar.where(F.col("date") <= F.lit(last_day))
    grid = bounds.join(F.broadcast(cal), cal["date"] >= bounds["first_date"]).select(
        "ticker", "date"
    )
    joined = grid.join(prices, ["ticker", "date"], "left")

    ordered = Window.partitionBy("ticker").orderBy("date")
    running = ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    joined = joined.withColumn("_idx", F.row_number().over(ordered))
    joined = joined.withColumn("is_filled", F.col("close").isNull())
    joined = joined.withColumn(
        "_last_obs_idx", F.last(F.when(~F.col("is_filled"), F.col("_idx")), True).over(running)
    )
    for column in ("close", "adj_close", "price", "source"):
        joined = joined.withColumn(column, F.last(column, True).over(running))
    filled = F.col("is_filled")
    joined = (
        joined.withColumn("open", F.when(filled, F.col("close")).otherwise(F.col("open")))
        .withColumn("high", F.when(filled, F.col("close")).otherwise(F.col("high")))
        .withColumn("low", F.when(filled, F.col("close")).otherwise(F.col("low")))
        .withColumn("volume", F.when(filled, F.lit(0.0)).otherwise(F.col("volume")))
        .withColumn("fill_gap", F.col("_idx") - F.col("_last_obs_idx"))
    )
    return joined.where(
        F.col("price").isNotNull() & (F.col("fill_gap") <= int(max_ffill_days))
    ).drop("_idx", "_last_obs_idx")


def add_features(frame: Any, windows: Sequence[int]) -> Any:
    """Per-ticker rolling features; null until a full window is available."""
    from pyspark.sql import Window  # noqa: PLC0415
    from pyspark.sql import functions as F  # noqa: PLC0415

    ordered = Window.partitionBy("ticker").orderBy("date")
    frame = frame.withColumn("_n", F.row_number().over(ordered))
    frame = frame.withColumn("log_return", F.log(F.col("price") / F.lag("price").over(ordered)))
    frame = frame.withColumn("day_of_week", F.dayofweek("date"))
    for n in sorted({int(w) for w in windows}):
        rolling = ordered.rowsBetween(-(n - 1), Window.currentRow)
        full_returns = F.col("_n") > n  # n returns need n + 1 prices
        full_prices = F.col("_n") >= n
        frame = (
            frame.withColumn(
                f"ret_mean_{n}", F.when(full_returns, F.avg("log_return").over(rolling))
            )
            .withColumn(
                f"volatility_{n}", F.when(full_returns, F.stddev_samp("log_return").over(rolling))
            )
            .withColumn(f"ma_{n}", F.when(full_prices, F.avg("price").over(rolling)))
            .withColumn(f"price_to_ma_{n}", F.col("price") / F.col(f"ma_{n}") - 1)
            .withColumn(
                f"volume_ma_{n}",
                F.when(
                    full_prices,
                    F.avg(F.when(~F.col("is_filled"), F.col("volume"))).over(rolling),
                ),
            )
        )
    return frame.drop("_n")


def clean_news(news: Any) -> tuple[Any, dict[str, int]]:
    """Drop blank / duplicate articles and add ``week_start`` (Monday, UTC).

    Returns:
        ``(clean_frame, counts)`` where counts holds rows removed per rule.
    """
    from pyspark.sql import Window  # noqa: PLC0415
    from pyspark.sql import functions as F  # noqa: PLC0415

    total = news.count()
    news = news.where(F.trim(F.coalesce(F.col("headline"), F.lit(""))) != "")
    after_blank = news.count()
    news = news.dropDuplicates(["ticker", "news_id"])
    after_id = news.count()
    news = news.withColumn(
        "headline_norm", F.lower(F.trim(F.regexp_replace("headline", r"\s+", " ")))
    )
    first = Window.partitionBy("ticker", "headline_norm").orderBy("published_at", "news_id")
    news = (
        news.withColumn("_rank", F.row_number().over(first))
        .where(F.col("_rank") == 1)
        .drop("_rank", "headline_norm")
    )
    news = news.withColumn("text", F.trim(F.coalesce(F.col("text"), F.col("headline"))))
    news = news.withColumn("week_start", F.to_date(F.date_trunc("week", "published_at")))
    news = news.cache()
    after_headline = news.count()
    return news, {
        "blank_headlines": total - after_blank,
        "duplicate_ids": after_blank - after_id,
        "duplicate_headlines": after_id - after_headline,
    }


def weekly_news(news: Any) -> Any:
    """Article counts per (ticker, week) -- the sentiment chart's x-axis and bar labels."""
    from pyspark.sql import functions as F  # noqa: PLC0415

    return news.groupBy("ticker", "week_start").agg(
        F.count("*").alias("article_count"),
        F.countDistinct("source").alias("distinct_sources"),
        F.min("published_at").alias("first_published_at"),
        F.max("published_at").alias("last_published_at"),
    )


# --------------------------------------------------------------------------- #
# Deterministic output
# --------------------------------------------------------------------------- #
def write_deterministic(
    frame: Any, path: Path, order: Sequence[str], partition_by: str | None = None
) -> int:
    """Write Parquet so identical data always yields identical bytes and names.

    Spark names files ``part-00000-<random uuid>.snappy.parquet`` and adds
    ``_SUCCESS``/``.crc`` side files; DVC would see a "new" output every run.
    We write one sorted file per partition into a temp directory, delete the
    side files, rename each data file to ``part-00000.parquet`` and swap the
    directory into place.

    Returns:
        Number of data files written.
    """
    tmp = path.with_name(path.name + ".__tmp__")
    shutil.rmtree(tmp, ignore_errors=True)
    if partition_by:
        writer = (
            frame.repartition(partition_by)
            .sortWithinPartitions(*order)
            .write.mode("overwrite")
            .partitionBy(partition_by)
        )
    else:
        writer = frame.repartition(1).sortWithinPartitions(*order).write.mode("overwrite")
    writer.parquet(str(tmp))

    files = 0
    for directory in sorted({p.parent for p in tmp.rglob("*")} | {tmp}):
        for side in [p for p in directory.iterdir() if p.is_file() and p.name[0] in "._"]:
            side.unlink()
        data = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".parquet")
        if len(data) > 1:
            raise RuntimeError(f"expected one file per partition in {directory}, got {len(data)}")
        if data:
            data[0].rename(directory / "part-00000.parquet")
            files += 1
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    return files


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_etl(
    spark: Any,
    etl: dict[str, Any],
    export: dict[str, Any],
    root: Path,
    output_root: Path | None = None,
    replicate: int = 1,
) -> dict[str, Any]:
    """Execute every step, write outputs and return the metrics document.

    Args:
        spark: Active SparkSession.
        etl: ``etl`` params section.
        export: ``export`` params section (input paths).
        root: Project root.
        output_root: Where outputs go (default ``root``; the benchmark uses a temp dir).
        replicate: Multiply the input ``replicate`` times with synthetic tickers
            (benchmark only) to measure how throughput scales with data volume.
    """
    from pyspark.sql import functions as F  # noqa: PLC0415

    out = output_root or root
    timer = StepTimer()

    started = time.monotonic()
    prices = spark.read.parquet(str(root / export["prices_path"]))
    news = spark.read.parquet(str(root / export["news_path"]))
    if replicate > 1:
        copies = spark.range(replicate).withColumnRenamed("id", "_copy")

        def _replicate(df: Any) -> Any:
            return (
                df.crossJoin(F.broadcast(copies))
                .withColumn(
                    "ticker",
                    F.when(F.col("_copy") == 0, F.col("ticker")).otherwise(
                        F.concat_ws("_", "ticker", F.col("_copy").cast("string"))
                    ),
                )
                .drop("_copy")
            )

        prices, news = _replicate(prices), _replicate(news)
    prices, news = prices.cache(), news.cache()
    raw_price_rows, raw_news_rows = prices.count(), news.count()
    timer.record(
        "read_snapshot", started, raw_price_rows + raw_news_rows, raw_price_rows + raw_news_rows
    )

    started = time.monotonic()
    first, last = prices.agg(F.min("date"), F.max("date")).first()
    calendar = trading_calendar(spark, first, last)
    aligned = align_to_calendar(with_model_price(prices), calendar, int(etl["max_ffill_days"]))
    aligned = aligned.cache()
    aligned_rows = aligned.count()
    filled_rows = aligned.where(F.col("is_filled")).count()
    matched = aligned.where(~F.col("is_filled")).count()
    timer.record("align_calendar", started, raw_price_rows, aligned_rows)

    started = time.monotonic()
    features = add_features(aligned, etl["feature_windows"]).cache()
    feature_rows = features.count()
    timer.record("engineer_features", started, aligned_rows, feature_rows)

    started = time.monotonic()
    news_clean, removed = clean_news(news)
    news_rows = news_clean.count()
    weekly = weekly_news(news_clean).cache()
    weekly_rows = weekly.count()
    timer.record("clean_aggregate_news", started, raw_news_rows, news_rows + weekly_rows)

    started = time.monotonic()
    files = write_deterministic(
        features, out / etl["processed_prices_path"], PRICE_OUTPUT_ORDER, partition_by="ticker"
    )
    files += write_deterministic(
        news_clean, out / etl["processed_news_path"], NEWS_OUTPUT_ORDER, partition_by="ticker"
    )
    files += write_deterministic(weekly, out / etl["news_weekly_path"], ["ticker", "week_start"])
    written = feature_rows + news_rows + weekly_rows
    timer.record("write_parquet", started, written, written)

    for frame in (prices, news, aligned, features, news_clean, weekly):
        frame.unpersist()

    total_seconds = round(sum(s["seconds"] for s in timer.steps), 3)
    rows_in = raw_price_rows + raw_news_rows
    return {
        "inputs": {
            "price_rows": raw_price_rows,
            "news_rows": raw_news_rows,
            "replicate": replicate,
        },
        "outputs": {
            "price_rows": feature_rows,
            "news_rows": news_rows,
            "news_weekly_rows": weekly_rows,
            "files": files,
            "tickers": features.select("ticker").distinct().count(),
        },
        "quality": {
            "price_rows_forward_filled": filled_rows,
            "price_rows_dropped": raw_price_rows - matched,
            "news_removed": removed,
        },
        "steps": timer.steps,
        "totals": {
            "processing_seconds": total_seconds,
            "rows_in": rows_in,
            "rows_per_s": round(rows_in / total_seconds, 1) if total_seconds > 0 else None,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.etl.process``)."""
    argparse.ArgumentParser(description="Spark ETL: raw snapshot -> processed data").parse_args(
        argv
    )
    settings = get_settings()
    configure_logging("etl", settings.log_level, settings.root / "logs")
    etl, export = settings.section("etl"), settings.section("export")
    with stage_run("process", settings) as run:
        spark, startup = build_spark(etl)
        try:
            metrics = run_etl(spark, etl, export, settings.root)
            metrics["spark"] = {
                "version": spark.version,
                "master": etl["spark_master"],
                "driver_memory": etl["driver_memory"],
                "shuffle_partitions": int(etl["shuffle_partitions"]),
                "startup_seconds": startup,
            }
        finally:
            spark.stop()
        path = settings.root / etl["metrics_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run.metrics = {
            "rows_in": metrics["totals"]["rows_in"],
            "rows_per_s": metrics["totals"]["rows_per_s"],
            "processing_seconds": metrics["totals"]["processing_seconds"],
            "forward_filled": metrics["quality"]["price_rows_forward_filled"],
        }
    print(json.dumps({"totals": metrics["totals"], "quality": metrics["quality"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
