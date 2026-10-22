"""DVC stage ``export``: snapshot MongoDB into versioned Parquet files.

Why a snapshot at all? MongoDB is the *operational* store and keeps changing.
Training on it directly would make results irreproducible. The export freezes
the current state into files that DVC hashes and stores in its remote, so
``git checkout <commit> && dvc checkout`` restores exactly the data a model
was trained on (guidelines: "every experiment reproducible from a commit hash").

Determinism matters: the stage runs on every ``dvc repro`` (``always_changed``)
but writes byte-identical files when the data has not changed, so DVC sees the
same hash and skips every downstream stage. Hence:

* rows are sorted by a total order,
* volatile bookkeeping fields (``_id``, ``ingested_at``, ``updated_at``) are dropped,
* the manifest contains no wall-clock timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo.database import Database

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.mongo import NEWS, PRICES, get_client
from finsight.common.runs import stage_run

log = get_logger(__name__)

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]
NEWS_COLUMNS = [
    "ticker",
    "news_id",
    "published_at",
    "headline",
    "summary",
    "text",
    "source",
    "url",
    "provider",
    "category",
]


def load_prices(db: Database, tickers: Sequence[str]) -> pd.DataFrame:
    """Read all stored bars for ``tickers`` into a typed, sorted DataFrame."""
    projection = {c: 1 for c in PRICE_COLUMNS} | {"_id": 0}
    docs = list(db[PRICES].find({"ticker": {"$in": list(tickers)}}, projection))
    frame = pd.DataFrame(docs, columns=PRICE_COLUMNS)
    # Trading days are calendar dates; storing them as `date` (not timestamps)
    # removes any time-zone ambiguity for Spark and pandas downstream.
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.date
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["ticker"] = frame["ticker"].astype("string")
    frame["source"] = frame["source"].astype("string")
    return frame.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def load_news(db: Database, tickers: Sequence[str]) -> pd.DataFrame:
    """Read all stored articles for ``tickers`` into a typed, sorted DataFrame."""
    docs = []
    for doc in db[NEWS].find({"ticker": {"$in": list(tickers)}}, {"_id": 0}):
        extra = doc.get("extra") or {}
        docs.append({**{c: doc.get(c) for c in NEWS_COLUMNS}, "category": extra.get("category")})
    frame = pd.DataFrame(docs, columns=NEWS_COLUMNS)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    for column in NEWS_COLUMNS:
        if column != "published_at":
            frame[column] = frame[column].astype("string")
    return frame.sort_values(["ticker", "published_at", "news_id"], kind="mergesort").reset_index(
        drop=True
    )


def file_sha256(path: Path) -> str:
    """Content hash recorded in the manifest (and logged to MLflow from Day 3)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame deterministically (no index, fixed compression).

    Timestamps are stored as microseconds: pandas defaults to nanoseconds,
    which Spark rejects (``PARQUET_TYPE_ILLEGAL: TIMESTAMP(NANOS)``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression="snappy",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )


def build_manifest(
    prices: pd.DataFrame, news: pd.DataFrame, written: dict[str, Path], display: dict[str, str]
) -> dict:
    """Describe the snapshot: row counts, coverage and content hashes.

    Args:
        prices: Exported price frame.
        news: Exported news frame.
        written: Absolute paths of the written files (hashed).
        display: Repo-relative paths recorded in the manifest (portable).
    """

    def _range(series: pd.Series) -> list[str | None]:
        if series.empty:
            return [None, None]
        return [str(series.min())[:10], str(series.max())[:10]]

    return {
        "prices": {
            "path": display["prices"],
            "rows": int(len(prices)),
            "tickers": int(prices["ticker"].nunique()),
            "date_range": _range(prices["date"]),
            "rows_per_source": {str(k): int(v) for k, v in prices["source"].value_counts().items()},
            "sha256": file_sha256(written["prices"]),
        },
        "news": {
            "path": display["news"],
            "rows": int(len(news)),
            "tickers": int(news["ticker"].nunique()),
            "date_range": _range(news["published_at"]),
            "sha256": file_sha256(written["news"]),
        },
    }


def export_snapshot(
    db: Database, tickers: Sequence[str], params: dict[str, Any], root: Path
) -> dict:
    """Run the export and return the manifest (also written to disk)."""
    paths = {"prices": root / params["prices_path"], "news": root / params["news_path"]}
    prices = load_prices(db, tickers)
    news = load_news(db, tickers)
    if prices.empty:
        raise FinSightError("No prices in MongoDB; run the ingestion backfill first.")
    write_parquet(prices, paths["prices"])
    write_parquet(news, paths["news"])
    manifest = build_manifest(
        prices, news, paths, {"prices": params["prices_path"], "news": params["news_path"]}
    )
    manifest_path = root / params["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.data.export``)."""
    argparse.ArgumentParser(description="Export MongoDB snapshot to Parquet").parse_args(argv)
    settings = get_settings()
    configure_logging("export", settings.log_level, settings.root / "logs")
    try:
        with stage_run("export", settings) as run:
            client = get_client(settings)
            try:
                manifest = export_snapshot(
                    client[settings.mongo_db],
                    settings.symbols,
                    settings.section("export"),
                    settings.root,
                )
            finally:
                client.close()
            run.metrics = {
                "price_rows": manifest["prices"]["rows"],
                "news_rows": manifest["news"]["rows"],
                "prices_sha256": manifest["prices"]["sha256"][:12],
            }
    except FinSightError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
