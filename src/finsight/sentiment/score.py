"""DVC stage ``score_sentiment``: score every article, aggregate per week, publish.

Incremental by design
---------------------
``data/sentiment/article_scores.parquet`` is a *persistent* DVC output (DVC
does not delete it before the stage runs). Each row carries the article's
text hash and the model fingerprint (:func:`~finsight.sentiment.scoring.model_version`).
On every run only articles that are new, whose text changed, or that were
scored by a different model version are sent through the model -- the daily
run scores ~50-100 new articles instead of ~10,000.

Progress and checkpoints (Day 6)
--------------------------------
A full re-score (e.g. after a new model is promoted) takes 10-60 minutes on a
laptop. Articles are scored in chunks of ``sentiment.checkpoint_every``; after
each chunk a progress line is logged (done / total, rate, time left) and every
score so far is saved to ``article_scores.partial.parquet``. If the run is
interrupted, the next run reads the checkpoint and only scores what is left.
The checkpoint is deleted once the stage finishes.

Outputs
-------
* ``data/sentiment/article_scores.parquet`` -- per article: label, confidence, score.
* ``data/sentiment/weekly.parquet`` -- per (ticker, week): article count, mean
  score, positive / negative / neutral counts, and the top positive and top
  negative headlines (the *explanation* shown next to each bar in the portal).
* ``reports/sentiment_scoring.json`` -- DVC metrics: how many scored vs reused,
  throughput, label shares.
* MongoDB ``news_sentiment`` and ``sentiment_weekly`` -- what the API serves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.sentiment.scoring import NEGATIVE, NEUTRAL, POSITIVE, clean_texts, labels_for

log = get_logger(__name__)

NEWS_FIELDS = [
    "ticker",
    "news_id",
    "published_at",
    "week_start",
    "headline",
    "text",
    "source",
    "url",
]
SCORE_COLUMNS = [*NEWS_FIELDS, "text_sha1", "model_version", "score", "confidence", "label"]


def text_hash(texts: Sequence[str]) -> list[str]:
    """Short content hash per text (detects edited articles)."""
    return [hashlib.sha1(t.encode("utf-8"), usedforsecurity=False).hexdigest() for t in texts]


def load_news(path: Path) -> pd.DataFrame:
    """Processed news from the Spark ETL, typed and in a fixed order."""
    frame = pd.read_parquet(path)
    for column in NEWS_FIELDS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[NEWS_FIELDS].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["news_id"] = frame["news_id"].astype(str)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["week_start"] = pd.to_datetime(frame["week_start"]).dt.date
    frame["text"] = clean_texts(frame["text"].tolist())
    frame["text_sha1"] = text_hash(frame["text"].tolist())
    return frame.sort_values(["ticker", "news_id"], kind="mergesort").reset_index(drop=True)


CACHE_COLUMNS = ["ticker", "news_id", "text_sha1", "model_version", "score", "confidence"]
ScoreFn = Callable[[list[str]], "np.ndarray | tuple[np.ndarray, np.ndarray]"]


def checkpoint_path(cache_path: Path) -> Path:
    """Where an interrupted run's scores are kept (next to the cache)."""
    return cache_path.with_name(cache_path.stem + ".partial.parquet")


def _read_cache_file(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "confidence" not in frame.columns:
        frame["confidence"] = np.round((1 + frame["score"].abs()) / 2, 6)
    return frame[CACHE_COLUMNS]


def load_cache(path: Path) -> pd.DataFrame:
    """Previously scored articles (empty frame on the first run).

    Rows from an interrupted run's checkpoint are added on top, so a restart
    resumes instead of starting over. Day 4 caches have no ``confidence``
    column; they were all written by the two-label DistilBERT, for which
    confidence = (1 + |score|) / 2 exactly.
    """
    parts = [_read_cache_file(p) for p in (path, checkpoint_path(path)) if p.exists()]
    if not parts:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    if len(parts) == 1:
        return parts[0]
    keys = ["ticker", "news_id", "text_sha1", "model_version"]
    return pd.concat(parts, ignore_index=True).drop_duplicates(keys, keep="last")


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write via a temporary file, so a crash never leaves half a file behind."""
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(path)


class ProgressLogger:
    """Logs scoring progress; time left is estimated from characters, not rows.

    Texts are scored shortest-first, so a rate in articles per second would
    promise far too early a finish. Characters remaining / characters per
    second is a much better estimate.
    """

    def __init__(
        self,
        total: int,
        total_chars: int,
        clock: Callable[[], float] = time.perf_counter,
        started: float | None = None,
    ) -> None:
        """Track ``total`` articles of ``total_chars`` characters from ``started``."""
        self.total, self.total_chars, self.clock = total, max(total_chars, 1), clock
        self.started = clock() if started is None else started

    def __call__(self, done: int, done_chars: int) -> dict[str, Any]:
        """Log one progress line and return its fields."""
        elapsed = max(self.clock() - self.started, 1e-9)
        chars_per_s = done_chars / elapsed
        left = (self.total_chars - done_chars) / chars_per_s if chars_per_s else None
        fields = {
            "done": done,
            "total": self.total,
            "percent": round(100 * done / self.total, 1) if self.total else 100.0,
            "articles_per_s": round(done / elapsed, 1),
            "elapsed_s": round(elapsed),
            "eta_s": round(left) if left is not None else None,
        }
        eta = f"~{left / 60:.0f} min left" if left is not None else "time left unknown"
        log.info(
            f"scored {done}/{self.total} ({fields['percent']}%), "
            f"{fields['articles_per_s']} articles/s, {eta}",
            extra=fields,
        )
        return fields


def _split(result: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Accept ``scores`` or ``(scores, confidences)`` from a score function."""
    if isinstance(result, tuple):
        scores, confidence = (np.asarray(r, dtype="float64") for r in result)
    else:
        scores = np.asarray(result, dtype="float64")
        confidence = np.round((1 + np.abs(scores)) / 2, 6)  # two-label models
    if len(scores) != n:
        raise ValueError(f"score function returned {len(scores)} scores for {n} texts")
    return scores, confidence


ChunkFn = Callable[[pd.DataFrame, "ChunkProgress"], None]


class ChunkProgress(NamedTuple):
    """How far a scoring run has got (passed to ``on_chunk``)."""

    done: int
    total: int
    done_chars: int
    total_chars: int


def score_incremental(
    news: pd.DataFrame,
    cache: pd.DataFrame,
    version: str,
    score_fn: ScoreFn,
    chunk_size: int | None = None,
    on_chunk: ChunkFn | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reuse cached scores where valid; score the rest with ``score_fn``.

    Args:
        news: Articles from :func:`load_news`.
        cache: Earlier scores from :func:`load_cache`.
        version: Fingerprint of the production model; other versions are re-scored.
        score_fn: Texts -> scores, or ``(scores, confidences)``.
        chunk_size: Score at most this many texts per ``score_fn`` call
            (``None``: all at once). Texts go shortest-first, as in the scorer.
        on_chunk: Called after each chunk with ``(scored_so_far, progress)``;
            ``scored_so_far`` holds every row scored in this run so far (cache
            columns), which is what the checkpoint saves.

    Returns:
        ``(news_with_score, counts)`` -- counts of ``reused`` and ``scored`` rows.
    """
    keep = ["ticker", "news_id", "text_sha1", "score", "confidence"]
    valid = cache[cache["model_version"] == version][keep]
    merged = news.merge(valid, on=["ticker", "news_id", "text_sha1"], how="left")
    todo = merged["score"].isna().to_numpy()
    if todo.any():
        lengths = merged.loc[todo, "text"].str.len()
        order = lengths.sort_values(kind="mergesort").index.to_numpy()
        step = chunk_size or len(order)
        total_chars = int(lengths.sum())
        done = done_chars = 0
        for start in range(0, len(order), step):
            idx = order[start : start + step]
            texts = merged.loc[idx, "text"].tolist()
            scores, confidence = _split(score_fn(texts), len(texts))
            merged.loc[idx, "score"] = scores
            merged.loc[idx, "confidence"] = confidence
            done += len(idx)
            done_chars += int(lengths.loc[idx].sum())
            if on_chunk is not None:
                so_far = merged.loc[order[:done], ["ticker", "news_id", "text_sha1"]].copy()
                so_far["model_version"] = version
                so_far["score"] = merged.loc[order[:done], "score"].astype("float64")
                so_far["confidence"] = merged.loc[order[:done], "confidence"].astype("float64")
                on_chunk(
                    so_far[CACHE_COLUMNS],
                    ChunkProgress(done, len(order), done_chars, total_chars),
                )
    merged["score"] = merged["score"].astype("float64")
    merged["confidence"] = merged["confidence"].astype("float64")
    merged["model_version"] = version
    return merged, {"reused": int((~todo).sum()), "scored": int(todo.sum())}


def finalise(scored: pd.DataFrame, neutral_band: float) -> pd.DataFrame:
    """Add the label (neutral band applied); keep the model's own confidence."""
    out = scored.copy()
    out["label"] = labels_for(out["score"].to_numpy(), neutral_band).astype(str)
    if "confidence" not in out.columns:
        out["confidence"] = np.round((1 + out["score"].abs()) / 2, 6)
    return (
        out[SCORE_COLUMNS]
        .sort_values(["ticker", "news_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _top(group: pd.DataFrame, n: int, positive: bool) -> list[dict[str, Any]]:
    side = group[group["label"] == (POSITIVE if positive else NEGATIVE)]
    side = side.sort_values(["score", "news_id"], ascending=[not positive, True], kind="mergesort")
    return [
        {
            "headline": r.headline,
            "score": round(float(r.score), 4),
            "source": r.source,
            "url": r.url,
            "published_at": r.published_at.isoformat(),
        }
        for r in side.head(n).itertuples()
    ]


def weekly_aggregate(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """One row per (ticker, week): counts, mean score and explanatory headlines."""
    rows = []
    for (ticker, week), group in scored.groupby(["ticker", "week_start"], sort=True):
        labels = group["label"].value_counts()
        rows.append(
            {
                "ticker": ticker,
                "week_start": week,
                "article_count": int(len(group)),
                "mean_score": round(float(group["score"].mean()), 6),
                "positive": int(labels.get(POSITIVE, 0)),
                "negative": int(labels.get(NEGATIVE, 0)),
                "neutral": int(labels.get(NEUTRAL, 0)),
                "top_positive": _top(group, top_n, positive=True),
                "top_negative": _top(group, top_n, positive=False),
            }
        )
    return pd.DataFrame(rows)


def _as_datetime(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def publish(db: Any, scored: pd.DataFrame, weekly: pd.DataFrame, run_id: str) -> dict[str, int]:
    """Upsert articles and weeks into MongoDB; remove documents no longer present.

    Idempotent: re-publishing the same frames changes nothing.
    """
    from pymongo import ReplaceOne  # noqa: PLC0415

    from finsight.common.mongo import NEWS_SENTIMENT, SENTIMENT_WEEKLY  # noqa: PLC0415

    article_ops = []
    for r in scored.itertuples(index=False):
        doc = {
            "ticker": r.ticker,
            "news_id": r.news_id,
            "published_at": r.published_at.to_pydatetime(),
            "week_start": _as_datetime(r.week_start),
            "headline": r.headline,
            "source": r.source,
            "url": r.url,
            "label": r.label,
            "score": float(r.score),
            "confidence": float(r.confidence),
            "model_version": r.model_version,
        }
        article_ops.append(ReplaceOne({"ticker": r.ticker, "news_id": r.news_id}, doc, upsert=True))
    week_ops = []
    for r in weekly.to_dict("records"):
        doc = {**r, "week_start": _as_datetime(r["week_start"]), "sentiment_run": run_id}
        week_ops.append(
            ReplaceOne({"ticker": r["ticker"], "week_start": doc["week_start"]}, doc, upsert=True)
        )
    counts = {"articles_upserted": 0, "weeks_upserted": 0, "stale_removed": 0}
    if article_ops:
        res = db[NEWS_SENTIMENT].bulk_write(article_ops, ordered=False)
        counts["articles_upserted"] = res.upserted_count + res.modified_count
    if week_ops:
        res = db[SENTIMENT_WEEKLY].bulk_write(week_ops, ordered=False)
        counts["weeks_upserted"] = res.upserted_count + res.modified_count
    keep = set(zip(scored["ticker"], scored["news_id"], strict=True))
    stale = [
        d["_id"]
        for d in db[NEWS_SENTIMENT].find({}, {"ticker": 1, "news_id": 1})
        if (d["ticker"], d["news_id"]) not in keep
    ]
    keep_weeks = {(r["ticker"], _as_datetime(r["week_start"])) for r in weekly.to_dict("records")}
    stale_weeks = [
        d["_id"]
        for d in db[SENTIMENT_WEEKLY].find({}, {"ticker": 1, "week_start": 1})
        if (d["ticker"], d["week_start"].replace(tzinfo=UTC)) not in keep_weeks
    ]
    if stale:
        counts["stale_removed"] += (
            db[NEWS_SENTIMENT].delete_many({"_id": {"$in": stale}}).deleted_count
        )
    if stale_weeks:
        counts["stale_removed"] += (
            db[SENTIMENT_WEEKLY].delete_many({"_id": {"$in": stale_weeks}}).deleted_count
        )
    return counts


def resolve_production(params: dict[str, Any], root: Path) -> dict[str, Any]:
    """Production sentiment model: pointer from the promotion report + local path."""
    from finsight.registry.pointer import (  # noqa: PLC0415
        load_promotion,
        local_model_dir,
        production_entry,
    )

    entry = production_entry(load_promotion(root / params["promotion_report"]), "sentiment")
    candidate = params["candidates"][entry["candidate"]]
    path = local_model_dir(entry, root / candidate["model_dir"], root / "models" / "registry_cache")
    return {**entry, "path": path}


def run_scoring(
    params: dict[str, Any],
    news_path: Path,
    root: Path,
    score_fn: ScoreFn | None = None,
    production: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score, aggregate and write the stage outputs (Mongo publishing is separate).

    Args:
        params: The ``sentiment`` section of ``params.yaml`` (+ ``promotion_report``).
        news_path: Processed news directory.
        root: Project root.
        score_fn: Injected scorer (tests); defaults to the production pyfunc model.
        production: Injected production pointer (tests); default: the promotion report.
    """
    production = production or resolve_production(params, root)
    version, band = production["model_version"], float(production["neutral_band"])
    if score_fn is None:
        import mlflow  # noqa: PLC0415
        import torch  # noqa: PLC0415

        model = mlflow.pyfunc.load_model(str(production["path"]))
        model.unwrap_python_model().scorer.batch_size = int(params["batch_size"])
        torch.set_num_threads(int(params["torch_threads"]))

        def score_fn(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
            out = model.predict(pd.DataFrame({"text": texts}))
            return out["score"].to_numpy(), out["confidence"].to_numpy()

    news = load_news(news_path)
    cache_path = root / params["scores_path"]
    partial = checkpoint_path(cache_path)
    cache = load_cache(cache_path)
    if partial.exists():
        log.info("resuming from checkpoint", extra={"checkpoint": str(partial)})
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    progress: ProgressLogger | None = None

    def on_chunk(so_far: pd.DataFrame, state: ChunkProgress) -> None:
        nonlocal progress
        if progress is None:
            progress = ProgressLogger(state.total, state.total_chars, started=started)
        rows = pd.concat([cache, so_far], ignore_index=True) if len(cache) else so_far
        write_parquet_atomic(rows, partial)
        progress(state.done, state.done_chars)

    started = time.perf_counter()
    merged, counts = score_incremental(
        news,
        cache,
        version,
        score_fn,
        chunk_size=int(params.get("checkpoint_every") or 0) or None,
        on_chunk=on_chunk,
    )
    seconds = time.perf_counter() - started
    scored = finalise(merged, band)
    weekly = weekly_aggregate(scored, int(params["top_headlines"]))

    write_parquet_atomic(scored, cache_path)
    weekly_path = root / params["weekly_path"]
    weekly.to_parquet(weekly_path, index=False, compression="snappy")
    partial.unlink(missing_ok=True)  # everything is in the cache now

    shares = scored["label"].value_counts(normalize=True)
    report = {
        "model_version": version,
        "articles": int(len(scored)),
        "scored_this_run": counts["scored"],
        "reused_from_cache": counts["reused"],
        "scoring_seconds": round(seconds, 2),
        "throughput_articles_per_s": (
            round(counts["scored"] / seconds, 1) if counts["scored"] else None
        ),
        "weeks": int(len(weekly)),
        "mean_score": round(float(scored["score"].mean()), 4),
        "share_positive": round(float(shares.get(POSITIVE, 0.0)), 4),
        "share_negative": round(float(shares.get(NEGATIVE, 0.0)), 4),
        "share_neutral": round(float(shares.get(NEUTRAL, 0.0)), 4),
    }
    report["production_version"] = production.get("version")
    report["candidate"] = production.get("candidate")
    return {"report": report, "scored": scored, "weekly": weekly, "run_id": production["run_id"]}


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.sentiment.score``)."""
    parser = argparse.ArgumentParser(description="Score news sentiment (incremental)")
    parser.add_argument("--no-publish", action="store_true", help="skip the MongoDB upsert")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("score_sentiment", settings.log_level, settings.root / "logs")
    params = {
        **settings.section("sentiment"),
        "promotion_report": settings.section("registry")["report"],
    }
    news_path = settings.root / settings.section("etl")["processed_news_path"]
    try:
        with stage_run("score_sentiment", settings) as run:
            outcome = run_scoring(params, news_path, settings.root)
            report = outcome["report"]
            if not args.no_publish:
                from finsight.common.mongo import ensure_indexes, get_client  # noqa: PLC0415

                client = get_client(settings)
                try:
                    db = client[settings.mongo_db]
                    ensure_indexes(db)
                    report["published"] = publish(
                        db, outcome["scored"], outcome["weekly"], outcome["run_id"]
                    )
                finally:
                    client.close()
            path = settings.root / params["scoring_report"]
            path.parent.mkdir(parents=True, exist_ok=True)
            compact = {k: v for k, v in report.items() if k != "published"}
            path.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
            run.metrics = compact
    except (FinSightError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"score_sentiment failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
