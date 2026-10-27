"""Ingestion orchestration: plan -> fetch -> upsert -> record.

Design notes (also summarised in docs/DESIGN_DECISIONS.md):

* **One code path for backfill and daily runs.** ``incremental`` gap-fills
  from each ticker's last stored date, and falls back to the full backfill
  window for a ticker with no data. So a missed day, a crashed run or a new
  ticker heal themselves on the next run.
* **Quota-aware provider chain.** MarketStack is tried first while the ledger
  allows it; on quota exhaustion or provider failure the ticker is fetched
  from the fallback provider. A ticker failing on *both* is recorded as an
  error and the run continues (graceful degradation -> status ``partial``).
* **Observability.** Each run writes a record to ``pipeline_runs`` (start ->
  finish, per-ticker results, API calls, rows, duration, throughput) and
  ``reports/ingestion_last_run.json``. The pipeline console (Day 7) and the
  Prometheus exporter (Day 8) read exactly these records.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from finsight.common.exceptions import ProviderError, QuotaExceededError
from finsight.common.logging import bind_run_id, get_logger
from finsight.common.timeutils import latest_expected_session, to_utc_midnight
from finsight.ingestion.providers.base import NewsProvider, PriceProvider
from finsight.ingestion.quota import QuotaLedger
from finsight.ingestion.repository import IngestionRepository

log = get_logger(__name__)

MODES = ("backfill", "incremental")


@dataclass(frozen=True)
class FetchTask:
    """One planned provider call."""

    kind: str  # "prices" | "news"
    ticker: str
    date_from: date
    date_to: date


@dataclass
class RunReport:
    """Everything recorded about one ingestion run."""

    run_id: str
    mode: str
    started_at: datetime
    pipeline: str = "ingestion"
    status: str = "running"
    finished_at: datetime | None = None
    duration_s: float = 0.0
    prices: dict[str, Any] = field(
        default_factory=lambda: {"fetched": 0, "inserted": 0, "updated": 0, "per_ticker": {}}
    )
    news: dict[str, Any] = field(
        default_factory=lambda: {"fetched": 0, "inserted": 0, "updated": 0, "per_ticker": {}}
    )
    api_calls: dict[str, int] = field(default_factory=dict)
    provider_rows: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    tasks_ok: int = 0
    tasks_skipped: int = 0
    throughput_rows_per_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (datetimes kept as objects for Mongo)."""
        return asdict(self)


class IngestionService:
    """Coordinates providers, quota ledger and repository for one run."""

    def __init__(
        self,
        repo: IngestionRepository,
        params: dict[str, Any],
        symbols: Sequence[str],
        *,
        price_provider: PriceProvider | None,
        fallback_provider: PriceProvider | None,
        news_provider: NewsProvider | None,
        ledger: QuotaLedger | None,
        report_dir: Path | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Wire the service.

        Args:
            repo: Mongo repository.
            params: The ``ingestion`` section of ``params.yaml``.
            symbols: Ticker universe.
            price_provider: Primary (metered) price provider, or ``None`` if not configured.
            fallback_provider: Secondary price provider, or ``None`` to disable fallback.
            news_provider: News provider, or ``None`` if not configured.
            ledger: Quota ledger for the primary provider (required if it is metered).
            report_dir: Where to write ``ingestion_last_run.json`` (``None`` = skip).
            clock: Injected clock for deterministic tests.
        """
        self.repo = repo
        self.params = params
        self.symbols = [s.upper() for s in symbols]
        self.primary = price_provider
        self.fallback = fallback_provider
        self.news = news_provider
        self.ledger = ledger
        self.report_dir = report_dir
        self.clock = clock
        self._primary_disabled = False

    # ================================================================ planning
    def plan(
        self, mode: str, tickers: Sequence[str] | None = None, *, prices: bool, news: bool
    ) -> list[FetchTask]:
        """Return the provider calls a run would make (no network, no writes)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        selected = [t.upper() for t in (tickers or self.symbols)]
        unknown = sorted(set(selected) - set(self.symbols))
        if unknown:
            raise ValueError(f"tickers not in configs/tickers.yaml: {unknown}")
        tasks: list[FetchTask] = []
        if prices:
            tasks += self._plan_prices(mode, selected)
        if news:
            tasks += self._plan_news(mode, selected)
        return tasks

    def _plan_prices(self, mode: str, tickers: list[str]) -> list[FetchTask]:
        now = self.clock()
        end = latest_expected_session(now)
        window_start = now.date() - timedelta(days=int(self.params["price_backfill_days"]))
        tasks = []
        for ticker in tickers:
            last = self.repo.last_price_date(ticker)
            if mode == "backfill" or last is None:
                start = window_start
            else:
                start = last.date() + timedelta(days=1)
            if start > end:
                log.info("prices up to date", extra={"ticker": ticker, "last": str(last)})
                continue
            tasks.append(FetchTask("prices", ticker, start, end))
        return tasks

    def _plan_news(self, mode: str, tickers: list[str]) -> list[FetchTask]:
        today = self.clock().date()
        window = max(1, int(self.params["news_window_days"]))
        backfill_start = today - timedelta(days=int(self.params["news_backfill_days"]))
        tasks = []
        for ticker in tickers:
            last = self.repo.last_news_date(ticker)
            # Re-read the last stored day: late-published articles are caught and
            # duplicates are absorbed by the unique (ticker, news_id) upsert.
            start = backfill_start if mode == "backfill" or last is None else last.date()
            cursor = start
            while cursor <= today:
                stop = min(cursor + timedelta(days=window - 1), today)
                tasks.append(FetchTask("news", ticker, cursor, stop))
                cursor = stop + timedelta(days=1)
        return tasks

    def estimate_cost(self, tasks: Sequence[FetchTask]) -> dict[str, int]:
        """Estimated metered requests per provider for a plan."""
        cost: dict[str, int] = {}
        if self.primary is not None:
            cost[self.primary.name] = sum(
                self.primary.cost(t.ticker, t.date_from, t.date_to)
                for t in tasks
                if t.kind == "prices"
            )
        if self.news is not None:
            cost[self.news.name] = sum(1 for t in tasks if t.kind == "news")
        return cost

    # =============================================================== execution
    def run(
        self,
        mode: str,
        tickers: Sequence[str] | None = None,
        *,
        prices: bool = True,
        news: bool = True,
    ) -> RunReport:
        """Execute a run end-to-end and persist its record.

        Never raises for provider failures (they are recorded in the report);
        only programming/database errors propagate, after the run record has
        been marked ``failed``.
        """
        report = RunReport(run_id=uuid.uuid4().hex[:12], mode=mode, started_at=self.clock())
        started = time.monotonic()
        with bind_run_id(report.run_id):
            tasks = self.plan(mode, tickers, prices=prices, news=news)
            self.repo.start_run(
                {
                    "run_id": report.run_id,
                    "pipeline": report.pipeline,
                    "mode": mode,
                    "started_at": report.started_at,
                    "planned_calls": len(tasks),
                    # Day 7: links this record to the scheduler run that started it.
                    "pipeline_run_id": os.getenv("FINSIGHT_PIPELINE_RUN_ID") or None,
                }
            )
            log.info("ingestion started", extra={"mode": mode, "planned_calls": len(tasks)})
            try:
                self._execute(tasks, report)
            except Exception as exc:
                report.status = "failed"
                report.errors.append({"stage": "run", "error": f"{exc.__class__.__name__}: {exc}"})
                self._finish(report, started)
                log.exception("ingestion crashed")
                raise
            failures = len(report.errors)
            not_done = len(tasks) - report.tasks_ok
            if not_done == 0:
                report.status = "success"  # includes "nothing to do"
            elif report.tasks_ok == 0:
                report.status = "failed"
            else:
                report.status = "partial"
            self._finish(report, started)
            log.info(
                "ingestion finished",
                extra={
                    "status": report.status,
                    "duration_s": report.duration_s,
                    "price_rows": report.prices["fetched"],
                    "news_rows": report.news["fetched"],
                    "api_calls": report.api_calls,
                    "errors": failures,
                },
            )
        return report

    def _execute(self, tasks: list[FetchTask], report: RunReport) -> None:
        """Run every task; count successes (skipped/failed tasks are 'not done')."""
        news_auth_failed = False
        for task in tasks:
            if task.kind == "prices":
                ok = self._run_price_task(task, report)
            elif news_auth_failed:
                report.tasks_skipped += 1
                continue
            else:
                ok, news_auth_failed = self._run_news_task(task, report)
            report.tasks_ok += int(ok)

    # ------------------------------------------------------------------ prices
    def _run_price_task(self, task: FetchTask, report: RunReport) -> bool:
        """Fetch + store one ticker's prices. Returns True on success."""
        bars, provider_name, error = self._fetch_prices_with_fallback(task, report)
        entry: dict[str, Any] = {
            "from": task.date_from.isoformat(),
            "to": task.date_to.isoformat(),
            "provider": provider_name,
        }
        if error is not None:
            entry["error"] = error
            report.errors.append({"stage": "prices", "ticker": task.ticker, "error": error})
            report.prices["per_ticker"][task.ticker] = entry
            return False
        # Defensive: keep only bars inside the requested range.
        lo = to_utc_midnight(task.date_from)
        hi = to_utc_midnight(task.date_to)
        bars = [b for b in bars if lo <= b.date <= hi]
        result = self.repo.upsert_prices(bars)
        entry.update(rows=len(bars), inserted=result.inserted, updated=result.updated)
        report.prices["fetched"] += len(bars)
        report.prices["inserted"] += result.inserted
        report.prices["updated"] += result.updated
        report.provider_rows[provider_name] = report.provider_rows.get(provider_name, 0) + len(bars)
        report.prices["per_ticker"][task.ticker] = entry
        return True

    def _fetch_prices_with_fallback(
        self, task: FetchTask, report: RunReport
    ) -> tuple[list, str, str | None]:
        """Try the primary provider (quota permitting), then the fallback.

        A planned range always ends on an expected trading session, so an
        *empty* response is treated as a failure too (yfinance, for instance,
        returns an empty frame instead of raising when Yahoo is unreachable).

        Returns:
            ``(bars, provider_used, error_message_or_None)``.
        """
        primary_error: str | None = None
        if self.primary is not None and not self._primary_disabled:
            bars, primary_error = self._try_primary(self.primary, task, report)
            if bars:
                return bars, self.primary.name, None
            if bars is not None:  # call succeeded but returned nothing
                primary_error = f"{self.primary.name}: no bars returned"

        if self.fallback is None:
            return [], "none", primary_error or "no price provider configured"
        self._count(report, self.fallback.name, 1)
        try:
            bars = self.fallback.fetch_eod(task.ticker, task.date_from, task.date_to)
        except ProviderError as exc:
            fallback_error = str(exc)
        else:
            if bars:
                return bars, self.fallback.name, None
            fallback_error = f"{self.fallback.name}: no bars returned"
        message = f"{primary_error} | {fallback_error}" if primary_error else fallback_error
        return [], self.fallback.name, message

    def _try_primary(
        self, provider: PriceProvider, task: FetchTask, report: RunReport
    ) -> tuple[list | None, str | None]:
        """One attempt on the metered primary provider.

        Returns:
            ``(bars, None)`` on success, ``(None, error_message)`` otherwise.
        """
        cost = provider.cost(task.ticker, task.date_from, task.date_to)
        metered = provider.metered and self.ledger is not None
        if metered:
            try:
                self.ledger.reserve_calls(cost)  # type: ignore[union-attr]
            except QuotaExceededError as exc:  # our own budget guard said no
                self._primary_disabled = True
                log.warning("price quota budget reached; switching to fallback")
                return None, str(exc)
        try:
            bars = provider.fetch_eod(task.ticker, task.date_from, task.date_to)
        except QuotaExceededError as exc:  # the API itself says the quota is gone
            self._primary_disabled = True
            if metered:
                self.ledger.mark_exhausted()  # type: ignore[union-attr]
            self._count(report, provider.name, cost)
            log.warning("primary price provider reported quota exhausted")
            return None, str(exc)
        except ProviderError as exc:
            # MarketStack does not bill failed requests: give the reservation back.
            if metered:
                self.ledger.refund(cost)  # type: ignore[union-attr]
            self._count(report, provider.name, 0)
            log.warning(
                "primary price provider failed", extra={"ticker": task.ticker, "error": str(exc)}
            )
            return None, str(exc)
        self._count(report, provider.name, cost)
        return bars, None

    # -------------------------------------------------------------------- news
    def _run_news_task(self, task: FetchTask, report: RunReport) -> tuple[bool, bool]:
        """Fetch + store one news window.

        Returns:
            ``(ok, stop_news)`` -- ``stop_news`` is True after an authentication
            failure, since every further news call would fail the same way.
        """
        if self.news is None:
            report.errors.append(
                {"stage": "news", "ticker": task.ticker, "error": "no news provider configured"}
            )
            return False, True
        per = report.news["per_ticker"].setdefault(
            task.ticker, {"windows": 0, "rows": 0, "inserted": 0, "errors": 0}
        )
        try:
            articles = self.news.fetch_news(task.ticker, task.date_from, task.date_to)
            self._count(report, self.news.name, 1)
        except ProviderError as exc:
            self._count(report, self.news.name, 1)
            per["errors"] += 1
            report.errors.append(
                {
                    "stage": "news",
                    "ticker": task.ticker,
                    "window": f"{task.date_from}..{task.date_to}",
                    "error": str(exc),
                }
            )
            return False, exc.status in (401, 403)
        result = self.repo.upsert_news(articles)
        per["windows"] += 1
        per["rows"] += len(articles)
        per["inserted"] += result.inserted
        report.news["fetched"] += len(articles)
        report.news["inserted"] += result.inserted
        report.news["updated"] += result.updated
        return True, False

    # ------------------------------------------------------------------ repair
    def repair(self, dry_run: bool = False) -> dict[str, Any]:
        """Replace stored bars that break the price rules (e.g. a close of 0).

        For each invalid bar, the *fallback* provider (unmetered) is asked for
        that single day. A valid bar replaces the stored one with its raw
        ``close`` and ``adj_close = None``, so the ticker's model price falls
        back to raw closes. Mixing two providers' adjustment factors in one
        series would add artificial jumps (see DATA_CARD B6). If no valid bar
        comes back, the stored bar is deleted and the ETL forward-fills the
        one-day gap.

        Args:
            dry_run: Only report what would change.

        Returns:
            A summary with one action per bad bar, also stored as an
            ``ingestion`` run record with ``mode = "repair"``.
        """
        bad = self.repo.invalid_prices()
        actions: list[dict[str, Any]] = []
        run_id = uuid.uuid4().hex[:12]
        started_at, started = self.clock(), time.monotonic()
        with bind_run_id(run_id):
            for doc in bad:
                day = doc["date"].date()
                entry = {
                    "ticker": doc["ticker"],
                    "date": day.isoformat(),
                    "stored": {k: doc.get(k) for k in ("close", "adj_close", "volume", "source")},
                }
                if dry_run:
                    actions.append({**entry, "action": "would_repair"})
                    continue
                replacement = None
                if self.fallback is not None:
                    try:
                        bars = self.fallback.fetch_eod(doc["ticker"], day, day)
                        replacement = next((b for b in bars if b.date.date() == day), None)
                    except ProviderError as exc:
                        entry["fallback_error"] = str(exc)
                if replacement is not None:
                    fixed = replace(replacement, adj_close=None)
                    self.repo.upsert_prices([fixed])
                    actions.append(
                        {
                            **entry,
                            "action": "replaced",
                            "new_close": fixed.close,
                            "new_source": fixed.source,
                        }
                    )
                else:
                    self.repo.delete_price(doc["ticker"], doc["date"])
                    actions.append({**entry, "action": "deleted"})
                log.info("price bar repaired", extra=actions[-1])
            summary = {
                "run_id": run_id,
                "dry_run": dry_run,
                "invalid_bars": len(bad),
                "replaced": sum(a["action"] == "replaced" for a in actions),
                "deleted": sum(a["action"] == "deleted" for a in actions),
                "actions": actions,
            }
            if not dry_run:
                self.repo.start_run(
                    {
                        "run_id": run_id,
                        "pipeline": "ingestion",
                        "mode": "repair",
                        "started_at": started_at,
                        "planned_calls": len(bad),
                    }
                )
                self.repo.finish_run(
                    run_id,
                    {
                        "status": "success",
                        "finished_at": self.clock(),
                        "duration_s": round(time.monotonic() - started, 3),
                        "repair": {k: v for k, v in summary.items() if k != "run_id"},
                    },
                )
        return summary

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _count(report: RunReport, provider: str, n: int) -> None:
        report.api_calls[provider] = report.api_calls.get(provider, 0) + n

    def _finish(self, report: RunReport, started_monotonic: float) -> None:
        report.finished_at = self.clock()
        report.duration_s = round(time.monotonic() - started_monotonic, 3)
        rows = report.prices["fetched"] + report.news["fetched"]
        report.throughput_rows_per_s = (
            round(rows / report.duration_s, 2) if report.duration_s > 0 else 0.0
        )
        record = report.to_dict()
        record.pop("run_id")
        self.repo.finish_run(report.run_id, record)
        if self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            path = self.report_dir / "ingestion_last_run.json"
            path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
