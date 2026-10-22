"""Command-line entry point for ingestion.

Examples::

    python -m finsight.ingestion.cli backfill --dry-run    # show API cost, no calls
    python -m finsight.ingestion.cli backfill              # one-time historical load
    python -m finsight.ingestion.cli incremental           # daily gap-fill (scheduler)
    python -m finsight.ingestion.cli incremental --tickers AAPL MSFT --news-only
    python -m finsight.ingestion.cli status                # coverage, quota, last runs
    python -m finsight.ingestion.cli repair --dry-run      # list impossible stored bars
    python -m finsight.ingestion.cli quota --set-used 12   # sync with provider dashboard

Exit codes (consumed by the scheduler on Day 7):
    0 success, 3 partial success, 1 failed, 2 usage/configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from finsight.common.config import Settings, get_settings
from finsight.common.exceptions import ConfigError, FinSightError
from finsight.common.http import HttpClient
from finsight.common.logging import configure_logging, get_logger
from finsight.common.mongo import ensure_indexes, get_client
from finsight.ingestion.providers.finnhub import FinnhubNewsProvider
from finsight.ingestion.providers.marketstack import MarketStackProvider, classify_marketstack
from finsight.ingestion.providers.yfinance_provider import YFinanceProvider
from finsight.ingestion.quota import QuotaLedger
from finsight.ingestion.repository import IngestionRepository
from finsight.ingestion.service import IngestionService

log = get_logger(__name__)
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 3


def build_service(settings: Settings, db: Any) -> IngestionService:
    """Assemble providers, ledger and repository from settings (composition root)."""
    params = settings.section("ingestion")
    http_kwargs = {
        "timeout_s": float(params["http_timeout_s"]),
        "max_retries": int(params["http_max_retries"]),
        "backoff_base_s": float(params["http_backoff_base_s"]),
        "backoff_max_s": float(params["http_backoff_max_s"]),
    }

    primary = None
    try:
        primary = MarketStackProvider(
            settings.require("marketstack_api_key"),
            settings.marketstack_base_url,
            HttpClient("marketstack", classify=classify_marketstack, **http_kwargs),
        )
    except ConfigError:
        log.warning("MARKETSTACK_API_KEY not set; prices will come from the fallback provider")

    fallback = (
        YFinanceProvider() if str(params.get("price_fallback_provider")) == "yfinance" else None
    )

    news = None
    try:
        news = FinnhubNewsProvider(
            settings.require("finnhub_api_key"),
            settings.finnhub_base_url,
            HttpClient(
                "finnhub", min_interval_s=float(params["finnhub_min_interval_s"]), **http_kwargs
            ),
            max_articles=int(params["max_articles_per_window"]),
        )
    except ConfigError:
        log.warning("FINNHUB_API_KEY not set; news ingestion disabled")

    ledger = QuotaLedger(
        db,
        "marketstack",
        budget=int(params["marketstack_monthly_budget"]),
        reserve=int(params["marketstack_reserve"]),
    )
    return IngestionService(
        IngestionRepository(db),
        params,
        settings.symbols,
        price_provider=primary,
        fallback_provider=fallback,
        news_provider=news,
        ledger=ledger,
        report_dir=settings.root / "reports",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finsight-ingest", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("backfill", "re-fetch the full historical window for every ticker"),
        ("incremental", "gap-fill from each ticker's last stored date (daily job)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--tickers", nargs="+", help="subset of symbols (default: all)")
        group = p.add_mutually_exclusive_group()
        group.add_argument("--prices-only", action="store_true")
        group.add_argument("--news-only", action="store_true")
        p.add_argument(
            "--dry-run", action="store_true", help="print the plan and API cost; make no calls"
        )
    repair = sub.add_parser(
        "repair",
        help="re-fetch stored bars with impossible values (e.g. close = 0) from the fallback",
    )
    repair.add_argument("--dry-run", action="store_true", help="list bad bars; change nothing")
    sub.add_parser("status", help="coverage per ticker, quota usage and recent runs")
    quota = sub.add_parser("quota", help="show or correct the MarketStack request ledger")
    quota.add_argument("--set-used", type=int, help="overwrite this month's used-call counter")
    return parser


def _cmd_run(args: argparse.Namespace, service: IngestionService) -> int:
    prices, news = not args.news_only, not args.prices_only
    tasks = service.plan(args.command, args.tickers, prices=prices, news=news)
    cost = service.estimate_cost(tasks)
    remaining = service.ledger.remaining() if service.ledger else None
    if args.dry_run:
        price_tasks = [t for t in tasks if t.kind == "prices"]
        print(
            f"Plan ({args.command}): {len(price_tasks)} price calls, "
            f"{len(tasks) - len(price_tasks)} news calls"
        )
        for t in price_tasks:
            print(f"  prices {t.ticker:<5} {t.date_from} -> {t.date_to}")
        print(f"Estimated metered calls: {cost}")
        print(f"MarketStack usable calls left this month: {remaining}")
        if (
            service.primary is not None
            and remaining is not None
            and cost.get("marketstack", 0) > remaining
        ):
            print("WARNING: not enough MarketStack quota; remaining tickers will use the fallback.")
        return EXIT_OK

    report = service.run(args.command, args.tickers, prices=prices, news=news)
    summary = {
        "run_id": report.run_id,
        "status": report.status,
        "duration_s": report.duration_s,
        "price_rows": report.prices["fetched"],
        "news_rows": report.news["fetched"],
        "api_calls": report.api_calls,
        "provider_rows": report.provider_rows,
        "throughput_rows_per_s": report.throughput_rows_per_s,
        "errors": report.errors[:10],
    }
    print(json.dumps(summary, indent=2, default=str))
    return {"success": EXIT_OK, "partial": EXIT_PARTIAL}.get(report.status, EXIT_FAILED)


def _day(value: Any) -> str:
    """Format a stored datetime as YYYY-MM-DD (``-`` when missing)."""
    return value.date().isoformat() if value is not None else "-"


def _cmd_status(service: IngestionService) -> int:
    coverage = service.repo.coverage()
    header = ("ticker", "price rows", "first", "last", "news rows", "last news")
    print("{:<6} {:>10} {:>11} {:>11} {:>10} {:>11}".format(*header))
    for symbol in service.symbols:
        row = coverage.get(symbol, {})
        p, n = row.get("prices", {}), row.get("news", {})
        print(
            "{:<6} {:>10} {:>11} {:>11} {:>10} {:>11}".format(
                symbol,
                p.get("rows", 0),
                _day(p.get("first")),
                _day(p.get("last")),
                n.get("rows", 0),
                _day(n.get("last")),
            )
        )
    if service.ledger:
        print(
            f"\nMarketStack ledger {service.ledger.period}: used {service.ledger.used()} / "
            f"budget {service.ledger.budget} (reserve {service.ledger.reserve}), "
            f"auto-usable left {service.ledger.remaining()}"
        )
    print("\nRecent ingestion runs:")
    for run in service.repo.recent_runs("ingestion", 5):
        errors = len(run.get("errors", []))
        print(
            f"  {run.get('started_at')!s:<32} {run.get('run_id')} {run.get('mode'):<11} "
            f"{run.get('status'):<8} {run.get('duration_s', 0)}s errors={errors}"
        )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()
        configure_logging("ingestion", settings.log_level, settings.root / "logs")
        client = get_client(settings)
    except FinSightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        db = client[settings.mongo_db]
        ensure_indexes(db)
        service = build_service(settings, db)
        if args.command == "status":
            return _cmd_status(service)
        if args.command == "repair":
            summary = service.repair(dry_run=args.dry_run)
            print(json.dumps(summary, indent=2, default=str))
            return EXIT_OK
        if args.command == "quota":
            if args.set_used is not None:
                service.ledger.set_used(args.set_used)
            print(
                f"{service.ledger.period}: used {service.ledger.used()} / {service.ledger.budget}"
            )
            return EXIT_OK
        return _cmd_run(args, service)
    except (FinSightError, ValueError) as exc:
        log.error("ingestion command failed", extra={"error": str(exc)})
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
