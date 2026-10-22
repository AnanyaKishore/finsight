"""DVC stage ``validate``: automated data-quality checks on the exported snapshot.

The suite borrows the vocabulary of Great Expectations (Module 2, Handout 3):
each *expectation* is a named, testable assertion with a severity, and the
*suite* result is a JSON document (``reports/data_quality.json``) that DVC
tracks as a metric and the Pipeline Console renders on Day 7.

Severity policy:

* ``error`` -- the data would silently corrupt training or serving (schema
  broken, duplicate keys, impossible prices, too little history). The stage
  exits non-zero and ``dvc repro`` stops: no model is trained on bad data.
* ``warn``  -- worth knowing but legitimately possible (a few missing trading
  days, a genuine market shock, stale news). Reported, never blocking.
* ``info``  -- descriptive statistics (provider mix, coverage).

Why custom code instead of the ``great_expectations`` package: the suite is
~20 checks on a few thousand rows; GE would add a large dependency tree and a
config-heavy project layout for no extra capability here. The behaviour that
matters is the same -- declarative checks, a persisted report, a gate that
*fails* on a dataset with known problems (verified by the unit tests).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import DataValidationError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.common.timeutils import is_trading_day, latest_expected_session

log = get_logger(__name__)

ERROR, WARN, INFO = "error", "warn", "info"
MAX_LISTED = 25  # cap list-valued details so the report stays readable

PRICE_SCHEMA = {
    "ticker": "string",
    "date": "date",
    "open": "float",
    "high": "float",
    "low": "float",
    "close": "float",
    "adj_close": "float",
    "volume": "float",
    "source": "string",
}
NEWS_SCHEMA = {
    "ticker": "string",
    "news_id": "string",
    "published_at": "datetime",
    "headline": "string",
}


@dataclass
class CheckResult:
    """Outcome of one expectation."""

    name: str
    severity: str
    success: bool
    observed: Any = None
    details: Any = None

    @property
    def blocking(self) -> bool:
        """True when this result must stop the pipeline."""
        return self.severity == ERROR and not self.success


@dataclass
class SuiteResult:
    """All expectation outcomes for one snapshot."""

    as_of_session: str
    checks: list[CheckResult] = field(default_factory=list)
    per_ticker: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True when no blocking check failed."""
        return not any(c.blocking for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready report (deterministic: no wall-clock timestamp)."""
        failed = [c for c in self.checks if not c.success]
        return {
            "status": "pass" if self.success else "fail",
            "as_of_session": self.as_of_session,
            "summary": {
                "checks": len(self.checks),
                "passed": len(self.checks) - len(failed),
                "errors": sum(c.severity == ERROR for c in failed),
                "warnings": sum(c.severity == WARN for c in failed),
            },
            "checks": [_jsonable(asdict(c)) for c in self.checks],
            "per_ticker": _jsonable(self.per_ticker),
        }


def _jsonable(value: Any) -> Any:
    """Convert numpy / pandas / date objects into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, float):
        return None if value != value else round(value, 6)
    if isinstance(value, date | datetime | pd.Timestamp):
        return (
            str(value)[:10]
            if isinstance(value, date) and not isinstance(value, datetime)
            else str(value)
        )
    return value


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def trading_days_between(start: date, end: date) -> list[date]:
    """NYSE trading days in ``[start, end]``."""
    days, current = [], start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def model_price(prices: pd.DataFrame) -> pd.Series:
    """Price series used for modelling.

    Per ticker: ``adj_close`` when *every* bar of that ticker has it (split- and
    dividend-adjusted), otherwise ``close``. Mixing the two within one series
    would create artificial jumps. The Spark ETL applies the same rule.
    """
    has_all_adj = prices.groupby("ticker")["adj_close"].transform(lambda s: s.notna().all())
    return prices["adj_close"].where(has_all_adj, prices["close"])


def _schema_check(frame: pd.DataFrame, schema: dict[str, str], name: str) -> CheckResult:
    missing = [c for c in schema if c not in frame.columns]
    wrong = []
    for column, kind in schema.items():
        if column in missing:
            continue
        series = frame[column]
        ok = {
            "float": pd.api.types.is_float_dtype(series),
            "string": pd.api.types.is_string_dtype(series) or series.dtype == object,
            "datetime": pd.api.types.is_datetime64_any_dtype(series),
            "date": series.dropna().map(lambda v: isinstance(v, date)).all(),
        }[kind]
        if not ok:
            wrong.append(f"{column}: expected {kind}, got {series.dtype}")
    return CheckResult(
        name,
        ERROR,
        not missing and not wrong,
        observed=list(frame.columns),
        details={"missing": missing, "wrong_type": wrong},
    )


# --------------------------------------------------------------------------- #
# Price expectations
# --------------------------------------------------------------------------- #
def validate_prices(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    params: dict[str, Any],
    as_of: date,
    suite: SuiteResult,
) -> None:
    """Append all price expectations to ``suite``."""
    schema = _schema_check(prices, PRICE_SCHEMA, "prices.schema")
    suite.checks.append(schema)
    if not schema.success:
        return  # later checks would only produce confusing secondary errors

    add = suite.checks.append
    nulls = {c: int(prices[c].isna().sum()) for c in ("ticker", "date", "close")}
    add(CheckResult("prices.required_not_null", ERROR, sum(nulls.values()) == 0, nulls))
    optional_nulls = {c: int(prices[c].isna().sum()) for c in ("open", "high", "low", "volume")}
    add(
        CheckResult(
            "prices.optional_not_null", WARN, sum(optional_nulls.values()) == 0, optional_nulls
        )
    )

    dupes = prices[prices.duplicated(["ticker", "date"], keep=False)]
    add(
        CheckResult(
            "prices.unique_ticker_date",
            ERROR,
            dupes.empty,
            int(len(dupes)),
            dupes[["ticker", "date"]].head(MAX_LISTED).astype(str).values.tolist(),
        )
    )

    bad_price = prices[(prices["close"] <= 0) | (prices["adj_close"] <= 0) | (prices["volume"] < 0)]
    add(
        CheckResult(
            "prices.positive_values",
            ERROR,
            bad_price.empty,
            int(len(bad_price)),
            bad_price[["ticker", "date", "close", "adj_close", "volume", "source"]]
            .head(MAX_LISTED)
            .astype(str)
            .values.tolist()
            + (["fix: python -m finsight.ingestion.cli repair"] if len(bad_price) else []),
        )
    )

    tol = 1e-6 * prices["close"].abs()
    ohlc_bad = prices[
        (prices["low"] > prices[["open", "close"]].min(axis=1) + tol)
        | (prices["high"] < prices[["open", "close"]].max(axis=1) - tol)
    ]
    add(
        CheckResult(
            "prices.ohlc_consistent",
            WARN,
            ohlc_bad.empty,
            int(len(ohlc_bad)),
            ohlc_bad[["ticker", "date"]].head(MAX_LISTED).astype(str).values.tolist(),
        )
    )

    present = set(prices["ticker"].unique())
    absent = sorted(set(tickers) - present)
    add(
        CheckResult(
            "prices.universe_complete", ERROR, not absent, sorted(present), {"absent": absent}
        )
    )

    non_trading = prices[~prices["date"].map(is_trading_day)]
    add(
        CheckResult(
            "prices.only_trading_days",
            WARN,
            non_trading.empty,
            int(len(non_trading)),
            non_trading[["ticker", "date"]].head(MAX_LISTED).astype(str).values.tolist(),
        )
    )

    # ---- per-ticker checks -------------------------------------------------
    priced = prices.assign(price=model_price(prices)).sort_values(["ticker", "date"])
    # Non-positive prices are already reported by prices.positive_values; mask
    # them here so they do not also produce infinite "returns".
    positive = priced["price"].where(priced["price"] > 0)
    priced["log_return"] = positive.groupby(priced["ticker"]).transform(
        lambda s: np.log(s / s.shift(1))
    )
    global_end = priced["date"].max()
    missing_warn, missing_err, short, stale, extreme, outliers = {}, {}, {}, {}, [], []
    expected_last = as_of
    lag_limit = int(params["max_staleness_days"])

    for ticker, group in priced.groupby("ticker"):
        dates = set(group["date"])
        calendar = trading_days_between(group["date"].min(), global_end)
        missing = [d for d in calendar if d not in dates]
        rows = int(len(group))
        last = group["date"].max()
        lag = len(trading_days_between(last + timedelta(days=1), expected_last))
        returns = group["log_return"].dropna()
        median = returns.median()
        mad = (returns - median).abs().median() * 1.4826
        z = (returns - median) / mad if mad and mad > 0 else returns * 0
        flagged = group.loc[z[z.abs() > float(params["return_outlier_z"])].index]
        too_big = group[group["log_return"].abs() > float(params["max_abs_log_return"])]

        suite.per_ticker[ticker] = {
            "rows": rows,
            "first_date": group["date"].min(),
            "last_date": last,
            "expected_trading_days": len(calendar),
            "missing_trading_days": len(missing),
            "missing_dates": missing[:MAX_LISTED],
            "sessions_behind": lag,
            "return_mean": returns.mean(),
            "return_std": returns.std(),
            "outlier_days": int(len(flagged)),
            "sources": sorted(group["source"].dropna().unique().tolist()),
            "price_field": "adj_close" if group["adj_close"].notna().all() else "close",
        }
        if len(missing) > int(params["max_missing_days_error"]):
            missing_err[ticker] = len(missing)
        elif len(missing) > int(params["max_missing_days_warn"]):
            missing_warn[ticker] = len(missing)
        if rows < int(params["min_history_rows"]):
            short[ticker] = rows
        if lag > lag_limit:
            stale[ticker] = lag
        outliers += [
            [ticker, str(r.date), round(float(r.log_return), 4)] for r in flagged.itertuples()
        ]
        extreme += [
            [ticker, str(r.date), round(float(r.log_return), 4)] for r in too_big.itertuples()
        ]

    add(CheckResult("prices.missing_days_blocking", ERROR, not missing_err, missing_err))
    add(CheckResult("prices.missing_days", WARN, not missing_warn, missing_warn))
    add(
        CheckResult(
            "prices.min_history",
            ERROR,
            not short,
            short,
            {"required_rows": int(params["min_history_rows"])},
        )
    )
    add(CheckResult("prices.fresh", WARN, not stale, stale, {"expected_last_session": as_of}))
    add(
        CheckResult(
            "prices.no_extreme_returns", ERROR, not extreme, len(extreme), extreme[:MAX_LISTED]
        )
    )
    add(
        CheckResult(
            "prices.return_outliers", WARN, not outliers, len(outliers), outliers[:MAX_LISTED]
        )
    )
    add(
        CheckResult(
            "prices.provider_mix",
            INFO,
            True,
            {str(k): int(v) for k, v in prices["source"].value_counts().items()},
        )
    )


# --------------------------------------------------------------------------- #
# News expectations
# --------------------------------------------------------------------------- #
def validate_news(
    news: pd.DataFrame, tickers: Sequence[str], params: dict[str, Any], suite: SuiteResult
) -> None:
    """Append all news expectations to ``suite``."""
    schema = _schema_check(news, NEWS_SCHEMA, "news.schema")
    suite.checks.append(schema)
    if not schema.success:
        return
    add = suite.checks.append
    nulls = {c: int(news[c].isna().sum()) for c in NEWS_SCHEMA}
    blank = int((news["headline"].fillna("").str.strip() == "").sum())
    add(
        CheckResult(
            "news.required_not_null",
            ERROR,
            sum(nulls.values()) + blank == 0,
            nulls,
            {"blank_headlines": blank},
        )
    )
    dupes = int(news.duplicated(["ticker", "news_id"]).sum())
    add(CheckResult("news.unique_ticker_id", ERROR, dupes == 0, dupes))
    absent = sorted(set(tickers) - set(news["ticker"].unique()))
    add(CheckResult("news.universe_complete", WARN, not absent, {"absent": absent}))

    if news.empty:
        return
    # Monday-start weeks, matching the sentiment aggregation key.
    week = (
        news["published_at"].dt.tz_convert("UTC").dt.normalize()
        - pd.to_timedelta(news["published_at"].dt.weekday, unit="D")
    ).dt.date
    counts = news.assign(week=week).groupby(["ticker", "week"]).size()
    all_weeks = sorted(week.unique())[1:-1]  # first/last weeks are partial by construction
    thin: dict[str, int] = {}
    for ticker in sorted(news["ticker"].unique()):
        per = counts.get(ticker, pd.Series(dtype=int))
        n_thin = sum(int(per.get(w, 0)) < int(params["min_articles_per_week"]) for w in all_weeks)
        if n_thin:
            thin[ticker] = n_thin
        suite.per_ticker.setdefault(ticker, {}).update(
            {
                "news_articles": int(per.sum()),
                "news_weeks": int((per > 0).sum()),
                "news_thin_weeks": n_thin,
            }
        )
    add(
        CheckResult(
            "news.weekly_coverage",
            WARN,
            not thin,
            thin,
            {
                "min_articles_per_week": int(params["min_articles_per_week"]),
                "weeks_checked": len(all_weeks),
            },
        )
    )
    empty_summary = (
        float((news["summary"].fillna("").str.strip() == "").mean()) if "summary" in news else 0.0
    )
    add(CheckResult("news.summary_present_fraction", INFO, True, round(1 - empty_summary, 4)))


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
def run_suite(
    prices: pd.DataFrame,
    news: pd.DataFrame,
    tickers: Sequence[str],
    params: dict[str, Any],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SuiteResult:
    """Run every expectation and return the suite result."""
    as_of = latest_expected_session(now())
    suite = SuiteResult(as_of_session=as_of.isoformat())
    validate_prices(prices, tickers, params, as_of, suite)
    validate_news(news, tickers, params, suite)
    return suite


def _print_summary(report: dict[str, Any]) -> None:
    print(f"\nData quality: {report['status'].upper()}  ({report['summary']})")
    for check in report["checks"]:
        mark = "ok  " if check["success"] else ("FAIL" if check["severity"] == ERROR else "warn")
        observed = json.dumps(check["observed"])
        print(f"  [{mark}] {check['name']:<32} {observed[:90]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.data.validate``)."""
    argparse.ArgumentParser(description="Validate the exported snapshot").parse_args(argv)
    settings = get_settings()
    configure_logging("validate", settings.log_level, settings.root / "logs")
    params = settings.section("validation")
    export = settings.section("export")
    try:
        with stage_run("validate", settings) as run:
            prices = pd.read_parquet(settings.root / export["prices_path"])
            news = pd.read_parquet(settings.root / export["news_path"])
            suite = run_suite(prices, news, settings.symbols, params)
            report = suite.to_dict()
            path = settings.root / params["report_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run.metrics = report["summary"] | {"status": report["status"]}
            _print_summary(report)
            if not suite.success:
                failing = [c.name for c in suite.checks if c.blocking]
                raise DataValidationError(f"blocking data-quality checks failed: {failing}")
    except DataValidationError as exc:
        print(f"\nvalidate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
