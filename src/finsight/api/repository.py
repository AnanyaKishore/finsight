"""MongoDB access for the backend (read-mostly; forecasts are the only writes).

Collections read: ``prices``, ``sentiment_weekly``, ``portfolios`` (written
by ingestion and the DVC pipeline). Collection written: ``forecasts`` -- one
document per (ticker, as_of trading day, model version), so

* repeated requests on the same day reuse one model call, and
* every day's forecast is kept, which the monitoring job (Day 8) compares with
  the prices that actually followed (rolling forecast error).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from pymongo import ASCENDING, DESCENDING

from finsight.common.mongo import FORECASTS, PORTFOLIOS, PRICES, SENTIMENT_WEEKLY
from finsight.common.timeutils import is_trading_day

PERIOD_DAYS = {"1W": 7, "1M": 31, "3M": 92, "6M": 183, "1Y": 366, "ALL": None}


def as_date(value: Any) -> date:
    """Mongo datetime (UTC midnight) or date -> ``date``."""
    return value.date() if isinstance(value, datetime) else value


def next_trading_days(after: date, count: int) -> list[date]:
    """The ``count`` NYSE trading days strictly after ``after``."""
    days, current = [], after
    while len(days) < count:
        current += timedelta(days=1)
        if is_trading_day(current):
            days.append(current)
    return days


class MarketRepository:
    """Queries used by the API routes."""

    def __init__(self, db: Any) -> None:
        """Wrap a ``pymongo`` (or mongomock) database."""
        self.db = db

    # ---------------------------------------------------------------- prices
    def latest_bars(self, ticker: str, n: int) -> list[dict[str, Any]]:
        """The ``n`` most recent bars, oldest first."""
        cursor = (
            self.db[PRICES]
            .find({"ticker": ticker}, {"_id": 0})
            .sort("date", DESCENDING)
            .limit(int(n))
        )
        return list(reversed(list(cursor)))

    def history(self, ticker: str, period: str) -> list[dict[str, Any]]:
        """Bars for a period ending at the latest stored bar, oldest first."""
        latest = self.latest_bars(ticker, 1)
        if not latest:
            return []
        query: dict[str, Any] = {"ticker": ticker}
        days = PERIOD_DAYS[period]
        if days is not None:
            query["date"] = {"$gte": latest[0]["date"] - timedelta(days=days)}
        return list(self.db[PRICES].find(query, {"_id": 0}).sort("date", ASCENDING))

    def model_prices(self, ticker: str, n: int) -> tuple[list[date], list[float]]:
        """Last ``n`` model prices with the pipeline's rule (``validate.model_price``).

        ``adj_close`` if every stored bar of the ticker has it, else ``close``
        -- the same series the LSTM was trained on, so no training/serving skew.
        """
        use_adj = self.db[PRICES].count_documents({"ticker": ticker, "adj_close": None}) == 0
        bars = self.latest_bars(ticker, n)
        field = "adj_close" if use_adj else "close"
        return [as_date(b["date"]) for b in bars], [float(b[field]) for b in bars]

    # ------------------------------------------------------------- forecasts
    def get_forecast(self, ticker: str, as_of: date, model_version: str | None) -> dict | None:
        """Stored forecast for this origin and model version (``None`` if absent)."""
        query = {"ticker": ticker, "as_of": _dt(as_of), "model_version": model_version}
        return self.db[FORECASTS].find_one(query, {"_id": 0})

    def latest_forecast(self, ticker: str) -> dict | None:
        """Most recent stored forecast of any version (used when the model is down)."""
        return self.db[FORECASTS].find_one(
            {"ticker": ticker},
            {"_id": 0},
            sort=[("as_of", DESCENDING), ("generated_at", DESCENDING)],
        )

    def save_forecast(self, doc: dict[str, Any]) -> None:
        """Upsert one forecast (idempotent per ticker / origin / model version)."""
        key = {k: doc[k] for k in ("ticker", "as_of", "model_version")}
        self.db[FORECASTS].replace_one(key, doc, upsert=True)

    # ------------------------------------------------------------- sentiment
    def sentiment_weeks(self, ticker: str, weeks: int) -> list[dict[str, Any]]:
        """Most recent ``weeks`` weekly aggregates, oldest first."""
        cursor = (
            self.db[SENTIMENT_WEEKLY]
            .find({"ticker": ticker}, {"_id": 0})
            .sort("week_start", DESCENDING)
            .limit(int(weeks))
        )
        return list(reversed(list(cursor)))

    def sentiment_model_version(self, ticker: str) -> str | None:
        """Model fingerprint of the newest scored article for ``ticker``."""
        doc = self.db["news_sentiment"].find_one(
            {"ticker": ticker}, {"model_version": 1}, sort=[("published_at", DESCENDING)]
        )
        return doc.get("model_version") if doc else None

    # ------------------------------------------------------------- portfolio
    def latest_portfolio(self) -> dict[str, Any] | None:
        """The newest optimisation result."""
        return self.db[PORTFOLIOS].find_one({}, {"_id": 0}, sort=[("as_of", DESCENDING)])

    # ---------------------------------------------------------------- health
    def ping(self) -> None:
        """Raise if MongoDB is unreachable."""
        self.db.client.admin.command("ping")


def _dt(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)
