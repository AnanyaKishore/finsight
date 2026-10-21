"""Finnhub company-news adapter (the proposal's "news source").

Why Finnhub instead of NewsAPI (used by the MLOps_AI_App baseline): NewsAPI's
free tier only returns articles from the last ~30 days, so the baseline's
52-week backfill could never work. Finnhub's free tier provides about a year
of company news at 60 calls/minute, enough for the 6-month news backfill.

Endpoint: ``GET /company-news?symbol=AAPL&from=YYYY-MM-DD&to=YYYY-MM-DD&token=...``
returns a JSON list of ``{id, datetime (unix), headline, summary, source, url,
related, category, image}``.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from typing import Any

from finsight.common.exceptions import ProviderError
from finsight.common.http import HttpClient
from finsight.common.logging import get_logger
from finsight.ingestion.models import NewsArticle
from finsight.ingestion.providers.base import NewsProvider

log = get_logger(__name__)


def _news_id(item: dict[str, Any]) -> str:
    """Stable id: Finnhub's numeric id, else a hash of the URL/headline."""
    raw_id = item.get("id")
    if raw_id not in (None, 0, "0", ""):
        return str(raw_id)
    basis = str(item.get("url") or item.get("headline") or "")
    return "h" + hashlib.sha1(basis.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]


def parse_news_item(item: dict[str, Any], ticker: str) -> NewsArticle | None:
    """Convert one API item to a :class:`NewsArticle`; ``None`` if unusable."""
    headline = str(item.get("headline") or "").strip()
    timestamp = item.get("datetime")
    if not headline or not isinstance(timestamp, int | float) or timestamp <= 0:
        return None
    return NewsArticle(
        ticker=ticker,
        news_id=_news_id(item),
        headline=headline,
        summary=str(item.get("summary") or "").strip(),
        source=str(item.get("source") or ""),
        url=str(item.get("url") or ""),
        published_at=datetime.fromtimestamp(timestamp, tz=UTC),
        provider="finnhub",
        extra={"category": item.get("category"), "related": item.get("related")},
    )


def sample_evenly(articles: list[NewsArticle], limit: int) -> list[NewsArticle]:
    """Keep at most ``limit`` articles spread evenly over time.

    Taking simply the newest N would bias a weekly window towards its last day;
    even spacing keeps the weekly sentiment representative while bounding the
    CPU cost of DistilBERT scoring.
    """
    if limit <= 0 or len(articles) <= limit:
        return articles
    ordered = sorted(articles, key=lambda a: a.published_at)
    step = len(ordered) / limit
    return [ordered[int(i * step)] for i in range(limit)]


class FinnhubNewsProvider(NewsProvider):
    """Fetch company news from Finnhub."""

    name = "finnhub"

    def __init__(
        self, api_key: str, base_url: str, http: HttpClient, max_articles: int = 40
    ) -> None:
        """Create the adapter.

        Args:
            api_key: Finnhub token (from ``.env``; never logged).
            base_url: API root, e.g. ``https://finnhub.io/api/v1``.
            http: Shared HTTP client (rate-limited to the free tier).
            max_articles: Cap per (ticker, window) call; see :func:`sample_evenly`.
        """
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/company-news"
        self._http = http
        self._max_articles = max_articles

    def fetch_news(self, ticker: str, date_from: date, date_to: date) -> list[NewsArticle]:
        """Fetch, parse, de-duplicate and down-sample one ticker/window."""
        payload = self._http.get_json(
            self._url,
            {
                "symbol": ticker,
                "from": date_from.isoformat(),
                "to": date_to.isoformat(),
                "token": self._api_key,
            },
        )
        if isinstance(payload, dict) and payload.get("error"):
            raise ProviderError(str(payload["error"]), provider=self.name, retryable=False)
        if not isinstance(payload, list):
            raise ProviderError("unexpected payload (not a list)", provider=self.name)

        seen: set[str] = set()
        articles: list[NewsArticle] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            article = parse_news_item(item, ticker)
            if article is None or article.news_id in seen:
                continue
            seen.add(article.news_id)
            articles.append(article)
        kept = sample_evenly(articles, self._max_articles)
        log.debug(
            "news window fetched",
            extra={
                "ticker": ticker,
                "from": str(date_from),
                "received": len(payload),
                "kept": len(kept),
            },
        )
        return kept
