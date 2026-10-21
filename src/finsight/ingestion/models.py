"""Plain data records produced by providers and stored by the repository.

Providers translate each API's JSON into these provider-independent records,
so the rest of the system never depends on MarketStack- or Finnhub-specific
field names (loose coupling; swapping a provider touches one module).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PriceBar:
    """One end-of-day OHLCV bar.

    Attributes:
        ticker: Upper-case symbol, e.g. ``"AAPL"``.
        date: Trading day as a UTC-midnight, timezone-aware datetime.
        open/high/low/close: Raw (unadjusted) prices in USD.
        volume: Shares traded.
        adj_close: Split/dividend-adjusted close when the provider supplies it.
        source: Provider that produced the bar (data lineage).
    """

    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float | None = None
    source: str = "unknown"

    def to_document(self) -> dict[str, Any]:
        """Mongo document representation."""
        return asdict(self)


@dataclass(frozen=True)
class NewsArticle:
    """One company news article.

    Attributes:
        ticker: Company symbol the article was requested for.
        news_id: Provider article id (string) used for de-duplication.
        headline: Article title.
        summary: Short description / lede (may be empty).
        source: Publisher name.
        url: Link to the article.
        published_at: Publication time, timezone-aware UTC.
        provider: Data provider (lineage), e.g. ``"finnhub"``.
    """

    ticker: str
    news_id: str
    headline: str
    summary: str
    source: str
    url: str
    published_at: datetime
    provider: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model_text(self) -> str:
        """Text fed to the sentiment model (headline + summary, truncated)."""
        text = self.headline.strip()
        if self.summary.strip():
            text = f"{text}. {self.summary.strip()}"
        return text[:1000]

    def to_document(self) -> dict[str, Any]:
        """Mongo document representation (adds the model input text)."""
        doc = asdict(self)
        doc["text"] = self.model_text
        return doc
