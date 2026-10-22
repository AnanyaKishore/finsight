"""Plain data records produced by providers and stored by the repository.

Providers translate each API's JSON into these provider-independent records,
so the rest of the system never depends on MarketStack- or Finnhub-specific
field names (loose coupling; swapping a provider touches one module).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def _number(value: Any) -> float | None:
    """Float, or ``None`` for missing / non-numeric / NaN / infinite values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def make_price_bar(
    ticker: str,
    day: datetime,
    *,
    open_: Any,
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
    adj_close: Any,
    source: str,
) -> PriceBar | None:
    """Build a :class:`PriceBar` from raw provider values, or reject it.

    Free data feeds occasionally return ``0`` (or nothing) instead of a price.
    A zero close turns into an infinite log return and corrupts every model
    window that contains it, so these rules are applied at the system boundary:

    * close missing or <= 0       -> bar rejected (the day becomes a gap, which the
      ETL forward-fills if short, or the fallback provider fills on ``repair``)
    * adj_close missing or <= 0   -> ``None`` (the ticker's model price then uses
      raw ``close``; see ``finsight.data.validate.model_price``)
    * open/high/low missing or <= 0 -> replaced by close
    * volume missing or < 0       -> 0
    """
    close_v = _number(close)
    if close_v is None or close_v <= 0:
        return None

    def _positive_or_close(value: Any) -> float:
        number = _number(value)
        return number if number is not None and number > 0 else close_v

    adj = _number(adj_close)
    vol = _number(volume)
    return PriceBar(
        ticker=ticker,
        date=day,
        open=_positive_or_close(open_),
        high=_positive_or_close(high),
        low=_positive_or_close(low),
        close=close_v,
        volume=vol if vol is not None and vol >= 0 else 0.0,
        adj_close=adj if adj is not None and adj > 0 else None,
        source=source,
    )


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
