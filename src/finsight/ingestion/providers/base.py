"""Provider interfaces (Strategy pattern).

:class:`~finsight.ingestion.service.IngestionService` depends only on these
abstract classes, so a provider can be swapped (or a fallback chained) without
touching the orchestration logic -- and tests can inject fakes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from finsight.ingestion.models import NewsArticle, PriceBar


class PriceProvider(ABC):
    """Source of end-of-day price bars."""

    #: Short name used in logs, run records and the ``source`` field.
    name: str = "abstract"
    #: Whether calls consume a metered monthly quota (checked by the ledger).
    metered: bool = False

    @abstractmethod
    def fetch_eod(self, ticker: str, date_from: date, date_to: date) -> list[PriceBar]:
        """Return bars for ``ticker`` with ``date_from <= date <= date_to`` (inclusive)."""

    def cost(self, ticker: str, date_from: date, date_to: date) -> int:
        """Quota units one :meth:`fetch_eod` call costs (default: 1 per ticker)."""
        return 1


class NewsProvider(ABC):
    """Source of company news articles."""

    name: str = "abstract"

    @abstractmethod
    def fetch_news(self, ticker: str, date_from: date, date_to: date) -> list[NewsArticle]:
        """Return articles for ``ticker`` published within ``[date_from, date_to]``."""
