"""yfinance adapter -- *fallback* price source only.

Why a fallback exists: MarketStack's free plan allows 100 requests/month and
charges one request per ticker, so 10 tickers x ~21 trading days = 210 > 100.
Rather than silently stopping, the ingestion degrades gracefully to Yahoo
Finance via ``yfinance`` (no key, unmetered, unofficial). Every bar records its
``source``, so lineage is preserved and the data-quality stage (Day 2) can
report the provider mix.

``yfinance`` is imported lazily so unit tests and other images do not need it.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from finsight.common.exceptions import ProviderError
from finsight.common.logging import get_logger
from finsight.common.timeutils import to_utc_midnight
from finsight.ingestion.models import PriceBar
from finsight.ingestion.providers.base import PriceProvider

log = get_logger(__name__)


class YFinanceProvider(PriceProvider):
    """Fetch EOD bars from Yahoo Finance through the ``yfinance`` package."""

    name = "yfinance"
    metered = False

    def __init__(self, downloader: Any | None = None) -> None:
        """Create the adapter.

        Args:
            downloader: Callable with the signature of ``yfinance.download``;
                injected in tests. Defaults to the real function (lazy import).
        """
        self._download = downloader

    def _downloader(self) -> Any:
        if self._download is None:
            try:
                import yfinance  # noqa: PLC0415 -- optional dependency, imported on demand
            except ImportError as exc:
                raise ProviderError(
                    "yfinance is not installed", provider=self.name, retryable=False
                ) from exc
            self._download = yfinance.download
        return self._download

    def fetch_eod(self, ticker: str, date_from: date, date_to: date) -> list[PriceBar]:
        """Fetch daily bars; ``yfinance``'s ``end`` is exclusive so we add one day."""
        try:
            frame = self._downloader()(
                ticker,
                start=date_from.isoformat(),
                end=(date_to + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
        except Exception as exc:  # yfinance raises many ad-hoc exception types
            raise ProviderError(
                f"download failed: {exc}", provider=self.name, retryable=True
            ) from exc

        if frame is None or len(frame) == 0:
            return []
        bars: list[PriceBar] = []
        for index, row in frame.iterrows():
            try:
                close = float(row["Close"])
            except (KeyError, TypeError, ValueError):
                continue
            if close != close:  # NaN check without numpy
                continue
            day = index.date() if hasattr(index, "date") else date.fromisoformat(str(index)[:10])
            adj = row.get("Adj Close")
            bars.append(
                PriceBar(
                    ticker=ticker,
                    date=to_utc_midnight(day),
                    open=float(row.get("Open", close)),
                    high=float(row.get("High", close)),
                    low=float(row.get("Low", close)),
                    close=close,
                    volume=float(row.get("Volume", 0) or 0),
                    adj_close=None if adj is None or adj != adj else float(adj),
                    source=self.name,
                )
            )
        return bars
