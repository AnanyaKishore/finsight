"""MarketStack end-of-day price adapter (primary price source, per the proposal).

API facts that shape this module (free plan, verified Oct 2026):

* 100 requests / month, up to 12 months of history, HTTPS supported.
* **Every ticker in a call counts as one request**, even if the response is
  empty. We therefore request one ticker per call (failures stay isolated per
  ticker and the cost is identical) and skip tickers that are already up to date.
* Errors arrive as ``{"error": {"code": ..., "message": ...}}``;
  ``usage_limit_reached`` means the monthly quota is gone -> QuotaExceededError,
  which the ingestion service answers by switching to the fallback provider.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from finsight.common.exceptions import ProviderError, QuotaExceededError
from finsight.common.http import HttpClient
from finsight.common.logging import get_logger
from finsight.common.timeutils import to_utc_midnight
from finsight.ingestion.models import PriceBar
from finsight.ingestion.providers.base import PriceProvider

log = get_logger(__name__)

PAGE_LIMIT = 1000  # maximum rows per page allowed by the API
_QUOTA_CODES = {"usage_limit_reached"}
_FATAL_CODES = {
    "invalid_access_key",
    "missing_access_key",
    "inactive_user",
    "https_access_restricted",
    "function_access_restricted",
}


def classify_marketstack(status: int, body: Any) -> None:
    """Map MarketStack error payloads to FinSight exceptions.

    Raises:
        QuotaExceededError: Monthly usage limit reached.
        ProviderError: Non-retryable account/configuration errors.
    """
    if not isinstance(body, dict) or "error" not in body:
        return
    error = body.get("error") or {}
    code = str(error.get("code", "")).lower()
    message = str(error.get("message", "unknown error"))
    if code in _QUOTA_CODES:
        raise QuotaExceededError(f"monthly quota exhausted: {message}", provider="marketstack")
    if code in _FATAL_CODES:
        raise ProviderError(
            f"{code}: {message}", provider="marketstack", retryable=False, status=status
        )
    # Anything else (rate_limit_reached, internal_error...) falls through to the
    # generic HTTP-status rules (429/5xx retryable).


def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def parse_eod_row(row: dict[str, Any], ticker: str) -> PriceBar | None:
    """Convert one API row to a :class:`PriceBar`; return ``None`` if unusable."""
    raw_date = row.get("date")
    close = _to_float(row.get("close"))
    if not raw_date or close is None:
        return None
    try:
        day = date.fromisoformat(str(raw_date)[:10])
    except ValueError:
        return None
    open_, high, low = (_to_float(row.get(k)) for k in ("open", "high", "low"))
    return PriceBar(
        ticker=ticker,
        date=to_utc_midnight(day),
        open=open_ if open_ is not None else close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=_to_float(row.get("volume")) or 0.0,
        adj_close=_to_float(row.get("adj_close")),
        source="marketstack",
    )


class MarketStackProvider(PriceProvider):
    """Fetch EOD bars from MarketStack (``/eod`` endpoint)."""

    name = "marketstack"
    metered = True

    def __init__(self, api_key: str, base_url: str, http: HttpClient) -> None:
        """Create the adapter.

        Args:
            api_key: MarketStack access key (from ``.env``; never logged).
            base_url: API root, e.g. ``https://api.marketstack.com/v2``.
            http: Shared HTTP client configured with :func:`classify_marketstack`.
        """
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/eod"
        self._http = http

    def cost(self, ticker: str, date_from: date, date_to: date) -> int:
        """One request per page; a page holds up to 1000 trading days."""
        approx_trading_days = max(1, int((date_to - date_from).days * 252 / 365) + 1)
        return math.ceil(approx_trading_days / PAGE_LIMIT)

    def fetch_eod(self, ticker: str, date_from: date, date_to: date) -> list[PriceBar]:
        """Fetch all pages of EOD data for one ticker (inclusive date range)."""
        bars: list[PriceBar] = []
        offset = 0
        while True:
            payload = self._http.get_json(
                self._url,
                {
                    "access_key": self._api_key,
                    "symbols": ticker,
                    "date_from": date_from.isoformat(),
                    "date_to": date_to.isoformat(),
                    "sort": "ASC",
                    "limit": PAGE_LIMIT,
                    "offset": offset,
                },
            )
            rows = payload.get("data") if isinstance(payload, dict) else None
            if rows is None:
                raise ProviderError(
                    "unexpected payload (no 'data' key)", provider=self.name, retryable=False
                )
            skipped = 0
            for row in rows:
                bar = parse_eod_row(row, ticker)
                if bar is None:
                    skipped += 1
                else:
                    bars.append(bar)
            if skipped:
                log.warning("skipped malformed rows", extra={"ticker": ticker, "skipped": skipped})

            pagination = payload.get("pagination") or {}
            count = int(pagination.get("count", len(rows)) or 0)
            total = int(pagination.get("total", 0) or 0)
            offset += count
            if count == 0 or offset >= total:
                break
        bars.sort(key=lambda b: b.date)
        return bars
