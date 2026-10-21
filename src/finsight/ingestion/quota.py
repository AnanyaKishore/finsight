"""Persistent API request ledger (quota guard).

Each metered provider has one Mongo document per calendar month::

    {"provider": "marketstack", "period": "2026-10", "calls": 37, "budget": 100}

:meth:`QuotaLedger.reserve` *atomically* checks and increments the counter
before a call is made, so even concurrent jobs can never overspend the budget.
When the provider itself reports the quota as exhausted, :meth:`mark_exhausted`
syncs the ledger so later runs go straight to the fallback provider.

Note: MarketStack's billing month starts on your sign-up day, not the 1st.
The calendar month is a conservative approximation; use
``python -m finsight.ingestion.cli quota --set-used N`` to sync the counter
with the number shown on the MarketStack dashboard.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from pymongo.database import Database

from finsight.common.exceptions import QuotaExceededError
from finsight.common.logging import get_logger
from finsight.common.mongo import API_USAGE

log = get_logger(__name__)


class QuotaLedger:
    """Monthly request budget for one provider."""

    def __init__(
        self,
        db: Database,
        provider: str,
        budget: int,
        reserve: int = 0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Create a ledger.

        Args:
            db: Mongo database.
            provider: Provider name (document key).
            budget: Requests allowed per month.
            reserve: Requests kept back for emergencies / manual debugging.
            clock: Injected clock (tests).
        """
        self._coll = db[API_USAGE]
        self.provider = provider
        self.budget = budget
        self.reserve = reserve
        self._clock = clock

    @property
    def period(self) -> str:
        """Current ledger period, e.g. ``"2026-10"``."""
        return self._clock().strftime("%Y-%m")

    @property
    def usable(self) -> int:
        """Requests that may be spent automatically this month."""
        return max(0, self.budget - self.reserve)

    def _ensure(self) -> None:
        self._coll.update_one(
            {"provider": self.provider, "period": self.period},
            {"$setOnInsert": {"calls": 0, "created_at": self._clock()}},
            upsert=True,
        )

    def used(self) -> int:
        """Requests already spent this period."""
        doc = self._coll.find_one({"provider": self.provider, "period": self.period})
        return int(doc["calls"]) if doc else 0

    def remaining(self) -> int:
        """Requests still available for automatic use this period."""
        return max(0, self.usable - self.used())

    def reserve_calls(self, n: int) -> None:
        """Atomically reserve ``n`` requests or raise.

        Raises:
            QuotaExceededError: If fewer than ``n`` usable requests remain.
        """
        self._ensure()
        result = self._coll.update_one(
            {"provider": self.provider, "period": self.period, "calls": {"$lte": self.usable - n}},
            {"$inc": {"calls": n}, "$set": {"budget": self.budget, "updated_at": self._clock()}},
        )
        if result.modified_count != 1:
            raise QuotaExceededError(
                f"ledger: {self.used()}/{self.usable} usable requests spent in {self.period}",
                provider=self.provider,
            )

    def refund(self, n: int) -> None:
        """Give back ``n`` reserved requests (e.g. the call was never sent)."""
        self._coll.update_one(
            {"provider": self.provider, "period": self.period}, {"$inc": {"calls": -n}}
        )

    def mark_exhausted(self) -> None:
        """Provider said the quota is gone: pin the counter at the budget."""
        self._ensure()
        self._coll.update_one(
            {"provider": self.provider, "period": self.period},
            {"$max": {"calls": self.budget}, "$set": {"updated_at": self._clock()}},
        )
        log.warning("quota marked exhausted", extra={"provider": self.provider})

    def set_used(self, calls: int) -> None:
        """Manually sync the counter with the provider dashboard."""
        self._ensure()
        self._coll.update_one(
            {"provider": self.provider, "period": self.period},
            {"$set": {"calls": int(calls), "updated_at": self._clock()}},
        )
