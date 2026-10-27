"""When does the daily run happen? Pure functions, so every rule is testable.

Rule: a daily run is **due** when today (in the configured time zone) is a run
day, the clock is past the daily time, and no daily run has started today.
Because the check repeats every minute, a laptop that was asleep or switched
off at 06:30 catches up as soon as it is back, and never runs twice in a day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class Schedule:
    """Daily run time, run days and time zone."""

    at: time
    days: frozenset[int]  # 0 = Monday
    tz: ZoneInfo

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Schedule:
        """Build from the ``scheduler`` section of ``params.yaml``."""
        hour, minute = (int(x) for x in str(params["daily_time"]).split(":"))
        days = frozenset(DAY_NAMES.index(str(d).lower()[:3]) for d in params["run_days"])
        if not days:
            raise ValueError("scheduler.run_days must name at least one day")
        return cls(at=time(hour, minute), days=days, tz=ZoneInfo(params["timezone"]))

    def local(self, now: datetime) -> datetime:
        """``now`` (timezone-aware) in the schedule's time zone."""
        return now.astimezone(self.tz)

    def slot(self, day: date) -> datetime:
        """The run time on ``day``."""
        return datetime.combine(day, self.at, tzinfo=self.tz)

    def is_due(self, now: datetime, last_daily_start: datetime | None) -> bool:
        """Whether a daily run should start now."""
        local = self.local(now)
        if local.weekday() not in self.days or local < self.slot(local.date()):
            return False
        return last_daily_start is None or self.local(last_daily_start).date() < local.date()

    def is_catch_up(self, now: datetime) -> bool:
        """A run starting more than 15 minutes after its slot is a catch-up."""
        local = self.local(now)
        return local - self.slot(local.date()) > timedelta(minutes=15)

    def next_run(self, now: datetime, last_daily_start: datetime | None) -> datetime:
        """The next time a daily run will start (``now`` if one is due)."""
        if self.is_due(now, last_daily_start):
            return self.local(now)
        local = self.local(now)
        for offset in range(0, 8):
            day = local.date() + timedelta(days=offset)
            if day.weekday() not in self.days:
                continue
            slot = self.slot(day)
            already = last_daily_start is not None and self.local(last_daily_start).date() >= day
            if slot > local and not already:
                return slot
        raise AssertionError("unreachable: a run day exists within a week")
