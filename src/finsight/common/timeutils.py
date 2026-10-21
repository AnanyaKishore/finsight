"""Date helpers for US equity end-of-day (EOD) data.

All dates are stored as timezone-aware UTC midnights. The helper
:func:`latest_expected_session` tells the ingestion whether a *new* trading
day can exist yet; skipping tickers that are already up to date saves
MarketStack quota (every ticker in a call costs one request even if the
response is empty).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
# MarketStack/Yahoo usually publish the day's EOD bar within ~2h of the 16:00 close.
EOD_AVAILABLE_AFTER = time(18, 0)

# NYSE full-day holidays up to the project deadline window (extend as needed).
US_MARKET_HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2025, 11, 27),
        date(2025, 12, 25),
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    }
)


def to_utc_midnight(day: date) -> datetime:
    """Return ``day`` as a timezone-aware UTC midnight datetime."""
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def is_trading_day(day: date) -> bool:
    """Weekday and not a listed NYSE holiday."""
    return day.weekday() < 5 and day not in US_MARKET_HOLIDAYS


def previous_trading_day(day: date) -> date:
    """Closest trading day strictly before ``day``."""
    candidate = day - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def latest_expected_session(now: datetime | None = None) -> date:
    """Most recent trading day whose EOD bar should already be published.

    Args:
        now: Current instant (any timezone); defaults to the real clock.
    """
    current = (now or datetime.now(UTC)).astimezone(NEW_YORK)
    today = current.date()
    if is_trading_day(today) and current.time() >= EOD_AVAILABLE_AFTER:
        return today
    return previous_trading_day(today)


def week_start(moment: datetime) -> datetime:
    """Monday 00:00 UTC of the week containing ``moment`` (sentiment aggregation key)."""
    day = moment.astimezone(UTC).date()
    return to_utc_midnight(day - timedelta(days=day.weekday()))
