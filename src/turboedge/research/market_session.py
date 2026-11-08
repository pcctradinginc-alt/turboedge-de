"""Conservative German cash-market-session guard for research observation.

This is NOT a trading/execution calendar. It prevents scheduled EOD and
weekend/known-holiday snapshots from being misclassified as fresh tradeable
market evidence. An exchange-specific closure not in this calendar still
cannot generate a valid exit without actually fresh, timed bid/ask quotes.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
LOCAL_OPEN = time(8, 0)
# Leave a buffer before the late certificate close; the research fetch can
# take minutes. We never claim this equals an exchange's exact opening hours.
LOCAL_LAST_CAPTURE = time(21, 30)
ENTRY_WEEKDAYS = frozenset({0, 1, 2, 3})


def _easter_sunday(year: int) -> date:
    """Gregorian computus, for the fixed annual exchange holiday exclusions."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def is_conservative_market_day(day: date) -> bool:
    """Weekday, excluding the major non-trading German exchange holidays.

    The guard intentionally does not infer traded quotes from a holiday:
    unknown special closures still require real, fresh bid/ask timestamps.
    """
    if day.weekday() >= 5:
        return False
    easter = _easter_sunday(day.year)
    closures = {
        date(day.year, 1, 1),
        date(day.year, 5, 1),
        date(day.year, 12, 24),
        date(day.year, 12, 25),
        date(day.year, 12, 26),
        date(day.year, 12, 31),
        easter - timedelta(days=2),  # Good Friday
        easter + timedelta(days=1),  # Easter Monday
    }
    return day not in closures


def is_research_market_session(moment: datetime, *, entry: bool = False) -> bool:
    """Fail-closed window with automatic winter/summer time conversion."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("market session moment must have a timezone")
    local = moment.astimezone(BERLIN)
    if not is_conservative_market_day(local.date()):
        return False
    if entry and local.weekday() not in ENTRY_WEEKDAYS:
        # Friday entry would require a 24-48h weekend exit; exclude such
        # cohorts even if the issuer quotes a fresh price on Friday.
        return False
    return LOCAL_OPEN <= local.time().replace(tzinfo=None) <= LOCAL_LAST_CAPTURE


def observed_research_exit_is_market_quote(snapshot_time: datetime, retrieved_at: datetime) -> bool:
    """Both the issuer's quote and the archival receipt must be in-session."""
    return (
        is_research_market_session(snapshot_time)
        and is_research_market_session(retrieved_at)
    )


__all__ = [
    "is_conservative_market_day",
    "is_research_market_session",
    "observed_research_exit_is_market_quote",
]
