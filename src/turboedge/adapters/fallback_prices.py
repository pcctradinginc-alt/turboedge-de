"""yfinance daily OHLC fallback for underlying prices.

Only used as a fallback (Master Spec Section 5.2: "Inoffizielle Quellen
niemals als einzige tragende Abhaengigkeit verwenden" - never rely on an
unofficial source as the sole supporting dependency). Ticker symbols and
metadata are resolved from :mod:`turboedge.universe.underlying_map`, so
callers only ever deal in canonical ``underlying_id`` values, never raw
yfinance symbols.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import structlog
import yfinance as yf

from turboedge.adapters.base import AdapterError, AdapterMetadata, HealthCheckResult
from turboedge.storage.schemas import HealthStatus, UnderlyingBar
from turboedge.universe.underlying_map import get_underlying_meta

logger = structlog.get_logger(__name__)

_PARSER_VERSION = "1"
_SOURCE_NAME = "yfinance"
_COMPLETE_BAR_QUALITY = 0.9
_HEALTHCHECK_PROBE_UNDERLYING = "DAX"


@dataclass(frozen=True)
class _SessionClose:
    """A conservative exchange-local daily-bar close, used to derive ``available_at``."""

    tz: str
    hour: int
    minute: int = 0


# Conservative daily-bar availability per canonical underlying_id (kept local
# to this adapter, deliberately *not* added to underlying_map.py: this is
# fallback-source close-timing plumbing, not shared universe metadata).
#
# yfinance's daily bar for an equity index is timestamped at local midnight
# of the trading day, but the bar's OHLC is only fully known -- and thus only
# safe to use without look-ahead (CLAUDE.md rule 5/17) -- once that day's
# cash session has actually closed. DAX/ESTX50 use 22:00 Europe/Berlin
# (rather than the ~17:30 Xetra continuous-trading close) to also cover the
# Boerse Stuttgart/Tradegate late-trading closing-auction prints that can
# still land in the same yfinance daily bar.
#
# Any underlying_id *not* listed here (FX, metals, energy -- all effectively
# 24h markets -- and anything unrecognized) falls back to the conservative
# rule in `_session_close_utc`: the daily bar only becomes usable at the
# start of the *next* calendar day, in the bar's own (yfinance index)
# timezone.
_EQUITY_SESSION_CLOSE: dict[str, _SessionClose] = {
    "DAX": _SessionClose("Europe/Berlin", 22),
    "ESTX50": _SessionClose("Europe/Berlin", 22),
    "SPX": _SessionClose("America/New_York", 17),
    "NDX": _SessionClose("America/New_York", 17),
    "NKY": _SessionClose("Asia/Tokyo", 16),
    "UKX": _SessionClose("Europe/London", 17),
    "SMI": _SessionClose("Europe/Zurich", 18),
}


def _to_utc_datetime(ts: Any) -> datetime:
    """Convert a yfinance index entry to a tz-aware UTC ``datetime``.

    A tz-naive entry is documented as already being UTC (never guessed at
    another zone) -- this only matters for malformed/synthetic input, since
    real yfinance history indices are always tz-aware.
    """
    if isinstance(ts, pd.Timestamp):
        py_dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        py_dt = ts
    else:
        raise AdapterError(f"unexpected yfinance index type: {type(ts)!r}")
    if py_dt.tzinfo is None:
        py_dt = py_dt.replace(tzinfo=UTC)
    return py_dt.astimezone(UTC)


def _index_tz_and_trade_date(ts: Any) -> tuple[Any, date]:
    """The yfinance index entry's OWN timezone and calendar date within it.

    yfinance daily bars are indexed at local midnight of the exchange (or,
    for 24h markets, the data-vendor's own trading-day cutoff) in the
    instrument's *own* timezone -- e.g. ``^GDAXI`` uses Europe/Berlin,
    ``^NDX`` uses America/New_York, FX/futures pairs use whatever
    yfinance/exchange timezone the venue reports. Converting to UTC first
    and then taking ``.date()`` (the previous bug) silently shifts the
    trading date for any exchange west or east of UTC. A tz-naive entry is
    documented as already being UTC (never guessed).
    """
    if isinstance(ts, pd.Timestamp):
        py_dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        py_dt = ts
    else:
        raise AdapterError(f"unexpected yfinance index type: {type(ts)!r}")
    if py_dt.tzinfo is None:
        py_dt = py_dt.replace(tzinfo=UTC)
    return py_dt.tzinfo, py_dt.date()


def _zone_arg(tz: Any) -> Any:
    """Best-effort stable zone identifier, reusable as a fresh ``pd.Timestamp(tz=...)``.

    Prefers an IANA zone key (pytz's ``.zone``, ``zoneinfo.ZoneInfo``'s
    ``.key``) over the raw tzinfo instance: a pytz tzinfo attached to one
    particular timestamp is a DST-snapshotted fixed offset, and reusing its
    zone *name* (rather than that instance) lets pandas relocalize with
    correct DST rules for a *different* calendar date -- needed for the
    24h-market "+1 day" rule below. Falls back to the tzinfo object itself
    for fixed-offset zones (e.g. plain UTC) that have neither attribute.
    """
    return getattr(tz, "zone", None) or getattr(tz, "key", None) or tz


def _session_close_utc(trade_date: date, index_tz: Any, underlying_id: str) -> datetime:
    """Conservative UTC instant at which the daily bar for ``trade_date`` is usable."""
    equity_close = _EQUITY_SESSION_CLOSE.get(underlying_id)
    if equity_close is not None:
        local_close = pd.Timestamp(
            year=trade_date.year,
            month=trade_date.month,
            day=trade_date.day,
            hour=equity_close.hour,
            minute=equity_close.minute,
            tz=equity_close.tz,
        )
    else:
        # 24h markets (FX/metals/energy) and any unrecognized underlying_id:
        # the yfinance daily-bar boundary is not a real session close, so
        # treat the bar as usable only from the start of the *next*
        # calendar day, in the bar's own timezone (conservative).
        next_day = trade_date + timedelta(days=1)
        local_close = pd.Timestamp(
            year=next_day.year,
            month=next_day.month,
            day=next_day.day,
            tz=_zone_arg(index_tz),
        )
    utc_close = local_close.tz_convert(UTC).to_pydatetime()
    return utc_close


def _normalize_history(
    history: pd.DataFrame, *, underlying_id: str, retrieved_at: datetime
) -> list[UnderlyingBar]:
    """Convert a yfinance ``history()`` DataFrame into UnderlyingBar records.

    For each row, the trading date is taken in the yfinance index's *own*
    timezone (never UTC-shifted first -- see `_index_tz_and_trade_date`).
    ``ts`` is that trading date normalized to 00:00 UTC (a calendar-date
    marker, not a real instant). ``available_at``/``observation_time`` is a
    conservative session close (`_session_close_utc`): equity indices use a
    fixed local close time (CLAUDE.md rule 5: no look-ahead), 24h markets
    (FX/metals/energy) and unrecognized underlyings use the start of the
    next calendar day. Any bar whose ``available_at`` is still in the future
    relative to ``retrieved_at`` is excluded outright -- this is a generic
    per-row check, not just a "last row" special case, since a fetch can
    legitimately span multiple not-yet-closed sessions (e.g. combined with a
    24h-market bar just past its yfinance cutoff). Ambiguous bars are never
    optimistically included (CLAUDE.md rule 17).

    Rows with NaN OHLC or internally inconsistent OHLC
    (``high < max(open, close)`` or ``low > min(open, close)``) are skipped
    with a warning log rather than raising, since these are data-quality
    blemishes in individual rows, not a malformed-schema failure. Duplicate
    trading dates (observed in practice from yfinance) keep the *last*
    occurrence.
    """
    if history is None or history.empty:
        return []

    order: list[date] = []
    by_trade_date: dict[date, UnderlyingBar] = {}
    # Befund 5 (2026-09-13 measurement session): a per-row logger.warning for
    # every skipped bar produced hundreds of near-identical lines in a single
    # fetch (e.g. an underlying with a long illiquid/pre-listing history).
    # Individual rows are still logged, but at DEBUG (detail preserved for
    # someone actively debugging a specific date); one aggregated summary
    # line per skip reason is emitted at WARNING after the loop instead, with
    # a count and the first/last affected trading day.
    skip_counts: dict[str, int] = {}
    skip_first: dict[str, date] = {}
    skip_last: dict[str, date] = {}

    def _record_skip(reason: str, trade_date: date) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
        skip_first.setdefault(reason, trade_date)
        skip_last[reason] = trade_date

    for ts, row in history.iterrows():
        source_timestamp = _to_utc_datetime(ts)
        index_tz, trade_date = _index_tz_and_trade_date(ts)

        try:
            open_ = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(
                f"malformed yfinance row for {underlying_id!r} at {ts!r}: {exc}"
            ) from exc

        if any(pd.isna(v) for v in (open_, high, low, close)):
            _record_skip("nan_ohlc", trade_date)
            logger.debug(
                "fallback_prices_skip_nan_ohlc",
                underlying_id=underlying_id,
                trade_date=trade_date.isoformat(),
            )
            continue

        if high < max(open_, close) or low > min(open_, close):
            _record_skip("inconsistent_ohlc", trade_date)
            logger.debug(
                "fallback_prices_skip_inconsistent_ohlc",
                underlying_id=underlying_id,
                trade_date=trade_date.isoformat(),
                open=open_,
                high=high,
                low=low,
                close=close,
            )
            continue

        available_at = _session_close_utc(trade_date, index_tz, underlying_id)
        if available_at > retrieved_at:
            # Bar not yet closed (or not yet conservatively usable) as of
            # retrieved_at -- excluded outright, never optimistically
            # resolved (CLAUDE.md rule 17).
            continue

        volume_raw = row.get("Volume")
        volume = float(volume_raw) if volume_raw is not None and not pd.isna(volume_raw) else None

        bar_ts = datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=UTC)

        bar = UnderlyingBar(
            underlying_id=underlying_id,
            ts=bar_ts,
            interval="1d",
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            observation_time=available_at,
            available_at=available_at,
            retrieved_at=retrieved_at,
            source_timestamp=source_timestamp,
            source=_SOURCE_NAME,
            parser_version=_PARSER_VERSION,
            is_stale=False,
            quality_score=_COMPLETE_BAR_QUALITY,
        )

        if trade_date in by_trade_date:
            logger.warning(
                "fallback_prices_duplicate_trade_date",
                underlying_id=underlying_id,
                trade_date=trade_date.isoformat(),
            )
        else:
            order.append(trade_date)
        by_trade_date[trade_date] = bar  # duplicates: last occurrence wins

    for reason, count in skip_counts.items():
        logger.warning(
            "fallback_prices_skip_summary",
            underlying_id=underlying_id,
            reason=reason,
            count=count,
            first_trade_date=skip_first[reason].isoformat(),
            last_trade_date=skip_last[reason].isoformat(),
        )

    return [by_trade_date[d] for d in order]


class YFinancePriceAdapter:
    """Fetches daily OHLC bars for a canonical underlying_id via yfinance."""

    def __init__(self, *, lookback_days: int = 400) -> None:
        self._default_lookback_days = lookback_days

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=_SOURCE_NAME,
            kind="price",
            version=_PARSER_VERSION,
            homepage="https://finance.yahoo.com/",
        )

    def fetch(self, *, underlying_id: str, lookback_days: int | None = None) -> pd.DataFrame:
        ticker_symbol = get_underlying_meta(underlying_id).yfinance_ticker
        period = f"{lookback_days or self._default_lookback_days}d"
        try:
            ticker = yf.Ticker(ticker_symbol)
            history = ticker.history(period=period, interval="1d", auto_adjust=False)
        except Exception as exc:  # yfinance raises a variety of exception types
            raise AdapterError(f"yfinance fetch failed for {ticker_symbol!r}: {exc}") from exc
        if not isinstance(history, pd.DataFrame):
            raise AdapterError(
                f"yfinance returned unexpected type {type(history)!r} for {ticker_symbol!r}"
            )
        return history

    def normalize(
        self, raw: pd.DataFrame, *, underlying_id: str, **_kwargs: Any
    ) -> list[UnderlyingBar]:
        return _normalize_history(raw, underlying_id=underlying_id, retrieved_at=datetime.now(UTC))

    def fetch_daily_bars(
        self, underlying_id: str, *, lookback_days: int | None = None
    ) -> list[UnderlyingBar]:
        raw = self.fetch(underlying_id=underlying_id, lookback_days=lookback_days)
        return self.normalize(raw, underlying_id=underlying_id)

    def healthcheck(self) -> HealthCheckResult:
        checked_at = datetime.now(UTC)
        start = time.monotonic()
        try:
            bars = self.fetch_daily_bars(_HEALTHCHECK_PROBE_UNDERLYING, lookback_days=5)
        except AdapterError as exc:
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=str(exc),
            )
        latency_ms = (time.monotonic() - start) * 1000
        if not bars:
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.WARN,
                ok=False,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message=f"no bars returned for probe underlying {_HEALTHCHECK_PROBE_UNDERLYING}",
            )
        return HealthCheckResult(
            source=_SOURCE_NAME,
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=latency_ms,
            checked_at=checked_at,
            message=f"fetched {len(bars)} bar(s), latest close={bars[-1].close}",
        )
