"""Cboe volatility-index adapter (Workstream W12-A).

Six official daily series, all from Cboe's own published CSV endpoint
(`cdn-api.cboe.com/api/global/us_indices/daily_prices/<INDEX>_History.csv`):

    VIX     S&P 500 30-day implied volatility          OHLC, from 1990-01-02
    VIX9D   S&P 500 9-day implied volatility           OHLC, from 2011-01-04
    VIX3M   S&P 500 3-month implied volatility         OHLC, from 2009-09-18
    VVIX    volatility of VIX ("vol of vol")           close, from 2006-03-06
    OVX     crude oil ETF implied volatility           close, from 2009-09-18
    GVZ     gold ETF implied volatility                close, from 2009-09-18

Chosen over the yfinance tickers the W9 challengers use (`^VIX`/`^VIX9D`/
`^VIX3M`) for two measured reasons, not on principle: it is a published
contract rather than an inferred one (docs/data_sources.md records yfinance
as "unofficial/inferred, no published API contract"), and it is *slower*,
which here is a feature -- on 2026-09-25 17:00 UTC the Cboe file carried
data only through 09-24 while yfinance already served 09-25. The official
file therefore reflects what was genuinely published at a given moment,
which is the quantity `available_at` is supposed to encode.

`robots.txt` check (2026-09-25): `cboe.com/robots.txt` disallows only
`/book/` and `*market_statistics/volume_reports/` with no crawl delay and
no rule touching `/api/`; `cdn-api.cboe.com/robots.txt` answers 403, which
RFC 9309 says a fetcher may treat as "no restrictions". The endpoint itself
answers 200 to an honest user agent. No access control is bypassed anywhere
in this module.

AVAILABILITY MODEL -- the load-bearing decision here. A trading day's close
is not published while that day is still trading, so `available_at` is set
to the *following* calendar day at 00:00 UTC. That is deliberately
conservative: the real publication happens on the evening of the trading
day itself (VIX is computed until 16:15 ET = 20:15 UTC), so this discards a
few usable hours. It cannot leak, and it comfortably covers the pipeline's
07:40 UTC morning scan. Tightening it requires *measuring* publication
latency over several days, not assuming it -- see
`features/availability.py`.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime, time, timedelta
from typing import Any

import structlog

from turboedge.adapters.base import (
    AdapterError,
    AdapterMetadata,
    HealthCheckResult,
    HttpClient,
)
from turboedge.storage.schemas import ExternalObservation, HealthStatus

logger = structlog.get_logger(__name__)

_SOURCE_NAME = "cboe"
_PARSER_VERSION = "1"
_SOURCE_VERSION = "cboe_daily_prices_csv"
_BASE_URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices"
_UNIT = "index_points"
_FREQUENCY = "daily"

#: Index -> which columns that index's CSV actually carries. Cboe publishes
#: OHLC for the three S&P 500 volatility term points and close-only for the
#: other three; both shapes are stored as one row per (series_id, date) with
#: `series_id` naming the field, so a close-only family never carries three
#: null columns just to match the OHLC ones.
_SERIES_COLUMNS: dict[str, tuple[str, ...]] = {
    "VIX": ("OPEN", "HIGH", "LOW", "CLOSE"),
    "VIX9D": ("OPEN", "HIGH", "LOW", "CLOSE"),
    "VIX3M": ("OPEN", "HIGH", "LOW", "CLOSE"),
    "VVIX": ("VVIX",),
    "OVX": ("OVX",),
    "GVZ": ("GVZ",),
}

DEFAULT_INDICES: tuple[str, ...] = ("VIX", "VIX9D", "VIX3M", "VVIX", "OVX", "GVZ")

_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)


def series_id_for(index: str, column: str) -> str:
    """Canonical series id, e.g. ``VIX.CLOSE`` or ``VVIX.CLOSE``.

    Close-only indices name their single column after themselves in the CSV
    header (``DATE,VVIX``); that is normalized to ``.CLOSE`` so every family
    exposes a uniform ``<INDEX>.CLOSE`` regardless of the upstream header,
    and a feature builder never has to special-case which shape it got.
    """
    normalized = "CLOSE" if column.upper() == index.upper() else column.upper()
    return f"{index.upper()}.{normalized}"


def _available_at(observation_date: datetime) -> datetime:
    """Conservative publication time: the following calendar day, 00:00 UTC.

    See the module docstring -- this discards a few genuinely usable evening
    hours in exchange for being unable to leak.
    """
    return datetime.combine((observation_date + timedelta(days=1)).date(), time(0, 0), tzinfo=UTC)


def parse_history_csv(
    text: str, index: str, *, retrieved_at: datetime
) -> list[ExternalObservation]:
    """Parse one ``<INDEX>_History.csv`` payload into ExternalObservations.

    Cboe's format is ``DATE,OPEN,HIGH,LOW,CLOSE`` (OHLC indices) or
    ``DATE,<INDEX>`` (close-only), with US-style ``MM/DD/YYYY`` dates.

    Rows whose value is empty or unparseable are skipped and counted, never
    imputed (CLAUDE.md rule 29) -- a missing volatility print is missing
    information, and filling it forward would silently manufacture a feature
    value on a day the index did not publish one.

    Raises:
        AdapterError: if the header is absent or carries none of the columns
            this index is expected to provide -- that is a contract change,
            not a bad row, and must fail loudly.
    """
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    if "DATE" not in fieldnames:
        raise AdapterError(f"{index}: CSV has no DATE column (header={fieldnames!r})")

    expected = _SERIES_COLUMNS.get(index.upper(), ("CLOSE",))
    present = [c for c in expected if c in fieldnames]
    if not present:
        raise AdapterError(
            f"{index}: none of the expected columns {expected!r} present "
            f"(header={fieldnames!r}) -- upstream contract changed"
        )

    observations: list[ExternalObservation] = []
    skipped = 0
    for row in reader:
        raw_date = (row.get("DATE") or "").strip()
        if not raw_date:
            skipped += 1
            continue
        try:
            observation_time = datetime.strptime(raw_date, "%m/%d/%Y").replace(tzinfo=UTC)
        except _PARSE_ERROR_TYPES:
            skipped += 1
            continue
        available_at = _available_at(observation_time)
        for column in present:
            raw_value = (row.get(column) or "").strip()
            if not raw_value:
                skipped += 1
                continue
            try:
                value = float(raw_value)
            except _PARSE_ERROR_TYPES:
                skipped += 1
                continue
            observations.append(
                ExternalObservation(
                    series_id=series_id_for(index, column),
                    value=value,
                    unit=_UNIT,
                    frequency=_FREQUENCY,
                    source_version=_SOURCE_VERSION,
                    observation_time=observation_time,
                    available_at=available_at,
                    retrieved_at=retrieved_at,
                    source=_SOURCE_NAME,
                    parser_version=_PARSER_VERSION,
                    quality_score=1.0,
                )
            )
    if skipped:
        logger.info("cboe_rows_skipped", index=index, skipped=skipped)
    return observations


class CboeVolatilityAdapter:
    """Fetches Cboe's official daily volatility-index history."""

    def __init__(
        self,
        http_client: HttpClient,
        *,
        indices: tuple[str, ...] = DEFAULT_INDICES,
        base_url: str = _BASE_URL,
    ) -> None:
        self._http = http_client
        self._indices = indices
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return _SOURCE_NAME

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=_SOURCE_NAME,
            kind="external_series",
            version=_PARSER_VERSION,
            homepage="https://www.cboe.com/tradable-products/vix/vix-historical-data",
        )

    def _url(self, index: str) -> str:
        return f"{self._base_url}/{index.upper()}_History.csv"

    def fetch(self, index: str) -> str:
        return self._http.get_text(self._url(index))

    def normalize(self, raw: str, index: str, **_kwargs: Any) -> list[ExternalObservation]:
        return parse_history_csv(raw, index, retrieved_at=datetime.now(UTC))

    def fetch_observations(self) -> list[ExternalObservation]:
        """Every configured index, concatenated.

        One index failing does not discard the others: a partial external
        layer is still usable (the feature builder simply has fewer series),
        whereas raising would make the whole family unavailable because one
        endpoint was briefly down. Failures are logged and surfaced through
        `healthcheck()`.
        """
        retrieved_at = datetime.now(UTC)
        out: list[ExternalObservation] = []
        for index in self._indices:
            try:
                text = self.fetch(index)
                parsed = parse_history_csv(text, index, retrieved_at=retrieved_at)
            except Exception as exc:
                logger.warning("cboe_index_fetch_failed", index=index, error=str(exc))
                continue
            logger.info("cboe_index_fetched", index=index, observations=len(parsed))
            out.extend(parsed)
        return out

    def healthcheck(self) -> HealthCheckResult:
        started = datetime.now(UTC)
        try:
            text = self.fetch(self._indices[0])
            parsed = parse_history_csv(text, self._indices[0], retrieved_at=started)
        except Exception as exc:
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=(datetime.now(UTC) - started).total_seconds() * 1000.0,
                checked_at=started,
                message=f"{self._indices[0]} fetch/parse failed: {exc}",
            )
        latency_ms = (datetime.now(UTC) - started).total_seconds() * 1000.0
        if not parsed:
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=latency_ms,
                checked_at=started,
                message=f"{self._indices[0]} returned no usable observations",
            )
        latest = max(o.observation_time for o in parsed)
        age_days = (started - latest).days
        # A daily series is expected to lag by a day (see the availability
        # model); a week means publication has actually stopped.
        status = HealthStatus.PASS if age_days <= 7 else HealthStatus.WARN
        return HealthCheckResult(
            source=_SOURCE_NAME,
            status=status,
            ok=status is HealthStatus.PASS,
            latency_ms=latency_ms,
            checked_at=started,
            message=(
                f"{len(parsed)} observation(s) for {self._indices[0]}, "
                f"latest {latest.date().isoformat()} ({age_days}d old)"
            ),
        )


__all__ = [
    "DEFAULT_INDICES",
    "CboeVolatilityAdapter",
    "parse_history_csv",
    "series_id_for",
]
