"""FRED / ALFRED adapter, for the External Data Factory (Wave 1).

Implements `turboedge.external.adapter.ExternalSeriesAdapter` against the
Federal Reserve Bank of St. Louis's public web service,
``https://api.stlouisfed.org/fred/series/observations``, documented at
https://fred.stlouisfed.org/docs/api/fred/series_observations.html.

DOCUMENTATION-BASED, NOT LIVE-VERIFIED: no `FRED_API_KEY` is available in
this environment, and every FRED endpoint requires one for every request
(including a plain, non-revised fetch). This module is built strictly from
the published API reference fetched during development, and every response
shape it parses (`observations` list, `date`/`value`/`realtime_start`/
`realtime_end` fields, the `.` missing-value marker, the JSON error body
`{"error_code": ..., "error_message": ...}`) comes from that documentation,
not from an observed live response. See the adapter report for exactly what
was fetched.

CREDENTIAL: `FRED_API_KEY` is read from the environment on every fetch, never
cached beyond one call, and never allowed to silently fall back to an
unauthenticated request -- `_require_api_key` raises `FredCredentialError`
(a typed `AdapterError`) naming the environment variable when it is unset.
The key is baked into the *actual* request's query parameters (FRED accepts
no other transport for it) but never into anything this adapter persists:
`FetchedPayload.url` and `.request_fingerprint` are built from a separate,
credential-free parameter dict, never from `response.request.url` (which
httpx has already filled in with the real, secret query string). This is
also enforced by `RawPayload._no_credentials_in_fingerprint` downstream, but
this adapter does not rely on that backstop -- it never constructs the
credentialed string in the first place for anything that gets archived.

VINTAGE SEMANTICS -- the reason this adapter matters. FRED is also known as
ALFRED (ArchivaL FRED) once `realtime_start`/`realtime_end` are used: every
observation FRED serves carries its own `realtime_start`/`realtime_end`
window, the period during which that specific *value* was the one FRED
would have returned if asked "what is this series as of today". Requesting
a wide realtime window (`realtime_start` at the series' earliest possible
date, `realtime_end` today) returns every historical revision of every
observation period, each tagged with the window during which it was current
-- this is `fetch()`, the adapter's primary, PIT-safe path. Requesting a
narrow window (`realtime_start == realtime_end == today`) returns exactly
one row per observation period: whichever vintage is in effect right now
(`fetch_current()`). Both request shapes return the *true* `realtime_start`
of each row (FRED does not truncate it to the query window), so:

  - `vintage_time` is always `realtime_start` (the day this specific vintage
    became FRED's answer for that period) -- known and honest in both modes.
  - `revision_index` (0 = first release, 1 = first revision, ...) can only be
    computed from the *wide* window: sorting each observation period's rows
    by `realtime_start` gives the true revision order only when *all*
    revisions were actually returned. A narrow-window (`fetch_current`)
    response gives exactly one row per period with no way to know where it
    sits in that period's revision history, so `revision_index` is left
    `None` there rather than guessed as `0` -- a single glimpse of "the
    current value" is not evidence that it was also the first one.

IMPORTANT DISTINCTION (do not conflate these): a FRED *release date* -- the
calendar date a statistical agency (BLS, BEA, ...) announced a figure to the
public -- is a different fact from `realtime_start`, the date FRED's own
vintage record shows that value becoming "current". FRED exposes true
release-date information only through a separate endpoint
(`/fred/release/dates`), which this adapter does not call. `realtime_start`
is therefore mapped only to `vintage_time` ("which published revision this
value is" -- storage/schemas.py's own definition of that field) and never to
`source_release_time`, which is left `None` here rather than populated with
a value that is not actually the publisher's release date.

AVAILABILITY: FRED gives no intraday timestamp for when a vintage became
retrievable, only a calendar date (`realtime_start`). `available_at` is
therefore always produced by `external.adapter.resolve_available_at()` using
that date and the series' declared `CONSERVATIVE_DATE` lag -- never invented
here, and never conflated with the observation period itself.

MISSING VALUES: FRED encodes a missing observation as the literal string
`"."` in the `value` field (documented convention, not a `null`). Skipped
without a warning -- this is a normal, expected shape of the API, not a
sign the schema moved (mirrors `adapters/ecb_data.py`'s treatment of a
`null` SDMX observation).

RATE LIMITING: FRED returns HTTP 429 when a client exceeds its rate limit.
No bespoke handling is added here -- `adapters/base.HttpClient` already
retries 429/5xx with exponential backoff + jitter through
`_is_retryable_error`, and every request in this module goes through it.

CURATED SERIES: `CURATED_SERIES_IDS` lists long-standing, widely-known FRED
series identifiers (industrial production, retail sales, payrolls, initial
claims, CPI/PCE inflation, key policy and Treasury rates, credit spreads,
financial conditions). These are **not verified against a live request** in
this environment -- there is no key to verify them with -- they are recorded
here as well-known identifiers for whoever wires up `configs/sources.yaml`
next, who should still confirm the first live pull of each returns data
before enabling it.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, time
from typing import Any, Final
from urllib.parse import urlencode

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "fred"
_PARSER_VERSION = "1"
_SOURCE_VERSION = "fred_series_observations_json"
_DEFAULT_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_API_KEY_ENV_VAR = "FRED_API_KEY"
_MISSING_VALUE_MARKER = "."

#: FRED's own documented defaults for an unrestricted request: `1776-07-04`
#: is the literal default `observation_start` FRED's docs give (the series
#: doesn't need to actually start then; it just means "no lower bound").
_EARLIEST_REALTIME_START: Final = date(1776, 7, 4)

#: Errors expected from a malformed/unexpected upstream payload; anything
#: else is a programming error and should propagate rather than be swallowed.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

#: Long-standing, well-known FRED series ids named in the task brief. NOT
#: verified live (no FRED_API_KEY in this environment) -- see module
#: docstring "CURATED SERIES".
CURATED_SERIES_IDS: Final[tuple[str, ...]] = (
    "INDPRO",
    "RSAFS",
    "PAYEMS",
    "ICSA",
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    "PCEPILFE",
    "DFF",
    "DGS2",
    "DGS10",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "NFCI",
)

__all__ = [
    "CURATED_SERIES_IDS",
    "FredAdapter",
    "FredCredentialError",
    "parse_series_observations",
]


class FredCredentialError(AdapterError):
    """`FRED_API_KEY` is not set. Never substituted with an unauthenticated
    request or a fabricated key -- every FRED endpoint requires a real one."""


def _require_api_key() -> str:
    api_key = os.environ.get(_API_KEY_ENV_VAR)
    if not api_key:
        raise FredCredentialError(
            f"{_API_KEY_ENV_VAR} is not set. FRED requires an API key for every request "
            "(https://fred.stlouisfed.org/docs/api/api_key.html); refusing to make an "
            "unauthenticated request or substitute a fabricated key."
        )
    return api_key


def _public_url(base_url: str, params: dict[str, str]) -> str:
    """Build the archivable request URL from parameters that never include
    the credential -- callers must pass a dict with `api_key` already
    excluded. This is the only thing ever written to `FetchedPayload.url` /
    `.request_fingerprint`; the real, credentialed request is sent
    separately (see `FredAdapter._fetch`)."""
    return f"{base_url}?{urlencode(params)}"


def parse_series_observations(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
    vintage_aware: bool,
) -> ParseResult:
    """Parse one `fred/series/observations` JSON payload into `ExternalObservation`s.

    Pure and network-free: everything needed comes from `content`, `spec`
    and `vintage_aware` (itself derived from the archived `FetchedPayload`'s
    `dataset`, never from a fresh network call -- see `FredAdapter.parse`).

    `vintage_aware=True` means the payload was fetched with a wide
    `realtime_start`/`realtime_end` window (the full ALFRED history) and
    revision order can be trusted; `vintage_aware=False` means it was
    fetched with `realtime_start == realtime_end` (current values only) and
    `revision_index` cannot be honestly assigned -- see module docstring
    "VINTAGE SEMANTICS".

    Any structural surprise -- not valid JSON, a non-object body, a missing
    `observations` key, a row missing `date`/`value`, an unparseable date or
    value -- is reported as a warning and the affected row (or the whole
    payload) is skipped, never guessed at. The documented `"."` missing-value
    marker is the one expected shape and is skipped silently.
    """
    warnings: list[str] = []

    try:
        raw: Any = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: payload is not valid UTF-8 JSON: {exc}",),
            missing_series=(spec.series_id,),
        )

    if not isinstance(raw, dict):
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: FRED response is not a JSON object",),
            missing_series=(spec.series_id,),
        )

    if "error_code" in raw or "error_message" in raw:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: FRED returned an error body "
                f"(error_code={raw.get('error_code')!r}, "
                f"error_message={raw.get('error_message')!r})",
            ),
            missing_series=(spec.series_id,),
        )

    rows = raw.get("observations")
    if not isinstance(rows, list):
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: FRED response missing an 'observations' list "
                f"(top-level keys={sorted(raw.keys())!r}) -- upstream contract may have changed",
            ),
            missing_series=(spec.series_id,),
        )

    # (observation date) -> [(realtime_start, value), ...]
    by_date: dict[date, list[tuple[date, float]]] = {}
    for row in rows:
        if not isinstance(row, dict) or "date" not in row or "value" not in row:
            warnings.append(
                f"{spec.qualified_id}: observation row missing required field(s): {row!r}"
            )
            continue

        raw_value = row["value"]
        if not isinstance(raw_value, str):
            warnings.append(
                f"{spec.qualified_id}: observation 'value' is not a string: {raw_value!r}"
            )
            continue
        if raw_value.strip() == _MISSING_VALUE_MARKER:
            # Documented FRED convention for "no observation this period" --
            # normal, not a warning.
            continue

        try:
            value = float(raw_value)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(
                f"{spec.qualified_id}: unparseable observation value {raw_value!r} "
                f"on {row.get('date')!r}: {exc}"
            )
            continue

        try:
            observation_day = date.fromisoformat(row["date"])
            raw_realtime_start = row.get("realtime_start", row["date"])
            if not isinstance(raw_realtime_start, str):
                raise ValueError(f"realtime_start is not a string: {raw_realtime_start!r}")
            realtime_start = date.fromisoformat(raw_realtime_start)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(f"{spec.qualified_id}: unparseable date in row {row!r}: {exc}")
            continue

        by_date.setdefault(observation_day, []).append((realtime_start, value))

    observations: list[ExternalObservation] = []
    for observation_day, vintages in by_date.items():
        vintages_sorted = sorted(vintages, key=lambda v: v[0])
        if not vintage_aware and len(vintages_sorted) > 1:
            warnings.append(
                f"{spec.qualified_id}: 'current values' fetch (realtime_start == "
                f"realtime_end) returned {len(vintages_sorted)} rows for {observation_day} "
                "-- expected exactly one; possible upstream schema drift"
            )

        for idx, (realtime_start, value) in enumerate(vintages_sorted):
            available_at, precision = resolve_available_at(spec, realtime_start)
            observations.append(
                ExternalObservation(
                    series_id=spec.series_id,
                    value=value,
                    unit=spec.unit,
                    frequency=spec.frequency,
                    source_version=_SOURCE_VERSION,
                    observation_time=datetime.combine(observation_day, time(0, 0), tzinfo=UTC),
                    available_at=available_at,
                    retrieved_at=retrieved_at,
                    source=_SOURCE_ID,
                    parser_version=_PARSER_VERSION,
                    quality_score=1.0,
                    is_stale=False,
                    # See module docstring "IMPORTANT DISTINCTION": realtime_start is
                    # when this vintage became current in FRED's own record, not the
                    # publisher's official release date (a separate FRED endpoint this
                    # adapter does not call) -- never written here.
                    source_release_time=None,
                    vintage_time=datetime.combine(realtime_start, time(0, 0), tzinfo=UTC),
                    availability_precision=str(precision),
                    revision_index=idx if vintage_aware else None,
                )
            )

    return ParseResult(observations=observations, warnings=tuple(warnings), missing_series=())


class FredAdapter:
    """Fetches FRED/ALFRED series observations: current values or full vintage history.

    One instance handles every FRED series a `SeriesSpec` can name --
    `spec.native_identifier` is the FRED series id sent to the endpoint
    (e.g. `"INDPRO"`); `spec.series_id` is TurboEdge's own id, used only for
    the emitted `ExternalObservation`s.
    """

    def __init__(self, http: HttpClient, *, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        """The protocol entry point and the PIT-safe path (spec requirement):
        the full ALFRED vintage history, `realtime_start` at FRED's own
        "no lower bound" default through today. `since`, when given, narrows
        `observation_start` only -- it can only return *more* periods than
        `since` onward would need, never fewer, satisfying the protocol's
        "never return less than since onwards"."""
        today = datetime.now(UTC).date()
        return self._fetch(
            spec,
            realtime_start=_EARLIEST_REALTIME_START,
            realtime_end=today,
            since=since,
            vintage_aware=True,
        )

    def fetch_current(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        """Only the vintage currently in effect for each period -- no
        revision history. Not the point-in-time-safe path; use `fetch()` or
        `fetch_vintages()` for research that must reconstruct history
        honestly. See module docstring "VINTAGE SEMANTICS"."""
        today = datetime.now(UTC).date()
        return self._fetch(
            spec, realtime_start=today, realtime_end=today, since=since, vintage_aware=False
        )

    def fetch_vintages(
        self,
        spec: SeriesSpec,
        *,
        realtime_start: date,
        realtime_end: date,
        since: date | None = None,
    ) -> FetchedPayload:
        """Explicit ALFRED vintage window, for a caller that wants a bounded
        slice of revision history rather than everything `fetch()` returns."""
        return self._fetch(
            spec,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            since=since,
            vintage_aware=True,
        )

    def _fetch(
        self,
        spec: SeriesSpec,
        *,
        realtime_start: date,
        realtime_end: date,
        since: date | None,
        vintage_aware: bool,
    ) -> FetchedPayload:
        api_key = _require_api_key()
        public_params: dict[str, str] = {
            "series_id": spec.native_identifier,
            "realtime_start": realtime_start.isoformat(),
            "realtime_end": realtime_end.isoformat(),
            "file_type": "json",
        }
        if since is not None:
            public_params["observation_start"] = since.isoformat()
        request_params = {**public_params, "api_key": api_key}

        retrieved_at = datetime.now(UTC)
        # `_request` (shared by every adapter on this contract, see
        # `adapters/ecb_data.py`) is the one place HttpClient applies retry
        # (incl. 429), per-host rate limiting and the honest User-Agent, and
        # hands back the real `httpx.Response` so `http_status`/`content_type`
        # are recorded rather than invented. Its `.request.url` is NOT used
        # here (unlike ecb_data.py) because httpx has filled that in with the
        # real, credentialed query string -- `url`/`request_fingerprint` are
        # built only from `public_params` below.
        response = self._http._request("GET", self._base_url, params=request_params)

        mode = "vintages" if vintage_aware else "current"
        public_url = _public_url(self._base_url, public_params)
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=f"{spec.series_id}:{mode}",
            url=public_url,
            content=response.content,
            http_status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            retrieved_at=retrieved_at,
            request_fingerprint=f"GET {public_url}",
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        vintage_aware = not payload.dataset.endswith(":current")
        return parse_series_observations(
            payload.content, spec, retrieved_at=payload.retrieved_at, vintage_aware=vintage_aware
        )
