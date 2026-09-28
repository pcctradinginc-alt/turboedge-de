"""U.S. EIA Open Data v2 adapter, for the External Data Factory (Wave 2).

DOCUMENTATION-BASED, MOSTLY NOT LIVE-VERIFIED: no `EIA_API_KEY` is available
in this environment. The route-based URL structure, parameter encoding and
response/error envelopes below come from EIA's own official v2 documentation
(https://www.eia.gov/opendata/documentation.php, fetched during development)
plus one live, unauthenticated (no key obtained or used) probe of
`https://api.eia.gov/v2/petroleum/stoc/wstk/data/` on 2026-09-28, which
independently confirmed the base URL, the route path shape, and the
undocumented-until-you-hit-it 403 error body byte-for-byte (see "THE
CREDENTIAL FAILURE MODE" below). No route's actual data shape (column names,
values, facet values) has been observed live -- see "ROUTES" below for
exactly which routes are documentation-only.

URL / ROUTE STRUCTURE
----------------------
    https://api.eia.gov/v2/<route>/data/
e.g. https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key=...

PARAMETERS (official v2 documentation)
---------------------------------------
    api_key               required; a QUERY PARAMETER, not a header (unlike
                           GIE's `x-key` header -- see `adapters/gie.py`).
    frequency=<freq>       periodicity, when a route supports more than one
                           (e.g. "weekly", "monthly", "annual").
    data[0]=<column>       which data column(s) to return; the bracketed
                           index form (`data[0]=`, `data[1]=`, ...) rather
                           than the equivalent repeated `data[]=` form, to
                           keep this adapter's single-column request
                           unambiguous.
    facets[<name>][]=<v>   filter on one facet; the doubly-bracketed form is
                           the one place this parameter encoding is easy to
                           get subtly wrong (a missing trailing `[]` silently
                           returns the wrong / unfiltered column per the
                           official docs' own warning).
    start=YYYY-MM-DD       lower bound (inclusive) on `period`.
    end=YYYY-MM-DD         upper bound (inclusive) on `period`.
    offset=<n>             paging: rows to skip.
    length=<n>             paging: rows to return, capped at 5000 for JSON
                           (`_MAX_PAGE_LENGTH` below).
    sort[0][column]=period,
    sort[0][direction]=asc  deterministic ordering, so repeated/paged fetches
                           are reproducible archives, not an upstream-decided
                           order that could vary between calls.

RESPONSE ENVELOPE
------------------
    {"response": {"total": "<n>", "dateFormat": "YYYY-MM or YYYY-MM-DD",
                   "frequency": "...", "data": [{"period": ..., "<facet>":
                   ..., "<column>": ..., "<column>-units": ...}, ...]},
     "request": {...}, "apiVersion": "..."}

The `dateFormat` field is EIA's own admission that `period` is
route-dependent: `"YYYY-MM or YYYY-MM-DD"` (confirmed from the official
documentation). `_parse_eia_period` below tries `YYYY-MM-DD`, then `YYYY-MM`,
then bare `YYYY`, by string length -- not a guess, a direct reading of that
documented ambiguity.

THE CREDENTIAL FAILURE MODE -- ordinary HTTP status, unlike GIE's trap
------------------------------------------------------------------------
Verified LIVE on 2026-09-28 (`curl https://api.eia.gov/v2/petroleum/stoc/wstk/data/`,
no key supplied): **HTTP 403** with body

    {"error": {"code": "API_KEY_MISSING",
               "message": "No api_key was supplied.  Please register for one
                           at https://www.eia.gov/opendata/register.php"}}

Unlike AGSI/ALSI (`adapters/gie.py`'s "THE TRAP"), this is an ordinary
non-2xx HTTP status. `HttpClient._request` already calls
`response.raise_for_status()` on every request, so a 403 becomes an
`AdapterHttpError` immediately -- no bespoke body-inspection is needed here,
and none is added: `_require_api_key` simply refuses to ever attempt the
request without a key (CLAUDE.md rule 4, "never fall back to an
unauthenticated request"), and any live rejection of a *present but invalid*
key would surface the same honest way, through the same existing mechanism.

CREDENTIAL HANDLING -- query parameter, like FRED (unlike GIE)
-------------------------------------------------------------------
The official docs are explicit: "must appear in URL, not headers". This
mirrors `adapters/fred.py` exactly, not `adapters/gie.py`: `EIA_API_KEY` is
read fresh on every `fetch()` call via `_require_api_key` (raises
`EiaCredentialError`, a typed `AdapterError`, if unset), baked into the real
request's query parameters, but never into `FetchedPayload.url` or
`.request_fingerprint` -- those are built only from a separate,
credential-free parameter dict (`public_params`), exactly as `fred.py`'s
module docstring describes. A test injects a real-looking secret via
`monkeypatch.setenv` and asserts it is absent from `.url` and
`.request_fingerprint`.

NATIVE IDENTIFIER FORMAT -- fixed by this workstream's contract
---------------------------------------------------------------
    "<route>|<facet=value,facet=value>|<data column>"
    e.g. "petroleum/stoc/wstk|series=WCESTUS1|value"
An empty facets segment is allowed (no filter), matching the same convention
`adapters/portwatch.py` uses for its own third segment (per this workstream's
contract).

ROUTES -- named in the task brief, NOT independently verified live here
-------------------------------------------------------------------------
    petroleum/stoc/wstk               (weekly petroleum stocks)
    natural-gas/stor/wkly             (weekly natural gas storage)
    electricity/rto/daily-region-data (daily regional electricity)
    petroleum/pri/spt                 (petroleum spot prices)
No `EIA_API_KEY` is available in this environment, so none of these routes'
actual facet names, column names, or data shape have been confirmed against
a live, authenticated response -- only the base URL and the generic v2
envelope/error shapes were confirmed live (see above). `SUGGESTED_ROUTES`
below records them as documentation-only, exactly as `adapters/fred.py`'s
`CURATED_SERIES_IDS` records its own un-verified series list, and whoever
wires up `configs/external_data.yaml` should confirm the first live pull of
each returns data before enabling it.

HANDLING
--------
Paging: `offset`/`length` against `response.total`, capped at
`_MAX_PAGE_LENGTH` (5000) per page; pages are concatenated into one
deterministic archived JSON envelope per fetch, the same approach
`adapters/gie.py` uses for its own multi-page fetches.
A `null` value for a row (a period with no report) is skipped without a
warning -- the documented, normal shape of this API, not schema drift.
Rate limiting: EIA publishes no `robots.txt` (per this workstream's
contract), so the default 1 request/second applies; `cli_external.py`
already wires `_MIN_INTERVAL_S["eia"] = 1.0`. No bespoke retry logic is
added here -- `HttpClient` already retries 429/5xx with backoff.

AVAILABILITY
------------
Availability is date-only (`period`); every `available_at` goes through
`resolve_available_at()` (CLAUDE.md rule 5). `source_release_time` is left
`None` (EIA's v2 API exposes no per-row release timestamp). `vintage_time`
is taken from the HTTP `Date` response header of the first page fetched, the
best honestly available evidence of when this archived copy was current.
`revision_index` is always `None`: this workstream's contract states EIA
exposes no vintages, and none of the named routes above are a revision-aware
endpoint (unlike `adapters/fred.py`'s ALFRED realtime windows).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from email.utils import parsedate_to_datetime
from typing import Any, Final
from urllib.parse import urlencode

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "eia"
_PARSER_VERSION = "1"
_SOURCE_VERSION = "eia_opendata_v2"
_API_KEY_ENV_VAR = "EIA_API_KEY"
_DEFAULT_BASE_URL = "https://api.eia.gov/v2"

#: Documented cap for JSON output (official v2 documentation).
_MAX_PAGE_LENGTH: Final = 5000
#: Defensive only, mirrors `adapters/gie.py`'s own safety cap.
_MAX_PAGES_SAFETY_CAP: Final = 2000

_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

#: Named in the task brief; NOT independently verified live (module
#: docstring "ROUTES").
SUGGESTED_ROUTES: Final[tuple[str, ...]] = (
    "petroleum/stoc/wstk",
    "natural-gas/stor/wkly",
    "electricity/rto/daily-region-data",
    "petroleum/pri/spt",
)

__all__ = [
    "SUGGESTED_ROUTES",
    "EiaAdapter",
    "EiaCredentialError",
    "build_native_identifier",
    "parse_eia_payload",
    "parse_native_identifier",
]


class EiaCredentialError(AdapterError):
    """`EIA_API_KEY` is not set. Never substituted with an unauthenticated
    request or a fabricated key -- every EIA v2 route requires one."""


def _require_api_key() -> str:
    api_key = os.environ.get(_API_KEY_ENV_VAR)
    if not api_key:
        raise EiaCredentialError(
            f"{_API_KEY_ENV_VAR} is not set. EIA's v2 API requires an API key for every "
            "request (https://www.eia.gov/opendata/register.php); refusing to make an "
            "unauthenticated request or substitute a fabricated key."
        )
    return api_key


def build_native_identifier(route: str, facets: Mapping[str, str], data_column: str) -> str:
    """The fixed format this workstream's contract specifies:
    `"<route>|<facet=value,facet=value>|<data column>"`."""
    if not route or not data_column:
        raise AdapterError(
            "eia: 'route' and 'data_column' are required to build a native_identifier"
        )
    if "|" in route or "|" in data_column:
        raise AdapterError("eia: native_identifier segments must not themselves contain '|'")
    facets_str = ",".join(f"{name}={value}" for name, value in facets.items())
    return f"{route}|{facets_str}|{data_column}"


def parse_native_identifier(raw: str) -> tuple[str, dict[str, str], str]:
    """Inverse of `build_native_identifier`. Raises `AdapterError` on malformed input."""
    parts = raw.split("|")
    if len(parts) != 3:
        raise AdapterError(
            f"eia: native_identifier {raw!r} does not match "
            "'<route>|<facet=value,facet=value>|<data column>'"
        )
    route, facets_str, data_column = parts
    if not route or not data_column:
        raise AdapterError(
            f"eia: native_identifier {raw!r} must have a non-empty route and data column"
        )
    facets: dict[str, str] = {}
    if facets_str:
        for pair in facets_str.split(","):
            name, sep, value = pair.partition("=")
            if not sep or not name:
                raise AdapterError(
                    f"eia: malformed facet segment {pair!r} in native_identifier {raw!r}"
                )
            facets[name] = value
    return route, facets, data_column


def _parse_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_number(raw: Any) -> float | None:
    if isinstance(raw, int | float):
        return float(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_eia_period(raw: str) -> date | None:
    """`period` is `"YYYY-MM-DD"`, `"YYYY-MM"` or bare `"YYYY"` depending on
    the route's frequency -- the official v2 documentation's own
    `dateFormat` field admits exactly this ambiguity (module docstring).
    Tried by exact string length, most to least specific; never guesses a
    day-of-month or month for a shorter label beyond the conventional "first
    of the period" anchor `adapters/eu_bcs.py` also uses for its own
    'YYYY-MM' labels."""
    if len(raw) == 10:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None
    if len(raw) == 7:
        try:
            return date.fromisoformat(f"{raw}-01")
        except ValueError:
            return None
    if len(raw) == 4:
        try:
            return date.fromisoformat(f"{raw}-01-01")
        except ValueError:
            return None
    return None


def parse_eia_payload(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
    vintage_time: datetime | None = None,
) -> ParseResult:
    """Pure parse function: bytes + spec -> observations. No network I/O.

    Exposed at module level so tests can exercise it directly against a
    fixture, mirroring every other adapter on this contract.
    """
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: payload is not valid UTF-8 JSON: {exc}",),
            missing_series=(spec.series_id,),
        )
    if not isinstance(raw, dict):
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: eia response is not a JSON object",),
            missing_series=(spec.series_id,),
        )

    response_obj = raw.get("response")
    if not isinstance(response_obj, dict):
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: eia response has no 'response' object "
                f"(top-level keys={sorted(raw.keys())!r}) -- upstream contract may have "
                "changed, or this is an error body (see module docstring)",
            ),
            missing_series=(spec.series_id,),
        )

    rows = response_obj.get("data")
    if not isinstance(rows, list):
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: eia response.data is not a list "
                f"(got {type(rows).__name__}) -- upstream contract may have changed",
            ),
            missing_series=(spec.series_id,),
        )

    _route, facets, data_column = parse_native_identifier(spec.native_identifier)

    warnings: list[str] = []
    observations: list[ExternalObservation] = []
    matched_any_row = False
    skipped_null = 0

    for row in rows:
        if not isinstance(row, dict):
            warnings.append(f"{spec.qualified_id}: non-object entry in response.data: {row!r}")
            continue

        # Defensive re-check of the server-side facet filter (mirrors
        # adapters/eu_bcs.py's own client-side re-match of indic/geo/s_adj):
        # a row for a facet combination other than the one requested is
        # schema drift, not silently mixed in.
        if any(str(row.get(name)) != value for name, value in facets.items()):
            continue
        matched_any_row = True

        period_raw = row.get("period")
        if not isinstance(period_raw, str):
            warnings.append(f"{spec.qualified_id}: row has no 'period': {row!r}")
            continue
        observation_day = _parse_eia_period(period_raw)
        if observation_day is None:
            warnings.append(f"{spec.qualified_id}: unparseable period {period_raw!r}")
            continue

        if data_column not in row:
            warnings.append(
                f"{spec.qualified_id}: row has no {data_column!r} column "
                f"(available: {sorted(row.keys())!r}) -- upstream contract may have changed"
            )
            continue
        raw_value = row[data_column]
        if raw_value is None:
            # Documented v2 convention: a null value for a period with no
            # report -- skip, no warning (this workstream's contract).
            skipped_null += 1
            continue

        value = _parse_number(raw_value)
        if value is None:
            warnings.append(
                f"{spec.qualified_id}: unparseable value {raw_value!r} for column "
                f"{data_column!r} on period {period_raw!r}"
            )
            continue

        observation_time = datetime.combine(observation_day, time(0, 0), tzinfo=UTC)
        available_at, precision = resolve_available_at(spec, observation_day)
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=value,
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=_SOURCE_VERSION,
                observation_time=observation_time,
                available_at=available_at,
                retrieved_at=retrieved_at,
                source=_SOURCE_ID,
                parser_version=_PARSER_VERSION,
                quality_score=1.0,
                is_stale=False,
                source_release_time=None,
                vintage_time=vintage_time,
                availability_precision=str(precision),
                revision_index=None,
            )
        )

    if not matched_any_row and rows:
        warnings.append(
            f"{spec.qualified_id}: response.data had {len(rows)} row(s) but none matched "
            f"facets {facets!r}"
        )
        return ParseResult(
            observations=observations, warnings=tuple(warnings), missing_series=(spec.series_id,)
        )

    return ParseResult(observations=observations, warnings=tuple(warnings))


class EiaAdapter:
    """Fetches and parses U.S. EIA Open Data v2 series. See module docstring."""

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
        api_key = _require_api_key()
        route, facets, data_column = parse_native_identifier(spec.native_identifier)
        url = f"{self._base_url}/{route}/data/"

        # Credential-free by construction: `api_key` is added only to
        # `request_params` below, never to `public_params` (module docstring
        # "CREDENTIAL HANDLING", mirrors adapters/fred.py exactly).
        public_params: dict[str, str] = {"frequency": spec.frequency, "data[0]": data_column}
        for name, value in facets.items():
            public_params[f"facets[{name}][]"] = value
        if since is not None:
            public_params["start"] = since.isoformat()
        public_params["sort[0][column]"] = "period"
        public_params["sort[0][direction]"] = "asc"
        public_params["length"] = str(_MAX_PAGE_LENGTH)

        retrieved_at = datetime.now(UTC)
        all_rows: list[Any] = []
        first_status = 0
        first_content_type = ""
        date_header: str | None = None
        first_date_format: Any = None
        offset = 0
        total = 0

        for page in range(_MAX_PAGES_SAFETY_CAP):
            request_params = {**public_params, "offset": str(offset), "api_key": api_key}
            response = self._http._request("GET", url, params=request_params)
            if page == 0:
                first_status = response.status_code
                first_content_type = response.headers.get("content-type", "")
                date_header = response.headers.get("date")
            try:
                envelope = response.json()
            except ValueError as exc:
                raise AdapterError(
                    f"eia: page at offset {offset} response is not valid JSON: {exc}"
                ) from exc

            resp_obj = envelope.get("response") if isinstance(envelope, dict) else None
            if not isinstance(resp_obj, dict):
                break
            if first_date_format is None:
                first_date_format = resp_obj.get("dateFormat")
            page_rows = resp_obj.get("data")
            if isinstance(page_rows, list):
                all_rows.extend(page_rows)
            else:
                page_rows = []
            try:
                total = int(resp_obj.get("total", len(all_rows)))
            except (TypeError, ValueError):
                total = len(all_rows)

            if not page_rows or len(page_rows) < _MAX_PAGE_LENGTH or len(all_rows) >= total:
                break
            offset += _MAX_PAGE_LENGTH

        combined = {"response": {"total": total, "dateFormat": first_date_format, "data": all_rows}}
        content = json.dumps(combined, sort_keys=True, default=str).encode("utf-8")

        public_url = f"{url}?{urlencode(sorted(public_params.items()))}"
        header_dict = {"date": date_header} if date_header else {}

        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=spec.series_id,
            url=public_url,
            content=content,
            http_status=first_status,
            content_type=first_content_type,
            retrieved_at=retrieved_at,
            request_fingerprint=f"GET {public_url}",
            headers=header_dict,
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        vintage_time = _parse_http_date(payload.headers.get("date"))
        return parse_eia_payload(
            payload.content, spec, retrieved_at=payload.retrieved_at, vintage_time=vintage_time
        )
