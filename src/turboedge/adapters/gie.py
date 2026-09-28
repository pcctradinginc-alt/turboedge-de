"""Gas Infrastructure Europe: AGSI (EU gas storage) and ALSI (EU LNG), for the
External Data Factory (Wave 2).

Two datasets, one API family (same REST shape, same paging, same credential
mechanism), one shared module with two small classes: `AgsiAdapter`
(`source_id="agsi"`, https://agsi.gie.eu/api) and `AlsiAdapter`
(`source_id="alsi"`, https://alsi.gie.eu/api). One `GIE_API_KEY` covers both
(GIE issues a single key that can be scoped to AGSI, ALSI, or both).

DOCUMENTATION SOURCE -- fetched live during development, 2026-09-28
--------------------------------------------------------------------
The authoritative source is GIE's own "User Manual: API access to AGSI /
ALSI", v007 (4 October 2022), fetched from
https://www.gie.eu/transparency-platform/GIE_API_documentation_v007.pdf --
found via GIE's own site and confirmed to be the newest version published (no
v008+ exists as of 2026-09-28). Cross-checked against the live, unauthenticated
pages https://agsi.gie.eu/data-definition and https://alsi.gie.eu/data-definition
(2026-09-28), which add narrative/footnote detail but name no JSON field not
already in the v007 manual's own field tables.

CONFIRMED FIELD NAMES (v007 manual §2.3/§2.4, "Data field library")
---------------------------------------------------------------------
AGSI `data[]` row: name, code, {consumption} (country-level only),
{consumptionFull} (country-level only), url, gasDayStart, gasInStorage,
injection, netWithdrawal, withdrawal, workingGasVolume, injectionCapacity,
withdrawalCapacity, status ("E"/"C"/"N"), trend, full, info, {children}.
This confirms every field the task brief named (gasDayStart, gasInStorage,
full, trend, injection, withdrawal, workingGasVolume) plus several more.

ALSI `data[]` row: name, code, url, gasDayStart, inventory, sendOut, dtmi,
dtrs, Info, {children}.

CORRECTION to the task brief: the brief expected ALSI fields "lngInventory"
and "full". Neither appears anywhere in the v007 manual's ALSI table, nor on
the live https://alsi.gie.eu/data-definition page (which documents exactly
"LNG Inventory" / "Send-Out" / "DTMI" / "DTRS", no percentage-full field at
all). The real field name for the inventory column is **`inventory`**, not
`lngInventory`. Per CLAUDE.md rule 1 ("never invent a series identifier/field
name"), this module uses `inventory`, not the brief's guess, and does not
invent an ALSI "full" field. (The doc tables also show `info` lowercase for
AGSI but `Info` capitalised for ALSI -- an apparent documentation
inconsistency, harmless here since this module never reads that field.)

Both AGSI and ALSI paging fields (v007 manual §7.2): `last_page` (total
pages), `total` (rows on this page), `data` (array of rows). `page` starts at
1; `size` defaults to 30, capped at 300 (`_MAX_PAGE_SIZE` below).

NUMBER FORMAT -- an unresolved ambiguity in the official docs
-----------------------------------------------------------------
The v007 manual's own worked examples disagree on decimal separator: the
cURL example (§6.4) shows `"gasInStorage":"326.9226"` (period), while the
REST-parameter example two pages later (§7.1) shows the *same field*, for
the *same country* (Germany), as `"gasInStorage": "63,9469"` (comma). No live
API key is available in this environment to observe which is actually
served. `_parse_number` below accepts both (period-decimal first, then a
comma-as-decimal fallback), mirroring `adapters/destatis_truck.py`'s own
defensive comma-decimal handling -- this is a documented ambiguity, not a
guess, and is exercised by a dedicated test.

THE TRAP -- verified LIVE by this workstream, 2026-09-28 (no key used)
---------------------------------------------------------------------
Both `https://agsi.gie.eu/api` and `https://alsi.gie.eu/api`, queried with NO
API key at all (a request that requires no credential and obtains none),
answered **HTTP 200** with:

    AGSI: {"last_page":0,"total":0,"dataset":"storage ERROR",
           "error":"access denied","message":"Invalid or missing API key",
           "data":[]}
    ALSI: {"last_page":0,"total":0,"dataset":"lng ERROR",
           "error":"access denied","message":"Invalid or missing API key",
           "data":[]}

confirming exactly the shape this workstream's contract described. An
adapter that trusts the HTTP status code would read this as "zero rows
published today" -- silent, plausible, and exactly the failure mode that
would make the readiness engine report a healthy empty series.
`parse_gie_payload` below detects an `"error"` key or a `dataset` value
ending in `" ERROR"` and raises `GieApiError` -- never a warning, never an
empty `ParseResult` -- covered by
`test_gie.py::test_trap_http_200_with_error_body_raises`, which replays this
exact byte-for-byte observed body as its fixture
(`tests/fixtures/external/gie/agsi_error_no_key.json` /
`alsi_error_no_key.json`).

CREDENTIAL HANDLING -- header, not query parameter
-----------------------------------------------------
Confirmed by the v007 manual's own cURL example (§6.4):
`curl https://agsi.gie.eu/api?type=eu --header "x-key: YOUR_API_KEY"`. The
key travels in the `x-key` request HEADER. `GIE_API_KEY` is read fresh on
every `fetch()` call, never cached, and never silently substituted with an
unauthenticated request -- `_require_api_key` raises `GieCredentialError` (a
typed `AdapterError`) naming the environment variable. Because the key is a
header value and never a query parameter, `FetchedPayload.url` and
`.request_fingerprint` (built only from `country`/`facility`/`from`/`size`/
`page`) are credential-free by construction; `.headers` never carries
anything but the response's own `Date` header (see below). A test still
injects a real-looking secret via `monkeypatch.setenv` and asserts it is
absent from `.url`, `.request_fingerprint` and `.headers`, per this
workstream's contract.

PAGINATION AND ARCHIVAL
-----------------------
`fetch()` requests `size=300` (the documented cap) and follows `last_page`
until every page is retrieved (or a defensive `_MAX_PAGES_SAFETY_CAP` is hit,
which only a pagination bug upstream could reach for any real dataset here).
All pages' `data` rows are concatenated into one combined, deterministic JSON
envelope, which is what gets archived as `FetchedPayload.content` -- one
payload per fetch regardless of how many HTTP requests it took, per this
workstream's contract. The trap response (`last_page: 0`) naturally short
-circuits this to exactly one request.

RATE LIMITING
-------------
Neither `agsi.gie.eu` nor `alsi.gie.eu` serves a `robots.txt`
(confirmed live 2026-09-28: both return non-200/empty). Per this
workstream's contract, the default in that case is 1 request/second, which
`cli_external.py` already wires via `_MIN_INTERVAL_S["agsi"] =
_MIN_INTERVAL_S["alsi"] = 1.0`. This module does not construct its own
`HttpClient` (the mandatory constructor takes one), so it relies on the
caller (the CLI, or a test) to set `min_interval_s=1.0` -- documented here so
a caller constructing `HttpClient` directly for this adapter knows the rate
to use. GIE's own documented limit (§7.3) is far looser (60 calls/minute
before a temporary queue), so 1 req/s is comfortably inside it.

NATIVE IDENTIFIER FORMAT -- fixed by this workstream's contract
---------------------------------------------------------------
    "<country>|<facility-or-empty>|<field>"
    e.g. "DE||gasInStorage"          (country-level aggregate)
         "DE|21W000000000078N|full"  (individual facility, EIC from the
                                       v007 manual's own worked example)

This identifier deliberately has no `company` segment, even though GIE's own
docs recommend sending `country`+`company`+`facility` together to fully
disambiguate a facility EIC that happens to collide across companies within
one country (v007 manual §7.1, §10 v003 changelog note). Since the identifier
format is fixed by this workstream's contract (not redesigned here), `fetch()`
sends only `country` (+ `facility`, when given) -- a facility whose EIC
collides across companies within the same country cannot be disambiguated by
this adapter. This is a known, documented limitation, not silently patched
over.

AVAILABILITY
------------
Availability is date-only (`gasDayStart`); every `available_at` goes through
`resolve_available_at()` (CLAUDE.md rule 5) -- this module never invents an
intraday publication time. For context when choosing a `SeriesSpec`'s
`conservative_release_lag_hours`: GIE's own "Gas Day" runs 04:00/05:00 UTC to
04:00/05:00 UTC the following day (not midnight UTC), and publication
happens at 19:30 CET/CEST (17:30-18:30 UTC) on the day *after* the gas day
ends, with a second pass at 23:00 CET/CEST -- from a naive midnight-UTC
anchor on the gas day's own date, a lag of at least ~48 hours is needed to
never treat a value as usable before GIE's own publication run. This module
does not choose that lag; it is the `SeriesSpec` author's responsibility.

`source_release_time` is left `None` (GIE publishes no per-row release
timestamp, only the twice-daily publication schedule above). `vintage_time`
is taken from the HTTP `Date` response header of the first page fetched
(this workstream's contract: "vintage_time from the HTTP Date header where
nothing better exists") -- not a GIE-native concept, but the best honestly
available evidence of when *this archived copy* was current.
`revision_index` is always `None`: AGSI/ALSI does not expose a revision
history endpoint (retroactive corrections happen silently, per the v007
manual's own "Retroactive corrections" note), so this source's
`backfill_class` should be `FORWARD_ONLY`, matching `adapters/eu_bcs.py`'s
treatment of the same kind of silently-revised source.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, time
from email.utils import parsedate_to_datetime
from typing import Any, Final
from urllib.parse import urlencode

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_PARSER_VERSION = "1"
_API_KEY_ENV_VAR = "GIE_API_KEY"

_AGSI_SOURCE_ID = "agsi"
_ALSI_SOURCE_ID = "alsi"
_AGSI_DEFAULT_BASE_URL = "https://agsi.gie.eu/api"
_ALSI_DEFAULT_BASE_URL = "https://alsi.gie.eu/api"

#: Documented cap (v007 manual §7.2): "the 'size' is capped at 300".
_MAX_PAGE_SIZE: Final = 300
#: Defensive only -- no real AGSI/ALSI dataset needs anywhere near this many
#: pages (300/page * 2000 = 600,000 gas-days = ~1,600 years of daily data);
#: this exists purely to stop a pagination bug upstream from looping forever.
_MAX_PAGES_SAFETY_CAP: Final = 2000

#: Errors expected from a malformed/unexpected upstream payload; anything
#: else is a programming error and should propagate rather than be swallowed.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)


class GieCredentialError(AdapterError):
    """`GIE_API_KEY` is not set. Never substituted with an unauthenticated
    request or a fabricated key -- both AGSI and ALSI require one."""


class GieApiError(AdapterError):
    """AGSI/ALSI answered HTTP 200 but the body reports an error (THE TRAP,
    see module docstring) -- typically a missing/invalid API key. This is
    never treated as "zero rows today"."""


def _require_api_key() -> str:
    api_key = os.environ.get(_API_KEY_ENV_VAR)
    if not api_key:
        raise GieCredentialError(
            f"{_API_KEY_ENV_VAR} is not set. AGSI and ALSI both require an API key, sent in "
            "the 'x-key' request header (https://agsi.gie.eu/account); refusing to make an "
            "unauthenticated request or substitute a fabricated key."
        )
    return api_key


def build_native_identifier(country: str, facility: str, field: str) -> str:
    """The fixed format this workstream's contract specifies:
    `"<country>|<facility-or-empty>|<field>"`. Raises on an obviously
    malformed input (empty required segment, or a segment containing the
    separator itself)."""
    if not country or not field:
        raise AdapterError("gie: 'country' and 'field' are required to build a native_identifier")
    if "|" in country or "|" in facility or "|" in field:
        raise AdapterError("gie: native_identifier segments must not themselves contain '|'")
    return f"{country}|{facility}|{field}"


def parse_native_identifier(raw: str) -> tuple[str, str, str]:
    """Inverse of `build_native_identifier`. Raises `AdapterError` on malformed input."""
    parts = raw.split("|")
    if len(parts) != 3:
        raise AdapterError(
            f"gie: native_identifier {raw!r} does not match '<country>|<facility-or-empty>|<field>'"
        )
    country, facility, field = parts
    if not country or not field:
        raise AdapterError(
            f"gie: native_identifier {raw!r} must have a non-empty country and field"
        )
    return country, facility, field


def _parse_http_date(value: str | None) -> datetime | None:
    """Best-effort parse of an HTTP `Date` response header into a tz-aware
    UTC datetime. Returns `None` on anything unparseable rather than
    raising -- this is a best-available fallback (`vintage_time`), not a
    load-bearing field."""
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
    """Parse one AGSI/ALSI data value. See module docstring "NUMBER FORMAT"
    for why both a period and a comma decimal separator are accepted."""
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
        pass
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def parse_gie_payload(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
    source_id: str,
    vintage_time: datetime | None = None,
) -> ParseResult:
    """Pure parse function: bytes + spec -> observations. No network I/O.

    Shared by `AgsiAdapter.parse` and `AlsiAdapter.parse` (`source_id`
    distinguishes them only for error messages and the emitted
    `ExternalObservation.source`). Exposed at module level so tests can
    exercise it directly against a fixture.
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
            warnings=(f"{spec.qualified_id}: {source_id} response is not a JSON object",),
            missing_series=(spec.series_id,),
        )

    # THE TRAP (module docstring): HTTP 200 with an error body. Never
    # silently treated as "zero rows published today".
    dataset_label = raw.get("dataset")
    if "error" in raw or (isinstance(dataset_label, str) and dataset_label.endswith(" ERROR")):
        raise GieApiError(
            f"{source_id} reported an error for {spec.qualified_id}: "
            f"error={raw.get('error')!r} message={raw.get('message')!r} "
            f"dataset={dataset_label!r} -- this is NOT an empty result; the request was "
            f"rejected (commonly a missing/invalid {_API_KEY_ENV_VAR})"
        )

    country, facility, field = parse_native_identifier(spec.native_identifier)
    expected_code = facility if facility else country

    rows_raw = raw.get("data", [])
    if isinstance(rows_raw, dict):
        # A single-day query (`date=` param) returns `data` as one object
        # rather than an array (v007 manual p.15 vs p.16 examples). fetch()
        # below never sends `date=` (it always requests a range), but parse()
        # stays defensive in case an archived payload from a different query
        # shape is ever replayed.
        rows: list[Any] = [rows_raw]
    elif isinstance(rows_raw, list):
        rows = rows_raw
    else:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: 'data' is neither an object nor an array "
                f"(got {type(rows_raw).__name__}) -- upstream contract may have changed",
            ),
            missing_series=(spec.series_id,),
        )

    warnings: list[str] = []
    observations: list[ExternalObservation] = []
    matched_any_code = False
    skipped_no_value = 0

    for row in rows:
        if not isinstance(row, dict):
            warnings.append(f"{spec.qualified_id}: non-object entry in 'data': {row!r}")
            continue

        code = row.get("code")
        if code != expected_code:
            # Defensive: a response covering a different dataset than the one
            # requested (e.g. an unexpected aggregation level) is schema
            # drift, not silently mixed in.
            continue
        matched_any_code = True

        gas_day_raw = row.get("gasDayStart")
        if not isinstance(gas_day_raw, str):
            warnings.append(f"{spec.qualified_id}: row has no 'gasDayStart': {row!r}")
            continue
        try:
            observation_day = date.fromisoformat(gas_day_raw)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(f"{spec.qualified_id}: unparseable gasDayStart {gas_day_raw!r}: {exc}")
            continue

        # AGSI-only field, absent from ALSI rows (module docstring). "N" ==
        # "no data" -- documented, normal, skipped without a warning, the
        # same treatment FRED gives its own "." missing-value marker.
        status = row.get("status")
        if status == "N":
            continue

        if field not in row:
            skipped_no_value += 1
            continue
        raw_value = row[field]
        if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
            skipped_no_value += 1
            continue

        value = _parse_number(raw_value)
        if value is None:
            warnings.append(
                f"{spec.qualified_id}: unparseable value {raw_value!r} for field {field!r} "
                f"on {gas_day_raw}"
            )
            continue

        # AGSI's own documented quality flag ("E"=estimated, "C"=confirmed)
        # translated into this schema's quality_score -- not an imputation,
        # a direct mapping of a field the publisher already provides for
        # exactly this purpose. ALSI rows have no 'status' field at all, so
        # this stays at the default 1.0 for every ALSI observation.
        quality_score = 0.7 if status == "E" else 1.0

        observation_time = datetime.combine(observation_day, time(0, 0), tzinfo=UTC)
        available_at, precision = resolve_available_at(spec, observation_day)
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=value,
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=f"{source_id}_api_v2",
                observation_time=observation_time,
                available_at=available_at,
                retrieved_at=retrieved_at,
                source=source_id,
                parser_version=_PARSER_VERSION,
                quality_score=quality_score,
                is_stale=False,
                source_release_time=None,
                vintage_time=vintage_time,
                availability_precision=str(precision),
                revision_index=None,
            )
        )

    if skipped_no_value:
        warnings.append(
            f"{spec.qualified_id}: {skipped_no_value} row(s) had no usable value for field "
            f"{field!r} (missing/blank), skipped, not imputed"
        )
    if not matched_any_code:
        warnings.append(f"{spec.qualified_id}: no rows in payload matched code {expected_code!r}")
        return ParseResult(
            observations=observations, warnings=tuple(warnings), missing_series=(spec.series_id,)
        )

    return ParseResult(observations=observations, warnings=tuple(warnings))


class _GieAdapterMixin:
    """Shared `fetch`/`parse` implementation for `AgsiAdapter`/`AlsiAdapter`.

    Not itself a complete adapter (no `__init__`): each concrete subclass
    must set `self._http`/`self._base_url` (via its own mandatory
    constructor) and the class attribute `_source_id`.
    """

    _source_id: str
    _http: HttpClient
    _base_url: str

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        api_key = _require_api_key()
        country, facility, _field = parse_native_identifier(spec.native_identifier)

        # Credential-free by construction (module docstring): the key never
        # enters this dict, only the `x-key` header passed separately below.
        public_params: dict[str, str] = {"country": country}
        if facility:
            public_params["facility"] = facility
        if since is not None:
            public_params["from"] = since.isoformat()
        public_params["size"] = str(_MAX_PAGE_SIZE)

        headers = {"x-key": api_key}
        retrieved_at = datetime.now(UTC)

        all_rows: list[Any] = []
        first_envelope: dict[str, Any] | None = None
        first_status = 0
        first_content_type = ""
        date_header: str | None = None
        last_page = 1
        page = 1
        while page <= last_page and page <= _MAX_PAGES_SAFETY_CAP:
            page_params = {**public_params, "page": str(page)}
            response = self._http._request(
                "GET", self._base_url, params=page_params, headers=headers
            )
            if page == 1:
                first_status = response.status_code
                first_content_type = response.headers.get("content-type", "")
                date_header = response.headers.get("date")
            try:
                envelope = response.json()
            except ValueError as exc:
                raise AdapterError(
                    f"{self._source_id}: page {page} response is not valid JSON: {exc}"
                ) from exc
            if first_envelope is None:
                first_envelope = envelope if isinstance(envelope, dict) else {}

            if isinstance(envelope, dict):
                page_data = envelope.get("data", [])
                if isinstance(page_data, list):
                    all_rows.extend(page_data)
                elif page_data:
                    all_rows.append(page_data)
                reported_last_page = envelope.get("last_page")
                if isinstance(reported_last_page, int) and reported_last_page > last_page:
                    last_page = reported_last_page
            page += 1

        # One deterministic archived payload per fetch, regardless of how
        # many pages it took (this workstream's contract).
        combined: dict[str, Any] = dict(first_envelope or {})
        combined["data"] = all_rows
        content = json.dumps(combined, sort_keys=True).encode("utf-8")

        public_url = f"{self._base_url}?{urlencode(sorted(public_params.items()))}"
        # Only the response's own `Date` header is carried in `.headers` --
        # never the `x-key` request header (module docstring "CREDENTIAL
        # HANDLING"). `parse()` re-derives `vintage_time` from this.
        header_dict = {"date": date_header} if date_header else {}

        return FetchedPayload(
            source=self._source_id,
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
        return parse_gie_payload(
            payload.content,
            spec,
            retrieved_at=payload.retrieved_at,
            source_id=self._source_id,
            vintage_time=vintage_time,
        )


class AgsiAdapter(_GieAdapterMixin):
    """Fetches and parses AGSI (EU gas storage) series. See module docstring."""

    _source_id = _AGSI_SOURCE_ID

    def __init__(self, http: HttpClient, *, base_url: str = _AGSI_DEFAULT_BASE_URL) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")


class AlsiAdapter(_GieAdapterMixin):
    """Fetches and parses ALSI (EU LNG) series. See module docstring."""

    _source_id = _ALSI_SOURCE_ID

    def __init__(self, http: HttpClient, *, base_url: str = _ALSI_DEFAULT_BASE_URL) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")


__all__ = [
    "AgsiAdapter",
    "AlsiAdapter",
    "GieApiError",
    "GieCredentialError",
    "build_native_identifier",
    "parse_gie_payload",
    "parse_native_identifier",
]
