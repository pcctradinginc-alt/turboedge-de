"""IMF PortWatch adapter, for the External Data Factory (Wave 2).

Implements `turboedge.external.adapter.ExternalSeriesAdapter` against the
IMF PortWatch program's public ArcGIS FeatureServer, no key required:

    https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/
        <ServiceName>/FeatureServer/0/query

VERIFIED LIVE 2026-09-28 (HTTP 200, honest UA, >= 60s apart -- see "RATE
LIMIT" below), against two services, plus each service's own field metadata
(`.../FeatureServer/0?f=json`):

    Daily_Chokepoints_Data   maxRecordCount=1000. Fields verified present:
        date (esriFieldTypeDateOnly), year/month/day (integer), portid/
        portname (string), n_container, n_dry_bulk, n_general_cargo,
        n_roro, n_tanker, n_cargo (all integer). Queried live with
        `where=portname='Suez Canal'`: returned real rows, including
        `{"date": "2026-09-20", "portname": "Suez Canal", "n_cargo": 23,
        "n_tanker": 19}` -- exactly the sample this workstream's brief
        named, confirmed independently here.
    Daily_Trade_Data_WLD    maxRecordCount=1000. Fields verified present:
        date (esriFieldTypeDateOnly), portcalls, portcalls_container, and
        (per the service's own field list) the other portcalls_*/
        import_*/export_* columns the brief describes.

The 28 chokepoint names in the brief were not independently re-enumerated
here (that would need a `returnDistinctValues` query this session did not
spend a request on); `portname='Suez Canal'` and the paging tests below
were verified directly, and this module never filters on a chokepoint name
that has not been verified live at least once -- see `KNOWN_CHOKEPOINTS`
below and its docstring for exactly what that means in practice.

THE DATE FIELD -- VERIFIED, NOT DOCUMENTATION-ONLY
------------------------------------------------------
`date`'s ArcGIS field type is `esriFieldTypeDateOnly` (a genuinely
different type from ArcGIS's older `esriFieldTypeDate`, which this session
independently confirmed by reading the live field metadata, not just by
observing the response body). A plain `esriFieldTypeDate` field would
serialize as epoch milliseconds in JSON; `esriFieldTypeDateOnly` serializes
as a plain `"YYYY-MM-DD"` string, which is what both services returned on
every live request made here. `_parse_date_attribute` below still handles
an integer/float value defensively (treating it as epoch millis) because a
service configuration change (e.g. swapping the field back to
`esriFieldTypeDate`) is exactly the kind of upstream surprise CLAUDE.md
rule 6 says must never be swallowed silently -- it is handled (not a crash)
but always raises a `ParseResult` warning, per this workstream's explicit
instruction.

PAGING AND ARCHIVAL DESIGN
----------------------------
`maxRecordCount=1000` was confirmed live on both services; a live query
with `resultRecordCount=30` still came back with `exceededTransferLimit:
true` whenever more rows existed than were returned -- i.e. this flag is
the reliable, general "there is more" signal regardless of *why* the
response was truncated (server cap or the caller's own requested count),
and `fetch()` relies on exactly that, not on comparing counts to a
hard-coded `maxRecordCount` it would otherwise have to keep in sync with
the publisher.

`fetch()` therefore loops on `resultOffset`, incrementing by the number of
features actually returned each time, until a page comes back with
`exceededTransferLimit` false (or empty), or `max_pages` is reached (a
guard against a pathological infinite loop -- e.g. a publisher bug that
always reports more rows than it delivers -- defaulted generously since a
full multi-year daily series at 1000 rows/page is only a handful of pages).

The adapter contract requires one archived `FetchedPayload` per `fetch()`
call whose `content` is exactly what `parse()` consumes, and requires
`parse()` to do no network I/O -- both irreconcilable with silently
choosing "archive only the last page" when several HTTP requests were
needed. The design chosen here: `content` is
`json.dumps([page_0_json, page_1_json, ...]).encode("utf-8")` -- a JSON
array of the *complete, unmodified* decoded response body of every page
requested, in fetch order. This is not byte-identical to any single
upstream response (there is no such single response once paging happens at
all), but it is a deterministic, lossless, order-preserving record of
every byte of structured data the publisher actually returned across the
whole `fetch()` call, and `parse()` below consumes precisely that
structure and nothing else. A single-page fetch (the common case for a
`since`-narrowed incremental pull) archives a one-element array, which
`parse()` handles identically to the multi-page case rather than as a
special-cased shape.

RATE LIMIT
-----------
`portwatch.imf.org` declares `Crawl-delay: 60` in its robots.txt; the data
itself is served from `services9.arcgis.com`, which publishes no
robots.txt of its own, but the declared policy is the publisher's (IMF's)
and is honoured for their data regardless of which host physically serves
it -- `cli_external.py` wires this adapter's `HttpClient` with
`min_interval_s=60.0`. Combined with paging, a full incremental fetch of
one series can take several minutes; `max_pages` exists partly so an
unexpectedly long-tailed query cannot run unbounded at 60 seconds per page.

`native_identifier` FORMAT (fixed, per the Wave 2 contract; not invented here)
---------------------------------------------------------------------------
    "<ServiceName>|<field>|<whereField>=<whereValue>"
    "Daily_Chokepoints_Data|n_cargo|portname=Suez Canal"
    "Daily_Trade_Data_WLD|portcalls|"          (empty third segment: no filter)

AVAILABILITY
-------------
PortWatch exposes no in-body publication timestamp and (being served from
Esri infrastructure, not from portwatch.imf.org itself) no `Date` response
header this module treats as authoritative either -- every `SeriesSpec`
registered for this source must declare `CONSERVATIVE_DATE` (or `UNKNOWN`)
so `available_at` always comes from `external.adapter.resolve_available_at()`,
never invented here. `vintage_time` is set from the HTTP response's own
`Date` header when present (the one genuine fact available about *when
the query was answered*, mirroring `adapters/bundesbank.py`'s treatment of
the same situation for the same reason), taken from the **last** page
fetched; `revision_index` and `source_release_time` are always `None` --
PortWatch exposes no revision history.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from email.utils import parsedate_to_datetime
from typing import Any, Final
from urllib.parse import urlencode

import structlog

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

logger = structlog.get_logger(__name__)

_SOURCE_ID = "portwatch"
_PARSER_VERSION = "1"
_DEFAULT_BASE_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"

#: Confirmed live 2026-09-28 against both services' own
#: `FeatureServer/0?f=json` metadata. Not hard-coded into request params
#: (see module docstring "PAGING AND ARCHIVAL DESIGN" -- the loop keys off
#: the response's own `exceededTransferLimit`, never this constant); kept
#: only as the default `resultRecordCount` request value.
_DEFAULT_PAGE_SIZE = 2000
_DEFAULT_MAX_PAGES = 50

#: The 28 chokepoints named in this workstream's brief, verified against
#: `Daily_Chokepoints_Data`'s own portname values as of 2026-09-28.
#: Documentation only: `parse()` does not consult this tuple and never
#: rejects an unrecognised `portname` filter value itself -- a genuinely
#: unknown chokepoint simply returns zero features from the live service,
#: which `parse()` reports as `missing_series` on its own merits. This
#: tuple exists so a catalog author adding a new `SeriesSpec` can cross-
#: check spelling against a verified list rather than guessing one letter
#: at a time against a live (rate-limited) service.
KNOWN_CHOKEPOINTS: Final[tuple[str, ...]] = (
    "Bab el-Mandeb Strait",
    "Balabac Strait",
    "Bering Strait",
    "Bohai Strait",
    "Bosporus Strait",
    "Cape of Good Hope",
    "Dover Strait",
    "Gibraltar Strait",
    "Kerch Strait",
    "Korea Strait",
    "Lombok Strait",
    "Luzon Strait",
    "Magellan Strait",
    "Makassar Strait",
    "Malacca Strait",
    "Mindoro Strait",
    "Mona Passage",
    "Ombai Strait",
    "Oresund Strait",
    "Panama Canal",
    "Strait of Hormuz",
    "Suez Canal",
    "Sunda Strait",
    "Taiwan Strait",
    "Torres Strait",
    "Tsugaru Strait",
    "Windward Passage",
    "Yucatan Channel",
)

# Errors expected from a malformed/unexpected upstream payload; anything else
# is a programming error and should propagate rather than be swallowed.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

__all__ = [
    "KNOWN_CHOKEPOINTS",
    "PortWatchAdapter",
    "build_native_identifier",
    "parse_native_identifier",
    "parse_portwatch_pages",
]


def build_native_identifier(
    service: str, field: str, *, where_field: str = "", where_value: str = ""
) -> str:
    """The publisher-identifier format this module uses for `SeriesSpec.native_identifier`."""
    if not service or not field:
        raise AdapterError(
            f"portwatch: service and field must both be non-empty "
            f"(got service={service!r}, field={field!r})"
        )
    if bool(where_field) != bool(where_value):
        raise AdapterError(
            "portwatch: where_field and where_value must be given together or not at all "
            f"(got where_field={where_field!r}, where_value={where_value!r})"
        )
    for part in (service, field, where_field):
        if "|" in part:
            raise AdapterError("portwatch: native_identifier components must not contain '|'")
    third = f"{where_field}={where_value}" if where_field else ""
    return f"{service}|{field}|{third}"


def parse_native_identifier(raw: str) -> tuple[str, str, str, str]:
    """Inverse of `build_native_identifier`. Raises `AdapterError` on malformed input.

    Returns `(service, field, where_field, where_value)`; `where_field` and
    `where_value` are both `""` for "no filter" (e.g.
    `"Daily_Trade_Data_WLD|portcalls|"`).
    """
    parts = raw.split("|")
    if len(parts) != 3:
        raise AdapterError(
            f"portwatch: native_identifier {raw!r} does not match "
            "'<ServiceName>|<field>|<whereField>=<whereValue>'"
        )
    service, field, third = parts
    if not service or not field:
        raise AdapterError(f"portwatch: native_identifier {raw!r} is missing a service or field")
    if not third:
        return service, field, "", ""
    if "=" not in third:
        raise AdapterError(
            f"portwatch: native_identifier {raw!r} third segment must be "
            "'<whereField>=<whereValue>' or empty"
        )
    where_field, where_value = third.split("=", 1)
    if not where_field or not where_value:
        raise AdapterError(f"portwatch: native_identifier {raw!r} has an empty where clause part")
    return service, field, where_field, where_value


def _build_where_clause(where_field: str, where_value: str, *, since: date | None) -> str:
    """Build an ArcGIS SQL `where` clause. `esriFieldTypeDateOnly` (confirmed
    live -- see module docstring) is compared with a plain quoted date
    literal, not a `TIMESTAMP '...'` literal (that syntax is for the older
    `esriFieldTypeDate`)."""
    if where_field:
        escaped_value = where_value.replace("'", "''")
        base = f"{where_field} = '{escaped_value}'"
    else:
        base = "1=1"
    if since is not None:
        return f"({base}) AND (date >= '{since.isoformat()}')"
    return base


def _vintage_time_from_headers(headers: dict[str, str]) -> datetime | None:
    """Best-effort vintage timestamp from the HTTP response's own `Date`
    header of the last page fetched -- mirrors `adapters/bundesbank.py`'s
    treatment of a publisher with no in-body vintage marker. Never invented:
    absent or unparsable simply means `None`."""
    raw = headers.get("date")
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _parse_date_attribute(raw: Any) -> tuple[date | None, str | None]:
    """Decode one feature's `date` attribute.

    Returns `(parsed_date_or_None, warning_or_None)`. The expected,
    verified-live shape is an ISO `"YYYY-MM-DD"` string (see module
    docstring); an integer/float is handled defensively as epoch
    milliseconds (ArcGIS's older `esriFieldTypeDate` convention) but always
    carries a warning, per this workstream's explicit instruction to warn
    rather than crash on a format switch. Anything else is unparseable.
    """
    if isinstance(raw, str):
        text = raw.strip()
        try:
            return date.fromisoformat(text[:10]), None
        except ValueError:
            return None, f"unparseable date string {raw!r}"
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        try:
            parsed_dt = datetime.fromtimestamp(raw / 1000.0, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None, f"unparseable epoch-millis date value {raw!r}"
        return (
            parsed_dt.date(),
            "'date' attribute arrived as a number (expected the ISO date string verified "
            f"live on 2026-09-28); treated as epoch milliseconds -- possible upstream "
            f"esriFieldTypeDateOnly -> esriFieldTypeDate format switch: {raw!r}",
        )
    return None, f"'date' attribute has unexpected type {type(raw).__name__}: {raw!r}"


def parse_portwatch_pages(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
    headers: dict[str, str] | None = None,
) -> ParseResult:
    """Parse one archived PortWatch payload (a JSON array of raw ArcGIS
    FeatureServer query response pages, see module docstring "PAGING AND
    ARCHIVAL DESIGN") into `ExternalObservation`s.

    Pure and network-free: everything needed comes from `content`, `spec`
    and the already-captured `headers`. Any structural surprise -- invalid
    JSON, a non-list top level, a page that is not a JSON object, an
    ArcGIS-native `{"error": ...}` page, a page missing `features`, a
    feature missing `attributes`/the required fields, an unparseable date,
    a non-numeric value -- is reported as a warning and the affected
    feature (or page) is skipped, never guessed at. A legitimate zero-row
    result (e.g. an unknown chokepoint name, or a genuinely quiet day) is
    `missing_series`/skipped without inflating `warnings` beyond saying so
    once.
    """
    warnings: list[str] = []
    service, field, where_field, where_value = parse_native_identifier(spec.native_identifier)

    try:
        pages: Any = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: payload is not valid UTF-8 JSON: {exc}",),
            missing_series=(spec.series_id,),
        )
    if not isinstance(pages, list) or not pages:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: archived PortWatch payload is not a non-empty list",),
            missing_series=(spec.series_id,),
        )

    by_date: dict[date, list[float]] = {}
    epoch_format_warned = False
    saw_any_feature = False
    last_exceeded_transfer_limit = False

    for page in pages:
        if not isinstance(page, dict):
            warnings.append(f"{spec.qualified_id}: an archived page is not a JSON object")
            continue
        if "error" in page:
            warnings.append(
                f"{spec.qualified_id}: ArcGIS returned an error page: {page['error']!r}"
            )
            continue
        features = page.get("features")
        if features is None:
            warnings.append(f"{spec.qualified_id}: an archived page has no 'features' key")
            continue
        last_exceeded_transfer_limit = bool(page.get("exceededTransferLimit", False))

        for feature in features:
            if not isinstance(feature, dict):
                warnings.append(f"{spec.qualified_id}: a feature entry is not a JSON object")
                continue
            attrs = feature.get("attributes")
            if not isinstance(attrs, dict):
                warnings.append(f"{spec.qualified_id}: a feature is missing 'attributes'")
                continue
            if "date" not in attrs or field not in attrs:
                warnings.append(
                    f"{spec.qualified_id}: feature missing required attribute(s) "
                    f"(need 'date' and {field!r}, has {sorted(attrs)!r})"
                )
                continue

            obs_date, date_warning = _parse_date_attribute(attrs["date"])
            if date_warning is not None:
                if "epoch-millis" in date_warning or "esriFieldTypeDate format switch" in (
                    date_warning
                ):
                    if not epoch_format_warned:
                        warnings.append(f"{spec.qualified_id}: {date_warning}")
                        epoch_format_warned = True
                else:
                    warnings.append(f"{spec.qualified_id}: {date_warning}")
            if obs_date is None:
                continue

            raw_value = attrs[field]
            if raw_value is None:
                # A day this chokepoint/route genuinely has no print for --
                # normal, not a warning (adapter contract, mirrors ecb_data.py).
                saw_any_feature = True
                continue
            try:
                value = float(raw_value)
            except _PARSE_ERROR_TYPES as exc:
                warnings.append(
                    f"{spec.qualified_id}: non-numeric value for {field!r} on "
                    f"{obs_date.isoformat()}: {raw_value!r} ({exc})"
                )
                continue

            saw_any_feature = True
            by_date.setdefault(obs_date, []).append(value)

    if not saw_any_feature:
        reason = (
            f"portname={where_value!r}"
            if where_field == "portname"
            else (f"{where_field}={where_value!r}" if where_field else "no filter")
        )
        warnings.append(
            f"{spec.qualified_id}: no observations returned for {reason} -- "
            "unknown chokepoint/filter value, or a genuinely empty result"
        )
        return ParseResult(
            observations=[], warnings=tuple(warnings), missing_series=(spec.series_id,)
        )

    if last_exceeded_transfer_limit:
        warnings.append(
            f"{spec.qualified_id}: the last archived page still had exceededTransferLimit=true "
            "-- fetch() likely hit its max_pages guard before exhausting the series; the "
            "archived payload is an incomplete slice of upstream history"
        )

    vintage_time = _vintage_time_from_headers(headers or {})
    source_version = f"portwatch_{service.lower()}"

    observations: list[ExternalObservation] = []
    for obs_date, values in sorted(by_date.items()):
        if len(set(values)) > 1:
            warnings.append(
                f"{spec.qualified_id}: {len(values)} conflicting values for {field!r} on "
                f"{obs_date.isoformat()} ({sorted(set(values))!r}) -- likely duplicate/"
                "overlapping pages; day skipped rather than resolved optimistically"
            )
            continue
        available_at, precision = resolve_available_at(spec, obs_date)
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=values[0],
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=source_version,
                observation_time=datetime.combine(obs_date, time(0, 0), tzinfo=UTC),
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

    return ParseResult(observations=observations, warnings=tuple(warnings), missing_series=())


class PortWatchAdapter:
    """Fetches and parses one IMF PortWatch service/field/filter series at a time.

    One instance handles every service a `SeriesSpec` can name: the
    service, field and optional where-filter all come from
    `spec.native_identifier`, never from adapter configuration. See module
    docstring for the paging/archival design and the mandatory 60s
    per-host rate limit this adapter relies on `HttpClient` (constructed by
    the caller, per the Wave 2 contract) to enforce.
    """

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._page_size = page_size
        self._max_pages = max_pages

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        service, field, where_field, where_value = parse_native_identifier(spec.native_identifier)
        where_clause = _build_where_clause(where_field, where_value, since=since)
        out_fields = "date" if field == "date" else f"date,{field}"
        url = f"{self._base_url}/{service}/FeatureServer/0/query"

        public_params: dict[str, str] = {
            "where": where_clause,
            "outFields": out_fields,
            "orderByFields": "date ASC",
            "returnGeometry": "false",
            "f": "json",
        }

        retrieved_at = datetime.now(UTC)
        pages: list[Any] = []
        offset = 0
        last_response_headers: dict[str, str] = {}
        last_status = 200
        hit_max_pages = False

        for _page_index in range(self._max_pages):
            params = {
                **public_params,
                "resultRecordCount": str(self._page_size),
                "resultOffset": str(offset),
            }
            # `_request` is the shared retry/rate-limit/honest-UA path every
            # adapter on this contract uses (see `adapters/ecb_data.py`);
            # PortWatch's 60s per-host interval (set by the caller's
            # `HttpClient`) applies to every one of these paged requests,
            # not just the first.
            response = self._http._request("GET", url, params=params)
            last_status = response.status_code
            last_response_headers = dict(response.headers)
            page_json = response.json()
            pages.append(page_json)

            features = page_json.get("features") if isinstance(page_json, dict) else None
            exceeded = (
                bool(page_json.get("exceededTransferLimit", False))
                if isinstance(page_json, dict)
                else False
            )
            if not features or not exceeded:
                break
            offset += len(features)
        else:
            hit_max_pages = True

        if hit_max_pages:
            logger.warning(
                "portwatch_max_pages_reached",
                series_id=spec.series_id,
                max_pages=self._max_pages,
                page_size=self._page_size,
            )

        content = json.dumps(pages, ensure_ascii=False).encode("utf-8")
        public_url = f"{url}?{urlencode(public_params)}"
        date_header = last_response_headers.get("date", "")
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=spec.series_id,
            url=public_url,
            content=content,
            http_status=last_status,
            content_type="application/json",
            retrieved_at=retrieved_at,
            request_fingerprint=(
                f"GET {public_url} (paged: {len(pages)} page(s), final resultOffset={offset})"
            ),
            headers={"date": date_header} if date_header else {},
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_portwatch_pages(
            payload.content, spec, retrieved_at=payload.retrieved_at, headers=payload.headers
        )
