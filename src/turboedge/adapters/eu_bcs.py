"""EU Business & Consumer Surveys (BCS) adapter, via the Eurostat dissemination API.

Two datasets, both verified live on 2026-09-28 (HTTP 200, JSON-stat 2.0 body,
honest User-Agent identifying this project, one request per host per second):

    ei_bssi_m_r2  "Economic sentiment and confidence indicators by sector -
                   monthly data"
      https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/
        ei_bssi_m_r2?format=JSON&lang=EN&geo=DE&lastTimePeriod=2
      Confirmed indicator codes present in this dataset's own `dimension`
      metadata (not guessed): BS-ESI-I (Economic sentiment indicator,
      SA-only -- no NSA series is published for it, confirmed via the
      response's own `positions-with-no-data`), BS-CCI-BAL (Construction),
      BS-ICI-BAL (Industrial), BS-RCI-BAL (Retail), BS-CSMCI-BAL (Consumer),
      BS-SCI-BAL (Services) -- each of the five sectoral indicators has both
      NSA and SA published.

      IMPORTANT CORRECTION to this workstream's brief: the brief expected an
      "Employment Expectations Indicator" in this dataset. It is not there.
      Querying `indic=BS-EEI-BAL` against this dataset returns an empty
      `indic` category (verified live, see
      tests/fixtures/external/eu_bcs/ei_bssi_m_r2_unknown_indicator_EEI.json)
      -- i.e. that code does not exist in `ei_bssi_m_r2` for any geo. Per
      CLAUDE.md rule 1 ("never invent a series identifier"), no
      Employment Expectations series is registered by this module. If it
      exists at all it is a different Eurostat dataset and would need its
      own live verification before being added here.

    ei_bsco_m     "Consumers confidence indicator and survey results -
                   monthly data"
      https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/
        ei_bsco_m?format=JSON&lang=EN&geo=DE&lastTimePeriod=2
      Confirmed indicator `BS-CSMCI` (the headline consumer confidence
      indicator, distinct dataset/code from `BS-CSMCI-BAL` above but the
      same underlying concept) plus 11 more granular survey-question
      balances, all visible in this dataset's own `dimension.indic`
      metadata. Only `BS-CSMCI` is wired up as a default series below; the
      parser is fully generic (see `decode_jsonstat`), so any of the other
      11 codes can be added by constructing another `SeriesSpec` with the
      matching `native_identifier` -- no code change needed here.

    Euro-area geo code: neither dataset accepts the bare code "geo=EA"
    (verified live: the response's `dimension.geo.category.index` comes
    back empty, see ei_bssi_m_r2_unknown_geo_EA.json). Eurostat renames the
    aggregate as euro-area membership changes; the code valid *today*
    (2026-09-28, following Bulgaria's 2026-01-01 accession) is "EA21"
    ("Euro area - 21 countries (from 2026)"), confirmed live for both
    datasets and both back to at least 2025. A future accession will bump
    it again -- this is exactly the kind of identifier CLAUDE.md rule 1
    means by "verify, don't guess", and it should be re-verified rather
    than assumed to still be EA21 far in the future.

JSON-stat 2.0 decoding
-----------------------
`value` is a sparse mapping from a single flat string index to a number.
The flat index must be decoded against `id` (the dimension order) and
`size` (each dimension's cardinality) using row-major (C-order) unraveling:
the *last* listed dimension varies fastest. This was verified by hand
against the live ei_bssi_m_r2 payload: for `id=[freq,indic,s_adj,geo,time]`
and `size=[1,6,2,1,2]`, key "6" decodes to (freq=M, indic=BS-ESI-I [pos 1],
s_adj=SA [pos 1], geo=DE [pos 0], time=2026-07 [pos 0]) -> 92.8, which
matches both the sign/magnitude expected for ESI and the fact that
`positions-with-no-data` separately confirms ESI has no NSA data (keys 4
and 5, which would have been indic=1/s_adj=0, are simply absent from
`value` -- the map is genuinely sparse with gaps, not merely reordered).
`decode_jsonstat` below implements this generically over whatever
dimensions a payload actually declares (so `ei_bsco_m`'s extra `unit`
dimension needs no special-casing) and is covered by
`tests/adapters/test_eu_bcs.py::test_decode_jsonstat_matches_hand_checked_mapping`.

Point-in-time honesty
----------------------
This API serves only the *current* revision of the whole time series --
there is no vintage/revision history endpoint, confirmed by the absence of
any such parameter in the dissemination API and by the dataset-level
(not observation-level) `updated` timestamp being the only revision signal
available. That single `updated` timestamp is stored as `vintage_time` on
every observation from a given fetch (it says "this is the state of the
value as of this dataset revision", which is honest); `revision_index` is
left `None` because Eurostat does not number revisions. Every `SeriesSpec`
registered for this source must declare `availability_precision=UNKNOWN`:
`resolve_available_at()` (the *only* place `available_at` is decided, per
CLAUDE.md rule 5) then produces a placeholder `available_at` that strict
point-in-time research automatically excludes (`STRICT_PIT_PRECISIONS`
does not include UNKNOWN). This is the honest reflection of the fact that
a re-fetch can silently change a "2024-03" value years after the fact with
no record of when the change happened, which makes the series'
`backfill_class` FORWARD_ONLY, not HISTORICAL_PIT_SAFE or even
HISTORICAL_CONSERVATIVE -- only observations TurboEdge itself has snapshotted
going forward are usable as confirmatory evidence.

`observation_time` is set to the first day of the reported month (the
conventional label for a monthly period). The `observation_day` passed
into `resolve_available_at()` is deliberately the *last* day of that same
month, not the first: the real-world BCS release for month M typically
appears near the end of month M itself (the payload captured on
2026-09-28 example, "updated": "2026-08-28T11:00:00+0200", is consistent
with the August release having appeared on 2026-08-28). Anchoring on the
last day rather than the first avoids a needlessly-early placeholder, even
though UNKNOWN precision already means strict research must ignore the
exact value.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from typing import Any

import structlog

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec
from turboedge.storage.schemas import ExternalObservation

logger = structlog.get_logger(__name__)

_SOURCE_ID = "eu_bcs"
_PARSER_VERSION = "1"
_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
_USER_AGENT = "TurboEdge-DE-research/0.1 (contact: pcctradinginc@gmail.com)"
#: Eurostat publishes no crawl-delay in robots.txt for this host; one
#: request per second is CLAUDE.md's own general default for an
#: unspecified host, not something this dataset demanded.
_MIN_INTERVAL_S = 1.0

_NATIVE_ID_RE = re.compile(
    r"^(?P<dataset>[a-z0-9_]+)\|indic=(?P<indic>[A-Za-z0-9_-]+)"
    r"\|geo=(?P<geo>[A-Za-z0-9]+)\|s_adj=(?P<s_adj>[A-Za-z0-9]+)$"
)
_TIME_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")

#: Indicator codes actually confirmed present in `ei_bssi_m_r2`'s own
#: dimension metadata (2026-09-28). Documentation only -- `parse()` does
#: not consult this dict; it trusts each payload's own metadata every time.
EI_BSSI_M_R2_INDICATORS: dict[str, str] = {
    "BS-ESI-I": "Economic sentiment indicator (SA only)",
    "BS-CCI-BAL": "Construction confidence indicator",
    "BS-ICI-BAL": "Industrial confidence indicator",
    "BS-RCI-BAL": "Retail confidence indicator",
    "BS-CSMCI-BAL": "Consumer confidence indicator",
    "BS-SCI-BAL": "Services confidence indicator",
}

#: Verified 2026-09-28: the current euro-area aggregate code. Re-verify
#: before trusting this far in the future -- see module docstring.
GEO_EURO_AREA = "EA21"
GEO_GERMANY = "DE"


def build_native_identifier(dataset: str, indic: str, geo: str, s_adj: str) -> str:
    """The publisher-identifier format this module uses for `SeriesSpec.native_identifier`.

    Encodes exactly what is needed to both build the fetch URL and pick the
    right cell out of the decoded JSON-stat cube: which dataset, which
    indicator code, which geo code, which seasonal-adjustment code.
    """
    return f"{dataset}|indic={indic}|geo={geo}|s_adj={s_adj}"


def parse_native_identifier(raw: str) -> tuple[str, str, str, str]:
    """Inverse of `build_native_identifier`. Raises on malformed input."""
    match = _NATIVE_ID_RE.match(raw)
    if not match:
        raise AdapterError(
            f"eu_bcs: native_identifier {raw!r} does not match "
            "'<dataset>|indic=<code>|geo=<code>|s_adj=<code>'"
        )
    return match["dataset"], match["indic"], match["geo"], match["s_adj"]


def make_series_spec(
    *,
    dataset: str,
    indic: str,
    geo: str,
    s_adj: str,
    series_id: str,
    name: str,
    category: str,
    unit: str = "balance",
) -> SeriesSpec:
    """Convenience constructor for a correctly-shaped eu_bcs `SeriesSpec`.

    Always UNKNOWN precision / FORWARD_ONLY backfill class -- see the
    "Point-in-time honesty" section of the module docstring for why this
    source cannot honestly declare anything stronger.
    """
    return SeriesSpec(
        source=_SOURCE_ID,
        series_id=series_id,
        name=name,
        category=category,
        unit=unit,
        frequency="monthly",
        native_identifier=build_native_identifier(dataset, indic, geo, s_adj),
        availability_precision=AvailabilityPrecision.UNKNOWN,
        backfill_class=BackfillClass.FORWARD_ONLY,
    )


def default_series_specs() -> list[SeriesSpec]:
    """The indicators this module has actually verified live, for DE and EA21.

    Seasonally-adjusted (SA) only, since that is the conventional
    cross-country-comparable headline figure and the one guaranteed to
    exist for every indicator here (BS-ESI-I has no NSA series at all).
    NSA series are also confirmed available for the five sectoral
    indicators and can be added the same way if a caller wants them.
    """
    specs: list[SeriesSpec] = []
    for geo in (GEO_GERMANY, GEO_EURO_AREA):
        for indic, description in EI_BSSI_M_R2_INDICATORS.items():
            suffix = indic.removeprefix("BS-").removesuffix("-BAL").removesuffix("-I")
            specs.append(
                make_series_spec(
                    dataset="ei_bssi_m_r2",
                    indic=indic,
                    geo=geo,
                    s_adj="SA",
                    series_id=f"EU_BCS.{geo}.{suffix}.SA",
                    name=f"{description} ({geo}, SA)",
                    category="business_consumer_survey",
                )
            )
    specs.append(
        make_series_spec(
            dataset="ei_bsco_m",
            indic="BS-CSMCI",
            geo=GEO_GERMANY,
            s_adj="SA",
            series_id="EU_BCS.DE.CSMCI_DETAIL.SA",
            name="Consumer confidence indicator, detailed survey (DE, SA)",
            category="business_consumer_survey",
        )
    )
    return specs


def _unravel_row_major(flat_index: int, sizes: Sequence[int]) -> list[int]:
    """Decode a JSON-stat flat index into per-dimension positions.

    Row-major / C-order: the *last* dimension in `sizes` varies fastest.
    Equivalent to `numpy.unravel_index(flat_index, sizes, order="C")`,
    verified against numpy for the hand-checked example in the module
    docstring, but implemented without a numpy dependency since it is a
    handful of lines of pure arithmetic.
    """
    positions = [0] * len(sizes)
    remaining = flat_index
    for i in range(len(sizes) - 1, -1, -1):
        size = sizes[i]
        if size <= 0:
            raise AdapterError(f"eu_bcs: JSON-stat dimension {i} has non-positive size {size}")
        positions[i] = remaining % size
        remaining //= size
    if remaining != 0:
        raise AdapterError(f"eu_bcs: value index {flat_index} out of bounds for sizes {sizes!r}")
    return positions


def decode_jsonstat(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode a JSON-stat 2.0 dataset body into flat rows.

    Each returned row is `{<dim_id>: <category_code>, ..., "value": <float>}`
    for one non-missing cell. Deliberately generic over whatever dimensions
    the payload declares in `id` -- it does not assume `freq`/`indic`/
    `s_adj`/`geo`/`time` specifically, so it decodes both `ei_bssi_m_r2`
    (5 dims) and `ei_bsco_m` (6 dims, extra `unit`) without change.

    Raises `AdapterError` for structural problems (missing keys, id/size
    length mismatch, a value index outside the declared shape, a dimension
    with no category matching a position) -- those are upstream contract
    changes, not ordinary missing data, and must fail loudly rather than
    silently produce an empty or mislabelled result.
    """
    try:
        dim_ids: list[str] = payload["id"]
        sizes: list[int] = payload["size"]
        value_map: dict[str, Any] = payload["value"]
        dimension_meta: dict[str, Any] = payload["dimension"]
    except KeyError as exc:
        raise AdapterError(f"eu_bcs: JSON-stat payload missing required key {exc}") from exc

    if len(dim_ids) != len(sizes):
        raise AdapterError(
            f"eu_bcs: 'id' has {len(dim_ids)} dimensions but 'size' has {len(sizes)}"
        )

    position_to_code: list[dict[int, str]] = []
    for dim_id in dim_ids:
        try:
            index_map = dimension_meta[dim_id]["category"]["index"]
        except (KeyError, TypeError) as exc:
            raise AdapterError(
                f"eu_bcs: dimension {dim_id!r} has no dimension.category.index metadata"
            ) from exc
        position_to_code.append({pos: code for code, pos in index_map.items()})

    rows: list[dict[str, Any]] = []
    for flat_key, value in value_map.items():
        try:
            flat_index = int(flat_key)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"eu_bcs: non-integer value key {flat_key!r}") from exc
        positions = _unravel_row_major(flat_index, sizes)
        row: dict[str, Any] = {}
        for dim_id, pos, lookup in zip(dim_ids, positions, position_to_code, strict=True):
            code = lookup.get(pos)
            if code is None:
                raise AdapterError(
                    f"eu_bcs: value index {flat_index} decodes to position {pos} for "
                    f"dimension {dim_id!r}, which declares no such category"
                )
            row[dim_id] = code
        row["value"] = value
        rows.append(row)
    return rows


def _month_bounds(time_label: str) -> tuple[date, date] | None:
    """First and last calendar day of a JSON-stat 'YYYY-MM' time label."""
    import calendar

    match = _TIME_RE.match(time_label)
    if not match:
        return None
    year, month = int(match["year"]), int(match["month"])
    if not (1 <= month <= 12):
        return None
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    return first, last


def parse_jsonstat_payload(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
) -> ParseResult:
    """Pure parse function: bytes + spec -> observations. No network I/O.

    Exposed at module level (rather than only as a method) so tests can
    exercise it directly against a captured fixture.
    """
    warnings: list[str] = []
    try:
        raw = json.loads(content.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise AdapterError(f"eu_bcs: payload is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AdapterError(f"eu_bcs: payload is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict) or raw.get("class") != "dataset" or raw.get("version") != "2.0":
        got_class = raw.get("class") if isinstance(raw, dict) else type(raw).__name__
        got_version = raw.get("version") if isinstance(raw, dict) else None
        raise AdapterError(
            f"eu_bcs: payload does not look like a JSON-stat 2.0 dataset "
            f"(class={got_class!r}, version={got_version!r})"
        )

    dataset, indic, geo, s_adj = parse_native_identifier(spec.native_identifier)

    updated_raw = raw.get("updated")
    vintage_time: datetime | None = None
    if isinstance(updated_raw, str):
        try:
            vintage_time = datetime.fromisoformat(updated_raw).astimezone(UTC)
        except ValueError:
            warnings.append(f"eu_bcs: could not parse 'updated' timestamp {updated_raw!r}")
    else:
        warnings.append("eu_bcs: payload has no 'updated' timestamp")

    rows = decode_jsonstat(raw)
    matched = [
        row
        for row in rows
        if row.get("indic") == indic and row.get("geo") == geo and row.get("s_adj") == s_adj
    ]

    dim_meta = raw.get("dimension", {})
    indic_known = indic in dim_meta.get("indic", {}).get("category", {}).get("index", {})
    geo_known = geo in dim_meta.get("geo", {}).get("category", {}).get("index", {})

    if not matched:
        reasons = []
        if not indic_known:
            reasons.append(f"indicator {indic!r} not present in this dataset's metadata")
        if not geo_known:
            reasons.append(f"geo {geo!r} not present in this dataset's metadata")
        if not reasons:
            reasons.append("no observations for this indic/geo/s_adj combination in this payload")
        warnings.append(f"{spec.qualified_id}: " + "; ".join(reasons))
        return ParseResult(
            observations=[], warnings=tuple(warnings), missing_series=(spec.series_id,)
        )

    by_time: dict[str, list[dict[str, Any]]] = {}
    for row in matched:
        by_time.setdefault(row["time"], []).append(row)

    observations: list[ExternalObservation] = []
    for time_label, group in sorted(by_time.items()):
        if len({g["value"] for g in group}) > 1:
            warnings.append(
                f"{spec.qualified_id}: ambiguous JSON-stat rows for time {time_label!r} "
                f"(an unresolved extra dimension produced {len(group)} distinct values); skipped"
            )
            continue
        bounds = _month_bounds(time_label)
        if bounds is None:
            warnings.append(f"{spec.qualified_id}: unparseable time label {time_label!r}; skipped")
            continue
        first_day, last_day = bounds
        observation_time = datetime.combine(first_day, time(0, 0), tzinfo=UTC)
        available_at, precision = resolve_available_at(spec, last_day)
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=float(group[0]["value"]),
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=dataset,
                observation_time=observation_time,
                available_at=available_at,
                retrieved_at=retrieved_at,
                source=_SOURCE_ID,
                parser_version=_PARSER_VERSION,
                quality_score=1.0,
                vintage_time=vintage_time,
                availability_precision=str(precision),
                revision_index=None,
            )
        )
    return ParseResult(observations=observations, warnings=tuple(warnings))


class EuBcsAdapter:
    """Fetches and parses EU Business & Consumer Survey series from Eurostat."""

    def __init__(self, http_client: HttpClient | None = None, *, base_url: str = _BASE_URL) -> None:
        self._http = http_client or HttpClient(
            user_agent=_USER_AGENT,
            min_interval_s=_MIN_INTERVAL_S,
        )
        self._base_url = base_url.rstrip("/")

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        dataset, indic, geo, _s_adj = parse_native_identifier(spec.native_identifier)
        params = {"format": "JSON", "lang": "EN", "geo": geo, "indic": indic}
        if since is not None:
            params["sinceTimePeriod"] = f"{since.year:04d}-{since.month:02d}"
        query = "&".join(f"{key}={value}" for key, value in params.items())
        url = f"{self._base_url}/{dataset}?{query}"

        retrieved_at = datetime.now(UTC)
        content = self._http.get_bytes(url)
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=dataset,
            url=url,
            content=content,
            http_status=200,
            content_type="application/json",
            retrieved_at=retrieved_at,
            request_fingerprint=url,
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_jsonstat_payload(payload.content, spec, retrieved_at=payload.retrieved_at)


__all__ = [
    "EI_BSSI_M_R2_INDICATORS",
    "GEO_EURO_AREA",
    "GEO_GERMANY",
    "EuBcsAdapter",
    "build_native_identifier",
    "decode_jsonstat",
    "default_series_specs",
    "make_series_spec",
    "parse_jsonstat_payload",
    "parse_native_identifier",
]
