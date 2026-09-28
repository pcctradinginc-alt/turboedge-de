"""Eurostat dissemination API -- generic over any dataset.

`adapters/eu_bcs.py` already speaks to this API, but its identifier format
is hard-wired to the business-and-consumer-survey cube's own dimensions
(`indic`, `geo`, `s_adj`). Transport statistics have different ones --
`tra_meas`, `schedule`, `tra_cov`, `natvessr`, `unit` -- and a freight
dataset cannot be expressed in the BCS shape at all.

So this module is the general case: any dataset code, any set of dimension
filters. It reuses `eu_bcs.decode_jsonstat`, which is already fully generic
over whatever dimensions a payload declares, rather than reimplementing the
flat-index unravelling that is the easy thing to get subtly wrong.

Verified live 2026-09-28
------------------------
  rail_go_quartal  quarterly rail freight, THS_T and MIO_TKM, DE, to 2026-Q2
  avia_gooc        MONTHLY air freight and mail, tonnes, DE, to 2026-08
  iww_go_qnave     quarterly inland waterway freight, DE, to 2026-Q2
  road_go_ta_tott  road freight -- ANNUAL ONLY, and therefore not configured:
                   Destatis' daily truck-toll mileage index is a far better
                   road indicator and is already live.

Every dimension must be pinned
------------------------------
A Eurostat dataset is a cube, not a series. `avia_gooc` alone carries nine
`tra_meas` values, four `schedule` values and nine `tra_cov` values; leaving
any of them unpinned would silently return several cells for one configured
series, and this adapter would have to pick one or merge them. It does
neither -- an identifier that does not resolve to exactly one cell per
period is reported as ambiguous and returns nothing, which is a question
rather than a wrong number.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any, Final

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.adapters.eu_bcs import decode_jsonstat
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "eurostat"
_PARSER_VERSION = "1"
_DEFAULT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

#: The dimension that varies within one configured series. Everything else
#: must be pinned by the identifier.
TIME_DIMENSION: Final = "time"

_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)


class EurostatError(AdapterError):
    """The payload was not a usable Eurostat JSON-stat response."""


def build_native_identifier(dataset: str, filters: dict[str, str]) -> str:
    """`"<dataset>|<dim=val,dim=val>"`."""
    encoded = ",".join(f"{k}={v}" for k, v in sorted(filters.items()))
    return f"{dataset}|{encoded}"


def parse_native_identifier(raw: str) -> tuple[str, dict[str, str]]:
    dataset, _, tail = raw.partition("|")
    dataset = dataset.strip()
    if not dataset:
        raise EurostatError(f"eurostat: native_identifier {raw!r} has no dataset code")
    filters: dict[str, str] = {}
    for chunk in tail.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, sep, value = chunk.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise EurostatError(f"eurostat: filter {chunk!r} in {raw!r} is not 'dim=value'")
        if key == TIME_DIMENSION:
            raise EurostatError(
                f"eurostat: {raw!r} pins {TIME_DIMENSION!r}, which is the dimension the "
                "series varies along; a pinned time would yield a single observation"
            )
        filters[key] = value
    if not filters:
        raise EurostatError(
            f"eurostat: {raw!r} pins no dimension. A Eurostat dataset is a cube, not a "
            "series -- an unpinned identifier would return many cells per period"
        )
    return dataset, filters


def _period_to_date(period: str) -> date:
    """Eurostat writes `2026`, `2026-08`, `2026-Q2` or `2026-08-31`.

    The date returned is the period's **start**. That is the honest anchor:
    a quarter's figure describes the quarter, and dating it to the end would
    imply the period was over when the value refers to its whole span. The
    publication lag that decides usability is handled separately, by
    `resolve_available_at`.
    """
    text = period.strip()
    if "Q" in text:
        year, _, quarter = text.partition("-Q")
        return date(int(year), 3 * (int(quarter) - 1) + 1, 1)
    parts = text.split("-")
    if len(parts) == 1:
        return date(int(parts[0]), 1, 1)
    if len(parts) == 2:
        return date(int(parts[0]), int(parts[1]), 1)
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def parse_eurostat_payload(
    content: bytes, spec: SeriesSpec, *, dataset: str, filters: dict[str, str]
) -> tuple[list[ExternalObservation], list[str]]:
    """Select the one cell per period that `filters` pins, as observations."""
    warnings: list[str] = []
    try:
        body: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EurostatError(f"{spec.qualified_id}: response is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise EurostatError(f"{spec.qualified_id}: expected a JSON object at the top level")

    declared = list(body.get("id") or [])
    unknown = sorted(set(filters) - set(declared))
    if unknown:
        warnings.append(
            f"identifier pins dimension(s) {unknown} that {dataset} does not declare; "
            f"it has {declared}"
        )
        return [], warnings
    unpinned = [d for d in declared if d != TIME_DIMENSION and d not in filters]
    if unpinned:
        # Not an error upstream -- a genuine ambiguity in our own catalog.
        warnings.append(
            f"identifier leaves {unpinned} unpinned, so {dataset} returns more than one "
            "cell per period; refusing to pick one"
        )
        return [], warnings

    rows = decode_jsonstat(body)
    selected = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    if not selected:
        warnings.append(f"no cell in {dataset} matches {filters}")
        return [], warnings

    by_period: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_period.setdefault(str(row.get(TIME_DIMENSION)), []).append(row)

    unit = str(filters.get("unit") or spec.unit)
    observations: list[ExternalObservation] = []
    ambiguous = 0
    for period, matches in sorted(by_period.items()):
        if len(matches) > 1:
            ambiguous += 1
            continue
        try:
            period_start = _period_to_date(period)
            value = float(matches[0]["value"])
        except _PARSE_ERROR_TYPES:
            warnings.append(f"unusable period/value pair: {period!r}")
            continue
        observed = datetime.combine(period_start, datetime.min.time(), tzinfo=UTC)
        available_at, precision = resolve_available_at(spec, period_start)
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=value,
                unit=unit,
                frequency=spec.frequency,
                source_version=dataset,
                observation_time=observed,
                available_at=available_at,
                retrieved_at=datetime.now(UTC),
                source=_SOURCE_ID,
                parser_version=_PARSER_VERSION,
                quality_score=1.0,
                availability_precision=str(precision),
                vintage_time=_updated_at(body),
            )
        )
    if ambiguous:
        warnings.append(f"{ambiguous} period(s) matched more than one cell and were skipped")
    return observations, warnings


def _updated_at(body: dict[str, Any]) -> datetime | None:
    """The dataset's own `updated` field -- the closest thing Eurostat gives
    to a vintage. Not a release time for any individual observation."""
    raw = body.get("updated")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).astimezone(UTC)
    except ValueError:
        return None


class EurostatAdapter:
    """`ExternalSeriesAdapter` over any Eurostat dissemination dataset."""

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
        dataset, filters = parse_native_identifier(spec.native_identifier)
        # The filters are sent to the API as well as applied at parse time.
        # Sending them keeps the payload small; applying them again is what
        # guarantees the selected cell is the configured one even if the API
        # ever ignores or loosens a filter.
        params = {"format": "JSON", "lang": "EN", **filters}
        if since is not None:
            params["sinceTimePeriod"] = since.isoformat()
        url = f"{self._base_url}/{dataset}"
        response = self._http._request("GET", url, params=params)
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        public_url = f"{url}?{query}"
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=dataset,
            url=public_url,
            content=response.content,
            http_status=response.status_code,
            content_type=response.headers.get("content-type", "application/json"),
            retrieved_at=datetime.now(UTC),
            request_fingerprint=f"GET {public_url}",
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        dataset, filters = parse_native_identifier(spec.native_identifier)
        observations, warnings = parse_eurostat_payload(
            payload.content, spec, dataset=dataset, filters=filters
        )
        return ParseResult(
            observations=observations,
            warnings=tuple(warnings),
            missing_series=() if observations else (spec.series_id,),
        )


__all__ = [
    "TIME_DIMENSION",
    "EurostatAdapter",
    "EurostatError",
    "build_native_identifier",
    "parse_eurostat_payload",
    "parse_native_identifier",
]
