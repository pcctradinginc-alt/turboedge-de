"""Energy-Charts (Fraunhofer ISE) -- German and European electricity data.

Verified live 2026-09-28
------------------------
`https://api.energy-charts.info`, no credential, no registration. Every
endpoint below answered HTTP 200 with JSON:

  /v2/public_power?country=de           21 series at PT15M, 96 points/day,
                                        including `load`, `residual_load`
                                        and every generation type
  /v2/public_power_forecast?country=de&production_type=load&forecast_type=day-ahead
                                        192 points, the day-ahead load forecast
  /v2/price?bzn=DE-LU                   series `day_ahead_price`, EUR/MWh

Why this source exists alongside ENTSO-E
----------------------------------------
It serves the same underlying grid data, and it is the pragmatic choice for
two reasons that matter more than provenance purity:

* **The licence travels in the payload.** Every response carries a `license`
  field, verbatim: "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from
  Bundesnetzagentur | SMARD.de". No licence page to read, nothing to leave
  at REVIEW_REQUIRED. `parse()` checks that the declared licence still
  matches what was reviewed and warns when it changes -- a silent relicence
  is exactly the kind of thing nobody notices for a year.
* **No key, so no waiting.** ENTSO-E requires an email request and up to
  three working days; this works now.

It is nonetheless a **secondary** source: Fraunhofer ISE re-serves data
originating with ENTSO-E and the Bundesnetzagentur. One more hop that can
break or silently change, and its revision behaviour is not documented.
That is why it is HISTORICAL_CONSERVATIVE rather than PIT-safe, and why
running ENTSO-E alongside it later is worth the registration: two renderings
of one published quantity are the cheapest cross-source check available.

`generated_at` is NOT a publication timestamp
---------------------------------------------
The response carries `generated_at`, which looks like exactly what a
point-in-time archive wants. It is not. Measured 2026-09-28: two calls two
seconds apart returned an identical value (a short server cache), but a call
two minutes later returned the new request time. It is response-generation
time, not the moment the data became public.

So availability here goes through `resolve_available_at()` with the series'
declared conservative lag, like every other source that publishes no release
time. Treating `generated_at` as a release timestamp would have produced an
EXACT_TIMESTAMP claim that is simply false, and strict point-in-time research
would then have trusted a cutoff that never existed.

Resolution versus effective sample
----------------------------------
The data arrive at PT15M -- 96 observations per day. Every one is stored,
because the intraday shape is real information. But 96 quarter-hours of one
day are not 96 independent observations, and `external/evidence.py` counts
distinct observation *dates*, so the effective sample is days. The configured
`frequency` is `daily` for exactly that reason (spec §36).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import (
    FetchedPayload,
    ParseResult,
    resolve_available_at,
    resolve_forecast_available_at,
)
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "energy_charts"
_PARSER_VERSION = "1"
_DEFAULT_BASE_URL = "https://api.energy-charts.info"

#: The licence every v2 response has declared since this adapter was written.
#: Checked on every parse: a changed licence is a governance event, not a
#: detail, and it must not pass unnoticed into an archive.
EXPECTED_LICENCE_PREFIX = "CC BY 4.0"

#: How much history a first fetch takes. 400 days clears the daily
#: readiness profile's 180-observation and 270-day thresholds with room to
#: spare, without pulling the 56 MB that three years costs.
DEFAULT_HISTORY_DAYS = 400

_NATIVE_ID_RE = re.compile(r"^(?P<endpoint>[^|]+)\|(?P<params>[^|]*)\|(?P<series>[^|]+)$")

_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError, AttributeError)


class EnergyChartsError(AdapterError):
    """The payload was not a usable Energy-Charts v2 response."""


@dataclass(frozen=True, slots=True)
class _Target:
    endpoint: str
    params: dict[str, str]
    series_id: str


def build_native_identifier(endpoint: str, params: dict[str, str], series_id: str) -> str:
    """`"<endpoint>|<k=v,k=v>|<series id>"`, the catalog's agreed format."""
    encoded = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{endpoint}|{encoded}|{series_id}"


def parse_native_identifier(raw: str) -> _Target:
    match = _NATIVE_ID_RE.match(raw.strip())
    if match is None:
        raise EnergyChartsError(
            f"energy_charts: native_identifier {raw!r} does not match "
            "'<endpoint>|<k=v,k=v>|<series id>'"
        )
    params: dict[str, str] = {}
    for chunk in match["params"].split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise EnergyChartsError(
                f"energy_charts: parameter {chunk!r} in {raw!r} is not 'key=value'"
            )
        key, value = chunk.split("=", 1)
        params[key.strip()] = value.strip()
    return _Target(endpoint=match["endpoint"].strip(), params=params, series_id=match["series"])


def parse_v2_payload(
    content: bytes,
    spec: SeriesSpec,
    *,
    target: _Target,
    issued_at: datetime | None = None,
) -> tuple[list[ExternalObservation], list[str]]:
    """Turn one v2 response into observations for exactly one series.

    A v2 body looks like::

        {"resolution": "PT15M", "unit": "MW", "license": "CC BY 4.0 ...",
         "series": [{"id": "load", "name": "Load"}],
         "data": [{"timestamp": "2026-09-27T00:00:00+02:00",
                   "values": {"load": 51234.0}}, ...]}

    Everything that is not a clean number at a clean timestamp becomes a
    warning rather than a guess: a null value is skipped (normal for the
    current, still-filling day), an unparseable timestamp is skipped and
    reported, and a missing series id is reported so the caller can mark it
    `missing_series` instead of silently returning nothing.
    """
    warnings: list[str] = []
    try:
        body: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EnergyChartsError(f"{spec.qualified_id}: response is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise EnergyChartsError(f"{spec.qualified_id}: expected a JSON object at the top level")

    licence = str(body.get("license") or body.get("license_info") or "")
    if not licence:
        warnings.append("response declared no licence")
    elif not licence.startswith(EXPECTED_LICENCE_PREFIX):
        warnings.append(
            f"licence changed: expected {EXPECTED_LICENCE_PREFIX!r}, response says {licence!r}"
        )

    available_ids = {str(s.get("id")) for s in body.get("series") or [] if isinstance(s, dict)}
    if available_ids and target.series_id not in available_ids:
        warnings.append(
            f"series {target.series_id!r} not in this response; available: "
            f"{sorted(available_ids)[:12]}"
        )
        return [], warnings

    rows = body.get("data")
    if not isinstance(rows, list):
        raise EnergyChartsError(
            f"{spec.qualified_id}: response has no 'data' list (keys: {sorted(body)[:10]})"
        )

    issued_at = issued_at or datetime.now(UTC)
    unit = str(body.get("unit") or spec.unit)
    resolution = str(body.get("resolution") or "")
    observations: list[ExternalObservation] = []
    bad_timestamps = 0

    for row in rows:
        if not isinstance(row, dict):
            bad_timestamps += 1
            continue
        raw_ts = row.get("timestamp")
        values = row.get("values")
        if not isinstance(values, dict):
            bad_timestamps += 1
            continue
        value = values.get(target.series_id)
        if value is None:
            # Normal: the current day is still filling in, and some series
            # are simply absent for some intervals.
            continue
        try:
            observed = datetime.fromisoformat(str(raw_ts)).astimezone(UTC)
            numeric = float(value)
        except _PARSE_ERROR_TYPES:
            bad_timestamps += 1
            continue

        if spec.forecast_series:
            # A forecast describes a period that has not happened yet, so
            # its availability is anchored to when we obtained it, never to
            # the period it predicts.
            available_at, precision = resolve_forecast_available_at(spec, issued_at)
        else:
            available_at, precision = resolve_available_at(spec, observed.date())
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=numeric,
                unit=unit,
                frequency=spec.frequency,
                source_version=f"{target.endpoint}:{resolution or 'unknown'}",
                observation_time=observed,
                available_at=available_at,
                retrieved_at=datetime.now(UTC),
                source=_SOURCE_ID,
                parser_version=_PARSER_VERSION,
                quality_score=1.0,
                availability_precision=str(precision),
            )
        )

    if bad_timestamps:
        warnings.append(f"{bad_timestamps} row(s) had an unusable timestamp or value shape")
    return observations, warnings


class EnergyChartsAdapter:
    """`ExternalSeriesAdapter` over the Energy-Charts v2 JSON API."""

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        default_history_days: int = DEFAULT_HISTORY_DAYS,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._history_days = default_history_days

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        target = parse_native_identifier(spec.native_identifier)
        params = dict(target.params)
        # The API windows on whole days and defaults to *today only* when no
        # start is given -- 96 points, which would never accumulate into a
        # usable history. So a first fetch takes `default_history_days`, and
        # later ones resume from `since`.
        #
        # The window is bounded rather than "everything": a three-year pull
        # measured 56 MB (96,092 points) on 2026-09-28, and the raw archive
        # is content-addressed, so re-fetching all of it daily would store a
        # fresh copy every day.
        if target.endpoint.endswith("public_power_forecast"):
            # A forecast endpoint serves the current issue only; windowing it
            # backwards returns nothing useful.
            pass
        else:
            # Both bounds are required. Measured 2026-09-28: `start` alone
            # returns that single day (96 points), not the range -- which
            # would silently give the readiness engine one day of history
            # and no error to explain it.
            today = datetime.now(UTC).date()
            start = since or (today - timedelta(days=self._history_days))
            params["start"] = start.isoformat()
            params["end"] = today.isoformat()
        url = f"{self._base_url}/{target.endpoint.lstrip('/')}"
        response = self._http._request("GET", url, params=params)
        # `HttpClient` exposes no public accessor for the raw response, and
        # FetchedPayload needs a genuine status and content type rather than
        # an invented one. Every merged adapter in this package does the same;
        # see the note in adapters/ecb_data.py.
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        public_url = f"{url}?{query}" if query else url
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=spec.series_id,
            url=public_url,
            content=response.content,
            http_status=response.status_code,
            content_type=response.headers.get("content-type", "application/json"),
            retrieved_at=datetime.now(UTC),
            request_fingerprint=f"GET {public_url}",
            headers={"date": response.headers.get("date", "")},
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        target = parse_native_identifier(spec.native_identifier)
        observations, warnings = parse_v2_payload(
            payload.content, spec, target=target, issued_at=payload.retrieved_at
        )
        missing = () if observations else (spec.series_id,)
        return ParseResult(
            observations=observations, warnings=tuple(warnings), missing_series=missing
        )


__all__ = [
    "EXPECTED_LICENCE_PREFIX",
    "EnergyChartsAdapter",
    "EnergyChartsError",
    "build_native_identifier",
    "parse_native_identifier",
    "parse_v2_payload",
]
