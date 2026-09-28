"""General ECB Data Portal adapter (SDMX-JSON), for the External Data Factory.

Implements `turboedge.external.adapter.ExternalSeriesAdapter` against
``https://data-api.ecb.europa.eu/service/data/{flowRef}/{key}`` -- the ECB's
public SDMX 2.1 REST web service, documented for programmatic use at
https://data.ecb.europa.eu/help/api/overview. This module is deliberately
generic over *any* ECB dataflow/key pair a `SeriesSpec` names; it does not
special-case a single series the way `adapters/ecb.py` does.

`adapters/ecb.py` already exists and fetches the euro short-term rate (EST)
for financing-spread pricing (`pricing/financing.py`). That module is not
touched, modified or absorbed here: it is a narrow, pricing-critical path
with its own fallback-on-failure behaviour, and this module is a separate,
general-purpose series adapter for the point-in-time research pipeline. The
two happen to be able to fetch the same underlying EST series -- that is a
coincidence of the source, not a reason to merge the adapters.

VERIFIED LIVE (2026-09-28, HTTP 200, SDMX-JSON body, one request >= 1.2s
apart, honest UA) -- see the module's adapter report for full detail:

    FM/B.U2.EUR.4F.KR.MRR_FR.LEV               main refinancing rate
    EST/B.EU000A2X2A25.WT                      euro short-term rate (EUR STR)
    BSI/M.U2.Y.V.M30.X.1.U2.2300.Z01.E         M3 monetary aggregate
    EXR/D.USD.EUR.SP00.A                       USD/EUR reference rate
    YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y        10y AAA euro area spot yield
    MIR/M.U2.B.A2A.A.R.A.2240.EUR.N            bank lending rate to households

`MIR/M.U2.B.A2A.AM.R.A.2240.EUR.N` (note the extra `M` in the fourth
position) returns HTTP 404 and must not be used -- confirmed live, not
guessed.

ROBOTS.TXT: `data-api.ecb.europa.eu/robots.txt` 301-redirects to
`data.ecb.europa.eu/robots.txt` (a *different* host: the Drupal-based
ECB statistics portal, not the SDMX API). That file's `User-agent: *`
block disallows only portal paths (`/admin/`, `/core/`, `/search/`,
login/logout, etc.) -- nothing under `/service/`, which does not exist on
that host at all. It separately disallows a list of named bots (generic
HTTP-library user agents, AI crawlers) by literal product token; this
adapter's honest, non-generic User-Agent (set by `HttpClient`) matches none
of those tokens. `/service/data/...` is the documented public SDMX API
surface, not scraped portal content, so this is read as "no restriction on
the API we are calling," analogous to the reasoning already recorded in
`adapters/cboe.py` for a robots.txt that does not address the endpoint in
use. No access control is bypassed: no auth, no CAPTCHA, no signature
reconstruction.

PARSING: SDMX-JSON nests one series under `dataSets[0].series[<key>]`,
whose `observations` map a positional index (as a string) to
`[value, ...attributes]`. The index maps into
`structure.dimensions.observation[0].values[index].id`, which carries the
time period label -- `YYYY-MM-DD` (daily), `YYYY-MM` (monthly) or `YYYY`
(annual) for the series verified here. A `null` value is a period the
publisher has not (yet) published a print for; that is normal and is
skipped without a warning, per the adapter contract. An unrecognised label
shape, missing top-level keys, or a non-JSON body *are* warnings: they mean
the schema moved, not that a business day was quiet.

VINTAGES: this API does not expose historical vintages (no `?vintage=` or
equivalent) -- every value returned is the *currently* known (i.e. latest
revised) print for its period, regardless of how old that period is. Every
observation therefore gets `vintage_time` set to the payload's
`header.prepared` timestamp (when *this* query was answered) and
`revision_index=None` (no revision sequence is knowable). `source_release_time`
is left `None`: `header.prepared` is when the API produced this response,
not when the publisher originally released a given historical print, and
setting it would misrepresent a decades-old observation as freshly
released.

AVAILABILITY: every configured series here declares `CONSERVATIVE_DATE`
availability with an explicit lag (`SeriesSpec.conservative_release_lag_hours`)
because this API is date-only -- it never hands back an intraday
publication timestamp. `available_at` is therefore always produced by
`external.adapter.resolve_available_at()`, never invented here.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from typing import Any

from turboedge.adapters.base import HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "ecb"
_PARSER_VERSION = "1"
_DEFAULT_BASE_URL = "https://data-api.ecb.europa.eu"
_ACCEPT_HEADER = "application/vnd.sdmx.data+json;version=1.0.0, application/json;q=0.9"

# Errors expected from a malformed/unexpected upstream payload; anything else
# is a programming error and should propagate rather than be swallowed.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

__all__ = [
    "EcbDataAdapter",
    "parse_sdmx_json",
    "split_native_identifier",
]


def split_native_identifier(native_identifier: str) -> tuple[str, str]:
    """Split a `SeriesSpec.native_identifier` of the form ``"FLOW/KEY"``.

    Splits on the *first* ``/`` only: the key itself is a dot-separated
    dimension tuple (e.g. ``B.U2.EUR.4F.KR.MRR_FR.LEV``) and never contains
    a further ``/``, but treating this as "split once" rather than "split on
    every /" keeps that assumption explicit instead of accidental.
    """
    if "/" not in native_identifier:
        raise ValueError(f"ECB native_identifier must be 'FLOW/KEY', got {native_identifier!r}")
    flow_ref, key = native_identifier.split("/", 1)
    if not flow_ref or not key:
        raise ValueError(f"ECB native_identifier missing flow or key: {native_identifier!r}")
    return flow_ref, key


def _source_version_for(spec: SeriesSpec) -> str:
    flow_ref, _ = split_native_identifier(spec.native_identifier)
    return f"ecb_sdmx_jsondata_{flow_ref.lower()}"


def _parse_time_period(label: str) -> date:
    """Parse an SDMX `TIME_PERIOD` dimension value into a calendar date.

    Only the shapes actually observed across the verified series are
    handled (daily, monthly, annual); an unrecognised shape raises
    `ValueError`, which callers turn into a warning rather than a crash --
    a new label shape is a contract change, not something to guess about.
    """
    if len(label) == 10:
        return date.fromisoformat(label)
    if len(label) == 7:
        year_str, month_str = label.split("-")
        return date(int(year_str), int(month_str), 1)
    if len(label) == 4:
        return date(int(label), 1, 1)
    raise ValueError(f"unrecognized ECB TIME_PERIOD label: {label!r}")


def parse_sdmx_json(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
) -> ParseResult:
    """Parse one ECB SDMX-JSON payload into `ExternalObservation`s.

    Pure and network-free (spec requirement): everything needed to
    reconstruct the series comes from `content` and `spec`. Any structural
    surprise -- not valid JSON, missing `dataSets`/`structure`/`header`, an
    empty or absent series, an unrecognised time-period label, a
    non-numeric value -- is reported as a warning and the observation (or
    payload) is skipped, never guessed at.
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
            warnings=(f"{spec.qualified_id}: ECB response is not a JSON object",),
            missing_series=(spec.series_id,),
        )

    try:
        data_sets = raw["dataSets"]
        obs_dim_values = raw["structure"]["dimensions"]["observation"][0]["values"]
        header_prepared = raw["header"]["prepared"]
    except _PARSE_ERROR_TYPES as exc:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: unexpected ECB SDMX-JSON structure, missing {exc}",),
            missing_series=(spec.series_id,),
        )

    vintage_time: datetime | None
    try:
        vintage_time = datetime.fromisoformat(header_prepared).astimezone(UTC)
    except _PARSE_ERROR_TYPES as exc:
        warnings.append(
            f"{spec.qualified_id}: unparsable header.prepared {header_prepared!r}: {exc}"
        )
        vintage_time = None

    if not data_sets or not isinstance(data_sets[0], dict):
        return ParseResult(
            observations=[],
            warnings=(*warnings, f"{spec.qualified_id}: ECB response has no dataSets[0]"),
            missing_series=(spec.series_id,),
        )

    series_map = data_sets[0].get("series")
    if not series_map:
        return ParseResult(
            observations=[],
            warnings=tuple(warnings),
            missing_series=(spec.series_id,),
        )

    # A single-key query (one flow, one fully-specified key) yields exactly
    # one series entry. Its dimension-index key (e.g. "0:0:0:0:0") depends
    # on the flow's own dimension count and is not something a caller
    # should have to predict, so the first (only) entry is taken positionally.
    series = next(iter(series_map.values()))
    observations_raw = series.get("observations", {}) if isinstance(series, dict) else {}

    observations: list[ExternalObservation] = []
    source_version = _source_version_for(spec)
    for index_str, values in observations_raw.items():
        try:
            index = int(index_str)
            period_label = obs_dim_values[index]["id"]
            observation_day = _parse_time_period(period_label)
            raw_value = values[0]
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(
                f"{spec.qualified_id}: malformed observation at index {index_str!r}: {exc}"
            )
            continue

        if raw_value is None:
            # A period the publisher has not printed a value for yet.
            # Normal, not a warning (adapter contract).
            continue

        try:
            value = float(raw_value)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(
                f"{spec.qualified_id}: non-numeric value at index {index_str!r}: "
                f"{raw_value!r} ({exc})"
            )
            continue

        available_at, precision = resolve_available_at(spec, observation_day)
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=value,
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=source_version,
                observation_time=datetime.combine(observation_day, time(0, 0), tzinfo=UTC),
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


class EcbDataAdapter:
    """Fetches and parses one ECB dataflow/key series at a time.

    One instance handles every ECB series a `SeriesSpec` can name: the
    flow and key both come from `spec.native_identifier` ("FLOW/KEY"), not
    from adapter configuration, so adding a new verified ECB series is a
    catalog change, not a code change.
    """

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        last_n_observations: int | None = None,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        #: Applied only when `since` is not given (a full/initial fetch);
        #: `since` always takes priority and uses `startPeriod` instead, per
        #: the adapter contract.
        self._last_n_observations = last_n_observations

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        flow_ref, key = split_native_identifier(spec.native_identifier)
        url = f"{self._base_url}/service/data/{flow_ref}/{key}"
        params: dict[str, str] = {"format": "jsondata"}
        if since is not None:
            params["startPeriod"] = since.isoformat()
        elif self._last_n_observations is not None:
            params["lastNObservations"] = str(self._last_n_observations)

        retrieved_at = datetime.now(UTC)
        # HttpClient's public accessors (get_json/get_text/get_bytes) return
        # only the decoded body, but FetchedPayload must record the *actual*
        # HTTP status and content-type the publisher sent -- never invented,
        # per the "archived payload proves what was parsed" contract.
        # `_request` is the one place HttpClient already applies the shared
        # retry policy, per-host rate limiting and honest User-Agent and
        # returns the raw `httpx.Response`; reused here so this adapter is
        # throttled and retried exactly like every other one, rather than
        # re-implementing that policy or fabricating a status code.
        response = self._http._request(
            "GET", url, params=params, headers={"Accept": _ACCEPT_HEADER}
        )
        request_url = str(response.request.url)
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=spec.series_id,
            url=request_url,
            content=response.content,
            http_status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            retrieved_at=retrieved_at,
            request_fingerprint=f"GET {request_url}",
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_sdmx_json(payload.content, spec, retrieved_at=payload.retrieved_at)
