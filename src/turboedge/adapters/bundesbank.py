"""Deutsche Bundesbank SDMX-CSV adapter, for the External Data Factory.

Implements `turboedge.external.adapter.ExternalSeriesAdapter` against
``https://api.statistiken.bundesbank.de/rest/data/{flow}/{key}`` -- the
Bundesbank's public SDMX 2.1 REST time-series API.

CRITICAL AND VERIFIED LIVE: this endpoint answers HTTP 406 to a plain
`GET` (its default representation is not one this adapter can use); it must
be asked for ``Accept: text/csv`` explicitly. With that header it returns a
semicolon-separated CSV with a UTF-8 BOM. There is no SDMX-JSON support on
this host -- `Accept: application/json` also 406s.

VERIFIED LIVE (2026-09-28, HTTP 200 with real data, one request >= 1.2s
apart, honest UA):

    BBEX3/D.USD.EUR.BB.AC.000                  ECB euro reference rate, USD per EUR
                                                (given; re-verified here)
    BBDE1/M.DE.Y.BAA1.A2P300000.G.C.I21.A       German industrial production index,
                                                Industry (B+C), 2021=100, calendar-
                                                and seasonally adjusted
    BBDE1/M.DE.Y.AEA1.A2P300000.F.C.I21.A       German manufacturing new orders index,
                                                Industry (B+C), constant prices,
                                                2021=100, calendar- and seasonally
                                                adjusted
    BBIN1/M.DE.BBK.BBKBAS2.EUR.ME               German statutory base rate
                                                ("Basiszinssatz" gem. Section 247
                                                BGB), end-of-month, % p.a.

DISCOVERY METHOD (not guessed): the three `BBDE1`/`BBIN1` keys were not
known in advance. They were found by (1) listing every registered dataflow
via `GET /metadata/dataflow/BBK` (`Accept: application/xml` -- the only
representation this metadata endpoint accepts; `application/json` and
`text/csv` both 406), (2) fetching the matching `DataStructure` via
`GET /metadata/datastructure/BBK/<dsd id>?references=children` to read its
dimension order and each dimension's codelist (concept/classification
codes such as `BAA1` "Output", `AEA1` "Orders received", `A2P300000`
"Mining, quarrying and manufacturing sector (B+C)", `BBKBAS2` "Base rate as
per Civil Code"), and (3) issuing a wildcard `GET .../data/{flow}/{partial
key with blank positions}` to see which combination of the remaining
dimensions the publisher actually has data under, before requesting that
exact key on its own and confirming HTTP 200 with real observations. A
plausible-looking key that returned no rows this way
(`BBIN1/D.DE.BBK.BBKBAS1.EUR._Z`, guessed before the wildcard step) was
discarded rather than reported as verified.

ROBOTS.TXT: `api.statistiken.bundesbank.de/robots.txt` returns HTTP 404
with the API's own "no route matches" JSON body -- this host serves no
robots.txt at all, which RFC 9309 treats as no restriction (the same
reasoning already recorded in `adapters/cboe.py` for a 403 on that
endpoint's robots.txt). No access control is bypassed: no auth, no
CAPTCHA, no signature reconstruction -- only the documented `Accept`
negotiation above.

PARSING: `TIME_PERIOD` and `OBS_VALUE`, plus `BBK_ID`/`BBK_UNIT`, sit at a
*fixed* position in the column layout across every dataflow tested here
even though the *dimension* columns before them (and the commentary/diff
columns after `BBK_TITLE`) differ per dataflow (`BBEX3`'s header has no
`BBK_STD_AREA`; `BBDE1`'s and `BBIN1`'s do, with different dimension names
again). The parser therefore looks these four columns up by name via
`csv.DictReader`, never by position, and treats their absence as schema
drift (a warning), not a crash. `OBS_VALUE` of `.` (or blank) is the
publisher's own missing-observation marker -- confirmed on real French/US-
holiday gaps in `BBEX3` -- and is skipped without a warning, per the
adapter contract. A numeral containing a comma is treated as German
decimal notation (comma decimal separator, optional `.` thousands
separator stripped first); every value actually observed live used a
plain `.` decimal point, so both are exercised defensively rather than
assumed.

VINTAGES: like the ECB API, this endpoint exposes no per-row revision or
vintage marker -- every value returned is the currently known print for its
period. There is also no in-body "prepared" timestamp analogous to ECB's
SDMX-JSON `header.prepared`, so `fetch()` additionally records the HTTP
response's own `Date` header (a real, observed fact about the response,
never invented) into `FetchedPayload.headers`; `parse()` uses it, when
present, as `vintage_time` for every observation in that payload -- "this
is what querying at this moment returned" -- and leaves it `None`
otherwise. `revision_index` is always `None` (no revision sequence is
knowable) and `source_release_time` is always `None` (the moment a
specific historical print was first released is not recoverable from this
API).

AVAILABILITY: every series here is date-only (monthly or daily figures,
never an intraday timestamp), so `SeriesSpec` declares `CONSERVATIVE_DATE`
with an explicit lag and `available_at` always comes from
`external.adapter.resolve_available_at()` -- never invented here.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime, time
from email.utils import parsedate_to_datetime

from turboedge.adapters.base import HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "bundesbank"
_PARSER_VERSION = "1"
_DEFAULT_BASE_URL = "https://api.statistiken.bundesbank.de/rest"
_ACCEPT_HEADER = "text/csv"

#: Columns this parser relies on by name. Verified present, at the same
#: relative position after the (per-dataflow) dimension columns, across
#: BBEX3, BBDE1 and BBIN1. Their absence means the CSV contract changed.
_REQUIRED_COLUMNS: tuple[str, ...] = ("TIME_PERIOD", "OBS_VALUE", "BBK_ID", "BBK_UNIT")

#: The publisher's own missing-observation markers (confirmed live: a
#: bank holiday or weekend in a daily series prints `.` in OBS_VALUE).
_MISSING_VALUE_MARKERS = frozenset({"", "."})

# Errors expected from a malformed/unexpected upstream payload; anything else
# is a programming error and should propagate rather than be swallowed.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

__all__ = [
    "BundesbankAdapter",
    "parse_bundesbank_csv",
    "split_native_identifier",
]


def split_native_identifier(native_identifier: str) -> tuple[str, str]:
    """Split a `SeriesSpec.native_identifier` of the form ``"FLOW/KEY"``.

    Splits on the *first* ``/`` only, mirroring `adapters/ecb_data.py`: the
    key is a dot-separated dimension tuple and never contains a further
    ``/``.
    """
    if "/" not in native_identifier:
        raise ValueError(
            f"Bundesbank native_identifier must be 'FLOW/KEY', got {native_identifier!r}"
        )
    flow, key = native_identifier.split("/", 1)
    if not flow or not key:
        raise ValueError(f"Bundesbank native_identifier missing flow or key: {native_identifier!r}")
    return flow, key


def _source_version_for(spec: SeriesSpec) -> str:
    flow, _ = split_native_identifier(spec.native_identifier)
    return f"bundesbank_sdmx_csv_{flow.lower()}"


def _parse_time_period(label: str) -> date:
    """Parse a Bundesbank `TIME_PERIOD` cell into a calendar date.

    Only the shapes actually observed across the verified series are
    handled (daily `YYYY-MM-DD`, monthly `YYYY-MM`, annual `YYYY`); an
    unrecognised shape raises `ValueError`, turned into a warning by the
    caller rather than crashing the whole payload.
    """
    text = label.strip()
    if len(text) == 10:
        return date.fromisoformat(text)
    if len(text) == 7:
        year_str, month_str = text.split("-")
        return date(int(year_str), int(month_str), 1)
    if len(text) == 4:
        return date(int(text), 1, 1)
    raise ValueError(f"unrecognized Bundesbank TIME_PERIOD label: {label!r}")


def _parse_obs_value(raw: str) -> float:
    """Parse an `OBS_VALUE` cell, tolerating German decimal-comma notation.

    Every value observed live from this API used a plain `.` decimal
    point, but the adapter contract calls for handling comma decimals
    defensively rather than assuming they never occur. A comma is treated
    as the decimal separator (with any `.` thousands separator stripped
    first); its absence means the value is already plain-decimal.
    """
    text = raw.strip()
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def _vintage_time_from_headers(headers: dict[str, str]) -> datetime | None:
    """Best-effort vintage timestamp from the HTTP response's own `Date`.

    This API has no in-body "prepared" timestamp the way ECB's SDMX-JSON
    does, so the one genuine fact available about *when this response was
    produced* is the HTTP `Date` header `fetch()` captured. Never invented:
    absent or unparsable simply means `None`.
    """
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


def parse_bundesbank_csv(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
    headers: dict[str, str] | None = None,
) -> ParseResult:
    """Parse one Bundesbank SDMX-CSV payload into `ExternalObservation`s.

    Pure and network-free: everything needed comes from `content`, `spec`
    and the already-captured `headers` (never re-fetched). Any structural
    surprise -- undecodable bytes, a missing required column, an
    unrecognised time-period label, a non-numeric value -- is reported as a
    warning and the observation (or payload) is skipped, never guessed at.
    A missing `OBS_VALUE` (`.` or blank) is the publisher's normal way of
    saying "no observation for this period" and is skipped silently.
    """
    warnings: list[str] = []

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: payload is not valid UTF-8 CSV: {exc}",),
            missing_series=(spec.series_id,),
        )

    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    fieldnames = reader.fieldnames or []
    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
    if missing_columns:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: unexpected Bundesbank CSV column set, missing "
                f"{missing_columns!r} (header={fieldnames!r})",
            ),
            missing_series=(spec.series_id,),
        )

    rows = list(reader)
    if not rows:
        return ParseResult(observations=[], warnings=(), missing_series=(spec.series_id,))

    vintage_time = _vintage_time_from_headers(headers or {})
    source_version = _source_version_for(spec)

    observations: list[ExternalObservation] = []
    for row in rows:
        raw_period = (row.get("TIME_PERIOD") or "").strip()
        raw_value = (row.get("OBS_VALUE") or "").strip()

        if not raw_period:
            warnings.append(f"{spec.qualified_id}: row with empty TIME_PERIOD skipped")
            continue

        if raw_value in _MISSING_VALUE_MARKERS:
            # The publisher's own "no observation this period" marker.
            # Normal, not a warning (adapter contract).
            continue

        try:
            observation_day = _parse_time_period(raw_period)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(f"{spec.qualified_id}: malformed TIME_PERIOD {raw_period!r}: {exc}")
            continue

        try:
            value = _parse_obs_value(raw_value)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(
                f"{spec.qualified_id}: non-numeric OBS_VALUE {raw_value!r} at {raw_period!r}: {exc}"
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


class BundesbankAdapter:
    """Fetches and parses one Bundesbank dataflow/key series at a time.

    One instance handles every Bundesbank series a `SeriesSpec` can name:
    the flow and key both come from `spec.native_identifier` ("FLOW/KEY"),
    not from adapter configuration.
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
        #: Applied only when `since` is not given; `since` always takes
        #: priority and uses `startPeriod` instead.
        self._last_n_observations = last_n_observations

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        flow, key = split_native_identifier(spec.native_identifier)
        url = f"{self._base_url}/data/{flow}/{key}"
        params: dict[str, str] = {}
        if since is not None:
            params["startPeriod"] = since.isoformat()
        elif self._last_n_observations is not None:
            params["lastNObservations"] = str(self._last_n_observations)

        retrieved_at = datetime.now(UTC)
        # This host answers HTTP 406 without an explicit `Accept: text/csv`
        # (verified live -- it does not speak SDMX-JSON at all). HttpClient's
        # public accessors don't expose the raw `httpx.Response` that
        # FetchedPayload needs for a genuine http_status/content_type/date,
        # so `_request` (HttpClient's own shared retry/rate-limit/error path)
        # is reused here rather than duplicated or faked -- see the matching
        # comment in `adapters/ecb_data.py`.
        response = self._http._request(
            "GET", url, params=params or None, headers={"Accept": _ACCEPT_HEADER}
        )
        request_url = str(response.request.url)
        response_date = response.headers.get("date")
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=spec.series_id,
            url=request_url,
            content=response.content,
            http_status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            retrieved_at=retrieved_at,
            request_fingerprint=f"GET {request_url}",
            headers={"date": response_date} if response_date else {},
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_bundesbank_csv(
            payload.content,
            spec,
            retrieved_at=payload.retrieved_at,
            headers=payload.headers,
        )
