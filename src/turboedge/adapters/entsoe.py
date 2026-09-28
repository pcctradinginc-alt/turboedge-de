"""ENTSO-E Transparency Platform adapter, for the External Data Factory (Wave 2).

Implements `turboedge.external.adapter.ExternalSeriesAdapter` against the
ENTSO-E Transparency Platform's RESTful API, ``https://web-api.tp.entsoe.eu/api``,
documented at the official Postman collection
(https://documenter.getpostman.com/view/7009892/2s93JtP3F6) and the
Transparency Platform's own "RESTful API - user guide".

CREDENTIAL: no `ENTSOE_SECURITY_TOKEN` is available in this development
environment, and every ENTSO-E endpoint requires one (query parameter
``securityToken``) for every request. **No live request was made by this
adapter during development** -- everything below is built from the official
Postman collection's own worked examples (each one a real, published
response captured by ENTSO-E, viewed live in a browser on 2026-09-28), not
invented. See "VERIFIED VS DOCUMENTED" below for exactly what that means for
each document type. `_require_security_token` raises `EntsoeCredentialError`
(a typed `AdapterError`) naming the environment variable when it is unset,
mirroring `adapters/fred.py`; the token is baked into the *actual* request's
query parameters (ENTSO-E accepts no other transport for it) but never into
anything this adapter persists -- `FetchedPayload.url` and
`.request_fingerprint` are built only from a separate, credential-free
parameter dict, exactly like `adapters/fred.py`.

VERIFIED VS DOCUMENTED
-----------------------
The task lead verified *live*, without a token, on 2026-09-28: requesting
``https://web-api.tp.entsoe.eu/api`` returns HTTP 401 with an XML
`Acknowledgement_MarketDocument` in namespace
``urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0``. That fact
(base URL, auth-failure shape, that namespace) is taken as verified, not
guessed.

Everything else below -- request parameter names, the `GL_MarketDocument`
namespace, the `Period`/`Point`/`resolution`/`position` shape, the specific
document types -- comes from viewing the official ENTSO-E Postman
collection's own worked examples in a browser on 2026-09-28 (not a live call
made by this adapter, but ENTSO-E's own published, real example
request/response pairs for exactly these endpoints):

  * 6.1.A "Actual Total Load" (`documentType=A65&processType=A16
    &outBiddingZone_Domain=10YCZ-CEPS-----N&periodStart=202303030000
    &periodEnd=202303060000`) -> `GL_MarketDocument` in namespace
    ``urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0``,
    `businessType=A04`, `objectAggregation=A01`,
    `outBiddingZone_Domain.mRID` (same name as the request parameter),
    `Period/resolution=PT60M`, `Point/position` starting at 1.
  * 6.1.B "Day-ahead Total Load Forecast" -- identical shape, `documentType=
    A65&processType=A01`.
  * 16.1.B&C "Actual Generation per Production Type" (`documentType=A75
    &processType=A16&in_Domain=10Y1001A1001A83F&...`) -> the same
    `GL_MarketDocument` namespace, `businessType=A01`,
    `objectAggregation=A08`, but the response element is
    **`inBiddingZone_Domain.mRID`** -- a different name from the
    `in_Domain` request parameter that selected it (the collection's own
    note: "Time series with inBiddingZone_Domain attribute reflects
    Generation values while outBiddingZone_Domain reflects Consumption
    values" -- one `in_Domain`-scoped request can return both generation-
    and consumption-side `TimeSeries`, distinguished only by which of the
    two element names is present). The production type itself is **not** a
    direct `TimeSeries` child: it is nested as `MktPSRType/psrType`
    (confirmed from the same live-viewed example,
    `<MktPSRType><psrType>B01</psrType></MktPSRType>`). `Period/resolution`
    in this example is `PT15M`, confirming the quarter-hourly case.
  * 12.1.G "Cross-Border Physical Flows" (`documentType=A11&out_Domain=...
    &in_Domain=...`) -> a **different** root/namespace,
    `Publication_MarketDocument` in
    ``urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0`` (this exact
    namespace was viewed live on two sibling Market-folder examples, 11.1
    "Implicit Allocations" and 12.1.E "Congestion Income", which share the
    identical `TimeSeries`/`Period`/`Point` shape and use `in_Domain.mRID`/
    `out_Domain.mRID` element names matching their request parameters
    exactly -- unlike the GL family's renamed bidding-zone elements). This
    session could not get the documentation site to render 12.1.G's own
    response body directly (a documentation-site rendering limitation, not
    a data gap); its structural claims for A11 rest on this sibling-example
    pattern plus 12.1.G's own confirmed request-parameter list, root
    element name and "up to 1 year" limit note, not a literally-viewed A11
    response body. This is reported as documented, not verified, below and
    in the test suite.
  * Every endpoint's own page states: "Request limit: Each request may
    cover a period of up to 1 year" -- viewed live on 6.1.A, 6.1.B, 11.1,
    12.1.E and 12.1.G's own pages independently, hence `_MAX_WINDOW_DAYS`
    below and the windowed-fetch logic.
  * The `Acknowledgement_MarketDocument` "no matching data" shape (a
    `Reason/code`=`999` with a `Reason/text` such as "No matching data
    found for Data item ...", and the same code 999 also used for an
    authentication failure with different text) could not be rendered as a
    live official example in this session -- it is reconstructed from
    public, cross-corroborated documentation of ENTSO-E's own behaviour
    (independent sources agree on the shape and on code 999 being reused
    for both cases), consistent with the lead's own live
    401-with-Acknowledgement-document probe. Because the *code* does not
    distinguish the two cases, this adapter's no-data/error split is driven
    entirely by the `Reason/text` content (see `_looks_like_no_data`); the
    code is logged in the warning/error message but never used to decide
    behaviour.

Given the above, `tests/fixtures/external/entsoe/*.xml` are all **hand-built
fixtures**, labelled as such in both this module and the test module's
docstrings: the `GL_MarketDocument` (A65, A75) and `Publication_
MarketDocument` (A11) fixtures transcribe the structure of the live-viewed
Postman examples above (some values/ids trimmed for fixture size, never
reshaped), and the `Acknowledgement_MarketDocument` fixtures follow the
reconstructed shape described above.

PARSING: SDMX/CIM-flavoured XML, parsed with `xml.etree.ElementTree`. The
default namespace differs *by document type* (`GL_MarketDocument` vs
`Publication_MarketDocument` vs `Acknowledgement_MarketDocument` each have
their own, and even within one document family ENTSO-E has changed the
version suffix historically) -- every lookup here matches on **local name**
only (`_local_name` strips the ``{namespace}`` prefix ElementTree leaves on
every tag), never the full namespace URI, so a future version-suffix bump
upstream does not silently stop this parser from finding its own elements.

TIMESTAMP RECONSTRUCTION (the reason this adapter is its own task): a
``<Period>`` gives ``timeInterval/start`` (an absolute UTC instant,
``Z``-suffixed ISO 8601) and a ``resolution`` (an ISO 8601 duration --
``PT15M``, ``PT30M``, ``PT60M`` are the shapes ENTSO-E uses for these
document types). Each ``<Point>`` gives a 1-based ``position``, not a
timestamp. This module computes::

    observation_time = period_start + (position - 1) * resolution

`_parse_iso8601_duration` hand-rolls the ``P[nD][T[nH][nM][nS]]`` grammar
(stdlib only -- this task may not touch `pyproject.toml`, so no `isodate`
dependency was added) for exactly the shapes ENTSO-E uses. This is tested
explicitly in `tests/adapters/test_entsoe.py` against a hand-checked mapping
for a `PT15M` period (position 1 -> the period's own `start`; position 2 ->
`start + 15min`; etc., verified by hand against the fixture's own encoded
values), plus `PT60M`.

MISSING POSITIONS: legal (task brief: "same as previous" in some document
types) and never interpolated or forward-filled (CLAUDE.md rule 29) -- a
`Period` whose point count is less than `(end - start) / resolution` gets a
warning naming the missing position(s) and simply has fewer observations,
never a fabricated one.

ACKNOWLEDGEMENT VS DATA: the root element's local name is checked, not just
the HTTP status (both 401-with-Acknowledgement and 200-with-Acknowledgement
are real ENTSO-E behaviours per the task brief). An `Acknowledgement_
MarketDocument` whose `Reason/text` matches `_looks_like_no_data` (a
publisher "no data for this query" statement) becomes an empty `ParseResult`
with a warning, matching every other adapter's "publisher had nothing to
say" convention; any other `Acknowledgement_MarketDocument` (an auth
failure, a malformed query, ...) raises `EntsoeQueryError` -- an explicit,
task-directed exception to the usual "never raise from parse(), always warn"
rule, because an authentication/query failure disguised as an empty series
is exactly the silent-failure trap this task exists to avoid (mirrors the
AGSI/ALSI HTTP-200-with-error-body trap the sibling Wave 2 adapters were
warned about).

WINDOWED FETCHING: ENTSO-E rejects a `periodStart`/`periodEnd` window longer
than about a year (`_MAX_WINDOW_DAYS = 364`, kept one day under the
documented year to avoid a leap-year edge landing exactly on the
publisher's own boundary). `fetch()` splits a longer request into
consecutive sub-year windows, fetches each in turn through the shared,
rate-limited `HttpClient` (no bespoke throttling is added here), and
concatenates the raw responses byte-for-byte into one `FetchedPayload`. No
artificial envelope or delimiter is invented for this: every ENTSO-E
response is already a complete, self-describing XML document starting with
its own `<?xml ...?>` prolog, so simple back-to-back concatenation is
already unambiguous and losslessly reversible -- `_split_xml_documents`
below splits on the lookahead pattern `(?=<\\?xml)` to recover the original
per-window documents years later, with no invented format to document or
maintain. A single-window fetch (the common case) therefore archives
exactly the one raw publisher response, byte for byte, like every other
adapter on this contract.

AVAILABILITY: ENTSO-E publishes with a short, documented delay and gives no
intraday publication timestamp in the payload (only `createdDateTime`, which
is this *document's* generation time, not a release-time guarantee for a
specific historical period). `available_at` is therefore always produced by
`external.adapter.resolve_available_at()` using the series' declared
`CONSERVATIVE_DATE` lag -- never invented here. `createdDateTime` is used
only for `vintage_time` (per the task brief), and the document's own
`<revisionNumber>` (present and equal to `1` in every example viewed) is
carried into `revision_index` (`revisionNumber - 1`) since it is a real,
publisher-supplied field describing which revision of this response this
is -- distinct from, and not a substitute for, an intraday `available_at`.

DOCUMENT TYPES SUPPORTED
-------------------------
    A65 + processType=A16   actual total load             (verified shape, see above)
    A65 + processType=A01   day-ahead load forecast        (verified shape, see above)
    A75 + processType=A16   actual generation per type     (verified shape, see above)
    A11                      cross-border physical flows    (documented, not directly
                                                              viewed for this exact type --
                                                              see "VERIFIED VS DOCUMENTED")

Bidding-zone EIC code confirmed by the task lead's own live probe:
`GERMANY_LUXEMBOURG_EIC = "10Y1001A1001A82H"`. No other EIC code is baked in
here -- CLAUDE.md rule 1 ("never invent a series identifier"): a caller
wiring up `configs/external_data.yaml` must supply and verify any other
domain code itself.

NATIVE IDENTIFIER FORMAT (fixed by the Wave 2 contract, not invented here):
``"<documentType>|<domain EIC>|<extra=value,extra=value>"``, e.g.
``"A65|10Y1001A1001A82H|processType=A16"``. The single `<domain EIC>` slot
is mapped to whichever request parameter actually selects the domain for
that `documentType` (`outBiddingZone_Domain` for A65, `in_Domain` for A75
and A11 -- see `_REQUEST_DOMAIN_PARAM`); anything else the request needs
(`processType`, a generation `psrType` filter, or A11's second domain via
`out_Domain=<EIC>`) goes in the `extra` segment. A75 in particular *should*
always be given a `psrType` extra: without one, ENTSO-E returns every
production type in one payload, and since one `SeriesSpec.series_id` must
mean one physical quantity, this adapter refuses to guess which returned
`TimeSeries` that series_id means (see `_series_identity_key` below) rather
than silently averaging or picking one.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, time, timedelta
from typing import Final
from urllib.parse import urlencode

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "entsoe"
_PARSER_VERSION = "1"
_DEFAULT_BASE_URL = "https://web-api.tp.entsoe.eu/api"
_TOKEN_ENV_VAR = "ENTSOE_SECURITY_TOKEN"

#: Documented "up to 1 year" request-window limit, kept one day under to
#: avoid landing exactly on the publisher's own boundary (leap years etc.).
_MAX_WINDOW_DAYS: Final = 364

#: Used only when `fetch()` is called with `since=None`. ENTSO-E has no
#: "give me full history" request shape (unlike FRED/ECB) -- every request
#: needs an explicit, bounded periodStart/periodEnd, and this adapter will
#: not invent how far back a caller's backfill should reach. A caller doing
#: a real historical backfill is expected to always pass `since` explicitly;
#: this default only covers a routine "what's new" incremental fetch.
_DEFAULT_LOOKBACK_DAYS: Final = 7

#: EIC code the task lead verified live (2026-09-28) as the one that
#: produced the documented 401/Acknowledgement probe. No other EIC is
#: baked in here -- CLAUDE.md rule 1.
GERMANY_LUXEMBOURG_EIC: Final = "10Y1001A1001A82H"

#: documentType -> the request query-parameter name that carries the
#: domain EIC for that document type (see module docstring).
_REQUEST_DOMAIN_PARAM: Final[dict[str, str]] = {
    "A65": "outBiddingZone_Domain",
    "A75": "in_Domain",
    "A11": "in_Domain",
}
#: documentType -> the *response* element (local name) carrying the same
#: domain EIC. Deliberately a separate table from `_REQUEST_DOMAIN_PARAM`:
#: for A75 the response element (`inBiddingZone_Domain.mRID`) is not the
#: same name as the request parameter (`in_Domain`) -- see module docstring.
_RESPONSE_DOMAIN_ELEMENT: Final[dict[str, str]] = {
    "A65": "outBiddingZone_Domain.mRID",
    "A75": "inBiddingZone_Domain.mRID",
    "A11": "in_Domain.mRID",
}

_ACKNOWLEDGEMENT_ROOT = "Acknowledgement_MarketDocument"
_SUPPORTED_DATA_ROOTS = frozenset({"GL_MarketDocument", "Publication_MarketDocument"})

#: Substrings (checked case-insensitively) that mark an Acknowledgement's
#: `Reason/text` as the documented "no matching data" case rather than a
#: genuine auth/query error -- see module docstring "VERIFIED VS DOCUMENTED".
_NO_DATA_TEXT_MARKERS: Final = ("no matching data", "no data found", "no data available")

_NATIVE_ID_RE = re.compile(r"^(?P<document_type>[^|]+)\|(?P<domain>[^|]*)\|(?P<extra>.*)$")

_ISO8601_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

_XML_DECL_SPLIT_RE = re.compile(rb"(?=<\?xml)")

# Errors expected from a malformed/unexpected upstream payload; anything
# else is a programming error and should propagate rather than be swallowed.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

__all__ = [
    "GERMANY_LUXEMBOURG_EIC",
    "EntsoeAdapter",
    "EntsoeCredentialError",
    "EntsoeQueryError",
    "build_native_identifier",
    "parse_gl_market_document",
    "parse_native_identifier",
]


class EntsoeCredentialError(AdapterError):
    """`ENTSOE_SECURITY_TOKEN` is not set. Never substituted with an
    unauthenticated request or a fabricated token -- every ENTSO-E endpoint
    requires a real one for every request."""


class EntsoeQueryError(AdapterError):
    """ENTSO-E returned an `Acknowledgement_MarketDocument` describing a
    genuine query/auth failure (not the documented "no matching data" case).

    Raised from `parse()` by explicit task-brief direction: silently
    returning zero observations for a bad credential or a malformed query
    would be indistinguishable from a genuinely quiet publisher, which is
    exactly the failure mode this adapter must not have (module docstring
    "ACKNOWLEDGEMENT VS DATA").
    """


def _require_security_token() -> str:
    token = os.environ.get(_TOKEN_ENV_VAR)
    if not token:
        raise EntsoeCredentialError(
            f"{_TOKEN_ENV_VAR} is not set. ENTSO-E requires a security token for every "
            "request; refusing to make an unauthenticated request or substitute a "
            "fabricated token."
        )
    return token


def _public_url(base_url: str, params: dict[str, str]) -> str:
    """Build the archivable request URL from parameters that never include
    the credential -- callers must pass a dict with `securityToken` already
    excluded. The only thing ever written to `FetchedPayload.url` /
    `.request_fingerprint`; the real, credentialed request is sent
    separately (see `EntsoeAdapter._fetch_window`)."""
    return f"{base_url}?{urlencode(params)}"


def parse_native_identifier(raw: str) -> tuple[str, str, dict[str, str]]:
    """Parse a `SeriesSpec.native_identifier` of the fixed Wave 2 entsoe
    format ``"<documentType>|<domain EIC>|<extra=value,extra=value>"``.

    Returns `(document_type, domain_eic, extra)`. Pure string parsing only
    -- it does not validate that `document_type` is one this adapter
    actually supports (callers, e.g. `EntsoeAdapter.fetch`, do that, so a
    parse error and an "unsupported document type" error stay distinct).
    Raises `AdapterError` on a malformed shape, never guesses one.
    """
    match = _NATIVE_ID_RE.match(raw)
    if not match:
        raise AdapterError(
            f"entsoe: native_identifier {raw!r} does not match "
            "'<documentType>|<domain EIC>|<extra=value,extra=value>'"
        )
    document_type = match["document_type"]
    domain_eic = match["domain"]
    extra_str = match["extra"]
    if not document_type:
        raise AdapterError(f"entsoe: native_identifier missing documentType: {raw!r}")

    extra: dict[str, str] = {}
    if extra_str:
        for pair in extra_str.split(","):
            if "=" not in pair:
                raise AdapterError(
                    f"entsoe: malformed extra parameter {pair!r} in native_identifier {raw!r}"
                )
            key, _, value = pair.partition("=")
            if not key:
                raise AdapterError(
                    f"entsoe: malformed extra parameter {pair!r} in native_identifier {raw!r}"
                )
            extra[key] = value
    return document_type, domain_eic, extra


def build_native_identifier(document_type: str, domain_eic: str, **extra: str) -> str:
    """Inverse of `parse_native_identifier`."""
    extra_str = ",".join(f"{key}={value}" for key, value in extra.items())
    return f"{document_type}|{domain_eic}|{extra_str}"


def _local_name(tag: str) -> str:
    """Strip ElementTree's `{namespace}` prefix, if any. Every lookup in
    this module matches on the result -- never the raw `tag` -- so a
    namespace-URI version bump upstream cannot silently break parsing (see
    module docstring "PARSING")."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children_local(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in elem if _local_name(child.tag) == name]


def _child_local(elem: ET.Element | None, name: str) -> ET.Element | None:
    if elem is None:
        return None
    children = _children_local(elem, name)
    return children[0] if children else None


def _child_text_local(elem: ET.Element | None, name: str) -> str | None:
    child = _child_local(elem, name)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _psr_type(time_series: ET.Element) -> str | None:
    return _child_text_local(_child_local(time_series, "MktPSRType"), "psrType")


def _looks_like_no_data(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _NO_DATA_TEXT_MARKERS)


def _parse_iso8601_duration(text: str) -> timedelta:
    """Hand-rolled `P[nD][T[nH][nM][nS]]` parser for exactly the durations
    ENTSO-E uses (`PT15M`, `PT30M`, `PT60M`; `P1D` for daily document
    types). Stdlib only -- see module docstring for why `isodate` was not
    added as a dependency. Raises `ValueError` on anything else, which
    callers turn into a warning."""
    match = _ISO8601_DURATION_RE.match(text.strip())
    if not match or not any(match.groups()):
        raise ValueError(f"unrecognized ISO 8601 duration: {text!r}")
    parts = {key: int(value) for key, value in match.groupdict().items() if value is not None}
    return timedelta(
        days=parts.get("days", 0),
        hours=parts.get("hours", 0),
        minutes=parts.get("minutes", 0),
        seconds=parts.get("seconds", 0),
    )


def _parse_utc_instant(text: str) -> datetime:
    """ENTSO-E's `Z`-suffixed ISO 8601 instants (e.g. `2023-08-15T22:00Z`).
    `datetime.fromisoformat` (Python >= 3.11, this project requires >= 3.12)
    accepts a trailing `Z` directly."""
    return datetime.fromisoformat(text).astimezone(UTC)


def _split_xml_documents(content: bytes) -> list[bytes]:
    """Split a possibly multi-window-concatenated payload back into its
    individual raw XML documents (module docstring "WINDOWED FETCHING").
    A single-window payload (the common case) yields a one-element list
    containing the content unchanged."""
    return [chunk for chunk in _XML_DECL_SPLIT_RE.split(content) if chunk.strip()]


def _series_identity_key(time_series: ET.Element) -> tuple[str | None, ...]:
    """The dimensions that distinguish one physically-distinct series from
    another within a single payload -- currently just the generation
    `psrType`, the only documented case where one request can legitimately
    return more than one distinct quantity (module docstring, A75 note).
    Used only to detect and refuse an ambiguous payload, never to merge or
    pick between series."""
    return (_psr_type(time_series),)


def _time_series_matches(
    time_series: ET.Element,
    *,
    document_type: str,
    domain_eic: str,
    extra: dict[str, str],
) -> bool:
    """Whether `time_series` is the one this `SeriesSpec` actually asked
    for. A domain/EIC mismatch is not itself a warning -- for A75 in
    particular, ENTSO-E may legitimately return neighbouring consumption-
    side `TimeSeries` alongside the requested generation-side ones (module
    docstring), and silently ignoring those is correct, not an oddity."""
    domain_element = _RESPONSE_DOMAIN_ELEMENT[document_type]
    if _child_text_local(time_series, domain_element) != domain_eic:
        return False

    if document_type == "A11":
        expected_out = extra.get("out_Domain")
        if expected_out is not None and _child_text_local(time_series, "out_Domain.mRID") != (
            expected_out
        ):
            return False

    if document_type == "A75":
        expected_psr = extra.get("psrType")
        if expected_psr is not None and _psr_type(time_series) != expected_psr:
            return False

    return True


def _source_version_for(document_type: str) -> str:
    return f"entsoe_{document_type.lower()}"


def _parse_period(
    period: ET.Element,
    *,
    spec: SeriesSpec,
    vintage_time: datetime | None,
    revision_index: int | None,
    retrieved_at: datetime,
    warnings: list[str],
) -> list[ExternalObservation]:
    time_interval = _child_local(period, "timeInterval")
    start_raw = _child_text_local(time_interval, "start")
    end_raw = _child_text_local(time_interval, "end")
    resolution_raw = _child_text_local(period, "resolution")

    if not start_raw or not end_raw or not resolution_raw:
        warnings.append(
            f"{spec.qualified_id}: Period missing timeInterval/start, "
            "timeInterval/end or resolution"
        )
        return []

    try:
        period_start = _parse_utc_instant(start_raw)
        period_end = _parse_utc_instant(end_raw)
        resolution = _parse_iso8601_duration(resolution_raw)
    except _PARSE_ERROR_TYPES as exc:
        warnings.append(
            f"{spec.qualified_id}: unparsable Period timeInterval/resolution "
            f"({start_raw!r}, {end_raw!r}, {resolution_raw!r}): {exc}"
        )
        return []

    resolution_seconds = resolution.total_seconds()
    if resolution_seconds <= 0:
        warnings.append(f"{spec.qualified_id}: non-positive resolution {resolution_raw!r}")
        return []

    total_seconds = (period_end - period_start).total_seconds()
    if total_seconds <= 0:
        warnings.append(f"{spec.qualified_id}: Period end <= start ({start_raw!r} -> {end_raw!r})")
        return []

    expected_points, remainder = divmod(total_seconds, resolution_seconds)
    expected_points = int(expected_points)
    if remainder != 0:
        warnings.append(
            f"{spec.qualified_id}: Period {start_raw!r}-{end_raw!r} ({total_seconds:.0f}s) is "
            f"not an exact multiple of resolution {resolution_raw!r}"
        )

    points_by_position: dict[int, float] = {}
    for point in _children_local(period, "Point"):
        position_raw = _child_text_local(point, "position")
        quantity_raw = _child_text_local(point, "quantity")
        if position_raw is None or quantity_raw is None:
            warnings.append(f"{spec.qualified_id}: Point missing position or quantity")
            continue
        try:
            position = int(position_raw)
            quantity = float(quantity_raw)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(
                f"{spec.qualified_id}: unparsable Point position/quantity "
                f"({position_raw!r}, {quantity_raw!r}): {exc}"
            )
            continue
        if position < 1 or (expected_points and position > expected_points):
            warnings.append(
                f"{spec.qualified_id}: Point position {position} outside expected range "
                f"[1, {expected_points}] for Period {start_raw!r}-{end_raw!r}"
            )
            continue
        if position in points_by_position:
            warnings.append(
                f"{spec.qualified_id}: duplicate Point position {position} in one Period"
            )
            continue
        points_by_position[position] = quantity

    if expected_points:
        missing_positions = sorted(set(range(1, expected_points + 1)) - points_by_position.keys())
        if missing_positions:
            # Legal (module docstring "MISSING POSITIONS") -- never
            # interpolated/forward-filled, just recorded as a gap.
            warnings.append(
                f"{spec.qualified_id}: Period {start_raw!r}-{end_raw!r} missing position(s) "
                f"{missing_positions} -- not interpolated, gap recorded (CLAUDE.md rule 29)"
            )

    source_version = _source_version_for(parse_native_identifier(spec.native_identifier)[0])
    observations: list[ExternalObservation] = []
    for position, quantity in points_by_position.items():
        # The core reconstruction this adapter exists for: a Point carries
        # only a 1-based position, never a timestamp (module docstring
        # "TIMESTAMP RECONSTRUCTION").
        observation_time = period_start + (position - 1) * resolution
        available_at, precision = resolve_available_at(spec, observation_time.date())
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=quantity,
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=source_version,
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
                revision_index=revision_index,
            )
        )
    return observations


def parse_gl_market_document(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
) -> ParseResult:
    """Parse one (possibly multi-window-concatenated) ENTSO-E payload into
    `ExternalObservation`s. Pure and network-free (spec requirement):
    everything needed comes from `content` and `spec`.

    Handles `GL_MarketDocument` (A65, A75), `Publication_MarketDocument`
    (A11) and `Acknowledgement_MarketDocument` (see module docstring
    "ACKNOWLEDGEMENT VS DATA" for the no-data-vs-error split). Any other
    structural surprise -- unparsable XML, an unrecognised root element, a
    `TimeSeries`/`Period` missing required fields -- is reported as a
    warning and the affected chunk/series/period is skipped, never guessed
    at, *except* a genuine Acknowledgement error, which raises
    `EntsoeQueryError` by explicit task direction.
    """
    warnings: list[str] = []

    try:
        document_type, domain_eic, extra = parse_native_identifier(spec.native_identifier)
    except AdapterError as exc:
        return ParseResult(observations=[], warnings=(str(exc),), missing_series=(spec.series_id,))

    if document_type not in _RESPONSE_DOMAIN_ELEMENT:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: unsupported ENTSO-E documentType {document_type!r} "
                f"(supported: {sorted(_RESPONSE_DOMAIN_ELEMENT)})",
            ),
            missing_series=(spec.series_id,),
        )
    if not domain_eic:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: native_identifier is missing the domain EIC",),
            missing_series=(spec.series_id,),
        )

    chunks = _split_xml_documents(content)
    if not chunks:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: empty payload",),
            missing_series=(spec.series_id,),
        )

    observations: list[ExternalObservation] = []
    for chunk in chunks:
        try:
            root = ET.fromstring(chunk)
        except ET.ParseError as exc:
            warnings.append(f"{spec.qualified_id}: unparsable XML document: {exc}")
            continue

        root_name = _local_name(root.tag)

        if root_name == _ACKNOWLEDGEMENT_ROOT:
            reason = _child_local(root, "Reason")
            code = _child_text_local(reason, "code")
            text = _child_text_local(reason, "text") or ""
            if text and _looks_like_no_data(text):
                warnings.append(
                    f"{spec.qualified_id}: ENTSO-E reported no matching data "
                    f"(Reason code={code!r}): {text}"
                )
                continue
            raise EntsoeQueryError(
                f"{spec.qualified_id}: ENTSO-E returned an Acknowledgement_MarketDocument "
                f"error (Reason code={code!r}): {text!r}"
            )

        if root_name not in _SUPPORTED_DATA_ROOTS:
            warnings.append(f"{spec.qualified_id}: unrecognized root element {root_name!r}")
            continue

        # A requested `processType` (e.g. A16=actual vs A01=day-ahead
        # forecast) is only exposed at document level (`process.
        # processType`), not per-TimeSeries -- checked here so a document
        # whose process differs from what this series_id means is never
        # silently folded in (defence in depth: a correctly-built `fetch()`
        # never mixes processTypes across its own windows, but `parse()`
        # must stay correct for any archived payload, not only ones this
        # adapter itself produced).
        expected_process_type = extra.get("processType")
        if expected_process_type is not None:
            actual_process_type = _child_text_local(root, "process.processType")
            if actual_process_type != expected_process_type:
                warnings.append(
                    f"{spec.qualified_id}: document process.processType "
                    f"{actual_process_type!r} does not match requested processType "
                    f"{expected_process_type!r} -- document skipped"
                )
                continue

        created_raw = _child_text_local(root, "createdDateTime")
        vintage_time: datetime | None = None
        if created_raw:
            try:
                vintage_time = _parse_utc_instant(created_raw)
            except _PARSE_ERROR_TYPES as exc:
                warnings.append(
                    f"{spec.qualified_id}: unparsable createdDateTime {created_raw!r}: {exc}"
                )
        else:
            warnings.append(f"{spec.qualified_id}: document missing createdDateTime")

        revision_raw = _child_text_local(root, "revisionNumber")
        revision_index: int | None = None
        if revision_raw is not None:
            try:
                revision_index = int(revision_raw) - 1
            except _PARSE_ERROR_TYPES as exc:
                warnings.append(
                    f"{spec.qualified_id}: unparsable revisionNumber {revision_raw!r}: {exc}"
                )

        all_series = _children_local(root, "TimeSeries")
        if not all_series:
            warnings.append(f"{spec.qualified_id}: {root_name} contained no TimeSeries elements")
            continue

        matched_series = [
            ts
            for ts in all_series
            if _time_series_matches(
                ts, document_type=document_type, domain_eic=domain_eic, extra=extra
            )
        ]
        if not matched_series:
            warnings.append(
                f"{spec.qualified_id}: no TimeSeries in this document matched "
                f"documentType={document_type!r} domain={domain_eic!r} extra={extra!r}"
            )
            continue

        identity_keys = {_series_identity_key(ts) for ts in matched_series}
        if len(identity_keys) > 1:
            warnings.append(
                f"{spec.qualified_id}: payload mixes {len(identity_keys)} distinct series "
                "under one native_identifier (ambiguous psrType) -- provide a disambiguating "
                "'psrType' in native_identifier extras; no observations emitted for this "
                "document rather than guessing which series_id means"
            )
            continue

        for time_series in matched_series:
            for period in _children_local(time_series, "Period"):
                observations.extend(
                    _parse_period(
                        period,
                        spec=spec,
                        vintage_time=vintage_time,
                        revision_index=revision_index,
                        retrieved_at=retrieved_at,
                        warnings=warnings,
                    )
                )

    missing_series = () if observations else (spec.series_id,)
    return ParseResult(
        observations=observations, warnings=tuple(warnings), missing_series=missing_series
    )


def _window_bounds(
    start: datetime, end: datetime, *, max_days: int
) -> list[tuple[datetime, datetime]]:
    if start >= end:
        raise AdapterError(f"entsoe: window start {start} must be before end {end}")
    max_delta = timedelta(days=max_days)
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + max_delta, end)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows


class EntsoeAdapter:
    """Fetches and parses ENTSO-E Transparency Platform series.

    One instance handles every ENTSO-E series a `SeriesSpec` can name --
    `spec.native_identifier` carries the documentType, domain EIC and any
    extra request parameters (see module docstring "NATIVE IDENTIFIER
    FORMAT"); `spec.series_id` is TurboEdge's own id, used only for the
    emitted `ExternalObservation`s.

    Rate limiting: `web-api.tp.entsoe.eu` serves no `robots.txt` (per the
    Wave 2 contract), so this adapter relies on the shared `HttpClient`'s
    per-host rate limiter -- the caller wiring this adapter up must
    construct that `HttpClient` with `min_interval_s=1.0` (1 request/second,
    the contract's documented default for a host with no declared
    `Crawl-delay`). This adapter does not set that itself: `HttpClient` is
    shared across every adapter's requests to a host and its rate-limit
    policy is the caller's configuration, not a per-adapter constant (see
    `adapters/base.HttpClient`).
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
        token = _require_security_token()
        document_type, domain_eic, extra = parse_native_identifier(spec.native_identifier)

        domain_param = _REQUEST_DOMAIN_PARAM.get(document_type)
        if domain_param is None:
            raise AdapterError(
                f"entsoe: unsupported documentType {document_type!r} "
                f"(supported: {sorted(_REQUEST_DOMAIN_PARAM)})"
            )
        if not domain_eic:
            raise AdapterError(
                f"entsoe: native_identifier {spec.native_identifier!r} is missing the "
                "domain EIC segment"
            )

        until = datetime.now(UTC)
        start = (
            datetime.combine(since, time(0, 0), tzinfo=UTC)
            if since is not None
            else until - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        )
        windows = _window_bounds(start, until, max_days=_MAX_WINDOW_DAYS)

        raw_chunks: list[bytes] = []
        public_urls: list[str] = []
        last_status = 0
        last_content_type = ""
        for window_start, window_end in windows:
            public_params: dict[str, str] = {
                "documentType": document_type,
                domain_param: domain_eic,
                "periodStart": window_start.astimezone(UTC).strftime("%Y%m%d%H%M"),
                "periodEnd": window_end.astimezone(UTC).strftime("%Y%m%d%H%M"),
                **extra,
            }
            request_params = {**public_params, "securityToken": token}

            # `_request` (shared by every adapter on this contract) is the
            # one place HttpClient applies retry, per-host rate limiting and
            # the honest User-Agent, and hands back the real `httpx.Response`
            # so `http_status`/`content_type` are recorded rather than
            # invented. Not used via `.request.url` (unlike ecb_data.py)
            # because httpx has filled that in with the real, credentialed
            # query string -- `url`/`request_fingerprint` are built only
            # from `public_params` below (mirrors adapters/fred.py).
            response = self._http._request("GET", self._base_url, params=request_params)
            raw_chunks.append(response.content)
            public_urls.append(_public_url(self._base_url, public_params))
            last_status = response.status_code
            last_content_type = response.headers.get("content-type", "")

        retrieved_at = datetime.now(UTC)
        # No delimiter needed: each raw response is already a complete,
        # self-describing XML document (module docstring "WINDOWED
        # FETCHING") -- plain concatenation is already losslessly
        # reversible via `_split_xml_documents`.
        content = b"".join(raw_chunks)
        dataset = spec.series_id if len(windows) == 1 else f"{spec.series_id}:{len(windows)}windows"
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=dataset,
            url=public_urls[0],
            content=content,
            http_status=last_status,
            content_type=last_content_type or "application/xml",
            retrieved_at=retrieved_at,
            request_fingerprint="; ".join(f"GET {url}" for url in public_urls),
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_gl_market_document(payload.content, spec, retrieved_at=payload.retrieved_at)
