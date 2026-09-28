"""Kiel Trade Indicator (KTI) adapter, for the External Data Factory (Wave 2).

Implements `turboedge.external.adapter.ExternalSeriesAdapter` against the
Kiel Institute for the World Economy's public CSV publication endpoint,
``https://trade.kielinstitut.de/KTI/<file>.csv`` -- plain, unauthenticated
HTTPS, no key, no query parameters, one file per request.

HOST: the brief's original host, ``trade.ifw-kiel.de``, 301-redirects to
``trade.kielinstitut.de`` (the institute's post-rename domain). This module
requests the latter directly rather than relying on the redirect, per the
project's usual practice of never depending on an upstream redirect chain
when the final host is already known.

VERIFIED LIVE 2026-09-28 (HTTP 200, ``content-type: text/csv``, honest UA,
>= 1 request/second, confirmed against ``https://trade.kielinstitut.de/
robots.txt`` returning HTTP 404 -- no robots.txt at all, so the project's
general 1 req/s default applies, not a declared crawl-delay):

    plot_ships_red_sea.csv                 timestamp,n            last 2026-09-25 = 46
    plot_ships_cape_good_hope.csv          timestamp,n            last 2026-09-24 = 29
    plot_ships_panama_canal.csv            timestamp,n            last 2025-01-24 = 47   STALE
    plot_portcalls_china.csv               timestamp,portname,n   last 2025-01-12
    plot_freight_rates_china_northern_europe_global.csv
                                            name,code,timestamp,rate  last 2025-01-27 STALE

Confirmed HTTP 404 (also live, 2026-09-28): ``plot_draft_global.csv``.

``plot_ships_panama_canal.csv``, ``plot_portcalls_china.csv`` and the
freight-rates file are frozen at their January 2025 values today -- the
publisher has stopped updating them, while ``plot_ships_red_sea.csv`` and
``plot_ships_cape_good_hope.csv`` are current as of yesterday. This module
does **not** hard-code that distinction anywhere: every file is parsed by
the same generic logic below, and it is the catalog's `SeriesSpec.enabled`
flag plus the readiness/staleness engine's job -- not this adapter's -- to
decide whether a frozen file should still be ingested. A file simply
returning the same last row on every fetch is not this adapter's problem to
solve; silently special-casing "these five files are dead" here would be
exactly the kind of guess CLAUDE.md rule 1 forbids (today's freeze is not
guaranteed permanent, and a future re-verification could find the publisher
resumed updates).

THREE CSV SHAPES, ONE GENERIC PARSER
-------------------------------------
Every file observed live uses one of three column layouts:

  A. ``timestamp,n``                     (plot_ships_*)
  B. ``timestamp,portname,n``            (plot_portcalls_*)
  C. ``name,code,timestamp,rate``        (plot_freight_rates_*)

Shape B and C both multiplex several independent series (one per port, or
one per named route) into a single file; shape A carries exactly one
series per file. `SeriesSpec.native_identifier` (format below) names which
column holds the value to extract and, for a multiplexed file, which
name/portname row-filter selects the one series wanted. The parser never
guesses which entity a multiplexed file's un-filtered rows belong to: a
file with a name/portname column but no filter in `native_identifier` is
schema drift from this adapter's point of view (ambiguous, ``missing_series``),
not something the code resolves for you. `plot_draft_global.csv` and
`plot_speed_global.csv` (both confirmed HTTP 404 above) are simply never
requested by a correctly configured `SeriesSpec`; requesting them anyway
raises `AdapterHttpError` from `fetch()` the same way any other 404 would
(the shared `HttpClient._request` retry/rate-limit path applies
`response.raise_for_status()` uniformly across every adapter on this
contract -- no bespoke 404 handling is added here).

`native_identifier` FORMAT (fixed, per the Wave 2 contract; not invented here)
-------------------------------------------------------------------------------
    "<csv basename without .csv>|<value column>|<optional name filter>"

    "plot_ships_red_sea|n|"            shape A, no filter needed
    "plot_portcalls_china|n|Shenzhen"  shape B, filtered to the Shenzhen rows
    "plot_freight_rates_china_northern_europe_global|rate|China to Northern Europe"
                                        shape C, filtered to that named route

The third segment is empty (trailing ``|`` with nothing after it) for a
single-series file; it is the exact, case-sensitive value of whichever
name-ish column (``portname`` or ``name``) the file declares, for a
multiplexed one.

POINT-IN-TIME HONESTY
----------------------
The publisher exposes no publication timestamp anywhere -- not in the CSV
body (`timestamp` is the *observation* date, not a release date), not in a
dataset-level field, and (per this module's explicit instruction, unlike
`adapters/bundesbank.py`) not even via the HTTP response's own `Date`
header: `vintage_time` and `source_release_time` are always `None` here,
and `revision_index` is always `None` (no revision sequence is knowable --
there is no vintage/history endpoint at all). Every `SeriesSpec` registered
for this source is date-only, so `available_at` always comes from
`external.adapter.resolve_available_at()` -- never invented here.

AMBIGUITY, NEVER RESOLVED OPTIMISTICALLY
------------------------------------------
If the same (filtered) series reports two different values for the same
`timestamp` -- which should never happen for real data, but is exactly the
kind of upstream surprise this module must not paper over (CLAUDE.md rule
17) -- that day is dropped and reported as a warning rather than picking
either value.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from typing import Final

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "kiel_trade"
_PARSER_VERSION = "1"
_DEFAULT_BASE_URL = "https://trade.kielinstitut.de/KTI"

_TIMESTAMP_COLUMN = "timestamp"
#: Column names this parser recognises as "the row's entity label", checked
#: in this order. Neither file observed live declares both.
_NAME_COLUMNS: Final[tuple[str, ...]] = ("portname", "name")

# Errors expected from a malformed/unexpected upstream payload; anything else
# is a programming error and should propagate rather than be swallowed.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

#: Files actually confirmed live on 2026-09-28 (HTTP 200) and their column
#: shape -- documentation only. `parse()` does not consult this dict; it
#: trusts each payload's own CSV header every time, and does not hard-code
#: which of these are still being updated (see module docstring). A file
#: not listed here can still be requested; it just has not been verified by
#: this module's author.
VERIFIED_FILES: Final[dict[str, str]] = {
    "plot_ships_red_sea": "timestamp,n",
    "plot_ships_cape_good_hope": "timestamp,n",
    "plot_ships_panama_canal": "timestamp,n",
    "plot_stationary_ships": "timestamp,n",
    "plot_fleet_global": "timestamp,n",
    "plot_portcalls_china": "timestamp,portname,n",
    "plot_freight_rates_china_northern_europe_global": "name,code,timestamp,rate",
}

#: Confirmed HTTP 404 live on 2026-09-28. Documentation only.
KNOWN_MISSING_FILES: Final[tuple[str, ...]] = ("plot_draft_global", "plot_speed_global")

__all__ = [
    "KNOWN_MISSING_FILES",
    "VERIFIED_FILES",
    "KielTradeAdapter",
    "build_native_identifier",
    "parse_kiel_csv",
    "parse_native_identifier",
]


def build_native_identifier(basename: str, value_column: str, name_filter: str = "") -> str:
    """The publisher-identifier format this module uses for `SeriesSpec.native_identifier`."""
    if not basename or not value_column:
        raise AdapterError(
            f"kiel_trade: basename and value_column must both be non-empty "
            f"(got basename={basename!r}, value_column={value_column!r})"
        )
    if "|" in basename or "|" in value_column or "|" in name_filter:
        raise AdapterError("kiel_trade: native_identifier components must not contain '|'")
    return f"{basename}|{value_column}|{name_filter}"


def parse_native_identifier(raw: str) -> tuple[str, str, str]:
    """Inverse of `build_native_identifier`. Raises `AdapterError` on malformed input."""
    parts = raw.split("|")
    if len(parts) != 3:
        raise AdapterError(
            f"kiel_trade: native_identifier {raw!r} does not match "
            "'<basename>|<value column>|<optional name filter>'"
        )
    basename, value_column, name_filter = parts
    if not basename or not value_column:
        raise AdapterError(
            f"kiel_trade: native_identifier {raw!r} is missing a basename or value column"
        )
    return basename, value_column, name_filter


def _detect_name_column(fieldnames: Sequence[str]) -> str | None:
    for candidate in _NAME_COLUMNS:
        if candidate in fieldnames:
            return candidate
    return None


def parse_kiel_csv(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
) -> ParseResult:
    """Parse one Kiel Trade Indicator CSV payload into `ExternalObservation`s.

    Pure and network-free: everything needed comes from `content` and
    `spec`. Generic over all three shapes documented in the module
    docstring -- driven entirely by `spec.native_identifier`'s value column
    and optional name filter, never by the filename. Any structural
    surprise (undecodable bytes, no `timestamp` column, the declared value
    column absent, a name filter given for a file with no name/portname
    column, a multiplexed file with no filter given, an unparseable date, a
    non-numeric value, two different values for the same day) is reported
    as a warning; the affected row, or the whole payload, is skipped rather
    than guessed at.
    """
    warnings: list[str] = []
    basename, value_column, name_filter = parse_native_identifier(spec.native_identifier)

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: payload is not valid UTF-8 CSV: {exc}",),
            missing_series=(spec.series_id,),
        )

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    if _TIMESTAMP_COLUMN not in fieldnames:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: {basename}.csv has no {_TIMESTAMP_COLUMN!r} column "
                f"(found {fieldnames!r}) -- upstream contract may have changed",
            ),
            missing_series=(spec.series_id,),
        )
    if value_column not in fieldnames:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: declared value column {value_column!r} not present in "
                f"{basename}.csv (found {fieldnames!r})",
            ),
            missing_series=(spec.series_id,),
        )

    name_column = _detect_name_column(fieldnames)
    if name_filter and name_column is None:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: name filter {name_filter!r} given but {basename}.csv "
                f"has no name/portname column (found {fieldnames!r})",
            ),
            missing_series=(spec.series_id,),
        )
    if name_column is not None and not name_filter:
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: {basename}.csv multiplexes several series via "
                f"{name_column!r} but native_identifier gave no name filter -- refusing to "
                "guess which one this series means",
            ),
            missing_series=(spec.series_id,),
        )

    rows = list(reader)
    if name_column is not None:
        matched_rows = [r for r in rows if (r.get(name_column) or "").strip() == name_filter]
        if not matched_rows:
            return ParseResult(
                observations=[],
                warnings=(
                    f"{spec.qualified_id}: no rows found for {name_column}={name_filter!r} "
                    f"in {basename}.csv",
                ),
                missing_series=(spec.series_id,),
            )
    else:
        matched_rows = rows

    by_date: dict[date, list[float]] = {}
    skipped_bad_dates = 0
    skipped_blank_or_bad_values = 0
    for row in matched_rows:
        raw_ts = (row.get(_TIMESTAMP_COLUMN) or "").strip()
        try:
            obs_date = date.fromisoformat(raw_ts)
        except _PARSE_ERROR_TYPES:
            skipped_bad_dates += 1
            continue

        raw_value = (row.get(value_column) or "").strip()
        if not raw_value:
            skipped_blank_or_bad_values += 1
            continue
        try:
            value = float(raw_value)
        except _PARSE_ERROR_TYPES:
            skipped_blank_or_bad_values += 1
            continue

        by_date.setdefault(obs_date, []).append(value)

    if skipped_bad_dates:
        warnings.append(
            f"{spec.qualified_id}: {skipped_bad_dates} row(s) with an unparseable "
            f"{_TIMESTAMP_COLUMN!r} skipped"
        )
    if skipped_blank_or_bad_values:
        warnings.append(
            f"{spec.qualified_id}: {skipped_blank_or_bad_values} row(s) with a blank/"
            f"non-numeric {value_column!r} skipped, not imputed"
        )

    observations: list[ExternalObservation] = []
    source_version = f"kiel_trade_{basename}"
    for obs_date, values in sorted(by_date.items()):
        if len(set(values)) > 1:
            warnings.append(
                f"{spec.qualified_id}: {basename}.csv has {len(values)} conflicting "
                f"{value_column!r} values for {obs_date.isoformat()} "
                f"({sorted(set(values))!r}); day skipped rather than resolved optimistically"
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
                vintage_time=None,
                availability_precision=str(precision),
                revision_index=None,
            )
        )

    return ParseResult(observations=observations, warnings=tuple(warnings), missing_series=())


class KielTradeAdapter:
    """Fetches and parses one Kiel Trade Indicator CSV file's series at a time.

    One instance handles every file/column/filter combination a `SeriesSpec`
    can name; the file basename, value column and optional name filter all
    come from `spec.native_identifier`, never from adapter configuration.
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
        """Retrieve the current CSV file. `since` is unusable here: the
        publisher serves one full-history file per basename, never a partial
        one, so this never returns less than `since` onwards -- it always
        returns everything."""
        basename, _value_column, _name_filter = parse_native_identifier(spec.native_identifier)
        url = f"{self._base_url}/{basename}.csv"

        retrieved_at = datetime.now(UTC)
        # `_request` is the shared retry/rate-limit/honest-UA path every
        # adapter on this contract uses (see `adapters/ecb_data.py`); it
        # raises `AdapterHttpError` for a 404 (e.g. plot_draft_global.csv)
        # before this method ever has to fabricate an http_status.
        response = self._http._request("GET", url)
        request_url = str(response.request.url)
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=basename,
            url=request_url,
            content=response.content,
            http_status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            retrieved_at=retrieved_at,
            request_fingerprint=f"GET {request_url}",
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_kiel_csv(payload.content, spec, retrieved_at=payload.retrieved_at)
