"""Destatis daily truck-toll mileage index ("taeglicher Lkw-Maut-Fahrleistungsindex").

Verified live 2026-09-28
------------------------
The data are published as an XLSX statistical report (GENESIS-Online table
42191-0001), updated weekly (Thursdays), with daily values from 2008-01-01.
Two concrete download paths were confirmed live, both HTTP 200 with
content-type `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`:

  * The versioned link this workstream's brief supplied
    (`...-5421901.xlsx?__blob=publicationFile&v=24`) -- confirmed still
    current: the landing page's own HTML links to exactly this `v=24` URL
    today.
  * The *same* file with the `v=` parameter dropped entirely
    (`...-5421901.xlsx?__blob=publicationFile`) -- also HTTP 200, same
    content-type, `Last-Modified: Thu, 24 Sep 2026 09:46:09 GMT`. This is
    used as the fallback when the landing page cannot be parsed, per the
    workstream brief's instruction not to hard-code one `v=` forever.

`fetch()` therefore does not hard-code a `v=` value. It first fetches the
landing page
  https://www.destatis.de/DE/Themen/Branchen-Unternehmen/Industrie-Verarbeitendes-Gewerbe/Tabellen/Lkw-Maut-Fahrleistungsindex-Daten.html
and extracts whatever `...-5421901.xlsx?__blob=publicationFile...` href is
currently published there (the href in the live HTML is HTML-entity
double-escaped, `&amp;amp;v=24`, unescaped here accordingly); if that page
is unreachable or its markup no longer contains a matching link, it falls
back to the un-versioned URL above, which independently returns the current
file.

Crawl-delay: 30 (verified 2026-09-28 against https://www.destatis.de/robots.txt)
-----------------------------------------------------------------------------
`HttpClient`'s per-host rate limiter (`base.py:_HostRateLimiter`, `.wait()`
called before every request regardless of path) is keyed by hostname alone,
so constructing this adapter's `HttpClient` with `min_interval_s=30.0`
enforces the 30-second minimum interval uniformly: between the landing-page
request and the XLSX request inside one `fetch()` call, and across repeated
`fetch()` calls sharing one adapter instance. `HttpClient` can express this
directly; nothing in this module needs its own throttling logic.

No new dependency: hand-rolled XLSX reading
--------------------------------------------
`polars.read_excel` needs one of `fastexcel`, `openpyxl` or `xlsx2csv`
(checked: `./.venv/bin/python -c "import polars as pl; pl.read_excel(...)"`
against the real downloaded file raises `ModuleNotFoundError: fastexcel`);
none of those, nor `pandas`'s own Excel engines, are installed in this
project's `.venv`. Rather than adding a dependency for this one file
format, this module reads the XLSX directly with `zipfile` + `xml.etree.
ElementTree` (both stdlib): an XLSX is a zip of XML parts, and the specific
sheet this adapter reads is simple enough (flat table, no merged cells,
one header row) that this is under 100 lines. This also sidesteps German
locale formatting entirely: Destatis ships the report as several sheets,
and the one this module reads is deliberately the *machine-readable*
one -- named `csv-<table>` (here `csv-42191-b01`; the human-facing display
sheet `42191-b01` uses German `DD.MM.YYYY`/comma-decimal *text* formatting
and is NOT used here) -- whose `Datum` and `Indexwert` cells are native
XLSX numeric types (an Excel date serial number and a plain float,
respectively), not locale-formatted strings. `_parse_number` still carries
a defensive comma-decimal fallback in case a future export changes that,
but it is untested against any real fixture because the real file does not
need it.

If a library-based approach is preferred instead of this hand-rolled
reader, the dependency that would be needed is `fastexcel>=0.12` (Rust
`calamine` backend, what `polars.read_excel` uses by default) or
`openpyxl>=3.1` (pure Python, works with `pandas.read_excel`) -- reported
here per the workstream brief's instruction not to edit `pyproject.toml`
directly.

Two published series (Saisonbereinigung column, used verbatim as this
module's `SeriesSpec.native_identifier`, stripped of the source's trailing
column-width padding)
----------------------------------------------------------------------
  "unbereinigt"                          raw / not adjusted
  "Kalender- und saisonbereinigt (KSB)"  calendar- and seasonally-adjusted

Both cover the same 2008-01-01 to present date range for
Gebiet="Deutschland insgesamt" (the only region in the file today; any
other Gebiet value that appears in the future is warned about, not
silently mixed in).

Point-in-time honesty
----------------------
"Historical release timing cannot be reconstructed for this series: old
daily values were published in weekly Thursday batches, and the file
records no publication timestamp at all (unlike eu_bcs, there is not even
a dataset-wide 'updated' field to fall back on -- `source_release_time`
and `vintage_time` are therefore left `None` for every observation).
`available_at` is decided *only* by `resolve_available_at()`
(`external/adapter.py`, CLAUDE.md rule 5) from whatever
`availability_precision` the caller's `SeriesSpec` declares -- this module
does not invent its own rule. A `SeriesSpec` for this source must declare
either `UNKNOWN` (safe default; `resolve_available_at` returns a
non-authoritative placeholder that strict point-in-time research
automatically excludes) or `CONSERVATIVE_DATE` with an explicit
`conservative_release_lag_hours` chosen generously enough to cover the
weekly batch delay -- a daily value from day D is not necessarily published
until the Thursday covering that week, which can be up to ~13 days later
(6 days to the end of its own week, up to 7 more to the following
Thursday); a lag shorter than that would risk treating a value as usable
before its batch was actually released. This module never sets
`EXACT_DATE`/`EXACT_TIMESTAMP` and `resolve_available_at` will raise if a
`SeriesSpec` tries to declare `EXACT_TIMESTAMP` without supplying an actual
timestamp, which this adapter never does.

Licensing -- REVIEW_REQUIRED, do not guess
--------------------------------------------
Neither the landing page nor the report's own detail page states explicit
reuse/licence terms (checked for "Lizenz", "Datenlizenz", "CC BY",
"Nutzungsbedingungen", "Copyright" in both pages' HTML on 2026-09-28: none
present). The downloaded XLSX itself, however, carries an "Impressum"
sheet with the statement (German, verbatim): "Vervielfaeltigung und
Verbreitung, auch auszugsweise, mit Quellenangabe gestattet." --
"Reproduction and distribution, even in part, with attribution, is
permitted." That is not a formal licence name (not "CC BY" or "Datenlizenz
Deutschland" by name) and this module does not attempt to classify it.
Per the workstream brief, this source is registered `REVIEW_REQUIRED`
until a human resolves the licence question; `NON_INGESTING_STATUSES`
(`external/schemas.py`) keeps a `REVIEW_REQUIRED` source from being
auto-ingested regardless of `enabled`.
"""

from __future__ import annotations

import html
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, date, datetime, time, timedelta
from urllib.parse import urljoin

import structlog

from turboedge.adapters.base import AdapterError, AdapterHttpError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

logger = structlog.get_logger(__name__)

_SOURCE_ID = "destatis_truck"
_PARSER_VERSION = "1"
_DATASET = "lkw_maut_fahrleistungsindex"
_USER_AGENT = "TurboEdge-DE-research/0.1 (contact: pcctradinginc@gmail.com)"
#: robots.txt for www.destatis.de: "Crawl-delay: 30" (verified 2026-09-28).
_MIN_INTERVAL_S = 30.0

_LANDING_URL = (
    "https://www.destatis.de/DE/Themen/Branchen-Unternehmen/"
    "Industrie-Verarbeitendes-Gewerbe/Tabellen/Lkw-Maut-Fahrleistungsindex-Daten.html"
)
#: Confirmed live 2026-09-28: same file, HTTP 200, without a `v=` parameter.
_FALLBACK_XLSX_URL = (
    "https://www.destatis.de/DE/Themen/Branchen-Unternehmen/"
    "Industrie-Verarbeitendes-Gewerbe/Publikationen/Downloads-Konjunktur/"
    "statistischer-bericht-lkw-maut-fahrleistungsindex-5421901.xlsx?__blob=publicationFile"
)

_HREF_RE = re.compile(
    r'href="([^"]*statistischer-bericht-lkw-maut-fahrleistungsindex-\d+\.xlsx\?'
    r'[^"]*__blob=publicationFile[^"]*)"',
    re.IGNORECASE,
)

_XML_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_R_ID_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

_REQUIRED_HEADERS = ("Statistik", "Gebiet", "Datum", "Wochentag", "Saisonbereinigung", "Indexwert")
_EXPECTED_GEBIET = "Deutschland insgesamt"

#: Verified 2026-09-28 against the live file's `xl/sharedStrings.xml`
#: (trailing spaces in the source are column-width padding, stripped here).
ADJUSTMENT_RAW = "unbereinigt"
ADJUSTMENT_CALENDAR_SEASONALLY_ADJUSTED = "Kalender- und saisonbereinigt (KSB)"

#: Excel's day-zero under the (unpatched, but irrelevant post-1900) 1900
#: date system: serial 1 == 1899-12-31, so serial N == this date + N days.
_EXCEL_EPOCH = date(1899, 12, 30)


def _extract_download_url(landing_html: str, page_url: str) -> str | None:
    """Find the current `...xlsx?__blob=publicationFile...` link on the landing page.

    The live href is HTML-entity-escaped more than once (`&amp;amp;v=24`);
    unescape repeatedly until stable rather than assuming a fixed depth.
    """
    match = _HREF_RE.search(landing_html)
    if not match:
        return None
    href = match.group(1)
    previous = None
    while previous != href:
        previous = href
        href = html.unescape(href)
    return urljoin(page_url, href)


def _load_workbook_sheet_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    """Sheet name -> archive path, resolved through workbook.xml + its rels."""
    workbook_xml = ET.fromstring(zf.read("xl/workbook.xml"))
    rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {
        rel.get("Id"): rel.get("Target") for rel in rels_xml.findall("rel:Relationship", _XML_NS)
    }
    sheets_el = workbook_xml.find("m:sheets", _XML_NS)
    if sheets_el is None:
        raise AdapterError("destatis_truck: workbook.xml has no <sheets> element")
    paths: dict[str, str] = {}
    for sheet_el in sheets_el:
        name = sheet_el.get("name")
        rid = sheet_el.get(_R_ID_ATTR)
        target = rid_to_target.get(rid) if rid else None
        if name is None or target is None:
            continue
        paths[name] = target if target.startswith("xl/") else f"xl/{target}"
    return paths


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    text_tag = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
    return ["".join(t.text or "" for t in si.iter(text_tag)) for si in root]


def _column_letters(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def _iter_sheet_rows(
    zf: zipfile.ZipFile, path: str, shared_strings: list[str]
) -> list[dict[str, str | None]]:
    root = ET.fromstring(zf.read(path))
    sheet_data = root.find("m:sheetData", _XML_NS)
    if sheet_data is None:
        return []
    rows: list[dict[str, str | None]] = []
    for row_el in sheet_data:
        row: dict[str, str | None] = {}
        for cell_el in row_el:
            ref = cell_el.get("r")
            if not ref:
                continue
            col = _column_letters(ref)
            v_el = cell_el.find("m:v", _XML_NS)
            if v_el is None or v_el.text is None:
                row[col] = None
                continue
            row[col] = shared_strings[int(v_el.text)] if cell_el.get("t") == "s" else v_el.text
        rows.append(row)
    return rows


def _excel_serial_to_date(raw: str | None) -> date | None:
    if raw is None:
        return None
    try:
        serial = float(raw)
    except ValueError:
        return None
    if serial != int(serial):
        # A time-of-day fraction is not expected for this column; treat as
        # schema drift rather than silently truncating it.
        return None
    return _EXCEL_EPOCH + timedelta(days=int(serial))


def _parse_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    # Defensive German-locale fallback (thousands '.' then decimal ',') --
    # not exercised by the real fixture, see module docstring.
    normalized = text.replace(".", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_xlsx_payload(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
) -> ParseResult:
    """Pure parse function: XLSX bytes + spec -> observations. No network I/O."""
    warnings: list[str] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise AdapterError(f"destatis_truck: payload is not a valid XLSX/zip file: {exc}") from exc

    try:
        shared_strings = _load_shared_strings(zf)
        sheet_paths = _load_workbook_sheet_paths(zf)
    except ET.ParseError as exc:
        raise AdapterError(f"destatis_truck: could not parse workbook XML: {exc}") from exc

    csv_sheets = {name: path for name, path in sheet_paths.items() if name.startswith("csv-")}
    if len(csv_sheets) != 1:
        raise AdapterError(
            "destatis_truck: expected exactly one 'csv-' prefixed machine-readable "
            f"sheet, found {sorted(csv_sheets)!r} -- upstream workbook structure changed"
        )
    sheet_name, sheet_path = next(iter(csv_sheets.items()))

    try:
        rows = _iter_sheet_rows(zf, sheet_path, shared_strings)
    except ET.ParseError as exc:
        raise AdapterError(f"destatis_truck: could not parse sheet {sheet_name!r}: {exc}") from exc
    if not rows:
        raise AdapterError(f"destatis_truck: sheet {sheet_name!r} has no rows")

    header_row, data_rows = rows[0], rows[1:]
    header_to_col = {(value or "").strip(): col for col, value in header_row.items() if value}
    missing_headers = [h for h in _REQUIRED_HEADERS if h not in header_to_col]
    if missing_headers:
        raise AdapterError(
            f"destatis_truck: sheet {sheet_name!r} is missing expected column(s) "
            f"{missing_headers!r} (found {sorted(header_to_col)!r}) -- upstream contract changed"
        )

    col_gebiet = header_to_col["Gebiet"]
    col_datum = header_to_col["Datum"]
    col_sais = header_to_col["Saisonbereinigung"]
    col_value = header_to_col["Indexwert"]
    target_adjustment = spec.native_identifier.strip()

    observations: list[ExternalObservation] = []
    matched_any_adjustment = False
    skipped_blank_values = 0
    skipped_bad_dates = 0
    unexpected_gebiet: set[str] = set()

    for row in data_rows:
        adjustment = (row.get(col_sais) or "").strip()
        if adjustment != target_adjustment:
            continue
        matched_any_adjustment = True

        gebiet = (row.get(col_gebiet) or "").strip()
        if gebiet != _EXPECTED_GEBIET:
            unexpected_gebiet.add(gebiet)
            continue

        obs_date = _excel_serial_to_date(row.get(col_datum))
        if obs_date is None:
            skipped_bad_dates += 1
            continue

        value = _parse_number(row.get(col_value))
        if value is None:
            skipped_blank_values += 1
            continue

        observation_time = datetime.combine(obs_date, time(0, 0), tzinfo=UTC)
        available_at, precision = resolve_available_at(spec, obs_date)
        observations.append(
            ExternalObservation(
                series_id=spec.series_id,
                value=value,
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=f"destatis_genesis_{sheet_name}",
                observation_time=observation_time,
                available_at=available_at,
                retrieved_at=retrieved_at,
                source=_SOURCE_ID,
                parser_version=_PARSER_VERSION,
                quality_score=1.0,
                availability_precision=str(precision),
            )
        )

    if skipped_blank_values:
        warnings.append(
            f"destatis_truck: {skipped_blank_values} row(s) with blank/unparseable "
            "Indexwert skipped, not imputed"
        )
    if skipped_bad_dates:
        warnings.append(
            f"destatis_truck: {skipped_bad_dates} row(s) with an unparseable Datum cell skipped"
        )
    if unexpected_gebiet:
        warnings.append(
            f"destatis_truck: row(s) for unexpected Gebiet value(s) {sorted(unexpected_gebiet)!r} "
            f"were present but not emitted (this parser only emits {_EXPECTED_GEBIET!r})"
        )

    missing_series: tuple[str, ...] = ()
    if not matched_any_adjustment:
        warnings.append(
            f"{spec.qualified_id}: no rows found for Saisonbereinigung={target_adjustment!r} "
            f"in sheet {sheet_name!r}"
        )
        missing_series = (spec.series_id,)

    return ParseResult(
        observations=observations, warnings=tuple(warnings), missing_series=missing_series
    )


class DestatisTruckAdapter:
    """Fetches and parses the Destatis daily truck-toll mileage index."""

    def __init__(self, http_client: HttpClient | None = None) -> None:
        self._http = http_client or HttpClient(
            user_agent=_USER_AGENT,
            min_interval_s=_MIN_INTERVAL_S,
        )

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def _discover_url(self) -> str:
        try:
            landing_html = self._http.get_text(_LANDING_URL)
        except AdapterHttpError as exc:
            logger.warning("destatis_truck_landing_page_unreachable", error=str(exc))
            return _FALLBACK_XLSX_URL
        discovered = _extract_download_url(landing_html, _LANDING_URL)
        if discovered is None:
            logger.warning("destatis_truck_landing_page_link_not_found")
            return _FALLBACK_XLSX_URL
        return discovered

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        """Retrieve the current XLSX report. `since` is unusable here: the
        publisher serves one full-history file, never a partial one, so
        this never returns less than `since` onwards -- it always returns
        everything."""
        url = self._discover_url()
        retrieved_at = datetime.now(UTC)
        content = self._http.get_bytes(url)
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=_DATASET,
            url=url,
            content=content,
            http_status=200,
            content_type=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            retrieved_at=retrieved_at,
            request_fingerprint=url,
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_xlsx_payload(payload.content, spec, retrieved_at=payload.retrieved_at)


__all__ = [
    "ADJUSTMENT_CALENDAR_SEASONALLY_ADJUSTED",
    "ADJUSTMENT_RAW",
    "DestatisTruckAdapter",
    "parse_xlsx_payload",
]
