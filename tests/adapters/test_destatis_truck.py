"""Contract tests for the Destatis daily truck-toll mileage index adapter.

The main fixture (`lkw_maut_fahrleistungsindex_sample.xlsx`) is a real file
downloaded live on 2026-09-28 from
  https://www.destatis.de/.../statistischer-bericht-lkw-maut-fahrleistungsindex-5421901.xlsx?__blob=publicationFile&v=24
trimmed from 13,675 data rows to 600 (the first and last 150 rows of each
of the two `Saisonbereinigung` blocks the real file contains) using only
Python's stdlib `zipfile`/`xml.etree.ElementTree` -- i.e. the same
machinery the adapter itself uses to read it, applied here to shrink
rather than parse. Every other part of the archive (styles, shared
strings, the header row, cell types) is untouched, so this is a real
XLSX with a real-but-shortened data sheet, not a hand-written one.

A handful of tests below need a payload with a specific defect (a blank
value cell, a missing header, an unexpected Gebiet) that the real file
does not contain; those construct a mutated copy of the real fixture's
bytes in memory (`_mutate_sheet9`) rather than hand-authoring a fake XLSX
from scratch, so the "host" file structure stays authentic even where one
cell is deliberately broken.

`landing_page_snippet.html` is a real (trimmed) excerpt of
https://www.destatis.de/.../Lkw-Maut-Fahrleistungsindex-Daten.html
captured 2026-09-28, containing the live, HTML-entity-double-escaped
`<a href="...xlsx?__blob=publicationFile&amp;amp;v=24">` link this
adapter's discovery regex must find and unescape.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from turboedge.adapters.base import AdapterError, AdapterHttpError
from turboedge.adapters.destatis_truck import (
    ADJUSTMENT_CALENDAR_SEASONALLY_ADJUSTED,
    ADJUSTMENT_RAW,
    DestatisTruckAdapter,
    _extract_download_url,
    parse_xlsx_payload,
)
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec

FIXTURES = Path(__file__).parent.parent / "fixtures" / "external" / "destatis"
XLSX_PATH = FIXTURES / "lkw_maut_fahrleistungsindex_sample.xlsx"
RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_SHEET_PATH = "xl/worksheets/sheet9.xml"
#: Shared-string indices in the real fixture (verified by inspection).
_SST_GEBIET_DEUTSCHLAND = 123
_SST_DIENSTAG = 115


def load_bytes() -> bytes:
    return XLSX_PATH.read_bytes()


def make_spec(
    *,
    native_identifier: str = ADJUSTMENT_RAW,
    precision: AvailabilityPrecision = AvailabilityPrecision.UNKNOWN,
    lag_hours: float | None = None,
) -> SeriesSpec:
    return SeriesSpec(
        source="destatis_truck",
        series_id="DESTATIS.LKW_MAUT.TEST",
        name="Daily truck-toll mileage index (test)",
        category="freight_activity",
        unit="index_2015_100",
        frequency="daily",
        native_identifier=native_identifier,
        availability_precision=precision,
        backfill_class=(
            BackfillClass.FORWARD_ONLY
            if precision is AvailabilityPrecision.UNKNOWN
            else BackfillClass.HISTORICAL_CONSERVATIVE
        ),
        conservative_release_lag_hours=lag_hours,
    )


def _mutate_sheet9(xlsx_bytes: bytes, mutate) -> bytes:
    """Return a copy of the fixture with `mutate(sheetdata_element)` applied
    to the `csv-42191-b01` sheet's `<sheetData>`, everything else untouched."""
    zin = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    root = ET.fromstring(zin.read(_SHEET_PATH))
    sheetdata = root.find("m:sheetData", _NS)
    mutate(sheetdata)
    new_xml = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    out = io.BytesIO()
    zout = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == _SHEET_PATH:
            data = new_xml
        zout.writestr(item, data)
    zout.close()
    return out.getvalue()


def _cell(row, col_letter: str):
    for c in row:
        if "".join(ch for ch in c.get("r") if ch.isalpha()) == col_letter:
            return c
    raise AssertionError(f"column {col_letter} not found in row {row.get('r')}")


# --- normal parsing --------------------------------------------------------


def test_raw_variant_parses_expected_row_count_and_boundary_values() -> None:
    result = parse_xlsx_payload(load_bytes(), make_spec(), retrieved_at=RETRIEVED_AT)
    assert not result.warnings
    assert not result.missing_series
    assert len(result.observations) == 300  # 150 + 150 trimmed rows

    first, last = result.observations[0], result.observations[-1]
    assert first.observation_time == datetime(2008, 1, 1, tzinfo=UTC)
    assert first.value == 7.0
    assert last.observation_time == datetime(2026, 9, 19, tzinfo=UTC)
    assert last.value == 45.0
    for obs in result.observations:
        assert obs.series_id == "DESTATIS.LKW_MAUT.TEST"
        assert obs.source == "destatis_truck"
        assert obs.vintage_time is None  # no per-observation timing is recorded at all
        assert obs.revision_index is None


def test_ksb_variant_parses_a_disjoint_set_of_rows() -> None:
    result = parse_xlsx_payload(
        load_bytes(),
        make_spec(native_identifier=ADJUSTMENT_CALENDAR_SEASONALLY_ADJUSTED),
        retrieved_at=RETRIEVED_AT,
    )
    assert not result.missing_series
    assert len(result.observations) == 300
    first, last = result.observations[0], result.observations[-1]
    assert first.observation_time == datetime(2008, 1, 1, tzinfo=UTC)
    assert first.value == 71.5
    assert last.observation_time == datetime(2026, 9, 19, tzinfo=UTC)
    assert last.value == 97.0

    raw_result = parse_xlsx_payload(load_bytes(), make_spec(), retrieved_at=RETRIEVED_AT)
    raw_values = {o.value for o in raw_result.observations}
    ksb_values = {o.value for o in result.observations}
    assert raw_values != ksb_values  # two genuinely different series, not a duplicate


# --- availability precision pass-through -----------------------------------


def test_unknown_precision_yields_day_after_placeholder() -> None:
    result = parse_xlsx_payload(load_bytes(), make_spec(), retrieved_at=RETRIEVED_AT)
    first = result.observations[0]
    assert first.observation_time == datetime(2008, 1, 1, tzinfo=UTC)
    assert first.available_at == datetime(2008, 1, 2, tzinfo=UTC)
    assert first.availability_precision == "UNKNOWN"


def test_conservative_date_precision_applies_declared_lag() -> None:
    spec = make_spec(precision=AvailabilityPrecision.CONSERVATIVE_DATE, lag_hours=312.0)  # 13 days
    result = parse_xlsx_payload(load_bytes(), spec, retrieved_at=RETRIEVED_AT)
    first = result.observations[0]
    assert first.observation_time == datetime(2008, 1, 1, tzinfo=UTC)
    assert first.available_at == datetime(2008, 1, 14, 0, 0, tzinfo=UTC)
    assert first.availability_precision == "CONSERVATIVE_DATE"


def test_exact_timestamp_precision_is_refused_not_fabricated() -> None:
    """This adapter never supplies an exact per-observation timestamp
    (module docstring: the file records no publication timing at all), so
    a SeriesSpec that declares EXACT_TIMESTAMP must make resolve_available_at
    refuse rather than let the adapter invent one -- but SeriesSpec itself
    doesn't require a lag for EXACT_TIMESTAMP, so this is caught downstream
    in resolve_available_at, exercised here via the real parse path."""
    spec = SeriesSpec(
        source="destatis_truck",
        series_id="DESTATIS.LKW_MAUT.BAD",
        name="bad spec",
        category="freight_activity",
        unit="index_2015_100",
        frequency="daily",
        native_identifier=ADJUSTMENT_RAW,
        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP,
        backfill_class=BackfillClass.HISTORICAL_PIT_SAFE,
    )
    with pytest.raises(ValueError, match="refusing to substitute"):
        parse_xlsx_payload(load_bytes(), spec, retrieved_at=RETRIEVED_AT)


# --- missing values / schema drift -----------------------------------------


def test_blank_indexwert_is_skipped_not_imputed() -> None:
    def blank_out_first_value(sheetdata) -> None:
        rows = list(sheetdata)
        target_row = rows[1]  # first data row (row 0 is the header)
        value_cell = _cell(target_row, "G")
        for v in list(value_cell):
            value_cell.remove(v)
        value_cell.attrib.pop("t", None)

    mutated = _mutate_sheet9(load_bytes(), blank_out_first_value)
    result = parse_xlsx_payload(mutated, make_spec(), retrieved_at=RETRIEVED_AT)
    assert len(result.observations) == 299  # one fewer than the clean fixture
    assert any("blank/unparseable Indexwert" in w for w in result.warnings)
    assert datetime(2008, 1, 1, tzinfo=UTC) not in {o.observation_time for o in result.observations}


def test_unparseable_datum_is_skipped_and_warned() -> None:
    def corrupt_first_date(sheetdata) -> None:
        rows = list(sheetdata)
        date_cell = _cell(rows[1], "C")
        date_cell.find("m:v", _NS).text = "not-a-date"

    mutated = _mutate_sheet9(load_bytes(), corrupt_first_date)
    result = parse_xlsx_payload(mutated, make_spec(), retrieved_at=RETRIEVED_AT)
    assert len(result.observations) == 299
    assert any("unparseable Datum" in w for w in result.warnings)


def test_unexpected_gebiet_is_warned_and_excluded() -> None:
    def swap_gebiet(sheetdata) -> None:
        rows = list(sheetdata)
        gebiet_cell = _cell(rows[1], "B")
        gebiet_cell.find("m:v", _NS).text = str(_SST_DIENSTAG)

    mutated = _mutate_sheet9(load_bytes(), swap_gebiet)
    result = parse_xlsx_payload(mutated, make_spec(), retrieved_at=RETRIEVED_AT)
    assert len(result.observations) == 299
    assert any("unexpected Gebiet" in w for w in result.warnings)


def test_missing_indexwert_column_raises_not_returns_empty() -> None:
    def rename_header(sheetdata) -> None:
        header = next(iter(sheetdata))
        header_cell = _cell(header, "G")
        # Point the shared-string index at "Wochentag" (col E's own header
        # text) instead of "Indexwert" -- a renamed/removed column.
        header_cell.find("m:v", _NS).text = str(80)  # "Wochentag" per module

    mutated = _mutate_sheet9(load_bytes(), rename_header)
    with pytest.raises(AdapterError, match="missing expected column"):
        parse_xlsx_payload(mutated, make_spec(), retrieved_at=RETRIEVED_AT)


def test_unknown_saisonbereinigung_variant_is_missing_series() -> None:
    spec = make_spec(native_identifier="a variant that does not exist")
    result = parse_xlsx_payload(load_bytes(), spec, retrieved_at=RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert result.warnings


def test_not_a_zip_raises() -> None:
    with pytest.raises(AdapterError, match="not a valid XLSX"):
        parse_xlsx_payload(b"definitely not a zip file", make_spec(), retrieved_at=RETRIEVED_AT)


# --- landing-page discovery / fallback / rate limiting ----------------------


def test_extract_download_url_unescapes_and_resolves_real_snippet() -> None:
    html_text = (FIXTURES / "landing_page_snippet.html").read_text(encoding="utf-8")
    page_url = (
        "https://www.destatis.de/DE/Themen/Branchen-Unternehmen/"
        "Industrie-Verarbeitendes-Gewerbe/Tabellen/Lkw-Maut-Fahrleistungsindex-Daten.html"
    )
    url = _extract_download_url(html_text, page_url)
    assert url is not None
    assert url.startswith("https://www.destatis.de/")
    assert "statistischer-bericht-lkw-maut-fahrleistungsindex-5421901.xlsx" in url
    assert "__blob=publicationFile" in url
    assert "v=24" in url
    assert "&amp;" not in url  # fully unescaped


def test_extract_download_url_returns_none_when_absent() -> None:
    assert _extract_download_url("<html><body>nothing here</body></html>", "https://x/") is None


class _StubHttpClient:
    def __init__(self, *, landing_html: str | None, xlsx_bytes: bytes) -> None:
        self._landing_html = landing_html
        self._xlsx_bytes = xlsx_bytes
        self.requested_urls: list[str] = []

    def get_text(self, url: str, **_kwargs: object) -> str:
        self.requested_urls.append(url)
        if self._landing_html is None:
            raise AdapterHttpError("landing page unavailable")
        return self._landing_html

    def get_bytes(self, url: str, **_kwargs: object) -> bytes:
        self.requested_urls.append(url)
        return self._xlsx_bytes


def test_fetch_discovers_url_from_landing_page() -> None:
    html_text = (FIXTURES / "landing_page_snippet.html").read_text(encoding="utf-8")
    stub = _StubHttpClient(landing_html=html_text, xlsx_bytes=load_bytes())
    adapter = DestatisTruckAdapter(http_client=stub)  # type: ignore[arg-type]

    payload = adapter.fetch(make_spec())

    assert "statistischer-bericht-lkw-maut-fahrleistungsindex-5421901.xlsx" in payload.url
    assert "v=24" in payload.url
    assert payload.content == stub._xlsx_bytes
    assert payload.source == "destatis_truck"
    assert len(stub.requested_urls) == 2  # landing page, then the xlsx itself


def test_fetch_falls_back_when_landing_page_unreachable() -> None:
    stub = _StubHttpClient(landing_html=None, xlsx_bytes=load_bytes())
    adapter = DestatisTruckAdapter(http_client=stub)  # type: ignore[arg-type]

    payload = adapter.fetch(make_spec())

    assert "__blob=publicationFile" in payload.url
    assert "v=" not in payload.url  # the un-versioned fallback URL
    assert payload.content == stub._xlsx_bytes


def test_fetch_falls_back_when_link_pattern_not_found() -> None:
    stub = _StubHttpClient(landing_html="<html>no matching link</html>", xlsx_bytes=load_bytes())
    adapter = DestatisTruckAdapter(http_client=stub)  # type: ignore[arg-type]

    payload = adapter.fetch(make_spec())

    assert "__blob=publicationFile" in payload.url
    assert "v=" not in payload.url


def test_fetch_url_and_fingerprint_have_no_credential_markers() -> None:
    stub = _StubHttpClient(landing_html=None, xlsx_bytes=load_bytes())
    adapter = DestatisTruckAdapter(http_client=stub)  # type: ignore[arg-type]
    payload = adapter.fetch(make_spec())
    for marker in ("api_key=", "apikey=", "token=", "password=", "appid="):
        assert marker not in payload.url.lower()
        assert marker not in payload.request_fingerprint.lower()


def test_default_http_client_honours_the_30s_crawl_delay() -> None:
    """robots.txt for www.destatis.de: 'Crawl-delay: 30' (verified
    2026-09-28). The adapter's default HttpClient must be constructed with
    a matching per-host minimum interval; HttpClient's own rate limiter
    (keyed by host, applied to every request regardless of path) does the
    rest -- this adapter has no throttling logic of its own to test."""
    adapter = DestatisTruckAdapter()
    assert adapter._http._min_interval_s >= 30.0


def test_adapter_identity() -> None:
    adapter = DestatisTruckAdapter(http_client=_StubHttpClient(landing_html=None, xlsx_bytes=b""))  # type: ignore[arg-type]
    assert adapter.source_id == "destatis_truck"
    assert adapter.parser_version
