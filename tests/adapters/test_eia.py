"""Tests for the U.S. EIA Open Data v2 adapter (`adapters/eia.py`).

Offline by default. `petroleum_stoc_wstk_normal.json` and
`natural_gas_stor_wkly_monthly_period.json` are **hand-built from the
official EIA v2 API documentation**
(https://www.eia.gov/opendata/documentation.php) -- no `EIA_API_KEY` is
available in this environment, so no real route's data has been observed.

`eia_error_api_key_missing.json` is different: it is hand-built to match the
documented error shape, but this exact byte-for-byte body was *also*
independently observed on a real, unauthenticated (no key obtained or used)
live request to `https://api.eia.gov/v2/petroleum/stoc/wstk/data/` during
this workstream's development, on 2026-09-28 (HTTP 403) -- see
`adapters/eia.py`'s module docstring "THE CREDENTIAL FAILURE MODE".
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters.base import AdapterHttpError, HttpClient
from turboedge.adapters.eia import (
    SUGGESTED_ROUTES,
    EiaAdapter,
    EiaCredentialError,
    build_native_identifier,
    parse_eia_payload,
    parse_native_identifier,
)
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "eia"
_ROUTE = "petroleum/stoc/wstk"
_URL = f"https://api.eia.gov/v2/{_ROUTE}/data/"
_RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.json").read_bytes()


def _spec(
    *,
    series_id: str = "EIA.WCESTUS1.value",
    route: str = _ROUTE,
    facets: dict[str, str] | None = None,
    data_column: str = "value",
    unit: str = "MBBL",
    frequency: str = "weekly",
    lag_hours: float = 48.0,
) -> SeriesSpec:
    return SeriesSpec(
        source="eia",
        series_id=series_id,
        name="Weekly US crude oil ending stocks",
        category="energy",
        unit=unit,
        frequency=frequency,
        native_identifier=build_native_identifier(
            route, facets or {"series": "WCESTUS1"}, data_column
        ),
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.FORWARD_ONLY,
        conservative_release_lag_hours=lag_hours,
    )


# --------------------------------------------------------------------------
# native_identifier round trip
# --------------------------------------------------------------------------


def test_build_and_parse_native_identifier_round_trip() -> None:
    raw = build_native_identifier("petroleum/stoc/wstk", {"series": "WCESTUS1"}, "value")
    assert raw == "petroleum/stoc/wstk|series=WCESTUS1|value"
    assert parse_native_identifier(raw) == ("petroleum/stoc/wstk", {"series": "WCESTUS1"}, "value")


def test_build_native_identifier_supports_multiple_facets() -> None:
    raw = build_native_identifier(
        "electricity/rto/daily-region-data", {"respondent": "US48", "type": "D"}, "value"
    )
    route, facets, column = parse_native_identifier(raw)
    assert route == "electricity/rto/daily-region-data"
    assert facets == {"respondent": "US48", "type": "D"}
    assert column == "value"


def test_native_identifier_allows_empty_facets() -> None:
    raw = build_native_identifier("petroleum/pri/spt", {}, "value")
    assert raw == "petroleum/pri/spt||value"
    assert parse_native_identifier(raw) == ("petroleum/pri/spt", {}, "value")


def test_parse_native_identifier_rejects_wrong_segment_count() -> None:
    with pytest.raises(Exception, match="does not match"):
        parse_native_identifier("petroleum/stoc/wstk|value")


def test_parse_native_identifier_rejects_malformed_facet() -> None:
    with pytest.raises(Exception, match="malformed facet"):
        parse_native_identifier("petroleum/stoc/wstk|series-without-equals|value")


# --------------------------------------------------------------------------
# normal parse
# --------------------------------------------------------------------------


def test_parse_petroleum_stoc_wstk_normal() -> None:
    spec = _spec()
    result = parse_eia_payload(
        _fixture_bytes("petroleum_stoc_wstk_normal"), spec, retrieved_at=_RETRIEVED_AT
    )

    # 4 rows: one null (skipped), one matches a different facet (skipped),
    # two are real WCESTUS1 observations.
    assert len(result.observations) == 2
    by_day = {o.observation_time.date(): o for o in result.observations}
    assert by_day[date(2026, 9, 5)].value == pytest.approx(419876)
    assert by_day[date(2026, 9, 19)].value == pytest.approx(421345)
    assert all(o.series_id == spec.series_id for o in result.observations)
    assert all(o.source == "eia" for o in result.observations)
    assert all(o.unit == "MBBL" for o in result.observations)
    assert all(o.frequency == "weekly" for o in result.observations)


def test_parse_null_value_row_skipped_without_warning() -> None:
    spec = _spec()
    result = parse_eia_payload(
        _fixture_bytes("petroleum_stoc_wstk_normal"), spec, retrieved_at=_RETRIEVED_AT
    )
    days = {o.observation_time.date() for o in result.observations}
    assert date(2026, 9, 12) not in days
    assert result.warnings == ()


def test_parse_facet_mismatch_row_excluded() -> None:
    """The fixture's 4th row is series=WCRSTUS1 (gasoline), not WCESTUS1
    (crude); requesting WCESTUS1 must never pick it up."""
    spec = _spec()
    result = parse_eia_payload(
        _fixture_bytes("petroleum_stoc_wstk_normal"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert all(o.value != pytest.approx(225110) for o in result.observations)


def test_parse_monthly_period_and_numeric_json_values() -> None:
    """Covers both the 'YYYY-MM' period format and a JSON-native numeric
    value (as opposed to the other fixture's quoted-string values) --
    `_parse_number` must accept both."""
    spec = _spec(
        series_id="EIA.NUS.stor.value",
        route="natural-gas/stor/wkly",
        facets={"duoarea": "NUS"},
        unit="BCF",
        frequency="monthly",
    )
    result = parse_eia_payload(
        _fixture_bytes("natural_gas_stor_wkly_monthly_period"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    assert len(result.observations) == 2
    by_month = {o.observation_time.date(): o.value for o in result.observations}
    assert by_month[date(2026, 7, 1)] == pytest.approx(3120.5)
    assert by_month[date(2026, 8, 1)] == pytest.approx(3350)


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash
# --------------------------------------------------------------------------


def test_non_json_payload_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_eia_payload(b"not json at all", spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "JSON" in result.warnings[0]


def test_error_body_has_no_response_key_and_warns_gracefully() -> None:
    """The real, live-observed EIA credential-failure body has no
    'response' object at all -- parse() must not crash if this is ever
    replayed (in normal operation fetch() raises before archiving it)."""
    spec = _spec()
    result = parse_eia_payload(
        _fixture_bytes("eia_error_api_key_missing"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "response" in result.warnings[0]


def test_response_data_not_a_list_warns() -> None:
    spec = _spec()
    payload = json.dumps({"response": {"total": "0", "data": "oops"}}).encode("utf-8")
    result = parse_eia_payload(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)


def test_non_object_row_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.dumps({"response": {"total": "1", "data": ["not-a-row"]}}).encode("utf-8")
    result = parse_eia_payload(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert any("non-object entry" in w for w in result.warnings)


def test_row_missing_data_column_warns() -> None:
    spec = _spec()
    payload = json.dumps(
        {"response": {"total": "1", "data": [{"period": "2026-09-05", "series": "WCESTUS1"}]}}
    ).encode("utf-8")
    result = parse_eia_payload(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert any("has no" in w and "column" in w for w in result.warnings)


def test_unparseable_period_warns() -> None:
    spec = _spec()
    payload = json.dumps(
        {
            "response": {
                "total": "1",
                "data": [{"period": "not-a-date", "series": "WCESTUS1", "value": "1.0"}],
            }
        }
    ).encode("utf-8")
    result = parse_eia_payload(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert any("unparseable period" in w for w in result.warnings)


def test_unparseable_value_warns() -> None:
    spec = _spec()
    payload = json.dumps(
        {
            "response": {
                "total": "1",
                "data": [{"period": "2026-09-05", "series": "WCESTUS1", "value": "not-a-number"}],
            }
        }
    ).encode("utf-8")
    result = parse_eia_payload(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert any("unparseable value" in w for w in result.warnings)


# --------------------------------------------------------------------------
# availability precision correctness
# --------------------------------------------------------------------------


def test_availability_is_conservative_date_with_declared_lag() -> None:
    spec = _spec(lag_hours=48.0)
    result = parse_eia_payload(
        _fixture_bytes("petroleum_stoc_wstk_normal"), spec, retrieved_at=_RETRIEVED_AT
    )
    by_day = {o.observation_time.date(): o for o in result.observations}
    row = by_day[date(2026, 9, 5)]
    assert row.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    assert row.available_at == datetime(2026, 9, 7, tzinfo=UTC)
    assert row.revision_index is None
    assert row.source_release_time is None


# --------------------------------------------------------------------------
# fetch() -- credential handling (query parameter, like FRED, unlike GIE)
# --------------------------------------------------------------------------


def test_fetch_raises_typed_error_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    with pytest.raises(EiaCredentialError, match="EIA_API_KEY"):
        adapter.fetch(_spec())


@respx.mock
def test_fetch_sends_api_key_and_bracket_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "test-key-should-never-be-archived")
    route = respx.get(_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("petroleum_stoc_wstk_normal"))
    )
    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec())

    assert route.called
    sent = route.calls.last.request
    assert sent.url.params["api_key"] == "test-key-should-never-be-archived"
    assert sent.url.params["data[0]"] == "value"
    assert sent.url.params["facets[series][]"] == "WCESTUS1"
    assert sent.url.params["frequency"] == "weekly"
    assert isinstance(payload, FetchedPayload)
    assert payload.source == "eia"
    assert payload.http_status == 200


@respx.mock
def test_fetch_url_and_fingerprint_never_contain_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "totally-secret-eia-key-0123456789ab"
    monkeypatch.setenv("EIA_API_KEY", secret)
    respx.get(_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("petroleum_stoc_wstk_normal"))
    )
    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec())

    lowered = f"{payload.url} {payload.request_fingerprint}".lower()
    assert secret not in payload.url
    assert secret not in payload.request_fingerprint
    for marker in ("api_key=", "apikey=", "token=", "password="):
        assert marker not in lowered


@respx.mock
def test_fetch_since_sets_start_param(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "test-key")
    respx.get(_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("petroleum_stoc_wstk_normal"))
    )
    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    adapter.fetch(_spec(), since=date(2026, 1, 1))

    sent = respx.calls.last.request
    assert sent.url.params["start"] == "2026-01-01"


@respx.mock
def test_fetch_paginates_via_offset_against_total(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "test-key")

    def _page(offset: int, total: int, rows: list[dict[str, object]]) -> dict[str, object]:
        return {"response": {"total": str(total), "dateFormat": "YYYY-MM-DD", "data": rows}}

    # Simulate a route whose page size is artificially small by returning
    # fewer rows than _MAX_PAGE_LENGTH only on the final page; here we just
    # confirm the offset increases and pages are combined when the server
    # reports more total rows than the first page returned.
    row = {"period": "2026-09-05", "series": "WCESTUS1", "value": "1.0"}
    first_rows = [dict(row) for _ in range(5000)]
    second_rows = [{"period": "2026-09-06", "series": "WCESTUS1", "value": "2.0"}]

    route = respx.get(_URL)
    route.side_effect = [
        httpx.Response(200, json=_page(0, 5001, first_rows)),
        httpx.Response(200, json=_page(5000, 5001, second_rows)),
    ]
    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec())

    assert route.call_count == 2
    combined = json.loads(payload.content)
    assert len(combined["response"]["data"]) == 5001
    offsets_requested = [call.request.url.params.get("offset") for call in route.calls]
    assert offsets_requested == ["0", "5000"]


# --------------------------------------------------------------------------
# THE (ordinary) credential failure mode -- HTTP 403, not a 200 trap
# --------------------------------------------------------------------------


@respx.mock
def test_fetch_with_invalid_key_surfaces_as_adapter_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike AGSI/ALSI, EIA's credential failure is an ordinary non-2xx
    HTTP status; HttpClient's existing raise_for_status() handling already
    covers it -- no bespoke body-inspection is added in eia.py."""
    monkeypatch.setenv("EIA_API_KEY", "an-invalid-key")
    respx.get(_URL).mock(
        return_value=httpx.Response(403, content=_fixture_bytes("eia_error_api_key_missing"))
    )
    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0", max_retries=1))

    with pytest.raises(AdapterHttpError):
        adapter.fetch(_spec())


# --------------------------------------------------------------------------
# parse() does no network I/O; adapter identity
# --------------------------------------------------------------------------


def test_parse_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parse() must not perform network I/O")

    monkeypatch.setattr(httpx.Client, "request", _boom)

    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec()
    payload = FetchedPayload(
        source="eia",
        dataset=spec.series_id,
        url=_URL,
        content=_fixture_bytes("petroleum_stoc_wstk_normal"),
        http_status=200,
        content_type="application/json",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_URL}",
    )
    result = adapter.parse(payload, spec)
    assert len(result.observations) == 2


def test_adapter_identity() -> None:
    adapter = EiaAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    assert adapter.source_id == "eia"
    assert adapter.parser_version == "1"


def test_suggested_routes_are_the_ones_named_in_the_task_brief() -> None:
    assert set(SUGGESTED_ROUTES) == {
        "petroleum/stoc/wstk",
        "natural-gas/stor/wkly",
        "electricity/rto/daily-region-data",
        "petroleum/pri/spt",
    }
