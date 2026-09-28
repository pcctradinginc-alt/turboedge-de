"""Tests for the IMF PortWatch adapter (`adapters/portwatch.py`).

Offline by default: parsing tests replay real, captured ArcGIS FeatureServer
response pages under `tests/fixtures/external/portwatch/` (each either a
direct live 2026-09-28 capture, wrapped in the one-JSON-array-per-`fetch()`
archival shape this adapter uses, or -- where noted -- a hand-crafted
payload exercising a shape never observed live, such as an epoch-millis
date or a naturally-completing two-page series). Only the fetch-shape tests
touch HTTP, and even those are mocked with `respx` -- no test in this file
makes a real network call.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters.base import AdapterError, AdapterHttpError, HttpClient
from turboedge.adapters.portwatch import (
    PortWatchAdapter,
    build_native_identifier,
    parse_native_identifier,
    parse_portwatch_pages,
)
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "portwatch"
_BASE_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
_RETRIEVED_AT = datetime(2026, 9, 28, 16, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.json").read_bytes()


def _pages_payload(*pages: dict) -> bytes:
    return json.dumps(list(pages)).encode("utf-8")


def _spec(
    *,
    series_id: str = "portwatch_suez_canal_n_cargo",
    native_identifier: str = "Daily_Chokepoints_Data|n_cargo|portname=Suez Canal",
    unit: str = "vessel_count",
    frequency: str = "daily",
    lag_hours: float = 24.0,
) -> SeriesSpec:
    return SeriesSpec(
        source="portwatch",
        series_id=series_id,
        name="Suez Canal daily cargo vessel transits (IMF PortWatch)",
        category="trade_flow",
        unit=unit,
        frequency=frequency,
        native_identifier=native_identifier,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.FORWARD_ONLY,
        conservative_release_lag_hours=lag_hours,
    )


# --------------------------------------------------------------------------
# native_identifier build/parse
# --------------------------------------------------------------------------


def test_build_and_parse_native_identifier_round_trip_with_filter() -> None:
    raw = build_native_identifier(
        "Daily_Chokepoints_Data", "n_cargo", where_field="portname", where_value="Suez Canal"
    )
    assert raw == "Daily_Chokepoints_Data|n_cargo|portname=Suez Canal"
    assert parse_native_identifier(raw) == (
        "Daily_Chokepoints_Data",
        "n_cargo",
        "portname",
        "Suez Canal",
    )


def test_build_and_parse_native_identifier_round_trip_no_filter() -> None:
    raw = build_native_identifier("Daily_Trade_Data_WLD", "portcalls")
    assert raw == "Daily_Trade_Data_WLD|portcalls|"
    assert parse_native_identifier(raw) == ("Daily_Trade_Data_WLD", "portcalls", "", "")


def test_build_native_identifier_rejects_unpaired_where_clause() -> None:
    with pytest.raises(AdapterError):
        build_native_identifier("Daily_Chokepoints_Data", "n_cargo", where_field="portname")
    with pytest.raises(AdapterError):
        build_native_identifier("Daily_Chokepoints_Data", "n_cargo", where_value="Suez Canal")


def test_parse_native_identifier_rejects_wrong_segment_count() -> None:
    with pytest.raises(AdapterError, match="does not match"):
        parse_native_identifier("Daily_Chokepoints_Data|n_cargo")
    with pytest.raises(AdapterError, match="does not match"):
        parse_native_identifier("Daily_Chokepoints_Data|n_cargo|portname=Suez Canal|extra")


def test_parse_native_identifier_rejects_malformed_where_clause() -> None:
    with pytest.raises(AdapterError):
        parse_native_identifier("Daily_Chokepoints_Data|n_cargo|portname")
    with pytest.raises(AdapterError):
        parse_native_identifier("Daily_Chokepoints_Data|n_cargo|=Suez Canal")


# --------------------------------------------------------------------------
# normal parse -- live-captured, single page
# --------------------------------------------------------------------------


def test_parse_suez_canal_n_cargo_normal() -> None:
    """This fixture is a real single-page, 30-row capture (`resultRecordCount=30`)
    of a series with thousands of rows of history, so ArcGIS's own
    `exceededTransferLimit=true` is genuinely present and correctly surfaced
    as the "incomplete slice" warning -- not a schema problem."""
    spec = _spec()
    result = parse_portwatch_pages(
        _fixture_bytes("suez_canal_n_cargo_normal"), spec, retrieved_at=_RETRIEVED_AT
    )

    assert len(result.warnings) == 1
    assert "incomplete" in result.warnings[0]
    assert result.missing_series == ()
    by_day = {o.observation_time.date(): o for o in result.observations}
    latest = by_day[date(2026, 9, 20)]
    assert latest.value == pytest.approx(23.0)
    assert latest.series_id == "portwatch_suez_canal_n_cargo"
    assert latest.unit == "vessel_count"
    assert latest.frequency == "daily"
    assert latest.source == "portwatch"
    assert latest.source_version == "portwatch_daily_chokepoints_data"
    assert latest.quality_score == 1.0
    assert latest.is_stale is False
    assert latest.source_release_time is None
    assert latest.revision_index is None


def test_parse_trade_wld_portcalls_normal_no_filter() -> None:
    spec = _spec(
        series_id="portwatch_wld_portcalls",
        native_identifier="Daily_Trade_Data_WLD|portcalls|",
        unit="port_calls",
    )
    result = parse_portwatch_pages(
        _fixture_bytes("trade_wld_portcalls_normal"), spec, retrieved_at=_RETRIEVED_AT
    )
    # Same real 30-row/thousands-of-rows situation as the chokepoints fixture above.
    assert len(result.warnings) == 1
    assert "incomplete" in result.warnings[0]
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2025, 4, 25)
    assert latest.value == pytest.approx(4676.0)
    assert latest.source_version == "portwatch_daily_trade_data_wld"


# --------------------------------------------------------------------------
# paging: exceededTransferLimit handling
# --------------------------------------------------------------------------


def test_paging_merges_pages_that_complete_naturally() -> None:
    """Synthetic (not live): two pages, the second reporting
    exceededTransferLimit=false -- the natural end of a series."""
    spec = _spec()
    result = parse_portwatch_pages(
        _fixture_bytes("suez_canal_n_cargo_paged_complete"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    assert result.missing_series == ()
    assert len(result.observations) == 4
    values_by_day = {o.observation_time.date(): o.value for o in result.observations}
    assert values_by_day == {
        date(2026, 9, 1): 20.0,
        date(2026, 9, 2): 21.0,
        date(2026, 9, 3): 22.0,
        date(2026, 9, 4): 23.0,
    }


def test_paging_still_exceeded_on_last_page_warns_incomplete() -> None:
    """Live-captured (resultRecordCount=5, resultOffset 0 then 5): both real
    pages report exceededTransferLimit=true, meaning fetch() stopped (e.g.
    hit max_pages) before the upstream series was exhausted."""
    spec = _spec()
    result = parse_portwatch_pages(
        _fixture_bytes("suez_canal_n_cargo_paged_incomplete"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert len(result.observations) == 10
    assert result.missing_series == ()
    assert len(result.warnings) == 1
    assert "incomplete" in result.warnings[0]
    by_day = {o.observation_time.date(): o.value for o in result.observations}
    assert by_day[date(2019, 1, 1)] == 43.0
    assert by_day[date(2019, 1, 10)] == 33.0


# --------------------------------------------------------------------------
# unknown chokepoint -> missing_series
# --------------------------------------------------------------------------


def test_unknown_chokepoint_is_missing_series_not_a_crash() -> None:
    spec = _spec(native_identifier="Daily_Chokepoints_Data|n_cargo|portname=Not A Real Strait")
    result = parse_portwatch_pages(
        _fixture_bytes("unknown_chokepoint"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "Not A Real Strait" in result.warnings[0]


# --------------------------------------------------------------------------
# date-format switch: ISO string (verified live) vs epoch millis (defensive)
# --------------------------------------------------------------------------


def test_epoch_millis_date_is_handled_with_a_warning_not_a_crash() -> None:
    """Hand-crafted: every live response used an ISO date string (module
    docstring), but a future upstream change back to the older
    esriFieldTypeDate convention would serialize `date` as epoch
    milliseconds. 1600128000000 ms == 2020-09-15T00:00:00Z."""
    spec = _spec()
    page = {
        "exceededTransferLimit": False,
        "features": [
            {"attributes": {"date": 1600128000000, "portname": "Suez Canal", "n_cargo": 12}},
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)

    assert len(result.observations) == 1
    assert result.observations[0].observation_time.date() == date(2020, 9, 15)
    assert result.observations[0].value == pytest.approx(12.0)
    assert len(result.warnings) == 1
    assert "epoch" in result.warnings[0]


def test_epoch_millis_warning_is_not_repeated_per_row() -> None:
    spec = _spec()
    page = {
        "exceededTransferLimit": False,
        "features": [
            {"attributes": {"date": 1600128000000, "portname": "Suez Canal", "n_cargo": 12}},
            {"attributes": {"date": 1600214400000, "portname": "Suez Canal", "n_cargo": 13}},
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 2
    assert len(result.warnings) == 1


def test_unparseable_date_string_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    page = {
        "exceededTransferLimit": False,
        "features": [
            {"attributes": {"date": "not-a-date", "portname": "Suez Canal", "n_cargo": 12}},
            {"attributes": {"date": "2026-09-20", "portname": "Suez Canal", "n_cargo": 23}},
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 1
    assert len(result.warnings) == 1
    assert "unparseable date" in result.warnings[0]


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash
# --------------------------------------------------------------------------


def test_non_utf8_payload_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_portwatch_pages(b"\xff\xfe not valid utf-8", spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1


def test_non_list_top_level_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_portwatch_pages(b'{"features": []}', spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)


def test_empty_list_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_portwatch_pages(b"[]", spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)


def test_arcgis_error_page_is_a_warning_and_missing_series() -> None:
    spec = _spec()
    page = {"error": {"code": 400, "message": "Invalid where clause"}}
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "error" in result.warnings[0].lower()


def test_page_missing_features_key_is_a_warning() -> None:
    spec = _spec()
    page = {"exceededTransferLimit": False}
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "'features'" in result.warnings[0]


def test_feature_missing_attributes_is_skipped_with_a_warning() -> None:
    spec = _spec()
    page = {
        "exceededTransferLimit": False,
        "features": [
            {"geometry": None},
            {"attributes": {"date": "2026-09-20", "portname": "Suez Canal", "n_cargo": 23}},
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 1
    assert len(result.warnings) == 1
    assert "attributes" in result.warnings[0]


def test_feature_missing_required_field_is_skipped_with_a_warning() -> None:
    spec = _spec()
    page = {
        "exceededTransferLimit": False,
        "features": [
            {"attributes": {"date": "2026-09-20", "portname": "Suez Canal"}},  # no n_cargo
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "missing required attribute" in result.warnings[0]


def test_non_numeric_value_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    page = {
        "exceededTransferLimit": False,
        "features": [
            {"attributes": {"date": "2026-09-20", "portname": "Suez Canal", "n_cargo": "lots"}},
            {"attributes": {"date": "2026-09-21", "portname": "Suez Canal", "n_cargo": 30}},
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 1
    assert len(result.warnings) == 1
    assert "non-numeric" in result.warnings[0]


def test_null_value_is_a_normal_quiet_day_not_a_warning() -> None:
    spec = _spec()
    page = {
        "exceededTransferLimit": False,
        "features": [
            {"attributes": {"date": "2026-09-20", "portname": "Suez Canal", "n_cargo": None}},
            {"attributes": {"date": "2026-09-21", "portname": "Suez Canal", "n_cargo": 30}},
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 1
    assert result.warnings == ()


def test_conflicting_values_for_same_day_across_pages_are_skipped() -> None:
    spec = _spec()
    page0 = {
        "exceededTransferLimit": True,
        "features": [
            {"attributes": {"date": "2026-09-20", "portname": "Suez Canal", "n_cargo": 23}}
        ],
    }
    page1 = {
        "exceededTransferLimit": False,
        "features": [
            {"attributes": {"date": "2026-09-20", "portname": "Suez Canal", "n_cargo": 99}}
        ],
    }
    result = parse_portwatch_pages(_pages_payload(page0, page1), spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert any("conflicting" in w for w in result.warnings)


# --------------------------------------------------------------------------
# availability precision
# --------------------------------------------------------------------------


def test_availability_precision_is_conservative_date_with_declared_lag() -> None:
    spec = _spec(lag_hours=20.0)
    result = parse_portwatch_pages(
        _fixture_bytes("suez_canal_n_cargo_normal"), spec, retrieved_at=_RETRIEVED_AT
    )
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    expected_available_at = datetime(2026, 9, 20, 20, 0, tzinfo=UTC)
    assert latest.available_at == expected_available_at
    assert latest.available_at > latest.observation_time


# --------------------------------------------------------------------------
# vintage handling (best-effort, from the HTTP Date header; never invented)
# --------------------------------------------------------------------------


def test_vintage_time_comes_from_response_date_header_when_present() -> None:
    spec = _spec()
    result = parse_portwatch_pages(
        _fixture_bytes("suez_canal_n_cargo_normal"),
        spec,
        retrieved_at=_RETRIEVED_AT,
        headers={"date": "Mon, 28 Sep 2026 14:00:00 GMT"},
    )
    expected = datetime(2026, 9, 28, 14, 0, 0, tzinfo=UTC)
    for obs in result.observations:
        assert obs.vintage_time == expected


def test_vintage_time_is_none_without_a_date_header() -> None:
    spec = _spec()
    result = parse_portwatch_pages(
        _fixture_bytes("suez_canal_n_cargo_normal"), spec, retrieved_at=_RETRIEVED_AT, headers=None
    )
    for obs in result.observations:
        assert obs.vintage_time is None


# --------------------------------------------------------------------------
# fetch() -- request shape, paging loop, max_pages guard, HTTP 404
# --------------------------------------------------------------------------


@respx.mock
def test_fetch_single_page_request_shape() -> None:
    spec = _spec()
    page = {"exceededTransferLimit": False, "features": []}
    route = respx.get(f"{_BASE_URL}/Daily_Chokepoints_Data/FeatureServer/0/query").mock(
        return_value=httpx.Response(
            200, json=page, headers={"date": "Mon, 28 Sep 2026 14:00:00 GMT"}
        )
    )
    adapter = PortWatchAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    payload = adapter.fetch(spec)

    assert route.call_count == 1
    sent = route.calls.last.request
    assert sent.url.params["where"] == "portname = 'Suez Canal'"
    assert sent.url.params["outFields"] == "date,n_cargo"
    assert sent.url.params["orderByFields"] == "date ASC"
    assert sent.url.params["returnGeometry"] == "false"
    assert sent.url.params["f"] == "json"
    assert sent.url.params["resultOffset"] == "0"

    assert isinstance(payload, FetchedPayload)
    assert payload.source == "portwatch"
    assert payload.dataset == spec.series_id
    assert payload.content_type == "application/json"
    assert payload.headers.get("date") == "Mon, 28 Sep 2026 14:00:00 GMT"
    pages = json.loads(payload.content)
    assert pages == [page]
    assert "api_key" not in payload.url.lower()


@respx.mock
def test_fetch_pages_until_exceeded_transfer_limit_is_false() -> None:
    spec = _spec()
    page0 = {
        "exceededTransferLimit": True,
        "features": [{"attributes": {"date": "2026-09-01", "n_cargo": 1}}] * 2,
    }
    page1 = {
        "exceededTransferLimit": False,
        "features": [{"attributes": {"date": "2026-09-03", "n_cargo": 3}}],
    }
    route = respx.get(f"{_BASE_URL}/Daily_Chokepoints_Data/FeatureServer/0/query").mock(
        side_effect=[httpx.Response(200, json=page0), httpx.Response(200, json=page1)]
    )
    adapter = PortWatchAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    payload = adapter.fetch(spec)

    assert route.call_count == 2
    first_offset = route.calls[0].request.url.params["resultOffset"]
    second_offset = route.calls[1].request.url.params["resultOffset"]
    assert first_offset == "0"
    assert second_offset == "2"  # advanced by len(page0['features'])
    pages = json.loads(payload.content)
    assert pages == [page0, page1]


@respx.mock
def test_fetch_stops_at_max_pages_guard() -> None:
    spec = _spec()
    always_exceeded = {
        "exceededTransferLimit": True,
        "features": [{"attributes": {"date": "2026-09-01", "n_cargo": 1}}],
    }
    route = respx.get(f"{_BASE_URL}/Daily_Chokepoints_Data/FeatureServer/0/query").mock(
        return_value=httpx.Response(200, json=always_exceeded)
    )
    adapter = PortWatchAdapter(
        HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL, max_pages=3
    )

    payload = adapter.fetch(spec)

    assert route.call_count == 3
    pages = json.loads(payload.content)
    assert len(pages) == 3
    # parse() sees exceededTransferLimit still true on the last archived
    # page and reports the series as an incomplete slice.
    result = parse_portwatch_pages(payload.content, spec, retrieved_at=_RETRIEVED_AT)
    assert any("incomplete" in w for w in result.warnings)


@respx.mock
def test_fetch_applies_since_as_an_additional_date_filter() -> None:
    spec = _spec()
    page = {"exceededTransferLimit": False, "features": []}
    respx.get(f"{_BASE_URL}/Daily_Chokepoints_Data/FeatureServer/0/query").mock(
        return_value=httpx.Response(200, json=page)
    )
    adapter = PortWatchAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    adapter.fetch(spec, since=date(2026, 9, 1))

    sent = respx.calls.last.request
    where = sent.url.params["where"]
    assert "portname = 'Suez Canal'" in where
    assert "date >= '2026-09-01'" in where


@respx.mock
def test_fetch_no_filter_series_uses_1_equals_1() -> None:
    spec = _spec(native_identifier="Daily_Trade_Data_WLD|portcalls|", series_id="wld")
    page = {"exceededTransferLimit": False, "features": []}
    respx.get(f"{_BASE_URL}/Daily_Trade_Data_WLD/FeatureServer/0/query").mock(
        return_value=httpx.Response(200, json=page)
    )
    adapter = PortWatchAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    adapter.fetch(spec)

    sent = respx.calls.last.request
    assert sent.url.params["where"] == "1=1"
    assert sent.url.params["outFields"] == "date,portcalls"


@respx.mock
def test_fetch_raises_adapter_http_error_on_404() -> None:
    spec = _spec()
    respx.get(f"{_BASE_URL}/Daily_Chokepoints_Data/FeatureServer/0/query").mock(
        return_value=httpx.Response(404)
    )
    adapter = PortWatchAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    with pytest.raises(AdapterHttpError):
        adapter.fetch(spec)


# --------------------------------------------------------------------------
# parse() does no network I/O
# --------------------------------------------------------------------------


def test_parse_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """`parse()` must be pure: no HttpClient, no network access, ever."""
    import httpx as httpx_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parse() must not perform network I/O")

    monkeypatch.setattr(httpx_module.Client, "request", _boom)

    adapter = PortWatchAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec()
    payload = FetchedPayload(
        source="portwatch",
        dataset=spec.series_id,
        url=f"{_BASE_URL}/Daily_Chokepoints_Data/FeatureServer/0/query?where=portname%3D%27Suez+Canal%27",
        content=_fixture_bytes("suez_canal_n_cargo_normal"),
        http_status=200,
        content_type="application/json",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint="GET .../query (paged: 1 page(s), final resultOffset=0)",
    )

    result = adapter.parse(payload, spec)

    assert result.observations
