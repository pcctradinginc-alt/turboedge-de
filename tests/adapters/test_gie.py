"""Tests for the GIE AGSI/ALSI adapters (`adapters/gie.py`).

Offline by default. The two "normal" fixtures
(`agsi_de_normal.json`/`agsi_facility_de.json`/`alsi_be_normal.json`) are
**hand-built from the official GIE "User Manual: API access to AGSI/ALSI"
v007 (4 October 2022)** field tables and worked examples -- no `GIE_API_KEY`
is available in this environment, so no real data payload has been observed.

The two error fixtures (`agsi_error_no_key.json`/`alsi_error_no_key.json`)
are different: they are hand-built to match the documented/expected error
shape, but this exact byte-for-byte shape was *also* independently observed
on a real, unauthenticated (no key obtained or used -- the request needs
none) live request to `https://agsi.gie.eu/api` / `https://alsi.gie.eu/api`
during this workstream's development, on 2026-09-28. That live check
required no credential and is what "THE TRAP" in `adapters/gie.py`'s module
docstring documents. Only these two fixtures carry that additional live
confirmation; every other fixture in this directory is documentation-only.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters.base import HttpClient
from turboedge.adapters.gie import (
    AgsiAdapter,
    AlsiAdapter,
    GieApiError,
    GieCredentialError,
    build_native_identifier,
    parse_gie_payload,
    parse_native_identifier,
)
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "gie"
_AGSI_URL = "https://agsi.gie.eu/api"
_ALSI_URL = "https://alsi.gie.eu/api"
_RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.json").read_bytes()


def _spec(
    *,
    source: str = "agsi",
    series_id: str = "AGSI.DE.gasInStorage",
    country: str = "DE",
    facility: str = "",
    field: str = "gasInStorage",
    unit: str = "TWh",
    lag_hours: float = 48.0,
) -> SeriesSpec:
    return SeriesSpec(
        source=source,
        series_id=series_id,
        name="Germany gas in storage",
        category="energy",
        unit=unit,
        frequency="daily",
        native_identifier=build_native_identifier(country, facility, field),
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.FORWARD_ONLY,
        conservative_release_lag_hours=lag_hours,
    )


# --------------------------------------------------------------------------
# native_identifier round trip
# --------------------------------------------------------------------------


def test_build_and_parse_native_identifier_round_trip() -> None:
    raw = build_native_identifier("DE", "21W000000000078N", "gasInStorage")
    assert raw == "DE|21W000000000078N|gasInStorage"
    assert parse_native_identifier(raw) == ("DE", "21W000000000078N", "gasInStorage")


def test_build_native_identifier_allows_empty_facility() -> None:
    raw = build_native_identifier("DE", "", "full")
    assert raw == "DE||full"
    assert parse_native_identifier(raw) == ("DE", "", "full")


def test_parse_native_identifier_rejects_wrong_segment_count() -> None:
    with pytest.raises(Exception, match="does not match"):
        parse_native_identifier("DE|full")


# --------------------------------------------------------------------------
# normal parse -- country level (AGSI)
# --------------------------------------------------------------------------


def test_parse_agsi_country_level_gas_in_storage() -> None:
    spec = _spec(field="gasInStorage")
    result = parse_gie_payload(
        _fixture_bytes("agsi_de_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )

    # row1 (C), row2 (E) have gasInStorage; row3 (status N) always skipped;
    # row4 has gasInStorage=null -- skipped for THIS field specifically.
    assert len(result.observations) == 2
    by_day = {o.observation_time.date(): o for o in result.observations}
    assert by_day[date(2026, 9, 24)].value == pytest.approx(853.4219)
    assert by_day[date(2026, 9, 25)].value == pytest.approx(854.1000)
    assert all(o.series_id == spec.series_id for o in result.observations)
    assert all(o.unit == "TWh" for o in result.observations)
    assert all(o.source == "agsi" for o in result.observations)
    assert all(o.parser_version == "1" for o in result.observations)
    assert any("no usable value" in w for w in result.warnings)


def test_parse_agsi_status_e_gets_lower_quality_score_than_status_c() -> None:
    spec = _spec(field="gasInStorage")
    result = parse_gie_payload(
        _fixture_bytes("agsi_de_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    by_day = {o.observation_time.date(): o for o in result.observations}
    assert by_day[date(2026, 9, 24)].quality_score == pytest.approx(1.0)  # status C
    assert by_day[date(2026, 9, 25)].quality_score == pytest.approx(0.7)  # status E


def test_parse_agsi_status_n_row_skipped_for_any_field() -> None:
    """The 2026-09-26 row has status='N' ('no data') -- it must never
    surface, regardless of which field is requested."""
    spec = _spec(field="workingGasVolume")
    result = parse_gie_payload(
        _fixture_bytes("agsi_de_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    days = {o.observation_time.date() for o in result.observations}
    assert date(2026, 9, 26) not in days


def test_parse_agsi_full_field_present_on_row_where_gas_in_storage_is_null() -> None:
    """Row 2026-09-27 has gasInStorage=null but full='83.60' -- a missing
    value must only suppress the field actually requested."""
    spec = _spec(field="full", unit="pct")
    result = parse_gie_payload(
        _fixture_bytes("agsi_de_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    by_day = {o.observation_time.date(): o for o in result.observations}
    assert date(2026, 9, 27) in by_day
    assert by_day[date(2026, 9, 27)].value == pytest.approx(83.60)


# --------------------------------------------------------------------------
# normal parse -- facility level (AGSI)
# --------------------------------------------------------------------------


def test_parse_agsi_facility_level_matches_facility_code_not_country() -> None:
    spec = _spec(
        series_id="AGSI.DE.Haidach.full",
        country="DE",
        facility="21W000000000078N",
        field="full",
        unit="pct",
    )
    result = parse_gie_payload(
        _fixture_bytes("agsi_facility_de"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    assert result.warnings == ()
    assert len(result.observations) == 2
    values = {o.observation_time.date(): o.value for o in result.observations}
    assert values[date(2026, 9, 26)] == pytest.approx(41.13)
    assert values[date(2026, 9, 27)] == pytest.approx(41.67)


def test_parse_agsi_code_mismatch_produces_warning_and_missing_series() -> None:
    """Requesting country=FR against a payload whose rows are all code='DE'
    must not silently return France's non-existent data."""
    spec = _spec(country="FR", field="gasInStorage")
    result = parse_gie_payload(
        _fixture_bytes("agsi_de_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("no rows in payload matched code" in w for w in result.warnings)


# --------------------------------------------------------------------------
# normal parse -- ALSI
# --------------------------------------------------------------------------


def test_parse_alsi_inventory() -> None:
    spec = _spec(
        source="alsi", series_id="ALSI.BE.inventory", country="BE", field="inventory", unit="km3"
    )
    result = parse_gie_payload(
        _fixture_bytes("alsi_be_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="alsi"
    )
    assert result.warnings == ()
    assert len(result.observations) == 3
    by_day = {o.observation_time.date(): o.value for o in result.observations}
    assert by_day[date(2026, 9, 27)] == pytest.approx(270.55)
    assert all(o.source == "alsi" for o in result.observations)
    # ALSI rows have no 'status' field -- quality_score stays at the default.
    assert all(o.quality_score == pytest.approx(1.0) for o in result.observations)


def test_parse_alsi_send_out_null_row_skipped() -> None:
    spec = _spec(
        source="alsi", series_id="ALSI.BE.sendOut", country="BE", field="sendOut", unit="GWh/d"
    )
    result = parse_gie_payload(
        _fixture_bytes("alsi_be_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="alsi"
    )
    assert len(result.observations) == 2
    days = {o.observation_time.date() for o in result.observations}
    assert date(2026, 9, 26) not in days
    assert any("no usable value" in w for w in result.warnings)


# --------------------------------------------------------------------------
# THE TRAP -- HTTP 200 with an error body
# --------------------------------------------------------------------------


def test_agsi_trap_http_200_with_error_body_raises() -> None:
    spec = _spec(field="gasInStorage")
    with pytest.raises(GieApiError, match=r"access denied|Invalid or missing API key"):
        parse_gie_payload(
            _fixture_bytes("agsi_error_no_key"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
        )


def test_alsi_trap_http_200_with_error_body_raises() -> None:
    spec = _spec(source="alsi", series_id="ALSI.BE.inventory", country="BE", field="inventory")
    with pytest.raises(GieApiError, match=r"access denied|Invalid or missing API key"):
        parse_gie_payload(
            _fixture_bytes("alsi_error_no_key"), spec, retrieved_at=_RETRIEVED_AT, source_id="alsi"
        )


def test_trap_is_never_read_as_a_healthy_empty_result() -> None:
    """The whole point of the trap: `total: 0` and `data: []` alone must
    never be mistaken for 'the publisher had no data today' when an 'error'
    key is present."""
    spec = _spec(field="gasInStorage")
    payload = json.loads(_fixture_bytes("agsi_error_no_key"))
    assert payload["total"] == 0
    assert payload["data"] == []
    assert "error" in payload
    with pytest.raises(GieApiError):
        parse_gie_payload(
            json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
        )


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash (except the trap, which always raises)
# --------------------------------------------------------------------------


def test_non_json_payload_warns_and_returns_no_observations() -> None:
    spec = _spec(field="gasInStorage")
    result = parse_gie_payload(
        b"not json at all", spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "JSON" in result.warnings[0]


def test_missing_data_key_warns_and_returns_no_observations() -> None:
    spec = _spec(field="gasInStorage")
    payload = json.dumps({"last_page": 1, "total": 0}).encode("utf-8")
    result = parse_gie_payload(payload, spec, retrieved_at=_RETRIEVED_AT, source_id="agsi")
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)


def test_data_as_single_object_is_handled_defensively() -> None:
    """A single-day ('date=') query returns 'data' as one object rather than
    an array (v007 manual p.15 vs p.16). fetch() never sends 'date=', but
    parse() must not crash if such a payload is ever replayed."""
    spec = _spec(field="gasInStorage")
    payload = json.dumps(
        {
            "last_page": 1,
            "total": 1,
            "dataset": "DE",
            "data": {
                "name": "Germany",
                "code": "DE",
                "gasDayStart": "2026-09-27",
                "gasInStorage": "853.4219",
                "status": "C",
            },
        }
    ).encode("utf-8")
    result = parse_gie_payload(payload, spec, retrieved_at=_RETRIEVED_AT, source_id="agsi")
    assert len(result.observations) == 1
    assert result.observations[0].value == pytest.approx(853.4219)


def test_data_as_unexpected_type_warns_and_returns_no_observations() -> None:
    spec = _spec(field="gasInStorage")
    payload = json.dumps({"last_page": 1, "total": 0, "data": "not-a-list-or-object"}).encode(
        "utf-8"
    )
    result = parse_gie_payload(payload, spec, retrieved_at=_RETRIEVED_AT, source_id="agsi")
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)


def test_non_object_row_in_data_is_a_warning_not_a_crash() -> None:
    spec = _spec(field="gasInStorage")
    payload = json.dumps({"last_page": 1, "total": 1, "data": ["not-a-row"]}).encode("utf-8")
    result = parse_gie_payload(payload, spec, retrieved_at=_RETRIEVED_AT, source_id="agsi")
    assert result.observations == []
    assert any("non-object entry" in w for w in result.warnings)


# --------------------------------------------------------------------------
# number format -- documented period/comma ambiguity (module docstring)
# --------------------------------------------------------------------------


def test_comma_decimal_value_is_parsed_via_fallback() -> None:
    spec = _spec(field="gasInStorage")
    payload = json.loads(_fixture_bytes("agsi_de_normal"))
    payload["data"][0]["gasInStorage"] = "63,9469"
    result = parse_gie_payload(
        json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    by_day = {o.observation_time.date(): o.value for o in result.observations}
    assert by_day[date(2026, 9, 24)] == pytest.approx(63.9469)


def test_unparseable_value_is_a_warning_not_a_crash() -> None:
    spec = _spec(field="gasInStorage")
    payload = json.loads(_fixture_bytes("agsi_de_normal"))
    payload["data"][0]["gasInStorage"] = "not-a-number"
    result = parse_gie_payload(
        json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    assert any("unparseable value" in w for w in result.warnings)
    days = {o.observation_time.date() for o in result.observations}
    assert date(2026, 9, 24) not in days


# --------------------------------------------------------------------------
# availability precision correctness
# --------------------------------------------------------------------------


def test_availability_is_conservative_date_from_gas_day_with_declared_lag() -> None:
    spec = _spec(field="gasInStorage", lag_hours=48.0)
    result = parse_gie_payload(
        _fixture_bytes("agsi_de_normal"), spec, retrieved_at=_RETRIEVED_AT, source_id="agsi"
    )
    by_day = {o.observation_time.date(): o for o in result.observations}
    row = by_day[date(2026, 9, 24)]
    assert row.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    assert row.available_at == datetime(2026, 9, 26, tzinfo=UTC)
    assert row.revision_index is None
    assert row.source_release_time is None


# --------------------------------------------------------------------------
# fetch() -- credential handling
# --------------------------------------------------------------------------


def test_agsi_fetch_raises_typed_error_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GIE_API_KEY", raising=False)
    adapter = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    with pytest.raises(GieCredentialError, match="GIE_API_KEY"):
        adapter.fetch(_spec(field="gasInStorage"))


def test_alsi_fetch_raises_typed_error_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GIE_API_KEY", raising=False)
    adapter = AlsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    with pytest.raises(GieCredentialError, match="GIE_API_KEY"):
        adapter.fetch(_spec(source="alsi", country="BE", field="inventory"))


@respx.mock
def test_fetch_sends_key_in_x_key_header_not_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "totally-secret-gie-key-0123456789ab"
    monkeypatch.setenv("GIE_API_KEY", secret)
    route = respx.get(_AGSI_URL).mock(
        return_value=httpx.Response(
            200,
            content=_fixture_bytes("agsi_de_normal"),
            headers={"date": "Mon, 28 Sep 2026 12:00:00 GMT"},
        )
    )
    adapter = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec(field="gasInStorage"))

    assert route.called
    sent = route.calls.last.request
    assert sent.headers["x-key"] == secret
    assert "x-key" not in sent.url.params
    assert "api_key" not in sent.url.params
    assert isinstance(payload, FetchedPayload)
    assert payload.source == "agsi"
    assert payload.http_status == 200


@respx.mock
def test_fetch_url_fingerprint_and_headers_never_contain_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "totally-secret-gie-key-0123456789ab"
    monkeypatch.setenv("GIE_API_KEY", secret)
    respx.get(_AGSI_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("agsi_de_normal"))
    )
    adapter = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec(field="gasInStorage"))

    assert secret not in payload.url
    assert secret not in payload.request_fingerprint
    assert secret not in json.dumps(payload.headers)
    lowered = f"{payload.url} {payload.request_fingerprint} {payload.headers}".lower()
    for marker in ("x-key=", "api_key=", "apikey=", "token="):
        assert marker not in lowered


@respx.mock
def test_fetch_includes_country_and_facility_params_when_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIE_API_KEY", "test-key")
    route = respx.get(_AGSI_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("agsi_facility_de"))
    )
    adapter = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    adapter.fetch(
        _spec(
            series_id="AGSI.DE.Haidach.full",
            country="DE",
            facility="21W000000000078N",
            field="full",
        )
    )

    sent = route.calls.last.request
    assert sent.url.params["country"] == "DE"
    assert sent.url.params["facility"] == "21W000000000078N"
    assert sent.url.params["size"] == "300"


@respx.mock
def test_fetch_paginates_and_archives_one_combined_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIE_API_KEY", "test-key")
    row1 = {
        "name": "Germany",
        "code": "DE",
        "gasDayStart": "2026-09-24",
        "gasInStorage": "1.0",
        "status": "C",
    }
    row2 = {
        "name": "Germany",
        "code": "DE",
        "gasDayStart": "2026-09-25",
        "gasInStorage": "2.0",
        "status": "C",
    }
    page1 = {"last_page": 2, "total": 1, "dataset": "DE", "data": [row1]}
    page2 = {"last_page": 2, "total": 1, "dataset": "DE", "data": [row2]}
    route = respx.get(_AGSI_URL)
    route.side_effect = [
        httpx.Response(200, json=page1),
        httpx.Response(200, json=page2),
    ]
    adapter = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec(field="gasInStorage"))

    assert route.call_count == 2
    combined = json.loads(payload.content)
    assert len(combined["data"]) == 2
    parsed = adapter.parse(payload, _spec(field="gasInStorage"))
    assert len(parsed.observations) == 2


@respx.mock
def test_fetch_stops_after_one_request_on_the_trap(monkeypatch: pytest.MonkeyPatch) -> None:
    """last_page=0 in the trap body must not cause extra requests."""
    monkeypatch.setenv("GIE_API_KEY", "wrong-or-missing")
    route = respx.get(_AGSI_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("agsi_error_no_key"))
    )
    adapter = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec(field="gasInStorage"))

    assert route.call_count == 1
    with pytest.raises(GieApiError):
        adapter.parse(payload, _spec(field="gasInStorage"))


# --------------------------------------------------------------------------
# parse() does no network I/O; adapter identity
# --------------------------------------------------------------------------


def test_parse_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parse() must not perform network I/O")

    monkeypatch.setattr(httpx.Client, "request", _boom)

    adapter = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec(field="gasInStorage")
    payload = FetchedPayload(
        source="agsi",
        dataset=spec.series_id,
        url=_AGSI_URL,
        content=_fixture_bytes("agsi_de_normal"),
        http_status=200,
        content_type="application/json",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_AGSI_URL}",
    )
    result = adapter.parse(payload, spec)
    assert len(result.observations) == 2


def test_adapter_identity() -> None:
    agsi = AgsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    alsi = AlsiAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    assert agsi.source_id == "agsi"
    assert alsi.source_id == "alsi"
    assert agsi.parser_version == "1"
    assert alsi.parser_version == "1"
