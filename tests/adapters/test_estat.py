"""Tests for the Japanese e-Stat adapter (`adapters/estat.py`).

This source is an addition beyond the written specification (see the module
docstring). Offline by default: parsing tests replay JSON fixtures under
`tests/fixtures/external/estat/`. Those fixtures are **hand-built from the
published e-Stat API 3.0 documentation** (fetched during development), not
captured from a live response -- no `ESTAT_APP_ID` is available in this
environment. Only the fetch-shape tests touch HTTP, and even those are
mocked with `respx`; no test in this file makes a real network call.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters.base import HttpClient
from turboedge.adapters.estat import EstatAdapter, EstatCredentialError, parse_stats_data
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, RawPayload, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "estat"
_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
_RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.json").read_bytes()


def _spec(
    *,
    series_id: str = "prefecture_population",
    native_identifier: str = "0000000000",
    unit: str = "people",
    frequency: str = "annual",
    lag_hours: float = 24.0 * 7,
) -> SeriesSpec:
    return SeriesSpec(
        source="estat",
        series_id=series_id,
        name="Prefecture population (example)",
        category="demographics",
        unit=unit,
        frequency=frequency,
        native_identifier=native_identifier,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
        conservative_release_lag_hours=lag_hours,
    )


# --------------------------------------------------------------------------
# normal parse
# --------------------------------------------------------------------------


def test_parse_success_yields_one_observation_per_nonempty_cell() -> None:
    spec = _spec()
    result = parse_stats_data(
        _fixture_bytes("population_by_prefecture_success"), spec, retrieved_at=_RETRIEVED_AT
    )

    assert result.warnings == ()
    assert result.missing_series == ()
    # 4 VALUE rows in the fixture, one has an empty "$" -> 3 observations.
    assert len(result.observations) == 3


def test_parse_resolves_time_class_name_to_calendar_date() -> None:
    spec = _spec()
    result = parse_stats_data(
        _fixture_bytes("population_by_prefecture_success"), spec, retrieved_at=_RETRIEVED_AT
    )
    days = {o.observation_time.date() for o in result.observations}
    assert days == {date(2024, 1, 1), date(2025, 1, 1)}


def test_parse_disambiguates_series_id_by_varying_non_time_dimension() -> None:
    """The fixture varies `@area` (Tokyo vs Osaka) in addition to `@time` --
    both must survive as distinct series, not collide under one series_id."""
    spec = _spec()
    result = parse_stats_data(
        _fixture_bytes("population_by_prefecture_success"), spec, retrieved_at=_RETRIEVED_AT
    )
    series_ids = {o.series_id for o in result.observations}
    assert series_ids == {
        "prefecture_population#area=13000",
        "prefecture_population#area=27000",
    }


def test_parse_field_provenance() -> None:
    spec = _spec()
    result = parse_stats_data(
        _fixture_bytes("population_by_prefecture_success"), spec, retrieved_at=_RETRIEVED_AT
    )
    tokyo_2025 = next(
        o
        for o in result.observations
        if o.series_id == "prefecture_population#area=13000"
        and o.observation_time.date() == date(2025, 1, 1)
    )
    assert tokyo_2025.value == pytest.approx(14050000)
    assert tokyo_2025.unit == "people"
    assert tokyo_2025.frequency == "annual"
    assert tokyo_2025.source == "estat"
    assert tokyo_2025.source_version == "estat_getstatsdata_json_v3"
    assert tokyo_2025.parser_version == "1"
    assert tokyo_2025.quality_score == 1.0
    assert tokyo_2025.retrieved_at == _RETRIEVED_AT
    # No vintage/revision concept for e-Stat, and no fabricated release time.
    assert tokyo_2025.vintage_time is None
    assert tokyo_2025.revision_index is None
    assert tokyo_2025.source_release_time is None


# --------------------------------------------------------------------------
# missing value handling (empty "$")
# --------------------------------------------------------------------------


def test_empty_value_cell_is_skipped_without_a_warning() -> None:
    spec = _spec()
    result = parse_stats_data(
        _fixture_bytes("population_by_prefecture_success"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    osaka_series = [
        o for o in result.observations if o.series_id == "prefecture_population#area=27000"
    ]
    assert len(osaka_series) == 1  # 2025 cell is empty in the fixture and must be dropped
    assert osaka_series[0].observation_time.date() == date(2024, 1, 1)


# --------------------------------------------------------------------------
# STATUS error envelope -> warning, body never parsed as data
# --------------------------------------------------------------------------


def test_nonzero_status_is_an_error_and_body_is_not_parsed_as_data() -> None:
    spec = _spec()
    result = parse_stats_data(
        _fixture_bytes("unknown_stats_data_id_error"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "STATUS=100" in result.warnings[0]


@pytest.mark.parametrize("status", [0, 1, 2])
def test_status_zero_through_two_are_success(status: int) -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("population_by_prefecture_success"))
    payload["GET_STATS_DATA"]["RESULT"]["STATUS"] = status
    result = parse_stats_data(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)
    assert result.warnings == ()
    assert len(result.observations) == 3


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash
# --------------------------------------------------------------------------


def test_non_json_payload_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_stats_data(b"not json at all", spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "JSON" in result.warnings[0]


def test_missing_get_stats_data_key_warns() -> None:
    spec = _spec()
    payload = json.dumps({"unexpected": "shape"}).encode("utf-8")
    result = parse_stats_data(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "GET_STATS_DATA" in result.warnings[0]


def test_missing_result_status_warns() -> None:
    spec = _spec()
    payload = json.dumps({"GET_STATS_DATA": {"STATISTICAL_DATA": {}}}).encode("utf-8")
    result = parse_stats_data(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert "RESULT.STATUS" in result.warnings[0]


def test_missing_data_inf_value_warns() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("population_by_prefecture_success"))
    del payload["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]
    result = parse_stats_data(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert "DATA_INF.VALUE" in result.warnings[0]


def test_value_row_missing_time_dimension_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("population_by_prefecture_success"))
    del payload["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"][0]["@time"]
    result = parse_stats_data(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 2  # 3 - 1 dropped
    assert len(result.warnings) == 1
    assert "@time" in result.warnings[0]


def test_unresolvable_time_code_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("population_by_prefecture_success"))
    payload["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"][0]["@time"] = "Q1-FY2024"
    result = parse_stats_data(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 2
    assert len(result.warnings) == 1
    assert "could not resolve a calendar date" in result.warnings[0]


def test_non_numeric_nonempty_value_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("population_by_prefecture_success"))
    payload["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"][0]["$"] = "X"
    result = parse_stats_data(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 2
    assert len(result.warnings) == 1
    assert "unparseable VALUE" in result.warnings[0]


def test_single_member_dimension_and_single_row_are_not_list_wrapped_but_still_parse() -> None:
    """A dimension/row with exactly one member is a bare JSON object in
    e-Stat's response, not a one-item list (its origin is XML) -- this
    exercises `_as_list`'s normalisation for both `CLASS_OBJ` (one
    dimension: `time`) and `DATA_INF.VALUE` (one row), the two places a
    real single-category table would hit this shape."""
    spec = _spec()
    payload = {
        "GET_STATS_DATA": {
            "RESULT": {"STATUS": 0, "ERROR_MSG": ""},
            "STATISTICAL_DATA": {
                "CLASS_INF": {
                    "CLASS_OBJ": {
                        "@id": "time",
                        "@name": "time",
                        "CLASS": {"@code": "2030000000", "@name": "2030年", "@level": "1"},
                    }
                },
                "DATA_INF": {
                    "VALUE": {"@tab": "001", "@time": "2030000000", "@unit": "人", "$": "1000"}
                },
            },
        }
    }
    result = parse_stats_data(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)
    assert result.warnings == ()
    assert len(result.observations) == 1
    assert result.observations[0].observation_time.date() == date(2030, 1, 1)
    assert result.observations[0].value == pytest.approx(1000)


# --------------------------------------------------------------------------
# availability precision correctness
# --------------------------------------------------------------------------


def test_availability_is_conservative_date_with_declared_lag() -> None:
    spec = _spec(lag_hours=48.0)
    result = parse_stats_data(
        _fixture_bytes("population_by_prefecture_success"), spec, retrieved_at=_RETRIEVED_AT
    )
    obs_2025 = next(o for o in result.observations if o.observation_time.date() == date(2025, 1, 1))
    assert obs_2025.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    assert obs_2025.available_at == datetime(2025, 1, 3, tzinfo=UTC)
    assert obs_2025.available_at > obs_2025.observation_time


# --------------------------------------------------------------------------
# fetch() -- request shape, credential handling, credential-free fingerprint
# --------------------------------------------------------------------------


def test_fetch_raises_typed_error_when_app_id_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ESTAT_APP_ID", raising=False)
    adapter = EstatAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    with pytest.raises(EstatCredentialError, match="ESTAT_APP_ID"):
        adapter.fetch(_spec())


@respx.mock
def test_fetch_sends_stats_data_id_and_app_id_to_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESTAT_APP_ID", "test-app-id-should-never-be-archived")
    route = respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("population_by_prefecture_success"))
    )
    adapter = EstatAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec())

    assert route.called
    sent = route.calls.last.request
    assert sent.url.params["appId"] == "test-app-id-should-never-be-archived"
    assert sent.url.params["statsDataId"] == "0000000000"
    assert isinstance(payload, FetchedPayload)
    assert payload.source == "estat"
    assert payload.dataset == _spec().series_id
    assert payload.http_status == 200


@respx.mock
def test_fetch_url_and_fingerprint_never_contain_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "totally-secret-estat-app-id-0123456789"
    monkeypatch.setenv("ESTAT_APP_ID", secret)
    respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("population_by_prefecture_success"))
    )
    adapter = EstatAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec())

    lowered = f"{payload.url} {payload.request_fingerprint}".lower()
    assert secret not in payload.url
    assert secret not in payload.request_fingerprint
    for marker in ("appid=", "app_id=", "token=", "password="):
        assert marker not in lowered


def test_raw_payload_rejects_a_fingerprint_that_slipped_through_with_a_credential() -> None:
    with pytest.raises(ValueError, match="still carries a value"):
        RawPayload(
            payload_id="estat:prefecture_population:deadbeef",
            source="estat",
            dataset="prefecture_population",
            request_fingerprint=(
                "GET https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
                "?statsDataId=0000000000&appId=SECRET"
            ),
            retrieved_at=_RETRIEVED_AT,
            http_status=200,
            content_type="application/json",
            byte_size=10,
            payload_hash="a" * 64,
            stored_path="/tmp/x.json",
            parser_version="1",
        )


def test_parse_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """`parse()` must be pure: no HttpClient, no network access, ever."""
    import httpx as httpx_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parse() must not perform network I/O")

    monkeypatch.setattr(httpx_module.Client, "request", _boom)

    adapter = EstatAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec()
    payload = FetchedPayload(
        source="estat",
        dataset=spec.series_id,
        url=f"{_BASE_URL}?statsDataId=0000000000",
        content=_fixture_bytes("population_by_prefecture_success"),
        http_status=200,
        content_type="application/json",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_BASE_URL}?statsDataId=0000000000",
    )

    result = adapter.parse(payload, spec)
    assert len(result.observations) == 3


def test_adapter_identity() -> None:
    adapter = EstatAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    assert adapter.source_id == "estat"
    assert adapter.parser_version == "1"
