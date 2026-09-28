"""Tests for the general ECB Data Portal adapter (`adapters/ecb_data.py`).

Offline by default: parsing tests replay real, captured SDMX-JSON fixtures
under `tests/fixtures/external/ecb/`. Only the fetch-shape tests touch HTTP,
and even those are mocked with `respx` -- no test in this file makes a real
network call.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters.base import HttpClient
from turboedge.adapters.ecb_data import (
    EcbDataAdapter,
    parse_sdmx_json,
    split_native_identifier,
)
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "ecb"
_BASE_URL = "https://data-api.ecb.europa.eu"
_RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.json").read_bytes()


def _spec(
    *,
    series_id: str = "eur_usd_reference_rate",
    native_identifier: str = "EXR/D.USD.EUR.SP00.A",
    unit: str = "usd_per_eur",
    frequency: str = "daily",
    lag_hours: float = 20.0,
) -> SeriesSpec:
    return SeriesSpec(
        source="ecb",
        series_id=series_id,
        name="USD/EUR reference rate",
        category="fx",
        unit=unit,
        frequency=frequency,
        native_identifier=native_identifier,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.FORWARD_ONLY,
        conservative_release_lag_hours=lag_hours,
    )


# --------------------------------------------------------------------------
# split_native_identifier
# --------------------------------------------------------------------------


def test_split_native_identifier_splits_on_first_slash() -> None:
    assert split_native_identifier("EXR/D.USD.EUR.SP00.A") == ("EXR", "D.USD.EUR.SP00.A")


def test_split_native_identifier_rejects_missing_slash() -> None:
    with pytest.raises(ValueError, match="FLOW/KEY"):
        split_native_identifier("EXR-D.USD.EUR.SP00.A")


def test_split_native_identifier_rejects_empty_flow_or_key() -> None:
    with pytest.raises(ValueError):
        split_native_identifier("/D.USD.EUR.SP00.A")
    with pytest.raises(ValueError):
        split_native_identifier("EXR/")


# --------------------------------------------------------------------------
# parse_sdmx_json -- normal parse (daily and monthly)
# --------------------------------------------------------------------------


def test_parse_daily_series_exr_usd() -> None:
    spec = _spec()
    result = parse_sdmx_json(_fixture_bytes("exr_usd"), spec, retrieved_at=_RETRIEVED_AT)

    assert result.warnings == ()
    assert result.missing_series == ()
    assert len(result.observations) == 5

    by_day = {o.observation_time.date(): o for o in result.observations}
    latest = by_day[date(2026, 9, 25)]
    assert latest.value == pytest.approx(1.1403)
    assert latest.series_id == "eur_usd_reference_rate"
    assert latest.unit == "usd_per_eur"
    assert latest.frequency == "daily"
    assert latest.source == "ecb"
    assert latest.source_version == "ecb_sdmx_jsondata_exr"
    assert latest.quality_score == 1.0
    assert latest.is_stale is False
    assert latest.retrieved_at == _RETRIEVED_AT


def test_parse_monthly_series_uses_first_of_month() -> None:
    spec = _spec(
        series_id="bank_lending_rate_households",
        native_identifier="MIR/M.U2.B.A2A.A.R.A.2240.EUR.N",
        unit="percent",
        frequency="monthly",
        lag_hours=24.0 * 30,
    )
    result = parse_sdmx_json(_fixture_bytes("mir_lending"), spec, retrieved_at=_RETRIEVED_AT)

    assert result.warnings == ()
    assert len(result.observations) == 5
    days = sorted(o.observation_time.date() for o in result.observations)
    assert days == [
        date(2026, 3, 1),
        date(2026, 4, 1),
        date(2026, 5, 1),
        date(2026, 6, 1),
        date(2026, 7, 1),
    ]


# --------------------------------------------------------------------------
# vintage / revision handling
# --------------------------------------------------------------------------


def test_every_observation_shares_header_prepared_as_vintage_and_no_revision_index() -> None:
    spec = _spec()
    result = parse_sdmx_json(_fixture_bytes("exr_usd"), spec, retrieved_at=_RETRIEVED_AT)

    expected_vintage = datetime.fromisoformat("2026-09-25T16:01:07.128+02:00").astimezone(UTC)
    assert len(result.observations) == 5
    for obs in result.observations:
        assert obs.vintage_time == expected_vintage
        assert obs.revision_index is None
        assert obs.source_release_time is None  # never known for a historical print


# --------------------------------------------------------------------------
# missing value handling
# --------------------------------------------------------------------------


def test_null_observation_is_skipped_without_a_warning() -> None:
    spec = _spec()
    full = parse_sdmx_json(_fixture_bytes("exr_usd"), spec, retrieved_at=_RETRIEVED_AT)
    with_null = parse_sdmx_json(
        _fixture_bytes("exr_usd_with_null"), spec, retrieved_at=_RETRIEVED_AT
    )

    assert len(with_null.observations) == len(full.observations) - 1
    assert with_null.warnings == ()
    assert date(2026, 9, 23) not in {o.observation_time.date() for o in with_null.observations}


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash
# --------------------------------------------------------------------------


def test_non_json_payload_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_sdmx_json(b"not json at all", spec, retrieved_at=_RETRIEVED_AT)

    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "JSON" in result.warnings[0]


def test_missing_top_level_keys_warns_and_returns_no_observations() -> None:
    spec = _spec()
    payload = json.dumps({"unexpected": "shape"}).encode("utf-8")
    result = parse_sdmx_json(payload, spec, retrieved_at=_RETRIEVED_AT)

    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "unexpected ECB SDMX-JSON structure" in result.warnings[0]


def test_empty_series_map_warns_missing_series() -> None:
    spec = _spec()
    payload = json.dumps(
        {
            "header": {"prepared": "2026-09-25T16:01:07.128+02:00"},
            "dataSets": [{"series": {}}],
            "structure": {"dimensions": {"observation": [{"values": []}]}},
        }
    ).encode("utf-8")
    result = parse_sdmx_json(payload, spec, retrieved_at=_RETRIEVED_AT)

    assert result.observations == []
    assert result.missing_series == (spec.series_id,)


def test_unrecognized_time_period_label_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("exr_usd"))
    obs_values = payload["structure"]["dimensions"]["observation"][0]["values"]
    obs_values[4]["id"] = "2026-Q3"  # a shape this parser does not handle
    result = parse_sdmx_json(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)

    # The other four observations still parse; only the odd one is dropped
    # with a warning instead of aborting the whole payload.
    assert len(result.observations) == 4
    assert len(result.warnings) == 1
    assert "TIME_PERIOD" in result.warnings[0] or "malformed observation" in result.warnings[0]


def test_non_numeric_value_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("exr_usd"))
    series = payload["dataSets"][0]["series"]
    key = next(iter(series))
    series[key]["observations"]["4"][0] = "not-a-number"
    result = parse_sdmx_json(json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)

    assert len(result.observations) == 4
    assert len(result.warnings) == 1
    assert "non-numeric" in result.warnings[0]


# --------------------------------------------------------------------------
# availability precision correctness
# --------------------------------------------------------------------------


def test_availability_precision_is_conservative_date_with_declared_lag() -> None:
    spec = _spec(lag_hours=20.0)
    result = parse_sdmx_json(_fixture_bytes("exr_usd"), spec, retrieved_at=_RETRIEVED_AT)

    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    expected_available_at = datetime(2026, 9, 25, 20, 0, tzinfo=UTC)
    assert latest.available_at == expected_available_at
    assert latest.available_at > latest.observation_time  # never leaks same-day


# --------------------------------------------------------------------------
# fetch() -- request shape, credential-free fingerprint
# --------------------------------------------------------------------------


@respx.mock
def test_fetch_builds_url_from_native_identifier_and_uses_last_n_observations() -> None:
    spec = _spec()
    route = respx.get(f"{_BASE_URL}/service/data/EXR/D.USD.EUR.SP00.A").mock(
        return_value=httpx.Response(
            200,
            content=_fixture_bytes("exr_usd"),
            headers={"content-type": "application/vnd.sdmx.data+json;version=1.0.0"},
        )
    )
    adapter = EcbDataAdapter(
        HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL, last_n_observations=5
    )

    payload = adapter.fetch(spec)

    assert route.called
    sent_request = route.calls.last.request
    assert sent_request.url.params["lastNObservations"] == "5"
    assert sent_request.url.params["format"] == "jsondata"
    assert "startPeriod" not in sent_request.url.params

    assert isinstance(payload, FetchedPayload)
    assert payload.source == "ecb"
    assert payload.dataset == spec.series_id
    assert payload.http_status == 200
    assert "sdmx" in payload.content_type or "json" in payload.content_type
    assert payload.content == _fixture_bytes("exr_usd")


@respx.mock
def test_fetch_uses_start_period_when_since_given() -> None:
    spec = _spec()
    respx.get(f"{_BASE_URL}/service/data/EXR/D.USD.EUR.SP00.A").mock(
        return_value=httpx.Response(200, content=_fixture_bytes("exr_usd"))
    )
    adapter = EcbDataAdapter(
        HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL, last_n_observations=5
    )

    payload = adapter.fetch(spec, since=date(2026, 9, 1))

    request = respx.calls.last.request
    assert request.url.params["startPeriod"] == "2026-09-01"
    assert "lastNObservations" not in request.url.params
    assert "2026-09-01" in payload.url


@respx.mock
def test_fetch_request_fingerprint_carries_no_credential_marker() -> None:
    spec = _spec()
    respx.get(f"{_BASE_URL}/service/data/EXR/D.USD.EUR.SP00.A").mock(
        return_value=httpx.Response(200, content=_fixture_bytes("exr_usd"))
    )
    adapter = EcbDataAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    payload = adapter.fetch(spec)

    lowered = f"{payload.request_fingerprint} {payload.url}".lower()
    for marker in ("api_key=", "apikey=", "appid=", "token=", "password="):
        assert marker not in lowered
    # ECB's public SDMX API needs no credential at all -- there should be no
    # Authorization-like query parameter of any kind.
    assert "key=" not in lowered or "lastnobservations" in lowered


def test_parse_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """`parse()` must be pure: no HttpClient, no network access, ever."""
    import httpx as httpx_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parse() must not perform network I/O")

    monkeypatch.setattr(httpx_module.Client, "request", _boom)

    adapter = EcbDataAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec()
    payload = FetchedPayload(
        source="ecb",
        dataset=spec.series_id,
        url=f"{_BASE_URL}/service/data/EXR/D.USD.EUR.SP00.A?format=jsondata",
        content=_fixture_bytes("exr_usd"),
        http_status=200,
        content_type="application/vnd.sdmx.data+json",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint="GET https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A",
    )

    result = adapter.parse(payload, spec)
    assert len(result.observations) == 5


def test_adapter_identity() -> None:
    adapter = EcbDataAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    assert adapter.source_id == "ecb"
    assert adapter.parser_version == "1"
