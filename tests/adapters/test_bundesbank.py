"""Tests for the Deutsche Bundesbank adapter (`adapters/bundesbank.py`).

Offline by default: parsing tests replay real, captured SDMX-CSV fixtures
under `tests/fixtures/external/bundesbank/`. Only the fetch-shape tests
touch HTTP, and even those are mocked with `respx` -- no test in this file
makes a real network call.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters.base import HttpClient
from turboedge.adapters.bundesbank import (
    BundesbankAdapter,
    parse_bundesbank_csv,
    split_native_identifier,
)
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "bundesbank"
_BASE_URL = "https://api.statistiken.bundesbank.de/rest"
_RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.csv").read_bytes()


def _spec(
    *,
    series_id: str = "eur_usd_reference_rate",
    native_identifier: str = "BBEX3/D.USD.EUR.BB.AC.000",
    unit: str = "usd_per_eur",
    frequency: str = "daily",
    lag_hours: float = 20.0,
) -> SeriesSpec:
    return SeriesSpec(
        source="bundesbank",
        series_id=series_id,
        name="ECB euro reference rate, USD per EUR",
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
    assert split_native_identifier("BBEX3/D.USD.EUR.BB.AC.000") == (
        "BBEX3",
        "D.USD.EUR.BB.AC.000",
    )


def test_split_native_identifier_rejects_missing_slash() -> None:
    with pytest.raises(ValueError, match="FLOW/KEY"):
        split_native_identifier("BBEX3-D.USD.EUR.BB.AC.000")


def test_split_native_identifier_rejects_empty_flow_or_key() -> None:
    with pytest.raises(ValueError):
        split_native_identifier("/D.USD.EUR.BB.AC.000")
    with pytest.raises(ValueError):
        split_native_identifier("BBEX3/")


# --------------------------------------------------------------------------
# parse_bundesbank_csv -- normal parse (daily and monthly)
# --------------------------------------------------------------------------


def test_parse_daily_series_eur_usd() -> None:
    spec = _spec()
    result = parse_bundesbank_csv(
        _fixture_bytes("eur_usd_reference_rate"), spec, retrieved_at=_RETRIEVED_AT
    )

    assert result.warnings == ()
    assert result.missing_series == ()
    assert len(result.observations) == 5

    by_day = {o.observation_time.date(): o for o in result.observations}
    latest = by_day[date(2026, 9, 25)]
    assert latest.value == pytest.approx(1.1403)
    assert latest.series_id == "eur_usd_reference_rate"
    assert latest.unit == "usd_per_eur"
    assert latest.frequency == "daily"
    assert latest.source == "bundesbank"
    assert latest.source_version == "bundesbank_sdmx_csv_bbex3"
    assert latest.quality_score == 1.0
    assert latest.is_stale is False
    assert latest.retrieved_at == _RETRIEVED_AT


def test_parse_monthly_industrial_production() -> None:
    spec = _spec(
        series_id="industrial_production_index",
        native_identifier="BBDE1/M.DE.Y.BAA1.A2P300000.G.C.I21.A",
        unit="index_2021_100",
        frequency="monthly",
        lag_hours=24.0 * 30,
    )
    result = parse_bundesbank_csv(
        _fixture_bytes("industrial_production"), spec, retrieved_at=_RETRIEVED_AT
    )

    assert result.warnings == ()
    assert len(result.observations) == 5
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2026, 7, 1)
    assert latest.value == pytest.approx(90.1)


def test_parse_monthly_manufacturing_orders() -> None:
    spec = _spec(
        series_id="manufacturing_new_orders_index",
        native_identifier="BBDE1/M.DE.Y.AEA1.A2P300000.F.C.I21.A",
        unit="index_2021_100",
        frequency="monthly",
        lag_hours=24.0 * 30,
    )
    result = parse_bundesbank_csv(
        _fixture_bytes("manufacturing_orders"), spec, retrieved_at=_RETRIEVED_AT
    )

    assert result.warnings == ()
    assert len(result.observations) == 5
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2026, 7, 1)
    assert latest.value == pytest.approx(95.5)


def test_parse_monthly_base_rate() -> None:
    spec = _spec(
        series_id="statutory_base_rate",
        native_identifier="BBIN1/M.DE.BBK.BBKBAS2.EUR.ME",
        unit="percent",
        frequency="monthly",
        lag_hours=24.0 * 30,
    )
    result = parse_bundesbank_csv(_fixture_bytes("base_rate"), spec, retrieved_at=_RETRIEVED_AT)

    assert result.warnings == ()
    assert len(result.observations) == 5
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2026, 9, 1)
    assert latest.value == pytest.approx(1.52)


# --------------------------------------------------------------------------
# missing value handling (real publisher gaps: weekend + New Year's Day)
# --------------------------------------------------------------------------


def test_missing_obs_value_marker_is_skipped_without_a_warning() -> None:
    spec = _spec()
    result = parse_bundesbank_csv(
        _fixture_bytes("eur_usd_reference_rate_with_gaps"), spec, retrieved_at=_RETRIEVED_AT
    )

    # 7 calendar days requested, 4 real prints (2025-12-30/31, 2026-01-02/05):
    # 2026-01-01 (New Year's Day) and 01-03/01-04 (weekend) are "." in the
    # real payload.
    assert result.warnings == ()
    assert len(result.observations) == 4
    days = {o.observation_time.date() for o in result.observations}
    assert date(2026, 1, 1) not in days
    assert date(2026, 1, 3) not in days
    assert date(2026, 1, 4) not in days
    assert date(2026, 1, 2) in days


# --------------------------------------------------------------------------
# German decimal-comma handling
# --------------------------------------------------------------------------


def test_comma_decimal_obs_value_is_parsed() -> None:
    spec = _spec(series_id="statutory_base_rate", unit="percent", frequency="monthly")
    text = _fixture_bytes("base_rate").decode("utf-8-sig")
    lines = text.splitlines()
    header, first_row, *rest = lines
    # Rewrite just the OBS_VALUE field (position 8, 0-indexed) of one row to
    # German comma-decimal notation, keeping every other field authentic.
    fields = first_row.split(";")
    fields[8] = fields[8].replace(".", ",")
    mutated = ";".join(fields)
    payload = "﻿" + "\n".join([header, mutated, *rest])

    result = parse_bundesbank_csv(payload.encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)

    assert result.warnings == ()
    assert len(result.observations) == 5


def test_thousands_separator_and_comma_decimal_combination() -> None:
    from turboedge.adapters.bundesbank import _parse_obs_value

    assert _parse_obs_value("1.234,56") == pytest.approx(1234.56)
    assert _parse_obs_value("92.1") == pytest.approx(92.1)
    assert _parse_obs_value("1,52") == pytest.approx(1.52)


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash
# --------------------------------------------------------------------------


def test_non_csv_payload_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_bundesbank_csv(b"\xff\xfe not valid utf-8", spec, retrieved_at=_RETRIEVED_AT)

    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1


def test_missing_required_columns_warns_and_returns_no_observations() -> None:
    spec = _spec()
    payload = "COL_A;COL_B\nfoo;bar\n".encode("utf-8-sig")
    result = parse_bundesbank_csv(payload, spec, retrieved_at=_RETRIEVED_AT)

    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "unexpected Bundesbank CSV column set" in result.warnings[0]


def test_empty_body_warns_missing_series() -> None:
    spec = _spec()
    header = "DATAFLOW;TIME_PERIOD;OBS_VALUE;BBK_ID;BBK_UNIT\n"
    result = parse_bundesbank_csv(header.encode("utf-8-sig"), spec, retrieved_at=_RETRIEVED_AT)

    assert result.observations == []
    assert result.missing_series == (spec.series_id,)


def test_unrecognized_time_period_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    text = _fixture_bytes("eur_usd_reference_rate").decode("utf-8-sig")
    lines = text.splitlines()
    header, *rows = lines
    mutated_rows = list(rows)
    fields = mutated_rows[0].split(";")
    time_period_idx = header.split(";").index("TIME_PERIOD")
    fields[time_period_idx] = "2026-Q3"  # a shape this parser does not handle
    mutated_rows[0] = ";".join(fields)
    payload = "﻿" + "\n".join([header, *mutated_rows])

    result = parse_bundesbank_csv(payload.encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)

    assert len(result.observations) == 4
    assert len(result.warnings) == 1
    assert "TIME_PERIOD" in result.warnings[0]


def test_non_numeric_value_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    text = _fixture_bytes("eur_usd_reference_rate").decode("utf-8-sig")
    lines = text.splitlines()
    header, *rows = lines
    mutated_rows = list(rows)
    fields = mutated_rows[0].split(";")
    obs_value_idx = header.split(";").index("OBS_VALUE")
    fields[obs_value_idx] = "not-a-number"
    mutated_rows[0] = ";".join(fields)
    payload = "﻿" + "\n".join([header, *mutated_rows])

    result = parse_bundesbank_csv(payload.encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT)

    assert len(result.observations) == 4
    assert len(result.warnings) == 1
    assert "non-numeric" in result.warnings[0]


# --------------------------------------------------------------------------
# availability precision correctness
# --------------------------------------------------------------------------


def test_availability_precision_is_conservative_date_with_declared_lag() -> None:
    spec = _spec(lag_hours=20.0)
    result = parse_bundesbank_csv(
        _fixture_bytes("eur_usd_reference_rate"), spec, retrieved_at=_RETRIEVED_AT
    )

    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    expected_available_at = datetime(2026, 9, 25, 20, 0, tzinfo=UTC)
    assert latest.available_at == expected_available_at
    assert latest.available_at > latest.observation_time  # never leaks same-day


# --------------------------------------------------------------------------
# vintage handling (best-effort, from the HTTP Date header; never invented)
# --------------------------------------------------------------------------


def test_vintage_time_comes_from_response_date_header_when_present() -> None:
    spec = _spec()
    result = parse_bundesbank_csv(
        _fixture_bytes("eur_usd_reference_rate"),
        spec,
        retrieved_at=_RETRIEVED_AT,
        headers={"date": "Mon, 28 Sep 2026 08:07:30 GMT"},
    )
    assert len(result.observations) == 5
    expected = datetime(2026, 9, 28, 8, 7, 30, tzinfo=UTC)
    for obs in result.observations:
        assert obs.vintage_time == expected
        assert obs.revision_index is None
        assert obs.source_release_time is None


def test_vintage_time_is_none_without_a_date_header() -> None:
    spec = _spec()
    result = parse_bundesbank_csv(
        _fixture_bytes("eur_usd_reference_rate"), spec, retrieved_at=_RETRIEVED_AT, headers=None
    )
    assert len(result.observations) == 5
    for obs in result.observations:
        assert obs.vintage_time is None


# --------------------------------------------------------------------------
# fetch() -- request shape, Accept header, credential-free fingerprint
# --------------------------------------------------------------------------


@respx.mock
def test_fetch_sends_accept_text_csv_and_uses_last_n_observations() -> None:
    spec = _spec()
    route = respx.get(f"{_BASE_URL}/data/BBEX3/D.USD.EUR.BB.AC.000").mock(
        return_value=httpx.Response(
            200,
            content=_fixture_bytes("eur_usd_reference_rate"),
            headers={"content-type": "text/csv", "date": "Mon, 28 Sep 2026 08:07:30 GMT"},
        )
    )
    adapter = BundesbankAdapter(
        HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL, last_n_observations=5
    )

    payload = adapter.fetch(spec)

    assert route.called
    sent_request = route.calls.last.request
    assert sent_request.headers["accept"] == "text/csv"
    assert sent_request.url.params["lastNObservations"] == "5"
    assert "startPeriod" not in sent_request.url.params

    assert isinstance(payload, FetchedPayload)
    assert payload.source == "bundesbank"
    assert payload.dataset == spec.series_id
    assert payload.http_status == 200
    assert payload.content_type == "text/csv"
    assert payload.headers.get("date") == "Mon, 28 Sep 2026 08:07:30 GMT"
    assert payload.content == _fixture_bytes("eur_usd_reference_rate")


@respx.mock
def test_fetch_uses_start_period_when_since_given() -> None:
    spec = _spec()
    respx.get(f"{_BASE_URL}/data/BBEX3/D.USD.EUR.BB.AC.000").mock(
        return_value=httpx.Response(200, content=_fixture_bytes("eur_usd_reference_rate"))
    )
    adapter = BundesbankAdapter(
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
    respx.get(f"{_BASE_URL}/data/BBEX3/D.USD.EUR.BB.AC.000").mock(
        return_value=httpx.Response(200, content=_fixture_bytes("eur_usd_reference_rate"))
    )
    adapter = BundesbankAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    payload = adapter.fetch(spec)

    lowered = f"{payload.request_fingerprint} {payload.url}".lower()
    for marker in ("api_key=", "apikey=", "appid=", "token=", "password="):
        assert marker not in lowered


def test_parse_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """`parse()` must be pure: no HttpClient, no network access, ever."""
    import httpx as httpx_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parse() must not perform network I/O")

    monkeypatch.setattr(httpx_module.Client, "request", _boom)

    adapter = BundesbankAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec()
    payload = FetchedPayload(
        source="bundesbank",
        dataset=spec.series_id,
        url=f"{_BASE_URL}/data/BBEX3/D.USD.EUR.BB.AC.000",
        content=_fixture_bytes("eur_usd_reference_rate"),
        http_status=200,
        content_type="text/csv",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_BASE_URL}/data/BBEX3/D.USD.EUR.BB.AC.000",
    )

    result = adapter.parse(payload, spec)
    assert len(result.observations) == 5


def test_adapter_identity() -> None:
    adapter = BundesbankAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    assert adapter.source_id == "bundesbank"
    assert adapter.parser_version == "1"
