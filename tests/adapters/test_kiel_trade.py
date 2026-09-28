"""Tests for the Kiel Trade Indicator adapter (`adapters/kiel_trade.py`).

Offline by default: parsing tests replay real, captured CSV fixtures under
`tests/fixtures/external/kiel/` (each trimmed from a live 2026-09-28 fetch,
see the adapter module docstring for exactly what was verified). Only the
fetch-shape tests touch HTTP, and even those are mocked with `respx` -- no
test in this file makes a real network call.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters.base import AdapterError, AdapterHttpError, HttpClient
from turboedge.adapters.kiel_trade import (
    KielTradeAdapter,
    build_native_identifier,
    parse_kiel_csv,
    parse_native_identifier,
)
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "kiel"
_BASE_URL = "https://trade.kielinstitut.de/KTI"
_RETRIEVED_AT = datetime(2026, 9, 28, 14, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.csv").read_bytes()


def _spec(
    *,
    series_id: str = "kiel_red_sea_ships",
    native_identifier: str = "plot_ships_red_sea|n|",
    unit: str = "ship_count",
    frequency: str = "daily",
    lag_hours: float = 24.0,
) -> SeriesSpec:
    return SeriesSpec(
        source="kiel_trade",
        series_id=series_id,
        name="Ships in the Red Sea (Kiel Trade Indicator)",
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


def test_build_and_parse_native_identifier_round_trip_no_filter() -> None:
    raw = build_native_identifier("plot_ships_red_sea", "n")
    assert raw == "plot_ships_red_sea|n|"
    assert parse_native_identifier(raw) == ("plot_ships_red_sea", "n", "")


def test_build_and_parse_native_identifier_round_trip_with_filter() -> None:
    raw = build_native_identifier("plot_portcalls_china", "n", "Shenzhen")
    assert raw == "plot_portcalls_china|n|Shenzhen"
    assert parse_native_identifier(raw) == ("plot_portcalls_china", "n", "Shenzhen")


def test_build_native_identifier_rejects_empty_basename_or_column() -> None:
    with pytest.raises(AdapterError):
        build_native_identifier("", "n")
    with pytest.raises(AdapterError):
        build_native_identifier("plot_ships_red_sea", "")


def test_parse_native_identifier_rejects_wrong_segment_count() -> None:
    with pytest.raises(AdapterError, match="does not match"):
        parse_native_identifier("plot_ships_red_sea|n")
    with pytest.raises(AdapterError, match="does not match"):
        parse_native_identifier("plot_ships_red_sea|n|extra|segment")


def test_parse_native_identifier_rejects_missing_basename_or_column() -> None:
    with pytest.raises(AdapterError):
        parse_native_identifier("|n|")
    with pytest.raises(AdapterError):
        parse_native_identifier("plot_ships_red_sea||")


# --------------------------------------------------------------------------
# normal parse -- shape A: timestamp,n (single series per file)
# --------------------------------------------------------------------------


def test_parse_shape_a_red_sea_current() -> None:
    spec = _spec()
    result = parse_kiel_csv(_fixture_bytes("plot_ships_red_sea"), spec, retrieved_at=_RETRIEVED_AT)

    assert result.warnings == ()
    assert result.missing_series == ()
    by_day = {o.observation_time.date(): o for o in result.observations}
    latest = by_day[date(2026, 9, 25)]
    assert latest.value == pytest.approx(46.0)
    assert latest.series_id == "kiel_red_sea_ships"
    assert latest.unit == "ship_count"
    assert latest.frequency == "daily"
    assert latest.source == "kiel_trade"
    assert latest.source_version == "kiel_trade_plot_ships_red_sea"
    assert latest.quality_score == 1.0
    assert latest.is_stale is False
    assert latest.source_release_time is None
    assert latest.vintage_time is None
    assert latest.revision_index is None


def test_parse_shape_a_cape_of_good_hope_current() -> None:
    spec = _spec(
        series_id="kiel_cape_of_good_hope_ships",
        native_identifier="plot_ships_cape_good_hope|n|",
    )
    result = parse_kiel_csv(
        _fixture_bytes("plot_ships_cape_good_hope"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2026, 9, 24)
    assert latest.value == pytest.approx(29.0)


def test_parse_shape_a_panama_canal_stale() -> None:
    """A frozen/stale file parses identically to a fresh one -- this adapter
    never special-cases staleness (module docstring)."""
    spec = _spec(
        series_id="kiel_panama_canal_ships", native_identifier="plot_ships_panama_canal|n|"
    )
    result = parse_kiel_csv(
        _fixture_bytes("plot_ships_panama_canal"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2025, 1, 24)
    assert latest.value == pytest.approx(47.0)


# --------------------------------------------------------------------------
# normal parse -- shape B: timestamp,portname,n (multiplexed by port)
# --------------------------------------------------------------------------


def test_parse_shape_b_portcalls_filtered_to_one_port() -> None:
    spec = _spec(
        series_id="kiel_portcalls_shenzhen",
        native_identifier="plot_portcalls_china|n|Shenzhen",
        unit="port_calls",
        frequency="weekly",
    )
    result = parse_kiel_csv(
        _fixture_bytes("plot_portcalls_china"), spec, retrieved_at=_RETRIEVED_AT
    )

    assert result.warnings == ()
    assert result.missing_series == ()
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2025, 1, 12)
    assert latest.value == pytest.approx(158.0)
    # Only Shenzhen's rows were extracted -- Shanghai/Hong Kong/Ningbo values
    # for the same dates must not leak in as duplicates.
    same_day = [o for o in result.observations if o.observation_time.date() == date(2025, 1, 12)]
    assert len(same_day) == 1


def test_parse_shape_b_other_port_gives_a_different_series() -> None:
    spec = _spec(
        series_id="kiel_portcalls_ningbo", native_identifier="plot_portcalls_china|n|Ningbo"
    )
    result = parse_kiel_csv(
        _fixture_bytes("plot_portcalls_china"), spec, retrieved_at=_RETRIEVED_AT
    )
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2025, 1, 12)
    assert latest.value == pytest.approx(106.0)


# --------------------------------------------------------------------------
# normal parse -- shape C: name,code,timestamp,rate (multiplexed by route)
# --------------------------------------------------------------------------


def test_parse_shape_c_freight_rate_filtered_to_one_route() -> None:
    spec = _spec(
        series_id="kiel_freight_china_to_northern_europe",
        native_identifier=(
            "plot_freight_rates_china_northern_europe_global|rate|China to Northern Europe"
        ),
        unit="usd_per_feu",
        frequency="daily",
    )
    result = parse_kiel_csv(
        _fixture_bytes("plot_freight_rates_china_northern_europe_global"),
        spec,
        retrieved_at=_RETRIEVED_AT,
    )
    assert result.warnings == ()
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2025, 1, 27)
    assert latest.value == pytest.approx(3863.0)


def test_parse_shape_c_global_average_route() -> None:
    spec = _spec(
        series_id="kiel_freight_global_average",
        native_identifier="plot_freight_rates_china_northern_europe_global|rate|Global average",
    )
    result = parse_kiel_csv(
        _fixture_bytes("plot_freight_rates_china_northern_europe_global"),
        spec,
        retrieved_at=_RETRIEVED_AT,
    )
    assert result.warnings == ()
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.observation_time.date() == date(2025, 1, 27)
    assert latest.value == pytest.approx(3631.0)


# --------------------------------------------------------------------------
# unknown file/filter -> missing_series
# --------------------------------------------------------------------------


def test_unknown_port_filter_is_missing_series_not_a_crash() -> None:
    spec = _spec(
        series_id="kiel_portcalls_nairobi",
        native_identifier="plot_portcalls_china|n|Nairobi",
    )
    result = parse_kiel_csv(
        _fixture_bytes("plot_portcalls_china"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "Nairobi" in result.warnings[0]


def test_filter_given_for_a_file_with_no_name_column_is_missing_series() -> None:
    spec = _spec(
        series_id="kiel_red_sea_bogus_filter", native_identifier="plot_ships_red_sea|n|Foo"
    )
    result = parse_kiel_csv(_fixture_bytes("plot_ships_red_sea"), spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "no name/portname column" in result.warnings[0]


def test_multiplexed_file_with_no_filter_is_missing_series() -> None:
    spec = _spec(series_id="kiel_portcalls_unfiltered", native_identifier="plot_portcalls_china|n|")
    result = parse_kiel_csv(
        _fixture_bytes("plot_portcalls_china"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "refusing to guess" in result.warnings[0]


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash
# --------------------------------------------------------------------------


def test_missing_timestamp_column_warns_and_returns_no_observations() -> None:
    spec = _spec()
    payload = b"date,n\n2026-09-25,46\n"
    result = parse_kiel_csv(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "timestamp" in result.warnings[0]


def test_declared_value_column_absent_warns_and_returns_no_observations() -> None:
    spec = _spec(native_identifier="plot_ships_red_sea|count|")
    result = parse_kiel_csv(_fixture_bytes("plot_ships_red_sea"), spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "count" in result.warnings[0]


def test_non_utf8_payload_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_kiel_csv(b"\xff\xfe not valid utf-8", spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1


def test_conflicting_values_for_same_day_are_skipped_not_guessed() -> None:
    spec = _spec()
    payload = b"timestamp,n\n2026-09-24,44\n2026-09-24,99\n2026-09-25,46\n"
    result = parse_kiel_csv(payload, spec, retrieved_at=_RETRIEVED_AT)
    days = {o.observation_time.date() for o in result.observations}
    assert date(2026, 9, 24) not in days
    assert date(2026, 9, 25) in days
    assert len(result.warnings) == 1
    assert "conflicting" in result.warnings[0]


def test_blank_and_non_numeric_values_are_skipped_with_a_warning() -> None:
    spec = _spec()
    payload = b"timestamp,n\n2026-09-23,\n2026-09-24,not-a-number\n2026-09-25,46\n"
    result = parse_kiel_csv(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 1
    assert result.observations[0].observation_time.date() == date(2026, 9, 25)
    assert len(result.warnings) == 1
    assert "blank/non-numeric" in result.warnings[0]


def test_unparseable_dates_are_skipped_with_a_warning() -> None:
    spec = _spec()
    payload = b"timestamp,n\nnot-a-date,44\n2026-09-25,46\n"
    result = parse_kiel_csv(payload, spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 1
    assert len(result.warnings) == 1
    assert "unparseable" in result.warnings[0]


# --------------------------------------------------------------------------
# availability precision
# --------------------------------------------------------------------------


def test_availability_precision_is_conservative_date_with_declared_lag() -> None:
    spec = _spec(lag_hours=20.0)
    result = parse_kiel_csv(_fixture_bytes("plot_ships_red_sea"), spec, retrieved_at=_RETRIEVED_AT)
    latest = max(result.observations, key=lambda o: o.observation_time)
    assert latest.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    expected_available_at = datetime(2026, 9, 25, 20, 0, tzinfo=UTC)
    assert latest.available_at == expected_available_at
    assert latest.available_at > latest.observation_time


# --------------------------------------------------------------------------
# fetch() -- request shape, dataset, HTTP 404
# --------------------------------------------------------------------------


@respx.mock
def test_fetch_requests_the_basename_csv_and_returns_its_bytes() -> None:
    spec = _spec()
    route = respx.get(f"{_BASE_URL}/plot_ships_red_sea.csv").mock(
        return_value=httpx.Response(
            200,
            content=_fixture_bytes("plot_ships_red_sea"),
            headers={"content-type": "text/csv"},
        )
    )
    adapter = KielTradeAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

    payload = adapter.fetch(spec)

    assert route.called
    assert isinstance(payload, FetchedPayload)
    assert payload.source == "kiel_trade"
    assert payload.dataset == "plot_ships_red_sea"
    assert payload.http_status == 200
    assert payload.content_type == "text/csv"
    assert payload.content == _fixture_bytes("plot_ships_red_sea")
    assert payload.url.endswith("plot_ships_red_sea.csv")


@respx.mock
def test_fetch_raises_adapter_http_error_on_404() -> None:
    """`plot_draft_global.csv` is confirmed HTTP 404 live (module docstring);
    a `SeriesSpec` should never reference it, but if one did, `fetch()` must
    fail loudly rather than archive a 404 error page as if it were data."""
    spec = _spec(native_identifier="plot_draft_global|n|")
    respx.get(f"{_BASE_URL}/plot_draft_global.csv").mock(return_value=httpx.Response(404))
    adapter = KielTradeAdapter(HttpClient(user_agent="turboedge-test/1.0"), base_url=_BASE_URL)

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

    adapter = KielTradeAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec()
    payload = FetchedPayload(
        source="kiel_trade",
        dataset="plot_ships_red_sea",
        url=f"{_BASE_URL}/plot_ships_red_sea.csv",
        content=_fixture_bytes("plot_ships_red_sea"),
        http_status=200,
        content_type="text/csv",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_BASE_URL}/plot_ships_red_sea.csv",
    )

    result = adapter.parse(payload, spec)

    assert result.observations
