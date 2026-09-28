"""Tests for the FRED/ALFRED adapter (`adapters/fred.py`).

Offline by default: parsing tests replay JSON fixtures under
`tests/fixtures/external/fred/`. Those fixtures are **hand-built from the
published FRED API documentation** (fetched during development), not
captured from a live response -- no `FRED_API_KEY` is available in this
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
from turboedge.adapters.fred import (
    CURATED_SERIES_IDS,
    FredAdapter,
    FredCredentialError,
    parse_series_observations,
)
from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, RawPayload, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "fred"
_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.json").read_bytes()


def _spec(
    *,
    series_id: str = "INDPRO",
    native_identifier: str = "INDPRO",
    unit: str = "index_2017_100",
    frequency: str = "monthly",
    lag_hours: float = 24.0,
) -> SeriesSpec:
    return SeriesSpec(
        source="fred",
        series_id=series_id,
        name="Industrial Production Index",
        category="macro",
        unit=unit,
        frequency=frequency,
        native_identifier=native_identifier,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.HISTORICAL_PIT_SAFE,
        conservative_release_lag_hours=lag_hours,
    )


# --------------------------------------------------------------------------
# normal parse
# --------------------------------------------------------------------------


def test_parse_current_values_five_months() -> None:
    spec = _spec()
    result = parse_series_observations(
        _fixture_bytes("indpro_current"), spec, retrieved_at=_RETRIEVED_AT, vintage_aware=False
    )

    assert result.warnings == ()
    assert result.missing_series == ()
    assert len(result.observations) == 5

    by_month = {o.observation_time.date(): o for o in result.observations}
    latest = by_month[date(2026, 8, 1)]
    assert latest.value == pytest.approx(103.7002)
    assert latest.series_id == "INDPRO"
    assert latest.unit == "index_2017_100"
    assert latest.frequency == "monthly"
    assert latest.source == "fred"
    assert latest.source_version == "fred_series_observations_json"
    assert latest.parser_version == "1"
    assert latest.quality_score == 1.0
    assert latest.retrieved_at == _RETRIEVED_AT


def test_current_values_have_no_revision_index_but_do_have_vintage_time() -> None:
    """A single-vintage-per-period fetch cannot honestly claim to know that
    vintage's position in the full revision history (module docstring
    "VINTAGE SEMANTICS"), but it *does* know that vintage's own effective
    date -- the two must not be conflated."""
    spec = _spec()
    result = parse_series_observations(
        _fixture_bytes("indpro_current"), spec, retrieved_at=_RETRIEVED_AT, vintage_aware=False
    )
    by_month = {o.observation_time.date(): o for o in result.observations}
    latest = by_month[date(2026, 8, 1)]

    assert latest.revision_index is None
    assert latest.vintage_time == datetime(2026, 9, 16, tzinfo=UTC)
    assert latest.source_release_time is None


# --------------------------------------------------------------------------
# vintage / revision handling -- the highest-value part of this adapter
# --------------------------------------------------------------------------


def test_vintage_aware_fetch_assigns_revision_index_in_realtime_order() -> None:
    spec = _spec(series_id="PAYEMS", native_identifier="PAYEMS", unit="thousands_of_persons")
    result = parse_series_observations(
        _fixture_bytes("payems_vintages"), spec, retrieved_at=_RETRIEVED_AT, vintage_aware=True
    )

    assert result.warnings == ()
    assert len(result.observations) == 5

    june = sorted(
        (o for o in result.observations if o.observation_time.date() == date(2026, 6, 1)),
        key=lambda o: o.vintage_time,  # type: ignore[arg-type, return-value]
    )
    assert [o.revision_index for o in june] == [0, 1, 2]
    assert [o.value for o in june] == [
        pytest.approx(158041),
        pytest.approx(158203),
        pytest.approx(157987),
    ]
    assert [o.vintage_time for o in june] == [
        datetime(2026, 7, 3, tzinfo=UTC),
        datetime(2026, 8, 6, tzinfo=UTC),
        datetime(2026, 9, 4, tzinfo=UTC),
    ]

    july = sorted(
        (o for o in result.observations if o.observation_time.date() == date(2026, 7, 1)),
        key=lambda o: o.vintage_time,  # type: ignore[arg-type, return-value]
    )
    assert [o.revision_index for o in july] == [0, 1]

    # Revision order is never conflated with the publisher's official
    # release date: this adapter does not claim to know that at all.
    assert all(o.source_release_time is None for o in result.observations)


def test_current_fetch_warns_if_more_than_one_row_per_period_appears() -> None:
    """A `realtime_start == realtime_end` (current) fetch should return
    exactly one row per period; more than one is upstream schema drift and
    must be a warning, not a silently-guessed revision_index=0/1/2."""
    spec = _spec(series_id="PAYEMS", native_identifier="PAYEMS", unit="thousands_of_persons")
    result = parse_series_observations(
        _fixture_bytes("payems_vintages"), spec, retrieved_at=_RETRIEVED_AT, vintage_aware=False
    )

    assert len(result.observations) == 5
    assert all(o.revision_index is None for o in result.observations)
    assert len(result.warnings) == 2  # one per period that had >1 row
    assert all("expected exactly one" in w for w in result.warnings)


# --------------------------------------------------------------------------
# missing value handling ("." marker)
# --------------------------------------------------------------------------


def test_dot_marker_is_skipped_without_a_warning() -> None:
    spec = _spec(series_id="CPIAUCSL", native_identifier="CPIAUCSL", unit="index_1982_84_100")
    result = parse_series_observations(
        _fixture_bytes("cpiaucsl_with_missing"),
        spec,
        retrieved_at=_RETRIEVED_AT,
        vintage_aware=False,
    )

    assert result.warnings == ()
    assert len(result.observations) == 2  # 4 rows in fixture, 2 are "."
    parsed_months = {o.observation_time.date() for o in result.observations}
    assert parsed_months == {date(2026, 6, 1), date(2026, 8, 1)}


def test_dot_marker_never_parsed_as_zero() -> None:
    spec = _spec(series_id="CPIAUCSL", native_identifier="CPIAUCSL")
    result = parse_series_observations(
        _fixture_bytes("cpiaucsl_with_missing"),
        spec,
        retrieved_at=_RETRIEVED_AT,
        vintage_aware=False,
    )
    assert all(o.value != 0.0 for o in result.observations)


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash
# --------------------------------------------------------------------------


def test_non_json_payload_warns_and_returns_no_observations() -> None:
    spec = _spec()
    result = parse_series_observations(
        b"not json at all", spec, retrieved_at=_RETRIEVED_AT, vintage_aware=True
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert len(result.warnings) == 1
    assert "JSON" in result.warnings[0]


def test_missing_observations_key_warns_and_returns_no_observations() -> None:
    spec = _spec()
    payload = json.dumps({"realtime_start": "2026-09-28"}).encode("utf-8")
    result = parse_series_observations(
        payload, spec, retrieved_at=_RETRIEVED_AT, vintage_aware=True
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "observations" in result.warnings[0]


def test_error_body_warns_and_does_not_crash() -> None:
    """FRED's documented JSON error shape (`error_code`/`error_message`)
    must never be parsed as if it were an observations payload."""
    spec = _spec()
    payload = json.dumps(
        {
            "error_code": 400,
            "error_message": "Bad Request. The value for variable api_key is not registered.",
        }
    ).encode("utf-8")
    result = parse_series_observations(
        payload, spec, retrieved_at=_RETRIEVED_AT, vintage_aware=True
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert "error_code" in result.warnings[0] or "400" in result.warnings[0]


def test_row_missing_required_field_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("indpro_current"))
    del payload["observations"][2]["value"]
    result = parse_series_observations(
        json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT, vintage_aware=False
    )
    assert len(result.observations) == 4
    assert len(result.warnings) == 1
    assert "missing required field" in result.warnings[0]


def test_unparseable_value_is_a_warning_not_a_crash() -> None:
    spec = _spec()
    payload = json.loads(_fixture_bytes("indpro_current"))
    payload["observations"][0]["value"] = "not-a-number"
    result = parse_series_observations(
        json.dumps(payload).encode("utf-8"), spec, retrieved_at=_RETRIEVED_AT, vintage_aware=False
    )
    assert len(result.observations) == 4
    assert len(result.warnings) == 1
    assert "unparseable observation value" in result.warnings[0]


# --------------------------------------------------------------------------
# availability precision correctness
# --------------------------------------------------------------------------


def test_availability_is_conservative_date_from_vintage_day_not_observation_period() -> None:
    """`available_at` must be derived from when the *vintage* became
    effective (`realtime_start`), not from the economic period it describes
    (`observation_time`) -- those differ by weeks to months for a monthly
    series like INDPRO."""
    spec = _spec(lag_hours=24.0)
    result = parse_series_observations(
        _fixture_bytes("indpro_current"), spec, retrieved_at=_RETRIEVED_AT, vintage_aware=False
    )
    latest = max(result.observations, key=lambda o: o.observation_time)

    assert latest.observation_time.date() == date(2026, 8, 1)
    assert latest.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    # realtime_start for this row is 2026-09-16 (see fixture); +24h lag.
    assert latest.available_at == datetime(2026, 9, 17, tzinfo=UTC)
    # available_at is nowhere near observation_time -- it tracks the vintage,
    # which for a monthly macro series lands ~6 weeks after the period ends.
    assert (latest.available_at.date() - latest.observation_time.date()).days > 30


# --------------------------------------------------------------------------
# fetch() -- request shape, credential handling, credential-free fingerprint
# --------------------------------------------------------------------------


def test_fetch_raises_typed_error_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    with pytest.raises(FredCredentialError, match="FRED_API_KEY"):
        adapter.fetch(_spec())


@respx.mock
def test_fetch_sends_wide_realtime_window_and_api_key_to_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key-should-never-be-archived")
    route = respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("indpro_current"))
    )
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec())

    assert route.called
    sent = route.calls.last.request
    assert sent.url.params["api_key"] == "test-key-should-never-be-archived"
    assert sent.url.params["series_id"] == "INDPRO"
    assert sent.url.params["realtime_start"] == "1776-07-04"
    assert sent.url.params["file_type"] == "json"
    assert isinstance(payload, FetchedPayload)
    assert payload.source == "fred"
    assert payload.dataset == "INDPRO:vintages"
    assert payload.http_status == 200


@respx.mock
def test_fetch_current_uses_narrow_realtime_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("indpro_current"))
    )
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch_current(_spec())

    sent = respx.calls.last.request
    assert sent.url.params["realtime_start"] == sent.url.params["realtime_end"]
    assert payload.dataset == "INDPRO:current"


@respx.mock
def test_fetch_vintages_accepts_explicit_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("payems_vintages"))
    )
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch_vintages(
        _spec(series_id="PAYEMS", native_identifier="PAYEMS"),
        realtime_start=date(2026, 1, 1),
        realtime_end=date(2026, 9, 1),
    )

    sent = respx.calls.last.request
    assert sent.url.params["realtime_start"] == "2026-01-01"
    assert sent.url.params["realtime_end"] == "2026-09-01"
    assert payload.dataset == "PAYEMS:vintages"


@respx.mock
def test_fetch_since_narrows_observation_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("indpro_current"))
    )
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    adapter.fetch(_spec(), since=date(2026, 1, 1))

    sent = respx.calls.last.request
    assert sent.url.params["observation_start"] == "2026-01-01"


@respx.mock
def test_fetch_url_and_fingerprint_never_contain_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "totally-secret-fred-key-0123456789ab"
    monkeypatch.setenv("FRED_API_KEY", secret)
    respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("indpro_current"))
    )
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_spec())

    lowered = f"{payload.url} {payload.request_fingerprint}".lower()
    assert secret not in payload.url
    assert secret not in payload.request_fingerprint
    for marker in ("api_key=", "apikey=", "token=", "password="):
        assert marker not in lowered


def test_raw_payload_rejects_a_fingerprint_that_slipped_through_with_a_credential() -> None:
    """Backstop assertion (contract requirement): even if some future change
    to this adapter accidentally left a credential in the fingerprint,
    `RawPayload` itself refuses to validate -- this pins that behaviour down
    for the exact marker this adapter's credential parameter uses."""
    with pytest.raises(ValueError, match="still carries a value"):
        RawPayload(
            payload_id="fred:INDPRO:deadbeef",
            source="fred",
            dataset="INDPRO:vintages",
            request_fingerprint=(
                "GET https://api.stlouisfed.org/fred/series/observations?series_id=INDPRO&api_key=SECRET"
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

    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec()
    payload = FetchedPayload(
        source="fred",
        dataset="INDPRO:current",
        url=f"{_BASE_URL}?series_id=INDPRO&file_type=json",
        content=_fixture_bytes("indpro_current"),
        http_status=200,
        content_type="application/json",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_BASE_URL}?series_id=INDPRO",
    )

    result = adapter.parse(payload, spec)
    assert len(result.observations) == 5
    assert all(o.revision_index is None for o in result.observations)


def test_parse_dispatches_on_dataset_suffix_for_vintage_awareness() -> None:
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _spec(series_id="PAYEMS", native_identifier="PAYEMS")
    vintages_payload = FetchedPayload(
        source="fred",
        dataset="PAYEMS:vintages",
        url=f"{_BASE_URL}?series_id=PAYEMS",
        content=_fixture_bytes("payems_vintages"),
        http_status=200,
        content_type="application/json",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_BASE_URL}?series_id=PAYEMS",
    )
    result = adapter.parse(vintages_payload, spec)
    assert any(o.revision_index is not None for o in result.observations)


def test_adapter_identity() -> None:
    adapter = FredAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    assert adapter.source_id == "fred"
    assert adapter.parser_version == "1"


def test_curated_series_ids_are_the_ones_named_in_the_task_brief() -> None:
    assert set(CURATED_SERIES_IDS) == {
        "INDPRO",
        "RSAFS",
        "PAYEMS",
        "ICSA",
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "PCEPILFE",
        "DFF",
        "DGS2",
        "DGS10",
        "BAMLC0A0CM",
        "BAMLH0A0HYM2",
        "NFCI",
    }
