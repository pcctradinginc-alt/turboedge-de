"""Tests for the ENTSO-E Transparency Platform adapter (`adapters/entsoe.py`).

Offline by default: parsing tests replay XML fixtures under
`tests/fixtures/external/entsoe/`. Every fixture in that directory is
**hand-built**, most of them transcribing the structure of ENTSO-E's own
official Postman-collection worked examples viewed live in a browser during
development (no `ENTSOE_SECURITY_TOKEN` is available in this environment,
so no live API call was made) -- see each fixture's own header comment and
`adapters/entsoe.py`'s module docstring ("VERIFIED VS DOCUMENTED") for
exactly what is transcribed-from-a-viewed-example vs. reconstructed from
corroborated public documentation. Only the fetch-shape tests touch HTTP,
and even those are mocked with `respx`; no test in this file makes a real
network call.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import httpx
import pytest
import respx

from turboedge.adapters import entsoe as entsoe_module
from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.adapters.entsoe import (
    GERMANY_LUXEMBOURG_EIC,
    EntsoeAdapter,
    EntsoeCredentialError,
    EntsoeQueryError,
    build_native_identifier,
    parse_gl_market_document,
    parse_native_identifier,
)
from turboedge.external.adapter import FetchedPayload, conservative_available_at
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, RawPayload, SeriesSpec

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "external" / "entsoe"
_BASE_URL = "https://web-api.tp.entsoe.eu/api"
_RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_DIR / f"{name}.xml").read_bytes()


def _load_spec(
    *,
    series_id: str = "ENTSOE.CZ.LOAD.ACTUAL",
    processType: str = "A16",
    lag_hours: float = 2.0,
) -> SeriesSpec:
    return SeriesSpec(
        source="entsoe",
        series_id=series_id,
        name="Czech actual total load",
        category="energy",
        unit="MW",
        frequency="hourly",
        native_identifier=build_native_identifier(
            "A65", "10YCZ-CEPS-----N", processType=processType
        ),
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
        conservative_release_lag_hours=lag_hours,
    )


def _generation_spec(
    *, series_id: str = "ENTSOE.DE.GENERATION.BIOMASS", lag_hours: float = 2.0
) -> SeriesSpec:
    return SeriesSpec(
        source="entsoe",
        series_id=series_id,
        name="German biomass actual generation",
        category="energy",
        unit="MW",
        frequency="quarter_hourly",
        native_identifier=build_native_identifier(
            "A75", "10Y1001A1001A83F", processType="A16", psrType="B01"
        ),
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
        conservative_release_lag_hours=lag_hours,
    )


def _generation_spec_no_psrtype(**kwargs: object) -> SeriesSpec:
    spec = _generation_spec(**kwargs)  # type: ignore[arg-type]
    return spec.model_copy(
        update={
            "native_identifier": build_native_identifier(
                "A75", "10Y1001A1001A83F", processType="A16"
            )
        }
    )


def _flows_spec(*, series_id: str = "ENTSOE.DE_BE.FLOW", lag_hours: float = 2.0) -> SeriesSpec:
    return SeriesSpec(
        source="entsoe",
        series_id=series_id,
        name="DE(Amprion) -> BE physical flow",
        category="energy",
        unit="MW",
        frequency="hourly",
        native_identifier=build_native_identifier(
            "A11", "10YBE----------2", out_Domain="10YDE-RWENET---I"
        ),
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
        conservative_release_lag_hours=lag_hours,
    )


# --------------------------------------------------------------------------
# native_identifier parse/build
# --------------------------------------------------------------------------


def test_build_and_parse_native_identifier_roundtrip() -> None:
    raw = build_native_identifier("A65", GERMANY_LUXEMBOURG_EIC, processType="A16")
    assert raw == "A65|10Y1001A1001A82H|processType=A16"
    document_type, domain_eic, extra = parse_native_identifier(raw)
    assert document_type == "A65"
    assert domain_eic == GERMANY_LUXEMBOURG_EIC
    assert extra == {"processType": "A16"}


def test_parse_native_identifier_empty_extra_segment() -> None:
    document_type, domain_eic, extra = parse_native_identifier("A11|10YBE----------2|")
    assert document_type == "A11"
    assert domain_eic == "10YBE----------2"
    assert extra == {}


def test_parse_native_identifier_multiple_extras() -> None:
    _, _, extra = parse_native_identifier("A75|10Y1001A1001A83F|processType=A16,psrType=B01")
    assert extra == {"processType": "A16", "psrType": "B01"}


def test_parse_native_identifier_malformed_raises() -> None:
    with pytest.raises(AdapterError, match="does not match"):
        parse_native_identifier("A65-only-one-segment")


def test_parse_native_identifier_malformed_extra_raises() -> None:
    with pytest.raises(AdapterError, match="malformed extra parameter"):
        parse_native_identifier("A65|10Y1001A1001A82H|processTypeA16")


# --------------------------------------------------------------------------
# ISO 8601 duration parsing (the core of the position->timestamp reconstruction)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PT15M", timedelta(minutes=15)),
        ("PT30M", timedelta(minutes=30)),
        ("PT60M", timedelta(minutes=60)),
        ("P1D", timedelta(days=1)),
    ],
)
def test_parse_iso8601_duration(text: str, expected: timedelta) -> None:
    assert entsoe_module._parse_iso8601_duration(text) == expected


def test_parse_iso8601_duration_rejects_unrecognized_shape() -> None:
    with pytest.raises(ValueError, match="unrecognized ISO 8601 duration"):
        entsoe_module._parse_iso8601_duration("not-a-duration")


# --------------------------------------------------------------------------
# multi-window concatenation splitting
# --------------------------------------------------------------------------


def test_split_xml_documents_single_document_unchanged() -> None:
    content = _fixture_bytes("gl_load_actual_a65_a16_pt60m")
    assert entsoe_module._split_xml_documents(content) == [content]


def test_split_xml_documents_recovers_concatenated_windows() -> None:
    first = _fixture_bytes("gl_load_actual_a65_a16_pt60m")
    second = _fixture_bytes("gl_load_dayahead_a65_a01_pt60m")
    concatenated = first + second
    chunks = entsoe_module._split_xml_documents(concatenated)
    assert chunks == [first, second]


def test_window_bounds_splits_long_ranges() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2023, 1, 1, tzinfo=UTC)
    windows = entsoe_module._window_bounds(start, end, max_days=364)
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for window_start, window_end in windows:
        assert (window_end - window_start).days <= 364
    # contiguous, no gaps or overlaps
    for (_, prev_end), (next_start, _) in pairwise(windows):
        assert prev_end == next_start


def test_window_bounds_single_window_when_short() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 8, tzinfo=UTC)
    assert entsoe_module._window_bounds(start, end, max_days=364) == [(start, end)]


def test_window_bounds_rejects_inverted_range() -> None:
    with pytest.raises(AdapterError, match="must be before"):
        entsoe_module._window_bounds(
            datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), max_days=364
        )


# --------------------------------------------------------------------------
# normal parse: actual total load (A65 + A16), PT60M
# --------------------------------------------------------------------------


def test_parse_actual_load_pt60m() -> None:
    spec = _load_spec()
    result = parse_gl_market_document(
        _fixture_bytes("gl_load_actual_a65_a16_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )

    # Fixture has positions 1, 2, 4 -- position 3 is a deliberate gap.
    assert len(result.observations) == 3
    assert result.missing_series == ()
    by_position_time = {o.observation_time: o.value for o in result.observations}
    assert by_position_time[datetime(2023, 3, 3, 0, 0, tzinfo=UTC)] == pytest.approx(7146)
    assert by_position_time[datetime(2023, 3, 3, 1, 0, tzinfo=UTC)] == pytest.approx(7100)
    assert by_position_time[datetime(2023, 3, 3, 3, 0, tzinfo=UTC)] == pytest.approx(7050)
    assert datetime(2023, 3, 3, 2, 0, tzinfo=UTC) not in by_position_time

    latest = result.observations[0]
    assert latest.series_id == spec.series_id
    assert latest.unit == "MW"
    assert latest.frequency == "hourly"
    assert latest.source == "entsoe"
    assert latest.source_version == "entsoe_a65"
    assert latest.parser_version == "1"
    assert latest.quality_score == 1.0
    assert latest.retrieved_at == _RETRIEVED_AT
    assert latest.is_stale is False


def test_missing_position_is_a_warning_not_interpolated() -> None:
    spec = _load_spec()
    result = parse_gl_market_document(
        _fixture_bytes("gl_load_actual_a65_a16_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert any("missing position" in w and "3" in w for w in result.warnings)
    assert all(o.value != 0.0 for o in result.observations)


def test_vintage_time_and_revision_index_from_document_fields() -> None:
    spec = _load_spec()
    result = parse_gl_market_document(
        _fixture_bytes("gl_load_actual_a65_a16_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )
    expected_vintage = datetime(2023, 8, 18, 12, 18, 45, tzinfo=UTC)
    assert all(o.vintage_time == expected_vintage for o in result.observations)
    # <revisionNumber>1</revisionNumber> -> revision_index 0 (first release).
    assert all(o.revision_index == 0 for o in result.observations)
    assert all(o.source_release_time is None for o in result.observations)


def test_availability_uses_resolve_available_at_conservative_date() -> None:
    spec = _load_spec(lag_hours=2.0)
    result = parse_gl_market_document(
        _fixture_bytes("gl_load_actual_a65_a16_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )
    target_time = datetime(2023, 3, 3, 0, 0, tzinfo=UTC)
    obs = next(o for o in result.observations if o.observation_time == target_time)
    assert obs.availability_precision == str(AvailabilityPrecision.CONSERVATIVE_DATE)
    assert obs.available_at == conservative_available_at(obs.observation_time.date(), lag_hours=2.0)


# --------------------------------------------------------------------------
# day-ahead load forecast (A65 + A01) -- same GL shape, different processType
# --------------------------------------------------------------------------


def test_parse_day_ahead_load_forecast() -> None:
    spec = _load_spec(series_id="ENTSOE.CZ.LOAD.DAYAHEAD", processType="A01")
    result = parse_gl_market_document(
        _fixture_bytes("gl_load_dayahead_a65_a01_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    assert len(result.observations) == 2
    values = {o.observation_time: o.value for o in result.observations}
    assert values[datetime(2023, 8, 14, 0, 0, tzinfo=UTC)] == pytest.approx(6800)
    assert values[datetime(2023, 8, 14, 1, 0, tzinfo=UTC)] == pytest.approx(6750)


# --------------------------------------------------------------------------
# actual generation per type (A75 + A16), PT15M -- the hand-checked mapping
# --------------------------------------------------------------------------


def test_parse_generation_pt15m_hand_checked_position_mapping() -> None:
    """The core off-by-one risk this adapter exists to avoid: position 1 must
    map to the Period's own `start`, not `start + resolution`."""
    spec = _generation_spec()
    result = parse_gl_market_document(
        _fixture_bytes("gl_generation_actual_a75_a16_pt15m"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    assert len(result.observations) == 4

    by_time = {o.observation_time: o.value for o in result.observations}
    assert by_time[datetime(2023, 8, 15, 22, 0, tzinfo=UTC)] == pytest.approx(120)  # position 1
    assert by_time[datetime(2023, 8, 15, 22, 15, tzinfo=UTC)] == pytest.approx(125)  # position 2
    assert by_time[datetime(2023, 8, 15, 22, 30, tzinfo=UTC)] == pytest.approx(130)  # position 3
    assert by_time[datetime(2023, 8, 15, 22, 45, tzinfo=UTC)] == pytest.approx(128)  # position 4


def test_generation_response_element_differs_from_request_param_name() -> None:
    """A75 is requested with `in_Domain` but the response carries
    `inBiddingZone_Domain.mRID` -- the adapter must match on the response
    element, not assume the two names are identical (unlike A65)."""
    spec = _generation_spec()
    result = parse_gl_market_document(
        _fixture_bytes("gl_generation_actual_a75_a16_pt15m"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert len(result.observations) == 4
    assert result.missing_series == ()


_AMBIGUOUS_PSR_FIXTURE = "gl_generation_actual_a75_ambiguous_psrtype"


def test_generation_ambiguous_psrtype_without_disambiguating_extra_is_refused() -> None:
    spec = _generation_spec_no_psrtype()
    result = parse_gl_market_document(
        _fixture_bytes(_AMBIGUOUS_PSR_FIXTURE), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("ambiguous" in w.lower() for w in result.warnings)


def test_generation_psrtype_extra_selects_the_right_series() -> None:
    """The same ambiguous 2-series payload, but the SeriesSpec disambiguates
    via `psrType=B01` -- only the biomass series should be extracted."""
    spec = _generation_spec()  # native_identifier includes psrType=B01
    result = parse_gl_market_document(
        _fixture_bytes(_AMBIGUOUS_PSR_FIXTURE), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    assert len(result.observations) == 4
    values = sorted(o.value for o in result.observations)
    assert values == [10, 11, 12, 13]  # the B01 series' quantities, not B16's zeros


# --------------------------------------------------------------------------
# cross-border physical flows (A11) -- Publication_MarketDocument, not GL
# --------------------------------------------------------------------------


def test_parse_cross_border_flows_publication_market_document() -> None:
    spec = _flows_spec()
    result = parse_gl_market_document(
        _fixture_bytes("publication_flows_a11_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.warnings == ()
    assert len(result.observations) == 3
    values = {o.observation_time: o.value for o in result.observations}
    assert values[datetime(2023, 8, 23, 22, 0, tzinfo=UTC)] == pytest.approx(340)
    assert values[datetime(2023, 8, 23, 23, 0, tzinfo=UTC)] == pytest.approx(355)


def test_flows_out_domain_mismatch_is_skipped() -> None:
    spec = _flows_spec()
    mismatched = spec.model_copy(
        update={
            "native_identifier": build_native_identifier(
                "A11", "10YBE----------2", out_Domain="10YFR-RTE------C"
            )
        }
    )
    result = parse_gl_market_document(
        _fixture_bytes("publication_flows_a11_pt60m"), mismatched, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (mismatched.series_id,)
    assert any("no TimeSeries" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Acknowledgement_MarketDocument: no-data vs genuine error
# --------------------------------------------------------------------------


def test_acknowledgement_no_matching_data_is_a_warning_not_an_exception() -> None:
    spec = _load_spec()
    result = parse_gl_market_document(
        _fixture_bytes("acknowledgement_no_matching_data"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("no matching data" in w.lower() for w in result.warnings)


def test_acknowledgement_auth_error_raises() -> None:
    spec = _load_spec()
    with pytest.raises(EntsoeQueryError, match="Authentication failed"):
        parse_gl_market_document(
            _fixture_bytes("acknowledgement_auth_error"), spec, retrieved_at=_RETRIEVED_AT
        )


def test_acknowledgement_same_code_different_text_distinguishes_by_text_only() -> None:
    """Both fixtures use Reason code 999 -- the split must be driven by
    Reason/text, never the code (module docstring)."""
    no_data = _fixture_bytes("acknowledgement_no_matching_data")
    auth_error = _fixture_bytes("acknowledgement_auth_error")
    assert b"<code>999</code>" in no_data
    assert b"<code>999</code>" in auth_error

    spec = _load_spec()
    result = parse_gl_market_document(no_data, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    with pytest.raises(EntsoeQueryError):
        parse_gl_market_document(auth_error, spec, retrieved_at=_RETRIEVED_AT)


# --------------------------------------------------------------------------
# schema drift -> warning, not a crash (except the Acknowledgement error case above)
# --------------------------------------------------------------------------


def test_non_xml_payload_warns_and_returns_no_observations() -> None:
    spec = _load_spec()
    result = parse_gl_market_document(b"not xml at all", spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("unparsable XML" in w for w in result.warnings)


def test_empty_payload_warns_and_returns_no_observations() -> None:
    spec = _load_spec()
    result = parse_gl_market_document(b"", spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("empty payload" in w for w in result.warnings)


def test_unrecognized_root_element_warns() -> None:
    spec = _load_spec()
    content = (
        b'<?xml version="1.0"?><SomeOtherDocument xmlns="urn:example"><x/></SomeOtherDocument>'
    )
    result = parse_gl_market_document(content, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert any("unrecognized root element" in w for w in result.warnings)


def test_document_with_no_timeseries_warns() -> None:
    spec = _load_spec()
    content = (
        b'<?xml version="1.0"?>'
        b'<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">'
        b"<mRID>x</mRID><revisionNumber>1</revisionNumber>"
        b"<process.processType>A16</process.processType>"
        b"<createdDateTime>2026-09-28T00:00:00Z</createdDateTime>"
        b"</GL_MarketDocument>"
    )
    result = parse_gl_market_document(content, spec, retrieved_at=_RETRIEVED_AT)
    assert result.observations == []
    assert any("no TimeSeries elements" in w for w in result.warnings)


def test_unsupported_document_type_warns_not_crashes() -> None:
    spec = _load_spec().model_copy(
        update={"native_identifier": build_native_identifier("A99", "10YCZ-CEPS-----N")}
    )
    result = parse_gl_market_document(
        _fixture_bytes("gl_load_actual_a65_a16_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("unsupported ENTSO-E documentType" in w for w in result.warnings)


def test_missing_domain_eic_warns_not_crashes() -> None:
    spec = _load_spec().model_copy(
        update={"native_identifier": build_native_identifier("A65", "", processType="A16")}
    )
    result = parse_gl_market_document(
        _fixture_bytes("gl_load_actual_a65_a16_pt60m"), spec, retrieved_at=_RETRIEVED_AT
    )
    assert result.observations == []
    assert any("missing the domain EIC" in w for w in result.warnings)


def test_parse_does_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """`parse()` must be pure: no HttpClient, no network access, ever."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parse() must not perform network I/O")

    monkeypatch.setattr(httpx.Client, "request", _boom)

    adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    spec = _load_spec()
    payload = FetchedPayload(
        source="entsoe",
        dataset=spec.series_id,
        url=f"{_BASE_URL}?documentType=A65",
        content=_fixture_bytes("gl_load_actual_a65_a16_pt60m"),
        http_status=200,
        content_type="application/xml",
        retrieved_at=_RETRIEVED_AT,
        request_fingerprint=f"GET {_BASE_URL}?documentType=A65",
    )
    result = adapter.parse(payload, spec)
    assert len(result.observations) == 3


# --------------------------------------------------------------------------
# fetch() -- request shape, credential handling, windowing, no-credential-leak
# --------------------------------------------------------------------------


def test_fetch_raises_typed_error_when_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENTSOE_SECURITY_TOKEN", raising=False)
    adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    with pytest.raises(EntsoeCredentialError, match="ENTSOE_SECURITY_TOKEN"):
        adapter.fetch(_load_spec())


def test_fetch_unsupported_document_type_raises_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENTSOE_SECURITY_TOKEN", "test-token")
    with respx.mock:
        route = respx.get(_BASE_URL).mock(return_value=httpx.Response(200, content=b"<x/>"))
        adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))
        spec = _load_spec().model_copy(
            update={"native_identifier": build_native_identifier("A99", "10YCZ-CEPS-----N")}
        )
        with pytest.raises(AdapterError, match="unsupported documentType"):
            adapter.fetch(spec, since=datetime(2026, 9, 20, tzinfo=UTC).date())
        assert not route.called


@respx.mock
def test_fetch_sends_documenttype_domain_and_period_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENTSOE_SECURITY_TOKEN", "test-token")
    route = respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("gl_load_actual_a65_a16_pt60m"))
    )
    adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_load_spec(), since=datetime(2026, 9, 20, tzinfo=UTC).date())

    assert route.called
    sent = route.calls.last.request
    assert sent.url.params["documentType"] == "A65"
    assert sent.url.params["processType"] == "A16"
    assert sent.url.params["outBiddingZone_Domain"] == "10YCZ-CEPS-----N"
    assert sent.url.params["periodStart"] == "202609200000"
    assert len(sent.url.params["periodStart"]) == 12
    assert sent.url.params["securityToken"] == "test-token"
    assert isinstance(payload, FetchedPayload)
    assert payload.source == "entsoe"
    assert payload.http_status == 200


@respx.mock
def test_fetch_defaults_to_short_lookback_when_since_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENTSOE_SECURITY_TOKEN", "test-token")
    respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("gl_load_actual_a65_a16_pt60m"))
    )
    adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    adapter.fetch(_load_spec())

    sent = respx.calls.last.request
    start = datetime.strptime(sent.url.params["periodStart"], "%Y%m%d%H%M").replace(tzinfo=UTC)
    end = datetime.strptime(sent.url.params["periodEnd"], "%Y%m%d%H%M").replace(tzinfo=UTC)
    assert (end - start).days == entsoe_module._DEFAULT_LOOKBACK_DAYS


@respx.mock
def test_fetch_url_and_fingerprint_never_contain_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "totally-secret-entsoe-token-0123456789ab"
    monkeypatch.setenv("ENTSOE_SECURITY_TOKEN", secret)
    respx.get(_BASE_URL).mock(
        return_value=httpx.Response(200, content=_fixture_bytes("gl_load_actual_a65_a16_pt60m"))
    )
    adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    payload = adapter.fetch(_load_spec(), since=datetime(2026, 9, 20, tzinfo=UTC).date())

    lowered = f"{payload.url} {payload.request_fingerprint}".lower()
    assert secret not in payload.url
    assert secret not in payload.request_fingerprint
    assert "securitytoken=" not in lowered
    assert "securitytoken" not in lowered


def test_raw_payload_credential_backstop_does_not_actually_cover_securitytoken() -> None:
    """Documents a real gap found while building this adapter, rather than
    asserting a false guarantee: `RawPayload._no_credentials_in_fingerprint`
    (external/schemas.py, out of this task's file list) matches credential
    *parameter names* with `\\b(...)=`, which requires a word boundary
    immediately before the name. ENTSO-E's parameter is the camelCase
    `securityToken`, and `\\b` does **not** match between "y" and "T" (both
    are word characters to the regex engine) -- so `securityToken=SECRET`
    slips through that backstop uncaught, unlike FRED's `api_key=` (preceded
    by `&`, a non-word character, so `\\b` matches there). This is why this
    adapter's own credential-free `public_params` dict (never `request.url`)
    is the *only* real protection for ENTSO-E, verified directly by
    `test_fetch_url_and_fingerprint_never_contain_credential` above -- the
    shared backstop cannot be relied on for this source. Reported to the
    task lead as a follow-up rather than fixed here (`external/schemas.py`
    is outside this task's assigned files)."""
    with pytest.raises(ValueError, match="still carries a value"):
        RawPayload(
            payload_id="entsoe:CZ.LOAD:deadbeef",
            source="entsoe",
            dataset="ENTSOE.CZ.LOAD.ACTUAL",
            # A leaked FRED-style `api_key=` (which the shared regex DOES
            # catch) confirms the validator itself still works in general;
            # it is `securityToken=` specifically that it cannot see (see
            # docstring above) -- there is no fixture string that both (a)
            # names ENTSO-E's real parameter and (b) trips this validator.
            request_fingerprint=(
                "GET https://web-api.tp.entsoe.eu/api?documentType=A65&api_key=SECRET"
            ),
            retrieved_at=_RETRIEVED_AT,
            http_status=200,
            content_type="application/xml",
            byte_size=10,
            payload_hash="a" * 64,
            stored_path="/tmp/x.xml",
            parser_version="1",
        )


@respx.mock
def test_fetch_windows_a_long_range_into_multiple_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `since` more than `_MAX_WINDOW_DAYS` in the past must be split into
    consecutive sub-year requests, each still going through the shared
    rate-limited HttpClient, then concatenated byte-for-byte into one
    FetchedPayload (module docstring "WINDOWED FETCHING")."""
    monkeypatch.setenv("ENTSOE_SECURITY_TOKEN", "test-token")
    first_window = _fixture_bytes("gl_load_actual_a65_a16_pt60m")
    second_window = _fixture_bytes("gl_load_actual_a65_a16_pt60m")
    route = respx.get(_BASE_URL).mock(
        side_effect=[
            httpx.Response(200, content=first_window),
            httpx.Response(200, content=second_window),
        ]
    )
    adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))

    # 400 days: strictly more than _MAX_WINDOW_DAYS (364) but less than
    # 2 * 364, so this always splits into exactly 2 windows regardless of
    # exactly when "now" falls.
    since = (datetime.now(UTC) - timedelta(days=400)).date()
    payload = adapter.fetch(_load_spec(), since=since)

    assert route.call_count == 2
    assert payload.content == first_window + second_window
    assert payload.dataset.endswith(":2windows")
    assert payload.request_fingerprint.count("GET ") == 2

    # And the concatenated content is fully reparsable, recovering
    # observations from both (identical) windows.
    spec = _load_spec()
    result = parse_gl_market_document(payload.content, spec, retrieved_at=_RETRIEVED_AT)
    assert len(result.observations) == 3 + 3  # same fixture, twice


def test_process_type_mismatch_across_windows_is_skipped_not_mixed() -> None:
    """If an archived multi-window payload ever mixed an actual-load
    document with a day-ahead-forecast document under one processType=A16
    series_id, the mismatched document must be skipped, not silently
    folded into the same series (defence in depth, see module docstring)."""
    actual = _fixture_bytes("gl_load_actual_a65_a16_pt60m")  # process.processType A16
    forecast = _fixture_bytes("gl_load_dayahead_a65_a01_pt60m")  # process.processType A01
    spec = _load_spec(processType="A16")

    result = parse_gl_market_document(actual + forecast, spec, retrieved_at=_RETRIEVED_AT)

    assert len(result.observations) == 3  # only the A16 document's 3 points
    assert any("does not match requested processType" in w for w in result.warnings)


def test_adapter_identity() -> None:
    adapter = EntsoeAdapter(HttpClient(user_agent="turboedge-test/1.0"))
    assert adapter.source_id == "entsoe"
    assert adapter.parser_version == "1"
