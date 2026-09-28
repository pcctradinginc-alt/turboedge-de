"""Contract tests for the EU Business & Consumer Surveys adapter.

Fixtures are real Eurostat JSON-stat 2.0 payloads captured live on
2026-09-28 from the two dissemination-API endpoints this workstream
verified (see the module docstring in `turboedge.adapters.eu_bcs`):

    ei_bssi_m_r2_DE.json                geo=DE, 2 months, all 6 indicators
    ei_bssi_m_r2_EA21.json              geo=EA21, 1 month, all 6 indicators
    ei_bsco_m_DE.json                   geo=DE, 2 months, 12 indicators
    ei_bssi_m_r2_unknown_geo_EA.json    geo=EA (bare code -- does not exist)
    ei_bssi_m_r2_unknown_indicator_EEI.json  indic=BS-EEI-BAL -- does not exist

The DE fixture's `value` map is hand-decoded in the module docstring and
below; the expected numbers are copied from that by-hand decode, not
derived from the code under test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from turboedge.adapters.base import AdapterError
from turboedge.adapters.eu_bcs import (
    EuBcsAdapter,
    build_native_identifier,
    decode_jsonstat,
    make_series_spec,
    parse_jsonstat_payload,
    parse_native_identifier,
)
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass

FIXTURES = Path(__file__).parent.parent / "fixtures" / "external" / "eu_bcs"
RETRIEVED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def load_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def load_json(name: str) -> dict:
    return json.loads(load_bytes(name))


def esi_spec(geo: str = "DE", s_adj: str = "SA", dataset: str = "ei_bssi_m_r2"):
    return make_series_spec(
        dataset=dataset,
        indic="BS-ESI-I",
        geo=geo,
        s_adj=s_adj,
        series_id=f"EU_BCS.{geo}.ESI.{s_adj}",
        name="Economic sentiment indicator",
        category="business_consumer_survey",
    )


def cci_spec(geo: str = "DE", s_adj: str = "SA"):
    return make_series_spec(
        dataset="ei_bssi_m_r2",
        indic="BS-CCI-BAL",
        geo=geo,
        s_adj=s_adj,
        series_id=f"EU_BCS.{geo}.CCI.{s_adj}",
        name="Construction confidence indicator",
        category="business_consumer_survey",
    )


# --- JSON-stat decode -------------------------------------------------


def test_decode_jsonstat_matches_hand_checked_mapping() -> None:
    """Pin the row-major flat-index decode against values worked out by hand.

    id=[freq,indic,s_adj,geo,time], size=[1,6,2,1,2] -> index = indic*4 +
    s_adj*2 + time (freq and geo both have size 1 and contribute nothing).
    Cross-checked against `numpy.unravel_index(i, size, order="C")` while
    writing this adapter.
    """
    raw = load_json("ei_bssi_m_r2_DE.json")
    rows = decode_jsonstat(raw)
    by_key = {(r["indic"], r["s_adj"], r["time"]): r["value"] for r in rows}

    assert by_key[("BS-CCI-BAL", "NSA", "2026-07")] == -10.1
    assert by_key[("BS-CCI-BAL", "NSA", "2026-08")] == -10.2
    assert by_key[("BS-CCI-BAL", "SA", "2026-07")] == -14.3
    assert by_key[("BS-ESI-I", "SA", "2026-07")] == 92.8
    assert by_key[("BS-ESI-I", "SA", "2026-08")] == 94.1
    assert by_key[("BS-SCI-BAL", "SA", "2026-08")] == 5.5

    # ESI has no NSA series at all: the sparse map has a genuine gap here,
    # not merely an out-of-order key.
    assert ("BS-ESI-I", "NSA", "2026-07") not in by_key
    assert ("BS-ESI-I", "NSA", "2026-08") not in by_key

    # freq and geo are constant (freq=M, geo=DE) across every row.
    assert {r["freq"] for r in rows} == {"M"}
    assert {r["geo"] for r in rows} == {"DE"}


def test_decode_jsonstat_handles_extra_dimension_generically() -> None:
    """ei_bsco_m has 6 dimensions (extra `unit`), ei_bssi_m_r2 has 5.

    The same decoder must handle both without special-casing dimension
    names or count.
    """
    raw = load_json("ei_bsco_m_DE.json")
    rows = decode_jsonstat(raw)
    assert rows
    assert {r["unit"] for r in rows} == {"BAL"}
    by_key = {(r["indic"], r["s_adj"], r["time"]): r["value"] for r in rows}
    assert by_key[("BS-CSMCI", "NSA", "2026-07")] == -13.8
    assert by_key[("BS-CSMCI", "SA", "2026-08")] == -12.7


def test_decode_jsonstat_missing_required_key_raises() -> None:
    with pytest.raises(AdapterError, match="missing required key"):
        decode_jsonstat({"id": ["a"], "size": [1]})


def test_decode_jsonstat_id_size_length_mismatch_raises() -> None:
    with pytest.raises(AdapterError, match="dimensions but"):
        decode_jsonstat(
            {
                "id": ["a", "b"],
                "size": [1],
                "value": {},
                "dimension": {"a": {"category": {"index": {"x": 0}}}},
            }
        )


# --- parse_jsonstat_payload: normal cases ------------------------------


def test_esi_de_sa_parses_two_months_with_correct_availability() -> None:
    spec = esi_spec(geo="DE", s_adj="SA")
    result = parse_jsonstat_payload(
        load_bytes("ei_bssi_m_r2_DE.json"), spec, retrieved_at=RETRIEVED_AT
    )
    assert not result.warnings
    assert not result.missing_series
    assert [o.value for o in result.observations] == [92.8, 94.1]

    july, august = result.observations
    assert july.observation_time == datetime(2026, 7, 1, tzinfo=UTC)
    assert august.observation_time == datetime(2026, 8, 1, tzinfo=UTC)
    # available_at is anchored to the day *after* the end of the described
    # month (see module docstring), not the day after it starts.
    assert july.available_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert august.available_at == datetime(2026, 9, 1, tzinfo=UTC)
    for obs in result.observations:
        assert obs.series_id == spec.series_id
        assert obs.source == "eu_bcs"
        assert obs.availability_precision == "UNKNOWN"
        assert obs.revision_index is None
        assert obs.vintage_time == datetime(2026, 8, 28, 9, 0, tzinfo=UTC)


def test_cci_de_nsa_and_sa_both_parse() -> None:
    nsa = parse_jsonstat_payload(
        load_bytes("ei_bssi_m_r2_DE.json"), cci_spec(s_adj="NSA"), retrieved_at=RETRIEVED_AT
    )
    sa = parse_jsonstat_payload(
        load_bytes("ei_bssi_m_r2_DE.json"), cci_spec(s_adj="SA"), retrieved_at=RETRIEVED_AT
    )
    assert [o.value for o in nsa.observations] == [-10.1, -10.2]
    assert [o.value for o in sa.observations] == [-14.3, -14.3]


def test_ea21_geo_parses() -> None:
    """EA21 is the euro-area code verified live today; a bare "EA" is not
    (covered separately below)."""
    spec = esi_spec(geo="EA21", s_adj="SA")
    result = parse_jsonstat_payload(
        load_bytes("ei_bssi_m_r2_EA21.json"), spec, retrieved_at=RETRIEVED_AT
    )
    assert not result.missing_series
    assert [o.value for o in result.observations] == [98.4]
    assert result.observations[0].observation_time == datetime(2026, 8, 1, tzinfo=UTC)


def test_ei_bsco_m_csmci_de_sa_parses() -> None:
    spec = make_series_spec(
        dataset="ei_bsco_m",
        indic="BS-CSMCI",
        geo="DE",
        s_adj="SA",
        series_id="EU_BCS.DE.CSMCI_DETAIL.SA",
        name="Consumer confidence indicator, detailed survey",
        category="business_consumer_survey",
    )
    result = parse_jsonstat_payload(
        load_bytes("ei_bsco_m_DE.json"), spec, retrieved_at=RETRIEVED_AT
    )
    assert not result.missing_series
    assert [o.value for o in result.observations] == [-13.8, -12.7]


# --- missing / unverified identifiers -----------------------------------


def test_esi_nsa_de_is_reported_missing_not_fabricated() -> None:
    """ESI has no NSA variant published at all -- must not silently emit
    nothing with no signal; must land in `missing_series` with a warning."""
    spec = esi_spec(geo="DE", s_adj="NSA")
    result = parse_jsonstat_payload(
        load_bytes("ei_bssi_m_r2_DE.json"), spec, retrieved_at=RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert result.warnings


def test_unknown_geo_bare_ea_is_reported_missing() -> None:
    """Live-verified: geo=EA (bare code) returns an empty geo dimension."""
    spec = esi_spec(geo="EA", s_adj="SA")
    result = parse_jsonstat_payload(
        load_bytes("ei_bssi_m_r2_unknown_geo_EA.json"), spec, retrieved_at=RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("geo" in w and "not present" in w for w in result.warnings)


def test_unknown_indicator_eei_is_reported_missing() -> None:
    """Live-verified: BS-EEI-BAL does not exist in ei_bssi_m_r2 at all --
    this workstream's brief was wrong to expect an Employment Expectations
    Indicator here, and the adapter must not invent one (CLAUDE.md rule 1)."""
    spec = make_series_spec(
        dataset="ei_bssi_m_r2",
        indic="BS-EEI-BAL",
        geo="DE",
        s_adj="SA",
        series_id="EU_BCS.DE.EEI.SA",
        name="Employment expectations indicator (unverified)",
        category="business_consumer_survey",
    )
    result = parse_jsonstat_payload(
        load_bytes("ei_bssi_m_r2_unknown_indicator_EEI.json"), spec, retrieved_at=RETRIEVED_AT
    )
    assert result.observations == []
    assert result.missing_series == (spec.series_id,)
    assert any("indicator" in w and "not present" in w for w in result.warnings)


# --- schema drift / malformed payloads -----------------------------------


def test_non_jsonstat_payload_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(AdapterError, match="does not look like a JSON-stat"):
        parse_jsonstat_payload(b'{"hello": "world"}', esi_spec(), retrieved_at=RETRIEVED_AT)


def test_invalid_json_raises() -> None:
    with pytest.raises(AdapterError, match="not valid JSON"):
        parse_jsonstat_payload(b"{not json", esi_spec(), retrieved_at=RETRIEVED_AT)


def test_missing_updated_field_warns_but_does_not_crash() -> None:
    raw = load_json("ei_bssi_m_r2_DE.json")
    del raw["updated"]
    result = parse_jsonstat_payload(
        json.dumps(raw).encode("utf-8"), esi_spec(), retrieved_at=RETRIEVED_AT
    )
    assert result.observations  # still parses
    assert any("no 'updated' timestamp" in w for w in result.warnings)
    assert all(o.vintage_time is None for o in result.observations)


# --- native_identifier ----------------------------------------------------


def test_native_identifier_round_trip() -> None:
    raw = build_native_identifier("ei_bssi_m_r2", "BS-ESI-I", "DE", "SA")
    assert parse_native_identifier(raw) == ("ei_bssi_m_r2", "BS-ESI-I", "DE", "SA")


def test_native_identifier_malformed_raises() -> None:
    with pytest.raises(AdapterError, match="does not match"):
        parse_native_identifier("not-the-right-shape")


def test_make_series_spec_declares_unknown_precision_and_forward_only() -> None:
    spec = esi_spec()
    assert spec.availability_precision is AvailabilityPrecision.UNKNOWN
    assert spec.backfill_class is BackfillClass.FORWARD_ONLY


# --- fetch(): no credentials, correct wiring ------------------------------


class _StubHttpClient:
    """Minimal stand-in for HttpClient -- records the URL, returns a fixture."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requested_urls: list[str] = []

    def get_bytes(self, url: str, **_kwargs: object) -> bytes:
        self.requested_urls.append(url)
        return self.body


def test_fetch_builds_url_without_credentials_and_fingerprint_is_clean() -> None:
    stub = _StubHttpClient(load_bytes("ei_bssi_m_r2_DE.json"))
    adapter = EuBcsAdapter(http_client=stub)  # type: ignore[arg-type]
    spec = esi_spec()

    payload = adapter.fetch(spec)

    assert payload.source == "eu_bcs"
    assert "ei_bssi_m_r2" in payload.url
    assert "indic=BS-ESI-I" in payload.url
    assert "geo=DE" in payload.url
    for marker in ("api_key=", "apikey=", "token=", "password=", "appid="):
        assert marker not in payload.url.lower()
        assert marker not in payload.request_fingerprint.lower()
    assert payload.content == stub.body
    assert stub.requested_urls == [payload.url]


def test_adapter_identity() -> None:
    adapter = EuBcsAdapter(http_client=_StubHttpClient(b"{}"))  # type: ignore[arg-type]
    assert adapter.source_id == "eu_bcs"
    assert adapter.parser_version
