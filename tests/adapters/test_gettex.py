"""Contract tests for the gettex (Boerse Muenchen) adapter.

No real network access (respx mocks every HTTP call). Real-capture fixtures
under ``tests/fixtures/gettex/`` come from the Round 3 data-source research
session -- see ``docs/data_sources.md`` (section 9 and follow-up 9a) and
``src/turboedge/adapters/gettex.py``'s module docstring for the research
context every test below is grounded in.

gettex exposes no ``ratio``/``currency``/``maturity``/``quanto`` field for
any product, so unlike ``tests/adapters/test_issuer_feeds.py`` (where the
source-provided ``ratio`` is just parsed), most of this adapter's behavior is
the *derivation* itself -- most tests here therefore use a synthetic
row-builder (``_gettex_product``, same "test-only synthetic-record builder"
pattern already used throughout ``test_issuer_feeds.py``) constructed so the
underlying spot, financing level, leverage, ratio and bid/ask are all
mutually exact (no floating-point derivation noise), making the
derive-then-verify pipeline's pass/fail boundary precisely testable. The real
fixtures are used for schema-shape realism and end-to-end smoke coverage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import structlog.testing

from turboedge.adapters.base import HttpClient
from turboedge.adapters.gettex import (
    _GETTEX_DEFAULT_BASE_URL,
    _GETTEX_PRODUCTS_PATH,
    _GETTEX_UNDERLYING_IDS,
    _RATIO_GRID,
    GettexAdapter,
    GettexRowError,
    _extract_quote,
    _fx_candidates_from_ratio_raw_at_1,
    _gettex_factory,
    _leverage_scaled_ratio_snap_tolerance,
    _leverage_scaled_verification_tolerance,
    _normalize_issuer,
    _ratio_reliability_for_outcome,
    _register,
    _robust_fx_median,
    _robust_reference_spot,
    _snap_ratio,
    _spot_from_leverage,
)
from turboedge.adapters.registry import PRODUCT_ADAPTER_FACTORIES, ProductSourceAdapter
from turboedge.config import SourceConfig
from turboedge.storage.schemas import (
    Direction,
    FieldReliability,
    HealthStatus,
    ProductSnapshot,
    ProductType,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "gettex"
PRODUCTS_URL = f"{_GETTEX_DEFAULT_BASE_URL}{_GETTEX_PRODUCTS_PATH}"


def _load_fixture(name: str) -> Any:
    return json.loads((FIXTURES_DIR / name).read_text())


def _fixed_clock(when: datetime) -> Callable[[], datetime]:
    return lambda: when


def _gettex_adapter(**kwargs: Any) -> GettexAdapter:
    http = HttpClient(user_agent="test-agent/1.0")
    return GettexAdapter(http, **kwargs)


def _gettex_page_response(
    products: list[dict[str, Any]], *, filtered_count: int | None = None, rows_per_page: int = 100
) -> dict[str, Any]:
    return {
        "data": {
            "groups": {"products": products},
            "pagination": {
                "page": 1,
                "rowsPerPage": rows_per_page,
                "filteredCount": filtered_count if filtered_count is not None else len(products),
                "totalCount": 837596,
            },
        },
        "status": "ok",
    }


_EPOCH_2026_09_11T12_00 = int(datetime(2026, 9, 11, 12, 0, tzinfo=UTC).timestamp() * 1000)


def _gettex_product(
    isin: str,
    *,
    direction: str = "long",
    spot: float = 25000.0,
    financing_level: float,
    ko_level: float | None = None,
    ratio: float = 0.01,
    fx: float = 1.0,
    issuer: str = "BNP Paribas",
    underlying_name: str = "DAX (Performance)",
    wkn: str | None = None,
    include_ask: bool = True,
    include_bid: bool = True,
    leverage_override: float | None = None,
    quote_timestamp_ms: int | None = _EPOCH_2026_09_11T12_00,
) -> dict[str, Any]:
    """Build one synthetic gettex `leverageProducts` row.

    Constructed so bid/ask exactly encode ``ratio`` at the given ``spot``
    (intrinsic pricing, +/-1% spread averaging to the exact mid) and
    ``leverage`` is the exact geometric ``spot / |spot - financing_level|``
    identity -- so the adapter's derivation recovers ``spot`` and ``ratio``
    exactly (up to float rounding), making pass/fail assertions precise.
    """
    moneyness = spot - financing_level if direction == "long" else financing_level - spot
    if moneyness <= 0:
        raise ValueError("moneyness must be positive for a valid long/short row")
    intrinsic = moneyness * ratio / fx
    bid = intrinsic * 0.99
    ask = intrinsic * 1.01
    leverage = (
        leverage_override if leverage_override is not None else spot / abs(spot - financing_level)
    )
    ko = ko_level if ko_level is not None else financing_level

    def quote_field(value: float | None) -> dict[str, Any]:
        if value is None:
            return {"valueTuple": None}
        return {
            "valueTuple": {
                "value": value,
                "size": 1000.0,
                "timestamp": quote_timestamp_ms,
            }
        }

    return {
        "wkn": {"value": wkn or isin[-6:]},
        "isin": {"value": isin},
        "underlying": {"value": underlying_name},
        "underlyings.price": {
            "valueTuple": {
                "value": spot,
                "size": 0.0,
                "timestamp": quote_timestamp_ms,
            }
        },
        "issuer": {"value": issuer},
        "structure.direction": {"value": direction},
        "financingLevelRefCurAbsolute": {"value": financing_level},
        "koLevelRefCurAbsolute": {"value": ko},
        "leverage": {"value": leverage},
        "bid": quote_field(bid if include_bid else 0.0),
        "ask": quote_field(ask if include_ask else 0.0),
    }


def _long_short_batch(
    *,
    spot: float = 25000.0,
    ratio: float = 0.01,
    n_each: int = 12,
    underlying_name: str = "DAX (Performance)",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in range(1, n_each + 1):
        rows.append(
            _gettex_product(
                f"DE000LONG{k:03d}",
                direction="long",
                spot=spot,
                financing_level=spot - 100.0 * k,
                ratio=ratio,
                underlying_name=underlying_name,
            )
        )
        rows.append(
            _gettex_product(
                f"DE000SHRT{k:03d}",
                direction="short",
                spot=spot,
                financing_level=spot + 100.0 * k,
                ratio=ratio,
                underlying_name=underlying_name,
            )
        )
    return rows


# ===========================================================================
# metadata / protocol / registration
# ===========================================================================


def test_gettex_metadata_and_name() -> None:
    adapter = _gettex_adapter()
    meta = adapter.metadata()
    assert meta.name == "gettex"
    assert meta.kind == "product"
    assert adapter.name == "gettex"


def test_gettex_satisfies_product_source_adapter_protocol() -> None:
    assert isinstance(_gettex_adapter(), ProductSourceAdapter)


def test_gettex_registration_is_idempotent() -> None:
    assert "gettex" in PRODUCT_ADAPTER_FACTORIES
    factory_before = PRODUCT_ADAPTER_FACTORIES["gettex"]
    _register()
    _register()
    assert "gettex" in PRODUCT_ADAPTER_FACTORIES
    assert PRODUCT_ADAPTER_FACTORIES["gettex"] is factory_before


def test_gettex_factory_builds_adapter_from_source_config() -> None:
    src = SourceConfig(
        enabled=True,
        base_url="https://gettex.wsd.com",
        timeout_s=20.0,
        min_interval_s=0.0,
        max_pages=5,
        user_agent="test-agent/1.0",
    )
    http = HttpClient(user_agent=src.user_agent)
    adapter = _gettex_factory(src, http)
    assert adapter.name == "gettex"


# ===========================================================================
# unit: leverage-implied reference spot (S_ref) derivation
# ===========================================================================


def test_spot_from_leverage_long_hand_example() -> None:
    # S=25000, F=20000 -> leverage = S / (S - F) = 5.0
    assert _spot_from_leverage(Direction.LONG, 20000.0, 5.0) == pytest.approx(25000.0)


def test_spot_from_leverage_short_hand_example() -> None:
    # S=25000, F=30000 -> leverage = S / (F - S) = 5.0
    assert _spot_from_leverage(Direction.SHORT, 30000.0, 5.0) == pytest.approx(25000.0)


def test_spot_from_leverage_degenerate_long_leverage_returns_none() -> None:
    # leverage <= 1 is a singularity/non-physical for LONG (denominator <= 0).
    assert _spot_from_leverage(Direction.LONG, 20000.0, 1.0) is None
    assert _spot_from_leverage(Direction.LONG, 20000.0, 0.5) is None


def test_robust_reference_spot_direction_balanced_median() -> None:
    # A small biased sample: LONG estimates cluster slightly high, SHORT
    # slightly low (mirrors the real wrapper-margin bias documented in
    # pricing/cross_issuer.py) -- the direction-balanced average should land
    # between the two per-direction medians.
    candidates = [
        (25010.0, Direction.LONG),
        (25012.0, Direction.LONG),
        (24990.0, Direction.SHORT),
        (24988.0, Direction.SHORT),
    ]
    estimate = _robust_reference_spot(candidates, mad_k=5.0)
    assert estimate is not None
    assert estimate.direction_balanced is True
    assert estimate.value == pytest.approx((25011.0 + 24989.0) / 2.0)


def test_robust_reference_spot_mad_filters_outlier() -> None:
    candidates = [(25000.0 + i, Direction.LONG) for i in range(9)] + [(99999.0, Direction.LONG)]
    estimate = _robust_reference_spot(candidates, mad_k=5.0)
    assert estimate is not None
    assert estimate.n_rejected == 1
    assert abs(estimate.value - 25004.0) < 10.0


def test_robust_reference_spot_empty_returns_none() -> None:
    assert _robust_reference_spot([], mad_k=5.0) is None


# ===========================================================================
# unit: ratio grid snapping
# ===========================================================================


def test_snap_ratio_exact_grid_value() -> None:
    assert _snap_ratio(0.01, 0.03) == pytest.approx(0.01)


def test_snap_ratio_within_tolerance_snaps() -> None:
    # 2% below 0.01 -- within the default 3% tolerance.
    assert _snap_ratio(0.0098, 0.03) == pytest.approx(0.01)


def test_snap_ratio_at_exact_tolerance_boundary_snaps() -> None:
    # exactly 3% deviation from 0.01 -- the boundary is inclusive (<=).
    assert _snap_ratio(0.0103, 0.03) == pytest.approx(0.01)


def test_snap_ratio_just_outside_tolerance_rejected() -> None:
    # just over 3% deviation from every grid value -- must NOT snap.
    assert _snap_ratio(0.010301, 0.03) is None


def test_snap_ratio_between_two_grid_values_rejected() -> None:
    # 0.03 sits ~50%/~33% away from 0.02/0.05 -- nowhere near either.
    assert _snap_ratio(0.03, 0.03) is None


def test_snap_ratio_non_positive_rejected() -> None:
    assert _snap_ratio(0.0, 0.03) is None
    assert _snap_ratio(-0.01, 0.03) is None


# ===========================================================================
# unit: issuer normalization
# ===========================================================================


def test_normalize_issuer_known_mappings() -> None:
    assert _normalize_issuer("BNP Paribas") == "BNP Paribas"
    assert _normalize_issuer("Goldman Sachs Bank Europe SE") == "Goldman Sachs"
    assert _normalize_issuer("HSBC Trinkaus & Burkhardt GmbH") == "HSBC"
    assert _normalize_issuer("UniCredit Bank GmbH") == "UniCredit"


def test_normalize_issuer_unknown_passes_through_unchanged() -> None:
    assert _normalize_issuer("Some New Issuer AG") == "Some New Issuer AG"
    assert _normalize_issuer("  Padded Issuer  ") == "Padded Issuer"


# ===========================================================================
# unit: bid/ask zero-sentinel extraction (Pitfall 2)
# ===========================================================================


def test_extract_quote_zero_sentinel_treated_as_missing() -> None:
    value, _size, ts = _extract_quote(
        {"valueTuple": {"value": 0.0, "size": 0.0, "timestamp": _EPOCH_2026_09_11T12_00}}
    )
    assert value is None
    assert ts is not None  # diagnostics kept even though value is nulled


def test_extract_quote_missing_value_tuple_treated_as_missing() -> None:
    assert _extract_quote({}) == (None, None, None)
    assert _extract_quote(None) == (None, None, None)


def test_extract_quote_live_value_parsed() -> None:
    value, size, ts = _extract_quote(
        {"valueTuple": {"value": 1.23, "size": 500.0, "timestamp": _EPOCH_2026_09_11T12_00}}
    )
    assert value == pytest.approx(1.23)
    assert size == pytest.approx(500.0)
    assert ts is not None


# ===========================================================================
# fetch_products: derivation success (>=20 products, both directions)
# ===========================================================================


@respx.mock
def test_fetch_products_derives_ratio_for_at_least_20_long_and_short() -> None:
    rows = _long_short_batch(spot=25000.0, ratio=0.01, n_each=12)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(
        max_pages=1, rows_per_page=100, clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC))
    )
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) >= 20
    assert all(isinstance(s, ProductSnapshot) for s in snapshots)
    assert all(s.ratio == pytest.approx(0.01) for s in snapshots)
    # Phase B: every snapshot this call emits passed derive-then-verify
    # (outcome == "ok"), so ratio_reliability is always DERIVED_VERIFIED;
    # barrier/financing_level are raw gettex feed fields (no derivation),
    # so SOURCE_REPORTED.
    assert all(s.ratio_reliability == FieldReliability.DERIVED_VERIFIED for s in snapshots)
    assert all(s.barrier_reliability == FieldReliability.SOURCE_REPORTED for s in snapshots)
    assert all(s.financing_level_reliability == FieldReliability.SOURCE_REPORTED for s in snapshots)
    assert all(s.currency == "EUR" for s in snapshots)
    assert all(s.underlying_currency == "EUR" for s in snapshots)
    assert all(s.quanto is None for s in snapshots)  # EUR underlying -> not applicable
    assert all(s.product_type == ProductType.TURBO_OPEN_END for s in snapshots)
    assert all(s.open_end is True for s in snapshots)
    assert all(s.maturity is None for s in snapshots)
    assert all(s.knocked_out is False for s in snapshots)
    directions = {s.direction for s in snapshots}
    assert directions == {Direction.LONG, Direction.SHORT}
    assert not adapter.last_errors


@respx.mock
def test_fetch_products_bid_only_row_still_derives_ratio() -> None:
    row = _gettex_product(
        "DE000BIDONE1", direction="long", financing_level=24500.0, include_ask=False
    )
    # Pad with enough fresh, fully-quoted rows so a robust S_ref exists.
    rows = [row, *_long_short_batch(n_each=10)]
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    bid_only = [s for s in snapshots if s.isin == "DE000BIDONE1"]
    assert len(bid_only) == 1
    snapshot = bid_only[0]
    assert snapshot.ask is None
    assert snapshot.bid_only is True
    assert snapshot.quote_presence is False
    assert snapshot.quality_score == pytest.approx(0.3)
    assert snapshot.ratio == pytest.approx(0.01)


# ===========================================================================
# fetch_products: verification failure on a manipulated row
# ===========================================================================


@respx.mock
def test_verification_fails_and_drops_manipulated_row() -> None:
    good_rows = _long_short_batch(spot=25000.0, ratio=0.01, n_each=10)

    # Manipulated: priced as if ratio were 0.0102 (2% off-grid, snaps to
    # 0.01), but that snap no longer reproduces S_ref within the 0.3%
    # implied-spot tolerance (worked out in the module docstring / commit
    # message: financing_level=20000 gives a 0.4% deviation).
    financing_level = 20000.0
    spot = 25000.0
    moneyness = spot - financing_level
    manipulated_ratio = 0.0102
    price = moneyness * manipulated_ratio
    manipulated = {
        "wkn": {"value": "MANIPU"},
        "isin": {"value": "DE000MANIPU1"},
        "underlying": {"value": "DAX (Performance)"},
        "underlyings.price": {
            "valueTuple": {"value": spot, "size": 0.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
        "issuer": {"value": "BNP Paribas"},
        "structure.direction": {"value": "long"},
        "financingLevelRefCurAbsolute": {"value": financing_level},
        "koLevelRefCurAbsolute": {"value": financing_level},
        "leverage": {"value": spot / moneyness},
        "bid": {
            "valueTuple": {"value": price, "size": 1000.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
        "ask": {
            "valueTuple": {"value": price, "size": 1000.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
    }

    respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([*good_rows, manipulated]))
    )
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert "DE000MANIPU1" not in {s.isin for s in snapshots}
    errors_for_row = [e for e in adapter.last_errors if e.isin == "DE000MANIPU1"]
    assert len(errors_for_row) == 1
    assert (
        "verification" in errors_for_row[0].error.lower()
        or "quanto" in errors_for_row[0].error.lower()
    )
    # Phase B: a row that fails verification never reaches a ProductSnapshot
    # at all (`ratio` is a required, pricing-critical field -- there is
    # nothing to attach an UNVERIFIED ratio to), which is exactly what
    # `test_ratio_reliability_for_outcome_maps_every_non_ok_outcome_to_unverified`
    # below verifies directly at the mapping-function level.


# ===========================================================================
# Phase B ("Produktstammdaten haerten"): ratio_reliability mapping
# ===========================================================================


def test_ratio_reliability_for_outcome_ok_is_derived_verified() -> None:
    assert _ratio_reliability_for_outcome("ok") == FieldReliability.DERIVED_VERIFIED


@pytest.mark.parametrize("outcome", ["ratio_rejected", "verification_failed", "quanto_ambiguous"])
def test_ratio_reliability_for_outcome_maps_every_non_ok_outcome_to_unverified(
    outcome: str,
) -> None:
    # Mirrors _derive_ratio's actual outcome strings -- see fetch_products'
    # outcome handling. Every one of these currently means the row never
    # reaches a ProductSnapshot at all (see the manipulated-row test above),
    # but the mapping itself must stay correct and testable independent of
    # that fact.
    assert _ratio_reliability_for_outcome(outcome) == FieldReliability.UNVERIFIED


# ===========================================================================
# fetch_products: reference_spot cross-check
# ===========================================================================


@respx.mock
def test_reference_spot_mismatch_drops_all_products_and_warns() -> None:
    rows = _long_short_batch(spot=25000.0, ratio=0.01, n_each=10)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    healthcheck_route = respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response(rows[:10]))
    )
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["DAX"], reference_spot=26000.0)

    assert snapshots == []
    assert "DAX" in adapter._reference_spot_mismatches
    mismatch_events = [e for e in logs if e.get("event") == "gettex_reference_spot_mismatch"]
    assert len(mismatch_events) == 1

    result = adapter.healthcheck()
    assert result.status == HealthStatus.WARN
    assert "reference_spot_mismatch" in result.message
    assert healthcheck_route.called


@respx.mock
def test_reference_spot_within_tolerance_accepted() -> None:
    rows = _long_short_batch(spot=25000.0, ratio=0.01, n_each=10)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    # 0.1% off -- comfortably within the default 0.5% tolerance.
    snapshots = adapter.fetch_products(["DAX"], reference_spot=25025.0)
    assert len(snapshots) >= 20


@respx.mock
def test_all_stale_rows_yield_no_reference_spot_and_no_products() -> None:
    """Pitfall 1: an all-stale fetch (every row's quote far older than
    stale_after_s) must not contribute to S_ref at all -- with zero fresh
    candidates, no reference spot can be derived, so the whole underlying is
    skipped for this fetch (never priced from a stale/absent reference)."""
    fixture = _load_fixture("dax_turbosendlos_sorted_leverage_sample.json")
    rows = fixture["data"]["groups"]["products"]
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["DAX"])

    assert snapshots == []
    assert "DAX" in adapter._reference_spot_mismatches
    assert any(e.get("event") == "gettex_no_reference_spot" for e in logs)


# ===========================================================================
# fetch_products: currency / quanto inference for non-EUR underlyings
# ===========================================================================


@respx.mock
def test_quanto_hypothesis_accepted_when_unambiguous() -> None:
    # NDX is USD-denominated. Price the rows as genuinely quanto (fx=1) --
    # only the fx=1 hypothesis should verify; the fx=fx_hint hypothesis must
    # NOT also happen to verify (fx_hint is deliberately far from 1).
    rows = _long_short_batch(spot=20000.0, ratio=0.01, n_each=10, underlying_name="NASDAQ 100")
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["NDX"], fx_hint=1.08)

    assert len(snapshots) >= 20
    assert all(s.quanto is True for s in snapshots)
    assert all(s.currency == "EUR" for s in snapshots)
    assert all(s.underlying_currency == "USD" for s in snapshots)


@respx.mock
def test_non_quanto_hypothesis_accepted_when_unambiguous() -> None:
    # Price the rows as genuinely non-quanto: price includes the EURUSD fx
    # factor (fx=1.08), so only the fx=fx_hint hypothesis verifies.
    fx = 1.08
    rows = _long_short_batch(spot=20000.0, ratio=0.01, n_each=10, underlying_name="NASDAQ 100")
    # _gettex_product bakes fx=1.0 into price by default; rebuild with fx.
    rows = []
    for k in range(1, 11):
        rows.append(
            _gettex_product(
                f"DE000NQLG{k:03d}",
                direction="long",
                spot=20000.0,
                financing_level=20000.0 - 100.0 * k,
                ratio=0.01,
                fx=fx,
                underlying_name="NASDAQ 100",
            )
        )
        rows.append(
            _gettex_product(
                f"DE000NQSH{k:03d}",
                direction="short",
                spot=20000.0,
                financing_level=20000.0 + 100.0 * k,
                ratio=0.01,
                fx=fx,
                underlying_name="NASDAQ 100",
            )
        )
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["NDX"], fx_hint=fx)

    assert len(snapshots) >= 20
    assert all(s.quanto is False for s in snapshots)


@respx.mock
def test_missing_fx_hint_for_non_eur_underlying_limits_to_quanto_hypothesis() -> None:
    # True fx != 1 (non-quanto pricing), but no fx_hint supplied -- only the
    # fx=1 hypothesis can be tried, and it must not be silently accepted as
    # quanto just because it's the only one tested: at an 8% fx mismatch the
    # single fx=1 hypothesis's ratio_raw is far enough off that it doesn't
    # even snap to a canonical grid value (ratio_rejected); a smaller
    # mismatch would instead snap and fail the implied-spot check
    # (verification_failed) -- either way, never accepted.
    fx = 1.08
    rows = []
    for k in range(1, 11):
        rows.append(
            _gettex_product(
                f"DE000NHLG{k:03d}",
                direction="long",
                spot=20000.0,
                financing_level=20000.0 - 100.0 * k,
                ratio=0.01,
                fx=fx,
                underlying_name="NASDAQ 100",
            )
        )
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["NDX"])  # no fx_hint

    assert snapshots == []
    assert len(adapter.last_errors) == 10
    assert all(
        "ratio_raw" in e.error.lower() or "verification" in e.error.lower()
        for e in adapter.last_errors
    )


# ===========================================================================
# fetch_products: barrier != financing_level -> UNKNOWN product_type
# ===========================================================================


@respx.mock
def test_barrier_not_equal_financing_level_yields_unknown_product_type() -> None:
    good_rows = _long_short_batch(n_each=10)
    mismatched = _gettex_product(
        "DE000MISMAT1", direction="long", financing_level=24500.0, ko_level=24000.0
    )
    respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([*good_rows, mismatched]))
    )
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    matches = [s for s in snapshots if s.isin == "DE000MISMAT1"]
    assert len(matches) == 1
    snapshot = matches[0]
    assert snapshot.product_type == ProductType.UNKNOWN
    assert snapshot.open_end is False
    assert snapshot.maturity is None
    assert snapshot.ratio > 0  # still priced -- just not classified as open-end
    assert any(
        e.isin == "DE000MISMAT1" and "barrier" in e.error.lower() for e in adapter.last_errors
    )


# ===========================================================================
# fetch_products: pagination + max_pages
# ===========================================================================


@respx.mock
def test_pagination_across_multiple_pages_and_max_pages_limit() -> None:
    page1 = _long_short_batch(n_each=2)  # 4 rows
    page2 = [
        _gettex_product("DE000PAGE2A1", direction="long", financing_level=24000.0),
        _gettex_product("DE000PAGE2B1", direction="short", financing_level=26000.0),
        _gettex_product("DE000PAGE2C1", direction="long", financing_level=23000.0),
        _gettex_product("DE000PAGE2D1", direction="short", financing_level=27000.0),
    ]  # also 4 rows -- a full page, so pagination would continue past max_pages
    respx.get(PRODUCTS_URL).mock(
        side_effect=[
            httpx.Response(
                200, json=_gettex_page_response(page1, filtered_count=20, rows_per_page=4)
            ),
            httpx.Response(
                200, json=_gettex_page_response(page2, filtered_count=20, rows_per_page=4)
            ),
        ]
    )
    adapter = _gettex_adapter(
        max_pages=2, rows_per_page=4, clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC))
    )
    snapshots = adapter.fetch_products(["DAX"])

    isins = {s.isin for s in snapshots}
    assert isins == {
        "DE000LONG001",
        "DE000SHRT001",
        "DE000LONG002",
        "DE000SHRT002",
        "DE000PAGE2A1",
        "DE000PAGE2B1",
        "DE000PAGE2C1",
        "DE000PAGE2D1",
    }
    # filteredCount=20, only 2 pages * 4 rows = 8 fetched -> partial_universe.
    assert adapter._partial_universe["DAX"] == (8, 20)


@respx.mock
def test_pagination_stops_early_on_short_page() -> None:
    # Low-leverage (large-moneyness) rows -- unlike `_long_short_batch`'s
    # default 100-point-per-k strikes (leverage ~250 at k=1, excluded from
    # S_ref by the numerical-conditioning cap), so the pair alone still
    # yields a usable S_ref.
    only_page = [
        _gettex_product("DE000ONLYPG1", direction="long", financing_level=22000.0),
        _gettex_product("DE000ONLYPG2", direction="short", financing_level=28000.0),
    ]  # 2 rows, less than rows_per_page
    route = respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response(only_page, rows_per_page=50))
    )
    adapter = _gettex_adapter(
        max_pages=20, rows_per_page=50, clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC))
    )
    snapshots = adapter.fetch_products(["DAX"])

    assert route.call_count == 1
    assert len(snapshots) == 2
    assert "DAX" not in adapter._partial_universe


@respx.mock
def test_unresolved_underlying_is_skipped_cleanly() -> None:
    route = respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([]))
    )
    adapter = _gettex_adapter()
    # XAU (gold) has no known gettex `underlying` numeric id in this project.
    snapshots = adapter.fetch_products(["XAU"])

    assert snapshots == []
    assert route.call_count == 0


# ===========================================================================
# fetch_products: schema drift
# ===========================================================================


@respx.mock
def test_schema_drift_missing_field_goes_to_last_errors_not_raised() -> None:
    good = _gettex_product("DE000GOODRW1", direction="long", financing_level=24500.0)
    bad = _gettex_product("DE000BADROW1", direction="long", financing_level=24000.0)
    del bad["leverage"]  # required field missing
    extra_good = _long_short_batch(n_each=9)  # pad for a robust S_ref
    respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([good, bad, *extra_good]))
    )
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert "DE000BADROW1" not in {s.isin for s in snapshots}
    assert "DE000GOODRW1" in {s.isin for s in snapshots}
    bad_errors = [e for e in adapter.last_errors if e.isin == "DE000BADROW1"]
    assert len(bad_errors) == 1
    assert isinstance(bad_errors[0], GettexRowError)


@respx.mock
def test_healthcheck_fail_on_schema_drift() -> None:
    malformed_product = {"isin": {"value": "DE000ONLYISN"}}  # missing everything else
    respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([malformed_product]))
    )
    adapter = _gettex_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.FAIL
    assert result.ok is False


@respx.mock
def test_healthcheck_warn_on_empty_products() -> None:
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response([])))
    adapter = _gettex_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.WARN
    assert result.ok is True


@respx.mock
def test_healthcheck_pass_on_healthy_probe() -> None:
    rows = _long_short_batch(n_each=2)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.PASS
    assert result.ok is True


# ===========================================================================
# fetch_products: underlying label / requested id mismatch is never trusted
# ===========================================================================


@respx.mock
def test_underlying_label_mismatch_row_dropped() -> None:
    good_rows = _long_short_batch(n_each=10)
    wrong_label = _gettex_product(
        "DE000WRONGLB",
        direction="long",
        financing_level=24500.0,
        underlying_name="S&P 500",  # requested DAX, row claims S&P 500
    )
    respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([*good_rows, wrong_label]))
    )
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert "DE000WRONGLB" not in {s.isin for s in snapshots}
    assert any(e.isin == "DE000WRONGLB" for e in adapter.last_errors)


# ===========================================================================
# fetch_products: real-fixture smoke coverage
# ===========================================================================


@respx.mock
def test_real_fixture_dax_sample_parses_without_raising() -> None:
    fixture = _load_fixture("dax_turbosendlos_sample.json")
    rows = fixture["data"]["groups"]["products"]
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 16, 0, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["DAX"])

    # Real, noisy data -- no strict count assertion, just structural sanity.
    for s in snapshots:
        assert s.ratio in _RATIO_GRID or any(abs(s.ratio - g) / g < 1e-6 for g in _RATIO_GRID)
        assert s.currency == "EUR"
        if s.bid is not None and s.ask is not None:
            assert s.bid <= s.ask

    summary_events = [e for e in logs if e.get("event") == "gettex_fetch_summary"]
    assert len(summary_events) == 1
    # This particular 15-row research capture happens to be entirely
    # high-leverage (>200) rows -- with the numerical-conditioning cap on
    # which rows are S_ref-eligible (see _DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT),
    # that can legitimately mean no reference spot is derivable from this
    # fixture alone (`total` only counts rows reached once S_ref exists).
    # Both outcomes are acceptable structural sanity for a real, uncurated
    # sample; a "no reference spot" skip must at least be the honest,
    # explicit reason (never silently zero snapshots for an unstated cause).
    if not snapshots:
        assert "DAX" in adapter._reference_spot_mismatches
    else:
        assert summary_events[0]["total"] >= len(rows)


@respx.mock
def test_real_fixture_multi_underlying_smoke() -> None:
    dax = _load_fixture("dax_turbosendlos_sample.json")["data"]["groups"]["products"]
    ndx = _load_fixture("nasdaq100_sample.json")["data"]["groups"]["products"]
    spx = _load_fixture("sp500_sample.json")["data"]["groups"]["products"]
    estx50 = _load_fixture("eurostoxx50_sample.json")["data"]["groups"]["products"]

    def _route(underlying_id: int, rows: list[dict[str, Any]]) -> None:
        respx.get(PRODUCTS_URL, params={"underlying": underlying_id}).mock(
            return_value=httpx.Response(200, json=_gettex_page_response(rows))
        )

    _route(_GETTEX_UNDERLYING_IDS["DAX"], dax)
    _route(_GETTEX_UNDERLYING_IDS["NDX"], ndx)
    _route(_GETTEX_UNDERLYING_IDS["SPX"], spx)
    _route(_GETTEX_UNDERLYING_IDS["ESTX50"], estx50)

    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 16, 0, tzinfo=UTC)))
    # Should not raise for any of the four canonical underlyings this
    # research session mapped.
    snapshots = adapter.fetch_products(["DAX", "NDX", "SPX", "ESTX50"])
    assert isinstance(snapshots, list)


# ===========================================================================
# Befund 2 fix 1/3: leverage-scaled ratio-snap + verification tolerance
# ===========================================================================


def test_leverage_scaled_ratio_snap_tolerance_unscaled_at_reference_leverage() -> None:
    assert _leverage_scaled_ratio_snap_tolerance(20.0, 0.03, 20.0, 0.50) == pytest.approx(0.03)
    # below the reference leverage, never *shrinks* below the base value.
    assert _leverage_scaled_ratio_snap_tolerance(5.0, 0.03, 20.0, 0.50) == pytest.approx(0.03)


def test_leverage_scaled_ratio_snap_tolerance_widens_and_caps() -> None:
    assert _leverage_scaled_ratio_snap_tolerance(100.0, 0.03, 20.0, 0.50) == pytest.approx(0.15)
    # 0.03 * (1000/20) = 1.5 -- capped at 0.50.
    assert _leverage_scaled_ratio_snap_tolerance(1000.0, 0.03, 20.0, 0.50) == pytest.approx(0.50)


def test_leverage_scaled_verification_tolerance_tightens_inversely() -> None:
    assert _leverage_scaled_verification_tolerance(20.0, 0.003, 20.0) == pytest.approx(0.003)
    assert _leverage_scaled_verification_tolerance(200.0, 0.003, 20.0) == pytest.approx(0.0003)
    # below the reference leverage, never *widens* beyond the base value.
    assert _leverage_scaled_verification_tolerance(5.0, 0.003, 20.0) == pytest.approx(0.003)


@respx.mock
def test_high_leverage_dax_row_recovered_by_leverage_scaled_snap_tolerance() -> None:
    """A single-hypothesis (EUR/DAX) row whose ratio_raw deviates ~7.4% from
    the true grid ratio -- rejected outright at the old fixed 3% tolerance --
    is recovered once its leverage (here ~83, well above the reference 20)
    widens the effective snap tolerance, and still passes the *unwidened*
    0.3% implied-spot verification because the row's price is genuinely
    consistent with the true ratio (the 7.4% gap is deliberately manufactured
    by pricing the row against a *different*, non-canonical ratio and letting
    the nearest-grid-point search find the true one)."""
    spot = 25000.0
    financing_level = 24700.0  # moneyness=300 -> leverage=25000/300=83.3
    true_ratio = 0.01
    off_grid_ratio = 0.0108  # ~8% off the nearest grid value (0.01)
    price = (spot - financing_level) * off_grid_ratio  # priced as if ratio were 0.0108
    row = {
        "wkn": {"value": "HILEV1"},
        "isin": {"value": "DE000HILEV01"},
        "underlying": {"value": "DAX (Performance)"},
        "underlyings.price": {
            "valueTuple": {"value": spot, "size": 0.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
        "issuer": {"value": "BNP Paribas"},
        "structure.direction": {"value": "long"},
        "financingLevelRefCurAbsolute": {"value": financing_level},
        "koLevelRefCurAbsolute": {"value": financing_level},
        "leverage": {"value": spot / (spot - financing_level)},
        "bid": {
            "valueTuple": {"value": price, "size": 1000.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
        "ask": {
            "valueTuple": {"value": price, "size": 1000.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
    }
    # Pad with a robust, direction-balanced, moderate-leverage batch so S_ref
    # is cleanly established at exactly `spot` regardless of the row above.
    good_rows = _long_short_batch(spot=spot, ratio=true_ratio, n_each=12)
    respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([row, *good_rows]))
    )
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    matches = [s for s in snapshots if s.isin == "DE000HILEV01"]
    assert len(matches) == 1
    # The nearest grid ratio to 0.0108 is 0.01 -- recovered, not the
    # off-grid 0.0108 itself (which is never a valid canonical ratio).
    assert matches[0].ratio == pytest.approx(0.01)


@respx.mock
def test_low_leverage_row_still_rejected_outside_widened_tolerance() -> None:
    """At the reference leverage (no widening), a ~7.4%-off ratio must still
    be rejected exactly as before this fix -- the widening is leverage-gated,
    not a blanket loosening."""
    spot = 25000.0
    financing_level = 23750.0  # moneyness=1250 -> leverage=20.0 (reference, unscaled tolerance)
    off_grid_ratio = 0.0108
    price = (spot - financing_level) * off_grid_ratio
    row = {
        "wkn": {"value": "LOLEV1"},
        "isin": {"value": "DE000LOLEV01"},
        "underlying": {"value": "DAX (Performance)"},
        "underlyings.price": {
            "valueTuple": {"value": spot, "size": 0.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
        "issuer": {"value": "BNP Paribas"},
        "structure.direction": {"value": "long"},
        "financingLevelRefCurAbsolute": {"value": financing_level},
        "koLevelRefCurAbsolute": {"value": financing_level},
        "leverage": {"value": spot / (spot - financing_level)},
        "bid": {
            "valueTuple": {"value": price, "size": 1000.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
        "ask": {
            "valueTuple": {"value": price, "size": 1000.0, "timestamp": _EPOCH_2026_09_11T12_00}
        },
    }
    good_rows = _long_short_batch(spot=spot, ratio=0.01, n_each=12)
    respx.get(PRODUCTS_URL).mock(
        return_value=httpx.Response(200, json=_gettex_page_response([row, *good_rows]))
    )
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert "DE000LOLEV01" not in {s.isin for s in snapshots}
    assert any(e.isin == "DE000LOLEV01" for e in adapter.last_errors)


# ===========================================================================
# Befund 2 fix 2: per-issuer fx/quanto majority learning
# ===========================================================================


def test_fx_candidates_from_ratio_raw_at_1_filters_to_plausible_band() -> None:
    # true ratio=0.01, true fx=1.14 -> ratio_raw_at_1 = price/moneyness = 0.01/1.14
    ratio_raw_at_1 = 0.01 / 1.14
    candidates = _fx_candidates_from_ratio_raw_at_1(ratio_raw_at_1, 0.5, 2.0)
    # 0.01/ratio_raw_at_1 == 1.14 is one of the 14 grid-inversion candidates.
    assert any(abs(c - 1.14) < 1e-9 for c in candidates)
    assert all(0.5 <= c <= 2.0 for c in candidates)


def test_fx_candidates_non_positive_input_returns_empty() -> None:
    assert _fx_candidates_from_ratio_raw_at_1(0.0, 0.5, 2.0) == []
    assert _fx_candidates_from_ratio_raw_at_1(-1.0, 0.5, 2.0) == []


def test_robust_fx_median_filters_outlier() -> None:
    candidates = [1.14] * 9 + [1.9]
    assert _robust_fx_median(candidates, mad_k=5.0) == pytest.approx(1.14)


def test_robust_fx_median_empty_returns_none() -> None:
    assert _robust_fx_median([], mad_k=5.0) is None


# Distinct ratios cycled across a batch's rows (Bezugsverhaeltnis varies by
# strike/product in real gettex data): using a single fixed ratio for every
# row is an unrealistic monoculture that creates an artificial grid-inversion
# tie (the "wrong" grid neighbor two steps away lands at exactly the same fx
# for every row, competing evenly with the true cluster) -- never observed on
# real, ratio-diverse data (this project's own live NDX validation recovered
# 85-87% with this exact learning method), so the synthetic batch below
# mirrors that diversity instead of the pathological single-ratio case.
_FX_LEARNING_RATIOS: tuple[float, ...] = (0.01, 0.001, 0.1)


def _fx_learning_batch(
    *,
    fx: float,
    spot: float = 25000.0,
    n_each: int = 15,
    underlying_name: str = "NASDAQ 100",
    issuer: str = "BNP Paribas",
    tag: str = "FXL",
) -> list[dict[str, Any]]:
    """``tag`` must be exactly 3 chars -- ISINs are ``DE000`` + tag(3) +
    direction(1) + k(3 digits) == 12 alphanumeric chars, the schema's
    required ISIN length."""
    assert len(tag) == 3
    rows: list[dict[str, Any]] = []
    for k in range(1, n_each + 1):
        ratio = _FX_LEARNING_RATIOS[k % len(_FX_LEARNING_RATIOS)]
        rows.append(
            _gettex_product(
                f"DE000{tag}L{k:03d}",
                direction="long",
                spot=spot,
                financing_level=spot - 200.0 * k,
                ratio=ratio,
                fx=fx,
                issuer=issuer,
                underlying_name=underlying_name,
            )
        )
        rows.append(
            _gettex_product(
                f"DE000{tag}S{k:03d}",
                direction="short",
                spot=spot,
                financing_level=spot + 200.0 * k,
                ratio=ratio,
                fx=fx,
                issuer=issuer,
                underlying_name=underlying_name,
            )
        )
    return rows


@respx.mock
def test_fx_learned_from_majority_recovers_ambiguous_rows_without_fx_hint() -> None:
    """The generic ProductSourceAdapter contract never passes fx_hint (Befund
    2) -- with none supplied, a genuinely non-quanto NDX issuer's rows must
    still be recovered via GettexAdapter._learn_fx_by_issuer's per-issuer
    majority-learned fx, not left stuck on the quanto-only hypothesis."""
    true_fx = 1.14
    rows = _fx_learning_batch(fx=true_fx, n_each=15)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["NDX"])  # no fx_hint

    assert len(snapshots) >= 20
    assert all(s.quanto is False for s in snapshots)
    learned_key = "NDX:BNP Paribas"
    assert learned_key in adapter.last_learned_fx
    assert adapter.last_learned_fx[learned_key] == pytest.approx(true_fx, rel=0.05)

    learned_events = [e for e in logs if e.get("event") == "gettex_fx_learned"]
    assert len(learned_events) == 1
    assert learned_events[0]["issuer"] == "BNP Paribas"
    assert learned_events[0]["underlying_id"] == "NDX"


@respx.mock
def test_fx_learning_low_support_group_falls_back_to_conservative_rejection() -> None:
    """A group whose grid-inversion candidates don't converge on a fx value
    that actually explains a majority of its own rows must be rejected by
    the majority (`fx_learning_min_verified_fraction`) gate -- those rows
    stay exactly as conservatively rejected as before this fix, never
    mispriced from a low-confidence learned fx."""
    # A small, internally-inconsistent batch: 4 sub-groups, each genuinely
    # priced under a DIFFERENT, widely-separated true fx (so no single
    # candidate cluster -- including any accidental grid-neighbor tie -- can
    # ever cover more than 1/4 of the group), each with a diverse-ratio
    # 3-row sub-batch (see `_FX_LEARNING_RATIOS`). No single learned fx can
    # explain a majority of the combined 24 rows.
    rows = []
    for i, fx in enumerate([0.6, 0.9, 1.3, 1.8]):
        rows.extend(_fx_learning_batch(fx=fx, n_each=3, tag=f"M{i}X", issuer="UniCredit"))
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        adapter.fetch_products(["NDX"])

    assert "NDX:UniCredit" not in adapter.last_learned_fx
    rejected_events = [
        e for e in logs if e.get("event") == "gettex_fx_learning_rejected_low_support"
    ]
    insufficient_events = [
        e for e in logs if e.get("event") == "gettex_fx_learning_insufficient_candidates"
    ]
    # Either the proposed fx failed the majority-verified check, or there
    # weren't even enough candidates to propose one -- both are the same
    # conservative "did not learn" outcome from the caller's perspective.
    assert rejected_events or insufficient_events


@respx.mock
def test_explicit_fx_hint_takes_precedence_over_learned_fx() -> None:
    """When the caller does supply fx_hint, it is used as-is -- the internal
    per-issuer learning step must not even run (no last_learned_fx entry)."""
    true_fx = 1.14
    rows = _fx_learning_batch(fx=true_fx, n_each=15)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    snapshots = adapter.fetch_products(["NDX"], fx_hint=true_fx)

    assert len(snapshots) >= 20
    assert adapter.last_learned_fx == {}


# ===========================================================================
# Befund 2 fix 3: quanto-ambiguity safety net (leverage-tightened verify)
# ===========================================================================


@respx.mock
def test_systematic_fx_offset_never_accepted_across_leverage_range() -> None:
    """Regression guard for the exact false-accept this fix closes: a
    uniform, systematic fx mismatch (here 8%, mimicking a genuinely
    non-quanto book mistaken for quanto) must be rejected at every leverage
    from a moderate level up through the S_ref-eligible cap (200) when only
    the quanto (fx=1) hypothesis is tested (no fx_hint, and too few/too
    uniform rows for fx-learning to find anything to converge on) -- not
    silently accepted just because a high-leverage row's widened snap
    tolerance let it reach the verification step at all."""
    spot = 20000.0
    true_fx = 1.08
    rows = []
    for k in range(1, 11):
        rows.append(
            _gettex_product(
                f"DE000SYSOFF{k:03d}",
                direction="long",
                spot=spot,
                financing_level=spot - 100.0 * k,
                ratio=0.01,
                fx=true_fx,
                underlying_name="NASDAQ 100",
            )
        )
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["NDX"])  # no fx_hint

    assert snapshots == []


# ===========================================================================
# Coordinator finding (2026-09-13): BNP outage must not zero out gettex --
# daily_close_reference fallback when reference_spot is unavailable
# ===========================================================================


@respx.mock
def test_daily_close_reference_used_when_reference_spot_missing() -> None:
    """Simulates a BNP outage: the pipeline's only external reference_spot
    source failed, so `reference_spot` is None -- but a yfinance-style same-
    day daily close is available and, within the wider daily-close tolerance,
    confirms the internally-derived S_ref. Products must still be returned
    (gettex's S_ref never actually depended on BNP -- see module docstring)."""
    true_spot = 25000.0
    rows = _long_short_batch(spot=true_spot, ratio=0.01, n_each=12)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    # 0.4% off the true spot -- comfortably within the default 2% daily-close
    # tolerance, well outside the tight 0.5% live-quote tolerance (proving
    # the wider, not the tighter, path was actually exercised).
    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(
            ["DAX"], reference_spot=None, daily_close_reference=true_spot * 1.004
        )

    assert len(snapshots) >= 20
    assert "DAX" not in adapter._reference_spot_mismatches
    confirmed_events = [
        e for e in logs if e.get("event") == "gettex_daily_close_reference_confirmed"
    ]
    assert len(confirmed_events) == 1


@respx.mock
def test_daily_close_reference_only_used_when_reference_spot_absent() -> None:
    """A live reference_spot, when supplied, still takes priority -- the
    daily-close fallback is not consulted (and could not save a mismatch at
    the tight tolerance even if it were within its own wider band)."""
    true_spot = 25000.0
    rows = _long_short_batch(spot=true_spot, ratio=0.01, n_each=12)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    # reference_spot is 1% off (outside the tight 0.5% tolerance) even though
    # daily_close_reference is exact -- the live reference_spot's tight check
    # must still be the one applied and must still reject.
    snapshots = adapter.fetch_products(
        ["DAX"],
        reference_spot=true_spot * 1.01,
        daily_close_reference=true_spot,
    )

    assert snapshots == []
    assert "DAX" in adapter._reference_spot_mismatches


@respx.mock
def test_daily_close_reference_mismatch_still_rejects() -> None:
    """The wider daily-close tolerance is not a rubber stamp -- a daily close
    that's genuinely far from S_ref (a stale/wrong/unit-mismatched value)
    still drops the underlying's products, exactly as the tight live-quote
    check would."""
    true_spot = 25000.0
    rows = _long_short_batch(spot=true_spot, ratio=0.01, n_each=12)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(
            ["DAX"], reference_spot=None, daily_close_reference=true_spot * 1.10
        )

    assert snapshots == []
    assert "DAX" in adapter._reference_spot_mismatches
    mismatch_events = [e for e in logs if e.get("event") == "gettex_daily_close_reference_mismatch"]
    assert len(mismatch_events) == 1


@respx.mock
def test_no_external_reference_at_all_still_derives_and_verifies() -> None:
    """Neither reference_spot nor daily_close_reference supplied -- S_ref
    (always internally derived, module docstring step 1) stands on its own,
    exactly as it did before this fix."""
    rows = _long_short_batch(spot=25000.0, ratio=0.01, n_each=12)
    respx.get(PRODUCTS_URL).mock(return_value=httpx.Response(200, json=_gettex_page_response(rows)))
    adapter = _gettex_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 12, 5, tzinfo=UTC)))

    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) >= 20
    assert "DAX" not in adapter._reference_spot_mismatches
