from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    FieldReliability,
    ManualPosition,
    ProductSnapshot,
    ProductType,
    UnderlyingBar,
)


def test_valid_product_snapshot_roundtrips(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    snapshot = make_product_snapshot()
    dumped = snapshot.model_dump(mode="json")
    restored = ProductSnapshot.model_validate(dumped)
    assert restored == snapshot


def test_product_snapshot_field_reliability_defaults_to_unverified(
    make_product_snapshot,  # type: ignore[no-untyped-def]
) -> None:
    """Phase B ("Produktstammdaten haerten"): a source that never sets these
    fields (e.g. adapters/csv_import.py) honestly reports UNVERIFIED rather
    than silently inheriting a stronger level it never earned (CLAUDE.md
    rule 29)."""
    snapshot = make_product_snapshot()
    assert snapshot.ratio_reliability == FieldReliability.UNVERIFIED
    assert snapshot.barrier_reliability == FieldReliability.UNVERIFIED
    assert snapshot.financing_level_reliability == FieldReliability.UNVERIFIED


@pytest.mark.parametrize(
    "level",
    [
        FieldReliability.SOURCE_REPORTED,
        FieldReliability.CROSS_SOURCE_VERIFIED,
        FieldReliability.DERIVED_VERIFIED,
        FieldReliability.UNVERIFIED,
    ],
)
def test_product_snapshot_field_reliability_every_level_assignable_and_roundtrips(
    make_product_snapshot,  # type: ignore[no-untyped-def]
    level: FieldReliability,
) -> None:
    snapshot = make_product_snapshot(
        ratio_reliability=level,
        barrier_reliability=level,
        financing_level_reliability=level,
    )
    assert snapshot.ratio_reliability == level
    assert snapshot.barrier_reliability == level
    assert snapshot.financing_level_reliability == level
    dumped = snapshot.model_dump(mode="json")
    assert dumped["ratio_reliability"] == level.value
    restored = ProductSnapshot.model_validate(dumped)
    assert restored == snapshot


def test_isin_must_be_12_alnum_chars(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError, match="ISIN"):
        make_product_snapshot(isin="TOO_SHORT")


def test_isin_lowercase_is_normalized_to_uppercase(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    snapshot = make_product_snapshot(isin="de000abc1234")
    assert snapshot.isin == "DE000ABC1234"


def test_ratio_must_be_positive(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError, match="> 0"):
        make_product_snapshot(ratio=0.0)
    with pytest.raises(ValidationError):
        make_product_snapshot(ratio=-0.01)


def test_quality_score_must_be_within_unit_interval(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        make_product_snapshot(quality_score=1.5)
    with pytest.raises(ValidationError):
        make_product_snapshot(quality_score=-0.1)


def test_naive_datetime_rejected(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_product_snapshot(observation_time=datetime(2026, 9, 10, 12, 0))  # naive


def test_tz_aware_non_utc_datetime_is_accepted(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    from datetime import timedelta, timezone

    berlin = timezone(timedelta(hours=2))
    snapshot = make_product_snapshot(observation_time=datetime(2026, 9, 10, 17, 30, tzinfo=berlin))
    assert snapshot.observation_time.utcoffset() == timedelta(hours=2)


def test_underlying_bar_valid() -> None:
    bar = UnderlyingBar(
        underlying_id="DAX",
        ts=datetime(2026, 9, 9, tzinfo=UTC),
        open=18000.0,
        high=18100.0,
        low=17950.0,
        close=18050.0,
        volume=1_000_000.0,
        observation_time=datetime(2026, 9, 9, tzinfo=UTC),
        available_at=datetime(2026, 9, 9, 22, 0, tzinfo=UTC),
        retrieved_at=datetime(2026, 9, 10, 6, 0, tzinfo=UTC),
        source="yfinance",
        parser_version="1",
        quality_score=0.9,
    )
    assert bar.interval == "1d"
    assert bar.is_stale is False


def test_candidate_evaluation_costs_json_shape() -> None:
    costs = CostDecomposition(
        ask=4.86,
        bid=4.80,
        mid=4.83,
        intrinsic=4.70,
        trading_spread_component=0.03,
        fair_gap_premium=0.02,
        financing_drag=0.01,
        issuer_margin=0.10,
        spread_pct=0.0123,
        gap_premium_pct=0.0041,
        financing_drag_pct=0.0021,
        issuer_margin_pct=0.0206,
    )
    candidate = CandidateEvaluation(
        run_id="20260910T120000Z-abc123456789",
        candidate_id="cand-1",
        isin="DE000ABC1234",
        wkn="ABC123",
        issuer="TestBank",
        underlying_id="DAX",
        direction=Direction.LONG,
        category=Category.WATCH,
        reasons=["cost_rank ok"],
        leverage=5.2,
        leverage_bucket="5-10",
        distance_to_barrier_pct=0.03,
        distance_to_barrier_sigma=1.5,
        costs=costs,
        realized_financing_spread=0.02,
        financing_cost_horizon_pct={"3d": 0.001, "7d": 0.003},
        cross_issuer_residual_zscore=0.5,
        issuer_markup_score=0.1,
        quote_dislocation_score=0.05,
        wrapper_edge=None,
        liquidity_factor=0.8,
        integrity_passed=True,
        lcb_ev=None,
        cost_rank_score=0.12,
    )
    assert candidate.lcb_ev is None  # ACTIONABLE gate depends on this staying None
    assert candidate.category != Category.ACTIONABLE


def test_manual_position_requires_positive_qty_and_price() -> None:
    with pytest.raises(ValidationError):
        ManualPosition(
            wkn="ABC123",
            qty=0,
            entry_price=4.86,
            entry_date=date(2026, 9, 10),
            created_at=datetime(2026, 9, 10, tzinfo=UTC),
            updated_at=datetime(2026, 9, 10, tzinfo=UTC),
        )


_ISIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(isin_chars=st.text(alphabet=_ISIN_ALPHABET, min_size=12, max_size=12))
def test_hypothesis_any_12_char_alnum_isin_is_accepted(  # type: ignore[no-untyped-def]
    isin_chars: str, make_product_snapshot
) -> None:
    snapshot = make_product_snapshot(isin=isin_chars)
    assert snapshot.isin == isin_chars.upper()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(bad_len=st.integers(min_value=1, max_value=30).filter(lambda n: n != 12))
def test_hypothesis_wrong_length_isin_is_rejected(bad_len: int, make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        make_product_snapshot(isin="A" * bad_len)


def test_product_type_enum_values() -> None:
    assert {pt.value for pt in ProductType} == {
        "turbo_open_end",
        "turbo_classic",
        "mini_future",
        "unknown",
    }
