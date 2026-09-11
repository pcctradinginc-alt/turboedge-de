from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.ranking.liquidity import liquidity_factor, quote_size_coverage, spread_quality


def test_liquidity_factor_geometric_mean() -> None:
    value = liquidity_factor(1.0, 1.0, 1.0, 1.0)
    assert value == pytest.approx(1.0)


def test_liquidity_factor_known_value() -> None:
    value = liquidity_factor(0.5, 0.5, 0.5, 0.5)
    assert value == pytest.approx(0.5)


def test_liquidity_factor_zero_input_floors_at_eps_not_zero() -> None:
    value = liquidity_factor(0.0, 1.0, 1.0, 1.0)
    # geometric_mean(eps, 1, 1, 1) == eps ** 0.25, a small but strictly
    # positive number -- not an exact, hard-clamped zero.
    assert value == pytest.approx(1e-6**0.25)
    assert 0.0 < value < 0.1


def test_liquidity_factor_clips_above_one() -> None:
    assert liquidity_factor(2.0, 2.0, 2.0, 2.0) == pytest.approx(1.0)


def test_quote_size_coverage_full() -> None:
    assert quote_size_coverage(ask_size=1000.0, required_notional=1000.0, ask=1.0) == pytest.approx(
        1.0
    )


def test_quote_size_coverage_partial() -> None:
    assert quote_size_coverage(ask_size=500.0, required_notional=1000.0, ask=1.0) == pytest.approx(
        0.5
    )


def test_quote_size_coverage_clips_to_one() -> None:
    assert quote_size_coverage(ask_size=5000.0, required_notional=1000.0, ask=1.0) == pytest.approx(
        1.0
    )


def test_quote_size_coverage_missing_size_is_zero() -> None:
    assert quote_size_coverage(ask_size=None, required_notional=1000.0, ask=1.0) == 0.0


def test_quote_size_coverage_rejects_nonpositive_required_notional() -> None:
    with pytest.raises(ValueError):
        quote_size_coverage(ask_size=100.0, required_notional=0.0, ask=1.0)


def test_spread_quality_zero_spread_is_perfect() -> None:
    assert spread_quality(0.0, 0.03) == pytest.approx(1.0)


def test_spread_quality_at_max_is_zero() -> None:
    assert spread_quality(0.03, 0.03) == pytest.approx(0.0)


def test_spread_quality_beyond_max_clips_to_zero() -> None:
    assert spread_quality(0.10, 0.03) == 0.0


def test_spread_quality_rejects_nonpositive_max() -> None:
    with pytest.raises(ValueError):
        spread_quality(0.01, 0.0)


@given(
    a=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
    b=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
    c=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
    d=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
)
def test_liquidity_factor_always_in_unit_interval(a: float, b: float, c: float, d: float) -> None:
    value = liquidity_factor(a, b, c, d)
    assert 0.0 <= value <= 1.0
