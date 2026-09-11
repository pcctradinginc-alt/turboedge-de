from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.features.product import (
    BarrierDistance,
    distance_to_barrier,
    ewma_volatility,
    freshness_score,
    leverage_bucket,
    quote_age_seconds,
    spread_pct,
)
from turboedge.storage.schemas import Direction


def test_spread_pct_relative_to_mid() -> None:
    # bid=4.80, ask=4.86 -> mid=4.83, spread=0.06 -> 0.06/4.83
    assert spread_pct(4.80, 4.86) == pytest.approx(0.06 / 4.83)


def test_spread_pct_rejects_bid_above_ask() -> None:
    with pytest.raises(ValueError):
        spread_pct(5.0, 4.9)


def test_spread_pct_rejects_nonpositive_mid() -> None:
    with pytest.raises(ValueError):
        spread_pct(0.0, 0.0)


def test_quote_age_seconds() -> None:
    quote_ts = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    now = quote_ts + timedelta(seconds=45)
    assert quote_age_seconds(quote_ts, now) == pytest.approx(45.0)


def test_freshness_score_none_age_is_zero() -> None:
    assert freshness_score(None) == 0.0


def test_freshness_score_zero_age_is_one() -> None:
    assert freshness_score(0.0, half_life_s=60.0) == pytest.approx(1.0)


def test_freshness_score_half_life() -> None:
    assert freshness_score(60.0, half_life_s=60.0) == pytest.approx(0.5, rel=1e-6)


def test_freshness_score_negative_age_clamped() -> None:
    assert freshness_score(-10.0, half_life_s=60.0) == pytest.approx(1.0)


def test_freshness_score_rejects_nonpositive_half_life() -> None:
    with pytest.raises(ValueError):
        freshness_score(10.0, half_life_s=0.0)


@pytest.mark.parametrize(
    ("lev", "bucket"),
    [
        (1.5, "<2"),
        (2.0, "2-3"),
        (2.9, "2-3"),
        (3.0, "3-4"),
        (4.0, "4-5"),
        (5.0, "5-6"),
        (6.0, "6-8"),
        (7.9, "6-8"),
        (8.0, "8-10"),
        (9.9, "8-10"),
        (10.0, "10-15"),
        (14.9, "10-15"),
        (15.0, ">15"),
        (25.0, ">15"),
    ],
)
def test_leverage_bucket_edges(lev: float, bucket: str) -> None:
    assert leverage_bucket(lev) == bucket


def test_distance_to_barrier_long() -> None:
    result = distance_to_barrier(
        spot=24000.0, barrier=22000.0, direction=Direction.LONG, daily_vol=0.01, horizon_days=5.0
    )
    assert isinstance(result, BarrierDistance)
    assert result.abs == pytest.approx(2000.0)
    assert result.pct == pytest.approx(2000.0 / 24000.0)
    assert result.sigma == pytest.approx(result.pct / (0.01 * math.sqrt(5.0)))


def test_distance_to_barrier_short() -> None:
    result = distance_to_barrier(
        spot=20000.0, barrier=22000.0, direction=Direction.SHORT, daily_vol=0.01, horizon_days=5.0
    )
    assert result.abs == pytest.approx(2000.0)
    assert result.pct > 0.0


def test_distance_to_barrier_rejects_nonpositive_inputs() -> None:
    with pytest.raises(ValueError):
        distance_to_barrier(0.0, 22000.0, Direction.LONG, 0.01, 5.0)
    with pytest.raises(ValueError):
        distance_to_barrier(24000.0, 22000.0, Direction.LONG, 0.0, 5.0)
    with pytest.raises(ValueError):
        distance_to_barrier(24000.0, 22000.0, Direction.LONG, 0.01, 0.0)


def test_ewma_volatility_constant_returns() -> None:
    returns = np.full(10, 0.01)
    vol = ewma_volatility(returns, lam=0.94)
    assert vol.shape == returns.shape
    # constant return series converges its EWMA variance to the return^2 itself.
    assert vol[-1] == pytest.approx(0.01, rel=1e-6)


def test_ewma_volatility_no_look_ahead() -> None:
    returns = np.array([0.01, -0.02, 0.015, 0.03, -0.01])
    full = ewma_volatility(returns, lam=0.9)
    prefix = ewma_volatility(returns[:3], lam=0.9)
    np.testing.assert_allclose(full[:3], prefix)


def test_ewma_volatility_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ewma_volatility(np.array([]))


def test_ewma_volatility_rejects_bad_lambda() -> None:
    with pytest.raises(ValueError):
        ewma_volatility(np.array([0.01, 0.02]), lam=1.0)


@given(lev=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
def test_leverage_bucket_is_total(lev: float) -> None:
    # Property: leverage_bucket always returns one of the known labels for
    # any non-negative leverage (totality of the mapping).
    bucket = leverage_bucket(lev)
    assert bucket in {"<2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-10", "10-15", ">15"}
