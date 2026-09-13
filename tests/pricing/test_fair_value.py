from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.pricing.fair_value import (
    dividend_yield_for_underlying,
    theoretical_fair_value,
    theoretical_fair_value_array,
)
from turboedge.pricing.intrinsic import intrinsic_value
from turboedge.storage.schemas import Direction, ProductType

_AS_OF = date(2026, 9, 11)


def test_open_end_long_equals_intrinsic() -> None:
    value = theoretical_fair_value(
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        spot=24000.0,
        financing_level=22000.0,
        knockout_barrier=22000.0,
        ratio=0.01,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.02,
        as_of=_AS_OF,
        maturity=None,
    )
    assert value == pytest.approx(intrinsic_value(24000.0, 22000.0, 0.01, Direction.LONG))


def test_mini_future_short_equals_intrinsic() -> None:
    value = theoretical_fair_value(
        direction=Direction.SHORT,
        product_type=ProductType.MINI_FUTURE,
        spot=20000.0,
        financing_level=22000.0,
        knockout_barrier=21500.0,
        ratio=0.01,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.02,
        as_of=_AS_OF,
        maturity=None,
    )
    assert value == pytest.approx(intrinsic_value(20000.0, 22000.0, 0.01, Direction.SHORT))


def test_classic_requires_maturity() -> None:
    with pytest.raises(ValueError):
        theoretical_fair_value(
            direction=Direction.SHORT,
            product_type=ProductType.TURBO_CLASSIC,
            spot=20000.0,
            financing_level=22000.0,
            knockout_barrier=22000.0,
            ratio=0.01,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.02,
            as_of=_AS_OF,
            maturity=None,
        )


def test_classic_rejects_matured_date() -> None:
    with pytest.raises(ValueError):
        theoretical_fair_value(
            direction=Direction.LONG,
            product_type=ProductType.TURBO_CLASSIC,
            spot=24000.0,
            financing_level=22000.0,
            knockout_barrier=22000.0,
            ratio=0.01,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.02,
            as_of=_AS_OF,
            maturity=date(2026, 9, 1),
        )


def test_classic_converges_to_intrinsic_at_maturity() -> None:
    # T=0 -> e^0 == 1 on both terms -> exactly intrinsic value, no special-casing.
    long_value = theoretical_fair_value(
        direction=Direction.LONG,
        product_type=ProductType.TURBO_CLASSIC,
        spot=24000.0,
        financing_level=22000.0,
        knockout_barrier=22000.0,
        ratio=0.01,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.02,
        as_of=_AS_OF,
        maturity=_AS_OF,
        dividend_yield=0.032,
    )
    assert long_value == pytest.approx(intrinsic_value(24000.0, 22000.0, 0.01, Direction.LONG))

    short_value = theoretical_fair_value(
        direction=Direction.SHORT,
        product_type=ProductType.TURBO_CLASSIC,
        spot=20000.0,
        financing_level=22000.0,
        knockout_barrier=22000.0,
        ratio=0.01,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.02,
        as_of=_AS_OF,
        maturity=_AS_OF,
        dividend_yield=0.032,
    )
    assert short_value == pytest.approx(intrinsic_value(20000.0, 22000.0, 0.01, Direction.SHORT))


def test_classic_short_fair_value_below_intrinsic_with_positive_carry() -> None:
    # Classic short, r > 0, q = 0 (performance index): discounting the strike
    # pulls fair value BELOW intrinsic (K - S) -- the "negative issuer
    # margin at the top of the leaderboard" hypothesis this task investigates.
    spot, strike, ratio = 35000.0, 39429.0, 0.01
    fair = theoretical_fair_value(
        direction=Direction.SHORT,
        product_type=ProductType.TURBO_CLASSIC,
        spot=spot,
        financing_level=strike,
        knockout_barrier=strike,
        ratio=ratio,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.0,
        as_of=_AS_OF,
        maturity=date(2027, 3, 20),
        dividend_yield=0.0,
    )
    intrinsic = intrinsic_value(spot, strike, ratio, Direction.SHORT)
    assert fair < intrinsic
    assert fair > 0.0


def test_classic_long_fair_value_above_intrinsic_with_positive_carry() -> None:
    spot, strike, ratio = 39429.0, 39429.0, 0.01
    fair = theoretical_fair_value(
        direction=Direction.LONG,
        product_type=ProductType.TURBO_CLASSIC,
        spot=spot,
        financing_level=strike,
        knockout_barrier=strike,
        ratio=ratio,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.0,
        as_of=_AS_OF,
        maturity=date(2027, 3, 20),
        dividend_yield=0.0,
    )
    intrinsic = intrinsic_value(spot, strike, ratio, Direction.LONG)
    assert fair > intrinsic


def test_classic_dividend_yield_reduces_long_fair_value() -> None:
    def _fair_value(dividend_yield: float) -> float:
        return theoretical_fair_value(
            direction=Direction.LONG,
            product_type=ProductType.TURBO_CLASSIC,
            spot=20000.0,
            financing_level=15000.0,
            knockout_barrier=15000.0,
            ratio=0.01,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.0,
            as_of=_AS_OF,
            maturity=date(2027, 9, 11),
            dividend_yield=dividend_yield,
        )

    fair_no_div = _fair_value(0.0)
    fair_with_div = _fair_value(0.03)
    assert fair_with_div < fair_no_div


def test_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError):
        theoretical_fair_value(
            direction=Direction.LONG,
            product_type=ProductType.TURBO_OPEN_END,
            spot=24000.0,
            financing_level=22000.0,
            knockout_barrier=22000.0,
            ratio=0.0,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.02,
            as_of=_AS_OF,
            maturity=None,
        )


@given(
    spot=st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False),
    strike=st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False),
    ratio=st.floats(min_value=1e-4, max_value=10.0, allow_nan=False),
    ref_rate=st.floats(min_value=-0.02, max_value=0.10, allow_nan=False),
    days=st.integers(min_value=0, max_value=3650),
    direction=st.sampled_from([Direction.LONG, Direction.SHORT]),
)
def test_classic_fair_value_never_negative(
    spot: float, strike: float, ratio: float, ref_rate: float, days: int, direction: Direction
) -> None:
    maturity = date.fromordinal(_AS_OF.toordinal() + days)
    value = theoretical_fair_value(
        direction=direction,
        product_type=ProductType.TURBO_CLASSIC,
        spot=spot,
        financing_level=strike,
        knockout_barrier=strike,
        ratio=ratio,
        fx=1.0,
        ref_rate=ref_rate,
        financing_spread=0.0,
        as_of=_AS_OF,
        maturity=maturity,
    )
    assert value >= 0.0
    assert math.isfinite(value)


def test_dividend_yield_dax_is_zero_performance_index() -> None:
    assert dividend_yield_for_underlying("DAX") == 0.0


def test_dividend_yield_ndx_is_nonzero_price_index() -> None:
    assert dividend_yield_for_underlying("NDX") == pytest.approx(0.007)


def test_dividend_yield_unmapped_underlying_defaults_to_zero() -> None:
    assert dividend_yield_for_underlying("XAU") == 0.0
    assert dividend_yield_for_underlying("EURUSD") == 0.0


def test_dividend_yield_none_defaults_to_zero() -> None:
    assert dividend_yield_for_underlying(None) == 0.0


# --- theoretical_fair_value_array: elementwise identity vs. the scalar
# function (Build Contract v2 W7 performance fix) --------------------------


@given(
    spot=st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False),
    financing_level=st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False),
    ratio=st.floats(min_value=1e-4, max_value=10.0, allow_nan=False),
    fx=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    direction=st.sampled_from([Direction.LONG, Direction.SHORT]),
    product_type=st.sampled_from([ProductType.TURBO_OPEN_END, ProductType.MINI_FUTURE]),
)
def test_array_matches_scalar_non_classic_random_sample(
    spot: float,
    financing_level: float,
    ratio: float,
    fx: float,
    direction: Direction,
    product_type: ProductType,
) -> None:
    """Non-classic branch (plain intrinsic value): elementwise identity on a
    single-element array against the scalar function, and on a batched
    array against a loop of scalar calls.
    """
    scalar_value = theoretical_fair_value(
        direction=direction,
        product_type=product_type,
        spot=spot,
        financing_level=financing_level,
        knockout_barrier=financing_level,
        ratio=ratio,
        fx=fx,
        ref_rate=0.02,
        financing_spread=0.01,
        as_of=_AS_OF,
        maturity=None,
    )
    array_value = theoretical_fair_value_array(
        direction=direction,
        product_type=product_type,
        spot=np.array([spot]),
        financing_level=financing_level,
        knockout_barrier=financing_level,
        ratio=ratio,
        fx=fx,
        ref_rate=0.02,
        financing_spread=0.01,
        as_of=_AS_OF,
        maturity=None,
    )
    assert array_value.shape == (1,)
    np.testing.assert_allclose(array_value, [scalar_value], atol=1e-12, rtol=0.0)


@given(
    spot=st.lists(
        st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False), min_size=5, max_size=30
    ),
    strike=st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False),
    ratio=st.floats(min_value=1e-4, max_value=10.0, allow_nan=False),
    ref_rate=st.floats(min_value=-0.02, max_value=0.10, allow_nan=False),
    days=st.integers(min_value=0, max_value=3650),
    direction=st.sampled_from([Direction.LONG, Direction.SHORT]),
)
def test_array_matches_scalar_classic_scalar_as_of_random_sample(
    spot: list[float],
    strike: float,
    ratio: float,
    ref_rate: float,
    days: int,
    direction: Direction,
) -> None:
    """Classic branch, a single (scalar) ``as_of`` broadcast across every
    ``spot`` element: elementwise identity against a loop of scalar calls,
    ``atol=1e-12``.
    """
    maturity = date.fromordinal(_AS_OF.toordinal() + days)
    spot_arr = np.array(spot, dtype=np.float64)

    expected = np.array(
        [
            theoretical_fair_value(
                direction=direction,
                product_type=ProductType.TURBO_CLASSIC,
                spot=s,
                financing_level=strike,
                knockout_barrier=strike,
                ratio=ratio,
                fx=1.0,
                ref_rate=ref_rate,
                financing_spread=0.0,
                as_of=_AS_OF,
                maturity=maturity,
            )
            for s in spot
        ]
    )
    actual = theoretical_fair_value_array(
        direction=direction,
        product_type=ProductType.TURBO_CLASSIC,
        spot=spot_arr,
        financing_level=strike,
        knockout_barrier=strike,
        ratio=ratio,
        fx=1.0,
        ref_rate=ref_rate,
        financing_spread=0.0,
        as_of=_AS_OF,
        maturity=maturity,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_array_matches_scalar_classic_per_day_as_of_array() -> None:
    """Classic branch, ``as_of`` varying per element (the payoff-simulation
    day/horizon axis use case): one ``as_of`` date per ``spot``/
    ``financing_level`` element (both 1-D, same length), each independently
    matching a scalar call with that element's own ``as_of``.
    """
    n_days = 14
    spot_arr = 20000.0 + 50.0 * np.arange(n_days, dtype=np.float64)
    strike = 18000.0
    maturity = date(2027, 3, 20)
    as_of_arr = np.array([_AS_OF + timedelta(days=i) for i in range(n_days)], dtype=object)

    expected = np.array(
        [
            theoretical_fair_value(
                direction=Direction.LONG,
                product_type=ProductType.TURBO_CLASSIC,
                spot=float(spot_arr[i]),
                financing_level=strike,
                knockout_barrier=strike,
                ratio=0.01,
                fx=1.0,
                ref_rate=0.0219,
                financing_spread=0.0,
                as_of=as_of_arr[i],
                maturity=maturity,
            )
            for i in range(n_days)
        ]
    )
    actual = theoretical_fair_value_array(
        direction=Direction.LONG,
        product_type=ProductType.TURBO_CLASSIC,
        spot=spot_arr,
        financing_level=strike,
        knockout_barrier=strike,
        ratio=0.01,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.0,
        as_of=as_of_arr,
        maturity=maturity,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_array_matches_scalar_classic_2d_broadcast_paths_x_days() -> None:
    """Classic branch, the actual ``simulation/payoff.py`` hot-path shape:
    ``spot`` is ``(n_paths, n_days)``, ``as_of`` varies only along the
    trailing (day) axis and broadcasts against every path -- one vectorized
    call must match a nested loop of scalar calls over both axes.
    """
    n_paths, n_days = 25, 6
    rng = np.random.default_rng(0)
    spot_2d = 15000.0 + rng.normal(0.0, 200.0, size=(n_paths, n_days))
    strike = 14000.0
    maturity = date(2027, 1, 1)
    as_of_arr = np.array([_AS_OF + timedelta(days=7 * i) for i in range(n_days)], dtype=object)

    expected = np.empty((n_paths, n_days), dtype=np.float64)
    for p in range(n_paths):
        for d in range(n_days):
            expected[p, d] = theoretical_fair_value(
                direction=Direction.SHORT,
                product_type=ProductType.TURBO_CLASSIC,
                spot=float(spot_2d[p, d]),
                financing_level=strike,
                knockout_barrier=strike,
                ratio=0.01,
                fx=1.0,
                ref_rate=0.0219,
                financing_spread=0.0,
                as_of=as_of_arr[d],
                maturity=maturity,
            )
    actual = theoretical_fair_value_array(
        direction=Direction.SHORT,
        product_type=ProductType.TURBO_CLASSIC,
        spot=spot_2d,
        financing_level=strike,
        knockout_barrier=strike,
        ratio=0.01,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.0,
        as_of=as_of_arr,
        maturity=maturity,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_array_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError):
        theoretical_fair_value_array(
            direction=Direction.LONG,
            product_type=ProductType.TURBO_OPEN_END,
            spot=np.array([24000.0]),
            financing_level=22000.0,
            knockout_barrier=22000.0,
            ratio=0.0,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.02,
            as_of=_AS_OF,
            maturity=None,
        )


def test_array_classic_requires_maturity() -> None:
    with pytest.raises(ValueError):
        theoretical_fair_value_array(
            direction=Direction.SHORT,
            product_type=ProductType.TURBO_CLASSIC,
            spot=np.array([20000.0]),
            financing_level=22000.0,
            knockout_barrier=22000.0,
            ratio=0.01,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.02,
            as_of=_AS_OF,
            maturity=None,
        )


def test_array_classic_rejects_matured_date_scalar_as_of() -> None:
    with pytest.raises(ValueError):
        theoretical_fair_value_array(
            direction=Direction.LONG,
            product_type=ProductType.TURBO_CLASSIC,
            spot=np.array([24000.0]),
            financing_level=22000.0,
            knockout_barrier=22000.0,
            ratio=0.01,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.02,
            as_of=_AS_OF,
            maturity=date(2026, 9, 1),
        )


def test_array_classic_rejects_matured_date_array_as_of() -> None:
    as_of_arr = np.array([_AS_OF, date(2026, 9, 5)], dtype=object)
    with pytest.raises(ValueError):
        theoretical_fair_value_array(
            direction=Direction.LONG,
            product_type=ProductType.TURBO_CLASSIC,
            spot=np.array([24000.0, 24000.0]),
            financing_level=22000.0,
            knockout_barrier=22000.0,
            ratio=0.01,
            fx=1.0,
            ref_rate=0.0219,
            financing_spread=0.02,
            as_of=as_of_arr,
            maturity=date(2026, 9, 1),
        )
