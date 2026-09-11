from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.pricing.intrinsic import implied_underlying, intrinsic_value, leverage
from turboedge.storage.schemas import Direction


def test_intrinsic_value_dax_long() -> None:
    # DAX Long, S=24000, F=22000, ratio=0.01 -> 2000 * 0.01 = 20.00
    assert intrinsic_value(24000.0, 22000.0, 0.01, Direction.LONG) == pytest.approx(20.00)


def test_intrinsic_value_dax_short_otm() -> None:
    # Same market, Short leg is out of the money -> 0.
    assert intrinsic_value(24000.0, 22000.0, 0.01, Direction.SHORT) == pytest.approx(0.0)


def test_intrinsic_value_dax_short_itm() -> None:
    # Short ITM: S=20000, F=22000, ratio=0.01 -> 2000 * 0.01 = 20.00
    assert intrinsic_value(20000.0, 22000.0, 0.01, Direction.SHORT) == pytest.approx(20.00)


def test_intrinsic_value_usd_underlying_with_fx() -> None:
    # Quanto-free USD underlying (e.g. gold), EUR product, fx=1.10 (USD per EUR).
    # S=2400 USD, F=2200 USD, ratio=0.01, fx=1.10
    # -> max(2400-2200,0)*0.01/1.10 = 20*0.01/1.10 = 0.181818...
    value = intrinsic_value(2400.0, 2200.0, 0.01, Direction.LONG, fx=1.10)
    assert value == pytest.approx(200.0 * 0.01 / 1.10)


def test_intrinsic_value_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError):
        intrinsic_value(24000.0, 22000.0, 0.0, Direction.LONG)


def test_implied_underlying_long_roundtrip() -> None:
    # ask = intrinsic for an ITM long; implied_underlying should recover spot.
    spot, f, ratio = 24000.0, 22000.0, 0.01
    price = intrinsic_value(spot, f, ratio, Direction.LONG)
    assert implied_underlying(price, f, ratio, Direction.LONG) == pytest.approx(spot)


def test_implied_underlying_short_roundtrip() -> None:
    spot, f, ratio = 20000.0, 22000.0, 0.01
    price = intrinsic_value(spot, f, ratio, Direction.SHORT)
    assert implied_underlying(price, f, ratio, Direction.SHORT) == pytest.approx(spot)


def test_implied_underlying_price_must_be_positive() -> None:
    with pytest.raises(ValueError):
        implied_underlying(0.0, 22000.0, 0.01, Direction.LONG)


def test_implied_underlying_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError):
        implied_underlying(20.0, 22000.0, 0.0, Direction.LONG)


def test_leverage_basic() -> None:
    # S=24000, ask=20, ratio=0.01 -> 24000*0.01/20 = 12
    assert leverage(24000.0, 20.0, 0.01) == pytest.approx(12.0)


def test_leverage_with_fx() -> None:
    lev = leverage(2400.0, 20.0, 0.01, fx=1.10)
    assert lev == pytest.approx(2400.0 * 0.01 / 1.10 / 20.0)


def test_leverage_price_must_be_positive() -> None:
    with pytest.raises(ValueError):
        leverage(24000.0, 0.0, 0.01)


def test_leverage_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError):
        leverage(24000.0, 20.0, -0.01)


@given(
    spot=st.floats(min_value=1.0, max_value=1_000_000, allow_nan=False),
    financing_level=st.floats(min_value=1.0, max_value=1_000_000, allow_nan=False),
    ratio=st.floats(min_value=1e-6, max_value=10.0, allow_nan=False),
    fx=st.floats(min_value=1e-3, max_value=100.0, allow_nan=False),
    direction=st.sampled_from([Direction.LONG, Direction.SHORT]),
)
def test_intrinsic_value_never_negative(
    spot: float, financing_level: float, ratio: float, fx: float, direction: Direction
) -> None:
    value = intrinsic_value(spot, financing_level, ratio, direction, fx=fx)
    assert value >= 0.0
    assert math.isfinite(value)


@given(
    spot=st.floats(min_value=100.0, max_value=1_000_000, allow_nan=False),
    financing_level=st.floats(min_value=1.0, max_value=99.0, allow_nan=False),
    ratio=st.floats(min_value=1e-6, max_value=10.0, allow_nan=False),
    fx=st.floats(min_value=1e-3, max_value=100.0, allow_nan=False),
)
def test_implied_underlying_roundtrip_itm_long(
    spot: float, financing_level: float, ratio: float, fx: float
) -> None:
    # spot > financing_level guaranteed by the disjoint ranges above -> strictly ITM.
    price = intrinsic_value(spot, financing_level, ratio, Direction.LONG, fx=fx)
    recovered = implied_underlying(price, financing_level, ratio, Direction.LONG, fx=fx)
    assert recovered == pytest.approx(spot, rel=1e-6)


@given(
    spot=st.floats(min_value=1.0, max_value=99.0, allow_nan=False),
    financing_level=st.floats(min_value=100.0, max_value=1_000_000, allow_nan=False),
    ratio=st.floats(min_value=1e-6, max_value=10.0, allow_nan=False),
    fx=st.floats(min_value=1e-3, max_value=100.0, allow_nan=False),
)
def test_implied_underlying_roundtrip_itm_short(
    spot: float, financing_level: float, ratio: float, fx: float
) -> None:
    price = intrinsic_value(spot, financing_level, ratio, Direction.SHORT, fx=fx)
    recovered = implied_underlying(price, financing_level, ratio, Direction.SHORT, fx=fx)
    assert recovered == pytest.approx(spot, rel=1e-6)
