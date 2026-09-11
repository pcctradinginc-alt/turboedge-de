from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.pricing.issuer_margin import decompose_ask
from turboedge.storage.schemas import Direction


def test_decompose_ask_components_sum_to_ask_long() -> None:
    result = decompose_ask(
        bid=4.80,
        ask=4.86,
        spot=24000.0,
        financing_level=22000.0,
        ratio=0.01,
        direction=Direction.LONG,
        fair_gap_premium=0.03,
        financing_spread=0.025,
        ref_rate=0.03,
    )
    total = (
        result.intrinsic
        + result.trading_spread_component
        + result.fair_gap_premium
        + result.financing_drag
        + result.issuer_margin
    )
    assert total == pytest.approx(result.ask, abs=1e-9)


def test_decompose_ask_components_sum_to_ask_short() -> None:
    result = decompose_ask(
        bid=5.10,
        ask=5.18,
        spot=20000.0,
        financing_level=22000.0,
        ratio=0.01,
        direction=Direction.SHORT,
        fair_gap_premium=0.02,
        financing_spread=0.018,
        ref_rate=0.03,
    )
    total = (
        result.intrinsic
        + result.trading_spread_component
        + result.fair_gap_premium
        + result.financing_drag
        + result.issuer_margin
    )
    assert total == pytest.approx(result.ask, abs=1e-9)


def test_decompose_ask_field_values() -> None:
    result = decompose_ask(
        bid=4.80,
        ask=4.86,
        spot=24000.0,
        financing_level=22000.0,
        ratio=0.01,
        direction=Direction.LONG,
        fair_gap_premium=0.03,
        financing_spread=0.025,
        ref_rate=0.03,
    )
    assert result.mid == pytest.approx(4.83)
    assert result.intrinsic == pytest.approx(20.0)
    assert result.trading_spread_component == pytest.approx(4.86 - 4.83)
    expected_financing_drag = 22000.0 * (0.03 + 0.025) / 360.0 * 0.01
    assert result.financing_drag == pytest.approx(expected_financing_drag)
    premium = 4.83 - 20.0
    expected_margin = premium - 0.03 - expected_financing_drag
    assert result.issuer_margin == pytest.approx(expected_margin)
    assert result.spread_pct == pytest.approx(result.trading_spread_component / result.ask)
    assert result.issuer_margin_pct == pytest.approx(result.issuer_margin / result.ask)


def test_decompose_ask_short_financing_drag_can_be_negative() -> None:
    result = decompose_ask(
        bid=5.10,
        ask=5.18,
        spot=20000.0,
        financing_level=22000.0,
        ratio=0.01,
        direction=Direction.SHORT,
        fair_gap_premium=0.02,
        financing_spread=0.01,  # below ref_rate -> negative financing_drag (a credit)
        ref_rate=0.03,
    )
    assert result.financing_drag < 0.0


def test_decompose_ask_rejects_bid_above_ask() -> None:
    with pytest.raises(ValueError):
        decompose_ask(
            bid=5.0,
            ask=4.9,
            spot=24000.0,
            financing_level=22000.0,
            ratio=0.01,
            direction=Direction.LONG,
            fair_gap_premium=0.0,
            financing_spread=0.025,
            ref_rate=0.03,
        )


def test_decompose_ask_rejects_nonpositive_ask() -> None:
    with pytest.raises(ValueError):
        decompose_ask(
            bid=0.0,
            ask=0.0,
            spot=24000.0,
            financing_level=22000.0,
            ratio=0.01,
            direction=Direction.LONG,
            fair_gap_premium=0.0,
            financing_spread=0.025,
            ref_rate=0.03,
        )


@given(
    bid=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
    spread=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    spot=st.floats(min_value=1000.0, max_value=30000.0, allow_nan=False),
    financing_level=st.floats(min_value=1000.0, max_value=30000.0, allow_nan=False),
    fair_gap_premium=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    financing_spread=st.floats(min_value=-0.05, max_value=0.1, allow_nan=False),
    ref_rate=st.floats(min_value=-0.02, max_value=0.1, allow_nan=False),
    direction=st.sampled_from([Direction.LONG, Direction.SHORT]),
)
def test_decompose_ask_sum_identity_property(
    bid: float,
    spread: float,
    spot: float,
    financing_level: float,
    fair_gap_premium: float,
    financing_spread: float,
    ref_rate: float,
    direction: Direction,
) -> None:
    ask = bid + spread
    if ask <= 0:
        return
    result = decompose_ask(
        bid=bid,
        ask=ask,
        spot=spot,
        financing_level=financing_level,
        ratio=0.01,
        direction=direction,
        fair_gap_premium=fair_gap_premium,
        financing_spread=financing_spread,
        ref_rate=ref_rate,
    )
    total = (
        result.intrinsic
        + result.trading_spread_component
        + result.fair_gap_premium
        + result.financing_drag
        + result.issuer_margin
    )
    assert total == pytest.approx(result.ask, abs=1e-6)
