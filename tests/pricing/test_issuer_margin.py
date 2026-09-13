from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.pricing.fair_value import theoretical_fair_value
from turboedge.pricing.issuer_margin import decompose_ask
from turboedge.storage.schemas import Direction, ProductType


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
    # ask == mid + trading_spread_component (mid decomposes into
    # fair_value + fair_gap_premium + financing_drag + issuer_margin, never
    # `intrinsic` -- see decompose_ask's docstring for why the two differ
    # for a turbo_classic).
    mid_reconstructed = (
        result.fair_gap_premium + result.financing_drag + result.issuer_margin + result.intrinsic
    )
    total = mid_reconstructed + result.trading_spread_component
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
    # ask == mid + trading_spread_component (mid decomposes into
    # fair_value + fair_gap_premium + financing_drag + issuer_margin, never
    # `intrinsic` -- see decompose_ask's docstring for why the two differ
    # for a turbo_classic).
    mid_reconstructed = (
        result.fair_gap_premium + result.financing_drag + result.issuer_margin + result.intrinsic
    )
    total = mid_reconstructed + result.trading_spread_component
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


def test_decompose_ask_classic_uses_fair_value_not_intrinsic_for_margin() -> None:
    """turbo_classic: issuer_margin is computed against the carry-adjusted
    fair value, not plain intrinsic -- `intrinsic` on the result stays the
    plain inner value (unchanged meaning), so for a classic (where fair
    value genuinely differs from intrinsic) `mid - intrinsic` and
    `issuer_margin + fair_gap_premium + financing_drag` are NOT the same
    number anymore (Build Contract W1)."""
    as_of = date(2026, 9, 11)
    maturity = date(2027, 3, 20)
    spot, strike, ratio = 35000.0, 39429.225, 0.001
    fair_value = theoretical_fair_value(
        direction=Direction.SHORT,
        product_type=ProductType.TURBO_CLASSIC,
        spot=spot,
        financing_level=strike,
        knockout_barrier=strike,
        ratio=ratio,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.02,
        as_of=as_of,
        maturity=maturity,
    )
    # Price the certificate right at its own fair value (no real market
    # margin) to isolate the carry effect cleanly.
    ask = fair_value + 0.001
    bid = fair_value - 0.001
    result = decompose_ask(
        bid=bid,
        ask=ask,
        spot=spot,
        financing_level=strike,
        ratio=ratio,
        direction=Direction.SHORT,
        fair_gap_premium=0.0,
        financing_spread=0.02,
        ref_rate=0.0219,
        product_type=ProductType.TURBO_CLASSIC,
        knockout_barrier=strike,
        as_of=as_of,
        maturity=maturity,
    )

    # financing_drag is 0 for classics (already inside the present-valued
    # fair value, not a separate daily accrual).
    assert result.financing_drag == 0.0
    # issuer_margin measured against fair value is ~0 (priced at fair value).
    assert result.issuer_margin == pytest.approx(0.0, abs=1e-6)
    # intrinsic stays the true inner value; for this deep-carry SHORT
    # example (discounting the strike pulls fair value BELOW intrinsic --
    # see fair_value.py's module docstring), intrinsic is genuinely ABOVE
    # fair value -- so the OLD (pre-fix) intrinsic-based premium would have
    # been strongly negative even though the product is priced exactly at
    # its fair value.
    assert result.intrinsic > fair_value
    old_style_premium = result.mid - result.intrinsic
    assert old_style_premium < -0.01


def test_decompose_ask_classic_requires_knockout_barrier_and_as_of() -> None:
    with pytest.raises(ValueError):
        decompose_ask(
            bid=13.0,
            ask=13.1,
            spot=35000.0,
            financing_level=39429.225,
            ratio=0.001,
            direction=Direction.SHORT,
            fair_gap_premium=0.0,
            financing_spread=0.02,
            ref_rate=0.0219,
            product_type=ProductType.TURBO_CLASSIC,
        )


def test_decompose_ask_classic_sum_identity_via_recovered_fair_value() -> None:
    """The two identities documented on decompose_ask hold for a classic
    too, once `fair_value` is recovered (it is not a stored field)."""
    as_of = date(2026, 9, 11)
    maturity = date(2027, 3, 20)
    result = decompose_ask(
        bid=13.80,
        ask=13.90,
        spot=35000.0,
        financing_level=39429.225,
        ratio=0.001,
        direction=Direction.SHORT,
        fair_gap_premium=0.0,
        financing_spread=0.02,
        ref_rate=0.0219,
        product_type=ProductType.TURBO_CLASSIC,
        knockout_barrier=39429.225,
        as_of=as_of,
        maturity=maturity,
    )
    recovered_fair_value = (
        result.mid - result.fair_gap_premium - result.financing_drag - result.issuer_margin
    )
    expected_fair_value = theoretical_fair_value(
        direction=Direction.SHORT,
        product_type=ProductType.TURBO_CLASSIC,
        spot=35000.0,
        financing_level=39429.225,
        knockout_barrier=39429.225,
        ratio=0.001,
        fx=1.0,
        ref_rate=0.0219,
        financing_spread=0.02,
        as_of=as_of,
        maturity=maturity,
    )
    assert recovered_fair_value == pytest.approx(expected_fair_value, abs=1e-9)
    assert result.ask == pytest.approx(result.mid + result.trading_spread_component, abs=1e-9)


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
    # See the two comparable tests above: this identity holds for the
    # default (non-classic) product_type, where fair_value == intrinsic.
    mid_reconstructed = (
        result.fair_gap_premium + result.financing_drag + result.issuer_margin + result.intrinsic
    )
    total = mid_reconstructed + result.trading_spread_component
    assert total == pytest.approx(result.ask, abs=1e-6)
