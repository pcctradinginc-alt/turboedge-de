"""Ask-price decomposition into intrinsic value plus its cost components.

Formula reference: Master Spec §13.2 ("Kostenzerlegung") and the Build
Contract's "Formeln (verbindlich)" section.

The ask price is decomposed as::

    Ask = Intrinsic Value
        + Fair Gap Premium
        + Trading Spread Component
        + Financing Drag
        + Issuer Margin

so that "issuer margin" is whatever is left over after every cost the
issuer can legitimately point to (financing, gap risk, the quoted spread
itself) has been subtracted out -- never a point estimate we take on faith
from the ask price alone (CLAUDE.md rule 14).
"""

from __future__ import annotations

from turboedge.pricing.intrinsic import intrinsic_value
from turboedge.storage.schemas import CostDecomposition, Direction

_FINANCING_DRAG_CALENDAR_DAYS = 1.0
_DAY_COUNT_BASIS = 360.0


def decompose_ask(
    bid: float,
    ask: float,
    spot: float,
    financing_level: float,
    ratio: float,
    direction: Direction,
    fair_gap_premium: float,
    financing_spread: float,
    ref_rate: float,
    fx: float = 1.0,
) -> CostDecomposition:
    """Decompose ``ask`` into intrinsic value, gap premium, spread, financing
    drag and issuer margin.

    ``financing_spread`` is the spread ``s`` to use for the one-day financing
    drag term -- callers resolve this beforehand, preferring
    ``realized_financing_spread`` (``pricing/financing.py``) and falling back
    to ``configs/risk.yaml``'s ``default_financing_spread`` when no clean
    financing-level history is available (Master Spec CLAUDE.md rule 13);
    this function does not know about that fallback itself.

    Steps (all absolute figures in product currency, per certificate)::

        mid = (bid + ask) / 2
        trading_spread_component = ask - mid
        intrinsic = intrinsic_value(spot, financing_level, ratio, direction, fx)
        premium = mid - intrinsic
        financing_drag = 1 calendar day of financing cost
            (Long:  F * (r + s) / 360 * ratio / fx
             Short: F * (s - r) / 360 * ratio / fx)
        issuer_margin = premium - fair_gap_premium - financing_drag

    ``issuer_margin`` can be negative (e.g. a temporarily underpriced
    product); a strongly negative value is a data-quality signal handled by
    ``pricing/integrity.py``, not clamped away here.

    Raises:
        ValueError: if ``bid > ask`` or ``ask <= 0``.
    """
    if bid > ask:
        raise ValueError(f"bid ({bid!r}) must be <= ask ({ask!r})")
    if not (ask > 0):
        raise ValueError(f"ask must be > 0, got {ask!r}")

    mid = (bid + ask) / 2.0
    trading_spread_component = ask - mid
    intrinsic = intrinsic_value(spot, financing_level, ratio, direction, fx)
    premium = mid - intrinsic

    if direction == Direction.LONG:
        rate_term = ref_rate + financing_spread
    else:
        rate_term = financing_spread - ref_rate
    financing_drag = (
        financing_level * rate_term * _FINANCING_DRAG_CALENDAR_DAYS / _DAY_COUNT_BASIS * ratio / fx
    )

    issuer_margin = premium - fair_gap_premium - financing_drag

    return CostDecomposition(
        ask=ask,
        bid=bid,
        mid=mid,
        intrinsic=intrinsic,
        trading_spread_component=trading_spread_component,
        fair_gap_premium=fair_gap_premium,
        financing_drag=financing_drag,
        issuer_margin=issuer_margin,
        spread_pct=trading_spread_component / ask,
        gap_premium_pct=fair_gap_premium / ask,
        financing_drag_pct=financing_drag / ask,
        issuer_margin_pct=issuer_margin / ask,
    )
