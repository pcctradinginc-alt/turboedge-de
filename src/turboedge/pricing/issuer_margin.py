"""Ask-price decomposition into intrinsic value plus its cost components.

Formula reference: Master Spec §13.2 ("Kostenzerlegung"), the Build
Contract's "Formeln (verbindlich)" section, and Build Contract v2's W1
pricing-fix section (``pricing/fair_value.py``).

The ask price is decomposed as::

    Ask = Fair Value
        + Fair Gap Premium
        + Trading Spread Component
        + Financing Drag
        + Issuer Margin

so that "issuer margin" is whatever is left over after every cost the
issuer can legitimately point to (financing, gap risk, the quoted spread
itself, AND -- new in this decomposition -- the theoretical carry a fixed-
maturity ``turbo_classic`` genuinely owes its buyer/seller) has been
subtracted out -- never a point estimate we take on faith from the ask
price alone (CLAUDE.md rule 14).

``fair_value`` (``pricing/fair_value.theoretical_fair_value``) replaces the
plain ``intrinsic_value`` this decomposition used before: for
``turbo_open_end``/``mini_future`` the two are identical (their financing
cost already rolls into ``financing_level`` day by day, Build Contract
"Formeln (verbindlich)"), but a ``turbo_classic`` (fixed strike, fixed
maturity) genuinely trades below/above raw intrinsic value by a real,
legitimate present-value/carry term -- treating that carry as "issuer
margin" would misclassify an honest pricing effect as an anomaly. The
``intrinsic`` field on :class:`~turboedge.storage.schemas.CostDecomposition`
is unchanged in meaning (the true inner/moneyness value, ``mid - intrinsic``
is NOT how margin is computed anymore); ``fair_value`` itself is not stored
as a separate field (``CostDecomposition``'s schema is out of this module's
scope) but is fully recoverable as ``intrinsic + (mid - trading_spread... )``
-- see the two identities documented on :func:`decompose_ask` below.
"""

from __future__ import annotations

from datetime import date

from turboedge.pricing.fair_value import theoretical_fair_value
from turboedge.pricing.intrinsic import intrinsic_value
from turboedge.storage.schemas import CostDecomposition, Direction, ProductType

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
    *,
    product_type: ProductType = ProductType.TURBO_OPEN_END,
    knockout_barrier: float | None = None,
    as_of: date | None = None,
    maturity: date | None = None,
    dividend_yield: float = 0.0,
) -> CostDecomposition:
    """Decompose ``ask`` into fair value, gap premium, spread, financing
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
        fair_value = theoretical_fair_value(..., product_type, as_of, maturity,
                                             dividend_yield)
                     # == intrinsic for turbo_open_end/mini_future (the
                     # default product_type); carry-adjusted for
                     # turbo_classic (Build Contract W1)
        premium_over_fair = mid - fair_value
        financing_drag = 0.0 if product_type is TURBO_CLASSIC else 1 calendar
            day of financing cost
            (Long:  F * (r + s) / 360 * ratio / fx
             Short: F * (s - r) / 360 * ratio / fx)
            -- a classic's financing cost is already inside its
            present-valued ``fair_value``, not a separate daily accrual
            (Build Contract W1: "financing_drag fuer Classics = 0, da im
            Preis enthalten").
        issuer_margin = premium_over_fair - fair_gap_premium - financing_drag

    Two identities always hold (the second is the "sum identity" tested by
    ``tests/pricing/test_issuer_margin.py``'s property test) -- neither
    requires ``intrinsic`` on the right-hand side, since ``intrinsic`` and
    ``fair_value`` genuinely differ for a ``turbo_classic``::

        ask == mid + trading_spread_component
        mid == fair_value + fair_gap_premium + financing_drag + issuer_margin

    ``intrinsic`` on the returned :class:`CostDecomposition` is always the
    plain inner/moneyness value (``pricing/intrinsic.intrinsic_value``),
    unchanged in meaning from before this fix -- only what ``issuer_margin``
    is computed *relative to* changed. ``fair_value`` is not itself a stored
    field (unchanged ``CostDecomposition`` schema); recover it as
    ``mid - fair_gap_premium - financing_drag - issuer_margin`` when needed.

    ``issuer_margin`` can be negative (e.g. a temporarily underpriced
    product); a strongly negative value is a data-quality signal handled by
    ``pricing/integrity.py``, not clamped away here.

    Args:
        product_type: defaults to ``TURBO_OPEN_END`` (``fair_value ==
            intrinsic``, identical behavior to before this fix) so existing
            callers that only ever priced open-end/mini-future products need
            not change. Pass ``ProductType.TURBO_CLASSIC`` (with
            ``knockout_barrier``, ``as_of`` and a real ``maturity``) to price
            a fixed-maturity classic turbo's real carry instead.
        knockout_barrier: required (not ``None``) when ``product_type is
            TURBO_CLASSIC`` (passed through to ``theoretical_fair_value``
            for interface uniformity; unused by its formula).
        as_of: valuation date; required (not ``None``) when ``product_type
            is TURBO_CLASSIC``.
        maturity: the certificate's fixed maturity date; required (not
            ``None``) when ``product_type is TURBO_CLASSIC`` --
            ``theoretical_fair_value`` raises ``ValueError`` otherwise
            (CLAUDE.md rule 29: never silently priced as open-end).
        dividend_yield: underlying's continuous dividend yield, decimal;
            only used by the classic branch (see
            ``pricing/fair_value.dividend_yield_for_underlying``).

    Raises:
        ValueError: if ``bid > ask``, ``ask <= 0``, or (via
            ``theoretical_fair_value``) ``product_type is TURBO_CLASSIC``
            with a missing/invalid ``knockout_barrier``, ``as_of`` or
            ``maturity``.
    """
    if bid > ask:
        raise ValueError(f"bid ({bid!r}) must be <= ask ({ask!r})")
    if not (ask > 0):
        raise ValueError(f"ask must be > 0, got {ask!r}")

    mid = (bid + ask) / 2.0
    trading_spread_component = ask - mid
    intrinsic = intrinsic_value(spot, financing_level, ratio, direction, fx)

    if product_type == ProductType.TURBO_CLASSIC:
        if knockout_barrier is None or as_of is None:
            raise ValueError(
                "turbo_classic requires knockout_barrier and as_of to compute a fair value"
            )
        fair_value = theoretical_fair_value(
            direction=direction,
            product_type=product_type,
            spot=spot,
            financing_level=financing_level,
            knockout_barrier=knockout_barrier,
            ratio=ratio,
            fx=fx,
            ref_rate=ref_rate,
            financing_spread=financing_spread,
            as_of=as_of,
            maturity=maturity,
            dividend_yield=dividend_yield,
        )
        financing_drag = 0.0
    else:
        # turbo_open_end / mini_future / unknown: fair value == intrinsic
        # (their financing cost is already rolled into financing_level day
        # by day) -- reuse the value already computed above rather than
        # calling theoretical_fair_value again for the identical result.
        fair_value = intrinsic
        if direction == Direction.LONG:
            rate_term = ref_rate + financing_spread
        else:
            rate_term = financing_spread - ref_rate
        financing_drag = (
            financing_level
            * rate_term
            * _FINANCING_DRAG_CALENDAR_DAYS
            / _DAY_COUNT_BASIS
            * ratio
            / fx
        )

    premium_over_fair = mid - fair_value
    issuer_margin = premium_over_fair - fair_gap_premium - financing_drag

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
