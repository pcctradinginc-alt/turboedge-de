"""Theoretical fair value for turbo/knockout certificates.

Formula reference: Build Contract v2, section "W1 -- Pricing fix".

Not every certificate's fair value equals its intrinsic value. A
``turbo_classic`` (fixed strike ``K``, fixed maturity ``T``) prices in the
cost of carrying that fixed strike to expiry -- present-valuing the intrinsic
payoff back to today, exactly like a forward contract's fair value differs
from its "if settled today" value::

    Long:  FV = max(S * e^{-q*T} - K * e^{-r*T}, 0) * ratio / fx
    Short: FV = max(K * e^{-r*T} - S * e^{-q*T}, 0) * ratio / fx

with ``T`` in years (act/365) and ``q`` the underlying's continuous dividend
yield (0 for a performance index such as the DAX, which already reinvests
dividends into the index level -- see the ``dividend_yield`` argument).

``turbo_open_end`` and ``mini_future`` roll their financing level ``F``
forward daily (``pricing/financing.py``: ``F(t+1) = F(t) * (1 + (r+/-s) *
dt/360)``), so their cost of carry is already embedded in ``F`` itself; fair
value for those product types is simply the intrinsic value
(``pricing/intrinsic.py``), with no separate discounting term.

At ``as_of == maturity`` (``T == 0``), the classic-turbo formula reduces
exactly to the intrinsic value (``e^0 == 1`` on both terms), so the two
branches agree at expiry with no special-casing needed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import numpy.typing as npt
import structlog

from turboedge.pricing.intrinsic import intrinsic_value
from turboedge.storage.schemas import Direction, ProductType

logger = structlog.get_logger(__name__)

_DAY_COUNT_BASIS_YEARS = 365.0

# Continuous annualized dividend yield ``q`` per ``underlying_id``, for
# ``theoretical_fair_value``'s classic-turbo branch (Build Contract W1:
# "Dividendenrendite per Underlying als Modul-Konstante mit Quelle/
# Begruendung"). Static, documented approximations -- not a live feed.
#
# DAX: 0.0 -- the DAX is a *performance* (total-return) index: Deutsche
# Boerse's index methodology reinvests every constituent's net dividend back
# into the index level itself, so there is no separate dividend leakage a
# forward/carry formula needs to discount away (unlike a *price* index).
# NDX: the Nasdaq-100 is a *price* index (dividends are NOT reinvested into
# the level), but its constituents have historically paid a low, fairly
# stable yield -- roughly 0.6-0.9% p.a. over the past several years per
# publicly reported Nasdaq-100 fact sheets/index methodology documents. 0.007
# (0.7%) is used as a single-point mid-of-range approximation. This is a
# coarse, static estimate, not a live index-dividend feed -- refine with a
# real data source before relying on it beyond this milestone's WATCH-only
# scope (no ACTIONABLE output depends on it yet).
_DIVIDEND_YIELD_BY_UNDERLYING: dict[str, float] = {
    "DAX": 0.0,
    "NDX": 0.007,
}


def dividend_yield_for_underlying(underlying_id: str | None) -> float:
    """Continuous dividend yield ``q`` to pass as ``theoretical_fair_value(dividend_yield=...)``.

    Returns ``0.0`` for ``None`` or any ``underlying_id`` not in
    :data:`_DIVIDEND_YIELD_BY_UNDERLYING` -- e.g. a genuine total-return/
    performance index not yet catalogued here, or an FX/commodity underlying
    (the dividend-yield concept does not apply; FX/commodity cost-of-carry is
    out of scope for this milestone's ``theoretical_fair_value``) -- logged
    rather than silently guessing a nonzero value (CLAUDE.md rule 29: never
    silently impute pricing-critical data). ``0.0`` is itself the safe,
    documented default (no dividend adjustment), not a guess.
    """
    if underlying_id is None:
        return 0.0
    yld = _DIVIDEND_YIELD_BY_UNDERLYING.get(underlying_id)
    if yld is None:
        logger.warning("dividend_yield_unmapped_defaulting_to_zero", underlying_id=underlying_id)
        return 0.0
    return yld


def theoretical_fair_value(
    *,
    direction: Direction,
    product_type: ProductType,
    spot: float,
    financing_level: float,
    knockout_barrier: float,
    ratio: float,
    fx: float,
    ref_rate: float,
    financing_spread: float,
    as_of: date,
    maturity: date | None,
    dividend_yield: float = 0.0,
) -> float:
    """Theoretical (no-issuer-margin) fair value of one certificate, in product currency.

    ``turbo_classic`` present-values the fixed strike/spot to ``maturity``
    (see module docstring); ``turbo_open_end`` and ``mini_future`` return
    plain intrinsic value, since their financing cost is already rolled into
    ``financing_level`` day by day.

    ``knockout_barrier`` and ``financing_spread`` are accepted for interface
    uniformity with ``ProductTerms``/``decompose_ask`` (every caller passes
    the same field set regardless of product type) but are not used by this
    formula directly: a classic turbo's fixed strike is discounted at the
    reference rate ``ref_rate`` only -- the issuer's own funding spread on
    top of that is issuer margin, not a "fair" cost
    (``pricing/issuer_margin.py`` keeps ``financing_drag`` at 0 for classics
    precisely because it is already inside this present-value term).

    Args:
        as_of: valuation date.
        maturity: the certificate's fixed maturity date. Required (not
            ``None``) when ``product_type is ProductType.TURBO_CLASSIC``.
        dividend_yield: underlying's annualized continuous dividend yield,
            decimal (e.g. ``0.032`` for 3.2%). Use ``0.0`` for performance
            indices (e.g. DAX) that already reinvest dividends into the
            index level; only used by the classic branch.

    Raises:
        ValueError: if ``ratio <= 0`` / ``fx <= 0``, or
            ``product_type is ProductType.TURBO_CLASSIC`` and ``maturity``
            is ``None`` or before ``as_of`` (CLAUDE.md rule 29: a classic
            turbo's fair value is undefined without a real maturity date --
            never silently treated as open-end).
    """
    if not (ratio > 0):
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    if not (fx > 0):
        raise ValueError(f"fx must be > 0, got {fx!r}")

    if product_type != ProductType.TURBO_CLASSIC:
        return intrinsic_value(spot, financing_level, ratio, direction, fx)

    if maturity is None:
        raise ValueError(
            "turbo_classic requires a maturity date to compute a fair value "
            "(carry cannot be priced without a fixed horizon)"
        )
    days = (maturity - as_of).days
    if days < 0:
        raise ValueError(
            f"maturity {maturity!r} is before as_of {as_of!r}; a matured "
            "classic turbo has no forward-looking fair value"
        )
    t_years = days / _DAY_COUNT_BASIS_YEARS

    discounted_spot = spot * math.exp(-dividend_yield * t_years)
    discounted_strike = financing_level * math.exp(-ref_rate * t_years)
    if direction == Direction.LONG:
        moneyness = discounted_spot - discounted_strike
    else:
        moneyness = discounted_strike - discounted_spot
    return max(moneyness, 0.0) * ratio / fx


def _t_years_array(
    maturity: date, as_of: date | Sequence[date] | npt.NDArray[np.object_]
) -> npt.NDArray[np.float64] | float:
    """Elapsed time to ``maturity`` in years (act/365), from a scalar or an
    array/sequence of ``as_of`` dates -- the array-broadcastable counterpart
    of the scalar ``days = (maturity - as_of).days`` computation inside
    :func:`theoretical_fair_value`.

    A scalar ``date`` returns a plain ``float`` (broadcasts trivially); an
    array/sequence returns an ``NDArray`` of the same shape, one element per
    ``as_of`` entry -- date subtraction itself is not a numpy ufunc, so each
    element is computed with a (cheap: this is sized to the day/horizon axis,
    never the path axis) Python-level ``.days`` lookup, then packed into an
    array for the caller's numpy broadcasting.

    Raises:
        ValueError: if any implied day count is negative (matches the
            scalar function's "matured classic turbo" guard).
    """
    if isinstance(as_of, date):
        days = (maturity - as_of).days
        if days < 0:
            raise ValueError(
                f"maturity {maturity!r} is before as_of {as_of!r}; a matured "
                "classic turbo has no forward-looking fair value"
            )
        return days / _DAY_COUNT_BASIS_YEARS

    as_of_arr = np.asarray(as_of, dtype=object)
    flat = as_of_arr.reshape(-1)
    days_arr = np.array([(maturity - d).days for d in flat], dtype=np.float64)
    if np.any(days_arr < 0):
        bad = flat[int(np.argmax(days_arr < 0))]
        raise ValueError(
            f"maturity {maturity!r} is before as_of {bad!r}; a matured "
            "classic turbo has no forward-looking fair value"
        )
    result: npt.NDArray[np.float64] = (days_arr / _DAY_COUNT_BASIS_YEARS).reshape(as_of_arr.shape)
    return result


def theoretical_fair_value_array(
    *,
    direction: Direction,
    product_type: ProductType,
    spot: npt.NDArray[np.float64],
    financing_level: npt.NDArray[np.float64] | float,
    knockout_barrier: float,
    ratio: float,
    fx: float,
    ref_rate: float,
    financing_spread: float,
    as_of: date | Sequence[date] | npt.NDArray[np.object_],
    maturity: date | None,
    dividend_yield: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Vectorized counterpart of :func:`theoretical_fair_value` -- the exact
    same formula (see module docstring), evaluated elementwise with numpy
    instead of once per scalar call. Added for the ``simulation/payoff.py``
    hot path (Build Contract v2 W7 performance fix): evaluating this
    function once per path in a Python loop dominated payoff-simulation
    runtime (a real scan's product x horizon x path count made that loop
    the bottleneck by roughly an order of magnitude over budget).

    ``spot`` may be any shape. ``financing_level`` and ``as_of`` may each be
    either a scalar (broadcasts to every element of ``spot``) or an array
    broadcastable against ``spot`` under ordinary numpy rules -- e.g. shape
    ``(horizon_days,)`` against a ``(n_paths, horizon_days)`` ``spot``, since
    numpy right-aligns trailing dimensions, letting one call price every
    path on every day (or every requested horizon) in a single vectorized
    expression. An array ``as_of`` must hold ``datetime.date`` objects
    (dtype ``object``); date arithmetic itself is not vectorized by numpy
    (see :func:`_t_years_array`), but that arithmetic is only ever sized to
    the day/horizon axis, never the (large) path axis, so it is not a
    performance concern.

    For ``product_type is not ProductType.TURBO_CLASSIC`` this is simply
    :func:`turboedge.pricing.intrinsic.intrinsic_value` evaluated
    elementwise (``as_of``/``maturity`` unused, matching the scalar
    function); ``as_of`` need not even be a valid broadcastable shape in
    that case.

    Result identity: for every element, this returns the identical value
    :func:`theoretical_fair_value` would return for the corresponding
    scalar inputs, up to floating-point evaluation-order rounding (``numpy``
    ufuncs vs. ``math``) -- see
    ``tests/pricing/test_fair_value.py::test_array_matches_scalar_random_sample``
    (elementwise, ``atol=1e-12``) and
    ``tests/simulation/test_payoff.py::test_vectorized_matches_reference_implementation``
    (full payoff-distribution equivalence) for the cross-checks this
    identity relies on.

    Raises:
        ValueError: same conditions as :func:`theoretical_fair_value`
            (``ratio``/``fx`` non-positive; missing ``maturity``, or any
            resulting time-to-maturity negative, for ``TURBO_CLASSIC``).
    """
    if not (ratio > 0):
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    if not (fx > 0):
        raise ValueError(f"fx must be > 0, got {fx!r}")

    spot_arr = np.asarray(spot, dtype=np.float64)

    if product_type != ProductType.TURBO_CLASSIC:
        fin_arr = np.asarray(financing_level, dtype=np.float64)
        moneyness = spot_arr - fin_arr if direction == Direction.LONG else fin_arr - spot_arr
        result: npt.NDArray[np.float64] = np.maximum(moneyness, 0.0) * ratio / fx
        return result

    if maturity is None:
        raise ValueError(
            "turbo_classic requires a maturity date to compute a fair value "
            "(carry cannot be priced without a fixed horizon)"
        )

    t_years = _t_years_array(maturity, as_of)

    fin_arr = np.asarray(financing_level, dtype=np.float64)
    discounted_spot = spot_arr * np.exp(-dividend_yield * t_years)
    discounted_strike = fin_arr * np.exp(-ref_rate * t_years)
    moneyness_classic = (
        discounted_spot - discounted_strike
        if direction == Direction.LONG
        else discounted_strike - discounted_spot
    )
    classic_result: npt.NDArray[np.float64] = np.maximum(moneyness_classic, 0.0) * ratio / fx
    return classic_result
