"""Implied financing spread inference from consecutive financing levels.

Formula reference: Master Spec §13.3 ("Impliziten Finanzierungsspread
messen") and the Build Contract's "Formeln (verbindlich)" section.

Issuers roll the financing level ``F`` forward each day by the reference
rate plus their own funding spread (actual/360). Given two consecutive
observations we can invert that relationship to recover the spread the
issuer is actually charging, which is materially more informative than any
single ask-price snapshot: it isolates financing cost from everything else
baked into the quoted price (gap premium, trading spread, issuer margin).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from turboedge.storage.schemas import Direction

_DAY_COUNT_BASIS = 360.0


def implied_financing_spread(
    f_prev: float,
    f_next: float,
    days: float,
    ref_rate: float,
    direction: Direction,
) -> float:
    """Recover the issuer's funding spread ``s`` from two financing levels.

    The roll-forward relationship (Long) is::

        F(t+1) = F(t) * (1 + (r + s) * dt / 360)

    which inverts to::

        s = (F(t+1)/F(t) - 1) * 360/dt - r

    Short certificates roll the level in the opposite direction, so the sign
    of the spread contribution flips::

        s = r - (F(t+1)/F(t) - 1) * 360/dt

    Args:
        f_prev: financing level ``F(t)`` (must be > 0).
        f_next: financing level ``F(t+1)``.
        days: calendar days ``dt`` between the two observations (must be > 0).
        ref_rate: reference rate ``r`` (decimal, e.g. 0.03 for 3%).
        direction: LONG or SHORT.

    Raises:
        ValueError: if ``f_prev <= 0`` or ``days <= 0``.
    """
    if not (f_prev > 0):
        raise ValueError(f"f_prev must be > 0, got {f_prev!r}")
    if not (days > 0):
        raise ValueError(f"days must be > 0, got {days!r}")
    annualized_change = (f_next / f_prev - 1.0) * _DAY_COUNT_BASIS / days
    if direction == Direction.LONG:
        return annualized_change - ref_rate
    return ref_rate - annualized_change


@dataclass(frozen=True, slots=True)
class FinancingSpreadObservation:
    """One consecutive-pair financing-spread observation.

    ``adjustment_suspected`` flags pairs whose implied spread magnitude is
    implausibly large for ordinary interest accrual -- almost always caused
    by a dividend or roll adjustment to the financing level rather than a
    genuine change in funding cost. Such observations are excluded from
    :func:`realized_financing_spread`.
    """

    start: datetime
    end: datetime
    days: int
    spread: float
    adjustment_suspected: bool


def financing_spread_history(
    levels: Sequence[tuple[datetime, float]],
    ref_rate: float,
    direction: Direction,
    max_daily_jump: float,
) -> list[FinancingSpreadObservation]:
    """Build a :class:`FinancingSpreadObservation` per consecutive pair of levels.

    ``levels`` need not be pre-sorted; observations are paired in
    chronological order. ``days`` is the calendar-day gap between the two
    timestamps (``(end - start).days``); pairs with a non-positive gap
    (duplicate or out-of-order timestamps) are skipped.

    A pair's implied spread is flagged ``adjustment_suspected`` (dividend or
    roll adjustment, Master Spec §13.3) whenever ``abs(spread) >
    max_daily_jump`` -- ordinary day-to-day funding spread does not jump by
    more than a few percentage points, so an implied spread beyond the
    configured threshold indicates the financing-level move was not (only)
    interest accrual.

    Raises:
        ValueError: if ``max_daily_jump <= 0``.
    """
    if not (max_daily_jump > 0):
        raise ValueError(f"max_daily_jump must be > 0, got {max_daily_jump!r}")
    ordered = sorted(levels, key=lambda item: item[0])
    observations: list[FinancingSpreadObservation] = []
    for (start, f_prev), (end, f_next) in pairwise(ordered):
        days = (end - start).days
        if days <= 0:
            continue
        spread = implied_financing_spread(f_prev, f_next, days, ref_rate, direction)
        adjustment_suspected = abs(spread) > max_daily_jump
        observations.append(
            FinancingSpreadObservation(
                start=start,
                end=end,
                days=days,
                spread=spread,
                adjustment_suspected=adjustment_suspected,
            )
        )
    return observations


def realized_financing_spread(
    observations: Sequence[FinancingSpreadObservation],
) -> float | None:
    """Robust (median) realized financing spread over non-flagged observations.

    Returns ``None`` if there are no clean (non-``adjustment_suspected``)
    observations to draw from -- callers must fall back to
    ``configs/risk.yaml``'s ``default_financing_spread`` in that case (Master
    Spec CLAUDE.md rule 13).
    """
    clean = [obs.spread for obs in observations if not obs.adjustment_suspected]
    if not clean:
        return None
    return statistics.median(clean)


def financing_cost_over_horizon(
    financing_level: float,
    spread: float,
    ref_rate: float,
    trading_days: float,
    ratio: float,
    direction: Direction,
    fx: float = 1.0,
) -> float:
    """Financing cost accrued over a holding horizon, in product currency.

    Calendar days are approximated from trading days as ``trading_days *
    7/5`` (Build Contract, "Formeln (verbindlich)").

    Long:  ``F * (r + s) * days / 360 * ratio / fx``
    Short: ``F * (s - r) * days / 360 * ratio / fx`` (can be negative, i.e.
    a carry *credit* to the holder, when the issuer's spread is smaller than
    the reference rate).

    Raises:
        ValueError: if ``ratio <= 0``, ``fx <= 0`` or ``trading_days < 0``.
    """
    if not (ratio > 0):
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    if not (fx > 0):
        raise ValueError(f"fx must be > 0, got {fx!r}")
    if trading_days < 0:
        raise ValueError(f"trading_days must be >= 0, got {trading_days!r}")
    calendar_days = trading_days * 7.0 / 5.0
    rate_term = ref_rate + spread if direction == Direction.LONG else spread - ref_rate
    return financing_level * rate_term * calendar_days / _DAY_COUNT_BASIS * ratio / fx
