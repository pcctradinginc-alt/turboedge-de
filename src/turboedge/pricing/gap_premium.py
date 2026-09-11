"""Overnight/weekend gap distributions and the fair gap premium.

Formula reference: Master Spec §11.2 ("Overnight und Weekend getrennt"),
§13.4 ("Gap-Prämie"), and the Build Contract's "Formeln (verbindlich)"
section.

Not every excess of the ask price over intrinsic value is issuer margin
(CLAUDE.md rule 14): a knock-out product can gap through its financing
level overnight or over a weekend, which is a real risk the issuer prices
in. This module estimates that cost empirically from the underlying's own
historical overnight jumps rather than assuming it away.

For a Mini Future (``ProductType.MINI_FUTURE``, ``barrier != financing
level``) the issuer's loss only materializes once the underlying gaps
*through the financing level* ``F`` itself -- the knockout barrier is a
stop-loss buffer the issuer uses to close out the position, not the level at
which the issuer stops earning/losing. The formulas below are therefore
always applied with ``financing_level``, never with ``knockout_barrier``,
for every product type; this is the "same formula with F, documented" case
called out in the Build Contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from turboedge.storage.schemas import Direction, UnderlyingBar

_WEEKEND_GAP_CALENDAR_DAYS = 1


@dataclass(frozen=True, slots=True)
class GapDistribution:
    """Empirical log-gap samples, split by overnight type.

    Both arrays hold ``g = ln(Open_t / Close_{t-1})`` log-gaps: ``weeknight_gaps``
    for ordinary one-calendar-day overnights, ``weekend_gaps`` for gaps that
    span more than one calendar day (weekend or holiday closures).
    """

    weeknight_gaps: npt.NDArray[np.float64]
    weekend_gaps: npt.NDArray[np.float64]


def gap_distribution_from_bars(bars: Sequence[UnderlyingBar]) -> GapDistribution:
    """Build a :class:`GapDistribution` from consecutive daily OHLC bars.

    Bars are sorted by ``ts`` first. For each consecutive pair, the overnight
    log-gap ``g = ln(Open_t / Close_{t-1})`` is computed and bucketed into
    ``weeknight_gaps`` (calendar-day gap == 1) or ``weekend_gaps`` (calendar-day
    gap > 1, i.e. a weekend or holiday).
    """
    ordered = sorted(bars, key=lambda bar: bar.ts)
    weeknight: list[float] = []
    weekend: list[float] = []
    for prev_bar, bar in pairwise(ordered):
        gap_days = (bar.ts.date() - prev_bar.ts.date()).days
        if gap_days <= 0:
            continue
        g = float(np.log(bar.open / prev_bar.close))
        if gap_days > _WEEKEND_GAP_CALENDAR_DAYS:
            weekend.append(g)
        else:
            weeknight.append(g)
    return GapDistribution(
        weeknight_gaps=np.asarray(weeknight, dtype=np.float64),
        weekend_gaps=np.asarray(weekend, dtype=np.float64),
    )


def expected_gap_loss_per_night(
    spot: float,
    financing_level: float,
    ratio: float,
    direction: Direction,
    gaps: npt.NDArray[np.float64],
    fx: float = 1.0,
) -> float:
    """Expected issuer loss from one overnight gap, in product currency.

    Long:  ``E[max(F - S * e^g, 0)] * ratio / fx``
    Short: ``E[max(S * e^g - F, 0)] * ratio / fx``

    The expectation is the empirical mean over ``gaps`` (a sample of
    observed log-gaps, typically ``GapDistribution.weeknight_gaps`` or
    ``.weekend_gaps``).

    Raises:
        ValueError: if ``gaps`` is empty, or ``ratio <= 0`` / ``fx <= 0``.
    """
    if gaps.size == 0:
        raise ValueError("gaps sample is empty; cannot estimate an expected gap loss")
    if not (ratio > 0):
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    if not (fx > 0):
        raise ValueError(f"fx must be > 0, got {fx!r}")
    projected = spot * np.exp(gaps)
    if direction == Direction.LONG:
        losses = np.maximum(financing_level - projected, 0.0)
    else:
        losses = np.maximum(projected - financing_level, 0.0)
    return float(np.mean(losses)) * ratio / fx


def fair_gap_premium(
    spot: float,
    financing_level: float,
    ratio: float,
    direction: Direction,
    dist: GapDistribution,
    next_night_is_weekend: bool,
    fx: float = 1.0,
) -> float:
    """Fair gap premium for a single upcoming overnight (as baked into the ask).

    Uses ``dist.weekend_gaps`` when ``next_night_is_weekend`` is True, else
    ``dist.weeknight_gaps``. See :func:`expected_gap_loss_per_night` for the
    underlying formula.
    """
    gaps = dist.weekend_gaps if next_night_is_weekend else dist.weeknight_gaps
    return expected_gap_loss_per_night(spot, financing_level, ratio, direction, gaps, fx)


def gap_premium_over_horizon(
    spot: float,
    financing_level: float,
    ratio: float,
    direction: Direction,
    dist: GapDistribution,
    trading_days: int,
    fx: float = 1.0,
) -> float:
    """Total fair gap premium over a multi-day holding horizon.

    The number of overnights held is approximated as ``trading_days``, split
    into ``trading_days // 5`` weekend nights (drawn from
    ``dist.weekend_gaps``) and the remainder as ordinary weeknights (drawn
    from ``dist.weeknight_gaps``); this mirrors the ``calendar_days =
    trading_days * 7/5`` approximation used elsewhere in the pricing engine
    (one weekend per five trading days).

    Raises:
        ValueError: if ``trading_days < 0``, or the required gap sample
            (weeknight and/or weekend) needed for a non-zero number of
            nights of that type is empty.
    """
    if trading_days < 0:
        raise ValueError(f"trading_days must be >= 0, got {trading_days!r}")
    weekend_nights = trading_days // 5
    weeknights = trading_days - weekend_nights
    total = 0.0
    if weeknights > 0:
        total += weeknights * expected_gap_loss_per_night(
            spot, financing_level, ratio, direction, dist.weeknight_gaps, fx
        )
    if weekend_nights > 0:
        total += weekend_nights * expected_gap_loss_per_night(
            spot, financing_level, ratio, direction, dist.weekend_gaps, fx
        )
    return total
