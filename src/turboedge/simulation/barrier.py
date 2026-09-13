"""Barrier-touch detection: discrete path scan, Brownian-bridge control value,
and ambiguous-day flagging (Master Spec §11.3, §26).

Discrete daily OHLC simulation can only see a day's open/high/low/close, not
the continuous intraday path -- so a day whose ``[low, high]`` range merely
*straddles* the barrier without the open or the recorded extreme actually
reaching it would be silently missed if we only ever compared closes.
:func:`first_hit_index` therefore checks each day's open (gap-through) and
its far extreme (low for Long, high for Short) against the barrier.

:func:`brownian_bridge_hit_probability` is the analytic first-passage
control value (Master Spec §11.3: "analytische First-Passage-Näherung als
Kontrollwert") used to sanity-check the discrete simulation's touch rate
against a continuous-time approximation, independent of the discretization.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from turboedge.simulation.paths import PathSet
from turboedge.storage.schemas import Direction


def first_hit_index(paths: PathSet, barrier: float, direction: Direction) -> npt.NDArray[np.int64]:
    """Day index (0-based) of the first barrier touch per path, ``-1`` if never touched.

    Long: touched on day ``d`` if ``open[d] <= barrier`` (gap-through at the
    open, already below the barrier before the session starts) or
    ``low[d] <= barrier``. Short is the mirror image with ``>=`` against
    ``open``/``high``.
    """
    if direction == Direction.LONG:
        hit_day = (paths.open <= barrier) | (paths.low <= barrier)
    else:
        hit_day = (paths.open >= barrier) | (paths.high >= barrier)

    any_hit = hit_day.any(axis=1)
    first_idx = np.argmax(hit_day, axis=1)  # first True index; 0 if row is all-False
    return np.where(any_hit, first_idx, -1).astype(np.int64)


def brownian_bridge_hit_probability(
    s0: float, s1: float, barrier: float, sigma_intraday: float, direction: Direction
) -> float:
    """Analytic probability that a Brownian bridge from ``s0`` to ``s1`` touches ``barrier``.

    ``sigma_intraday`` is the (single-day) log-return volatility, and the
    bridge is over log-price. For Long (barrier below the path, touched from
    above) with both endpoints strictly above the barrier::

        P(touch) = exp(-2 * ln(s0/B) * ln(s1/B) / sigma_intraday**2)

    For Short (barrier above the path) with both endpoints strictly below
    the barrier, the identical formula applies (``ln(s0/B)`` and ``ln(s1/B)``
    are both negative, so their product -- and hence the formula -- is
    unchanged; this is the standard symmetric first-passage-probability
    result for a Brownian bridge, not a direction-specific special case).

    If the barrier was already reached at (or passed by) either endpoint
    given ``direction`` -- i.e. the "safe interior" condition
    (``s0 > B and s1 > B`` for Long, ``s0 < B and s1 < B`` for Short) does
    not hold -- the touch is certain and this returns ``1.0``.

    Raises:
        ValueError: if ``s0 <= 0``, ``s1 <= 0``, ``barrier <= 0`` or
            ``sigma_intraday <= 0``.
    """
    if not (s0 > 0 and s1 > 0 and barrier > 0):
        raise ValueError(f"s0, s1 and barrier must be > 0, got {s0!r}, {s1!r}, {barrier!r}")
    if not (sigma_intraday > 0):
        raise ValueError(f"sigma_intraday must be > 0, got {sigma_intraday!r}")

    if direction == Direction.LONG:
        safe = s0 > barrier and s1 > barrier
    else:
        safe = s0 < barrier and s1 < barrier
    if not safe:
        return 1.0

    exponent = -2.0 * math.log(s0 / barrier) * math.log(s1 / barrier) / (sigma_intraday**2)
    return float(math.exp(exponent))


def ambiguous_days(
    paths: PathSet, barrier: float, other_level: float, direction: Direction
) -> npt.NDArray[np.bool_]:
    """``(n_paths, horizon_days)``: day where the ``[low, high]`` range contains
    *both* ``barrier`` and ``other_level``.

    With only daily OHLC available, the order in which two levels inside the
    same day's range were touched is unknown (Master Spec §26 "Ambiguous
    Bars"). Any such day is flagged here so callers resolve it
    conservatively (CLAUDE.md rule 17: assume the less favorable order --
    typically, the barrier was touched first) rather than optimistically.
    ``direction`` is accepted for interface symmetry with
    :func:`first_hit_index`/callers that already carry it; the containment
    check itself does not depend on it (touching an interval is direction-agnostic).
    """
    del direction
    day_low = np.minimum(paths.low, paths.high)
    day_high = np.maximum(paths.low, paths.high)
    contains_barrier = (day_low <= barrier) & (day_high >= barrier)
    contains_other = (day_low <= other_level) & (day_high >= other_level)
    return contains_barrier & contains_other
