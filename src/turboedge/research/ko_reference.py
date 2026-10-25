"""An independent reference for the barrier-touch probability, and the two
things that turned out to be wrong with how it was being compared.

This module does **not** re-implement the knock-out grid.
``simulation/ko_calibration.build_ko_calibration_dataset`` already walks
every (as-of bar, horizon, direction, sigma level) cell and records
``(p_ko_raw, realized_ko)``; ``backtest/ko_calibration.py`` already
walk-forward evaluates calibrators against it (W10,
``docs/measured_results.md`` §3.1). What was missing is a *third* column to
compare both against -- an analytic value that depends on no simulation at
all -- plus a diagnostic for whether the simulated paths actually carry the
volatility the barrier was placed with.

Three things are measured here, documented in
``docs/ko_reference_2026-10-25.md``:

1. **The analytic Brownian reference is not a better model, it is a
   symmetric one.** Over a 128-cell grid on real DAX/S&P/EURUSD/gold bars
   it lands inside the realised rate's 95 % block-bootstrap interval in
   36 % of cells against the simulation's 67 %. Reality is not symmetric:
   the realised long/short touch ratio rises from 1.19 at k=1.5 to 2.01 at
   k=3, is 2.10/2.75 on the two equity indices, 1.09 on EURUSD and 0.78 on
   gold. A driftless Brownian motion cannot express that at any parameter,
   so a closed form is the right tool for checking numerics (see
   :mod:`turboedge.research.first_passage`) and the wrong tool for
   predicting an index.

2. **The simulation over-disperses relative to the volatility the barrier
   is placed with.** The total-horizon sigma realised across simulated
   paths is 1.41x the causal EWMA sigma on average (DAX 1.36, S&P 1.22,
   EURUSD 1.71, gold 1.33), because the bootstrap draws from a 400-day
   lookback while the barrier is placed with a ~16-day half-life EWMA
   estimate. That is a regime mismatch, not a bug: volatility is
   right-skewed, so a long window's dispersion usually exceeds today's.
   It is also sufficient to explain the documented over-prediction (§3) --
   the simulation reads high in 113 of 128 cells.

3. **A percentage offset is not a log distance, and the error is
   asymmetric.** The dataset builder places barriers at ``spot*(1 ± a)``
   with ``a = k*sigma*sqrt(h)``. Since ``exp(-a) > 1-a``, the long barrier
   lands *further* than ``a`` in log space and the short barrier *closer*,
   so one nominal ``k`` means two different log distances. At k=2 that
   alone favours short touches by 1.39x; at k=4, which is in
   ``DEFAULT_SIGMA_LEVELS``, by 10.9x. :func:`linear_offset_bias` computes
   it. No production decision reads those barriers -- live candidates carry
   the issuer's real barrier -- so this distorts a *measurement*, not a
   trade. It does mean the long/short split of §3's figures is partly
   geometry.

Nothing in ``src/turboedge`` outside ``research/`` imports this module, and
a test enforces that. It computes no EV, opens no trial, and changes no
gate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from turboedge.research.first_passage import BarrierSpec, bm_hit_probability
from turboedge.storage.schemas import Direction


def effective_log_k(
    *, spot: float, barrier: float, sigma_daily: float, horizon_days: int, direction: Direction
) -> float:
    """The log-space ``k`` a concrete barrier actually implies.

    The analytic reference ``2*Phi(-k)`` is a statement about a *log*
    distance of ``k*sigma*sqrt(h)``. Comparing it against a barrier that
    was placed as a percentage offset silently compares two different
    distances, which is finding 3 in the module docstring. This converts
    whatever barrier is on the table into the quantity the formula speaks
    about, so the comparison is like-for-like.

    Raises:
        ValueError: on a non-positive spot, barrier, sigma or horizon, or a
            barrier on the wrong side of spot (a long barrier at or above
            spot, a short one at or below) -- which is not a far-away
            barrier but an already-touched one, and has no log distance to
            report.
    """
    if not spot > 0:
        raise ValueError(f"spot must be > 0, got {spot!r}")
    if not barrier > 0:
        raise ValueError(f"barrier must be > 0, got {barrier!r}")
    if not sigma_daily > 0:
        raise ValueError(f"sigma_daily must be > 0, got {sigma_daily!r}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days!r}")
    log_distance = (
        math.log(spot / barrier) if direction is Direction.LONG else math.log(barrier / spot)
    )
    if not log_distance > 0:
        raise ValueError(
            f"barrier {barrier!r} is on the wrong side of spot {spot!r} for {direction} "
            "-- it is already touched, not at a distance"
        )
    return log_distance / (sigma_daily * math.sqrt(horizon_days))


def analytic_touch_probability(
    *,
    spot: float,
    barrier: float,
    sigma_daily: float,
    horizon_days: int,
    direction: Direction,
    log_drift_daily: float = 0.0,
) -> float:
    """Brownian touch probability for a real barrier, with no drift by default.

    Parameterised by the drift of the **log** price, not by a geometric
    ``mu``, and signed so that a positive value moves *away* from the
    barrier on either side. That is deliberate: with ``log_drift_daily=0``
    this is exactly ``2*Phi(-k)`` for the ``k`` that :func:`effective_log_k`
    reports, which is the reference the grid comparison needs. Passing a
    geometric ``mu`` of zero would instead leave the Ito drift
    ``-sigma^2/2`` in place and quietly tilt every long barrier closer --
    about 4 % on the probability at k=2 on daily equity volatility, enough
    to matter and easy to mistake for a market asymmetry. Convert with
    ``log_drift = mu - sigma**2 / 2`` when a geometric drift is what is in
    hand.

    Delegates the first-passage law itself to
    :func:`turboedge.research.first_passage.bm_hit_probability` so there is
    one implementation of it in the repository, and that one carries the
    exactness test.
    """
    spec = BarrierSpec(
        spot=spot,
        barrier=barrier,
        direction=direction,
        mu=0.0,
        sigma=sigma_daily,
        horizon=float(horizon_days),
    )
    signed = log_drift_daily if direction is Direction.LONG else -log_drift_daily
    return bm_hit_probability(spec.log_distance, signed, sigma_daily, float(horizon_days))


@dataclass(frozen=True, slots=True)
class LinearOffsetBias:
    """What one nominal ``k`` means on each side under a percentage offset.

    ``ratio`` is ``p_short / p_long``: the long/short asymmetry that the
    placement alone produces, before any market asymmetry. 1.0 would mean
    the offset is harmless at this ``k``.
    """

    k_nominal: float
    k_effective_long: float
    k_effective_short: float
    p_long: float
    p_short: float

    @property
    def ratio(self) -> float:
        return self.p_short / self.p_long if self.p_long > 0 else math.inf


def linear_offset_bias(k: float, sigma_daily: float, horizon_days: int) -> LinearOffsetBias:
    """Quantify finding 3: a linear offset splits one ``k`` into two.

    Raises:
        ValueError: if ``k``, ``sigma_daily`` or ``horizon_days`` is not
            positive, or if ``k*sigma*sqrt(h) >= 1``, where the long
            barrier would sit at or below zero. That is not an edge case to
            clamp -- a percentage offset of 100 % has no log distance, and
            silently moving the barrier would hide exactly the distortion
            this function exists to report.
    """
    if not k > 0:
        raise ValueError(f"k must be > 0, got {k!r}")
    if not sigma_daily > 0:
        raise ValueError(f"sigma_daily must be > 0, got {sigma_daily!r}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days!r}")
    scale = sigma_daily * math.sqrt(horizon_days)
    a = k * scale
    if a >= 1.0:
        raise ValueError(
            f"a linear offset of {a:.4f} (k={k}, sigma={sigma_daily}, h={horizon_days}) "
            "puts the long barrier at or below zero; it has no log distance"
        )
    k_long = -math.log1p(-a) / scale
    k_short = math.log1p(a) / scale
    return LinearOffsetBias(
        k_nominal=k,
        k_effective_long=k_long,
        k_effective_short=k_short,
        p_long=2.0 * float(norm.cdf(-k_long)),
        p_short=2.0 * float(norm.cdf(-k_short)),
    )


def horizon_sigma_ratio(*, realized_sigma: float, sigma_daily: float, horizon_days: int) -> float:
    """Finding 2 as one number: simulated dispersion / intended dispersion.

    ``realized_sigma`` is :attr:`~turboedge.simulation.paths.PathSet.
    realized_sigma`, the standard deviation across paths of the total
    horizon log return. The barrier was placed with ``sigma_daily *
    sqrt(horizon_days)``, so a ratio above 1 means the paths are more
    dispersed than the barrier distance assumes and P(KO) reads high for
    that reason alone. 1.0 is consistency, not accuracy -- a consistent
    simulation can still have the wrong shape.
    """
    if not sigma_daily > 0:
        raise ValueError(f"sigma_daily must be > 0, got {sigma_daily!r}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days!r}")
    return realized_sigma / (sigma_daily * math.sqrt(horizon_days))


def moving_block_bootstrap_ci(
    hits: npt.NDArray[np.bool_],
    *,
    block: int,
    reps: int = 2000,
    alpha: float = 0.05,
    seed: int = 11,
) -> tuple[float, float]:
    """Confidence interval for a realised touch rate from overlapping windows.

    Consecutive windows share up to ``horizon - 1`` days, so a binomial
    interval on ``n`` windows would claim an independence the sample does
    not have and come out far too narrow. The blocks must therefore be at
    least as long as the overlap (``block >= horizon`` -- the caller owns
    that choice because only the caller knows the horizon in window units).

    Returns ``(nan, nan)`` rather than a misleading interval when the
    sample is shorter than three blocks.

    Raises:
        ValueError: if ``block < 1``, ``reps < 1``, or ``alpha`` is outside
            ``(0, 1)``.
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block!r}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    arr = np.asarray(hits, dtype=np.float64)
    n = arr.size
    if n < block * 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    offsets = np.arange(block)
    # (reps, n_blocks, block) -> flatten to (reps, n_blocks*block), trim to n.
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(reps, -1)[:, :n]
    means = arr[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


@dataclass(frozen=True, slots=True)
class MethodAgreement:
    """How well one method's probabilities match a realised frequency."""

    name: str
    cells: int
    mean_absolute_error: float
    max_absolute_error: float
    cells_inside_ci: int
    cells_over_predicting: int

    @property
    def coverage(self) -> float:
        """Share of cells where the method sits inside the realised CI.

        Compared against the interval's own nominal level (0.95 by
        default), this is the honest summary: a method can have a small
        mean error and still sit outside the interval in most cells.
        """
        return self.cells_inside_ci / self.cells if self.cells else float("nan")


def compare_methods(
    *,
    realised: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    predictions: dict[str, Sequence[float]],
) -> list[MethodAgreement]:
    """Rank candidate P(KO) methods against realised rates over a grid.

    Deliberately reports mean error *and* interval coverage: the two
    disagree here, and which one is used decides the conclusion. Over the
    128-cell grid the simulation wins on both, but at k=2 alone the
    analytic reference wins on mean error -- which is how a two-cell
    comparison produced the wrong answer once already.

    A cell with a ``nan`` interval bound counts toward the error summary
    and not toward coverage; ``cells`` reports the error denominator.

    Raises:
        ValueError: if any sequence length differs, or ``predictions`` is
            empty.
    """
    if not predictions:
        raise ValueError("predictions must not be empty")
    n = len(realised)
    if len(ci_low) != n or len(ci_high) != n:
        raise ValueError(f"ci bounds must have length {n}")
    real = np.asarray(realised, dtype=np.float64)
    lo = np.asarray(ci_low, dtype=np.float64)
    hi = np.asarray(ci_high, dtype=np.float64)
    out: list[MethodAgreement] = []
    for name, preds in predictions.items():
        if len(preds) != n:
            raise ValueError(f"predictions[{name!r}] must have length {n}, got {len(preds)}")
        p = np.asarray(preds, dtype=np.float64)
        err = np.abs(p - real)
        finite_ci = np.isfinite(lo) & np.isfinite(hi)
        inside = int(np.sum(finite_ci & (p >= lo) & (p <= hi)))
        out.append(
            MethodAgreement(
                name=name,
                cells=n,
                mean_absolute_error=float(np.mean(err)),
                max_absolute_error=float(np.max(err)),
                cells_inside_ci=inside,
                cells_over_predicting=int(np.sum(p > real)),
            )
        )
    return sorted(out, key=lambda m: m.mean_absolute_error)


__all__ = [
    "LinearOffsetBias",
    "MethodAgreement",
    "analytic_touch_probability",
    "compare_methods",
    "effective_log_k",
    "horizon_sigma_ratio",
    "linear_offset_bias",
    "moving_block_bootstrap_ci",
]
