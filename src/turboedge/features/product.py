"""Product-level quote and barrier-distance features.

Formula reference: Master Spec §12 ("MAE-basierte Barrierelogik") and §17
("Optimale Kombination aus Produkt und Horizont", leverage buckets).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from turboedge.storage.schemas import Direction

_LN2 = np.log(2.0)

# Master Spec §17 leverage buckets, plus a "<2" catch-all for products below
# the analysis band (Build Contract's DEINE DATEIEN description explicitly
# lists "<2" alongside the §17 buckets).
_LEVERAGE_BUCKET_EDGES: tuple[tuple[float, str], ...] = (
    (2.0, "<2"),
    (3.0, "2-3"),
    (4.0, "3-4"),
    (5.0, "4-5"),
    (6.0, "5-6"),
    (8.0, "6-8"),
    (10.0, "8-10"),
    (15.0, "10-15"),
)
_LEVERAGE_BUCKET_OVERFLOW = ">15"


def spread_pct(bid: float, ask: float) -> float:
    """Quoted spread relative to the mid price: ``(ask - bid) / mid``.

    Raises:
        ValueError: if ``bid > ask`` or the mid price is not > 0.
    """
    if bid > ask:
        raise ValueError(f"bid ({bid!r}) must be <= ask ({ask!r})")
    mid = (bid + ask) / 2.0
    if not (mid > 0):
        raise ValueError(f"mid price must be > 0, got {mid!r}")
    return (ask - bid) / mid


def quote_age_seconds(quote_ts: datetime, now: datetime) -> float:
    """Seconds elapsed between a quote's timestamp and ``now`` (can be negative)."""
    return (now - quote_ts).total_seconds()


def freshness_score(age_s: float | None, half_life_s: float = 60.0) -> float:
    """Exponential-decay freshness score in ``(0, 1]``; ``None`` age scores 0.

    ``score = exp(-ln(2) * age_s / half_life_s)``, i.e. the score halves
    every ``half_life_s`` seconds of age. A negative age (clock skew) is
    clamped to zero before decaying, so freshness never exceeds 1.

    Raises:
        ValueError: if ``half_life_s <= 0``.
    """
    if not (half_life_s > 0):
        raise ValueError(f"half_life_s must be > 0, got {half_life_s!r}")
    if age_s is None:
        return 0.0
    clamped_age = max(age_s, 0.0)
    return float(np.exp(-_LN2 * clamped_age / half_life_s))


def leverage_bucket(lev: float) -> str:
    """Map a leverage value to its analysis bucket (Master Spec §17)."""
    for upper_bound, label in _LEVERAGE_BUCKET_EDGES:
        if lev < upper_bound:
            return label
    return _LEVERAGE_BUCKET_OVERFLOW


@dataclass(frozen=True, slots=True)
class BarrierDistance:
    """Distance from spot to the knockout barrier, in three units."""

    pct: float
    abs: float
    sigma: float


def distance_to_barrier(
    spot: float,
    barrier: float,
    direction: Direction,
    daily_vol: float,
    horizon_days: float,
) -> BarrierDistance:
    """Distance from ``spot`` to ``barrier``, in absolute, percent and sigma units.

    ``abs``/``pct`` are signed so that a positive value means the barrier has
    not yet been touched (Long: spot above barrier; Short: spot below
    barrier). ``sigma`` normalizes the percent distance by the underlying's
    expected move over ``horizon_days`` trading days at ``daily_vol``:
    ``sigma = pct / (daily_vol * sqrt(horizon_days))``.

    Raises:
        ValueError: if ``spot <= 0``, ``daily_vol <= 0`` or ``horizon_days <= 0``.
    """
    if not (spot > 0):
        raise ValueError(f"spot must be > 0, got {spot!r}")
    if not (daily_vol > 0):
        raise ValueError(f"daily_vol must be > 0, got {daily_vol!r}")
    if not (horizon_days > 0):
        raise ValueError(f"horizon_days must be > 0, got {horizon_days!r}")

    abs_distance = spot - barrier if direction == Direction.LONG else barrier - spot
    pct_distance = abs_distance / spot
    sigma_distance = pct_distance / (daily_vol * float(np.sqrt(horizon_days)))
    return BarrierDistance(pct=pct_distance, abs=abs_distance, sigma=sigma_distance)


def ewma_volatility(
    log_returns: npt.NDArray[np.float64], lam: float = 0.94
) -> npt.NDArray[np.float64]:
    """Exponentially-weighted daily volatility of a log-return series.

    ``var[0] = log_returns[0]**2``, then ``var[t] = lam*var[t-1] +
    (1-lam)*log_returns[t]**2`` -- each ``var[t]`` (and hence ``sigma[t] =
    sqrt(var[t])``) only uses returns up to and including index ``t``, so
    using it to normalize a signal computed "as of" day ``t`` introduces no
    look-ahead bias (CLAUDE.md rule 4).

    Raises:
        ValueError: if ``log_returns`` is empty or not 1-dimensional, or if
            ``lam`` is not in ``(0, 1)``.
    """
    arr = np.asarray(log_returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"log_returns must be 1-dimensional, got shape {arr.shape!r}")
    if arr.size == 0:
        raise ValueError("log_returns must not be empty")
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam!r}")

    variance = np.empty_like(arr)
    variance[0] = arr[0] ** 2
    for t in range(1, arr.size):
        variance[t] = lam * variance[t - 1] + (1.0 - lam) * arr[t] ** 2
    return np.sqrt(variance)
