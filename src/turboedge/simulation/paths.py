"""Simulated underlying price paths (Master Spec §11).

Public entry point: :func:`simulate_paths`. Combines the empirical block
bootstrap, regime-conditioned bootstrap, vol-standardized bootstrap (all
three in ``bootstrap.py``) and a parametric Monte Carlo method into a single
vectorized numpy pipeline that produces a full daily OHLC :class:`PathSet`
(not just terminal prices) -- Turbo/knockout risk is path-dependent (Master
Spec §11 preamble), so every simulated day needs its own open/high/low/close.

Calendar (documented, CLAUDE.md rule 33): the simulation calendar is
Monday-Friday trading days starting the first calendar day strictly after
``start``'s date (``start`` is treated like a "close" reference timestamp,
consistent with ``spot0`` being the entry/as-of price; the first simulated
day is the next trading session). German public holidays are **not**
modeled -- a holiday is treated as an ordinary weekday, which understates
gap risk around holiday closures. This mirrors the ``trading_days * 7/5``
weekend approximation already used throughout ``pricing/*.py`` and is
flagged here as a known simplification, not silently assumed away.

Default ``method`` and its validated bias (Build Contract v2 W5 follow-ups,
``scratchpad/w5_simulation_validation.md`` -- read that file for full tables):
the default is **``vol_scaled_bootstrap``**, changed from the original
``block_bootstrap`` default based on measured calibration, not by assumption.
On the same 558 ^GDAXI start dates (2015-2025), k in {1.5,2,3,4} sigma, h in
{5,10} days, both directions:

- ``block_bootstrap`` and ``monte_carlo`` both carry a large, quantified
  conservative (over-predicting) P(KO) bias -- mean |simulated - realized|
  of 0.055 and 0.113 respectively at the trading-relevant k=1.5-2 sigma
  range (``monte_carlo`` up to ~2-3x worse than ``block_bootstrap``,
  especially Short). A dedicated ablation **ruled out** i.i.d. weekend
  sampling against blocked weekday sampling as the driver: making weekday
  sampling i.i.d. too left P(KO) essentially unchanged.
- ``regime_bootstrap``'s vol-regime bucket was found to be **dormant by
  construction** at the original ``min_bucket_days=250``: with
  ``lookback_days=750`` the weekday tercile bucket is always ~195-199 days
  (0% activation across all 558 dates), so it silently degenerated to
  ``block_bootstrap`` on every date tested. Fixed by lowering
  ``RegimeBootstrapConfig.min_bucket_days`` to ``150`` (see
  ``bootstrap.py`` module docstring for why the lookback was *not* raised
  instead). Once fixed, its calibration improved dramatically: mean
  |diff| 0.022 at k=1.5-2.
- ``vol_scaled_bootstrap`` (new: standardize each historical day by its own
  trailing, no-look-ahead EWMA volatility, block-bootstrap the standardized
  shocks, rescale by the *current* EWMA volatility -- conditions on regime
  without any bucket-size threshold at all) matched or slightly beat the
  fixed ``regime_bootstrap``: mean |diff| 0.019 at k=1.5-2, 0.012 across all
  16 (h, k, direction) combinations tested -- the best of the four methods,
  hence the new default.

**Residual bias, stated plainly even for the new default:** ``vol_scaled_bootstrap``
still over-predicts P(KO) on average at the k=1.5-2 sigma range that matters
most for real turbo barrier placements -- mean signed diff (realized -
simulated) of -0.019 there, i.e. still conservative, just ~2.8x smaller than
the old ``block_bootstrap`` default (-0.055) and ~5.8x smaller than
``monte_carlo`` (-0.113). **This residual bias is directional and known: it
makes the system UNDER-state net EV and OVER-state knockout risk, which
biases the (not-yet-built) Wave 2 ACTIONABLE gate toward proposing FEWER
trades, never toward proposing bad ones on false confidence.** Callers of
P(KO) from this module (LCB/EV gating, Wave 2 ranking) should still treat it
as a mildly conservative, not perfectly calibrated, estimate -- the residual
is now small enough to be a tuning problem for Wave 2, not a modeling
blocker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

from turboedge.features.product import ewma_volatility
from turboedge.simulation.bootstrap import (
    DailyDraws,
    RegimeBootstrapConfig,
    block_bootstrap_daily,
    regime_bootstrap_daily,
    vol_scaled_bootstrap_daily,
)
from turboedge.simulation.overnight import DailyComponents, daily_components_from_bars
from turboedge.storage.schemas import UnderlyingBar

SimulationMethod = Literal[
    "block_bootstrap", "regime_bootstrap", "monte_carlo", "vol_scaled_bootstrap"
]

_DEFAULT_BLOCK_SIZE = 5
_DEFAULT_LOOKBACK_DAYS = 750
_EWMA_LAMBDA = 0.94
_JUMP_SIGMA_MULTIPLE = 4.0
# Student-t degrees of freedom bounds (numerical robustness only): below 4
# the theoretical kurtosis is undefined/infinite, above ~60 a t-distribution
# is visually indistinguishable from Normal.
_MIN_STUDENT_T_DF = 4.5
_MAX_STUDENT_T_DF = 60.0
_MIN_EXCESS_KURTOSIS_FOR_STUDENT_T = 0.05


@dataclass(frozen=True, slots=True)
class PathSet:
    """A batch of simulated daily OHLC paths for one underlying.

    ``open``/``high``/``low``/``close`` all have shape ``(n_paths,
    horizon_days)``; ``open[:, d]`` is the price at the open of simulated
    trading day ``d`` (index 0 = the first trading day after ``start``),
    already reflecting that day's overnight/weekend gap from the previous
    close (or from ``spot0`` for day 0). ``weekend_before[d]`` is True iff
    the gap immediately before day ``d`` spans a weekend/holiday (see
    ``simulate_paths`` calendar docs).
    """

    underlying_id: str
    spot0: float
    start: datetime
    open: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    close: npt.NDArray[np.float64]
    weekend_before: npt.NDArray[np.bool_]
    method: str


def _next_business_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:  # Saturday=5, Sunday=6
        nxt += timedelta(days=1)
    return nxt


def _simulation_calendar(start: date, horizon_days: int) -> npt.NDArray[np.bool_]:
    """``(horizon_days,)`` weekend-gap flags for the Mon-Fri calendar starting after ``start``."""
    dates: list[date] = []
    cur = start
    for _ in range(horizon_days):
        cur = _next_business_day(cur)
        dates.append(cur)
    weekend_before = np.empty(horizon_days, dtype=np.bool_)
    weekend_before[0] = (dates[0] - start).days > 1
    for i in range(1, horizon_days):
        weekend_before[i] = (dates[i] - dates[i - 1]).days > 1
    return weekend_before


def _lookback_components(
    bars: Sequence[UnderlyingBar], start: datetime, lookback_days: int
) -> DailyComponents:
    history = [b for b in bars if b.ts < start]
    history.sort(key=lambda b: b.ts)
    if lookback_days > 0:
        history = history[-(lookback_days + 1) :]  # +1: first bar only seeds the first gap
    if len(history) < 2:
        raise ValueError(
            f"need at least 2 historical bars strictly before start={start!r} to bootstrap "
            f"from, got {len(history)}"
        )
    return daily_components_from_bars(history)


def _monte_carlo_draws(
    components: DailyComponents,
    weekend_before: npt.NDArray[np.bool_],
    n_paths: int,
    rng: np.random.Generator,
) -> DailyDraws:
    """Monte Carlo method (Master Spec §11.1.C): drift + realized volatility +
    jump component + empirical overnight distribution.

    The overnight gap and the intraday high/low *shape* are drawn from the
    empirical distribution (i.i.d., i.e. ``block_bootstrap_daily`` with
    ``block_size=1`` -- see ``bootstrap.py``); the intraday close-to-open
    return is instead replaced by a parametric diffusion (Normal, or
    Student-t with degrees of freedom moment-matched to the historical
    excess kurtosis of the total daily log return) scaled by the most recent
    EWMA daily volatility, plus an additive jump term drawn with the
    historical empirical frequency of ``|total daily log return| > 4 *
    (contemporaneous EWMA volatility)`` days, sized from that same jump
    sample. ``hi``/``lo`` are then clamped against the new intraday value so
    ``High >= max(Open,Close)`` / ``Low <= min(Open,Close)`` still holds.
    """
    empirical = block_bootstrap_daily(components, weekend_before, n_paths, rng, block_size=1)

    total_return = components.gap + components.intraday
    if total_return.size < 2:
        raise ValueError("need at least 2 historical daily components for monte_carlo method")

    ewma_sigma = ewma_volatility(total_return, _EWMA_LAMBDA)
    current_sigma = float(ewma_sigma[-1])
    if not (current_sigma > 0):
        current_sigma = float(np.std(total_return)) or 1e-6

    excess_kurtosis = float(scipy_stats.kurtosis(total_return, fisher=True, bias=False))
    use_student_t = excess_kurtosis > _MIN_EXCESS_KURTOSIS_FOR_STUDENT_T
    if use_student_t:
        df = float(np.clip(6.0 / excess_kurtosis + 4.0, _MIN_STUDENT_T_DF, _MAX_STUDENT_T_DF))
    else:
        df = float("inf")

    horizon_days = weekend_before.size
    if use_student_t:
        z = rng.standard_t(df, size=(n_paths, horizon_days))
        z *= np.sqrt((df - 2.0) / df)
    else:
        z = rng.standard_normal(size=(n_paths, horizon_days))
    diffusive = z * current_sigma

    jump_mask = np.abs(total_return) > _JUMP_SIGMA_MULTIPLE * ewma_sigma
    jump_sample = total_return[jump_mask]
    p_jump = float(jump_mask.mean())
    jump_term = np.zeros((n_paths, horizon_days), dtype=np.float64)
    if p_jump > 0.0 and jump_sample.size > 0:
        occurs = rng.random(size=(n_paths, horizon_days)) < p_jump
        n_occur = int(occurs.sum())
        if n_occur > 0:
            draws = rng.choice(jump_sample, size=n_occur, replace=True)
            jump_term[occurs] = draws

    new_intraday = diffusive + jump_term
    hi = np.maximum(empirical.hi, np.maximum(new_intraday, 0.0))
    lo = np.minimum(empirical.lo, np.minimum(new_intraday, 0.0))
    return DailyDraws(gap=empirical.gap, intraday=new_intraday, hi=hi, lo=lo)


def _apply_drift_tilt(draws: DailyDraws, drift_log_return: float | None) -> DailyDraws:
    """Additive per-day recentering of ``intraday``/``hi``/``lo`` so that the
    realized mean total horizon log return matches ``drift_log_return``
    (``None`` => 0, i.e. the unconditional empirical drift of the resampled
    paths is removed -- see ``simulate_paths`` docstring). ``gap`` is left
    untouched: it represents genuine overnight/weekend jump risk calibrated
    from historical data, not a trend view, and must not be corrupted by a
    forecast's directional tilt (CLAUDE.md rule 16).

    The shift is added identically to ``intraday``, ``hi`` and ``lo`` so the
    *shape* of each day's range around its close is preserved exactly
    (dispersion unaffected); a cheap clamp afterwards guards the
    ``High>=max(Open,Close)`` / ``Low<=min(Open,Close)`` invariants against
    the (for realistic drift magnitudes, negligible) edge case where the
    shift pushes them out of order.
    """
    horizon_days = draws.gap.shape[1]
    target = 0.0 if drift_log_return is None else drift_log_return
    current_mean_total = float((draws.gap + draws.intraday).sum(axis=1).mean())
    shift_per_day = (target - current_mean_total) / horizon_days

    intraday = draws.intraday + shift_per_day
    hi = draws.hi + shift_per_day
    lo = draws.lo + shift_per_day
    hi = np.maximum(hi, np.maximum(intraday, 0.0))
    lo = np.minimum(lo, np.minimum(intraday, 0.0))
    return DailyDraws(gap=draws.gap, intraday=intraday, hi=hi, lo=lo)


def _draws_to_pathset(
    draws: DailyDraws,
    underlying_id: str,
    spot0: float,
    start: datetime,
    weekend_before: npt.NDArray[np.bool_],
    method: str,
) -> PathSet:
    day_total = draws.gap + draws.intraday
    log_close_cum = np.cumsum(day_total, axis=1)
    close = spot0 * np.exp(log_close_cum)

    n_paths = draws.gap.shape[0]
    prev_close = np.concatenate(
        [np.full((n_paths, 1), spot0, dtype=np.float64), close[:, :-1]], axis=1
    )
    open_ = prev_close * np.exp(draws.gap)
    high = open_ * np.exp(draws.hi)
    low = open_ * np.exp(draws.lo)

    return PathSet(
        underlying_id=underlying_id,
        spot0=spot0,
        start=start,
        open=open_,
        high=high,
        low=low,
        close=close,
        weekend_before=weekend_before,
        method=method,
    )


def simulate_paths(
    bars: Sequence[UnderlyingBar],
    *,
    spot0: float,
    start: datetime,
    horizon_days: int,
    n_paths: int,
    rng: np.random.Generator,
    drift_log_return: float | None = None,
    method: SimulationMethod = "vol_scaled_bootstrap",
    block_size: int = _DEFAULT_BLOCK_SIZE,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> PathSet:
    """Simulate ``n_paths`` daily OHLC paths of length ``horizon_days`` (Master Spec §11).

    ``bars`` must contain daily OHLC history strictly before ``start``
    (CLAUDE.md rule 4: no look-ahead -- bars at or after ``start`` are
    silently excluded from the *bootstrap sample*, not used as a shortcut).
    Only the most recent ``lookback_days`` (+1, to seed the first gap) such
    bars are used.

    ``drift_log_return`` is the target mean **total** log return over the
    whole horizon (``None`` => the unconditional empirical drift of the
    resampled distribution is removed, i.e. zero drift) -- see
    :func:`_apply_drift_tilt`.

    Raises:
        ValueError: if ``spot0 <= 0``, ``horizon_days <= 0``, ``n_paths <=
            0``, fewer than 2 usable historical bars are available, or (for
            ``block_bootstrap``/``regime_bootstrap``) a required weekday or
            weekend historical subsequence is empty.
    """
    if not (spot0 > 0):
        raise ValueError(f"spot0 must be > 0, got {spot0!r}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days!r}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths!r}")

    components = _lookback_components(bars, start, lookback_days)
    weekend_before = _simulation_calendar(start.date(), horizon_days)

    if method == "block_bootstrap":
        draws = block_bootstrap_daily(components, weekend_before, n_paths, rng, block_size)
    elif method == "regime_bootstrap":
        draws = regime_bootstrap_daily(
            components, weekend_before, n_paths, rng, block_size, RegimeBootstrapConfig()
        )
    elif method == "monte_carlo":
        draws = _monte_carlo_draws(components, weekend_before, n_paths, rng)
    elif method == "vol_scaled_bootstrap":
        draws = vol_scaled_bootstrap_daily(components, weekend_before, n_paths, rng, block_size)
    else:
        raise ValueError(f"unknown method {method!r}")

    draws = _apply_drift_tilt(draws, drift_log_return)

    underlying_id = bars[0].underlying_id if bars else "UNKNOWN"
    return _draws_to_pathset(draws, underlying_id, spot0, start, weekend_before, method)


__all__ = ["PathSet", "SimulationMethod", "simulate_paths"]
