"""Empirical dataset for out-of-sample calibration of the path-simulation
knock-out probability P(KO) (Workstream W10; ``docs/measured_results.md``
§3 -- Workstream W5's original P(KO)-calibration study is no longer in this
repository, so this rebuilds the empirical dataset from scratch rather than
reusing lost code).

Design (CLAUDE.md rules 4-8, 15-17, 33):

- For every historical "as of" bar (a candidate ``prediction_time``), every
  supported horizon, direction and standardized barrier distance, this
  module produces one :class:`KoCalibrationObservation` pairing:

  - ``p_ko_raw``: the fraction of simulated paths (``simulation/paths.py::
    simulate_paths`` + ``simulation/barrier.py::first_hit_index``) that
    touch the barrier within the horizon, using *only* information
    available at ``prediction_time`` (bars with ``available_at <=
    prediction_time``, rule 5) -- the same computation the scan pipeline
    would perform.
  - ``realized_ko``: whether the barrier was *actually* touched afterwards,
    read directly off the real subsequent bars (their ``high``/``low``,
    never ``close`` -- rule 15, KO is path-dependent). This is the ground
    truth label the calibrator is fit against; it is never used as an
    input to ``p_ko_raw`` itself.

- ``sigma`` (the EWMA daily volatility used to standardize the barrier
  distance) and the volatility ``regime_bucket`` are both computed *only*
  from bars with ``available_at <= prediction_time``
  (``features/volatility.py::ewma_volatility``) -- no look-ahead.

- Barrier levels use the same standardized-distance convention as
  ``features/product.py::distance_to_barrier``
  (``pct_distance = k * daily_vol * sqrt(horizon_days)``), so a
  ``sigma_k=1.5`` observation here means exactly what
  ``distance_to_barrier_sigma=1.5`` means everywhere else in the codebase.

- One :class:`~turboedge.simulation.paths.PathSet` is simulated per "as of"
  bar, at the *largest* configured horizon; shorter horizons reuse the same
  simulated paths (sliced to their own first ``h`` days) rather than
  resimulating, since resampling is the expensive step and the barrier
  level (not the path) is what changes per horizon/sigma/direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from turboedge.features.volatility import ewma_volatility
from turboedge.simulation.paths import SimulationMethod, simulate_paths
from turboedge.storage.schemas import Direction, UnderlyingBar

#: Master Spec / docs/measured_results.md §3 grid.
DEFAULT_HORIZONS: tuple[int, ...] = (3, 5, 7, 10, 14)
DEFAULT_SIGMA_LEVELS: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)

# Minimum number of historical bars (strictly before the candidate "as of"
# bar) required before that bar is used as a prediction_time at all: enough
# for features/volatility.py's EWMA warmup (20 return-indexed observations)
# plus a further cushion so the regime-bucket tercile split (computed from
# the trailing EWMA-sigma series up to that point, see
# ``_regime_bucket_series``) has a non-degenerate sample. Deliberately much
# smaller than simulate_paths' own default 750-day lookback -- the
# bootstrap itself tolerates a shorter history (it just draws from what is
# available), it is the *feature* (EWMA vol, regime) that needs the warmup.
_MIN_HISTORY_BARS = 130
_REGIME_BUCKETS: tuple[str, str, str] = ("low_vol", "mid_vol", "high_vol")


@dataclass(frozen=True, slots=True)
class KoCalibrationObservation:
    """One (as-of bar, horizon, direction, sigma) knock-out observation.

    ``t0_index`` is this observation's "as of" bar's position in the
    underlying's own chronologically sorted bar list -- the sample-axis
    index :class:`~turboedge.backtest.purged_cv.PurgedWalkForwardSplit`
    purges/embargoes on (each underlying has its own independent bar axis;
    see ``backtest/ko_calibration.py`` for why walk-forward is run per
    underlying rather than pooling axes).
    """

    underlying_id: str
    t0_index: int
    prediction_time: datetime
    horizon_days: int
    direction: Direction
    sigma_k: float
    regime_bucket: str
    p_ko_raw: float
    realized_ko: bool


def _daily_sigma_series(closes: np.ndarray) -> np.ndarray:
    """EWMA daily volatility aligned to ``closes`` (index ``i`` uses only
    ``closes[:i+1]`` -- causal, see ``features/volatility.py``)."""
    return ewma_volatility(closes)


def _regime_bucket(sigma_history: np.ndarray, current_sigma: float) -> str:
    """Tercile bucket of ``current_sigma`` within ``sigma_history`` (all
    valid, non-NaN EWMA-sigma values strictly before the current bar) --
    ``"low_vol"``/``"mid_vol"``/``"high_vol"``. Causal: ``sigma_history``
    only ever contains values already computed from bars at or before the
    current "as of" bar (see :func:`build_ko_calibration_dataset`).
    """
    valid = sigma_history[~np.isnan(sigma_history)]
    if valid.size < 20:
        return "unknown"
    lo, hi = np.percentile(valid, [33.33, 66.67])
    if current_sigma <= lo:
        return _REGIME_BUCKETS[0]
    if current_sigma >= hi:
        return _REGIME_BUCKETS[2]
    return _REGIME_BUCKETS[1]


def build_ko_calibration_dataset(
    bars: Sequence[UnderlyingBar],
    underlying_id: str,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    sigma_levels: Sequence[float] = DEFAULT_SIGMA_LEVELS,
    n_paths: int,
    method: SimulationMethod,
    block_size: int,
    lookback_days: int,
    rng: np.random.Generator,
    step_days: int = 1,
) -> list[KoCalibrationObservation]:
    """Build the empirical KO-calibration dataset for one underlying.

    Iterates over historical "as of" bars (every ``step_days``-th bar with
    enough preceding history, up to the last bar that still has
    ``max(horizons)`` realized trading days ahead of it), simulating one
    :class:`~turboedge.simulation.paths.PathSet` per "as of" bar (at
    ``horizon_days = max(horizons)``) and deriving every
    ``(horizon, direction, sigma_k)`` observation from it.

    ``step_days`` trades dataset density for runtime -- simulating paths
    for every single historical day is the dominant cost. This is a
    documented, applied-before-any-result-is-seen stride (never a post-hoc
    choice of which days to include), not cherry-picking: every ``step_days``-
    th calendar bar is used, unconditionally.

    Raises:
        ValueError: if ``bars`` has too few usable rows for even one "as of"
            bar (fewer than ``_MIN_HISTORY_BARS + max(horizons) + 1``).
    """
    if not horizons:
        raise ValueError("horizons must not be empty")
    if not sigma_levels:
        raise ValueError("sigma_levels must not be empty")
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1, got {step_days!r}")

    sorted_bars = sorted(bars, key=lambda b: b.ts)
    n = len(sorted_bars)
    max_h = max(horizons)
    if n < _MIN_HISTORY_BARS + max_h + 1:
        raise ValueError(
            f"need at least {_MIN_HISTORY_BARS + max_h + 1} bars for underlying "
            f"{underlying_id!r} (got {n}: {_MIN_HISTORY_BARS} history warmup + "
            f"{max_h} horizon + 1)"
        )
    closes = np.array([b.close for b in sorted_bars], dtype=np.float64)
    sigma_series = _daily_sigma_series(closes)

    observations: list[KoCalibrationObservation] = []
    for t0 in range(_MIN_HISTORY_BARS, n - max_h, step_days):
        bar_t0 = sorted_bars[t0]
        prediction_time = bar_t0.available_at
        spot0 = bar_t0.close
        daily_sigma = float(sigma_series[t0])
        if not np.isfinite(daily_sigma) or daily_sigma <= 0.0:
            continue

        # No-look-ahead: only bars whose information was actually available
        # at prediction_time (rule 5). Chronological order + available_at
        # monotonic with ts in every adapter this codebase ships, but this
        # filters explicitly rather than assuming it.
        history = [b for b in sorted_bars[: t0 + 1] if b.available_at <= prediction_time]
        if len(history) < 2:
            continue
        regime = _regime_bucket(sigma_series[:t0], daily_sigma)

        try:
            path_set = simulate_paths(
                history,
                spot0=spot0,
                start=prediction_time,
                horizon_days=max_h,
                n_paths=n_paths,
                rng=rng,
                method=method,
                block_size=block_size,
                lookback_days=lookback_days,
            )
        except ValueError:
            continue

        future_bars = sorted_bars[t0 + 1 : t0 + 1 + max_h]
        if len(future_bars) < max_h:
            continue
        future_low = np.array([b.low for b in future_bars], dtype=np.float64)
        future_high = np.array([b.high for b in future_bars], dtype=np.float64)

        for horizon_days in horizons:
            open_h = path_set.open[:, :horizon_days]
            low_h = path_set.low[:, :horizon_days]
            high_h = path_set.high[:, :horizon_days]
            fut_low_h = future_low[:horizon_days]
            fut_high_h = future_high[:horizon_days]
            sqrt_h = float(np.sqrt(horizon_days))

            for direction in (Direction.LONG, Direction.SHORT):
                for sigma_k in sigma_levels:
                    pct_distance = sigma_k * daily_sigma * sqrt_h
                    if direction == Direction.LONG:
                        barrier = spot0 * (1.0 - pct_distance)
                        hit = (open_h <= barrier) | (low_h <= barrier)
                        realized = bool(np.any(fut_low_h <= barrier))
                    else:
                        barrier = spot0 * (1.0 + pct_distance)
                        hit = (open_h >= barrier) | (high_h >= barrier)
                        realized = bool(np.any(fut_high_h >= barrier))
                    p_ko_raw = float(np.mean(np.any(hit, axis=1)))

                    observations.append(
                        KoCalibrationObservation(
                            underlying_id=underlying_id,
                            t0_index=t0,
                            prediction_time=prediction_time,
                            horizon_days=horizon_days,
                            direction=direction,
                            sigma_k=sigma_k,
                            regime_bucket=regime,
                            p_ko_raw=p_ko_raw,
                            realized_ko=realized,
                        )
                    )
    return observations


__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_SIGMA_LEVELS",
    "KoCalibrationObservation",
    "build_ko_calibration_dataset",
]
