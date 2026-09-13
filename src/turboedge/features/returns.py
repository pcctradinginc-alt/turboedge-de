"""Return-family features and the combined per-bar feature frame.

Formula reference: Master Spec §8.1 ("Returns"); Build Contract v2 item 1.
``build_feature_frame`` is the single entry point models use to turn a
chronological bar sequence into a dense, purely-causal feature matrix: row
``i`` uses only ``bars[0..i]`` (CLAUDE.md rule 4), so slicing the input to
any earlier length never changes an already-computed row. Insufficient
history is ``NaN`` -- never imputed (Build Contract v2 item 1); callers that
train a model must drop NaN rows themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt

from turboedge.features.trend import breakout_distance, ols_trend, robust_trend_distance
from turboedge.features.volatility import (
    downside_semivariance,
    ewma_volatility,
    garman_klass_volatility,
    kurtosis,
    parkinson_volatility,
    realized_volatility,
    skewness,
    volatility_of_volatility,
)
from turboedge.storage.schemas import UnderlyingBar

#: h-day log-return lookbacks used both as features (§8.1) and, elsewhere, as
#: the protected TSMOM lookback superset check.
RETURN_HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 20, 63, 126)
_TREND_WINDOWS: tuple[int, ...] = (20, 63)
_DIST_WINDOW = 63
_VOL_SHORT = 20
_VOL_LONG = 63


def bars_as_of(bars: Sequence[UnderlyingBar], as_of: datetime) -> list[UnderlyingBar]:
    """Bars actually observable "as of" ``as_of``, sorted ascending by ``ts``.

    Enforces CLAUDE.md rule 5 (``available_at <= prediction_time``) once,
    centrally, so every forecast model applies the same look-ahead cutoff
    before deriving any feature or label.
    """
    eligible = [b for b in bars if b.available_at <= as_of]
    eligible.sort(key=lambda b: b.ts)
    return eligible


def log_return_nd(closes: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    """``out[t] = ln(closes[t] / closes[t-n])``; ``NaN`` for ``t < n``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    arr = np.asarray(closes, dtype=np.float64)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    if arr.shape[0] > n:
        out[n:] = np.log(arr[n:] / arr[:-n])
    return out


def build_feature_frame(
    bars: Sequence[UnderlyingBar],
) -> tuple[list[datetime], npt.NDArray[np.float64], list[str]]:
    """Dense causal feature matrix for a chronological bar sequence.

    Returns ``(dates, X, names)`` with ``X.shape == (len(bars), len(names))``;
    row ``i`` is the feature vector "as of" ``bars[i]`` (uses only
    ``bars[0..i]``). ``bars`` must already be the caller's as-of-filtered,
    chronologically sorted sequence (see :func:`bars_as_of`) -- this function
    does not itself apply any ``available_at`` cutoff.

    Feature families (Master Spec §8.1-§8.4, narrowed by Build Contract v2
    item 1): log returns at 1/2/5/10/20/63/126 days; realized/EWMA/Parkinson/
    Garman-Klass volatility and vol-of-vol; rolling OLS slope + t-stat at
    20/63 days, distance to a robust (Theil-Sen-style) trend and breakout
    distance to the 63-day high/low, both in volatility units; and 63-day
    skewness, kurtosis and downside semivariance.

    Raises:
        ValueError: if ``bars`` is empty, not sorted ascending by ``ts``, or
            mixes more than one ``underlying_id``.
    """
    if not bars:
        raise ValueError("bars must not be empty")
    underlying_ids = {b.underlying_id for b in bars}
    if len(underlying_ids) > 1:
        raise ValueError(f"bars must all share one underlying_id, got {underlying_ids!r}")
    dates = [b.ts for b in bars]
    if dates != sorted(dates):
        raise ValueError("bars must be sorted ascending by ts")

    closes = np.array([b.close for b in bars], dtype=np.float64)
    opens = np.array([b.open for b in bars], dtype=np.float64)
    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)

    columns: dict[str, npt.NDArray[np.float64]] = {}
    for n in RETURN_HORIZONS:
        columns[f"log_return_{n}d"] = log_return_nd(closes, n)

    columns[f"realized_vol_{_VOL_SHORT}d"] = realized_volatility(closes, _VOL_SHORT)
    columns[f"realized_vol_{_VOL_LONG}d"] = realized_volatility(closes, _VOL_LONG)
    columns["ewma_vol"] = ewma_volatility(closes)
    columns[f"parkinson_vol_{_VOL_SHORT}d"] = parkinson_volatility(highs, lows, _VOL_SHORT)
    columns[f"parkinson_vol_{_VOL_LONG}d"] = parkinson_volatility(highs, lows, _VOL_LONG)
    columns[f"garman_klass_vol_{_VOL_SHORT}d"] = garman_klass_volatility(
        opens, highs, lows, closes, _VOL_SHORT
    )
    columns[f"garman_klass_vol_{_VOL_LONG}d"] = garman_klass_volatility(
        opens, highs, lows, closes, _VOL_LONG
    )
    columns[f"vol_of_vol_{_VOL_SHORT}d"] = volatility_of_volatility(closes, _VOL_SHORT, _VOL_SHORT)

    for w in _TREND_WINDOWS:
        slope, tstat = ols_trend(closes, w)
        columns[f"ols_slope_{w}d"] = slope
        columns[f"ols_tstat_{w}d"] = tstat

    vol_for_distance = columns[f"realized_vol_{_VOL_LONG}d"]
    columns[f"robust_trend_distance_{_DIST_WINDOW}d"] = robust_trend_distance(
        closes, vol_for_distance, _DIST_WINDOW
    )
    dist_high, dist_low = breakout_distance(closes, highs, lows, vol_for_distance, _DIST_WINDOW)
    columns[f"breakout_dist_high_{_DIST_WINDOW}d"] = dist_high
    columns[f"breakout_dist_low_{_DIST_WINDOW}d"] = dist_low

    columns[f"skew_{_DIST_WINDOW}d"] = skewness(closes, _DIST_WINDOW)
    columns[f"kurtosis_{_DIST_WINDOW}d"] = kurtosis(closes, _DIST_WINDOW)
    columns[f"downside_semivariance_{_DIST_WINDOW}d"] = downside_semivariance(closes, _DIST_WINDOW)

    names = list(columns.keys())
    x = np.column_stack([columns[name] for name in names])
    return dates, x, names
