"""Trend-family features: OLS slope/t-stat, robust-trend distance, breakout distance.

Formula reference: Master Spec §8.2 ("Trend"). All functions are pure,
causal (rolling) transforms: ``out[t]`` uses only ``closes[..t]`` (and, for
breakout distance, ``highs[..t]``/``lows[..t]``), so truncating the input
never changes an already-computed value (CLAUDE.md rule 4). Insufficient
history is ``NaN`` -- never imputed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def ols_trend(
    closes: npt.NDArray[np.float64], window: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rolling OLS slope and t-statistic of ``ln(close)`` on a linear time index.

    For each ``t``, fits ``ln(close) = a + b*i`` (``i = 0..window-1``) over
    the trailing ``window`` bars ``closes[t-window+1 .. t]`` and returns the
    slope ``b`` (in log-return-per-day units) and its t-statistic
    ``b / SE(b)``. ``NaN`` while fewer than ``window`` bars are available.
    """
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window!r}")
    y_all = np.log(np.asarray(closes, dtype=np.float64))
    n = y_all.shape[0]
    slope = np.full(n, np.nan, dtype=np.float64)
    tstat = np.full(n, np.nan, dtype=np.float64)

    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    sxx = float(np.sum(x_centered**2))
    dof = window - 2

    for t in range(window - 1, n):
        y = y_all[t - window + 1 : t + 1]
        y_centered = y - y.mean()
        b = float(np.sum(x_centered * y_centered) / sxx)
        resid = y_centered - b * x_centered
        resid_var = float(np.sum(resid**2) / dof)
        se_b = float(np.sqrt(resid_var / sxx)) if resid_var > 0 else 0.0
        slope[t] = b
        tstat[t] = (b / se_b) if se_b > 0 else 0.0
    return slope, tstat


def robust_trend_distance(
    closes: npt.NDArray[np.float64], vol: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Distance (in vol units) of ``ln(close_t)`` from a robust trend line fit over ``window`` bars.

    The robust trend slope is the median of the pairwise slopes from the
    window's first observation to every later one within the window (a
    cheap, O(window)-per-bar Theil-Sen-style estimator, robust to a handful
    of outlying returns without the full O(window^2) pairwise cost); the
    intercept is the median of ``y_i - slope*x_i`` over the window. The
    output is ``(y_t - fitted_t) / vol_t``, i.e. how far the current
    (log) price sits above/below that robust trend, normalized by ``vol``
    (typically a realized-volatility feature aligned to ``closes``).
    """
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window!r}")
    y_all = np.log(np.asarray(closes, dtype=np.float64))
    vol_arr = np.asarray(vol, dtype=np.float64)
    if vol_arr.shape != y_all.shape:
        raise ValueError("vol must have the same shape as closes")
    n = y_all.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)

    for t in range(window - 1, n):
        y = y_all[t - window + 1 : t + 1]
        v = vol_arr[t]
        if not (v > 0):
            continue
        slopes = (y[1:] - y[0]) / x[1:]
        slope = float(np.median(slopes))
        intercept = float(np.median(y - slope * x))
        fitted_t = intercept + slope * x[-1]
        out[t] = (y[-1] - fitted_t) / v
    return out


def breakout_distance(
    closes: npt.NDArray[np.float64],
    highs: npt.NDArray[np.float64],
    lows: npt.NDArray[np.float64],
    vol: npt.NDArray[np.float64],
    window: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Distance (in vol units) from ``close_t`` to the trailing ``window``-bar high/low.

    Returns ``(dist_to_high, dist_to_low)``:

    - ``dist_to_high = (close_t - rolling_max(high, window)) / (vol_t * close_t)``,
      ``<= 0``; ``0`` at a new ``window``-bar high (upside breakout).
    - ``dist_to_low = (close_t - rolling_min(low, window)) / (vol_t * close_t)``,
      ``>= 0``; ``0`` at a new ``window``-bar low (downside breakout).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    c = np.asarray(closes, dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    v = np.asarray(vol, dtype=np.float64)
    if not (c.shape == h.shape == lo.shape == v.shape):
        raise ValueError("closes, highs, lows and vol must have the same shape")

    roll_high = pd.Series(h).rolling(window=window, min_periods=window).max().to_numpy()
    roll_low = pd.Series(lo).rolling(window=window, min_periods=window).min().to_numpy()
    denom = v * c
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_high = np.where(denom > 0, (c - roll_high) / denom, np.nan)
        dist_low = np.where(denom > 0, (c - roll_low) / denom, np.nan)
    return dist_high, dist_low
