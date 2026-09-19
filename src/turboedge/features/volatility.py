"""Volatility- and distribution-shape features.

Formula reference: Master Spec §8.3 ("Volatilitaet") and §8.4
("Verteilungsform"); Build Contract v2 item 1 narrows the challenger set to
realized vol, EWMA vol, Parkinson, Garman-Klass, vol-of-vol, skewness,
kurtosis and downside semivariance.

Every function is a pure, causal (rolling) transform over OHLC arrays:
``out[t]`` uses only observations up to and including index ``t``, so
truncating the input to any earlier length never changes an
already-computed value (CLAUDE.md rule 4, "kein Look-ahead"). Insufficient
history is ``NaN`` -- never imputed (Build Contract v2 item 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from turboedge.features.product import ewma_volatility as _ewma_daily_vol

if TYPE_CHECKING:
    from turboedge.storage.schemas import UnderlyingBar

_LN2 = float(np.log(2.0))
_GK_CONST = 2.0 * _LN2 - 1.0
_EWMA_WARMUP = 20


def log_returns(closes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Daily log returns ``ln(close[t] / close[t-1])``; length ``len(closes) - 1``."""
    arr = np.asarray(closes, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"closes must be 1-dimensional, got shape {arr.shape!r}")
    return np.diff(np.log(arr))


def _align_from_returns(series: npt.NDArray[np.float64], n_closes: int) -> npt.NDArray[np.float64]:
    """Left-pad a length-``n_closes - 1`` return-indexed series with one leading NaN."""
    out = np.full(n_closes, np.nan, dtype=np.float64)
    out[1:] = series
    return out


def realized_volatility(closes: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Rolling sample-std of daily log returns over a trailing ``window``, aligned to ``closes``.

    ``out[t]`` uses the ``window`` returns ending at ``t`` (i.e. bars
    ``t-window .. t``); ``NaN`` while fewer than ``window`` returns are
    available.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    arr = np.asarray(closes, dtype=np.float64)
    rets = log_returns(arr)
    roll = pd.Series(rets).rolling(window=window, min_periods=window).std(ddof=1)
    return _align_from_returns(roll.to_numpy(), arr.shape[0])


def ewma_volatility(
    closes: npt.NDArray[np.float64], lam: float = 0.94, warmup: int = _EWMA_WARMUP
) -> npt.NDArray[np.float64]:
    """EWMA daily volatility of log returns (``turboedge.features.product.ewma_volatility``).

    The recursive EWMA estimate is technically defined from the very first
    return onward, but its first few values are a poor (effectively
    unregularized) estimate of volatility. The first ``warmup`` return-indexed
    values are therefore masked to ``NaN`` rather than treated as valid
    signal, on top of the single leading NaN every return-indexed feature
    carries (no return defined for the first bar).
    """
    arr = np.asarray(closes, dtype=np.float64)
    rets = log_returns(arr)
    if rets.size == 0:
        return np.full(arr.shape[0], np.nan, dtype=np.float64)
    ewma = _ewma_daily_vol(rets, lam=lam)
    ewma = ewma.copy()
    ewma[: min(warmup, ewma.size)] = np.nan
    return _align_from_returns(ewma, arr.shape[0])


def causal_ewma_sigma(
    bars: Sequence[UnderlyingBar],
    as_of: datetime,
    *,
    lam: float = 0.94,
    warmup: int = _EWMA_WARMUP,
) -> float:
    """EWMA daily volatility "as of" ``as_of``, built only from bars whose
    ``available_at <= as_of`` (CLAUDE.md rule 5).

    This is ``sigma_t`` in Phase D's normalized forecasting target
    ``y_h = ln(P_t+h / P_t) / (sigma_t * sqrt(h))`` (``docs/measured_results.md``
    Phase D, backtest/walkforward.py). Filters and sorts ``bars`` itself
    (rather than requiring a pre-filtered, pre-sorted sequence like
    :func:`ewma_volatility` does) so every caller applies the exact same
    look-ahead cutoff; returns ``NaN`` when there is not enough eligible
    history to form even a single EWMA estimate.
    """
    eligible = [b for b in bars if b.available_at <= as_of]
    eligible.sort(key=lambda b: b.ts)
    if len(eligible) < 2:
        return float("nan")
    closes = np.array([b.close for b in eligible], dtype=np.float64)
    sigma_series = ewma_volatility(closes, lam=lam, warmup=warmup)
    return float(sigma_series[-1])


def normalized_horizon_target(
    closes: npt.NDArray[np.float64], sigma: npt.NDArray[np.float64], horizon: int
) -> npt.NDArray[np.float64]:
    """Phase D primary forecasting target, vectorized: ``y_h[t] = ln(closes[t+h]/closes[t]) /
    (sigma[t] * sqrt(h))``.

    ``closes`` and ``sigma`` must be aligned, chronological (ts-ascending)
    and ``sigma[t]`` must already be a *causal* volatility estimate "as of"
    bar ``t`` (e.g. :func:`ewma_volatility` applied to the same, already
    as-of-filtered ``closes`` -- its own causal-prefix guarantee is what
    makes this vectorized batch form equivalent to calling
    :func:`causal_ewma_sigma` separately for every ``t``, the same
    equivalence :class:`~turboedge.models.directional.TsmomForecastModel`
    relies on for its own vectorized score series). Output is aligned to
    ``closes``; ``NaN`` wherever ``t + h`` is out of range or ``sigma[t]``
    is ``NaN``/``<= 0`` (never divided-by-zero, never silently imputed).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon!r}")
    c = np.asarray(closes, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if c.shape != s.shape:
        raise ValueError("closes and sigma must have the same shape")
    n = c.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return out
    denom = s[: n - horizon] * np.sqrt(horizon)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.log(c[horizon:] / c[: n - horizon]) / denom
    valid = denom > 0.0
    out[: n - horizon] = np.where(valid, ratio, np.nan)
    return out


def parkinson_volatility(
    highs: npt.NDArray[np.float64], lows: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Parkinson (1980) high-low range volatility estimator, rolling over ``window`` bars.

    ``sigma_t = sqrt(mean(ln(high/low)^2) / (4 * ln 2))`` over the trailing
    window; uses only each bar's own (already-closed) high/low, so it is
    causal without any additional lag.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    if h.shape != lo.shape:
        raise ValueError("highs and lows must have the same shape")
    sq = np.log(h / lo) ** 2
    roll_mean = pd.Series(sq).rolling(window=window, min_periods=window).mean()
    return np.sqrt(roll_mean.to_numpy() / (4.0 * _LN2))


def garman_klass_volatility(
    opens: npt.NDArray[np.float64],
    highs: npt.NDArray[np.float64],
    lows: npt.NDArray[np.float64],
    closes: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]:
    """Garman-Klass (1980) OHLC volatility estimator, rolling over ``window`` bars.

    ``var_t = mean(0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2)`` over the window.
    The per-bar term can occasionally go slightly negative for individual
    windows (a known property of the GK estimator on noisy data); the
    windowed mean is floored at 0 before the square root since variance
    cannot legitimately be negative -- this is a mathematical floor on the
    estimator, not an imputation of missing data.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    o = np.asarray(opens, dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    if not (o.shape == h.shape == lo.shape == c.shape):
        raise ValueError("opens, highs, lows and closes must have the same shape")
    term = 0.5 * np.log(h / lo) ** 2 - _GK_CONST * np.log(c / o) ** 2
    roll_mean = pd.Series(term).rolling(window=window, min_periods=window).mean().to_numpy()
    variance = np.where(np.isnan(roll_mean), np.nan, np.clip(roll_mean, 0.0, None))
    return np.sqrt(variance)


def volatility_of_volatility(
    closes: npt.NDArray[np.float64], vol_window: int, vol_of_vol_window: int
) -> npt.NDArray[np.float64]:
    """Rolling std of the realized-volatility series itself ("vol of vol")."""
    rv = realized_volatility(closes, vol_window)
    roll = (
        pd.Series(rv).rolling(window=vol_of_vol_window, min_periods=vol_of_vol_window).std(ddof=1)
    )
    return roll.to_numpy()


def skewness(closes: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Rolling (Fisher-Pearson, bias-adjusted) skewness of daily log returns."""
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window!r}")
    rets = log_returns(np.asarray(closes, dtype=np.float64))
    roll = pd.Series(rets).rolling(window=window, min_periods=window).skew()
    return _align_from_returns(roll.to_numpy(), np.asarray(closes).shape[0])


def kurtosis(closes: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Rolling excess kurtosis (normal == 0) of daily log returns."""
    if window < 4:
        raise ValueError(f"window must be >= 4, got {window!r}")
    rets = log_returns(np.asarray(closes, dtype=np.float64))
    roll = pd.Series(rets).rolling(window=window, min_periods=window).kurt()
    return _align_from_returns(roll.to_numpy(), np.asarray(closes).shape[0])


def downside_semivariance(closes: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Rolling mean of squared negative daily log returns (0 contributed by non-negative days)."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    rets = log_returns(np.asarray(closes, dtype=np.float64))
    neg_sq = np.where(rets < 0.0, rets**2, 0.0)
    roll = pd.Series(neg_sq).rolling(window=window, min_periods=window).mean()
    return _align_from_returns(roll.to_numpy(), np.asarray(closes).shape[0])
