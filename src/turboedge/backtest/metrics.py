"""Calibration and P&L metrics.

Formula reference: Master Spec §10 ("Calibration") and §38 ("Metrics").
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

_DEFAULT_ECE_BINS = 10
_LOG_LOSS_EPS = 1e-12
# Regularization strength for the calibration-intercept/slope logistic fit
# (Cox calibration, Steyerberg et al.): large C approximates the classical
# unregularized MLE while still converging cleanly on small/separable
# samples (some (horizon, direction, sigma_bucket) cells have very few or
# even zero realized positives -- see calibration_slope_intercept).
_COX_CALIBRATION_C = 1e4


def _as_prob_and_label(
    p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    p_arr = np.asarray(p, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if p_arr.shape != y_arr.shape:
        raise ValueError(f"p and y must have the same shape, got {p_arr.shape!r}/{y_arr.shape!r}")
    if p_arr.size == 0:
        raise ValueError("p and y must not be empty")
    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        raise ValueError("p must be within [0, 1]")
    if np.any((y_arr != 0.0) & (y_arr != 1.0)):
        raise ValueError("y must be binary (0.0 or 1.0)")
    return p_arr, y_arr


def brier_score(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Mean squared error between predicted probability and binary outcome, in ``[0, 1]``."""
    p_arr, y_arr = _as_prob_and_label(p, y)
    return float(np.mean((p_arr - y_arr) ** 2))


def log_loss(
    p: npt.NDArray[np.float64], y: npt.NDArray[np.float64], eps: float = _LOG_LOSS_EPS
) -> float:
    """Binary log loss (cross-entropy); ``p`` clipped to ``[eps, 1-eps]`` to avoid ``-inf``."""
    p_arr, y_arr = _as_prob_and_label(p, y)
    clipped = np.clip(p_arr, eps, 1.0 - eps)
    return float(-np.mean(y_arr * np.log(clipped) + (1.0 - y_arr) * np.log(1.0 - clipped)))


def reliability_curve(
    p: npt.NDArray[np.float64], y: npt.NDArray[np.float64], n_bins: int = _DEFAULT_ECE_BINS
) -> list[tuple[float, float, int]]:
    """Per-bin ``(mean_predicted_p, mean_observed_frequency, count)`` over equal-width bins.

    Only non-empty bins are returned, in ascending bin order.
    """
    p_arr, y_arr = _as_prob_and_label(p, y)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, n_bins - 1)
    curve: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        curve.append((float(np.mean(p_arr[mask])), float(np.mean(y_arr[mask])), count))
    return curve


def expected_calibration_error(
    p: npt.NDArray[np.float64], y: npt.NDArray[np.float64], n_bins: int = _DEFAULT_ECE_BINS
) -> float:
    """Weighted-average ``|mean_predicted - mean_observed|`` over equal-width ``[0,1]`` bins."""
    p_arr, _ = _as_prob_and_label(p, y)
    curve = reliability_curve(p, y, n_bins=n_bins)
    total = p_arr.size
    return float(sum(count * abs(mean_p - mean_y) for mean_p, mean_y, count in curve) / total)


def calibration_slope_intercept(
    p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Cox calibration intercept and slope (Steyerberg et al.): fit
    ``y ~ intercept + slope * logit(p)`` by logistic regression.

    Perfect calibration is ``intercept == 0.0`` and ``slope == 1.0``;
    ``slope < 1`` means predictions are too extreme (overconfident away
    from the base rate), ``slope > 1`` too conservative (underconfident);
    ``intercept != 0`` means the predictions are systematically offset
    from the true base rate even where their relative ordering is fine.

    ``p`` is clipped to ``[eps, 1-eps]`` before taking ``logit`` (a
    predicted probability of exactly 0 or 1 has no finite logit). Returns
    ``(nan, nan)`` when ``y`` has only one class -- the slope/intercept are
    then not identifiable (this is common for far-out-of-the-money barrier
    buckets where realized knock-outs never or always occur in a fold).
    """
    p_arr, y_arr = _as_prob_and_label(p, y)
    if np.unique(y_arr).size < 2:
        return float("nan"), float("nan")
    eps = 1e-6
    clipped = np.clip(p_arr, eps, 1.0 - eps)
    logit_p = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000, C=_COX_CALIBRATION_C)
    model.fit(logit_p, y_arr)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return intercept, slope


def calibration_intercept(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Cox calibration intercept; see :func:`calibration_slope_intercept`."""
    return calibration_slope_intercept(p, y)[0]


def calibration_slope(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Cox calibration slope; see :func:`calibration_slope_intercept`."""
    return calibration_slope_intercept(p, y)[1]


def mean_signed_error(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Mean ``(y - p)``: the signed bias of predictions against outcomes.

    Positive means predictions understate the realized frequency (e.g. a
    P(KO) too low -- an unsafe direction for a risk gate); negative means
    predictions overstate it (e.g. a P(KO) too high -- the conservative
    direction documented for the raw path-simulation P(KO) in
    ``docs/measured_results.md`` §3). Unlike :func:`brier_score`, sign is
    preserved rather than squared away, so this is the metric to check when
    asking "which direction is the bias in", not "how large is the error".
    """
    p_arr, y_arr = _as_prob_and_label(p, y)
    return float(np.mean(y_arr - p_arr))


def absolute_calibration_error(
    p: npt.NDArray[np.float64], y: npt.NDArray[np.float64], n_bins: int = _DEFAULT_ECE_BINS
) -> float:
    """Unweighted mean ``|mean_predicted - mean_observed|`` over non-empty
    equal-width bins.

    Contrast with :func:`expected_calibration_error`, which weights each
    bin's contribution by its observation count: a bin with few
    observations counts the same here as a densely populated one, so a
    sparsely populated but badly miscalibrated region (e.g. a large,
    rarely-touched barrier-distance bucket) is not diluted away by a much
    larger, well-calibrated bin the way it would be in ECE.
    """
    curve = reliability_curve(p, y, n_bins=n_bins)
    if not curve:
        raise ValueError("no non-empty bins")
    return float(np.mean([abs(mean_p - mean_y) for mean_p, mean_y, _count in curve]))


def sharpe(returns: npt.NDArray[np.float64], periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe ratio (mean/std, ``ddof=1``) of a return series; ``0`` if flat."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        raise ValueError("returns must have at least 2 observations")
    std = float(np.std(r, ddof=1))
    # A "flat" (constant) series can still produce a tiny nonzero floating-
    # point std from summation rounding; treat anything below this relative
    # tolerance as exactly flat rather than reporting a spurious huge ratio.
    if std <= 1e-12 * (1.0 + abs(float(np.mean(r)))):
        return 0.0
    return float(np.mean(r) / std * np.sqrt(periods_per_year))


def sortino(returns: npt.NDArray[np.float64], periods_per_year: float = 252.0) -> float:
    """Annualized Sortino ratio (mean / downside-deviation) of a per-period return series."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        raise ValueError("returns must have at least 2 observations")
    downside = r[r < 0.0]
    if downside.size == 0:
        return float("inf") if np.mean(r) > 0 else 0.0
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std <= 1e-12 * (1.0 + abs(float(np.mean(r)))):
        return 0.0
    return float(np.mean(r) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: npt.NDArray[np.float64]) -> float:
    """Maximum peak-to-trough drawdown of the compounded equity curve, as a negative fraction.

    ``returns`` are treated as simple per-period returns compounding
    multiplicatively (``equity = cumprod(1 + returns)``); e.g. ``-0.25``
    means a 25% peak-to-trough decline.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        raise ValueError("returns must not be empty")
    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(np.min(drawdown))


def profit_factor(returns: npt.NDArray[np.float64]) -> float:
    """Gross profit / gross loss (``inf`` if there are no losing periods)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        raise ValueError("returns must not be empty")
    gains = float(np.sum(r[r > 0.0]))
    losses = float(-np.sum(r[r < 0.0]))
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def hit_rate(returns: npt.NDArray[np.float64]) -> float:
    """Fraction of strictly positive periods."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        raise ValueError("returns must not be empty")
    return float(np.mean(r > 0.0))
