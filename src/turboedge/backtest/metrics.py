"""Calibration and P&L metrics.

Formula reference: Master Spec §10 ("Calibration") and §38 ("Metrics").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


# --- Phase D: distributional evaluation (CRPS, pinball loss, interval coverage) -------------
#
# docs/measured_results.md Phase D: the existing binary metrics above (brier_score,
# log_loss, ece, hit_rate) only ever score p_up/label -- they structurally cannot
# reward a model whose full predictive distribution (mean/sigma/quantiles/ES,
# models.forecast.HorizonForecast) is informative but whose binary hit rate is
# unremarkable. These functions score the distribution itself, via the quantiles
# every ForecastModel already produces.


def pinball_loss(y_true: float, q_pred: float, tau: float) -> float:
    """Quantile (pinball) loss of one predicted ``tau``-quantile against one realized value.

    ``L_tau(y, q) = tau * (y - q)`` if ``y >= q``, else ``(1 - tau) * (q - y)``
    -- equivalently ``max(tau * (y - q), (tau - 1) * (y - q))``, the form used
    here. Always ``>= 0``; ``0`` only for a perfect quantile hit. Minimized in
    expectation exactly at the true ``tau``-quantile of ``y``'s distribution,
    which is what makes it the standard proper scoring rule for one quantile
    (Koenker & Bassett 1978; Gneiting & Raftery 2007).
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be within (0, 1), got {tau!r}")
    diff = float(y_true) - float(q_pred)
    return float(max(tau * diff, (tau - 1.0) * diff))


def mean_pinball_loss(
    y_true: npt.NDArray[np.float64], q_pred: npt.NDArray[np.float64], tau: float
) -> float:
    """Mean pinball loss (see :func:`pinball_loss`) across many ``(y_true, q_pred)`` pairs."""
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be within (0, 1), got {tau!r}")
    y = np.asarray(y_true, dtype=np.float64)
    q = np.asarray(q_pred, dtype=np.float64)
    if y.shape != q.shape:
        raise ValueError(f"y_true and q_pred must have the same shape, got {y.shape!r}/{q.shape!r}")
    if y.size == 0:
        raise ValueError("y_true and q_pred must not be empty")
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def crps_from_quantiles(y_true: float, quantiles: Mapping[float, float]) -> float:
    """Quantile-based CRPS approximation for one realized value against a
    distribution given only by a finite set of quantiles.

    ``CRPS(F, y) = 2 * integral_0^1 pinball_tau(y, F^-1(tau)) dtau`` exactly,
    for the true quantile function ``F^-1`` (Gneiting & Raftery 2007, eq. 21;
    Matheson & Winkler 1976); with only a finite quantile grid (e.g. the
    ``q05/q25/q50/q75/q95`` every :class:`~turboedge.models.forecast.HorizonForecast`
    carries) the integral is approximated by ``2 * mean`` of the pinball loss
    over the given ``{tau: quantile_value}`` levels -- the standard
    quantile-averaging CRPS approximation used across forecast-evaluation
    literature (e.g. Bracher et al. 2021, "Evaluating epidemic forecasts").
    Coarser grids give a coarser (typically slightly conservative/smoothed)
    approximation, never a wrong sign or direction of comparison between two
    models scored on the same grid.
    """
    if not quantiles:
        raise ValueError("quantiles must not be empty")
    losses = [pinball_loss(y_true, q, tau) for tau, q in quantiles.items()]
    return float(2.0 * np.mean(losses))


def mean_crps_from_quantiles(
    y_true: npt.NDArray[np.float64], quantiles_per_obs: Sequence[Mapping[float, float]]
) -> float:
    """Mean :func:`crps_from_quantiles` across many observations."""
    y = np.asarray(y_true, dtype=np.float64)
    if y.shape[0] != len(quantiles_per_obs):
        raise ValueError("y_true and quantiles_per_obs must have the same length")
    if y.size == 0:
        raise ValueError("y_true must not be empty")
    values = [crps_from_quantiles(float(yi), q) for yi, q in zip(y, quantiles_per_obs, strict=True)]
    return float(np.mean(values))


def pinball_loss_by_level(
    y_true: npt.NDArray[np.float64], quantiles_per_obs: Sequence[Mapping[float, float]]
) -> dict[float, float]:
    """Mean pinball loss per quantile level ``tau``, across every observation carrying that level.

    ``quantiles_per_obs[i]`` need not carry the exact same set of levels for
    every observation; the mean for a given ``tau`` is only taken over
    observations that actually have it.
    """
    y = np.asarray(y_true, dtype=np.float64)
    if y.shape[0] != len(quantiles_per_obs):
        raise ValueError("y_true and quantiles_per_obs must have the same length")
    if y.size == 0:
        raise ValueError("y_true must not be empty")
    levels: set[float] = set()
    for q in quantiles_per_obs:
        levels.update(q.keys())
    result: dict[float, float] = {}
    for tau in sorted(levels):
        losses = [
            pinball_loss(float(yi), qi[tau], tau)
            for yi, qi in zip(y, quantiles_per_obs, strict=True)
            if tau in qi
        ]
        if losses:
            result[tau] = float(np.mean(losses))
    return result


def interval_coverage(
    y_true: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> float:
    """Fraction of ``y_true`` falling within the closed interval ``[lower, upper]``, elementwise.

    For a well-calibrated central prediction interval (e.g. ``[q05, q95]``,
    nominal coverage 90%), this should be close to the nominal level out of
    sample; a materially lower empirical coverage means the interval is too
    narrow (overconfident) -- exactly the failure mode this function is
    meant to catch (``docs/measured_results.md`` Phase D coverage checks).
    """
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must have the same shape")
    if y.size == 0:
        raise ValueError("y_true must not be empty")
    return float(np.mean((y >= lo) & (y <= hi)))
