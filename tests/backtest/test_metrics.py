from __future__ import annotations

import numpy as np
import pytest

from turboedge.backtest.metrics import (
    absolute_calibration_error,
    brier_score,
    calibration_intercept,
    calibration_slope,
    calibration_slope_intercept,
    expected_calibration_error,
    hit_rate,
    log_loss,
    max_drawdown,
    mean_signed_error,
    profit_factor,
    reliability_curve,
    sharpe,
    sortino,
)


def test_brier_score_hand_example() -> None:
    p = np.array([0.8, 0.2, 0.6])
    y = np.array([1.0, 0.0, 0.0])
    expected = np.mean([(0.8 - 1.0) ** 2, (0.2 - 0.0) ** 2, (0.6 - 0.0) ** 2])
    assert brier_score(p, y) == pytest.approx(expected)


def test_brier_score_perfect_predictions_is_zero() -> None:
    p = np.array([1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, 1.0])
    assert brier_score(p, y) == pytest.approx(0.0)


def test_brier_score_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        brier_score(np.array([1.5]), np.array([1.0]))
    with pytest.raises(ValueError):
        brier_score(np.array([0.5]), np.array([0.3]))


def test_log_loss_hand_example() -> None:
    p = np.array([0.9, 0.1])
    y = np.array([1.0, 0.0])
    expected = -np.mean([np.log(0.9), np.log(0.9)])
    assert log_loss(p, y) == pytest.approx(expected)


def test_log_loss_clips_extreme_probabilities() -> None:
    p = np.array([1.0, 0.0])
    y = np.array([1.0, 0.0])
    val = log_loss(p, y)
    assert np.isfinite(val)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_reliability_curve_and_ece_perfect_calibration() -> None:
    rng = np.random.default_rng(0)
    n = 10000
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float64)
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece < 0.02  # well-calibrated by construction
    curve = reliability_curve(p, y, n_bins=10)
    assert sum(count for _, _, count in curve) == n


def test_expected_calibration_error_bad_calibration_is_large() -> None:
    n = 1000
    p = np.full(n, 0.9)
    y = np.zeros(n)  # model says 90% up, but never happens
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece == pytest.approx(0.9, abs=1e-6)


def test_sharpe_flat_returns_is_zero() -> None:
    r = np.full(20, 0.001)
    assert sharpe(r) == pytest.approx(0.0)


def test_sharpe_positive_for_positive_mean() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, size=500)
    assert sharpe(r) > 0.0


def test_sortino_ignores_upside_volatility() -> None:
    # All positive returns, no downside -> Sortino should be +inf.
    r = np.array([0.01, 0.02, 0.03, 0.01])
    assert sortino(r) == float("inf")


def test_max_drawdown_hand_example() -> None:
    # equity path: 1 -> 1.1 -> 0.99 -> 1.05
    r = np.array([0.10, -0.10, 0.06060606])
    dd = max_drawdown(r)
    equity = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(equity)
    expected = float(np.min((equity - running_max) / running_max))
    assert dd == pytest.approx(expected)
    assert dd < 0.0


def test_profit_factor_hand_example() -> None:
    r = np.array([0.1, -0.05, 0.2, -0.1])
    expected = (0.1 + 0.2) / (0.05 + 0.1)
    assert profit_factor(r) == pytest.approx(expected)


def test_profit_factor_no_losses_is_inf() -> None:
    r = np.array([0.1, 0.2])
    assert profit_factor(r) == float("inf")


def test_hit_rate_hand_example() -> None:
    r = np.array([0.1, -0.1, 0.2, -0.05, 0.0])
    assert hit_rate(r) == pytest.approx(2.0 / 5.0)


def test_calibration_slope_intercept_perfectly_calibrated_is_near_identity() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=2000)
    y = (rng.uniform(size=2000) < p).astype(np.float64)
    intercept, slope = calibration_slope_intercept(p, y)
    assert intercept == pytest.approx(0.0, abs=0.2)
    assert slope == pytest.approx(1.0, abs=0.2)


def test_calibration_slope_intercept_single_class_is_nan() -> None:
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0.0, 0.0, 0.0])
    intercept, slope = calibration_slope_intercept(p, y)
    assert np.isnan(intercept)
    assert np.isnan(slope)
    assert np.isnan(calibration_intercept(p, y))
    assert np.isnan(calibration_slope(p, y))


def test_mean_signed_error_hand_example() -> None:
    p = np.array([0.8, 0.2, 0.6])
    y = np.array([1.0, 0.0, 0.0])
    expected = np.mean([1.0 - 0.8, 0.0 - 0.2, 0.0 - 0.6])
    assert mean_signed_error(p, y) == pytest.approx(expected)


def test_mean_signed_error_overprediction_is_negative() -> None:
    # p systematically higher than realized frequency -> negative signed error
    # (the conservative direction documented for raw path-simulation P(KO)).
    p = np.full(100, 0.5)
    y = np.zeros(100)
    assert mean_signed_error(p, y) == pytest.approx(-0.5)


def test_absolute_calibration_error_well_calibrated_is_near_zero() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = (rng.uniform(size=5000) < p).astype(np.float64)
    assert absolute_calibration_error(p, y, n_bins=10) == pytest.approx(0.0, abs=0.03)


def test_absolute_calibration_error_differs_from_ece_weighting() -> None:
    # One heavily-populated, well-calibrated bin and one sparse, badly
    # miscalibrated bin: ECE (count-weighted) is dominated by the large bin;
    # the unweighted absolute_calibration_error is not.
    p = np.concatenate([np.full(1000, 0.5), np.full(2, 0.95)])
    y = np.concatenate([np.array([1.0, 0.0] * 500), np.array([0.0, 0.0])])
    ece = expected_calibration_error(p, y, n_bins=10)
    ace = absolute_calibration_error(p, y, n_bins=10)
    assert ace > ece
