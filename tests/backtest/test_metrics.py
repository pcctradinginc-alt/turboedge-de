from __future__ import annotations

import numpy as np
import pytest

from turboedge.backtest.metrics import (
    absolute_calibration_error,
    brier_score,
    calibration_intercept,
    calibration_slope,
    calibration_slope_intercept,
    crps_from_quantiles,
    expected_calibration_error,
    hit_rate,
    interval_coverage,
    log_loss,
    max_drawdown,
    mean_crps_from_quantiles,
    mean_pinball_loss,
    mean_signed_error,
    pinball_loss,
    pinball_loss_by_level,
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


# --- Phase D: pinball loss, CRPS, interval coverage -------------------------------------------


def test_pinball_loss_hand_examples() -> None:
    # y >= q branch: L = tau * (y - q)
    assert pinball_loss(1.0, 0.8, tau=0.5) == pytest.approx(0.5 * 0.2)
    # y < q branch: L = (1 - tau) * (q - y)
    assert pinball_loss(1.0, 1.5, tau=0.9) == pytest.approx(0.1 * 0.5)
    # exact hit is zero loss regardless of tau
    assert pinball_loss(2.0, 2.0, tau=0.05) == pytest.approx(0.0)
    assert pinball_loss(2.0, 2.0, tau=0.95) == pytest.approx(0.0)


def test_pinball_loss_rejects_bad_tau() -> None:
    with pytest.raises(ValueError):
        pinball_loss(1.0, 0.5, tau=0.0)
    with pytest.raises(ValueError):
        pinball_loss(1.0, 0.5, tau=1.0)


def test_mean_pinball_loss_hand_example() -> None:
    y = np.array([1.0, 1.0])
    q = np.array([0.8, 1.2])
    expected = np.mean([pinball_loss(1.0, 0.8, 0.3), pinball_loss(1.0, 1.2, 0.3)])
    assert mean_pinball_loss(y, q, tau=0.3) == pytest.approx(expected)


def test_crps_from_quantiles_hand_example() -> None:
    # y=0 against a symmetric 5-quantile distribution; every pinball term
    # computed by hand (max(tau*diff, (tau-1)*diff), diff = y - q):
    #   tau=.05, q=-2  -> diff=2  -> max(.1, -1.9)  = .1
    #   tau=.25, q=-.5 -> diff=.5 -> max(.125, -.375) = .125
    #   tau=.50, q=0   -> diff=0  -> 0
    #   tau=.75, q=.5  -> diff=-.5-> max(-.375, .125) = .125
    #   tau=.95, q=2   -> diff=-2 -> max(-1.9, .1)   = .1
    # mean = 0.45 / 5 = 0.09; CRPS = 2 * mean = 0.18
    quantiles = {0.05: -2.0, 0.25: -0.5, 0.50: 0.0, 0.75: 0.5, 0.95: 2.0}
    assert crps_from_quantiles(0.0, quantiles) == pytest.approx(0.18)


def test_crps_from_quantiles_perfect_point_mass_is_low() -> None:
    # A (degenerate) "distribution" whose every quantile equals the realized
    # value scores zero -- CRPS's minimum.
    quantiles = {0.05: 1.0, 0.25: 1.0, 0.50: 1.0, 0.75: 1.0, 0.95: 1.0}
    assert crps_from_quantiles(1.0, quantiles) == pytest.approx(0.0)


def test_crps_from_quantiles_rejects_empty() -> None:
    with pytest.raises(ValueError):
        crps_from_quantiles(0.0, {})


def test_mean_crps_from_quantiles_matches_manual_average() -> None:
    q_a = {0.05: -2.0, 0.25: -0.5, 0.50: 0.0, 0.75: 0.5, 0.95: 2.0}
    q_b = {0.05: -1.0, 0.25: -0.25, 0.50: 0.0, 0.75: 0.25, 0.95: 1.0}
    y = np.array([0.0, 0.0])
    expected = np.mean([crps_from_quantiles(0.0, q_a), crps_from_quantiles(0.0, q_b)])
    assert mean_crps_from_quantiles(y, [q_a, q_b]) == pytest.approx(expected)


def test_pinball_loss_by_level_matches_manual_mean_per_level() -> None:
    q_list = [{0.5: 0.8, 0.9: 1.5}, {0.5: 1.2, 0.9: 1.1}]
    y = np.array([1.0, 1.0])
    result = pinball_loss_by_level(y, q_list)
    assert result[0.5] == pytest.approx(
        np.mean([pinball_loss(1.0, 0.8, 0.5), pinball_loss(1.0, 1.2, 0.5)])
    )
    assert result[0.9] == pytest.approx(
        np.mean([pinball_loss(1.0, 1.5, 0.9), pinball_loss(1.0, 1.1, 0.9)])
    )


def test_interval_coverage_hand_example() -> None:
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    lower = np.full(5, -1.0)
    upper = np.full(5, 1.0)
    assert interval_coverage(y, lower, upper) == pytest.approx(1.0)


def test_interval_coverage_detects_deliberately_too_narrow_interval() -> None:
    """A miscalibrated, deliberately-too-narrow interval must show up as
    empirical coverage far below its nominal level -- this is the whole
    point of measuring coverage instead of trusting the model's own claimed
    quantile levels."""
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=5000)
    # True 90% interval for N(0,1) is roughly [-1.645, 1.645]; shrink it by
    # 10x so it is deliberately far too narrow.
    lower = np.full(y.shape, -0.1645)
    upper = np.full(y.shape, 0.1645)
    coverage = interval_coverage(y, lower, upper)
    assert coverage < 0.20  # nominal was 0.90 -- badly, detectably miscalibrated
    # the same interval at its correct (un-shrunk) width should recover
    # coverage close to the nominal 90%
    wide_coverage = interval_coverage(y, lower * 10.0, upper * 10.0)
    assert wide_coverage == pytest.approx(0.90, abs=0.03)
    assert wide_coverage > coverage


def test_interval_coverage_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        interval_coverage(np.array([0.0, 0.0]), np.array([-1.0]), np.array([1.0, 1.0]))
