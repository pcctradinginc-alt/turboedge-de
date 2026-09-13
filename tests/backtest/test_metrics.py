from __future__ import annotations

import numpy as np
import pytest

from turboedge.backtest.metrics import (
    brier_score,
    expected_calibration_error,
    hit_rate,
    log_loss,
    max_drawdown,
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
