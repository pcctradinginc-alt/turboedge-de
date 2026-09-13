from __future__ import annotations

import numpy as np
import pytest

from turboedge.models.calibration import (
    CalibrationConfig,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    fit_calibrator,
)


def _make_calibration_data(
    n: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw_p = rng.uniform(0.0, 1.0, size=n)
    # Miscalibrated: true P(up) is a squashed version of raw_p, so a
    # calibrator has something real to correct.
    true_p = raw_p**2
    labels = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(np.float64)
    weights = np.ones(n)
    return raw_p, labels, weights


def test_isotonic_calibrator_is_monotonic_in_raw_p() -> None:
    raw_p, labels, weights = _make_calibration_data()
    cfg = CalibrationConfig(method="isotonic")
    calibrator = fit_calibrator(cfg, raw_p, labels, weights)
    assert isinstance(calibrator, IsotonicCalibrator)
    grid = np.linspace(0.0, 1.0, 50)
    calibrated = calibrator.predict(grid)
    assert np.all(np.diff(calibrated) >= -1e-12)  # non-decreasing
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_platt_calibrator_output_in_unit_interval() -> None:
    raw_p, labels, weights = _make_calibration_data(seed=1)
    cfg = CalibrationConfig(method="platt")
    calibrator = fit_calibrator(cfg, raw_p, labels, weights)
    assert isinstance(calibrator, PlattCalibrator)
    grid = np.linspace(0.0, 1.0, 50)
    calibrated = calibrator.predict(grid)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_platt_calibrator_is_monotonic_for_informative_raw_p() -> None:
    raw_p, labels, weights = _make_calibration_data(seed=2)
    cfg = CalibrationConfig(method="platt")
    calibrator = fit_calibrator(cfg, raw_p, labels, weights)
    grid = np.linspace(0.0, 1.0, 50)
    calibrated = calibrator.predict(grid)
    # Platt scaling is a monotone (sigmoid) function of raw_p whenever the
    # fitted coefficient is positive, which it will be here since higher
    # raw_p really does mean higher true_p by construction.
    assert np.all(np.diff(calibrated) >= -1e-9)


def test_fit_calibrator_falls_back_to_identity_below_min_samples() -> None:
    raw_p = np.array([0.1, 0.9, 0.5])
    labels = np.array([0.0, 1.0, 1.0])
    weights = np.ones(3)
    cfg = CalibrationConfig(method="isotonic", min_samples=10)
    calibrator = fit_calibrator(cfg, raw_p, labels, weights)
    assert isinstance(calibrator, IdentityCalibrator)
    out = calibrator.predict(np.array([0.0, 1.0]))
    assert out[0] == pytest.approx(calibrator.eps)
    assert out[1] == pytest.approx(1.0 - calibrator.eps)


def test_fit_calibrator_falls_back_to_identity_for_single_class() -> None:
    raw_p = np.linspace(0.0, 1.0, 50)
    labels = np.ones(50)  # only one label class present
    weights = np.ones(50)
    cfg = CalibrationConfig(min_samples=5)
    calibrator = fit_calibrator(cfg, raw_p, labels, weights)
    assert isinstance(calibrator, IdentityCalibrator)


def test_fit_calibrator_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_calibrator(CalibrationConfig(), np.array([0.1, 0.2]), np.array([0.0]), np.array([1.0]))
