from __future__ import annotations

import numpy as np
import pytest

from turboedge.models.quantile import (
    QUANTILE_LEVELS,
    RidgeReturnModel,
    RidgeReturnModelConfig,
    expected_shortfall,
    fit_weighted_ridge,
    weighted_quantile,
    weighted_std,
)


def test_quantile_levels_keys() -> None:
    assert set(QUANTILE_LEVELS) == {"q05", "q25", "q50", "q75", "q95"}


def test_weighted_quantile_equal_weights_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=1000)
    weights = np.ones_like(values)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        wq = weighted_quantile(values, weights, q)
        npq = np.quantile(values, q)
        assert wq == pytest.approx(npq, abs=0.05)  # interpolation conventions differ slightly


def test_weighted_quantile_hand_example() -> None:
    values = np.array([10.0, 20.0, 30.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert weighted_quantile(values, weights, 0.5) == pytest.approx(20.0)


def test_weighted_quantile_concentrates_on_heavy_weight() -> None:
    values = np.array([1.0, 100.0])
    weights = np.array([1000.0, 1.0])
    median = weighted_quantile(values, weights, 0.5)
    assert median == pytest.approx(1.0, abs=0.5)


def test_weighted_quantile_rejects_bad_q() -> None:
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0]), np.array([1.0]), 1.5)


def test_weighted_std_matches_unweighted_for_uniform_weights() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    expected = float(np.std(values, ddof=0))
    assert weighted_std(values, weights) == pytest.approx(expected)


def test_expected_shortfall_is_below_quantile() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=2000)
    weights = np.ones_like(values)
    q05 = weighted_quantile(values, weights, 0.05)
    es = expected_shortfall(values, weights, level=0.05)
    assert es <= q05


def test_fit_weighted_ridge_recovers_known_slope_noiseless() -> None:
    n = 200
    x = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = 3.0 * x[:, 0] + 2.0
    weights = np.ones(n)
    fit = fit_weighted_ridge(x, y, weights, alpha=1e-8, fit_intercept=True)
    assert fit.coef[0] == pytest.approx(3.0, abs=1e-3)
    assert fit.intercept == pytest.approx(2.0, abs=1e-3)
    assert np.allclose(fit.residuals, 0.0, atol=1e-3)
    assert fit.n_effective == pytest.approx(n)


def test_ridge_return_model_predict_and_quantiles() -> None:
    n = 300
    rng = np.random.default_rng(2)
    x = rng.normal(size=(n, 3))
    y = 0.01 * x[:, 0] - 0.005 * x[:, 1] + rng.normal(0.0, 0.001, size=n)
    weights = np.ones(n)
    model = RidgeReturnModel(RidgeReturnModelConfig(alpha=1.0))
    model.fit(x, y, weights)
    pred = model.predict(x[:5])
    assert pred.shape == (5,)
    quantiles = model.residual_quantiles()
    assert set(quantiles) == {"q05", "q25", "q50", "q75", "q95"}
    ordered = [quantiles[k] for k in ("q05", "q25", "q50", "q75", "q95")]
    assert ordered == sorted(ordered)
    assert model.residual_std() >= 0.0
    assert model.n_effective() == pytest.approx(n)


def test_ridge_return_model_requires_fit_before_use() -> None:
    model = RidgeReturnModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 2)))
