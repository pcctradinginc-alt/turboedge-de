"""Weighted ridge regression and empirical (weighted) quantile/ES helpers.

Formula reference: Build Contract v2 item 3b. ``RidgeReturnModel`` is the
mean/quantile engine behind ``LogisticDirectionModel``: a ridge-regularized
linear map from standardized features to the h-day log return, whose
training residuals (weighted by average uniqueness) supply empirical
quantiles and expected shortfall around any new mean prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

#: Canonical quantile levels/keys shared by every ``HorizonForecast.quantiles``.
QUANTILE_LEVELS: dict[str, float] = {
    "q05": 0.05,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q95": 0.95,
}


def weighted_quantile(
    values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], q: float
) -> float:
    """Weighted sample quantile (Hazen-style interpolation on the weighted empirical CDF)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if v.size == 0:
        raise ValueError("values must not be empty")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be within [0, 1], got {q!r}")
    if np.sum(w) == 0.0:
        raise ValueError("weights must not all be zero")

    order = np.argsort(v, kind="stable")
    v_sorted = v[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum = cum / np.sum(w_sorted)
    return float(np.interp(q, cum, v_sorted))


def weighted_std(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    """Weighted (population, ``ddof=0``) standard deviation."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if v.size == 0 or np.sum(w) == 0.0:
        raise ValueError("values/weights must be non-empty with positive total weight")
    mean = float(np.average(v, weights=w))
    var = float(np.average((v - mean) ** 2, weights=w))
    return float(np.sqrt(max(var, 0.0)))


def expected_shortfall(
    values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], level: float = 0.05
) -> float:
    """Weighted mean of ``values`` at or below their weighted ``level``-quantile."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    q = weighted_quantile(v, w, level)
    mask = v <= q
    if not np.any(mask):
        return q
    return float(np.average(v[mask], weights=w[mask]))


@dataclass(frozen=True, slots=True)
class WeightedRidgeFit:
    coef: npt.NDArray[np.float64]
    intercept: float
    residuals: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    n_effective: float


def fit_weighted_ridge(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    alpha: float,
    fit_intercept: bool = True,
) -> WeightedRidgeFit:
    """Sample-weighted ridge regression (``sklearn.linear_model.Ridge``), plus residuals."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2-dimensional, got shape {x_arr.shape!r}")
    if x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[0] != w_arr.shape[0]:
        raise ValueError("x, y and weights must have matching first dimension")
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(x_arr, y_arr, sample_weight=w_arr)
    predicted: npt.NDArray[np.float64] = model.predict(x_arr)
    residuals = y_arr - predicted
    return WeightedRidgeFit(
        coef=np.asarray(model.coef_, dtype=np.float64).reshape(-1),
        intercept=float(model.intercept_) if fit_intercept else 0.0,
        residuals=residuals,
        weights=w_arr,
        n_effective=float(np.sum(w_arr)),
    )


def predict_weighted_ridge(
    fit: WeightedRidgeFit, x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=np.float64)
    result: npt.NDArray[np.float64] = x_arr @ fit.coef + fit.intercept
    return result


class RidgeReturnModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha: float = 1.0


class RidgeReturnModel:
    """Ridge regression from standardized features to an h-day log return.

    Fit once per horizon on (already-standardized) training features;
    exposes the training residual distribution (weighted by average
    uniqueness) for empirical quantile/ES construction around any new mean
    prediction.
    """

    def __init__(self, config: RidgeReturnModelConfig | None = None) -> None:
        self._config = config or RidgeReturnModelConfig()
        self._fit: WeightedRidgeFit | None = None

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ) -> None:
        self._fit = fit_weighted_ridge(
            x, y, sample_weight, alpha=self._config.alpha, fit_intercept=True
        )

    def _require_fit(self) -> WeightedRidgeFit:
        if self._fit is None:
            raise RuntimeError("RidgeReturnModel.fit() must be called before use")
        return self._fit

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return predict_weighted_ridge(self._require_fit(), x)

    def residual_quantiles(self) -> dict[str, float]:
        fit = self._require_fit()
        return {
            key: weighted_quantile(fit.residuals, fit.weights, q)
            for key, q in QUANTILE_LEVELS.items()
        }

    def residual_expected_shortfall_05(self) -> float:
        fit = self._require_fit()
        return expected_shortfall(fit.residuals, fit.weights, level=0.05)

    def residual_std(self) -> float:
        fit = self._require_fit()
        return weighted_std(fit.residuals, fit.weights)

    def n_effective(self) -> float:
        return self._require_fit().n_effective

    @property
    def coef(self) -> npt.NDArray[np.float64]:
        return self._require_fit().coef

    @property
    def intercept(self) -> float:
        return self._require_fit().intercept
