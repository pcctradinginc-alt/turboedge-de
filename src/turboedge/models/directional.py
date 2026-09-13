"""Directional forecast models: TSMOM-mapped, logistic+ridge, and null baselines.

Build Contract v2, W4 interface (§3.4-3.5, §7, §9-§10). Each model implements
the ``ForecastModel`` protocol (``models/forecast.py``): ``fit`` trains only
on bars with ``available_at <= as_of``; ``predict`` maps the *current*
feature/signal state (again "as of" ``as_of``) to a full predictive
distribution per horizon. Neither method ever looks at a bar beyond
``as_of`` (CLAUDE.md rules 4-5).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from turboedge.backtest.purged_cv import PurgedWalkForwardSplit, average_uniqueness
from turboedge.features.product import ewma_volatility as _ewma_daily_vol
from turboedge.features.returns import bars_as_of, build_feature_frame
from turboedge.models.calibration import CalibrationConfig, fit_calibrator
from turboedge.models.forecast import HORIZONS, HorizonForecast
from turboedge.models.protected_baseline import TsmomConfig, compute_tsmom
from turboedge.models.quantile import (
    QUANTILE_LEVELS,
    RidgeReturnModel,
    RidgeReturnModelConfig,
    expected_shortfall,
    weighted_quantile,
    weighted_std,
)
from turboedge.storage.schemas import UnderlyingBar

_ROUND_NDIGITS = 10


def _model_hash(
    *,
    class_name: str,
    hyperparams: Mapping[str, object],
    training_end: datetime,
    feature_names: Sequence[str],
    coefficients: Mapping[str, object],
) -> str:
    """sha256 over class name, hyperparameters, training-end date, feature names and coefficients.

    Rounds all float leaves in ``coefficients`` to avoid spurious hash churn
    from floating-point noise across otherwise-identical fits (CLAUDE.md rule
    33: every prediction must be fully reproducibly traceable).
    """

    def _round(obj: object) -> object:
        if isinstance(obj, float):
            return round(obj, _ROUND_NDIGITS)
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [_round(v) for v in obj]
        return obj

    payload = {
        "class_name": class_name,
        "hyperparams": hyperparams,
        "training_end": training_end.isoformat(),
        "feature_names": list(feature_names),
        "coefficients": _round(coefficients),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _closes_and_dates(
    bars: Sequence[UnderlyingBar],
) -> tuple[npt.NDArray[np.float64], list[datetime]]:
    closes = np.array([b.close for b in bars], dtype=np.float64)
    dates = [b.ts for b in bars]
    return closes, dates


def _underlying_id_of(bars: Sequence[UnderlyingBar]) -> str:
    ids = {b.underlying_id for b in bars}
    if len(ids) != 1:
        raise ValueError(f"bars must all share one underlying_id, got {ids!r}")
    return next(iter(ids))


def _tsmom_score_series(
    closes: npt.NDArray[np.float64], cfg: TsmomConfig
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorized batch equivalent of calling ``compute_tsmom(closes[:t+1], cfg)`` for every ``t``.

    ``protected_baseline.compute_tsmom`` re-derives the EWMA volatility from
    scratch on every call, over whatever prefix it is given; since that EWMA
    recursion only ever depends on earlier elements (``var[t]`` depends only
    on ``var[t-1]`` and ``log_returns[t]``), computing it once over the
    *entire* series yields exactly the same ``var[t]`` (and hence z-score) as
    computing it separately per prefix -- this is what lets this function
    batch what would otherwise be an O(n^2) repeated-recompute loop into a
    single O(n) pass (equivalence is covered by
    ``tests/models/test_directional.py::test_tsmom_score_series_matches_compute_tsmom``).
    ``protected_baseline.py`` itself is never modified (Build Contract v2:
    "protected_baseline.py NICHT verändern (nur benutzen)").

    Returns ``(scores, sigmas)`` aligned to ``closes``: ``sigmas[t]`` is the
    EWMA daily volatility "as of" bar ``t`` (the same ``sigma_t`` TSMOM
    normalizes by); both are ``NaN`` for ``t < max(cfg.lookbacks)``.
    """
    n = closes.shape[0]
    scores = np.full(n, np.nan, dtype=np.float64)
    sigmas = np.full(n, np.nan, dtype=np.float64)
    max_lb = max(cfg.lookbacks)
    if n < max_lb + 1:
        return scores, sigmas

    log_rets = np.diff(np.log(closes))  # log_rets[i] = ln(close[i+1]/close[i])
    sigma_series = _ewma_daily_vol(log_rets, lam=cfg.ewma_lambda)  # aligned: sigma "as of" bar i+1
    # sigma "as of" bar t (t >= 1) is sigma_series[t - 1].
    sigmas[1:] = sigma_series

    for t in range(max_lb, n):
        sigma_t = sigmas[t]
        if not (sigma_t > 0):
            continue
        clipped_zs = []
        for k in cfg.lookbacks:
            raw_z = float(np.log(closes[t] / closes[t - k]) / (sigma_t * np.sqrt(k)))
            clipped_zs.append(float(np.clip(raw_z, -cfg.clip, cfg.clip)))
        scores[t] = float(np.mean(clipped_zs))
    return scores, sigmas


def _fit_scalar_weighted_ridge(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    prior_n: float,
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """Single-coefficient (no intercept), sample-weighted ridge regression ``y ~ beta*x``.

    Shrinkage toward 0 is expressed as ``prior_n`` "phantom" observations at
    the typical (weighted-mean) squared magnitude of ``x`` with ``y=0`` --
    i.e. ``beta = sum(w*x*y) / (sum(w*x^2) + prior_n * mean_w(x^2))``. This
    keeps the shrinkage strength scale-invariant in ``x`` (unlike a fixed
    additive ridge penalty, which would over- or under-shrink depending on
    ``x``'s magnitude) and reduces to plain weighted OLS as the effective
    sample size grows large relative to ``prior_n``.

    Returns ``(beta, se_beta, residuals)``.
    """
    sxx = float(np.sum(weights * x**2))
    sxy = float(np.sum(weights * x * y))
    w_sum = float(np.sum(weights))
    mean_wxx = sxx / w_sum if w_sum > 0 else 0.0
    lam = prior_n * mean_wxx
    denom = sxx + lam
    beta = sxy / denom if denom > 0 else 0.0
    residuals = y - beta * x
    resid_var = float(np.average(residuals**2, weights=weights))
    se_beta = float(np.sqrt(resid_var / denom)) if denom > 0 else 0.0
    return beta, se_beta, residuals


class TsmomForecastConfig(BaseModel):
    """Hyperparameters of :class:`TsmomForecastModel`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "tsmom_forecast_v1"
    ridge_prior_n: float = 10.0
    min_train_samples: int = 60
    calibration_folds: int = 4
    calibration: CalibrationConfig = CalibrationConfig()


class TsmomForecastModel:
    """Maps the protected ``tsmom_horizon_norm_v1`` score to a full predictive distribution.

    Build Contract v2 item 3a. ``compute_tsmom``/``TsmomConfig`` (the
    protected baseline itself) are used, never modified: for each horizon
    ``h``, ``mean_h = beta_h * score_t * sigma_h`` is fit by weighted ridge
    regression (average-uniqueness sample weights, shrinkage toward 0 for a
    small effective sample), quantiles come from the empirical distribution
    of standardized training residuals (rescaled by the new ``sigma_h``),
    and ``p_up`` is ``Phi(mean_h / sigma_h)`` recalibrated by an
    isotonic/Platt calibrator fit on out-of-sample folds *within* the
    training window.
    """

    signal_family = "tsmom"

    def __init__(self, config: TsmomForecastConfig | None = None) -> None:
        self._config = config or TsmomForecastConfig()
        self.model_id = self._config.model_id
        self._tsmom_cfg = TsmomConfig()  # protected defaults -- never altered here
        self._fitted = False
        self._betas: dict[int, float] = {}
        self._se_betas: dict[int, float] = {}
        self._z_quantiles: dict[int, dict[str, float]] = {}
        self._z_es05: dict[int, float] = {}
        self._n_train: dict[int, int] = {}
        self._n_effective: dict[int, float] = {}
        self._calibrators: dict[int, object] = {}
        self._training_end: datetime | None = None
        self._underlying_id: str | None = None

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        self._underlying_id = _underlying_id_of(eligible)
        closes, dates = _closes_and_dates(eligible)
        n = closes.shape[0]
        cfg = self._tsmom_cfg
        max_lb = max(cfg.lookbacks)
        if n < max_lb + 1 + self._config.min_train_samples:
            raise ValueError(
                f"insufficient history to fit TsmomForecastModel as of {as_of!r}: "
                f"need >= {max_lb + 1 + self._config.min_train_samples} bars, got {n}"
            )
        scores, sigmas = _tsmom_score_series(closes, cfg)
        self._training_end = dates[-1]

        for h in HORIZONS:
            valid_t = np.arange(max_lb, n - h)
            mask = ~np.isnan(scores[valid_t]) & (sigmas[valid_t] > 0)
            valid_t = valid_t[mask]
            if valid_t.size < self._config.min_train_samples:
                raise ValueError(
                    f"insufficient samples for horizon {h}d: "
                    f"{valid_t.size} < {self._config.min_train_samples}"
                )
            t0 = valid_t
            t1 = valid_t + h
            weights = average_uniqueness(t0, t1)
            sigma_h = sigmas[valid_t] * np.sqrt(h)
            x = scores[valid_t] * sigma_h
            y = np.log(closes[valid_t + h] / closes[valid_t])

            beta, se_beta, resid = _fit_scalar_weighted_ridge(
                x, y, weights, prior_n=self._config.ridge_prior_n
            )
            z = resid / sigma_h
            self._betas[h] = beta
            self._se_betas[h] = se_beta
            self._n_train[h] = int(valid_t.size)
            self._n_effective[h] = float(np.sum(weights))
            self._z_quantiles[h] = {
                key: weighted_quantile(z, weights, q) for key, q in QUANTILE_LEVELS.items()
            }
            self._z_es05[h] = expected_shortfall(z, weights, level=0.05)

            raw_p_oos, label_oos, w_oos = self._internal_oos_calibration(
                valid_t=valid_t, scores=scores, sigmas=sigmas, closes=closes, h=h
            )
            self._calibrators[h] = fit_calibrator(
                self._config.calibration, raw_p_oos, label_oos, w_oos
            )
        self._fitted = True

    def _internal_oos_calibration(
        self,
        *,
        valid_t: npt.NDArray[np.int64],
        scores: npt.NDArray[np.float64],
        sigmas: npt.NDArray[np.float64],
        closes: npt.NDArray[np.float64],
        h: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Refit beta on each purged CV fold's train subset; collect OOS raw p_up + labels."""
        m = valid_t.size
        splitter = PurgedWalkForwardSplit(
            horizon=h, embargo=h, n_splits=min(self._config.calibration_folds, max(m // 20, 1))
        )
        t0_all = valid_t
        t1_all = valid_t + h
        raw_p_parts: list[npt.NDArray[np.float64]] = []
        label_parts: list[npt.NDArray[np.float64]] = []
        weight_parts: list[npt.NDArray[np.float64]] = []
        for train_idx, test_idx in splitter.split(t0_all, t1_all):
            train_t = valid_t[train_idx]
            train_weights = average_uniqueness(train_t, train_t + h)
            sigma_h_train = sigmas[train_t] * np.sqrt(h)
            x_train = scores[train_t] * sigma_h_train
            y_train = np.log(closes[train_t + h] / closes[train_t])
            beta, _, _ = _fit_scalar_weighted_ridge(
                x_train, y_train, train_weights, prior_n=self._config.ridge_prior_n
            )
            test_t = valid_t[test_idx]
            sigma_h_test = sigmas[test_t] * np.sqrt(h)
            mean_test = beta * scores[test_t] * sigma_h_test
            with np.errstate(invalid="ignore", divide="ignore"):
                raw_p_test = norm.cdf(np.where(sigma_h_test > 0, mean_test / sigma_h_test, 0.0))
            y_test = np.log(closes[test_t + h] / closes[test_t])
            label_test = (y_test > 0).astype(np.float64)
            test_weights = average_uniqueness(test_t, test_t + h)
            raw_p_parts.append(raw_p_test)
            label_parts.append(label_test)
            weight_parts.append(test_weights)

        if not raw_p_parts:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        return (
            np.concatenate(raw_p_parts),
            np.concatenate(label_parts),
            np.concatenate(weight_parts),
        )

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("TsmomForecastModel.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        closes, _dates = _closes_and_dates(eligible)
        cfg = self._tsmom_cfg
        result = compute_tsmom(closes, cfg)  # protected baseline, used as-is
        score_now = result.score
        sigma_now = result.sigma

        forecasts = []
        for h in horizons:
            if h not in self._betas:
                raise ValueError(
                    f"horizon {h}d was not fit; fitted horizons: {sorted(self._betas)}"
                )
            beta = self._betas[h]
            sigma_h = sigma_now * np.sqrt(h)
            mean = beta * score_now * sigma_h
            raw_p = float(norm.cdf(mean / sigma_h)) if sigma_h > 0 else 0.5
            calibrator = self._calibrators[h]
            p_up = float(np.asarray(calibrator.predict(np.array([raw_p])))[0])  # type: ignore[attr-defined]
            quantiles = {key: mean + sigma_h * z for key, z in self._z_quantiles[h].items()}
            es05 = mean + sigma_h * self._z_es05[h]
            uncertainty = abs(score_now * sigma_h) * self._se_betas[h]
            forecasts.append(
                HorizonForecast(
                    underlying_id=underlying_id,
                    horizon_days=h,
                    prediction_time=as_of,
                    p_up=p_up,
                    mean=float(mean),
                    sigma=float(sigma_h),
                    quantiles=quantiles,
                    expected_shortfall_05=float(es05),
                    uncertainty=float(uncertainty),
                    model_id=self.model_id,
                    model_hash=self.model_hash(),
                    signal_family=self.signal_family,
                    n_train=self._n_train[h],
                    n_effective=self._n_effective[h],
                )
            )
        return forecasts

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("TsmomForecastModel.fit() must be called before model_hash()")
        return _model_hash(
            class_name="TsmomForecastModel",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=["tsmom_score"],
            coefficients={"beta": self._betas, "se_beta": self._se_betas},
        )


class LogisticDirectionModelConfig(BaseModel):
    """Hyperparameters of :class:`LogisticDirectionModel`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "logistic_direction_v1"
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    ridge_alpha: float = 1.0
    min_train_samples: int = 80
    cv_folds: int = 3


class LogisticDirectionModel:
    """L2-logistic direction model on standardized features, plus a ridge mean/quantile engine.

    Build Contract v2 item 3b. Per horizon: a small grid over ``C`` is
    selected by (average-uniqueness-weighted) purged CV log loss on the
    training window, then the winning ``LogisticRegression`` is refit on all
    valid rows for ``p_up``; a :class:`~turboedge.models.quantile.RidgeReturnModel`
    on the same standardized features gives the mean and (mean + empirical
    training-residual quantiles) the quantiles/expected shortfall.
    """

    signal_family = "logit"

    def __init__(self, config: LogisticDirectionModelConfig | None = None) -> None:
        self._config = config or LogisticDirectionModelConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._feature_names: list[str] = []
        self._scaler_mean: npt.NDArray[np.float64] | None = None
        self._scaler_std: npt.NDArray[np.float64] | None = None
        self._logit: dict[int, LogisticRegression] = {}
        self._best_c: dict[int, float] = {}
        self._ridge: dict[int, RidgeReturnModel] = {}
        self._n_train: dict[int, int] = {}
        self._n_effective: dict[int, float] = {}
        self._training_end: datetime | None = None
        self._underlying_id: str | None = None

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        self._underlying_id = _underlying_id_of(eligible)
        dates, x_full, names = build_feature_frame(eligible)
        self._feature_names = names
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        n = x_full.shape[0]
        self._training_end = dates[-1]

        row_valid = ~np.isnan(x_full).any(axis=1)

        for h in HORIZONS:
            valid_t = np.arange(0, n - h)
            valid_t = valid_t[row_valid[valid_t]]
            if valid_t.size < self._config.min_train_samples:
                raise ValueError(
                    f"insufficient samples for horizon {h}d: "
                    f"{valid_t.size} < {self._config.min_train_samples}"
                )
            x_valid = x_full[valid_t]
            y_ret = np.log(closes[valid_t + h] / closes[valid_t])
            y_label = (y_ret > 0).astype(np.float64)
            weights = average_uniqueness(valid_t, valid_t + h)

            if self._scaler_mean is None or self._scaler_std is None:
                self._scaler_mean = np.nanmean(x_valid, axis=0)
                std = np.nanstd(x_valid, axis=0, ddof=0)
                std[std == 0.0] = 1.0
                self._scaler_std = std
            scaler_mean = self._scaler_mean
            scaler_std = self._scaler_std
            x_std = (x_valid - scaler_mean) / scaler_std

            best_c = self._select_c(x_std, y_label, weights, h)
            logit = LogisticRegression(C=best_c, max_iter=2000)
            logit.fit(x_std, y_label, sample_weight=weights)

            ridge = RidgeReturnModel(RidgeReturnModelConfig(alpha=self._config.ridge_alpha))
            ridge.fit(x_std, y_ret, weights)

            self._logit[h] = logit
            self._best_c[h] = best_c
            self._ridge[h] = ridge
            self._n_train[h] = int(valid_t.size)
            self._n_effective[h] = float(np.sum(weights))
        self._fitted = True

    def _select_c(
        self,
        x_std: npt.NDArray[np.float64],
        y_label: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        h: int,
    ) -> float:
        m = x_std.shape[0]
        n_splits = min(self._config.cv_folds, max(m // 20, 1))
        if n_splits < 1 or np.unique(y_label).size < 2:
            return self._config.c_grid[len(self._config.c_grid) // 2]
        splitter = PurgedWalkForwardSplit(horizon=h, embargo=h, n_splits=n_splits)
        t0 = np.arange(m)
        t1 = t0 + h
        best_c = self._config.c_grid[0]
        best_loss = np.inf
        for c in self._config.c_grid:
            losses = []
            for train_idx, test_idx in splitter.split(t0, t1):
                if np.unique(y_label[train_idx]).size < 2 or test_idx.size == 0:
                    continue
                model = LogisticRegression(C=c, max_iter=2000)
                model.fit(x_std[train_idx], y_label[train_idx], sample_weight=weights[train_idx])
                p = model.predict_proba(x_std[test_idx])[:, 1]
                eps = 1e-12
                p_clip = np.clip(p, eps, 1.0 - eps)
                y_test = y_label[test_idx]
                w_test = weights[test_idx]
                loss = float(
                    -np.average(
                        y_test * np.log(p_clip) + (1.0 - y_test) * np.log(1.0 - p_clip),
                        weights=w_test,
                    )
                )
                losses.append(loss)
            if losses:
                mean_loss = float(np.mean(losses))
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_c = c
        return best_c

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("LogisticDirectionModel.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        _dates, x_full, names = build_feature_frame(eligible)
        if names != self._feature_names:
            raise ValueError("feature set at predict time does not match the fitted feature set")
        x_last = x_full[-1]
        if np.isnan(x_last).any():
            raise ValueError(f"insufficient feature history to predict as of {as_of!r}")
        assert self._scaler_mean is not None and self._scaler_std is not None
        x_std = ((x_last - self._scaler_mean) / self._scaler_std).reshape(1, -1)

        forecasts = []
        for h in horizons:
            if h not in self._logit:
                raise ValueError(
                    f"horizon {h}d was not fit; fitted horizons: {sorted(self._logit)}"
                )
            p_up = float(self._logit[h].predict_proba(x_std)[0, 1])
            ridge = self._ridge[h]
            mean = float(ridge.predict(x_std)[0])
            resid_quantiles = ridge.residual_quantiles()
            quantiles = {key: mean + val for key, val in resid_quantiles.items()}
            es05 = mean + ridge.residual_expected_shortfall_05()
            sigma = ridge.residual_std()
            n_eff = self._n_effective[h]
            uncertainty = sigma / np.sqrt(n_eff) if n_eff > 0 else sigma
            forecasts.append(
                HorizonForecast(
                    underlying_id=underlying_id,
                    horizon_days=h,
                    prediction_time=as_of,
                    p_up=p_up,
                    mean=mean,
                    sigma=float(sigma),
                    quantiles=quantiles,
                    expected_shortfall_05=float(es05),
                    uncertainty=float(uncertainty),
                    model_id=self.model_id,
                    model_hash=self.model_hash(),
                    signal_family=self.signal_family,
                    n_train=self._n_train[h],
                    n_effective=n_eff,
                )
            )
        return forecasts

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("LogisticDirectionModel.fit() must be called before model_hash()")
        coefficients = {
            str(h): {
                "logit_coef": self._logit[h].coef_.reshape(-1).tolist(),
                "logit_intercept": float(self._logit[h].intercept_[0]),
                "best_c": self._best_c[h],
                "ridge_coef": self._ridge[h].coef.tolist(),
                "ridge_intercept": self._ridge[h].intercept,
            }
            for h in self._logit
        }
        return _model_hash(
            class_name="LogisticDirectionModel",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=self._feature_names,
            coefficients=coefficients,
        )


class NullModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "null_v1"
    min_train_samples: int = 20


class NullModel:
    """Unconditional benchmark: the training window's own realized h-day return distribution.

    Build Contract v2 item 3c. Ignores every feature; ``p_up``/``mean``/
    quantiles are the (average-uniqueness-weighted) unconditional empirical
    distribution of h-day log returns observed in the training window, the
    same for any ``as_of`` bar's feature state. This is the "does the
    challenger actually beat doing nothing clever" reference.
    """

    signal_family = "null"

    def __init__(self, config: NullModelConfig | None = None) -> None:
        self._config = config or NullModelConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._p_up: dict[int, float] = {}
        self._mean: dict[int, float] = {}
        self._sigma: dict[int, float] = {}
        self._quantiles: dict[int, dict[str, float]] = {}
        self._es05: dict[int, float] = {}
        self._n_train: dict[int, int] = {}
        self._n_effective: dict[int, float] = {}
        self._training_end: datetime | None = None

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)  # validates a single underlying_id
        closes, dates = _closes_and_dates(eligible)
        n = closes.shape[0]
        self._training_end = dates[-1]

        for h in HORIZONS:
            valid_t = np.arange(0, n - h)
            if valid_t.size < self._config.min_train_samples:
                raise ValueError(
                    f"insufficient samples for horizon {h}d: "
                    f"{valid_t.size} < {self._config.min_train_samples}"
                )
            y = np.log(closes[valid_t + h] / closes[valid_t])
            weights = average_uniqueness(valid_t, valid_t + h)
            self._p_up[h] = float(np.average((y > 0).astype(np.float64), weights=weights))
            self._mean[h] = float(np.average(y, weights=weights))
            self._sigma[h] = weighted_std(y, weights)
            self._quantiles[h] = {
                key: weighted_quantile(y, weights, q) for key, q in QUANTILE_LEVELS.items()
            }
            self._es05[h] = expected_shortfall(y, weights, level=0.05)
            self._n_train[h] = int(valid_t.size)
            self._n_effective[h] = float(np.sum(weights))
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("NullModel.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)

        forecasts = []
        for h in horizons:
            if h not in self._p_up:
                raise ValueError(f"horizon {h}d was not fit; fitted horizons: {sorted(self._p_up)}")
            n_eff = self._n_effective[h]
            uncertainty = self._sigma[h] / np.sqrt(n_eff) if n_eff > 0 else self._sigma[h]
            forecasts.append(
                HorizonForecast(
                    underlying_id=underlying_id,
                    horizon_days=h,
                    prediction_time=as_of,
                    p_up=self._p_up[h],
                    mean=self._mean[h],
                    sigma=self._sigma[h],
                    quantiles=dict(self._quantiles[h]),
                    expected_shortfall_05=self._es05[h],
                    uncertainty=float(uncertainty),
                    model_id=self.model_id,
                    model_hash=self.model_hash(),
                    signal_family=self.signal_family,
                    n_train=self._n_train[h],
                    n_effective=n_eff,
                )
            )
        return forecasts

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("NullModel.fit() must be called before model_hash()")
        return _model_hash(
            class_name="NullModel",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=[],
            coefficients={"p_up": self._p_up, "mean": self._mean},
        )
