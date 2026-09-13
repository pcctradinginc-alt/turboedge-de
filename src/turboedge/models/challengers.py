"""Challenger signal families (Build Contract v2, W9): is there ANY real OOS edge over the null?

Six ``ForecastModel``-conforming families, one signal-family hypothesis each
(pre-registered in ``scratchpad/w9_challenger_results.md`` before any of them
were measured), probing different explanations for why the protected TSMOM
baseline lost to the trivial unconditional (null) benchmark in W4's
walk-forward measurement (worse Brier in 20/20 combinations, worse ECE in
nearly all):

- :class:`VolTargetedTsmom` -- same trend sign, vol-targeted confidence.
- :class:`LowVolRegimeTrend` -- trend gated to calm-vol regimes only.
- :class:`ShortHorizonReversal` -- 1-3 day vol-normalized mean reversion.
- :class:`VixTermStructure` -- VIX9D/VIX/VIX3M term slope as a risk filter.
- :class:`CrossAssetLeadLag` -- US close/VIX/EURUSD/rates overnight lead-lag (Master Spec §8.6).
- :class:`SeasonalityTurnOfMonth` -- turn-of-month calendar effect (deliberately simple reference).

Five of the six (everything except ``SeasonalityTurnOfMonth``) share one
small internal engine: a per-horizon sample-weighted ridge regression from a
family-specific causal feature vector to the h-day log return
(``_fit_multi_feature_horizon``/``_predict_from_fit``), with ``p_up`` from
``Phi(mean/sigma)`` recalibrated by an isotonic/Platt calibrator fit on
out-of-sample purged-CV folds *within* the training window -- the same
overall shape as ``models/directional.py``'s ``TsmomForecastModel``, kept
deliberately closed-form/fast (no per-fold hyperparameter grid search, unlike
``LogisticDirectionModel``) so a full walk-forward sweep across families x
underlyings x horizons stays tractable. ``SeasonalityTurnOfMonth`` uses a
simpler bucketed-empirical-distribution engine (like ``NullModel``, split by
calendar bucket).

``models/directional.py``, ``models/forecast.py``, ``models/protected_baseline.py``,
``backtest/*`` and ``features/{returns,volatility,trend,product}.py`` are
used read-only, never modified (Build Contract v2 file-ownership rules).
``CrossAssetLeadLag`` and ``VixTermStructure`` need auxiliary underlyings
(US indices, VIX family, EURUSD, ^TNX) that the ``ForecastModel`` protocol's
``fit``/``predict`` signature has no room for (both take only the *primary*
underlying's bars); those two classes take the auxiliary bar sequences as
constructor arguments instead, aligned onto the primary bar axis via
``turboedge.features.cross_asset.align_auxiliary_series`` -- which enforces
``available_at <= primary_bar.available_at`` per auxiliary entry, so holding
each auxiliary series' *full* history in the instance for the whole
walk-forward run is safe (never a look-ahead) and avoids re-slicing
auxiliary data on every refit.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm

from turboedge.backtest.purged_cv import PurgedWalkForwardSplit, average_uniqueness
from turboedge.features.cross_asset import (
    align_auxiliary_series,
    own_level,
    own_level_diff_1d,
    own_log_return_1d,
)
from turboedge.features.returns import bars_as_of
from turboedge.features.volatility import ewma_volatility
from turboedge.models.calibration import CalibrationConfig, Calibrator, fit_calibrator
from turboedge.models.forecast import HORIZONS, HorizonForecast
from turboedge.models.quantile import (
    QUANTILE_LEVELS,
    WeightedRidgeFit,
    expected_shortfall,
    fit_weighted_ridge,
    predict_weighted_ridge,
    weighted_quantile,
    weighted_std,
)
from turboedge.storage.schemas import UnderlyingBar

_ROUND_NDIGITS = 10
_TSMOM_LOOKBACKS: tuple[int, ...] = (21, 63, 126)
_TSMOM_CLIP = 3.0
_EWMA_LAMBDA = 0.94


# --- shared small helpers (self-contained; models/directional.py NOT imported) ---------------


def _challenger_model_hash(
    *,
    class_name: str,
    hyperparams: Mapping[str, object],
    training_end: datetime,
    feature_names: Sequence[str],
    coefficients: Mapping[str, object],
) -> str:
    """sha256 over class name, hyperparameters, training-end date, feature names and coefficients.

    Same reproducibility contract as ``models/directional.py``'s private
    ``_model_hash`` (CLAUDE.md rule 33); reimplemented locally rather than
    importing a private symbol from a sibling, not-owned module.
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


def _underlying_id_of(bars: Sequence[UnderlyingBar]) -> str:
    ids = {b.underlying_id for b in bars}
    if len(ids) != 1:
        raise ValueError(f"bars must all share one underlying_id, got {ids!r}")
    return next(iter(ids))


def _zscore_lookback(
    closes: npt.NDArray[np.float64], sigma: npt.NDArray[np.float64], k: int, clip: float
) -> npt.NDArray[np.float64]:
    """``clip(ln(close[t]/close[t-k]) / (sigma[t]*sqrt(k)), -clip, clip)``; ``NaN`` where undefined.

    Vectorized, causal (``out[t]`` only uses ``closes[..t]``/``sigma[t]``),
    the same TSMOM-style normalized-momentum building block used by every
    trend-based challenger below. ``sigma[t] <= 0`` or non-finite yields
    ``NaN`` (never a spurious clipped extreme from a division by ~0).
    """
    n = closes.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= k:
        return out
    sig_slice = sigma[k:]
    valid = np.isfinite(sig_slice) & (sig_slice > 0.0)
    raw = np.full(sig_slice.shape[0], np.nan, dtype=np.float64)
    raw[valid] = np.log(closes[k:][valid] / closes[:-k][valid]) / (sig_slice[valid] * np.sqrt(k))
    out[k:] = np.clip(raw, -clip, clip)
    return out


def _trend_score_series(
    closes: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    lookbacks: Sequence[int],
    clip: float,
) -> npt.NDArray[np.float64]:
    """Mean of clipped z-scores across ``lookbacks`` (the protected tsmom_horizon_norm_v1 shape,
    reimplemented here -- ``models/protected_baseline.py`` itself is never modified)."""
    z_stack = np.stack([_zscore_lookback(closes, sigma, k, clip) for k in lookbacks], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        score = np.nanmean(z_stack, axis=0)
    score[np.all(np.isnan(z_stack), axis=0)] = np.nan
    return score


def _trailing_tercile_gate(
    sigma: npt.NDArray[np.float64], window: int, min_periods: int, tercile: float
) -> npt.NDArray[np.float64]:
    """``1.0`` where ``sigma[t]`` is at/below the trailing ``tercile`` quantile of ``sigma``.

    ``0.0`` above it, ``NaN`` while fewer than ``min_periods`` trailing
    values are available. The rolling quantile at ``t`` only uses
    ``sigma[..t]`` (pandas ``rolling`` is causal by construction), so this
    introduces no look-ahead beyond what ``sigma`` itself already carries.
    """
    threshold = (
        pd.Series(sigma)
        .rolling(window=window, min_periods=min_periods)
        .quantile(tercile)
        .to_numpy()
    )
    valid = np.isfinite(sigma) & np.isfinite(threshold)
    gate = np.full(sigma.shape[0], np.nan, dtype=np.float64)
    gate[valid] = (sigma[valid] <= threshold[valid]).astype(np.float64)
    return gate


# --- shared multi-feature ridge + internal-OOS-calibration engine ----------------------------


@dataclass(frozen=True, slots=True)
class _HorizonFit:
    ridge: WeightedRidgeFit
    quantile_offsets: dict[str, float]
    es05_offset: float
    sigma: float
    calibrator: Calibrator
    n_train: int
    n_effective: float


def _fit_multi_feature_horizon(
    x: npt.NDArray[np.float64],
    y_ret: npt.NDArray[np.float64],
    t0: npt.NDArray[np.int64],
    t1: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64],
    *,
    h: int,
    ridge_alpha: float,
    calibration_cfg: CalibrationConfig,
    calibration_folds: int,
) -> _HorizonFit:
    """Fit ``mean_h ~ ridge(x)``, empirical residual quantiles/ES, and an internal-OOS calibrator.

    ``x``/``y_ret``/``weights`` must already be the horizon's valid training
    rows (no NaN); ``t0``/``t1`` are their label windows (``t1 = t0 + h``),
    used to purge+embargo the internal calibration folds exactly like
    ``TsmomForecastModel._internal_oos_calibration``.
    """
    fit = fit_weighted_ridge(x, y_ret, weights, alpha=ridge_alpha)
    quantile_offsets = {
        key: weighted_quantile(fit.residuals, weights, q) for key, q in QUANTILE_LEVELS.items()
    }
    es05_offset = expected_shortfall(fit.residuals, weights, level=0.05)
    sigma = weighted_std(fit.residuals, weights)

    m = x.shape[0]
    n_splits = min(calibration_folds, max(m // 20, 1))
    raw_p_parts: list[npt.NDArray[np.float64]] = []
    label_parts: list[npt.NDArray[np.float64]] = []
    w_parts: list[npt.NDArray[np.float64]] = []
    splitter = PurgedWalkForwardSplit(horizon=h, embargo=h, n_splits=n_splits)
    for train_idx, test_idx in splitter.split(t0, t1):
        if train_idx.size < 5 or test_idx.size == 0:
            continue
        fold_fit = fit_weighted_ridge(
            x[train_idx], y_ret[train_idx], weights[train_idx], alpha=ridge_alpha
        )
        fold_sigma = weighted_std(fold_fit.residuals, weights[train_idx])
        mean_test = predict_weighted_ridge(fold_fit, x[test_idx])
        if fold_sigma > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                raw_p_test = norm.cdf(mean_test / fold_sigma)
        else:
            raw_p_test = np.full(test_idx.shape[0], 0.5)
        label_test = (y_ret[test_idx] > 0.0).astype(np.float64)
        raw_p_parts.append(np.asarray(raw_p_test, dtype=np.float64))
        label_parts.append(label_test)
        w_parts.append(weights[test_idx])

    if raw_p_parts:
        raw_p_oos = np.concatenate(raw_p_parts)
        label_oos = np.concatenate(label_parts)
        w_oos = np.concatenate(w_parts)
    else:
        raw_p_oos = np.zeros(0, dtype=np.float64)
        label_oos = np.zeros(0, dtype=np.float64)
        w_oos = np.zeros(0, dtype=np.float64)
    calibrator = fit_calibrator(calibration_cfg, raw_p_oos, label_oos, w_oos)

    return _HorizonFit(
        ridge=fit,
        quantile_offsets=quantile_offsets,
        es05_offset=es05_offset,
        sigma=sigma,
        calibrator=calibrator,
        n_train=int(m),
        n_effective=float(np.sum(weights)),
    )


def _predict_from_fit(
    fit: _HorizonFit, x_now: npt.NDArray[np.float64]
) -> tuple[float, float, float, dict[str, float], float]:
    """Returns ``(mean, sigma, p_up, quantiles, expected_shortfall_05)`` for one feature row."""
    mean = float(predict_weighted_ridge(fit.ridge, x_now.reshape(1, -1))[0])
    sigma = fit.sigma
    raw_p = float(norm.cdf(mean / sigma)) if sigma > 0 else 0.5
    p_up = float(np.asarray(fit.calibrator.predict(np.array([raw_p])))[0])
    quantiles = {key: mean + val for key, val in fit.quantile_offsets.items()}
    es05 = mean + fit.es05_offset
    return mean, sigma, p_up, quantiles, es05


def _fit_all_horizons(
    closes: npt.NDArray[np.float64],
    *,
    horizons: Sequence[int],
    feature_fn: Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]],
    min_train_samples: int,
    ridge_alpha: float,
    calibration_cfg: CalibrationConfig,
    calibration_folds: int,
) -> dict[int, _HorizonFit]:
    """Loop ``_fit_multi_feature_horizon`` over every horizon, given a ``(t_idx,h)->X`` builder."""
    n = closes.shape[0]
    fits: dict[int, _HorizonFit] = {}
    for h in horizons:
        t_idx = np.arange(0, n - h, dtype=np.int64)
        feat = feature_fn(t_idx, h) if t_idx.size > 0 else np.zeros((0, 1), dtype=np.float64)
        row_valid = np.isfinite(feat).all(axis=1)
        t_valid = t_idx[row_valid]
        if t_valid.size < min_train_samples:
            raise ValueError(
                f"insufficient samples for horizon {h}d: {t_valid.size} < {min_train_samples}"
            )
        x = feat[row_valid]
        y = np.log(closes[t_valid + h] / closes[t_valid])
        weights = average_uniqueness(t_valid, t_valid + h)
        fits[h] = _fit_multi_feature_horizon(
            x,
            y,
            t_valid,
            t_valid + h,
            weights,
            h=h,
            ridge_alpha=ridge_alpha,
            calibration_cfg=calibration_cfg,
            calibration_folds=calibration_folds,
        )
    return fits


def _predict_ridge_forecasts(
    fits: Mapping[int, _HorizonFit],
    feature_fn: Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]],
    last_idx: int,
    *,
    underlying_id: str,
    as_of: datetime,
    model_id: str,
    model_hash: str,
    signal_family: str,
    horizons: Sequence[int],
) -> list[HorizonForecast]:
    forecasts: list[HorizonForecast] = []
    for h in horizons:
        if h not in fits:
            raise ValueError(f"horizon {h}d was not fit; fitted horizons: {sorted(fits)}")
        fit = fits[h]
        x_now = feature_fn(np.array([last_idx], dtype=np.int64), h)[0]
        if not np.all(np.isfinite(x_now)):
            raise ValueError(
                f"insufficient feature history to predict as of {as_of!r} for horizon {h}d"
            )
        mean, sigma, p_up, quantiles, es05 = _predict_from_fit(fit, x_now)
        n_eff = fit.n_effective
        uncertainty = sigma / np.sqrt(n_eff) if n_eff > 0 else sigma
        forecasts.append(
            HorizonForecast(
                underlying_id=underlying_id,
                horizon_days=h,
                prediction_time=as_of,
                p_up=p_up,
                mean=mean,
                sigma=sigma,
                quantiles=quantiles,
                expected_shortfall_05=es05,
                uncertainty=float(uncertainty),
                model_id=model_id,
                model_hash=model_hash,
                signal_family=signal_family,
                n_train=fit.n_train,
                n_effective=n_eff,
            )
        )
    return forecasts


def _ridge_coefficients_for_hash(fits: Mapping[int, _HorizonFit]) -> dict[str, object]:
    return {
        str(h): {"coef": fit.ridge.coef.tolist(), "intercept": fit.ridge.intercept}
        for h, fit in fits.items()
    }


# --- 1. VolTargetedTsmom ----------------------------------------------------------------------


class VolTargetedTsmomConfig(BaseModel):
    """Hyperparameters of :class:`VolTargetedTsmom`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "voltarget_tsmom_v1"
    lookbacks: tuple[int, ...] = _TSMOM_LOOKBACKS
    clip: float = _TSMOM_CLIP
    ewma_lambda: float = _EWMA_LAMBDA
    vol_multiplier_min: float = 0.3
    vol_multiplier_max: float = 3.0
    ridge_alpha: float = 1.0
    min_train_samples: int = 60
    calibration_folds: int = 4
    calibration: CalibrationConfig = CalibrationConfig()


class VolTargetedTsmom:
    """Same TSMOM sign, but the predicted-return magnitude is scaled by a vol-target multiplier.

    Build Contract v2 W9 item 2a. ``multiplier(t) = clip(target_vol / sigma(t), lo, hi)``, with
    ``target_vol`` the training window's own average EWMA daily vol (a
    training-set constant, never leaking beyond ``as_of``). Tests whether
    TSMOM's out-of-sample failure (W4) is a sizing/confidence problem rather
    than a directional one.
    """

    signal_family = "voltarget_tsmom"

    def __init__(self, config: VolTargetedTsmomConfig | None = None) -> None:
        self._config = config or VolTargetedTsmomConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._fits: dict[int, _HorizonFit] = {}
        self._target_vol: float | None = None
        self._training_end: datetime | None = None

    def _score_and_sigma(
        self, eligible: Sequence[UnderlyingBar]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        sigma = ewma_volatility(closes, lam=self._config.ewma_lambda)
        score = _trend_score_series(closes, sigma, self._config.lookbacks, self._config.clip)
        return score, sigma

    def _feature_fn(
        self, eligible: Sequence[UnderlyingBar]
    ) -> Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]]:
        score, sigma = self._score_and_sigma(eligible)
        target_vol = self._target_vol
        if target_vol is None:
            raise RuntimeError("VolTargetedTsmom.fit() must be called before predicting")
        with np.errstate(invalid="ignore", divide="ignore"):
            multiplier = np.clip(
                target_vol / sigma, self._config.vol_multiplier_min, self._config.vol_multiplier_max
            )
        invalid_sigma = ~np.isfinite(sigma) | (sigma <= 0.0)
        multiplier[invalid_sigma] = np.nan
        x_raw = score * multiplier

        def feature_fn(t_idx: npt.NDArray[np.int64], h: int) -> npt.NDArray[np.float64]:
            return (x_raw[t_idx] * sigma[t_idx] * np.sqrt(h)).reshape(-1, 1)

        return feature_fn

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        self._training_end = eligible[-1].ts if eligible else as_of
        _score, sigma = self._score_and_sigma(eligible)
        finite_sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]
        if finite_sigma.size == 0:
            raise ValueError(f"insufficient history to fit VolTargetedTsmom as of {as_of!r}")
        self._target_vol = float(np.mean(finite_sigma))
        feature_fn = self._feature_fn(eligible)
        self._fits = _fit_all_horizons(
            closes,
            horizons=HORIZONS,
            feature_fn=feature_fn,
            min_train_samples=self._config.min_train_samples,
            ridge_alpha=self._config.ridge_alpha,
            calibration_cfg=self._config.calibration,
            calibration_folds=self._config.calibration_folds,
        )
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("VolTargetedTsmom.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        feature_fn = self._feature_fn(eligible)
        return _predict_ridge_forecasts(
            self._fits,
            feature_fn,
            len(eligible) - 1,
            underlying_id=underlying_id,
            as_of=as_of,
            model_id=self.model_id,
            model_hash=self.model_hash(),
            signal_family=self.signal_family,
            horizons=horizons,
        )

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("VolTargetedTsmom.fit() must be called before model_hash()")
        return _challenger_model_hash(
            class_name="VolTargetedTsmom",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=["trend_score_voltarget"],
            coefficients={
                "target_vol": self._target_vol,
                "per_horizon": _ridge_coefficients_for_hash(self._fits),
            },
        )


# --- 2. LowVolRegimeTrend ---------------------------------------------------------------------


class LowVolRegimeTrendConfig(BaseModel):
    """Hyperparameters of :class:`LowVolRegimeTrend`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "lowvol_regime_trend_v1"
    lookbacks: tuple[int, ...] = _TSMOM_LOOKBACKS
    clip: float = _TSMOM_CLIP
    ewma_lambda: float = _EWMA_LAMBDA
    regime_window: int = 252
    regime_min_periods: int = 60
    regime_tercile: float = 1.0 / 3.0
    ridge_alpha: float = 1.0
    min_train_samples: int = 60
    calibration_folds: int = 4
    calibration: CalibrationConfig = CalibrationConfig()


class LowVolRegimeTrend:
    """The same trend score, active only in the bottom vol tercile; unconditional (via ridge
    intercept) elsewhere.

    Build Contract v2 W9 item 2b. Regime gate is a trailing (causal) rolling
    tercile of EWMA daily vol -- see :func:`_trailing_tercile_gate`. Tests
    whether trend-following's OOS failure (W4) is concentrated in
    higher-vol regimes where trend continuation is known to break down.
    """

    signal_family = "lowvol_regime_trend"

    def __init__(self, config: LowVolRegimeTrendConfig | None = None) -> None:
        self._config = config or LowVolRegimeTrendConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._fits: dict[int, _HorizonFit] = {}
        self._training_end: datetime | None = None

    def _score_sigma_gate(
        self, eligible: Sequence[UnderlyingBar]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        sigma = ewma_volatility(closes, lam=self._config.ewma_lambda)
        score = _trend_score_series(closes, sigma, self._config.lookbacks, self._config.clip)
        gate = _trailing_tercile_gate(
            sigma,
            self._config.regime_window,
            self._config.regime_min_periods,
            self._config.regime_tercile,
        )
        return score, sigma, gate

    def _feature_fn(
        self, eligible: Sequence[UnderlyingBar]
    ) -> Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]]:
        score, sigma, gate = self._score_sigma_gate(eligible)
        x_raw = score * gate

        def feature_fn(t_idx: npt.NDArray[np.int64], h: int) -> npt.NDArray[np.float64]:
            return (x_raw[t_idx] * sigma[t_idx] * np.sqrt(h)).reshape(-1, 1)

        return feature_fn

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        self._training_end = eligible[-1].ts if eligible else as_of
        feature_fn = self._feature_fn(eligible)
        self._fits = _fit_all_horizons(
            closes,
            horizons=HORIZONS,
            feature_fn=feature_fn,
            min_train_samples=self._config.min_train_samples,
            ridge_alpha=self._config.ridge_alpha,
            calibration_cfg=self._config.calibration,
            calibration_folds=self._config.calibration_folds,
        )
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("LowVolRegimeTrend.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        feature_fn = self._feature_fn(eligible)
        return _predict_ridge_forecasts(
            self._fits,
            feature_fn,
            len(eligible) - 1,
            underlying_id=underlying_id,
            as_of=as_of,
            model_id=self.model_id,
            model_hash=self.model_hash(),
            signal_family=self.signal_family,
            horizons=horizons,
        )

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("LowVolRegimeTrend.fit() must be called before model_hash()")
        return _challenger_model_hash(
            class_name="LowVolRegimeTrend",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=["trend_score_gated"],
            coefficients=_ridge_coefficients_for_hash(self._fits),
        )


# --- 3. ShortHorizonReversal ------------------------------------------------------------------


class ShortHorizonReversalConfig(BaseModel):
    """Hyperparameters of :class:`ShortHorizonReversal`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "reversal_short_horizon_v1"
    reversal_window: int = 3
    clip: float = 3.0
    ewma_lambda: float = _EWMA_LAMBDA
    ridge_alpha: float = 1.0
    min_train_samples: int = 60
    calibration_folds: int = 4
    calibration: CalibrationConfig = CalibrationConfig()


class ShortHorizonReversal:
    """Vol-normalized ``reversal_window``-day mean-reversion: negative of a short-horizon z-score.

    Build Contract v2 W9 item 2d. Master Spec §3.4 requires a *stricter*
    evidence bar for mean-reversion families specifically than for trend
    families; the walk-forward measurement in
    ``scratchpad/w9_challenger_results.md`` applies that stricter ladder
    when judging this family's results (not a change made in this module).
    """

    signal_family = "reversal_short_horizon"

    def __init__(self, config: ShortHorizonReversalConfig | None = None) -> None:
        self._config = config or ShortHorizonReversalConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._fits: dict[int, _HorizonFit] = {}
        self._training_end: datetime | None = None

    def _score_and_sigma(
        self, eligible: Sequence[UnderlyingBar]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        sigma = ewma_volatility(closes, lam=self._config.ewma_lambda)
        score = -_zscore_lookback(closes, sigma, self._config.reversal_window, self._config.clip)
        return score, sigma

    def _feature_fn(
        self, eligible: Sequence[UnderlyingBar]
    ) -> Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]]:
        score, sigma = self._score_and_sigma(eligible)

        def feature_fn(t_idx: npt.NDArray[np.int64], h: int) -> npt.NDArray[np.float64]:
            return (score[t_idx] * sigma[t_idx] * np.sqrt(h)).reshape(-1, 1)

        return feature_fn

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        self._training_end = eligible[-1].ts if eligible else as_of
        feature_fn = self._feature_fn(eligible)
        self._fits = _fit_all_horizons(
            closes,
            horizons=HORIZONS,
            feature_fn=feature_fn,
            min_train_samples=self._config.min_train_samples,
            ridge_alpha=self._config.ridge_alpha,
            calibration_cfg=self._config.calibration,
            calibration_folds=self._config.calibration_folds,
        )
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("ShortHorizonReversal.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        feature_fn = self._feature_fn(eligible)
        return _predict_ridge_forecasts(
            self._fits,
            feature_fn,
            len(eligible) - 1,
            underlying_id=underlying_id,
            as_of=as_of,
            model_id=self.model_id,
            model_hash=self.model_hash(),
            signal_family=self.signal_family,
            horizons=horizons,
        )

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("ShortHorizonReversal.fit() must be called before model_hash()")
        return _challenger_model_hash(
            class_name="ShortHorizonReversal",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=["reversal_score"],
            coefficients=_ridge_coefficients_for_hash(self._fits),
        )


# --- 4. VixTermStructure ----------------------------------------------------------------------


class VixTermStructureConfig(BaseModel):
    """Hyperparameters of :class:`VixTermStructure`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "vix_term_structure_v1"
    lookbacks: tuple[int, ...] = _TSMOM_LOOKBACKS
    clip: float = _TSMOM_CLIP
    ewma_lambda: float = _EWMA_LAMBDA
    ridge_alpha: float = 1.0
    min_train_samples: int = 80
    calibration_folds: int = 4
    calibration: CalibrationConfig = CalibrationConfig()


class VixTermStructure:
    """Trend score plus VIX9D/VIX and VIX/VIX3M term-structure slope as a risk-on/off feature.

    Build Contract v2 W9 item 2e. ``vix_bars``/``vix9d_bars``/``vix3m_bars``
    are auxiliary underlying histories (yfinance ``^VIX``/``^VIX9D``/
    ``^VIX3M``), aligned onto the primary axis via
    ``features.cross_asset.align_auxiliary_series`` (no look-ahead: each
    entry only uses the most recent auxiliary bar with
    ``available_at <= primary_bar.available_at``).
    """

    signal_family = "vix_term_structure"

    def __init__(
        self,
        vix_bars: Sequence[UnderlyingBar],
        vix9d_bars: Sequence[UnderlyingBar],
        vix3m_bars: Sequence[UnderlyingBar],
        config: VixTermStructureConfig | None = None,
    ) -> None:
        self._vix_bars = sorted(vix_bars, key=lambda b: b.ts)
        self._vix9d_bars = sorted(vix9d_bars, key=lambda b: b.ts)
        self._vix3m_bars = sorted(vix3m_bars, key=lambda b: b.ts)
        self._config = config or VixTermStructureConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._fits: dict[int, _HorizonFit] = {}
        self._training_end: datetime | None = None
        self._feature_names = ["trend_score", "vix9d_over_vix_minus_1", "vix_over_vix3m_minus_1"]

    def _score_and_sigma(
        self, eligible: Sequence[UnderlyingBar]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        sigma = ewma_volatility(closes, lam=self._config.ewma_lambda)
        score = _trend_score_series(closes, sigma, self._config.lookbacks, self._config.clip)
        return score, sigma

    def _term_features(
        self, eligible: Sequence[UnderlyingBar]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        vix_level = align_auxiliary_series(eligible, self._vix_bars, own_level(self._vix_bars))
        vix9d_level = align_auxiliary_series(
            eligible, self._vix9d_bars, own_level(self._vix9d_bars)
        )
        vix3m_level = align_auxiliary_series(
            eligible, self._vix3m_bars, own_level(self._vix3m_bars)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            term_short = np.where(vix_level > 0.0, vix9d_level / vix_level - 1.0, np.nan)
            term_long = np.where(vix3m_level > 0.0, vix_level / vix3m_level - 1.0, np.nan)
        return term_short, term_long

    def _feature_fn(
        self, eligible: Sequence[UnderlyingBar]
    ) -> Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]]:
        score, sigma = self._score_and_sigma(eligible)
        term_short, term_long = self._term_features(eligible)

        def feature_fn(t_idx: npt.NDArray[np.int64], h: int) -> npt.NDArray[np.float64]:
            trend_col = score[t_idx] * sigma[t_idx] * np.sqrt(h)
            return np.column_stack([trend_col, term_short[t_idx], term_long[t_idx]])

        return feature_fn

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        self._training_end = eligible[-1].ts if eligible else as_of
        feature_fn = self._feature_fn(eligible)
        self._fits = _fit_all_horizons(
            closes,
            horizons=HORIZONS,
            feature_fn=feature_fn,
            min_train_samples=self._config.min_train_samples,
            ridge_alpha=self._config.ridge_alpha,
            calibration_cfg=self._config.calibration,
            calibration_folds=self._config.calibration_folds,
        )
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("VixTermStructure.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        feature_fn = self._feature_fn(eligible)
        return _predict_ridge_forecasts(
            self._fits,
            feature_fn,
            len(eligible) - 1,
            underlying_id=underlying_id,
            as_of=as_of,
            model_id=self.model_id,
            model_hash=self.model_hash(),
            signal_family=self.signal_family,
            horizons=horizons,
        )

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("VixTermStructure.fit() must be called before model_hash()")
        return _challenger_model_hash(
            class_name="VixTermStructure",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=self._feature_names,
            coefficients=_ridge_coefficients_for_hash(self._fits),
        )


# --- 5. CrossAssetLeadLag ---------------------------------------------------------------------


class CrossAssetLeadLagConfig(BaseModel):
    """Hyperparameters of :class:`CrossAssetLeadLag`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "cross_asset_leadlag_v1"
    ridge_alpha: float = 1.0
    min_train_samples: int = 80
    calibration_folds: int = 4
    calibration: CalibrationConfig = CalibrationConfig()


class CrossAssetLeadLag:
    """Overnight cross-asset lead/lag: US close, VIX change, EURUSD, 10y yield change -> primary.

    Build Contract v2 W9 item 2c / Master Spec §8.6. All five auxiliary
    series are aligned onto the primary axis via
    ``features.cross_asset.align_auxiliary_series``: for a EU-morning
    prediction, that alignment naturally resolves to "yesterday's" US close
    (available only after the US close, which is after the EU close) and
    excludes the same trading day's US close, purely from each bar's own
    ``available_at`` -- no special-cased calendar logic here (see
    ``features/cross_asset.py``'s module docstring). Bund yield is
    substituted with ``^TNX`` (US 10y): no reliable free-yfinance German
    Bund-yield ticker was available in this session's time budget; global
    duration/rates moves are correlated enough for this to be a defensible
    proxy, documented explicitly as a deviation from the spec's exact list
    (pre-registered in ``scratchpad/w9_challenger_results.md`` before
    measurement, not decided after).
    """

    signal_family = "cross_asset_leadlag"

    def __init__(
        self,
        spx_bars: Sequence[UnderlyingBar],
        ndx_bars: Sequence[UnderlyingBar],
        vix_bars: Sequence[UnderlyingBar],
        eurusd_bars: Sequence[UnderlyingBar],
        tnx_bars: Sequence[UnderlyingBar],
        config: CrossAssetLeadLagConfig | None = None,
    ) -> None:
        self._spx_bars = sorted(spx_bars, key=lambda b: b.ts)
        self._ndx_bars = sorted(ndx_bars, key=lambda b: b.ts)
        self._vix_bars = sorted(vix_bars, key=lambda b: b.ts)
        self._eurusd_bars = sorted(eurusd_bars, key=lambda b: b.ts)
        self._tnx_bars = sorted(tnx_bars, key=lambda b: b.ts)
        self._config = config or CrossAssetLeadLagConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._fits: dict[int, _HorizonFit] = {}
        self._training_end: datetime | None = None
        self._feature_names = [
            "spx_ret_1d",
            "ndx_ret_1d",
            "vix_chg_1d",
            "eurusd_ret_1d",
            "tnx_chg_1d",
        ]

    def _features(self, eligible: Sequence[UnderlyingBar]) -> npt.NDArray[np.float64]:
        spx_ret = own_log_return_1d(self._spx_bars)
        ndx_ret = own_log_return_1d(self._ndx_bars)
        vix_chg = own_level_diff_1d(self._vix_bars)
        eurusd_ret = own_log_return_1d(self._eurusd_bars)
        tnx_chg = own_level_diff_1d(self._tnx_bars)
        return np.column_stack(
            [
                align_auxiliary_series(eligible, self._spx_bars, spx_ret),
                align_auxiliary_series(eligible, self._ndx_bars, ndx_ret),
                align_auxiliary_series(eligible, self._vix_bars, vix_chg),
                align_auxiliary_series(eligible, self._eurusd_bars, eurusd_ret),
                align_auxiliary_series(eligible, self._tnx_bars, tnx_chg),
            ]
        )

    def _feature_fn(
        self, eligible: Sequence[UnderlyingBar]
    ) -> Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]]:
        feat = self._features(eligible)

        def feature_fn(t_idx: npt.NDArray[np.int64], h: int) -> npt.NDArray[np.float64]:
            return feat[t_idx]

        return feature_fn

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        self._training_end = eligible[-1].ts if eligible else as_of
        feature_fn = self._feature_fn(eligible)
        self._fits = _fit_all_horizons(
            closes,
            horizons=HORIZONS,
            feature_fn=feature_fn,
            min_train_samples=self._config.min_train_samples,
            ridge_alpha=self._config.ridge_alpha,
            calibration_cfg=self._config.calibration,
            calibration_folds=self._config.calibration_folds,
        )
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("CrossAssetLeadLag.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        feature_fn = self._feature_fn(eligible)
        return _predict_ridge_forecasts(
            self._fits,
            feature_fn,
            len(eligible) - 1,
            underlying_id=underlying_id,
            as_of=as_of,
            model_id=self.model_id,
            model_hash=self.model_hash(),
            signal_family=self.signal_family,
            horizons=horizons,
        )

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("CrossAssetLeadLag.fit() must be called before model_hash()")
        return _challenger_model_hash(
            class_name="CrossAssetLeadLag",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=self._feature_names,
            coefficients=_ridge_coefficients_for_hash(self._fits),
        )


# --- 6. SeasonalityTurnOfMonth -----------------------------------------------------------------


_TURN_OF_MONTH_LOW_DAY = 3
_TURN_OF_MONTH_HIGH_DAY = 26


def _turn_of_month_bucket(ts: datetime) -> bool:
    """``True`` if ``ts`` falls in the last ~4 or first 3 calendar days of its month.

    A deliberately simple calendar-day proxy for "turn of month" (not a
    trading-day count) -- see Section 1.1's rationale in
    ``scratchpad/w9_challenger_results.md``: this family is included as a
    boringly simple reference, not because it is expected to survive costs.
    """
    day = ts.day
    return day <= _TURN_OF_MONTH_LOW_DAY or day >= _TURN_OF_MONTH_HIGH_DAY


@dataclass(frozen=True, slots=True)
class _SeasonalBucketStats:
    p_up: float
    mean: float
    sigma: float
    quantiles: dict[str, float]
    es05: float
    n_train: int
    n_effective: float


class SeasonalityTurnOfMonthConfig(BaseModel):
    """Hyperparameters of :class:`SeasonalityTurnOfMonth`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "seasonality_turn_of_month_v1"
    min_train_samples: int = 20


class SeasonalityTurnOfMonth:
    """Bucketed empirical h-day return distribution: turn-of-month calendar days vs. the rest.

    Build Contract v2 W9 item 2f. Same "unconditional empirical distribution"
    engine as ``models/directional.py``'s ``NullModel`` (weighted by average
    uniqueness), just split into two calendar buckets instead of one -- a
    deliberately simple reference family, per the task's own framing.
    """

    signal_family = "seasonality_turn_of_month"

    def __init__(self, config: SeasonalityTurnOfMonthConfig | None = None) -> None:
        self._config = config or SeasonalityTurnOfMonthConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._stats: dict[int, dict[bool, _SeasonalBucketStats]] = {}
        self._training_end: datetime | None = None

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = np.array([b.close for b in eligible], dtype=np.float64)
        dates = [b.ts for b in eligible]
        n = closes.shape[0]
        self._training_end = dates[-1] if dates else as_of
        bucket_flags = np.array([_turn_of_month_bucket(d) for d in dates], dtype=bool)

        stats: dict[int, dict[bool, _SeasonalBucketStats]] = {}
        for h in HORIZONS:
            t_idx = np.arange(0, n - h, dtype=np.int64)
            y = np.log(closes[t_idx + h] / closes[t_idx]) if t_idx.size > 0 else np.zeros(0)
            weights_all = average_uniqueness(t_idx, t_idx + h)
            bucket_stats: dict[bool, _SeasonalBucketStats] = {}
            for bucket in (True, False):
                mask = bucket_flags[t_idx] == bucket if t_idx.size > 0 else np.zeros(0, dtype=bool)
                if int(np.sum(mask)) < self._config.min_train_samples:
                    raise ValueError(
                        f"insufficient samples for horizon {h}d bucket={bucket}: "
                        f"{int(np.sum(mask))} < {self._config.min_train_samples}"
                    )
                yb = y[mask]
                wb = weights_all[mask]
                bucket_stats[bucket] = _SeasonalBucketStats(
                    p_up=float(np.average((yb > 0.0).astype(np.float64), weights=wb)),
                    mean=float(np.average(yb, weights=wb)),
                    sigma=weighted_std(yb, wb),
                    quantiles={
                        key: weighted_quantile(yb, wb, q) for key, q in QUANTILE_LEVELS.items()
                    },
                    es05=expected_shortfall(yb, wb, level=0.05),
                    n_train=int(yb.size),
                    n_effective=float(np.sum(wb)),
                )
            stats[h] = bucket_stats
        self._stats = stats
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError("SeasonalityTurnOfMonth.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        bucket = (
            _turn_of_month_bucket(eligible[-1].ts) if eligible else _turn_of_month_bucket(as_of)
        )

        forecasts: list[HorizonForecast] = []
        for h in horizons:
            if h not in self._stats:
                raise ValueError(
                    f"horizon {h}d was not fit; fitted horizons: {sorted(self._stats)}"
                )
            s = self._stats[h][bucket]
            n_eff = s.n_effective
            uncertainty = s.sigma / np.sqrt(n_eff) if n_eff > 0 else s.sigma
            forecasts.append(
                HorizonForecast(
                    underlying_id=underlying_id,
                    horizon_days=h,
                    prediction_time=as_of,
                    p_up=s.p_up,
                    mean=s.mean,
                    sigma=s.sigma,
                    quantiles=dict(s.quantiles),
                    expected_shortfall_05=s.es05,
                    uncertainty=float(uncertainty),
                    model_id=self.model_id,
                    model_hash=self.model_hash(),
                    signal_family=self.signal_family,
                    n_train=s.n_train,
                    n_effective=n_eff,
                )
            )
        return forecasts

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError("SeasonalityTurnOfMonth.fit() must be called before model_hash()")
        coefficients = {
            str(h): {str(bucket): {"p_up": s.p_up, "mean": s.mean} for bucket, s in buckets.items()}
            for h, buckets in self._stats.items()
        }
        return _challenger_model_hash(
            class_name="SeasonalityTurnOfMonth",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            feature_names=["turn_of_month_bucket"],
            coefficients=coefficients,
        )


__all__ = [
    "CrossAssetLeadLag",
    "CrossAssetLeadLagConfig",
    "LowVolRegimeTrend",
    "LowVolRegimeTrendConfig",
    "SeasonalityTurnOfMonth",
    "SeasonalityTurnOfMonthConfig",
    "ShortHorizonReversal",
    "ShortHorizonReversalConfig",
    "VixTermStructure",
    "VixTermStructureConfig",
    "VolTargetedTsmom",
    "VolTargetedTsmomConfig",
]
