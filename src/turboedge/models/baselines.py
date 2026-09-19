"""Phase D pre-registered baseline distributional models.

``docs/measured_results.md`` Phase D ("Bewertungsziel auf die Verteilung
umstellen"): before any new challenger is measured, four baselines are
measured in this fixed order --

    a) unconditional empirical distribution        -- ``models.directional.NullModel`` (unchanged)
    b) regime-conditional empirical distribution    -- :class:`RegimeConditionalEmpiricalModel`
    c) regularized linear location model            -- :class:`RegularizedLinearLocationModel`
    d) robust location-scale (Student-t) model       -- :class:`RobustLocationScaleModel`

All three new classes here share one modeling idea, distinct from every
existing model in ``models/directional.py``/``models/challengers.py``: they
fit their location/scale/shape parameters on the *normalized* per-horizon
target

    y_h(t) = ln(P_t+h / P_t) / (sigma_t * sqrt(h))

(``features/volatility.py::normalized_horizon_target``, ``sigma_t`` the
causal EWMA volatility "as of" bar ``t``, ``features/volatility.py::ewma_volatility``)
instead of the raw log return. Pooling in this normalized space is what lets
a regime-conditional or unconditional historical sample -- collected across
very different volatility periods -- be combined coherently; every model
here converts its normalized-space forecast back into an ordinary
``HorizonForecast`` (mean/sigma/quantiles/ES in ordinary log-return units,
via the *current* ``sigma_t``) before returning, so every existing
downstream consumer (``backtest/walkforward.py``, ``pipeline/scan.py``'s
path/payoff simulation) is completely unaffected by this internal
normalization -- Master Spec/CLAUDE.md rule 4-5 (no look-ahead) and the
Phase D brief's explicit requirement to convert back to a return
distribution before the payoff engine.

``models/protected_baseline.py`` (the ``tsmom_horizon_norm_v1`` score) is
never imported or modified here -- these are independent baselines, not a
mapping of the protected score (that mapping already exists and is measured
separately as ``TsmomForecastModel``); :class:`RegularizedLinearLocationModel`
re-derives the *same shape* of trend z-score locally (as
``models/challengers.py`` already does for its own challengers), never the
protected module's output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.stats import t as student_t

from turboedge.backtest.purged_cv import average_uniqueness
from turboedge.features.returns import bars_as_of
from turboedge.features.volatility import ewma_volatility, normalized_horizon_target
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


def _underlying_id_of(bars: Sequence[UnderlyingBar]) -> str:
    ids = {b.underlying_id for b in bars}
    if len(ids) != 1:
        raise ValueError(f"bars must all share one underlying_id, got {ids!r}")
    return next(iter(ids))


def _closes(bars: Sequence[UnderlyingBar]) -> npt.NDArray[np.float64]:
    return np.array([b.close for b in bars], dtype=np.float64)


def _baseline_model_hash(
    *,
    class_name: str,
    hyperparams: Mapping[str, object],
    training_end: datetime,
    coefficients: Mapping[str, object],
) -> str:
    """sha256 over class name, hyperparameters, training-end date and coefficients.

    Same reproducibility contract as ``models/directional.py``'s
    ``_model_hash``/``models/challengers.py``'s ``_challenger_model_hash``
    (CLAUDE.md rule 33), reimplemented locally per that same established
    per-module convention rather than importing a private cross-module helper.
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
        "coefficients": _round(coefficients),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _empirical_distribution(
    y: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> tuple[float, float, dict[str, float], float, float]:
    """``(p_up, mean, quantiles, es05, sigma)`` of the weighted empirical distribution of ``y``."""
    p_up = float(np.average((y > 0.0).astype(np.float64), weights=weights))
    mean = float(np.average(y, weights=weights))
    quantiles = {key: weighted_quantile(y, weights, q) for key, q in QUANTILE_LEVELS.items()}
    es05 = expected_shortfall(y, weights, level=0.05)
    sigma = weighted_std(y, weights)
    return p_up, mean, quantiles, es05, sigma


# ---------------------------------------------------------------------------
# b) Regime-conditional empirical distribution
# ---------------------------------------------------------------------------


def _regime_bucket_series(
    sigma: npt.NDArray[np.float64], window: int, min_periods: int, low_edge: float, high_edge: float
) -> npt.NDArray[np.float64]:
    """Causal, pre-registered 3-bucket regime label per bar: ``0.0`` (low vol), ``1.0`` (mid),
    ``2.0`` (high vol), ``NaN`` while fewer than ``min_periods`` trailing values are available.

    Buckets are a trailing rolling ``low_edge``/``high_edge`` quantile of
    ``sigma`` itself (same technique as ``models/challengers.py``'s
    ``_trailing_tercile_gate``, extended from one threshold to two so the
    bucket *scheme* -- three pre-registered volatility regimes -- is fixed in
    advance rather than picked post-hoc; the *threshold values* are trailing
    and causal like every other rolling feature in this codebase, never
    computed using bars beyond the current one).
    """
    s = pd.Series(sigma)
    low_thr = s.rolling(window=window, min_periods=min_periods).quantile(low_edge).to_numpy()
    high_thr = s.rolling(window=window, min_periods=min_periods).quantile(high_edge).to_numpy()
    valid = np.isfinite(sigma) & np.isfinite(low_thr) & np.isfinite(high_thr)
    bucket = np.where(sigma <= low_thr, 0.0, np.where(sigma >= high_thr, 2.0, 1.0))
    out = np.full(sigma.shape[0], np.nan, dtype=np.float64)
    out[valid] = bucket[valid]
    return out


class RegimeConditionalEmpiricalConfig(BaseModel):
    """Hyperparameters of :class:`RegimeConditionalEmpiricalModel`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "regime_conditional_empirical_v1"
    ewma_lambda: float = 0.94
    regime_window: int = 252
    regime_min_periods: int = 60
    low_edge: float = 1.0 / 3.0
    high_edge: float = 2.0 / 3.0
    min_train_samples: int = 40
    #: Below this many samples in the current regime bucket, fall back to the
    #: unconditional (all-bucket) empirical distribution rather than fit a
    #: noisy, tiny-sample conditional one -- never silently promoted back
    #: once thin, always the honest unconditional fallback.
    min_bucket_samples: int = 30


class RegimeConditionalEmpiricalModel:
    """Baseline (b): the empirical distribution of the normalized target ``y_h``, conditioned on
    a trailing, pre-registered 3-bucket volatility regime (:func:`_regime_bucket_series`).

    Falls back to the fully unconditional empirical distribution of ``y_h``
    (baseline (a)'s own construction, just in normalized-target space) when
    the current regime has too few training samples, or when the regime
    itself cannot be determined (insufficient trailing history) -- see
    :attr:`RegimeConditionalEmpiricalConfig.min_bucket_samples`.
    """

    signal_family = "regime_conditional_empirical"

    def __init__(self, config: RegimeConditionalEmpiricalConfig | None = None) -> None:
        self._config = config or RegimeConditionalEmpiricalConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        # Per horizon: {bucket_label (0.0/1.0/2.0): (p_up, mean, quantiles, es05, sigma, n, n_eff)}
        self._by_bucket: dict[
            int, dict[float, tuple[float, float, dict[str, float], float, float, int, float]]
        ] = {}
        self._unconditional: dict[
            int, tuple[float, float, dict[str, float], float, float, int, float]
        ] = {}
        self._training_end: datetime | None = None

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = _closes(eligible)
        n = closes.shape[0]
        cfg = self._config
        sigma = ewma_volatility(closes, lam=cfg.ewma_lambda)
        regime = _regime_bucket_series(
            sigma, cfg.regime_window, cfg.regime_min_periods, cfg.low_edge, cfg.high_edge
        )
        self._training_end = eligible[-1].ts if eligible else as_of

        for h in HORIZONS:
            y = normalized_horizon_target(closes, sigma, h)
            valid_t = np.arange(0, n - h)
            mask = np.isfinite(y[valid_t]) & np.isfinite(regime[valid_t])
            valid_t = valid_t[mask]
            if valid_t.size < cfg.min_train_samples:
                raise ValueError(
                    f"insufficient samples for horizon {h}d: "
                    f"{valid_t.size} < {cfg.min_train_samples}"
                )
            y_valid = y[valid_t]
            w_all = average_uniqueness(valid_t, valid_t + h)
            self._unconditional[h] = (
                *_empirical_distribution(y_valid, w_all),
                int(valid_t.size),
                float(np.sum(w_all)),
            )

            buckets: dict[
                float, tuple[float, float, dict[str, float], float, float, int, float]
            ] = {}
            for bucket_label in (0.0, 1.0, 2.0):
                bucket_mask = regime[valid_t] == bucket_label
                bt = valid_t[bucket_mask]
                if bt.size < cfg.min_bucket_samples:
                    continue
                y_b = y[bt]
                w_b = average_uniqueness(bt, bt + h)
                buckets[bucket_label] = (
                    *_empirical_distribution(y_b, w_b),
                    int(bt.size),
                    float(np.sum(w_b)),
                )
            self._by_bucket[h] = buckets
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError(
                "RegimeConditionalEmpiricalModel.fit() must be called before predict()"
            )
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        closes = _closes(eligible)
        cfg = self._config
        sigma = ewma_volatility(closes, lam=cfg.ewma_lambda)
        sigma_now = float(sigma[-1])
        if not (sigma_now > 0):
            raise ValueError(f"EWMA volatility as of {as_of!r} is zero/NaN; cannot normalize")
        # Rolling quantile bucketing only ever looks back `regime_window` bars,
        # so restricting to that trailing slice before recomputing it gives
        # the exact same last-index bucket in O(window) instead of O(n) --
        # this is called once per out-of-sample point in
        # backtest/walkforward.py, so the difference matters at scale.
        sigma_tail = sigma[-cfg.regime_window :]
        regime_tail = _regime_bucket_series(
            sigma_tail, cfg.regime_window, cfg.regime_min_periods, cfg.low_edge, cfg.high_edge
        )
        bucket_now = float(regime_tail[-1]) if np.isfinite(regime_tail[-1]) else None

        forecasts = []
        for h in horizons:
            if h not in self._unconditional:
                raise ValueError(
                    f"horizon {h}d was not fit; fitted horizons: {sorted(self._unconditional)}"
                )
            chosen = None
            if bucket_now is not None:
                chosen = self._by_bucket.get(h, {}).get(bucket_now)
            if chosen is None:
                chosen = self._unconditional[h]
            p_up, mean_y, quantiles_y, es05_y, sigma_y, n_train, n_eff = chosen
            sigma_h = sigma_now * np.sqrt(h)
            quantiles = {key: val * sigma_h for key, val in quantiles_y.items()}
            uncertainty = (sigma_y * sigma_h) / np.sqrt(n_eff) if n_eff > 0 else sigma_y * sigma_h
            forecasts.append(
                HorizonForecast(
                    underlying_id=underlying_id,
                    horizon_days=h,
                    prediction_time=as_of,
                    p_up=p_up,
                    mean=float(mean_y * sigma_h),
                    sigma=float(sigma_y * sigma_h),
                    quantiles=quantiles,
                    expected_shortfall_05=float(es05_y * sigma_h),
                    uncertainty=float(uncertainty),
                    model_id=self.model_id,
                    model_hash=self.model_hash(),
                    signal_family=self.signal_family,
                    n_train=n_train,
                    n_effective=n_eff,
                )
            )
        return forecasts

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError(
                "RegimeConditionalEmpiricalModel.fit() must be called before model_hash()"
            )
        return _baseline_model_hash(
            class_name="RegimeConditionalEmpiricalModel",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            coefficients={
                "unconditional_mean": {str(h): v[1] for h, v in self._unconditional.items()},
                "bucket_mean": {
                    str(h): {str(b): v[1] for b, v in buckets.items()}
                    for h, buckets in self._by_bucket.items()
                },
            },
        )


# ---------------------------------------------------------------------------
# c) Regularized linear location model
# ---------------------------------------------------------------------------

_TREND_LOOKBACKS: tuple[int, ...] = (21, 63, 126)
_TREND_CLIP = 3.0


def _trend_zscore_series(
    closes: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    lookbacks: Sequence[int],
    clip: float,
) -> npt.NDArray[np.float64]:
    """Mean of clipped, EWMA-vol-normalized log-return z-scores across ``lookbacks``, vectorized.

    Same *shape* as the protected ``tsmom_horizon_norm_v1`` score, re-derived
    locally (``models/protected_baseline.py`` is used elsewhere but never
    imported here) -- the same convention ``models/challengers.py`` already
    follows for its own independent challengers. ``out[t]`` is ``NaN`` unless
    every lookback has enough history *and* ``sigma[t] > 0`` (plain
    ``np.mean``, not ``nanmean``, so a single invalid lookback correctly
    NaNs the whole score, matching the un-vectorized reference semantics
    this replaces).
    """
    n = closes.shape[0]
    sigma_positive = sigma > 0.0
    z_stack = np.full((len(lookbacks), n), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        for row, k in enumerate(lookbacks):
            if n <= k:
                continue
            raw = np.log(closes[k:] / closes[: n - k]) / (sigma[k:] * np.sqrt(k))
            z_stack[row, k:] = np.where(sigma_positive[k:], np.clip(raw, -clip, clip), np.nan)
    return np.asarray(np.mean(z_stack, axis=0), dtype=np.float64)


def _trend_zscore_now(
    closes: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    lookbacks: Sequence[int],
    clip: float,
) -> float:
    """:func:`_trend_zscore_series`'s value at the *last* index only, computed directly in
    ``O(len(lookbacks))`` instead of rebuilding the full ``O(n)`` history.

    Used by :meth:`RegularizedLinearLocationModel.predict`, which is called
    once per out-of-sample point in ``backtest/walkforward.py`` -- the same
    efficiency trick ``models/protected_baseline.py::compute_tsmom`` already
    relies on for exactly this reason.
    """
    n = closes.shape[0]
    t = n - 1
    sigma_t = sigma[t]
    if not (sigma_t > 0):
        return float("nan")
    zs: list[float] = []
    for k in lookbacks:
        if t - k < 0:
            return float("nan")
        raw = float(np.log(closes[t] / closes[t - k]) / (sigma_t * np.sqrt(k)))
        zs.append(float(np.clip(raw, -clip, clip)))
    return float(np.mean(zs))


class RegularizedLinearLocationConfig(BaseModel):
    """Hyperparameters of :class:`RegularizedLinearLocationModel`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "regularized_linear_location_v1"
    lookbacks: tuple[int, ...] = _TREND_LOOKBACKS
    clip: float = _TREND_CLIP
    ewma_lambda: float = 0.94
    ridge_alpha: float = 5.0
    min_train_samples: int = 60


class RegularizedLinearLocationModel:
    """Baseline (c): ridge regression of the normalized target ``y_h`` on a single, pre-registered
    causal trend z-score (same shape as the protected TSMOM score, re-derived independently).

    A deliberately simple, strongly regularized location model -- one
    feature, no internal probability recalibration (unlike
    ``TsmomForecastModel``/``LogisticDirectionModel``) -- meant as a
    pre-registered baseline to beat, not a challenger.
    """

    signal_family = "regularized_linear_location"

    def __init__(self, config: RegularizedLinearLocationConfig | None = None) -> None:
        self._config = config or RegularizedLinearLocationConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._fits: dict[int, WeightedRidgeFit] = {}
        self._quantile_offsets: dict[int, dict[str, float]] = {}
        self._es05_offset: dict[int, float] = {}
        self._sigma_y: dict[int, float] = {}
        self._training_end: datetime | None = None

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = _closes(eligible)
        n = closes.shape[0]
        cfg = self._config
        sigma = ewma_volatility(closes, lam=cfg.ewma_lambda)
        score = _trend_zscore_series(closes, sigma, cfg.lookbacks, cfg.clip)
        self._training_end = eligible[-1].ts if eligible else as_of

        for h in HORIZONS:
            y = normalized_horizon_target(closes, sigma, h)
            valid_t = np.arange(0, n - h)
            mask = np.isfinite(y[valid_t]) & np.isfinite(score[valid_t])
            valid_t = valid_t[mask]
            if valid_t.size < cfg.min_train_samples:
                raise ValueError(
                    f"insufficient samples for horizon {h}d: "
                    f"{valid_t.size} < {cfg.min_train_samples}"
                )
            x = score[valid_t].reshape(-1, 1)
            y_valid = y[valid_t]
            weights = average_uniqueness(valid_t, valid_t + h)
            fit = fit_weighted_ridge(x, y_valid, weights, alpha=cfg.ridge_alpha, fit_intercept=True)
            self._fits[h] = fit
            self._quantile_offsets[h] = {
                key: weighted_quantile(fit.residuals, fit.weights, q)
                for key, q in QUANTILE_LEVELS.items()
            }
            self._es05_offset[h] = expected_shortfall(fit.residuals, fit.weights, level=0.05)
            self._sigma_y[h] = weighted_std(fit.residuals, fit.weights)
        self._fitted = True

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]:
        if not self._fitted:
            raise RuntimeError(
                "RegularizedLinearLocationModel.fit() must be called before predict()"
            )
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        closes = _closes(eligible)
        cfg = self._config
        sigma = ewma_volatility(closes, lam=cfg.ewma_lambda)
        sigma_now = float(sigma[-1])
        if not (sigma_now > 0):
            raise ValueError(f"EWMA volatility as of {as_of!r} is zero/NaN; cannot normalize")
        score_now = _trend_zscore_now(closes, sigma, cfg.lookbacks, cfg.clip)
        if not np.isfinite(score_now):
            raise ValueError(f"insufficient trend-score history to predict as of {as_of!r}")

        forecasts = []
        for h in horizons:
            if h not in self._fits:
                raise ValueError(f"horizon {h}d was not fit; fitted horizons: {sorted(self._fits)}")
            fit = self._fits[h]
            mean_y = float(predict_weighted_ridge(fit, np.array([[score_now]]))[0])
            sigma_h = sigma_now * np.sqrt(h)
            quantiles = {
                key: (mean_y + off) * sigma_h for key, off in self._quantile_offsets[h].items()
            }
            es05 = (mean_y + self._es05_offset[h]) * sigma_h
            sigma_y = self._sigma_y[h]
            n_eff = fit.n_effective
            p_up = float(
                np.average(((mean_y + fit.residuals) > 0.0).astype(np.float64), weights=fit.weights)
            )
            uncertainty = (sigma_y * sigma_h) / np.sqrt(n_eff) if n_eff > 0 else sigma_y * sigma_h
            forecasts.append(
                HorizonForecast(
                    underlying_id=underlying_id,
                    horizon_days=h,
                    prediction_time=as_of,
                    p_up=float(np.clip(p_up, 0.0, 1.0)),
                    mean=float(mean_y * sigma_h),
                    sigma=float(sigma_y * sigma_h),
                    quantiles=quantiles,
                    expected_shortfall_05=float(es05),
                    uncertainty=float(uncertainty),
                    model_id=self.model_id,
                    model_hash=self.model_hash(),
                    signal_family=self.signal_family,
                    n_train=int(fit.residuals.shape[0]),
                    n_effective=n_eff,
                )
            )
        return forecasts

    def model_hash(self) -> str:
        if not self._fitted or self._training_end is None:
            raise RuntimeError(
                "RegularizedLinearLocationModel.fit() must be called before model_hash()"
            )
        return _baseline_model_hash(
            class_name="RegularizedLinearLocationModel",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            coefficients={
                str(h): {"coef": fit.coef.tolist(), "intercept": fit.intercept}
                for h, fit in self._fits.items()
            },
        )


# ---------------------------------------------------------------------------
# d) Robust location-scale (Student-t) model
# ---------------------------------------------------------------------------


def _student_t_expected_shortfall(loc: float, scale: float, df: float, alpha: float) -> float:
    """Analytic expected shortfall of a location-scale Student-t distribution at level ``alpha``.

    Standard closed form (McNeil, Frey & Embrechts, "Quantitative Risk
    Management", eq. 2.28), requires ``df > 1``.
    """
    x_alpha = float(student_t.ppf(alpha, df))
    dens = float(student_t.pdf(x_alpha, df))
    es_standard = -(dens / alpha) * (df + x_alpha**2) / (df - 1.0)
    return loc + scale * es_standard


class RobustLocationScaleConfig(BaseModel):
    """Hyperparameters of :class:`RobustLocationScaleModel`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = "robust_location_scale_t_v1"
    ewma_lambda: float = 0.94
    #: Fixed, pre-registered degrees of freedom -- never fit from data (that
    #: would just be another parametric estimation step to overfit on).
    #: df=5 is a standard, moderately fat-tailed choice for daily financial
    #: returns (Praetz 1972; Blattberg & Gonedes 1974).
    student_t_df: float = 5.0
    min_train_samples: int = 40

    @field_validator("student_t_df")
    @classmethod
    def _df_must_exceed_two(cls, v: float) -> float:
        if not (v > 2.0):
            raise ValueError(f"student_t_df must be > 2 (finite variance/ES), got {v!r}")
        return v


class RobustLocationScaleModel:
    """Baseline (d): robust (median/MAD) location-scale fit of the normalized target ``y_h``,
    mapped onto a fixed-shape Student-t distribution for quantiles/ES.

    Unconditional (like baseline (a)), but robust to outliers in both
    location and scale, and with heavier, non-Gaussian tails than
    ``NullModel``'s purely empirical quantiles allow at small sample sizes.
    """

    signal_family = "robust_location_scale_t"

    def __init__(self, config: RobustLocationScaleConfig | None = None) -> None:
        self._config = config or RobustLocationScaleConfig()
        self.model_id = self._config.model_id
        self._fitted = False
        self._loc: dict[int, float] = {}
        self._scale: dict[int, float] = {}
        self._n_train: dict[int, int] = {}
        self._n_effective: dict[int, float] = {}
        self._training_end: datetime | None = None

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None:
        eligible = bars_as_of(bars, as_of)
        _underlying_id_of(eligible)
        closes = _closes(eligible)
        n = closes.shape[0]
        cfg = self._config
        sigma = ewma_volatility(closes, lam=cfg.ewma_lambda)
        self._training_end = eligible[-1].ts if eligible else as_of

        for h in HORIZONS:
            y = normalized_horizon_target(closes, sigma, h)
            valid_t = np.arange(0, n - h)
            valid_t = valid_t[np.isfinite(y[valid_t])]
            if valid_t.size < cfg.min_train_samples:
                raise ValueError(
                    f"insufficient samples for horizon {h}d: "
                    f"{valid_t.size} < {cfg.min_train_samples}"
                )
            y_valid = y[valid_t]
            weights = average_uniqueness(valid_t, valid_t + h)
            loc = weighted_quantile(y_valid, weights, 0.5)
            mad = weighted_quantile(np.abs(y_valid - loc), weights, 0.5)
            # Normal-consistent MAD-to-scale constant (1.4826), applied here as a
            # simple, standard robust scale estimator for the Student-t scale
            # parameter -- not an exact MLE of the t-scale, but a well-known,
            # cheap, breakdown-robust approximation.
            scale = max(mad * 1.4826, 1e-8)
            self._loc[h] = loc
            self._scale[h] = scale
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
            raise RuntimeError("RobustLocationScaleModel.fit() must be called before predict()")
        eligible = bars_as_of(bars, as_of)
        underlying_id = _underlying_id_of(eligible)
        closes = _closes(eligible)
        cfg = self._config
        sigma = ewma_volatility(closes, lam=cfg.ewma_lambda)
        sigma_now = float(sigma[-1])
        if not (sigma_now > 0):
            raise ValueError(f"EWMA volatility as of {as_of!r} is zero/NaN; cannot normalize")
        df = cfg.student_t_df

        forecasts = []
        for h in horizons:
            if h not in self._loc:
                raise ValueError(f"horizon {h}d was not fit; fitted horizons: {sorted(self._loc)}")
            loc = self._loc[h]
            scale = self._scale[h]
            sigma_h = sigma_now * np.sqrt(h)
            quantiles = {
                key: (loc + scale * float(student_t.ppf(q, df))) * sigma_h
                for key, q in QUANTILE_LEVELS.items()
            }
            es05_y = _student_t_expected_shortfall(loc, scale, df, 0.05)
            p_up = float(student_t.cdf(loc / scale, df))
            var_t = scale**2 * df / (df - 2.0)
            sigma_y = float(np.sqrt(var_t))
            n_eff = self._n_effective[h]
            uncertainty = (sigma_y * sigma_h) / np.sqrt(n_eff) if n_eff > 0 else sigma_y * sigma_h
            forecasts.append(
                HorizonForecast(
                    underlying_id=underlying_id,
                    horizon_days=h,
                    prediction_time=as_of,
                    p_up=float(np.clip(p_up, 0.0, 1.0)),
                    mean=float(loc * sigma_h),
                    sigma=float(sigma_y * sigma_h),
                    quantiles=quantiles,
                    expected_shortfall_05=float(es05_y * sigma_h),
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
            raise RuntimeError("RobustLocationScaleModel.fit() must be called before model_hash()")
        return _baseline_model_hash(
            class_name="RobustLocationScaleModel",
            hyperparams=self._config.model_dump(mode="json"),
            training_end=self._training_end,
            coefficients={"loc": self._loc, "scale": self._scale},
        )
