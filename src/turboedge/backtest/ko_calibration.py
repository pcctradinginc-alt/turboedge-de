"""Out-of-sample calibration of the path-simulation knock-out probability
P(KO) (Workstream W10; ``docs/measured_results.md`` §3).

``simulation/ko_calibration.py`` builds the raw ``(p_ko_raw, realized_ko)``
observation dataset; this module fits and *walk-forward evaluates*
candidate calibrators against it, and decides whether any of them should
become the production default.

Walk-forward design (CLAUDE.md rules 6-8; Master Spec §27.4, §28):

- Each underlying has its own bar-index time axis
  (:class:`~turboedge.simulation.ko_calibration.KoCalibrationObservation.t0_index`).
  :class:`~turboedge.backtest.purged_cv.PurgedWalkForwardSplit` (purged +
  embargoed by ``horizon_days``) is run *independently per
  (underlying, horizon)* -- exactly the granularity
  ``backtest/walkforward.py::walk_forward_evaluate`` already uses for
  forecast models. Deliberately **not** pooled into one cross-underlying
  time axis: ``PurgedWalkForwardSplit`` purges/embargoes against a single
  sorted ``t0``/``t1`` axis, and two different underlyings' bar indices
  are not comparable positions in time (DAX bar 400 and NDX bar 400 are
  different calendar dates) -- inventing a joint axis would either merge
  unrelated indices or require assumptions this module does not make.
  Every underlying's OOS predictions ARE still pooled afterwards, but only
  for *reporting* (final metric aggregation), never for fitting.
- Within each fold, a calibrator is fit **per (direction, regime_bucket)
  stratum** on the fold's train observations only (mirrors
  ``P(realized_KO | p_ko_raw, horizon, direction, regime)`` -- horizon is
  already fixed by the fold's own axis, so the calibrator only needs to
  vary by direction/regime within it) and applied to that fold's test
  observations in the same stratum -- so every predicted probability
  reported here is genuinely out-of-sample, both in time (purged/embargoed
  by the split) and in calibrator-fit (train/test never share observations).
- Monotonicity: isotonic regression is monotone by construction; Platt
  (logistic) scaling is monotone in its single input. Neither candidate can
  produce a non-monotone mapping.

Candidates: identity (baseline; also what "raw" without any promoted
calibrator effectively is), isotonic and Platt/logistic
(``models/calibration.py``, reused rather than reimplemented). A monotone
GAM was *not* added: isotonic regression already is the nonparametric
monotone estimator for a single input (raw_p -> calibrated_p) -- a GAM
would add a smoothing-spline machinery on top of the same one-dimensional
monotone-mapping problem without a structural reason to expect it to beat
isotonic here (no additional covariates for it to combine), so per the
"only if statistically justified" instruction, it is skipped, and that
decision is stated rather than silently omitted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt

from turboedge.backtest import metrics as bt_metrics
from turboedge.backtest.purged_cv import PurgedWalkForwardSplit
from turboedge.models.calibration import (
    CalibrationConfig,
    Calibrator,
    IdentityCalibrator,
    fit_calibrator,
)
from turboedge.simulation.ko_calibration import KoCalibrationObservation
from turboedge.storage.schemas import Direction

CalibratorMethod = Literal["identity", "isotonic", "platt"]
CANDIDATE_METHODS: tuple[CalibratorMethod, ...] = ("identity", "isotonic", "platt")

#: docs/measured_results.md §3: the barrier distances closest to real turbo
#: placements, where the ranking/gates.py ACTIONABLE gate actually bites.
TAIL_RISK_SIGMA_BUCKETS: tuple[float, ...] = (1.5, 2.0)
#: How much more the mean-signed-error is allowed to move toward the unsafe
#: (under-predicting-KO) direction at the tail-risk buckets, relative to raw,
#: before a candidate is disqualified from promotion regardless of how much
#: it improves calibration elsewhere. 1 percentage point of realized-outcome
#: frequency -- small on purpose, since the conservative bias is a
#: deliberately protected safety margin (docs/measured_results.md §3), not
#: a defect to be closed at any cost.
_TAIL_RISK_DEGRADATION_TOLERANCE = 0.01
#: A candidate must not be materially worse on Brier either (guards against
#: a calibrator that improves ECE/ACE narrowly while overall prediction
#: quality regresses).
_BRIER_DEGRADATION_TOLERANCE = 0.0


@dataclass(frozen=True, slots=True)
class KoCalibrationPrediction:
    """One out-of-sample (raw or calibrated) prediction, fully attributable."""

    underlying_id: str
    prediction_time: datetime
    horizon_days: int
    direction: Direction
    sigma_k: float
    regime_bucket: str
    p_ko_raw: float
    p_ko_calibrated: float
    realized_ko: bool
    method: str


@dataclass(frozen=True, slots=True)
class KoCalibrationMetrics:
    n: int
    brier: float
    calibration_intercept: float
    calibration_slope: float
    ece: float
    mean_signed_error: float
    absolute_calibration_error: float


@dataclass(frozen=True, slots=True)
class KoCalibrationMethodResult:
    method: str
    overall: KoCalibrationMetrics
    by_horizon: dict[int, KoCalibrationMetrics]
    by_direction: dict[str, KoCalibrationMetrics]
    by_sigma_bucket: dict[float, KoCalibrationMetrics]
    by_underlying: dict[str, KoCalibrationMetrics]
    by_regime: dict[str, KoCalibrationMetrics]
    predictions: list[KoCalibrationPrediction] = field(repr=False)


@dataclass(frozen=True, slots=True)
class KoCalibrationRunResult:
    raw: KoCalibrationMethodResult
    candidates: dict[str, KoCalibrationMethodResult]
    promoted_method: str | None
    promotion_reason: str


def _metrics_from(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> KoCalibrationMetrics:
    intercept, slope = bt_metrics.calibration_slope_intercept(p, y)
    return KoCalibrationMetrics(
        n=int(p.size),
        brier=bt_metrics.brier_score(p, y),
        calibration_intercept=intercept,
        calibration_slope=slope,
        ece=bt_metrics.expected_calibration_error(p, y),
        mean_signed_error=bt_metrics.mean_signed_error(p, y),
        absolute_calibration_error=bt_metrics.absolute_calibration_error(p, y),
    )


def _group_metrics(
    preds: Sequence[KoCalibrationPrediction], key: Callable[[KoCalibrationPrediction], object]
) -> dict[object, KoCalibrationMetrics]:
    groups: dict[object, list[KoCalibrationPrediction]] = {}
    for pred in preds:
        groups.setdefault(key(pred), []).append(pred)
    out: dict[object, KoCalibrationMetrics] = {}
    for k, group in groups.items():
        p = np.array([g.p_ko_calibrated for g in group], dtype=np.float64)
        y = np.array([1.0 if g.realized_ko else 0.0 for g in group], dtype=np.float64)
        out[k] = _metrics_from(p, y)
    return out


def _build_method_result(
    method: str, preds: list[KoCalibrationPrediction]
) -> KoCalibrationMethodResult:
    p = np.array([g.p_ko_calibrated for g in preds], dtype=np.float64)
    y = np.array([1.0 if g.realized_ko else 0.0 for g in preds], dtype=np.float64)
    return KoCalibrationMethodResult(
        method=method,
        overall=_metrics_from(p, y),
        by_horizon=_group_metrics(preds, lambda pr: pr.horizon_days),  # type: ignore[arg-type]
        by_direction=_group_metrics(preds, lambda pr: pr.direction.value),  # type: ignore[arg-type]
        by_sigma_bucket=_group_metrics(preds, lambda pr: pr.sigma_k),  # type: ignore[arg-type]
        by_underlying=_group_metrics(preds, lambda pr: pr.underlying_id),  # type: ignore[arg-type]
        by_regime=_group_metrics(preds, lambda pr: pr.regime_bucket),  # type: ignore[arg-type]
        predictions=preds,
    )


def run_ko_calibration(
    observations: Sequence[KoCalibrationObservation],
    *,
    min_train: int,
    step: int,
    candidate_methods: Sequence[CalibratorMethod] = CANDIDATE_METHODS,
    calibration_min_samples: int = 20,
) -> KoCalibrationRunResult:
    """Walk-forward fit and out-of-sample evaluate every candidate calibrator.

    For each ``(underlying_id, horizon_days)`` pair present in
    ``observations``, runs an independent
    :class:`~turboedge.backtest.purged_cv.PurgedWalkForwardSplit`
    (``embargo = horizon_days``) over that pair's own observations, ordered
    by ``t0_index``. In every fold: the raw baseline is recorded directly
    (``p_ko_calibrated = p_ko_raw``, no fitting); for every candidate
    method, one calibrator is fit per ``(direction, regime_bucket)`` stratum
    on the fold's training observations and applied to the fold's test
    observations in the same stratum (a stratum absent from a fold's
    training data falls back to the raw value for that fold -- never an
    error, since a candidate calibrator must never be *worse than doing
    nothing* just because one fold had no training data for a rare
    stratum).

    Raises:
        ValueError: if ``observations`` is empty.
    """
    if not observations:
        raise ValueError("observations must not be empty")

    by_key: dict[tuple[str, int], list[KoCalibrationObservation]] = {}
    for obs in observations:
        by_key.setdefault((obs.underlying_id, obs.horizon_days), []).append(obs)

    raw_preds: list[KoCalibrationPrediction] = []
    candidate_preds: dict[str, list[KoCalibrationPrediction]] = {m: [] for m in candidate_methods}

    for (_underlying_id, horizon_days), obs_list in by_key.items():
        obs_list = sorted(obs_list, key=lambda o: o.t0_index)
        t0_arr = np.array([o.t0_index for o in obs_list], dtype=np.int64)
        t1_arr = t0_arr + horizon_days
        splitter = PurgedWalkForwardSplit(
            horizon=horizon_days, embargo=horizon_days, min_train=min_train, step=step
        )
        for train_idx, test_idx in splitter.split(t0_arr, t1_arr):
            train_obs = [obs_list[i] for i in train_idx.tolist()]
            test_obs = [obs_list[i] for i in test_idx.tolist()]

            for o in test_obs:
                raw_preds.append(
                    KoCalibrationPrediction(
                        underlying_id=o.underlying_id,
                        prediction_time=o.prediction_time,
                        horizon_days=o.horizon_days,
                        direction=o.direction,
                        sigma_k=o.sigma_k,
                        regime_bucket=o.regime_bucket,
                        p_ko_raw=o.p_ko_raw,
                        p_ko_calibrated=o.p_ko_raw,
                        realized_ko=o.realized_ko,
                        method="raw",
                    )
                )

            strata: dict[tuple[Direction, str], list[KoCalibrationObservation]] = {}
            for o in train_obs:
                strata.setdefault((o.direction, o.regime_bucket), []).append(o)

            for method in candidate_methods:
                calibrators: dict[tuple[Direction, str], Calibrator] = {}
                for stratum_key, group in strata.items():
                    raw_p = np.array([g.p_ko_raw for g in group], dtype=np.float64)
                    labels = np.array(
                        [1.0 if g.realized_ko else 0.0 for g in group], dtype=np.float64
                    )
                    weights = np.ones_like(raw_p)
                    if method == "identity":
                        calibrators[stratum_key] = IdentityCalibrator()
                    else:
                        cfg = CalibrationConfig(method=method, min_samples=calibration_min_samples)
                        calibrators[stratum_key] = fit_calibrator(cfg, raw_p, labels, weights)

                for o in test_obs:
                    stratum_key = (o.direction, o.regime_bucket)
                    calibrator = calibrators.get(stratum_key)
                    if calibrator is None:
                        calibrated = o.p_ko_raw
                    else:
                        calibrated = float(
                            calibrator.predict(np.array([o.p_ko_raw], dtype=np.float64))[0]
                        )
                    candidate_preds[method].append(
                        KoCalibrationPrediction(
                            underlying_id=o.underlying_id,
                            prediction_time=o.prediction_time,
                            horizon_days=o.horizon_days,
                            direction=o.direction,
                            sigma_k=o.sigma_k,
                            regime_bucket=o.regime_bucket,
                            p_ko_raw=o.p_ko_raw,
                            p_ko_calibrated=calibrated,
                            realized_ko=o.realized_ko,
                            method=method,
                        )
                    )

    if not raw_preds:
        raise ValueError(
            "walk-forward split produced no out-of-sample folds for any "
            "(underlying, horizon) pair -- check min_train/step against the "
            "dataset size"
        )

    raw_result = _build_method_result("raw", raw_preds)
    candidate_results = {
        method: _build_method_result(method, preds) for method, preds in candidate_preds.items()
    }
    promoted_method, reason = _decide_promotion(raw_result, candidate_results)

    return KoCalibrationRunResult(
        raw=raw_result,
        candidates=candidate_results,
        promoted_method=promoted_method,
        promotion_reason=reason,
    )


def _tail_risk_ok(raw: KoCalibrationMethodResult, candidate: KoCalibrationMethodResult) -> bool:
    """docs/measured_results.md §3 promotion rule's safety condition: a
    candidate must not materially worsen the conservative
    (over-predicting-KO) bias at the trading-relevant sigma buckets.

    A bucket missing from either side (e.g. no observations at that sigma
    in this dataset) is skipped, not treated as a failure -- absence of
    data is not evidence of harm.
    """
    for bucket in TAIL_RISK_SIGMA_BUCKETS:
        raw_m = raw.by_sigma_bucket.get(bucket)
        cand_m = candidate.by_sigma_bucket.get(bucket)
        if raw_m is None or cand_m is None:
            continue
        if np.isnan(raw_m.mean_signed_error) or np.isnan(cand_m.mean_signed_error):
            continue
        # mean_signed_error = mean(realized - predicted); more positive is
        # the unsafe direction (predicted P(KO) too low). A candidate whose
        # signed error rises materially above raw's is disqualified.
        if cand_m.mean_signed_error > raw_m.mean_signed_error + _TAIL_RISK_DEGRADATION_TOLERANCE:
            return False
    return True


def _calibration_improves(
    raw: KoCalibrationMethodResult, candidate: KoCalibrationMethodResult
) -> bool:
    """Both ECE and absolute_calibration_error must improve (strictly, not
    tie) over raw, and Brier must not regress -- a candidate that trades
    calibration for worse discrimination is not a genuine improvement."""
    if candidate.overall.brier > raw.overall.brier + _BRIER_DEGRADATION_TOLERANCE:
        return False
    if candidate.overall.ece >= raw.overall.ece:
        return False
    return candidate.overall.absolute_calibration_error < raw.overall.absolute_calibration_error


def _decide_promotion(
    raw: KoCalibrationMethodResult, candidates: dict[str, KoCalibrationMethodResult]
) -> tuple[str | None, str]:
    """docs/measured_results.md §3 promotion rule: a calibrator is promoted
    only if it improves OOS calibration over raw P(KO) without materially
    worsening the conservative tail-risk bias at 1.5-2 sigma. Never promotes
    "identity" over "raw" (they are numerically the same predictions, clipped
    only) -- identity is included as a candidate purely as the walk-forward-
    evaluated floor every other candidate must beat, not as a promotable
    outcome in its own right.
    """
    eligible: dict[str, KoCalibrationMethodResult] = {
        m: r for m, r in candidates.items() if m != "identity"
    }
    passing: list[str] = []
    rejected: dict[str, str] = {}
    for method, result in eligible.items():
        if not _calibration_improves(raw, result):
            rejected[method] = (
                f"did not improve OOS calibration over raw (brier={result.overall.brier:.5f} "
                f"vs raw {raw.overall.brier:.5f}, ece={result.overall.ece:.5f} vs raw "
                f"{raw.overall.ece:.5f}, ace={result.overall.absolute_calibration_error:.5f} "
                f"vs raw {raw.overall.absolute_calibration_error:.5f})"
            )
            continue
        if not _tail_risk_ok(raw, result):
            rejected[method] = (
                "improved overall calibration but materially worsened the conservative "
                f"tail-risk bias at sigma buckets {TAIL_RISK_SIGMA_BUCKETS} "
                "(mean_signed_error moved toward under-predicting P(KO))"
            )
            continue
        passing.append(method)

    if not passing:
        detail = "; ".join(f"{m}: {reason}" for m, reason in rejected.items())
        return None, f"no candidate cleared the promotion rule ({detail})" if detail else (
            "no candidate methods were evaluated"
        )

    best = min(passing, key=lambda m: eligible[m].overall.absolute_calibration_error)
    reason = (
        f"{best} improved OOS calibration over raw P(KO) "
        f"(brier {eligible[best].overall.brier:.5f} vs raw {raw.overall.brier:.5f}, "
        f"ece {eligible[best].overall.ece:.5f} vs raw {raw.overall.ece:.5f}, "
        f"ace {eligible[best].overall.absolute_calibration_error:.5f} vs raw "
        f"{raw.overall.absolute_calibration_error:.5f}) without materially worsening the "
        f"conservative tail-risk bias at sigma buckets {TAIL_RISK_SIGMA_BUCKETS}"
    )
    return best, reason


__all__ = [
    "CANDIDATE_METHODS",
    "TAIL_RISK_SIGMA_BUCKETS",
    "KoCalibrationMethodResult",
    "KoCalibrationMetrics",
    "KoCalibrationPrediction",
    "KoCalibrationRunResult",
    "run_ko_calibration",
]
