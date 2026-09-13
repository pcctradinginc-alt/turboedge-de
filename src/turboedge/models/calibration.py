"""Probability calibration: isotonic regression and Platt scaling.

Formula reference: Master Spec §10 ("Calibration"). Used by
``models/directional.py`` to map a model's raw ``P(return > 0)`` estimate to
a calibrated probability, fit on out-of-sample folds *within* the training
window (never on the same data the raw estimate was produced from) so the
mapping does not simply memorize training noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

_IDENTITY_EPS = 1e-6


class CalibrationConfig(BaseModel):
    """Hyperparameters for :func:`fit_calibrator`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["isotonic", "platt"] = "isotonic"
    #: Minimum number of out-of-sample (raw_p, label) pairs required to fit a
    #: calibrator; below this (or with a single-class label set) calibration
    #: is skipped in favor of :class:`IdentityCalibrator`, never silently
    #: producing an overfit mapping from too little data.
    min_samples: int = 20


class Calibrator(Protocol):
    def predict(self, raw_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class IdentityCalibrator:
    """Fallback calibrator: clips to ``[eps, 1-eps]`` without remapping (too little OOS data)."""

    eps: float = _IDENTITY_EPS

    def predict(self, raw_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        arr = np.asarray(raw_p, dtype=np.float64)
        return np.clip(arr, self.eps, 1.0 - self.eps)


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    model: IsotonicRegression

    def predict(self, raw_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        arr = np.asarray(raw_p, dtype=np.float64)
        out: npt.NDArray[np.float64] = self.model.predict(arr)
        return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    model: LogisticRegression

    def predict(self, raw_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        arr = np.asarray(raw_p, dtype=np.float64).reshape(-1, 1)
        out: npt.NDArray[np.float64] = self.model.predict_proba(arr)[:, 1]
        return out


def fit_calibrator(
    cfg: CalibrationConfig,
    raw_p: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> Calibrator:
    """Fit an isotonic or Platt calibrator mapping raw ``P(up)`` to a calibrated probability.

    ``raw_p``/``labels``/``weights`` should come from out-of-sample folds
    within the training window (see ``TsmomForecastModel``); falls back to
    :class:`IdentityCalibrator` when there is too little data (< ``cfg.min_samples``)
    or only one label class present (isotonic/Platt would be degenerate).
    """
    raw = np.asarray(raw_p, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if raw.shape != y.shape or raw.shape != w.shape:
        raise ValueError("raw_p, labels and weights must have the same shape")
    if raw.size < cfg.min_samples or np.unique(y).size < 2:
        return IdentityCalibrator()

    if cfg.method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw, y, sample_weight=w)
        return IsotonicCalibrator(iso)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(raw.reshape(-1, 1), y, sample_weight=w)
    return PlattCalibrator(platt)
