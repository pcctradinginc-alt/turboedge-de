"""Quantile models for the MAE/MFE excursion targets (RO-MAE-PREDICTION / RO-MFE-PREDICTION).

The research question this module exists to answer is whether an
entry-time feature vector (``feature_snapshot`` plus the other
entry-time-available forward-ledger columns) can forecast a turbo
position's maximum adverse excursion (``mae``) or maximum favourable
excursion (``mfe``) better than the position's unconditional empirical
distribution already does. The honest default expectation, per
``docs/measured_results.md``, is NO -- so the unconditional baseline
(:class:`NullExcursionModel`) is built as a first-class model, not an
afterthought, and is given every advantage the conditional model gets
(the same average-uniqueness sample weights). If
:class:`ConditionalExcursionModel` cannot beat it out of sample, that is
the answer, not a bug to work around.

Sign convention (verified against ``turboedge.learning.labeler.resolve_exit``,
not assumed): for each ledger entry, ``product_returns = [bid / entry_ask - 1
for every snapshot with a bid seen during the exit window]``, and then
``mfe = max(product_returns)``, ``mae = min(product_returns)``. Because both
are extrema of the *same* list, ``mae <= mfe`` always holds, but neither is
guaranteed to be single-signed: if a product's bid never rose above
``entry_ask`` during the window, its ``mfe`` is itself <= 0; if it never fell
below, its ``mae`` is itself >= 0. The typical case is ``mae <= 0`` (worst
drawdown) and ``mfe >= 0`` (best run-up), but this module does not assume
that -- it fits/predicts whatever sign the target column actually has.

Both model classes share one fit/predict shape so the evaluation harness
(owned elsewhere) can loop over either via the :class:`ExcursionModel`
protocol:

    model.fit(x_train, y_train, sample_weight=weights_train)
    predictions = model.predict(x_test)  # list[ExcursionPrediction], one per row

Determinism: neither :class:`~sklearn.preprocessing.StandardScaler` nor
:class:`~sklearn.linear_model.QuantileRegressor` (solver ``"highs"``, an
exact linear-program solver, not a stochastic one) exposes a
``random_state`` parameter, and :func:`turboedge.models.quantile.weighted_quantile`
uses a stable sort -- so there is no source of randomness to seed, and
fit/predict on the same input is exactly reproducible byte-for-byte. This
is verified by a determinism test in the test suite rather than merely
asserted here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from itertools import pairwise
from typing import Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, model_validator
from sklearn.linear_model import QuantileRegressor  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from turboedge.models.quantile import weighted_quantile

__all__ = [
    "DEFAULT_QUANTILES",
    "ConditionalExcursionModel",
    "ExcursionModel",
    "ExcursionPrediction",
    "ExcursionTarget",
    "NullExcursionModel",
]

#: Canonical quantile levels requested when a caller does not specify its own.
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Samples required per regression parameter (one slope per feature, plus the
#: intercept) for :class:`ConditionalExcursionModel`. 10 is the standard
#: rule-of-thumb floor below which a linear fit's parameter estimates are
#: dominated by noise rather than signal (see e.g. Harrell, "Regression
#: Modeling Strategies", on events/parameters ratios); below it a per-tau fit
#: is not "conditional on features", it is memorising rows.
_MIN_SAMPLES_PER_FEATURE = 10


class ExcursionTarget(StrEnum):
    """Which excursion column a model instance is fit to."""

    MAE = "mae"
    MFE = "mfe"


class ExcursionPrediction(BaseModel):
    """One row's predicted quantile curve for one excursion target.

    ``quantiles`` maps ``tau -> predicted value`` and must be non-decreasing
    in ``tau`` -- both concrete models enforce this before constructing the
    prediction (see ``ConditionalExcursionModel``'s crossing-sort), and this
    validator is the defensive backstop that catches a violation directly
    rather than letting a silently-invalid prediction reach a caller.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: ExcursionTarget
    quantiles: dict[float, float]

    @model_validator(mode="after")
    def _check_quantiles(self) -> Self:
        if not self.quantiles:
            raise ValueError("quantiles must not be empty")
        items = sorted(self.quantiles.items())
        taus = [tau for tau, _ in items]
        if any(not (0.0 <= tau <= 1.0) for tau in taus):
            raise ValueError(f"quantile levels (tau) must be within [0, 1], got {taus!r}")
        values = [value for _, value in items]
        if any(later < earlier for earlier, later in pairwise(values)):
            raise ValueError(f"quantiles must be non-decreasing in tau, got {dict(items)!r}")
        return self


@runtime_checkable
class ExcursionModel(Protocol):
    """Structural interface shared by every excursion model.

    Both :class:`NullExcursionModel` and :class:`ConditionalExcursionModel`
    satisfy this without inheriting from it, so the evaluation harness can
    hold a ``list[ExcursionModel]`` and loop over ``fit``/``predict``
    identically regardless of which concrete class it is.
    """

    model_id: str

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> None: ...

    def predict(self, x: npt.NDArray[np.float64]) -> list[ExcursionPrediction]: ...


def _validate_quantile_levels(levels: Sequence[float]) -> tuple[float, ...]:
    """Shared validation for both models' ``quantile_levels`` constructor argument."""
    levels_t = tuple(float(q) for q in levels)
    if not levels_t:
        raise ValueError("quantile_levels must not be empty")
    if any(not (0.0 < q < 1.0) for q in levels_t):
        raise ValueError(f"quantile_levels must be within (0, 1), got {levels_t!r}")
    if len(set(levels_t)) != len(levels_t):
        raise ValueError(f"quantile_levels must not contain duplicates, got {levels_t!r}")
    if list(levels_t) != sorted(levels_t):
        raise ValueError(f"quantile_levels must be given in ascending order, got {levels_t!r}")
    return levels_t


def _validate_fit_inputs(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Checks common to both models' ``fit``: shapes, non-negativity, degeneracy.

    Raises ``ValueError`` naming the specific cause for every degenerate case
    the contract calls out (mismatched shapes, all-zero weights, zero-variance
    target) -- callers add their own additional, model-specific minimum-sample
    checks on top of this.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2-dimensional, got shape {x_arr.shape!r}")
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y_arr.shape!r}")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"x and y must have matching first dimension, got {x_arr.shape[0]} and {y_arr.shape[0]}"
        )
    n = y_arr.shape[0]
    if n == 0:
        raise ValueError("x and y must not be empty")
    if sample_weight is None:
        w_arr = np.ones(n, dtype=np.float64)
    else:
        w_arr = np.asarray(sample_weight, dtype=np.float64)
        if w_arr.shape != (n,):
            raise ValueError(f"sample_weight must have shape ({n},), got {w_arr.shape!r}")
        if np.any(w_arr < 0.0):
            raise ValueError("sample_weight must be non-negative")
    if float(np.sum(w_arr)) <= 0.0:
        raise ValueError("sample_weight must not sum to zero (all weights are zero)")
    if float(np.std(y_arr)) == 0.0:
        raise ValueError(
            "y has zero variance -- every quantile of a constant target is that "
            "constant, so there is nothing to fit"
        )
    return x_arr, y_arr, w_arr


class NullExcursionModel:
    """Unconditional weighted empirical quantiles of the training targets.

    This is the baseline every conditional model must beat. It is not a
    straw man: it uses the *same* average-uniqueness ``sample_weight`` the
    conditional model gets, via a from-scratch weighted-quantile
    implementation (sorted values, cumulative weight, linear interpolation --
    see :func:`turboedge.models.quantile.weighted_quantile`'s Hazen-style
    convention, reused here rather than reimplemented so both models are
    scored against literally the same quantile estimator). A null model that
    silently ignored the weights while the conditional model used them would
    make any "improvement" partly an artifact of that asymmetry rather than
    of genuinely conditioning on features.
    """

    model_id = "excursion_null_v1"

    def __init__(
        self,
        target: ExcursionTarget,
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        self._target = target
        self._quantile_levels = _validate_quantile_levels(quantile_levels)
        # The tightest requested tail (closest to 0 or 1) needs at least one
        # order statistic's worth of weight to be represented at all
        # (1 / tail samples); independently, there must be at least twice as
        # many samples as distinct quantile levels requested, so the levels
        # are resolving a real distribution rather than interpolating
        # between a handful of points. Both bounds are named/reasoned, not
        # arbitrary, and both come from `quantile_levels` itself.
        tail = min(min(q, 1.0 - q) for q in self._quantile_levels)
        self._min_samples = max(2 * len(self._quantile_levels), math.ceil(1.0 / tail))
        self._y: npt.NDArray[np.float64] | None = None
        self._w: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Fit on ``(x, y)``. ``x``'s values are ignored (this model is
        unconditional) -- only its row count is checked against ``y``, so
        both models share one call signature for the harness's benefit."""
        _, y_arr, w_arr = _validate_fit_inputs(x, y, sample_weight)
        n = y_arr.shape[0]
        if n < self._min_samples:
            raise ValueError(
                f"need at least {self._min_samples} samples to estimate the tail quantile "
                f"{min(min(q, 1.0 - q) for q in self._quantile_levels):.3f} reliably, got {n}"
            )
        self._y = y_arr
        self._w = w_arr

    def predict(self, x: npt.NDArray[np.float64]) -> list[ExcursionPrediction]:
        if self._y is None or self._w is None:
            raise RuntimeError("NullExcursionModel.fit() must be called before predict()")
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 2-dimensional, got shape {x_arr.shape!r}")
        # weighted_quantile(q) is monotone non-decreasing in q by construction
        # (piecewise-linear interpolation over a sorted, cumulative-weight
        # axis), so no crossing-sort is needed here -- unlike the
        # per-tau-independent conditional model below.
        quantiles = {tau: weighted_quantile(self._y, self._w, tau) for tau in self._quantile_levels}
        return [
            ExcursionPrediction(target=self._target, quantiles=dict(quantiles))
            for _ in range(x_arr.shape[0])
        ]


class ConditionalExcursionModel:
    """Linear quantile regression on standardized entry-time features.

    One ``sklearn.linear_model.QuantileRegressor`` per requested tau, each
    L1-regularised (``alpha``) on features standardised by a
    ``StandardScaler`` fit on the training fold only (never on test data --
    CLAUDE.md rule 4/no look-ahead applies to feature scaling too).

    Fitting each tau independently can produce crossing quantiles (e.g. a
    fitted q75 below the fitted q50 for some row). This is not hidden: the
    raw per-tau predictions are sorted per row to restore monotonicity
    (:class:`ExcursionPrediction` would otherwise reject them), and
    ``crossing_count`` / ``crossing_rate`` after ``predict`` report exactly
    how often the raw fit crossed before that correction -- a high rate is
    itself evidence the model is overfitting a thin sample, and a reviewer
    comparing this model to the null needs to see it, not have it quietly
    fixed away.
    """

    model_id = "excursion_quantile_v1"

    def __init__(
        self,
        target: ExcursionTarget,
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        alpha: float = 0.1,
    ) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        self._target = target
        self._quantile_levels = _validate_quantile_levels(quantile_levels)
        self._alpha = alpha
        self._scaler: StandardScaler | None = None
        self._models: dict[float, QuantileRegressor] | None = None
        self._n_features: int | None = None
        self._crossing_count: int | None = None
        self._n_predictions: int | None = None

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> None:
        x_arr, y_arr, w_arr = _validate_fit_inputs(x, y, sample_weight)
        n, n_features = x_arr.shape
        if n_features < 1:
            raise ValueError("x must have at least one feature column")
        if n < n_features:
            raise ValueError(
                f"fewer training samples ({n}) than features ({n_features}) -- "
                "the per-tau linear fit is unidentified"
            )
        min_samples = (n_features + 1) * _MIN_SAMPLES_PER_FEATURE
        if n < min_samples:
            raise ValueError(
                f"need at least {min_samples} samples ({n_features} features + intercept, "
                f"{_MIN_SAMPLES_PER_FEATURE} samples/parameter) for a stable per-tau fit, "
                f"got {n}"
            )

        scaler = StandardScaler()
        scaler.fit(x_arr, sample_weight=w_arr)
        x_scaled = scaler.transform(x_arr)

        fitted: dict[float, QuantileRegressor] = {}
        for tau in self._quantile_levels:
            model = QuantileRegressor(quantile=tau, alpha=self._alpha, solver="highs")
            try:
                model.fit(x_scaled, y_arr, sample_weight=w_arr)
            except Exception as exc:
                # Never fall back to the null model's answer here -- a
                # convergence failure must surface as a failure, or the
                # baseline comparison this module exists to support becomes
                # meaningless (a "conditional" model that's secretly the
                # null model on a bad fold would look like a tie, not a
                # failure).
                raise RuntimeError(
                    f"QuantileRegressor failed to fit tau={tau!r} "
                    f"(alpha={self._alpha!r}, n={n}, n_features={n_features}): {exc}"
                ) from exc
            fitted[tau] = model

        self._scaler = scaler
        self._models = fitted
        self._n_features = n_features

    def predict(self, x: npt.NDArray[np.float64]) -> list[ExcursionPrediction]:
        if self._scaler is None or self._models is None or self._n_features is None:
            raise RuntimeError("ConditionalExcursionModel.fit() must be called before predict()")
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 2-dimensional, got shape {x_arr.shape!r}")
        if x_arr.shape[1] != self._n_features:
            raise ValueError(
                f"x has {x_arr.shape[1]} feature columns, model was fit on {self._n_features}"
            )

        x_scaled = self._scaler.transform(x_arr)
        raw = np.column_stack(
            [self._models[tau].predict(x_scaled) for tau in self._quantile_levels]
        )  # shape (n_rows, n_quantiles), columns in ascending tau order
        sorted_raw = np.sort(raw, axis=1)
        # A row's raw predictions equal their own sorted version iff they
        # were already non-decreasing across tau; any difference means that
        # row's per-tau fits crossed and had to be corrected.
        crossed_rows = np.any(raw != sorted_raw, axis=1)
        self._crossing_count = int(np.sum(crossed_rows))
        self._n_predictions = int(x_arr.shape[0])

        predictions: list[ExcursionPrediction] = []
        for i in range(x_arr.shape[0]):
            quantiles = {
                tau: float(sorted_raw[i, j]) for j, tau in enumerate(self._quantile_levels)
            }
            predictions.append(ExcursionPrediction(target=self._target, quantiles=quantiles))
        return predictions

    @property
    def crossing_count(self) -> int:
        """Rows whose raw per-tau predictions were not already monotone in tau.

        Only meaningful after ``predict`` has been called at least once.
        """
        if self._crossing_count is None:
            raise RuntimeError("predict() must be called before crossing_count is available")
        return self._crossing_count

    @property
    def crossing_rate(self) -> float:
        """``crossing_count`` as a fraction of rows predicted."""
        if self._crossing_count is None or not self._n_predictions:
            raise RuntimeError("predict() must be called before crossing_rate is available")
        return self._crossing_count / self._n_predictions
