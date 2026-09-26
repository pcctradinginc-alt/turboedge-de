from __future__ import annotations

import numpy as np
import pytest

from turboedge.backtest.metrics import mean_pinball_loss
from turboedge.models.excursion import (
    DEFAULT_QUANTILES,
    ConditionalExcursionModel,
    ExcursionModel,
    ExcursionPrediction,
    ExcursionTarget,
    NullExcursionModel,
)

# --------------------------------------------------------------------------
# Weighted empirical quantiles (NullExcursionModel), hand-computed
# --------------------------------------------------------------------------


def test_null_model_hand_computed_weighted_median() -> None:
    """Hand-computed weighted median: y=[1,2,3], w=[1,1,8] (heavy weight on 3).

    Using the Hazen-style convention (``cum = cumsum(w) - 0.5*w``, normalised,
    then linearly interpolated against the sorted values -- the same
    convention documented in ``turboedge.models.quantile.weighted_quantile``,
    which this model reuses):

        cum_raw = [0.5, 1.5, 6.0], total = 10.0
        cum     = [0.05, 0.15, 0.6]

    tau=0.5 falls between index 1 (cum=0.15, v=2) and index 2 (cum=0.6, v=3):

        frac = (0.5 - 0.15) / (0.6 - 0.15) = 0.35 / 0.45 = 7/9
        expected = 2 + 7/9 * (3 - 2) = 2 + 7/9 ≈ 2.777778
    """
    model = NullExcursionModel(ExcursionTarget.MFE, quantile_levels=(0.5,))
    x = np.zeros((3, 1))
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 8.0])
    model.fit(x, y, sample_weight=w)

    [prediction] = model.predict(np.zeros((1, 1)))
    expected = 2.0 + 7.0 / 9.0
    assert prediction.quantiles[0.5] == pytest.approx(expected, abs=1e-9)


def test_null_model_weights_change_the_output() -> None:
    """Equal weights vs. weight concentrated on the largest value must give
    different medians -- a null model that silently dropped `sample_weight`
    (e.g. called `np.quantile` on the raw values) would fail this."""
    x = np.zeros((4, 1))
    y = np.array([1.0, 2.0, 3.0, 4.0])

    equal = NullExcursionModel(ExcursionTarget.MAE, quantile_levels=(0.5,))
    equal.fit(x, y, sample_weight=np.ones(4))
    [equal_pred] = equal.predict(np.zeros((1, 1)))

    skewed = NullExcursionModel(ExcursionTarget.MAE, quantile_levels=(0.5,))
    skewed.fit(x, y, sample_weight=np.array([1.0, 1.0, 1.0, 50.0]))
    [skewed_pred] = skewed.predict(np.zeros((1, 1)))

    assert equal_pred.quantiles[0.5] == pytest.approx(2.5, abs=1e-9)
    assert skewed_pred.quantiles[0.5] > equal_pred.quantiles[0.5]
    assert skewed_pred.quantiles[0.5] != pytest.approx(equal_pred.quantiles[0.5])


# --------------------------------------------------------------------------
# ConditionalExcursionModel recovers a known linear relationship
# --------------------------------------------------------------------------


def test_conditional_model_recovers_linear_slope_sign_and_magnitude() -> None:
    """y depends on x0 through both location (slope 4.0) and dispersion
    (heteroscedastic noise scaling with |x0|) -- a feature-dependent quantile
    structure, not just a feature-dependent mean. The median (tau=0.5) tracks
    the location term because the noise is symmetric about zero, so the
    fitted q50 curve's slope should recover the true slope's sign and rough
    magnitude regardless of the added heteroscedasticity.
    """
    rng = np.random.default_rng(1234)
    n = 3000
    x = rng.normal(size=(n, 1))
    noise = rng.normal(size=n) * (1.0 + 2.0 * np.abs(x[:, 0]))
    y = 4.0 * x[:, 0] + noise

    model = ConditionalExcursionModel(ExcursionTarget.MFE, quantile_levels=(0.5,))
    model.fit(x, y, sample_weight=np.ones(n))

    low, high = model.predict(np.array([[-2.0], [2.0]]))
    slope_estimate = (high.quantiles[0.5] - low.quantiles[0.5]) / 4.0

    assert slope_estimate > 0.0  # correct sign
    assert slope_estimate == pytest.approx(4.0, rel=0.3)  # rough magnitude


# --------------------------------------------------------------------------
# Monotonicity enforcement and the crossing counter
# --------------------------------------------------------------------------


class _FakeRegressor:
    """Stand-in for a fitted ``QuantileRegressor`` that returns a fixed,
    per-row prediction column regardless of input -- used to force a
    deterministic crossing scenario without depending on QuantileRegressor
    actually crossing on some particular random draw (which is not
    guaranteed on well-conditioned data, and would make this test flaky)."""

    def __init__(self, column: np.ndarray) -> None:
        self._column = column

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._column[: x.shape[0]]


def test_monotonicity_enforced_and_crossing_counted() -> None:
    model = ConditionalExcursionModel(ExcursionTarget.MAE, quantile_levels=(0.25, 0.5, 0.75))
    # Fit for real first so the scaler/feature-count bookkeeping is genuine,
    # then swap in fake per-tau regressors that deliberately cross on some
    # rows -- this isolates the sort-and-count logic in `predict` from
    # whatever QuantileRegressor happens to produce on a given fold.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 1))
    y = rng.normal(size=200)
    model.fit(x, y, sample_weight=np.ones(200))

    # Row 0: correctly ordered (0.25 < 0.5 < 0.75) -> no crossing.
    # Row 1: q0.75 predicted below q0.25 -> crossing.
    # Row 2: q0.5 predicted below q0.25 -> crossing.
    model._models = {
        0.25: _FakeRegressor(np.array([0.0, 5.0, 3.0])),
        0.5: _FakeRegressor(np.array([1.0, 4.0, 1.0])),
        0.75: _FakeRegressor(np.array([2.0, 3.0, 4.0])),
    }

    predictions = model.predict(np.zeros((3, 1)))

    assert [predictions[0].quantiles[t] for t in (0.25, 0.5, 0.75)] == [0.0, 1.0, 2.0]
    for pred in predictions:
        values = [pred.quantiles[t] for t in (0.25, 0.5, 0.75)]
        assert values == sorted(values)

    assert model.crossing_count == 2
    assert model.crossing_rate == pytest.approx(2 / 3)


def test_crossing_properties_raise_before_predict() -> None:
    model = ConditionalExcursionModel(ExcursionTarget.MAE)
    with pytest.raises(RuntimeError):
        _ = model.crossing_count
    with pytest.raises(RuntimeError):
        _ = model.crossing_rate


# --------------------------------------------------------------------------
# ExcursionPrediction validator
# --------------------------------------------------------------------------


def test_excursion_prediction_rejects_crossing_quantiles() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        ExcursionPrediction(target=ExcursionTarget.MAE, quantiles={0.25: 1.0, 0.75: 0.5})


def test_excursion_prediction_rejects_empty_quantiles() -> None:
    with pytest.raises(ValueError):
        ExcursionPrediction(target=ExcursionTarget.MAE, quantiles={})


def test_excursion_prediction_frozen_and_extra_forbidden() -> None:
    pred = ExcursionPrediction(target=ExcursionTarget.MFE, quantiles={0.5: 1.0})
    with pytest.raises(ValueError):
        pred.target = ExcursionTarget.MAE  # frozen
    with pytest.raises(ValueError):
        ExcursionPrediction(target=ExcursionTarget.MFE, quantiles={0.5: 1.0}, extra_field=1)


# --------------------------------------------------------------------------
# Degenerate input -> explicit errors (never a silent meaningless fit)
# --------------------------------------------------------------------------


def test_null_model_rejects_too_few_samples() -> None:
    model = NullExcursionModel(ExcursionTarget.MAE)  # default quantiles, tail=0.05 -> min 20
    x = np.zeros((5, 1))
    y = np.arange(5.0)
    with pytest.raises(ValueError, match="need at least"):
        model.fit(x, y)


def test_conditional_model_rejects_too_few_samples_per_feature() -> None:
    model = ConditionalExcursionModel(ExcursionTarget.MAE)
    n, n_features = 15, 1  # below (1+1)*10 = 20
    x = np.zeros((n, n_features))
    y = np.arange(float(n))
    with pytest.raises(ValueError, match="need at least"):
        model.fit(x, y)


def test_conditional_model_rejects_fewer_samples_than_features() -> None:
    model = ConditionalExcursionModel(ExcursionTarget.MAE)
    x = np.zeros((3, 5))
    y = np.arange(3.0)
    with pytest.raises(ValueError, match="fewer training samples"):
        model.fit(x, y)


@pytest.mark.parametrize("model_cls", [NullExcursionModel, ConditionalExcursionModel])
def test_rejects_zero_variance_target(model_cls: type) -> None:
    model = model_cls(ExcursionTarget.MAE)
    n = 50
    x = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = np.full(n, 3.0)
    with pytest.raises(ValueError, match="zero variance"):
        model.fit(x, y)


@pytest.mark.parametrize("model_cls", [NullExcursionModel, ConditionalExcursionModel])
def test_rejects_all_zero_weights(model_cls: type) -> None:
    model = model_cls(ExcursionTarget.MAE)
    n = 50
    x = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = np.linspace(0.0, 1.0, n)
    with pytest.raises(ValueError, match="all weights are zero"):
        model.fit(x, y, sample_weight=np.zeros(n))


@pytest.mark.parametrize("model_cls", [NullExcursionModel, ConditionalExcursionModel])
def test_rejects_negative_weights(model_cls: type) -> None:
    model = model_cls(ExcursionTarget.MAE)
    n = 50
    x = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    w[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(x, y, sample_weight=w)


@pytest.mark.parametrize("model_cls", [NullExcursionModel, ConditionalExcursionModel])
def test_rejects_mismatched_shapes(model_cls: type) -> None:
    model = model_cls(ExcursionTarget.MAE)
    x = np.zeros((10, 1))
    y = np.zeros(9)
    with pytest.raises(ValueError, match="matching first dimension"):
        model.fit(x, y)


@pytest.mark.parametrize("model_cls", [NullExcursionModel, ConditionalExcursionModel])
def test_predict_before_fit_raises(model_cls: type) -> None:
    model = model_cls(ExcursionTarget.MAE)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.zeros((1, 1)))


def test_conditional_model_rejects_zero_feature_columns() -> None:
    model = ConditionalExcursionModel(ExcursionTarget.MAE)
    x = np.zeros((50, 0))
    y = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError, match="at least one feature"):
        model.fit(x, y)


@pytest.mark.parametrize(
    "levels",
    [
        (),
        (0.0, 0.5),
        (1.0,),
        (0.5, 0.5),
        (0.75, 0.25),
    ],
)
def test_invalid_quantile_levels_rejected(levels: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        NullExcursionModel(ExcursionTarget.MAE, quantile_levels=levels)
    with pytest.raises(ValueError):
        ConditionalExcursionModel(ExcursionTarget.MAE, quantile_levels=levels)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_null_model_deterministic() -> None:
    rng = np.random.default_rng(99)
    x = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    w = rng.uniform(0.1, 1.0, size=100)

    a = NullExcursionModel(ExcursionTarget.MAE)
    a.fit(x, y, sample_weight=w)
    b = NullExcursionModel(ExcursionTarget.MAE)
    b.fit(x, y, sample_weight=w)

    preds_a = a.predict(x)
    preds_b = b.predict(x)
    for pa, pb in zip(preds_a, preds_b, strict=True):
        assert pa.quantiles == pb.quantiles


def test_conditional_model_deterministic() -> None:
    rng = np.random.default_rng(99)
    x = rng.normal(size=(200, 2))
    y = rng.normal(size=200) + x[:, 0]
    w = rng.uniform(0.1, 1.0, size=200)

    a = ConditionalExcursionModel(ExcursionTarget.MFE)
    a.fit(x, y, sample_weight=w)
    b = ConditionalExcursionModel(ExcursionTarget.MFE)
    b.fit(x, y, sample_weight=w)

    preds_a = a.predict(x)
    preds_b = b.predict(x)
    for pa, pb in zip(preds_a, preds_b, strict=True):
        assert pa.quantiles == pb.quantiles


# --------------------------------------------------------------------------
# predict returns one prediction per row
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_rows", [1, 2, 17])
def test_null_model_one_prediction_per_row(n_rows: int) -> None:
    model = NullExcursionModel(ExcursionTarget.MAE)
    x_train = np.zeros((30, 1))
    y_train = np.linspace(-1.0, 1.0, 30)
    model.fit(x_train, y_train)
    predictions = model.predict(np.zeros((n_rows, 1)))
    assert len(predictions) == n_rows


@pytest.mark.parametrize("n_rows", [1, 2, 17])
def test_conditional_model_one_prediction_per_row(n_rows: int) -> None:
    model = ConditionalExcursionModel(ExcursionTarget.MAE)
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(50, 1))
    y_train = rng.normal(size=50)
    model.fit(x_train, y_train)
    predictions = model.predict(rng.normal(size=(n_rows, 1)))
    assert len(predictions) == n_rows


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_both_models_satisfy_excursion_model_protocol() -> None:
    null_model = NullExcursionModel(ExcursionTarget.MAE)
    conditional_model = ConditionalExcursionModel(ExcursionTarget.MAE)
    assert isinstance(null_model, ExcursionModel)
    assert isinstance(conditional_model, ExcursionModel)


# --------------------------------------------------------------------------
# MANDATORY anti-triviality tests
# --------------------------------------------------------------------------


def _total_pinball(
    y_true: np.ndarray, predictions: list[ExcursionPrediction], taus: tuple[float, ...]
) -> float:
    total = 0.0
    for tau in taus:
        q_pred = np.array([p.quantiles[tau] for p in predictions])
        total += mean_pinball_loss(y_true, q_pred, tau)
    return total


def test_conditional_model_beats_null_when_target_depends_on_feature() -> None:
    """Target genuinely depends on the feature (location AND dispersion) --
    a conditional model that ignored its features and just reproduced the
    unconditional distribution would tie the null, not beat it. This must
    fail if `ConditionalExcursionModel.fit`/`predict` silently drop `x`."""
    rng = np.random.default_rng(2024)
    taus = DEFAULT_QUANTILES

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, 1))
        noise = rng.normal(size=n) * (1.0 + 1.5 * np.abs(x[:, 0]))
        y = 3.0 * x[:, 0] + noise
        return x, y

    x_train, y_train = sample(2000)
    x_test, y_test = sample(4000)

    null_model = NullExcursionModel(ExcursionTarget.MFE, quantile_levels=taus)
    null_model.fit(x_train, y_train, sample_weight=np.ones(len(y_train)))
    null_total = _total_pinball(y_test, null_model.predict(x_test), taus)

    conditional_model = ConditionalExcursionModel(ExcursionTarget.MFE, quantile_levels=taus)
    conditional_model.fit(x_train, y_train, sample_weight=np.ones(len(y_train)))
    conditional_total = _total_pinball(y_test, conditional_model.predict(x_test), taus)

    assert conditional_total < null_total


def test_conditional_model_does_not_meaningfully_beat_null_on_pure_noise() -> None:
    """Target is pure noise, independent of every feature. A conditional
    model fitting noise (rather than correctly finding ~no signal) would
    show a spuriously large improvement over the null; this is the more
    important of the two anti-triviality checks for this project, whose
    default expectation is that conditioning on entry-time features finds
    nothing (see the module docstring and docs/measured_results.md)."""
    rng = np.random.default_rng(555)
    taus = DEFAULT_QUANTILES
    n_features = 3

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, n_features))
        y = rng.normal(size=n)  # independent of x
        return x, y

    x_train, y_train = sample(3000)
    x_test, y_test = sample(5000)

    null_model = NullExcursionModel(ExcursionTarget.MAE, quantile_levels=taus)
    null_model.fit(x_train, y_train, sample_weight=np.ones(len(y_train)))
    null_total = _total_pinball(y_test, null_model.predict(x_test), taus)

    conditional_model = ConditionalExcursionModel(ExcursionTarget.MAE, quantile_levels=taus)
    conditional_model.fit(x_train, y_train, sample_weight=np.ones(len(y_train)))
    conditional_total = _total_pinball(y_test, conditional_model.predict(x_test), taus)

    # Allow at most a 5% relative improvement -- sampling noise alone can
    # produce a small, spurious edge, but a model actually fitting the noise
    # would show a much larger one.
    margin = 0.05 * null_total
    assert conditional_total >= null_total - margin
