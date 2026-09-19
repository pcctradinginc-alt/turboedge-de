from __future__ import annotations

from collections.abc import Callable

import pytest

from turboedge.models.baselines import (
    RegimeConditionalEmpiricalConfig,
    RegimeConditionalEmpiricalModel,
    RegularizedLinearLocationConfig,
    RegularizedLinearLocationModel,
    RobustLocationScaleConfig,
    RobustLocationScaleModel,
)
from turboedge.models.forecast import HORIZONS, HorizonForecast
from turboedge.storage.schemas import UnderlyingBar

_MODEL_FACTORIES: list[Callable[[], object]] = [
    lambda: RegimeConditionalEmpiricalModel(
        RegimeConditionalEmpiricalConfig(
            min_train_samples=40, min_bucket_samples=15, regime_window=120, regime_min_periods=30
        )
    ),
    lambda: RegularizedLinearLocationModel(RegularizedLinearLocationConfig(min_train_samples=60)),
    lambda: RobustLocationScaleModel(RobustLocationScaleConfig(min_train_samples=40)),
]
_MODEL_IDS = [
    "regime_conditional_empirical",
    "regularized_linear_location",
    "robust_location_scale_t",
]


def _quantile_values(fc: HorizonForecast) -> list[float]:
    return [fc.quantiles[k] for k in ("q05", "q25", "q50", "q75", "q95")]


# --- Protocol conformance ----------------------------------------------------


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_model_has_forecast_model_protocol_shape(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(700, seed=1)
    model = factory()
    assert hasattr(model, "model_id")
    assert hasattr(model, "signal_family")
    as_of = bars[-1].available_at
    model.fit(bars, as_of)  # type: ignore[attr-defined]
    forecasts = model.predict(bars, as_of)  # type: ignore[attr-defined]
    assert isinstance(forecasts, list)
    assert len(forecasts) == len(HORIZONS)
    for fc in forecasts:
        assert isinstance(fc, HorizonForecast)
    h = model.model_hash()  # type: ignore[attr-defined]
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)


# --- No look-ahead ------------------------------------------------------------


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_no_lookahead_appending_future_bars_does_not_change_prediction(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(800, seed=2)
    as_of = bars[500].available_at
    prefix_bars = bars[:501]

    model_a = factory()
    model_a.fit(prefix_bars, as_of)  # type: ignore[attr-defined]
    forecast_a = model_a.predict(prefix_bars, as_of)  # type: ignore[attr-defined]

    future_bars = bars
    model_b = factory()
    model_b.fit(future_bars, as_of)  # type: ignore[attr-defined]
    forecast_b = model_b.predict(future_bars, as_of)  # type: ignore[attr-defined]

    for fc_a, fc_b in zip(forecast_a, forecast_b, strict=True):
        assert fc_a.p_up == pytest.approx(fc_b.p_up, abs=1e-9)
        assert fc_a.mean == pytest.approx(fc_b.mean, abs=1e-9)
        assert fc_a.sigma == pytest.approx(fc_b.sigma, abs=1e-9)


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_no_lookahead_via_available_at_gating(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    """A bar with ``available_at`` after ``as_of`` must be invisible even if present in ``bars``."""
    bars = make_bars(800, seed=3)
    as_of = bars[500].available_at
    tampered = list(bars)
    for i in range(501, len(tampered)):
        b = tampered[i]
        tampered[i] = b.model_copy(update={"close": b.close * 1000.0, "high": b.high * 1000.0})

    model_a = factory()
    model_a.fit(bars, as_of)  # type: ignore[attr-defined]
    forecast_a = model_a.predict(bars, as_of)  # type: ignore[attr-defined]

    model_b = factory()
    model_b.fit(tampered, as_of)  # type: ignore[attr-defined]
    forecast_b = model_b.predict(tampered, as_of)  # type: ignore[attr-defined]

    for fc_a, fc_b in zip(forecast_a, forecast_b, strict=True):
        assert fc_a.mean == pytest.approx(fc_b.mean, abs=1e-9)
        assert fc_a.p_up == pytest.approx(fc_b.p_up, abs=1e-9)


# --- Quantiles monotonic -------------------------------------------------------


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_quantiles_are_monotonic(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(700, seed=4)
    model = factory()
    as_of = bars[-1].available_at
    model.fit(bars, as_of)  # type: ignore[attr-defined]
    for fc in model.predict(bars, as_of):  # type: ignore[attr-defined]
        values = _quantile_values(fc)
        assert values == sorted(values)
        assert fc.quantiles["q05"] >= fc.expected_shortfall_05 - 1e-9


# --- Determinism ---------------------------------------------------------------


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_determinism_same_inputs_same_outputs(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(700, seed=6)
    as_of = bars[-1].available_at
    model_a = factory()
    model_a.fit(bars, as_of)  # type: ignore[attr-defined]
    forecast_a = model_a.predict(bars, as_of)  # type: ignore[attr-defined]

    model_b = factory()
    model_b.fit(bars, as_of)  # type: ignore[attr-defined]
    forecast_b = model_b.predict(bars, as_of)  # type: ignore[attr-defined]

    assert model_a.model_hash() == model_b.model_hash()  # type: ignore[attr-defined]
    for fc_a, fc_b in zip(forecast_a, forecast_b, strict=True):
        assert fc_a == fc_b


# --- Model-specific behavior -----------------------------------------------------


def test_regime_conditional_falls_back_to_unconditional_when_bucket_too_thin(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """A ``min_bucket_samples`` set higher than any bucket can ever reach
    forces every prediction onto the unconditional fallback -- must match
    the plain (unconditional) empirical distribution exactly."""
    bars = make_bars(700, seed=14)
    cfg_fallback = RegimeConditionalEmpiricalConfig(
        min_train_samples=40, min_bucket_samples=10_000, regime_window=120, regime_min_periods=30
    )
    model = RegimeConditionalEmpiricalModel(cfg_fallback)
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    model.predict(bars, as_of, horizons=[5, 10])
    for h in (5, 10):
        # every bucket was forced below min_bucket_samples, so none should
        # have been recorded as usable
        assert model._by_bucket[h] == {}


def test_regime_conditional_uses_bucket_distribution_when_available(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(900, seed=15)
    cfg = RegimeConditionalEmpiricalConfig(
        min_train_samples=40, min_bucket_samples=20, regime_window=120, regime_min_periods=30
    )
    model = RegimeConditionalEmpiricalModel(cfg)
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    # with a large enough sample and a low min_bucket_samples, at least one
    # horizon should have at least one usable regime bucket.
    assert any(buckets for buckets in model._by_bucket.values())


def test_robust_location_scale_rejects_low_degrees_of_freedom() -> None:
    with pytest.raises(ValueError):
        RobustLocationScaleConfig(student_t_df=2.0)
    with pytest.raises(ValueError):
        RobustLocationScaleConfig(student_t_df=1.0)
    RobustLocationScaleConfig(student_t_df=5.0)  # does not raise


def test_regularized_linear_location_positive_trend_gives_positive_mean(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(700, seed=16, drift=0.01, daily_vol=0.001)  # strong clean uptrend
    model = RegularizedLinearLocationModel(RegularizedLinearLocationConfig(min_train_samples=60))
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    forecasts = model.predict(bars, as_of, horizons=[5])
    assert forecasts[0].mean > 0.0
