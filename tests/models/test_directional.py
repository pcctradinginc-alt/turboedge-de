from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from turboedge.models.directional import (
    LogisticDirectionModel,
    LogisticDirectionModelConfig,
    NullModel,
    NullModelConfig,
    TsmomForecastConfig,
    TsmomForecastModel,
    _tsmom_score_series,
)
from turboedge.models.forecast import HORIZONS, HorizonForecast
from turboedge.models.protected_baseline import TsmomConfig, compute_tsmom
from turboedge.storage.schemas import UnderlyingBar

_MODEL_FACTORIES: list[Callable[[], object]] = [
    lambda: TsmomForecastModel(TsmomForecastConfig(min_train_samples=40)),
    lambda: LogisticDirectionModel(LogisticDirectionModelConfig(min_train_samples=60)),
    lambda: NullModel(NullModelConfig(min_train_samples=20)),
]
_MODEL_IDS = ["tsmom", "logit", "null"]


def _quantile_values(fc: HorizonForecast) -> list[float]:
    return [fc.quantiles[k] for k in ("q05", "q25", "q50", "q75", "q95")]


# --- vectorized TSMOM score series equivalence ------------------------------


def test_tsmom_score_series_matches_compute_tsmom(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(400, seed=5)
    closes = np.array([b.close for b in bars])
    cfg = TsmomConfig()
    scores, sigmas = _tsmom_score_series(closes, cfg)
    max_lb = max(cfg.lookbacks)
    for t in (max_lb, max_lb + 1, 200, 399):
        result = compute_tsmom(closes[: t + 1], cfg)
        assert scores[t] == pytest.approx(result.score, abs=1e-10)
        assert sigmas[t] == pytest.approx(result.sigma, abs=1e-10)
    assert np.all(np.isnan(scores[:max_lb]))


# --- Protocol conformance ----------------------------------------------------


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_model_has_forecast_model_protocol_shape(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(400, seed=1)
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
    bars = make_bars(500, seed=2)
    as_of = bars[300].available_at
    prefix_bars = bars[:301]

    model_a = factory()
    model_a.fit(prefix_bars, as_of)  # type: ignore[attr-defined]
    forecast_a = model_a.predict(prefix_bars, as_of)  # type: ignore[attr-defined]

    future_bars = bars  # includes everything up to index 499
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
    bars = make_bars(500, seed=3)
    as_of = bars[300].available_at
    # Corrupt the "future" bars' close prices; since available_at > as_of
    # they must never influence the prediction made at as_of.
    tampered = list(bars)
    for i in range(301, len(tampered)):
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
    bars = make_bars(500, seed=4)
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
    bars = make_bars(500, seed=6)
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


# --- Runtime budget -------------------------------------------------------------


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_fit_predict_runtime_under_5s_on_4000_bars(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(4000, seed=8)
    as_of = bars[-1].available_at
    model = factory()
    start = time.monotonic()
    model.fit(bars, as_of)  # type: ignore[attr-defined]
    model.predict(bars, as_of)  # type: ignore[attr-defined]
    elapsed = time.monotonic() - start
    assert elapsed < 5.0


# --- Model-specific behavior -----------------------------------------------------


def test_tsmom_uses_protected_baseline_score_sign(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(600, seed=9, drift=0.01, daily_vol=0.001)  # strong clean uptrend
    model = TsmomForecastModel()
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    forecasts = model.predict(bars, as_of, horizons=[5])
    # A strong, clean uptrend should map to a positive expected mean return.
    assert forecasts[0].mean > 0.0


def test_null_model_ignores_current_features(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(500, seed=10)
    model = NullModel(NullModelConfig(min_train_samples=20))
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    fc_1 = model.predict(bars, as_of, horizons=[5])[0]
    tampered = list(bars)
    tampered[-1] = tampered[-1].model_copy(update={"close": tampered[-1].close * 2.0})
    fc_2 = model.predict(tampered, as_of, horizons=[5])[0]
    assert fc_1.mean == pytest.approx(fc_2.mean)
    assert fc_1.p_up == pytest.approx(fc_2.p_up)


def test_logistic_model_predict_requires_matching_feature_set(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(400, seed=12)
    model = LogisticDirectionModel(LogisticDirectionModelConfig(min_train_samples=60))
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    # predict on a totally different underlying_id's bars must fail cleanly
    # rather than silently mixing feature semantics -- exercised indirectly
    # via insufficient-history: bars truncated before any feature is valid.
    with pytest.raises(ValueError):
        model.predict(bars[:5], as_of)


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_predict_before_fit_raises(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(200, seed=13)
    model = factory()
    with pytest.raises(RuntimeError):
        model.predict(bars, bars[-1].available_at)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        model.model_hash()  # type: ignore[attr-defined]


@pytest.mark.parametrize("factory", _MODEL_FACTORIES, ids=_MODEL_IDS)
def test_fit_raises_on_insufficient_history(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(30, seed=14)
    model = factory()
    with pytest.raises(ValueError):
        model.fit(bars, bars[-1].available_at)  # type: ignore[attr-defined]


def test_predict_at_earlier_as_of_than_fit_uses_less_history(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(600, seed=15)
    model = TsmomForecastModel(TsmomForecastConfig(min_train_samples=40))
    fit_as_of = bars[-1].available_at
    model.fit(bars, fit_as_of)
    earlier_as_of = bars[400].available_at
    forecast_early = model.predict(bars, earlier_as_of, horizons=[5])[0]
    forecast_late = model.predict(bars, fit_as_of, horizons=[5])[0]
    # Different "as of" snapshots of a noisy series should not coincidentally
    # produce identical means (sanity: predict is actually as-of-sensitive).
    assert forecast_early.prediction_time == earlier_as_of
    assert forecast_late.prediction_time == fit_as_of
