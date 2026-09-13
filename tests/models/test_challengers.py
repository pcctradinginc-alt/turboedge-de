from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.models.challengers import (
    CrossAssetLeadLag,
    CrossAssetLeadLagConfig,
    LowVolRegimeTrend,
    LowVolRegimeTrendConfig,
    SeasonalityTurnOfMonth,
    SeasonalityTurnOfMonthConfig,
    ShortHorizonReversal,
    ShortHorizonReversalConfig,
    VixTermStructure,
    VixTermStructureConfig,
    VolTargetedTsmom,
    VolTargetedTsmomConfig,
    _trailing_tercile_gate,
    _turn_of_month_bucket,
    _zscore_lookback,
)
from turboedge.models.forecast import HORIZONS, HorizonForecast
from turboedge.storage.schemas import UnderlyingBar

# --- factories for the 4 single-underlying challengers (no auxiliary bars needed) -------------

_SIMPLE_FACTORIES: list[Callable[[], object]] = [
    lambda: VolTargetedTsmom(VolTargetedTsmomConfig(min_train_samples=40)),
    lambda: LowVolRegimeTrend(
        LowVolRegimeTrendConfig(min_train_samples=40, regime_window=100, regime_min_periods=30)
    ),
    lambda: ShortHorizonReversal(ShortHorizonReversalConfig(min_train_samples=40)),
    lambda: SeasonalityTurnOfMonth(SeasonalityTurnOfMonthConfig(min_train_samples=15)),
]
_SIMPLE_IDS = ["voltarget_tsmom", "lowvol_regime_trend", "reversal_short_horizon", "seasonality"]


def _quantile_values(fc: HorizonForecast) -> list[float]:
    return [fc.quantiles[k] for k in ("q05", "q25", "q50", "q75", "q95")]


# --- Protocol conformance ----------------------------------------------------------------------


@pytest.mark.parametrize("factory", _SIMPLE_FACTORIES, ids=_SIMPLE_IDS)
def test_model_has_forecast_model_protocol_shape(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(500, seed=1)
    model = factory()
    assert hasattr(model, "model_id")
    assert hasattr(model, "signal_family")
    assert isinstance(model.signal_family, str) and model.signal_family  # type: ignore[attr-defined]
    as_of = bars[-1].available_at
    model.fit(bars, as_of)  # type: ignore[attr-defined]
    forecasts = model.predict(bars, as_of)  # type: ignore[attr-defined]
    assert isinstance(forecasts, list)
    assert len(forecasts) == len(HORIZONS)
    for fc in forecasts:
        assert isinstance(fc, HorizonForecast)
        assert 0.0 <= fc.p_up <= 1.0
    h = model.model_hash()  # type: ignore[attr-defined]
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)  # valid hex


# --- No look-ahead (appending future bars must not change an as-of prediction) ----------------


@pytest.mark.parametrize("factory", _SIMPLE_FACTORIES, ids=_SIMPLE_IDS)
def test_no_lookahead_appending_future_bars_does_not_change_prediction(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(500, seed=2)
    as_of = bars[300].available_at
    prefix_bars = bars[:301]

    model_a = factory()
    model_a.fit(prefix_bars, as_of)  # type: ignore[attr-defined]
    forecast_a = model_a.predict(prefix_bars, as_of)  # type: ignore[attr-defined]

    model_b = factory()
    model_b.fit(bars, as_of)  # type: ignore[attr-defined]
    forecast_b = model_b.predict(bars, as_of)  # type: ignore[attr-defined]

    for fc_a, fc_b in zip(forecast_a, forecast_b, strict=True):
        assert fc_a.p_up == pytest.approx(fc_b.p_up, abs=1e-9)
        assert fc_a.mean == pytest.approx(fc_b.mean, abs=1e-9)
        assert fc_a.sigma == pytest.approx(fc_b.sigma, abs=1e-9)


@pytest.mark.parametrize("factory", _SIMPLE_FACTORIES, ids=_SIMPLE_IDS)
def test_no_lookahead_via_available_at_gating(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(500, seed=3)
    as_of = bars[300].available_at
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


# --- Quantiles monotonic + calibration in [0,1] -------------------------------------------------


@pytest.mark.parametrize("factory", _SIMPLE_FACTORIES, ids=_SIMPLE_IDS)
def test_quantiles_are_monotonic_and_p_up_in_unit_interval(
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
        assert 0.0 <= fc.p_up <= 1.0


# --- Determinism ---------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _SIMPLE_FACTORIES, ids=_SIMPLE_IDS)
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


# --- predict/model_hash before fit, insufficient history --------------------------------------


@pytest.mark.parametrize("factory", _SIMPLE_FACTORIES, ids=_SIMPLE_IDS)
def test_predict_before_fit_raises(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(200, seed=13)
    model = factory()
    with pytest.raises(RuntimeError):
        model.predict(bars, bars[-1].available_at)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        model.model_hash()  # type: ignore[attr-defined]


@pytest.mark.parametrize("factory", _SIMPLE_FACTORIES, ids=_SIMPLE_IDS)
def test_fit_raises_on_insufficient_history(
    factory: Callable[[], object], make_bars: Callable[..., list[UnderlyingBar]]
) -> None:
    bars = make_bars(25, seed=14)
    model = factory()
    with pytest.raises(ValueError):
        model.fit(bars, bars[-1].available_at)  # type: ignore[attr-defined]


# --- family-specific sanity checks -------------------------------------------------------------


def test_voltarget_tsmom_positive_mean_in_clean_uptrend(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(600, seed=9, drift=0.01, daily_vol=0.001)
    model = VolTargetedTsmom(VolTargetedTsmomConfig(min_train_samples=40))
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    forecasts = model.predict(bars, as_of, horizons=[5])
    assert forecasts[0].mean > 0.0


def _make_mean_reverting_bars(
    n: int, *, seed: int, ar_coef: float = -0.35, daily_vol: float = 0.01
) -> list[UnderlyingBar]:
    """Synthetic bars whose daily log returns follow ``r_t = ar_coef*r_{t-1} + noise``.

    A genuinely negatively-autocorrelated return process, so a
    vol-normalized short-horizon reversal signal has a real, reliable
    (not noise-dependent) sign to recover -- unlike a plain random walk,
    where a fitted reversal beta's sign is arbitrary.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, daily_vol, size=n)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = ar_coef * rets[i - 1] + noise[i]
    closes = 100.0 * np.exp(np.cumsum(rets))
    start = datetime(2010, 1, 1, tzinfo=UTC)
    bars = []
    for i in range(n):
        ts = start + timedelta(days=i)
        c = float(closes[i])
        bars.append(
            UnderlyingBar(
                underlying_id="TEST",
                ts=ts,
                open=c,
                high=c * 1.001,
                low=c * 0.999,
                close=c,
                volume=1000.0,
                observation_time=ts,
                available_at=ts,
                retrieved_at=ts,
                source="synthetic",
                parser_version="1",
                quality_score=0.9,
            )
        )
    return bars


def test_reversal_short_horizon_learns_positive_beta_on_mean_reverting_data() -> None:
    """On a genuinely negatively-autocorrelated (mean-reverting) return
    process, the reversal family's fitted ridge coefficient must be
    positive: a large negative recent move (reversal score > 0) should be
    associated with a positive expected next-horizon return."""
    bars = _make_mean_reverting_bars(1200, seed=20, ar_coef=-0.4)
    model = ShortHorizonReversal(ShortHorizonReversalConfig(min_train_samples=40))
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    beta = model._fits[5].ridge.coef[0]  # type: ignore[attr-defined]
    assert beta > 0.0


def test_turn_of_month_bucket_boundaries() -> None:
    assert _turn_of_month_bucket(datetime(2024, 3, 1, tzinfo=UTC)) is True
    assert _turn_of_month_bucket(datetime(2024, 3, 3, tzinfo=UTC)) is True
    assert _turn_of_month_bucket(datetime(2024, 3, 4, tzinfo=UTC)) is False
    assert _turn_of_month_bucket(datetime(2024, 3, 25, tzinfo=UTC)) is False
    assert _turn_of_month_bucket(datetime(2024, 3, 26, tzinfo=UTC)) is True
    assert _turn_of_month_bucket(datetime(2024, 3, 31, tzinfo=UTC)) is True


def test_seasonality_bucket_stats_differ_between_buckets(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(1500, seed=21)
    model = SeasonalityTurnOfMonth(SeasonalityTurnOfMonthConfig(min_train_samples=15))
    as_of = bars[-1].available_at
    model.fit(bars, as_of)
    stats_by_h = model._stats  # type: ignore[attr-defined]
    for h in HORIZONS:
        buckets = stats_by_h[h]
        # Both buckets fitted independently -- generally different sample
        # means/effective sample sizes (not required to differ by much, but
        # must be genuinely two distinct empirical fits, not the same object).
        assert buckets[True] is not buckets[False]
        same_n = buckets[True].n_train == buckets[False].n_train
        same_mean = buckets[True].mean == pytest.approx(buckets[False].mean)
        assert not (same_n and same_mean)


def test_zscore_lookback_zero_or_nan_sigma_yields_nan() -> None:
    closes = np.array([100.0, 100.0, 100.0, 100.0, 110.0], dtype=np.float64)
    sigma = np.array([0.0, 0.0, 0.0, 0.0, 0.01], dtype=np.float64)
    out = _zscore_lookback(closes, sigma, k=2, clip=3.0)
    assert np.isnan(out[2])  # sigma[2] == 0 -> NaN, not an extreme clipped value
    assert np.isfinite(out[4])


def test_trailing_tercile_gate_causal_and_bounded() -> None:
    rng = np.random.default_rng(0)
    sigma = np.abs(rng.normal(0.01, 0.005, size=300))
    gate = _trailing_tercile_gate(sigma, window=100, min_periods=30, tercile=1.0 / 3.0)
    finite = gate[np.isfinite(gate)]
    assert set(np.unique(finite)).issubset({0.0, 1.0})
    assert np.all(np.isnan(gate[:29]))
    # Truncating the series must not change an already-computed gate value (causality).
    gate_prefix = _trailing_tercile_gate(sigma[:150], window=100, min_periods=30, tercile=1.0 / 3.0)
    assert gate[149] == pytest.approx(gate_prefix[149]) or (
        np.isnan(gate[149]) and np.isnan(gate_prefix[149])
    )


# --- Cross-asset-dependent challengers (VixTermStructure, CrossAssetLeadLag) ------------------


def _aux(
    make_bars: Callable[..., list[UnderlyingBar]], underlying_id: str, n: int, seed: int
) -> list[UnderlyingBar]:
    return make_bars(n, seed=seed, underlying_id=underlying_id)


def test_vix_term_structure_protocol_and_determinism(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(500, seed=30)
    vix = _aux(make_bars, "VIX", 500, 31)
    vix9d = _aux(make_bars, "VIX9D", 500, 32)
    vix3m = _aux(make_bars, "VIX3M", 500, 33)

    def factory() -> VixTermStructure:
        return VixTermStructure(vix, vix9d, vix3m, VixTermStructureConfig(min_train_samples=60))

    as_of = bars[-1].available_at
    model_a = factory()
    model_a.fit(bars, as_of)
    forecasts_a = model_a.predict(bars, as_of)
    assert len(forecasts_a) == len(HORIZONS)
    for fc in forecasts_a:
        assert 0.0 <= fc.p_up <= 1.0
        values = _quantile_values(fc)
        assert values == sorted(values)

    model_b = factory()
    model_b.fit(bars, as_of)
    forecasts_b = model_b.predict(bars, as_of)
    assert model_a.model_hash() == model_b.model_hash()
    for fc_a, fc_b in zip(forecasts_a, forecasts_b, strict=True):
        assert fc_a == fc_b


def test_vix_term_structure_no_lookahead_on_primary_bars(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(500, seed=34)
    vix = _aux(make_bars, "VIX", 500, 35)
    vix9d = _aux(make_bars, "VIX9D", 500, 36)
    vix3m = _aux(make_bars, "VIX3M", 500, 37)
    as_of = bars[300].available_at

    model_a = VixTermStructure(vix, vix9d, vix3m, VixTermStructureConfig(min_train_samples=60))
    model_a.fit(bars[:301], as_of)
    fc_a = model_a.predict(bars[:301], as_of)

    model_b = VixTermStructure(vix, vix9d, vix3m, VixTermStructureConfig(min_train_samples=60))
    model_b.fit(bars, as_of)
    fc_b = model_b.predict(bars, as_of)

    for a, b in zip(fc_a, fc_b, strict=True):
        assert a.mean == pytest.approx(b.mean, abs=1e-9)
        assert a.p_up == pytest.approx(b.p_up, abs=1e-9)


def test_vix_term_structure_ignores_future_auxiliary_bars(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """Appending a wildly-different-valued FUTURE vix bar (dated after as_of)
    must not change the prediction -- the model holds full aux history but
    align_auxiliary_series enforces available_at <= as_of per entry."""
    bars = make_bars(500, seed=38)
    vix = _aux(make_bars, "VIX", 500, 39)
    vix9d = _aux(make_bars, "VIX9D", 500, 40)
    vix3m = _aux(make_bars, "VIX3M", 500, 41)
    as_of = bars[300].available_at

    model_a = VixTermStructure(vix, vix9d, vix3m, VixTermStructureConfig(min_train_samples=60))
    model_a.fit(bars, as_of)
    fc_a = model_a.predict(bars, as_of)

    tampered_vix = list(vix)
    for i in range(301, len(tampered_vix)):
        b = tampered_vix[i]
        tampered_vix[i] = b.model_copy(update={"close": b.close * 50.0})
    model_b = VixTermStructure(
        tampered_vix, vix9d, vix3m, VixTermStructureConfig(min_train_samples=60)
    )
    model_b.fit(bars, as_of)
    fc_b = model_b.predict(bars, as_of)

    for a, b in zip(fc_a, fc_b, strict=True):
        assert a.mean == pytest.approx(b.mean, abs=1e-9)
        assert a.p_up == pytest.approx(b.p_up, abs=1e-9)


def test_cross_asset_leadlag_protocol_and_determinism(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(500, seed=50)
    spx = _aux(make_bars, "SPX", 500, 51)
    ndx = _aux(make_bars, "NDX", 500, 52)
    vix = _aux(make_bars, "VIX", 500, 53)
    eurusd = _aux(make_bars, "EURUSD", 500, 54)
    tnx = _aux(make_bars, "TNX", 500, 55)

    def factory() -> CrossAssetLeadLag:
        return CrossAssetLeadLag(
            spx, ndx, vix, eurusd, tnx, CrossAssetLeadLagConfig(min_train_samples=60)
        )

    as_of = bars[-1].available_at
    model_a = factory()
    model_a.fit(bars, as_of)
    forecasts_a = model_a.predict(bars, as_of)
    assert len(forecasts_a) == len(HORIZONS)
    for fc in forecasts_a:
        assert 0.0 <= fc.p_up <= 1.0
        values = _quantile_values(fc)
        assert values == sorted(values)

    model_b = factory()
    model_b.fit(bars, as_of)
    forecasts_b = model_b.predict(bars, as_of)
    assert model_a.model_hash() == model_b.model_hash()
    for fc_a, fc_b in zip(forecasts_a, forecasts_b, strict=True):
        assert fc_a == fc_b


def test_cross_asset_leadlag_ignores_future_auxiliary_bars(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(500, seed=56)
    spx = _aux(make_bars, "SPX", 500, 57)
    ndx = _aux(make_bars, "NDX", 500, 58)
    vix = _aux(make_bars, "VIX", 500, 59)
    eurusd = _aux(make_bars, "EURUSD", 500, 60)
    tnx = _aux(make_bars, "TNX", 500, 61)
    as_of = bars[300].available_at

    model_a = CrossAssetLeadLag(
        spx, ndx, vix, eurusd, tnx, CrossAssetLeadLagConfig(min_train_samples=60)
    )
    model_a.fit(bars, as_of)
    fc_a = model_a.predict(bars, as_of)

    tampered_spx = list(spx)
    for i in range(301, len(tampered_spx)):
        b = tampered_spx[i]
        tampered_spx[i] = b.model_copy(update={"close": b.close * 50.0})
    model_b = CrossAssetLeadLag(
        tampered_spx, ndx, vix, eurusd, tnx, CrossAssetLeadLagConfig(min_train_samples=60)
    )
    model_b.fit(bars, as_of)
    fc_b = model_b.predict(bars, as_of)

    for a, b in zip(fc_a, fc_b, strict=True):
        assert a.mean == pytest.approx(b.mean, abs=1e-9)
        assert a.p_up == pytest.approx(b.p_up, abs=1e-9)


def test_cross_asset_leadlag_no_lookahead_on_primary_bars(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(500, seed=62)
    spx = _aux(make_bars, "SPX", 500, 63)
    ndx = _aux(make_bars, "NDX", 500, 64)
    vix = _aux(make_bars, "VIX", 500, 65)
    eurusd = _aux(make_bars, "EURUSD", 500, 66)
    tnx = _aux(make_bars, "TNX", 500, 67)
    as_of = bars[300].available_at

    model_a = CrossAssetLeadLag(
        spx, ndx, vix, eurusd, tnx, CrossAssetLeadLagConfig(min_train_samples=60)
    )
    model_a.fit(bars[:301], as_of)
    fc_a = model_a.predict(bars[:301], as_of)

    model_b = CrossAssetLeadLag(
        spx, ndx, vix, eurusd, tnx, CrossAssetLeadLagConfig(min_train_samples=60)
    )
    model_b.fit(bars, as_of)
    fc_b = model_b.predict(bars, as_of)

    for a, b in zip(fc_a, fc_b, strict=True):
        assert a.mean == pytest.approx(b.mean, abs=1e-9)
        assert a.p_up == pytest.approx(b.p_up, abs=1e-9)
