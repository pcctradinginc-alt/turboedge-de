from __future__ import annotations

from datetime import UTC, datetime

import pytest

from turboedge.models.ensemble import combine_forecasts
from turboedge.models.forecast import HorizonForecast

_PREDICTION_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _forecast(
    *, model_id: str, mean: float, sigma: float, p_up: float, uncertainty: float = 0.001
) -> HorizonForecast:
    return HorizonForecast(
        underlying_id="DAX",
        horizon_days=5,
        prediction_time=_PREDICTION_TIME,
        p_up=p_up,
        mean=mean,
        sigma=sigma,
        quantiles={
            "q05": mean - 2 * sigma,
            "q25": mean - sigma,
            "q50": mean,
            "q75": mean + sigma,
            "q95": mean + 2 * sigma,
        },
        expected_shortfall_05=mean - 2.5 * sigma,
        uncertainty=uncertainty,
        model_id=model_id,
        model_hash="b" * 64,
        signal_family="test",
        n_train=100,
        n_effective=90.0,
    )


def test_combine_forecasts_equal_weights_averages_mean_and_p_up() -> None:
    f1 = _forecast(model_id="m1", mean=0.01, sigma=0.02, p_up=0.6)
    f2 = _forecast(model_id="m2", mean=-0.01, sigma=0.02, p_up=0.4)
    ens = combine_forecasts([f1, f2], {"m1": 1.0, "m2": 1.0})
    assert ens.mean == pytest.approx(0.0, abs=1e-9)
    assert ens.p_up == pytest.approx(0.5, abs=1e-9)
    assert ens.model_id == "ensemble"
    assert ens.signal_family == "ensemble"


def test_combine_forecasts_weighted_average_favors_higher_weight() -> None:
    f1 = _forecast(model_id="m1", mean=0.02, sigma=0.01, p_up=0.7)
    f2 = _forecast(model_id="m2", mean=0.0, sigma=0.01, p_up=0.5)
    ens = combine_forecasts([f1, f2], {"m1": 3.0, "m2": 1.0})
    expected_mean = (3.0 * 0.02 + 1.0 * 0.0) / 4.0
    assert ens.mean == pytest.approx(expected_mean)


def test_combine_forecasts_sigma_reflects_disagreement() -> None:
    # Two models that agree exactly -> ensemble sigma should equal each
    # component's sigma (no between-component variance).
    f1 = _forecast(model_id="m1", mean=0.01, sigma=0.02, p_up=0.55)
    f2 = _forecast(model_id="m2", mean=0.01, sigma=0.02, p_up=0.55)
    ens_agree = combine_forecasts([f1, f2], {"m1": 1.0, "m2": 1.0})
    assert ens_agree.sigma == pytest.approx(0.02, abs=1e-6)

    # Two models that disagree strongly on the mean -> higher ensemble sigma.
    f3 = _forecast(model_id="m1", mean=0.05, sigma=0.02, p_up=0.9)
    f4 = _forecast(model_id="m2", mean=-0.05, sigma=0.02, p_up=0.1)
    ens_disagree = combine_forecasts([f3, f4], {"m1": 1.0, "m2": 1.0})
    assert ens_disagree.sigma > ens_agree.sigma


def test_combine_forecasts_quantiles_monotonic() -> None:
    f1 = _forecast(model_id="m1", mean=0.01, sigma=0.02, p_up=0.6)
    f2 = _forecast(model_id="m2", mean=-0.02, sigma=0.03, p_up=0.35)
    ens = combine_forecasts([f1, f2], {"m1": 1.0, "m2": 2.0})
    values = [ens.quantiles[k] for k in ("q05", "q25", "q50", "q75", "q95")]
    assert values == sorted(values)


def test_combine_forecasts_uncertainty_includes_disagreement_term() -> None:
    f1 = _forecast(model_id="m1", mean=0.05, sigma=0.02, p_up=0.9, uncertainty=0.001)
    f2 = _forecast(model_id="m2", mean=-0.05, sigma=0.02, p_up=0.1, uncertainty=0.001)
    ens = combine_forecasts([f1, f2], {"m1": 1.0, "m2": 1.0})
    # Pure sqrt(sum(w^2 u^2)) term alone would be tiny; disagreement must dominate.
    naive_no_disagreement = ((0.5 * 0.001) ** 2 + (0.5 * 0.001) ** 2) ** 0.5
    assert ens.uncertainty > naive_no_disagreement + 0.01


def test_combine_forecasts_deterministic_with_fixed_seed() -> None:
    f1 = _forecast(model_id="m1", mean=0.01, sigma=0.02, p_up=0.6)
    f2 = _forecast(model_id="m2", mean=-0.02, sigma=0.03, p_up=0.35)
    ens_a = combine_forecasts([f1, f2], {"m1": 1.0, "m2": 2.0})
    ens_b = combine_forecasts([f1, f2], {"m1": 1.0, "m2": 2.0})
    assert ens_a == ens_b


def test_combine_forecasts_ignores_zero_weight_models() -> None:
    f1 = _forecast(model_id="m1", mean=0.01, sigma=0.02, p_up=0.6)
    f2 = _forecast(model_id="m2", mean=999.0, sigma=0.02, p_up=0.999)
    ens = combine_forecasts([f1, f2], {"m1": 1.0, "m2": 0.0})
    assert ens.mean == pytest.approx(0.01)
    assert ens.p_up == pytest.approx(0.6)


def test_combine_forecasts_rejects_empty() -> None:
    with pytest.raises(ValueError):
        combine_forecasts([], {"m1": 1.0})


def test_combine_forecasts_rejects_mismatched_horizon() -> None:
    f1 = _forecast(model_id="m1", mean=0.01, sigma=0.02, p_up=0.6)
    f2 = HorizonForecast(
        underlying_id="DAX",
        horizon_days=10,  # different horizon
        prediction_time=_PREDICTION_TIME,
        p_up=0.5,
        mean=0.0,
        sigma=0.02,
        quantiles={"q05": -0.04, "q25": -0.02, "q50": 0.0, "q75": 0.02, "q95": 0.04},
        expected_shortfall_05=-0.05,
        uncertainty=0.001,
        model_id="m2",
        model_hash="c" * 64,
        signal_family="test",
        n_train=100,
        n_effective=90.0,
    )
    with pytest.raises(ValueError):
        combine_forecasts([f1, f2], {"m1": 1.0, "m2": 1.0})


def test_combine_forecasts_rejects_no_positive_weight() -> None:
    f1 = _forecast(model_id="m1", mean=0.01, sigma=0.02, p_up=0.6)
    with pytest.raises(ValueError):
        combine_forecasts([f1], {"m1": 0.0})
