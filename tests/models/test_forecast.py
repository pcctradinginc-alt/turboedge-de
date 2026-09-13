from __future__ import annotations

from datetime import UTC, datetime

import pytest

from turboedge.models.forecast import HORIZONS, ForecastModel, HorizonForecast, build_default_models


def _valid_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        underlying_id="DAX",
        horizon_days=5,
        prediction_time=datetime(2026, 1, 1, tzinfo=UTC),
        p_up=0.55,
        mean=0.001,
        sigma=0.02,
        quantiles={"q05": -0.03, "q25": -0.01, "q50": 0.0, "q75": 0.01, "q95": 0.03},
        expected_shortfall_05=-0.04,
        uncertainty=0.0005,
        model_id="test_model",
        model_hash="a" * 64,
        signal_family="tsmom",
        n_train=100,
        n_effective=80.0,
    )
    base.update(overrides)
    return base


def test_horizons_constant() -> None:
    assert HORIZONS == (3, 5, 7, 10, 14)


def test_horizon_forecast_valid_construction() -> None:
    fc = HorizonForecast(**_valid_kwargs())
    assert fc.horizon_days == 5


@pytest.mark.parametrize(
    "override",
    [
        {"p_up": 1.5},
        {"p_up": -0.1},
        {"sigma": -1.0},
        {"uncertainty": -0.5},
        {"horizon_days": 0},
        {"n_train": -1},
        {"n_effective": -1.0},
        {"quantiles": {"q05": -0.03, "q25": -0.01, "q50": 0.0, "q75": 0.01}},  # missing q95
        {
            "quantiles": {
                "q05": -0.03,
                "q25": -0.01,
                "q50": 0.0,
                "q75": 0.01,
                "q95": 0.03,
                "extra": 1.0,
            }
        },
    ],
)
def test_horizon_forecast_rejects_invalid_fields(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HorizonForecast(**_valid_kwargs(**override))


def test_horizon_forecast_is_frozen() -> None:
    fc = HorizonForecast(**_valid_kwargs())
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError, dataclass-internal
        fc.p_up = 0.9  # type: ignore[misc]


def test_build_default_models_returns_three_protected_baselines() -> None:
    models = build_default_models()
    assert len(models) == 3
    signal_families = {m.signal_family for m in models}
    assert signal_families == {"tsmom", "logit", "null"}
    for model in models:
        assert isinstance(model, ForecastModel)
        assert hasattr(model, "model_id")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "model_hash")
