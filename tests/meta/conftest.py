"""Shared fixtures for the meta-layer tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.models.forecast import HorizonForecast
from turboedge.storage.schemas import (
    ModelRegistryEntry,
    ModelStatus,
    UnderlyingBar,
    WalkforwardResultRecord,
)

START = datetime(2024, 1, 2, tzinfo=UTC)


def make_bar(day: int, close: float) -> UnderlyingBar:
    ts = START + timedelta(days=day)
    return UnderlyingBar(
        underlying_id="DAX",
        ts=ts,
        open=close,
        high=close * 1.005,
        low=close * 0.995,
        close=close,
        volume=1000.0,
        observation_time=ts,
        available_at=ts,
        retrieved_at=ts,
        source="test",
        parser_version="1",
        quality_score=1.0,
    )


def make_bars(n: int = 400, *, seed: int = 7, drift: float = 0.0) -> list[UnderlyingBar]:
    rng = np.random.default_rng(seed)
    price, bars = 100.0, []
    for i in range(n):
        price *= float(np.exp(drift + rng.normal(0.0, 0.01)))
        bars.append(make_bar(i, price))
    return bars


def make_forecast(
    model_id: str,
    *,
    mean: float = 0.01,
    p_up: float = 0.55,
    sigma: float = 0.05,
    uncertainty: float = 0.01,
) -> HorizonForecast:
    return HorizonForecast(
        underlying_id="DAX",
        horizon_days=7,
        prediction_time=START,
        p_up=p_up,
        mean=mean,
        sigma=sigma,
        quantiles={
            "q05": mean - 2 * sigma,
            "q25": mean - 0.5 * sigma,
            "q50": mean,
            "q75": mean + 0.5 * sigma,
            "q95": mean + 2 * sigma,
        },
        expected_shortfall_05=mean - 2.5 * sigma,
        uncertainty=uncertainty,
        model_id=model_id,
        model_hash="h",
        signal_family=model_id,
        n_train=500,
        n_effective=400.0,
    )


def make_entry(model_id: str) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        model_id=model_id,
        model_hash="h",
        signal_family=model_id,
        status=ModelStatus.CHAMPION,
        weight=1.0,
        params={},
        created_at=START,
        updated_at=START,
    )


def make_walkforward(
    model_id: str, *, brier: float = 0.20, brier_null: float = 0.25, ece: float = 0.01
) -> WalkforwardResultRecord:
    return WalkforwardResultRecord(
        model_id=model_id,
        signal_family=model_id,
        underlying_id="DAX",
        horizon_days=7,
        evaluated_at=START,
        n_folds=20,
        brier=brier,
        brier_null=brier_null,
        log_loss=0.6,
        ece=ece,
        hit_rate=0.52,
        mean_oos_return=0.0,
        psr=0.5,
        n_effective=300.0,
        config_hash="c",
        params={},
        pinball_loss={},
    )


# Exposed as fixtures rather than imported directly: `tests/meta` is not a
# package (no __init__.py anywhere under tests/), so a relative import fails.
# This matches how tests/pipeline shares its builders.


@pytest.fixture
def bars() -> list[UnderlyingBar]:
    return make_bars()


@pytest.fixture
def bars_factory() -> Callable[..., list[UnderlyingBar]]:
    return make_bars


@pytest.fixture
def forecast_factory() -> Callable[..., HorizonForecast]:
    return make_forecast


@pytest.fixture
def entry_factory() -> Callable[..., ModelRegistryEntry]:
    return make_entry


@pytest.fixture
def walkforward_factory() -> Callable[..., WalkforwardResultRecord]:
    return make_walkforward


@pytest.fixture
def start() -> datetime:
    return START
