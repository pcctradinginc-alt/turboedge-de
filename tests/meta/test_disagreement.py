"""Model disagreement (M3)."""

from __future__ import annotations

from collections.abc import Callable

from turboedge.meta.disagreement import (
    aggregate_disagreement,
    direction_disagreement,
    forecast_dispersion,
    tail_risk_disagreement,
)


def test_identical_forecasts_have_no_disagreement(
    forecast_factory: Callable[..., object],
) -> None:
    f = [forecast_factory(m) for m in ("m1", "m2", "m3")]
    assert forecast_dispersion(f) == 0.0
    assert direction_disagreement(f) == 0.0
    assert tail_risk_disagreement(f) == 0.0
    assert aggregate_disagreement(f) == 0.0


def test_single_forecast_is_not_disagreement(
    forecast_factory: Callable[..., object],
) -> None:
    """One model cannot disagree with itself; that is not the same as
    consensus, but it must not read as conflict either."""
    assert aggregate_disagreement([forecast_factory("m1")]) == 0.0
    assert aggregate_disagreement([]) == 0.0


def test_opposite_directions_register_as_a_split(
    forecast_factory: Callable[..., object],
) -> None:
    f = [
        forecast_factory("m1", mean=0.05, p_up=0.80),
        forecast_factory("m2", mean=-0.05, p_up=0.20),
    ]
    assert direction_disagreement(f) == 0.5
    assert aggregate_disagreement(f) > 0.0


def test_tail_disagreement_is_separate_from_central_agreement(
    forecast_factory: Callable[..., object],
) -> None:
    """Two models can share a mean and still disagree entirely about the
    tails -- which for a knock-out product is the disagreement that counts.
    """
    f = [
        forecast_factory("m1", mean=0.0, sigma=0.02),
        forecast_factory("m2", mean=0.0, sigma=0.20),
    ]
    assert forecast_dispersion(f) == 0.0, "means are identical"
    assert tail_risk_disagreement(f) > 0.0, "tails are not"
    assert aggregate_disagreement(f) > 0.0


def test_aggregate_is_bounded(forecast_factory: Callable[..., object]) -> None:
    extreme = [
        forecast_factory("m1", mean=-5.0, p_up=0.01, sigma=0.001),
        forecast_factory("m2", mean=5.0, p_up=0.99, sigma=0.001),
    ]
    assert 0.0 <= aggregate_disagreement(extreme) <= 1.0
