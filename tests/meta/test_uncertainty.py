"""Uncertainty decomposition and abstention (M4)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from turboedge.meta.schemas import ModelTrust
from turboedge.meta.uncertainty import (
    assess,
    calibration_uncertainty,
    data_uncertainty,
    epistemic_uncertainty,
    product_data_uncertainty,
    regime_uncertainty,
)


def trust(score: float, *, calibration: float | None = None) -> ModelTrust:
    return ModelTrust(
        model_id="m",
        signal_family="f",
        calibration_quality=calibration,
        data_quality=1.0,
        drift_penalty=1.0,
        uncertainty_penalty=1.0,
        trust_score=score,
    )


def test_epistemic_uses_the_best_model_not_the_average() -> None:
    """One good model is not discredited by sitting next to poor ones."""
    assert epistemic_uncertainty([trust(0.8), trust(0.0), trust(0.0)]) == pytest.approx(0.2)


def test_no_models_means_maximal_epistemic_uncertainty() -> None:
    assert epistemic_uncertainty([]) == 1.0


def test_data_uncertainty_takes_the_worse_of_health_and_staleness() -> None:
    """Fresh quotes do not repair a failed health check, and vice versa."""
    assert data_uncertainty(1.0, 0.9) == 0.9
    assert data_uncertainty(0.2, 0.0) == 0.8


def test_unseen_regime_is_maximal_regime_uncertainty() -> None:
    assert regime_uncertainty(0) == 1.0
    assert regime_uncertainty(100) == 0.0
    assert 0.0 < regime_uncertainty(20) < 1.0


def test_unmeasured_calibration_is_maximal_not_neutral() -> None:
    """With `walkforward_results` empty, a stated probability has never been
    checked against outcomes. Treating that as neutral is how an
    uncalibrated number ends up sizing a position."""
    assert calibration_uncertainty([trust(0.5), trust(0.5)]) == 1.0
    assert calibration_uncertainty([trust(0.5, calibration=0.9)]) < 0.2


def test_product_data_uncertainty_is_its_own_axis() -> None:
    """Market data can be perfect while the certificate's own terms are
    unverified -- which alone makes a payoff calculation meaningless."""
    assert product_data_uncertainty(ratio_unverified_share=0.4, integrity_fail_share=0.0) == 0.4
    assert product_data_uncertainty(ratio_unverified_share=0.0, integrity_fail_share=0.6) == 0.6


def test_one_disqualifying_unknown_can_drive_abstention_alone(
    forecast_factory: Callable[..., object],
) -> None:
    """A weighted mean alone would let five comfortable numbers drown out one
    disqualifying one; the max term prevents that."""
    c = assess(
        trusts=[trust(0.9, calibration=0.95)],
        forecasts=[forecast_factory("m1")],
        data_quality=1.0,
        stale_share=0.0,
        regime_observation_count=0,  # never-seen regime: the single unknown
        ratio_unverified_share=0.0,
        integrity_fail_share=0.0,
    )
    assert c.regime_uncertainty == 1.0
    assert c.abstain_score >= 0.5


def test_everything_known_and_clean_yields_low_abstention(
    forecast_factory: Callable[..., object],
) -> None:
    c = assess(
        trusts=[trust(0.9, calibration=0.95), trust(0.85, calibration=0.9)],
        forecasts=[forecast_factory("m1"), forecast_factory("m2")],
        data_quality=1.0,
        stale_share=0.0,
        regime_observation_count=300,
        ratio_unverified_share=0.0,
        integrity_fail_share=0.0,
    )
    assert c.abstain_score < 0.3


def test_all_components_stay_in_unit_interval(
    forecast_factory: Callable[..., object],
) -> None:
    c = assess(
        trusts=[],
        forecasts=[],
        data_quality=-5.0,
        stale_share=9.0,
        regime_observation_count=0,
        ratio_unverified_share=9.0,
        integrity_fail_share=-3.0,
    )
    for value in c.model_dump().values():
        assert 0.0 <= value <= 1.0
