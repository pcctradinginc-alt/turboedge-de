"""Tests for the alpha-centric schemas (Alpha Factory, Phase A)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from turboedge.alpha.schemas import (
    PRODUCTION_STATES,
    TERMINAL_STATES,
    AlphaSource,
    AlphaStatus,
    EdgeAttribution,
    transition_allowed,
)
from turboedge.meta.research_opportunity import Estimate, InformationFamily

_NOW = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)


def _unknown() -> Estimate:
    return Estimate.unknown("not measured yet")


def _alpha(**over: object) -> AlphaSource:
    defaults: dict[str, object] = dict(
        alpha_id="A-1",
        name="issuer spread widening",
        family=InformationFamily.MICROSTRUCTURE,
        version="1",
        description="d",
        economic_hypothesis="h",
        created_at=_NOW,
        nominal_sample=_unknown(),
        effective_sample=_unknown(),
        expected_net_ev=_unknown(),
        lcb_net_ev=_unknown(),
        posterior_probability_positive=_unknown(),
        uncertainty_score=_unknown(),
        drift_score=_unknown(),
        decay_score=_unknown(),
    )
    defaults.update(over)
    return AlphaSource.model_validate(defaults)


def test_alpha_source_round_trips_and_keeps_estimate_provenance() -> None:
    """An unmeasured quantity must still read as unmeasured after a round trip.

    A `net_ev` of 0.0 because nothing was measured and one measured at zero are
    different claims; collapsing them is how an empty table becomes confidence.
    """
    restored = AlphaSource.model_validate_json(_alpha().model_dump_json())

    assert restored == _alpha()
    assert not restored.expected_net_ev.is_known
    assert restored.expected_net_ev.value is None


def test_disabled_is_reachable_from_every_production_state() -> None:
    """An emergency stop that has to walk a ladder is not an emergency stop."""
    for state in PRODUCTION_STATES:
        assert transition_allowed(state, AlphaStatus.DISABLED)


def test_disabled_is_not_reachable_from_a_terminal_state() -> None:
    for state in TERMINAL_STATES:
        assert not transition_allowed(state, AlphaStatus.DISABLED)


def test_the_research_ladder_cannot_be_skipped() -> None:
    """Untouched confirmation evidence is the one thing exploration cannot fake."""
    assert not transition_allowed(AlphaStatus.EXPLORATORY, AlphaStatus.NORMAL_PRODUCTION)
    assert not transition_allowed(AlphaStatus.VALIDATED, AlphaStatus.FORWARD_SHADOW)
    assert not transition_allowed(AlphaStatus.IDEA, AlphaStatus.CONFIRMATORY)


def test_demotion_is_graduated_one_rung_at_a_time() -> None:
    assert transition_allowed(AlphaStatus.NORMAL_PRODUCTION, AlphaStatus.LIMITED_PRODUCTION)
    assert transition_allowed(AlphaStatus.LIMITED_PRODUCTION, AlphaStatus.CANARY_PRODUCTION)
    assert transition_allowed(AlphaStatus.CANARY_PRODUCTION, AlphaStatus.FORWARD_SHADOW)
    assert transition_allowed(AlphaStatus.FORWARD_SHADOW, AlphaStatus.DORMANT)


def test_a_rejected_alpha_is_terminal_but_still_a_valid_object() -> None:
    """Failure memory is the point, not a cleanup target (§7, §38)."""
    rejected = _alpha(status=AlphaStatus.REJECTED)

    assert rejected.status is AlphaStatus.REJECTED
    assert not transition_allowed(AlphaStatus.REJECTED, AlphaStatus.EXPLORATORY)
    assert AlphaSource.model_validate_json(rejected.model_dump_json()) == rejected


def test_dormant_can_be_retested_but_rejected_cannot() -> None:
    assert transition_allowed(AlphaStatus.DORMANT, AlphaStatus.EXPLORATORY)
    assert not transition_allowed(AlphaStatus.REJECTED, AlphaStatus.EXPLORATORY)


@pytest.mark.parametrize(
    "status",
    [AlphaStatus.CONFIRMATORY, AlphaStatus.FORWARD_SHADOW, AlphaStatus.NORMAL_PRODUCTION],
)
def test_confirmatory_and_production_require_a_freeze_timestamp(status: AlphaStatus) -> None:
    """Confirmatory evidence means nothing if the hypothesis could still move."""
    with pytest.raises(ValidationError, match="requires frozen_at"):
        _alpha(status=status)

    assert _alpha(status=status, frozen_at=_NOW).frozen_at == _NOW


def test_exploratory_does_not_require_a_freeze() -> None:
    assert _alpha(status=AlphaStatus.EXPLORATORY).frozen_at is None


def _attribution(**over: object) -> EdgeAttribution:
    defaults: dict[str, object] = dict(
        decision_id="D-1",
        prediction_time=_NOW,
        underlying="DAX",
        horizon=7,
        total_incremental_net_ev=0.0,
    )
    defaults.update(over)
    return EdgeAttribution.model_validate(defaults)


def test_edge_attribution_reconciles_when_components_add_up() -> None:
    attribution = _attribution(
        total_incremental_net_ev=0.05,
        forecast_edge=-0.02,
        product_selection_edge=0.06,
        cost_selection_edge=0.01,
    )

    assert attribution.component_sum == pytest.approx(0.05)


def test_edge_attribution_raises_rather_than_silently_balancing() -> None:
    """An attribution that quietly balances itself is decoration.

    If the parts do not add up the decomposition is wrong, and the number it
    produces should not be used.
    """
    with pytest.raises(ValidationError, match="do not reconcile"):
        _attribution(total_incremental_net_ev=0.05, forecast_edge=0.01)


def test_the_residual_carries_what_the_components_do_not_explain() -> None:
    """The residual measures how much is NOT understood, and must be usable as such."""
    attribution = _attribution(
        total_incremental_net_ev=0.05,
        product_selection_edge=0.01,
        interaction_residual=0.04,
    )

    assert attribution.interaction_residual == pytest.approx(0.04)
    assert attribution.component_sum == pytest.approx(attribution.total_incremental_net_ev)


def test_value_cannot_all_hide_under_one_component_without_saying_so() -> None:
    """Spec §8: the decomposition exists so value cannot hide under 'model alpha'.

    Attributing everything to the forecast is allowed -- but it must be stated
    explicitly in that field, not smuggled in by leaving the total unexplained.
    """
    with pytest.raises(ValidationError):
        _attribution(total_incremental_net_ev=0.05)

    explicit = _attribution(total_incremental_net_ev=0.05, forecast_edge=0.05)
    assert explicit.forecast_edge == pytest.approx(0.05)
