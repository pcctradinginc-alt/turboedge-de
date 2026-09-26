"""Tests for the failed-hypothesis / positive-pattern research memory (§9, §10)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from turboedge.learning.failed_hypotheses import FailedHypothesis
from turboedge.meta.research_memory import (
    enrich,
    family_redundancy,
    measure_current_uncertainty,
    measure_data_availability,
    pattern_support,
    prior_failure_similarity,
    record_successful_pattern,
    similar_failures,
)
from turboedge.meta.research_opportunity import (
    Estimate,
    EstimateBasis,
    InformationFamily,
    ResearchOpportunity,
    ResearchStatus,
    SuccessfulResearchPattern,
)
from turboedge.meta.schemas import DecisionConfidence, MetaDecision, MetaDecisionKind

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def _declared(value: float = 0.5) -> Estimate:
    return Estimate.declared(value, note="test fixture")


def make_opportunity(
    hypothesis_id: str,
    *,
    information_family: InformationFamily,
    description: str = "",
    affected_underlyings: list[str] | None = None,
    affected_horizons: list[str] | None = None,
    status: ResearchStatus = ResearchStatus.PROPOSED,
) -> ResearchOpportunity:
    return ResearchOpportunity(
        hypothesis_id=hypothesis_id,
        description=description or hypothesis_id,
        information_family=information_family,
        affected_underlyings=affected_underlyings or [],
        affected_horizons=affected_horizons or [],
        expected_information_gain=_declared(),
        expected_economic_value=_declared(),
        probability_of_resolving_uncertainty=_declared(),
        implementation_cost=_declared(),
        implementation_complexity=_declared(),
        estimated_sample_size=_declared(),
        current_uncertainty=_declared(),
        data_availability=_declared(),
        leakage_risk=_declared(),
        overlap_with_existing_research=_declared(),
        status=status,
        approved_by="tester" if status is not ResearchStatus.PROPOSED else None,
    )


def make_failed(
    feature: str,
    horizons: list[str],
    *,
    status: str = "dormant",
    regime_change_note: str | None = None,
    recorded_at: datetime = NOW,
) -> FailedHypothesis:
    return FailedHypothesis(
        feature=feature,
        horizons=horizons,
        incremental_net_ev=-0.0001,
        effective_sample=48304,
        status=status,  # type: ignore[arg-type]
        trial_id=None,
        recorded_at=recorded_at,
        regime_change_note=regime_change_note,
    )


def make_decision(
    *,
    underlying_id: str = "DAX",
    horizon_days: int = 7,
    epistemic: float = 0.5,
    regime: float = 0.5,
    calibration: float = 0.5,
    product_data: float = 0.5,
    model_disagreement: float = 0.5,
) -> MetaDecision:
    return MetaDecision(
        run_id="run1",
        underlying_id=underlying_id,
        horizon_days=horizon_days,
        prediction_time=NOW,
        decided_at=NOW,
        volatility_regime="normal",
        trend_regime="neutral",
        regime_observation_count=50,
        available_models=["m1"],
        selected_models=["m1"],
        model_weights={"m1": 1.0},
        model_trust=[],
        confidence=DecisionConfidence(
            epistemic_uncertainty=epistemic,
            data_uncertainty=0.5,
            regime_uncertainty=regime,
            model_disagreement=model_disagreement,
            calibration_uncertainty=calibration,
            product_data_uncertainty=product_data,
            abstain_score=0.2,
        ),
        final_confidence=0.5,
        decision=MetaDecisionKind.WATCH_ONLY,
        reasons=["test"],
        config_hash="c",
    )


def make_pattern(
    *,
    information_family: InformationFamily = InformationFamily.VOLATILITY_SURFACE,
    feature: str = "vix_term_structure",
    stability: float = 0.8,
    decay_since_discovery: float | None = 0.1,
    discovered_at: datetime = NOW,
) -> SuccessfulResearchPattern:
    return SuccessfulResearchPattern(
        pattern_id="P1",
        information_family=information_family,
        feature=feature,
        underlying_id="DAX",
        horizon="5d",
        volatility_regime="normal",
        trend_regime="neutral",
        oos_effect=0.001,
        effective_sample=10000,
        stability=stability,
        economic_value=0.002,
        discovered_at=discovered_at,
        last_confirmed_at=None,
        decay_since_discovery=decay_since_discovery,
    )


# --------------------------------------------------------------------------
# similar_failures / prior_failure_similarity
# --------------------------------------------------------------------------


def test_vix_term_structure_overlaps_with_cboe_volatility_state() -> None:
    """The exact overlap named in the contract: `vix_term_structure` is a
    dormant failed hypothesis, and a catalog idea for a CBOE volatility
    state indicator sits in the same VOLATILITY_SURFACE family. Even
    though the words themselves barely overlap ("vix/term/structure" vs
    "cboe/volatility/state"), the family-keyword signal must catch it.

    This also serves as one of the two mandatory anti-triviality checks:
    an always-returns-0 similarity function would fail this assertion.
    """
    opportunity = make_opportunity(
        "cboe_volatility_state",
        information_family=InformationFamily.VOLATILITY_SURFACE,
        description="CBOE volatility state indicator from six official Cboe series",
        affected_horizons=["5d", "10d"],
    )
    failed = [
        make_failed(
            "vix_term_structure",
            ["3d", "5d", "7d", "10d", "14d"],
        )
    ]

    [result] = similar_failures(opportunity, failed)
    assert result.feature == "vix_term_structure"
    assert result.same_family is True
    assert result.similarity > 0.5, (
        f"expected the VIX-term-structure / CBOE-volatility-state overlap to score "
        f"high, got {result.similarity!r}"
    )

    prior = prior_failure_similarity(opportunity, failed)
    assert prior.basis is EstimateBasis.MEASURED
    assert prior.value is not None and prior.value > 0.5
    assert "vix_term_structure" in prior.note


def test_unrelated_family_and_vocabulary_scores_low() -> None:
    """The second mandatory anti-triviality check: an always-returns-1
    similarity function would fail this assertion. A calendar-seasonality
    failure has no family, token, or horizon overlap with an unrelated
    rates-curve opportunity."""
    opportunity = make_opportunity(
        "ecb_estr_curve_slope",
        information_family=InformationFamily.RATES_CREDIT,
        description="ECB overnight rate curve slope indicator",
        affected_horizons=["21d"],
    )
    failed = [make_failed("seasonality_turn_of_month", ["3d", "5d", "7d", "10d", "14d"])]

    [result] = similar_failures(opportunity, failed)
    assert result.same_family is False
    assert result.similarity == 0.0

    prior = prior_failure_similarity(opportunity, failed)
    assert prior.value == 0.0
    assert prior.basis is EstimateBasis.MEASURED


def test_retest_allowed_reduces_the_penalty() -> None:
    """A documented §9 exception must reduce the resemblance penalty, not
    the raw similarity measurement itself."""
    opportunity = make_opportunity(
        "cboe_volatility_state",
        information_family=InformationFamily.VOLATILITY_SURFACE,
        description="CBOE volatility state indicator",
        affected_horizons=["5d", "10d"],
    )
    dormant = [make_failed("vix_term_structure", ["3d", "5d", "7d", "10d", "14d"])]
    retestable = [
        make_failed(
            "vix_term_structure",
            ["3d", "5d", "7d", "10d", "14d"],
            status="retest_allowed",
            regime_change_note="2026 vol regime shift, new CBOE data source available",
        )
    ]

    sim_dormant = similar_failures(opportunity, dormant)[0]
    sim_retestable = similar_failures(opportunity, retestable)[0]
    # The raw structural similarity does not change...
    assert sim_dormant.similarity == sim_retestable.similarity
    assert sim_retestable.retest_allowed is True
    assert "retest_allowed" in sim_retestable.reason

    # ...but the penalty derived from it must be strictly lower.
    prior_dormant = prior_failure_similarity(opportunity, dormant)
    prior_retestable = prior_failure_similarity(opportunity, retestable)
    assert prior_dormant.value is not None and prior_retestable.value is not None
    assert prior_retestable.value < prior_dormant.value


def test_similar_failures_empty_registry_returns_empty_list() -> None:
    opportunity = make_opportunity("x", information_family=InformationFamily.VOLATILITY_SURFACE)
    assert similar_failures(opportunity, []) == []


def test_prior_failure_similarity_unknown_when_registry_empty() -> None:
    opportunity = make_opportunity("x", information_family=InformationFamily.VOLATILITY_SURFACE)
    result = prior_failure_similarity(opportunity, [])
    assert result.basis is EstimateBasis.UNKNOWN
    assert result.value is None


# --------------------------------------------------------------------------
# family_redundancy
# --------------------------------------------------------------------------


def test_family_redundancy_counts_only_other_non_rejected_same_family() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    others = [
        opportunity,  # self must be excluded
        make_opportunity("b", information_family=InformationFamily.VOLATILITY_SURFACE),
        make_opportunity(
            "c",
            information_family=InformationFamily.VOLATILITY_SURFACE,
            status=ResearchStatus.REJECTED,
        ),
        make_opportunity("d", information_family=InformationFamily.SENTIMENT),
    ]
    result = family_redundancy(opportunity, others)
    assert result.basis is EstimateBasis.MEASURED
    # peers = {b, c, d} (3 total); only b is same-family and non-REJECTED
    assert result.value == pytest.approx(1 / 3)


def test_family_redundancy_empty_others_is_measured_zero_not_unknown() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    result = family_redundancy(opportunity, [])
    assert result.basis is EstimateBasis.MEASURED
    assert result.value == 0.0


# --------------------------------------------------------------------------
# pattern_support
# --------------------------------------------------------------------------


def test_pattern_support_none_decay_yields_no_support() -> None:
    """CRITICAL rule: `decay_since_discovery is None` means unmeasured
    decay, which must contribute NO support -- not be treated as zero
    decay (which would be full, undiscounted strength)."""
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    patterns = [make_pattern(decay_since_discovery=None, stability=0.9)]
    result = pattern_support(opportunity, patterns, now=NOW)
    assert result.basis is EstimateBasis.MEASURED
    assert result.value == 0.0


def test_pattern_support_uses_measured_decay() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    patterns = [make_pattern(stability=0.8, decay_since_discovery=0.25)]
    result = pattern_support(opportunity, patterns, now=NOW)
    assert result.basis is EstimateBasis.MEASURED
    assert result.value == pytest.approx(0.8 * 0.75)


def test_pattern_support_takes_max_not_sum_across_patterns() -> None:
    """Several confirmed patterns in one family must not manufacture more
    confidence than the single best one earned -- this would fail against
    a naive summing implementation."""
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    patterns = [
        make_pattern(stability=0.6, decay_since_discovery=0.0),
        make_pattern(stability=0.5, decay_since_discovery=0.0),
        make_pattern(stability=0.4, decay_since_discovery=0.0),
    ]
    result = pattern_support(opportunity, patterns, now=NOW)
    assert result.value == pytest.approx(0.6)


def test_pattern_support_excludes_patterns_from_the_future() -> None:
    """No look-ahead: a pattern discovered after `now` cannot inform a
    decision made at `now`."""
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    future = NOW.replace(year=NOW.year + 1)
    patterns = [make_pattern(stability=0.9, decay_since_discovery=0.0, discovered_at=future)]
    result = pattern_support(opportunity, patterns, now=NOW)
    assert result.value == 0.0


def test_pattern_support_no_match_in_family_is_measured_zero() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.SENTIMENT)
    patterns = [make_pattern(information_family=InformationFamily.VOLATILITY_SURFACE)]
    result = pattern_support(opportunity, patterns, now=NOW)
    assert result.basis is EstimateBasis.MEASURED
    assert result.value == 0.0


def test_pattern_support_empty_patterns_is_unknown() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    result = pattern_support(opportunity, [], now=NOW)
    assert result.basis is EstimateBasis.UNKNOWN


# --------------------------------------------------------------------------
# measure_current_uncertainty
# --------------------------------------------------------------------------


def test_measure_current_uncertainty_empty_decisions_is_unknown() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    result = measure_current_uncertainty(opportunity, [])
    assert result.basis is EstimateBasis.UNKNOWN


def test_measure_current_uncertainty_maps_family_to_axes_and_restricts() -> None:
    opportunity = make_opportunity(
        "a",
        information_family=InformationFamily.VOLATILITY_SURFACE,
        affected_underlyings=["DAX"],
        affected_horizons=["7d"],
    )
    matching = make_decision(underlying_id="DAX", horizon_days=7, epistemic=0.8, regime=0.6)
    non_matching = make_decision(underlying_id="SPX", horizon_days=3, epistemic=0.1, regime=0.1)
    result = measure_current_uncertainty(opportunity, [matching, non_matching])
    assert result.basis is EstimateBasis.MEASURED
    # Only the matching decision is used: mean(epistemic=0.8, regime=0.6) = 0.7
    assert result.value == pytest.approx(0.7)
    assert "restricted" in result.note


def test_measure_current_uncertainty_falls_back_when_nothing_matches() -> None:
    opportunity = make_opportunity(
        "a",
        information_family=InformationFamily.VOLATILITY_SURFACE,
        affected_underlyings=["DAX"],
        affected_horizons=["7d"],
    )
    only_non_matching = make_decision(
        underlying_id="SPX", horizon_days=3, epistemic=0.2, regime=0.4
    )
    result = measure_current_uncertainty(opportunity, [only_non_matching])
    assert result.basis is EstimateBasis.MEASURED
    assert result.value == pytest.approx(0.3)
    assert "fallback" in result.note


# --------------------------------------------------------------------------
# measure_data_availability
# --------------------------------------------------------------------------


def test_measure_data_availability_known_and_reachable() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    result = measure_data_availability(opportunity, source_available={"cboe": True})
    assert result.basis is EstimateBasis.MEASURED
    assert result.value == 1.0


def test_measure_data_availability_known_and_unreachable() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    result = measure_data_availability(opportunity, source_available={"cboe": False})
    assert result.value == 0.0


def test_measure_data_availability_unknown_when_source_absent() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    result = measure_data_availability(opportunity, source_available={})
    assert result.basis is EstimateBasis.UNKNOWN


# --------------------------------------------------------------------------
# enrich
# --------------------------------------------------------------------------


def test_enrich_does_not_mutate_input_and_returns_measured_estimates() -> None:
    opportunity = make_opportunity(
        "cboe_volatility_state",
        information_family=InformationFamily.VOLATILITY_SURFACE,
        description="CBOE volatility state indicator",
        affected_underlyings=["DAX"],
        affected_horizons=["7d"],
    )
    original_dump = opportunity.model_dump()

    failed = [make_failed("vix_term_structure", ["3d", "5d", "7d", "10d", "14d"])]
    others = [make_opportunity("other", information_family=InformationFamily.VOLATILITY_SURFACE)]
    patterns = [make_pattern(stability=0.7, decay_since_discovery=0.2)]
    decisions = [make_decision(underlying_id="DAX", horizon_days=7, epistemic=0.6, regime=0.4)]

    updated, extra = enrich(
        opportunity,
        failed=failed,
        others=others,
        patterns=patterns,
        decisions=decisions,
        source_available={"cboe": True},
        now=NOW,
    )

    # Input untouched.
    assert opportunity.model_dump() == original_dump

    assert set(extra) == {"prior_failure_similarity", "family_redundancy", "pattern_support"}
    assert updated.current_uncertainty.basis is EstimateBasis.MEASURED
    assert updated.data_availability.basis is EstimateBasis.MEASURED
    assert updated.overlap_with_existing_research.basis is EstimateBasis.MEASURED
    assert extra["prior_failure_similarity"].basis is EstimateBasis.MEASURED
    assert extra["family_redundancy"].value == updated.overlap_with_existing_research.value

    # Everything else on the opportunity is left as-is.
    assert updated.hypothesis_id == opportunity.hypothesis_id
    assert updated.expected_information_gain == opportunity.expected_information_gain


def test_enrich_leaves_genuinely_unmeasurable_fields_unknown() -> None:
    opportunity = make_opportunity("a", information_family=InformationFamily.VOLATILITY_SURFACE)
    updated, extra = enrich(
        opportunity,
        failed=[],
        others=[],
        patterns=[],
        decisions=[],
        source_available={},
        now=NOW,
    )
    assert updated.current_uncertainty.basis is EstimateBasis.UNKNOWN
    assert updated.data_availability.basis is EstimateBasis.UNKNOWN
    assert extra["prior_failure_similarity"].basis is EstimateBasis.UNKNOWN
    assert extra["pattern_support"].basis is EstimateBasis.UNKNOWN
    # No other opportunities is a genuine measurement (zero possible
    # crowding), not an unknown.
    assert updated.overlap_with_existing_research.basis is EstimateBasis.MEASURED
    assert updated.overlap_with_existing_research.value == 0.0


# --------------------------------------------------------------------------
# record_successful_pattern
# --------------------------------------------------------------------------


def test_record_successful_pattern_leaves_decay_and_confirmation_unmeasured() -> None:
    pattern = record_successful_pattern(
        "P42",
        information_family=InformationFamily.VOLATILITY_SURFACE,
        feature="vix_term_structure",
        underlying_id="DAX",
        horizon="5d",
        volatility_regime="normal",
        trend_regime="neutral",
        oos_effect=0.0015,
        effective_sample=12000,
        stability=0.75,
        economic_value=0.002,
        discovered_at=NOW,
        trial_id="TR-1",
        note="measured in W12",
    )
    assert pattern.pattern_id == "P42"
    assert pattern.last_confirmed_at is None
    assert pattern.decay_since_discovery is None
    assert pattern.trial_id == "TR-1"
