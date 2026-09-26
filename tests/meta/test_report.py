"""Tests for the plain-text meta renderers."""

from __future__ import annotations

from datetime import UTC, datetime

from turboedge.meta.report import render_research_queue
from turboedge.meta.research_opportunity import (
    Estimate,
    InformationFamily,
    ResearchOpportunity,
    ResearchPriority,
    ResearchStatus,
    StoredOpportunity,
)


def _opportunity(
    hypothesis_id: str = "RO-ONE",
    *,
    status: ResearchStatus = ResearchStatus.PROPOSED,
    approved_by: str | None = None,
    status_note: str = "",
    current_uncertainty: Estimate | None = None,
) -> ResearchOpportunity:
    return ResearchOpportunity(
        hypothesis_id=hypothesis_id,
        description="A stated hypothesis.",
        information_family=InformationFamily.VOLATILITY_SURFACE,
        expected_information_gain=Estimate.declared(0.6, "judgement"),
        expected_economic_value=Estimate.declared(0.3, "judgement"),
        probability_of_resolving_uncertainty=Estimate.declared(0.7, "judgement"),
        implementation_cost=Estimate.declared(3.0, "engineer-days"),
        implementation_complexity=Estimate.declared(0.4, "judgement"),
        estimated_sample_size=Estimate.measured(520.0),
        current_uncertainty=current_uncertainty or Estimate.unknown("no meta decisions stored yet"),
        data_availability=Estimate.measured(1.0),
        leakage_risk=Estimate.declared(0.1, "judgement"),
        overlap_with_existing_research=Estimate.measured(0.8),
        status=status,
        approved_by=approved_by,
        status_note=status_note,
    )


def _priority(hypothesis_id: str, score: float, **kwargs: object) -> ResearchPriority:
    defaults: dict[str, object] = {
        "hypothesis_id": hypothesis_id,
        "score": score,
        "base_score": score * 2,
        "evidence_completeness": 0.9,
    }
    defaults.update(kwargs)
    return ResearchPriority.model_validate(defaults)


def test_empty_queue_says_so() -> None:
    assert render_research_queue([]) == "RESEARCH QUEUE: empty."


def test_queue_shows_rank_score_and_evidence_completeness() -> None:
    entries = [
        StoredOpportunity(
            opportunity=_opportunity("RO-ONE"),
            priority=_priority("RO-ONE", 0.0421),
            scored_at=datetime(2026, 9, 26, tzinfo=UTC),
        )
    ]
    out = render_research_queue(entries)

    assert "RO-ONE" in out
    assert "0.0421" in out
    assert "90%" in out
    assert "volatility_surface" in out


def test_queue_states_that_priorities_are_not_permission_to_implement() -> None:
    """§8: the system may prioritise, only a human may authorise work.

    The header has to say that, because a ranked list is otherwise easy to
    read as a plan.
    """
    out = render_research_queue([StoredOpportunity(opportunity=_opportunity())])
    assert "human approval" in out


def test_unknown_inputs_are_named_rather_than_hidden() -> None:
    entries = [
        StoredOpportunity(
            opportunity=_opportunity(),
            priority=_priority(
                "RO-ONE",
                0.02,
                unknown_inputs=["current_uncertainty", "data_availability"],
                evidence_completeness=0.8,
            ),
        )
    ]
    out = render_research_queue(entries)

    assert "current_uncertainty" in out
    assert "data_availability" in out


def test_declared_inputs_are_marked_as_judgement() -> None:
    """A score built from judgements must not read as a measurement.

    This is the one line that stops the queue from laundering declared priors
    into apparent evidence, so it is asserted explicitly.
    """
    out = render_research_queue([StoredOpportunity(opportunity=_opportunity())])

    assert "declared (judgement, not measured)" in out
    assert "implementation_cost" in out
    # A measured input must NOT appear on the declared line.
    declared_line = next(line for line in out.splitlines() if "declared (judgement" in line)
    assert "estimated_sample_size" not in declared_line
    assert "data_availability" not in declared_line


def test_unscored_entries_render_without_inventing_a_score() -> None:
    out = render_research_queue([StoredOpportunity(opportunity=_opportunity(), priority=None)])

    assert "not scored yet" in out
    assert "0.00" not in out


def test_limit_truncates_but_reports_the_full_count() -> None:
    entries = [
        StoredOpportunity(
            opportunity=_opportunity(f"RO-{i}"), priority=_priority(f"RO-{i}", 1.0 / (i + 1))
        )
        for i in range(5)
    ]
    out = render_research_queue(entries, limit=2)

    assert "5 open, showing 2" in out
    assert "RO-0" in out
    assert "RO-4" not in out


def test_non_proposed_entries_show_who_moved_them() -> None:
    entries = [
        StoredOpportunity(
            opportunity=_opportunity(
                status=ResearchStatus.MEASURED,
                approved_by="user (research wave 2)",
                status_note="measured, did not clear the ladder",
            ),
            priority=_priority("RO-ONE", 0.01),
        )
    ]
    out = render_research_queue(entries)

    assert "MEASURED" in out
    assert "user (research wave 2)" in out
    assert "did not clear the ladder" in out


def test_rendering_reflects_the_actual_ranking_order() -> None:
    """A renderer that sorted internally, or ignored order, would pass every
    other test here while showing the wrong queue."""
    entries = [
        StoredOpportunity(opportunity=_opportunity("RO-HIGH"), priority=_priority("RO-HIGH", 9.0)),
        StoredOpportunity(opportunity=_opportunity("RO-LOW"), priority=_priority("RO-LOW", 0.1)),
    ]
    out = render_research_queue(entries)
    lines = [line for line in out.splitlines() if line.strip().startswith(("1.", "2."))]

    assert "RO-HIGH" in lines[0]
    assert "RO-LOW" in lines[1]
