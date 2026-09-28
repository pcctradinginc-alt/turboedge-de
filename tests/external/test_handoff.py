"""Tests for the readiness -> ResearchQueue hand-off.

The boundary being protected: a dataset becoming ready may create a
*question*, never an approval, a trial or an expectation of alpha.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from turboedge.external.readiness import DataReadinessRecord, ReadinessState
from turboedge.external.schemas import AvailabilityPrecision
from turboedge.external.triggers import TriggerType, triggers_for_transition
from turboedge.meta.research_opportunity import (
    SYSTEM_ASSIGNABLE_STATES,
    EstimateBasis,
    InformationFamily,
    ResearchStatus,
)
from turboedge.meta.research_queue import ResearchQueue
from turboedge.storage.duckdb import Store

_NOW = datetime(2026, 9, 28, 6, 0, tzinfo=UTC)


@pytest.fixture
def queue(tmp_path: Path) -> Iterator[ResearchQueue]:
    with Store(tmp_path / "t.duckdb") as store:
        store.init_schema()
        yield ResearchQueue(store, failed_hypotheses_path=tmp_path / "failed.json")


def _triggers(state: ReadinessState = ReadinessState.EXPLORATORY_READY):
    record = DataReadinessRecord(
        source="ecb",
        series_id="ECB.M3",
        state=state,
        evaluated_at=_NOW,
        nominal_n=120,
        effective_n=120.0,
        independent_dates=120,
        calendar_span_days=3650,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        completeness=1.0,
        policy_version="1",
    )
    return triggers_for_transition(record)


def test_a_ready_dataset_creates_one_proposed_question(queue: ResearchQueue) -> None:
    inserted = queue.seed_from_data_readiness(_triggers())

    assert len(inserted) == 1
    hypothesis_id, _ = inserted[0]
    assert hypothesis_id == "RO-DATA-ECB-ECB-M3"

    stored = {
        s.opportunity.hypothesis_id: s.opportunity
        for s in queue._store.list_research_opportunities()
    }
    assert stored[hypothesis_id].status is ResearchStatus.PROPOSED
    assert stored[hypothesis_id].status in {ResearchStatus.PROPOSED}


def test_the_created_status_is_system_assignable(queue: ResearchQueue) -> None:
    queue.seed_from_data_readiness(_triggers())

    (stored,) = queue._store.list_research_opportunities()
    assert stored.opportunity.status in SYSTEM_ASSIGNABLE_STATES


def test_re_running_inserts_nothing_new(queue: ResearchQueue) -> None:
    queue.seed_from_data_readiness(_triggers())
    again = queue.seed_from_data_readiness(_triggers())

    assert again == []
    assert len(queue._store.list_research_opportunities()) == 1


def test_a_later_milestone_creates_no_second_question(queue: ResearchQueue) -> None:
    # Validation and confirmation readiness change what an existing question
    # may be tested on. They are partition decisions, not new hypotheses.
    confirmation = [
        t
        for t in _triggers(ReadinessState.CONFIRMATION_READY)
        if t.trigger_type is not TriggerType.EXPLORATORY_DATA_READY
    ]

    assert queue.seed_from_data_readiness(confirmation) == []


def test_only_sample_size_is_claimed_as_measured(queue: ResearchQueue) -> None:
    # Readiness measures evidence volume. It measures nothing about whether
    # the data predict anything, and the estimates must say so.
    queue.seed_from_data_readiness(_triggers())
    (stored,) = queue._store.list_research_opportunities()
    opportunity = stored.opportunity

    assert opportunity.estimated_sample_size.basis is EstimateBasis.MEASURED
    assert opportunity.estimated_sample_size.value == 120.0
    assert opportunity.expected_information_gain.basis is EstimateBasis.UNKNOWN
    assert opportunity.expected_economic_value.basis is EstimateBasis.UNKNOWN


def test_the_id_is_stable_across_policy_versions(queue: ResearchQueue) -> None:
    # Re-evaluating under a new policy must not create a second copy of a
    # question a human may already have approved or rejected.
    first = queue.seed_from_data_readiness(_triggers())
    record = DataReadinessRecord(
        source="ecb",
        series_id="ECB.M3",
        state=ReadinessState.EXPLORATORY_READY,
        evaluated_at=_NOW,
        nominal_n=120,
        effective_n=120.0,
        independent_dates=120,
        calendar_span_days=3650,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        completeness=1.0,
        policy_version="2",
    )
    second = queue.seed_from_data_readiness(triggers_for_transition(record))

    assert first and second == []


def test_the_question_is_filed_under_macro(queue: ResearchQueue) -> None:
    queue.seed_from_data_readiness(_triggers())

    (stored,) = queue._store.list_research_opportunities()
    assert stored.opportunity.information_family is InformationFamily.MACRO
