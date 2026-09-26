"""Tests for the Phase 2 research queue (§8) -- the governance boundary.

Nearly every test here is really testing one thing: that no method on
`ResearchQueue` can move a research opportunity out of `PROPOSED`, or alter
its already-recorded approval, without a named human and a stated reason on
the record. `catalog`, `research_memory` and `value_of_information` are used
for real (no stubs) -- a real `Store(":memory:")` is the only test double.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from turboedge.meta import catalog
from turboedge.meta.research_opportunity import (
    SYSTEM_ASSIGNABLE_STATES,
    Estimate,
    InformationFamily,
    ResearchOpportunity,
    ResearchStatus,
)
from turboedge.meta.research_queue import (
    ALLOWED_TRANSITIONS,
    IllegalTransition,
    ResearchQueue,
)
from turboedge.storage.duckdb import Store

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
_TERMINAL = frozenset({ResearchStatus.PROMOTED, ResearchStatus.REJECTED})


def _est(v: float) -> Estimate:
    return Estimate.measured(v)


def make_opportunity(hypothesis_id: str, **overrides: object) -> ResearchOpportunity:
    """A minimal, valid opportunity for tests that do not care about its
    ranking inputs -- only about lifecycle mechanics."""
    fields: dict[str, object] = dict(
        hypothesis_id=hypothesis_id,
        description=f"test opportunity {hypothesis_id}",
        information_family=InformationFamily.POSITIONING,
        expected_information_gain=_est(0.5),
        expected_economic_value=_est(0.5),
        probability_of_resolving_uncertainty=_est(0.5),
        implementation_cost=_est(0.5),
        implementation_complexity=_est(0.5),
        estimated_sample_size=_est(400.0),
        current_uncertainty=_est(0.5),
        data_availability=_est(0.8),
        leakage_risk=_est(0.1),
        overlap_with_existing_research=_est(0.1),
    )
    fields.update(overrides)
    return ResearchOpportunity(**fields)


def put(store: Store, hypothesis_id: str, status: ResearchStatus, **overrides: object) -> None:
    """Insert an opportunity directly at `status`, bypassing the queue's own
    governed lifecycle -- used only to set up test fixtures at an arbitrary
    starting state, never to test the governance itself."""
    fields: dict[str, object] = {"status": status}
    if status is not ResearchStatus.PROPOSED:
        fields["approved_by"] = overrides.pop("approved_by", "seed")
        fields["approved_at"] = overrides.pop("approved_at", NOW)
    fields.update(overrides)
    store.upsert_research_opportunity(make_opportunity(hypothesis_id, **fields))


@pytest.fixture
def store() -> Iterator[Store]:
    s = Store(":memory:")
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def failed_path(tmp_path: Path) -> Path:
    return tmp_path / "registry" / "failed_hypotheses.json"


@pytest.fixture
def queue(store: Store, failed_path: Path) -> ResearchQueue:
    return ResearchQueue(store, failed_hypotheses_path=failed_path)


# -- seeding ---------------------------------------------------------------


def test_seed_from_catalog_inserts_all_proposed(queue: ResearchQueue, store: Store) -> None:
    inserted = queue.seed_from_catalog(now=NOW)
    assert len(inserted) == 14
    assert len(set(inserted)) == 14  # no duplicate ids in the catalog itself
    stored = store.list_research_opportunities()
    assert len(stored) == 14
    assert all(s.opportunity.status == ResearchStatus.PROPOSED for s in stored)
    assert all(s.priority is None for s in stored)  # not scored yet


def test_seed_from_catalog_is_idempotent_and_preserves_human_edits(
    queue: ResearchQueue, store: Store
) -> None:
    first = queue.seed_from_catalog(now=NOW)
    target = first[0]
    queue.approve(target, approved_by="alice", note="worth a look", now=NOW)

    second = queue.seed_from_catalog(now=NOW + timedelta(days=1))
    assert second == []  # every id already present -- nothing new to insert

    still = store.get_research_opportunity(target)
    assert still is not None
    assert still.opportunity.status == ResearchStatus.APPROVED
    assert still.opportunity.approved_by == "alice"
    assert "worth a look" in still.opportunity.status_note


# -- rescore -----------------------------------------------------------


def test_rescore_never_changes_lifecycle_fields(queue: ResearchQueue, store: Store) -> None:
    ids = queue.seed_from_catalog(now=NOW)
    approved_id = ids[0]
    queue.approve(approved_id, approved_by="bob", note="go ahead", trial_id="TR-1", now=NOW)

    def snapshot() -> dict[str, tuple[object, ...]]:
        return {
            s.opportunity.hypothesis_id: (
                s.opportunity.status,
                s.opportunity.approved_by,
                s.opportunity.approved_at,
                s.opportunity.trial_id,
                s.opportunity.status_note,
            )
            for s in store.list_research_opportunities()
        }

    before = snapshot()
    queue.rescore(now=NOW + timedelta(hours=1))
    after = snapshot()
    assert before == after


def test_rescore_persists_enriched_estimates_and_stamps_scored_at(
    queue: ResearchQueue, store: Store
) -> None:
    queue.seed_from_catalog(now=NOW)
    results = queue.rescore(now=NOW)
    assert len(results) == 14
    assert all(r.priority is not None for r in results)
    assert all(r.scored_at == NOW for r in results)
    for r in results:
        stored = store.get_research_opportunity(r.opportunity.hypothesis_id)
        assert stored is not None
        assert stored.priority is not None
        assert stored.scored_at == NOW


# -- approve -----------------------------------------------------------


def test_approve_rejects_blank_approver_and_blank_note(queue: ResearchQueue) -> None:
    ids = queue.seed_from_catalog(now=NOW)
    with pytest.raises(ValueError, match="approved_by"):
        queue.approve(ids[0], approved_by="   ", note="fine", now=NOW)
    with pytest.raises(ValueError, match="note"):
        queue.approve(ids[0], approved_by="carol", note="   ", now=NOW)


def test_approve_refuses_an_already_approved_entry(queue: ResearchQueue, store: Store) -> None:
    put(store, "H-APPROVED", ResearchStatus.APPROVED, trial_id=None)
    with pytest.raises(IllegalTransition):
        queue.approve("H-APPROVED", approved_by="dave", note="again?", now=NOW)


# -- transition: illegal moves -------------------------------------------


_ALL_STATUSES = list(ResearchStatus)
_ILLEGAL_PAIRS = [
    (frm, to)
    for frm in _ALL_STATUSES
    for to in _ALL_STATUSES
    if to not in ALLOWED_TRANSITIONS.get(frm, frozenset())
]


@pytest.mark.parametrize(
    ("frm", "to"), _ILLEGAL_PAIRS, ids=[f"{f}->{t}" for f, t in _ILLEGAL_PAIRS]
)
def test_every_illegal_transition_raises(
    queue: ResearchQueue, store: Store, frm: ResearchStatus, to: ResearchStatus
) -> None:
    trial_id = "TR-SETUP" if frm is not ResearchStatus.PROPOSED else None
    put(store, "H-ILLEGAL", frm, trial_id=trial_id)
    with pytest.raises(IllegalTransition):
        queue.transition("H-ILLEGAL", to, actor="erin", note="attempting", now=NOW)


def test_transition_refuses_to_leave_proposed_even_via_its_own_allowed_edge(
    queue: ResearchQueue, store: Store
) -> None:
    """PROPOSED -> APPROVED *is* in ALLOWED_TRANSITIONS (it is a real edge in
    the lifecycle graph, used by `apply_measured_outcomes`'s path-finder),
    but `transition()` must still refuse to walk it -- `approve()` is the
    only function allowed to move an entry out of PROPOSED (§8). This is the
    transition this module deliberately forbids beyond what the table alone
    would allow."""
    put(store, "H-PROPOSED", ResearchStatus.PROPOSED)
    assert ResearchStatus.APPROVED in ALLOWED_TRANSITIONS[ResearchStatus.PROPOSED]
    with pytest.raises(IllegalTransition, match="approve"):
        queue.transition(
            "H-PROPOSED", ResearchStatus.APPROVED, actor="frank", note="skip?", now=NOW
        )


def test_running_without_trial_id_raises(queue: ResearchQueue, store: Store) -> None:
    put(store, "H-NOTRIAL", ResearchStatus.APPROVED, trial_id=None)
    with pytest.raises(ValueError, match="trial_id"):
        queue.transition("H-NOTRIAL", ResearchStatus.RUNNING, actor="erin", note="start", now=NOW)


def test_running_with_trial_id_succeeds(queue: ResearchQueue, store: Store) -> None:
    put(store, "H-TRIAL", ResearchStatus.APPROVED, trial_id="TR-9")
    result = queue.transition(
        "H-TRIAL", ResearchStatus.RUNNING, actor="erin", note="start", now=NOW
    )
    assert result.opportunity.status == ResearchStatus.RUNNING
    assert result.opportunity.trial_id == "TR-9"
    assert "start" in result.opportunity.status_note


def test_transition_requires_actor_and_note(queue: ResearchQueue, store: Store) -> None:
    put(store, "H-BLANK", ResearchStatus.APPROVED, trial_id="TR-1")
    with pytest.raises(ValueError, match="actor"):
        queue.transition("H-BLANK", ResearchStatus.RUNNING, actor=" ", note="ok", now=NOW)
    with pytest.raises(ValueError, match="note"):
        queue.transition("H-BLANK", ResearchStatus.RUNNING, actor="grace", note=" ", now=NOW)


def test_transition_never_overwrites_approved_by(queue: ResearchQueue, store: Store) -> None:
    put(store, "H-KEEP", ResearchStatus.APPROVED, approved_by="alice", trial_id="TR-1")
    moved = queue.transition(
        "H-KEEP", ResearchStatus.RUNNING, actor="bob", note="running now", now=NOW
    )
    assert moved.opportunity.approved_by == "alice"  # who authorised it, not who ran this hop


# -- apply_measured_outcomes ---------------------------------------------


def test_apply_measured_outcomes_lands_w12_entries_and_drops_terminal_from_ranked(
    queue: ResearchQueue, store: Store
) -> None:
    queue.seed_from_catalog(now=NOW)
    changed = queue.apply_measured_outcomes(now=NOW)
    assert set(changed) == set(catalog.W12_MEASURED_OUTCOMES.keys())

    ranked_ids = {s.opportunity.hypothesis_id for s in queue.ranked()}
    for hid, outcome in catalog.W12_MEASURED_OUTCOMES.items():
        stored = store.get_research_opportunity(hid)
        assert stored is not None
        assert stored.opportunity.status == outcome.status
        # governance invariant: never left PROPOSED without an approver
        assert stored.opportunity.approved_by
        if outcome.status in _TERMINAL:
            assert hid not in ranked_ids
        else:
            assert hid in ranked_ids

    # idempotent: re-applying moves nothing further
    changed_again = queue.apply_measured_outcomes(now=NOW + timedelta(days=1))
    assert changed_again == []


# -- ranked --------------------------------------------------------------


def test_ranked_unscored_sorts_last_and_limit_truncates(queue: ResearchQueue, store: Store) -> None:
    queue.seed_from_catalog(now=NOW)
    queue.rescore(now=NOW)
    baseline = queue.ranked()
    assert len(baseline) >= 2

    # Blank one entry's score directly to prove the "no score sorts last"
    # rule holds independent of whatever real scores the pipeline produces.
    unscored = baseline[0].opportunity
    store.upsert_research_opportunity(unscored, priority=None, scored_at=None)

    reranked = queue.ranked()
    assert reranked[-1].opportunity.hypothesis_id == unscored.hypothesis_id
    assert reranked[-1].priority is None

    limited = queue.ranked(limit=2)
    assert limited == reranked[:2]


def test_ranked_excludes_terminal_statuses(queue: ResearchQueue, store: Store) -> None:
    put(store, "H-PROMOTED", ResearchStatus.PROMOTED)
    put(store, "H-REJECTED", ResearchStatus.REJECTED)
    put(store, "H-DORMANT", ResearchStatus.DORMANT)
    ids = {s.opportunity.hypothesis_id for s in queue.ranked()}
    assert "H-PROMOTED" not in ids
    assert "H-REJECTED" not in ids
    assert "H-DORMANT" in ids  # dormant is parked, not finished -- still a candidate


# -- anti-triviality: rescore must actually discriminate ------------------


def test_rescore_produces_differentiated_non_alphabetical_ranking(
    queue: ResearchQueue, store: Store
) -> None:
    """A `rescore` that scored everything equally, or that merely echoed the
    store's alphabetical tie-break, would pass every other test in this file.
    This is the one that would catch it."""
    queue.seed_from_catalog(now=NOW)
    queue.apply_measured_outcomes(now=NOW)
    rescored = queue.rescore(now=NOW)

    active = [
        r for r in rescored if r.priority is not None and r.opportunity.status not in _TERMINAL
    ]
    assert len(active) >= 2
    scores = {r.priority.score for r in active if r.priority is not None}
    assert len(scores) >= 2, "rescore must not assign every active entry the same score"

    ranked_ids = [s.opportunity.hypothesis_id for s in queue.ranked()]
    alpha_ids = sorted(ranked_ids)
    assert ranked_ids != alpha_ids, "ranking must not equal a bare alphabetical sort"

    top = queue.ranked()[0]
    max_score = max(r.priority.score for r in active if r.priority is not None)
    assert top.priority is not None
    assert top.priority.score == pytest.approx(max_score)


def test_no_method_produces_non_system_assignable_status_without_approver(
    queue: ResearchQueue, store: Store
) -> None:
    queue.seed_from_catalog(now=NOW)
    queue.apply_measured_outcomes(now=NOW)
    queue.rescore(now=NOW)
    for stored in store.list_research_opportunities():
        opp = stored.opportunity
        if opp.status not in SYSTEM_ASSIGNABLE_STATES:
            assert opp.approved_by, f"{opp.hypothesis_id} is {opp.status} with no approved_by"

    # And the safety net this module relies on: the schema itself refuses to
    # construct such an object, independent of any check this module makes.
    base = make_opportunity("H-INVALID").model_dump()
    base.update(status=ResearchStatus.APPROVED, approved_by=None)
    with pytest.raises(ValidationError):
        ResearchOpportunity.model_validate(base)
