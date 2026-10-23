"""End-to-end: the autonomous research cycle against a real Store.

`test_controller.py` covers the judgement in isolation. This covers the
loop as it will actually run: persisted queue entries in, one decision
recorded, and the same decision reached again on a re-run without a second
row appearing.

The twelve properties the brief asks for, and where each is checked:

     1 a hypothesis is selected automatically      ..selects_without_human
     2 the decision is persisted                   ..records_its_decision
     3/4 the decision is evaluated and correct     test_controller.py
     5 a negative result prevents a retest         ..shelves_a_failed_family
     6 unconfirmed support is not promoted         ..never_promotes
     7 the next step is determined automatically   ..moves_on_after_screening
     8 an interrupted run resumes without double
       processing                                  ..is_idempotent_within_a_quarter
     9 the research budget cannot be exceeded      ..never_consumes_the_budget
    10 insufficient data means waiting             ..waits_for_data
    11 state survives separate processes           ..survives_a_store_reopen
    12 the production path is untouched            ..touches_no_production_table

What this does NOT demonstrate, and must not be read as demonstrating: a
completed statistical experiment, or any economic edge. The loop decides
that a screen is admissible and records it. Running the screen uses the
already-approved harnesses, which are separately scheduled, and no result
here says anything about profitability.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.learning import failed_hypotheses
from turboedge.meta.research_opportunity import (
    Estimate,
    InformationFamily,
    ResearchOpportunity,
    ResearchPriority,
    ResearchStatus,
)
from turboedge.research.controller import ActionKind
from turboedge.research.step import (
    DECISION_KIND,
    SCREEN_KIND,
    run_research_step,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ResearchTrial, TrialStatus

_NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
_QUARTER = "2026Q4"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "t.duckdb") as s:
        s.init_schema()
        yield s


@pytest.fixture
def failed_path(tmp_path: Path) -> Path:
    return tmp_path / "failed.json"


def _opportunity(
    hypothesis_id: str,
    *,
    family: InformationFamily = InformationFamily.MACRO,
    sample: float | None = 120.0,
) -> ResearchOpportunity:
    unknown = Estimate.unknown("not measured")
    measured_sample = (
        Estimate.measured(sample, "effective sample from readiness")
        if sample is not None
        else unknown
    )
    return ResearchOpportunity(
        hypothesis_id=hypothesis_id,
        description="does this series carry incremental net EV after costs?",
        information_family=family,
        affected_underlyings=["DAX"],
        affected_horizons=["5"],
        expected_information_gain=unknown,
        expected_economic_value=unknown,
        probability_of_resolving_uncertainty=Estimate.declared(0.5, "first screen"),
        implementation_cost=unknown,
        implementation_complexity=unknown,
        estimated_sample_size=measured_sample,
        current_uncertainty=unknown,
        data_availability=Estimate.measured(1.0, "passed every readiness gate"),
        leakage_risk=unknown,
        overlap_with_existing_research=unknown,
        status=ResearchStatus.PROPOSED,
    )


def _seed(store: Store, hypothesis_id: str, *, priority: float, **kw: object) -> None:
    store.upsert_research_opportunity(
        _opportunity(hypothesis_id, **kw),  # type: ignore[arg-type]
        priority=ResearchPriority(hypothesis_id=hypothesis_id, score=priority, base_score=priority),
        scored_at=_NOW,
    )


def _step(store: Store, failed_path: Path, *, now: datetime = _NOW, **kw: object):
    return run_research_step(store, failed_hypotheses_path=failed_path, now=now, **kw)  # type: ignore[arg-type]


def test_it_selects_a_hypothesis_without_a_human(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.3)
    _seed(store, "RO-B", priority=0.9)

    action = _step(store, failed_path)

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.hypothesis_id == "RO-B"


def test_it_records_its_decision(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.5)

    action = _step(store, failed_path)
    trials = store.list_research_trials(quarter=_QUARTER)

    assert len(trials) == 1
    assert trials[0].kind == DECISION_KIND
    assert action.hypothesis_id in trials[0].description
    assert str(action.kind) in trials[0].description


def test_it_is_idempotent_within_a_quarter(store: Store, failed_path: Path) -> None:
    # The loop runs on a schedule, and schedules retry. A re-run that
    # reaches the same conclusion must not add a second row -- an inflated
    # research ledger would misreport how much testing happened, which is
    # the number the multiple-testing correction depends on.
    _seed(store, "RO-A", priority=0.5)

    first = _step(store, failed_path)
    second = _step(store, failed_path, now=_NOW + timedelta(days=3))

    assert first.kind is second.kind
    assert first.hypothesis_id == second.hypothesis_id
    assert len(store.list_research_trials(quarter=_QUARTER)) == 1


def test_a_dry_run_reaches_the_same_conclusion_and_writes_nothing(
    store: Store, failed_path: Path
) -> None:
    _seed(store, "RO-A", priority=0.5)

    dry = _step(store, failed_path, dry_run=True)

    assert store.list_research_trials(quarter=_QUARTER) == []
    wet = _step(store, failed_path)
    assert dry.kind is wet.kind and dry.hypothesis_id == wet.hypothesis_id


def test_it_waits_when_the_sample_is_unmeasured(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.9, sample=None)

    action = _step(store, failed_path)

    assert action.kind is ActionKind.AWAIT_DATA


def test_it_waits_when_the_sample_is_too_small(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.9, sample=10.0)

    action = _step(store, failed_path)

    assert action.kind is ActionKind.AWAIT_DATA
    assert action.evidence["effective_sample"] == 10.0


def test_a_failed_family_is_shelved_rather_than_retested(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.9)
    for i in range(3):
        failed_hypotheses.append(
            failed_path,
            failed_hypotheses.new_hypothesis(
                feature=f"de_rail_freight_{i}",
                horizons=["5"],
                incremental_net_ev=-0.01,
                effective_sample=60,
                now=_NOW - timedelta(days=30),
            ),
        )

    action = _step(store, failed_path)

    assert action.kind is ActionKind.SHELVE_FAMILY
    assert action.family == str(InformationFamily.MACRO)


def test_it_never_consumes_the_confirmatory_budget(store: Store, failed_path: Path) -> None:
    # Decisions and screens are recorded in the same table as confirmatory
    # trials. If they counted, the loop would spend the quarter's six
    # primary hypotheses merely by thinking.
    _seed(store, "RO-A", priority=0.5)
    _step(store, failed_path)

    confirmatory = [
        t
        for t in store.list_research_trials(quarter=_QUARTER)
        if t.kind not in {SCREEN_KIND, DECISION_KIND}
    ]

    assert confirmatory == []


def test_the_budget_cannot_be_exceeded(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.9)
    for i in range(6):
        store.insert_research_trial(
            ResearchTrial(
                trial_id=f"{_QUARTER}_{i:03d}",
                kind="feature",
                description="a real confirmatory trial",
                created_at=_NOW,
                quarter=_QUARTER,
                status=TrialStatus.EXPERIMENTAL,
            )
        )

    action = _step(store, failed_path)

    # A full confirmatory budget must not stop exploratory screening, and
    # must not be spendable by it either.
    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.evidence["confirmatory_trials_this_quarter"] == 6


def test_it_never_promotes(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.9)

    _step(store, failed_path)

    assert store.list_alpha_sources() == []


def test_it_moves_on_once_everything_is_screened(store: Store, failed_path: Path) -> None:
    _seed(store, "RO-A", priority=0.9)
    store.insert_research_trial(
        ResearchTrial(
            trial_id=f"{_QUARTER}_SCREEN_A",
            kind=SCREEN_KIND,
            description="RO-A|screened, no support",
            created_at=_NOW,
            quarter=_QUARTER,
            status=TrialStatus.EXPERIMENTAL,
        )
    )

    action = _step(store, failed_path)

    assert action.kind is ActionKind.NO_ACTION
    assert "already been screened" in action.reason


def test_the_decision_survives_a_store_reopen(tmp_path: Path) -> None:
    db = tmp_path / "t.duckdb"
    failed = tmp_path / "failed.json"
    with Store(db) as store:
        store.init_schema()
        _seed(store, "RO-A", priority=0.5)
        first = run_research_step(store, failed_hypotheses_path=failed, now=_NOW)

    # A separate process, as the scheduled job will be.
    with Store(db) as store:
        again = run_research_step(
            store, failed_hypotheses_path=failed, now=_NOW + timedelta(days=7)
        )
        trials = store.list_research_trials(quarter=_QUARTER)

    assert first.hypothesis_id == again.hypothesis_id
    assert len(trials) == 1


def test_it_touches_no_production_table(store: Store, failed_path: Path) -> None:
    # The scan/trading path must be provably unaffected.
    _seed(store, "RO-A", priority=0.5)
    _step(store, failed_path)

    for table in (
        "signals",
        "candidate_sets",
        "forward_ledger",
        "positions_manual",
        "product_snapshots",
        "model_registry",
        "alpha_sources",
    ):
        count = store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert count is not None and count[0] == 0, table


def test_an_empty_queue_decides_nothing_rather_than_failing(
    store: Store, failed_path: Path
) -> None:
    action = _step(store, failed_path)

    assert action.kind is ActionKind.NO_ACTION
