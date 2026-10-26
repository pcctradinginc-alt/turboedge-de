"""End-to-end: the autonomous research cycle against a real Store.

`test_controller.py` covers the judgement in isolation. This covers the
loop as it will actually run: persisted queue entries in, a question
selected, an approved harness executed, its result and cost written down,
the ranking inputs it measured fed back, and the next question chosen from
the updated state.

Where each property the brief asks for is checked:

     1 a hypothesis is selected automatically   ..selects_a_hypothesis_without_a_human
     2 a screen is actually executed            ..executes_the_approved_harness
     3 spec/data version/cost are persisted     ..records_the_specification_and_the_cost
     4 errors are recorded, not swallowed       ..records_a_harness_failure
     5 no approved harness is handled           ..records_no_approved_harness_and_moves_on
     6 negative results change prioritisation   ..a_negative_screen_lowers_the_ranking_input
     7 positive results escalate, not promote   ..support_escalates_to_a_human
     8 the next question follows automatically  ..walks_past_unanswerable_questions
     9 a restart does not double-process        ..a_restart_does_not_re_run_the_screen
    10 insufficient data means waiting          ..waits_when_the_sample_is_too_small
    11 a shelved family stays shelved           ..a_shelved_family_is_not_re_decided
    12 the screen cap counts executions         ..the_cap_counts_executions_not_decisions
    13 the confirmatory budget is untouchable   ..never_consumes_the_confirmatory_budget
    14 state survives separate processes        ..the_screen_survives_a_store_reopen
    15 point-in-time reads are not bypassed     ..the_harness_registry_is_read_only
    16 the production path is untouched         ..touches_no_production_table

What this does NOT demonstrate, and must not be read as demonstrating: an
economic edge, or a confirmed hypothesis. Every screen here runs on
controlled or stubbed data; a CANDIDATE verdict means "worth a confirmatory
trial", and that trial still needs a named human.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from turboedge.learning import failed_hypotheses
from turboedge.meta.research_opportunity import (
    Estimate,
    EstimateBasis,
    InformationFamily,
    ResearchOpportunity,
    ResearchPriority,
    ResearchStatus,
)
from turboedge.research import harness as harness_module
from turboedge.research import step as step_module
from turboedge.research.controller import ActionKind
from turboedge.research.harness import ApprovedHarness, HarnessOutcome, resolve_harness
from turboedge.research.step import (
    DECISION_KIND,
    SCREEN_KIND,
    run_research_step,
    screen_id_for,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ResearchTrial, ScreenStatus, TrialStatus, UnderlyingBar

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
    information_gain: float | None = 0.7,
) -> ResearchOpportunity:
    unknown = Estimate.unknown("not measured")
    measured_sample = (
        Estimate.measured(sample, "effective sample from readiness")
        if sample is not None
        else unknown
    )
    gain = (
        Estimate.declared(information_gain, "catalogue judgement")
        if information_gain is not None
        else unknown
    )
    return ResearchOpportunity(
        hypothesis_id=hypothesis_id,
        description="does this series carry incremental net EV after costs?",
        information_family=family,
        affected_underlyings=["DAX"],
        affected_horizons=["5"],
        expected_information_gain=gain,
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


def _step(store: Store, failed_path: Path, *, now: datetime = _NOW, **kw: object) -> Any:
    return run_research_step(store, failed_hypotheses_path=failed_path, now=now, **kw)  # type: ignore[arg-type]


class _StubHarness:
    """A reviewed-procedure stand-in, so the loop's wiring is testable.

    The real registry holds one harness whose dataset needs a matured,
    labelled forward ledger. Driving the loop through a stub is how the
    wiring -- claim, execute, record, feed back -- gets covered on
    controlled data. The stub is a test fixture, not a generated
    procedure: the loop still only ever calls what the registry hands it.
    """

    def __init__(self, *, verdict: str = "NO_IMPROVEMENT", supported: bool = False) -> None:
        self.verdict = verdict
        self.supported = supported
        self.calls: list[str] = []
        self.as_of: list[datetime] = []
        self.raises: Exception | None = None

    def __call__(self, store: Store, hypothesis_id: str, now: datetime) -> HarnessOutcome:
        self.calls.append(hypothesis_id)
        self.as_of.append(now)
        if self.raises is not None:
            raise self.raises
        return HarnessOutcome(
            verdict=self.verdict,
            metrics={"effective_sample": 96.0, "crps_null": 0.4, "crps_conditional": 0.41},
            data_version="deadbeefcafe0001",
            supported=self.supported,
            spec={"horizon": 3, "embargo": 3, "min_train": 200, "step": 100},
        )


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubHarness:
    runner = _StubHarness()
    registered = ApprovedHarness(
        name="stub",
        version="1",
        hypothesis_ids=frozenset({"RO-A"}),
        description="test double",
        spec={"horizon": 3},
        run=runner,
    )
    monkeypatch.setattr(
        step_module, "resolve_harness", lambda hid: registered if hid == "RO-A" else None
    )
    return runner


# -- selection and execution ---------------------------------------------


def test_it_selects_a_hypothesis_without_a_human(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9)
    _seed(store, "RO-B", priority=0.3)

    action = _step(store, failed_path)

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.hypothesis_id == "RO-A"


def test_it_executes_the_approved_harness(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # The gap this closes: before, the loop recorded a decision and nothing
    # ever ran, so the screen cap never depleted and the same question was
    # re-decided forever.
    _seed(store, "RO-A", priority=0.9)

    _step(store, failed_path)

    assert stub.calls == ["RO-A"]
    screens = store.list_research_screens(quarter=_QUARTER)
    assert len(screens) == 1
    assert screens[0].status is ScreenStatus.COMPLETED
    assert screens[0].verdict == "NO_IMPROVEMENT"
    assert screens[0].harness == "stub"
    assert screens[0].harness_version == "1"


def test_it_records_the_specification_the_data_version_and_the_cost(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9)

    _step(store, failed_path)
    screen = store.list_research_screens(quarter=_QUARTER)[0]

    assert screen.spec["min_train"] == 200
    assert screen.data_version == "deadbeefcafe0001"
    assert screen.cost_seconds >= 0.0
    assert screen.finished_at is not None
    assert screen.metrics["effective_sample"] == 96.0


def test_it_records_a_harness_failure_rather_than_swallowing_it(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # A harness that raises because nothing has matured yet is the expected
    # early state. The loop must record it, not crash and not pretend a
    # result exists.
    _seed(store, "RO-A", priority=0.9)
    stub.raises = ValueError("produced zero usable rows")

    action = _step(store, failed_path)

    assert action.kind is ActionKind.START_EXPLORATORY
    screen = store.list_research_screens(quarter=_QUARTER)[0]
    assert screen.status is ScreenStatus.FAILED
    assert "produced zero usable rows" in (screen.error or "")
    assert screen.verdict is None


def test_a_failed_screen_is_not_support(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # Reading one's own failure to measure as a reason to escalate is the
    # single most dangerous thing this loop could do.
    _seed(store, "RO-A", priority=0.9)
    stub.raises = ValueError("no data")

    _step(store, failed_path)

    assert not store.list_research_screens(quarter=_QUARTER)[0].produced_support


# -- no approved harness --------------------------------------------------


def test_it_records_no_approved_harness_and_moves_on(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # Twelve of the sixteen catalogue questions have no harness. The loop
    # must say so and continue, not invent a procedure and not stop.
    _seed(store, "RO-UNANSWERABLE", priority=0.9)

    action = _step(store, failed_path)

    screens = store.list_research_screens(quarter=_QUARTER)
    assert [s.status for s in screens] == [ScreenStatus.NO_APPROVED_HARNESS]
    assert "research/harness.py" in (screens[0].error or "")
    assert action.kind is ActionKind.NO_ACTION
    assert "approved harness" in action.reason


def test_an_absent_harness_costs_no_screen_from_the_quarter(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # Nothing ran, so nothing may be charged. Otherwise a catalogue of
    # un-harnessed questions would exhaust the quarter without producing a
    # single measurement.
    _seed(store, "RO-UNANSWERABLE", priority=0.9)

    action = _step(store, failed_path)

    assert action.evidence["exploratory_screens_this_quarter"] == 0


def test_it_walks_past_unanswerable_questions_to_one_it_can_answer(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # The automatic selection of the next question, in one run. A weekly
    # schedule that skipped one entry per run would take three months to
    # reach the question it can actually screen.
    _seed(store, "RO-NO-HARNESS-1", priority=0.99)
    _seed(store, "RO-NO-HARNESS-2", priority=0.98)
    _seed(store, "RO-A", priority=0.5)

    action = _step(store, failed_path)

    assert stub.calls == ["RO-A"]
    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.hypothesis_id == "RO-A"
    statuses = {s.hypothesis_id: s.status for s in store.list_research_screens(quarter=_QUARTER)}
    assert statuses["RO-NO-HARNESS-1"] is ScreenStatus.NO_APPROVED_HARNESS
    assert statuses["RO-NO-HARNESS-2"] is ScreenStatus.NO_APPROVED_HARNESS
    assert statuses["RO-A"] is ScreenStatus.COMPLETED


def test_the_decision_walk_is_bounded(store: Store, failed_path: Path, stub: _StubHarness) -> None:
    # An unbounded re-decide loop on surprising state is how a scheduled
    # job becomes a spin.
    for i in range(20):
        _seed(store, f"RO-NONE-{i:02d}", priority=0.5 + i / 100)

    action = _step(store, failed_path)

    recorded = store.list_research_screens(quarter=_QUARTER)
    assert len(recorded) <= step_module.MAX_DECISIONS_PER_RUN
    assert action is not None


# -- feedback into prioritisation ----------------------------------------


def test_a_negative_screen_lowers_the_ranking_input_it_measured(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # Without this the loop re-ranks the same exhausted question to the top
    # forever: the declared optimism in the catalogue is never contradicted
    # by what was actually measured.
    _seed(store, "RO-A", priority=0.9, information_gain=0.7)

    _step(store, failed_path)

    stored = {
        i.opportunity.hypothesis_id: i.opportunity for i in store.list_research_opportunities()
    }
    gain = stored["RO-A"].expected_information_gain
    assert gain.basis is EstimateBasis.MEASURED
    assert gain.value == 0.0
    assert "NO_IMPROVEMENT" in gain.note


def test_a_screen_measures_the_sample_size(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9)

    _step(store, failed_path)

    stored = {
        i.opportunity.hypothesis_id: i.opportunity for i in store.list_research_opportunities()
    }
    sample = stored["RO-A"].estimated_sample_size
    assert sample.basis is EstimateBasis.MEASURED
    assert sample.value == 96.0
    assert "not rows" in sample.note


def test_a_positive_screen_does_not_raise_its_own_ranking(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # A loop that promoted its own question up the ranking on its own
    # screen evidence would be marking its own work.
    _seed(store, "RO-A", priority=0.9, information_gain=0.7)
    stub.verdict = "CANDIDATE"
    stub.supported = True

    _step(store, failed_path)

    stored = {
        i.opportunity.hypothesis_id: i.opportunity for i in store.list_research_opportunities()
    }
    gain = stored["RO-A"].expected_information_gain
    assert gain.basis is EstimateBasis.DECLARED
    assert gain.value == 0.7


def test_support_escalates_to_a_human_and_promotes_nothing(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9)
    stub.verdict = "CANDIDATE"
    stub.supported = True
    _step(store, failed_path)

    action = _step(store, failed_path, now=_NOW + timedelta(days=7))

    assert action.kind is ActionKind.AWAIT_CONFIRMATION_APPROVAL
    assert action.hypothesis_id == "RO-A"
    assert "named human" in action.reason
    assert store.list_alpha_sources() == []


def test_a_screen_never_moves_a_queue_entry_out_of_proposed(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9)
    stub.verdict = "CANDIDATE"
    stub.supported = True

    _step(store, failed_path)

    stored = store.list_research_opportunities()
    assert all(i.opportunity.status is ResearchStatus.PROPOSED for i in stored)
    assert all(i.opportunity.approved_by is None for i in stored)


# -- restarts and duplication -------------------------------------------


def test_a_restart_does_not_re_run_the_screen(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # The loop runs on a schedule, and schedules retry. Running the same
    # experiment twice and recording both would inflate the research ledger
    # with activity that never happened -- and that ledger is the
    # denominator of the multiple-testing correction.
    _seed(store, "RO-A", priority=0.9)

    _step(store, failed_path)
    _step(store, failed_path, now=_NOW + timedelta(days=3))

    assert stub.calls == ["RO-A"]
    assert len(store.list_research_screens(quarter=_QUARTER)) == 1


def test_an_interrupted_screen_leaves_evidence_and_is_not_retried(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # A process killed mid-screen (a job timeout, which is how 19 nights
    # were lost in September) must not leave the quarter looking untouched.
    _seed(store, "RO-A", priority=0.9)
    screen_id = screen_id_for(quarter=_QUARTER, hypothesis_id="RO-A", harness="stub", version="1")
    from turboedge.storage.schemas import ResearchScreen

    store.upsert_research_screen(
        ResearchScreen(
            screen_id=screen_id,
            trial_id=screen_id,
            hypothesis_id="RO-A",
            family=str(InformationFamily.MACRO),
            harness="stub",
            harness_version="1",
            spec={"horizon": 3},
            data_version="pending",
            quarter=_QUARTER,
            status=ScreenStatus.RUNNING,
            started_at=_NOW - timedelta(hours=2),
        )
    )

    _step(store, failed_path)

    assert stub.calls == []
    remaining = store.list_research_screens(quarter=_QUARTER)
    assert [s.status for s in remaining] == [ScreenStatus.RUNNING]


def test_a_new_harness_version_is_screened_again(
    store: Store, failed_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "Already answered" must not be permanent across a change of method.
    _seed(store, "RO-A", priority=0.9)
    runner = _StubHarness()

    def register(version: str) -> None:
        registered = ApprovedHarness(
            name="stub",
            version=version,
            hypothesis_ids=frozenset({"RO-A"}),
            description="test double",
            spec={"horizon": 3},
            run=runner,
        )
        monkeypatch.setattr(
            step_module, "resolve_harness", lambda hid: registered if hid == "RO-A" else None
        )

    register("1")
    _step(store, failed_path)
    register("2")
    _step(store, failed_path, now=_NOW + timedelta(days=7))

    assert runner.calls == ["RO-A", "RO-A"]
    versions = {s.harness_version for s in store.list_research_screens(quarter=_QUARTER)}
    assert versions == {"1", "2"}


def test_the_screen_survives_a_store_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "t.duckdb"
    failed = tmp_path / "failed.json"
    runner = _StubHarness()
    registered = ApprovedHarness(
        name="stub",
        version="1",
        hypothesis_ids=frozenset({"RO-A"}),
        description="test double",
        spec={"horizon": 3},
        run=runner,
    )
    monkeypatch.setattr(
        step_module, "resolve_harness", lambda hid: registered if hid == "RO-A" else None
    )

    with Store(db) as store:
        store.init_schema()
        _seed(store, "RO-A", priority=0.9)
        run_research_step(store, failed_hypotheses_path=failed, now=_NOW)

    # A separate process, as the scheduled job will be.
    with Store(db) as store:
        run_research_step(store, failed_hypotheses_path=failed, now=_NOW + timedelta(days=7))
        screens = store.list_research_screens(quarter=_QUARTER)

    assert runner.calls == ["RO-A"]
    assert len(screens) == 1
    assert screens[0].data_version == "deadbeefcafe0001"


# -- the brakes -----------------------------------------------------------


def test_the_cap_counts_executions_not_decisions(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # The specific defect: `SCREEN_KIND` rows were counted but never
    # written, so the cap was permanently zero and a decision was being
    # confused with an experiment.
    _seed(store, "RO-A", priority=0.9)

    first = _step(store, failed_path)
    assert first.evidence["exploratory_screens_this_quarter"] == 0

    second = _step(store, failed_path, now=_NOW + timedelta(days=7))
    assert second.evidence["exploratory_screens_this_quarter"] == 1


def test_one_run_executes_at_most_one_harness(
    store: Store, failed_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _StubHarness()
    registered = ApprovedHarness(
        name="stub",
        version="1",
        hypothesis_ids=frozenset({"RO-A", "RO-B"}),
        description="test double",
        spec={},
        run=runner,
    )
    monkeypatch.setattr(
        step_module, "resolve_harness", lambda hid: registered if hid in {"RO-A", "RO-B"} else None
    )
    _seed(store, "RO-A", priority=0.9)
    _seed(store, "RO-B", priority=0.8)

    _step(store, failed_path)

    assert len(runner.calls) == step_module.MAX_EXECUTIONS_PER_RUN


def test_the_quarterly_cap_stops_screening(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    from turboedge.research.controller import EXPLORATORY_SCREENS_PER_QUARTER
    from turboedge.storage.schemas import ResearchScreen

    _seed(store, "RO-A", priority=0.9)
    for i in range(EXPLORATORY_SCREENS_PER_QUARTER):
        store.upsert_research_screen(
            ResearchScreen(
                screen_id=f"{_QUARTER}_SCREEN_filler{i:02d}",
                trial_id=f"{_QUARTER}_SCREEN_filler{i:02d}",
                hypothesis_id=f"RO-OTHER-{i}",
                family=str(InformationFamily.MACRO),
                harness="stub",
                harness_version="1",
                spec={},
                data_version="x",
                quarter=_QUARTER,
                status=ScreenStatus.COMPLETED,
                verdict="NO_IMPROVEMENT",
                started_at=_NOW,
                finished_at=_NOW,
            )
        )

    action = _step(store, failed_path)

    assert action.kind is ActionKind.NO_ACTION
    assert "at the cap" in action.reason
    assert stub.calls == []


def test_waits_when_the_sample_is_unmeasured(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9, sample=None)

    action = _step(store, failed_path)

    assert action.kind is ActionKind.AWAIT_DATA
    assert stub.calls == []
    assert store.list_research_screens(quarter=_QUARTER) == []


def test_waits_when_the_sample_is_too_small(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9, sample=10.0)

    action = _step(store, failed_path)

    assert action.kind is ActionKind.AWAIT_DATA
    assert action.evidence["effective_sample"] == 10.0
    assert stub.calls == []


def test_never_consumes_the_confirmatory_budget(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # Decisions and screens are recorded in the same table as confirmatory
    # trials. If they counted, the loop would spend the quarter's six
    # primary hypotheses merely by thinking.
    _seed(store, "RO-A", priority=0.9)
    _step(store, failed_path)

    confirmatory = [
        t
        for t in store.list_research_trials(quarter=_QUARTER)
        if t.kind not in {SCREEN_KIND, DECISION_KIND}
    ]

    assert confirmatory == []


def test_a_full_confirmatory_budget_does_not_stop_screening(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
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

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.evidence["confirmatory_trials_this_quarter"] == 6


# -- shelving -------------------------------------------------------------


def test_a_failed_family_is_shelved_and_the_shelf_is_persisted(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
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

    _step(store, failed_path)

    shelved = store.list_shelved_families()
    assert [s.family for s in shelved] == [str(InformationFamily.MACRO)]
    assert shelved[0].failure_count == 3
    assert "failures" in shelved[0].reason


def test_a_shelved_family_is_not_re_decided_every_run(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # The defect this closes: shelving is checked before every other rule,
    # so an unrecorded shelf meant the loop reached SHELVE_FAMILY on every
    # run and never screened anything again.
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

    first = _step(store, failed_path)
    second = _step(store, failed_path, now=_NOW + timedelta(days=7))

    assert first.kind is not ActionKind.SHELVE_FAMILY or second.kind is not ActionKind.SHELVE_FAMILY
    assert second.kind is not ActionKind.SHELVE_FAMILY
    assert len(store.list_shelved_families()) == 1


def test_a_shelved_family_is_not_screened(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
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

    _step(store, failed_path)
    _step(store, failed_path, now=_NOW + timedelta(days=7))

    assert stub.calls == []


# -- the real registry ----------------------------------------------------


def test_only_reviewed_exploratory_harnesses_are_registered() -> None:
    # `synthetic-ev` is confirmatory and pre-registered; `ko-calibration`
    # writes a promotion decision. Neither is the loop's to start, and an
    # entry appearing here for either would be a governance change made in
    # the wrong place.
    names = {h.name for h in harness_module.APPROVED_HARNESSES}

    # `day_night` was added 2026-10-25: read-only, promotes nothing, and
    # allowed to answer that the sample is too small. The protective part
    # of this test is the two exclusions below, not the count.
    assert names == {"excursion", "day_night", "lottery_pricing"}
    assert "synthetic_ev" not in names and "synthetic-ev" not in names
    assert "ko_calibration" not in names and "ko-calibration" not in names


def test_the_registry_answers_exactly_the_reviewed_catalogue_questions() -> None:
    assert harness_module.harnessable_hypothesis_ids() == {
        "RO-MAE-PREDICTION",
        "RO-MFE-PREDICTION",
        "RO-DAY-NIGHT-MOMENTUM",
        "RO-LOTTERY-PRICING",
    }


def test_every_catalogue_question_either_resolves_or_is_honestly_unanswered() -> None:
    # The honest state of the system: four of sixteen are answerable, and
    # the other twelve say so rather than being quietly skipped. The
    # day/night question is the only one of the three that does not wait
    # on the forward ledger, which is why it is the one that could run.
    from turboedge.meta.catalog import CATALOG

    resolved = {o.hypothesis_id: resolve_harness(o.hypothesis_id) is not None for o in CATALOG}

    assert sum(resolved.values()) == 4
    assert len(resolved) >= 16


def test_a_supporting_verdict_comes_from_the_harness_vocabulary() -> None:
    # INSUFFICIENT_SAMPLE, NO_IMPROVEMENT and IMPROVEMENT_NOT_SIGNIFICANT
    # are all legitimate answers and none of them is support.
    assert {"CANDIDATE"} == harness_module.SUPPORTING_VERDICTS


def test_the_harness_registry_is_a_constant_of_direct_references() -> None:
    # No name lookup, no import by string, no caller-supplied parameter.
    # Extending it has to be a reviewed source change -- that is the only
    # mechanism by which this loop can acquire a new capability.
    import ast
    import inspect

    source = inspect.getsource(harness_module)
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {"eval", "exec", "compile", "import_module", "__import__", "getattr"}

    assert not (called & forbidden), sorted(called & forbidden)


# -- isolation ------------------------------------------------------------


def test_it_touches_no_production_table(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # The scan/trading path must be provably unaffected.
    _seed(store, "RO-A", priority=0.9)
    _step(store, failed_path)

    for table in (
        "signals",
        "candidate_sets",
        "forward_ledger",
        "positions_manual",
        "product_snapshots",
        "model_registry",
        "alpha_sources",
        "ko_calibration_promotion",
    ):
        count = store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert count is not None and count[0] == 0, table


def test_a_dry_run_reaches_the_same_conclusion_and_writes_nothing(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9)

    dry = _step(store, failed_path, dry_run=True)

    assert store.list_research_trials(quarter=_QUARTER) == []
    assert store.list_research_screens(quarter=_QUARTER) == []
    assert stub.calls == []

    wet = _step(store, failed_path)
    assert dry.kind is wet.kind and dry.hypothesis_id == wet.hypothesis_id


def test_an_empty_queue_decides_nothing_rather_than_failing(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    action = _step(store, failed_path)

    assert action.kind is ActionKind.NO_ACTION
    assert stub.calls == []


# -- bounded retry of a transient failure --------------------------------


def test_a_failed_screen_is_retried_once_the_data_may_have_changed(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # The common early failure is "nothing has matured and been labelled
    # yet", which is transient. Blocking the question until next quarter
    # would mean a screen that failed in week one is never retried when the
    # data arrives in week four.
    _seed(store, "RO-A", priority=0.9)
    stub.raises = ValueError("produced zero usable rows")
    _step(store, failed_path)
    stub.raises = None

    later = _NOW + timedelta(days=step_module.FAILED_SCREEN_RETRY_AFTER_DAYS)
    action = _step(store, failed_path, now=later)

    assert action.kind is ActionKind.START_EXPLORATORY
    assert stub.calls == ["RO-A", "RO-A"]
    screens = store.list_research_screens(quarter=_QUARTER)
    assert len(screens) == 1, "a retry is the same experiment, not a second one"
    assert screens[0].status is ScreenStatus.COMPLETED
    assert screens[0].metrics["attempts"] == 2


def test_a_failed_screen_is_not_retried_immediately(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    _seed(store, "RO-A", priority=0.9)
    stub.raises = ValueError("no data")
    _step(store, failed_path)

    _step(store, failed_path, now=_NOW + timedelta(days=3))

    assert stub.calls == ["RO-A"]


def test_retries_are_bounded(store: Store, failed_path: Path, stub: _StubHarness) -> None:
    # A persistent failure is a finding. Retrying it forever at no cost to
    # the quarterly cap would be an unbounded loop.
    _seed(store, "RO-A", priority=0.9)
    stub.raises = ValueError("still no data")

    now = _NOW
    for _ in range(4):  # 4 x 14 days stays inside 2026Q4
        _step(store, failed_path, now=now)
        now += timedelta(days=step_module.FAILED_SCREEN_RETRY_AFTER_DAYS)

    assert len(stub.calls) == step_module.MAX_SCREEN_ATTEMPTS
    assert store.list_research_screens(quarter=_QUARTER)[0].metrics["attempts"] == (
        step_module.MAX_SCREEN_ATTEMPTS
    )


def test_a_completed_screen_is_never_re_run(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # Re-running an answered screen on the same data and method would be
    # multiple testing with extra steps.
    _seed(store, "RO-A", priority=0.9)
    _step(store, failed_path)

    # Inside the same quarter: a new quarter legitimately opens the
    # question again, which is a different property.
    _step(store, failed_path, now=_NOW + timedelta(days=55))

    assert stub.calls == ["RO-A"]


def test_an_interrupted_screen_is_not_retried_automatically(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # A RUNNING row is a screen whose process died. Re-running it on a
    # schedule would hide a job that is being killed every time -- which is
    # exactly how 19 cancelled nights went unnoticed. That needs a human.
    from turboedge.storage.schemas import ResearchScreen

    _seed(store, "RO-A", priority=0.9)
    screen_id = screen_id_for(quarter=_QUARTER, hypothesis_id="RO-A", harness="stub", version="1")
    store.upsert_research_screen(
        ResearchScreen(
            screen_id=screen_id,
            trial_id=screen_id,
            hypothesis_id="RO-A",
            family=str(InformationFamily.MACRO),
            harness="stub",
            harness_version="1",
            spec={},
            data_version="pending",
            quarter=_QUARTER,
            status=ScreenStatus.RUNNING,
            started_at=_NOW,
        )
    )

    _step(store, failed_path, now=_NOW + timedelta(days=55))

    assert stub.calls == []


# -- point-in-time protection --------------------------------------------


def test_the_harness_is_given_the_decision_time_not_the_wall_clock(
    store: Store, failed_path: Path, stub: _StubHarness
) -> None:
    # CLAUDE.md rule 5 is `available_at <= prediction_time`. A harness
    # handed `datetime.now()` instead of the step's `now` would read rows
    # that became knowable after the moment the loop is reasoning about,
    # and a backdated re-run would silently see the future.
    _seed(store, "RO-A", priority=0.9)

    _step(store, failed_path)

    assert stub.as_of == [_NOW]


def test_the_real_harness_passes_the_decision_time_through() -> None:
    # Static, because invoking the real harness needs a matured forward
    # ledger. What matters is that `as_of` comes from the parameter and not
    # from the clock: `build_excursion_dataset` excludes entries whose
    # `prediction_time > as_of`, and that filter is only a point-in-time
    # guarantee if the loop passes the time it is reasoning about.
    import ast
    import inspect

    source = inspect.getsource(harness_module._run_excursion)
    module = ast.parse(inspect.getsource(harness_module))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_excursion"
    )
    as_of_args = [
        kw.value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "as_of"
    ]

    assert as_of_args, "the harness no longer passes as_of at all"
    assert all(isinstance(value, ast.Name) and value.id == "now" for value in as_of_args), (
        "as_of must be the decision time the loop passed in, never a fresh clock read"
    )
    assert "datetime.now" not in source, (
        "a clock read inside the harness would let a backdated re-run see rows "
        "that were not knowable at the decision time"
    )


# -- the complete cycle, in one test -------------------------------------


def test_the_complete_cycle_on_controlled_data(
    store: Store, failed_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Select -> execute -> measure -> store -> re-prioritise -> next question.

    The whole loop in one place, with every stage asserted, because the
    individual tests above can all pass while the chain between them is
    broken -- which is exactly what was wrong before: the decision was
    recorded, nothing ran, and the next run re-decided the same thing.
    """
    runner = _StubHarness()
    registered = ApprovedHarness(
        name="stub",
        version="1",
        hypothesis_ids=frozenset({"RO-FIRST", "RO-SECOND"}),
        description="test double",
        spec={"horizon": 3, "min_train": 200},
        run=runner,
    )
    monkeypatch.setattr(
        step_module,
        "resolve_harness",
        lambda hid: registered if hid in {"RO-FIRST", "RO-SECOND"} else None,
    )
    _seed(store, "RO-FIRST", priority=0.9, information_gain=0.8)
    _seed(store, "RO-SECOND", priority=0.4, information_gain=0.6)

    # 1. The highest-ranked question is selected, with no human involved.
    first = _step(store, failed_path)
    assert first.kind is ActionKind.START_EXPLORATORY
    assert first.hypothesis_id == "RO-FIRST"

    # 2. An approved harness actually ran, at the decision time.
    assert runner.calls == ["RO-FIRST"]
    assert runner.as_of == [_NOW]

    # 3. The result is stored with everything needed to reproduce it.
    screen = store.list_research_screens(quarter=_QUARTER)[0]
    assert screen.status is ScreenStatus.COMPLETED
    assert screen.verdict == "NO_IMPROVEMENT"
    assert screen.harness == "stub" and screen.harness_version == "1"
    assert screen.spec["min_train"] == 200
    assert screen.data_version == "deadbeefcafe0001"
    assert screen.cost_seconds >= 0.0
    assert screen.metrics["attempts"] == 1

    # 4. The attempt is in the trial ledger, as a screen and not a
    #    confirmation.
    kinds = [t.kind for t in store.list_research_trials(quarter=_QUARTER)]
    assert SCREEN_KIND in kinds
    assert all(k in {SCREEN_KIND, DECISION_KIND} for k in kinds)

    # 5. What the screen measured has changed the ranking inputs.
    stored = {
        i.opportunity.hypothesis_id: i.opportunity for i in store.list_research_opportunities()
    }
    assert stored["RO-FIRST"].expected_information_gain.basis is EstimateBasis.MEASURED
    assert stored["RO-FIRST"].expected_information_gain.value == 0.0
    assert stored["RO-FIRST"].estimated_sample_size.value == 96.0
    # Untouched: the loop measured one question, not both.
    assert stored["RO-SECOND"].expected_information_gain.basis is EstimateBasis.DECLARED

    # 6. The next run picks the next question by itself.
    second = _step(store, failed_path, now=_NOW + timedelta(days=7))
    assert second.kind is ActionKind.START_EXPLORATORY
    assert second.hypothesis_id == "RO-SECOND"
    assert runner.calls == ["RO-FIRST", "RO-SECOND"]

    # 7. Two screens executed, two charged to the quarter, and the
    #    confirmatory budget still untouched.
    assert len(store.list_research_screens(quarter=_QUARTER)) == 2
    third = _step(store, failed_path, now=_NOW + timedelta(days=14), dry_run=True)
    assert third.evidence["exploratory_screens_this_quarter"] == 2
    assert third.evidence["confirmatory_trials_this_quarter"] == 0

    # 8. Both questions answered negatively, so there is nothing left to
    #    do -- and that is reported rather than re-screened.
    assert third.kind is ActionKind.NO_ACTION
    assert store.list_alpha_sources() == []


# ---------------------------------------------------------------------
# Harness-measured readiness. Until 2026-10-25 the loop could never start
# a screen at all, for any question, because readiness came from
# catalogue fields that are DECLARED by design -- see
# `harness.HarnessReadiness` for the runs that showed it.
# ---------------------------------------------------------------------


def test_every_registered_harness_can_measure_its_own_readiness() -> None:
    """A harness without one leaves its question permanently unready.

    Optional in the dataclass so a harness can be registered before its
    check is written; this test is what stops that staying the case.
    """
    for harness in harness_module.APPROVED_HARNESSES:
        assert harness.readiness is not None, harness.name


def test_an_empty_store_is_measured_as_not_ready_rather_than_unknown(store: Store) -> None:
    """Zero is a measurement. `None` would be read as "wait" either way,
    but it would hide whether the count was taken."""
    readiness = harness_module.measure_readiness(store)

    assert set(readiness) == harness_module.harnessable_hypothesis_ids()
    for hypothesis_id, measured in readiness.items():
        assert not measured.ready, hypothesis_id
        assert measured.detail


def test_day_night_readiness_turns_true_once_enough_bars_exist(store: Store) -> None:
    """The decisive case: this screen needs bars, not labels.

    Its catalogue family is MICROSTRUCTURE, whose primary source is mapped
    to `bnp_paribas` -- a product feed the screen never reads. The
    family-level proxy therefore cannot answer for it, which is why the
    harness counts for itself.
    """
    needed = int(harness_module._DAY_NIGHT_SPEC["min_train"]) + 260
    base = datetime(2015, 1, 1, 17, 30, tzinfo=UTC)
    bars = [
        UnderlyingBar(
            observation_time=base + timedelta(days=i),
            available_at=base + timedelta(days=i),
            retrieved_at=base + timedelta(days=i),
            source="test",
            schema_version="1",
            parser_version="1",
            quality_score=1.0,
            underlying_id="DAX",
            ts=base + timedelta(days=i),
            interval="1d",
            open=100.0 + i * 0.01,
            high=101.0 + i * 0.01,
            low=99.0 + i * 0.01,
            close=100.5 + i * 0.01,
        )
        for i in range(needed)
    ]
    store.append_underlying_bars(bars)

    measured = harness_module.measure_readiness(store)["RO-DAY-NIGHT-MOMENTUM"]
    assert measured.ready
    assert measured.effective_sample is not None
    assert measured.effective_sample > 0.0
    assert "DAX" in measured.detail


def test_harness_readiness_overrides_the_catalogue_proxy() -> None:
    """`build_candidates` must prefer the counted value over the declared one."""
    from turboedge.meta.catalog import catalog_by_id
    from turboedge.meta.research_queue import StoredOpportunity
    from turboedge.research.harness import HarnessReadiness
    from turboedge.research.step import build_candidates

    opportunity = catalog_by_id()["RO-DAY-NIGHT-MOMENTUM"]
    stored = [StoredOpportunity(opportunity=opportunity, priority=None, scored_at=None)]

    # Without a measurement the catalogue proxy stands, and it never says yes.
    before = build_candidates(stored, screened_hypothesis_ids=frozenset())[0]
    assert not before.data_ready
    assert before.effective_sample is None

    after = build_candidates(
        stored,
        screened_hypothesis_ids=frozenset(),
        harness_readiness={
            "RO-DAY-NIGHT-MOMENTUM": HarnessReadiness(
                ready=True, effective_sample=8000.0, detail="counted"
            )
        },
    )[0]
    assert after.data_ready
    assert after.effective_sample == 8000.0


def test_a_question_without_harness_readiness_keeps_the_catalogue_proxy() -> None:
    """The override is additive; it must not change unmeasured questions."""
    from turboedge.meta.catalog import catalog_by_id
    from turboedge.meta.research_queue import StoredOpportunity
    from turboedge.research.harness import HarnessReadiness
    from turboedge.research.step import build_candidates

    other = catalog_by_id()["RO-CBOE-VOL-STATE"]
    stored = [StoredOpportunity(opportunity=other, priority=None, scored_at=None)]
    candidate = build_candidates(
        stored,
        screened_hypothesis_ids=frozenset(),
        harness_readiness={
            "RO-DAY-NIGHT-MOMENTUM": HarnessReadiness(
                ready=True, effective_sample=8000.0, detail="counted"
            )
        },
    )[0]

    assert not candidate.data_ready
    assert candidate.effective_sample is None
