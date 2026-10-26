"""One cycle of the autonomous research loop: read state, decide, record.

`controller.decide_next_action` holds the judgement and touches nothing.
This module is its counterpart: it assembles the controller's inputs from
what is already persisted, calls it once, and writes the decision down. It
adds no judgement of its own.

Idempotence is the property that matters, because this runs on a schedule
and schedules retry. A decision is recorded in `research_trials` under a
trial id derived deterministically from the quarter, the action and the
hypothesis, so a re-run of the same week's cycle writes the same row rather
than a second one. `research_trials` has `trial_id` as its primary key, so
the storage layer enforces that rather than this module remembering to.

That also gives resumability for free: an interrupted run leaves either no
row or the correct row, never half of one, and the next run reaches the
same conclusion from the same inputs.

What executing a screen means here
----------------------------------
When the decision is `START_EXPLORATORY`, this module runs it -- but only
by looking the question up in `research/harness.py` and calling the
procedure registered there. It never constructs a procedure. If no approved
harness answers the question, that is recorded as `NO_APPROVED_HARNESS` and
the loop moves on to the next question rather than inventing one.

Each execution is recorded in `research_screens` with the harness and its
version, the parameters actually used, a fingerprint of the data the
harness saw, the verdict, the metrics, any error, and the wall-clock cost.
A screen is claimed as `RUNNING` before the harness is invoked, so a
process that dies mid-screen leaves evidence that it tried, and the
deterministic `screen_id` means a re-run after a restart updates that row
instead of recording a second experiment that never happened.

Only executed screens count against the quarterly screen cap. A decision
is not an experiment: counting one as the other would let the loop exhaust
its quarter by thinking.

What this module does NOT do
----------------------------
It does not approve, promote, or move a queue entry out of PROPOSED. Those
require a named human (`meta/research_queue.approve`) and that boundary is
untouched here. It does not start a confirmatory trial, and the two
existing confirmatory or promotion-writing harnesses are deliberately
absent from the registry it reads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING

import structlog

from turboedge.learning import failed_hypotheses
from turboedge.meta.research_memory import family_for_feature
from turboedge.meta.research_opportunity import Estimate, EstimateBasis, StoredOpportunity
from turboedge.notifications.alpha import AlphaDispatchResult, alert_for_screen, send_alpha_alert
from turboedge.notifications.dedup import NotificationDeduplicator
from turboedge.notifications.gmail import GmailNotifier
from turboedge.research.controller import (
    ActionKind,
    Candidate,
    ResearchAction,
    decide_next_action,
)
from turboedge.research.harness import HarnessReadiness, measure_readiness, resolve_harness
from turboedge.storage.schemas import (
    ResearchScreen,
    ResearchTrial,
    ScreenStatus,
    ShelvedFamily,
    TrialStatus,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from pathlib import Path

    from turboedge.storage.duckdb import Store

log = structlog.get_logger(__name__)

#: `kind` written to `research_trials` for a recorded decision. Distinct
#: from the confirmatory trial kinds so that
#: `learning/trials.effective_number_of_trials` and the quarterly budget
#: can tell a decision record apart from a primary hypothesis. A screen
#: must never be countable as a confirmation.
DECISION_KIND = "autonomous_decision"

#: `kind` written to `research_trials` for an executed exploratory screen,
#: likewise not a confirmation. One such row exists for each row in
#: `research_screens`, so the trial ledger shows the experiment and the
#: screen table shows what it measured.
SCREEN_KIND = "exploratory_screen"

#: Screen states that mean a harness was actually invoked, and which
#: therefore count against the quarterly cap and cost something.
#: `NO_APPROVED_HARNESS` is excluded: nothing ran, nothing was learned, and
#: nothing should be charged for. `RUNNING` is included -- a screen that
#: died mid-flight consumed its cost whether or not it produced a verdict.
EXECUTED_SCREEN_STATUSES: frozenset[ScreenStatus] = frozenset(
    {ScreenStatus.RUNNING, ScreenStatus.COMPLETED, ScreenStatus.FAILED}
)

#: How many decisions one run may make. A run that reaches a question with
#: no approved harness records that and asks again, so a catalogue of
#: mostly un-harnessed questions does not cost one scheduled run per entry
#: to walk past. Bounded because an unbounded re-decide loop on surprising
#: state is how a scheduled job becomes a spin.
MAX_DECISIONS_PER_RUN = 6

#: How long a failed screen waits before it may be attempted again. The
#: common early failure is "nothing has matured and been labelled yet",
#: which is transient: blocking the question until next quarter would mean a
#: screen that failed in week one is not retried when the data arrives in
#: week four. Two weeks is long enough that a retry sees genuinely different
#: data rather than the same empty table.
FAILED_SCREEN_RETRY_AFTER_DAYS = 14

#: How many times one screen may be attempted in a quarter. A transient
#: failure deserves a retry; a persistent one is a finding, and retrying it
#: forever at no cost to the quarterly cap would be an unbounded loop.
MAX_SCREEN_ATTEMPTS = 3

#: How many harnesses one run may actually execute. One. The cap exists so
#: that a quarter's screens cannot all be spent in a single run, which
#: would defeat the point of spreading exploration over time, and so that
#: one run's wall-clock cost stays predictable.
MAX_EXECUTIONS_PER_RUN = 1

#: Effective sample a screen requires before it may start. Conservative and
#: not tuned: it is the monthly readiness profile's exploratory threshold
#: (`external/readiness.py:DEFAULT_PROFILES`), reused rather than invented
#: so that one number governs both "this dataset is usable" and "this
#: question may be screened".
REQUIRED_SCREEN_SAMPLE = 48.0


def _quarter(now: datetime) -> str:
    return f"{now.year}Q{(now.month - 1) // 3 + 1}"


def decision_trial_id(action: ResearchAction, *, quarter: str) -> str:
    """A stable id for one decision, so a re-run cannot duplicate it.

    Derived from the quarter, the action kind and the hypothesis -- not from
    the clock. Two runs in the same quarter that reach the same conclusion
    about the same question are the same decision, and recording it twice
    would inflate the research ledger with activity that never happened.
    """
    material = f"{quarter}|{action.kind}|{action.hypothesis_id or '-'}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:10]
    return f"{quarter}_AUTO_{digest}"


def build_candidates(
    stored: Sequence[StoredOpportunity],
    *,
    screened_hypothesis_ids: frozenset[str],
    supported_hypothesis_ids: frozenset[str] = frozenset(),
    unharnessed_hypothesis_ids: frozenset[str] = frozenset(),
    harness_readiness: Mapping[str, HarnessReadiness] | None = None,
) -> list[Candidate]:
    """Project persisted opportunities onto the controller's flat view.

    `effective_sample` and `priority` stay `None` when the stored estimate
    is not MEASURED. That is the point: a DECLARED judgement about how much
    data exists is not a measurement, and the controller treats an unknown
    sample as a reason to wait rather than as a number to compare.

    `harness_readiness` overrides both for a question whose harness counted
    its own data (`research/harness.measure_readiness`). That is a
    measurement too, and a better one: the catalogue's
    `estimated_sample_size` is DECLARED by design and never becomes
    MEASURED, and its `data_availability` is the reachability of a
    *family's* primary source rather than of what a given screen reads.
    Together those made every question permanently unready and
    `START_EXPLORATORY` unreachable -- see `HarnessReadiness` for the run
    that showed it. A question with no harness readiness keeps the
    catalogue proxy unchanged.
    """
    out: list[Candidate] = []
    for item in stored:
        opportunity = item.opportunity
        sample = opportunity.estimated_sample_size
        effective = (
            float(sample.value)
            if sample.basis is EstimateBasis.MEASURED and sample.value is not None
            else None
        )
        availability = opportunity.data_availability
        data_ready = (
            availability.basis is EstimateBasis.MEASURED
            and availability.value is not None
            and availability.value >= 1.0
        )
        measured = (harness_readiness or {}).get(opportunity.hypothesis_id)
        readiness_detail: str | None = None
        if measured is not None:
            data_ready = measured.ready
            effective = measured.effective_sample
            readiness_detail = measured.detail
        out.append(
            Candidate(
                hypothesis_id=opportunity.hypothesis_id,
                family=str(opportunity.information_family),
                priority=item.priority.score if item.priority is not None else None,
                effective_sample=effective,
                required_sample=REQUIRED_SCREEN_SAMPLE,
                data_ready=data_ready,
                readiness_detail=readiness_detail,
                exploratory_done=opportunity.hypothesis_id in screened_hypothesis_ids,
                has_exploratory_support=opportunity.hypothesis_id in supported_hypothesis_ids,
                screenable=opportunity.hypothesis_id not in unharnessed_hypothesis_ids,
            )
        )
    return out


def screen_id_for(*, quarter: str, hypothesis_id: str, harness: str, version: str) -> str:
    """A stable id for one screen, so a restart cannot duplicate an experiment.

    Includes the harness version: a screen run under an older version must
    not suppress a re-run under a newer one, or the loop would treat
    "already answered" as permanent across a change of method. It does NOT
    include the data version, which is only knowable after the harness has
    run -- that is recorded on the row instead, so a reader can see which
    data produced the result even though the identity is per quarter.
    """
    material = f"{quarter}|{hypothesis_id}|{harness}|{version}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:10]
    return f"{quarter}_SCREEN_{digest}"


def _matches_current_harness(screen: ResearchScreen) -> bool:
    """Whether this screen was run by the harness now registered for it."""
    harness = resolve_harness(screen.hypothesis_id)
    return (
        harness is not None
        and screen.harness == harness.name
        and screen.harness_version == harness.version
    )


def _attempts(screen: ResearchScreen) -> int:
    recorded = screen.metrics.get("attempts")
    return int(recorded) if isinstance(recorded, int | float) else 1


def _is_retryable(screen: ResearchScreen, *, now: datetime) -> bool:
    """Whether a failed screen has waited long enough to be attempted again.

    Only `FAILED` qualifies. A `RUNNING` row is a screen whose process died,
    and re-running it automatically would hide a job that is being killed
    every time -- that one needs a human to look. A `COMPLETED` screen has
    an answer, and re-running it on the same data and method would be
    multiple testing with extra steps.
    """
    if screen.status is not ScreenStatus.FAILED or screen.finished_at is None:
        return False
    if _attempts(screen) >= MAX_SCREEN_ATTEMPTS:
        return False
    return (now - screen.finished_at).days >= FAILED_SCREEN_RETRY_AFTER_DAYS


def _record_screen_trial(
    store: Store, screen: ResearchScreen, *, action: ResearchAction, now: datetime
) -> None:
    """Mirror an executed screen into the trial ledger.

    `research_trials` is what `learning/trials.effective_number_of_trials`
    reads, and §27.1 requires every attempt to be counted there. The kind
    keeps it distinguishable from a confirmation, so a screen can never be
    mistaken for a primary hypothesis.

    A retry updates the existing row rather than adding one: §27.1 counts
    attempts at one question, and one experiment retried after a transient
    failure is one attempt, not two. `insert_research_trial` is a plain
    INSERT and would raise on the primary key, so the status is moved
    instead -- the screen table holds the attempt count.
    """
    if any(
        t.trial_id == screen.trial_id for t in store.list_research_trials(quarter=screen.quarter)
    ):
        store.update_research_trial_status(screen.trial_id, TrialStatus.EXPERIMENTAL)
        return
    store.insert_research_trial(
        ResearchTrial(
            trial_id=screen.trial_id,
            kind=SCREEN_KIND,
            description=(
                f"{screen.hypothesis_id}|{screen.harness}@{screen.harness_version}|"
                f"{screen.status.value}|{screen.verdict or '-'}|{action.reason}"
            )[:1000],
            created_at=now,
            quarter=screen.quarter,
            status=TrialStatus.EXPERIMENTAL,
        )
    )


def _feed_result_into_priority(
    store: Store,
    screen: ResearchScreen,
    *,
    stored: Sequence[StoredOpportunity],
    now: datetime,
) -> bool:
    """Turn a screen result into measured ranking inputs. Returns whether it did.

    Two inputs become MEASURED, and only these two:

    * `estimated_sample_size`, from the screen's own effective sample. The
      catalogue states it as a declared judgement and says a later stage
      should measure it; this is that stage. Effective, not nominal.
    * `expected_information_gain`, and only downwards, and only on a
      completed screen that found no improvement. A screen that measured
      "there is little here" has measured exactly that input, and leaving
      the declared optimism in place would make the loop re-rank the same
      exhausted question to the top forever.

    What this deliberately does NOT do is raise anything on a positive
    result. The escalation for support is `AWAIT_CONFIRMATION_APPROVAL`,
    which is a human's call; a loop that also promoted its own question up
    the ranking on its own screen evidence would be marking its own work.
    Nor does it touch `status`: only a named human moves an entry out of
    PROPOSED (§8).
    """
    if screen.status is not ScreenStatus.COMPLETED:
        return False
    match = next(
        (item for item in stored if item.opportunity.hypothesis_id == screen.hypothesis_id),
        None,
    )
    if match is None:
        return False

    effective = screen.metrics.get("effective_sample")
    updates: dict[str, Estimate] = {}
    if isinstance(effective, int | float):
        updates["estimated_sample_size"] = Estimate(
            basis=EstimateBasis.MEASURED,
            value=float(effective),
            note=(
                f"measured by screen {screen.screen_id} "
                f"({screen.harness}@{screen.harness_version}, data {screen.data_version}): "
                "independent observations, not rows"
            ),
        )

    if not screen.produced_support:
        current = match.opportunity.expected_information_gain
        previous = current.value if current.value is not None else 1.0
        if previous > 0.0:
            updates["expected_information_gain"] = Estimate(
                basis=EstimateBasis.MEASURED,
                value=0.0,
                note=(
                    f"screen {screen.screen_id} returned {screen.verdict!r} on "
                    f"{effective} effective observations; the declared gain of "
                    f"{previous} is superseded by that measurement. Raising it again "
                    "requires new data or a new method, not a re-run."
                ),
            )

    if not updates:
        return False

    store.upsert_research_opportunity(
        match.opportunity.model_copy(update=updates),
        priority=match.priority,
        scored_at=now,
    )
    return True


def _execute_screen(
    store: Store,
    action: ResearchAction,
    *,
    quarter: str,
    now: datetime,
    existing: Sequence[ResearchScreen],
) -> ResearchScreen | None:
    """Run the approved harness for this question, or record that there is none.

    Returns the recorded screen, or `None` when this screen was already
    recorded in this quarter -- which is what makes a restart safe: the
    deterministic id is looked up before anything is invoked, so an
    interrupted run never charges the quarter twice for one experiment.
    """
    hypothesis_id = action.hypothesis_id
    if hypothesis_id is None:  # pragma: no cover - START_EXPLORATORY always carries one
        return None
    harness = resolve_harness(hypothesis_id)

    if harness is None:
        screen_id = screen_id_for(
            quarter=quarter, hypothesis_id=hypothesis_id, harness="-", version="-"
        )
        if any(s.screen_id == screen_id for s in existing):
            return None
        screen = ResearchScreen(
            screen_id=screen_id,
            trial_id=screen_id,
            hypothesis_id=hypothesis_id,
            family=action.family or "-",
            harness="-",
            harness_version="-",
            spec={},
            data_version="-",
            quarter=quarter,
            status=ScreenStatus.NO_APPROVED_HARNESS,
            error=(
                f"no approved harness answers {hypothesis_id}. The loop does not "
                "build one; registering a reviewed procedure in "
                "`research/harness.py` is a human change."
            ),
            started_at=now,
            finished_at=now,
        )
        store.upsert_research_screen(screen)
        _record_screen_trial(store, screen, action=action, now=now)
        log.info(
            "research_screen_no_approved_harness",
            hypothesis_id=hypothesis_id,
            family=action.family,
            screen_id=screen_id,
        )
        return screen

    screen_id = screen_id_for(
        quarter=quarter,
        hypothesis_id=hypothesis_id,
        harness=harness.name,
        version=harness.version,
    )
    previous = next((s for s in existing if s.screen_id == screen_id), None)
    if previous is not None and not _is_retryable(previous, now=now):
        # Already attempted this quarter under this harness version. Not
        # re-run: the cost was already paid and the record already exists.
        log.info(
            "research_screen_already_recorded",
            screen_id=screen_id,
            status=previous.status.value,
        )
        return None
    attempt = _attempts(previous) + 1 if previous is not None else 1

    # Claimed before the harness is invoked. A process that dies here leaves
    # a RUNNING row, which counts as executed and is visible as unfinished,
    # rather than leaving no trace of an experiment that consumed its cost.
    claim = ResearchScreen(
        screen_id=screen_id,
        trial_id=screen_id,
        hypothesis_id=hypothesis_id,
        family=action.family or "-",
        harness=harness.name,
        harness_version=harness.version,
        spec=dict(harness.spec),
        data_version="pending",
        quarter=quarter,
        status=ScreenStatus.RUNNING,
        metrics={"attempts": attempt},
        started_at=now,
    )
    store.upsert_research_screen(claim)
    _record_screen_trial(store, claim, action=action, now=now)

    started = perf_counter()
    try:
        outcome = harness.run(store, hypothesis_id, now)
    except Exception as exc:
        cost = perf_counter() - started
        failed = claim.model_copy(
            update={
                "status": ScreenStatus.FAILED,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "cost_seconds": cost,
                "metrics": {"attempts": attempt},
                "finished_at": now,
            }
        )
        store.upsert_research_screen(failed)
        log.warning(
            "research_screen_failed",
            screen_id=screen_id,
            hypothesis_id=hypothesis_id,
            error=failed.error,
            cost_seconds=round(cost, 3),
        )
        return failed

    cost = perf_counter() - started
    done = claim.model_copy(
        update={
            "status": ScreenStatus.COMPLETED,
            "verdict": outcome.verdict,
            "metrics": {
                **outcome.metrics,
                "supported": outcome.supported,
                "attempts": attempt,
            },
            "spec": outcome.spec or dict(harness.spec),
            "data_version": outcome.data_version,
            "cost_seconds": cost,
            "finished_at": now,
        }
    )
    store.upsert_research_screen(done)
    log.info(
        "research_screen_completed",
        screen_id=screen_id,
        hypothesis_id=hypothesis_id,
        harness=f"{harness.name}@{harness.version}",
        verdict=outcome.verdict,
        supported=outcome.supported,
        data_version=outcome.data_version,
        cost_seconds=round(cost, 3),
    )
    return done


def _shelve(store: Store, action: ResearchAction, *, now: datetime) -> None:
    """Persist a shelving decision so the loop gets past it.

    Shelving is checked before every other rule, so an unrecorded shelf is a
    loop that reaches the same conclusion on every run and screens nothing.
    """
    if action.family is None:  # pragma: no cover - SHELVE_FAMILY always carries one
        return
    store.upsert_shelved_family(
        ShelvedFamily(
            family=action.family,
            shelved_at=now,
            failure_count=int(action.evidence.get("family_failures", 0)),
            reason=action.reason[:1000],
        )
    )
    log.info("research_family_shelved", family=action.family, reason=action.reason)


def _notify_screen(
    store: Store,
    screen: ResearchScreen,
    *,
    stored: Sequence[StoredOpportunity],
    notifier: GmailNotifier | None,
    recipients: Sequence[str],
    now: datetime,
    run_url: str | None,
    run_id: str | None,
) -> AlphaDispatchResult | None:
    """Turn a completed screen into a notification, automatically.

    This is the link the operator asked for: a module and a manual test
    command are not an automatic path. A screen that finishes here decides
    its own notification type, checks the scientific status, builds the
    message, attempts delivery, records success, and leaves a failure
    eligible for the next run.

    `None` means no message was warranted -- a screen that found nothing
    is not news, and `alert_for_screen` returns `None` for every
    unsupported verdict, `LOOKAHEAD_SUSPECT` among them.

    The wording follows the question's `research_kind`, so a COST_EDGE
    result is announced as a Kostenvorteil-Kandidat with an explicit "this
    is not alpha" banner. That is the condition the 2026-10-26
    authorisation for autonomous product-selection research was granted
    under.
    """
    entry = next((o for o in stored if o.opportunity.hypothesis_id == screen.hypothesis_id), None)
    if entry is None:
        # A screen for a question not in the queue should not happen; it is
        # reported rather than notified about, because the wording depends
        # on a classification that is not available.
        log.warning(
            "research_screen_not_in_queue",
            screen_id=screen.screen_id,
            hypothesis_id=screen.hypothesis_id,
        )
        return None

    opportunity = entry.opportunity
    alert = alert_for_screen(
        hypothesis_id=opportunity.hypothesis_id,
        research_kind=opportunity.research_kind,
        verdict=screen.verdict or "",
        supported=bool(screen.metrics.get("supported")),
        description=opportunity.description,
        mechanism=opportunity.expected_information_gain.note or "nicht benannt",
        underlyings=opportunity.affected_underlyings,
        horizon_days=None,
        metrics=screen.metrics,
        screen_id=screen.screen_id,
        run_url=run_url,
        run_id=run_id,
    )
    if alert is None:
        return None

    result = send_alpha_alert(
        notifier,
        NotificationDeduplicator(store),
        alert,
        recipients=list(recipients) or ["unconfigured@example.invalid"],
        now=now,
    )
    log.info(
        "research_screen_notified",
        screen_id=screen.screen_id,
        hypothesis_id=screen.hypothesis_id,
        research_kind=str(opportunity.research_kind),
        sent=result.sent,
        suppressed=result.suppressed_as_duplicate,
        error=result.error,
    )
    return result


def run_research_step(
    store: Store,
    *,
    failed_hypotheses_path: Path,
    now: datetime,
    confirmatory_budget: int = 6,
    dry_run: bool = False,
    notifier: GmailNotifier | None = None,
    recipients: Sequence[str] = (),
    run_url: str | None = None,
    run_id: str | None = None,
) -> ResearchAction:
    """Decide, execute and record one research step. Idempotent within a quarter.

    Reads: the research queue, this quarter's trials and screens, the
    persisted shelf, and the negative memory. Writes: the shelf, at most one
    executed screen with its trial row, the measured ranking inputs that
    screen produced, and at most one decision row.

    Returns the last action decided. A run may decide more than once: a
    question with no approved harness is recorded as such and the loop asks
    again, up to `MAX_DECISIONS_PER_RUN`, so a catalogue of mostly
    un-harnessed questions does not cost one scheduled run per entry to walk
    past. It executes at most `MAX_EXECUTIONS_PER_RUN` harness.

    `dry_run` decides nothing differently -- it only skips every write,
    including the execution. The same inputs produce the same reported
    action either way, which is what makes a dry run useful for checking
    what the scheduled run will do.
    """
    quarter = _quarter(now)
    stored = store.list_research_opportunities()
    screens = list(store.list_research_screens(quarter=quarter))
    trials = store.list_research_trials(quarter=quarter)
    blocked = [entry.family for entry in store.list_shelved_families()]

    # A recorded failure carries a `feature`, not a family, so the family is
    # resolved through the same token overlap `_failure_similarity` already
    # uses (`meta/research_memory.family_for_feature`). An unattributable
    # feature counts against nothing -- the conservative direction, since it
    # leaves research open rather than shelving a family on a guess.
    failures = failed_hypotheses.load(failed_hypotheses_path)
    family_counts: dict[str, int] = {}
    unattributable = 0
    for entry in failures:
        family = family_for_feature(entry.feature)
        if family is None:
            unattributable += 1
            continue
        key = str(family)
        family_counts[key] = family_counts.get(key, 0) + 1

    # Only genuine primary hypotheses count against the confirmatory
    # budget. A recorded decision or a screen must never be countable as
    # one, or the loop would spend the budget by thinking about it.
    confirmatory = sum(1 for t in trials if t.kind not in {SCREEN_KIND, DECISION_KIND})

    executions = 0
    action: ResearchAction | None = None
    for _attempt in range(MAX_DECISIONS_PER_RUN):
        # Derived from the screen table on every pass, so a screen executed
        # earlier in this same run is immediately visible to the next
        # decision rather than being re-chosen.
        executed = [s for s in screens if s.status in EXECUTED_SCREEN_STATUSES]
        # Screened means screened *by the harness version now registered*.
        # `screen_id_for` includes the version so that a method change is not
        # suppressed by an older answer, and this has to agree with it or the
        # two would disagree about whether a question is still open. Bumping a
        # version therefore reopens questions -- a deliberate human act, and
        # bounded by the quarterly cap like any other screen.
        screened_ids = frozenset(
            s.hypothesis_id
            for s in screens
            if s.status in EXECUTED_SCREEN_STATUSES
            and _matches_current_harness(s)
            and not _is_retryable(s, now=now)
        )
        # Support is evidence, and evidence does not expire with a version
        # bump: a question that a screen already backed stays backed.
        supported_ids = frozenset(s.hypothesis_id for s in screens if s.produced_support)
        no_harness_ids = frozenset(
            s.hypothesis_id for s in screens if s.status is ScreenStatus.NO_APPROVED_HARNESS
        )

        action = decide_next_action(
            candidates=build_candidates(
                stored,
                screened_hypothesis_ids=screened_ids,
                supported_hypothesis_ids=supported_ids,
                unharnessed_hypothesis_ids=no_harness_ids,
                harness_readiness=measure_readiness(store),
            ),
            blocked_families=blocked,
            failed_family_counts=family_counts,
            exploratory_screens_this_quarter=len(executed),
            confirmatory_trials_this_quarter=confirmatory,
            confirmatory_budget=confirmatory_budget,
            now=now,
        )

        log.info(
            "research_step_decided",
            kind=str(action.kind),
            hypothesis_id=action.hypothesis_id,
            family=action.family,
            reason=action.reason,
            executed_screens_this_quarter=len(executed),
            unattributable_failures=unattributable,
            **{f"evidence_{k}": v for k, v in action.evidence.items()},
        )

        if dry_run:
            return action

        if action.kind is ActionKind.SHELVE_FAMILY:
            _shelve(store, action, now=now)
            blocked.append(action.family or "")
            # Re-decide: the whole point of persisting the shelf is that the
            # loop gets past it instead of stopping here every week.
            continue

        if action.kind is not ActionKind.START_EXPLORATORY:
            break

        if executions >= MAX_EXECUTIONS_PER_RUN:
            break

        screen = _execute_screen(store, action, quarter=quarter, now=now, existing=screens)
        if screen is None:
            # Already recorded this quarter and nothing changed, so asking
            # again would reach the same answer. Stop rather than spin.
            break
        screens.append(screen)
        if screen.status is ScreenStatus.NO_APPROVED_HARNESS:
            # Nothing ran and nothing was charged. Ask again so the next
            # question gets its turn in this same run.
            continue
        executions += 1
        if _feed_result_into_priority(store, screen, stored=stored, now=now):
            stored = store.list_research_opportunities()
        _notify_screen(
            store,
            screen,
            stored=stored,
            notifier=notifier,
            recipients=recipients,
            now=now,
            run_url=run_url,
            run_id=run_id,
        )
        break

    assert action is not None  # MAX_DECISIONS_PER_RUN >= 1
    trial_id = decision_trial_id(action, quarter=quarter)
    if trial_id not in {t.trial_id for t in trials}:
        store.insert_research_trial(
            ResearchTrial(
                trial_id=trial_id,
                kind=DECISION_KIND,
                description=f"{action.hypothesis_id or '-'}|{action.kind}|{action.reason}"[:1000],
                created_at=now,
                quarter=quarter,
                status=TrialStatus.EXPERIMENTAL,
            )
        )
    return action


__all__ = [
    "DECISION_KIND",
    "EXECUTED_SCREEN_STATUSES",
    "FAILED_SCREEN_RETRY_AFTER_DAYS",
    "MAX_DECISIONS_PER_RUN",
    "MAX_EXECUTIONS_PER_RUN",
    "MAX_SCREEN_ATTEMPTS",
    "REQUIRED_SCREEN_SAMPLE",
    "SCREEN_KIND",
    "ActionKind",
    "build_candidates",
    "decision_trial_id",
    "run_research_step",
    "screen_id_for",
]
