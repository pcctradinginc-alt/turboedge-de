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

What this module does NOT do
----------------------------
It does not execute an experiment. The loop decides that a screen is
admissible and records it; running one uses the already-approved harnesses
(`backtest`, `research excursion`), which are separately scheduled. That
separation is deliberate: the user's standing rule is that the system may
not generate and launch new executable procedures of its own, and a module
that both chose a method and ran it would be one step from doing exactly
that.

It does not approve, promote, or move a queue entry out of PROPOSED. Those
require a named human (`meta/research_queue.approve`) and that boundary is
untouched here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from turboedge.learning import failed_hypotheses
from turboedge.meta.research_memory import family_for_feature
from turboedge.meta.research_opportunity import EstimateBasis, StoredOpportunity
from turboedge.research.controller import (
    ActionKind,
    Candidate,
    ResearchAction,
    decide_next_action,
)
from turboedge.storage.schemas import ResearchTrial, TrialStatus

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

#: Prefix for a recorded exploratory screen, likewise not a confirmation.
SCREEN_KIND = "exploratory_screen"

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
) -> list[Candidate]:
    """Project persisted opportunities onto the controller's flat view.

    `effective_sample` and `priority` stay `None` when the stored estimate
    is not MEASURED. That is the point: a DECLARED judgement about how much
    data exists is not a measurement, and the controller treats an unknown
    sample as a reason to wait rather than as a number to compare.
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
        out.append(
            Candidate(
                hypothesis_id=opportunity.hypothesis_id,
                family=str(opportunity.information_family),
                priority=item.priority.score if item.priority is not None else None,
                effective_sample=effective,
                required_sample=REQUIRED_SCREEN_SAMPLE,
                data_ready=data_ready,
                exploratory_done=opportunity.hypothesis_id in screened_hypothesis_ids,
                has_exploratory_support=opportunity.hypothesis_id in supported_hypothesis_ids,
            )
        )
    return out


def run_research_step(
    store: Store,
    *,
    failed_hypotheses_path: Path,
    now: datetime,
    confirmatory_budget: int = 6,
    dry_run: bool = False,
) -> ResearchAction:
    """Decide and record one research step. Idempotent within a quarter.

    Reads: the research queue, the recorded trials for this quarter, and the
    negative memory. Writes: at most one `research_trials` row.

    `dry_run` decides nothing differently -- it only skips the write, so the
    same inputs produce the same reported action either way. A dry run that
    could reach a different conclusion would be useless for checking what
    the scheduled run will do.
    """
    quarter = _quarter(now)
    stored = store.list_research_opportunities()
    trials = store.list_research_trials(quarter=quarter)

    screened = frozenset(
        t.description.split("|", 1)[0].strip()
        for t in trials
        if t.kind == SCREEN_KIND and "|" in t.description
    )
    screens_this_quarter = sum(1 for t in trials if t.kind == SCREEN_KIND)
    # Only genuine primary hypotheses count against the confirmatory
    # budget. A recorded decision or a screen must never be countable as
    # one, or the loop would spend the budget by thinking about it.
    confirmatory = sum(1 for t in trials if t.kind not in {SCREEN_KIND, DECISION_KIND})

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

    action = decide_next_action(
        candidates=build_candidates(stored, screened_hypothesis_ids=screened),
        failed_family_counts=family_counts,
        exploratory_screens_this_quarter=screens_this_quarter,
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
        unattributable_failures=unattributable,
        **{f"evidence_{k}": v for k, v in action.evidence.items()},
    )

    if dry_run:
        return action

    trial_id = decision_trial_id(action, quarter=quarter)
    existing = {t.trial_id for t in trials}
    if trial_id in existing:
        # The same conclusion as a previous run this quarter. Nothing to
        # add; re-writing it would only move its timestamp.
        return action

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
    "REQUIRED_SCREEN_SAMPLE",
    "SCREEN_KIND",
    "ActionKind",
    "build_candidates",
    "decision_trial_id",
    "run_research_step",
]
