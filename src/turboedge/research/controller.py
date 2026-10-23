"""Which research step comes next -- decided, not chosen by a human.

The pieces of an autonomous research loop all existed before this module:
`meta/research_queue.py` holds the questions and the governance boundary,
`meta/value_of_information.py` ranks them, `learning/failed_hypotheses.py`
is the negative memory with its retest gate, `learning/trials.py` enforces
the quarterly budget, `research/partitions.py` separates exploration from
confirmation, and `backtest/` has the harnesses. What was missing was the
one decision that tied them together, and it was being made by a person.

This module makes that decision and nothing else. It is a pure function:
given the ranked queue, the negative memory, the data readiness, the
quarter's trial count and the clock, it returns exactly one action with the
measured reasons behind it. No I/O, no side effects, no network -- so the
whole of the loop's judgement is testable without a database, and a
disagreement about what it decided is settled by reading inputs rather than
by re-running a pipeline.

The autonomous lane is deliberately narrow
------------------------------------------
`ResearchQueue.approve` refuses to run without a named human, by design
(§8), and that is not weakened here. The loop operates in the one lane that
needs no approval:

    exploratory screening, DEVELOPMENT partition only, every run logged.

It may **not** start a confirmatory trial. Those consume the quarterly
budget of six primary hypotheses, they are the ones whose p-values mean
something, and they stay human-gated. The loop can reach the point of
saying "this is ready to confirm" and then stops, which is the correct
place for a machine to stop.

It also may not promote anything. `alpha/promotion.py` still has seven
unset gates and `GOVERNANCE.md` disagrees with `configs/ranking.yaml` about
position and cluster limits in both directions. Until both are resolved,
`PROMOTION_LOCKED` is the honest state and this module returns it as a
reason rather than treating an absent criterion as a satisfied one.

What stops it running away
--------------------------
Three independent brakes, none of which the loop can loosen:

* **Exploratory screens are counted.** They do not consume the confirmatory
  budget -- they are not confirmations -- but a cap per quarter exists so
  that "exploration" cannot become an unbounded search that launders
  multiple testing. Spec §27.1: every attempt counts.
* **The negative memory gates retests.** A family that has failed is not
  re-screened because it was renamed; `learning/failed_hypotheses.allow_retest`
  decides, and it requires a materially different condition.
* **Insufficient data means waiting.** `AWAIT_DATA` is a real outcome and
  the most common one early on. Shortening a confirmation window to get a
  result sooner is the one thing this loop must never do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

#: How many exploratory screens one quarter may contain. Separate from the
#: confirmatory budget of six (GOVERNANCE.md §1.2) and deliberately larger:
#: a screen on DEVELOPMENT data makes no claim and costs no inferential
#: credibility. It is capped at all because an unbounded screen count is
#: multiple testing by another name, and because an autonomous process with
#: no ceiling is one nobody can reason about.
EXPLORATORY_SCREENS_PER_QUARTER = 12

#: Research families the loop may screen without human approval. A family
#: not listed here is out of bounds -- the loop may rank it, propose it and
#: say it is ready, but not start it. Extending this list is a governance
#: change and therefore a human decision.
PERMITTED_EXPLORATORY_FAMILIES: frozenset[str] = frozenset(
    {
        "macro",
        "rates_credit",
        "sentiment",
        "cross_asset",
        "volatility_surface",
        "positioning",
        "breadth_dispersion",
        "path_statistics",
        "microstructure",
        "event_risk",
    }
)


class ActionKind(StrEnum):
    """The five things the loop may decide. There is no sixth."""

    #: Run a screen on DEVELOPMENT data for one question. No approval needed.
    START_EXPLORATORY = "START_EXPLORATORY"
    #: The best question's data are not yet sufficient. Wait; do not shorten.
    AWAIT_DATA = "AWAIT_DATA"
    #: A question has exploratory support and needs a confirmatory trial,
    #: which is human-gated. The loop stops here on purpose.
    AWAIT_CONFIRMATION_APPROVAL = "AWAIT_CONFIRMATION_APPROVAL"
    #: A family has failed repeatedly; stop spending screens on it.
    SHELVE_FAMILY = "SHELVE_FAMILY"
    #: Nothing is admissible. An explicitly correct outcome.
    NO_ACTION = "NO_ACTION"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One research question as the controller sees it.

    Deliberately a flat, primitive-only view rather than the full
    `StoredOpportunity`: the decision depends on these fields and nothing
    else, and keeping the boundary narrow is what makes the function
    testable without constructing a database row.

    `effective_sample` and `priority` are `None` when unmeasured. They are
    never defaulted to zero or to a guess -- an unknown sample size is a
    reason to wait, not a reason to proceed with a small one.
    """

    hypothesis_id: str
    family: str
    priority: float | None
    effective_sample: float | None
    required_sample: float
    data_ready: bool
    exploratory_done: bool
    has_exploratory_support: bool


@dataclass(frozen=True, slots=True)
class ResearchAction:
    """One decision, with the inputs that produced it.

    `evidence` carries the measured values the decision rested on, so a
    recorded action can be re-derived later without guessing what the
    controller saw. That is what makes the loop auditable rather than merely
    automatic.
    """

    kind: ActionKind
    reason: str
    hypothesis_id: str | None = None
    family: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def decide_next_action(
    *,
    candidates: Sequence[Candidate],
    blocked_families: Sequence[str] = (),
    failed_family_counts: dict[str, int] | None = None,
    exploratory_screens_this_quarter: int,
    confirmatory_trials_this_quarter: int,
    confirmatory_budget: int,
    now: datetime,
    shelve_after_failures: int = 3,
    promotion_unlocked: bool = False,
) -> ResearchAction:
    """Pick the next research step, or decide there is not one.

    Order matters and is the whole design. Each check answers a question
    that makes the later ones moot:

    1. Is the exploratory quarter spent?  -> stop; waiting is not failure.
    2. Has a family failed enough times?  -> shelve it before spending more.
    3. Does anything already have exploratory support? -> that is the most
       valuable next step, and it needs a human, so say so.
    4. Is the best admissible question's data sufficient? -> screen it.
    5. Otherwise -> wait for data, naming what is missing.

    `promotion_unlocked` exists only to be False. It is threaded through so
    that the reason "promotion is locked" is recorded in the action's
    evidence rather than being invisible, and so that the day the criteria
    are complete, exactly one call site changes.
    """
    failures = dict(failed_family_counts or {})
    blocked = set(blocked_families)
    base_evidence: dict[str, Any] = {
        "candidates": len(candidates),
        "exploratory_screens_this_quarter": exploratory_screens_this_quarter,
        "exploratory_cap": EXPLORATORY_SCREENS_PER_QUARTER,
        "confirmatory_trials_this_quarter": confirmatory_trials_this_quarter,
        "confirmatory_budget": confirmatory_budget,
        "promotion_unlocked": promotion_unlocked,
        "decided_at": now.isoformat(),
    }

    # 2. A family that keeps failing is shelved before anything else is
    #    considered, so the loop stops paying for it even in a quarter that
    #    still has room.
    for family, count in sorted(failures.items()):
        if count >= shelve_after_failures and family not in blocked:
            return ResearchAction(
                kind=ActionKind.SHELVE_FAMILY,
                family=family,
                reason=(
                    f"family {family!r} has {count} recorded failures, at or above the "
                    f"shelving threshold of {shelve_after_failures}; further screens "
                    "would spend the quarter re-testing a family the record already "
                    "answers"
                ),
                evidence={**base_evidence, "family_failures": count},
            )

    admissible = [
        c
        for c in candidates
        if c.family in PERMITTED_EXPLORATORY_FAMILIES and c.family not in blocked
    ]

    # 3. Exploratory support already exists somewhere: the next genuinely
    #    informative step is a confirmatory trial, and that is not the
    #    loop's to start.
    supported = [c for c in admissible if c.has_exploratory_support]
    if supported:
        best = max(supported, key=lambda c: (c.priority or 0.0, c.hypothesis_id))
        room = confirmatory_budget - confirmatory_trials_this_quarter
        return ResearchAction(
            kind=ActionKind.AWAIT_CONFIRMATION_APPROVAL,
            hypothesis_id=best.hypothesis_id,
            family=best.family,
            reason=(
                f"{best.hypothesis_id} has exploratory support and needs a "
                "confirmatory trial. Confirmatory trials consume the quarterly "
                f"primary-hypothesis budget ({confirmatory_trials_this_quarter} of "
                f"{confirmatory_budget} used) and require a named human via "
                "`research approve`; this loop does not start them."
            ),
            evidence={**base_evidence, "confirmatory_budget_remaining": room},
        )

    # 1. The exploratory ceiling. Checked after shelving and support so that
    #    a spent quarter still reports the more useful of the two states.
    if exploratory_screens_this_quarter >= EXPLORATORY_SCREENS_PER_QUARTER:
        return ResearchAction(
            kind=ActionKind.NO_ACTION,
            reason=(
                f"{exploratory_screens_this_quarter} exploratory screens already run "
                f"this quarter, at the cap of {EXPLORATORY_SCREENS_PER_QUARTER}. An "
                "uncapped screen count is multiple testing under another name."
            ),
            evidence=base_evidence,
        )

    if not admissible:
        return ResearchAction(
            kind=ActionKind.NO_ACTION,
            reason=(
                "no candidate is in a permitted exploratory family and not shelved. "
                f"Permitted: {sorted(PERMITTED_EXPLORATORY_FAMILIES)}"
            ),
            evidence=base_evidence,
        )

    pending = [c for c in admissible if not c.exploratory_done]
    if not pending:
        return ResearchAction(
            kind=ActionKind.NO_ACTION,
            reason=(
                "every admissible candidate has already been screened; none produced "
                "exploratory support. Nothing new is learned by screening them again "
                "on the same data."
            ),
            evidence=base_evidence,
        )

    # 4./5. Best-ranked question first; an unmeasured priority sorts last
    #       rather than being treated as zero-and-therefore-comparable.
    ranked = sorted(
        pending,
        key=lambda c: (c.priority is None, -(c.priority or 0.0), c.hypothesis_id),
    )
    best = ranked[0]

    if not best.data_ready or best.effective_sample is None:
        return ResearchAction(
            kind=ActionKind.AWAIT_DATA,
            hypothesis_id=best.hypothesis_id,
            family=best.family,
            reason=(
                f"{best.hypothesis_id} ranks highest but its data are not ready "
                f"(data_ready={best.data_ready}, effective_sample="
                f"{best.effective_sample}). Waiting is the correct step; shortening "
                "the window to get a result sooner is not available to this loop."
            ),
            evidence={**base_evidence, "required_sample": best.required_sample},
        )

    if best.effective_sample < best.required_sample:
        return ResearchAction(
            kind=ActionKind.AWAIT_DATA,
            hypothesis_id=best.hypothesis_id,
            family=best.family,
            reason=(
                f"{best.hypothesis_id} has an effective sample of "
                f"{best.effective_sample:.1f} against the {best.required_sample:.0f} "
                "its screen requires. Effective, not nominal: rows from one day are "
                "one observation repeated."
            ),
            evidence={
                **base_evidence,
                "effective_sample": best.effective_sample,
                "required_sample": best.required_sample,
            },
        )

    return ResearchAction(
        kind=ActionKind.START_EXPLORATORY,
        hypothesis_id=best.hypothesis_id,
        family=best.family,
        reason=(
            f"{best.hypothesis_id} ranks highest among admissible unscreened "
            f"candidates, its family is permitted, its effective sample "
            f"({best.effective_sample:.1f}) meets the required "
            f"{best.required_sample:.0f}, and the quarter has "
            f"{EXPLORATORY_SCREENS_PER_QUARTER - exploratory_screens_this_quarter} "
            "screens left. DEVELOPMENT partition only."
        ),
        evidence={
            **base_evidence,
            "priority": best.priority,
            "effective_sample": best.effective_sample,
            "required_sample": best.required_sample,
        },
    )


__all__ = [
    "EXPLORATORY_SCREENS_PER_QUARTER",
    "PERMITTED_EXPLORATORY_FAMILIES",
    "ActionKind",
    "Candidate",
    "ResearchAction",
    "decide_next_action",
]
