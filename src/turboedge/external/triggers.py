"""Research triggers: the hand-off from "data exists" to "someone may look".

Spec §43 asks for one thing that is easy to state and easy to get wrong:
when a series first becomes ready, emit exactly one event -- not one every
day the series remains ready. A daily re-emission would flood the research
queue with duplicates of the same question and quietly inflate the
multiple-testing denominator in `GOVERNANCE.md` §11.2, which is how a real
finding gets buried under its own notifications.

The boundary this module holds is the same one `meta/research_queue.py`
holds: **a trigger is a proposal, never an approval.** It records that a
dataset became usable. It does not start research, does not consume a
research budget unit, does not touch CONFIRMATION or FORWARD data, and does
not create anything past `ResearchStatus.PROPOSED`. Spec §44 states the
distinction this enforces: a dataset being research-ready says nothing at
all about whether an alpha is there, and most of the time there will not be.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from turboedge.external.readiness import (
    READINESS_LEVEL,
    DataReadinessRecord,
    ReadinessState,
)
from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime


class TriggerType(StrEnum):
    """One per readiness milestone worth telling the research layer about."""

    EXPLORATORY_DATA_READY = "EXPLORATORY_DATA_READY"
    VALIDATION_DATA_READY = "VALIDATION_DATA_READY"
    CONFIRMATION_DATA_READY = "CONFIRMATION_DATA_READY"
    FORWARD_SAMPLE_MATURE = "FORWARD_SAMPLE_MATURE"


#: Which state first earns which trigger. A series that jumps straight from
#: COLLECTING to CONFIRMATION_READY (a PIT-safe backfill with deep history
#: can do exactly that) earns every trigger it passed through, so nothing is
#: lost just because the data arrived all at once.
TRIGGER_FOR_STATE: dict[ReadinessState, TriggerType] = {
    ReadinessState.EXPLORATORY_READY: TriggerType.EXPLORATORY_DATA_READY,
    ReadinessState.VALIDATION_READY: TriggerType.VALIDATION_DATA_READY,
    ReadinessState.CONFIRMATION_READY: TriggerType.CONFIRMATION_DATA_READY,
    ReadinessState.FORWARD_MATURE: TriggerType.FORWARD_SAMPLE_MATURE,
}


class ResearchTrigger(BaseModel):
    """One emitted, persisted, idempotent readiness event.

    `dedup_key` is what makes "exactly once" a property of the data rather
    than of the caller remembering. It excludes the timestamp on purpose: the
    same series reaching the same milestone under the same policy is the same
    event, whether it is noticed today or next week.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    series_id: str
    trigger_type: TriggerType
    emitted_at: TzAwareDatetime

    from_state: ReadinessState | None
    to_state: ReadinessState

    policy_version: str
    effective_n: float
    calendar_span_days: int

    #: Set once a `ResearchOpportunity` has been proposed for this trigger,
    #: so a restart cannot propose the same question twice.
    handed_off_at: TzAwareDatetime | None = None
    hypothesis_id: str | None = None

    note: str = ""
    schema_version: str = SCHEMA_VERSION

    @property
    def dedup_key(self) -> str:
        return f"{self.source}|{self.series_id}|{self.trigger_type}|{self.policy_version}"

    @model_validator(mode="after")
    def _handoff_is_recorded_completely(self) -> Self:
        """A hand-off without a timestamp, or a timestamp without a
        hypothesis, is a half-written record that a restart would read as
        either "not yet done" or "done, but untraceable"."""
        if (self.handed_off_at is None) != (self.hypothesis_id is None):
            raise ValueError(
                f"{self.dedup_key}: handed_off_at and hypothesis_id must be set "
                "together -- a partial hand-off cannot be resumed safely"
            )
        return self


def triggers_for_transition(
    record: DataReadinessRecord,
    *,
    already_emitted: frozenset[str] = frozenset(),
) -> list[ResearchTrigger]:
    """Every trigger `record` newly earns, oldest milestone first.

    Args:
        record: the readiness evaluation just produced.
        already_emitted: `dedup_key`s of triggers persisted previously. Pass
            the stored set, not an empty one -- it is the only thing keeping
            a daily evaluation from re-emitting a milestone reached months
            ago.

    Returns an empty list for a series that is not ready, that has not moved,
    or whose milestones were all emitted before. A demotion emits nothing:
    "this got worse" is a readiness state, and re-emitting the milestone when
    it recovers would be the daily-flood failure by another route.
    """
    level = READINESS_LEVEL[record.state]
    if level < 1:
        return []

    out: list[ResearchTrigger] = []
    for state, trigger_type in TRIGGER_FOR_STATE.items():
        if READINESS_LEVEL[state] > level:
            continue
        candidate = ResearchTrigger(
            source=record.source,
            series_id=record.series_id,
            trigger_type=trigger_type,
            emitted_at=record.evaluated_at,
            from_state=record.previous_state,
            to_state=record.state,
            policy_version=record.policy_version,
            effective_n=record.effective_n,
            calendar_span_days=record.calendar_span_days,
        )
        if candidate.dedup_key in already_emitted:
            continue
        out.append(candidate)

    out.sort(key=lambda t: READINESS_LEVEL[_state_for(t.trigger_type)])
    return out


def _state_for(trigger_type: TriggerType) -> ReadinessState:
    for state, candidate in TRIGGER_FOR_STATE.items():
        if candidate is trigger_type:
            return state
    raise KeyError(trigger_type)  # pragma: no cover - exhaustive by construction
