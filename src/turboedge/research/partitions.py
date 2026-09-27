"""Chronological research partitions -- the exploration/confirmation firewall.

An alpha factory needs to try many things. Trying many things inflates false
positives. The resolution is not a looser threshold, it is a hard split
between where you may look and where you may conclude (spec §18, §19, §51).

    DEVELOPMENT   explore freely; conclusions from here are never evidence
    VALIDATION    select candidates; reuse is tracked
    CONFIRMATION  one shot, frozen before the test, never reopened
    FORWARD       accumulates in real time, append-only, never rewritten

The load-bearing rule is that a CONFIRMATION boundary cannot move once frozen.
If it can, "out-of-sample" means "out of the sample I chose after seeing the
answer", which is the failure this whole structure exists to prevent.

**What this module cannot enforce, and says so rather than pretending.** No
type system can tell whether a boundary was chosen before or after someone
looked at a result. `created_at`, `frozen_at` and `git_commit` make the
sequence auditable after the fact; they do not make it impossible. A boundary
set after a result exists is a governance violation even when every validator
here passes. This repository has already shown the pattern is real -- its
2026Q3 adaptation budget of 6 stands at 13, every overrun individually
justified in writing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime


class PartitionType(StrEnum):
    """What a stretch of time may be used for."""

    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    CONFIRMATION = "CONFIRMATION"
    FORWARD = "FORWARD"


class PartitionAction(StrEnum):
    """What a partition permits.

    A typed set rather than free text: "allowed_actions: exploratory analysis
    ok" is a note, not a constraint, and cannot be checked by anything.
    """

    EXPLORE = "EXPLORE"
    SELECT_CANDIDATES = "SELECT_CANDIDATES"
    CONFIRM = "CONFIRM"
    ACCUMULATE = "ACCUMULATE"


#: What each partition type permits. CONFIRMATION allows only CONFIRM -- no
#: exploring in the confirmation window, which is how a "one-shot" test quietly
#: becomes a search.
DEFAULT_ACTIONS: dict[PartitionType, frozenset[PartitionAction]] = {
    PartitionType.DEVELOPMENT: frozenset({PartitionAction.EXPLORE}),
    PartitionType.VALIDATION: frozenset(
        {PartitionAction.EXPLORE, PartitionAction.SELECT_CANDIDATES}
    ),
    PartitionType.CONFIRMATION: frozenset({PartitionAction.CONFIRM}),
    PartitionType.FORWARD: frozenset({PartitionAction.ACCUMULATE}),
}


class PartitionFrozenError(RuntimeError):
    """Raised when a frozen CONFIRMATION partition's boundaries are moved."""


class ResearchPartition(BaseModel):
    """One chronological slice of data with a declared purpose.

    Mutable by design -- a FORWARD partition's `end_time` advances as data
    accumulates -- so the freeze rule is enforced in `rebound`, the one place
    boundaries may change, rather than by making the whole object immutable.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    partition_id: str
    name: str
    partition_type: PartitionType
    start_time: TzAwareDatetime
    #: None means open-ended, which only FORWARD may be.
    end_time: TzAwareDatetime | None = None
    created_at: TzAwareDatetime
    frozen_at: TzAwareDatetime | None = None
    purpose: str = ""
    allowed_actions: frozenset[PartitionAction] = Field(default_factory=frozenset)
    git_commit: str | None = None
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.end_time is None and self.partition_type is not PartitionType.FORWARD:
            raise ValueError(
                f"{self.partition_type} needs an end_time; only FORWARD may stay open-ended, "
                "because only FORWARD is still being written"
            )
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError(f"end_time {self.end_time} must be after start_time {self.start_time}")

        permitted = DEFAULT_ACTIONS[self.partition_type]
        if not self.allowed_actions:
            object.__setattr__(self, "allowed_actions", permitted)
        elif not self.allowed_actions <= permitted:
            raise ValueError(
                f"{self.partition_type} permits {sorted(permitted)}, got "
                f"{sorted(self.allowed_actions)} -- widening a partition's permissions is how "
                "a one-shot confirmation window becomes a search"
            )
        return self

    @property
    def is_frozen(self) -> bool:
        return self.frozen_at is not None

    def permits(self, action: PartitionAction) -> bool:
        return action in self.allowed_actions

    def freeze(self, *, at: TzAwareDatetime) -> None:
        """Lock this partition's boundaries.

        Idempotent only in the sense that re-freezing with a different time is
        refused: the first freeze is the one that counts, and moving it would
        move the audit trail with it.
        """
        if self.is_frozen and self.frozen_at != at:
            raise PartitionFrozenError(
                f"{self.partition_id} was frozen at {self.frozen_at}; re-freezing at {at} "
                "would rewrite the timestamp that makes the freeze auditable"
            )
        self.frozen_at = at

    def rebound(
        self, *, start_time: TzAwareDatetime | None = None, end_time: TzAwareDatetime | None = None
    ) -> None:
        """Move this partition's boundaries. Refused once frozen.

        The single rule this module exists for. A CONFIRMATION window whose
        edges can move after freezing does not provide out-of-sample evidence;
        it provides evidence out of whichever sample produced the desired
        answer.

        FORWARD is the exception for `end_time` only: it advances as data
        arrives, which is not a boundary choice but the passage of time. Its
        `start_time` is still locked by a freeze, so already-partitioned
        history cannot be reassigned (§51).
        """
        if self.is_frozen:
            moving_forward_end = (
                self.partition_type is PartitionType.FORWARD
                and start_time is None
                and end_time is not None
            )
            if not moving_forward_end:
                raise PartitionFrozenError(
                    f"{self.partition_id} ({self.partition_type}) was frozen at {self.frozen_at}; "
                    "its boundaries cannot be moved. A confirmation window that can be "
                    "re-cut after the fact is not out-of-sample evidence."
                )
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time


__all__ = [
    "DEFAULT_ACTIONS",
    "PartitionAction",
    "PartitionFrozenError",
    "PartitionType",
    "ResearchPartition",
]
