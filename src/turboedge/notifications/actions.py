"""Human action requests -- the notification layer.

TurboEdge should require no routine dashboard watching. Nothing is emailed for
a successful scan; a message arrives when, and only when, a human decision is
actually needed.

**The distinction that matters here** follows from GOVERNANCE.md §1.1a, where
routine alpha promotion became evidence-gated and automatic:

* `ALPHA_PROMOTION_NOTICE` **informs**. An alpha cleared frozen criteria and
  was promoted. Nothing is being asked; the human retains veto, disable and
  emergency stop, but need not act for the promotion to stand.
* `ALPHA_PROMOTION_REQUIRED` **asks**, and is retained unchanged for the six
  categories §1.1a still reserves to the human: hard portfolio risk limits,
  maximum aggregate exposure, the promotion criteria themselves, governance
  rules, confirmation methodology, the allowed instrument universe, and any
  introduction of automatic broker execution.

Both exist deliberately. Collapsing them would leave the difference between
"you may want to know" and "nothing proceeds without you" to convention, and
that is the one distinction in this file worth encoding. A NOTICE therefore
cannot carry an approval, and a REQUIRED cannot resolve without a named
approver -- both enforced below.

Deduplication reuses `notifications/dedup.py`; nothing new is built for it.
Phase A defines the schema and the dedup key only and sends nothing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from turboedge.notifications.dedup import notification_hash
from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime


class ActionType(StrEnum):
    """Why a human is being contacted. See the module docstring on the two
    promotion members -- they are not redundant."""

    TRADE_DECISION_REQUIRED = "TRADE_DECISION_REQUIRED"
    POSITION_ACTION_REQUIRED = "POSITION_ACTION_REQUIRED"
    RESEARCH_APPROVAL_REQUIRED = "RESEARCH_APPROVAL_REQUIRED"
    ALPHA_PROMOTION_REQUIRED = "ALPHA_PROMOTION_REQUIRED"
    ALPHA_PROMOTION_NOTICE = "ALPHA_PROMOTION_NOTICE"
    SYSTEM_FAILURE_REQUIRES_ATTENTION = "SYSTEM_FAILURE_REQUIRES_ATTENTION"
    DATA_QUALITY_FAILURE_REQUIRES_ATTENTION = "DATA_QUALITY_FAILURE_REQUIRES_ATTENTION"


#: Types that only inform. They may be acknowledged but never approved or
#: rejected, because there is no decision pending to approve.
INFORMATIONAL_TYPES: frozenset[ActionType] = frozenset({ActionType.ALPHA_PROMOTION_NOTICE})


class Urgency(StrEnum):
    ROUTINE = "ROUTINE"
    ELEVATED = "ELEVATED"
    URGENT = "URGENT"


class ActionStatus(StrEnum):
    OPEN = "OPEN"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    RESOLVED = "RESOLVED"


#: Statuses that assert a human decided something, as opposed to merely noting
#: that a matter is closed. Only these require an approver.
DECISION_STATUSES: frozenset[ActionStatus] = frozenset(
    {ActionStatus.APPROVED, ActionStatus.REJECTED}
)


class HumanActionRequest(BaseModel):
    """One thing a human is being told about, or asked to decide.

    Carries its own evidence and the exact command to act, because a
    notification that requires the reader to go and reconstruct the situation
    is a dashboard with extra steps.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_id: str
    action_type: ActionType
    created_at: TzAwareDatetime
    urgency: Urgency
    reason: str
    evidence_summary: str
    recommended_action: str
    alternative_actions: list[str] = Field(default_factory=list)
    #: After this, the request stops being actionable -- a trade proposal whose
    #: quotes have gone stale, for instance. Absent when it does not decay.
    deadline: TzAwareDatetime | None = None
    #: The literal command, where one exists. Not a description of it.
    cli_command: str | None = None
    report_reference: str | None = None

    status: ActionStatus = ActionStatus.OPEN
    resolved_at: TzAwareDatetime | None = None
    resolved_by: str | None = None
    #: Incremented on escalation. The rule is escalate an unresolved urgent
    #: action once and then stop; a second reminder is a notification system
    #: training its reader to ignore it.
    escalation_count: int = Field(default=0, ge=0)

    schema_version: str = SCHEMA_VERSION

    @property
    def is_informational(self) -> bool:
        return self.action_type in INFORMATIONAL_TYPES

    @model_validator(mode="after")
    def _check_status_is_coherent(self) -> Self:
        if self.is_informational and self.status in DECISION_STATUSES:
            raise ValueError(
                f"{self.action_type} only informs, so it cannot be {self.status}: there is no "
                "pending decision to approve or reject. Use RESOLVED to close it, or "
                "ALPHA_PROMOTION_REQUIRED if a human decision is genuinely needed "
                "(GOVERNANCE.md §1.1a)"
            )
        if self.status in DECISION_STATUSES and not (self.resolved_by or "").strip():
            raise ValueError(
                f"status {self.status} requires resolved_by: an approval with no named approver "
                "is not an audit trail"
            )
        if self.status is not ActionStatus.OPEN and self.resolved_at is None:
            raise ValueError(f"status {self.status} requires resolved_at")
        return self

    def dedup_key(self) -> str:
        """Stable hash for `NotificationDeduplicator`.

        Keyed on the action's identity and its type, not on its prose: a reason
        string that is reworded must not resend, and a materially changed
        situation gets a new `action_id` rather than a reworded old one.
        """
        return notification_hash(
            self.action_id,
            str(self.action_type),
            {"urgency": str(self.urgency)},
        )

    def should_escalate(self, *, max_escalations: int = 1) -> bool:
        """Whether an unresolved URGENT request may be re-sent.

        Once, by default. The failure mode a notification layer has to avoid is
        not silence, it is being tuned out.
        """
        return (
            self.status is ActionStatus.OPEN
            and self.urgency is Urgency.URGENT
            and self.escalation_count < max_escalations
        )


__all__ = [
    "DECISION_STATUSES",
    "INFORMATIONAL_TYPES",
    "ActionStatus",
    "ActionType",
    "HumanActionRequest",
    "Urgency",
]
