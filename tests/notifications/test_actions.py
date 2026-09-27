"""Tests for the human action notification layer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from turboedge.notifications.actions import (
    ActionStatus,
    ActionType,
    HumanActionRequest,
    Urgency,
)

_NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _request(**over: object) -> HumanActionRequest:
    defaults: dict[str, object] = {
        "action_id": "ACT-1",
        "action_type": ActionType.TRADE_DECISION_REQUIRED,
        "created_at": _NOW,
        "urgency": Urgency.ELEVATED,
        "reason": "a candidate cleared every gate",
        "evidence_summary": "lcb_ev 0.004, p_ko 0.02, cluster ok",
        "recommended_action": "review and place manually if you agree",
    }
    defaults.update(over)
    return HumanActionRequest.model_validate(defaults)


def test_a_promotion_notice_cannot_be_approved_or_rejected() -> None:
    """The distinction GOVERNANCE.md §1.1a created, encoded rather than assumed.

    A NOTICE reports a promotion that already happened under frozen criteria.
    There is no pending decision, so approving it would be theatre.
    """
    for status in (ActionStatus.APPROVED, ActionStatus.REJECTED):
        with pytest.raises(ValidationError, match="only informs"):
            _request(
                action_type=ActionType.ALPHA_PROMOTION_NOTICE,
                status=status,
                resolved_at=_NOW,
                resolved_by="cc",
            )


def test_a_promotion_notice_can_still_be_closed() -> None:
    """Informational does not mean permanent; it means nothing is being asked."""
    closed = _request(
        action_type=ActionType.ALPHA_PROMOTION_NOTICE,
        status=ActionStatus.RESOLVED,
        resolved_at=_NOW,
    )

    assert closed.is_informational
    assert closed.status is ActionStatus.RESOLVED


def test_a_decision_cannot_be_recorded_without_a_named_approver() -> None:
    """An approval with no approver is not an audit trail."""
    with pytest.raises(ValidationError, match="requires resolved_by"):
        _request(
            action_type=ActionType.ALPHA_PROMOTION_REQUIRED,
            status=ActionStatus.APPROVED,
            resolved_at=_NOW,
        )

    approved = _request(
        action_type=ActionType.ALPHA_PROMOTION_REQUIRED,
        status=ActionStatus.APPROVED,
        resolved_at=_NOW,
        resolved_by="cc",
    )
    assert approved.resolved_by == "cc"


def test_both_promotion_types_exist_and_differ() -> None:
    """Collapsing them would leave 'you may want to know' versus 'nothing
    proceeds without you' to convention."""
    assert not _request(action_type=ActionType.ALPHA_PROMOTION_REQUIRED).is_informational
    assert _request(action_type=ActionType.ALPHA_PROMOTION_NOTICE).is_informational


def test_a_closed_request_must_say_when() -> None:
    with pytest.raises(ValidationError, match="requires resolved_at"):
        _request(status=ActionStatus.EXPIRED)


def test_dedup_key_is_stable_against_rewording() -> None:
    """A reworded reason must not resend.

    A materially changed situation gets a new action_id instead.
    """
    a = _request(reason="a candidate cleared every gate")
    b = _request(reason="candidate passed all gates")

    assert a.dedup_key() == b.dedup_key()
    assert _request(action_id="ACT-2").dedup_key() != a.dedup_key()


def test_dedup_key_changes_with_urgency() -> None:
    """An escalation in urgency is new information, not a repeat."""
    assert _request(urgency=Urgency.URGENT).dedup_key() != _request().dedup_key()


def test_urgent_requests_escalate_once_and_then_stop() -> None:
    """The failure mode a notification layer must avoid is not silence, it is
    being tuned out."""
    urgent = _request(urgency=Urgency.URGENT)

    assert urgent.should_escalate()
    urgent.escalation_count = 1
    assert not urgent.should_escalate()


def test_non_urgent_and_resolved_requests_never_escalate() -> None:
    assert not _request(urgency=Urgency.ROUTINE).should_escalate()
    assert not _request(
        urgency=Urgency.URGENT,
        status=ActionStatus.RESOLVED,
        resolved_at=_NOW,
    ).should_escalate()


def test_a_request_carries_what_is_needed_to_act_without_a_dashboard() -> None:
    """A notification that makes the reader reconstruct the situation is a
    dashboard with extra steps."""
    r = _request(
        cli_command="turboedge position add --wkn ABC123 --qty 100 --price 4.86",
        report_reference="reports/weekly/2026-W39.json",
        alternative_actions=["skip this candidate", "wait for the next scan"],
        deadline=_NOW,
    )

    assert r.cli_command and r.cli_command.startswith("turboedge ")
    assert r.report_reference
    assert len(r.alternative_actions) == 2
    assert r.evidence_summary
