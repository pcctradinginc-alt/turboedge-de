"""Tests for the exploration/confirmation firewall."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from turboedge.research.partitions import (
    PartitionAction,
    PartitionFrozenError,
    PartitionType,
    ResearchPartition,
)

_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_T1 = datetime(2025, 1, 1, tzinfo=UTC)
_T2 = datetime(2026, 1, 1, tzinfo=UTC)
_NOW = datetime(2026, 9, 27, tzinfo=UTC)


def _partition(ptype: PartitionType, **over: object) -> ResearchPartition:
    defaults: dict[str, object] = {
        "partition_id": f"P-{ptype}",
        "name": str(ptype),
        "partition_type": ptype,
        "start_time": _T0,
        "end_time": None if ptype is PartitionType.FORWARD else _T1,
        "created_at": _T0,
    }
    defaults.update(over)
    return ResearchPartition.model_validate(defaults)


def test_a_frozen_confirmation_partition_cannot_be_re_cut() -> None:
    """The single rule this module exists for.

    A confirmation window whose edges can move after freezing does not give
    out-of-sample evidence; it gives evidence out of whichever sample produced
    the desired answer.
    """
    p = _partition(PartitionType.CONFIRMATION)
    p.freeze(at=_NOW)

    with pytest.raises(PartitionFrozenError, match="not out-of-sample evidence"):
        p.rebound(end_time=_T2)
    with pytest.raises(PartitionFrozenError):
        p.rebound(start_time=_T1)

    assert p.start_time == _T0
    assert p.end_time == _T1


def test_an_unfrozen_partition_can_still_be_rebound() -> None:
    """Freezing is the act that locks it, so before that it must be editable."""
    p = _partition(PartitionType.CONFIRMATION)
    p.rebound(end_time=_T2)

    assert p.end_time == _T2


def test_forward_end_time_advances_even_when_frozen_but_its_start_does_not() -> None:
    """A FORWARD window growing is the passage of time, not a boundary choice.

    Its start stays locked so already-partitioned history cannot be reassigned.
    """
    p = _partition(PartitionType.FORWARD)
    p.freeze(at=_NOW)

    p.rebound(end_time=_T2)
    assert p.end_time == _T2

    with pytest.raises(PartitionFrozenError):
        p.rebound(start_time=_T1)


def test_refreezing_at_a_different_time_is_refused() -> None:
    """Moving the freeze timestamp would move the audit trail with it."""
    p = _partition(PartitionType.CONFIRMATION)
    p.freeze(at=_NOW)

    p.freeze(at=_NOW)  # idempotent at the same instant
    with pytest.raises(PartitionFrozenError, match="auditable"):
        p.freeze(at=_T2)


def test_only_forward_may_be_open_ended() -> None:
    """An open-ended window is only honest while it is still being written."""
    assert _partition(PartitionType.FORWARD).end_time is None

    for ptype in (PartitionType.DEVELOPMENT, PartitionType.VALIDATION, PartitionType.CONFIRMATION):
        with pytest.raises(ValidationError, match="only FORWARD"):
            _partition(ptype, end_time=None)


def test_confirmation_permits_only_confirmation() -> None:
    """No exploring inside the confirmation window.

    That is how a one-shot test quietly becomes a search.
    """
    p = _partition(PartitionType.CONFIRMATION)

    assert p.permits(PartitionAction.CONFIRM)
    assert not p.permits(PartitionAction.EXPLORE)
    assert not p.permits(PartitionAction.SELECT_CANDIDATES)


def test_a_partition_cannot_grant_itself_wider_permissions() -> None:
    with pytest.raises(ValidationError, match="widening"):
        _partition(
            PartitionType.CONFIRMATION,
            allowed_actions=frozenset({PartitionAction.CONFIRM, PartitionAction.EXPLORE}),
        )


def test_default_permissions_match_the_partition_type() -> None:
    assert _partition(PartitionType.DEVELOPMENT).permits(PartitionAction.EXPLORE)
    assert _partition(PartitionType.VALIDATION).permits(PartitionAction.SELECT_CANDIDATES)
    assert _partition(PartitionType.FORWARD).permits(PartitionAction.ACCUMULATE)
    assert not _partition(PartitionType.DEVELOPMENT).permits(PartitionAction.CONFIRM)


def test_boundaries_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="must be after"):
        _partition(PartitionType.DEVELOPMENT, start_time=_T1, end_time=_T0)


def test_provenance_is_carried_so_the_sequence_is_auditable() -> None:
    """The module cannot prove a boundary was set before a result was seen.

    It can make the sequence checkable afterwards, and that is what these
    fields are for -- the docstring says plainly that it is not enforcement.
    """
    p = _partition(PartitionType.CONFIRMATION, git_commit="abc123", purpose="2026Q4-003")
    p.freeze(at=_NOW)

    assert p.created_at == _T0
    assert p.frozen_at == _NOW
    assert p.git_commit == "abc123"
    assert p.is_frozen
