"""Tests for system generations -- the guard against self-congratulation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from turboedge.research.generations import (
    GenerationEvidenceError,
    GenerationStatus,
    SystemGeneration,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 4, 1, tzinfo=UTC)
_T2 = datetime(2026, 7, 1, tzinfo=UTC)
_T3 = datetime(2026, 10, 1, tzinfo=UTC)


def _gen(gid: str, **over: object) -> SystemGeneration:
    defaults: dict[str, object] = {
        "generation_id": gid,
        "created_at": _T0,
        "git_commit": "abc123",
        "config_hash": "cfg1",
    }
    defaults.update(over)
    return SystemGeneration.model_validate(defaults)


def _measured(gid: str, net_ev: float, start: datetime, end: datetime, **over: object):
    return _gen(gid, evaluation_start=start, evaluation_end=end, forward_net_ev=net_ev, **over)


def test_a_generation_starts_unevaluated() -> None:
    """The honest default, and where most generations should stay."""
    g = _gen("G-1")

    assert g.status is GenerationStatus.UNEVALUATED
    assert not g.has_forward_evidence


def test_a_better_backtest_cannot_mark_a_generation_improved() -> None:
    """The entire point of the object.

    A generation with no forward window cannot be promoted however good its
    historical numbers look.
    """
    parent = _measured("G-1", 0.001, _T0, _T1)
    child = _gen("G-2", parent_generation_id="G-1")

    with pytest.raises(GenerationEvidenceError, match="not improved because its backtest is"):
        child.mark_improved(reason="backtest looks better", against=parent)

    assert child.status is GenerationStatus.UNEVALUATED


def test_forward_evidence_on_a_later_window_can_mark_it_improved() -> None:
    parent = _measured("G-1", 0.001, _T0, _T1)
    child = _measured("G-2", 0.004, _T2, _T3, parent_generation_id="G-1")

    child.mark_improved(reason="forward net EV 0.004 vs parent 0.001 over Q3", against=parent)

    assert child.status is GenerationStatus.IMPROVED
    assert "0.004" in child.status_reason


def test_a_favourable_baseline_cannot_be_shopped_for() -> None:
    """Comparing against anything but the declared parent is baseline shopping."""
    declared_parent = _measured("G-1", 0.009, _T0, _T1)
    weaker_other = _measured("G-0", 0.000, _T0, _T1)
    child = _measured("G-2", 0.004, _T2, _T3, parent_generation_id="G-1")

    with pytest.raises(GenerationEvidenceError, match="shopping for a favourable baseline"):
        child.mark_improved(reason="better than G-0", against=weaker_other)

    with pytest.raises(GenerationEvidenceError, match="not above parent"):
        child.mark_improved(reason="…", against=declared_parent)


def test_better_than_something_unmeasured_is_not_a_comparison() -> None:
    parent = _gen("G-1")
    child = _measured("G-2", 0.004, _T2, _T3, parent_generation_id="G-1")

    with pytest.raises(GenerationEvidenceError, match="not a comparison"):
        child.mark_improved(reason="parent never measured", against=parent)


def test_the_same_evaluation_window_is_a_backtest_in_disguise() -> None:
    """Two configurations judged over identical data is not forward evidence."""
    parent = _measured("G-1", 0.001, _T0, _T1)
    child = _measured("G-2", 0.004, _T0, _T1, parent_generation_id="G-1")

    with pytest.raises(GenerationEvidenceError, match="backtest comparison"):
        child.mark_improved(reason="higher net EV", against=parent)


def test_a_half_open_evaluation_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        _gen("G-1", evaluation_start=_T0)


def test_any_verdict_must_state_its_basis() -> None:
    """A verdict with no stated basis cannot be disagreed with."""
    with pytest.raises(ValidationError, match="requires status_reason"):
        _gen("G-1", status=GenerationStatus.NOT_IMPROVED)

    assert (
        _gen("G-1", status=GenerationStatus.NOT_IMPROVED, status_reason="forward EV fell").status
        is GenerationStatus.NOT_IMPROVED
    )


def test_inconclusive_is_recordable() -> None:
    """Without it, 'we looked and could not tell' has nowhere to go -- and
    tends to become a quiet IMPROVED."""
    g = _gen("G-1", status=GenerationStatus.INCONCLUSIVE, status_reason="effective sample 3")

    assert g.status is GenerationStatus.INCONCLUSIVE


def test_a_window_without_an_outcome_is_not_forward_evidence() -> None:
    """An evaluation that was started is not one that concluded."""
    g = _gen("G-1", evaluation_start=_T0, evaluation_end=_T1)

    assert not g.has_forward_evidence
