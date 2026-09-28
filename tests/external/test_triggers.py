"""Tests for research triggers -- above all, "exactly once"."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from turboedge.external.readiness import (
    DataReadinessRecord,
    ReadinessState,
)
from turboedge.external.schemas import AvailabilityPrecision
from turboedge.external.triggers import (
    ResearchTrigger,
    TriggerType,
    triggers_for_transition,
)

_NOW = datetime(2026, 9, 28, 6, 0, tzinfo=UTC)


def _record(state: ReadinessState, **over: object) -> DataReadinessRecord:
    defaults: dict[str, object] = dict(
        source="ecb",
        series_id="ECB.ESTR",
        state=state,
        evaluated_at=_NOW,
        nominal_n=400,
        effective_n=400.0,
        independent_dates=400,
        calendar_span_days=600,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        completeness=1.0,
        policy_version="1",
    )
    defaults.update(over)
    return DataReadinessRecord(**defaults)  # type: ignore[arg-type]


def test_reaching_exploratory_emits_one_trigger() -> None:
    out = triggers_for_transition(_record(ReadinessState.EXPLORATORY_READY))

    assert [t.trigger_type for t in out] == [TriggerType.EXPLORATORY_DATA_READY]


def test_the_same_state_tomorrow_emits_nothing() -> None:
    first = triggers_for_transition(_record(ReadinessState.EXPLORATORY_READY))
    emitted = frozenset(t.dedup_key for t in first)

    again = triggers_for_transition(
        _record(
            ReadinessState.EXPLORATORY_READY, evaluated_at=datetime(2026, 9, 29, 6, 0, tzinfo=UTC)
        ),
        already_emitted=emitted,
    )

    assert again == []


def test_a_jump_straight_to_confirmation_earns_every_milestone_passed() -> None:
    # A PIT-safe backfill with deep history arrives all at once. Nothing
    # should be lost just because the data did not trickle in.
    out = triggers_for_transition(_record(ReadinessState.CONFIRMATION_READY))

    assert [t.trigger_type for t in out] == [
        TriggerType.EXPLORATORY_DATA_READY,
        TriggerType.VALIDATION_DATA_READY,
        TriggerType.CONFIRMATION_DATA_READY,
    ]


def test_advancing_a_level_emits_only_the_new_milestone() -> None:
    earlier = triggers_for_transition(_record(ReadinessState.EXPLORATORY_READY))
    emitted = frozenset(t.dedup_key for t in earlier)

    out = triggers_for_transition(_record(ReadinessState.VALIDATION_READY), already_emitted=emitted)

    assert [t.trigger_type for t in out] == [TriggerType.VALIDATION_DATA_READY]


def test_a_not_ready_series_emits_nothing() -> None:
    for state in (
        ReadinessState.COLLECTING,
        ReadinessState.PIT_VALIDATED,
        ReadinessState.DEGRADED,
        ReadinessState.BLOCKED,
        ReadinessState.DISABLED,
    ):
        assert triggers_for_transition(_record(state)) == []


def test_a_recovery_after_degradation_does_not_re_emit() -> None:
    first = triggers_for_transition(_record(ReadinessState.EXPLORATORY_READY))
    emitted = frozenset(t.dedup_key for t in first)
    # degraded, then healthy again
    assert triggers_for_transition(_record(ReadinessState.DEGRADED), already_emitted=emitted) == []
    recovered = triggers_for_transition(
        _record(ReadinessState.EXPLORATORY_READY), already_emitted=emitted
    )

    assert recovered == []


def test_dedup_key_excludes_the_timestamp_but_includes_the_policy() -> None:
    a = triggers_for_transition(_record(ReadinessState.EXPLORATORY_READY))[0]
    b = triggers_for_transition(
        _record(ReadinessState.EXPLORATORY_READY, evaluated_at=datetime(2027, 1, 1, tzinfo=UTC))
    )[0]
    c = triggers_for_transition(_record(ReadinessState.EXPLORATORY_READY, policy_version="2"))[0]

    assert a.dedup_key == b.dedup_key
    assert a.dedup_key != c.dedup_key


def test_a_partial_handoff_is_refused() -> None:
    base = triggers_for_transition(_record(ReadinessState.EXPLORATORY_READY))[0]

    with pytest.raises(ValueError, match="handed_off_at and hypothesis_id"):
        ResearchTrigger(**{**base.model_dump(), "handed_off_at": _NOW})
    with pytest.raises(ValueError, match="handed_off_at and hypothesis_id"):
        ResearchTrigger(**{**base.model_dump(), "hypothesis_id": "H-1"})

    complete = ResearchTrigger(
        **{**base.model_dump(), "handed_off_at": _NOW, "hypothesis_id": "H-1"}
    )
    assert complete.hypothesis_id == "H-1"
