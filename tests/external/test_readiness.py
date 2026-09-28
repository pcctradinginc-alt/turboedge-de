"""Tests for the readiness engine.

The rules worth protecting are the ones that stop a dataset looking usable
before it is: quality before quantity, forward-only series counting only
their own forward archive, and thresholds that cannot be edited in place.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from turboedge.external.readiness import (
    DEFAULT_PROFILES,
    READINESS_LEVEL,
    DataReadinessRecord,
    ReadinessPolicy,
    ReadinessState,
    RegimeCoverage,
    SeriesEvidence,
    evaluate_readiness,
)
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass

_NOW = datetime(2026, 9, 28, 6, 0, tzinfo=UTC)
_POLICY = ReadinessPolicy(policy_version="test-1")


def _evidence(**over: object) -> SeriesEvidence:
    defaults: dict[str, object] = dict(
        source="ecb",
        series_id="ECB.ESTR",
        frequency="daily",
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        nominal_n=400,
        independent_dates=400,
        effective_n=400.0,
        first_observation=date(2025, 1, 1),
        last_observation=date(2026, 9, 27),
        schema_valid=True,
        pit_integrity=True,
        provenance_complete=True,
        missingness=0.0,
    )
    defaults.update(over)
    return SeriesEvidence(**defaults)  # type: ignore[arg-type]


def test_a_complete_daily_series_reaches_confirmation_ready() -> None:
    record = evaluate_readiness(_evidence(), _POLICY, now=_NOW)

    assert record.state is ReadinessState.CONFIRMATION_READY
    assert record.blocking_reasons == ()
    assert record.research_usable_from == date(2025, 1, 1)


def test_schema_failure_blocks_before_anything_else_is_considered() -> None:
    # Deliberately gives a huge sample: quality must be judged first, so a
    # broken series never reports as "nearly ready, just needs more data".
    record = evaluate_readiness(
        _evidence(
            schema_valid=False, nominal_n=10_000, independent_dates=10_000, effective_n=10_000.0
        ),
        _POLICY,
        now=_NOW,
    )

    assert record.state is ReadinessState.BLOCKED
    assert any("schema" in r for r in record.blocking_reasons)


def test_an_unresolved_parser_warning_blocks() -> None:
    record = evaluate_readiness(
        _evidence(unresolved_warnings=("unexpected column FOO",)), _POLICY, now=_NOW
    )

    assert record.state is ReadinessState.BLOCKED


def test_pit_integrity_failure_stops_at_schema_validated() -> None:
    record = evaluate_readiness(_evidence(pit_integrity=False), _POLICY, now=_NOW)

    assert record.state is ReadinessState.SCHEMA_VALIDATED
    assert record.research_usable_from is None


def test_a_stale_series_is_degraded_not_ready() -> None:
    record = evaluate_readiness(_evidence(last_observation=date(2026, 1, 1)), _POLICY, now=_NOW)

    assert record.state is ReadinessState.DEGRADED
    assert record.freshness_days == 270


def test_excessive_missingness_stops_at_pit_validated() -> None:
    record = evaluate_readiness(_evidence(missingness=0.4), _POLICY, now=_NOW)

    assert record.state is ReadinessState.PIT_VALIDATED
    assert any("missingness" in r for r in record.blocking_reasons)


def test_insufficient_effective_sample_blocks_even_with_many_rows() -> None:
    # 10,000 rows, 40 distinct dates. Row count must not decide this.
    record = evaluate_readiness(
        _evidence(nominal_n=10_000, independent_dates=40, effective_n=40.0),
        _POLICY,
        now=_NOW,
    )

    assert record.state is ReadinessState.PIT_VALIDATED
    assert any("effective sample" in r for r in record.blocking_reasons)


def test_insufficient_calendar_span_blocks_even_with_enough_observations() -> None:
    record = evaluate_readiness(
        _evidence(first_observation=date(2026, 6, 1), last_observation=date(2026, 9, 27)),
        _POLICY,
        now=_NOW,
    )

    assert record.state is ReadinessState.PIT_VALIDATED
    assert any("calendar span" in r for r in record.blocking_reasons)


def test_forward_only_series_ignores_the_publishers_history() -> None:
    # A decade of backfilled history and almost no self-archived data. The
    # publisher revises silently, so its history is not evidence this system
    # ever observed -- using it would be the fake retrospective backtest.
    record = evaluate_readiness(
        _evidence(
            backfill_class=BackfillClass.FORWARD_ONLY,
            availability_precision=AvailabilityPrecision.UNKNOWN,
            nominal_n=4000,
            independent_dates=4000,
            effective_n=4000.0,
            first_observation=date(2015, 1, 1),
            forward_n=5,
            forward_effective_n=5.0,
            forward_first_observation=date(2026, 9, 23),
        ),
        _POLICY,
        now=_NOW,
    )

    assert record.state is ReadinessState.PIT_VALIDATED
    assert any("forward-archived" in r for r in record.blocking_reasons)


def test_forward_only_series_becomes_ready_on_its_own_archive() -> None:
    record = evaluate_readiness(
        _evidence(
            backfill_class=BackfillClass.FORWARD_ONLY,
            nominal_n=400,
            independent_dates=400,
            effective_n=400.0,
            first_observation=date(2015, 1, 1),
            forward_n=300,
            forward_effective_n=300.0,
            forward_first_observation=date(2025, 9, 1),
        ),
        _POLICY,
        now=_NOW,
    )

    assert READINESS_LEVEL[record.state] >= 1
    # Research may only start where TurboEdge's own archive starts.
    assert record.research_usable_from == date(2025, 9, 1)


def test_unknown_availability_precision_cannot_reach_confirmation() -> None:
    record = evaluate_readiness(
        _evidence(availability_precision=AvailabilityPrecision.UNKNOWN),
        _POLICY,
        now=_NOW,
    )

    assert record.state is ReadinessState.VALIDATION_READY


def test_inferred_precision_also_cannot_reach_confirmation() -> None:
    record = evaluate_readiness(
        _evidence(availability_precision=AvailabilityPrecision.INFERRED),
        _POLICY,
        now=_NOW,
    )

    assert record.state is ReadinessState.VALIDATION_READY


def test_forward_maturity_needs_both_span_and_forward_sample() -> None:
    record = evaluate_readiness(
        _evidence(
            forward_n=200,
            forward_effective_n=200.0,
            forward_first_observation=date(2025, 1, 1),
        ),
        _POLICY,
        now=_NOW,
    )

    assert record.state is ReadinessState.FORWARD_MATURE


def test_a_disabled_series_is_disabled_whatever_its_data() -> None:
    record = evaluate_readiness(_evidence(), _POLICY, now=_NOW, enabled=False)

    assert record.state is ReadinessState.DISABLED
    assert record.level == -1


def test_evaluation_is_deterministic() -> None:
    evidence = _evidence()
    a = evaluate_readiness(evidence, _POLICY, now=_NOW)
    b = evaluate_readiness(evidence, _POLICY, now=_NOW)

    assert a == b


def test_first_reached_at_is_preserved_across_re_evaluations() -> None:
    first = evaluate_readiness(_evidence(), _POLICY, now=_NOW)
    later = evaluate_readiness(
        _evidence(), _POLICY, now=datetime(2026, 9, 29, 6, 0, tzinfo=UTC), previous=first
    )

    assert later.first_reached_at == first.first_reached_at == _NOW


def test_policy_is_frozen() -> None:
    # A threshold that can be edited in place is a threshold that can be
    # lowered the week an alpha needs it (GOVERNANCE.md §11, spec §55).
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        _POLICY.max_missingness = 0.9  # type: ignore[misc]
    with pytest.raises(Exception):  # noqa: B017
        DEFAULT_PROFILES["daily"].min_exploratory_observations = 1  # type: ignore[misc]


def test_policy_refuses_an_unknown_frequency_rather_than_substituting() -> None:
    with pytest.raises(KeyError, match="no readiness profile"):
        _POLICY.profile_for("fortnightly")


def test_a_ready_record_cannot_carry_blocking_reasons() -> None:
    with pytest.raises(ValueError, match="cannot carry blocking"):
        DataReadinessRecord(
            source="s",
            series_id="x",
            state=ReadinessState.EXPLORATORY_READY,
            evaluated_at=_NOW,
            nominal_n=1,
            effective_n=1.0,
            independent_dates=1,
            calendar_span_days=1,
            availability_precision=AvailabilityPrecision.EXACT_DATE,
            completeness=1.0,
            blocking_reasons=("something",),
            policy_version="1",
        )


def test_effective_sample_may_not_exceed_nominal() -> None:
    with pytest.raises(ValueError, match="exceeds nominal_n"):
        _evidence(nominal_n=10, effective_n=11.0)


def test_regime_coverage_is_metadata_not_a_gate() -> None:
    record = evaluate_readiness(
        _evidence(regime_coverage=RegimeCoverage.LIMITED), _POLICY, now=_NOW
    )

    assert record.regime_coverage is RegimeCoverage.LIMITED
    assert READINESS_LEVEL[record.state] >= 1
