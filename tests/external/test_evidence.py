"""Tests for evidence measurement -- effective sample, not row count."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from turboedge.external.evidence import (
    build_evidence,
    count_revisions,
    estimate_missingness,
    latest_vintage_per_period,
)
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SPEC = SeriesSpec(
    source="ecb",
    series_id="ECB.ESTR",
    name="n",
    category="c",
    unit="percent",
    frequency="daily",
    native_identifier="EST/B.EU000A2X2A25.WT",
    availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
    backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
    conservative_release_lag_hours=36,
)


def _obs(day: date, *, value: float = 1.0, vintage_offset_days: int = 1, **over: object):
    observation_time = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    defaults: dict[str, object] = dict(
        series_id="ECB.ESTR",
        value=value,
        unit="percent",
        frequency="daily",
        source_version="v1",
        observation_time=observation_time,
        available_at=observation_time + timedelta(days=vintage_offset_days),
        retrieved_at=observation_time + timedelta(days=vintage_offset_days),
        source="ecb",
        parser_version="1",
        quality_score=1.0,
        availability_precision=str(AvailabilityPrecision.CONSERVATIVE_DATE),
    )
    defaults.update(over)
    return ExternalObservation(**defaults)  # type: ignore[arg-type]


def test_many_rows_of_one_period_count_as_one_observation() -> None:
    # Three vintages of the same day. Counting them as three would make a
    # heavily revised series look better sampled than a stable one.
    rows = [_obs(date(2026, 1, 5), vintage_offset_days=d) for d in (1, 2, 3)]

    evidence = build_evidence(_SPEC, rows)

    assert evidence.nominal_n == 1
    assert evidence.effective_n == 1.0
    assert count_revisions(rows) == 2


def test_the_newest_vintage_wins() -> None:
    rows = [
        _obs(date(2026, 1, 5), value=1.0, vintage_offset_days=1),
        _obs(date(2026, 1, 5), value=2.0, vintage_offset_days=5),
    ]

    kept = latest_vintage_per_period(rows)

    assert [o.value for o in kept] == [2.0]


def test_event_driven_data_count_events_not_days() -> None:
    # 500 daily rows containing eight decision events is closer to 8 than 500.
    rows = [_obs(date(2026, 1, 1) + timedelta(days=i)) for i in range(500)]
    events = [date(2026, 1, 1) + timedelta(days=45 * i) for i in range(8)]

    evidence = build_evidence(_SPEC, rows, event_dates=events)

    assert evidence.independent_dates == 500
    assert evidence.effective_n == 8.0
    assert evidence.independent_event_count == 8


def test_availability_before_the_observation_fails_pit_integrity() -> None:
    rows = [_obs(date(2026, 1, 5), vintage_offset_days=-1)]

    evidence = build_evidence(_SPEC, rows)

    assert not evidence.pit_integrity
    assert any("precedes" in w for w in evidence.unresolved_warnings)


def test_the_weakest_precision_in_the_series_governs() -> None:
    rows = [
        _obs(date(2026, 1, 5), availability_precision=str(AvailabilityPrecision.EXACT_TIMESTAMP)),
        _obs(date(2026, 1, 6), availability_precision=str(AvailabilityPrecision.UNKNOWN)),
    ]

    evidence = build_evidence(_SPEC, rows)

    assert evidence.availability_precision is AvailabilityPrecision.UNKNOWN


def test_a_row_without_a_recorded_precision_is_unknown_not_fine() -> None:
    # The 93,851 pre-existing Cboe/CFTC rows are exactly this case.
    rows = [_obs(date(2026, 1, 5), availability_precision=None)]

    evidence = build_evidence(_SPEC, rows)

    assert evidence.availability_precision is AvailabilityPrecision.UNKNOWN


def test_forward_evidence_counts_only_what_was_archived_after_collection_start() -> None:
    old = [_obs(date(2025, 1, 1) + timedelta(days=i)) for i in range(10)]
    new = [_obs(date(2026, 9, 1) + timedelta(days=i)) for i in range(5)]

    evidence = build_evidence(_SPEC, old + new, collection_start=datetime(2026, 8, 1, tzinfo=UTC))

    assert evidence.nominal_n == 15
    assert evidence.forward_n == 5
    assert evidence.forward_first_observation == date(2026, 9, 1)


def test_missingness_is_measured_against_the_series_own_span() -> None:
    # Ten consecutive days: complete, even though the series is young.
    complete = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]
    holed = [date(2026, 1, 1) + timedelta(days=2 * i) for i in range(5)]

    assert estimate_missingness(complete, frequency="daily")[0] == 0.0
    assert estimate_missingness(holed, frequency="daily")[0] > 0.4


def test_unknown_cadence_reports_zero_missingness_and_says_so() -> None:
    value, notes = estimate_missingness(
        [date(2026, 1, 1), date(2026, 1, 2)], frequency="fortnightly"
    )

    assert value == 0.0
    assert notes and "fortnightly" in notes[0]


def test_an_empty_series_is_not_schema_valid() -> None:
    evidence = build_evidence(_SPEC, [])

    assert not evidence.schema_valid
    assert not evidence.pit_integrity
    assert evidence.effective_n == 0.0


def test_backfilled_history_is_not_forward_evidence() -> None:
    # The first fetch pulls the whole history in one request, so every row
    # is retrieved "now". Counting that as forward evidence would hand a
    # FORWARD_ONLY series instant maturity it has not earned -- which is
    # precisely the fake retrospective backtest the class exists to prevent.
    collection_start = datetime(2026, 9, 28, tzinfo=UTC)
    history = [
        _obs(date(2015, 1, 1) + timedelta(days=30 * i), vintage_offset_days=1) for i in range(120)
    ]
    for obs in history:
        object.__setattr__(obs, "retrieved_at", collection_start)

    evidence = build_evidence(_SPEC, history, collection_start=collection_start)

    assert evidence.nominal_n == 120
    assert evidence.forward_n == 0
    assert evidence.forward_first_observation is None


def test_an_observation_after_collection_start_is_forward_evidence() -> None:
    collection_start = datetime(2026, 9, 1, tzinfo=UTC)
    rows = [
        _obs(date(2026, 8, 30)),  # before: backfill
        _obs(date(2026, 9, 5)),  # after: genuinely accumulated
        _obs(date(2026, 9, 6)),
    ]

    evidence = build_evidence(_SPEC, rows, collection_start=collection_start)

    assert evidence.forward_n == 2
    assert evidence.forward_first_observation == date(2026, 9, 5)


def test_the_staleness_allowance_grows_with_the_publication_lag() -> None:
    # German industrial production for July is 89 days old in late
    # September and publishing exactly on schedule.
    from turboedge.external.readiness import ReadinessPolicy

    policy = ReadinessPolicy(policy_version="t")
    assert policy.staleness_limit_days("daily") == 45
    assert policy.staleness_limit_days("monthly") == 76
    assert policy.staleness_limit_days("monthly", release_lag_days=42.0) == 118
    assert policy.staleness_limit_days("unknown-cadence", release_lag_days=99.0) == 45


def test_an_event_series_is_complete_not_incomplete() -> None:
    # The ECB publishes the policy rate only when it changes: 48 values
    # since 1999. Measured on the first live run, treating that as a
    # business-daily series reported missingness 0.993 -- "99.3% of the
    # data is absent" about a series that is entirely present.
    spec = _SPEC.model_copy(update={"event_driven": True, "frequency": "business_daily"})
    decisions = [
        _obs(date(2020, 1, 23)),
        _obs(date(2022, 7, 27)),
        _obs(date(2023, 9, 20)),
        _obs(date(2025, 6, 11)),
    ]

    evidence = build_evidence(spec, decisions)

    assert evidence.missingness == 0.0
    assert evidence.frequency == "event"
    assert evidence.effective_n == 4.0
    assert evidence.independent_event_count == 4


def test_an_event_series_is_judged_against_the_event_profile() -> None:
    from turboedge.external.readiness import ReadinessPolicy, ReadinessState, evaluate_readiness

    spec = _SPEC.model_copy(update={"event_driven": True})
    # Four decisions is complete data and nowhere near enough evidence.
    thin = build_evidence(spec, [_obs(date(2020 + i, 1, 23)) for i in range(4)])
    record = evaluate_readiness(
        thin, ReadinessPolicy(policy_version="t"), now=datetime(2026, 9, 28, tzinfo=UTC)
    )

    assert record.state is ReadinessState.PIT_VALIDATED
    assert any("effective sample" in r for r in record.blocking_reasons)
    assert not any("missingness" in r for r in record.blocking_reasons)
