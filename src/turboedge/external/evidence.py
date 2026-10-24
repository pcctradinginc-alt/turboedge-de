"""Measuring what a series actually contains, before any policy is applied.

Kept separate from `readiness.py` on purpose: this module measures, that one
decides. The split is what allows a stored evaluation to be replayed under a
new policy version without refetching a byte, and it keeps the thresholds in
one reviewable place instead of scattered through counting code.

The counting rule that matters is spec §36: **effective sample, not row
count.** This repository has already been burned by the alternative. The
weekly tournament reported three significant signal families at n = 60,774
until the denominator was recomputed properly and the same data gave 12,324
effective observations and nothing significant (`docs/measured_results.md`
§6.12). A macro series is worse, not better: twenty daily rows of a monthly
indicator that only moves once a month are one observation repeated twenty
times, and counting them as twenty is how a dataset looks ready a year
before it is.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import cast

from turboedge.external.readiness import (
    EXPECTED_PERIOD_DAYS,
    RegimeCoverage,
    SeriesEvidence,
)
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec
from turboedge.storage.schemas import ExternalObservation


@dataclass(frozen=True, slots=True)
class ExternalSeriesAggregate:
    """Everything `build_evidence` measures, as database aggregates.

    Computed by `Store.external_series_aggregate` in one scan. Exists so
    the readiness engine never has to materialise a series: it needs
    counts, a span, a weakest precision and two integrity flags, and all of
    those are aggregates. Measured 2026-10-23, materialising them instead
    is what pushed the external ingest past its 90-minute budget.

    `precisions` is the set over the latest vintage of each period, so the
    weakest one can be chosen without loading a row.
    """

    nominal_n: int
    independent_dates: int
    first_observation: date | None
    last_observation: date | None
    pit_violations: int
    provenance_gaps: int
    forward_dates: int
    forward_first_observation: date | None
    precisions: frozenset[str | None]


def evidence_from_aggregate(
    spec: SeriesSpec,
    aggregate: ExternalSeriesAggregate,
    *,
    collection_start: datetime | None = None,
    schema_valid: bool = True,
    parser_warnings: Sequence[str] = (),
    regime_coverage: RegimeCoverage = RegimeCoverage.LIMITED,
) -> SeriesEvidence:
    """`build_evidence` without the rows.

    Must return exactly what `build_evidence` returns for the same data.
    `tests/external/test_evidence_aggregate.py` asserts that over
    randomised series including revisions, intraday resolution, absent
    precisions and point-in-time violations -- two paths that could
    silently diverge are only safe while the equivalence is enforced.

    Deliberately does not take `event_dates`. A caller with an explicit
    event calendar has the rows anyway and should use `build_evidence`; for
    an `event_driven` series without one, the events *are* the observation
    dates, which the aggregate already has.
    """
    warnings = list(parser_warnings)
    has_rows = aggregate.nominal_n > 0

    if spec.event_driven:
        missingness = 0.0
    else:
        # Missingness needs the span and the count, not the dates.
        missingness = _missingness_from_span(
            first=aggregate.first_observation,
            last=aggregate.last_observation,
            observed=aggregate.independent_dates,
            frequency=spec.frequency,
        )

    effective_n = float(aggregate.independent_dates)
    event_count = 0
    independent_event_count = 0
    if spec.event_driven:
        # The observations are the events.
        event_count = aggregate.independent_dates
        independent_event_count = aggregate.independent_dates

    pit_integrity = spec.forecast_series or aggregate.pit_violations == 0
    if not pit_integrity:
        warnings.append(
            f"{spec.series_id}: {aggregate.pit_violations} observation(s) have an "
            "available_at that precedes their observation_time"
        )

    return SeriesEvidence(
        source=spec.source,
        series_id=spec.series_id,
        frequency="event" if spec.event_driven else spec.frequency,
        backfill_class=spec.backfill_class,
        availability_precision=_weakest_precision(
            {p for p in aggregate.precisions if p is not None}
            | ({str(AvailabilityPrecision.UNKNOWN)} if None in aggregate.precisions else set())
        ),
        nominal_n=aggregate.nominal_n,
        independent_dates=aggregate.independent_dates,
        effective_n=effective_n,
        first_observation=aggregate.first_observation,
        last_observation=aggregate.last_observation,
        collection_start=collection_start,
        forward_n=aggregate.forward_dates,
        forward_effective_n=float(aggregate.forward_dates),
        forward_first_observation=aggregate.forward_first_observation,
        event_count=event_count,
        independent_event_count=independent_event_count,
        schema_valid=schema_valid and has_rows,
        pit_integrity=pit_integrity and has_rows,
        provenance_complete=aggregate.provenance_gaps == 0 and has_rows,
        missingness=missingness,
        unresolved_warnings=tuple(warnings),
        regime_coverage=regime_coverage,
        release_lag_days=(spec.conservative_release_lag_hours or 0.0) / 24.0,
    )


def _missingness_from_span(
    *, first: date | None, last: date | None, observed: int, frequency: str
) -> float:
    """`estimate_missingness` expressed over a span plus a count.

    The row-based version takes the list of unique periods and uses only
    its min, its max and its length -- so the same number falls out of the
    aggregate without the list.
    """
    if first is None or last is None or observed < 2:
        return 0.0
    expected_period = EXPECTED_PERIOD_DAYS.get(frequency)
    if expected_period is None:
        return 0.0
    span_days = (last - first).days
    expected = span_days / expected_period + 1.0
    if expected <= 0:
        return 0.0
    return min(1.0, max(0.0, (expected - observed) / expected))


def latest_vintage_per_period(
    observations: Sequence[ExternalObservation],
) -> list[ExternalObservation]:
    """One row per observation period: the newest vintage of each.

    Revisions are kept in storage forever (that is the point of the
    `available_at` key), but counting them as separate evidence would let a
    heavily revised series look better-sampled than a stable one. Three
    prints of one month are one month of evidence.
    """
    newest: dict[datetime, ExternalObservation] = {}
    for obs in observations:
        current = newest.get(obs.observation_time)
        if current is None or obs.available_at > current.available_at:
            newest[obs.observation_time] = obs
    return sorted(newest.values(), key=lambda o: o.observation_time)


def count_revisions(observations: Sequence[ExternalObservation]) -> int:
    """How many rows are revisions rather than first prints."""
    per_period = Counter(o.observation_time for o in observations)
    return sum(max(0, n - 1) for n in per_period.values())


def estimate_missingness(
    periods: Sequence[date], *, frequency: str
) -> tuple[float, tuple[str, ...]]:
    """Fraction of expected observations absent from the observed span.

    Measured against the series' own span, not against an ideal start date:
    the question is whether the data have holes, not whether history goes
    back far enough. Span sufficiency is a separate check, and conflating
    the two would report a young, complete series as incomplete.
    """
    if len(periods) < 2:
        return 0.0, ("too few observations to estimate missingness",)
    expected_period = EXPECTED_PERIOD_DAYS.get(frequency)
    if expected_period is None:
        return 0.0, (f"no expected cadence known for frequency {frequency!r}",)

    span_days = (max(periods) - min(periods)).days
    expected = span_days / expected_period + 1.0
    if expected <= 0:
        return 0.0, ()
    missing = max(0.0, (expected - len(periods)) / expected)
    return min(1.0, missing), ()


def build_evidence(
    spec: SeriesSpec,
    observations: Sequence[ExternalObservation],
    *,
    collection_start: datetime | None = None,
    schema_valid: bool = True,
    parser_warnings: Sequence[str] = (),
    event_dates: Sequence[date] | None = None,
    regime_coverage: RegimeCoverage = RegimeCoverage.LIMITED,
) -> SeriesEvidence:
    """Measure one series from its stored observations.

    `collection_start` is when TurboEdge itself began archiving this series.
    Everything observed at or after it counts as forward evidence -- evidence
    the system genuinely accumulated rather than inherited from a publisher's
    revised history. For a FORWARD_ONLY series that distinction is the whole
    difference between real and imaginary out-of-sample data.
    """
    deduped = latest_vintage_per_period(observations)
    periods = [o.observation_time.astimezone(UTC).date() for o in deduped]
    unique_periods = sorted(set(periods))

    warnings = list(parser_warnings)
    if spec.event_driven and event_dates is None:
        # For an event series the observations *are* the events: the
        # publisher emits a value when something happens and is silent
        # otherwise, so silence is the data, not a gap in it.
        event_dates = unique_periods
    if spec.event_driven:
        missingness, notes = 0.0, cast("tuple[str, ...]", ())
    else:
        missingness, notes = estimate_missingness(unique_periods, frequency=spec.frequency)
    # A cadence we cannot model is a gap in our knowledge, not in the data;
    # it must not masquerade as a parser warning that blocks readiness.
    _ = notes

    # Effective sample is the number of distinct periods, never the row
    # count. For an event-driven series it is the number of independent
    # events, which can be far smaller still: 500 daily rows containing
    # eight ECB decisions are closer to 8 than to 500 (spec §37).
    independent_dates = len(unique_periods)
    effective_n = float(independent_dates)
    event_count = 0
    independent_event_count = 0
    if event_dates is not None:
        event_count = len(event_dates)
        independent_event_count = len(set(event_dates))
        effective_n = float(min(effective_n, independent_event_count))

    # Forward evidence is an observation whose *period* began at or after
    # TurboEdge started collecting -- not one that happened to be downloaded
    # after that moment. The first fetch of a source pulls its entire
    # history in one request, so `retrieved_at >= collection_start` would
    # classify twenty years of backfill as forward evidence and hand a
    # FORWARD_ONLY series instant maturity it has not earned. That
    # distinction is the whole reason FORWARD_ONLY exists.
    forward_periods = sorted(
        {
            o.observation_time.astimezone(UTC).date()
            for o in deduped
            if collection_start is not None and o.observation_time >= collection_start
        }
    )

    # PIT integrity is a property of the data, checked here rather than
    # asserted by the adapter: every row must carry an availability that is
    # not before the period it describes, and must declare how well that
    # availability is known.
    # Precision is collected over EVERY row before the point-in-time check
    # runs, rather than inside the same loop. That loop `break`s on the
    # first violation, so collecting there made the reported precision
    # depend on how far the iteration had got: the same data could yield a
    # different `availability_precision` depending on row order. A series is
    # only as honest as its weakest row, wherever a violation happens to sit.
    precisions: set[str] = {
        obs.availability_precision or AvailabilityPrecision.UNKNOWN for obs in deduped
    }

    pit_integrity = True
    for obs in deduped:
        # For ordinary data, holding a value before the period it describes
        # is look-ahead and the ordering is an invariant worth enforcing.
        #
        # A forecast has no such invariant in either direction. It is
        # published before its target period, so available_at precedes
        # observation_time by design; and one issue covers periods that
        # have already elapsed as well (the day-ahead load forecast fetched
        # this afternoon still carries this morning's intervals), so
        # available_at follows observation_time for those. Neither is a
        # defect. What protects a forecast study from look-ahead is the
        # read-time filter `available_at <= prediction_time`, which is
        # correct in both directions and needs no help here.
        if not spec.forecast_series and obs.available_at < obs.observation_time:
            pit_integrity = False
            warnings.append(
                f"{obs.series_id}: available_at {obs.available_at.isoformat()} precedes "
                f"observation_time {obs.observation_time.isoformat()}"
            )
            break

    # The weakest precision present governs the series. One UNKNOWN row in an
    # otherwise exact series makes the series unsuitable for strict work,
    # because nothing downstream filters row-by-row.
    precision = _weakest_precision(precisions)

    provenance_complete = all(
        obs.parser_version and obs.source and obs.source_version for obs in deduped
    )

    return SeriesEvidence(
        source=spec.source,
        series_id=spec.series_id,
        frequency="event" if spec.event_driven else spec.frequency,
        backfill_class=spec.backfill_class,
        availability_precision=precision,
        nominal_n=len(deduped),
        independent_dates=independent_dates,
        effective_n=effective_n,
        first_observation=unique_periods[0] if unique_periods else None,
        last_observation=unique_periods[-1] if unique_periods else None,
        collection_start=collection_start,
        forward_n=len(forward_periods),
        forward_effective_n=float(len(forward_periods)),
        forward_first_observation=forward_periods[0] if forward_periods else None,
        event_count=event_count,
        independent_event_count=independent_event_count,
        schema_valid=schema_valid and bool(deduped),
        pit_integrity=pit_integrity and bool(deduped),
        provenance_complete=provenance_complete and bool(deduped),
        missingness=missingness,
        unresolved_warnings=tuple(warnings),
        regime_coverage=regime_coverage,
        release_lag_days=(spec.conservative_release_lag_hours or 0.0) / 24.0,
    )


#: Weakest first. A series is only as point-in-time honest as its worst row.
_PRECISION_ORDER: tuple[AvailabilityPrecision, ...] = (
    AvailabilityPrecision.UNKNOWN,
    AvailabilityPrecision.INFERRED,
    AvailabilityPrecision.CONSERVATIVE_DATE,
    AvailabilityPrecision.EXACT_DATE,
    AvailabilityPrecision.EXACT_TIMESTAMP,
)


def _weakest_precision(values: set[str]) -> AvailabilityPrecision:
    if not values:
        return AvailabilityPrecision.UNKNOWN
    for candidate in _PRECISION_ORDER:
        if str(candidate) in values:
            return candidate
    # An unrecognised precision string is not a reason to assume the best.
    return AvailabilityPrecision.UNKNOWN


def is_backfill_pit_safe(backfill_class: BackfillClass) -> bool:
    """Whether a class's *history* may serve as strict point-in-time evidence."""
    return backfill_class is BackfillClass.HISTORICAL_PIT_SAFE
