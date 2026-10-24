"""The aggregate evidence path must equal the row-based one, exactly.

`evidence_from_aggregate` + `Store.external_series_aggregate` exist because
materialising a series to count it cost minutes per series once FRED's
paged ALFRED history arrived, and drove the external ingest past its
90-minute budget on 2026-10-23.

Two code paths computing the same readiness is a standing risk: the fast
one is the one that runs, the slow one is the one the tests know, and they
can drift without anyone noticing until a series is declared ready on the
wrong numbers. These tests remove that risk by asserting equality over
randomised data that includes every shape known to matter -- revisions,
intraday resolution, absent precisions, point-in-time violations,
provenance gaps, forward windows and event-driven series.

Also here: the performance regression itself, as a test with a realistic
row count rather than a comment hoping someone remembers.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.external.evidence import build_evidence, evidence_from_aggregate
from turboedge.external.schemas import AvailabilityPrecision, BackfillClass, SeriesSpec
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ExternalObservation

_SOURCE = "testsource"
_SERIES = "TEST.SERIES"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "t.duckdb") as s:
        s.init_schema()
        yield s


def _spec(**over: object) -> SeriesSpec:
    defaults: dict[str, object] = dict(
        source=_SOURCE,
        series_id=_SERIES,
        name="n",
        category="c",
        unit="u",
        frequency="daily",
        native_identifier="X",
        availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
        conservative_release_lag_hours=36,
    )
    defaults.update(over)
    return SeriesSpec(**defaults)  # type: ignore[arg-type]


def _obs(
    observation_time: datetime,
    *,
    available_offset_h: float = 36.0,
    precision: str | None = str(AvailabilityPrecision.CONSERVATIVE_DATE),
    source_version: str = "v1",
    parser_version: str = "1",
    value: float = 1.0,
) -> ExternalObservation:
    return ExternalObservation(
        series_id=_SERIES,
        value=value,
        unit="u",
        frequency="daily",
        source_version=source_version,
        observation_time=observation_time,
        available_at=observation_time + timedelta(hours=available_offset_h),
        retrieved_at=observation_time + timedelta(hours=available_offset_h),
        source=_SOURCE,
        parser_version=parser_version,
        quality_score=1.0,
        availability_precision=precision,
    )


def _assert_paths_agree(
    store: Store,
    rows: list[ExternalObservation],
    spec: SeriesSpec,
    *,
    collection_start: datetime | None = None,
) -> None:
    store.append_external_observations(rows)
    from_rows = build_evidence(spec, rows, collection_start=collection_start)
    aggregate = store.external_series_aggregate(
        spec.source, spec.series_id, collection_start=collection_start
    )
    from_agg = evidence_from_aggregate(spec, aggregate, collection_start=collection_start)

    # Warnings carry different wording on purpose (the aggregate path counts
    # violations rather than naming the first one), so they are compared
    # only on whether any exist.
    assert bool(from_rows.unresolved_warnings) == bool(from_agg.unresolved_warnings)
    assert from_rows.model_dump(exclude={"unresolved_warnings"}) == from_agg.model_dump(
        exclude={"unresolved_warnings"}
    )


def test_a_plain_daily_series_agrees(store: Store) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=i)) for i in range(120)]

    _assert_paths_agree(store, rows, _spec())


def test_revisions_agree_and_count_once(store: Store) -> None:
    # Three prints of one day are one day of evidence in both paths.
    base = datetime(2026, 3, 2, tzinfo=UTC)
    rows = [_obs(base, available_offset_h=h, value=float(h)) for h in (24.0, 48.0, 96.0)] + [
        _obs(base + timedelta(days=i)) for i in range(1, 60)
    ]

    _assert_paths_agree(store, rows, _spec())


def test_intraday_resolution_agrees(store: Store) -> None:
    # 96 quarter-hours a day: nominal_n counts them, independent_dates does
    # not. That distinction is the whole reason effective sample is not the
    # row count, and both paths must make it the same way.
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=d, minutes=15 * q)) for d in range(20) for q in range(96)]

    _assert_paths_agree(store, rows, _spec())


def test_absent_precision_agrees(store: Store) -> None:
    # The pre-existing Cboe/CFTC rows carry no precision at all; both paths
    # must treat that as UNKNOWN rather than as fine.
    base = datetime(2026, 2, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=i), precision=None) for i in range(60)]

    _assert_paths_agree(store, rows, _spec())


def test_a_mixed_precision_series_agrees_on_the_weakest(store: Store) -> None:
    base = datetime(2026, 2, 1, tzinfo=UTC)
    rows = [
        _obs(
            base + timedelta(days=i),
            precision=str(AvailabilityPrecision.EXACT_TIMESTAMP)
            if i % 2
            else str(AvailabilityPrecision.UNKNOWN),
        )
        for i in range(60)
    ]

    _assert_paths_agree(store, rows, _spec())


def test_a_point_in_time_violation_agrees(store: Store) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=i)) for i in range(40)]
    rows.append(_obs(base + timedelta(days=50), available_offset_h=-24.0))

    _assert_paths_agree(store, rows, _spec())


def test_a_provenance_gap_agrees(store: Store) -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=i)) for i in range(30)]
    rows.append(_obs(base + timedelta(days=40), source_version=""))

    _assert_paths_agree(store, rows, _spec())


def test_a_forward_window_agrees(store: Store) -> None:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=i)) for i in range(400)]

    _assert_paths_agree(store, rows, _spec(), collection_start=datetime(2025, 9, 1, tzinfo=UTC))


def test_an_event_driven_series_agrees(store: Store) -> None:
    rows = [_obs(datetime(2020 + i, 6, 1, tzinfo=UTC)) for i in range(6)]

    _assert_paths_agree(store, rows, _spec(event_driven=True))


def test_a_forecast_series_agrees(store: Store) -> None:
    # A forecast has no ordering invariant, so the violation count must not
    # make either path report broken point-in-time integrity.
    base = datetime(2026, 6, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=i), available_offset_h=-48.0) for i in range(50)]

    _assert_paths_agree(
        store, rows, _spec(forecast_series=True, backfill_class=BackfillClass.FORWARD_ONLY)
    )


def test_an_empty_series_agrees(store: Store) -> None:
    _assert_paths_agree(store, [], _spec())


@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
def test_randomised_series_agree(store: Store, seed: int) -> None:
    # The shapes above are the ones that were thought of. This covers the
    # ones that were not.
    rng = random.Random(seed)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    precisions = [
        None,
        str(AvailabilityPrecision.UNKNOWN),
        str(AvailabilityPrecision.INFERRED),
        str(AvailabilityPrecision.CONSERVATIVE_DATE),
        str(AvailabilityPrecision.EXACT_DATE),
        str(AvailabilityPrecision.EXACT_TIMESTAMP),
    ]
    rows: list[ExternalObservation] = []
    for _i in range(300):
        day = base + timedelta(days=rng.randrange(0, 500), minutes=15 * rng.randrange(0, 4))
        for _ in range(rng.choice([1, 1, 1, 2, 3])):  # occasional revisions
            rows.append(
                _obs(
                    day,
                    available_offset_h=rng.choice([-12.0, 6.0, 24.0, 36.0, 120.0]),
                    precision=rng.choice(precisions),
                    source_version=rng.choice(["v1", "v1", "v1", ""]),
                    value=rng.random(),
                )
            )

    _assert_paths_agree(store, rows, _spec(), collection_start=base + timedelta(days=250))


def _bulk_insert(store: Store, *, days: int, per_day: int, base: datetime) -> int:
    """Insert rows in the database rather than via pydantic models.

    The point of the test below is how long the *read* path takes. Building
    200,000 `ExternalObservation` objects to set it up would add minutes of
    pydantic validation to the fixture and measure the wrong thing.
    """
    store._conn.execute(
        """
        INSERT INTO external_observations BY NAME
        SELECT
            ? AS series_id,
            ? + INTERVAL 1 DAY * d + INTERVAL 15 MINUTE * q AS observation_time,
            1.0 AS value, 'u' AS unit, 'daily' AS frequency, 'v1' AS source_version,
            ? + INTERVAL 1 DAY * d + INTERVAL 15 MINUTE * q + INTERVAL 36 HOUR
                AS available_at,
            ? + INTERVAL 1 DAY * d + INTERVAL 15 MINUTE * q + INTERVAL 36 HOUR
                AS retrieved_at,
            ? AS source, '1' AS schema_version, '1' AS parser_version,
            FALSE AS is_stale, 1.0 AS quality_score,
            'CONSERVATIVE_DATE' AS availability_precision
        FROM generate_series(0, ? - 1) AS t(d)
        CROSS JOIN generate_series(0, ? - 1) AS u(q)
        """,
        [_SERIES, base, base, base, _SOURCE, days, per_day],
    )
    return days * per_day


def test_the_aggregate_path_is_fast_on_a_large_series(store: Store) -> None:
    # The regression this replaced. Measured 2026-10-23: the row-based path
    # constructed one pydantic object per stored row, twice per series per
    # run, which for FRED's paged ALFRED history meant minutes per series
    # and a cancelled 90-minute job.
    #
    # 200,000 rows is well below what NFCI actually produced and already far
    # beyond anything the old path handled in reasonable time. The threshold
    # is deliberately loose -- this must catch a return to per-row
    # materialisation, not police a CI runner's scheduling.
    base = datetime(2015, 1, 1, tzinfo=UTC)
    days, per_day = 2084, 96
    inserted = _bulk_insert(store, days=days, per_day=per_day, base=base)
    spec = _spec()

    started = time.perf_counter()
    aggregate = store.external_series_aggregate(spec.source, spec.series_id, collection_start=base)
    evidence = evidence_from_aggregate(spec, aggregate, collection_start=base)
    elapsed = time.perf_counter() - started

    assert evidence.nominal_n == inserted
    assert evidence.independent_dates == days
    assert evidence.effective_n == float(days)
    assert elapsed < 10.0, (
        f"the aggregate readiness path took {elapsed:.1f}s for {inserted} rows. It "
        "must stay an aggregate query; per-row materialisation is what cancelled "
        "the external ingest for 90 minutes."
    )


def test_the_row_based_path_is_what_the_aggregate_replaced(store: Store) -> None:
    # Guards the claim above rather than asserting it in a comment: on the
    # same data, reading the rows out must be materially more expensive than
    # the aggregate. Kept small enough to stay quick, and asserted as a
    # ratio so it does not depend on the runner's absolute speed.
    base = datetime(2019, 1, 1, tzinfo=UTC)
    _bulk_insert(store, days=400, per_day=96, base=base)
    spec = _spec()

    started = time.perf_counter()
    store.external_series_aggregate(spec.source, spec.series_id)
    aggregate_seconds = time.perf_counter() - started

    started = time.perf_counter()
    rows = store.list_external_observations(series_id=spec.series_id)
    build_evidence(spec, rows)
    row_seconds = time.perf_counter() - started

    assert len(rows) == 400 * 96
    assert row_seconds > aggregate_seconds, (
        f"materialising rows took {row_seconds:.2f}s, the aggregate "
        f"{aggregate_seconds:.2f}s -- if these are comparable, check that "
        "`external_series_aggregate` is still an aggregate query."
    )


def test_the_latest_observation_time_is_one_query(store: Store) -> None:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    rows = [_obs(base + timedelta(days=i)) for i in range(5000)]
    store.append_external_observations(rows)

    started = time.perf_counter()
    newest = store.latest_external_observation_time(_SOURCE, _SERIES)
    elapsed = time.perf_counter() - started

    assert newest == rows[-1].observation_time
    assert elapsed < 2.0


def test_the_latest_observation_time_is_none_for_an_unknown_series(store: Store) -> None:
    assert store.latest_external_observation_time(_SOURCE, "NOPE") is None
