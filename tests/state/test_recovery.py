"""Merging a recovered state database into the live one, additively.

The recovery this serves (`docs/state_recovery_2026-10-23.md`): eight runs
on 2026-10-23/24 packed state that the restore could not reach, and four
scans' `product_snapshots` are irreplaceable -- a bid/ask quote at a past
minute cannot be fetched again.

The test that matters most is `test_merging_twice_changes_nothing` together
with `test_a_table_without_a_primary_key_is_not_duplicated`:
`product_snapshots` has **no** primary key and `append_product_snapshots`
is a plain `INSERT`, so a naive merge would duplicate every row and double
the observation counts that retention thinning and every effective-sample
figure derive from.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.state.recovery import (
    NATURAL_KEYS,
    NEVER_MERGED,
    merge_recovered_state,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    ExternalObservation,
    ProductSnapshot,
    ProductType,
)

_AT = datetime(2026, 10, 23, 17, 57, tzinfo=UTC)


@pytest.fixture
def live(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "live.duckdb") as store:
        store.init_schema()
        yield store


def _recovered(tmp_path: Path, name: str = "recovered.duckdb") -> Path:
    path = tmp_path / name
    with Store(path) as store:
        store.init_schema()
    return path


def _snapshot(isin: str, *, at: datetime = _AT, source: str = "bnp") -> ProductSnapshot:
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer="BankA",
        venue="stuttgart",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=19_000.0,
        knockout_barrier=19_000.0,
        ratio=0.01,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        bid=9.98,
        ask=10.0,
        bid_size=100.0,
        ask_size=100.0,
        quote_timestamp=at,
        quote_presence=True,
        bid_only=False,
        knocked_out=False,
        underlying_price_ref=20_000.0,
        underlying_price_ref_timestamp=at,
        raw_hash=hashlib.sha256(f"{isin}{at}".encode()).hexdigest(),
        observation_time=at,
        available_at=at,
        retrieved_at=at,
        source=source,
        parser_version="1",
        is_stale=False,
        quality_score=0.9,
    )


def _observation(series: str, *, at: datetime = _AT) -> ExternalObservation:
    return ExternalObservation(
        series_id=series,
        value=1.0,
        unit="u",
        frequency="daily",
        source_version="v1",
        observation_time=at,
        available_at=at + timedelta(hours=36),
        retrieved_at=at + timedelta(hours=36),
        source="fred",
        parser_version="1",
        quality_score=1.0,
    )


# -- the core property ---------------------------------------------------


def test_rows_the_live_database_lacks_are_recovered(live: Store, tmp_path: Path) -> None:
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000LOST001")])

    report = merge_recovered_state(live, path)

    assert report.rows_inserted == 1
    assert "product_snapshots" in report.tables_changed
    assert len(live.list_product_snapshots()) == 1
    assert report.summary() == "1 row(s) recovered into 1 table(s)"


def test_a_table_without_a_primary_key_is_not_duplicated(live: Store, tmp_path: Path) -> None:
    # `product_snapshots` has no primary key, so nothing in the database
    # stops a second insert of the same row. The merge has to tell them
    # apart itself.
    shared = _snapshot("DE000SHARED1")
    live.append_product_snapshots([shared])
    path = _recovered(tmp_path)
    with Store(source_path := path) as source:
        source.append_product_snapshots([shared, _snapshot("DE000LOST001")])
    assert source_path == path

    report = merge_recovered_state(live, path)

    assert report.rows_inserted == 1, "only the row the live database lacked"
    assert len(live.list_product_snapshots()) == 2
    assert report.duplicates_introduced == ()


def test_merging_twice_changes_nothing(live: Store, tmp_path: Path) -> None:
    # Idempotence is what makes a merge safe to retry after an interruption,
    # and `docs/state_recovery_2026-10-23.md` requires it explicitly.
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000LOST001")])
        source.append_external_observations([_observation("NFCI")])

    first = merge_recovered_state(live, path)
    second = merge_recovered_state(live, path)

    assert first.rows_inserted == 2
    assert second.rows_inserted == 0
    assert second.summary() == "nothing new -- every row was already present"
    assert second.duplicates_introduced == ()


def test_an_existing_row_is_left_exactly_as_it_is(live: Store, tmp_path: Path) -> None:
    # Additive means additive: a disagreement between the two databases
    # about the same key is not resolved in the recovered database's favour.
    live.append_product_snapshots([_snapshot("DE000SHARED1")])
    path = _recovered(tmp_path)
    with Store(path) as source:
        conflicting = _snapshot("DE000SHARED1").model_copy(update={"ask": 99.0})
        source.append_product_snapshots([conflicting])

    merge_recovered_state(live, path)

    stored = live.list_product_snapshots()
    assert len(stored) == 1
    assert stored[0].ask == 10.0, "the live row must win; nothing is updated"


def test_a_primary_key_table_merges_on_its_own_key(live: Store, tmp_path: Path) -> None:
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_external_observations([_observation("NFCI"), _observation("DFF")])
    live.append_external_observations([_observation("NFCI")])

    report = merge_recovered_state(live, path)

    external = next(t for t in report.tables if t.table == "external_observations")
    assert external.key_source == "primary_key"
    assert external.rows_inserted == 1
    assert external.rows_after == 2


# -- fail-closed on unknown identity -------------------------------------


def test_a_table_with_no_known_identity_is_skipped_not_guessed(
    live: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without an identity there is no way to tell a row already present from
    # a new one. Inserting anyway would duplicate silently, which is the
    # single worst outcome of a recovery.
    monkeypatch.setitem(
        __import__("turboedge.state.recovery", fromlist=["NATURAL_KEYS"]).__dict__,
        "NATURAL_KEYS",
        {},
    )
    live.append_product_snapshots([_snapshot("DE000SHARED1")])
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000SHARED1")])

    report = merge_recovered_state(live, path)

    assert "product_snapshots" not in report.tables_changed
    assert "no primary key" in report.skipped["product_snapshots"]
    assert len(live.list_product_snapshots()) == 1, "nothing was duplicated"


def test_every_table_without_a_primary_key_is_configured_or_refused() -> None:
    # If a new PK-less table appears and nobody configures its identity, the
    # merge must skip it rather than duplicate it. This records which ones
    # are known today so that the set is a decision, not an accident.
    assert "product_snapshots" in NATURAL_KEYS
    assert NATURAL_KEYS["product_snapshots"] == ("isin", "observation_time", "source")
    assert "schema_migrations" in NEVER_MERGED


def test_the_schema_migration_log_is_never_merged(live: Store, tmp_path: Path) -> None:
    # It records what *this* database's schema has been through. Another
    # database's log is not this one's history.
    path = _recovered(tmp_path)

    report = merge_recovered_state(live, path)

    assert "schema_migrations" in report.skipped
    assert "own schema history" in report.skipped["schema_migrations"]


# -- what the report has to show -----------------------------------------


def test_the_report_carries_before_and_after_counts(live: Store, tmp_path: Path) -> None:
    live.append_product_snapshots([_snapshot("DE000SHARED1")])
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000LOST001")])

    report = merge_recovered_state(live, path)

    snapshots = next(t for t in report.tables if t.table == "product_snapshots")
    assert snapshots.rows_before == 1
    assert snapshots.rows_inserted == 1
    assert snapshots.rows_after == 2
    assert snapshots.consistent
    assert report.inconsistent == ()


def test_an_empty_recovered_database_is_a_no_op(live: Store, tmp_path: Path) -> None:
    live.append_product_snapshots([_snapshot("DE000SHARED1")])

    report = merge_recovered_state(live, _recovered(tmp_path))

    assert report.rows_inserted == 0
    assert len(live.list_product_snapshots()) == 1


def test_the_live_database_is_usable_afterwards(live: Store, tmp_path: Path) -> None:
    # The merge attaches and detaches a second database. If the detach were
    # missed, every later call would fail.
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000LOST001")])

    merge_recovered_state(live, path)

    assert live.table_counts()["product_snapshots"] == 1
    assert merge_recovered_state(live, path).rows_inserted == 0


def test_nothing_is_deleted(live: Store, tmp_path: Path) -> None:
    # The one invariant the brief states outright: no historical data is
    # removed by a recovery.
    live.append_product_snapshots([_snapshot(f"DE000KEEP{i:03d}") for i in range(5)])
    before = live.table_counts()

    merge_recovered_state(live, _recovered(tmp_path))

    after = live.table_counts()
    for table, count in before.items():
        assert after[table] >= count, f"{table} lost rows"


# -- two defects the first real dry run exposed --------------------------


def test_each_table_is_processed_exactly_once(live: Store, tmp_path: Path) -> None:
    # `duckdb_tables()` spans every attached database, so inside the ATTACH
    # block an unfiltered query returns each table twice -- once for the
    # live database and once for the recovered one. Run 40189568686 merged
    # every table twice and listed every name twice in its report. The
    # second pass inserted nothing, so no data was harmed, but the counts
    # were meaningless.
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000LOST001")])

    report = merge_recovered_state(live, path)

    tables = [t.table for t in report.tables]
    assert len(tables) == len(set(tables)), f"a table appears more than once: {tables}"
    assert len(report.tables_changed) == len(set(report.tables_changed))


def test_pre_existing_duplicates_are_not_blamed_on_the_merge(live: Store, tmp_path: Path) -> None:
    # Run 40189568686 refused to pack because `rejected_ratio_derivations`
    # had duplicate keys -- which were already there. Refusing was right;
    # the reason was wrong, and a report that cannot tell inherited from
    # introduced makes every recovery look unsafe.
    #
    # `product_snapshots` has no primary key, so two rows with the same
    # natural key can be inserted directly.
    shared = _snapshot("DE000DUPED01")
    live.append_product_snapshots([shared, shared])
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000LOST001")])

    report = merge_recovered_state(live, path)

    assert report.duplicates_introduced == (), (
        "duplicates that were already present must not be reported as introduced"
    )
    # And the table is skipped rather than merged on a key the data shows is
    # not unique -- the same fail-closed rule as having no key at all.
    assert "not unique in the existing data" in report.skipped["product_snapshots"]
    assert "product_snapshots" not in report.tables_changed


def test_a_table_whose_key_is_unique_still_merges(live: Store, tmp_path: Path) -> None:
    # The guard above must not block the normal case.
    live.append_product_snapshots([_snapshot("DE000UNIQUE1")])
    path = _recovered(tmp_path)
    with Store(path) as source:
        source.append_product_snapshots([_snapshot("DE000LOST001")])

    report = merge_recovered_state(live, path)

    assert "product_snapshots" in report.tables_changed
    assert report.rows_inserted == 1
    snapshots = next(t for t in report.tables if t.table == "product_snapshots")
    assert snapshots.duplicate_groups_before == 0
    assert snapshots.duplicates_introduced == 0
