from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.state.retention import DEFAULT_KEEP_DAYS, compact_product_snapshots
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ProductSnapshot


@pytest.fixture
def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    with Store(tmp_path / "turboedge.duckdb") as s:
        s.init_schema()
        yield s


def _snap_at(
    make_product_snapshot: Callable[..., ProductSnapshot], *, isin: str, when: datetime
) -> ProductSnapshot:
    return make_product_snapshot(
        isin=isin,
        quote_timestamp=when,
        observation_time=when,
        available_at=when,
        retrieved_at=when,
    )


def test_keeps_everything_within_keep_days(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    snaps = [
        _snap_at(make_product_snapshot, isin="DE000AAA1111", when=now - timedelta(days=d))
        for d in range(5)
    ]
    store.append_product_snapshots(snaps)

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.rows_before == 5
    assert report.rows_after == 5
    assert report.rows_removed == 0


def test_reduces_old_snapshots_to_one_per_isin_per_day(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    old_day = now - timedelta(days=100)
    # Three snapshots of the same ISIN on the same (old) UTC calendar day.
    snaps = [
        _snap_at(
            make_product_snapshot,
            isin="DE000AAA1111",
            when=old_day.replace(hour=h),
        )
        for h in (9, 13, 17)
    ]
    store.append_product_snapshots(snaps)

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.rows_before == 3
    assert report.rows_after == 1
    assert report.rows_removed == 2

    # The surviving row is the latest one (17:00), not an arbitrary one.
    remaining = store._conn.execute("SELECT quote_timestamp FROM product_snapshots").fetchall()
    assert len(remaining) == 1
    assert remaining[0][0].hour == 17


def test_reduces_to_one_per_isin_per_distinct_day(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    base = now - timedelta(days=100)
    snaps = [
        _snap_at(make_product_snapshot, isin="DE000AAA1111", when=base - timedelta(days=d))
        for d in range(4)  # 4 distinct old days, one snapshot each
    ]
    store.append_product_snapshots(snaps)

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.rows_before == 4
    assert report.rows_after == 4  # already 1/day -- nothing to reduce
    assert report.rows_removed == 0


def test_ledger_isin_kept_in_full_when_forward_ledger_exists(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    old_day = now - timedelta(days=100)

    protected_isin = "DE000PROTEC1"
    unprotected_isin = "DE000PLAIN01"

    snaps = []
    for h in (9, 13, 17):
        snaps.append(
            _snap_at(make_product_snapshot, isin=protected_isin, when=old_day.replace(hour=h))
        )
        snaps.append(
            _snap_at(make_product_snapshot, isin=unprotected_isin, when=old_day.replace(hour=h))
        )
    store.append_product_snapshots(snaps)

    # `store.init_schema()` (the `store` fixture) already creates the real
    # `forward_ledger` table (W6, storage/duckdb.py) -- replace it with a
    # minimal stand-in exercising only what retention.py actually queries
    # (table + `selected_isin` column via information_schema), decoupled
    # from W6's full NOT NULL column set.
    store._conn.execute("DROP TABLE IF EXISTS forward_ledger")
    store._conn.execute("CREATE TABLE forward_ledger (selected_isin VARCHAR NOT NULL)")
    store._conn.execute("INSERT INTO forward_ledger (selected_isin) VALUES (?)", [protected_isin])

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.forward_ledger_present is True
    assert report.protected_isin_count == 1
    assert report.rows_before == 6
    # protected_isin: all 3 kept; unprotected_isin: reduced to 1
    assert report.rows_after == 4
    assert report.rows_removed == 2

    rows = store._conn.execute(
        "SELECT isin, count(*) FROM product_snapshots GROUP BY isin"
    ).fetchall()
    counts = dict(rows)
    assert counts[protected_isin] == 3
    assert counts[unprotected_isin] == 1


def test_forward_ledger_absent_is_handled_gracefully(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """forward_ledger doesn't exist -- must not error, and nothing is
    treated as protected. Drops the table `store.init_schema()` already
    creates (W6) to exercise the "not migrated yet" code path this module
    must also tolerate."""
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    old_day = now - timedelta(days=100)
    snaps = [
        _snap_at(make_product_snapshot, isin="DE000AAA1111", when=old_day.replace(hour=h))
        for h in (9, 13)
    ]
    store.append_product_snapshots(snaps)
    store._conn.execute("DROP TABLE IF EXISTS forward_ledger")

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.forward_ledger_present is False
    assert report.protected_isin_count == 0
    assert report.rows_after == 1


def test_forward_ledger_without_isin_column_ignored(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """A forward_ledger table without an isin column (unexpected shape) must
    not be treated as a source of protected ISINs, and must not error."""
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    old_day = now - timedelta(days=100)
    snaps = [
        _snap_at(make_product_snapshot, isin="DE000AAA1111", when=old_day.replace(hour=h))
        for h in (9, 13)
    ]
    store.append_product_snapshots(snaps)
    store._conn.execute("DROP TABLE IF EXISTS forward_ledger")
    store._conn.execute("CREATE TABLE forward_ledger (entry_id VARCHAR NOT NULL)")

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.forward_ledger_present is False
    assert report.rows_after == 1


def test_default_keep_days_constant() -> None:
    assert DEFAULT_KEEP_DAYS == 45


def test_invalid_keep_days_raises(store: Store) -> None:
    with pytest.raises(ValueError, match="keep_days"):
        compact_product_snapshots(store, keep_days=0)
    with pytest.raises(ValueError, match="keep_days"):
        compact_product_snapshots(store, keep_days=-5)


def test_runs_checkpoint_and_reports_db_size(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    store.append_product_snapshots([_snap_at(make_product_snapshot, isin="DE000AAA1111", when=now)])

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.db_size_bytes_before >= 0
    assert report.db_size_bytes_after >= 0
