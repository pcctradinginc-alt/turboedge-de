from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.state.retention import (
    DEFAULT_HARD_DELETE_AFTER_DAYS,
    DEFAULT_KEEP_DAYS,
    compact_product_snapshots,
)
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

    # hard_delete_after_days is passed explicitly (well beyond old_day's 100
    # days) so this test isolates rule 1 (thinning) -- rule 2 (hard delete)
    # is covered separately below, and DEFAULT_HARD_DELETE_AFTER_DAYS (90)
    # would otherwise hard-delete this row outright rather than thin it.
    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

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

    # See the comment on the equivalent line in
    # test_reduces_old_snapshots_to_one_per_isin_per_day for why
    # hard_delete_after_days is explicit here.
    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

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

    # See the comment on the equivalent line in
    # test_reduces_old_snapshots_to_one_per_isin_per_day for why
    # hard_delete_after_days is explicit here.
    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

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

    # See the comment on the equivalent line in
    # test_reduces_old_snapshots_to_one_per_isin_per_day for why
    # hard_delete_after_days is explicit here.
    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

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

    # See the comment on the equivalent line in
    # test_reduces_old_snapshots_to_one_per_isin_per_day for why
    # hard_delete_after_days is explicit here.
    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

    assert report.forward_ledger_present is False
    assert report.rows_after == 1


def test_alternatives_isin_protected_in_full(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """An ISIN referenced only as a discarded alternative/counterfactual
    (forward_ledger.alternatives, Master Spec §21) -- never as
    selected_isin -- must be kept in full, exactly like the selected pick."""
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    old_day = now - timedelta(days=100)

    selected_isin = "DE000SELEC01"
    alt_isin = "DE000ALT0001"
    plain_isin = "DE000PLAINX1"

    snaps = []
    for isin in (selected_isin, alt_isin, plain_isin):
        for h in (9, 13, 17):
            snaps.append(_snap_at(make_product_snapshot, isin=isin, when=old_day.replace(hour=h)))
    store.append_product_snapshots(snaps)

    store._conn.execute("DROP TABLE IF EXISTS forward_ledger")
    store._conn.execute(
        "CREATE TABLE forward_ledger "
        "(selected_isin VARCHAR NOT NULL, alternatives VARCHAR NOT NULL)"
    )
    store._conn.execute(
        "INSERT INTO forward_ledger (selected_isin, alternatives) VALUES (?, ?)",
        [selected_isin, json.dumps([alt_isin])],
    )

    # See the comment on the equivalent line in
    # test_reduces_old_snapshots_to_one_per_isin_per_day for why
    # hard_delete_after_days is explicit here.
    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

    assert report.protected_isin_count == 2  # selected_isin + the one alternative
    rows = store._conn.execute(
        "SELECT isin, count(*) FROM product_snapshots GROUP BY isin"
    ).fetchall()
    counts = dict(rows)
    assert counts[selected_isin] == 3
    assert counts[alt_isin] == 3  # protected purely via `alternatives`
    assert counts[plain_isin] == 1  # thinned like any unprotected ISIN


def test_alternatives_column_absent_ignored(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """A forward_ledger without an `alternatives` column (unexpected shape)
    must not error -- protection falls back to selected_isin only."""
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    old_day = now - timedelta(days=100)
    selected_isin = "DE000SELEC01"
    snaps = [
        _snap_at(make_product_snapshot, isin=selected_isin, when=old_day.replace(hour=h))
        for h in (9, 13)
    ]
    store.append_product_snapshots(snaps)

    store._conn.execute("DROP TABLE IF EXISTS forward_ledger")
    store._conn.execute("CREATE TABLE forward_ledger (selected_isin VARCHAR NOT NULL)")
    store._conn.execute("INSERT INTO forward_ledger (selected_isin) VALUES (?)", [selected_isin])

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.protected_isin_count == 1
    assert store._conn.execute("SELECT count(*) FROM product_snapshots").fetchone()[0] == 2


def test_default_keep_days_constant() -> None:
    assert DEFAULT_KEEP_DAYS == 5


def test_default_hard_delete_after_days_constant() -> None:
    assert DEFAULT_HARD_DELETE_AFTER_DAYS == 90


def test_invalid_keep_days_raises(store: Store) -> None:
    with pytest.raises(ValueError, match="keep_days"):
        compact_product_snapshots(store, keep_days=0)
    with pytest.raises(ValueError, match="keep_days"):
        compact_product_snapshots(store, keep_days=-5)


def test_invalid_hard_delete_after_days_raises(store: Store) -> None:
    with pytest.raises(ValueError, match="hard_delete_after_days"):
        compact_product_snapshots(store, keep_days=45, hard_delete_after_days=0)
    with pytest.raises(ValueError, match="hard_delete_after_days"):
        compact_product_snapshots(store, keep_days=45, hard_delete_after_days=-5)


def test_hard_delete_after_days_must_exceed_keep_days(store: Store) -> None:
    with pytest.raises(ValueError, match="hard_delete_after_days"):
        compact_product_snapshots(store, keep_days=45, hard_delete_after_days=45)
    with pytest.raises(ValueError, match="hard_delete_after_days"):
        compact_product_snapshots(store, keep_days=45, hard_delete_after_days=10)


# --------------------------------------------------------------------------
# Hard delete (rule 2): rows still around after keep_days-thinning that are
# older than hard_delete_after_days are permanently removed, not just
# thinned -- unless the ISIN is ledger-protected, in which case it survives
# no matter how old (see the alternatives-protection tests above/below).
# --------------------------------------------------------------------------


def test_hard_delete_removes_rows_past_hard_cutoff(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    beyond_hard_cutoff = now - timedelta(days=500)  # older than hard_delete_after_days=400
    within_thin_zone = now - timedelta(days=100)  # thinned, not hard-deleted (100 < 400)

    isin = "DE000AAA1111"
    snaps = [
        _snap_at(make_product_snapshot, isin=isin, when=beyond_hard_cutoff),
        _snap_at(make_product_snapshot, isin=isin, when=within_thin_zone),
    ]
    store.append_product_snapshots(snaps)

    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

    assert report.rows_before == 2
    assert report.rows_hard_deleted == 1
    assert report.rows_after == 1
    remaining = store._conn.execute("SELECT quote_timestamp FROM product_snapshots").fetchall()
    assert len(remaining) == 1
    assert remaining[0][0] == within_thin_zone


def test_hard_delete_does_not_touch_rows_within_the_hard_cutoff(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """A row just inside hard_delete_after_days (but outside keep_days) is
    thinned like any other old row -- not hard-deleted."""
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    just_inside = now - timedelta(days=399)  # < 400: survives, thinned

    isin = "DE000AAA1111"
    store.append_product_snapshots([_snap_at(make_product_snapshot, isin=isin, when=just_inside)])

    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

    assert report.rows_hard_deleted == 0
    assert report.rows_after == 1


def test_hard_delete_protects_ledger_selected_isin_beyond_hard_cutoff(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    ancient = now - timedelta(days=1000)  # far beyond any sane hard_delete_after_days

    protected_isin = "DE000PROTEC1"
    unprotected_isin = "DE000PLAIN01"
    snaps = [
        _snap_at(make_product_snapshot, isin=protected_isin, when=ancient),
        _snap_at(make_product_snapshot, isin=unprotected_isin, when=ancient),
    ]
    store.append_product_snapshots(snaps)

    store._conn.execute("DROP TABLE IF EXISTS forward_ledger")
    store._conn.execute("CREATE TABLE forward_ledger (selected_isin VARCHAR NOT NULL)")
    store._conn.execute("INSERT INTO forward_ledger (selected_isin) VALUES (?)", [protected_isin])

    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

    assert report.rows_hard_deleted == 1  # only the unprotected one
    rows = store._conn.execute("SELECT isin FROM product_snapshots").fetchall()
    assert [r[0] for r in rows] == [protected_isin]


def test_hard_delete_protects_ledger_alternative_isin_beyond_hard_cutoff(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """An ISIN referenced only via `alternatives` survives the hard delete
    too -- protection is not limited to the entry actually taken."""
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    ancient = now - timedelta(days=1000)

    selected_isin = "DE000SELEC01"
    alt_isin = "DE000ALT0001"
    snaps = [
        _snap_at(make_product_snapshot, isin=selected_isin, when=ancient),
        _snap_at(make_product_snapshot, isin=alt_isin, when=ancient),
    ]
    store.append_product_snapshots(snaps)

    store._conn.execute("DROP TABLE IF EXISTS forward_ledger")
    store._conn.execute(
        "CREATE TABLE forward_ledger "
        "(selected_isin VARCHAR NOT NULL, alternatives VARCHAR NOT NULL)"
    )
    store._conn.execute(
        "INSERT INTO forward_ledger (selected_isin, alternatives) VALUES (?, ?)",
        [selected_isin, json.dumps([alt_isin])],
    )

    report = compact_product_snapshots(store, keep_days=45, hard_delete_after_days=400, now=now)

    assert report.rows_hard_deleted == 0
    rows = {r[0] for r in store._conn.execute("SELECT isin FROM product_snapshots").fetchall()}
    assert rows == {selected_isin, alt_isin}


def test_hard_delete_report_default_matches_module_default(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    store.append_product_snapshots([_snap_at(make_product_snapshot, isin="DE000AAA1111", when=now)])
    report = compact_product_snapshots(store, keep_days=45, now=now)
    assert report.hard_delete_after_days == DEFAULT_HARD_DELETE_AFTER_DAYS


def test_runs_checkpoint_and_reports_db_size(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    now = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
    store.append_product_snapshots([_snap_at(make_product_snapshot, isin="DE000AAA1111", when=now)])

    report = compact_product_snapshots(store, keep_days=45, now=now)

    assert report.db_size_bytes_before >= 0
    assert report.db_size_bytes_after >= 0


# --------------------------------------------------------------------------
# The physical-shrink case: real row deletion (rows_removed > 0), the one
# scenario never exercised above -- every test up to here either keeps
# everything or reduces same-day duplicates (rows_removed == 0 or small).
# Built via a bulk SQL INSERT rather than the ProductSnapshot fixture: this
# is a storage-layer volume test (needs enough rows to span multiple DuckDB
# row-group blocks before a size change is even observable -- see the
# scratchpad measurement script referenced in retention.py's module
# docstring), and constructing thousands of pydantic models is needlessly
# slow for that.
# --------------------------------------------------------------------------

_BULK_ISIN_COUNT = 25
_BULK_DAY_COUNT = 400
_BULK_SCANS_PER_DAY = 5  # matches pipeline.yml's 5 weekday scan crons
_BULK_ROW_COUNT = _BULK_ISIN_COUNT * _BULK_DAY_COUNT * _BULK_SCANS_PER_DAY  # 50,000


def _bulk_insert_product_snapshots(store: Store, *, now: datetime) -> None:
    """Insert ``_BULK_ROW_COUNT`` rows: ``_BULK_ISIN_COUNT`` ISINs, each with
    ``_BULK_SCANS_PER_DAY`` snapshots/day over ``_BULK_DAY_COUNT`` distinct
    UTC calendar days ending at ``now`` -- mirrors the real scheduler
    (pipeline.yml scans several times/day, forever) closely enough to
    exercise real, multi-day-spanning deletion once ``keep_days`` is
    smaller than ``_BULK_DAY_COUNT``.

    ``threads=1`` and ``force_compression='uncompressed'`` pin DuckDB to a
    single-threaded, non-adaptive writer: with the default (parallel,
    auto-compressing) writer, row-group packing -- and therefore the exact
    file size -- was observed to vary between otherwise-identical runs
    (verified empirically while writing this test, looping the exact insert
    + compact below in-process: file sizes after an identical compact call
    landed on one of two distinct values across repeated runs, ~15-20% of
    the time picking the larger one), which made the size assertions below
    flaky (DuckDB apparently samples data when choosing a per-column
    compression codec, and that choice isn't fully pinned down by
    ``threads=1`` alone). Forcing uncompressed storage removes that
    adaptive-codec source of nondeterminism -- confirmed byte-identical
    across 6 repeated trials of 4 compact calls each with both settings
    applied. This is purely a test-determinism device: production code
    never sets either pragma, since forcing uncompressed storage would
    defeat the whole point of ``db compact``.
    """
    store._conn.execute("SET threads=1")
    store._conn.execute("PRAGMA force_compression='uncompressed'")
    store._conn.execute("SET TimeZone='UTC'")
    store._conn.execute(
        f"""
        INSERT INTO product_snapshots (
            isin, wkn, issuer, venue, underlying_raw, underlying_id, direction,
            product_type, financing_level, knockout_barrier, ratio, currency,
            underlying_currency, quanto, open_end, maturity, first_trading_day,
            bid, ask, bid_size, ask_size, quote_timestamp, quote_presence,
            bid_only, knocked_out, trading_hours, product_age_days,
            underlying_price_ref, underlying_price_ref_timestamp, financing_rate,
            raw_hash, observation_time, available_at, retrieved_at,
            source_timestamp, source, schema_version, parser_version, is_stale,
            quality_score
        )
        SELECT
            'DE000TEST' || (i % {_BULK_ISIN_COUNT})::VARCHAR,
            'WKN' || (i % {_BULK_ISIN_COUNT})::VARCHAR,
            'TestBank', 'stuttgart', 'DAX', 'DAX', 'LONG', 'TURBO_OPEN_END',
            18000.0, 18000.0, 0.01, 'EUR', 'EUR', false, true, NULL, NULL,
            4.80, 4.86, 1000.0, 1000.0, ts, true, false, false,
            '09:00-22:00', 120, 18500.0, ts, NULL, md5(i::VARCHAR),
            ts, ts, ts, ts, 'test', '1', '1', false, 0.95
        FROM (
            -- `i % ISIN_COUNT` and `(i // ISIN_COUNT)` are independent (the
            -- latter sweeps 0..DAY_COUNT*SCANS_PER_DAY-1 exactly once per
            -- ISIN), so day_idx/scan_idx below land exactly
            -- SCANS_PER_DAY times in every (isin, day) bucket -- verified
            -- directly against this exact formula before writing this test.
            SELECT
                i,
                ?::TIMESTAMPTZ
                - (((i // {_BULK_ISIN_COUNT}) % {_BULK_DAY_COUNT}) || ' days')::INTERVAL
                - ((((i // {_BULK_ISIN_COUNT}) // {_BULK_DAY_COUNT}) * 3)
                    || ' hours')::INTERVAL AS ts
            FROM range({_BULK_ROW_COUNT}) t(i)
        )
        """,
        [now],
    )


def test_real_deletion_shrinks_file_on_disk(store: Store) -> None:
    """The core bug report: `db compact` must not just report fewer rows,
    it must make the .duckdb file physically smaller on disk when rows are
    actually removed -- not grow it (the reported +60%) and not leave it
    flat (CHECKPOINT/VACUUM alone, measured in the scratchpad script, do
    neither)."""
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    _bulk_insert_product_snapshots(store, now=now)
    store._conn.execute("CHECKPOINT")

    rows_before = store._conn.execute("SELECT count(*) FROM product_snapshots").fetchone()[0]
    size_before = store.path.stat().st_size
    assert rows_before == _BULK_ROW_COUNT

    report = compact_product_snapshots(store, keep_days=45, now=now)

    size_after = store.path.stat().st_size

    # A real deletion happened (this is the case rows_removed=0 in every
    # measurement run of the original bug never covered).
    assert report.rows_before == _BULK_ROW_COUNT
    assert report.rows_removed > 0
    assert report.rows_after < report.rows_before

    assert report.file_rewritten is True
    assert report.db_size_bytes_after == size_after
    # The actual regression under test: real row deletion must shrink the
    # file, not merely fail to grow it.
    assert size_after < size_before


def test_compact_is_idempotent_and_does_not_grow_the_file(store: Store) -> None:
    """A second `db compact` call back-to-back (nothing left to remove --
    the reported bug's exact rows_removed=0 case) must not grow the file,
    matching the reported +60%/oscillating growth from calling compact on
    an already-compacted database.

    Re-applies the same ``force_compression`` pragma
    ``_bulk_insert_product_snapshots`` used (see its docstring) before every
    call: a successful rewrite reconnects ``store._conn`` (a fresh
    connection, deliberately not inheriting session-local pragmas -- see
    ``_rewrite_database_file``), so this is re-armed each time to keep the
    *test's* size comparisons deterministic; nothing in the production code
    path depends on it.
    """
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    _bulk_insert_product_snapshots(store, now=now)
    store._conn.execute("CHECKPOINT")

    report1 = compact_product_snapshots(store, keep_days=45, now=now)
    size_after_1 = store.path.stat().st_size
    assert report1.rows_removed > 0

    store._conn.execute("PRAGMA force_compression='uncompressed'")
    report2 = compact_product_snapshots(store, keep_days=45, now=now)
    size_after_2 = store.path.stat().st_size

    # Nothing left to remove: already 1 row/(isin, day) outside keep_days.
    assert report2.rows_removed == 0
    assert size_after_2 <= size_after_1

    store._conn.execute("PRAGMA force_compression='uncompressed'")
    report3 = compact_product_snapshots(store, keep_days=45, now=now)
    size_after_3 = store.path.stat().st_size
    assert report3.rows_removed == 0
    assert size_after_3 <= size_after_1
    # A no-op compact must be a true fixed point, not just non-growing on
    # average: with nothing left to remove, re-rewriting the exact same
    # data must reproduce the exact same file, every time.
    assert size_after_3 == size_after_2


def test_rewrite_preserves_all_tables_and_row_content(
    store: Store, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """The file swap in `_rewrite_database_file` must not lose or corrupt
    data -- neither in `product_snapshots` itself nor in sibling tables
    (`COPY FROM DATABASE` copies the whole file, not just one table)."""
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    _bulk_insert_product_snapshots(store, now=now)

    store.start_run("run-1", command="scan", config_hash="deadbeef", git_commit="abc123")
    store.upsert_instruments(
        [_snap_at(make_product_snapshot, isin="DE000AAA1111", when=now - timedelta(days=1))]
    )
    store._conn.execute("CHECKPOINT")

    runs_before = store._conn.execute("SELECT run_id, command FROM runs").fetchall()
    instruments_before = store._conn.execute("SELECT isin FROM instruments").fetchall()

    report = compact_product_snapshots(store, keep_days=45, now=now)
    assert report.file_rewritten is True

    # Sibling tables untouched by the product_snapshots-specific row
    # reduction survive the whole-file rewrite unchanged.
    assert store._conn.execute("SELECT run_id, command FROM runs").fetchall() == runs_before
    assert store._conn.execute("SELECT isin FROM instruments").fetchall() == instruments_before

    # product_snapshots' surviving rows are real, well-formed data, not
    # truncated/corrupted by the rewrite -- for history older than
    # keep_days, exactly one row per (isin, day) (rows *within* keep_days
    # are deliberately kept at full multiplicity, so only the older slice
    # is checked here).
    cutoff = now - timedelta(days=45)
    per_isin_day = store._conn.execute(
        """
        SELECT isin, CAST(quote_timestamp AS DATE) AS d, count(*)
        FROM product_snapshots
        WHERE quote_timestamp < ?
        GROUP BY isin, d
        HAVING count(*) > 1
        """,
        [cutoff],
    ).fetchall()
    assert per_isin_day == []


def test_reopened_store_sees_compacted_data_after_rewrite(tmp_path: Path) -> None:
    """After a rewrite swaps the file, a *new* connection to the same path
    (e.g. the next CLI invocation) must see the compacted data -- not a
    stale pre-compaction copy left behind by a swap that silently failed or
    pointed the wrong file at the wrong name."""
    db_path = tmp_path / "turboedge.duckdb"
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)

    with Store(db_path) as store:
        store.init_schema()
        _bulk_insert_product_snapshots(store, now=now)
        store._conn.execute("CHECKPOINT")
        report = compact_product_snapshots(store, keep_days=45, now=now)
        assert report.file_rewritten is True
        expected_rows = report.rows_after

    with Store(db_path) as reopened:
        reopened.init_schema()
        rows = reopened._conn.execute("SELECT count(*) FROM product_snapshots").fetchone()[0]
        assert rows == expected_rows


# --------------------------------------------------------------------------
# Regression coverage for the current DEFAULT_KEEP_DAYS/DEFAULT_HARD_DELETE_
# AFTER_DAYS values (see the module docstring's "Why these defaults" for the
# full measurement this is a scaled-down version of, run against a realistic
# synthetic DB of 21,700 ISINs/scan x 5 scans/day x a multi-hundred-day span
# before choosing 5/90 over the old 45/400): the defaults must actually
# trigger both thinning *and* hard deletion at realistic scale, without ever
# touching a ledger-protected ISIN or leaving an ordinary ISIN's
# `financing_level_history()` (what `pricing/financing.py`'s spread
# inference is fed from) without >= 2 consecutive calendar days of history.
# --------------------------------------------------------------------------

_DEFAULT_SCALE_ISIN_COUNT = 300
_DEFAULT_SCALE_COLD_DAYS = 110  # > DEFAULT_HARD_DELETE_AFTER_DAYS (90): real deletions happen
_DEFAULT_SCALE_LEDGER_SELECTED = 3
_DEFAULT_SCALE_LEDGER_ALT_ONLY = 2
_DEFAULT_SCALE_LEDGER_TOTAL = _DEFAULT_SCALE_LEDGER_SELECTED + _DEFAULT_SCALE_LEDGER_ALT_ONLY

_SNAPSHOT_COLUMNS = (
    "isin, wkn, issuer, venue, underlying_raw, underlying_id, direction, "
    "product_type, financing_level, knockout_barrier, ratio, currency, "
    "underlying_currency, quanto, open_end, maturity, first_trading_day, "
    "bid, ask, bid_size, ask_size, quote_timestamp, quote_presence, "
    "bid_only, knocked_out, trading_hours, product_age_days, "
    "underlying_price_ref, underlying_price_ref_timestamp, financing_rate, "
    "raw_hash, observation_time, available_at, retrieved_at, "
    "source_timestamp, source, schema_version, parser_version, is_stale, "
    "quality_score"
)


def _seed_default_scale_database(store: Store, *, now: datetime) -> None:
    """Non-ledger ISINs get DEFAULT_KEEP_DAYS days of full (5 scans/day)
    resolution followed by daily resolution out to
    _DEFAULT_SCALE_COLD_DAYS (already-thinned, exactly what
    DEFAULT_KEEP_DAYS-based thinning would itself produce -- see the
    scratchpad measurement script this test mirrors for why that's a
    size-neutral shortcut); ledger ISINs get full 5 scans/day resolution for
    the entire span, unconditionally."""
    conn = store._conn
    conn.execute("SET TimeZone='UTC'")
    scans_per_day = 5
    hot_days = DEFAULT_KEEP_DAYS
    isin_count = _DEFAULT_SCALE_ISIN_COUNT

    n_hot = hot_days * scans_per_day * isin_count
    conn.execute(
        f"""
        INSERT INTO product_snapshots ({_SNAPSHOT_COLUMNS})
        SELECT
            'DE000NL' || lpad((i % {isin_count})::VARCHAR, 6, '0'),
            'WKN' || (i % {isin_count})::VARCHAR, 'TestBank', 'stuttgart',
            'DAX', 'DAX', 'LONG', 'TURBO_OPEN_END',
            18000.0 + (i % 97) * 0.5, 18000.0, 0.01, 'EUR', 'EUR',
            false, true, NULL, NULL, 4.80, 4.86, 1000.0, 1000.0, ts, true,
            false, false, '09:00-22:00', 120, 18500.0, ts, 0.035,
            md5(i::VARCHAR), ts, ts, ts, ts, 'test', '1', '1', false, 0.95
        FROM (
            SELECT i, ?::TIMESTAMPTZ
                - (((i // {isin_count}) // {scans_per_day}) || ' days')::INTERVAL
                - ((((i // {isin_count}) % {scans_per_day}) * 3) || ' hours')::INTERVAL AS ts
            FROM range({n_hot}) t(i)
        )
        """,
        [now],
    )

    n_cold_days = _DEFAULT_SCALE_COLD_DAYS - hot_days
    n_cold = n_cold_days * isin_count
    conn.execute(
        f"""
        INSERT INTO product_snapshots ({_SNAPSHOT_COLUMNS})
        SELECT
            'DE000NL' || lpad((i % {isin_count})::VARCHAR, 6, '0'),
            'WKN' || (i % {isin_count})::VARCHAR, 'TestBank', 'stuttgart',
            'DAX', 'DAX', 'LONG', 'TURBO_OPEN_END',
            18000.0 + (i % 97) * 0.5, 18000.0, 0.01, 'EUR', 'EUR',
            false, true, NULL, NULL, 4.80, 4.86, 1000.0, 1000.0, ts, true,
            false, false, '09:00-22:00', 120, 18500.0, ts, 0.035,
            md5(i::VARCHAR || '-cold'), ts, ts, ts, ts, 'test', '1', '1', false, 0.95
        FROM (
            SELECT i, ?::TIMESTAMPTZ
                - (({hot_days} + (i // {isin_count})) || ' days')::INTERVAL
                - '2 hours'::INTERVAL AS ts
            FROM range({n_cold}) t(i)
        )
        """,
        [now],
    )

    n_ledger = _DEFAULT_SCALE_COLD_DAYS * scans_per_day * _DEFAULT_SCALE_LEDGER_TOTAL
    conn.execute(
        f"""
        INSERT INTO product_snapshots ({_SNAPSHOT_COLUMNS})
        SELECT
            'DE000LG' || lpad((i % {_DEFAULT_SCALE_LEDGER_TOTAL})::VARCHAR, 6, '0'),
            'WKNLG' || (i % {_DEFAULT_SCALE_LEDGER_TOTAL})::VARCHAR, 'TestBank',
            'stuttgart', 'DAX', 'DAX', 'LONG', 'TURBO_OPEN_END',
            18000.0, 18000.0, 0.01, 'EUR', 'EUR', false, true, NULL, NULL,
            4.80, 4.86, 1000.0, 1000.0, ts, true, false, false,
            '09:00-22:00', 120, 18500.0, ts, 0.035,
            md5(i::VARCHAR || '-ledger'), ts, ts, ts, ts, 'test', '1', '1', false, 0.95
        FROM (
            SELECT i, ?::TIMESTAMPTZ
                - (((i // {_DEFAULT_SCALE_LEDGER_TOTAL}) // {scans_per_day}) || ' days')::INTERVAL
                - ((((i // {_DEFAULT_SCALE_LEDGER_TOTAL}) % {scans_per_day}) * 3)
                    || ' hours')::INTERVAL AS ts
            FROM range({n_ledger}) t(i)
        )
        """,
        [now],
    )

    conn.execute("DROP TABLE IF EXISTS forward_ledger")
    conn.execute(
        "CREATE TABLE forward_ledger (selected_isin VARCHAR NOT NULL, "
        "alternatives VARCHAR NOT NULL)"
    )
    alt_only = [
        f"DE000LG{str(i).zfill(6)}"
        for i in range(_DEFAULT_SCALE_LEDGER_SELECTED, _DEFAULT_SCALE_LEDGER_TOTAL)
    ]
    for i in range(_DEFAULT_SCALE_LEDGER_SELECTED):
        conn.execute(
            "INSERT INTO forward_ledger (selected_isin, alternatives) VALUES (?, ?)",
            [f"DE000LG{str(i).zfill(6)}", json.dumps(alt_only)],
        )


def test_defaults_compact_realistically_and_preserve_ledger_and_financing_history(
    store: Store,
) -> None:
    """Encodes the measurement behind DEFAULT_KEEP_DAYS=5/
    DEFAULT_HARD_DELETE_AFTER_DAYS=90 (see module docstring): at a
    realistic multi-hundred-day scale with the current *defaults* (no
    explicit keep_days/hard_delete_after_days override), thinning and real
    hard deletion both actually happen, no ledger-protected ISIN loses a
    single row, and financing-level history for ordinary ISINs never drops
    below 2 consecutive calendar days -- the exact bar
    `pricing/financing.py`'s spread inference needs."""
    now = datetime(2026, 9, 14, 20, 0, tzinfo=UTC)
    _seed_default_scale_database(store, now=now)

    rows_before = store._conn.execute("SELECT count(*) FROM product_snapshots").fetchone()[0]

    report = compact_product_snapshots(store, now=now)  # defaults only

    assert report.keep_days == DEFAULT_KEEP_DAYS
    assert report.hard_delete_after_days == DEFAULT_HARD_DELETE_AFTER_DAYS
    assert report.rows_before == rows_before
    # Both rules actually bite at this scale (not a no-op).
    assert report.rows_removed > 0
    assert report.rows_hard_deleted > 0
    assert report.rows_after < report.rows_before

    # Ledger ISINs: every one of their rows across the *entire* span
    # survives untouched, however old.
    ledger_counts = dict(
        store._conn.execute(
            "SELECT isin, count(*) FROM product_snapshots WHERE isin LIKE 'DE000LG%' GROUP BY isin"
        ).fetchall()
    )
    expected_per_ledger_isin = _DEFAULT_SCALE_COLD_DAYS * 5
    assert len(ledger_counts) == _DEFAULT_SCALE_LEDGER_TOTAL
    assert all(c == expected_per_ledger_isin for c in ledger_counts.values())

    # Ordinary ISINs: financing_level_history still has >= 2 consecutive
    # calendar days -- sampled across the ISIN range, not just isin 0.
    sample_isins = [
        f"DE000NL{str(i).zfill(6)}"
        for i in (0, _DEFAULT_SCALE_ISIN_COUNT // 2, _DEFAULT_SCALE_ISIN_COUNT - 1)
    ]
    for isin in sample_isins:
        history_days = sorted({ts.date() for ts, _ in store.financing_level_history(isin)})
        assert len(history_days) >= 2, f"{isin}: only {len(history_days)} day(s) of history"
        assert any(
            (history_days[i + 1] - history_days[i]).days == 1 for i in range(len(history_days) - 1)
        ), f"{isin}: no 2 consecutive calendar days in {history_days}"
