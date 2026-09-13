from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from turboedge.storage.duckdb import Store, StoreError
from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    HealthStatus,
    ManualPosition,
    NotificationRecord,
    PositionStatus,
    ProductSnapshot,
    SignalSnapshot,
    SourceHealthRecord,
    UnderlyingBar,
)


@pytest.fixture
def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    with Store(tmp_path / "turboedge.duckdb") as s:
        s.init_schema()
        yield s


def test_init_schema_creates_all_tables_empty(store: Store) -> None:
    counts = store.table_counts()
    assert set(counts) == {
        "runs",
        "instruments",
        "product_snapshots",
        "underlying_prices",
        "signals",
        "candidate_sets",
        "source_health",
        "positions_manual",
        "notifications_sent",
        "forward_ledger",
        "ledger_labels",
        "strategy_posteriors",
        "model_registry",
        "model_weight_history",
        "research_trials",
        "drift_events",
        "shadow_portfolio",
        "forecasts",
        "position_evaluations",
        "walkforward_results",
    }
    assert all(v == 0 for v in counts.values())


def test_init_schema_is_idempotent(store: Store) -> None:
    store.init_schema()
    store.init_schema()
    assert store.table_counts()["runs"] == 0


def test_run_lifecycle(store: Store) -> None:
    store.start_run("run-1", command="scan", config_hash="abc", git_commit="deadbeef")
    store.finish_run("run-1", status="ok")
    row = store._conn.execute(
        "SELECT status, finished_at FROM runs WHERE run_id = ?", ["run-1"]
    ).fetchone()
    assert row is not None
    assert row[0] == "ok"
    assert row[1] is not None


def test_append_and_upsert_product_snapshots(store: Store, make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    snap1 = make_product_snapshot(isin="DE000ABC1234", financing_level=18000.0)
    snap2 = make_product_snapshot(
        isin="DE000ABC1234",
        financing_level=18010.0,
        observation_time=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        quote_timestamp=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
    )
    n = store.append_product_snapshots([snap1, snap2])
    assert n == 2
    assert store.table_counts()["product_snapshots"] == 2

    upserted = store.upsert_instruments([snap1, snap2])
    assert upserted == 1  # same ISIN -> one instrument row
    assert store.table_counts()["instruments"] == 1

    # Upsert again with a newer snapshot for the same ISIN should keep first_seen_at.
    row = store._conn.execute(
        "SELECT first_seen_at, last_seen_at FROM instruments WHERE isin = ?",
        ["DE000ABC1234"],
    ).fetchone()
    assert row is not None
    first_seen, last_seen = row
    assert first_seen <= last_seen


def test_append_product_snapshots_empty_list_is_noop(store: Store) -> None:
    assert store.append_product_snapshots([]) == 0


def test_underlying_bars_roundtrip_and_latest(store: Store) -> None:
    bars = [
        UnderlyingBar(
            underlying_id="DAX",
            ts=datetime(2026, 9, 8 + i, tzinfo=UTC),
            open=18000.0 + i,
            high=18100.0 + i,
            low=17950.0 + i,
            close=18050.0 + i,
            volume=1000.0,
            observation_time=datetime(2026, 9, 8 + i, tzinfo=UTC),
            available_at=datetime(2026, 9, 8 + i, 22, tzinfo=UTC),
            retrieved_at=datetime(2026, 9, 8 + i, 22, 5, tzinfo=UTC),
            source="yfinance",
            parser_version="1",
            quality_score=0.9,
        )
        for i in range(5)
    ]
    n = store.append_underlying_bars(bars)
    assert n == 5

    latest = store.latest_underlying_bars("DAX", 3)
    assert len(latest) == 3
    assert [b.close for b in latest] == [18052.0, 18053.0, 18054.0]  # ascending, most recent last
    assert all(b.ts.tzinfo is not None for b in latest)


def test_append_signal_and_query(store: Store) -> None:
    signal = SignalSnapshot(
        signal_id="tsmom_horizon_norm_v1",
        signal_version_hash="hash123",
        underlying_id="DAX",
        prediction_time=datetime(2026, 9, 10, 8, 0, tzinfo=UTC),
        frozen_at=datetime(2026, 9, 10, 8, 1, tzinfo=UTC),
        score=0.42,
        components={"z21": 0.3, "z63": 0.5, "z126": 0.45},
        direction_hint=Direction.LONG,
        threshold=0.5,
        config_hash="cfg-hash",
        git_commit="deadbeef",
        data_snapshot_hash="snap-hash",
    )
    store.append_signal(signal)
    assert store.table_counts()["signals"] == 1
    row = store._conn.execute("SELECT components FROM signals").fetchone()
    assert row is not None
    import json

    assert json.loads(row[0]) == {"z21": 0.3, "z63": 0.5, "z126": 0.45}


def test_append_and_list_candidates(store: Store) -> None:
    costs = CostDecomposition(
        ask=4.86,
        bid=4.80,
        mid=4.83,
        intrinsic=4.70,
        trading_spread_component=0.03,
        fair_gap_premium=0.02,
        financing_drag=0.01,
        issuer_margin=0.10,
        spread_pct=0.0123,
        gap_premium_pct=0.0041,
        financing_drag_pct=0.0021,
        issuer_margin_pct=0.0206,
    )
    candidate = CandidateEvaluation(
        run_id="run-1",
        candidate_id="cand-1",
        isin="DE000ABC1234",
        wkn="ABC123",
        issuer="TestBank",
        underlying_id="DAX",
        direction=Direction.LONG,
        category=Category.WATCH,
        reasons=["ok"],
        leverage=5.0,
        leverage_bucket="5-10",
        distance_to_barrier_pct=0.03,
        distance_to_barrier_sigma=1.5,
        costs=costs,
        realized_financing_spread=0.02,
        financing_cost_horizon_pct={"3d": 0.001, "7d": 0.003},
        cross_issuer_residual_zscore=0.5,
        issuer_markup_score=0.1,
        quote_dislocation_score=0.05,
        wrapper_edge=None,
        liquidity_factor=0.8,
        integrity_passed=True,
        lcb_ev=None,
        cost_rank_score=0.12,
    )
    n = store.append_candidates([candidate])
    assert n == 1
    fetched = store.list_candidates("run-1")
    assert len(fetched) == 1
    assert fetched[0] == candidate


def test_append_source_health(store: Store) -> None:
    record = SourceHealthRecord(
        source="ecb",
        checked_at=datetime(2026, 9, 10, tzinfo=UTC),
        availability=1.0,
        freshness=0.9,
        missingness=0.0,
        schema_consistency=1.0,
        cross_source_agreement=None,
        score=0.95,
        status=HealthStatus.PASS,
        message="ok",
    )
    n = store.append_source_health([record])
    assert n == 1
    assert store.table_counts()["source_health"] == 1


def test_financing_level_history_one_per_calendar_day(store: Store, make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    snaps = [
        make_product_snapshot(
            isin="DE000ABC1234",
            financing_level=18000.0,
            observation_time=datetime(2026, 9, 8, 9, 0, tzinfo=UTC),
            quote_timestamp=datetime(2026, 9, 8, 9, 0, tzinfo=UTC),
        ),
        make_product_snapshot(
            isin="DE000ABC1234",
            financing_level=18001.0,
            observation_time=datetime(2026, 9, 8, 17, 0, tzinfo=UTC),
            quote_timestamp=datetime(2026, 9, 8, 17, 0, tzinfo=UTC),
        ),
        make_product_snapshot(
            isin="DE000ABC1234",
            financing_level=18010.0,
            observation_time=datetime(2026, 9, 9, 9, 0, tzinfo=UTC),
            quote_timestamp=datetime(2026, 9, 9, 9, 0, tzinfo=UTC),
        ),
    ]
    store.append_product_snapshots(snaps)
    history = store.financing_level_history("DE000ABC1234")
    assert len(history) == 2  # one per calendar day
    assert history[0][1] == 18001.0  # last observation of 2026-09-08
    assert history[1][1] == 18010.0
    assert history[0][0] < history[1][0]


def test_positions_lifecycle(store: Store) -> None:
    position = ManualPosition(
        wkn="ABC123",
        isin="DE000ABC1234",
        qty=100,
        entry_price=4.86,
        entry_date=date(2026, 9, 10),
        created_at=datetime(2026, 9, 10, tzinfo=UTC),
        updated_at=datetime(2026, 9, 10, tzinfo=UTC),
    )
    store.insert_position(position)

    open_positions = store.list_positions(PositionStatus.OPEN)
    assert len(open_positions) == 1
    assert open_positions[0].wkn == "ABC123"

    closed = store.close_position("ABC123", exit_price=5.42, exit_date=date(2026, 9, 15))
    assert closed.status == PositionStatus.CLOSED
    assert closed.exit_price == 5.42

    assert store.list_positions(PositionStatus.OPEN) == []
    assert len(store.list_positions(PositionStatus.CLOSED)) == 1


def test_close_position_without_open_lot_raises(store: Store) -> None:
    with pytest.raises(StoreError, match="no open position"):
        store.close_position("NOPE", exit_price=1.0, exit_date=date(2026, 9, 15))


def test_close_position_ambiguous_multiple_open_lots_raises(store: Store) -> None:
    for i in range(2):
        store.insert_position(
            ManualPosition(
                wkn="DUP123",
                isin="DE000ABC1234",
                qty=10,
                entry_price=1.0 + i,
                entry_date=date(2026, 9, 10),
                created_at=datetime(2026, 9, 10, tzinfo=UTC),
                updated_at=datetime(2026, 9, 10, tzinfo=UTC),
            )
        )
    with pytest.raises(StoreError, match="ambiguous"):
        store.close_position("DUP123", exit_price=2.0, exit_date=date(2026, 9, 15))


def test_notification_dedup(store: Store) -> None:
    assert store.notification_already_sent("hash-1") is False
    store.record_notification(
        NotificationRecord(
            notification_hash="hash-1",
            candidate_id="cand-1",
            category="WATCH",
            sent_at=datetime(2026, 9, 10, tzinfo=UTC),
            subject="test",
        )
    )
    assert store.notification_already_sent("hash-1") is True
    # recording the same hash again must not raise (ON CONFLICT DO NOTHING)
    store.record_notification(
        NotificationRecord(
            notification_hash="hash-1",
            candidate_id="cand-1",
            category="WATCH",
            sent_at=datetime(2026, 9, 10, tzinfo=UTC),
            subject="test again",
        )
    )


def test_store_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "turboedge.duckdb"
    with Store(nested) as s:
        s.init_schema()
    assert nested.exists()


# --------------------------------------------------------------------------
# Regression coverage: the DuckDB append paths (`append_product_snapshots`,
# `append_candidates`) go through `executemany` with parametrized `?`
# placeholders against a fixed DDL schema (never a pandas/polars DataFrame),
# so there is no dtype inference step for them to get wrong. These tests
# exercise the same large, heterogeneous (first-200-fields-None) batches used
# for the storage/snapshots.py Parquet regression to confirm that holds and
# every row round-trips.
# --------------------------------------------------------------------------


def test_append_product_snapshots_large_heterogeneous_batch(
    store: Store,
    tmp_path: Path,
    make_large_product_snapshot_batch: Callable[[int], list[ProductSnapshot]],
) -> None:
    records = make_large_product_snapshot_batch(5000)

    n = store.append_product_snapshots(records)

    assert n == 5000
    assert store.table_counts()["product_snapshots"] == 5000

    rows = store._conn.execute(
        "SELECT quanto, quote_presence, ask, quote_timestamp, maturity "
        "FROM product_snapshots ORDER BY isin ASC LIMIT 200"
    ).fetchall()
    assert len(rows) == 200
    assert all(row == (None, None, None, None, None) for row in rows)

    populated = store._conn.execute(
        "SELECT count(*) FROM product_snapshots WHERE quanto IS NOT NULL"
    ).fetchone()
    assert populated is not None
    assert populated[0] == 4800


def test_append_candidates_large_heterogeneous_batch(
    store: Store,
    make_large_candidate_batch: Callable[[int], list[CandidateEvaluation]],
) -> None:
    records = make_large_candidate_batch(5000)

    n = store.append_candidates(records)

    assert n == 5000
    assert store.table_counts()["candidate_sets"] == 5000

    fetched = store.list_candidates("run-large")
    assert len(fetched) == 5000
    assert fetched == sorted(fetched, key=lambda c: c.candidate_id)
    # spot-check the nested `costs` field round-trips through the JSON VARCHAR column
    populated = [c for c in fetched if c.costs is not None]
    assert len(populated) == 4800
    assert populated[0].costs is not None
    assert populated[0].costs.ask == 4.86


# --------------------------------------------------------------------------
# Additive schema migration: `Store.init_schema()` must be able to bring a
# DuckDB file created by an OLDER version of this codebase's DDL up to date
# (the real-world trigger: `scan-report.yml` restores `state/turboedge.duckdb`
# from the GitHub Actions cache across runs, so a file on disk can predate a
# newly added, nullable ProductSnapshot field like
# `underlying_price_ref_timestamp`) without dropping data or requiring a
# fresh state dir.
# --------------------------------------------------------------------------

# The pre-migration `product_snapshots` DDL -- byte-for-byte the current one
# in `storage/duckdb.py` minus the `underlying_price_ref_timestamp` column,
# reproducing exactly the on-disk shape a DuckDB file cached from a run
# before that field was added would have.
_OLD_PRODUCT_SNAPSHOTS_DDL = """
    CREATE TABLE product_snapshots (
        isin VARCHAR NOT NULL,
        wkn VARCHAR,
        issuer VARCHAR NOT NULL,
        venue VARCHAR NOT NULL,
        underlying_raw VARCHAR NOT NULL,
        underlying_id VARCHAR,
        direction VARCHAR NOT NULL,
        product_type VARCHAR NOT NULL,
        financing_level DOUBLE,
        knockout_barrier DOUBLE,
        ratio DOUBLE NOT NULL,
        currency VARCHAR NOT NULL,
        underlying_currency VARCHAR,
        quanto BOOLEAN,
        open_end BOOLEAN NOT NULL,
        maturity DATE,
        first_trading_day DATE,
        bid DOUBLE,
        ask DOUBLE,
        bid_size DOUBLE,
        ask_size DOUBLE,
        quote_timestamp TIMESTAMPTZ,
        quote_presence BOOLEAN,
        bid_only BOOLEAN NOT NULL,
        knocked_out BOOLEAN NOT NULL,
        trading_hours VARCHAR,
        product_age_days INTEGER,
        underlying_price_ref DOUBLE,
        raw_hash VARCHAR NOT NULL,
        observation_time TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        retrieved_at TIMESTAMPTZ NOT NULL,
        source_timestamp TIMESTAMPTZ,
        source VARCHAR NOT NULL,
        schema_version VARCHAR NOT NULL,
        parser_version VARCHAR NOT NULL,
        is_stale BOOLEAN NOT NULL,
        quality_score DOUBLE NOT NULL
    )
    """

_OLD_ROW_TIMESTAMP = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def _seed_old_product_snapshots_db(path: Path) -> None:
    """Create a DuckDB file at ``path`` with the pre-migration
    ``product_snapshots`` schema (no ``underlying_price_ref_timestamp``) and
    one row in it -- simulating exactly what CI restores from its cache."""
    conn = duckdb.connect(str(path))
    try:
        conn.execute(_OLD_PRODUCT_SNAPSHOTS_DDL)
        conn.execute(
            """
            INSERT INTO product_snapshots (
                isin, wkn, issuer, venue, underlying_raw, underlying_id, direction,
                product_type, financing_level, knockout_barrier, ratio, currency,
                underlying_currency, quanto, open_end, maturity, first_trading_day,
                bid, ask, bid_size, ask_size, quote_timestamp, quote_presence,
                bid_only, knocked_out, trading_hours, product_age_days,
                underlying_price_ref, raw_hash, observation_time, available_at,
                retrieved_at, source_timestamp, source, schema_version,
                parser_version, is_stale, quality_score
            ) VALUES (
                'DE000OLD0001', 'OLD001', 'TestBank', 'stuttgart', 'DAX', 'DAX',
                'long', 'turbo_open_end', 18000.0, 18000.0, 0.01, 'EUR', 'EUR',
                false, true, NULL, NULL, 4.80, 4.86, 1000.0, 1000.0, ?, true,
                false, false, '09:00-22:00', 100, 18500.0, 'oldhash',
                ?, ?, ?, ?, 'test_source', '1.0.0', '1', false, 0.9
            )
            """,
            [
                _OLD_ROW_TIMESTAMP,
                _OLD_ROW_TIMESTAMP,
                _OLD_ROW_TIMESTAMP,
                _OLD_ROW_TIMESTAMP,
                _OLD_ROW_TIMESTAMP,
            ],
        )
    finally:
        conn.close()


def test_init_schema_adds_missing_column_to_old_product_snapshots_table(
    tmp_path: Path,
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    db_path = tmp_path / "turboedge.duckdb"
    _seed_old_product_snapshots_db(db_path)

    with Store(db_path) as store:
        store.init_schema()

        # the missing column now exists, and the pre-existing row has NULL
        # for it (never guessed/backfilled) rather than the insert failing
        # or the column being silently skipped
        row = store._conn.execute(
            "SELECT isin, underlying_price_ref_timestamp FROM product_snapshots "
            "WHERE isin = 'DE000OLD0001'"
        ).fetchone()
        assert row is not None
        assert row[0] == "DE000OLD0001"
        assert row[1] is None

        # a fresh, fully-populated snapshot (with the new field set) can now
        # be appended without error
        new_snapshot = make_product_snapshot(isin="DE000NEW0001")
        assert new_snapshot.underlying_price_ref_timestamp is not None
        n = store.append_product_snapshots([new_snapshot])
        assert n == 1
        assert store.table_counts()["product_snapshots"] == 2

        # the migration was logged
        migrations = store.list_schema_migrations()
        assert (
            migrations[-1][1],
            migrations[-1][2],
            migrations[-1][3],
        ) == ("product_snapshots", "financing_rate", "add_column")
        migrations_after_first_call = len(migrations)

        # a second init_schema() call is a no-op: the column already exists,
        # so no new migration is logged
        store.init_schema()
        assert len(store.list_schema_migrations()) == migrations_after_first_call


def test_init_schema_migration_check_runs_for_every_managed_table(
    tmp_path: Path,
) -> None:
    """Not just `product_snapshots`: every table `Store` manages (e.g.
    `candidate_sets`) goes through the same additive-migration check, so a
    DuckDB file missing a column on any of them would be repaired the same
    way."""
    db_path = tmp_path / "turboedge.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        # pre-migration `candidate_sets`, missing `cost_rank_score` (the
        # last column in the current DDL).
        conn.execute(
            """
            CREATE TABLE candidate_sets (
                run_id VARCHAR NOT NULL,
                candidate_id VARCHAR NOT NULL,
                isin VARCHAR NOT NULL,
                wkn VARCHAR,
                issuer VARCHAR NOT NULL,
                underlying_id VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                reasons VARCHAR NOT NULL,
                leverage DOUBLE,
                leverage_bucket VARCHAR,
                distance_to_barrier_pct DOUBLE,
                distance_to_barrier_sigma DOUBLE,
                costs VARCHAR,
                realized_financing_spread DOUBLE,
                financing_cost_horizon_pct VARCHAR NOT NULL,
                cross_issuer_residual_zscore DOUBLE,
                issuer_markup_score DOUBLE,
                quote_dislocation_score DOUBLE,
                wrapper_edge DOUBLE,
                liquidity_factor DOUBLE,
                integrity_passed BOOLEAN NOT NULL,
                lcb_ev DOUBLE,
                PRIMARY KEY (run_id, candidate_id)
            )
            """
        )
    finally:
        conn.close()

    with Store(db_path) as store:
        store.init_schema()
        columns = {
            row[1] for row in store._conn.execute("PRAGMA table_info(candidate_sets)").fetchall()
        }
        assert "cost_rank_score" in columns
        migrations = store.list_schema_migrations()
        assert ("candidate_sets", "cost_rank_score", "add_column") in [
            (m[1], m[2], m[3]) for m in migrations
        ]


def test_init_schema_raises_on_incompatible_existing_column_type(tmp_path: Path) -> None:
    """A column that exists in both the on-disk table and the current DDL,
    but with a different type, is never auto-migrated -- that would risk
    silently truncating/reinterpreting data. `init_schema()` must refuse
    with a clear `StoreError` instead."""
    db_path = tmp_path / "turboedge.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        # `started_at` is TIMESTAMPTZ in the real DDL; here it is VARCHAR.
        conn.execute(
            """
            CREATE TABLE runs (
                run_id VARCHAR PRIMARY KEY,
                started_at VARCHAR NOT NULL,
                finished_at TIMESTAMPTZ,
                command VARCHAR NOT NULL,
                config_hash VARCHAR NOT NULL,
                git_commit VARCHAR,
                status VARCHAR NOT NULL,
                error VARCHAR
            )
            """
        )
    finally:
        conn.close()

    with Store(db_path) as store, pytest.raises(StoreError, match=r"runs\.started_at"):
        store.init_schema()


# --------------------------------------------------------------------------
# W6: a DuckDB file predating the Forward Ledger / Learning tables (Master
# Spec §20-27, §46) must gain them via `init_schema()`, without touching
# pre-existing tables/data -- the same "state/turboedge.duckdb restored from
# an older cached run" scenario the migration tests above cover for columns,
# but here for whole tables that did not exist at all yet.
# --------------------------------------------------------------------------


def test_init_schema_creates_w6_tables_on_a_pre_w6_database(tmp_path: Path) -> None:
    db_path = tmp_path / "turboedge.duckdb"
    # Simulate a database created before the W6 tables existed: only the
    # original `runs` table, with one pre-existing row.
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE runs (
                run_id VARCHAR PRIMARY KEY,
                started_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                command VARCHAR NOT NULL,
                config_hash VARCHAR NOT NULL,
                git_commit VARCHAR,
                status VARCHAR NOT NULL,
                error VARCHAR
            )
            """
        )
        conn.execute(
            "INSERT INTO runs "
            "(run_id, started_at, command, config_hash, git_commit, status, error) "
            "VALUES ('old-run', '2026-01-01T00:00:00+00:00', 'scan', 'cfg', NULL, 'ok', NULL)"
        )
    finally:
        conn.close()

    with Store(db_path) as store:
        store.init_schema()
        counts = store.table_counts()
        for table in (
            "forward_ledger",
            "ledger_labels",
            "strategy_posteriors",
            "model_registry",
            "model_weight_history",
            "research_trials",
            "drift_events",
            "shadow_portfolio",
        ):
            assert table in counts
            assert counts[table] == 0

        # Pre-existing table/data is untouched.
        assert counts["runs"] == 1
        row = store._conn.execute(
            "SELECT run_id, status FROM runs WHERE run_id = 'old-run'"
        ).fetchone()
        assert row == ("old-run", "ok")

        # A second init_schema() call remains idempotent.
        store.init_schema()
        assert store.table_counts()["forward_ledger"] == 0
