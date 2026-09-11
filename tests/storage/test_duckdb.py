from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

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
