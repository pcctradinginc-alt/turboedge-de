"""DuckDB-backed persistence for every TurboEdge-DE table.

``Store`` owns a single DuckDB connection and exposes one typed method per
table/operation the pipeline needs. All queries are parametrized (DuckDB
``?`` placeholders) - values are never interpolated into SQL strings, table
and column names are always fixed literals defined in this module.

Nested/structured fields (``CandidateEvaluation.costs``,
``financing_cost_horizon_pct``, ``SignalSnapshot.components``,
``CandidateEvaluation.reasons``) are stored as JSON-encoded ``VARCHAR``
columns rather than DuckDB's native ``JSON`` type, to avoid depending on
autoloading the ``json`` extension in offline/sandboxed environments.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb
import structlog

from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    Instrument,
    ManualPosition,
    NotificationRecord,
    PositionStatus,
    ProductSnapshot,
    SignalSnapshot,
    SourceHealthRecord,
    UnderlyingBar,
)

logger = structlog.get_logger(__name__)


class StoreError(Exception):
    """Raised for domain-level storage errors (not found, ambiguous, ...)."""


# --------------------------------------------------------------------------
# Schema DDL
# --------------------------------------------------------------------------

_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL,
        finished_at TIMESTAMPTZ,
        command VARCHAR NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR,
        status VARCHAR NOT NULL,
        error VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS instruments (
        isin VARCHAR PRIMARY KEY,
        wkn VARCHAR,
        issuer VARCHAR NOT NULL,
        underlying_id VARCHAR,
        underlying_raw VARCHAR NOT NULL,
        direction VARCHAR NOT NULL,
        product_type VARCHAR NOT NULL,
        ratio DOUBLE NOT NULL,
        currency VARCHAR NOT NULL,
        underlying_currency VARCHAR,
        quanto BOOLEAN,
        open_end BOOLEAN NOT NULL,
        maturity DATE,
        first_trading_day DATE,
        venue VARCHAR NOT NULL,
        first_seen_at TIMESTAMPTZ NOT NULL,
        last_seen_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS product_snapshots (
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
        underlying_price_ref_timestamp TIMESTAMPTZ,
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
    """,
    """
    CREATE TABLE IF NOT EXISTS underlying_prices (
        underlying_id VARCHAR NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        interval VARCHAR NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        volume DOUBLE,
        observation_time TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        retrieved_at TIMESTAMPTZ NOT NULL,
        source_timestamp TIMESTAMPTZ,
        source VARCHAR NOT NULL,
        schema_version VARCHAR NOT NULL,
        parser_version VARCHAR NOT NULL,
        is_stale BOOLEAN NOT NULL,
        quality_score DOUBLE NOT NULL,
        PRIMARY KEY (underlying_id, ts, interval, source)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        signal_id VARCHAR NOT NULL,
        signal_version_hash VARCHAR NOT NULL,
        underlying_id VARCHAR NOT NULL,
        prediction_time TIMESTAMPTZ NOT NULL,
        frozen_at TIMESTAMPTZ NOT NULL,
        score DOUBLE NOT NULL,
        components VARCHAR NOT NULL,
        direction_hint VARCHAR,
        threshold DOUBLE NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR,
        data_snapshot_hash VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_sets (
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
        cost_rank_score DOUBLE,
        PRIMARY KEY (run_id, candidate_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_health (
        source VARCHAR NOT NULL,
        checked_at TIMESTAMPTZ NOT NULL,
        availability DOUBLE NOT NULL,
        freshness DOUBLE NOT NULL,
        missingness DOUBLE NOT NULL,
        schema_consistency DOUBLE NOT NULL,
        cross_source_agreement DOUBLE,
        score DOUBLE NOT NULL,
        status VARCHAR NOT NULL,
        message VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions_manual (
        position_id VARCHAR PRIMARY KEY,
        wkn VARCHAR NOT NULL,
        isin VARCHAR,
        qty DOUBLE NOT NULL,
        entry_price DOUBLE NOT NULL,
        entry_date DATE NOT NULL,
        exit_price DOUBLE,
        exit_date DATE,
        status VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notifications_sent (
        notification_hash VARCHAR PRIMARY KEY,
        candidate_id VARCHAR,
        category VARCHAR NOT NULL,
        sent_at TIMESTAMPTZ NOT NULL,
        subject VARCHAR NOT NULL
    )
    """,
)

_ALL_TABLES: tuple[str, ...] = (
    "runs",
    "instruments",
    "product_snapshots",
    "underlying_prices",
    "signals",
    "candidate_sets",
    "source_health",
    "positions_manual",
    "notifications_sent",
)

# Append-only log of every additive column migration `Store.init_schema()`
# has ever applied (see `_migrate_table_columns` below). Deliberately NOT
# part of `_ALL_TABLES`/`table_counts()` -- it is bookkeeping metadata about
# the schema itself, not a pipeline data table.
_SCHEMA_MIGRATIONS_DDL = """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        applied_at TIMESTAMPTZ NOT NULL,
        table_name VARCHAR NOT NULL,
        column_name VARCHAR NOT NULL,
        action VARCHAR NOT NULL
    )
    """


# --------------------------------------------------------------------------
# additive schema migration helpers
# --------------------------------------------------------------------------


def _actual_table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """``{column_name: canonical_type}`` for a table that already exists.

    ``table`` is always one of the fixed literals in ``_ALL_TABLES`` -- never
    user input -- so interpolating it into the ``PRAGMA`` call is safe (same
    pattern as ``Store.table_counts``).
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1]: row[2] for row in rows}


def _probe_expected_columns(
    conn: duckdb.DuckDBPyConnection, table: str, ddl: str
) -> dict[str, str]:
    """``{column_name: canonical_type}`` that ``ddl`` declares for ``table``.

    ``ddl`` (one of ``_DDL_STATEMENTS``) is the single source of truth for a
    table's current, code-defined schema -- rather than hand-maintaining a
    second, parallel column list that could drift from it, this creates a
    throwaway ``TEMP TABLE`` from the *same* DDL text (table name swapped for
    a private probe name) and reads its columns back via
    ``PRAGMA table_info``, which is DuckDB's own canonicalization of the
    declared types (e.g. ``TIMESTAMPTZ`` -> ``TIMESTAMP WITH TIME ZONE``) --
    the same canonicalization ``_actual_table_columns`` reads from the real
    table, so the two are directly comparable. The probe table is dropped
    immediately after; it never persists past this call.
    """
    probe_name = f"__schema_probe_{table}"
    probe_ddl, n = re.subn(
        rf"CREATE TABLE IF NOT EXISTS {re.escape(table)}\b",
        f"CREATE TEMP TABLE IF NOT EXISTS {probe_name}",
        ddl,
        count=1,
    )
    if n != 1:
        raise AssertionError(
            f"could not derive a schema probe for table {table!r} from its DDL; "
            "the DDL text no longer matches the expected "
            "'CREATE TABLE IF NOT EXISTS <table> (' shape"
        )
    conn.execute(f"DROP TABLE IF EXISTS {probe_name}")
    try:
        conn.execute(probe_ddl)
        rows = conn.execute(f"PRAGMA table_info({probe_name})").fetchall()
    finally:
        conn.execute(f"DROP TABLE IF EXISTS {probe_name}")
    return {row[1]: row[2] for row in rows}


# --------------------------------------------------------------------------
# datetime helpers
# --------------------------------------------------------------------------


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _opt_to_utc(value: datetime | None) -> datetime | None:
    return _to_utc(value) if value is not None else None


def _from_db_dt(value: Any) -> datetime:
    """DuckDB may return a naive datetime for TIMESTAMPTZ depending on driver
    version/settings; we always write UTC, so a naive value is assumed UTC."""
    if not isinstance(value, datetime):
        raise TypeError(f"expected datetime from DuckDB, got {type(value)!r}")
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _from_db_opt_dt(value: Any) -> datetime | None:
    return _from_db_dt(value) if value is not None else None


class Store:
    """Owns one DuckDB connection and implements all TurboEdge-DE table access.

    Usage::

        with Store(state_dir / "turboedge.duckdb") as store:
            store.init_schema()
            store.append_product_snapshots(snapshots)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- schema ------------------------------------------------------------

    def init_schema(self) -> None:
        """Create every table if it does not already exist, then run any
        additive column migration each one still needs. Idempotent.

        ``CREATE TABLE IF NOT EXISTS`` alone does not add columns to a table
        that already exists -- which matters here because ``state/*.duckdb``
        is restored from the GitHub Actions cache (``scan-report.yml``)
        across runs. A DDL change like ``ProductSnapshot`` gaining
        ``underlying_price_ref_timestamp`` is invisible to a table created by
        an older cached run until this migration step adds the missing
        column explicitly. Column removal or type changes are NEVER applied
        automatically (see :func:`_migrate_table_columns`) -- only additive,
        nullable ``ALTER TABLE ... ADD COLUMN`` ever runs here.
        """
        self._conn.execute(_SCHEMA_MIGRATIONS_DDL)
        for table, statement in zip(_ALL_TABLES, _DDL_STATEMENTS, strict=True):
            self._conn.execute(statement)
            self._migrate_table_columns(table, statement)

    def _migrate_table_columns(self, table: str, ddl: str) -> None:
        """Additively migrate one table's columns to match ``ddl``.

        Compares the table's actual columns (``PRAGMA table_info``) against
        the columns ``ddl`` declares (derived from ``ddl`` itself via a
        throwaway ``TEMP TABLE`` probe, so the DDL string stays the single
        source of truth -- no parallel column list to keep in sync). Any
        column present in ``ddl`` but missing from the table is added via a
        nullable ``ALTER TABLE ... ADD COLUMN`` and logged to
        ``schema_migrations``. A column present in both with a different
        type raises :class:`StoreError` -- this migration only ever adds
        columns, it never changes or drops one.
        """
        expected = _probe_expected_columns(self._conn, table, ddl)
        actual = _actual_table_columns(self._conn, table)

        for name, col_type in expected.items():
            if name in actual:
                if actual[name].upper() != col_type.upper():
                    raise StoreError(
                        f"column {table}.{name} has type {actual[name]!r} in "
                        f"{self.path}, but the current schema expects "
                        f"{col_type!r}. Automatic migration only ever adds "
                        "missing columns -- it never changes or drops an "
                        "existing one. Resolve this manually (e.g. a "
                        "one-off migration script, or a fresh state dir)."
                    )
                continue
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
            applied_at = datetime.now(UTC)
            self._conn.execute(
                "INSERT INTO schema_migrations "
                "(applied_at, table_name, column_name, action) VALUES (?, ?, ?, ?)",
                [applied_at, table, name, "add_column"],
            )
            logger.info(
                "schema_migration_applied",
                table=table,
                column=name,
                action="add_column",
                column_type=col_type,
            )

    def table_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table in _ALL_TABLES:
            row = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()
            counts[table] = int(row[0]) if row is not None else 0
        return counts

    def list_schema_migrations(self) -> list[tuple[datetime, str, str, str]]:
        """Every additive column migration ``init_schema()`` has applied so
        far, as ``(applied_at, table_name, column_name, action)`` tuples,
        oldest first."""
        rows = self._conn.execute(
            "SELECT applied_at, table_name, column_name, action "
            "FROM schema_migrations ORDER BY applied_at ASC"
        ).fetchall()
        return [(_from_db_dt(row[0]), row[1], row[2], row[3]) for row in rows]

    # -- runs ----------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        *,
        command: str,
        config_hash: str,
        git_commit: str | None,
        started_at: datetime | None = None,
    ) -> None:
        started = _to_utc(started_at if started_at is not None else datetime.now(UTC))
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, started_at, finished_at, command, config_hash, git_commit, status, error
            )
            VALUES (?, ?, NULL, ?, ?, ?, 'running', NULL)
            """,
            [run_id, started, command, config_hash, git_commit],
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        finished = _to_utc(finished_at if finished_at is not None else datetime.now(UTC))
        self._conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, error = ? WHERE run_id = ?",
            [finished, status, error, run_id],
        )

    # -- product snapshots ---------------------------------------------------

    def append_product_snapshots(self, snapshots: Sequence[ProductSnapshot]) -> int:
        if not snapshots:
            return 0
        rows = [_product_snapshot_row(s) for s in snapshots]
        self._conn.executemany(
            f"INSERT INTO product_snapshots ({', '.join(_PRODUCT_SNAPSHOT_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_PRODUCT_SNAPSHOT_COLUMNS))})",
            rows,
        )
        return len(rows)

    # -- instruments -----------------------------------------------------------

    def upsert_instruments(self, snapshots: Sequence[ProductSnapshot]) -> int:
        """Refresh the `instruments` master table from a batch of snapshots.

        For each distinct ISIN in ``snapshots`` (keeping the snapshot with the
        latest ``observation_time``), preserve the existing ``first_seen_at``
        if a row already exists, else set it from the snapshot.
        """
        if not snapshots:
            return 0
        latest_by_isin: dict[str, ProductSnapshot] = {}
        for snap in snapshots:
            current = latest_by_isin.get(snap.isin)
            if current is None or snap.observation_time > current.observation_time:
                latest_by_isin[snap.isin] = snap

        count = 0
        for isin, snap in latest_by_isin.items():
            existing = self._conn.execute(
                "SELECT first_seen_at FROM instruments WHERE isin = ?", [isin]
            ).fetchone()
            first_seen = _from_db_dt(existing[0]) if existing is not None else None
            instrument = Instrument.from_snapshot(snap, first_seen_at=first_seen)
            self._conn.execute(
                """
                INSERT INTO instruments (
                    isin, wkn, issuer, underlying_id, underlying_raw, direction, product_type,
                    ratio, currency, underlying_currency, quanto, open_end, maturity,
                    first_trading_day, venue, first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (isin) DO UPDATE SET
                    wkn = excluded.wkn,
                    issuer = excluded.issuer,
                    underlying_id = excluded.underlying_id,
                    underlying_raw = excluded.underlying_raw,
                    direction = excluded.direction,
                    product_type = excluded.product_type,
                    ratio = excluded.ratio,
                    currency = excluded.currency,
                    underlying_currency = excluded.underlying_currency,
                    quanto = excluded.quanto,
                    open_end = excluded.open_end,
                    maturity = excluded.maturity,
                    first_trading_day = excluded.first_trading_day,
                    venue = excluded.venue,
                    last_seen_at = excluded.last_seen_at
                """,
                [
                    instrument.isin,
                    instrument.wkn,
                    instrument.issuer,
                    instrument.underlying_id,
                    instrument.underlying_raw,
                    instrument.direction.value,
                    instrument.product_type.value,
                    instrument.ratio,
                    instrument.currency,
                    instrument.underlying_currency,
                    instrument.quanto,
                    instrument.open_end,
                    instrument.maturity,
                    instrument.first_trading_day,
                    instrument.venue,
                    _to_utc(instrument.first_seen_at),
                    _to_utc(instrument.last_seen_at),
                ],
            )
            count += 1
        return count

    # -- underlying bars ---------------------------------------------------

    def append_underlying_bars(self, bars: Sequence[UnderlyingBar]) -> int:
        if not bars:
            return 0
        rows = [_underlying_bar_row(b) for b in bars]
        self._conn.executemany(
            """
            INSERT INTO underlying_prices (
                underlying_id, ts, interval, open, high, low, close, volume,
                observation_time, available_at, retrieved_at, source_timestamp,
                source, schema_version, parser_version, is_stale, quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (underlying_id, ts, interval, source) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                observation_time = excluded.observation_time,
                available_at = excluded.available_at,
                retrieved_at = excluded.retrieved_at,
                source_timestamp = excluded.source_timestamp,
                schema_version = excluded.schema_version,
                parser_version = excluded.parser_version,
                is_stale = excluded.is_stale,
                quality_score = excluded.quality_score
            """,
            rows,
        )
        return len(rows)

    def latest_underlying_bars(
        self, underlying_id: str, n: int, *, interval: str = "1d"
    ) -> list[UnderlyingBar]:
        rows = self._conn.execute(
            """
            SELECT underlying_id, ts, interval, open, high, low, close, volume,
                   observation_time, available_at, retrieved_at, source_timestamp,
                   source, schema_version, parser_version, is_stale, quality_score
            FROM underlying_prices
            WHERE underlying_id = ? AND interval = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            [underlying_id, interval, n],
        ).fetchall()
        bars = [_row_to_underlying_bar(row) for row in rows]
        bars.reverse()  # chronological order, oldest first
        return bars

    # -- signals ---------------------------------------------------------------

    def append_signal(self, signal: SignalSnapshot) -> None:
        self._conn.execute(
            """
            INSERT INTO signals (
                signal_id, signal_version_hash, underlying_id, prediction_time, frozen_at,
                score, components, direction_hint, threshold, config_hash, git_commit,
                data_snapshot_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                signal.signal_id,
                signal.signal_version_hash,
                signal.underlying_id,
                _to_utc(signal.prediction_time),
                _to_utc(signal.frozen_at),
                signal.score,
                json.dumps(signal.components, sort_keys=True),
                signal.direction_hint.value if signal.direction_hint is not None else None,
                signal.threshold,
                signal.config_hash,
                signal.git_commit,
                signal.data_snapshot_hash,
            ],
        )

    # -- candidate evaluations ---------------------------------------------

    def append_candidates(self, candidates: Sequence[CandidateEvaluation]) -> int:
        if not candidates:
            return 0
        rows = [_candidate_row(c) for c in candidates]
        self._conn.executemany(
            f"INSERT INTO candidate_sets ({', '.join(_CANDIDATE_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_CANDIDATE_COLUMNS))})",
            rows,
        )
        return len(rows)

    def list_candidates(self, run_id: str | None = None) -> list[CandidateEvaluation]:
        if run_id is None:
            rows = self._conn.execute(
                f"SELECT {', '.join(_CANDIDATE_COLUMNS)} FROM candidate_sets "
                "ORDER BY run_id ASC, candidate_id ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(_CANDIDATE_COLUMNS)} FROM candidate_sets "
                "WHERE run_id = ? ORDER BY candidate_id ASC",
                [run_id],
            ).fetchall()
        return [_row_to_candidate(row) for row in rows]

    # -- source health -----------------------------------------------------

    def append_source_health(self, records: Sequence[SourceHealthRecord]) -> int:
        if not records:
            return 0
        rows = [
            (
                r.source,
                _to_utc(r.checked_at),
                r.availability,
                r.freshness,
                r.missingness,
                r.schema_consistency,
                r.cross_source_agreement,
                r.score,
                r.status.value,
                r.message,
            )
            for r in records
        ]
        self._conn.executemany(
            """
            INSERT INTO source_health (
                source, checked_at, availability, freshness, missingness,
                schema_consistency, cross_source_agreement, score, status, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    # -- financing level history --------------------------------------------

    def financing_level_history(self, isin: str) -> list[tuple[datetime, float]]:
        """One (timestamp, financing_level) observation per calendar day.

        When multiple snapshots exist for the same UTC calendar day, the one
        with the latest timestamp (``quote_timestamp`` if present, else
        ``observation_time``) wins. Ordered ascending by day.
        """
        rows = self._conn.execute(
            """
            WITH dated AS (
                SELECT
                    COALESCE(quote_timestamp, observation_time) AS ts,
                    financing_level
                FROM product_snapshots
                WHERE isin = ? AND financing_level IS NOT NULL
            ),
            ranked AS (
                SELECT ts, financing_level,
                       ROW_NUMBER() OVER (
                           PARTITION BY CAST(ts AS DATE)
                           ORDER BY ts DESC
                       ) AS rn
                FROM dated
            )
            SELECT ts, financing_level FROM ranked WHERE rn = 1 ORDER BY ts ASC
            """,
            [isin],
        ).fetchall()
        return [(_from_db_dt(row[0]), float(row[1])) for row in rows]

    # -- manual positions ----------------------------------------------------

    def insert_position(self, position: ManualPosition) -> None:
        self._conn.execute(
            """
            INSERT INTO positions_manual (
                position_id, wkn, isin, qty, entry_price, entry_date,
                exit_price, exit_date, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                position.position_id,
                position.wkn,
                position.isin,
                position.qty,
                position.entry_price,
                position.entry_date,
                position.exit_price,
                position.exit_date,
                position.status.value,
                _to_utc(position.created_at),
                _to_utc(position.updated_at),
            ],
        )

    def list_positions(self, status: PositionStatus | None = None) -> list[ManualPosition]:
        if status is None:
            rows = self._conn.execute(
                f"SELECT {', '.join(_POSITION_COLUMNS)} FROM positions_manual "
                "ORDER BY created_at ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(_POSITION_COLUMNS)} FROM positions_manual "
                "WHERE status = ? ORDER BY created_at ASC",
                [status.value],
            ).fetchall()
        return [_row_to_position(row) for row in rows]

    def close_position(self, wkn: str, exit_price: float, exit_date: date) -> ManualPosition:
        open_rows = self._conn.execute(
            f"SELECT {', '.join(_POSITION_COLUMNS)} FROM positions_manual "
            "WHERE wkn = ? AND status = ?",
            [wkn, PositionStatus.OPEN.value],
        ).fetchall()
        if not open_rows:
            raise StoreError(f"no open position found for wkn={wkn!r}")
        if len(open_rows) > 1:
            ids = [row[0] for row in open_rows]
            raise StoreError(
                f"ambiguous close: {len(open_rows)} open positions for wkn={wkn!r} ({ids}); "
                "this ledger only supports closing by wkn when exactly one lot is open"
            )
        position = _row_to_position(open_rows[0])
        now = datetime.now(UTC)
        self._conn.execute(
            """
            UPDATE positions_manual
            SET exit_price = ?, exit_date = ?, status = ?, updated_at = ?
            WHERE position_id = ?
            """,
            [exit_price, exit_date, PositionStatus.CLOSED.value, now, position.position_id],
        )
        return position.model_copy(
            update={
                "exit_price": exit_price,
                "exit_date": exit_date,
                "status": PositionStatus.CLOSED,
                "updated_at": now,
            }
        )

    # -- notification dedup --------------------------------------------------

    def notification_already_sent(self, notification_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM notifications_sent WHERE notification_hash = ?",
            [notification_hash],
        ).fetchone()
        return row is not None

    def record_notification(self, record: NotificationRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO notifications_sent (
                notification_hash, candidate_id, category, sent_at, subject
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (notification_hash) DO NOTHING
            """,
            [
                record.notification_hash,
                record.candidate_id,
                record.category,
                _to_utc(record.sent_at),
                record.subject,
            ],
        )


# --------------------------------------------------------------------------
# row (de)serialization helpers
# --------------------------------------------------------------------------

_PRODUCT_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "isin",
    "wkn",
    "issuer",
    "venue",
    "underlying_raw",
    "underlying_id",
    "direction",
    "product_type",
    "financing_level",
    "knockout_barrier",
    "ratio",
    "currency",
    "underlying_currency",
    "quanto",
    "open_end",
    "maturity",
    "first_trading_day",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "quote_timestamp",
    "quote_presence",
    "bid_only",
    "knocked_out",
    "trading_hours",
    "product_age_days",
    "underlying_price_ref",
    "underlying_price_ref_timestamp",
    "raw_hash",
    "observation_time",
    "available_at",
    "retrieved_at",
    "source_timestamp",
    "source",
    "schema_version",
    "parser_version",
    "is_stale",
    "quality_score",
)


def _product_snapshot_row(s: ProductSnapshot) -> tuple[Any, ...]:
    return (
        s.isin,
        s.wkn,
        s.issuer,
        s.venue,
        s.underlying_raw,
        s.underlying_id,
        s.direction.value,
        s.product_type.value,
        s.financing_level,
        s.knockout_barrier,
        s.ratio,
        s.currency,
        s.underlying_currency,
        s.quanto,
        s.open_end,
        s.maturity,
        s.first_trading_day,
        s.bid,
        s.ask,
        s.bid_size,
        s.ask_size,
        _opt_to_utc(s.quote_timestamp),
        s.quote_presence,
        s.bid_only,
        s.knocked_out,
        s.trading_hours,
        s.product_age_days,
        s.underlying_price_ref,
        _opt_to_utc(s.underlying_price_ref_timestamp),
        s.raw_hash,
        _to_utc(s.observation_time),
        _to_utc(s.available_at),
        _to_utc(s.retrieved_at),
        _opt_to_utc(s.source_timestamp),
        s.source,
        s.schema_version,
        s.parser_version,
        s.is_stale,
        s.quality_score,
    )


def _underlying_bar_row(b: UnderlyingBar) -> tuple[Any, ...]:
    return (
        b.underlying_id,
        _to_utc(b.ts),
        b.interval,
        b.open,
        b.high,
        b.low,
        b.close,
        b.volume,
        _to_utc(b.observation_time),
        _to_utc(b.available_at),
        _to_utc(b.retrieved_at),
        _opt_to_utc(b.source_timestamp),
        b.source,
        b.schema_version,
        b.parser_version,
        b.is_stale,
        b.quality_score,
    )


def _row_to_underlying_bar(row: tuple[Any, ...]) -> UnderlyingBar:
    return UnderlyingBar(
        underlying_id=row[0],
        ts=_from_db_dt(row[1]),
        interval=row[2],
        open=row[3],
        high=row[4],
        low=row[5],
        close=row[6],
        volume=row[7],
        observation_time=_from_db_dt(row[8]),
        available_at=_from_db_dt(row[9]),
        retrieved_at=_from_db_dt(row[10]),
        source_timestamp=_from_db_opt_dt(row[11]),
        source=row[12],
        schema_version=row[13],
        parser_version=row[14],
        is_stale=row[15],
        quality_score=row[16],
    )


_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "candidate_id",
    "isin",
    "wkn",
    "issuer",
    "underlying_id",
    "direction",
    "category",
    "reasons",
    "leverage",
    "leverage_bucket",
    "distance_to_barrier_pct",
    "distance_to_barrier_sigma",
    "costs",
    "realized_financing_spread",
    "financing_cost_horizon_pct",
    "cross_issuer_residual_zscore",
    "issuer_markup_score",
    "quote_dislocation_score",
    "wrapper_edge",
    "liquidity_factor",
    "integrity_passed",
    "lcb_ev",
    "cost_rank_score",
)


def _candidate_row(c: CandidateEvaluation) -> tuple[Any, ...]:
    return (
        c.run_id,
        c.candidate_id,
        c.isin,
        c.wkn,
        c.issuer,
        c.underlying_id,
        c.direction.value,
        c.category.value,
        json.dumps(c.reasons),
        c.leverage,
        c.leverage_bucket,
        c.distance_to_barrier_pct,
        c.distance_to_barrier_sigma,
        json.dumps(c.costs.model_dump(mode="json")) if c.costs is not None else None,
        c.realized_financing_spread,
        json.dumps(c.financing_cost_horizon_pct, sort_keys=True),
        c.cross_issuer_residual_zscore,
        c.issuer_markup_score,
        c.quote_dislocation_score,
        c.wrapper_edge,
        c.liquidity_factor,
        c.integrity_passed,
        c.lcb_ev,
        c.cost_rank_score,
    )


def _row_to_candidate(row: tuple[Any, ...]) -> CandidateEvaluation:
    costs_json = row[13]
    return CandidateEvaluation(
        run_id=row[0],
        candidate_id=row[1],
        isin=row[2],
        wkn=row[3],
        issuer=row[4],
        underlying_id=row[5],
        direction=Direction(row[6]),
        category=Category(row[7]),
        reasons=json.loads(row[8]),
        leverage=row[9],
        leverage_bucket=row[10],
        distance_to_barrier_pct=row[11],
        distance_to_barrier_sigma=row[12],
        costs=CostDecomposition(**json.loads(costs_json)) if costs_json is not None else None,
        realized_financing_spread=row[14],
        financing_cost_horizon_pct=json.loads(row[15]),
        cross_issuer_residual_zscore=row[16],
        issuer_markup_score=row[17],
        quote_dislocation_score=row[18],
        wrapper_edge=row[19],
        liquidity_factor=row[20],
        integrity_passed=row[21],
        lcb_ev=row[22],
        cost_rank_score=row[23],
    )


_POSITION_COLUMNS: tuple[str, ...] = (
    "position_id",
    "wkn",
    "isin",
    "qty",
    "entry_price",
    "entry_date",
    "exit_price",
    "exit_date",
    "status",
    "created_at",
    "updated_at",
)


def _row_to_position(row: tuple[Any, ...]) -> ManualPosition:
    return ManualPosition(
        position_id=row[0],
        wkn=row[1],
        isin=row[2],
        qty=row[3],
        entry_price=row[4],
        entry_date=row[5],
        exit_price=row[6],
        exit_date=row[7],
        status=PositionStatus(row[8]),
        created_at=_from_db_dt(row[9]),
        updated_at=_from_db_dt(row[10]),
    )


__all__ = ["Store", "StoreError"]
