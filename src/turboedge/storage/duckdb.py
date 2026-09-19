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
    DriftEvent,
    ExitReason,
    ForecastRecord,
    Instrument,
    KoCalibrationPromotionRecord,
    KoCalibrationResultRecord,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
    ManualPosition,
    ModelRegistryEntry,
    ModelStatus,
    NotificationRecord,
    PositionEvaluation,
    PositionEvaluationStatus,
    PositionStatus,
    ProductSnapshot,
    ProductType,
    ResearchTrial,
    ShadowPortfolioKind,
    ShadowPosition,
    SignalSnapshot,
    SourceHealthRecord,
    TrialStatus,
    UnderlyingBar,
    WalkforwardResultRecord,
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
        financing_rate DOUBLE,
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
        financing_spread_source VARCHAR,
        premium_over_fair DOUBLE,
        premium_uncertainty_term DOUBLE,
        p_ko_raw DOUBLE,
        p_ko_calibrated DOUBLE,
        ko_calibrator_version VARCHAR,
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
    # -- W6: Forward Ledger, Learning & Governance (Master Spec §20-27, §46) --
    """
    CREATE TABLE IF NOT EXISTS forward_ledger (
        entry_id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        candidate_id VARCHAR NOT NULL,
        signal_id VARCHAR NOT NULL,
        signal_version_hash VARCHAR NOT NULL,
        trial_id VARCHAR NOT NULL,
        prediction_time TIMESTAMPTZ NOT NULL,
        underlying VARCHAR NOT NULL,
        direction VARCHAR NOT NULL,
        horizon_days INTEGER NOT NULL,
        regime_bucket VARCHAR,
        cluster_id VARCHAR,
        feature_hash VARCHAR NOT NULL,
        model_hash VARCHAR NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR,
        category VARCHAR NOT NULL,
        selected_wkn VARCHAR,
        selected_isin VARCHAR NOT NULL,
        issuer VARCHAR NOT NULL,
        entry_bid DOUBLE,
        entry_ask DOUBLE NOT NULL,
        entry_spread DOUBLE NOT NULL,
        entry_quote_timestamp TIMESTAMPTZ NOT NULL,
        entry_underlying_timestamp TIMESTAMPTZ NOT NULL,
        financing_level_entry DOUBLE,
        barrier_entry DOUBLE,
        ratio DOUBLE NOT NULL,
        fx DOUBLE NOT NULL,
        predicted_return DOUBLE NOT NULL,
        p_profit DOUBLE NOT NULL,
        p_ko DOUBLE NOT NULL,
        expected_shortfall DOUBLE NOT NULL,
        lcb_ev DOUBLE NOT NULL,
        uncertainty DOUBLE NOT NULL,
        shrinkage_intensity DOUBLE NOT NULL,
        is_shadow BOOLEAN NOT NULL,
        shadow_stratum VARCHAR,
        suggested_position_fraction DOUBLE,
        exit_due DATE NOT NULL,
        alternatives VARCHAR NOT NULL,
        feature_snapshot VARCHAR NOT NULL,
        status VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ledger_labels (
        entry_id VARCHAR PRIMARY KEY,
        labeled_at TIMESTAMPTZ NOT NULL,
        exit_bid DOUBLE,
        exit_quote_timestamp TIMESTAMPTZ,
        financing_level_exit DOUBLE,
        exit_reason VARCHAR NOT NULL,
        realized_selected_pnl DOUBLE,
        underlying_pnl DOUBLE,
        median_turbo_pnl DOUBLE,
        best_turbo_pnl DOUBLE,
        ideal_turbo_pnl DOUBLE,
        mfe DOUBLE,
        mae DOUBLE,
        ko_hit BOOLEAN NOT NULL,
        time_to_ko_days INTEGER,
        ambiguous_path BOOLEAN NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_posteriors (
        signal_family VARCHAR NOT NULL,
        horizon_days INTEGER NOT NULL,
        mu0 DOUBLE NOT NULL,
        kappa DOUBLE NOT NULL,
        alpha DOUBLE NOT NULL,
        beta DOUBLE NOT NULL,
        n DOUBLE NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (signal_family, horizon_days)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_registry (
        model_id VARCHAR PRIMARY KEY,
        model_hash VARCHAR NOT NULL,
        signal_family VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        weight DOUBLE NOT NULL,
        params VARCHAR NOT NULL,
        trial_id VARCHAR,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_weight_history (
        model_id VARCHAR NOT NULL,
        weight DOUBLE NOT NULL,
        utility DOUBLE,
        trial_id VARCHAR,
        recorded_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_trials (
        trial_id VARCHAR PRIMARY KEY,
        kind VARCHAR NOT NULL,
        description VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        quarter VARCHAR NOT NULL,
        status VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS drift_events (
        event_id VARCHAR PRIMARY KEY,
        detected_at TIMESTAMPTZ NOT NULL,
        stream_id VARCHAR NOT NULL,
        signal_family VARCHAR,
        metric VARCHAR NOT NULL,
        ph_statistic DOUBLE NOT NULL,
        threshold DOUBLE NOT NULL,
        action VARCHAR NOT NULL,
        details VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS shadow_portfolio (
        run_id VARCHAR NOT NULL,
        portfolio VARCHAR NOT NULL,
        isin VARCHAR NOT NULL,
        horizon_days INTEGER NOT NULL,
        entry_ask DOUBLE NOT NULL,
        exit_due DATE NOT NULL,
        realized_net_return DOUBLE,
        created_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (run_id, portfolio, isin, horizon_days)
    )
    """,
    # -- Integration wave (Contract v3): forecasts, position reevaluation, ----
    # walk-forward persistence. Additive only.
    """
    CREATE TABLE IF NOT EXISTS forecasts (
        run_id VARCHAR NOT NULL,
        underlying_id VARCHAR NOT NULL,
        horizon_days INTEGER NOT NULL,
        prediction_time TIMESTAMPTZ NOT NULL,
        frozen_at TIMESTAMPTZ NOT NULL,
        p_up DOUBLE NOT NULL,
        mean DOUBLE NOT NULL,
        sigma DOUBLE NOT NULL,
        quantiles VARCHAR NOT NULL,
        expected_shortfall_05 DOUBLE NOT NULL,
        uncertainty DOUBLE NOT NULL,
        model_id VARCHAR NOT NULL,
        model_hash VARCHAR NOT NULL,
        signal_family VARCHAR NOT NULL,
        n_train INTEGER NOT NULL,
        n_effective DOUBLE NOT NULL,
        component_weights VARCHAR NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR,
        PRIMARY KEY (run_id, underlying_id, horizon_days, model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS position_evaluations (
        position_id VARCHAR NOT NULL,
        as_of TIMESTAMPTZ NOT NULL,
        wkn VARCHAR NOT NULL,
        isin VARCHAR,
        underlying_id VARCHAR,
        status VARCHAR NOT NULL,
        reasons VARCHAR NOT NULL,
        current_bid DOUBLE,
        quote_timestamp TIMESTAMPTZ,
        remaining_horizon_days INTEGER,
        remaining_lcb_ev DOUBLE,
        remaining_p_ko DOUBLE,
        remaining_p_profit DOUBLE,
        unrealized_return DOUBLE,
        data_quality_ok BOOLEAN NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR,
        PRIMARY KEY (position_id, as_of)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS walkforward_results (
        model_id VARCHAR NOT NULL,
        model_hash VARCHAR,
        signal_family VARCHAR NOT NULL,
        underlying_id VARCHAR NOT NULL,
        horizon_days INTEGER NOT NULL,
        evaluated_at TIMESTAMPTZ NOT NULL,
        n_folds INTEGER NOT NULL,
        brier DOUBLE NOT NULL,
        brier_null DOUBLE,
        log_loss DOUBLE NOT NULL,
        ece DOUBLE NOT NULL,
        hit_rate DOUBLE NOT NULL,
        mean_oos_return DOUBLE NOT NULL,
        psr DOUBLE NOT NULL,
        n_effective DOUBLE NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR,
        params VARCHAR NOT NULL
    )
    """,
    # -- W10: KO-probability calibration (docs/measured_results.md §3, ---------
    # backtest/ko_calibration.py). Additive only.
    """
    CREATE TABLE IF NOT EXISTS ko_calibration_results (
        run_id VARCHAR NOT NULL,
        method VARCHAR NOT NULL,
        breakdown_dim VARCHAR NOT NULL,
        breakdown_value VARCHAR NOT NULL,
        n INTEGER NOT NULL,
        brier DOUBLE NOT NULL,
        calibration_intercept DOUBLE,
        calibration_slope DOUBLE,
        ece DOUBLE NOT NULL,
        mean_signed_error DOUBLE NOT NULL,
        absolute_calibration_error DOUBLE NOT NULL,
        evaluated_at TIMESTAMPTZ NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR,
        PRIMARY KEY (run_id, method, breakdown_dim, breakdown_value)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ko_calibration_promotion (
        run_id VARCHAR PRIMARY KEY,
        promoted_method VARCHAR,
        reason VARCHAR NOT NULL,
        evaluated_at TIMESTAMPTZ NOT NULL,
        config_hash VARCHAR NOT NULL,
        git_commit VARCHAR
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
    "ko_calibration_results",
    "ko_calibration_promotion",
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

    def product_snapshots_in_range(
        self, isin: str, start: datetime, end: datetime
    ) -> list[ProductSnapshot]:
        """All ``product_snapshots`` for ``isin`` whose effective timestamp
        (``quote_timestamp`` if present, else ``observation_time``) falls
        within ``[start, end]`` (inclusive), ordered ascending.

        Used by ``learning/labeler.py`` (exit-quote search, MFE/MAE) and
        ``learning/counterfactual.py`` (evaluating alternative products
        under the same exit rules).
        """
        rows = self._conn.execute(
            f"SELECT {', '.join(_PRODUCT_SNAPSHOT_COLUMNS)} FROM product_snapshots "
            "WHERE isin = ? AND COALESCE(quote_timestamp, observation_time) BETWEEN ? AND ? "
            "ORDER BY COALESCE(quote_timestamp, observation_time) ASC",
            [isin, _to_utc(start), _to_utc(end)],
        ).fetchall()
        return [_row_to_product_snapshot(row) for row in rows]

    def latest_product_snapshot_at_or_before(
        self, isin: str, as_of: datetime
    ) -> ProductSnapshot | None:
        """The most recent ``product_snapshots`` row for ``isin`` whose
        effective timestamp is ``<= as_of`` (strict no-look-ahead --
        CLAUDE.md rule 5), or ``None`` if none exists.

        Used by ``learning/counterfactual.py`` to reconstruct an
        alternative product's own entry terms (ask, barrier) as of the
        original prediction time, since the ledger only stores the
        alternative's ISIN, not its terms.
        """
        row = self._conn.execute(
            f"SELECT {', '.join(_PRODUCT_SNAPSHOT_COLUMNS)} FROM product_snapshots "
            "WHERE isin = ? AND COALESCE(quote_timestamp, observation_time) <= ? "
            "ORDER BY COALESCE(quote_timestamp, observation_time) DESC LIMIT 1",
            [isin, _to_utc(as_of)],
        ).fetchone()
        return _row_to_product_snapshot(row) if row is not None else None

    def get_instrument(self, isin: str) -> Instrument | None:
        """Master-data row for one ISIN (``product_type`` etc.), or
        ``None`` if never upserted. Used by ``learning/labeler.py`` to
        determine KO-residual treatment (turbo vs. mini-future)."""
        row = self._conn.execute(
            """
            SELECT isin, wkn, issuer, underlying_id, underlying_raw, direction,
                   product_type, ratio, currency, underlying_currency, quanto,
                   open_end, maturity, first_trading_day, venue, first_seen_at,
                   last_seen_at
            FROM instruments WHERE isin = ?
            """,
            [isin],
        ).fetchone()
        if row is None:
            return None
        return Instrument(
            isin=row[0],
            wkn=row[1],
            issuer=row[2],
            underlying_id=row[3],
            underlying_raw=row[4],
            direction=Direction(row[5]),
            product_type=ProductType(row[6]),
            ratio=row[7],
            currency=row[8],
            underlying_currency=row[9],
            quanto=row[10],
            open_end=row[11],
            maturity=row[12],
            first_trading_day=row[13],
            venue=row[14],
            first_seen_at=_from_db_dt(row[15]),
            last_seen_at=_from_db_dt(row[16]),
        )

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

    # -- forward ledger (W6, Master Spec §25) ---------------------------------

    def append_ledger_entries(self, entries: Sequence[LedgerEntry]) -> int:
        """Insert new forward-ledger rows.

        Idempotent per ``entry_id``: an entry whose ``entry_id`` already
        exists is silently skipped rather than raising or overwriting
        (``ON CONFLICT ... DO NOTHING``) -- the ledger is append-only
        (CLAUDE.md rule 33; Build Contract v2 W6 requirement 2). Returns the
        number of rows actually inserted (which can be less than
        ``len(entries)`` if some were already present).
        """
        if not entries:
            return 0
        rows = [_ledger_entry_row(e) for e in entries]
        before = self._conn.execute("SELECT count(*) FROM forward_ledger").fetchone()
        before_n = int(before[0]) if before is not None else 0
        self._conn.executemany(
            f"INSERT INTO forward_ledger ({', '.join(_LEDGER_ENTRY_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_LEDGER_ENTRY_COLUMNS))}) "
            "ON CONFLICT (entry_id) DO NOTHING",
            rows,
        )
        after = self._conn.execute("SELECT count(*) FROM forward_ledger").fetchone()
        after_n = int(after[0]) if after is not None else 0
        return after_n - before_n

    def get_ledger_entry(self, entry_id: str) -> LedgerEntry | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_LEDGER_ENTRY_COLUMNS)} FROM forward_ledger WHERE entry_id = ?",
            [entry_id],
        ).fetchone()
        return _row_to_ledger_entry(row) if row is not None else None

    def ledger_entries_due_for_labeling(self, as_of: datetime) -> list[LedgerEntry]:
        """Open entries whose ``exit_due`` date has arrived by ``as_of``."""
        rows = self._conn.execute(
            f"SELECT {', '.join(_LEDGER_ENTRY_COLUMNS)} FROM forward_ledger "
            "WHERE status = ? AND exit_due <= ? ORDER BY exit_due ASC, entry_id ASC",
            [LedgerEntryStatus.OPEN.value, _to_utc(as_of).date()],
        ).fetchall()
        return [_row_to_ledger_entry(row) for row in rows]

    def attach_ledger_label(self, label: LedgerLabel) -> None:
        """Attach the (append-only, never overwritten) exit-side label to an
        entry.

        Flips that entry's ``status`` to ``labeled``, *except* when
        ``label.exit_reason`` is ``EXPIRED_NO_DATA`` (truly no exit
        information could be found at all), in which case the entry's
        ``status`` becomes ``expired_no_data`` instead -- both are terminal
        states reachable only from ``open``.

        Raises :class:`StoreError` if the entry does not exist, or already
        has a label -- a label is a final fact, recorded once.
        """
        existing = self._conn.execute(
            "SELECT 1 FROM ledger_labels WHERE entry_id = ?", [label.entry_id]
        ).fetchone()
        if existing is not None:
            raise StoreError(
                f"ledger entry {label.entry_id!r} is already labeled; labels are "
                "append-only and are never overwritten"
            )
        entry_row = self._conn.execute(
            "SELECT status FROM forward_ledger WHERE entry_id = ?", [label.entry_id]
        ).fetchone()
        if entry_row is None:
            raise StoreError(f"no forward_ledger entry with entry_id={label.entry_id!r}")
        self._conn.execute(
            f"INSERT INTO ledger_labels ({', '.join(_LEDGER_LABEL_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_LEDGER_LABEL_COLUMNS))})",
            _ledger_label_row(label),
        )
        new_status = (
            LedgerEntryStatus.EXPIRED_NO_DATA
            if label.exit_reason is ExitReason.EXPIRED_NO_DATA
            else LedgerEntryStatus.LABELED
        )
        self._conn.execute(
            "UPDATE forward_ledger SET status = ? WHERE entry_id = ?",
            [new_status.value, label.entry_id],
        )

    def get_ledger_label(self, entry_id: str) -> LedgerLabel | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_LEDGER_LABEL_COLUMNS)} FROM ledger_labels WHERE entry_id = ?",
            [entry_id],
        ).fetchone()
        return _row_to_ledger_label(row) if row is not None else None

    def list_ledger_entries(
        self,
        *,
        run_id: str | None = None,
        underlying: str | None = None,
        status: LedgerEntryStatus | None = None,
        category: Category | None = None,
        is_shadow: bool | None = None,
    ) -> list[tuple[LedgerEntry, LedgerLabel | None]]:
        """Entries matching the given filters (AND-combined; an omitted
        filter is unconstrained), paired with their label if attached yet,
        ordered by ``prediction_time`` ascending."""
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("fl.run_id = ?")
            params.append(run_id)
        if underlying is not None:
            clauses.append("fl.underlying = ?")
            params.append(underlying)
        if status is not None:
            clauses.append("fl.status = ?")
            params.append(status.value)
        if category is not None:
            clauses.append("fl.category = ?")
            params.append(category.value)
        if is_shadow is not None:
            clauses.append("fl.is_shadow = ?")
            params.append(is_shadow)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        entry_cols = ", ".join(f"fl.{c}" for c in _LEDGER_ENTRY_COLUMNS)
        label_cols = ", ".join(f"ll.{c}" for c in _LEDGER_LABEL_COLUMNS)
        rows = self._conn.execute(
            f"SELECT {entry_cols}, {label_cols} FROM forward_ledger fl "
            f"LEFT JOIN ledger_labels ll ON ll.entry_id = fl.entry_id "
            f"{where} ORDER BY fl.prediction_time ASC, fl.entry_id ASC",
            params,
        ).fetchall()
        n_entry_cols = len(_LEDGER_ENTRY_COLUMNS)
        results: list[tuple[LedgerEntry, LedgerLabel | None]] = []
        for row in rows:
            entry = _row_to_ledger_entry(row[:n_entry_cols])
            label_part = row[n_entry_cols:]
            label = _row_to_ledger_label(label_part) if label_part[0] is not None else None
            results.append((entry, label))
        return results

    # -- strategy posteriors (W6, Master Spec §21) -----------------------------

    def get_strategy_posterior(
        self, signal_family: str, horizon_days: int
    ) -> tuple[float, float, float, float, float, datetime] | None:
        """``(mu0, kappa, alpha, beta, n, updated_at)`` for one
        ``(signal_family, horizon_days)`` pair, or ``None`` if never
        persisted (the caller should then fall back to its own prior). ``n``
        is a ``float`` (effective sample size) rather than an integer count,
        since down-weighted bootstrap-prior observations (Master Spec §29)
        can contribute a fractional amount."""
        row = self._conn.execute(
            "SELECT mu0, kappa, alpha, beta, n, updated_at FROM strategy_posteriors "
            "WHERE signal_family = ? AND horizon_days = ?",
            [signal_family, horizon_days],
        ).fetchone()
        if row is None:
            return None
        return (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            _from_db_dt(row[5]),
        )

    def upsert_strategy_posterior(
        self,
        signal_family: str,
        horizon_days: int,
        *,
        mu0: float,
        kappa: float,
        alpha: float,
        beta: float,
        n: float,
        updated_at: datetime,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO strategy_posteriors (
                signal_family, horizon_days, mu0, kappa, alpha, beta, n, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (signal_family, horizon_days) DO UPDATE SET
                mu0 = excluded.mu0, kappa = excluded.kappa, alpha = excluded.alpha,
                beta = excluded.beta, n = excluded.n, updated_at = excluded.updated_at
            """,
            [signal_family, horizon_days, mu0, kappa, alpha, beta, n, _to_utc(updated_at)],
        )

    # -- model registry (W6, Master Spec §20-21) -------------------------------

    def upsert_model_registry_entry(self, entry: ModelRegistryEntry) -> None:
        """Insert or refresh one model's registry row.

        ``created_at`` is preserved from any existing row on upsert (only
        the mutable fields and ``updated_at`` change) -- mirrors
        ``upsert_instruments``. A ``PROTECTED`` model (e.g. the TSMOM
        baseline) is upserted the same way as any other; nothing in this
        method ever deletes a row (CLAUDE.md rule 10).
        """
        existing = self._conn.execute(
            "SELECT created_at FROM model_registry WHERE model_id = ?", [entry.model_id]
        ).fetchone()
        created_at = _from_db_dt(existing[0]) if existing is not None else entry.created_at
        self._conn.execute(
            """
            INSERT INTO model_registry (
                model_id, model_hash, signal_family, status, weight, params,
                trial_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model_id) DO UPDATE SET
                model_hash = excluded.model_hash,
                signal_family = excluded.signal_family,
                status = excluded.status,
                weight = excluded.weight,
                params = excluded.params,
                trial_id = excluded.trial_id,
                updated_at = excluded.updated_at
            """,
            [
                entry.model_id,
                entry.model_hash,
                entry.signal_family,
                entry.status.value,
                entry.weight,
                json.dumps(entry.params, sort_keys=True),
                entry.trial_id,
                _to_utc(created_at),
                _to_utc(entry.updated_at),
            ],
        )

    def get_model_registry_entry(self, model_id: str) -> ModelRegistryEntry | None:
        row = self._conn.execute(
            "SELECT model_id, model_hash, signal_family, status, weight, params, "
            "trial_id, created_at, updated_at FROM model_registry WHERE model_id = ?",
            [model_id],
        ).fetchone()
        return _row_to_model_registry_entry(row) if row is not None else None

    def list_model_registry_entries(
        self, signal_family: str | None = None
    ) -> list[ModelRegistryEntry]:
        if signal_family is None:
            rows = self._conn.execute(
                "SELECT model_id, model_hash, signal_family, status, weight, params, "
                "trial_id, created_at, updated_at FROM model_registry ORDER BY model_id ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT model_id, model_hash, signal_family, status, weight, params, "
                "trial_id, created_at, updated_at FROM model_registry "
                "WHERE signal_family = ? ORDER BY model_id ASC",
                [signal_family],
            ).fetchall()
        return [_row_to_model_registry_entry(row) for row in rows]

    def append_model_weight_history(
        self,
        model_id: str,
        weight: float,
        *,
        utility: float | None,
        trial_id: str | None,
        recorded_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO model_weight_history "
            "(model_id, weight, utility, trial_id, recorded_at) VALUES (?, ?, ?, ?, ?)",
            [model_id, weight, utility, trial_id, _to_utc(recorded_at)],
        )

    def model_weight_history(
        self, model_id: str
    ) -> list[tuple[datetime, float, float | None, str | None]]:
        """``(recorded_at, weight, utility, trial_id)`` tuples for one model,
        oldest first -- the audit trail behind ``update_weights``."""
        rows = self._conn.execute(
            "SELECT recorded_at, weight, utility, trial_id FROM model_weight_history "
            "WHERE model_id = ? ORDER BY recorded_at ASC",
            [model_id],
        ).fetchall()
        return [(_from_db_dt(row[0]), float(row[1]), row[2], row[3]) for row in rows]

    # -- research trials (W6, Master Spec §27.1) -------------------------------

    def insert_research_trial(self, trial: ResearchTrial) -> None:
        self._conn.execute(
            "INSERT INTO research_trials "
            "(trial_id, kind, description, created_at, quarter, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                trial.trial_id,
                trial.kind,
                trial.description,
                _to_utc(trial.created_at),
                trial.quarter,
                trial.status.value,
            ],
        )

    def get_research_trial(self, trial_id: str) -> ResearchTrial | None:
        row = self._conn.execute(
            "SELECT trial_id, kind, description, created_at, quarter, status "
            "FROM research_trials WHERE trial_id = ?",
            [trial_id],
        ).fetchone()
        return _row_to_research_trial(row) if row is not None else None

    def count_research_trials_in_quarter(self, quarter: str) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM research_trials WHERE quarter = ?", [quarter]
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def list_research_trials(self, quarter: str | None = None) -> list[ResearchTrial]:
        if quarter is None:
            rows = self._conn.execute(
                "SELECT trial_id, kind, description, created_at, quarter, status "
                "FROM research_trials ORDER BY created_at ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT trial_id, kind, description, created_at, quarter, status "
                "FROM research_trials WHERE quarter = ? ORDER BY created_at ASC",
                [quarter],
            ).fetchall()
        return [_row_to_research_trial(row) for row in rows]

    def update_research_trial_status(self, trial_id: str, status: TrialStatus) -> None:
        self._conn.execute(
            "UPDATE research_trials SET status = ? WHERE trial_id = ?",
            [status.value, trial_id],
        )

    # -- drift events (W6, Master Spec §30) ------------------------------------

    def insert_drift_event(self, event: DriftEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO drift_events (
                event_id, detected_at, stream_id, signal_family, metric,
                ph_statistic, threshold, action, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event.event_id,
                _to_utc(event.detected_at),
                event.stream_id,
                event.signal_family,
                event.metric,
                event.ph_statistic,
                event.threshold,
                event.action,
                json.dumps(event.details, sort_keys=True),
            ],
        )

    def list_drift_events(self, stream_id: str | None = None) -> list[DriftEvent]:
        if stream_id is None:
            rows = self._conn.execute(
                "SELECT event_id, detected_at, stream_id, signal_family, metric, "
                "ph_statistic, threshold, action, details FROM drift_events "
                "ORDER BY detected_at ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT event_id, detected_at, stream_id, signal_family, metric, "
                "ph_statistic, threshold, action, details FROM drift_events "
                "WHERE stream_id = ? ORDER BY detected_at ASC",
                [stream_id],
            ).fetchall()
        return [_row_to_drift_event(row) for row in rows]

    # -- shadow portfolio (W6, Master Spec §46) --------------------------------

    def append_shadow_positions(self, positions: Sequence[ShadowPosition]) -> int:
        """Insert new shadow-portfolio rows, ignoring any that already exist
        for the same ``(run_id, portfolio, isin, horizon_days)`` key."""
        if not positions:
            return 0
        rows = [_shadow_position_row(p) for p in positions]
        before = self._conn.execute("SELECT count(*) FROM shadow_portfolio").fetchone()
        before_n = int(before[0]) if before is not None else 0
        self._conn.executemany(
            """
            INSERT INTO shadow_portfolio (
                run_id, portfolio, isin, horizon_days, entry_ask, exit_due,
                realized_net_return, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, portfolio, isin, horizon_days) DO NOTHING
            """,
            rows,
        )
        after = self._conn.execute("SELECT count(*) FROM shadow_portfolio").fetchone()
        after_n = int(after[0]) if after is not None else 0
        return after_n - before_n

    def update_shadow_position_realized_return(
        self,
        run_id: str,
        portfolio: ShadowPortfolioKind,
        isin: str,
        horizon_days: int,
        realized_net_return: float,
    ) -> None:
        self._conn.execute(
            "UPDATE shadow_portfolio SET realized_net_return = ? "
            "WHERE run_id = ? AND portfolio = ? AND isin = ? AND horizon_days = ?",
            [realized_net_return, run_id, portfolio.value, isin, horizon_days],
        )

    def list_shadow_positions(
        self,
        run_id: str | None = None,
        portfolio: ShadowPortfolioKind | None = None,
    ) -> list[ShadowPosition]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if portfolio is not None:
            clauses.append("portfolio = ?")
            params.append(portfolio.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT run_id, portfolio, isin, horizon_days, entry_ask, exit_due, "
            f"realized_net_return, created_at FROM shadow_portfolio {where} "
            "ORDER BY created_at ASC, isin ASC",
            params,
        ).fetchall()
        return [_row_to_shadow_position(row) for row in rows]

    # -- forecasts (Contract v3 integration wave) ------------------------------

    def append_forecasts(self, records: Sequence[ForecastRecord]) -> int:
        """Insert forecast rows, idempotent per ``(run_id, underlying_id,
        horizon_days, model_id)`` (one call per scan naturally writes each
        component model's forecast plus the combined ``model_id="ensemble"``
        row once)."""
        if not records:
            return 0
        rows = [_forecast_row(r) for r in records]
        self._conn.executemany(
            f"INSERT INTO forecasts ({', '.join(_FORECAST_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_FORECAST_COLUMNS))}) "
            "ON CONFLICT (run_id, underlying_id, horizon_days, model_id) DO NOTHING",
            rows,
        )
        return len(rows)

    def list_forecasts(
        self,
        *,
        run_id: str | None = None,
        underlying_id: str | None = None,
        model_id: str | None = None,
    ) -> list[ForecastRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if underlying_id is not None:
            clauses.append("underlying_id = ?")
            params.append(underlying_id)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT {', '.join(_FORECAST_COLUMNS)} FROM forecasts {where} "
            "ORDER BY prediction_time ASC, horizon_days ASC, model_id ASC",
            params,
        ).fetchall()
        return [_row_to_forecast(row) for row in rows]

    # -- position evaluations (Contract v3 Abschnitt D) -------------------------

    def append_position_evaluation(self, evaluation: PositionEvaluation) -> None:
        """Insert one evaluation row, idempotent per ``(position_id, as_of)``
        -- a second call with the identical ``as_of`` (e.g. a re-run within
        the same invocation) is a silent no-op rather than a constraint
        error, mirroring ``append_ledger_entries``' idempotency."""
        self._conn.execute(
            f"INSERT INTO position_evaluations ({', '.join(_POSITION_EVAL_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_POSITION_EVAL_COLUMNS))}) "
            "ON CONFLICT (position_id, as_of) DO NOTHING",
            _position_eval_row(evaluation),
        )

    def list_position_evaluations(self, position_id: str | None = None) -> list[PositionEvaluation]:
        if position_id is None:
            rows = self._conn.execute(
                f"SELECT {', '.join(_POSITION_EVAL_COLUMNS)} FROM position_evaluations "
                "ORDER BY position_id ASC, as_of ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(_POSITION_EVAL_COLUMNS)} FROM position_evaluations "
                "WHERE position_id = ? ORDER BY as_of ASC",
                [position_id],
            ).fetchall()
        return [_row_to_position_eval(row) for row in rows]

    def latest_position_evaluation(self, position_id: str) -> PositionEvaluation | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_POSITION_EVAL_COLUMNS)} FROM position_evaluations "
            "WHERE position_id = ? ORDER BY as_of DESC LIMIT 1",
            [position_id],
        ).fetchone()
        return _row_to_position_eval(row) if row is not None else None

    # -- walk-forward results (Contract v3 coordinator addition) ----------------

    def append_walkforward_results(self, records: Sequence[WalkforwardResultRecord]) -> int:
        """Append walk-forward evaluation rows (``turboedge backtest``).

        Not deduplicated (unlike the append-only ledger tables): a re-run of
        ``backtest`` for the same model/horizon is a new *measurement* worth
        keeping in full history, not an idempotent replay -- callers that
        only want the latest measurement should use
        :meth:`latest_walkforward_results`.
        """
        if not records:
            return 0
        rows = [_walkforward_row(r) for r in records]
        self._conn.executemany(
            f"INSERT INTO walkforward_results ({', '.join(_WALKFORWARD_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_WALKFORWARD_COLUMNS))})",
            rows,
        )
        return len(rows)

    def list_walkforward_results(
        self,
        *,
        signal_family: str | None = None,
        underlying_id: str | None = None,
        horizon_days: int | None = None,
    ) -> list[WalkforwardResultRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if signal_family is not None:
            clauses.append("signal_family = ?")
            params.append(signal_family)
        if underlying_id is not None:
            clauses.append("underlying_id = ?")
            params.append(underlying_id)
        if horizon_days is not None:
            clauses.append("horizon_days = ?")
            params.append(horizon_days)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT {', '.join(_WALKFORWARD_COLUMNS)} FROM walkforward_results {where} "
            "ORDER BY evaluated_at ASC",
            params,
        ).fetchall()
        return [_row_to_walkforward(row) for row in rows]

    def latest_walkforward_results(
        self, *, signal_family: str | None = None, underlying_id: str | None = None
    ) -> list[WalkforwardResultRecord]:
        """Most recent :class:`WalkforwardResultRecord` per ``(model_id,
        horizon_days)``, optionally filtered -- what
        ``reporting/weekly.run_research_tournament`` would read once it is
        wired to prefer this table over forward-ledger-only data (see
        Kurzbericht)."""
        all_rows = self.list_walkforward_results(
            signal_family=signal_family, underlying_id=underlying_id
        )
        latest: dict[tuple[str, int], WalkforwardResultRecord] = {}
        for r in all_rows:
            key = (r.model_id, r.horizon_days)
            existing = latest.get(key)
            if existing is None or r.evaluated_at > existing.evaluated_at:
                latest[key] = r
        return sorted(latest.values(), key=lambda r: (r.model_id, r.horizon_days))

    # -- KO-probability calibration (W10) --------------------------------------

    def append_ko_calibration_results(self, records: Sequence[KoCalibrationResultRecord]) -> int:
        """Append breakdown rows from one ``backtest.ko_calibration.run_ko_calibration``
        run. Not deduplicated (like ``append_walkforward_results``): a re-run
        is a new measurement, kept in full history."""
        if not records:
            return 0
        rows = [_ko_calibration_result_row(r) for r in records]
        self._conn.executemany(
            f"INSERT INTO ko_calibration_results ({', '.join(_KO_CALIBRATION_RESULT_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(_KO_CALIBRATION_RESULT_COLUMNS))})",
            rows,
        )
        return len(rows)

    def list_ko_calibration_results(
        self, *, run_id: str | None = None, method: str | None = None
    ) -> list[KoCalibrationResultRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if method is not None:
            clauses.append("method = ?")
            params.append(method)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT {', '.join(_KO_CALIBRATION_RESULT_COLUMNS)} FROM ko_calibration_results "
            f"{where} ORDER BY evaluated_at ASC, method ASC, "
            "breakdown_dim ASC, breakdown_value ASC",
            params,
        ).fetchall()
        return [_row_to_ko_calibration_result(row) for row in rows]

    def insert_ko_calibration_promotion(self, record: KoCalibrationPromotionRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO ko_calibration_promotion (
                run_id, promoted_method, reason, evaluated_at, config_hash, git_commit
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                record.run_id,
                record.promoted_method,
                record.reason,
                _to_utc(record.evaluated_at),
                record.config_hash,
                record.git_commit,
            ],
        )

    def get_ko_calibration_promotion(self, run_id: str) -> KoCalibrationPromotionRecord | None:
        row = self._conn.execute(
            "SELECT run_id, promoted_method, reason, evaluated_at, config_hash, git_commit "
            "FROM ko_calibration_promotion WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        return KoCalibrationPromotionRecord(
            run_id=row[0],
            promoted_method=row[1],
            reason=row[2],
            evaluated_at=_from_db_dt(row[3]),
            config_hash=row[4],
            git_commit=row[5],
        )

    def latest_ko_calibration_promotion(self) -> KoCalibrationPromotionRecord | None:
        """Most recently evaluated promotion decision across all runs, or
        ``None`` if ``run_ko_calibration`` has never been persisted."""
        row = self._conn.execute(
            "SELECT run_id, promoted_method, reason, evaluated_at, config_hash, git_commit "
            "FROM ko_calibration_promotion ORDER BY evaluated_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return KoCalibrationPromotionRecord(
            run_id=row[0],
            promoted_method=row[1],
            reason=row[2],
            evaluated_at=_from_db_dt(row[3]),
            config_hash=row[4],
            git_commit=row[5],
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
    "financing_rate",
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
        s.financing_rate,
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


def _row_to_product_snapshot(row: tuple[Any, ...]) -> ProductSnapshot:
    return ProductSnapshot(
        isin=row[0],
        wkn=row[1],
        issuer=row[2],
        venue=row[3],
        underlying_raw=row[4],
        underlying_id=row[5],
        direction=Direction(row[6]),
        product_type=ProductType(row[7]),
        financing_level=row[8],
        knockout_barrier=row[9],
        ratio=row[10],
        currency=row[11],
        underlying_currency=row[12],
        quanto=row[13],
        open_end=row[14],
        maturity=row[15],
        first_trading_day=row[16],
        bid=row[17],
        ask=row[18],
        bid_size=row[19],
        ask_size=row[20],
        quote_timestamp=_from_db_opt_dt(row[21]),
        quote_presence=row[22],
        bid_only=row[23],
        knocked_out=row[24],
        trading_hours=row[25],
        product_age_days=row[26],
        underlying_price_ref=row[27],
        underlying_price_ref_timestamp=_from_db_opt_dt(row[28]),
        financing_rate=row[29],
        raw_hash=row[30],
        observation_time=_from_db_dt(row[31]),
        available_at=_from_db_dt(row[32]),
        retrieved_at=_from_db_dt(row[33]),
        source_timestamp=_from_db_opt_dt(row[34]),
        source=row[35],
        schema_version=row[36],
        parser_version=row[37],
        is_stale=row[38],
        quality_score=row[39],
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
    "financing_spread_source",
    "premium_over_fair",
    "premium_uncertainty_term",
    "p_ko_raw",
    "p_ko_calibrated",
    "ko_calibrator_version",
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
        c.financing_spread_source,
        c.premium_over_fair,
        c.premium_uncertainty_term,
        c.p_ko_raw,
        c.p_ko_calibrated,
        c.ko_calibrator_version,
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
        financing_spread_source=row[24],
        premium_over_fair=row[25],
        premium_uncertainty_term=row[26],
        p_ko_raw=row[27],
        p_ko_calibrated=row[28],
        ko_calibrator_version=row[29],
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


_LEDGER_ENTRY_COLUMNS: tuple[str, ...] = (
    "entry_id",
    "run_id",
    "candidate_id",
    "signal_id",
    "signal_version_hash",
    "trial_id",
    "prediction_time",
    "underlying",
    "direction",
    "horizon_days",
    "regime_bucket",
    "cluster_id",
    "feature_hash",
    "model_hash",
    "config_hash",
    "git_commit",
    "category",
    "selected_wkn",
    "selected_isin",
    "issuer",
    "entry_bid",
    "entry_ask",
    "entry_spread",
    "entry_quote_timestamp",
    "entry_underlying_timestamp",
    "financing_level_entry",
    "barrier_entry",
    "ratio",
    "fx",
    "predicted_return",
    "p_profit",
    "p_ko",
    "expected_shortfall",
    "lcb_ev",
    "uncertainty",
    "shrinkage_intensity",
    "is_shadow",
    "shadow_stratum",
    "suggested_position_fraction",
    "exit_due",
    "alternatives",
    "feature_snapshot",
    "status",
)


def _ledger_entry_row(e: LedgerEntry) -> tuple[Any, ...]:
    return (
        e.entry_id,
        e.run_id,
        e.candidate_id,
        e.signal_id,
        e.signal_version_hash,
        e.trial_id,
        _to_utc(e.prediction_time),
        e.underlying,
        e.direction.value,
        e.horizon_days,
        e.regime_bucket,
        e.cluster_id,
        e.feature_hash,
        e.model_hash,
        e.config_hash,
        e.git_commit,
        e.category.value,
        e.selected_wkn,
        e.selected_isin,
        e.issuer,
        e.entry_bid,
        e.entry_ask,
        e.entry_spread,
        _to_utc(e.entry_quote_timestamp),
        _to_utc(e.entry_underlying_timestamp),
        e.financing_level_entry,
        e.barrier_entry,
        e.ratio,
        e.fx,
        e.predicted_return,
        e.p_profit,
        e.p_ko,
        e.expected_shortfall,
        e.lcb_ev,
        e.uncertainty,
        e.shrinkage_intensity,
        e.is_shadow,
        e.shadow_stratum,
        e.suggested_position_fraction,
        e.exit_due,
        json.dumps(list(e.alternatives)),
        json.dumps(e.feature_snapshot, sort_keys=True),
        e.status.value,
    )


def _row_to_ledger_entry(row: tuple[Any, ...]) -> LedgerEntry:
    return LedgerEntry(
        entry_id=row[0],
        run_id=row[1],
        candidate_id=row[2],
        signal_id=row[3],
        signal_version_hash=row[4],
        trial_id=row[5],
        prediction_time=_from_db_dt(row[6]),
        underlying=row[7],
        direction=Direction(row[8]),
        horizon_days=row[9],
        regime_bucket=row[10],
        cluster_id=row[11],
        feature_hash=row[12],
        model_hash=row[13],
        config_hash=row[14],
        git_commit=row[15],
        category=Category(row[16]),
        selected_wkn=row[17],
        selected_isin=row[18],
        issuer=row[19],
        entry_bid=row[20],
        entry_ask=row[21],
        entry_spread=row[22],
        entry_quote_timestamp=_from_db_dt(row[23]),
        entry_underlying_timestamp=_from_db_dt(row[24]),
        financing_level_entry=row[25],
        barrier_entry=row[26],
        ratio=row[27],
        fx=row[28],
        predicted_return=row[29],
        p_profit=row[30],
        p_ko=row[31],
        expected_shortfall=row[32],
        lcb_ev=row[33],
        uncertainty=row[34],
        shrinkage_intensity=row[35],
        is_shadow=row[36],
        shadow_stratum=row[37],
        suggested_position_fraction=row[38],
        exit_due=row[39],
        alternatives=json.loads(row[40]),
        feature_snapshot=json.loads(row[41]),
        status=LedgerEntryStatus(row[42]),
    )


_LEDGER_LABEL_COLUMNS: tuple[str, ...] = (
    "entry_id",
    "labeled_at",
    "exit_bid",
    "exit_quote_timestamp",
    "financing_level_exit",
    "exit_reason",
    "realized_selected_pnl",
    "underlying_pnl",
    "median_turbo_pnl",
    "best_turbo_pnl",
    "ideal_turbo_pnl",
    "mfe",
    "mae",
    "ko_hit",
    "time_to_ko_days",
    "ambiguous_path",
)


def _ledger_label_row(label: LedgerLabel) -> tuple[Any, ...]:
    return (
        label.entry_id,
        _to_utc(label.labeled_at),
        label.exit_bid,
        _opt_to_utc(label.exit_quote_timestamp),
        label.financing_level_exit,
        label.exit_reason.value,
        label.realized_selected_pnl,
        label.underlying_pnl,
        label.median_turbo_pnl,
        label.best_turbo_pnl,
        label.ideal_turbo_pnl,
        label.mfe,
        label.mae,
        label.ko_hit,
        label.time_to_ko_days,
        label.ambiguous_path,
    )


def _row_to_ledger_label(row: tuple[Any, ...]) -> LedgerLabel:
    return LedgerLabel(
        entry_id=row[0],
        labeled_at=_from_db_dt(row[1]),
        exit_bid=row[2],
        exit_quote_timestamp=_from_db_opt_dt(row[3]),
        financing_level_exit=row[4],
        exit_reason=ExitReason(row[5]),
        realized_selected_pnl=row[6],
        underlying_pnl=row[7],
        median_turbo_pnl=row[8],
        best_turbo_pnl=row[9],
        ideal_turbo_pnl=row[10],
        mfe=row[11],
        mae=row[12],
        ko_hit=row[13],
        time_to_ko_days=row[14],
        ambiguous_path=row[15],
    )


def _row_to_model_registry_entry(row: tuple[Any, ...]) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        model_id=row[0],
        model_hash=row[1],
        signal_family=row[2],
        status=ModelStatus(row[3]),
        weight=row[4],
        params=json.loads(row[5]),
        trial_id=row[6],
        created_at=_from_db_dt(row[7]),
        updated_at=_from_db_dt(row[8]),
    )


def _row_to_research_trial(row: tuple[Any, ...]) -> ResearchTrial:
    return ResearchTrial(
        trial_id=row[0],
        kind=row[1],
        description=row[2],
        created_at=_from_db_dt(row[3]),
        quarter=row[4],
        status=TrialStatus(row[5]),
    )


def _row_to_drift_event(row: tuple[Any, ...]) -> DriftEvent:
    return DriftEvent(
        event_id=row[0],
        detected_at=_from_db_dt(row[1]),
        stream_id=row[2],
        signal_family=row[3],
        metric=row[4],
        ph_statistic=row[5],
        threshold=row[6],
        action=row[7],
        details=json.loads(row[8]),
    )


def _shadow_position_row(p: ShadowPosition) -> tuple[Any, ...]:
    return (
        p.run_id,
        p.portfolio.value,
        p.isin,
        p.horizon_days,
        p.entry_ask,
        p.exit_due,
        p.realized_net_return,
        _to_utc(p.created_at),
    )


def _row_to_shadow_position(row: tuple[Any, ...]) -> ShadowPosition:
    return ShadowPosition(
        run_id=row[0],
        portfolio=ShadowPortfolioKind(row[1]),
        isin=row[2],
        horizon_days=row[3],
        entry_ask=row[4],
        exit_due=row[5],
        realized_net_return=row[6],
        created_at=_from_db_dt(row[7]),
    )


_FORECAST_COLUMNS: tuple[str, ...] = (
    "run_id",
    "underlying_id",
    "horizon_days",
    "prediction_time",
    "frozen_at",
    "p_up",
    "mean",
    "sigma",
    "quantiles",
    "expected_shortfall_05",
    "uncertainty",
    "model_id",
    "model_hash",
    "signal_family",
    "n_train",
    "n_effective",
    "component_weights",
    "config_hash",
    "git_commit",
)


def _forecast_row(r: ForecastRecord) -> tuple[Any, ...]:
    return (
        r.run_id,
        r.underlying_id,
        r.horizon_days,
        _to_utc(r.prediction_time),
        _to_utc(r.frozen_at),
        r.p_up,
        r.mean,
        r.sigma,
        json.dumps(r.quantiles, sort_keys=True),
        r.expected_shortfall_05,
        r.uncertainty,
        r.model_id,
        r.model_hash,
        r.signal_family,
        r.n_train,
        r.n_effective,
        json.dumps(r.component_weights, sort_keys=True),
        r.config_hash,
        r.git_commit,
    )


def _row_to_forecast(row: tuple[Any, ...]) -> ForecastRecord:
    return ForecastRecord(
        run_id=row[0],
        underlying_id=row[1],
        horizon_days=row[2],
        prediction_time=_from_db_dt(row[3]),
        frozen_at=_from_db_dt(row[4]),
        p_up=row[5],
        mean=row[6],
        sigma=row[7],
        quantiles=json.loads(row[8]),
        expected_shortfall_05=row[9],
        uncertainty=row[10],
        model_id=row[11],
        model_hash=row[12],
        signal_family=row[13],
        n_train=row[14],
        n_effective=row[15],
        component_weights=json.loads(row[16]),
        config_hash=row[17],
        git_commit=row[18],
    )


_POSITION_EVAL_COLUMNS: tuple[str, ...] = (
    "position_id",
    "as_of",
    "wkn",
    "isin",
    "underlying_id",
    "status",
    "reasons",
    "current_bid",
    "quote_timestamp",
    "remaining_horizon_days",
    "remaining_lcb_ev",
    "remaining_p_ko",
    "remaining_p_profit",
    "unrealized_return",
    "data_quality_ok",
    "config_hash",
    "git_commit",
)


def _position_eval_row(e: PositionEvaluation) -> tuple[Any, ...]:
    return (
        e.position_id,
        _to_utc(e.as_of),
        e.wkn,
        e.isin,
        e.underlying_id,
        e.status.value,
        json.dumps(e.reasons),
        e.current_bid,
        _opt_to_utc(e.quote_timestamp),
        e.remaining_horizon_days,
        e.remaining_lcb_ev,
        e.remaining_p_ko,
        e.remaining_p_profit,
        e.unrealized_return,
        e.data_quality_ok,
        e.config_hash,
        e.git_commit,
    )


def _row_to_position_eval(row: tuple[Any, ...]) -> PositionEvaluation:
    return PositionEvaluation(
        position_id=row[0],
        as_of=_from_db_dt(row[1]),
        wkn=row[2],
        isin=row[3],
        underlying_id=row[4],
        status=PositionEvaluationStatus(row[5]),
        reasons=json.loads(row[6]),
        current_bid=row[7],
        quote_timestamp=_from_db_opt_dt(row[8]),
        remaining_horizon_days=row[9],
        remaining_lcb_ev=row[10],
        remaining_p_ko=row[11],
        remaining_p_profit=row[12],
        unrealized_return=row[13],
        data_quality_ok=row[14],
        config_hash=row[15],
        git_commit=row[16],
    )


_WALKFORWARD_COLUMNS: tuple[str, ...] = (
    "model_id",
    "model_hash",
    "signal_family",
    "underlying_id",
    "horizon_days",
    "evaluated_at",
    "n_folds",
    "brier",
    "brier_null",
    "log_loss",
    "ece",
    "hit_rate",
    "mean_oos_return",
    "psr",
    "n_effective",
    "config_hash",
    "git_commit",
    "params",
)


def _walkforward_row(r: WalkforwardResultRecord) -> tuple[Any, ...]:
    return (
        r.model_id,
        r.model_hash,
        r.signal_family,
        r.underlying_id,
        r.horizon_days,
        _to_utc(r.evaluated_at),
        r.n_folds,
        r.brier,
        r.brier_null,
        r.log_loss,
        r.ece,
        r.hit_rate,
        r.mean_oos_return,
        r.psr,
        r.n_effective,
        r.config_hash,
        r.git_commit,
        json.dumps(r.params, sort_keys=True, default=str),
    )


def _row_to_walkforward(row: tuple[Any, ...]) -> WalkforwardResultRecord:
    return WalkforwardResultRecord(
        model_id=row[0],
        model_hash=row[1],
        signal_family=row[2],
        underlying_id=row[3],
        horizon_days=row[4],
        evaluated_at=_from_db_dt(row[5]),
        n_folds=row[6],
        brier=row[7],
        brier_null=row[8],
        log_loss=row[9],
        ece=row[10],
        hit_rate=row[11],
        mean_oos_return=row[12],
        psr=row[13],
        n_effective=row[14],
        config_hash=row[15],
        git_commit=row[16],
        params=json.loads(row[17]),
    )


_KO_CALIBRATION_RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "method",
    "breakdown_dim",
    "breakdown_value",
    "n",
    "brier",
    "calibration_intercept",
    "calibration_slope",
    "ece",
    "mean_signed_error",
    "absolute_calibration_error",
    "evaluated_at",
    "config_hash",
    "git_commit",
)


def _ko_calibration_result_row(r: KoCalibrationResultRecord) -> tuple[Any, ...]:
    return (
        r.run_id,
        r.method,
        r.breakdown_dim,
        r.breakdown_value,
        r.n,
        r.brier,
        r.calibration_intercept,
        r.calibration_slope,
        r.ece,
        r.mean_signed_error,
        r.absolute_calibration_error,
        _to_utc(r.evaluated_at),
        r.config_hash,
        r.git_commit,
    )


def _row_to_ko_calibration_result(row: tuple[Any, ...]) -> KoCalibrationResultRecord:
    return KoCalibrationResultRecord(
        run_id=row[0],
        method=row[1],
        breakdown_dim=row[2],
        breakdown_value=row[3],
        n=row[4],
        brier=row[5],
        calibration_intercept=row[6],
        calibration_slope=row[7],
        ece=row[8],
        mean_signed_error=row[9],
        absolute_calibration_error=row[10],
        evaluated_at=_from_db_dt(row[11]),
        config_hash=row[12],
        git_commit=row[13],
    )


__all__ = ["Store", "StoreError"]
