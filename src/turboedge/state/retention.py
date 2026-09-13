"""Bounded-retention compaction for ``product_snapshots``.

The public repo's always-on scheduler (``pipeline.yml``) appends a
``product_snapshots`` row per product per scan, several times a day,
forever -- left alone this grows the archived (and encrypted, uploaded)
DuckDB file without bound. :func:`compact_product_snapshots` (``turboedge db
compact``) reduces rows older than ``keep_days`` to one snapshot per
``(isin, UTC calendar day)`` -- the last snapshot of that day -- while
keeping the *full* history for any ISIN that appears in ``forward_ledger``,
since the label/learn pipeline (W6) needs the complete path, not a
once-a-day sample, to compute realized P&L, KO timing and MFE/MAE.

``forward_ledger`` is owned by another workstream (W6) and may not exist yet
at the time this module runs -- its presence (and its ``isin`` column) is
checked via ``information_schema`` before it is ever referenced in a query,
so this module works standalone against a database that doesn't have it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from duckdb import DuckDBPyConnection
from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.duckdb import Store

DEFAULT_KEEP_DAYS = 45

_LEDGER_TABLE = "forward_ledger"
# W6's `forward_ledger` schema (storage/duckdb.py) names the column
# `selected_isin`; `isin` is kept as a fallback in case that ever changes or
# a differently-shaped ledger table is used, since this module treats the
# exact column name as a soft (information_schema-checked) contract, not a
# hard-coded assumption about a table it does not own.
_LEDGER_ISIN_COLUMN_CANDIDATES: tuple[str, ...] = ("selected_isin", "isin")
_COMPACT_TABLE_NAME = "__product_snapshots_compact"


class RetentionConfig(BaseModel):
    """Defaults for ``turboedge db compact``; wired from YAML by the
    integration wave, not by this module."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    keep_days: int = Field(gt=0, default=DEFAULT_KEEP_DAYS)


@dataclass(frozen=True)
class RetentionReport:
    keep_days: int
    rows_before: int
    rows_after: int
    rows_removed: int
    protected_isin_count: int
    forward_ledger_present: bool
    db_size_bytes_before: int
    db_size_bytes_after: int


def _forward_ledger_isin_subquery(store: Store) -> str | None:
    """A SQL subquery selecting distinct protected ISINs, or ``None`` if
    ``forward_ledger`` does not exist yet, or exists but has none of
    :data:`_LEDGER_ISIN_COLUMN_CANDIDATES`."""
    conn = store._conn  # see `compact_product_snapshots` docstring
    table_rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [_LEDGER_TABLE],
    ).fetchall()
    if not table_rows:
        return None

    existing_columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ?",
            [_LEDGER_TABLE],
        ).fetchall()
    }
    for candidate in _LEDGER_ISIN_COLUMN_CANDIDATES:
        if candidate in existing_columns:
            return f"SELECT DISTINCT {candidate} FROM {_LEDGER_TABLE}"

    return None


def _db_size_bytes(store: Store) -> int:
    if str(store.path) == ":memory:" or not store.path.exists():
        return 0
    return store.path.stat().st_size


def _scalar_count(conn: DuckDBPyConnection, sql: str, params: Sequence[Any] = ()) -> int:
    """``SELECT count(*) ...`` helper: DuckDB's ``fetchone()`` is typed as
    ``tuple[Any, ...] | None``, but a ``count(*)`` query always returns
    exactly one row."""
    row = conn.execute(sql, list(params)).fetchone()
    if row is None:
        raise AssertionError(f"count query returned no row: {sql!r}")
    return int(row[0])


def compact_product_snapshots(
    store: Store,
    *,
    keep_days: int = DEFAULT_KEEP_DAYS,
    now: datetime | None = None,
) -> RetentionReport:
    """Reduce ``product_snapshots`` rows older than ``keep_days`` to one row
    per ``(isin, UTC calendar day)``, except ISINs present in
    ``forward_ledger`` (kept in full). Runs a ``CHECKPOINT`` afterwards.

    ``store`` must have already had ``init_schema()`` called (as every CLI
    command does) -- this function only touches ``product_snapshots``
    (rebuilt via ``CREATE TABLE ... AS SELECT`` + rename, since the table has
    no primary key to `DELETE` against individual rows) and reads
    ``information_schema`` for the optional ``forward_ledger`` table.

    Raises:
        ValueError: if ``keep_days`` is not positive.
    """
    if keep_days <= 0:
        raise ValueError(f"keep_days must be > 0, got {keep_days}")

    as_of = now if now is not None else datetime.now(UTC)
    cutoff = as_of - timedelta(days=keep_days)

    # `Store` (storage/duckdb.py) is outside this module's assignment and
    # does not expose its connection publicly; see
    # `state/archive.py::_checkpoint` for the same rationale.
    conn = store._conn

    # DuckDB's Python API returns TIMESTAMPTZ values converted to the
    # session's configured timezone (the OS timezone by default, e.g.
    # Europe/Berlin in CEST -- NOT necessarily UTC), and `CAST(... AS DATE)`
    # below truncates in that same session timezone. Every timestamp this
    # app stores is UTC (`storage/duckdb.py::_to_utc`), and the spec
    # requires bucketing by *UTC* calendar day regardless of the host
    # machine's local timezone -- pin the session explicitly rather than
    # relying on the runtime environment.
    conn.execute("SET TimeZone='UTC'")

    rows_before = _scalar_count(conn, "SELECT count(*) FROM product_snapshots")
    db_size_before = _db_size_bytes(store)

    ledger_subquery = _forward_ledger_isin_subquery(store)
    if ledger_subquery is not None:
        protected_count = _scalar_count(conn, f"SELECT count(*) FROM ({ledger_subquery})")
        is_protected_expr = f"(isin IN ({ledger_subquery}))"
    else:
        protected_count = 0
        is_protected_expr = "FALSE"

    conn.execute(f"DROP TABLE IF EXISTS {_COMPACT_TABLE_NAME}")
    conn.execute(
        f"""
        CREATE TABLE {_COMPACT_TABLE_NAME} AS
        SELECT * EXCLUDE (__rn, __is_protected) FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY isin, CAST(COALESCE(quote_timestamp, observation_time) AS DATE)
                    ORDER BY COALESCE(quote_timestamp, observation_time) DESC
                ) AS __rn,
                {is_protected_expr} AS __is_protected
            FROM product_snapshots
        )
        WHERE observation_time >= ? OR __is_protected OR __rn = 1
        """,
        [cutoff],
    )
    conn.execute("DROP TABLE product_snapshots")
    conn.execute(f"ALTER TABLE {_COMPACT_TABLE_NAME} RENAME TO product_snapshots")

    rows_after = _scalar_count(conn, "SELECT count(*) FROM product_snapshots")
    conn.execute("CHECKPOINT")
    db_size_after = _db_size_bytes(store)

    return RetentionReport(
        keep_days=keep_days,
        rows_before=rows_before,
        rows_after=rows_after,
        rows_removed=rows_before - rows_after,
        protected_isin_count=protected_count,
        forward_ledger_present=ledger_subquery is not None,
        db_size_bytes_before=db_size_before,
        db_size_bytes_after=db_size_after,
    )


__all__ = [
    "DEFAULT_KEEP_DAYS",
    "RetentionConfig",
    "RetentionReport",
    "compact_product_snapshots",
]
