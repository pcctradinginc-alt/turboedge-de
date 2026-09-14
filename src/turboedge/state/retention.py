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

Row-deletion alone does not shrink the ``.duckdb`` file. Measured against
this DuckDB version (1.5.x, see the module-level rewrite helper below for
the numbers): ``CREATE TABLE ... AS SELECT`` + ``DROP``/``RENAME`` +
``CHECKPOINT`` frees *blocks* inside the file but DuckDB tracks those in an
internal free list for reuse by later writes rather than returning them to
the OS via ``ftruncate`` -- the file never shrinks from this alone, and can
even grow (a full second copy of the surviving rows is written before the
old blocks are freed). ``VACUUM`` does not reclaim space either in this
version. The only thing that actually shrinks the file is rewriting every
table into a brand-new file with no free-list history and swapping it in;
:func:`_rewrite_database_file` does that via DuckDB's built-in ``COPY FROM
DATABASE ... TO ...``, verifies the copy is byte-for-byte identical in
row counts before touching the original, and swaps it in atomically.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import duckdb
import structlog
from duckdb import DuckDBPyConnection
from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.duckdb import Store

logger = structlog.get_logger(__name__)

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
    file_rewritten: bool


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


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded ``"`` per the SQL
    standard (DuckDB included). Defensive: our own table names are fixed
    literals, but a DuckDB catalog name is derived from the state
    directory's file stem, which is not under this module's control (e.g.
    a test fixture or a differently-named state dir could contain ``-`` or
    other characters that are invalid in an unquoted identifier)."""
    return '"' + name.replace('"', '""') + '"'


def _table_row_counts(conn: DuckDBPyConnection, catalog: str) -> dict[str, int]:
    """``{table_name: row_count}`` for every base table in ``catalog``,
    used to verify a rewritten database file is identical to the original
    before it is ever swapped in."""
    table_rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = 'main' AND table_type = 'BASE TABLE'",
        [catalog],
    ).fetchall()
    quoted_catalog = _quote_ident(catalog)
    return {
        name: _scalar_count(conn, f"SELECT count(*) FROM {quoted_catalog}.{_quote_ident(name)}")
        for (name,) in table_rows
    }


def _fsync_path(path: Any) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rewrite_database_file(store: Store) -> bool:
    """Physically rewrite ``store``'s ``.duckdb`` file to actually reclaim
    the space freed by :func:`compact_product_snapshots`'s row reduction
    (see the module docstring for why ``CHECKPOINT``/``VACUUM`` alone don't).

    Copies every table into a brand-new file via ``COPY FROM DATABASE``,
    verifies the new file has the exact same set of tables and the exact
    same row count in each one, and only then swaps it in atomically (temp
    file in the same directory, fsynced, ``os.replace`` -- the same pattern
    ``state/archive.py::pack_state`` uses for its own atomic writes). If
    anything goes wrong at any step -- the copy, the verification, or the
    swap -- the original file is left completely untouched and ``store``
    keeps working against it; a bigger database beats a corrupted one.

    Mutates ``store._conn`` in place (closes the old connection and opens a
    new one against the same path) on success; leaves it untouched on
    failure. Returns whether the rewrite happened.

    Known residual (measured, accepted): DuckDB's default (``auto``)
    per-column compression codec is chosen adaptively, and that choice is
    not perfectly deterministic across repeated ``COPY FROM DATABASE``
    calls on the same content -- on the real (small, single-scan)
    ``product_snapshots`` table this module ships tests against, repeated
    ``compact_product_snapshots`` calls with nothing left to remove were
    observed to settle into a *bounded* two-value oscillation (~10-20%
    above the fully-compacted minimum) rather than one fixed byte count.
    This is qualitatively different from the bug this module fixes: it
    doesn't compound (verified over 15 consecutive calls: never exceeded
    that ~20% band), and it starts small instead of growing there over
    repeated calls the way the free-list-bloat bug did (up to +90%
    cumulative after two calls, unbounded in principle). Forcing a fixed
    compression codec (``PRAGMA force_compression``) was tried to remove
    this: ``fsst``/``rle``/``dictionary`` did make the small real table
    perfectly reproducible, but ``fsst`` forced on a larger synthetic table
    with genuine, large-scale row deletion (35,500 of 50,000 rows removed)
    produced *zero* shrinkage across four repeated runs, where the
    unforced default reliably shrank the file -- i.e. forcing a codec
    trades this cosmetic small-table wobble for silently defeating the
    actual compaction on the workload this function exists for. Left
    unforced; the data said no fixed codec was safe across both scales.
    """
    if str(store.path) == ":memory:" or not store.path.exists():
        return False

    conn = store._conn
    tmp_path = store.path.with_name(f".{store.path.name}.rewrite-{uuid4().hex}")
    tmp_path.unlink(missing_ok=True)
    fresh_alias = f"rewrite_{uuid4().hex}"

    try:
        main_db_row = conn.execute("SELECT current_database()").fetchone()
        if main_db_row is None:
            raise AssertionError("current_database() returned no row")
        main_name = str(main_db_row[0])

        conn.execute(f"ATTACH {_sql_string(str(tmp_path))} AS {_quote_ident(fresh_alias)}")
        try:
            conn.execute(
                f"COPY FROM DATABASE {_quote_ident(main_name)} TO {_quote_ident(fresh_alias)}"
            )
            before_counts = _table_row_counts(conn, main_name)
            after_counts = _table_row_counts(conn, fresh_alias)
            if before_counts != after_counts:
                raise AssertionError(
                    "rewritten database does not match the original: "
                    f"before={before_counts} after={after_counts}"
                )
        finally:
            conn.execute(f"DETACH {_quote_ident(fresh_alias)}")
    except Exception:
        logger.warning("db_compact_rewrite_failed", db_path=str(store.path), exc_info=True)
        tmp_path.unlink(missing_ok=True)
        return False

    # Verified identical -- close the connection (it holds an OS-level lock
    # tied to the *original* file's inode; the swap below doesn't need, and
    # shouldn't have, an open writer on the file being replaced) and swap
    # the rewritten copy in.
    store.close()
    try:
        _fsync_path(tmp_path)
        os.replace(tmp_path, store.path)
    except OSError:
        logger.warning("db_compact_rewrite_swap_failed", db_path=str(store.path), exc_info=True)
        tmp_path.unlink(missing_ok=True)
        store._conn = duckdb.connect(str(store.path))
        return False
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        _fsync_path(store.path.parent)
    except OSError:
        # The swap already happened (data is correct on disk); this only
        # affects how durable the rename is against a concurrent crash, so
        # it's logged, not treated as a failure of the rewrite itself.
        logger.warning(
            "db_compact_rewrite_dir_fsync_failed", db_path=str(store.path), exc_info=True
        )

    store._conn = duckdb.connect(str(store.path))
    store._conn.execute("SET TimeZone='UTC'")
    return True


def _sql_string(value: str) -> str:
    """A single-quoted SQL string literal, escaping embedded ``'``."""
    return "'" + value.replace("'", "''") + "'"


def compact_product_snapshots(
    store: Store,
    *,
    keep_days: int = DEFAULT_KEEP_DAYS,
    now: datetime | None = None,
) -> RetentionReport:
    """Reduce ``product_snapshots`` rows older than ``keep_days`` to one row
    per ``(isin, UTC calendar day)``, except ISINs present in
    ``forward_ledger`` (kept in full). Runs a ``CHECKPOINT``, then rewrites
    the database file in place (see :func:`_rewrite_database_file` and the
    module docstring) so the row reduction actually shrinks the file on
    disk instead of just moving free space around inside it.

    ``store`` must have already had ``init_schema()`` called (as every CLI
    command does) -- this function only touches ``product_snapshots``
    (rebuilt via ``CREATE TABLE ... AS SELECT`` + rename, since the table has
    no primary key to `DELETE` against individual rows) and reads
    ``information_schema`` for the optional ``forward_ledger`` table. The
    file rewrite step, unlike the row reduction, touches (rewrites, then
    replaces) the whole database file -- if it fails for any reason the row
    reduction above is *not* rolled back (it already committed via
    ``CHECKPOINT``); only the physical shrink is skipped, and
    ``store``/``store._conn`` remain fully usable against the original
    file, just not yet compacted on disk. ``report.file_rewritten`` says
    which happened.

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

    file_rewritten = _rewrite_database_file(store)
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
        file_rewritten=file_rewritten,
    )


__all__ = [
    "DEFAULT_KEEP_DAYS",
    "RetentionConfig",
    "RetentionReport",
    "compact_product_snapshots",
]
