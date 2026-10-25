"""Additively merge a recovered state database into the current one.

WHY THIS EXISTS
---------------
On 2026-10-23/24 eight pipeline runs went red for a reason unrelated to the
state they packed, and `.github/actions/turboedge-state` selected restore
candidates by run conclusion, so all eight artifacts became unreachable.
The chain resumed from a day-old state. The cause is fixed (the restore now
treats the artifact's existence as the proof, since "Pack and upload state"
is the last step of every data job and gated on `if: success()`), but the
data already lost needs recovering by hand --
`docs/state_recovery_2026-10-23.md`.

The irreplaceable part is four scans' `product_snapshots`: a bid/ask quote
with its issuer, barrier and timestamp as it stood at a past minute cannot
be fetched again at any later date.

WHAT MAKES THIS SAFE
--------------------
**Additive only.** Every table is merged by inserting the rows the current
database does not already have, matched on a key. Nothing is updated,
nothing is deleted, and a row already present is left exactly as it is --
so merging the same source twice changes nothing the second time.

**Fail-closed on unknown identity.** A table is merged only when its
identity is known: either DuckDB reports a PRIMARY KEY, or this module
configures a natural key for it. A table with neither is skipped and
reported, never merged on a guess. `product_snapshots` is the case that
makes this necessary -- it has **no** primary key and
`Store.append_product_snapshots` is a plain `INSERT`, so a naive merge
would duplicate every row and double the observation counts that retention
thinning and every effective-sample figure derive from.

**Measured, not asserted.** The report carries per-table counts before and
after, how many rows were inserted, and how many duplicate key groups exist
afterwards. A merge that introduced duplicates is visible rather than
discovered months later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from turboedge.storage.duckdb import Store

logger = structlog.get_logger(__name__)

#: Identity for tables DuckDB reports no PRIMARY KEY for. Each is the
#: smallest column set that identifies one observation, so an insert can
#: tell "already have this" from "new".
#:
#: `product_snapshots` is the important one and the reason this mapping
#: exists at all: one row per (product, moment, source).
NATURAL_KEYS: Mapping[str, tuple[str, ...]] = {
    "product_snapshots": ("isin", "observation_time", "source"),
    "signals": ("signal_id", "signal_version_hash"),
    "source_health": ("source", "checked_at"),
    "model_weight_history": ("model_id", "recorded_at"),
    "walkforward_results": (
        "model_id",
        "model_hash",
        "underlying_id",
        "horizon_days",
        "evaluated_at",
    ),
    "rejected_ratio_derivations": ("isin", "observed_at", "source"),
}

#: Deliberately never merged. `schema_migrations` records what *this*
#: database's schema has been through; another database's migration log is
#: not this one's history, and merging it would misreport it.
NEVER_MERGED: frozenset[str] = frozenset({"schema_migrations"})


@dataclass(frozen=True, slots=True)
class TableMerge:
    """What happened to one table."""

    table: str
    key: tuple[str, ...]
    key_source: str
    rows_before: int
    rows_inserted: int
    rows_after: int
    duplicate_groups_before: int
    duplicate_groups_after: int

    @property
    def consistent(self) -> bool:
        """Whether the row count moved by exactly what was inserted."""
        return self.rows_after == self.rows_before + self.rows_inserted

    @property
    def duplicates_introduced(self) -> int:
        """Duplicate key groups this merge added, not ones it inherited.

        The first recovery dry run reported pre-existing duplicates as
        introduced and refused to pack on that basis -- correctly refusing,
        for the wrong reason.
        """
        return max(0, self.duplicate_groups_after - self.duplicate_groups_before)


@dataclass(frozen=True, slots=True)
class MergeReport:
    """The whole merge, in numbers a reviewer can check."""

    source: str
    tables: tuple[TableMerge, ...] = ()
    skipped: Mapping[str, str] = field(default_factory=dict)

    @property
    def rows_inserted(self) -> int:
        return sum(t.rows_inserted for t in self.tables)

    @property
    def tables_changed(self) -> tuple[str, ...]:
        return tuple(t.table for t in self.tables if t.rows_inserted)

    @property
    def duplicates_introduced(self) -> tuple[str, ...]:
        return tuple(t.table for t in self.tables if t.duplicates_introduced)

    @property
    def inconsistent(self) -> tuple[str, ...]:
        return tuple(t.table for t in self.tables if not t.consistent)

    def summary(self) -> str:
        """One line, without the source path.

        The caller knows which archive it passed and can say so far more
        briefly than an absolute path to a temporary directory, which wraps
        and buries the numbers that matter.
        """
        if not self.rows_inserted:
            return "nothing new -- every row was already present"
        return f"{self.rows_inserted} row(s) recovered into {len(self.tables_changed)} table(s)"


def _primary_keys(store: Store) -> dict[str, tuple[str, ...]]:
    rows = store._conn.execute(
        "SELECT table_name, constraint_column_names FROM duckdb_constraints() "
        "WHERE constraint_type = 'PRIMARY KEY'"
    ).fetchall()
    return {str(table): tuple(str(c) for c in columns) for table, columns in rows}


def _tables(store: Store) -> list[str]:
    """The LIVE database's tables, not every attached database's.

    `duckdb_tables()` spans all attached databases, so inside the ATTACH
    block an unfiltered query returns each table twice -- once for the live
    database and once for the recovered one. The first recovery dry run
    (40189568686) showed every table merged twice and every name listed
    twice in the report. The second pass inserted nothing, so no data was
    harmed, but the counts were meaningless.
    """
    rows = store._conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = current_database() "
        "ORDER BY table_name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _count(store: Store, sql: str) -> int:
    row = store._conn.execute(sql).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def _duplicate_groups(store: Store, table: str, key: Sequence[str]) -> int:
    columns = ", ".join(key)
    return _count(
        store,
        f"SELECT count(*) FROM (SELECT {columns} FROM {table} "
        f"GROUP BY {columns} HAVING count(*) > 1)",
    )


def merge_recovered_state(
    store: Store, recovered: Path, *, alias: str = "recovered"
) -> MergeReport:
    """Insert every row of `recovered` that `store` does not already have.

    `store` is the live database and is only ever added to. The caller is
    responsible for having made a copy first -- this function does not back
    anything up, because a backup that the thing being changed also owns is
    not a backup.
    """
    pks = _primary_keys(store)
    merges: list[TableMerge] = []
    skipped: dict[str, str] = {}

    store._conn.execute(f"ATTACH '{recovered}' AS {alias} (READ_ONLY)")
    try:
        recovered_tables = {
            str(row[0])
            for row in store._conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE database_name = ?", [alias]
            ).fetchall()
        }
        for table in _tables(store):
            if table in NEVER_MERGED:
                skipped[table] = "never merged: describes this database's own schema history"
                continue
            if table not in recovered_tables:
                skipped[table] = "absent from the recovered database"
                continue

            key = pks.get(table) or NATURAL_KEYS.get(table)
            if not key:
                # Fail-closed: without an identity there is no way to tell a
                # row already present from a new one, and inserting anyway
                # would duplicate silently.
                skipped[table] = (
                    "no primary key and no configured natural key -- not merged, because "
                    "a row already present cannot be told from a new one"
                )
                continue

            duplicates_before = _duplicate_groups(store, table, key)
            if duplicates_before:
                # The configured key is not unique in the data that is
                # already here, so it does not identify a row and merging on
                # it could skip genuinely distinct rows. Fail-closed, same as
                # having no key at all. Observed on
                # `rejected_ratio_derivations` in run 40189568686.
                skipped[table] = (
                    f"the key {list(key)} is not unique in the existing data "
                    f"({duplicates_before} duplicate group(s)) -- not merged, because it "
                    "cannot tell a row already present from a new one"
                )
                continue

            before = _count(store, f"SELECT count(*) FROM {table}")
            on = " AND ".join(f"c.{column} IS NOT DISTINCT FROM r.{column}" for column in key)
            inserted = store._conn.execute(
                f"INSERT INTO {table} SELECT r.* FROM {alias}.{table} r "
                f"WHERE NOT EXISTS (SELECT 1 FROM {table} c WHERE {on})"
            ).fetchone()
            after = _count(store, f"SELECT count(*) FROM {table}")
            merges.append(
                TableMerge(
                    table=table,
                    key=tuple(key),
                    key_source="primary_key" if table in pks else "natural_key",
                    rows_before=before,
                    rows_inserted=after - before,
                    rows_after=after,
                    duplicate_groups_before=duplicates_before,
                    duplicate_groups_after=_duplicate_groups(store, table, key),
                )
            )
            _ = inserted
    finally:
        store._conn.execute(f"DETACH {alias}")

    report = MergeReport(source=str(recovered), tables=tuple(merges), skipped=skipped)
    logger.info(
        "state_merge_complete",
        source=str(recovered),
        rows_inserted=report.rows_inserted,
        tables_changed=list(report.tables_changed),
        duplicates_introduced=list(report.duplicates_introduced),
        skipped=len(report.skipped),
    )
    return report


__all__ = [
    "NATURAL_KEYS",
    "NEVER_MERGED",
    "MergeReport",
    "TableMerge",
    "merge_recovered_state",
]
