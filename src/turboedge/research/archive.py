"""Point-in-time reads over the immutable Parquet archive.

A research result is only worth the data discipline behind it. This project
has already thrown away one statistically clean finding because the economics
did not survive costs (`docs/measured_results.md` §6.15); the one failure mode
that would be worse is a finding that does not survive the discovery that it
saw data it could not have had.

So the archive under `state/snapshots/` is exposed to research through a
protocol with exactly one read shape, and that read is point-in-time by
construction rather than by the caller remembering to filter:

* A row is returned only if it satisfies **both** conditions -- its declared
  `available_at` is at or before the cutoff, *and* the file containing it was
  written at or before the cutoff. The second is not redundant: a backfill
  writes rows whose declared availability long predates the moment the system
  actually held them, and a backtest that reads those rows is testing a
  machine that never existed.
* A table without an availability column cannot be read point-in-time at all,
  and raises. Returning every row with a warning would be the same look-ahead
  with a softer label.

There are no write methods here, deliberately. `storage/snapshots.py`'s
`write_snapshot_parquet` stays the only writer, and research reads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

#: The column that says when a row became knowable. Every record written
#: through `storage/snapshots.py` carries it (it is part of the observation
#: envelope), which is what makes a point-in-time read possible at all.
AVAILABILITY_COLUMN = "available_at"

#: `<UTC timestamp, second precision>-<uuid4 hex[:12]>` (`provenance.new_run_id`).
#: The timestamp prefix is when the file was written, and that is the second
#: half of the point-in-time test.
_RUN_ID_TIME_RE = re.compile(r"^(\d{8}T\d{6}Z)-")

_PARTITION_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


class ArchiveError(Exception):
    """Base class for archive read failures."""


class UnknownTable(ArchiveError):
    """No such table directory under `snapshots/`."""


class PointInTimeUnavailable(ArchiveError):
    """The table cannot be read point-in-time, so it is not read at all.

    Raised when the availability column is missing, or when a file's write
    time cannot be established from its name. Both are cases where the honest
    answer is "this cannot be reconstructed", and returning rows anyway would
    substitute a guess for a fact (CLAUDE.md rule 29).
    """


@runtime_checkable
class ResearchArchive(Protocol):
    """Read-only, point-in-time access to historical observations.

    Implementations must never return a row the system could not have held at
    `as_of`. Nothing in this protocol writes, deletes or repairs.
    """

    def tables(self) -> list[str]:
        """Table names present in the archive, sorted."""
        ...

    def partitions(self, table: str) -> list[date]:
        """Snapshot dates present for `table`, oldest first."""
        ...

    def files(
        self, table: str, *, start: date | None = None, end: date | None = None
    ) -> list[Path]:
        """Parquet files for `table` in `[start, end]`, chronologically."""
        ...

    def read_as_of(
        self,
        table: str,
        as_of: datetime,
        *,
        start: date | None = None,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Every row of `table` the system could have held at `as_of`."""
        ...


class ParquetResearchArchive:
    """`ResearchArchive` over `<state_dir>/snapshots/<table>/date=.../<run_id>.parquet`.

    The layout is the one `write_snapshot_parquet` already produces; this adds
    no new storage and migrates nothing.
    """

    def __init__(self, state_dir: Path) -> None:
        self._root = Path(state_dir) / "snapshots"

    @property
    def root(self) -> Path:
        return self._root

    def tables(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())

    def partitions(self, table: str) -> list[date]:
        return sorted(day for day, _ in self._partition_dirs(table))

    def files(
        self, table: str, *, start: date | None = None, end: date | None = None
    ) -> list[Path]:
        out: list[Path] = []
        for day, directory in self._partition_dirs(table):
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            out.extend(sorted(directory.glob("*.parquet")))
        return out

    def read_as_of(
        self,
        table: str,
        as_of: datetime,
        *,
        start: date | None = None,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Every row of `table` whose file was written at or before `as_of`
        and whose `available_at` is at or before `as_of`.

        Args:
            table: Logical table name, e.g. `"product_snapshots"`.
            as_of: The cutoff. Must be timezone-aware -- a naive cutoff would
                silently mean "whatever the machine's timezone is", which is
                not a defensible basis for a look-ahead guarantee.
            start: Optional lower bound on the snapshot date, to avoid reading
                history a study does not need. Purely a performance bound; it
                can only ever remove *older* rows, never bring newer ones in.
            columns: Optional projection. `available_at` is always read so the
                filter cannot be projected away, and is then kept in the
                result only if the caller asked for it.

        Raises:
            UnknownTable: the table is not in the archive.
            PointInTimeUnavailable: a file has no parseable write time, or the
                data has no `available_at` column.
        """
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        cutoff = as_of.astimezone(UTC)

        paths = [
            p for p in self.files(table, start=start, end=cutoff.date()) if _written_at(p) <= cutoff
        ]
        if not paths:
            return pl.DataFrame()

        wanted = None if columns is None else _with_availability(columns)
        frames = [pl.read_parquet(p, columns=wanted) for p in paths]
        frame = pl.concat(frames, how="vertical_relaxed")

        if AVAILABILITY_COLUMN not in frame.columns:
            raise PointInTimeUnavailable(
                f"{table!r} has no {AVAILABILITY_COLUMN!r} column, so it cannot be "
                "read point-in-time; refusing to return rows whose availability "
                "is unknown"
            )
        frame = frame.filter(pl.col(AVAILABILITY_COLUMN) <= cutoff)
        if columns is not None and AVAILABILITY_COLUMN not in columns:
            frame = frame.drop(AVAILABILITY_COLUMN)
        return frame

    # -- internals ---------------------------------------------------------

    def _partition_dirs(self, table: str) -> list[tuple[date, Path]]:
        table_dir = self._root / table
        if not table_dir.is_dir():
            raise UnknownTable(f"no table {table!r} under {self._root}; present: {self.tables()}")
        out: list[tuple[date, Path]] = []
        for child in table_dir.iterdir():
            if not child.is_dir():
                continue
            match = _PARTITION_RE.match(child.name)
            if match is None:
                continue
            out.append((date.fromisoformat(match.group(1)), child))
        return sorted(out)


def _with_availability(columns: Sequence[str]) -> list[str]:
    return list(columns) if AVAILABILITY_COLUMN in columns else [*columns, AVAILABILITY_COLUMN]


def _written_at(path: Path) -> datetime:
    match = _RUN_ID_TIME_RE.match(path.stem)
    if match is None:
        raise PointInTimeUnavailable(
            f"cannot establish when {path.name!r} was written: its name does not "
            "start with a run-id timestamp, so whether the system held these rows "
            "at a given time is unknowable"
        )
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
