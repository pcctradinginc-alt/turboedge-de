"""Tests for the point-in-time research archive.

The point of this module is one guarantee -- no row the system could not have
held at `as_of` -- so most of these tests are attempts to get such a row out.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from turboedge.research.archive import (
    ParquetResearchArchive,
    PointInTimeUnavailable,
    ResearchArchive,
    UnknownTable,
)

_DAY = date(2026, 9, 11)


def _write(
    root: Path,
    *,
    table: str = "quotes",
    day: date = _DAY,
    written_at: str = "20260911T090000Z",
    suffix: str = "aaaaaaaaaaaa",
    available_at: list[datetime] | None = None,
    isins: list[str] | None = None,
    include_availability: bool = True,
) -> Path:
    available_at = available_at or [datetime(2026, 9, 11, 8, 0, tzinfo=UTC)]
    isins = isins or [f"X{i}" for i in range(len(available_at))]
    data: dict[str, object] = {"isin": isins, "bid": [1.0] * len(isins)}
    if include_availability:
        data["available_at"] = available_at
    out_dir = root / "snapshots" / table / f"date={day.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{written_at}-{suffix}.parquet"
    pl.DataFrame(data).write_parquet(path)
    return path


def test_parquet_archive_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(ParquetResearchArchive(tmp_path), ResearchArchive)


def test_archive_exposes_no_way_to_write_or_delete() -> None:
    # write_snapshot_parquet stays the only writer; research reads.
    forbidden = ("write", "delete", "remove", "append", "repair")
    assert not [
        n
        for n in dir(ParquetResearchArchive)
        if not n.startswith("_") and any(f in n for f in forbidden)
    ]


def test_tables_and_partitions_are_discovered(tmp_path: Path) -> None:
    _write(tmp_path, table="quotes", day=date(2026, 9, 11))
    _write(tmp_path, table="quotes", day=date(2026, 9, 14), written_at="20260914T090000Z")
    _write(tmp_path, table="prices", written_at="20260911T100000Z")
    archive = ParquetResearchArchive(tmp_path)

    assert archive.tables() == ["prices", "quotes"]
    assert archive.partitions("quotes") == [date(2026, 9, 11), date(2026, 9, 14)]


def test_missing_archive_root_has_no_tables(tmp_path: Path) -> None:
    assert ParquetResearchArchive(tmp_path / "nothing").tables() == []


def test_unknown_table_raises(tmp_path: Path) -> None:
    _write(tmp_path)
    with pytest.raises(UnknownTable):
        ParquetResearchArchive(tmp_path).partitions("nope")


def test_read_returns_rows_available_before_the_cutoff(tmp_path: Path) -> None:
    _write(
        tmp_path,
        available_at=[
            datetime(2026, 9, 11, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 11, 8, 30, tzinfo=UTC),
        ],
        isins=["EARLY", "LATE"],
    )
    archive = ParquetResearchArchive(tmp_path)

    frame = archive.read_as_of("quotes", datetime(2026, 9, 11, 12, 0, tzinfo=UTC))

    assert sorted(frame["isin"].to_list()) == ["EARLY", "LATE"]


def test_a_row_that_became_available_after_the_cutoff_is_excluded(tmp_path: Path) -> None:
    # `available_at` is not simply the fetch time: a source with a publication
    # delay yields rows that only become usable *after* the file was written,
    # so the write-time check alone would let this row through.
    _write(
        tmp_path,
        written_at="20260911T080000Z",
        available_at=[
            datetime(2026, 9, 11, 7, 55, tzinfo=UTC),
            datetime(2026, 9, 11, 8, 15, tzinfo=UTC),
        ],
        isins=["BEFORE", "EMBARGOED"],
    )
    archive = ParquetResearchArchive(tmp_path)

    frame = archive.read_as_of("quotes", datetime(2026, 9, 11, 8, 0, tzinfo=UTC))

    assert frame["isin"].to_list() == ["BEFORE"]


def test_a_backfilled_row_is_excluded_even_though_it_claims_to_be_old(
    tmp_path: Path,
) -> None:
    # The whole reason the file's write time is checked as well: a backfill
    # declares an availability from long before the system actually held it.
    _write(
        tmp_path,
        day=date(2026, 9, 20),
        written_at="20260920T090000Z",
        available_at=[datetime(2026, 9, 11, 8, 0, tzinfo=UTC)],
        isins=["BACKFILLED"],
    )
    archive = ParquetResearchArchive(tmp_path)

    frame = archive.read_as_of("quotes", datetime(2026, 9, 11, 23, 59, tzinfo=UTC))

    assert frame.is_empty()


def test_a_file_written_later_the_same_day_is_excluded(tmp_path: Path) -> None:
    # Partition pruning alone would let this through: the partition date is
    # the cutoff date, but the file was written four hours after the cutoff.
    _write(
        tmp_path,
        written_at="20260911T160000Z",
        available_at=[datetime(2026, 9, 11, 7, 0, tzinfo=UTC)],
        isins=["SAME_DAY_LATER"],
    )
    archive = ParquetResearchArchive(tmp_path)

    assert archive.read_as_of("quotes", datetime(2026, 9, 11, 12, 0, tzinfo=UTC)).is_empty()


def test_a_table_without_an_availability_column_refuses_to_be_read(tmp_path: Path) -> None:
    _write(tmp_path, include_availability=False)
    archive = ParquetResearchArchive(tmp_path)

    with pytest.raises(PointInTimeUnavailable, match="available_at"):
        archive.read_as_of("quotes", datetime(2026, 9, 11, 12, 0, tzinfo=UTC))


def test_a_file_with_no_parseable_write_time_refuses_to_be_read(tmp_path: Path) -> None:
    out_dir = tmp_path / "snapshots" / "quotes" / "date=2026-09-11"
    out_dir.mkdir(parents=True)
    pl.DataFrame(
        {"isin": ["X"], "available_at": [datetime(2026, 9, 11, tzinfo=UTC)]}
    ).write_parquet(out_dir / "handwritten.parquet")
    archive = ParquetResearchArchive(tmp_path)

    with pytest.raises(PointInTimeUnavailable, match="written"):
        archive.read_as_of("quotes", datetime(2026, 9, 11, 12, 0, tzinfo=UTC))


def test_a_naive_cutoff_is_refused(tmp_path: Path) -> None:
    _write(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        ParquetResearchArchive(tmp_path).read_as_of("quotes", datetime(2026, 9, 11, 12, 0))


def test_reading_before_any_data_exists_is_empty_not_an_error(tmp_path: Path) -> None:
    _write(tmp_path)
    archive = ParquetResearchArchive(tmp_path)

    assert archive.read_as_of("quotes", datetime(2026, 9, 1, tzinfo=UTC)).is_empty()


def test_multiple_partitions_are_concatenated(tmp_path: Path) -> None:
    _write(tmp_path, day=date(2026, 9, 11), isins=["A"])
    _write(
        tmp_path,
        day=date(2026, 9, 14),
        written_at="20260914T090000Z",
        available_at=[datetime(2026, 9, 14, 8, 0, tzinfo=UTC)],
        isins=["B"],
    )
    archive = ParquetResearchArchive(tmp_path)

    frame = archive.read_as_of("quotes", datetime(2026, 9, 15, tzinfo=UTC))

    assert sorted(frame["isin"].to_list()) == ["A", "B"]


def test_start_only_removes_older_rows(tmp_path: Path) -> None:
    _write(tmp_path, day=date(2026, 9, 11), isins=["OLD"])
    _write(
        tmp_path,
        day=date(2026, 9, 14),
        written_at="20260914T090000Z",
        available_at=[datetime(2026, 9, 14, 8, 0, tzinfo=UTC)],
        isins=["NEW"],
    )
    archive = ParquetResearchArchive(tmp_path)

    frame = archive.read_as_of("quotes", datetime(2026, 9, 15, tzinfo=UTC), start=date(2026, 9, 12))

    assert frame["isin"].to_list() == ["NEW"]


def test_projection_cannot_drop_the_filter_column(tmp_path: Path) -> None:
    _write(
        tmp_path,
        written_at="20260911T080000Z",
        available_at=[
            datetime(2026, 9, 11, 7, 30, tzinfo=UTC),
            datetime(2026, 9, 11, 23, 0, tzinfo=UTC),
        ],
        isins=["BEFORE", "AFTER"],
    )
    archive = ParquetResearchArchive(tmp_path)

    frame = archive.read_as_of("quotes", datetime(2026, 9, 11, 12, 0, tzinfo=UTC), columns=["isin"])

    assert frame.columns == ["isin"]
    assert frame["isin"].to_list() == ["BEFORE"]


def test_a_cutoff_in_another_timezone_is_normalised(tmp_path: Path) -> None:
    from zoneinfo import ZoneInfo

    _write(
        tmp_path,
        available_at=[datetime(2026, 9, 11, 8, 30, tzinfo=UTC)],
        isins=["X"],
    )
    archive = ParquetResearchArchive(tmp_path)
    berlin = datetime(2026, 9, 11, 11, 0, tzinfo=ZoneInfo("Europe/Berlin"))  # 09:00Z

    assert archive.read_as_of("quotes", berlin)["isin"].to_list() == ["X"]


def test_files_lists_chronologically_within_the_requested_window(tmp_path: Path) -> None:
    first = _write(tmp_path, written_at="20260911T080000Z", suffix="aaaaaaaaaaaa")
    second = _write(tmp_path, written_at="20260911T160000Z", suffix="bbbbbbbbbbbb")
    _write(tmp_path, day=date(2026, 9, 20), written_at="20260920T080000Z")
    archive = ParquetResearchArchive(tmp_path)

    assert archive.files("quotes", end=date(2026, 9, 11)) == [first, second]


def test_it_reads_a_real_product_snapshot_archive() -> None:
    # The layout this is a protocol over is the one already on disk.
    archive = ParquetResearchArchive(Path("state"))
    if "product_snapshots" not in archive.tables():
        pytest.skip("no local state/snapshots archive")

    days = archive.partitions("product_snapshots")
    frame = archive.read_as_of(
        "product_snapshots", datetime.combine(days[0], datetime.max.time(), tzinfo=UTC)
    )

    assert not frame.is_empty()
    assert frame["available_at"].max() <= datetime.combine(days[0], datetime.max.time(), tzinfo=UTC)
