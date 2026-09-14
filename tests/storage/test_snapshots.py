from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import BaseModel

from turboedge.provenance import data_snapshot_hash
from turboedge.storage.schemas import CandidateEvaluation, ProductSnapshot
from turboedge.storage.snapshots import snapshot_paths_for_run_ids, write_snapshot_parquet


def test_write_snapshot_parquet_creates_expected_path(  # type: ignore[no-untyped-def]
    tmp_path: Path, make_product_snapshot
) -> None:
    snapshots = [
        make_product_snapshot(isin="DE000ABC1234"),
        make_product_snapshot(isin="DE000XYZ5678"),
    ]
    as_of = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)

    result = write_snapshot_parquet(
        "product_snapshots", snapshots, "run-123", tmp_path, snapshot_date=as_of
    )

    expected_path = (
        tmp_path / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run-123.parquet"
    )
    assert result.path == expected_path
    assert expected_path.exists()
    assert result.row_count == 2
    assert result.data_snapshot_hash == data_snapshot_hash(snapshots)


def test_write_snapshot_parquet_roundtrip_readable(tmp_path: Path, make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    snapshots = [make_product_snapshot(isin="DE000ABC1234")]
    result = write_snapshot_parquet("product_snapshots", snapshots, "run-abc", tmp_path)
    frame = pl.read_parquet(result.path)
    assert frame.height == 1
    assert frame["isin"][0] == "DE000ABC1234"


def test_write_snapshot_parquet_empty_records(tmp_path: Path) -> None:
    result = write_snapshot_parquet("signals", [], "run-empty", tmp_path)
    assert result.row_count == 0
    assert result.path.exists()


def test_write_snapshot_parquet_rejects_bad_table_name(  # type: ignore[no-untyped-def]
    tmp_path: Path, make_product_snapshot
) -> None:
    with pytest.raises(ValueError, match="invalid table name"):
        write_snapshot_parquet("bad/table", [make_product_snapshot()], "run-1", tmp_path)


def test_write_snapshot_parquet_rejects_empty_run_id(tmp_path: Path, make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="run_id"):
        write_snapshot_parquet("product_snapshots", [make_product_snapshot()], "", tmp_path)


# --------------------------------------------------------------------------
# Regression tests for the schema-inference bug (polars.exceptions.ComputeError
# "could not append value ... to the builder"): a real end-to-end scan over
# 4340 BNP Paribas products crashed here because pl.DataFrame(list_of_dicts)
# infers each column's dtype from only the first ~100 rows, and fields like
# `quanto`/`quote_presence` were `None` for longer than that before turning
# out to be `bool`. write_snapshot_parquet now derives an explicit schema
# from the pydantic model's field annotations instead of inferring one.
# --------------------------------------------------------------------------


def test_write_snapshot_parquet_large_heterogeneous_product_snapshot_batch(
    tmp_path: Path,
    make_large_product_snapshot_batch: Callable[[int], list[ProductSnapshot]],
) -> None:
    records = make_large_product_snapshot_batch(5000)

    result = write_snapshot_parquet("product_snapshots", records, "run-large-ps", tmp_path)

    assert result.row_count == 5000
    frame = pl.read_parquet(result.path)
    assert frame.height == 5000

    # explicit dtypes, not inferred ones
    assert frame.schema["quanto"] == pl.Boolean
    assert frame.schema["quote_presence"] == pl.Boolean
    assert frame.schema["ask"] == pl.Float64
    assert frame.schema["maturity"] == pl.Date
    assert frame.schema["quote_timestamp"] == pl.Datetime("us", "UTC")
    assert frame.schema["direction"] == pl.Utf8  # StrEnum -> Utf8
    assert frame.schema["product_age_days"] == pl.Int64

    # first 200 rows: the fields that started out None in the source data are
    # still None after the parquet roundtrip (no silent coercion to False/0/...)
    head = frame.head(200)
    assert head["quanto"].null_count() == 200
    assert head["quote_presence"].null_count() == 200
    assert head["ask"].null_count() == 200
    assert head["quote_timestamp"].null_count() == 200
    assert head["maturity"].null_count() == 200

    # from row 200 onward, every one of those fields is populated with its
    # real type (this is exactly the point at which the old code crashed)
    tail = frame.slice(200, frame.height - 200)
    assert tail["quanto"].null_count() == 0
    assert tail["quote_presence"].null_count() == 0
    assert tail["ask"].null_count() == 0
    assert tail["quote_timestamp"].null_count() == 0
    assert tail["maturity"].null_count() == 0
    assert set(tail["quanto"].to_list()) == {True, False}

    assert result.data_snapshot_hash == data_snapshot_hash(records)


def test_write_snapshot_parquet_large_heterogeneous_candidate_evaluation_batch(
    tmp_path: Path,
    make_large_candidate_batch: Callable[[int], list[CandidateEvaluation]],
) -> None:
    records = make_large_candidate_batch(5000)

    result = write_snapshot_parquet("candidate_sets", records, "run-large-cand", tmp_path)

    assert result.row_count == 5000
    frame = pl.read_parquet(result.path)
    assert frame.height == 5000

    # nested BaseModel and dict/list fields become canonical JSON strings
    assert frame.schema["costs"] == pl.Utf8
    assert frame.schema["financing_cost_horizon_pct"] == pl.Utf8
    assert frame.schema["reasons"] == pl.Utf8

    head = frame.head(200)
    assert head["costs"].null_count() == 200

    tail = frame.slice(200, frame.height - 200)
    assert tail["costs"].null_count() == 0
    decoded_costs = json.loads(tail["costs"][0])
    assert decoded_costs["ask"] == 4.86
    assert decoded_costs["issuer_margin"] == 0.10

    decoded_horizon = json.loads(tail["financing_cost_horizon_pct"][0])
    assert decoded_horizon == {"3d": 0.001, "7d": 0.003}

    assert result.data_snapshot_hash == data_snapshot_hash(records)


def test_write_snapshot_parquet_hash_deterministic(
    tmp_path: Path,
    make_large_product_snapshot_batch: Callable[[int], list[ProductSnapshot]],
) -> None:
    records = make_large_product_snapshot_batch(300)

    result1 = write_snapshot_parquet("product_snapshots", records, "run-det-1", tmp_path / "a")
    result2 = write_snapshot_parquet("product_snapshots", records, "run-det-2", tmp_path / "b")

    assert result1.data_snapshot_hash == result2.data_snapshot_hash
    assert result1.data_snapshot_hash == data_snapshot_hash(records)

    # order-sensitivity is unchanged: reordering the same records changes the hash
    reordered = list(reversed(records))
    result3 = write_snapshot_parquet("product_snapshots", reordered, "run-det-3", tmp_path / "c")
    assert result3.data_snapshot_hash != result1.data_snapshot_hash


class _UnsupportedField(BaseModel):
    value: complex


def test_write_snapshot_parquet_unsupported_annotation_raises_type_error(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unsupported annotation"):
        write_snapshot_parquet("bogus_table", [_UnsupportedField(value=1 + 2j)], "run-x", tmp_path)


# --------------------------------------------------------------------------
# snapshot_paths_for_run_ids -- the "only this run's new files" lookup that
# `turboedge state pack-snapshots` relies on for the incremental per-scan
# Parquet artifact (pipeline.yml's scan job), instead of re-uploading the
# whole accumulated state/snapshots/ history on every run.
# --------------------------------------------------------------------------


def test_snapshot_paths_for_run_ids_finds_only_the_requested_run_ids(  # type: ignore[no-untyped-def]
    tmp_path: Path, make_product_snapshot
) -> None:
    """Older history (a prior run's file) plus this run's new files coexist
    under state/snapshots/ -- only the requested run_ids' files come back,
    proving the lookup does not silently re-collect everything."""
    old = write_snapshot_parquet(
        "product_snapshots", [make_product_snapshot()], "run-old-1", tmp_path
    )
    new_dax = write_snapshot_parquet(
        "product_snapshots", [make_product_snapshot()], "run-new-dax", tmp_path
    )
    new_ndx = write_snapshot_parquet(
        "product_snapshots", [make_product_snapshot()], "run-new-ndx", tmp_path
    )

    found = snapshot_paths_for_run_ids(tmp_path, ["run-new-dax", "run-new-ndx"])

    assert found == sorted([new_dax.path, new_ndx.path])
    assert old.path not in found


def test_snapshot_paths_for_run_ids_spans_multiple_tables_and_dates(  # type: ignore[no-untyped-def]
    tmp_path: Path, make_product_snapshot
) -> None:
    """One turboedge scan-all run_id can appear under more than one table
    (e.g. product_snapshots and signals both written for the same run) --
    every matching file across every table/date partition is returned."""
    ps = write_snapshot_parquet(
        "product_snapshots",
        [make_product_snapshot()],
        "run-shared",
        tmp_path,
        snapshot_date=datetime(2026, 9, 10, tzinfo=UTC),
    )
    other_day = write_snapshot_parquet(
        "product_snapshots",
        [make_product_snapshot()],
        "run-shared",
        tmp_path,
        snapshot_date=datetime(2026, 9, 11, tzinfo=UTC),
    )

    found = snapshot_paths_for_run_ids(tmp_path, ["run-shared"])

    assert set(found) == {ps.path, other_day.path}


def test_snapshot_paths_for_run_ids_tolerates_unknown_and_duplicate_ids(
    tmp_path: Path, make_product_snapshot
) -> None:
    result = write_snapshot_parquet(
        "product_snapshots", [make_product_snapshot()], "run-known", tmp_path
    )

    found = snapshot_paths_for_run_ids(
        tmp_path, ["run-known", "run-known", "run-never-written", ""]
    )

    assert found == [result.path]


def test_snapshot_paths_for_run_ids_empty_input_returns_empty(tmp_path: Path) -> None:
    assert snapshot_paths_for_run_ids(tmp_path, []) == []


def test_snapshot_paths_for_run_ids_no_snapshots_dir_returns_empty(tmp_path: Path) -> None:
    assert not (tmp_path / "snapshots").exists()
    assert snapshot_paths_for_run_ids(tmp_path, ["run-1"]) == []


def test_snapshot_paths_for_run_ids_rejects_unsafe_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe run_id"):
        snapshot_paths_for_run_ids(tmp_path, ["../../etc/passwd"])


def test_snapshot_paths_for_run_ids_rejects_glob_metacharacters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe run_id"):
        snapshot_paths_for_run_ids(tmp_path, ["*"])
