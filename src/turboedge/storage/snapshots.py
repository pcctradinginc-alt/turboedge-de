"""Immutable Parquet archive of every scan run's persisted records.

DuckDB (``storage/duckdb.py``) is the mutable, queryable store used at
runtime. This module additionally writes an append-only, content-hashed
Parquet file per ``(table, run_id)`` under
``state/snapshots/<table>/date=YYYY-MM-DD/<run_id>.parquet`` so that any past
run's inputs/outputs can be reproduced byte-for-byte later, independent of
whatever the DuckDB tables have since been updated to (CLAUDE.md rule 33).

Schema inference pitfall
-------------------------
``pl.DataFrame(list_of_dicts)`` (and pandas' equivalent) infers a column's
dtype from only the first ``infer_schema_length`` rows (default 100). With
large, real-world batches (e.g. thousands of BNP Paribas turbo products) it
is common for a field such as ``quanto`` or ``quote_presence`` to be ``None``
for every one of the first 100 rows and only take on its real ``bool`` value
later, or for numeric fields to mix ``int``/``float`` shapes. Polars then
raises ``ComputeError: could not append value ... to the builder`` once a
later row does not fit the guessed dtype. Raising ``infer_schema_length``
only delays the failure to a bigger batch.

The fix here is to never infer a schema at all: for any batch, we derive an
explicit, exhaustive Arrow/Polars dtype per column directly from the
pydantic model's field annotations (see :func:`_build_field_plan`), convert
every record's field values to match that dtype up front, and build the
:class:`polars.DataFrame` with ``schema=...`` and ``strict=True`` - so a
genuinely unsupported/unexpected field type raises a clear ``TypeError`` at
schema-derivation time instead of silently guessing or failing deep inside
polars' Arrow builder.
"""

from __future__ import annotations

import json
import types
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import polars as pl
from pydantic import BaseModel

from turboedge.provenance import data_snapshot_hash


@dataclass(frozen=True)
class SnapshotResult:
    """Outcome of writing one immutable snapshot file."""

    path: Path
    table: str
    run_id: str
    row_count: int
    data_snapshot_hash: str


# --------------------------------------------------------------------------
# Explicit schema derivation (pydantic field annotation -> polars dtype)
# --------------------------------------------------------------------------

# How a field's Python value must be converted before handing it to
# ``pl.DataFrame``:
#   "plain"    - already the correct native type (str/int/float/bool/date),
#                passed through as-is.
#   "datetime" - a tz-aware ``datetime``, normalized to UTC.
#   "enum"     - a ``StrEnum`` member, replaced by its ``.value``.
#   "json"     - a ``dict``/``list``/nested ``BaseModel``, replaced by a
#                canonical (sorted-key, no incidental whitespace) JSON string.
_FieldKind = str  # Literal["plain", "datetime", "enum", "json"]


@dataclass(frozen=True)
class _FieldPlan:
    # polars dtypes are either a `DataType` instance (e.g. ``pl.Datetime("us",
    # "UTC")``) or a bare `DataType` subclass used as a singleton (e.g.
    # ``pl.Boolean``) - both are valid schema values for `pl.DataFrame`.
    dtype: pl.DataType | type[pl.DataType]
    kind: _FieldKind


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner_type, nullable)`` for a resolved pydantic annotation.

    Pydantic v2's ``model_fields[name].annotation`` already strips
    ``Annotated[...]`` metadata (e.g. our ``IsinStr``/``PositiveFloat``
    validators) down to the plain runtime type, so this only needs to peel
    off ``X | None`` (equivalently ``Optional[X]``). Polars columns are
    nullable regardless, so the boolean is informational only; it is
    returned so callers/tests can assert on it if needed.
    """
    origin = get_origin(annotation)
    if origin is not Union and origin is not types.UnionType:
        return annotation, False
    args = get_args(annotation)
    if type(None) not in args:
        return annotation, False
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) != 1:
        raise TypeError(
            f"unsupported union annotation with more than one non-None member: {annotation!r}"
        )
    return non_none[0], True


def _field_plan_for_annotation(annotation: Any, *, context: str) -> _FieldPlan:
    """Map one resolved pydantic field annotation to an explicit polars dtype.

    Raises ``TypeError`` for any annotation this module does not know how to
    represent - we never fall back to silent dtype guessing.
    """
    inner, _ = _unwrap_optional(annotation)

    if inner is bool:
        return _FieldPlan(pl.Boolean, "plain")
    if inner is int:
        return _FieldPlan(pl.Int64, "plain")
    if inner is float:
        return _FieldPlan(pl.Float64, "plain")
    if inner is str:
        return _FieldPlan(pl.Utf8, "plain")
    if inner is datetime:
        return _FieldPlan(pl.Datetime("us", "UTC"), "datetime")
    if inner is date:
        return _FieldPlan(pl.Date, "plain")
    if isinstance(inner, type) and issubclass(inner, StrEnum):
        return _FieldPlan(pl.Utf8, "enum")
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return _FieldPlan(pl.Utf8, "json")
    origin = get_origin(inner)
    if origin in (list, dict):
        return _FieldPlan(pl.Utf8, "json")

    raise TypeError(
        f"cannot derive an explicit Parquet/Arrow dtype for {context}: "
        f"unsupported annotation {inner!r}. Extend "
        "turboedge.storage.snapshots._field_plan_for_annotation to handle it "
        "explicitly rather than relying on schema inference."
    )


@cache
def _build_field_plan(model_cls: type[BaseModel]) -> dict[str, _FieldPlan]:
    """Derive the full, ordered field-name -> :class:`_FieldPlan` mapping.

    Cached per model class (there are only a handful of persisted model
    classes; the cache just avoids re-walking ``model_fields`` on every
    snapshot write).
    """
    return {
        name: _field_plan_for_annotation(field.annotation, context=f"{model_cls.__name__}.{name}")
        for name, field in model_cls.model_fields.items()
    }


def _json_encode(value: Any) -> str:
    """Canonical JSON encoding for dict/list/nested-BaseModel field values."""

    def _default(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, datetime):
            return obj.astimezone(UTC).isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, StrEnum):
            return obj.value
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def _convert_value(value: Any, plan: _FieldPlan) -> Any:
    if value is None:
        return None
    if plan.kind == "datetime":
        return value.astimezone(UTC)
    if plan.kind == "enum":
        return value.value
    if plan.kind == "json":
        return _json_encode(value)
    return value


def _records_to_frame(records: Sequence[BaseModel]) -> pl.DataFrame:
    """Build a :class:`polars.DataFrame` from ``records`` with an explicit schema.

    Unlike ``pl.DataFrame([r.model_dump(mode="json") for r in records])``,
    this never infers column dtypes from a sample of rows: the schema is
    derived once from ``type(records[0])``'s pydantic field annotations
    (:func:`_build_field_plan`), every value is converted to match it, and
    the frame is constructed with ``strict=True`` so any residual mismatch
    raises immediately instead of silently coercing or crashing deep inside
    an Arrow builder.
    """
    if not records:
        return pl.DataFrame()

    model_cls = type(records[0])
    plan = _build_field_plan(model_cls)
    schema = {name: field_plan.dtype for name, field_plan in plan.items()}
    columns: dict[str, list[Any]] = {name: [] for name in plan}

    for record in records:
        if type(record) is not model_cls:
            raise TypeError(
                "write_snapshot_parquet requires every record to be an instance of "
                f"the same pydantic model class; got {type(record).__name__!r} mixed "
                f"with {model_cls.__name__!r}"
            )
        for name, field_plan in plan.items():
            columns[name].append(_convert_value(getattr(record, name), field_plan))

    return pl.DataFrame(columns, schema=schema, strict=True)


def write_snapshot_parquet(
    table: str,
    records: Sequence[BaseModel],
    run_id: str,
    state_dir: Path,
    *,
    snapshot_date: datetime | None = None,
) -> SnapshotResult:
    """Write ``records`` as an immutable Parquet file for one table/run.

    Args:
        table: Logical table name (e.g. ``"product_snapshots"``). Must be a
            valid identifier - it becomes a path segment.
        records: Already-validated pydantic model instances, all of the same
            model class.
        run_id: The run this snapshot belongs to (see
            :func:`turboedge.provenance.new_run_id`). Becomes the file name;
            re-writing the same ``(table, run_id, day)`` overwrites the prior
            file, so callers must use a fresh run_id per scan to keep the
            archive append-only in practice.
        state_dir: Root state directory (``$TURBOEDGE_STATE_DIR``). The
            archive is written under ``<state_dir>/snapshots/...``.
        snapshot_date: Timestamp used to compute the ``date=YYYY-MM-DD``
            partition. Defaults to now (UTC).

    Returns:
        A :class:`SnapshotResult` with the written path, row count and the
        ``data_snapshot_hash`` of ``records`` (see
        :func:`turboedge.provenance.data_snapshot_hash`) - the same hash a
        caller would embed in a ``SignalSnapshot`` or run record to prove
        exactly which data this file contains.
    """
    if not table or not table.isidentifier():
        raise ValueError(f"invalid table name for snapshot path: {table!r}")
    if not run_id:
        raise ValueError("run_id must be a non-empty string")

    as_of = snapshot_date if snapshot_date is not None else datetime.now(UTC)
    day = as_of.astimezone(UTC).date().isoformat()

    out_dir = Path(state_dir) / "snapshots" / table / f"date={day}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.parquet"

    frame = _records_to_frame(records)
    frame.write_parquet(out_path)

    return SnapshotResult(
        path=out_path,
        table=table,
        run_id=run_id,
        row_count=len(records),
        data_snapshot_hash=data_snapshot_hash(records),
    )
