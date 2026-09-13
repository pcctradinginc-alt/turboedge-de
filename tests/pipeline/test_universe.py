"""Tests for pipeline/universe.py. No network; all adapters are in-memory fakes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from turboedge.adapters.base import AdapterError, ProductFetchContext
from turboedge.config import TurboEdgeConfig
from turboedge.pipeline.universe import NoProductsError, run_universe
from turboedge.provenance import new_run_id
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Direction, ProductSnapshot

_NOW = datetime(2026, 9, 10, 15, 30, tzinfo=UTC)


def _run_status(store: Store, run_id: str) -> tuple[str, str | None]:
    row = store._conn.execute(  # accessing the private connection: test-only introspection
        "SELECT status, error FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    assert row is not None, f"no runs row for {run_id!r}"
    return (row[0], row[1])


def test_run_universe_merges_persists_and_snapshots(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    long_a = dax_product_factory(
        isin="DE000LONG001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    short_b = dax_product_factory(
        isin="DE000SHRT001",
        issuer="BankB",
        direction=Direction.SHORT,
        financing_level=28000.0,
        quote_timestamp=_NOW,
    )

    adapter_a = make_product_adapter("source_a", products=[long_a])
    adapter_b = make_product_adapter("source_b", products=[short_b])

    run_id = new_run_id()
    result = run_universe(cfg, store, tmp_path, [adapter_a, adapter_b], ["DAX"], run_id=run_id)

    assert result.run_id == run_id
    assert {p.isin for p in result.products} == {"DE000LONG001", "DE000SHRT001"}
    assert result.source_errors == {}
    assert result.counts_by_source == {"source_a": 1, "source_b": 1}
    assert result.conflicts == []
    assert result.snapshot is not None
    assert result.snapshot.row_count == 2
    assert result.snapshot.path.exists()

    counts = store.table_counts()
    assert counts["product_snapshots"] == 2
    assert counts["instruments"] == 2
    assert counts["runs"] == 1
    assert _run_status(store, run_id) == ("ok", None)


def test_run_universe_detects_field_conflicts_between_sources(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    # Same ISIN, disagreeing financing_level (and, since it defaults from it,
    # knockout_barrier too) between two sources -> conflicts on both fields.
    version_a = dax_product_factory(
        isin="DE000CONF0001"[:12],
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    version_b = dax_product_factory(
        isin="DE000CONF0001"[:12],
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20500.0,
        quote_timestamp=_NOW,
    )

    adapter_a = make_product_adapter("source_a", products=[version_a])
    adapter_b = make_product_adapter("source_b", products=[version_b])

    result = run_universe(
        cfg, store, tmp_path, [adapter_a, adapter_b], ["DAX"], run_id=new_run_id()
    )

    assert len(result.products) == 1  # deduped to one winner
    conflict_fields = {c.field for c in result.conflicts}
    assert conflict_fields == {"financing_level", "knockout_barrier"}
    assert all(c.isin == version_a.isin for c in result.conflicts)


def test_run_universe_one_source_fails_other_succeeds(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    good = dax_product_factory(
        isin="DE000LONG001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter_ok = make_product_adapter("source_ok", products=[good])
    adapter_broken = make_product_adapter("source_broken", exception=AdapterError("upstream 503"))

    result = run_universe(
        cfg, store, tmp_path, [adapter_ok, adapter_broken], ["DAX"], run_id=new_run_id()
    )

    assert len(result.products) == 1
    assert result.source_errors == {"source_broken": "upstream 503"}
    assert result.counts_by_source == {"source_ok": 1}


def test_run_universe_all_sources_fail_raises_no_products_error(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
) -> None:
    adapter_a = make_product_adapter("source_a", exception=AdapterError("timeout"))
    adapter_b = make_product_adapter("source_b", exception=ValueError("bad parse"))

    run_id = new_run_id()
    with pytest.raises(NoProductsError) as excinfo:
        run_universe(cfg, store, tmp_path, [adapter_a, adapter_b], ["DAX"], run_id=run_id)

    assert excinfo.value.source_errors == {"source_a": "timeout", "source_b": "bad parse"}
    # nothing persisted for an all-failed universe fetch
    counts = store.table_counts()
    assert counts["product_snapshots"] == 0
    assert counts["instruments"] == 0
    status, error = _run_status(store, run_id)
    assert status == "error"
    assert error is not None


def test_run_universe_zero_products_raises_no_products_error(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
) -> None:
    adapter = make_product_adapter("source_a", products=[])
    with pytest.raises(NoProductsError) as excinfo:
        run_universe(cfg, store, tmp_path, [adapter], ["DAX"], run_id=new_run_id())
    assert excinfo.value.source_errors == {}


def test_run_universe_manage_run_records_error_status_on_failure(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
) -> None:
    adapter = make_product_adapter("source_a", exception=AdapterError("down"))
    run_id = new_run_id()
    with pytest.raises(NoProductsError):
        run_universe(cfg, store, tmp_path, [adapter], ["DAX"], run_id=run_id)

    status, error = _run_status(store, run_id)
    assert status == "error"
    assert error is not None and "down" in error


def test_run_universe_manage_run_false_does_not_touch_runs_table(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """When called as a sub-step (manage_run=False), no runs-table row is written here."""
    product = dax_product_factory(
        isin="DE000LONG001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter = make_product_adapter("source_a", products=[product])
    run_id = new_run_id()

    result = run_universe(cfg, store, tmp_path, [adapter], ["DAX"], run_id=run_id, manage_run=False)
    assert len(result.products) == 1
    assert store.table_counts()["runs"] == 0


def test_run_universe_threads_context_to_every_adapter(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """Befund 2 (2026-09-13 measurement session): ``run_universe`` must pass
    an explicit ``context`` through to every adapter's ``fetch_products`` --
    previously it always called ``fetch_products(underlying_ids)`` with
    nothing else, so a source like gettex that accepts an optional
    same-run daily-close cross-check never actually received one from the
    real pipeline."""
    product = dax_product_factory(
        isin="DE000LONG001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter = make_product_adapter("source_a", products=[product])
    context = ProductFetchContext(daily_close_reference={"DAX": 24000.0})

    run_universe(cfg, store, tmp_path, [adapter], ["DAX"], run_id=new_run_id(), context=context)

    assert adapter.last_context is context
    assert adapter.last_context is not None
    assert adapter.last_context.daily_close_reference == {"DAX": 24000.0}
