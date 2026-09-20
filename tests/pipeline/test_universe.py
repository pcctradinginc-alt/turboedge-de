"""Tests for pipeline/universe.py. No network; all adapters are in-memory fakes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import structlog.testing

from turboedge.adapters.base import AdapterError, ProductFetchContext
from turboedge.config import TurboEdgeConfig
from turboedge.pipeline.universe import NoProductsError, run_universe
from turboedge.provenance import new_run_id
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    ProductSnapshot,
    RatioDerivationOutcome,
    RejectedRatioDerivation,
)

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


def _rejected(isin: str) -> RejectedRatioDerivation:
    return RejectedRatioDerivation(
        isin=isin,
        wkn=None,
        issuer="BNP Paribas",
        underlying_id="DAX",
        underlying_raw="DAX (Performance)",
        direction=Direction.LONG,
        outcome=RatioDerivationOutcome.RATIO_REJECTED,
        detail="ratio_raw did not snap to the canonical grid within tolerance",
        leverage=12.5,
        financing_level=23000.0,
        knockout_barrier=23000.0,
        bid=None,
        ask=None,
        reference_spot=25000.0,
        quote_timestamp=_NOW,
        observed_at=_NOW,
        source="gettex",
        parser_version="gettex/1",
        raw_hash="deadbeef",
    )


def test_run_universe_persists_rejected_ratio_derivations(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """A source that derives a pricing-critical field hands back what it
    discarded, and the pipeline persists it alongside the products.

    Adapters that do not satisfy ``RejectedRatioDerivationSource`` are
    untouched -- the ``isinstance`` check skips them, which is why this is a
    separate protocol rather than another method every product source would
    have to implement.
    """
    product = dax_product_factory(
        isin="DE000GOOD001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter = make_product_adapter(
        "source_a",
        products=[product],
        rejected_ratio_derivations=[_rejected("DE000BAD0001"), _rejected("DE000BAD0002")],
    )

    run_universe(cfg, store, tmp_path, [adapter], ["DAX"], run_id=new_run_id())

    persisted = store.list_rejected_ratio_derivations()
    assert {r.isin for r in persisted} == {"DE000BAD0001", "DE000BAD0002"}
    # The discarded rows never cross over into the priceable universe: they
    # have no verified ratio, so nothing here may ever be priced, ranked or
    # gated. Asserted against `product_snapshots` specifically -- that is the
    # table `run_universe` actually writes, and the one everything
    # downstream reads.
    snapshot_isins = {
        row[0]
        for row in store._conn.execute(  # test-only introspection
            "SELECT DISTINCT isin FROM product_snapshots"
        ).fetchall()
    }
    assert snapshot_isins == {"DE000GOOD001"}
    assert adapter.drain_calls == 1


def test_rejected_ratio_derivations_survive_a_no_products_run(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
) -> None:
    """The case that decides where the write belongs.

    A source that rejected every row it saw contributes no products, so the
    run raises ``NoProductsError`` -- and that is exactly the run whose
    rejection records matter most. Persisting them after the raise would
    discard the research material precisely when the derivation failed
    hardest, which is the failure mode this record type exists to end.
    """
    adapter = make_product_adapter(
        "source_a",
        products=[],
        rejected_ratio_derivations=[_rejected("DE000ALLBAD1")],
    )

    with pytest.raises(NoProductsError):
        run_universe(cfg, store, tmp_path, [adapter], ["DAX"], run_id=new_run_id())

    persisted = store.list_rejected_ratio_derivations()
    assert [r.isin for r in persisted] == ["DE000ALLBAD1"]


def test_run_universe_ignores_adapters_without_the_rejection_protocol(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """An adapter that does not implement the optional protocol is skipped
    silently -- no crash, no empty rows written."""

    class MinimalAdapter:
        name = "minimal"

        def __init__(self, products: list[ProductSnapshot]) -> None:
            self._products = products

        def fetch_products(
            self, underlying_ids: Any, *, context: Any = None
        ) -> list[ProductSnapshot]:
            return list(self._products)

    product = dax_product_factory(
        isin="DE000MIN0001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter = MinimalAdapter([product])

    run_universe(cfg, store, tmp_path, [adapter], ["DAX"], run_id=new_run_id())  # type: ignore[list-item]

    assert store.list_rejected_ratio_derivations() == []


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


# --------------------------------------------------------------------------
# concurrent per-adapter fetch (Build Contract freshness/duration review,
# 2026-09-14): _run_universe_body now fetches every adapter's
# fetch_products() in its own thread (ThreadPoolExecutor) instead of one
# after another, since BNP/Citi/gettex are three independent hosts already
# rate-limited per-host (adapters/base.py's _HostRateLimiter). These tests
# cover the concurrency-specific behavior the sequential-loop tests above
# never exercised: every adapter is still called even when another fails or
# is slower, a failing adapter still lands in source_errors without
# blocking the others, and logging/counts_by_source stay deterministic in
# original adapter order regardless of which host's thread actually
# finishes first.
# --------------------------------------------------------------------------


def test_run_universe_calls_every_adapter_exactly_once(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    names = ("source_a", "source_b", "source_c")
    isins = ("DE000SRCA0001", "DE000SRCB0001", "DE000SRCC0001")
    products = {
        name: [
            dax_product_factory(
                isin=isin[:12],
                issuer=name,
                direction=Direction.LONG,
                financing_level=20000.0,
                quote_timestamp=_NOW,
            )
        ]
        for name, isin in zip(names, isins, strict=True)
    }
    adapters = [make_product_adapter(name, products=prods) for name, prods in products.items()]

    run_universe(cfg, store, tmp_path, adapters, ["DAX"], run_id=new_run_id())

    assert [a.fetch_calls for a in adapters] == [1, 1, 1]


def test_run_universe_slow_failure_does_not_block_faster_adapters(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """A slow, failing adapter must not prevent a faster adapter's products
    from being fetched and merged -- the opposite of the old sequential
    ``for adapter in adapters:`` loop, where a slow adapter earlier in the
    list would simply delay (not block) every later one, but a THREADING
    regression could plausibly let one thread's exception propagate and
    cancel the pool. It must not."""
    good = dax_product_factory(
        isin="DE000LONG001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )

    def _slow_failure() -> None:
        time.sleep(0.2)

    adapter_slow_broken = make_product_adapter(
        "source_slow_broken", exception=AdapterError("timeout"), on_fetch=_slow_failure
    )
    adapter_fast_ok = make_product_adapter("source_fast_ok", products=[good])

    result = run_universe(
        cfg, store, tmp_path, [adapter_slow_broken, adapter_fast_ok], ["DAX"], run_id=new_run_id()
    )

    assert result.source_errors == {"source_slow_broken": "timeout"}
    assert result.counts_by_source == {"source_fast_ok": 1}
    assert len(result.products) == 1
    assert adapter_fast_ok.fetch_calls == 1


def test_run_universe_counts_by_source_order_is_adapter_order_not_completion_order(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """``counts_by_source`` (and the matching ``universe_source_ok`` log
    lines) must be populated in the ORIGINAL adapter order passed to
    ``run_universe``, not in whichever order each thread happens to finish
    -- reproducible logs/diagnostics regardless of network timing. Here
    adapter "source_first" is deliberately the SLOWEST (sleeps longest) and
    "source_third" the fastest, so completion order is the exact reverse of
    adapter order; the assertion below only passes if the result iterates
    in adapter order."""
    p1 = dax_product_factory(
        isin="DE000FIRST01",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    p2 = dax_product_factory(
        isin="DE000SECND01",
        issuer="BankB",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    p3 = dax_product_factory(
        isin="DE000THIRD01",
        issuer="BankC",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter_first = make_product_adapter(
        "source_first", products=[p1], on_fetch=lambda: time.sleep(0.3)
    )
    adapter_second = make_product_adapter(
        "source_second", products=[p2], on_fetch=lambda: time.sleep(0.15)
    )
    adapter_third = make_product_adapter("source_third", products=[p3])

    with structlog.testing.capture_logs() as logs:
        result = run_universe(
            cfg,
            store,
            tmp_path,
            [adapter_first, adapter_second, adapter_third],
            ["DAX"],
            run_id=new_run_id(),
        )

    assert list(result.counts_by_source.keys()) == ["source_first", "source_second", "source_third"]

    ok_events = [e for e in logs if e.get("event") == "universe_source_ok"]
    assert [e["source"] for e in ok_events] == ["source_first", "source_second", "source_third"]


def test_run_universe_records_per_adapter_fetch_duration_on_success_and_failure(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """Every adapter's own fetch wall-clock time is measured (``time.monotonic``
    around ``fetch_products()``, ``finally``-guarded) and logged, whether it
    succeeded or raised -- diagnosing "why is fetch_duration_s high" needs to
    know if a slow, FAILING source was the cause, not just slow successful
    ones."""
    good = dax_product_factory(
        isin="DE000LONG001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter_ok = make_product_adapter(
        "source_ok", products=[good], on_fetch=lambda: time.sleep(0.05)
    )
    adapter_broken = make_product_adapter(
        "source_broken", exception=AdapterError("upstream 503"), on_fetch=lambda: time.sleep(0.05)
    )

    with structlog.testing.capture_logs() as logs:
        run_universe(
            cfg, store, tmp_path, [adapter_ok, adapter_broken], ["DAX"], run_id=new_run_id()
        )

    ok_event = next(e for e in logs if e.get("event") == "universe_source_ok")
    failed_event = next(e for e in logs if e.get("event") == "universe_source_failed")
    assert ok_event["fetch_duration_s"] is not None and ok_event["fetch_duration_s"] >= 0.0
    assert failed_event["fetch_duration_s"] is not None and failed_event["fetch_duration_s"] >= 0.0


def test_run_universe_adapters_actually_run_concurrently(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """The one test in this file that a regression back to the old
    sequential ``for adapter in adapters:`` loop cannot pass.

    Every one of the tests above this one (calls-every-adapter,
    failure-isolation, deterministic ordering, duration-on-failure) holds
    just as well under the OLD sequential loop -- a slow/failing adapter
    there only delays or is skipped, it never blocks a later one, so none
    of those assertions actually distinguish "fetched one after another"
    from "fetched concurrently". This test does: three fake adapters each
    block on a shared ``threading.Barrier(3)`` the instant their
    ``fetch_products()`` is called. A barrier only releases once all three
    parties have arrived. Under genuine concurrent fetching (one thread per
    adapter, as ``pipeline/universe.py`` does today), all three threads
    call ``fetch_products()`` at roughly the same time, all three reach the
    barrier within milliseconds of each other, and it releases immediately.
    Under a REGRESSION to the sequential loop, only one adapter's
    ``fetch_products()`` is ever running at a time -- the first adapter
    blocks alone at the barrier, waiting for two parties that can never
    arrive (the second and third adapters' ``fetch_products()`` calls
    haven't started yet, and never will until the first one returns) -- so
    the wait times out and raises ``BrokenBarrierError``, which this
    adapter's ``fetch_products()`` propagates as an ordinary fetch failure.
    A generous 5s barrier timeout keeps this from false-alarming on a
    loaded machine; the wall-clock assertion below is a secondary,
    ADDITIONAL check (never a substitute for the barrier itself, which is
    the reliable signal here).
    """
    n = 3
    barrier_timeout_s = 5.0
    barrier = threading.Barrier(n, timeout=barrier_timeout_s)

    # Distinct 12-character ISINs, passed through verbatim: `merge_snapshots`
    # deduplicates by ISIN, so three 13-character values truncated to 12
    # ("DE000BARR0001"[:12] == "DE000BARR0002"[:12]) would collapse into a
    # single merged product and the product-count assertion below would fail
    # even while the barrier proves the fetches really did run concurrently.
    isins = ("DE000BARR001", "DE000BARR002", "DE000BARR003")
    adapters = [
        make_product_adapter(
            f"source_{i}",
            products=[
                dax_product_factory(
                    isin=isin,
                    issuer=f"source_{i}",
                    direction=Direction.LONG,
                    financing_level=20000.0,
                    quote_timestamp=_NOW,
                )
            ],
            on_fetch=barrier.wait,
        )
        for i, isin in enumerate(isins)
    ]

    started = time.monotonic()
    result = run_universe(cfg, store, tmp_path, adapters, ["DAX"], run_id=new_run_id())
    elapsed = time.monotonic() - started

    assert result.source_errors == {}, (
        "one or more adapters timed out waiting at the barrier for the "
        "other two to arrive (BrokenBarrierError) -- this means "
        "fetch_products() calls are no longer running concurrently; "
        "parallelization was lost. source_errors: " + repr(result.source_errors)
    )
    assert len(result.products) == n
    # Secondary wall-clock sanity check, additional to the barrier above:
    # three concurrent barrier waits resolve in well under a second; three
    # SEQUENTIAL ones would each individually time out after
    # barrier_timeout_s, so a regression would take >= 3 * 5s = 15s here.
    assert elapsed < barrier_timeout_s
