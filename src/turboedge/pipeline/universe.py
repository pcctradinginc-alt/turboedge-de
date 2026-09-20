"""Universe pipeline: fetch products from every configured source, merge, persist.

Formula/order reference: Build Contract "Scan-Pipeline" step 5 and Master Spec
§43 ("Scanner Pipeline") steps 10-12. This module is also invoked standalone
by ``turboedge universe`` (CLI, owned by another agent) and by
``pipeline/scan.py`` step 5, always with the same contract: every configured
:class:`~turboedge.adapters.registry.ProductSourceAdapter` is queried, a
failure in one source is logged and skipped rather than aborting the whole
run (CLAUDE.md rule 30's "check source health" pairs with this: a source
being down must degrade gracefully, never crash the pipeline), and the merged
result is persisted to both DuckDB (mutable, queryable) and an immutable
per-run Parquet snapshot (CLAUDE.md rule 33).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from turboedge.adapters.base import ProductFetchContext
from turboedge.adapters.registry import ProductSourceAdapter, RejectedRatioDerivationSource
from turboedge.config import TurboEdgeConfig, config_hash
from turboedge.provenance import git_commit
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ProductSnapshot, RejectedRatioDerivation
from turboedge.storage.snapshots import SnapshotResult, write_snapshot_parquet
from turboedge.universe.discover import FieldConflict, merge_snapshots

logger = structlog.get_logger(__name__)

_SNAPSHOT_TABLE = "product_snapshots"


class NoProductsError(Exception):
    """Raised when every product source failed, or the merged result is empty.

    ``source_errors`` maps adapter name -> error message for every source
    that raised, so callers (the CLI's ``sources health``/``universe``/
    ``scan`` commands) can report exactly what went wrong rather than a bare
    "no products" message.
    """

    def __init__(self, source_errors: dict[str, str]) -> None:
        self.source_errors = dict(source_errors)
        if source_errors:
            detail = "; ".join(f"{name}: {msg}" for name, msg in source_errors.items())
            message = f"no products from any source (all sources failed): {detail}"
        else:
            message = "no products returned by any source (0 products, no source errors)"
        super().__init__(message)


@dataclass(frozen=True)
class UniverseResult:
    """Outcome of one :func:`run_universe` call."""

    run_id: str
    products: list[ProductSnapshot]
    conflicts: list[FieldConflict]
    source_errors: dict[str, str]
    counts_by_source: dict[str, int]
    snapshot: SnapshotResult | None


def run_universe(
    cfg: TurboEdgeConfig,
    store: Store,
    state_dir: str | Path,
    adapters: Sequence[ProductSourceAdapter],
    underlying_ids: Sequence[str],
    *,
    run_id: str,
    manage_run: bool = True,
    context: ProductFetchContext | None = None,
) -> UniverseResult:
    """Fetch products from every adapter, merge/dedupe, persist, snapshot.

    Each adapter's ``fetch_products()`` is called independently; an
    exception from one adapter is caught, logged, and recorded in
    ``source_errors`` -- it never aborts the whole run (other sources still
    contribute). If every adapter fails, or the merged result has zero
    products, :class:`NoProductsError` is raised (with whatever
    ``source_errors`` were collected) instead of silently persisting an
    empty universe.

    On success, the merged, deduplicated products are appended to
    ``product_snapshots``, ``instruments`` is upserted from them, and an
    immutable Parquet snapshot is written under
    ``<state_dir>/snapshots/product_snapshots/date=.../<run_id>.parquet``.

    Args:
        cfg: Loaded TurboEdge-DE configuration; used for ``config_hash`` when
            ``manage_run`` is True (adapters are already built/configured by
            the caller).
        store: Open DuckDB store to persist into.
        state_dir: Root state directory (``$TURBOEDGE_STATE_DIR``), used for
            the Parquet snapshot archive.
        adapters: Product-source adapters to query, e.g. from
            ``adapters.registry.build_product_adapters``.
        underlying_ids: Canonical underlying_id values to request from each
            adapter.
        run_id: The run this universe fetch belongs to (see
            ``turboedge.provenance.new_run_id``).
        manage_run: When True (default), this call owns the ``runs`` table
            row for ``run_id``: it calls ``store.start_run``/``finish_run``
            around the whole fetch (Build Contract: "store.start_run/
            finish_run ... in beiden Pipelines"), recording ``status="error"``
            with the exception message on failure. Set to False when this is
            invoked as a *sub-step* of another pipeline that already owns
            ``run_id``'s row for the same run (``pipeline.scan.run_scan``
            step 5 reuses the scan's own ``run_id`` here; starting a second
            ``runs`` row with that same primary key would fail) -- in that
            case the caller is responsible for the run lifecycle.

        context: Optional additive per-run context (Befund 2) threaded
            through to every adapter's ``fetch_products`` -- e.g. a same-run
            daily-close reference price per underlying, so a source like
            gettex can sanity-check its own internally-derived reference
            spot even when it's the only source available this run. See
            :class:`~turboedge.adapters.base.ProductFetchContext`.

    Raises:
        NoProductsError: if every adapter failed, or none returned any
            product that survived merging.
    """
    if not manage_run:
        return _run_universe_body(cfg, store, state_dir, adapters, underlying_ids, run_id, context)

    started_at = datetime.now(UTC)
    store.start_run(
        run_id,
        command="universe",
        config_hash=config_hash(cfg),
        git_commit=git_commit(),
        started_at=started_at,
    )
    try:
        result = _run_universe_body(
            cfg, store, state_dir, adapters, underlying_ids, run_id, context
        )
    except Exception as exc:
        store.finish_run(run_id, status="error", error=str(exc), finished_at=datetime.now(UTC))
        raise
    store.finish_run(run_id, status="ok", finished_at=datetime.now(UTC))
    return result


def _run_universe_body(
    cfg: TurboEdgeConfig,
    store: Store,
    state_dir: str | Path,
    adapters: Sequence[ProductSourceAdapter],
    underlying_ids: Sequence[str],
    run_id: str,
    context: ProductFetchContext | None = None,
) -> UniverseResult:
    del cfg  # not needed beyond config_hash(), which only the manage_run wrapper computes

    per_source_snapshots: list[Sequence[ProductSnapshot]] = []
    source_errors: dict[str, str] = {}
    counts_by_source: dict[str, int] = {}

    # Fetch every adapter CONCURRENTLY rather than one-after-another.
    #
    # This does not weaken the "politeness" rule (CLAUDE.md/Build Contract:
    # <=1 request/s per host, never parallelized against the same host): each
    # adapter owns its own `HttpClient` (`adapters.registry.build_http_client`,
    # one instance per `configs/sources.yaml` entry) with its own per-host
    # `_HostRateLimiter`, and BNP (derivate.bnpparibas.com), Citi
    # (de.citifirst.com) and gettex (gettex.wsd.com) are three distinct hosts
    # -- so within any one adapter, its own paginated requests stay exactly as
    # serialized (>= `min_interval_s` apart) as before. Only the *previously
    # accidental* serialization ACROSS adapters/hosts (the plain `for adapter
    # in adapters:` loop below, before this change) is removed: nothing
    # requires those independent per-host request streams to also wait on
    # each other's wall-clock time.
    #
    # Measured impact (Build Contract freshness/duration review,
    # 2026-09-14): before this change, a DAX scan's total fetch_duration_s
    # (evaluation_time - fetch_start, `pipeline/scan.py`) was close to the
    # SUM of each adapter's own fetch time (gettex's ~20-page pagination
    # dominates at ~1.5s/page; BNP and Citi each add their own double-digit
    # seconds on top) -- exactly the sequential-loop shape below. Running the
    # (at most 3, all product-)adapters concurrently instead bounds
    # fetch_duration_s by the SLOWEST single adapter rather than their sum,
    # which is what actually determines how old the freshest possible
    # candidate quote can be at evaluation time (`ranking/gates.py`'s
    # "quote_age_at_decision" REJECT reason, `configs/risk.yaml`'s
    # `max_quote_age_at_decision_s`).
    per_adapter_duration_s: dict[str, float] = {}

    def _fetch_one(adapter: ProductSourceAdapter) -> list[ProductSnapshot]:
        started = time.monotonic()
        try:
            return adapter.fetch_products(underlying_ids, context=context)
        finally:
            per_adapter_duration_s[adapter.name] = round(time.monotonic() - started, 1)

    with ThreadPoolExecutor(max_workers=max(len(adapters), 1)) as pool:
        future_by_adapter = {adapter: pool.submit(_fetch_one, adapter) for adapter in adapters}
        # Iterate in the original, deterministic adapter order (not
        # completion order) so logging/`source_errors`/`counts_by_source`
        # stay reproducible run-to-run regardless of which host happened to
        # answer first; `merge_snapshots` below is itself order-independent
        # (freshest-quote-wins, see `universe/discover.py::_pick_winner`), so
        # this ordering choice affects only log/diagnostic readability, never
        # which product snapshot wins a conflict.
        for adapter, future in future_by_adapter.items():
            try:
                products = future.result()
            except Exception as exc:
                logger.error(
                    "universe_source_failed",
                    source=adapter.name,
                    error=str(exc),
                    fetch_duration_s=per_adapter_duration_s.get(adapter.name),
                )
                source_errors[adapter.name] = str(exc)
                continue
            counts_by_source[adapter.name] = len(products)
            per_source_snapshots.append(products)
            logger.info(
                "universe_source_ok",
                source=adapter.name,
                count=len(products),
                fetch_duration_s=per_adapter_duration_s.get(adapter.name),
            )

    merge_result = merge_snapshots(per_source_snapshots)

    for conflict in merge_result.conflicts:
        logger.warning(
            "universe_field_conflict",
            isin=conflict.isin,
            field=conflict.field,
            values=conflict.values,
            sources=conflict.sources,
        )

    # Persist discarded ratio-derivation attempts BEFORE the empty-result
    # check below, not after. A source that rejected every row it saw
    # contributes no products at all, so an all-sources-empty run raises
    # `NoProductsError` -- and that is precisely the run whose rejection
    # records are most worth keeping. Writing them after the raise would
    # discard the research material exactly when the derivation failed
    # hardest, which is the failure mode `RejectedRatioDerivation` exists to
    # end (CLAUDE.md rule 31). These records are never merged: a discarded
    # attempt is not a product, has no verified ratio, and competes with no
    # other source for the same ISIN -- `merge_snapshots` is about picking a
    # winner between sources quoting the same instrument, which cannot
    # apply here.
    rejected_derivations: list[RejectedRatioDerivation] = []
    for adapter in adapters:
        if isinstance(adapter, RejectedRatioDerivationSource):
            rejected_derivations.extend(adapter.drain_rejected_ratio_derivations())
    if rejected_derivations:
        store.append_rejected_ratio_derivations(rejected_derivations)
        logger.info(
            "universe_rejected_ratio_derivations_persisted",
            count=len(rejected_derivations),
            by_outcome={
                outcome: sum(1 for r in rejected_derivations if r.outcome == outcome)
                for outcome in sorted({r.outcome for r in rejected_derivations})
            },
        )

    if not merge_result.products:
        raise NoProductsError(source_errors)

    store.append_product_snapshots(merge_result.products)
    store.upsert_instruments(merge_result.products)

    snapshot = write_snapshot_parquet(
        _SNAPSHOT_TABLE, merge_result.products, run_id, Path(state_dir)
    )

    logger.info(
        "universe_run_complete",
        run_id=run_id,
        product_count=len(merge_result.products),
        conflict_count=len(merge_result.conflicts),
        source_errors=list(source_errors),
    )

    return UniverseResult(
        run_id=run_id,
        products=merge_result.products,
        conflicts=merge_result.conflicts,
        source_errors=source_errors,
        counts_by_source=counts_by_source,
        snapshot=snapshot,
    )


__all__ = ["NoProductsError", "UniverseResult", "run_universe"]
