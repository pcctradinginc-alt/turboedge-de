"""Deep, slow, once-daily product archive -- research only, never the decision path.

Why this exists (Data Factory stage, CLAUDE.md "North Star"): gettex already
serves four issuers with live bid/ask (BNP, Goldman Sachs, UniCredit, HSBC),
but its adapter is capped at ``max_pages=20`` x 100 rows/page = 2,000 rows per
underlying against a DAX book of roughly 13,855 open-end turbos -- about 14%
of an already-working, already-validated, free source is being read. That cap
is the binding constraint on the one economic hypothesis this repo has never
tested: ``_pick_alternatives`` (``pipeline/scan.py``) needs comparable
products for the counterfactual selection edge, and most ledger entries had
none because BNP alone is ~92% of the universe at the live scan's depth.

Raising the cap on the LIVE scan path would be actively harmful, not helpful.
Measured: a scan's ``fetch_duration_s`` is already 227s (DAX) / 261s (NDX),
and ``configs/risk.yaml`` sets ``max_quote_age_at_decision_s: 450``. Full-depth
fetching adds roughly 120s per underlying at the project's <=1 request/s
politeness rule, which would push decision-time quote age past that gate and
cause products to be REJECTED on staleness -- more data would mean fewer
candidates, not more. So this fetch is deliberately decoupled from the
decision path: deep, slow, run once a day (the ``eod`` job, never one of the
five daily ``scan-all`` invocations), and it writes snapshots/instruments for
research only. It runs no pricing, no EV, no gates, no ledger entry and sends
no notification -- see ``run_product_archive``'s docstring.

Deliberately does NOT call ``pipeline.universe.run_universe``: that function
already persists (``Store.append_product_snapshots`` /
``Store.upsert_instruments``) as a side effect of its own return value, which
does not expose the ``upsert_instruments`` row count this module's
``ArchiveResult`` needs -- calling it and then persisting again here would
double-write ``product_snapshots`` (an append-only log, not a merge). Instead
this module reuses the same underlying utilities ``run_universe`` itself
reuses (``turboedge.universe.discover.merge_snapshots`` for per-ISIN
dedup/winner-selection) and the same warning-naming convention
``pipeline/scan.py`` uses when it turns a ``UniverseResult.source_errors``
entry into a warning (``f"product_source_failed:{name}:{err}"``,
``pipeline/scan.py`` around its ``run_universe`` call site) -- so a partial
fetch (one source down, others fine) is recorded as a warning here exactly as
it is there, never raised as an exception.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import structlog

from turboedge.adapters.base import ProductFetchContext
from turboedge.adapters.gettex import GettexAdapter
from turboedge.adapters.registry import ProductSourceAdapter
from turboedge.config import TurboEdgeConfig
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ProductSnapshot
from turboedge.universe.discover import merge_snapshots

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    """Outcome of one :func:`run_product_archive` call."""

    underlyings: tuple[str, ...]
    snapshots_written: int
    instruments_upserted: int
    #: issuer (e.g. "BNP Paribas", "Goldman Sachs", "UniCredit", "HSBC") ->
    #: number of merged snapshots seen for it this run. The point of this
    #: command is issuer BREADTH -- a run that comes back with only one key
    #: here should read as visibly disappointing, not as a quiet success.
    issuers_seen: dict[str, int]
    rows_fetched_by_underlying: dict[str, int]
    warnings: list[str]
    duration_s: float


def _add_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _boost_gettex_depth(adapters: Sequence[ProductSourceAdapter], max_pages: int) -> None:
    """Override every :class:`GettexAdapter` instance's page cap, for this call only.

    The registry (``adapters/registry.py``, out of scope for this change)
    builds ``GettexAdapter`` from ``configs/sources.yaml`` (``max_pages: 20``
    -- the value the live scan path must keep, since that cap is what keeps
    ``fetch_duration_s`` inside ``configs/risk.yaml``'s
    ``max_quote_age_at_decision_s`` gate). There is no config-level way to ask
    for a *second*, deeper-configured instance for one command without
    touching the registry, so this mutates the already-constructed instance's
    private ``_max_pages`` in place -- the smallest change that gets a
    genuinely deep fetch (150 pages x 100 rows/page ~= 15,000 rows, above
    DAX's observed ~13,855-row book) without altering ``registry.py`` or
    ``gettex.py``, both outside this task's file list, and without touching
    anything the scan path reads (a fresh set of adapters is built for every
    CLI invocation, so this never leaks into a later ``scan``/``scan-all``
    process). Every other adapter (BNP, Citi, CSV import) has no such cap and
    is untouched.
    """
    for adapter in adapters:
        if isinstance(adapter, GettexAdapter):
            adapter._max_pages = max_pages


def run_product_archive(
    cfg: TurboEdgeConfig,
    store: Store,
    *,
    product_adapters: Sequence[ProductSourceAdapter],
    underlying_ids: Sequence[str] | None = None,
    max_pages: int = 150,
    now: datetime | None = None,
) -> ArchiveResult:
    """Fetch every enabled product source at full depth and persist, nothing else.

    Deliberately does NOT run pricing, EV, gates, ledger writes or
    notifications -- this is the Data Factory stage, not the decision path
    (see module docstring). Every adapter is queried once with the full
    ``underlying_ids`` list; a failing adapter is recorded in ``warnings``
    (``product_source_failed:<name>:<error>``) and skipped, never raised --
    same convention ``pipeline/scan.py`` uses for
    ``UniverseResult.source_errors``. If every adapter fails, or the merged
    result is empty, that is *also* just a warning
    (``archive_no_products_fetched``) with an otherwise-empty
    :class:`ArchiveResult`, not an exception: a slow research archive with
    nothing to show for one day is not a reason to fail a CI job that also
    runs ``label``/``learn``/``position reevaluate``/``db compact`` in the
    same ``eod`` job.

    Args:
        cfg: Loaded TurboEdge-DE configuration. Used only to default
            ``underlying_ids`` to every enabled underlying
            (``cfg.universe.enabled_ids()``) when the caller does not supply
            an explicit list.
        store: Open DuckDB store to persist into.
        product_adapters: Already-constructed product-source adapters (e.g.
            from ``adapters.registry.build_product_adapters``) -- any
            :class:`~turboedge.adapters.gettex.GettexAdapter` among them has
            its page cap overridden to ``max_pages`` for this call only (see
            :func:`_boost_gettex_depth`); every other adapter is used as-is.
        underlying_ids: Canonical underlying_id values to request. Defaults
            to every enabled id in ``configs/universe.yaml``.
        max_pages: Deep page cap for gettex (default 150, i.e. ~15,000 rows
            at gettex's 100-rows/page, above DAX's observed ~13,855-row open-
            end turbo book). The live scan path is unaffected: it builds its
            own adapters from config (``max_pages: 20``) and never calls this
            function.
        now: Injectable clock for tests; defaults to ``datetime.now(UTC)``.

    Returns:
        :class:`ArchiveResult` with persisted counts and the issuer breakdown
        that is this command's entire point.
    """
    del now  # no pricing/EV/gates here -- nothing in this function is time-dependent
    started = time.monotonic()
    warnings: list[str] = []

    ids = list(underlying_ids) if underlying_ids else cfg.universe.enabled_ids()

    _boost_gettex_depth(product_adapters, max_pages)

    per_source_snapshots: list[Sequence[ProductSnapshot]] = []
    context = ProductFetchContext()
    for adapter in product_adapters:
        try:
            products = adapter.fetch_products(ids, context=context)
        except Exception as exc:
            logger.error("archive_source_failed", source=adapter.name, error=str(exc))
            _add_warning(warnings, f"product_source_failed:{adapter.name}:{exc}")
            continue
        per_source_snapshots.append(products)
        logger.info("archive_source_ok", source=adapter.name, count=len(products))

    merge_result = merge_snapshots(per_source_snapshots)
    products = merge_result.products

    if not products:
        _add_warning(warnings, "archive_no_products_fetched")

    snapshots_written = store.append_product_snapshots(products)
    instruments_upserted = store.upsert_instruments(products)

    issuers_seen: dict[str, int] = dict(Counter(p.issuer for p in products))

    rows_fetched_by_underlying: dict[str, int] = dict.fromkeys(ids, 0)
    for p in products:
        key = p.underlying_id or p.underlying_raw
        rows_fetched_by_underlying[key] = rows_fetched_by_underlying.get(key, 0) + 1

    duration_s = time.monotonic() - started
    logger.info(
        "archive_run_complete",
        underlyings=ids,
        snapshots_written=snapshots_written,
        instruments_upserted=instruments_upserted,
        issuers_seen=issuers_seen,
        duration_s=round(duration_s, 1),
    )

    return ArchiveResult(
        underlyings=tuple(ids),
        snapshots_written=snapshots_written,
        instruments_upserted=instruments_upserted,
        issuers_seen=issuers_seen,
        rows_fetched_by_underlying=rows_fetched_by_underlying,
        warnings=warnings,
        duration_s=round(duration_s, 1),
    )


__all__ = ["ArchiveResult", "run_product_archive"]
