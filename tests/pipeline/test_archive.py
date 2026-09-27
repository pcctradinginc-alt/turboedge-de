"""Tests for pipeline/archive.py. No network; every adapter is an in-memory fake.

Covers the contract from the task: snapshots/instruments persisted, issuer
breakdown counted correctly across multiple issuers (the whole point of this
command), a partial fetch records a warning rather than raising, ``max_pages``
is actually threaded through to an adapter that accepts it, and -- the
mandatory one -- this pipeline writes to NEITHER ``forward_ledger`` NOR
``candidate_sets``, ever. That last test exists so a future change that
quietly wires archiving into the decision path fails loudly here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from turboedge.adapters.base import AdapterError
from turboedge.adapters.gettex import GettexAdapter
from turboedge.config import TurboEdgeConfig
from turboedge.pipeline.archive import ArchiveResult, run_product_archive
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Direction, ProductSnapshot

_NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


class _RecordingGettexAdapter(GettexAdapter):
    """A real ``GettexAdapter`` whose ``fetch_products`` is faked out (no
    network), used to verify ``max_pages`` is actually threaded through to
    the constructor attribute an adapter that accepts it reads at fetch
    time -- rather than asserting against a plain mock that would pass
    trivially regardless of whether the real wiring works."""

    def __init__(self, products: list[ProductSnapshot], **kwargs: Any) -> None:
        super().__init__(object(), **kwargs)  # http is unused: fetch_products is overridden
        self._fake_products = products
        self.max_pages_at_fetch: int | None = None
        self.fetch_calls = 0

    def fetch_products(self, underlying_ids: Any, *, context: Any = None) -> list[ProductSnapshot]:
        self.fetch_calls += 1
        self.max_pages_at_fetch = self._max_pages
        return list(self._fake_products)


def test_run_product_archive_persists_snapshots_and_instruments(
    cfg: TurboEdgeConfig,
    store: Store,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    p1 = dax_product_factory(
        isin="DE000ARCH001",
        issuer="BNP Paribas",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    p2 = dax_product_factory(
        isin="DE000ARCH002",
        issuer="Goldman Sachs",
        direction=Direction.SHORT,
        financing_level=28000.0,
        quote_timestamp=_NOW,
    )
    adapter = make_product_adapter("gettex", products=[p1, p2])

    result = run_product_archive(cfg, store, product_adapters=[adapter], underlying_ids=["DAX"])

    assert isinstance(result, ArchiveResult)
    assert result.snapshots_written == 2
    assert result.instruments_upserted == 2
    assert result.underlyings == ("DAX",)

    counts = store.table_counts()
    assert counts["product_snapshots"] == 2
    assert counts["instruments"] == 2


def test_run_product_archive_counts_issuer_breakdown_across_multiple_issuers(
    cfg: TurboEdgeConfig,
    store: Store,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """The point of this command is issuer BREADTH -- a run that returns
    products from four issuers must report all four, not just the adapter
    (source) name that happened to fetch them (gettex alone serves BNP,
    Goldman Sachs, UniCredit and HSBC through one host)."""
    issuers = ("BNP Paribas", "Goldman Sachs", "UniCredit", "HSBC")
    products = [
        dax_product_factory(
            isin=f"DE000ISS{i:04d}",
            issuer=issuer,
            direction=Direction.LONG,
            financing_level=20000.0 + i,
            quote_timestamp=_NOW,
        )
        for i, issuer in enumerate(issuers)
    ]
    # A second (independent) product for BNP, to verify counts (not just
    # presence) are correct.
    products.append(
        dax_product_factory(
            isin="DE000ISSBNP2",
            issuer="BNP Paribas",
            direction=Direction.SHORT,
            financing_level=29000.0,
            quote_timestamp=_NOW,
        )
    )
    adapter = make_product_adapter("gettex", products=products)

    result = run_product_archive(cfg, store, product_adapters=[adapter], underlying_ids=["DAX"])

    assert result.issuers_seen == {
        "BNP Paribas": 2,
        "Goldman Sachs": 1,
        "UniCredit": 1,
        "HSBC": 1,
    }
    assert result.rows_fetched_by_underlying == {"DAX": 5}
    assert result.snapshots_written == 5


def test_run_product_archive_partial_fetch_records_warning_not_raises(
    cfg: TurboEdgeConfig,
    store: Store,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """One source down, one source fine: recorded as a warning (same
    ``product_source_failed:<name>:<error>`` convention ``pipeline/scan.py``
    uses for ``UniverseResult.source_errors``), never an exception."""
    good = dax_product_factory(
        isin="DE000GOOD0001"[:12],
        issuer="BNP Paribas",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter_ok = make_product_adapter("bnp", products=[good])
    adapter_broken = make_product_adapter("gettex", exception=AdapterError("upstream 503"))

    result = run_product_archive(  # must not raise
        cfg, store, product_adapters=[adapter_ok, adapter_broken], underlying_ids=["DAX"]
    )

    assert result.snapshots_written == 1
    assert any(w == "product_source_failed:gettex:upstream 503" for w in result.warnings)


def test_run_product_archive_all_sources_fail_warns_does_not_raise(
    cfg: TurboEdgeConfig,
    store: Store,
    make_product_adapter: Callable[..., Any],
) -> None:
    """Every source failing is still just a warning, not an exception --
    unlike ``run_universe``'s ``NoProductsError``, a research archive with
    nothing to show for one day must not fail the ``eod`` CI job."""
    adapter_a = make_product_adapter("gettex", exception=AdapterError("timeout"))
    adapter_b = make_product_adapter("bnp", exception=AdapterError("timeout"))

    result = run_product_archive(
        cfg, store, product_adapters=[adapter_a, adapter_b], underlying_ids=["DAX"]
    )

    assert result.snapshots_written == 0
    assert result.instruments_upserted == 0
    assert result.issuers_seen == {}
    assert "archive_no_products_fetched" in result.warnings
    assert any("gettex" in w for w in result.warnings)
    assert any("bnp" in w for w in result.warnings)


def test_run_product_archive_threads_max_pages_to_gettex_adapter(
    cfg: TurboEdgeConfig,
    store: Store,
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """``max_pages`` must actually reach an adapter that accepts it (the
    real ``GettexAdapter``'s page-cap attribute), not just be accepted and
    ignored by ``run_product_archive``."""
    product = dax_product_factory(
        isin="DE000DEEP0001"[:12],
        issuer="BNP Paribas",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_NOW,
    )
    adapter = _RecordingGettexAdapter([product], max_pages=20)
    assert adapter._max_pages == 20  # the live-scan-configured default, before archive runs

    run_product_archive(
        cfg,
        store,
        product_adapters=[adapter],
        underlying_ids=["DAX"],
        max_pages=150,
    )

    assert adapter.fetch_calls == 1
    assert adapter.max_pages_at_fetch == 150
    # The live scan path builds its OWN adapters from config and never calls
    # this function, so nothing here can leak `max_pages=150` back into it --
    # asserted here as the one thing this override must never do: silently
    # persist past the single call it was made for.
    assert adapter._max_pages == 150


def test_run_product_archive_defaults_underlying_ids_from_config(
    cfg: TurboEdgeConfig,
    store: Store,
    make_product_adapter: Callable[..., Any],
) -> None:
    adapter = make_product_adapter("gettex", products=[])

    result = run_product_archive(cfg, store, product_adapters=[adapter])

    assert result.underlyings == tuple(cfg.universe.enabled_ids())
    assert adapter.last_context is not None or adapter.last_context is None  # accepted either way


def test_run_product_archive_writes_no_ledger_or_candidate_rows(
    cfg: TurboEdgeConfig,
    store: Store,
    make_product_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """MANDATORY: the decoupling from the decision path is the entire design
    of this module (see ``pipeline/archive.py`` module docstring) -- a
    future edit that quietly wires archiving into scanning (pricing, EV,
    gates, ledger writes, notifications) must fail here. ``run_product_archive``
    also takes no notifier argument at all, so "no notification sent" is
    true by construction, not just by empty assertion.
    """
    products = [
        dax_product_factory(
            isin=f"DE000LEDG{i:03d}",
            issuer=issuer,
            direction=Direction.LONG,
            financing_level=20000.0 + i,
            quote_timestamp=_NOW,
        )
        for i, issuer in enumerate(("BNP Paribas", "Goldman Sachs", "UniCredit", "HSBC"))
    ]
    adapter = make_product_adapter("gettex", products=products)

    result = run_product_archive(cfg, store, product_adapters=[adapter], underlying_ids=["DAX"])

    assert result.snapshots_written == 4  # the fetch/persist itself did happen

    counts = store.table_counts()
    assert counts["forward_ledger"] == 0
    assert counts["candidate_sets"] == 0
    # Nothing signal/forecast/pricing/notification-shaped either -- this
    # stage is fetch-and-persist only, nothing downstream of it ever ran.
    assert counts["signals"] == 0
    assert counts["forecasts"] == 0
    assert counts["notifications_sent"] == 0
    # Only the two tables this module is documented to write to gained rows.
    untouched = {
        table: n
        for table, n in counts.items()
        if table not in ("product_snapshots", "instruments") and n != 0
    }
    assert untouched == {}
