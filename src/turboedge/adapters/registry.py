"""Adapter registry and factory for data source adapters.

Maintains a registry of product/reference-rate adapters, builds HTTP clients
with per-source configuration, and instantiates adapters only when enabled
in the config. Supports selective adapter activation via the `only` filter.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import structlog

from turboedge.adapters.base import (
    AdapterMetadata,
    HealthCheckResult,
    HttpClient,
    ProductFetchContext,
)
from turboedge.config import SourceConfig, TurboEdgeConfig
from turboedge.storage.schemas import ProductSnapshot, RejectedRatioDerivation

logger = structlog.get_logger(__name__)


@runtime_checkable
class ProductSourceAdapter(Protocol):
    """Structural contract for product (quote) source adapters.

    Every concrete product adapter (Deutsche Börse, Börse Stuttgart, issuer
    feeds) implements this protocol and is registered in PRODUCT_ADAPTER_FACTORIES.
    """

    @property
    def name(self) -> str:
        """Stable, unique adapter name (e.g., 'deutsche_boerse')."""
        ...

    def fetch_products(
        self,
        underlying_ids: Sequence[str],
        *,
        context: ProductFetchContext | None = None,
    ) -> list[ProductSnapshot]:
        """Fetch and return raw product data.

        Args:
            underlying_ids: List of underlying IDs to filter on (if applicable).
            context: Optional additive per-run context (Befund 2, see
                :class:`ProductFetchContext`) -- e.g. a same-run daily-close
                reference price per underlying, usable as a sanity
                cross-check. Every adapter must accept this kwarg; an
                adapter with no use for it simply ignores it.

        Returns:
            List of raw product records (type depends on the adapter).
        """
        ...

    def healthcheck(self) -> HealthCheckResult:
        """Perform a quick liveness/quality probe.

        Returns:
            HealthCheckResult indicating the adapter's availability and freshness.
        """
        ...

    def metadata(self) -> AdapterMetadata:
        """Return static metadata about this adapter.

        Returns:
            AdapterMetadata with name, kind, version, homepage.
        """
        ...


@runtime_checkable
class RejectedRatioDerivationSource(Protocol):
    """Optional add-on contract: an adapter that *derives* a pricing-critical
    field can hand back the attempts it discarded.

    Only ``adapters/gettex.py`` needs this today -- it is the one source that
    reconstructs ``ratio`` instead of reading it (no gettex endpoint reports
    a Bezugsverhaeltnis), so it is the one source that can reject a row for a
    reason worth studying. Every other adapter simply does not satisfy this
    protocol, and ``pipeline/universe.py``'s ``isinstance`` check skips it;
    nothing else changes.

    Deliberately a separate protocol rather than another method on
    :class:`ProductSourceAdapter`: that contract is what every product source
    must implement, and a CSV import or an issuer feed that reads ``ratio``
    straight from its source has nothing to report here. Equally deliberately
    a typed protocol rather than ``getattr(adapter, "...", ())`` in the
    pipeline -- duck-typing by attribute name would silently return nothing
    if this method were ever renamed, turning a wiring bug into permanent,
    invisible data loss, which is exactly the failure mode this whole record
    type exists to end (see :class:`RejectedRatioDerivation`).
    """

    def drain_rejected_ratio_derivations(self) -> Sequence[RejectedRatioDerivation]:
        """Return this fetch's discarded derivation attempts.

        "Drain" is the contract: the implementation must hand over the
        records it accumulated during the most recent ``fetch_products``
        call and must not return them again on a subsequent call, so a
        caller that persists them cannot write the same attempt twice.
        """
        ...


ProductAdapterFactory = Callable[[SourceConfig, HttpClient], ProductSourceAdapter]

# Module-level registry of available product adapters.
# Concrete adapter modules (e.g., adapters/deutsche_boerse.py) register
# themselves on import via register_product_adapter().
PRODUCT_ADAPTER_FACTORIES: dict[str, ProductAdapterFactory] = {}


class RegistryError(Exception):
    """Raised when adapter registration or lookup fails."""


def register_product_adapter(name: str, factory: ProductAdapterFactory) -> None:
    """Register a product adapter factory.

    Args:
        name: Unique adapter name (matches key in sources.yaml).
        factory: Callable that creates an adapter instance from (SourceConfig, HttpClient).

    Raises:
        ValueError: If an adapter with this name is already registered.
    """
    if name in PRODUCT_ADAPTER_FACTORIES:
        raise ValueError(
            f"product adapter {name!r} is already registered; duplicate registration is not allowed"
        )
    PRODUCT_ADAPTER_FACTORIES[name] = factory
    logger.debug("adapter_registered", name=name)


def load_builtin_adapters() -> None:
    """Import every built-in product adapter module so it self-registers.

    Each concrete adapter module (``csv_import``, ``issuer_feeds``, ...)
    calls ``register_product_adapter`` as an import-time side effect, guarded
    by its own "already registered" check. Plain Python module caching
    (``sys.modules``) already makes a repeated ``import`` of the same module
    a no-op, so this function is safe to call any number of times, from any
    number of call sites -- it never re-runs a module's registration logic
    after the first successful import.

    Imports are deliberately local to this function (not at module top
    level) to avoid a circular import: each adapter module itself imports
    from ``turboedge.adapters.registry`` to call ``register_product_adapter``
    -- the same lazy-import pattern already used by
    :func:`build_reference_healthchecks` for ``ecb``/``fallback_prices``.
    """
    importlib.import_module("turboedge.adapters.csv_import")
    importlib.import_module("turboedge.adapters.issuer_feeds")
    importlib.import_module("turboedge.adapters.gettex")


def build_http_client(src: SourceConfig) -> HttpClient:
    """Build an HttpClient from a SourceConfig.

    Args:
        src: Source configuration with timeout_s, min_interval_s, user_agent.

    Returns:
        HttpClient configured with retries, rate limiting, and User-Agent.
    """
    return HttpClient(
        user_agent=src.user_agent,
        timeout_s=src.timeout_s,
        min_interval_s=src.min_interval_s,
        max_retries=3,
    )


def build_product_adapters(
    cfg: TurboEdgeConfig, only: str | None = None
) -> list[ProductSourceAdapter]:
    """Instantiate all enabled product adapters.

    Only adapters that are:
    1. Marked `enabled: true` in cfg.sources
    2. Have a registered factory in PRODUCT_ADAPTER_FACTORIES
    3. Match the `only` filter (if provided)

    are instantiated. If `only` is provided but does not match any
    registered adapter, raises ValueError listing available names.

    Args:
        cfg: TurboEdgeConfig loaded from configs/*.yaml.
        only: Optional adapter name filter (exact match).

    Returns:
        List of instantiated ProductSourceAdapter instances.

    Raises:
        ValueError: If `only` is provided and matches no registered adapter.
    """
    load_builtin_adapters()

    adapters: list[ProductSourceAdapter] = []

    # Validate `only` if provided
    if only is not None and only not in PRODUCT_ADAPTER_FACTORIES:
        available = sorted(PRODUCT_ADAPTER_FACTORIES.keys())
        raise ValueError(
            f"unknown product adapter {only!r}; "
            f"available: {', '.join(available) or '(none registered)'}"
        )

    for source_name, source_cfg in cfg.sources.items():
        # Skip disabled sources
        if not source_cfg.enabled:
            continue

        # Skip if not in registry
        if source_name not in PRODUCT_ADAPTER_FACTORIES:
            logger.debug("adapter_not_registered", source=source_name)
            continue

        # Skip if filtered by `only`
        if only is not None and source_name != only:
            continue

        # Build and instantiate
        try:
            http = build_http_client(source_cfg)
            factory = PRODUCT_ADAPTER_FACTORIES[source_name]
            adapter = factory(source_cfg, http)
            adapters.append(adapter)
            logger.debug("adapter_instantiated", source=source_name, adapter_name=adapter.name)
        except Exception as exc:
            logger.error("adapter_instantiation_failed", source=source_name, error=str(exc))
            raise

    return adapters


def build_reference_healthchecks(cfg: TurboEdgeConfig) -> list[Callable[[], HealthCheckResult]]:
    """Build healthcheck callables for reference-rate sources (ECB, yfinance).

    Only sources that are enabled and have concrete implementations in this
    codebase are included.

    Args:
        cfg: TurboEdgeConfig loaded from configs/*.yaml.

    Returns:
        List of zero-argument callables, each returning a HealthCheckResult.
    """
    checks: list[Callable[[], HealthCheckResult]] = []

    # ECB EST (reference rate)
    if cfg.sources.get("ecb", None) is not None and cfg.sources["ecb"].enabled:
        try:
            from turboedge.adapters.ecb import EcbEstrAdapter

            http = build_http_client(cfg.sources["ecb"])
            fallback_rate = cfg.risk.reference_rate_fallback
            ecb_adapter = EcbEstrAdapter(http, fallback_rate=fallback_rate)
            checks.append(ecb_adapter.healthcheck)
            logger.debug("healthcheck_registered", source="ecb_estr")
        except Exception as exc:
            logger.error("ecb_adapter_setup_failed", error=str(exc))

    # yfinance underlying prices
    if cfg.sources.get("yfinance", None) is not None and cfg.sources["yfinance"].enabled:
        try:
            from turboedge.adapters.fallback_prices import YFinancePriceAdapter

            yf_adapter = YFinancePriceAdapter()
            checks.append(yf_adapter.healthcheck)
            logger.debug("healthcheck_registered", source="yfinance")
        except Exception as exc:
            logger.error("yfinance_adapter_setup_failed", error=str(exc))

    return checks


__all__ = [
    "PRODUCT_ADAPTER_FACTORIES",
    "ProductAdapterFactory",
    "ProductSourceAdapter",
    "RegistryError",
    "build_http_client",
    "build_product_adapters",
    "build_reference_healthchecks",
    "load_builtin_adapters",
    "register_product_adapter",
]
