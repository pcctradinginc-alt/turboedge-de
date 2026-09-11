"""Config-driven universe filters: leverage range, issuer allow-list, bid-only, knocked-out.

Leverage itself is computed by ``pricing/intrinsic.py`` (not part of this
milestone's scope); this module only *applies* bounds to an already-computed
value so it stays a pure, dependency-free filter step usable from
``universe/discover.py`` or the scan pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from turboedge.storage.schemas import ProductSnapshot


@dataclass(frozen=True)
class ProductFilter:
    """Declarative filter configuration (typically sourced from risk.yaml/universe.yaml)."""

    min_leverage: float | None = None
    max_leverage: float | None = None
    allowed_issuers: frozenset[str] | None = None  # None = no restriction
    exclude_bid_only: bool = True
    exclude_knocked_out: bool = True


def apply_filters(
    snapshots: Iterable[ProductSnapshot],
    product_filter: ProductFilter,
    *,
    leverage_lookup: Callable[[ProductSnapshot], float | None] | None = None,
) -> list[ProductSnapshot]:
    """Return the subset of ``snapshots`` that passes ``product_filter``.

    Args:
        snapshots: Candidate snapshots (already deduplicated/merged).
        product_filter: Bounds/allow-lists to apply.
        leverage_lookup: Required whenever ``min_leverage`` or
            ``max_leverage`` is set - this module does not compute leverage
            itself. Raises :class:`ValueError` up front if a leverage bound
            is configured but no lookup was supplied, rather than silently
            filtering out every snapshot.
    """
    needs_leverage = (
        product_filter.min_leverage is not None or product_filter.max_leverage is not None
    )
    if needs_leverage and leverage_lookup is None:
        raise ValueError("product_filter sets a leverage bound but no leverage_lookup was provided")

    result: list[ProductSnapshot] = []
    for snapshot in snapshots:
        if product_filter.exclude_bid_only and snapshot.bid_only:
            continue
        if product_filter.exclude_knocked_out and snapshot.knocked_out:
            continue
        if (
            product_filter.allowed_issuers is not None
            and snapshot.issuer not in product_filter.allowed_issuers
        ):
            continue
        if needs_leverage:
            assert leverage_lookup is not None  # narrowed by the check above
            leverage = leverage_lookup(snapshot)
            if leverage is None:
                continue
            if product_filter.min_leverage is not None and leverage < product_filter.min_leverage:
                continue
            if product_filter.max_leverage is not None and leverage > product_filter.max_leverage:
                continue
        result.append(snapshot)
    return result
