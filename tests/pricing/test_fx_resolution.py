"""Tests for pricing/fx_resolution.py (Befund 1, 2026-09-13 measurement
session): per-(issuer, underlying) fx resolved from the fetch's own product
data instead of a single external daily-close approximation.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from turboedge.pricing.fx_resolution import fx_by_isin, resolve_fx_by_issuer
from turboedge.storage.schemas import Direction, ProductSnapshot


def _price_for_spot(spot: float, financing_level: float, ratio: float, fx: float) -> float:
    """Inverse of implied_underlying (Long): price = (spot - F) * ratio / fx."""
    return (spot - financing_level) * ratio / fx


def _make_group(
    make_product_snapshot: Callable[..., ProductSnapshot],
    *,
    issuer: str,
    spot: float,
    fx: float,
    n: int = 8,
) -> list[ProductSnapshot]:
    """``n`` internally-consistent NDX-style products for one issuer, priced
    so that every one of them implies the same ``spot`` under ``fx`` --
    alternating long/short and varying financing levels the way a real
    product shelf would."""
    products = []
    for i in range(n):
        direction = Direction.LONG if i % 2 == 0 else Direction.SHORT
        financing_level = (
            spot - 3000.0 - i * 50.0 if direction == Direction.LONG else spot + 3000.0 + i * 50.0
        )
        ratio = 0.01
        price = _price_for_spot(spot, financing_level, ratio, fx)
        if direction == Direction.SHORT:
            price = (financing_level - spot) * ratio / fx
        products.append(
            make_product_snapshot(
                isin=f"DE000{issuer[:4].upper()}{i:03d}",
                issuer=issuer,
                underlying_id="NDX",
                underlying_currency="USD",
                currency="EUR",
                quanto=None,
                financing_level=financing_level,
                ratio=ratio,
                direction=direction,
                bid=price - 0.01,
                ask=price + 0.01,
            )
        )
    return products


def test_resolves_quanto_group_as_fx_one(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    products = _make_group(make_product_snapshot, issuer="QuantoBank", spot=21000.0, fx=1.0)

    resolutions = resolve_fx_by_issuer(products)

    assert "QuantoBank" in resolutions
    resolution = resolutions["QuantoBank"]
    assert resolution.quanto is True
    assert resolution.fx == pytest.approx(1.0)
    assert resolution.verified_fraction >= 0.5


def test_resolves_non_quanto_group_from_data_alone(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    true_fx = 1.10
    products = _make_group(make_product_snapshot, issuer="NonQuantoBank", spot=21000.0, fx=true_fx)

    resolutions = resolve_fx_by_issuer(products)

    assert "NonQuantoBank" in resolutions
    resolution = resolutions["NonQuantoBank"]
    assert resolution.quanto is False
    assert resolution.fx == pytest.approx(true_fx, rel=0.02)
    assert resolution.verified_fraction >= 0.5


def test_insufficient_products_left_unresolved(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """Fewer than the minimum product count for an issuer must not be
    guessed at all (CLAUDE.md rule 29) -- the issuer is simply absent from
    the result."""
    products = _make_group(make_product_snapshot, issuer="ThinBank", spot=21000.0, fx=1.0, n=2)

    resolutions = resolve_fx_by_issuer(products)

    assert "ThinBank" not in resolutions


def test_inconsistent_group_left_unresolved(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """Random, mutually-inconsistent prices (neither hypothesis clusters)
    must resolve to nothing rather than picking whichever hypothesis happens
    to look marginally better."""
    products = []
    for i in range(8):
        direction = Direction.LONG if i % 2 == 0 else Direction.SHORT
        financing_level = 18000.0 - i * 40.0 if direction == Direction.LONG else 24000.0 + i * 40.0
        # Prices with no consistent implied spot under any fx in the search band.
        price = 50.0 + 37.0 * (i % 5)
        products.append(
            make_product_snapshot(
                isin=f"DE000NOISY{i:02d}",
                issuer="NoisyBank",
                underlying_id="NDX",
                underlying_currency="USD",
                currency="EUR",
                quanto=None,
                financing_level=financing_level,
                ratio=0.01,
                direction=direction,
                bid=price - 0.01,
                ask=price + 0.01,
            )
        )

    resolutions = resolve_fx_by_issuer(products)

    assert "NoisyBank" not in resolutions


def test_fx_by_isin_maps_unresolved_issuer_to_nothing(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    resolved = _make_group(make_product_snapshot, issuer="QuantoBank", spot=21000.0, fx=1.0)
    unresolved = _make_group(make_product_snapshot, issuer="ThinBank", spot=21000.0, fx=1.0, n=2)
    products = [*resolved, *unresolved]

    resolutions = resolve_fx_by_issuer(products)
    mapping = fx_by_isin(products, resolutions)

    assert all(isin.startswith("DE000QUAN") for isin in mapping)
    assert len(mapping) == len(resolved)
    assert all(fx == pytest.approx(1.0) for fx in mapping.values())
