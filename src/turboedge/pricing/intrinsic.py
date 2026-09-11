"""Intrinsic value, implied underlying, and leverage for turbo certificates.

Formula reference: Master Spec §13.1 ("Innerer Wert") and the Build Contract's
"Formeln (verbindlich)" section.

FX convention (binding across the whole pricing engine): ``fx`` is the number
of units of the *underlying's* currency per 1 unit of the *product's*
currency (EUR for every product in this milestone, per
``ProductSnapshot.currency``). A EUR-denominated underlying therefore always
uses ``fx=1.0``; a USD-denominated underlying (e.g. SPX, XAU) uses the
EURUSD spot rate.
"""

from __future__ import annotations

from turboedge.storage.schemas import Direction


def _require_positive(name: str, value: float) -> None:
    if not (value > 0):
        raise ValueError(f"{name} must be > 0, got {value!r}")


def intrinsic_value(
    spot: float,
    financing_level: float,
    ratio: float,
    direction: Direction,
    fx: float = 1.0,
) -> float:
    """Intrinsic (moneyness) value of one certificate, in product currency.

    Long:  ``max(S - F, 0) * ratio / fx``
    Short: ``max(F - S, 0) * ratio / fx``

    Raises:
        ValueError: if ``ratio <= 0`` or ``fx <= 0``.
    """
    _require_positive("ratio", ratio)
    _require_positive("fx", fx)
    moneyness = spot - financing_level if direction == Direction.LONG else financing_level - spot
    return max(moneyness, 0.0) * ratio / fx


def implied_underlying(
    price: float,
    financing_level: float,
    ratio: float,
    direction: Direction,
    fx: float = 1.0,
) -> float:
    """Underlying spot implied by a certificate's (mid) price.

    Long:  ``F + price * fx / ratio``
    Short: ``F - price * fx / ratio``

    This is the exact algebraic inverse of :func:`intrinsic_value` in the
    in-the-money region (where ``price`` equals the intrinsic value with no
    time-value component), and is used both to price products directly and
    to build the cross-issuer consensus spot (see
    ``pricing/cross_issuer.py``).

    Raises:
        ValueError: if ``price <= 0`` or ``ratio <= 0`` or ``fx <= 0``.
    """
    _require_positive("price", price)
    _require_positive("ratio", ratio)
    _require_positive("fx", fx)
    if direction == Direction.LONG:
        return financing_level + price * fx / ratio
    return financing_level - price * fx / ratio


def leverage(spot: float, price: float, ratio: float, fx: float = 1.0) -> float:
    """Effective leverage of one certificate: ``S * ratio / fx / price``.

    Raises:
        ValueError: if ``price <= 0`` or ``ratio <= 0`` or ``fx <= 0``.
    """
    _require_positive("price", price)
    _require_positive("ratio", ratio)
    _require_positive("fx", fx)
    return spot * ratio / fx / price
