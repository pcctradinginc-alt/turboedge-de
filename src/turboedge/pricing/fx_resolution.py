"""Data-driven FX resolution for non-EUR underlyings (Befund 1, 2026-09-13
measurement session).

Problem measured live: for a USD underlying (Nasdaq 100, S&P 500),
``pipeline/scan.py`` used to price *every* product with a single EURUSD rate
taken from the previous *daily close* (``PriceSource.fetch_daily_bars(
"EURUSD", ...)``), applied uniformly regardless of whether a given issuer's
products are actually quanto (fx=1, no FX exposure by design) or not. Two
compounding errors: (1) a daily close is stale relative to intraday FX moves,
and (2) quanto products priced with any fx != 1 get a systematically wrong
implied intrinsic value. Both look exactly like a data-quality problem (a
huge share of NDX products came back ``bid_below_intrinsic``/
``implied_spot_deviation``) even though the products themselves were fine.

Fix, mirroring the already-proven pattern in ``adapters/gettex.py``'s
``GettexAdapter._learn_fx_by_issuer`` (test both the quanto and non-quanto
hypothesis, let internal consistency -- "the majority of the same fetch's
own rows agreeing" -- decide, never accept an unverified guess): for every
(issuer, underlying) group of products with a known ``ratio`` already
(BNP/Citi already report it -- unlike gettex, which has to *derive* ratio
from a canonical grid, so this module reuses ``pricing.intrinsic.
implied_underlying`` directly rather than gettex's grid-inversion machinery,
which solves a different unknown), both hypotheses are evaluated:

1. **Quanto (fx=1)**: every product's mid price implies an underlying spot
   via :func:`~turboedge.pricing.intrinsic.implied_underlying`. If those
   implied spots cluster tightly (robust MAD dispersion below
   ``verify_tolerance_pct``), fx=1 is accepted.
2. **Non-quanto**: search for the fx value that makes the group's implied
   spots cluster *most* tightly (minimizes MAD dispersion). If that
   candidate also clusters within ``verify_tolerance_pct`` (and is not
   trivially indistinguishable from fx=1), it is accepted instead.

Whichever hypothesis passes is used; if neither clusters tightly enough
(too few products, or genuinely inconsistent data), the group's fx is left
**unresolved** -- callers must skip those products with their own
``fx_unresolved`` reason (CLAUDE.md rule 29: never impute pricing-critical
data), never fold them into a generic data-quality bucket like
``bid_below_intrinsic`` that looks like a product-level defect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import structlog
from scipy.optimize import minimize_scalar

from turboedge.pricing.intrinsic import implied_underlying
from turboedge.storage.schemas import Direction, ProductSnapshot

logger = structlog.get_logger(__name__)

_MAD_TO_STD = 1.4826  # same convention as pricing/cross_issuer.py, adapters/gettex.py

_DEFAULT_VERIFY_TOLERANCE_PCT = 0.01  # 1%: issuer-feed products carry a real bid/ask
# spread and small time-value/wrapper-margin noise (unlike gettex's derived-ratio
# problem, ratio here is issuer-reported ground truth) -- tighter than this false-
# rejects good data, looser stops discriminating a wrong fx from a right one.
_DEFAULT_MIN_VERIFIED_FRACTION = 0.5  # "Mehrheit" (majority), same as gettex's fix 2
_DEFAULT_MIN_PRODUCTS = 5
_DEFAULT_FX_SEARCH_MIN = 0.5
_DEFAULT_FX_SEARCH_MAX = 2.0
_DEFAULT_MAD_K = 5.0
# A searched non-quanto fx this close to 1.0 is not meaningfully distinguishable
# from the quanto hypothesis -- treat it as quanto rather than a suspiciously
# precise "non-quanto" fx that is actually just fx=1 plus search noise.
_QUANTO_EQUIVALENCE_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class FxResolution:
    """Resolved FX for one (issuer, underlying) product group."""

    issuer: str
    fx: float
    quanto: bool
    n_used: int
    n_total: int
    verified_fraction: float


@dataclass(frozen=True, slots=True)
class _PriceRow:
    isin: str
    price: float
    financing_level: float
    ratio: float
    direction: Direction


def _usable_rows(products: Sequence[ProductSnapshot]) -> list[_PriceRow]:
    rows: list[_PriceRow] = []
    for p in products:
        if p.bid is None or p.financing_level is None or p.ratio is None or not (p.ratio > 0):
            continue
        price = (p.bid + p.ask) / 2.0 if p.ask is not None else p.bid
        rows.append(
            _PriceRow(
                isin=p.isin,
                price=price,
                financing_level=p.financing_level,
                ratio=p.ratio,
                direction=p.direction,
            )
        )
    return rows


def _implied_values(rows: Sequence[_PriceRow], fx: float) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            values.append(
                implied_underlying(row.price, row.financing_level, row.ratio, row.direction, fx)
            )
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float64)


def _mad_dispersion(values: np.ndarray) -> tuple[float, float]:
    """Returns (median, relative MAD dispersion). Dispersion is +inf for < 2 values."""
    if values.size < 2:
        return (float(values[0]) if values.size == 1 else 0.0, float("inf"))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scaled = mad * _MAD_TO_STD
    relative = scaled / abs(median) if median != 0 else float("inf")
    return median, relative


def _count_within_tolerance(values: np.ndarray, median: float, tolerance_pct: float) -> int:
    if values.size == 0 or median == 0:
        return 0
    deviation = np.abs(values - median) / abs(median)
    return int(np.sum(deviation <= tolerance_pct))


def _verified_fraction(values: np.ndarray, median: float, tolerance_pct: float) -> float:
    if values.size == 0:
        return 0.0
    return _count_within_tolerance(values, median, tolerance_pct) / values.size


def resolve_fx_by_issuer(
    products: Sequence[ProductSnapshot],
    *,
    verify_tolerance_pct: float = _DEFAULT_VERIFY_TOLERANCE_PCT,
    min_verified_fraction: float = _DEFAULT_MIN_VERIFIED_FRACTION,
    min_products: int = _DEFAULT_MIN_PRODUCTS,
    fx_search_min: float = _DEFAULT_FX_SEARCH_MIN,
    fx_search_max: float = _DEFAULT_FX_SEARCH_MAX,
) -> dict[str, FxResolution]:
    """Resolve fx (and quanto/non-quanto) per issuer, from this fetch's own data.

    Returns a mapping ``{issuer: FxResolution}`` containing only the issuer
    groups that could actually be resolved (see module docstring) -- an
    issuer not present in the result is ``fx_unresolved`` for every one of
    its products this call; callers must not fall back to a guessed value.
    """
    groups: dict[str, list[ProductSnapshot]] = {}
    for p in products:
        groups.setdefault(p.issuer, []).append(p)

    resolutions: dict[str, FxResolution] = {}
    for issuer, group in groups.items():
        rows = _usable_rows(group)
        if len(rows) < min_products:
            logger.info("fx_resolution_insufficient_products", issuer=issuer, n_products=len(rows))
            continue

        # Hypothesis 1: quanto (fx=1).
        quanto_values = _implied_values(rows, 1.0)
        quanto_median, _quanto_dispersion = _mad_dispersion(quanto_values)
        quanto_fraction = _verified_fraction(quanto_values, quanto_median, verify_tolerance_pct)

        # Hypothesis 2: non-quanto -- search for the fx that makes the
        # group's own implied spots cluster most tightly (minimizes the
        # MAD dispersion), mirroring gettex's "propose from data, then
        # verify" two-step rather than assuming any external fx.
        def _objective(fx: float, rows: Sequence[_PriceRow] = rows) -> float:
            values = _implied_values(rows, fx)
            if values.size < 2:
                return float("inf")
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            return mad

        search = minimize_scalar(
            _objective, bounds=(fx_search_min, fx_search_max), method="bounded"
        )
        fx_candidate = float(search.x)
        candidate_values = _implied_values(rows, fx_candidate)
        candidate_median, _candidate_dispersion = _mad_dispersion(candidate_values)
        candidate_fraction = _verified_fraction(
            candidate_values, candidate_median, verify_tolerance_pct
        )
        candidate_is_quanto_equivalent = abs(fx_candidate - 1.0) <= _QUANTO_EQUIVALENCE_TOLERANCE

        if quanto_fraction >= min_verified_fraction and quanto_fraction >= candidate_fraction:
            resolutions[issuer] = FxResolution(
                issuer=issuer,
                fx=1.0,
                quanto=True,
                n_used=_count_within_tolerance(quanto_values, quanto_median, verify_tolerance_pct),
                n_total=len(rows),
                verified_fraction=quanto_fraction,
            )
            logger.info(
                "fx_resolution_quanto",
                issuer=issuer,
                n_total=len(rows),
                verified_fraction=quanto_fraction,
            )
        elif candidate_fraction >= min_verified_fraction and not candidate_is_quanto_equivalent:
            resolutions[issuer] = FxResolution(
                issuer=issuer,
                fx=fx_candidate,
                quanto=False,
                n_used=_count_within_tolerance(
                    candidate_values, candidate_median, verify_tolerance_pct
                ),
                n_total=len(rows),
                verified_fraction=candidate_fraction,
            )
            logger.info(
                "fx_resolution_non_quanto",
                issuer=issuer,
                fx=fx_candidate,
                n_total=len(rows),
                verified_fraction=candidate_fraction,
            )
        else:
            logger.info(
                "fx_resolution_unresolved",
                issuer=issuer,
                n_total=len(rows),
                quanto_fraction=quanto_fraction,
                candidate_fx=fx_candidate,
                candidate_fraction=candidate_fraction,
            )

    return resolutions


def fx_by_isin(
    products: Sequence[ProductSnapshot], resolutions: Mapping[str, FxResolution]
) -> dict[str, float]:
    """Expand per-issuer :class:`FxResolution` into a per-ISIN fx map.

    A product whose issuer has no entry in ``resolutions`` is intentionally
    left out of the result -- callers (``pipeline/scan.py``) treat a missing
    ISIN as ``fx_unresolved`` rather than silently defaulting to any fx.
    """
    result: dict[str, float] = {}
    for p in products:
        resolution = resolutions.get(p.issuer)
        if resolution is not None:
            result[p.isin] = resolution.fx
    return result


__all__ = ["FxResolution", "fx_by_isin", "resolve_fx_by_issuer"]
