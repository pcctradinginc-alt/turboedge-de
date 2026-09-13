"""Direction and product-type classification from raw source text/fields.

Every adapter hands back issuer-specific vocabulary ("Long"/"Call"/"Bull" vs
"Short"/"Put"/"Bear"; "Mini Future"/"Smart Turbo"/"Turbo Pro" vs plain
"Turbo"). This module centralizes the mapping to the canonical
:class:`~turboedge.storage.schemas.Direction` /
:class:`~turboedge.storage.schemas.ProductType` enums so adapters stay thin
and every source is classified consistently.
"""

from __future__ import annotations

import math
import re
from datetime import date

from turboedge.storage.schemas import Direction, ProductType

_LONG_RE = re.compile(r"\b(long|call|bull(?:ish)?)\b", re.IGNORECASE)
_SHORT_RE = re.compile(r"\b(short|put|bear(?:ish)?)\b", re.IGNORECASE)

_MINI_FUTURE_KEYWORDS: tuple[str, ...] = (
    "mini future",
    "mini-future",
    "minifuture",
    "smart turbo",
    "turbo pro",
    # BNP Paribas' actual live product-name convention (confirmed live,
    # 2026-09-11 research session, e.g. "Mini Long auf den DAX(R)"/"Mini
    # Short auf den DAX(R)") never contains the word "future" at all --
    # without these two, the name-based branch below never actually fires
    # for BNP (it happened to still classify correctly via the structural
    # barrier-vs-financing-level fallback, but the name signal this branch
    # exists for was silently unused).
    "mini long",
    "mini short",
)


def classify_direction(raw_text: str) -> Direction | None:
    """Infer Long/Short from free text such as a product name or type field.

    Returns ``None`` (never guesses) when both or neither direction keyword
    is present, e.g. "Bull/Bear Zertifikat" (both) or unrelated text
    (neither) - callers should treat that as DATA_QUALITY.
    """
    if not raw_text:
        return None
    has_long = bool(_LONG_RE.search(raw_text))
    has_short = bool(_SHORT_RE.search(raw_text))
    if has_long and not has_short:
        return Direction.LONG
    if has_short and not has_long:
        return Direction.SHORT
    return None


def classify_product_type(
    *,
    type_text: str | None = None,
    financing_level: float | None = None,
    knockout_barrier: float | None = None,
    open_end: bool | None = None,
    maturity: date | None = None,
    tolerance: float = 1e-9,
) -> ProductType:
    """Infer the product type from a type label plus its structural fields.

    Rules (Contract / Master Spec Section 13.1):

    - ``TURBO_OPEN_END``: barrier == financing level, open-ended.
    - ``TURBO_CLASSIC``: barrier == strike, fixed maturity.
    - ``MINI_FUTURE``: barrier != financing level (stop-loss buffer),
      open-ended. Also matched by name ("Mini Future", "Smart Turbo",
      "Turbo Pro") regardless of the numeric relationship, since issuers use
      those labels even when the buffer happens to be (temporarily) zero.
    - ``UNKNOWN``: none of the above can be established.
    """
    text = (type_text or "").lower()
    if any(keyword in text for keyword in _MINI_FUTURE_KEYWORDS):
        return ProductType.MINI_FUTURE

    barrier_eq_financing = (
        financing_level is not None
        and knockout_barrier is not None
        and math.isclose(financing_level, knockout_barrier, rel_tol=tolerance, abs_tol=tolerance)
    )
    has_barrier_and_financing = financing_level is not None and knockout_barrier is not None

    if open_end is True:
        if barrier_eq_financing:
            return ProductType.TURBO_OPEN_END
        if has_barrier_and_financing:
            return ProductType.MINI_FUTURE
        return ProductType.UNKNOWN

    if open_end is False:
        return ProductType.TURBO_CLASSIC if maturity is not None else ProductType.UNKNOWN

    # open_end unknown: fall back to the strongest available structural signal.
    if maturity is not None and not barrier_eq_financing:
        return ProductType.TURBO_CLASSIC
    if barrier_eq_financing:
        return ProductType.TURBO_OPEN_END
    if has_barrier_and_financing:
        return ProductType.MINI_FUTURE
    return ProductType.UNKNOWN
