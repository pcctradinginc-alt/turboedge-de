"""Liquidity factor: geometric mean of four liquidity sub-scores.

Formula reference: Master Spec §18 ("Utility-Funktion")::

    liq = geometric_mean(
        quote_presence,
        quote_size_coverage,
        quote_freshness,
        spread_quality,
    )
"""

from __future__ import annotations

_EPS = 1e-6


def _clip_unit(value: float) -> float:
    return min(1.0, max(_EPS, value))


def liquidity_factor(
    quote_presence: float,
    quote_size_coverage: float,
    quote_freshness: float,
    spread_quality: float,
) -> float:
    """Geometric mean of the four liquidity sub-scores, each clipped to ``[eps, 1]``.

    The ``eps`` floor (``1e-6``) keeps the result a well-defined small
    positive number rather than exactly zero when one input is exactly
    zero, so downstream multiplicative use (``score_jh = LCB(U) * liq *
    ...``, Master Spec §18) degrades gracefully instead of hard-zeroing the
    whole score.
    """
    values = (
        _clip_unit(quote_presence),
        _clip_unit(quote_size_coverage),
        _clip_unit(quote_freshness),
        _clip_unit(spread_quality),
    )
    product = 1.0
    for value in values:
        product *= value
    return float(product ** (1.0 / len(values)))


def quote_size_coverage(
    ask_size: float | None,
    required_notional: float,
    ask: float | None,
) -> float:
    """Fraction of the required trade notional the displayed ask size can fill.

    ``coverage = min(1, ask_size * ask / required_notional)``, clipped to
    ``[0, 1]``. Missing or non-positive ``ask_size``/``ask`` yields ``0.0``
    (no imputation of an unknown quote size, CLAUDE.md rule 29).

    Raises:
        ValueError: if ``required_notional <= 0``.
    """
    if not (required_notional > 0):
        raise ValueError(f"required_notional must be > 0, got {required_notional!r}")
    if ask_size is None or ask is None or ask_size <= 0 or ask <= 0:
        return 0.0
    coverage = (ask_size * ask) / required_notional
    return float(min(1.0, max(0.0, coverage)))


def spread_quality(spread_pct: float, max_spread_pct: float) -> float:
    """Linear spread-quality score: 1 at zero spread, 0 at/beyond ``max_spread_pct``.

    ``quality = clip(1 - spread_pct / max_spread_pct, 0, 1)``.

    Raises:
        ValueError: if ``max_spread_pct <= 0``.
    """
    if not (max_spread_pct > 0):
        raise ValueError(f"max_spread_pct must be > 0, got {max_spread_pct!r}")
    quality = 1.0 - (spread_pct / max_spread_pct)
    return float(min(1.0, max(0.0, quality)))
