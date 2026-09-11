"""Canonical underlying_id resolution from raw source text.

Every data source names underlyings differently ("DAX", "DAX 40",
"DAX® (Performance)", "Euro / US-Dollar", ...). This module owns the single
canonical vocabulary (Master Spec Section 4 / Contract) that the rest of the
system uses: ``DAX, ESTX50, SPX, NDX, NKY, UKX, SMI, EURUSD, USDJPY, GBPUSD,
EURCHF, XAU, XAG, BRENT, WTI, NATGAS``, plus the metadata (yfinance ticker,
currency, asset class, correlation cluster) needed to fetch prices and to
apply cluster-risk limits later.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_STRIP_TRADEMARK_RE = re.compile(r"[®™©]")
_PUNCT_TO_SPACE_RE = re.compile(r"[/\-_.,;:()]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_alias(raw: str) -> str:
    """Normalize a raw underlying label for alias matching.

    Applies Unicode NFKC normalization, strips trademark glyphs, turns
    punctuation (``/ - _ . , ; : ( )``) into spaces, lowercases, and collapses
    whitespace. E.g. ``"DAX® (Performance)"`` -> ``"dax performance"``,
    ``"EUR/USD"`` -> ``"eur usd"``.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = _STRIP_TRADEMARK_RE.sub("", text)
    text = _PUNCT_TO_SPACE_RE.sub(" ", text)
    text = text.strip().lower()
    return _WHITESPACE_RE.sub(" ", text)


@dataclass(frozen=True)
class UnderlyingMeta:
    id: str
    name: str
    yfinance_ticker: str
    currency: str
    asset_class: str
    cluster: str
    aliases: tuple[str, ...]


_UNDERLYINGS: tuple[UnderlyingMeta, ...] = (
    UnderlyingMeta(
        id="DAX",
        name="DAX",
        yfinance_ticker="^GDAXI",
        currency="EUR",
        asset_class="index",
        cluster="de_equity",
        aliases=(
            "DAX",
            "DAX40",
            "DAX 40",
            "DAX® (Performance)",
            "DAX Performance Index",
            "GER40",
            "GER 40",
            "Deutscher Aktienindex",
        ),
    ),
    UnderlyingMeta(
        id="ESTX50",
        name="Euro Stoxx 50",
        yfinance_ticker="^STOXX50E",
        currency="EUR",
        asset_class="index",
        cluster="eu_equity",
        aliases=(
            "ESTX50",
            "Euro Stoxx 50",
            "EURO STOXX 50",
            "Eurostoxx 50",
            "Eurostoxx50",
            "SX5E",
        ),
    ),
    UnderlyingMeta(
        id="SPX",
        name="S&P 500",
        yfinance_ticker="^GSPC",
        currency="USD",
        asset_class="index",
        cluster="us_equity",
        aliases=(
            "SPX",
            "S&P 500",
            "S&P500",
            "SP500",
            "US500",
            "Standard & Poor's 500",
        ),
    ),
    UnderlyingMeta(
        id="NDX",
        name="Nasdaq 100",
        yfinance_ticker="^NDX",
        currency="USD",
        asset_class="index",
        cluster="us_equity",
        aliases=(
            "NDX",
            "Nasdaq 100",
            "Nasdaq100",
            "NASDAQ-100",
            "NASDAQ 100",
            "US100",
            "US Tech 100",
        ),
    ),
    UnderlyingMeta(
        id="NKY",
        name="Nikkei 225",
        yfinance_ticker="^N225",
        currency="JPY",
        asset_class="index",
        cluster="jp_equity",
        aliases=(
            "NKY",
            "Nikkei 225",
            "Nikkei225",
            "Nikkei",
            "JP225",
        ),
    ),
    UnderlyingMeta(
        id="UKX",
        name="FTSE 100",
        yfinance_ticker="^FTSE",
        currency="GBP",
        asset_class="index",
        cluster="uk_equity",
        aliases=(
            "UKX",
            "FTSE 100",
            "FTSE100",
            "UK100",
        ),
    ),
    UnderlyingMeta(
        id="SMI",
        name="SMI",
        yfinance_ticker="^SSMI",
        currency="CHF",
        asset_class="index",
        cluster="ch_equity",
        aliases=(
            "SMI",
            "Swiss Market Index",
            "SMI20",
        ),
    ),
    UnderlyingMeta(
        id="EURUSD",
        name="EUR/USD",
        yfinance_ticker="EURUSD=X",
        currency="USD",
        asset_class="fx",
        cluster="fx_eur",
        aliases=(
            "EURUSD",
            "EUR/USD",
            "EUR USD",
            "Euro / US-Dollar",
            "Euro/US-Dollar",
            "Euro US Dollar",
            "Euro-Dollar",
        ),
    ),
    UnderlyingMeta(
        id="USDJPY",
        name="USD/JPY",
        yfinance_ticker="USDJPY=X",
        currency="JPY",
        asset_class="fx",
        cluster="fx_jpy",
        aliases=(
            "USDJPY",
            "USD/JPY",
            "USD JPY",
            "US-Dollar / Japanischer Yen",
            "Dollar-Yen",
        ),
    ),
    UnderlyingMeta(
        id="GBPUSD",
        name="GBP/USD",
        yfinance_ticker="GBPUSD=X",
        currency="USD",
        asset_class="fx",
        cluster="fx_gbp",
        aliases=(
            "GBPUSD",
            "GBP/USD",
            "GBP USD",
            "Britisches Pfund / US-Dollar",
            "Pfund-Dollar",
        ),
    ),
    UnderlyingMeta(
        id="EURCHF",
        name="EUR/CHF",
        yfinance_ticker="EURCHF=X",
        currency="CHF",
        asset_class="fx",
        cluster="fx_chf",
        aliases=(
            "EURCHF",
            "EUR/CHF",
            "EUR CHF",
            "Euro / Schweizer Franken",
        ),
    ),
    UnderlyingMeta(
        id="XAU",
        name="Gold",
        yfinance_ticker="GC=F",
        currency="USD",
        asset_class="metal",
        cluster="metals",
        aliases=(
            "XAU",
            "XAUUSD",
            "XAU/USD",
            "Gold",
            "Goldpreis",
            "Gold Spot",
            "Feinunze Gold",
        ),
    ),
    UnderlyingMeta(
        id="XAG",
        name="Silver",
        yfinance_ticker="SI=F",
        currency="USD",
        asset_class="metal",
        cluster="metals",
        aliases=(
            "XAG",
            "XAGUSD",
            "XAG/USD",
            "Silber",
            "Silver",
            "Silberpreis",
            "Feinunze Silber",
        ),
    ),
    UnderlyingMeta(
        id="BRENT",
        name="Brent Crude",
        yfinance_ticker="BZ=F",
        currency="USD",
        asset_class="energy",
        cluster="energy",
        aliases=(
            "BRENT",
            "Brent",
            "Brent Crude",
            "Brent Crude Oil",
            "Brent Oel",
            "Brentoel",
            "Brent Oil",
        ),
    ),
    UnderlyingMeta(
        id="WTI",
        name="WTI Crude",
        yfinance_ticker="CL=F",
        currency="USD",
        asset_class="energy",
        cluster="energy",
        aliases=(
            "WTI",
            "WTI Crude",
            "WTI Crude Oil",
            "WTI Oel",
            "West Texas Intermediate",
        ),
    ),
    UnderlyingMeta(
        id="NATGAS",
        name="Natural Gas",
        yfinance_ticker="NG=F",
        currency="USD",
        asset_class="energy",
        cluster="energy",
        aliases=(
            "NATGAS",
            "Natural Gas",
            "Nat Gas",
            "Erdgas",
            "Naturgas",
        ),
    ),
)

UNDERLYINGS: dict[str, UnderlyingMeta] = {meta.id: meta for meta in _UNDERLYINGS}

# normalized alias string -> canonical id, built once at import time. Includes
# the canonical id and name themselves so e.g. "DAX" and "Gold" always match
# even if not separately listed among `aliases`.
_ALIAS_LOOKUP: dict[str, str] = {}
for _meta in _UNDERLYINGS:
    for _label in (_meta.id, _meta.name, *_meta.aliases):
        _ALIAS_LOOKUP[normalize_alias(_label)] = _meta.id

# Precomputed (normalized_alias, id, word-boundary pattern) tuples, longest
# alias first, used for the fuzzy/containment fallback in resolve_underlying_id.
_ALIAS_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    sorted(
        (
            (normalized, underlying_id, re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)"))
            for normalized, underlying_id in _ALIAS_LOOKUP.items()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def all_underlying_ids() -> list[str]:
    """Every canonical underlying_id, in declaration order."""
    return [meta.id for meta in _UNDERLYINGS]


def get_underlying_meta(underlying_id: str) -> UnderlyingMeta:
    """Look up metadata for a canonical id. Raises KeyError if unknown."""
    try:
        return UNDERLYINGS[underlying_id]
    except KeyError as exc:
        raise KeyError(f"unknown canonical underlying_id: {underlying_id!r}") from exc


def resolve_underlying_id(raw: str) -> str | None:
    """Resolve a raw, free-text underlying label to a canonical underlying_id.

    Two-tier matching:

    1. Exact match of the fully normalized string against the alias table
       (covers the vast majority of real feeds, which use a small, stable
       set of labels).
    2. A conservative containment fallback: the normalized input is scanned
       for any known alias as a whole "word" (word-boundary regex, longest
       alias wins so e.g. "Euro Stoxx 50" is preferred over a bare "50").
       This only fires when tier 1 finds nothing and returns ``None`` rather
       than a guess if the normalized input contains no known alias.

    Returns ``None`` (never raises) when no underlying can be confidently
    identified - callers should treat that as ``underlying_id = None`` /
    DATA_QUALITY, never guess.
    """
    if not raw or not raw.strip():
        return None
    normalized = normalize_alias(raw)

    direct = _ALIAS_LOOKUP.get(normalized)
    if direct is not None:
        return direct

    for _alias_normalized, underlying_id, pattern in _ALIAS_PATTERNS:
        if pattern.search(normalized):
            return underlying_id
    return None
