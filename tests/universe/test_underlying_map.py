from __future__ import annotations

import pytest

from turboedge.universe.underlying_map import (
    UNDERLYINGS,
    all_underlying_ids,
    get_underlying_meta,
    normalize_alias,
    resolve_underlying_id,
)

EXPECTED_IDS = {
    "DAX",
    "ESTX50",
    "SPX",
    "NDX",
    "NKY",
    "UKX",
    "SMI",
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "EURCHF",
    "XAU",
    "XAG",
    "BRENT",
    "WTI",
    "NATGAS",
}


def test_all_canonical_ids_present() -> None:
    assert set(all_underlying_ids()) == EXPECTED_IDS


@pytest.mark.parametrize(
    ("raw", "expected_id"),
    [
        ("DAX", "DAX"),
        ("DAX 40", "DAX"),
        ("DAX® (Performance)", "DAX"),
        ("DAX Performance Index", "DAX"),
        ("NASDAQ-100", "NDX"),
        ("Nasdaq 100", "NDX"),
        ("NDX", "NDX"),
        ("Gold", "XAU"),
        ("Goldpreis", "XAU"),
        ("XAU/USD", "XAU"),
        ("EUR/USD", "EURUSD"),
        ("Euro / US-Dollar", "EURUSD"),
        ("EURUSD", "EURUSD"),
        ("Euro Stoxx 50", "ESTX50"),
        ("S&P 500", "SPX"),
        ("Nikkei 225", "NKY"),
        ("FTSE 100", "UKX"),
        ("Swiss Market Index", "SMI"),
        ("USD/JPY", "USDJPY"),
        ("GBP/USD", "GBPUSD"),
        ("EUR/CHF", "EURCHF"),
        ("Silber", "XAG"),
        ("Brent", "BRENT"),
        ("WTI Crude Oil", "WTI"),
        ("Erdgas", "NATGAS"),
    ],
)
def test_resolve_underlying_id_known_aliases(raw: str, expected_id: str) -> None:
    assert resolve_underlying_id(raw) == expected_id


def test_resolve_underlying_id_is_case_and_whitespace_insensitive() -> None:
    assert resolve_underlying_id("  dax   40  ") == "DAX"
    assert resolve_underlying_id("eur/usd") == "EURUSD"


def test_resolve_underlying_id_unknown_returns_none() -> None:
    assert resolve_underlying_id("Bitcoin") is None
    assert resolve_underlying_id("") is None
    assert resolve_underlying_id("   ") is None


def test_resolve_underlying_id_containment_fallback() -> None:
    # Not a literal registered alias, but contains "DAX" as a whole word.
    assert resolve_underlying_id("DAX Call Optionsschein 18000") == "DAX"


def test_get_underlying_meta_known_id() -> None:
    meta = get_underlying_meta("DAX")
    assert meta.yfinance_ticker == "^GDAXI"
    assert meta.currency == "EUR"
    assert meta.asset_class == "index"


def test_get_underlying_meta_unknown_id_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_underlying_meta("NOPE")


def test_every_underlying_has_required_metadata() -> None:
    for underlying_id, meta in UNDERLYINGS.items():
        assert meta.id == underlying_id
        assert meta.yfinance_ticker
        assert meta.currency
        assert meta.asset_class
        assert meta.cluster
        assert len(meta.aliases) >= 1


def test_normalize_alias_strips_trademark_and_punctuation() -> None:
    assert normalize_alias("DAX® (Performance)") == "dax performance"
    assert normalize_alias("EUR/USD") == "eur usd"
    assert normalize_alias("  Multiple   Spaces  ") == "multiple spaces"
