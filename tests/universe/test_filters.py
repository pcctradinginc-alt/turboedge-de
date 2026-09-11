from __future__ import annotations

import pytest

from turboedge.universe.filters import ProductFilter, apply_filters


def test_exclude_bid_only_default(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    normal = make_product_snapshot(isin="DE000AAA1111", bid_only=False)
    bid_only = make_product_snapshot(isin="DE000BBB2222", bid_only=True)
    result = apply_filters([normal, bid_only], ProductFilter())
    assert [s.isin for s in result] == ["DE000AAA1111"]


def test_exclude_knocked_out_default(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    normal = make_product_snapshot(isin="DE000AAA1111", knocked_out=False)
    knocked_out = make_product_snapshot(isin="DE000BBB2222", knocked_out=True)
    result = apply_filters([normal, knocked_out], ProductFilter())
    assert [s.isin for s in result] == ["DE000AAA1111"]


def test_issuer_allow_list(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    a = make_product_snapshot(isin="DE000AAA1111", issuer="BankA")
    b = make_product_snapshot(isin="DE000BBB2222", issuer="BankB")
    result = apply_filters([a, b], ProductFilter(allowed_issuers=frozenset({"BankA"})))
    assert [s.isin for s in result] == ["DE000AAA1111"]


def test_leverage_bounds_require_lookup(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    snapshot = make_product_snapshot()
    with pytest.raises(ValueError, match="leverage_lookup"):
        apply_filters([snapshot], ProductFilter(min_leverage=2.0))


def test_leverage_bounds_applied_via_lookup(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    low = make_product_snapshot(isin="DE000AAA1111")
    mid = make_product_snapshot(isin="DE000BBB2222")
    high = make_product_snapshot(isin="DE000CCC3333")
    leverage_by_isin = {"DE000AAA1111": 1.0, "DE000BBB2222": 5.0, "DE000CCC3333": 30.0}

    result = apply_filters(
        [low, mid, high],
        ProductFilter(min_leverage=2.0, max_leverage=20.0),
        leverage_lookup=lambda s: leverage_by_isin[s.isin],
    )
    assert [s.isin for s in result] == ["DE000BBB2222"]


def test_leverage_lookup_returning_none_excludes_snapshot(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    snapshot = make_product_snapshot()
    result = apply_filters(
        [snapshot], ProductFilter(min_leverage=1.0), leverage_lookup=lambda s: None
    )
    assert result == []


def test_no_filters_returns_all(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    a = make_product_snapshot(isin="DE000AAA1111")
    b = make_product_snapshot(isin="DE000BBB2222")
    result = apply_filters(
        [a, b],
        ProductFilter(exclude_bid_only=False, exclude_knocked_out=False),
    )
    assert {s.isin for s in result} == {"DE000AAA1111", "DE000BBB2222"}
