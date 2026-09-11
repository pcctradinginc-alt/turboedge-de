from __future__ import annotations

from datetime import UTC, datetime

from turboedge.universe.discover import merge_snapshots


def test_merge_snapshots_dedupes_by_isin_preferring_freshest_quote(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    older = make_product_snapshot(
        isin="DE000ABC1234",
        source="deutsche_boerse",
        quote_timestamp=datetime(2026, 9, 10, 15, 0, tzinfo=UTC),
        quality_score=0.99,
    )
    newer = make_product_snapshot(
        isin="DE000ABC1234",
        source="boerse_stuttgart",
        quote_timestamp=datetime(2026, 9, 10, 15, 30, tzinfo=UTC),
        quality_score=0.5,
    )
    result = merge_snapshots([[older], [newer]])
    assert len(result.products) == 1
    assert result.products[0].source == "boerse_stuttgart"  # fresher wins despite lower quality


def test_merge_snapshots_ties_broken_by_quality_score(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    same_ts = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
    low_quality = make_product_snapshot(
        isin="DE000ABC1234", source="a", quote_timestamp=same_ts, quality_score=0.4
    )
    high_quality = make_product_snapshot(
        isin="DE000ABC1234", source="b", quote_timestamp=same_ts, quality_score=0.9
    )
    result = merge_snapshots([[low_quality], [high_quality]])
    assert result.products[0].source == "b"


def test_merge_snapshots_missing_quote_timestamp_loses_to_present_one(  # type: ignore[no-untyped-def]
    make_product_snapshot,
) -> None:
    no_ts = make_product_snapshot(isin="DE000ABC1234", source="a", quote_timestamp=None)
    with_ts = make_product_snapshot(
        isin="DE000ABC1234",
        source="b",
        quote_timestamp=datetime(2026, 9, 10, 15, 0, tzinfo=UTC),
    )
    result = merge_snapshots([[no_ts], [with_ts]])
    assert result.products[0].source == "b"


def test_merge_snapshots_no_conflict_for_matching_static_fields(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    a = make_product_snapshot(isin="DE000ABC1234", source="a", financing_level=18000.0, ratio=0.01)
    b = make_product_snapshot(isin="DE000ABC1234", source="b", financing_level=18000.0, ratio=0.01)
    result = merge_snapshots([[a], [b]])
    assert result.conflicts == []


def test_merge_snapshots_detects_financing_level_conflict(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    a = make_product_snapshot(isin="DE000ABC1234", source="a", financing_level=18000.0)
    b = make_product_snapshot(isin="DE000ABC1234", source="b", financing_level=19000.0)
    result = merge_snapshots([[a], [b]])
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.isin == "DE000ABC1234"
    assert conflict.field == "financing_level"
    assert set(conflict.sources) == {"a", "b"}


def test_merge_snapshots_detects_ratio_factor_conflict(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    a = make_product_snapshot(isin="DE000ABC1234", source="a", ratio=0.01)
    b = make_product_snapshot(isin="DE000ABC1234", source="b", ratio=0.1)  # off by 10x
    result = merge_snapshots([[a], [b]])
    fields = {c.field for c in result.conflicts}
    assert "ratio" in fields


def test_merge_snapshots_small_float_noise_is_not_a_conflict(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    a = make_product_snapshot(isin="DE000ABC1234", source="a", financing_level=18000.0)
    b = make_product_snapshot(isin="DE000ABC1234", source="b", financing_level=18000.0001)
    result = merge_snapshots([[a], [b]])
    assert result.conflicts == []


def test_merge_snapshots_handles_multiple_isins_independently(make_product_snapshot) -> None:  # type: ignore[no-untyped-def]
    a1 = make_product_snapshot(isin="DE000AAA1111", source="a")
    a2 = make_product_snapshot(isin="DE000BBB2222", source="a")
    result = merge_snapshots([[a1, a2]])
    assert {p.isin for p in result.products} == {"DE000AAA1111", "DE000BBB2222"}


def test_merge_snapshots_empty_input() -> None:
    result = merge_snapshots([])
    assert result.products == []
    assert result.conflicts == []
