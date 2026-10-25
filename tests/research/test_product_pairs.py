"""Counting comparable cross-issuer products in the archive.

`docs/research_plan_product_selection_cost_edge.md` §4 names this count as
the precondition for pre-registering that question: before asking whether
choosing between comparable products of different issuers buys a real cost
advantage, it has to be established how often the archive even contains
such a pair. `measured_results.md` §4 records one issuer dominating the
universe, so the expected answer is "rarely", and establishing that is the
point rather than a disappointment.

These tests pin the comparability definition, because a census is only
useful if "comparable" means the same thing every time it is run.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from turboedge.research.product_pairs import ComparabilityCriteria, census
from turboedge.storage.schemas import (
    Direction,
    FieldReliability,
    ProductSnapshot,
    ProductType,
)

_AT = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
_SPOT = 20_000.0


def _product(
    isin: str,
    *,
    issuer: str,
    barrier: float = 19_000.0,
    ask: float = 10.0,
    ratio: float = 0.01,
    direction: Direction = Direction.LONG,
    underlying: str | None = "DAX",
    at: datetime = _AT,
    spot: float | None = _SPOT,
    bid: float | None = 9.98,
    knocked_out: bool = False,
    bid_only: bool = False,
    ratio_reliability: FieldReliability = FieldReliability.SOURCE_REPORTED,
) -> ProductSnapshot:
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer=issuer,
        venue="stuttgart",
        underlying_raw=underlying or "unknown",
        underlying_id=underlying,
        direction=direction,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=barrier,
        knockout_barrier=barrier,
        ratio=ratio,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        bid=bid,
        ask=ask,
        bid_size=100.0,
        ask_size=100.0,
        quote_timestamp=at,
        quote_presence=True,
        bid_only=bid_only,
        knocked_out=knocked_out,
        underlying_price_ref=spot,
        underlying_price_ref_timestamp=at,
        ratio_reliability=ratio_reliability,
        raw_hash=hashlib.sha256(isin.encode()).hexdigest(),
        observation_time=at,
        available_at=at,
        retrieved_at=at,
        source="fake",
        parser_version="1",
        is_stale=False,
        quality_score=0.9,
    )


# -- the core count ------------------------------------------------------


def test_two_near_identical_products_from_two_issuers_are_a_pair() -> None:
    result = census(
        [_product("DE000PAIR001", issuer="BankA"), _product("DE000PAIR002", issuer="BankB")]
    )

    assert result.pair_count == 1
    assert result.pairs[0].issuer_pair == ("BankA", "BankB")
    assert result.pairs_by_underlying == {"DAX": 1}
    assert result.distinct_dates_by_underlying == {"DAX": 1}
    assert "comparable cross-issuer pair" in result.summary()


def test_two_products_from_the_same_issuer_are_not_a_pair() -> None:
    # The question is about choosing *between issuers*. Two products from one
    # issuer say nothing about issuer-level cost differences.
    result = census(
        [_product("DE000PAIR001", issuer="BankA"), _product("DE000PAIR002", issuer="BankA")]
    )

    assert result.pair_count == 0
    assert "no comparable cross-issuer pair" in result.summary()


def test_an_empty_archive_counts_zero_without_failing() -> None:
    result = census([])

    assert result.pair_count == 0
    assert result.snapshots_considered == 0


def test_three_issuers_at_one_moment_give_three_pairs() -> None:
    result = census(
        [
            _product("DE000PAIR001", issuer="BankA"),
            _product("DE000PAIR002", issuer="BankB"),
            _product("DE000PAIR003", issuer="BankC"),
        ]
    )

    assert result.pair_count == 3
    assert result.distinct_issuer_pairs == 3


def test_a_second_product_from_a_present_issuer_adds_no_pair() -> None:
    # One representative per issuer per bucket is enough; a second adds no
    # cross-issuer pair the first does not already provide, and counting it
    # would inflate the census with the dominant issuer's inventory.
    result = census(
        [
            _product("DE000PAIR001", issuer="BankA"),
            _product("DE000PAIR002", issuer="BankA"),
            _product("DE000PAIR003", issuer="BankB"),
        ]
    )

    assert result.pair_count == 1


# -- what makes two products NOT comparable ------------------------------


def test_different_underlyings_are_not_comparable() -> None:
    result = census(
        [
            _product("DE000PAIR001", issuer="BankA", underlying="DAX"),
            _product("DE000PAIR002", issuer="BankB", underlying="NDX"),
        ]
    )

    assert result.pair_count == 0


def test_opposite_directions_are_not_comparable() -> None:
    result = census(
        [
            _product("DE000PAIR001", issuer="BankA", direction=Direction.LONG),
            _product("DE000PAIR002", issuer="BankB", direction=Direction.SHORT, barrier=21_000.0),
        ]
    )

    assert result.pair_count == 0


def test_quotes_in_different_windows_are_not_comparable() -> None:
    # A cost comparison between two moments is a comparison of two markets.
    result = census(
        [
            _product("DE000PAIR001", issuer="BankA"),
            _product("DE000PAIR002", issuer="BankB", at=_AT + timedelta(minutes=5)),
        ],
        criteria=ComparabilityCriteria(window_seconds=60),
    )

    assert result.pair_count == 0


def test_a_far_apart_barrier_is_not_comparable() -> None:
    # 5 % vs 50 % distance to barrier is a different product, not a cheaper
    # one: the knock-out risk differs by an order of magnitude.
    near = _product("DE000PAIR001", issuer="BankA", barrier=19_000.0)  # 5 %
    far = _product("DE000PAIR002", issuer="BankB", barrier=10_000.0)  # 50 %

    assert census([near, far]).pair_count == 0


def test_a_far_apart_leverage_is_not_comparable() -> None:
    # leverage = spot * ratio / ask, so a 10x difference in ask is a 10x
    # difference in leverage at the same ratio.
    cheap = _product("DE000PAIR001", issuer="BankA", ask=10.0)
    pricey = _product("DE000PAIR002", issuer="BankB", ask=100.0)

    assert census([cheap, pricey]).pair_count == 0


def test_the_tolerances_are_relative_not_absolute() -> None:
    # The defect this pins: an earlier version bucketed leverage as
    # `value // (value * tolerance)`, which is `1 / tolerance` for every
    # input -- one bucket, so leverage was silently ignored and any two
    # products in a window counted as comparable.
    low = _product("DE000PAIR001", issuer="BankA", ask=100.0)  # leverage 2
    high = _product("DE000PAIR002", issuer="BankB", ask=6.67)  # leverage ~30

    assert census([low, high]).pair_count == 0, (
        "two products an order of magnitude apart in leverage were counted as "
        "comparable, which means the leverage dimension is being ignored"
    )


# -- what is excluded from a cost comparison -----------------------------


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"ask": None}, "no_ask"),
        ({"bid": None}, "no_bid"),
        ({"bid_only": True}, "no_ask"),
        ({"knocked_out": True}, "knocked_out"),
        ({"underlying": None}, "no_underlying_id"),
        ({"barrier": None}, "no_barrier"),
        ({"spot": None}, "no_underlying_reference"),
        ({"ratio_reliability": FieldReliability.UNVERIFIED}, "ratio_unverified"),
    ],
)
def test_unusable_products_are_dropped_with_a_reason(
    kwargs: dict[str, object], reason: str
) -> None:
    # Entry is at Ask and exit at Bid, so a round-trip cost needs both. An
    # unverified Bezugsverhaeltnis makes every per-unit figure meaningless,
    # because that is what the figure is divided by.
    result = census(
        [
            _product("DE000PAIR001", issuer="BankA", **kwargs),  # type: ignore[arg-type]
            _product("DE000PAIR002", issuer="BankB"),
        ]
    )

    assert result.pair_count == 0
    assert result.dropped.get(reason) == 1, result.dropped


def test_a_touched_barrier_is_dropped_even_without_the_flag() -> None:
    # Whether or not the issuer has flagged `knocked_out` yet, there is no
    # round trip left to cost.
    result = census(
        [
            _product("DE000PAIR001", issuer="BankA", barrier=21_000.0),  # above spot, Long
            _product("DE000PAIR002", issuer="BankB"),
        ]
    )

    assert result.pair_count == 0
    assert result.dropped.get("barrier_touched") == 1


def test_an_unverified_ratio_can_be_admitted_deliberately() -> None:
    # Only to measure how much the verification requirement costs in sample
    # size -- not for a cost comparison.
    products = [
        _product("DE000PAIR001", issuer="BankA", ratio_reliability=FieldReliability.UNVERIFIED),
        _product("DE000PAIR002", issuer="BankB", ratio_reliability=FieldReliability.UNVERIFIED),
    ]

    assert census(products).pair_count == 0
    assert (
        census(products, criteria=ComparabilityCriteria(require_verified_ratio=False)).pair_count
        == 1
    )


# -- effective sample ----------------------------------------------------


def test_distinct_dates_not_row_counts_are_the_sample_size() -> None:
    # Rows from one day are one observation repeated. Four pairs in one
    # afternoon are not four independent observations.
    snapshots = []
    for i in range(4):
        at = _AT + timedelta(hours=i)
        snapshots += [
            _product(f"DE000AAA{i:04d}", issuer="BankA", at=at),
            _product(f"DE000BBB{i:04d}", issuer="BankB", at=at),
        ]

    result = census(snapshots)

    assert result.pair_count == 4
    assert result.distinct_dates_by_underlying == {"DAX": 1}


def test_pairs_on_separate_days_are_separate_observations() -> None:
    snapshots = []
    for day in range(3):
        at = _AT + timedelta(days=day)
        snapshots += [
            _product(f"DE000AAA{day:04d}", issuer="BankA", at=at),
            _product(f"DE000BBB{day:04d}", issuer="BankB", at=at),
        ]

    result = census(snapshots)

    assert result.pair_count == 3
    assert result.distinct_dates_by_underlying == {"DAX": 3}


def test_the_census_computes_no_cost_and_opens_nothing() -> None:
    # It answers one question: how often does a comparable cross-issuer pair
    # exist. A cost figure here would be the experiment, not its
    # precondition.
    import inspect

    from turboedge.research import product_pairs

    source = inspect.getsource(product_pairs)

    for forbidden in ("net_ev", "lcb", "spread_cost", "p_value", "insert_research_trial"):
        assert forbidden not in source, f"{forbidden} appears in a census module"
