from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from turboedge.pricing.cross_issuer import (
    ConsensusSpot,
    CrossIssuerInput,
    consensus_spot,
    cross_issuer_scores,
    robust_zscore,
)
from turboedge.storage.schemas import Direction, ProductSnapshot


def _mid_for_spot(spot: float, financing_level: float, ratio: float) -> float:
    # Inverse of implied_underlying (Long, fx=1): mid = (spot - F) * ratio
    return (spot - financing_level) * ratio


def test_consensus_spot_rejects_outlier_and_factor_error(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    financing_level, ratio = 22000.0, 0.01
    normal_spots = [23990.0, 24000.0, 24010.0, 23995.0, 24005.0]
    products = []
    for i, spot in enumerate(normal_spots):
        mid = _mid_for_spot(spot, financing_level, ratio)
        products.append(
            make_product_snapshot(
                isin=f"DE000NORM{i:03d}",
                financing_level=financing_level,
                ratio=ratio,
                direction=Direction.LONG,
                bid=mid - 0.01,
                ask=mid + 0.01,
            )
        )
    # A ~100x ratio/factor data error: implied spot is ~100x the true consensus.
    outlier_mid = _mid_for_spot(24000.0 * 100.0, financing_level, ratio)
    products.append(
        make_product_snapshot(
            isin="DE000OUTLI01",
            financing_level=financing_level,
            ratio=ratio,
            direction=Direction.LONG,
            bid=outlier_mid - 0.01,
            ask=outlier_mid + 0.01,
        )
    )

    result = consensus_spot(products)
    assert isinstance(result, ConsensusSpot)
    assert result.value == pytest.approx(24000.0, rel=0.005)
    assert result.n_used == 5
    assert result.n_rejected == 1


def test_consensus_spot_skips_products_missing_pricing_data(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    financing_level, ratio = 22000.0, 0.01
    mid = _mid_for_spot(24000.0, financing_level, ratio)
    valid = make_product_snapshot(
        isin="DE000VALID01",
        financing_level=financing_level,
        ratio=ratio,
        bid=mid - 0.01,
        ask=mid + 0.01,
    )
    missing_financing = make_product_snapshot(
        isin="DE000MISS001",
        financing_level=None,
        ratio=ratio,
        bid=mid - 0.01,
        ask=mid + 0.01,
    )
    bid_above_ask = make_product_snapshot(
        isin="DE000BADQ001",
        financing_level=financing_level,
        ratio=ratio,
        bid=100.0,
        ask=1.0,
    )
    result = consensus_spot([valid, missing_financing, bid_above_ask])
    assert result.n_used == 1
    assert result.value == pytest.approx(24000.0, rel=1e-6)


def test_consensus_spot_uses_bid_only_fallback_when_few_full_quotes(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """Build Contract Task 2: with fewer than 5 full (bid+ask) quotes, a
    bid-only product (ask missing) contributes to the consensus using bid
    in place of mid, and ``used_bid_only_fallback`` is set so callers can
    surface a "consensus_from_bid_only" warning.
    """
    financing_level, ratio = 22000.0, 0.01
    mid = _mid_for_spot(24000.0, financing_level, ratio)
    full = make_product_snapshot(
        isin="DE000FULL001",
        financing_level=financing_level,
        ratio=ratio,
        bid=mid - 0.01,
        ask=mid + 0.01,
    )
    bid_only = make_product_snapshot(
        isin="DE000BIDO001",
        financing_level=financing_level,
        ratio=ratio,
        bid=mid,
        ask=None,
    )
    result = consensus_spot([full, bid_only])
    assert result.used_bid_only_fallback is True
    assert result.n_used == 2
    assert result.value == pytest.approx(24000.0, rel=0.01)


def test_consensus_spot_excludes_bid_only_when_enough_full_quotes(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """With >= 5 full quotes, a bid-only product is excluded entirely rather
    than (even slightly) skewing the consensus.
    """
    financing_level, ratio = 22000.0, 0.01
    normal_spots = [23990.0, 24000.0, 24010.0, 23995.0, 24005.0]
    products = []
    for i, spot in enumerate(normal_spots):
        mid = _mid_for_spot(spot, financing_level, ratio)
        products.append(
            make_product_snapshot(
                isin=f"DE000FULL{i:03d}",
                financing_level=financing_level,
                ratio=ratio,
                bid=mid - 0.01,
                ask=mid + 0.01,
            )
        )
    # A wildly different bid-only quote -- must not pollute the consensus
    # once 5 full quotes are already available.
    bid_only = make_product_snapshot(
        isin="DE000BIDO002",
        financing_level=financing_level,
        ratio=ratio,
        bid=_mid_for_spot(30000.0, financing_level, ratio),
        ask=None,
    )
    result = consensus_spot([*products, bid_only])
    assert result.used_bid_only_fallback is False
    assert result.n_used == 5
    assert result.value == pytest.approx(24000.0, rel=0.005)


def test_consensus_spot_one_sided_sample_is_not_direction_balanced(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """A sample with only one direction (e.g. only LONG quotes) cannot
    bias-cancel -- `direction_balanced` is False and the plain median is
    used, same as before the Build Contract BEFUND 1 fix.
    """
    financing_level, ratio = 22000.0, 0.01
    normal_spots = [23990.0, 24000.0, 24010.0, 23995.0, 24005.0]
    products = []
    for i, spot in enumerate(normal_spots):
        mid = _mid_for_spot(spot, financing_level, ratio)
        products.append(
            make_product_snapshot(
                isin=f"DE000ONESD{i:02d}",
                financing_level=financing_level,
                ratio=ratio,
                direction=Direction.LONG,
                bid=mid - 0.01,
                ask=mid + 0.01,
            )
        )
    result = consensus_spot(products)
    assert result.direction_balanced is False
    assert result.value == pytest.approx(24000.0, rel=0.005)


def test_consensus_spot_balances_long_short_wrapper_margin_bias(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """Build Contract BEFUND 1: a positive issuer margin baked into every
    mid price inflates LONG-implied spot and deflates SHORT-implied spot by
    the same amount (``mid = intrinsic + margin``, and
    ``implied_underlying`` adds/subtracts ``mid/ratio`` for Long/Short
    respectively). With an intentionally imbalanced sample (6 LONG quotes,
    median implied +9 vs. the true spot, against 3 SHORT quotes, median
    implied -8), a plain pooled median is skewed toward the more numerous,
    inflated LONG side (would land at +6); averaging the per-direction
    medians instead recovers a value much closer to the true spot.
    """
    # Separate financing levels per direction (structurally required: LONG
    # needs F below spot, SHORT needs F above spot -- both ~24000 here).
    financing_level_long, financing_level_short, ratio = 22000.0, 28000.0, 0.01
    true_spot = 24000.0
    # Implied-spot values each quote's mid is engineered to produce (jittered
    # so the sample's MAD is not degenerately zero -- see
    # test_consensus_spot_one_sided_sample_is_not_direction_balanced's
    # sibling investigation of that edge case).
    long_implied = [24004.0, 24006.0, 24008.0, 24010.0, 24012.0, 24014.0]
    short_implied = [23988.0, 23992.0, 23996.0]

    products = [
        make_product_snapshot(
            isin=f"DE000LONGB{i:02d}",
            financing_level=financing_level_long,
            ratio=ratio,
            direction=Direction.LONG,
            bid=_mid_for_spot(v, financing_level_long, ratio) - 0.001,
            ask=_mid_for_spot(v, financing_level_long, ratio) + 0.001,
        )
        for i, v in enumerate(long_implied)
    ] + [
        make_product_snapshot(
            isin=f"DE000SHRTB{i:02d}",
            financing_level=financing_level_short,
            ratio=ratio,
            direction=Direction.SHORT,
            bid=(financing_level_short - v) * ratio - 0.001,
            ask=(financing_level_short - v) * ratio + 0.001,
        )
        for i, v in enumerate(short_implied)
    ]

    result = consensus_spot(products)
    assert result.n_used == len(long_implied) + len(short_implied)  # no MAD rejection
    assert result.direction_balanced is True
    # Bias-cancelling average of the two direction medians (24009, 23992)
    # recovers a value close to the true spot; the plain pooled median of
    # all 9 accepted values (what the pre-fix code computed) is 24006 --
    # 12x further from the true spot than the balanced estimate.
    assert result.value == pytest.approx(24000.5, abs=0.01)
    assert abs(result.value - true_spot) < abs(24006.0 - true_spot)


def test_consensus_spot_raises_when_nothing_valid(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    missing_financing = make_product_snapshot(isin="DE000MISS002", financing_level=None)
    with pytest.raises(ValueError):
        consensus_spot([missing_financing])


def test_robust_zscore_known_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    z = robust_zscore(values)
    median = 3.0
    mad = float(np.median(np.abs(np.array(values) - median)))
    expected = (np.array(values) - median) / (1.4826 * mad)
    np.testing.assert_allclose(z, expected)


def test_robust_zscore_zero_mad_returns_zeros() -> None:
    z = robust_zscore([5.0, 5.0, 5.0])
    np.testing.assert_array_equal(z, np.zeros(3))


def _row(
    isin: str,
    issuer: str,
    margin: float,
    spread: float,
    underlying: str = "DAX",
    direction: Direction = Direction.LONG,
    bucket: str = "4-5",
) -> CrossIssuerInput:
    return CrossIssuerInput(
        isin=isin,
        issuer=issuer,
        underlying_id=underlying,
        direction=direction,
        leverage_bucket=bucket,
        normalized_spread=spread,
        issuer_margin_pct=margin,
        financing_spread=0.02,
        quote_age_s=5.0,
    )


def test_cross_issuer_scores_small_group_is_none() -> None:
    rows = [
        _row("DE000A0001", "BankA", 0.02, 0.01),
        _row("DE000A0002", "BankB", 0.03, 0.012),
    ]
    scores = cross_issuer_scores(rows)
    for s in scores.values():
        assert s.cross_issuer_residual_zscore is None
        assert s.issuer_markup_score is None
        assert s.quote_dislocation_score is None
        assert s.wrapper_edge is None


def test_cross_issuer_scores_group_of_three_or_more() -> None:
    rows = [
        _row("DE000A0001", "BankA", 0.01, 0.005),
        _row("DE000A0002", "BankB", 0.015, 0.006),
        _row("DE000A0003", "BankC", 0.08, 0.02),  # clear high-margin outlier
    ]
    scores = cross_issuer_scores(rows)
    for s in scores.values():
        assert s.cross_issuer_residual_zscore is not None
        assert s.quote_dislocation_score is not None
        assert s.wrapper_edge is not None

    # BankC has the highest margin -> highest residual z-score and largest
    # dislocation, and (being the most expensive) the most negative wrapper_edge.
    assert scores["DE000A0003"].cross_issuer_residual_zscore is not None
    assert scores["DE000A0001"].cross_issuer_residual_zscore is not None
    assert (
        scores["DE000A0003"].cross_issuer_residual_zscore
        > scores["DE000A0001"].cross_issuer_residual_zscore
    )
    assert scores["DE000A0003"].wrapper_edge is not None
    assert scores["DE000A0003"].wrapper_edge < 0.0


def test_cross_issuer_scores_issuer_markup_is_issuer_level_median() -> None:
    # BankA appears in two separate qualifying groups (different underlyings)
    # and is consistently the cheapest -> issuer_markup_score should reflect
    # the median of its own residual z-scores across both groups.
    rows = [
        _row("DE000G1A", "BankA", 0.01, 0.005, underlying="DAX"),
        _row("DE000G1B", "BankB", 0.03, 0.01, underlying="DAX"),
        _row("DE000G1C", "BankC", 0.05, 0.015, underlying="DAX"),
        _row("DE000G2A", "BankA", 0.01, 0.005, underlying="NDX"),
        _row("DE000G2B", "BankB", 0.03, 0.01, underlying="NDX"),
        _row("DE000G2C", "BankC", 0.05, 0.015, underlying="NDX"),
    ]
    scores = cross_issuer_scores(rows)
    bank_a_scores = [scores["DE000G1A"], scores["DE000G2A"]]
    for s in bank_a_scores:
        assert s.issuer_markup_score is not None
        assert s.issuer_markup_score < 0.0  # consistently below-median margin
