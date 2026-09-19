"""Tests for Phase F's shadow-portfolio construction (`_build_shadow_positions`
in `pipeline/scan.py`, Master Spec §46, CLAUDE.md guiding question).

These are deliberately white-box, unit-level tests of `_build_shadow_positions`
itself (constructing `_PricedProduct`/`ProductHorizonEvaluation` directly)
rather than driving the full stochastic forecast/path-simulation pipeline --
that gives exact, deterministic control over each arm's ranking/extremum
inputs, which a full `run_scan()` (random-walk bars, MC path simulation)
cannot offer without excessive seeding gymnastics. `test_scan_shadow_e2e.py`
covers the end-to-end wiring (`run_scan` -> `store.append_shadow_positions`)
separately.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import numpy as np

from turboedge.pipeline.scan import (
    _CALENDAR_DAYS_PER_TRADING_DAY,
    _SHADOW_TOP_N,
    _build_shadow_positions,
    _PricedProduct,
)
from turboedge.pricing.integrity import IntegrityReport
from turboedge.ranking.ev import ProductHorizonEvaluation
from turboedge.storage.schemas import (
    CostDecomposition,
    Direction,
    ProductSnapshot,
    ShadowPortfolioKind,
)

_EVAL_TIME = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)
_RUN_ID = "run-shadow-1"


def _expected_exit_due(horizon_days: int, *, evaluation_time: datetime = _EVAL_TIME) -> date:
    return (
        evaluation_time + timedelta(days=round(horizon_days * _CALENDAR_DAYS_PER_TRADING_DAY))
    ).date()


def _priced(
    dax_product_factory: Callable[..., ProductSnapshot],
    *,
    isin: str,
    direction: Direction,
    ask: float = 60.0,
    leverage_value: float = 4.0,
    spread_pct: float = 0.01,
    financing_drag_pct: float = 0.02,
) -> _PricedProduct:
    product = dax_product_factory(
        isin=isin,
        issuer="BankA",
        direction=direction,
        financing_level=18000.0 if direction == Direction.LONG else 30000.0,
        quote_timestamp=_EVAL_TIME,
    ).model_copy(update={"ask": ask, "bid": ask * (1 - spread_pct)})
    costs = CostDecomposition(
        ask=ask,
        bid=ask * (1 - spread_pct),
        mid=ask * (1 - spread_pct / 2),
        intrinsic=ask * 0.9,
        trading_spread_component=ask * spread_pct / 2,
        fair_gap_premium=0.0,
        financing_drag=ask * financing_drag_pct,
        issuer_margin=0.0,
        spread_pct=spread_pct,
        gap_premium_pct=0.0,
        financing_drag_pct=financing_drag_pct,
        issuer_margin_pct=0.0,
    )
    return _PricedProduct(
        product=product,
        bid=ask * (1 - spread_pct),
        ask=ask,
        spot=24000.0,
        fx=1.0,
        integrity=IntegrityReport(passed=True),
        leverage_value=leverage_value,
        leverage_bucket_value="mid",
        spread_pct_value=spread_pct,
        quote_age_s=0.0,
        source_quote_age_s=0.0,
        distance_pct=None,
        distance_sigma=None,
        realized_spread=0.01,
        used_default_spread=False,
        financing_spread_source="issuer_funding_rate",
        costs=costs,
        financing_cost_pct={},
        gap_premium_over_horizon_pct=0.0,
        liquidity=1.0,
    )


def _eval(
    *, isin: str, direction: Direction, horizon_days: int, score: float
) -> ProductHorizonEvaluation:
    return ProductHorizonEvaluation(
        isin=isin,
        underlying_id="DAX",
        direction=direction,
        horizon_days=horizon_days,
        mean_net_return=0.01,
        median_net_return=0.01,
        q05=-0.05,
        q95=0.05,
        p_profit=0.5,
        p_ko=0.05,
        es95=-0.1,
        mfe_median=0.02,
        mae_median=-0.02,
        mc_standard_error=0.001,
        shrunk_mean=0.01,
        shrinkage_intensity=0.5,
        lcb_net_return=0.0,
        utility=0.0,
        liquidity_factor=1.0,
        score=score,
        suggested_position_fraction=0.01,
        cluster_id="c1",
        reasons=[],
    )


def _build_pool(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> tuple[dict[str, ProductHorizonEvaluation], dict[str, _PricedProduct]]:
    """7 LONG, horizon=5 candidates (the eligible pool) with distinct
    score/leverage/spread/financing so every arm's extremum/ranking pick is
    unambiguous, plus a LONG horizon=7 decoy (higher score than most of the
    pool, but wrong horizon) and a SHORT horizon=5 decoy (very high score,
    but wrong direction) -- both must never appear in any arm's output.
    """
    best_by_isin: dict[str, ProductHorizonEvaluation] = {}
    priced: dict[str, _PricedProduct] = {}

    # score, leverage, spread_pct, financing_drag_pct -- all distinct so each
    # arm's extremum pick is unambiguous.
    specs = [
        ("DE000POL0001", 100.0, 5.0, 0.020, 0.030),  # top1 -> highest score
        ("DE000POL0002", 90.0, 3.0, 0.015, 0.025),
        ("DE000POL0003", 80.0, 8.0, 0.005, 0.010),  # lowest spread + lowest financing
        ("DE000POL0004", 70.0, 1.0, 0.030, 0.040),  # lowest leverage
        ("DE000POL0005", 60.0, 9.0, 0.025, 0.035),  # highest leverage
        ("DE000POL0006", 50.0, 6.0, 0.018, 0.028),
        ("DE000POL0007", 40.0, 7.0, 0.022, 0.032),
    ]
    for isin, score, lev, spread_pct, fin_pct in specs:
        best_by_isin[isin] = _eval(isin=isin, direction=Direction.LONG, horizon_days=5, score=score)
        priced[isin] = _priced(
            dax_product_factory,
            isin=isin,
            direction=Direction.LONG,
            ask=60.0 + score,  # distinct entry_ask per isin too
            leverage_value=lev,
            spread_pct=spread_pct,
            financing_drag_pct=fin_pct,
        )

    # Decoys: must never appear in any arm's output.
    best_by_isin["DE000DECOYH7"] = _eval(
        isin="DE000DECOYH7", direction=Direction.LONG, horizon_days=7, score=95.0
    )
    priced["DE000DECOYH7"] = _priced(
        dax_product_factory, isin="DE000DECOYH7", direction=Direction.LONG
    )
    best_by_isin["DE000DECOYSH"] = _eval(
        isin="DE000DECOYSH", direction=Direction.SHORT, horizon_days=5, score=99.0
    )
    priced["DE000DECOYSH"] = _priced(
        dax_product_factory, isin="DE000DECOYSH", direction=Direction.SHORT
    )

    return best_by_isin, priced


_POOL_ISINS = {f"DE000POL000{i}" for i in range(1, 8)}


def test_arms_match_direction_time_horizon_of_top1_and_exclude_decoys(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    best_by_isin, priced = _build_pool(dax_product_factory)
    warnings: list[str] = []

    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin=best_by_isin,
        priced=priced,
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=warnings,
    )

    assert positions, "expected at least one shadow row"
    expected_exit_due = _expected_exit_due(5)
    for p in positions:
        assert p.run_id == _RUN_ID
        assert p.isin in _POOL_ISINS, f"{p.isin} must come only from the direction/horizon pool"
        assert p.horizon_days == 5
        assert p.exit_due == expected_exit_due
        assert p.realized_net_return is None
        assert p.created_at == _EVAL_TIME
        assert p.entry_ask == priced[p.isin].ask

    # Decoys never appear anywhere.
    isins_used = {p.isin for p in positions}
    assert "DE000DECOYH7" not in isins_used
    assert "DE000DECOYSH" not in isins_used


def test_top1_top3_top5_are_best_n_by_existing_ranking(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    best_by_isin, priced = _build_pool(dax_product_factory)
    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin=best_by_isin,
        priced=priced,
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=[],
    )
    by_kind: dict[ShadowPortfolioKind, list[str]] = {}
    for p in positions:
        by_kind.setdefault(p.portfolio, []).append(p.isin)

    # Descending score order: 0001(100) 0002(90) 0003(80) 0004(70) 0005(60) ...
    assert by_kind[ShadowPortfolioKind.TOP1] == ["DE000POL0001"]
    assert set(by_kind[ShadowPortfolioKind.TOP3]) == {
        "DE000POL0001",
        "DE000POL0002",
        "DE000POL0003",
    }
    assert set(by_kind[ShadowPortfolioKind.TOP5]) == {
        "DE000POL0001",
        "DE000POL0002",
        "DE000POL0003",
        "DE000POL0004",
        "DE000POL0005",
    }
    # TOP1 is exactly the real TurboEdge selection for this scan+underlying.
    assert by_kind[ShadowPortfolioKind.TOP1][0] == "DE000POL0001"


def test_single_pick_arms_write_at_most_one_row(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """TOP1, RANDOM_VALID_TURBO, LOWEST_SPREAD, LOWEST_FINANCING_COST,
    HIGHEST_LEVERAGE, LOWEST_LEVERAGE, MEDIAN_PRODUCT are single-pick by
    construction -- exactly one row each given a non-empty pool. TOP3/TOP5
    are the sole, documented exception (one row per real constituent ISIN,
    since `ShadowPosition` has no basket field -- see
    `_build_shadow_positions`'s own docstring)."""
    best_by_isin, priced = _build_pool(dax_product_factory)
    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin=best_by_isin,
        priced=priced,
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=[],
    )
    counts: dict[ShadowPortfolioKind, int] = {}
    for p in positions:
        counts[p.portfolio] = counts.get(p.portfolio, 0) + 1

    single_pick_arms = [
        ShadowPortfolioKind.TOP1,
        ShadowPortfolioKind.RANDOM_VALID_TURBO,
        ShadowPortfolioKind.LOWEST_SPREAD,
        ShadowPortfolioKind.LOWEST_FINANCING_COST,
        ShadowPortfolioKind.HIGHEST_LEVERAGE,
        ShadowPortfolioKind.LOWEST_LEVERAGE,
        ShadowPortfolioKind.MEDIAN_PRODUCT,
    ]
    for kind in single_pick_arms:
        assert counts.get(kind, 0) <= 1, f"{kind} wrote more than one row"
        assert counts.get(kind, 0) == 1, f"{kind} should have written exactly one row"

    assert counts[ShadowPortfolioKind.TOP3] == 3
    assert counts[ShadowPortfolioKind.TOP5] == 5


def test_extremum_arms_pick_the_genuine_extremum(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    best_by_isin, priced = _build_pool(dax_product_factory)
    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin=best_by_isin,
        priced=priced,
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=[],
    )
    by_kind = {p.portfolio: p.isin for p in positions if p.portfolio not in _SHADOW_TOP_N}

    assert by_kind[ShadowPortfolioKind.LOWEST_SPREAD] == "DE000POL0003"  # spread_pct=0.005
    assert by_kind[ShadowPortfolioKind.LOWEST_FINANCING_COST] == "DE000POL0003"  # 0.010
    assert by_kind[ShadowPortfolioKind.HIGHEST_LEVERAGE] == "DE000POL0005"  # leverage=9.0
    assert by_kind[ShadowPortfolioKind.LOWEST_LEVERAGE] == "DE000POL0004"  # leverage=1.0
    # Median of 7, ranked by score descending: index (7-1)//2 = 3 -> 4th
    # element (scores 100,90,80,70,...) -> isin with score=70.
    assert by_kind[ShadowPortfolioKind.MEDIAN_PRODUCT] == "DE000POL0004"


def test_median_product_even_pool_picks_lower_of_two_middle_elements(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """An even-sized pool has two middle elements by score; the lower
    (worse) of the two is picked, deterministically, so the pick is always
    a real product, never an interpolated one (rule 21)."""
    best_by_isin: dict[str, ProductHorizonEvaluation] = {}
    priced: dict[str, _PricedProduct] = {}
    for isin, score in [
        ("DE000EVN0001", 100.0),
        ("DE000EVN0002", 90.0),
        ("DE000EVN0003", 80.0),
        ("DE000EVN0004", 70.0),
    ]:
        best_by_isin[isin] = _eval(isin=isin, direction=Direction.LONG, horizon_days=5, score=score)
        priced[isin] = _priced(dax_product_factory, isin=isin, direction=Direction.LONG)

    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin=best_by_isin,
        priced=priced,
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=[],
    )
    median_row = next(p for p in positions if p.portfolio == ShadowPortfolioKind.MEDIAN_PRODUCT)
    # ranked desc: [100, 90, 80, 70] -> index (4-1)//2 = 1 -> score=90 (the
    # lower of the two middle values {90, 80}).
    assert median_row.isin == "DE000EVN0002"


def test_random_valid_turbo_reproducible_with_same_seed_different_with_other_seed(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """CLAUDE.md rule 33: the injected, seeded numpy Generator -- never
    `random`, never a fresh unseeded Generator -- drives the draw, so the
    same seed always reproduces the same product and a different seed can
    draw a different one."""
    best_by_isin: dict[str, ProductHorizonEvaluation] = {}
    priced: dict[str, _PricedProduct] = {}
    for i in range(1, 6):
        isin = f"DE000RND000{i}"
        best_by_isin[isin] = _eval(
            isin=isin, direction=Direction.LONG, horizon_days=5, score=100.0 - i
        )
        priced[isin] = _priced(dax_product_factory, isin=isin, direction=Direction.LONG)

    def _random_pick(seed: int) -> str:
        positions = _build_shadow_positions(
            run_id=_RUN_ID,
            best_by_isin=best_by_isin,
            priced=priced,
            evaluation_time=_EVAL_TIME,
            rng=np.random.default_rng(seed),
            warnings=[],
        )
        row = next(p for p in positions if p.portfolio == ShadowPortfolioKind.RANDOM_VALID_TURBO)
        return row.isin

    pick_seed0_a = _random_pick(0)
    pick_seed0_b = _random_pick(0)
    pick_seed1 = _random_pick(1)

    assert pick_seed0_a == pick_seed0_b, "same seed must draw the same product"
    assert pick_seed0_a != pick_seed1, "a different seed must be able to draw a different product"


def test_empty_pool_writes_no_rows() -> None:
    warnings: list[str] = []
    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin={},
        priced={},
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=warnings,
    )
    assert positions == []
    assert "shadow_pool_empty" in warnings


def test_missing_measure_skips_only_that_arm_and_logs_reason(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """If `leverage_value` were ever unavailable (rule 29: never
    guessed/defaulted to 0), HIGHEST_LEVERAGE/LOWEST_LEVERAGE must write no
    row and log why -- without suppressing any other arm."""
    best_by_isin: dict[str, ProductHorizonEvaluation] = {}
    priced: dict[str, _PricedProduct] = {}
    for i in range(1, 4):
        isin = f"DE000NLV000{i}"
        best_by_isin[isin] = _eval(
            isin=isin, direction=Direction.LONG, horizon_days=5, score=100.0 - i
        )
        p = _priced(dax_product_factory, isin=isin, direction=Direction.LONG)
        # `_PricedProduct` types `leverage_value` as `float` (never `None`
        # in production -- see the docstring of `_build_shadow_positions`),
        # but the defensive skip-with-warning branch is exercised here via
        # an explicit runtime override (bypassing the frozen dataclass), to
        # prove it actually works should that invariant ever change.
        object.__setattr__(p, "leverage_value", None)
        priced[isin] = p

    warnings: list[str] = []
    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin=best_by_isin,
        priced=priced,
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=warnings,
    )
    kinds_written = {p.portfolio for p in positions}
    assert ShadowPortfolioKind.HIGHEST_LEVERAGE not in kinds_written
    assert ShadowPortfolioKind.LOWEST_LEVERAGE not in kinds_written
    assert f"shadow_arm_no_value:{ShadowPortfolioKind.HIGHEST_LEVERAGE.value}" in warnings
    assert f"shadow_arm_no_value:{ShadowPortfolioKind.LOWEST_LEVERAGE.value}" in warnings
    # Every other arm is unaffected.
    assert ShadowPortfolioKind.TOP1 in kinds_written
    assert ShadowPortfolioKind.LOWEST_SPREAD in kinds_written
    assert ShadowPortfolioKind.MEDIAN_PRODUCT in kinds_written


def test_pool_smaller_than_top_n_writes_as_many_rows_as_exist(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """A genuinely small eligible pool (2 candidates) is not an empty-pool
    condition (rule 29 is about missing DATA, not a small-but-real
    universe) -- TOP3/TOP5 write as many rows as the pool actually has."""
    best_by_isin: dict[str, ProductHorizonEvaluation] = {}
    priced: dict[str, _PricedProduct] = {}
    for isin, score in [("DE000SML0001", 100.0), ("DE000SML0002", 90.0)]:
        best_by_isin[isin] = _eval(isin=isin, direction=Direction.LONG, horizon_days=5, score=score)
        priced[isin] = _priced(dax_product_factory, isin=isin, direction=Direction.LONG)

    positions = _build_shadow_positions(
        run_id=_RUN_ID,
        best_by_isin=best_by_isin,
        priced=priced,
        evaluation_time=_EVAL_TIME,
        rng=np.random.default_rng(0),
        warnings=[],
    )
    counts: dict[ShadowPortfolioKind, int] = {}
    for p in positions:
        counts[p.portfolio] = counts.get(p.portfolio, 0) + 1
    assert counts[ShadowPortfolioKind.TOP1] == 1
    assert counts[ShadowPortfolioKind.TOP3] == 2
    assert counts[ShadowPortfolioKind.TOP5] == 2
