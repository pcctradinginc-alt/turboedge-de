"""Tests for the lottery-pricing screen (A02, Pi-CAPM).

Two tests carry most of the weight:

* ``test_scaling_every_price_in_a_cell_leaves_the_slope_unchanged`` is the
  endogeneity guard. Multiplying ask and bid by one factor leaves the
  relative spread untouched but changes any ask-derived leverage, so a
  regressor built from the price would move and barrier distance does not.
* ``test_a_slope_in_the_opposite_direction_is_not_a_candidate`` stops the
  screen from reporting a refutation as support. Pi-CAPM predicts a
  specific sign; a clean slope the other way is a real result and the
  opposite of this hypothesis.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.research.lottery_pricing import (
    MAX_QUOTE_AGE_SECONDS,
    MIN_CELLS,
    MIN_PRODUCTS_PER_CELL,
    evaluate,
)
from turboedge.storage.schemas import Direction, FieldReliability, ProductSnapshot

SPOT = 18_000.0
BASE = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)


def _snap(
    *,
    isin: str,
    issuer: str = "BankA",
    barrier_distance: float,
    relative_spread: float,
    minute_offset: int = 0,
    quote_age: float = 1.0,
    ratio: float = 0.01,
    premium_fraction: float = 0.002,
    direction: Direction = Direction.LONG,
    ratio_reliability: FieldReliability = FieldReliability.CROSS_SOURCE_VERIFIED,
    **over: object,
) -> ProductSnapshot:
    """One quote with an exactly prescribed barrier distance and spread."""
    retrieved = BASE + timedelta(minutes=minute_offset)
    absolute = SPOT * barrier_distance
    barrier = SPOT - absolute if direction is Direction.LONG else SPOT + absolute
    # ask per unit = intrinsic + premium*spot, so premium is controllable
    ask = (absolute + premium_fraction * SPOT) * ratio
    bid = ask * (1.0 - relative_spread)
    defaults: dict[str, object] = dict(
        isin=isin,
        wkn=isin[-6:],
        issuer=issuer,
        venue="stuttgart",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=direction,
        product_type="turbo_open_end",
        financing_level=barrier,
        knockout_barrier=barrier,
        ratio=ratio,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        maturity=None,
        first_trading_day=None,
        bid=bid,
        ask=ask,
        bid_size=1000.0,
        ask_size=1000.0,
        quote_timestamp=retrieved - timedelta(seconds=quote_age),
        quote_presence=True,
        bid_only=False,
        knocked_out=False,
        trading_hours="09:00-22:00",
        product_age_days=120,
        underlying_price_ref=SPOT,
        underlying_price_ref_timestamp=retrieved,
        raw_hash=hashlib.sha256(isin.encode()).hexdigest(),
        observation_time=retrieved - timedelta(seconds=quote_age),
        available_at=retrieved,
        retrieved_at=retrieved,
        source_timestamp=retrieved,
        source="test",
        parser_version="1",
        is_stale=False,
        quality_score=0.95,
        ratio_reliability=ratio_reliability,
    )
    defaults.update(over)
    return ProductSnapshot(**defaults)  # type: ignore[arg-type]


def _cells(
    n_cells: int,
    *,
    slope: float,
    noise: float = 0.0,
    seed: int = 5,
    products: int = 6,
    issuer: Callable[[int], str] = lambda i: "BankA",
) -> list[ProductSnapshot]:
    """`n_cells` minutes, each with `products` quotes on a prescribed slope.

    ``slope`` is d(relative spread)/d(barrier distance); the hypothesis
    predicts a negative one.
    """
    rng = np.random.default_rng(seed)
    out: list[ProductSnapshot] = []
    for cell in range(n_cells):
        for p in range(products):
            distance = 0.02 + 0.01 * p
            spread = 0.01 + slope * (distance - 0.045)
            if noise:
                spread += float(rng.normal(0.0, noise))
            out.append(
                _snap(
                    isin=f"DE000L{cell:03d}{p:03d}",
                    issuer=issuer(cell),
                    barrier_distance=distance,
                    relative_spread=max(spread, 1e-6),
                    minute_offset=cell,
                )
            )
    return out


class TestPlantedSlope:
    def test_a_planted_negative_slope_is_recovered(self) -> None:
        """Lottery-like products made expensive by construction."""
        result = evaluate(_cells(40, slope=-0.2, noise=0.0005), reps=500)

        assert result.cells_used == 40
        assert result.mean_spread_slope == pytest.approx(-0.2, rel=0.1)
        assert result.interval_excludes_zero
        assert result.spread_ci_high < 0.0
        assert result.verdict == "CANDIDATE"
        assert result.mean_economic_size_bps > 0.0
        assert result.median_economic_size_bps > 0.0

    def test_no_slope_gives_an_interval_through_zero(self) -> None:
        result = evaluate(_cells(40, slope=0.0, noise=0.0005), reps=500)

        assert not result.interval_excludes_zero
        assert result.spread_ci_low < 0.0 < result.spread_ci_high
        assert result.verdict == "NO_IMPROVEMENT"

    def test_a_slope_in_the_opposite_direction_is_not_a_candidate(self) -> None:
        """A clean refutation must not be reported as support.

        Pi-CAPM predicts lottery-like payoffs are *dearer*. If the spread
        instead narrows as the barrier nears, the hypothesis is refuted,
        and the verdict has to say so rather than celebrating a tight
        interval.
        """
        result = evaluate(_cells(40, slope=+0.2, noise=0.0005), reps=500)

        assert result.interval_excludes_zero
        assert result.mean_spread_slope > 0.0
        assert result.verdict == "NO_IMPROVEMENT"

    def test_a_clean_but_tiny_slope_fails_the_economic_criterion(self) -> None:
        """Declared in advance: a measurement without a consequence."""
        result = evaluate(_cells(40, slope=-0.0005, noise=1e-7), reps=500)

        assert result.interval_excludes_zero
        assert result.mean_spread_slope < 0.0
        assert result.median_economic_size_bps < 5.0
        assert result.verdict == "IMPROVEMENT_NOT_SIGNIFICANT"

    def test_the_verdict_uses_the_median_cell_not_the_mean(self) -> None:
        """On the real archive the two differ by a factor of seven.

        A handful of cells carry an enormous slope, so a mean-based size
        would pass the economic criterion on the strength of outliers while
        the representative cell sat below it. The median is the stricter
        test, which is why switching to it after seeing the data cannot
        flatter the result.
        """
        # 39 tiny-slope cells plus one enormous one: the mean clears 5 bps,
        # the median does not.
        snaps = _cells(39, slope=-0.0005, noise=1e-7) + [
            _snap(
                isin=f"DE000HUGE{p:03d}",
                barrier_distance=0.02 + 0.01 * p,
                relative_spread=max(0.01 - 5.0 * (0.02 + 0.01 * p - 0.045), 1e-6),
                minute_offset=500,
            )
            for p in range(6)
        ]
        result = evaluate(snaps, reps=500)
        assert result.mean_economic_size_bps > result.median_economic_size_bps
        assert result.median_economic_size_bps < 5.0
        assert result.verdict == "IMPROVEMENT_NOT_SIGNIFICANT"


class TestEndogeneityGuard:
    def test_scaling_every_price_in_a_cell_leaves_the_slope_unchanged(self) -> None:
        """The endogeneity guard the brief asks for, as an invariance.

        Multiplying ask and bid by one factor leaves the relative spread
        exactly as it was, so the estimate must not move. An ask-derived
        leverage regressor would move a great deal, which is why barrier
        distance is the regressor.
        """
        snaps = _cells(35, slope=-0.15, noise=0.0004)
        scaled = [
            s.model_copy(update={"ask": s.ask * 3.0, "bid": (s.bid or 0.0) * 3.0}) for s in snaps
        ]

        base = evaluate(snaps, reps=200)
        after = evaluate(scaled, reps=200)

        assert base.mean_spread_slope == pytest.approx(after.mean_spread_slope, rel=1e-9)
        assert base.cells_used == after.cells_used

    def test_the_regressor_does_not_depend_on_the_ratio_either(self) -> None:
        """A different Bezugsverhaeltnis is a different product size, not a
        different barrier distance."""
        snaps = _cells(35, slope=-0.15, noise=0.0004)
        re_ratioed = [
            s.model_copy(
                update={"ratio": 0.1, "ask": (s.ask or 0.0) * 10.0, "bid": (s.bid or 0.0) * 10.0}
            )
            for s in snaps
        ]
        assert evaluate(snaps, reps=200).mean_spread_slope == pytest.approx(
            evaluate(re_ratioed, reps=200).mean_spread_slope, rel=1e-9
        )


class TestFilters:
    @pytest.mark.parametrize(
        ("override", "reason"),
        [
            ({"knocked_out": True}, "knocked_out"),
            ({"bid": None}, "no_bid"),
            ({"ask": None}, "no_ask"),
            ({"bid_only": True}, "no_ask"),
            ({"knockout_barrier": None}, "no_barrier"),
            ({"underlying_price_ref": None}, "no_underlying_reference"),
            ({"quote_timestamp": None}, "no_quote_timestamp"),
            ({"ratio_reliability": FieldReliability.UNVERIFIED}, "ratio_unverified"),
        ],
    )
    def test_unusable_quotes_are_dropped_and_counted(
        self, override: dict[str, object], reason: str
    ) -> None:
        snaps = [
            _snap(
                isin="DE000FILT001",
                barrier_distance=0.03,
                relative_spread=0.01,
                **override,  # type: ignore[arg-type]
            )
        ]
        assert evaluate(snaps, reps=100).dropped[reason] == 1

    def test_a_stale_quote_is_excluded(self) -> None:
        """Cheapness and quote age were measured to move together, so a
        stale quote would import that association into this one."""
        snaps = [
            _snap(
                isin="DE000STALE01",
                barrier_distance=0.03,
                relative_spread=0.01,
                quote_age=MAX_QUOTE_AGE_SECONDS + 1.0,
            )
        ]
        assert evaluate(snaps, reps=100).dropped["quote_too_stale"] == 1

    def test_a_crossed_quote_is_refused_rather_than_producing_a_negative_spread(self) -> None:
        snaps = [
            _snap(
                isin="DE000CROSS01",
                barrier_distance=0.03,
                relative_spread=-0.05,  # bid above ask
            )
        ]
        assert evaluate(snaps, reps=100).dropped["crossed_quote"] == 1

    def test_a_touched_barrier_is_excluded(self) -> None:
        snaps = [
            _snap(
                isin="DE000TOUCH01",
                barrier_distance=0.03,
                relative_spread=0.01,
                knockout_barrier=SPOT + 10.0,  # above spot for a long
            )
        ]
        assert evaluate(snaps, reps=100).dropped["barrier_touched"] == 1

    def test_a_cell_below_the_minimum_product_count_is_dropped(self) -> None:
        result = evaluate(_cells(5, slope=-0.2, products=MIN_PRODUCTS_PER_CELL - 1), reps=100)
        assert result.cells_used == 0
        assert result.dropped["cell_too_small"] == 5

    def test_a_cell_with_one_barrier_distance_cannot_yield_a_slope(self) -> None:
        snaps = [
            _snap(
                isin=f"DE000FLAT{p:03d}",
                barrier_distance=0.03,
                relative_spread=0.01 + 0.0001 * p,
            )
            for p in range(6)
        ]
        result = evaluate(snaps, reps=100)
        assert result.dropped["no_barrier_variation_in_cell"] == 1
        assert result.cells_used == 0


class TestVerdictAndSampling:
    def test_too_few_cells_is_insufficient_not_negative(self) -> None:
        result = evaluate(_cells(MIN_CELLS - 1, slope=-0.2, noise=0.0005), reps=200)
        assert result.cells_used == MIN_CELLS - 1
        assert result.verdict == "INSUFFICIENT_SAMPLE"

    def test_an_empty_archive_reports_nothing_rather_than_a_null_result(self) -> None:
        result = evaluate([], reps=100)
        assert result.cells_used == 0
        assert math.isnan(result.mean_spread_slope)
        assert result.verdict == "INSUFFICIENT_SAMPLE"

    def test_issuer_and_minute_both_split_cells(self) -> None:
        """Two issuers in the same minute are two cells, not one.

        That is what differences out pricing style; pooling them would let
        a cheap issuer's level masquerade as a barrier-distance effect.
        """
        snaps = _cells(20, slope=-0.2, issuer=lambda i: "BankA") + _cells(
            20, slope=-0.2, issuer=lambda i: "BankB", seed=9
        )
        result = evaluate(snaps, reps=200)
        assert result.cells_used == 40
        assert sorted({c.issuer for c in result.cells}) == ["BankA", "BankB"]

    def test_results_are_reproducible(self) -> None:
        snaps = _cells(35, slope=-0.15, noise=0.0004)
        a = evaluate(snaps, reps=300, seed=1)
        b = evaluate(snaps, reps=300, seed=1)
        assert (a.spread_ci_low, a.spread_ci_high) == (b.spread_ci_low, b.spread_ci_high)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"reps": 10}, "reps must be >= 100"), ({"alpha": 1.0}, r"alpha must be in \(0, 1\)")],
    )
    def test_invalid_inputs_are_refused(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            evaluate([], **kwargs)  # type: ignore[arg-type]


def test_nothing_in_the_decision_path_imports_this_module() -> None:
    """A cost diagnostic may be reported; it must never size a trade."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "turboedge"
    decision_path = (
        "pipeline",
        "ranking",
        "simulation",
        "models",
        "alpha",
        "learning",
        "positions",
        "forecast",
        "meta",
    )
    pattern = re.compile(r"\blottery_pricing\b")
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.relative_to(root).parts[0] in decision_path and pattern.search(path.read_text())
    ]
    assert offenders == []
