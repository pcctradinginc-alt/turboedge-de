"""Tests for the barrier-touch reference and its two comparison diagnostics.

Three kinds of check, and the distinction matters because only the first
kind can fail for a reason that is purely mathematical:

1. Identities that must hold exactly -- the analytic value at a barrier
   placed in log space is ``2*Phi(-k)``, and a linear offset of the same
   nominal ``k`` is *not* that value.
2. Directions that follow from the geometry -- ``exp(-a) > 1-a`` forces the
   long barrier outward and the short barrier inward, so the bias always
   favours short touches.
3. Behaviour of the bootstrap interval under known dependence -- an
   overlapping sample must produce a wider interval than an independent one
   of the same length, or the interval is not doing its job.

The headline figures in ``docs/ko_reference_2026-10-25.md`` come from the
full 128-cell grid on real bars. These tests assert the mechanism, not
those numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from turboedge.research.ko_reference import (
    analytic_touch_probability,
    compare_methods,
    effective_log_k,
    horizon_sigma_ratio,
    linear_offset_bias,
    moving_block_bootstrap_ci,
)
from turboedge.storage.schemas import Direction


class TestEffectiveLogK:
    @pytest.mark.parametrize("k", [0.75, 1.5, 2.0, 3.0, 4.0])
    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    @pytest.mark.parametrize(("sigma", "horizon"), [(0.008, 3), (0.011, 10), (0.03, 14)])
    def test_a_log_placed_barrier_round_trips_to_its_own_k(
        self, k: float, direction: Direction, sigma: float, horizon: int
    ) -> None:
        """Placing at exp(-k*sigma*sqrt(h)) and reading back gives k."""
        a = k * sigma * math.sqrt(horizon)
        spot = 100.0
        barrier = spot * math.exp(-a if direction is Direction.LONG else a)
        got = effective_log_k(
            spot=spot,
            barrier=barrier,
            sigma_daily=sigma,
            horizon_days=horizon,
            direction=direction,
        )
        assert got == pytest.approx(k, rel=1e-12)

    def test_a_linear_offset_does_not_round_trip(self) -> None:
        """The whole point of finding 3: 1-a and exp(-a) are different barriers.

        If this ever passes, the two placements have been made equivalent
        somewhere and the bias measurement below is meaningless.
        """
        k, sigma, horizon = 2.0, 0.011, 10
        a = k * sigma * math.sqrt(horizon)
        long_k = effective_log_k(
            spot=100.0,
            barrier=100.0 * (1.0 - a),
            sigma_daily=sigma,
            horizon_days=horizon,
            direction=Direction.LONG,
        )
        short_k = effective_log_k(
            spot=100.0,
            barrier=100.0 * (1.0 + a),
            sigma_daily=sigma,
            horizon_days=horizon,
            direction=Direction.SHORT,
        )
        assert long_k > k
        assert short_k < k
        # And the two sides are not even the same distance from k.
        assert not math.isclose(long_k - k, k - short_k, rel_tol=1e-3)

    def test_an_already_touched_barrier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="wrong side of spot"):
            effective_log_k(
                spot=100.0,
                barrier=101.0,
                sigma_daily=0.01,
                horizon_days=5,
                direction=Direction.LONG,
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"spot": 0.0}, "spot must be > 0"),
            ({"barrier": -1.0}, "barrier must be > 0"),
            ({"sigma_daily": 0.0}, "sigma_daily must be > 0"),
            ({"horizon_days": 0}, "horizon_days must be >= 1"),
        ],
    )
    def test_invalid_inputs_are_refused(self, kwargs: dict[str, float], match: str) -> None:
        base = {
            "spot": 100.0,
            "barrier": 95.0,
            "sigma_daily": 0.01,
            "horizon_days": 5,
            "direction": Direction.LONG,
        }
        with pytest.raises(ValueError, match=match):
            effective_log_k(**{**base, **kwargs})  # type: ignore[arg-type]


class TestAnalyticReference:
    @pytest.mark.parametrize("k", [1.0, 1.5, 2.0, 2.5, 3.0])
    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_driftless_equals_two_phi_minus_k(self, k: float, direction: Direction) -> None:
        """The reflection-principle identity, as an equality.

        ``first_passage`` owns the proof of the general formula; this
        asserts that routing a *price* barrier through it lands on the same
        closed form, which is what makes the grid comparison meaningful.
        """
        sigma, horizon = 0.011, 10
        a = k * sigma * math.sqrt(horizon)
        spot = 100.0
        barrier = spot * math.exp(-a if direction is Direction.LONG else a)
        got = analytic_touch_probability(
            spot=spot,
            barrier=barrier,
            sigma_daily=sigma,
            horizon_days=horizon,
            direction=direction,
            log_drift_daily=0.0,
        )
        assert got == pytest.approx(2.0 * float(norm.cdf(-k)), abs=1e-12)

    def test_the_reference_is_symmetric_and_that_is_the_finding(self) -> None:
        """Long and short get the same number at the same log distance.

        Not a bug -- it is why the reference cannot represent the measured
        long/short ratio of 1.19 (k=1.5) to 2.01 (k=3) on real bars, which
        is finding 1. A symmetric model has no parameter for skew.
        """
        sigma, horizon, k = 0.011, 10, 2.0
        a = k * sigma * math.sqrt(horizon)
        long_p = analytic_touch_probability(
            spot=100.0,
            barrier=100.0 * math.exp(-a),
            sigma_daily=sigma,
            horizon_days=horizon,
            direction=Direction.LONG,
        )
        short_p = analytic_touch_probability(
            spot=100.0,
            barrier=100.0 * math.exp(a),
            sigma_daily=sigma,
            horizon_days=horizon,
            direction=Direction.SHORT,
        )
        assert long_p == pytest.approx(short_p, abs=1e-12)

    def test_drift_towards_the_barrier_raises_the_probability(self) -> None:
        common = {
            "spot": 100.0,
            "barrier": 95.0,
            "sigma_daily": 0.01,
            "horizon_days": 10,
            "direction": Direction.LONG,
        }
        down = analytic_touch_probability(**common, log_drift_daily=-0.002)  # type: ignore[arg-type]
        flat = analytic_touch_probability(**common, log_drift_daily=0.0)  # type: ignore[arg-type]
        up = analytic_touch_probability(**common, log_drift_daily=0.002)  # type: ignore[arg-type]
        assert down > flat > up


class TestLinearOffsetBias:
    @pytest.mark.parametrize("k", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    def test_the_bias_always_favours_short(self, k: float) -> None:
        """Geometry, not data: exp(-a) > 1-a for every a > 0."""
        bias = linear_offset_bias(k, 0.011, 10)
        assert bias.k_effective_long > k > bias.k_effective_short
        assert bias.p_short > bias.p_long
        assert bias.ratio > 1.0

    def test_the_bias_grows_with_the_barrier_distance(self) -> None:
        """At k=2 it is modest; by k=4 it is an order of magnitude.

        ``DEFAULT_SIGMA_LEVELS`` in ``simulation/ko_calibration.py`` goes to
        4.0, so this is inside the measured grid, not a hypothetical.
        """
        ratios = [linear_offset_bias(k, 0.011, 10).ratio for k in (1.5, 2.0, 2.5, 3.0, 4.0)]
        assert ratios == sorted(ratios)
        assert ratios[1] > 1.2  # k=2.0
        assert ratios[-1] > 5.0  # k=4.0

    def test_a_vanishing_offset_has_a_vanishing_bias(self) -> None:
        """Sanity floor: as a -> 0, 1-a and exp(-a) agree to first order."""
        bias = linear_offset_bias(0.05, 0.001, 1)
        assert bias.ratio == pytest.approx(1.0, abs=1e-3)

    def test_an_offset_reaching_zero_is_refused_not_clamped(self) -> None:
        with pytest.raises(ValueError, match="no log distance"):
            linear_offset_bias(4.0, 0.2, 10)

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((0.0, 0.01, 10), "k must be > 0"),
            ((2.0, 0.0, 10), "sigma_daily must be > 0"),
            ((2.0, 0.01, 0), "horizon_days must be >= 1"),
        ],
    )
    def test_invalid_inputs_are_refused(self, args: tuple[float, float, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            linear_offset_bias(*args)


class TestHorizonSigmaRatio:
    def test_a_consistent_simulation_scores_one(self) -> None:
        assert horizon_sigma_ratio(
            realized_sigma=0.011 * math.sqrt(10), sigma_daily=0.011, horizon_days=10
        ) == pytest.approx(1.0)

    def test_over_dispersion_scores_above_one(self) -> None:
        """The measured default is ~1.41; this asserts the sign only."""
        assert (
            horizon_sigma_ratio(
                realized_sigma=1.4 * 0.011 * math.sqrt(10), sigma_daily=0.011, horizon_days=10
            )
            > 1.0
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sigma_daily": 0.0}, "sigma_daily must be > 0"),
            ({"horizon_days": 0}, "horizon_days must be >= 1"),
        ],
    )
    def test_invalid_inputs_are_refused(self, kwargs: dict[str, float], match: str) -> None:
        base = {"realized_sigma": 0.03, "sigma_daily": 0.01, "horizon_days": 10}
        with pytest.raises(ValueError, match=match):
            horizon_sigma_ratio(**{**base, **kwargs})  # type: ignore[arg-type]


class TestMovingBlockBootstrapCi:
    def test_the_interval_brackets_an_independent_rate(self) -> None:
        rng = np.random.default_rng(3)
        hits = rng.random(2000) < 0.07
        lo, hi = moving_block_bootstrap_ci(hits, block=20)
        assert lo <= hits.mean() <= hi

    def test_overlap_widens_the_interval(self) -> None:
        """The reason this function exists instead of a binomial interval.

        Both samples have the same length and nearly the same mean; only
        the dependence differs. A binomial interval would be identical for
        both, which is how an overlapping sample gets reported as more
        precise than it is.
        """
        rng = np.random.default_rng(5)
        daily = (rng.random(4000) < 0.012).astype(float)
        # 14-day rolling windows: consecutive entries share 13 days.
        overlapping = np.convolve(daily, np.ones(14), "valid") > 0
        independent = rng.random(overlapping.size) < overlapping.mean()
        lo_o, hi_o = moving_block_bootstrap_ci(overlapping, block=28, seed=7)
        lo_i, hi_i = moving_block_bootstrap_ci(independent, block=28, seed=7)
        assert (hi_o - lo_o) > 1.5 * (hi_i - lo_i)

    def test_a_sample_too_short_for_three_blocks_returns_nan(self) -> None:
        lo, hi = moving_block_bootstrap_ci(np.ones(10, dtype=bool), block=20)
        assert math.isnan(lo) and math.isnan(hi)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"block": 0}, "block must be >= 1"),
            ({"reps": 0}, "reps must be >= 1"),
            ({"alpha": 1.0}, r"alpha must be in \(0, 1\)"),
        ],
    )
    def test_invalid_inputs_are_refused(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            moving_block_bootstrap_ci(np.ones(100, dtype=bool), **{"block": 10, **kwargs})  # type: ignore[arg-type]

    def test_the_interval_is_reproducible(self) -> None:
        hits = np.random.default_rng(1).random(600) < 0.1
        assert moving_block_bootstrap_ci(hits, block=20, seed=42) == moving_block_bootstrap_ci(
            hits, block=20, seed=42
        )


class TestCompareMethods:
    def test_ranking_is_by_mean_error_and_coverage_is_reported_separately(self) -> None:
        """The two can disagree, and the grid is a case where they did."""
        realised = [0.10, 0.05, 0.02]
        lo, hi = [0.08, 0.03, 0.01], [0.12, 0.07, 0.03]
        results = compare_methods(
            realised=realised,
            ci_low=lo,
            ci_high=hi,
            predictions={
                "close": [0.10, 0.05, 0.02],
                # Small mean error, but outside every interval.
                "biased": [0.13, 0.08, 0.04],
            },
        )
        assert [r.name for r in results] == ["close", "biased"]
        assert results[0].coverage == 1.0
        assert results[1].cells_inside_ci == 0
        assert results[1].cells_over_predicting == 3

    def test_a_nan_interval_counts_for_error_but_not_coverage(self) -> None:
        result = compare_methods(
            realised=[0.1, 0.1],
            ci_low=[0.05, float("nan")],
            ci_high=[0.15, float("nan")],
            predictions={"m": [0.1, 0.1]},
        )[0]
        assert result.cells == 2
        assert result.cells_inside_ci == 1
        assert result.coverage == pytest.approx(0.5)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must have length 2"):
            compare_methods(
                realised=[0.1, 0.1],
                ci_low=[0.0, 0.0],
                ci_high=[0.2, 0.2],
                predictions={"m": [0.1]},
            )

    def test_no_predictions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compare_methods(realised=[0.1], ci_low=[0.0], ci_high=[0.2], predictions={})


def test_nothing_in_production_imports_this_module() -> None:
    """The reference is a measuring instrument, not a model.

    Mirrors the same guard on ``research/first_passage.py``. If a
    production path ever needs an analytic P(KO), that is a deliberate
    model change with its own trial -- not an import.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "turboedge"
    pattern = re.compile(r"\bko_reference\b")
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.parent.name != "research" and pattern.search(path.read_text())
    ]
    assert offenders == []
