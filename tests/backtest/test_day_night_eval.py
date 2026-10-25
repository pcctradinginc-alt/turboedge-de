"""Tests for the day/night return decomposition and its look-ahead audit.

The most important test in this file is
``test_a_stale_open_manufactures_an_edge_the_audit_catches``. It plants the
exact defect that produced this experiment's first, wrong result -- an open
pulled toward the previous close -- and asserts that the rule's measured
mean rises while the clean variant stays flat. If that test ever fails, the
audit has stopped protecting anything.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.backtest.day_night_eval import (
    MIN_OOS_OBSERVATIONS,
    FamilyResult,
    SubHypothesis,
    benjamini_hochberg,
    decompose,
    evaluate_cell,
    evaluate_family,
    shares_a_price,
)
from turboedge.storage.schemas import UnderlyingBar

START = datetime(2000, 1, 3, 17, 30, tzinfo=UTC)


def _bar(day: int, *, open_: float, close: float, uid: str = "DAX") -> UnderlyingBar:
    ts = START + timedelta(days=day)
    return UnderlyingBar(
        observation_time=ts,
        available_at=ts,
        retrieved_at=ts,
        source="test",
        schema_version="1",
        parser_version="1",
        quality_score=1.0,
        underlying_id=uid,
        ts=ts,
        interval="1d",
        open=open_,
        high=max(open_, close) * 1.001,
        low=min(open_, close) * 0.999,
        close=close,
    )


def _series(
    n: int,
    *,
    seed: int = 7,
    stale_open_weight: float = 0.0,
    night_predicts_day: float = 0.0,
) -> list[UnderlyingBar]:
    """A synthetic price series with two optionally-planted defects/effects.

    ``stale_open_weight`` pulls the recorded open back toward the previous
    close, which is the stale-open artefact. ``night_predicts_day`` makes
    the session genuinely continue the overnight move, which is the real
    effect the experiment is looking for. They are separable here and the
    audit must tell them apart.
    """
    rng = np.random.default_rng(seed)
    bars: list[UnderlyingBar] = []
    close = 100.0
    for i in range(n):
        true_night = float(rng.normal(0.0, 0.007))
        true_open = close * math.exp(true_night)
        # A stale open sits between the previous close and the true open.
        recorded_open = close * math.exp(true_night * (1.0 - stale_open_weight))
        day = float(rng.normal(0.0, 0.011)) + night_predicts_day * true_night
        close = true_open * math.exp(day)
        bars.append(_bar(i, open_=recorded_open, close=close))
    return bars


class TestDecompose:
    def test_the_parts_sum_to_the_close_to_close_return(self) -> None:
        parts = decompose(_series(50))
        assert np.allclose(parts.night + parts.day, parts.close_to_close)

    def test_a_duplicate_bar_does_not_fabricate_a_zero_night(self) -> None:
        """A re-fetched day appears twice in the archive legitimately."""
        bars = _series(30)
        with_dup = [*bars, bars[10]]
        assert decompose(with_dup).night.size == decompose(bars).night.size

    def test_bars_are_sorted_before_decomposing(self) -> None:
        bars = _series(30)
        shuffled = [bars[i] for i in np.random.default_rng(1).permutation(len(bars))]
        assert np.allclose(decompose(shuffled).night, decompose(bars).night)

    def test_an_unusable_bar_breaks_the_chain_instead_of_stitching_over_it(self) -> None:
        """A night measured across a hole is a multi-day return, not a night.

        Dropping the bad bar alone would silently turn one gap into one
        overnight observation that spans two sessions.
        """
        bars = _series(20)
        broken = list(bars)
        broken[10] = _bar(10, open_=0.0, close=0.0)
        parts = decompose(broken)
        # One bar removed costs two night observations: its own and the one
        # that would have spanned the hole.
        assert parts.night.size == decompose(bars).night.size - 2

    def test_too_little_history_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2 usable"):
            decompose(_series(2))


class TestLookAheadAudit:
    def test_only_the_within_bar_pair_shares_a_price(self) -> None:
        assert shares_a_price(SubHypothesis.NIGHT_DAY)
        for sub in SubHypothesis:
            if sub is not SubHypothesis.NIGHT_DAY:
                assert not shares_a_price(sub)

    def test_a_stale_open_manufactures_an_edge_the_audit_catches(self) -> None:
        """The defect that produced this experiment's first, wrong result.

        With no genuine predictability planted at all, a stale open alone
        makes the shared-price rule look profitable, while the clean
        next-session variant stays flat. Measured on real data the same
        way round: corr(night, day) was +0.264 on the Euro Stoxx 50 and
        the rule's mean fell by a factor of 25 once the shared price went.
        """
        clean_bars = _series(3000, stale_open_weight=0.0, night_predicts_day=0.0)
        stale_bars = _series(3000, stale_open_weight=0.5, night_predicts_day=0.0)

        def mean_for(bars: list[UnderlyingBar], sub: SubHypothesis) -> float:
            return evaluate_cell(
                bars,
                underlying_id="SYNTH",
                sub_hypothesis=sub,
                min_train=500,
                step=500,
                null_draws=100,
            ).rule_mean

        shared_clean = mean_for(clean_bars, SubHypothesis.NIGHT_DAY)
        shared_stale = mean_for(stale_bars, SubHypothesis.NIGHT_DAY)
        next_stale = mean_for(stale_bars, SubHypothesis.NIGHT_NEXT_DAY)

        # The artefact inflates the shared-price cell...
        assert shared_stale > shared_clean
        # ...and leaves the clean cell with nothing, because nothing is there.
        assert abs(next_stale) < abs(shared_stale)

    def test_the_night_day_correlation_is_recorded_and_rises_with_staleness(self) -> None:
        """The diagnostic that identifies the mechanism rather than the symptom."""
        clean = evaluate_cell(
            _series(2000, stale_open_weight=0.0),
            underlying_id="SYNTH",
            sub_hypothesis=SubHypothesis.NIGHT_NEXT_DAY,
            min_train=500,
            step=500,
            null_draws=100,
        )
        stale = evaluate_cell(
            _series(2000, stale_open_weight=0.6),
            underlying_id="SYNTH",
            sub_hypothesis=SubHypothesis.NIGHT_NEXT_DAY,
            min_train=500,
            step=500,
            null_draws=100,
        )
        assert stale.night_day_correlation > clean.night_day_correlation + 0.1


class TestEvaluateCell:
    def test_a_planted_clean_effect_is_detected(self) -> None:
        """The procedure can find a real effect, so a null result means something."""
        result = evaluate_cell(
            _series(3000, night_predicts_day=0.8),
            underlying_id="SYNTH",
            sub_hypothesis=SubHypothesis.NIGHT_DAY,
            min_train=500,
            step=500,
            null_draws=500,
        )
        assert result.rule_mean > 0.0
        assert result.p_value < 0.05
        assert result.ci_low > 0.0

    def test_pure_noise_is_not_detected(self) -> None:
        result = evaluate_cell(
            _series(3000, seed=99),
            underlying_id="SYNTH",
            sub_hypothesis=SubHypothesis.NIGHT_NEXT_DAY,
            min_train=500,
            step=500,
            null_draws=500,
        )
        assert result.p_value > 0.05
        assert result.ci_low < 0.0 < result.ci_high

    def test_the_null_preserves_drift_so_an_always_long_rule_is_not_significant(self) -> None:
        """Why the null is a circular shift and not zero.

        A series with a strong upward drift and no predictability at all
        makes any mostly-long rule earn a positive mean. Against a zero
        null that reads as a discovery; against the shift null, which
        carries the same drift, it must not.

        Averaged over several seeds rather than asserted on one: a single
        draw lands near the threshold often enough that a one-seed version
        of this test would be flaky, and picking the seed that passes
        would be seed-shopping.
        """
        p_values: list[float] = []
        long_means: list[float] = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            bars: list[UnderlyingBar] = []
            close = 100.0
            for i in range(2500):
                night = float(rng.normal(0.0008, 0.006))  # strong overnight drift
                open_ = close * math.exp(night)
                close = open_ * math.exp(float(rng.normal(0.0008, 0.010)))
                bars.append(_bar(i, open_=open_, close=close))
            result = evaluate_cell(
                bars,
                underlying_id="DRIFT",
                sub_hypothesis=SubHypothesis.NIGHT_NEXT_DAY,
                min_train=500,
                step=500,
                null_draws=500,
                seed=seed,
            )
            p_values.append(result.p_value)
            long_means.append(result.always_long_mean)

        # The drift is really there, so a zero null would have fired.
        assert min(long_means) > 0.0
        # And the shift null is not fooled by it.
        assert float(np.mean(p_values)) > 0.10

    def test_break_even_cost_is_the_rule_mean(self) -> None:
        result = evaluate_cell(
            _series(2000),
            underlying_id="SYNTH",
            sub_hypothesis=SubHypothesis.DAY_DAY,
            min_train=500,
            step=500,
            null_draws=100,
        )
        assert result.break_even_round_trip_cost == result.rule_mean

    def test_a_short_sample_is_marked_insufficient(self) -> None:
        result = evaluate_cell(
            _series(700),
            underlying_id="SYNTH",
            sub_hypothesis=SubHypothesis.DAY_DAY,
            min_train=500,
            step=100,
            null_draws=100,
        )
        assert result.n_oos < MIN_OOS_OBSERVATIONS
        assert not result.sufficient

    def test_results_are_reproducible(self) -> None:
        kwargs = {
            "underlying_id": "SYNTH",
            "sub_hypothesis": SubHypothesis.DAY_DAY,
            "min_train": 500,
            "step": 500,
            "null_draws": 200,
        }
        bars = _series(2000)
        a = evaluate_cell(bars, **kwargs)  # type: ignore[arg-type]
        b = evaluate_cell(bars, **kwargs)  # type: ignore[arg-type]
        assert a.rule_mean == b.rule_mean
        assert a.p_value == b.p_value

    def test_too_few_null_draws_is_refused(self) -> None:
        with pytest.raises(ValueError, match="null_draws must be >= 100"):
            evaluate_cell(
                _series(1200),
                underlying_id="SYNTH",
                sub_hypothesis=SubHypothesis.DAY_DAY,
                null_draws=10,
            )


class TestBenjaminiHochberg:
    def test_a_single_tiny_p_value_survives(self) -> None:
        assert benjamini_hochberg([0.001, 0.5, 0.9], alpha=0.10) == [True, False, False]

    def test_nothing_survives_when_all_are_large(self) -> None:
        assert benjamini_hochberg([0.3, 0.5, 0.9], alpha=0.10) == [False, False, False]

    def test_a_larger_family_makes_survival_harder(self) -> None:
        """Adding cells only ever tightens the threshold.

        This is why the two clean sub-hypotheses were added to the family
        rather than replacing the original four: the correction could only
        get stricter, never more permissive.
        """
        # Rank-1 thresholds: 0.10/16 = 0.00625, 0.10/24 = 0.004167.
        assert benjamini_hochberg([0.005] + [0.9] * 15, alpha=0.10)[0]
        assert not benjamini_hochberg([0.005] + [0.9] * 23, alpha=0.10)[0]

    def test_nan_p_values_never_survive(self) -> None:
        assert benjamini_hochberg([float("nan"), 0.001], alpha=0.10) == [False, True]

    def test_an_empty_family_is_empty(self) -> None:
        assert benjamini_hochberg([], alpha=0.10) == []

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
    def test_an_invalid_alpha_is_refused(self, alpha: float) -> None:
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
            benjamini_hochberg([0.01], alpha=alpha)


class TestFamilyVerdict:
    @staticmethod
    def _cell(sub: SubHypothesis, *, ci_low: float, n_oos: int = 1000) -> object:
        from turboedge.backtest.day_night_eval import CellResult

        return CellResult(
            underlying_id="X",
            sub_hypothesis=sub,
            n_oos=n_oos,
            rule_mean=0.001,
            always_long_mean=0.0,
            directional_accuracy=0.52,
            p_value=0.001,
            ci_low=ci_low,
            ci_high=0.002,
            folds=5,
        )

    def test_a_shared_price_winner_is_a_lookahead_suspect_not_a_candidate(self) -> None:
        """The whole point. This is what the real data produced."""
        family = FamilyResult(
            cells=(self._cell(SubHypothesis.NIGHT_DAY, ci_low=0.0005),),  # type: ignore[arg-type]
            survives=(True,),
            alpha=0.10,
        )
        assert family.verdict == "LOOKAHEAD_SUSPECT"
        assert math.isnan(family.best_break_even_cost)

    def test_a_clean_winner_is_a_candidate(self) -> None:
        family = FamilyResult(
            cells=(self._cell(SubHypothesis.NIGHT_NEXT_DAY, ci_low=0.0005),),  # type: ignore[arg-type]
            survives=(True,),
            alpha=0.10,
        )
        assert family.verdict == "CANDIDATE"
        assert family.best_break_even_cost == pytest.approx(0.001)

    def test_a_surviving_p_value_with_an_interval_through_zero_is_not_a_candidate(self) -> None:
        """A direction without a magnitude cannot be sized."""
        family = FamilyResult(
            cells=(self._cell(SubHypothesis.NIGHT_NEXT_DAY, ci_low=-0.0005),),  # type: ignore[arg-type]
            survives=(True,),
            alpha=0.10,
        )
        assert family.verdict == "IMPROVEMENT_NOT_SIGNIFICANT"

    def test_an_undersized_sample_reports_insufficient(self) -> None:
        family = FamilyResult(
            cells=(self._cell(SubHypothesis.NIGHT_NEXT_DAY, ci_low=0.001, n_oos=10),),  # type: ignore[arg-type]
            survives=(True,),
            alpha=0.10,
        )
        assert family.verdict == "INSUFFICIENT_SAMPLE"

    def test_an_empty_family_reports_insufficient(self) -> None:
        assert FamilyResult(cells=(), survives=(), alpha=0.10).verdict == "INSUFFICIENT_SAMPLE"


class TestEvaluateFamily:
    def test_every_sub_hypothesis_runs_on_every_underlying(self) -> None:
        bars = {"A": _series(1500, seed=1), "B": _series(1500, seed=2)}
        family = evaluate_family(bars, min_train=500, step=500, null_draws=100)
        assert len(family.cells) == 2 * len(SubHypothesis)

    def test_the_cell_order_does_not_depend_on_dict_order(self) -> None:
        """BH ranks by position, so a reordered dict must not change the verdict."""
        a = evaluate_family(
            {"A": _series(1500, seed=1), "B": _series(1500, seed=2)},
            min_train=500,
            step=500,
            null_draws=100,
        )
        b = evaluate_family(
            {"B": _series(1500, seed=2), "A": _series(1500, seed=1)},
            min_train=500,
            step=500,
            null_draws=100,
        )
        assert [c.underlying_id for c in a.cells] == [c.underlying_id for c in b.cells]
        assert [c.p_value for c in a.cells] == [c.p_value for c in b.cells]

    def test_an_underlying_with_no_history_is_skipped_not_fatal(self) -> None:
        bars = {"GOOD": _series(1500), "EMPTY": []}
        family = evaluate_family(bars, min_train=500, step=500, null_draws=100)
        assert {c.underlying_id for c in family.cells} == {"GOOD"}
