"""Tests for `turboedge.meta.value_of_information` (§7).

Every test either checks arithmetic against a hand-computed number, or
checks that a specific honesty rule from the module's docstring actually
holds: unknown inputs are penalised and never treated as neutral, the
pattern-support bonus is bounded, the score never goes negative, and
`rank` is deterministic including its tie-break.
"""

from __future__ import annotations

import pytest

from turboedge.meta.research_opportunity import (
    Estimate,
    InformationFamily,
    ResearchOpportunity,
    ResearchPriority,
)
from turboedge.meta.value_of_information import (
    CURRENT_UNCERTAINTY_FLOOR,
    OVERLAP_WEIGHT,
    PATTERN_SUPPORT_BONUS_CAP,
    SMALL_SAMPLE_REFERENCE,
    UNKNOWN_INPUT_PENALTY,
    rank,
    score_opportunity,
    uncertainty_multiplier,
)

_UNUSED_OVERLAP = Estimate.declared(
    0.0, "not consumed by this scorer; overlap bookkeeping lives in research_queue"
)


def _opportunity(hypothesis_id: str = "H1", **overrides: Estimate) -> ResearchOpportunity:
    """A well-evidenced, cheap, high-value opportunity, every input known.

    Defaults are the hand-computed example used by
    `test_base_formula_hand_computed_example` and the anti-triviality test;
    individual estimates can be overridden per test.
    """
    fields = {
        "expected_information_gain": Estimate.measured(0.8, "held-out MI estimate"),
        "expected_economic_value": Estimate.declared(500.0, "typical position size x edge"),
        "probability_of_resolving_uncertainty": Estimate.declared(
            0.7, "similar studies resolved cleanly"
        ),
        "implementation_cost": Estimate.declared(10.0, "one afternoon, existing adapters"),
        "implementation_complexity": Estimate.declared(0.2, "reuses an existing feature"),
        "estimated_sample_size": Estimate.measured(150.0, "effective sample from pilot pull"),
        "current_uncertainty": Estimate.measured(0.9, "Phase 1 epistemic_uncertainty"),
        "data_availability": Estimate.measured(0.9, "source health check"),
        "leakage_risk": Estimate.declared(0.1, "feature already lagged correctly"),
        "overlap_with_existing_research": _UNUSED_OVERLAP,
    }
    fields.update(overrides)
    return ResearchOpportunity(
        hypothesis_id=hypothesis_id,
        description="test opportunity",
        information_family=InformationFamily.POSITIONING,
        **fields,
    )


def _score(
    opportunity: ResearchOpportunity,
    *,
    prior_failure_similarity: Estimate | None = None,
    family_redundancy: Estimate | None = None,
    pattern_support: Estimate | None = None,
) -> ResearchPriority:
    return score_opportunity(
        opportunity,
        prior_failure_similarity=prior_failure_similarity
        or Estimate.declared(0.0, "no similar failure on record"),
        family_redundancy=family_redundancy or Estimate.declared(0.1, "mostly novel family"),
        pattern_support=pattern_support or Estimate.declared(0.4, "moderate track record"),
    )


# --- base formula ------------------------------------------------------


def test_base_formula_hand_computed_example() -> None:
    """base = eig * eev * pru * uncertainty_multiplier(cu) / cost, by hand.

    eig=0.8, eev=500, pru=0.7, cu=0.9 -> um = 0.2 + 0.8*0.9 = 0.92
    base = 0.8 * 500 * 0.7 * 0.92 / 10 = 25.76

    Full score then applies every penalty and the bonus:
    leak(0.1)->0.9, sample(150/100 clipped)->1.0, redundancy(0.1)->0.9,
    similarity(0.0)->1.0, availability(0.9)->0.9, complexity(0.2)->0.8,
    bonus(pattern=0.4)->1.10
    score = 25.76 * 0.9 * 1.0 * 0.9 * 1.0 * 0.9 * 0.8 * 1.10 = 16.5255552
    """
    priority = _score(_opportunity())
    assert priority.base_score == pytest.approx(25.76)
    assert priority.score == pytest.approx(16.5255552)
    assert priority.unknown_inputs == []
    assert priority.evidence_completeness == pytest.approx(1.0)


def test_uncertainty_multiplier_floor_discounts_but_never_zeroes() -> None:
    assert uncertainty_multiplier(0.0) == pytest.approx(CURRENT_UNCERTAINTY_FLOOR)
    assert uncertainty_multiplier(0.0) > 0.0
    assert uncertainty_multiplier(1.0) == pytest.approx(1.0)


# --- individual penalties ------------------------------------------------


def test_leakage_risk_penalty_reduces_score() -> None:
    safe = _score(_opportunity(leakage_risk=Estimate.declared(0.0, "no leakage")))
    risky = _score(_opportunity(leakage_risk=Estimate.declared(0.9, "likely leaky")))
    assert risky.score < safe.score
    leak_factor = next(p for p in risky.penalties if p.name == "leakage_risk")
    assert leak_factor.value == pytest.approx(0.1)


def test_small_sample_penalty_ramps_linearly_to_reference() -> None:
    half = _score(
        _opportunity(estimated_sample_size=Estimate.measured(SMALL_SAMPLE_REFERENCE / 2, "pilot"))
    )
    full = _score(
        _opportunity(estimated_sample_size=Estimate.measured(SMALL_SAMPLE_REFERENCE, "pilot"))
    )
    sample_factor_half = next(p for p in half.penalties if p.name == "estimated_sample_size")
    sample_factor_full = next(p for p in full.penalties if p.name == "estimated_sample_size")
    assert sample_factor_half.value == pytest.approx(0.5)
    assert sample_factor_full.value == pytest.approx(1.0)
    assert half.score < full.score
    # Sample sizes far beyond the reference must not keep buying more credit.
    beyond = _score(
        _opportunity(
            estimated_sample_size=Estimate.measured(SMALL_SAMPLE_REFERENCE * 10, "large pull")
        )
    )
    assert beyond.score == pytest.approx(full.score)


def test_family_redundancy_penalty_reduces_score() -> None:
    novel = _score(_opportunity(), family_redundancy=Estimate.declared(0.0, "novel family"))
    redundant = _score(
        _opportunity(), family_redundancy=Estimate.declared(0.9, "near-duplicate of H0")
    )
    assert redundant.score < novel.score
    factor = next(p for p in redundant.penalties if p.name == "family_redundancy")
    assert factor.value == pytest.approx(0.1)


def test_prior_failure_similarity_penalty_reduces_score() -> None:
    clean = _score(
        _opportunity(), prior_failure_similarity=Estimate.declared(0.0, "nothing similar failed")
    )
    tainted = _score(
        _opportunity(),
        prior_failure_similarity=Estimate.declared(0.8, "close to failed hypothesis H0"),
    )
    assert tainted.score < clean.score
    factor = next(p for p in tainted.penalties if p.name == "prior_failure_similarity")
    assert factor.value == pytest.approx(0.2)
    reason = next(r for r in tainted.reasons if "prior failure similarity" in r)
    assert "0.80" in reason


def test_data_availability_used_directly_as_factor() -> None:
    priority = _score(
        _opportunity(data_availability=Estimate.measured(0.35, "half sources offline"))
    )
    factor = next(p for p in priority.penalties if p.name == "data_availability")
    assert factor.value == pytest.approx(0.35)


def test_implementation_complexity_penalty_reduces_score() -> None:
    simple = _score(_opportunity(implementation_complexity=Estimate.declared(0.0, "trivial")))
    complex_ = _score(
        _opportunity(implementation_complexity=Estimate.declared(0.9, "needs a new adapter"))
    )
    assert complex_.score < simple.score
    factor = next(p for p in complex_.penalties if p.name == "implementation_complexity")
    assert factor.value == pytest.approx(0.1)


# --- unknown-input handling ------------------------------------------------


def test_unknown_input_recorded_and_penalised() -> None:
    opp = _opportunity(leakage_risk=Estimate.unknown("no leakage audit run yet"))
    priority = _score(opp)
    assert priority.unknown_inputs == ["leakage_risk"]
    assert priority.evidence_completeness == pytest.approx(12 / 13)
    unknown_factor = next(p for p in priority.penalties if p.name == "leakage_risk")
    assert unknown_factor.value == pytest.approx(UNKNOWN_INPUT_PENALTY)
    assert unknown_factor.basis.value == "UNKNOWN"


def test_unknown_input_is_not_treated_as_a_neutral_one() -> None:
    """The load-bearing distinction: omitting an unknown factor from the
    arithmetic is, by itself, numerically identical to multiplying by
    1.0 -- so a genuinely known-and-neutral input (leakage_risk=0.0, which
    also yields a factor of 1.0) must score *higher* than the same
    opportunity with leakage_risk unknown. If unknown were silently
    treated as 1.0, these two scores would be equal.
    """
    known_neutral = _score(_opportunity(leakage_risk=Estimate.declared(0.0, "audited clean")))
    unknown = _score(_opportunity(leakage_risk=Estimate.unknown("no audit run yet")))
    assert unknown.score < known_neutral.score
    assert unknown.score == pytest.approx(known_neutral.score * UNKNOWN_INPUT_PENALTY)


def test_multiple_unknown_inputs_compound() -> None:
    opp = _opportunity(
        leakage_risk=Estimate.unknown("no audit"),
        implementation_complexity=Estimate.unknown("not scoped"),
    )
    priority = _score(opp)
    assert set(priority.unknown_inputs) == {"leakage_risk", "implementation_complexity"}
    assert priority.evidence_completeness == pytest.approx(11 / 13)
    # Omitting leakage_risk (factor 0.9) and implementation_complexity (factor
    # 0.8) from the arithmetic changes the score by more than the flat tax
    # alone -- the tax is on top of, not instead of, dropping those terms.
    fully_known = _score(_opportunity())
    without_the_two_factors = fully_known.score / (0.9 * 0.8)
    assert priority.score == pytest.approx(without_the_two_factors * UNKNOWN_INPUT_PENALTY**2)


def test_unknown_external_estimate_is_tracked_too() -> None:
    priority = _score(_opportunity(), pattern_support=Estimate.unknown("no track record yet"))
    assert "pattern_support" in priority.unknown_inputs
    assert priority.evidence_completeness == pytest.approx(12 / 13)
    # No bonus applied (skipped), plus the flat unknown-input tax.
    known = _score(_opportunity(), pattern_support=Estimate.declared(0.0, "no bonus, but known"))
    assert priority.score == pytest.approx(known.score * UNKNOWN_INPUT_PENALTY)


# --- pattern-support bonus is bounded ---------------------------------------


def test_pattern_support_bonus_is_capped() -> None:
    no_support = _score(_opportunity(), pattern_support=Estimate.declared(0.0, "no track record"))
    maxed = _score(_opportunity(), pattern_support=Estimate.declared(1.0, "strong track record"))
    assert maxed.score == pytest.approx(no_support.score * (1.0 + PATTERN_SUPPORT_BONUS_CAP))

    # A value above 1.0 (a mis-declared input) must not buy more than the cap.
    over_declared = _score(
        _opportunity(), pattern_support=Estimate.declared(5.0, "mis-declared, out of range")
    )
    assert over_declared.score == pytest.approx(maxed.score)


# --- never negative ----------------------------------------------------


def test_score_never_negative_even_with_a_negative_economic_value() -> None:
    opp = _opportunity(
        expected_economic_value=Estimate.declared(
            -50.0, "declared as net-negative: likely confirms a known null"
        )
    )
    priority = _score(opp)
    assert priority.base_score == 0.0
    assert priority.score == 0.0


# --- rank determinism --------------------------------------------------


def test_rank_sorts_by_score_descending() -> None:
    low = _score(_opportunity("LOW", implementation_cost=Estimate.declared(1000.0, "expensive")))
    high = _score(_opportunity("HIGH"))
    ordered = rank([low, high])
    assert [p.hypothesis_id for p in ordered] == ["HIGH", "LOW"]


def test_rank_tie_break_is_hypothesis_id_ascending() -> None:
    a = _score(_opportunity("B"))
    b = _score(_opportunity("A"))
    assert a.score == pytest.approx(b.score)  # identical inputs but for the id
    ordered = rank([a, b])
    assert [p.hypothesis_id for p in ordered] == ["A", "B"]
    # Order of the input sequence must not matter.
    ordered_reversed = rank([b, a])
    assert [p.hypothesis_id for p in ordered_reversed] == ["A", "B"]


# --- mandatory anti-triviality test --------------------------------------


def test_anti_triviality_well_evidenced_cheap_beats_poorly_evidenced_expensive() -> None:
    """A constant-returning `score_opportunity` would make good == bad here.

    `good` is cheap, well evidenced, and has a track record; `bad` is
    expensive, barely evidenced, redundant with a family that already
    failed, and has no track record. The ordering -- and the size of the
    gap -- is what a real scorer must produce and a degenerate one cannot.
    """
    good = _score(
        _opportunity(
            "GOOD",
            expected_information_gain=Estimate.measured(0.9, "clear MI signal"),
            expected_economic_value=Estimate.declared(800.0, "large potential edge"),
            probability_of_resolving_uncertainty=Estimate.declared(0.9, "clean design"),
            implementation_cost=Estimate.declared(5.0, "half a day"),
            implementation_complexity=Estimate.declared(0.1, "trivial reuse"),
            estimated_sample_size=Estimate.measured(400.0, "ample pilot data"),
            current_uncertainty=Estimate.measured(0.8, "materially uncertain today"),
            data_availability=Estimate.measured(0.95, "all sources healthy"),
            leakage_risk=Estimate.declared(0.05, "lagged correctly, reviewed"),
        ),
        prior_failure_similarity=Estimate.declared(0.0, "nothing similar failed"),
        family_redundancy=Estimate.declared(0.0, "genuinely novel angle"),
        pattern_support=Estimate.declared(0.8, "strong track record in this family"),
    )
    bad = _score(
        _opportunity(
            "BAD",
            expected_information_gain=Estimate.measured(0.1, "weak MI signal"),
            expected_economic_value=Estimate.declared(20.0, "small potential edge"),
            probability_of_resolving_uncertainty=Estimate.declared(0.2, "murky design"),
            implementation_cost=Estimate.declared(200.0, "needs a new adapter"),
            implementation_complexity=Estimate.declared(0.9, "novel infra required"),
            estimated_sample_size=Estimate.measured(10.0, "thin pilot data"),
            current_uncertainty=Estimate.measured(0.1, "already well understood"),
            data_availability=Estimate.measured(0.2, "most sources degraded"),
            leakage_risk=Estimate.declared(0.8, "hard to rule out leakage"),
        ),
        prior_failure_similarity=Estimate.declared(0.9, "near-duplicate of failed H0"),
        family_redundancy=Estimate.declared(0.9, "heavily redundant family"),
        pattern_support=Estimate.declared(0.0, "no track record"),
    )
    assert good.score > bad.score
    # Not just "higher" -- a wide, specific margin a constant scorer could
    # not produce (a constant scorer would give a ratio of exactly 1.0).
    assert good.score > bad.score * 50


# --- overlap with existing research (added by the lead) ----------------------


def test_overlap_with_existing_research_reduces_the_score() -> None:
    """Work already done must rank below work not yet done.

    Without this the queue would keep recommending finished research: four of
    the fourteen catalog questions were already measured in September 2026.
    """
    fresh = _score(_opportunity(overlap_with_existing_research=Estimate.measured(0.0)))
    done = _score(_opportunity(overlap_with_existing_research=Estimate.measured(1.0)))

    assert done.score < fresh.score
    assert done.score == pytest.approx(fresh.score * (1.0 - OVERLAP_WEIGHT))


def test_total_overlap_does_not_zero_the_score() -> None:
    """A fully-overlapping question must be unattractive, not invisible.

    §9 allows a retest with a new method, a new regime or a longer history, so
    an entry at total overlap has to remain rankable.
    """
    done = _score(_opportunity(overlap_with_existing_research=Estimate.measured(1.0)))
    assert done.score > 0.0


def test_overlap_penalty_is_distinct_from_family_redundancy() -> None:
    """The two must not be the same number under different names.

    Redundancy discounts a crowded queue; overlap discounts finished work. A
    single combined factor would make "already measured" indistinguishable
    from "several similar ideas are also queued".
    """
    overlapping = _score(
        _opportunity(overlap_with_existing_research=Estimate.measured(0.8)),
        family_redundancy=Estimate.measured(0.0),
    )
    crowded = _score(
        _opportunity(overlap_with_existing_research=Estimate.measured(0.0)),
        family_redundancy=Estimate.measured(0.8),
    )

    assert overlapping.score != crowded.score
    assert {p.name for p in overlapping.penalties} >= {
        "overlap_with_existing_research",
        "family_redundancy",
    }


def test_unknown_overlap_is_named_and_penalised_not_assumed_zero() -> None:
    priority = _score(
        _opportunity(overlap_with_existing_research=Estimate.unknown("no trial history read"))
    )
    known_zero = _score(_opportunity(overlap_with_existing_research=Estimate.measured(0.0)))

    assert "overlap_with_existing_research" in priority.unknown_inputs
    assert priority.score < known_zero.score
    assert priority.evidence_completeness < 1.0
