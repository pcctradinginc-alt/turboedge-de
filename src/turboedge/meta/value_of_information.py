"""Expected-Value-of-Information ranking for open research questions (§7).

`research_opportunity.py` explains why almost every quantity in a
`ResearchOpportunity` is a judgement rather than a measurement. This module
turns those judgements into a ranking, on the user's own sketch::

    research_priority = expected_information_gain * economic_relevance
                        * probability_of_resolving_uncertainty
                        / implementation_cost

with penalties for leakage risk, tiny effective sample, redundant feature
family, an already-failed similar hypothesis, and poor data quality, plus a
bounded bonus for a family with a track record (§10).

This is a heuristic, not a model: no fitting, no RL, nothing tuned against
an outcome, because nothing in this repository has forward data to tune
against yet. Every constant is a named, commented, honestly-arbitrary
default (see each constant below).

The one rule that makes the result trustworthy rather than merely
plausible: an `Estimate` with ``basis is UNKNOWN`` is never treated as
though it were a known, neutral value. Doing so would let the ranking
launder an absence of evidence into an ordinary-looking number -- the exact
failure `research_opportunity.py`'s docstring names. Concretely, for every
one of the twelve estimates this module consumes:

1. its field name is appended to ``ResearchPriority.unknown_inputs``;
2. the term it would have contributed is *omitted* from the arithmetic
   entirely -- not replaced by 1.0, not replaced by any other guess;
3. the final score is additionally multiplied by ``UNKNOWN_INPUT_PENALTY``
   once for that field, on top of the omission;
4. ``evidence_completeness`` records known/total across all twelve.

Step 3 is what makes "unknown" different from "known and happens to be
neutral": omitting a factor from a product is, by itself, arithmetically
identical to multiplying by 1.0 -- so omission alone would not distinguish
"we don't know" from "we know it doesn't matter". The flat penalty is the
part that keeps the two apart (see the module's test suite for a case
where this is checked directly).

Every number that ends up in `factors`, `penalties` or `reasons` is a
number this module actually computed -- never a narrative gloss on an
input it did not have.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from turboedge.meta.research_opportunity import (
    Estimate,
    EstimateBasis,
    PriorityFactor,
    ResearchOpportunity,
    ResearchPriority,
)

# --- constants -------------------------------------------------------------
# Every constant below is a set default, chosen for a defensible reason
# stated in its comment. None is tuned against a measured outcome: per
# CLAUDE.md / GOVERNANCE.md, this repository has no forward research
# results to tune against, and tuning a priority heuristic on results would
# be exactly the parameter fishing the project's governance forbids.

#: Applied once per unknown input, multiplicatively, on top of omitting
#: that input's term from the arithmetic (see module docstring point 3).
#: 0.5 mirrors `trust.MISSING_FACTOR_PENALTY`: two unknown inputs already
#: leave the score at a quarter of its otherwise-computed value, which is
#: the honest position when most of a research catalog is judgement calls.
UNKNOWN_INPUT_PENALTY = 0.5

#: Floor for `implementation_cost` in the denominator. A declared cost of
#: exactly 0 is a data-entry artifact, not a claim that research is free;
#: without a floor it would send the score to infinity. Set well below any
#: plausible declared cost so it only ever bites on that artifact.
MIN_COST = 0.01

#: Floor for `uncertainty_multiplier` (see below). A question the system
#: is already confident about is worth little to re-research, but a
#: confident answer can still be wrong, so it is discounted, not zeroed.
CURRENT_UNCERTAINTY_FLOOR = 0.2

#: Reference effective sample size for the "tiny sample" ramp. Taken
#: directly from GOVERNANCE.md §2's own promotion floor ("Effective sample
#: >= 100 distinct outcomes") rather than invented here: a study that
#: could not even clear the bar this repository already requires for
#: promotion is not one whose result would be interpretable.
SMALL_SAMPLE_REFERENCE = 100.0

#: Cap on the `pattern_support` bonus, as a fraction of the score it can
#: add (i.e. a bonus factor in [1.0, 1.25]). Deliberately small and
#: additive-bounded: §10 asks to reinforce positive research memory
#: without automatically overweighting old winners ("keine automatische
#: Überbewertung alter Gewinner"), and a track record should nudge a
#: ranking, not dominate one built from this opportunity's own estimates.
PATTERN_SUPPORT_BONUS_CAP = 0.25

#: How hard `prior_failure_similarity` bites. Below 1.0 for the same reason
#: as OVERLAP_WEIGHT, and it matters more here: similarity is measured
#: lexically, so a perfect 1.0 can mean two genuinely different questions that
#: happen to share vocabulary -- "cross-asset residual momentum" scores 1.00
#: against the failed "cross_asset_leadlag" on tokens alone. At weight 1.0
#: that would zero the score outright and make the entry invisible, letting a
#: string match silently close off a line of research. It should make it
#: unattractive instead, and leave the §9 retest exceptions reachable.
PRIOR_FAILURE_WEIGHT = 0.9

#: How hard `overlap_with_existing_research` bites. Below 1.0 on purpose:
#: total overlap should push a question far down the queue, but not to a
#: score of exactly zero, because a re-examination with a new method or a
#: longer history is sometimes the right call (§9's retest exceptions) and a
#: zero would make that entry invisible rather than merely unattractive.
OVERLAP_WEIGHT = 0.9

#: Total number of `Estimate` inputs this module consumes, for
#: `evidence_completeness`. Ten live on `ResearchOpportunity` itself, plus
#: the three passed in explicitly.
_TOTAL_TRACKED_INPUTS = 13


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def uncertainty_multiplier(current_uncertainty: float) -> float:
    """Scale the base score by how little the system already knows.

    `current_uncertainty` is the one field that links this module back to
    Phase 1's measured uncertainty axes (`meta.uncertainty`, all in
    [0, 1], higher meaning "less known"). A question the system already
    has a confident answer to is worth little to research further, so
    this multiplier shrinks toward `CURRENT_UNCERTAINTY_FLOOR` as
    `current_uncertainty` falls toward 0 -- but never to 0, because a
    confident answer can still be wrong.
    """
    return CURRENT_UNCERTAINTY_FLOOR + (1.0 - CURRENT_UNCERTAINTY_FLOOR) * _clip01(
        current_uncertainty
    )


def _small_sample_factor(estimated_sample_size: float) -> float:
    """Ramp from 0 up to 1 as the effective sample approaches the reference.

    A tiny effective sample makes any result uninterpretable regardless of
    how large the other factors are, so this ramps linearly rather than
    saturating quickly -- a sample at half the reference is worth roughly
    half credit, not full credit.
    """
    if estimated_sample_size <= 0:
        return 0.0
    return _clip01(estimated_sample_size / SMALL_SAMPLE_REFERENCE)


def _factor(
    name: str,
    estimate: Estimate,
    transform: Callable[[float], float],
) -> tuple[float | None, PriorityFactor]:
    """One named, transformed contribution from one `Estimate`.

    Returns `(None, factor)` when the estimate is unknown: `None` tells the
    caller to omit this term from the arithmetic entirely (module
    docstring point 2), while the returned `PriorityFactor` still records
    what actually happened to the score for this field -- the flat
    `UNKNOWN_INPUT_PENALTY`, not a fabricated stand-in for the missing
    value.
    """
    if estimate.is_known:
        assert estimate.value is not None  # guaranteed by Estimate's own validator
        value = transform(float(estimate.value))
        return value, PriorityFactor(
            name=name, value=float(value), basis=estimate.basis, note=estimate.note
        )
    return None, PriorityFactor(
        name=name,
        value=UNKNOWN_INPUT_PENALTY,
        basis=EstimateBasis.UNKNOWN,
        note=estimate.note,
    )


def score_opportunity(
    opportunity: ResearchOpportunity,
    *,
    prior_failure_similarity: Estimate,
    family_redundancy: Estimate,
    pattern_support: Estimate,
) -> ResearchPriority:
    """Score one research question against the twelve estimates it rests on.

    `prior_failure_similarity`, `family_redundancy` and `pattern_support`
    are passed in rather than computed here so this module stays a pure
    function of its inputs -- computing them would require importing
    `research_memory`/`research_queue`/the catalog, which would let this
    module's output depend on repository state it cannot see or test
    against directly.
    """
    unknown_inputs: list[str] = []
    known_count = 0

    def track(estimate: Estimate) -> None:
        nonlocal known_count
        if estimate.is_known:
            known_count += 1

    o = opportunity
    for est, name in (
        (o.expected_information_gain, "expected_information_gain"),
        (o.expected_economic_value, "expected_economic_value"),
        (o.probability_of_resolving_uncertainty, "probability_of_resolving_uncertainty"),
        (o.implementation_cost, "implementation_cost"),
        (o.current_uncertainty, "current_uncertainty"),
        (o.leakage_risk, "leakage_risk"),
        (o.estimated_sample_size, "estimated_sample_size"),
        (o.data_availability, "data_availability"),
        (o.implementation_complexity, "implementation_complexity"),
        (o.overlap_with_existing_research, "overlap_with_existing_research"),
        (prior_failure_similarity, "prior_failure_similarity"),
        (family_redundancy, "family_redundancy"),
        (pattern_support, "pattern_support"),
    ):
        track(est)
        if not est.is_known:
            unknown_inputs.append(name)

    factors: list[PriorityFactor] = []
    penalties: list[PriorityFactor] = []
    reasons: list[str] = []

    # --- base score: the opportunity's own estimates -----------------------
    eig, eig_f = _factor("expected_information_gain", o.expected_information_gain, lambda v: v)
    factors.append(eig_f)
    eev, eev_f = _factor("expected_economic_value", o.expected_economic_value, lambda v: v)
    factors.append(eev_f)
    pru, pru_f = _factor(
        "probability_of_resolving_uncertainty",
        o.probability_of_resolving_uncertainty,
        _clip01,
    )
    factors.append(pru_f)
    um, um_f = _factor("current_uncertainty", o.current_uncertainty, uncertainty_multiplier)
    factors.append(um_f)

    base_score = 1.0
    base_terms: list[str] = []
    for value, label in (
        (eig, f"information gain {eig:.4f}" if eig is not None else None),
        (eev, f"economic value {eev:.4f}" if eev is not None else None),
        (pru, f"resolution probability {pru:.2f}" if pru is not None else None),
        (um, f"uncertainty multiplier {um:.2f}" if um is not None else None),
    ):
        if value is not None:
            base_score *= value
            assert label is not None
            base_terms.append(label)

    cost, cost_f = _factor("implementation_cost", o.implementation_cost, lambda v: max(v, MIN_COST))
    factors.append(cost_f)
    if cost is not None:
        base_score /= cost
        base_terms.append(f"cost {cost:.4f}")

    base_score = max(base_score, 0.0)
    reasons.append(
        f"base score {base_score:.6f} from known base factors: " + ", ".join(base_terms)
        if base_terms
        else f"base score {base_score:.6f}: every base factor was unknown"
    )

    # --- multiplicative penalties, each in [0, 1] ---------------------------
    score = base_score

    leak, leak_f = _factor("leakage_risk", o.leakage_risk, lambda v: _clip01(1.0 - v))
    penalties.append(leak_f)
    if leak is not None:
        score *= leak
        reasons.append(
            f"leakage risk {o.leakage_risk.value:.2f} leaves a penalty factor of {leak:.2f}"
        )

    sample, sample_f = _factor(
        "estimated_sample_size", o.estimated_sample_size, _small_sample_factor
    )
    penalties.append(sample_f)
    if sample is not None:
        score *= sample
        reasons.append(
            f"estimated sample size {o.estimated_sample_size.value:.0f} against a reference "
            f"of {SMALL_SAMPLE_REFERENCE:.0f} gives a small-sample factor of {sample:.2f}"
        )

    redundancy, redundancy_f = _factor(
        "family_redundancy", family_redundancy, lambda v: _clip01(1.0 - v)
    )
    penalties.append(redundancy_f)
    if redundancy is not None:
        score *= redundancy
        reasons.append(
            f"family redundancy {family_redundancy.value:.2f} leaves a penalty factor of "
            f"{redundancy:.2f}"
        )

    # Distinct from `family_redundancy` above, and the distinction is load
    # bearing rather than pedantic. Redundancy asks how crowded this family is
    # with other *queued, unrun* work, so it discounts marginal value.
    # `overlap_with_existing_research` asks how much of this question has
    # *already been investigated* -- which is what stops the queue from
    # ranking work that is finished. In September 2026 four of the fourteen
    # catalog entries were in exactly that position.
    overlap, overlap_f = _factor(
        "overlap_with_existing_research",
        o.overlap_with_existing_research,
        lambda v: _clip01(1.0 - OVERLAP_WEIGHT * _clip01(v)),
    )
    penalties.append(overlap_f)
    if overlap is not None:
        score *= overlap
        assert o.overlap_with_existing_research.value is not None
        reasons.append(
            f"overlap with existing research "
            f"{o.overlap_with_existing_research.value:.2f} leaves a factor of {overlap:.2f}"
        )

    similarity, similarity_f = _factor(
        "prior_failure_similarity",
        prior_failure_similarity,
        lambda v: _clip01(1.0 - PRIOR_FAILURE_WEIGHT * _clip01(v)),
    )
    penalties.append(similarity_f)
    if similarity is not None:
        score *= similarity
        assert prior_failure_similarity.value is not None
        reasons.append(
            f"prior failure similarity {prior_failure_similarity.value:.2f} for a question "
            "in this family"
        )

    availability, availability_f = _factor("data_availability", o.data_availability, _clip01)
    penalties.append(availability_f)
    if availability is not None:
        score *= availability
        reasons.append(f"data availability {availability:.2f} used directly as a quality factor")

    complexity, complexity_f = _factor(
        "implementation_complexity", o.implementation_complexity, lambda v: _clip01(1.0 - v)
    )
    penalties.append(complexity_f)
    if complexity is not None:
        score *= complexity
        reasons.append(
            f"implementation complexity {o.implementation_complexity.value:.2f} leaves a "
            f"penalty factor of {complexity:.2f}"
        )

    # --- bounded bonus from positive research memory (§10) ------------------
    support, support_f = _factor(
        "pattern_support",
        pattern_support,
        lambda v: 1.0 + _clip01(v) * PATTERN_SUPPORT_BONUS_CAP,
    )
    factors.append(support_f)
    if support is not None:
        score *= support
        assert pattern_support.value is not None
        reasons.append(
            f"pattern support {pattern_support.value:.2f} adds a bonus factor of "
            f"{support:.3f} (capped at {1.0 + PATTERN_SUPPORT_BONUS_CAP:.2f})"
        )

    # --- unknown-input tax: once per unknown, on top of the omissions above -
    if unknown_inputs:
        tax = UNKNOWN_INPUT_PENALTY ** len(unknown_inputs)
        score *= tax
        reasons.append(
            f"{len(unknown_inputs)} unknown input(s) {sorted(unknown_inputs)} apply a combined "
            f"penalty of {tax:.4f} ({UNKNOWN_INPUT_PENALTY:.2f} each, on top of being omitted "
            "from the arithmetic above)"
        )

    score = max(score, 0.0)
    evidence_completeness = known_count / _TOTAL_TRACKED_INPUTS

    return ResearchPriority(
        hypothesis_id=o.hypothesis_id,
        score=score,
        base_score=base_score,
        factors=factors,
        penalties=penalties,
        unknown_inputs=sorted(unknown_inputs),
        evidence_completeness=evidence_completeness,
        reasons=reasons,
    )


def rank(priorities: Sequence[ResearchPriority]) -> list[ResearchPriority]:
    """Sort by score descending, tie-broken by `hypothesis_id` ascending.

    The tie-break exists so that two opportunities scored identically
    (most commonly: both entirely unscored, or both hitting the same
    unknown-input floor) still produce a deterministic order -- an
    unstable ranking would make the research queue's order non-
    reproducible, which `research_opportunity`'s reproducibility rule
    (mirroring Master Spec rule 33) exists to prevent.
    """
    return sorted(priorities, key=lambda p: (-p.score, p.hypothesis_id))


__all__ = [
    "CURRENT_UNCERTAINTY_FLOOR",
    "MIN_COST",
    "PATTERN_SUPPORT_BONUS_CAP",
    "SMALL_SAMPLE_REFERENCE",
    "UNKNOWN_INPUT_PENALTY",
    "rank",
    "score_opportunity",
    "uncertainty_multiplier",
]
