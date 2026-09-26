"""Tests for the fixed research catalog (Phase 2, §11).

The load-bearing property of this module is that it cannot manufacture
confidence: every catalog entry must be `PROPOSED`, the three genuinely
measurable inputs must stay `UNKNOWN` rather than defaulted, and no input
may claim `MEASURED` since nothing here reads repository state. The
degeneracy test guards against the other failure mode -- a catalog that
"validates" only because every entry was filled with the same placeholder
numbers, which would be as uninformative as an empty one.
"""

from __future__ import annotations

from turboedge.meta.catalog import (
    CATALOG,
    W12_MEASURED_OUTCOMES,
    catalog_by_id,
    catalog_for_family,
)
from turboedge.meta.research_opportunity import (
    EstimateBasis,
    InformationFamily,
    ResearchStatus,
)

_EXPECTED_IDS = {
    "RO-CBOE-VOL-STATE",
    "RO-EUREX-POSITIONING",
    "RO-EUWAX-SENTIMENT",
    "RO-CFTC-POSITIONING",
    "RO-RATES-CREDIT",
    "RO-MARKET-BREADTH",
    "RO-DISPERSION",
    "RO-EVENT-RISK",
    "RO-MAE-PREDICTION",
    "RO-MFE-PREDICTION",
    "RO-CROSS-ASSET-RESIDUAL-MOMENTUM",
    "RO-ISSUER-SPREAD-BEHAVIOUR",
    "RO-FINANCING-BEHAVIOUR",
    "RO-PRODUCT-SELECTION-EDGE",
}

# The three inputs another agent enriches from real repository state.
_MEASURABLE_LATER_FIELDS = (
    "current_uncertainty",
    "data_availability",
    "overlap_with_existing_research",
)

# Every other ranking input: must be DECLARED (never MEASURED, never UNKNOWN)
# in a module that reads no repository state.
_DECLARED_FIELDS = (
    "expected_information_gain",
    "expected_economic_value",
    "probability_of_resolving_uncertainty",
    "implementation_cost",
    "implementation_complexity",
    "estimated_sample_size",
    "leakage_risk",
)

_MIN_NOTE_LENGTH = 40

#: Repo-derived underlyings (configs/universe.yaml `enabled: true`) and the
#: horizon-string ladder (state/registry/failed_hypotheses.json), used to
#: check the catalog didn't invent its own vocabulary.
_REPO_UNDERLYINGS = {"DAX", "NDX", "EURUSD", "XAU"}
_REPO_HORIZONS = {"3d", "5d", "7d", "10d", "14d"}


def test_every_entry_is_valid_and_has_all_fourteen_ideas() -> None:
    """Every catalog entry already validated at import time (pydantic); this
    just asserts the exact set the user asked for is present, no more, no
    fewer."""
    assert len(CATALOG) == 14
    assert {o.hypothesis_id for o in CATALOG} == _EXPECTED_IDS


def test_hypothesis_ids_are_unique() -> None:
    ids = [o.hypothesis_id for o in CATALOG]
    assert len(ids) == len(set(ids))


def test_every_entry_stays_proposed() -> None:
    """The controller may prioritise, never approve. A catalog entry that
    ships in any other status would need an `approved_by` this static
    module never supplies -- and would silently smuggle a human decision
    into data that is supposed to be pre-decision."""
    for opportunity in CATALOG:
        assert opportunity.status is ResearchStatus.PROPOSED
        assert opportunity.approved_by is None


def test_no_entry_uses_measured() -> None:
    """This module reads no database, filesystem or network -- so nothing in
    it is entitled to claim MEASURED. A single MEASURED estimate anywhere
    would be exactly the laundered-guess failure the module docstring in
    research_opportunity.py names."""
    for opportunity in CATALOG:
        for field in (*_DECLARED_FIELDS, *_MEASURABLE_LATER_FIELDS):
            estimate = getattr(opportunity, field)
            assert estimate.basis is not EstimateBasis.MEASURED, (
                f"{opportunity.hypothesis_id}.{field} must not be MEASURED"
            )


def test_declared_fields_are_declared_with_substantive_notes() -> None:
    """A DECLARED estimate without a real reason cannot be argued with, which
    defeats the entire point of separating DECLARED from MEASURED. This
    would fail immediately against a catalog that used `Estimate.declared(x,
    "n/a")` or similar filler."""
    for opportunity in CATALOG:
        for field in _DECLARED_FIELDS:
            estimate = getattr(opportunity, field)
            assert estimate.basis is EstimateBasis.DECLARED, (
                f"{opportunity.hypothesis_id}.{field} must be DECLARED"
            )
            assert estimate.value is not None
            assert len(estimate.note.strip()) >= _MIN_NOTE_LENGTH, (
                f"{opportunity.hypothesis_id}.{field} note is too short to be a real justification"
            )


def test_the_three_measurable_fields_are_unknown_on_every_entry() -> None:
    """current_uncertainty, data_availability and overlap_with_existing_research
    are genuinely measurable from repository state -- this module must leave
    them UNKNOWN (with a note on what would measure them) rather than
    guessing, since it never reads that state itself."""
    for opportunity in CATALOG:
        for field in _MEASURABLE_LATER_FIELDS:
            estimate = getattr(opportunity, field)
            assert estimate.basis is EstimateBasis.UNKNOWN, (
                f"{opportunity.hypothesis_id}.{field} must be UNKNOWN, not {estimate.basis}"
            )
            assert estimate.value is None
            assert len(estimate.note.strip()) >= _MIN_NOTE_LENGTH


def test_implementation_cost_is_positive_engineer_days() -> None:
    for opportunity in CATALOG:
        assert opportunity.implementation_cost.value is not None
        assert opportunity.implementation_cost.value > 0


def test_bounded_fields_stay_in_unit_interval() -> None:
    unit_fields = (
        "expected_information_gain",
        "expected_economic_value",
        "probability_of_resolving_uncertainty",
        "implementation_complexity",
        "leakage_risk",
    )
    for opportunity in CATALOG:
        for field in unit_fields:
            value = getattr(opportunity, field).value
            assert value is not None
            assert 0.0 <= value <= 1.0, f"{opportunity.hypothesis_id}.{field} out of [0, 1]"


def test_sample_size_is_a_nonnegative_count() -> None:
    for opportunity in CATALOG:
        value = opportunity.estimated_sample_size.value
        assert value is not None
        assert value >= 0


def test_catalog_is_not_degenerate() -> None:
    """The mandatory anti-placeholder check: a catalog filled with the same
    number everywhere would pass every schema check above while conveying
    nothing. Both a cost axis and a sample-size axis (the two numbers most
    tempting to fake identically) must show real spread."""
    costs = {o.implementation_cost.value for o in CATALOG}
    assert len(costs) > 2, f"implementation_cost barely varies across entries: {costs}"

    sample_sizes = {o.estimated_sample_size.value for o in CATALOG}
    assert len(sample_sizes) > 2, f"estimated_sample_size barely varies: {sample_sizes}"

    gains = {o.expected_information_gain.value for o in CATALOG}
    assert len(gains) > 2, f"expected_information_gain barely varies across entries: {gains}"

    # Families must differ too -- a catalog that filed everything under one
    # family would defeat the redundancy-detection this taxonomy exists for.
    families = {o.information_family for o in CATALOG}
    assert len(families) >= 8


def test_catalog_by_id_round_trips() -> None:
    by_id = catalog_by_id()
    assert set(by_id) == _EXPECTED_IDS
    for opportunity in CATALOG:
        assert by_id[opportunity.hypothesis_id] is opportunity


def test_catalog_by_id_would_fail_on_a_duplicate_id() -> None:
    """catalog_by_id must not silently drop a collision -- if two entries ever
    shared an id, the dict comprehension would quietly keep only the last
    one. This test pins the real catalog's count as the guard: any future
    id collision shrinks len(by_id) below len(CATALOG), which this test
    would catch."""
    assert len(catalog_by_id()) == len(CATALOG)


def test_catalog_for_family_partitions_the_catalog() -> None:
    """Every entry must show up in exactly one family bucket, and the buckets
    together must reconstitute the whole catalog -- a filter that always
    returned [] or always returned everything would both pass a weaker
    test."""
    seen: set[str] = set()
    for family in InformationFamily:
        members = catalog_for_family(family)
        for opportunity in members:
            assert opportunity.information_family is family
            assert opportunity.hypothesis_id not in seen
            seen.add(opportunity.hypothesis_id)
    assert seen == _EXPECTED_IDS


def test_catalog_for_family_is_specific_not_a_passthrough() -> None:
    """Guards against a degenerate implementation that ignores `family` and
    just returns the whole catalog regardless of argument."""
    vol_surface = catalog_for_family(InformationFamily.VOLATILITY_SURFACE)
    assert [o.hypothesis_id for o in vol_surface] == ["RO-CBOE-VOL-STATE"]
    assert len(vol_surface) < len(CATALOG)


def test_underlyings_and_horizons_match_the_repo() -> None:
    for opportunity in CATALOG:
        assert opportunity.affected_underlyings, opportunity.hypothesis_id
        assert opportunity.affected_horizons, opportunity.hypothesis_id
        assert set(opportunity.affected_underlyings) <= _REPO_UNDERLYINGS, (
            f"{opportunity.hypothesis_id} names an underlying outside "
            f"{_REPO_UNDERLYINGS}: {opportunity.affected_underlyings}"
        )
        assert set(opportunity.affected_horizons) <= _REPO_HORIZONS, (
            f"{opportunity.hypothesis_id} names a horizon outside "
            f"{_REPO_HORIZONS}: {opportunity.affected_horizons}"
        )


def test_w12_measured_outcomes_keys_exist_in_catalog() -> None:
    catalog_ids = {o.hypothesis_id for o in CATALOG}
    assert W12_MEASURED_OUTCOMES
    for hypothesis_id in W12_MEASURED_OUTCOMES:
        assert hypothesis_id in catalog_ids


def test_w12_measured_outcomes_cover_exactly_the_four_researched_ideas() -> None:
    assert set(W12_MEASURED_OUTCOMES) == {
        "RO-CBOE-VOL-STATE",
        "RO-EUREX-POSITIONING",
        "RO-EUWAX-SENTIMENT",
        "RO-CFTC-POSITIONING",
    }


def test_w12_measured_outcomes_never_leave_the_catalog_entry_itself_non_proposed() -> None:
    """The mapping is meant to be applied by a later stage, not baked into
    this static catalog -- this pins that separation so a future edit can't
    quietly start setting `status=` on the CATALOG entries instead."""
    by_id = catalog_by_id()
    for hypothesis_id in W12_MEASURED_OUTCOMES:
        assert by_id[hypothesis_id].status is ResearchStatus.PROPOSED


def test_w12_measured_outcomes_distinguish_measured_from_untestable() -> None:
    """The four outcomes are not all in the same state, and the distinction
    sits where the lifecycle can actually carry it.

    Cboe and CFTC were measured and not promoted; Eurex was attempted and
    abandoned at Neff ~357 without producing anything to measure. Both are
    shelved-but-retestable, so both are DORMANT -- the seven states of §8 have
    no separate slot for "attempted, uninformative", and inventing one by
    parking Eurex in APPROVED would keep it in the actionable queue and
    present it as ready to run.

    So the status carries "can this be picked up again", while `trial_id`
    carries "did it ever produce a result" (pinned in the next test) and the
    note carries why. Euwax is the state that genuinely differs: blocked at
    the source, terminal.
    """
    statuses = {k: v.status for k, v in W12_MEASURED_OUTCOMES.items()}

    assert statuses["RO-CBOE-VOL-STATE"] == statuses["RO-CFTC-POSITIONING"]
    assert statuses["RO-EUWAX-SENTIMENT"] != statuses["RO-CBOE-VOL-STATE"]
    # Not all one status -- a lazy implementation assigning the same value to
    # every outcome still fails here.
    assert len(set(statuses.values())) >= 2
    # Nothing attempted-but-unmeasured may sit in a state the queue treats as
    # actionable; that is the whole reason Eurex is not APPROVED.
    assert statuses["RO-EUREX-POSITIONING"] is not ResearchStatus.APPROVED
    assert statuses["RO-EUREX-POSITIONING"] is not ResearchStatus.RUNNING


def test_w12_measured_outcomes_require_approval_and_a_note() -> None:
    for outcome in W12_MEASURED_OUTCOMES.values():
        assert outcome.approved_by.strip()
        assert len(outcome.note.strip()) >= _MIN_NOTE_LENGTH


def test_w12_measured_outcomes_trial_ids_present_only_when_actually_measured() -> None:
    """Cboe and CFTC were run to completion and have real trial ids; Eurex
    (abandoned) and Euwax (blocked) never produced a trial worth citing."""
    assert W12_MEASURED_OUTCOMES["RO-CBOE-VOL-STATE"].trial_id == "TR-2026Q3-abd750"
    assert W12_MEASURED_OUTCOMES["RO-CFTC-POSITIONING"].trial_id == "TR-2026Q3-31e266"
    assert W12_MEASURED_OUTCOMES["RO-EUREX-POSITIONING"].trial_id is None
    assert W12_MEASURED_OUTCOMES["RO-EUWAX-SENTIMENT"].trial_id is None
