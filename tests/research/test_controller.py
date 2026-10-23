"""Tests for the autonomous research decision.

The controller is the one place the loop exercises judgement, so these
tests are mostly attempts to make it do something it must not: start a
confirmatory trial, screen a shelved family, proceed on a sample it has
not got, or run forever.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from turboedge.research.controller import (
    EXPLORATORY_SCREENS_PER_QUARTER,
    PERMITTED_EXPLORATORY_FAMILIES,
    ActionKind,
    Candidate,
    decide_next_action,
)

_NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def _candidate(**over: object) -> Candidate:
    defaults: dict[str, object] = dict(
        hypothesis_id="RO-DATA-ECB-ECB-M3",
        family="macro",
        priority=0.7,
        effective_sample=120.0,
        required_sample=48.0,
        data_ready=True,
        exploratory_done=False,
        has_exploratory_support=False,
    )
    defaults.update(over)
    return Candidate(**defaults)  # type: ignore[arg-type]


def _decide(**over: object):
    kwargs: dict[str, object] = dict(
        candidates=[_candidate()],
        exploratory_screens_this_quarter=0,
        confirmatory_trials_this_quarter=0,
        confirmatory_budget=6,
        now=_NOW,
    )
    kwargs.update(over)
    return decide_next_action(**kwargs)  # type: ignore[arg-type]


def test_the_best_ready_candidate_is_screened() -> None:
    action = _decide()

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.hypothesis_id == "RO-DATA-ECB-ECB-M3"
    assert "DEVELOPMENT" in action.reason


def test_the_loop_never_starts_a_confirmatory_trial() -> None:
    # The single most important boundary. A question with exploratory
    # support is exactly where a machine would be tempted to continue, and
    # exactly where the quarterly budget and the p-values start mattering.
    action = _decide(candidates=[_candidate(has_exploratory_support=True)])

    assert action.kind is ActionKind.AWAIT_CONFIRMATION_APPROVAL
    assert "named human" in action.reason
    assert "research approve" in action.reason


def test_confirmation_is_withheld_even_with_budget_to_spare() -> None:
    action = _decide(
        candidates=[_candidate(has_exploratory_support=True)],
        confirmatory_trials_this_quarter=0,
        confirmatory_budget=6,
    )

    assert action.kind is ActionKind.AWAIT_CONFIRMATION_APPROVAL
    assert action.evidence["confirmatory_budget_remaining"] == 6


def test_an_insufficient_effective_sample_means_waiting() -> None:
    action = _decide(candidates=[_candidate(effective_sample=12.0, required_sample=48.0)])

    assert action.kind is ActionKind.AWAIT_DATA
    assert "Effective, not nominal" in action.reason


def test_an_unmeasured_sample_is_not_treated_as_zero_or_as_enough() -> None:
    # An unknown sample size is a reason to wait. Defaulting it either way
    # would be the invented estimate the repository forbids.
    action = _decide(candidates=[_candidate(effective_sample=None)])

    assert action.kind is ActionKind.AWAIT_DATA
    assert "effective_sample=None" in action.reason


def test_unready_data_means_waiting_not_screening() -> None:
    action = _decide(candidates=[_candidate(data_ready=False)])

    assert action.kind is ActionKind.AWAIT_DATA


def test_a_repeatedly_failing_family_is_shelved_before_anything_else() -> None:
    action = _decide(
        candidates=[_candidate()],
        failed_family_counts={"macro": 3},
    )

    assert action.kind is ActionKind.SHELVE_FAMILY
    assert action.family == "macro"


def test_shelving_takes_precedence_over_a_quarter_with_room() -> None:
    action = _decide(
        candidates=[_candidate()],
        failed_family_counts={"macro": 5},
        exploratory_screens_this_quarter=0,
    )

    assert action.kind is ActionKind.SHELVE_FAMILY


def test_an_already_shelved_family_is_not_shelved_again() -> None:
    # Otherwise the loop would return SHELVE_FAMILY forever and never look
    # at anything else.
    action = _decide(
        candidates=[_candidate(family="sentiment")],
        failed_family_counts={"macro": 9},
        blocked_families=["macro"],
    )

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.family == "sentiment"


def test_the_exploratory_quarter_has_a_hard_ceiling() -> None:
    action = _decide(exploratory_screens_this_quarter=EXPLORATORY_SCREENS_PER_QUARTER)

    assert action.kind is ActionKind.NO_ACTION
    assert "multiple testing" in action.reason


def test_a_family_outside_the_permitted_set_is_never_screened() -> None:
    # The loop may rank and propose anything; it may only *start* a screen
    # in a family governance has listed.
    action = _decide(candidates=[_candidate(family="not_a_real_family")])

    assert action.kind is ActionKind.NO_ACTION
    assert "permitted exploratory family" in action.reason


def test_everything_screened_and_nothing_supported_stops() -> None:
    action = _decide(candidates=[_candidate(exploratory_done=True)])

    assert action.kind is ActionKind.NO_ACTION
    assert "already been screened" in action.reason


def test_no_candidates_at_all_is_a_clean_no_action() -> None:
    action = _decide(candidates=[])

    assert action.kind is ActionKind.NO_ACTION


def test_an_unmeasured_priority_sorts_last_rather_than_as_zero() -> None:
    measured = _candidate(hypothesis_id="MEASURED", priority=0.1)
    unmeasured = _candidate(hypothesis_id="UNMEASURED", priority=None)

    action = _decide(candidates=[unmeasured, measured])

    assert action.hypothesis_id == "MEASURED"


def test_the_decision_is_deterministic_for_equal_priorities() -> None:
    # Two candidates with the same priority must not alternate between
    # runs: an autonomous loop whose choice depends on dict ordering is not
    # reproducible, and reproducibility is what makes its log worth keeping.
    a = _candidate(hypothesis_id="AAA", priority=0.5)
    b = _candidate(hypothesis_id="BBB", priority=0.5)

    first = _decide(candidates=[a, b])
    second = _decide(candidates=[b, a])

    assert first.hypothesis_id == second.hypothesis_id == "AAA"


def test_promotion_is_recorded_as_locked_in_every_action() -> None:
    # Not an absent field. The reason promotion cannot happen has to be
    # visible in the log of what the loop decided, or a future reader will
    # assume it simply never came up.
    for action in (
        _decide(),
        _decide(candidates=[_candidate(has_exploratory_support=True)]),
        _decide(candidates=[]),
    ):
        assert action.evidence["promotion_unlocked"] is False


def test_every_action_carries_the_inputs_it_was_decided_on() -> None:
    action = _decide()

    for key in (
        "candidates",
        "exploratory_screens_this_quarter",
        "exploratory_cap",
        "confirmatory_trials_this_quarter",
        "confirmatory_budget",
        "decided_at",
    ):
        assert key in action.evidence, key


@pytest.mark.parametrize("family", sorted(PERMITTED_EXPLORATORY_FAMILIES))
def test_each_permitted_family_can_actually_be_screened(family: str) -> None:
    action = _decide(candidates=[_candidate(family=family)])

    assert action.kind is ActionKind.START_EXPLORATORY
