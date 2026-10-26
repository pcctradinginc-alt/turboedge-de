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


def test_a_blocked_top_question_does_not_starve_the_loop() -> None:
    """The defect run 40235422444 exposed, pinned.

    Until 2026-10-25 the controller took the top-ranked candidate and
    returned AWAIT_DATA if its data were not ready, never looking further
    down the ranking. In production the top question waited on matured
    forward labels, of which there were none, so the loop sat idle
    indefinitely while a lower-ranked question with an approved harness
    and ready data was available the whole time.
    """
    action = _decide(
        candidates=[
            _candidate(hypothesis_id="RO-BLOCKED", priority=0.9, data_ready=False),
            _candidate(hypothesis_id="RO-RUNNABLE", priority=0.2, data_ready=True),
        ]
    )

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.hypothesis_id == "RO-RUNNABLE"
    # And the question it passed over is named, not silently dropped.
    skipped = action.evidence["skipped_for_data"]
    assert [s["hypothesis_id"] for s in skipped] == ["RO-BLOCKED"]
    assert "passed over for want of data" in action.reason


def test_priority_still_decides_among_questions_that_can_run() -> None:
    """Skipping a blocked question must not become ignoring the ranking."""
    action = _decide(
        candidates=[
            _candidate(hypothesis_id="RO-LOWER", priority=0.2),
            _candidate(hypothesis_id="RO-HIGHER", priority=0.9),
        ]
    )

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.hypothesis_id == "RO-HIGHER"
    assert action.evidence["skipped_for_data"] == []


def test_an_undersized_sample_is_skipped_like_missing_data() -> None:
    action = _decide(
        candidates=[
            _candidate(hypothesis_id="RO-THIN", priority=0.9, effective_sample=3.0),
            _candidate(hypothesis_id="RO-RUNNABLE", priority=0.2),
        ]
    )

    assert action.hypothesis_id == "RO-RUNNABLE"
    assert action.evidence["skipped_for_data"][0]["reason"] == (
        "effective_sample_below_requirement"
    )


def test_when_nothing_can_run_the_wait_names_the_best_question_and_the_rest() -> None:
    action = _decide(
        candidates=[
            _candidate(hypothesis_id="RO-BEST", priority=0.9, data_ready=False),
            _candidate(hypothesis_id="RO-ALSO", priority=0.2, effective_sample=1.0),
        ]
    )

    assert action.kind is ActionKind.AWAIT_DATA
    assert action.hypothesis_id == "RO-BEST"
    assert "The other 1 screenable candidate(s) are blocked too." in action.reason
    assert len(action.evidence["skipped_for_data"]) == 2


def test_a_skipped_question_explains_itself_in_the_harness_words() -> None:
    """Run 40385586039 reported thirteen blocked candidates and no reason.

    A blocked question whose cause is invisible cannot be acted on: the
    operator cannot tell a young archive from a filter that drops
    everything. The harness counted its own data, so its sentence is the
    one that belongs in the record.
    """
    action = _decide(
        candidates=[
            _candidate(
                hypothesis_id="RO-BLOCKED",
                priority=0.9,
                data_ready=False,
                readiness_detail="12 usable cell(s) covering 300 quote(s); needs 30",
            ),
            _candidate(hypothesis_id="RO-RUNNABLE", priority=0.2),
        ]
    )

    assert action.kind is ActionKind.START_EXPLORATORY
    skipped = action.evidence["skipped_for_data"]
    assert skipped[0]["detail"] == "12 usable cell(s) covering 300 quote(s); needs 30"


def test_a_question_without_a_harness_detail_records_none_rather_than_a_guess() -> None:
    action = _decide(candidates=[_candidate(data_ready=False)])

    assert action.kind is ActionKind.AWAIT_DATA
    assert action.evidence["skipped_for_data"][0]["detail"] is None


# ---------------------------------------------------------------------
# The operator's decision of 2026-10-26: product_selection admitted to
# autonomous exploration, under conditions. These tests are the
# conditions.
# ---------------------------------------------------------------------


def test_product_selection_is_admissible_for_autonomous_exploration() -> None:
    """The authorisation itself. Granted 2026-10-26."""
    assert "product_selection" in PERMITTED_EXPLORATORY_FAMILIES


def test_the_controller_can_now_select_a02_on_its_own() -> None:
    """Before the authorisation this question was not admissible at all.

    Run 40387338754 showed the consequence: RO-LOTTERY-PRICING appeared in
    the queue ranking and never in the skip list, because the family was
    excluded before ranking.
    """
    action = _decide(
        candidates=[
            _candidate(
                hypothesis_id="RO-LOTTERY-PRICING",
                family="product_selection",
                priority=0.5,
            )
        ]
    )

    assert action.kind is ActionKind.START_EXPLORATORY
    assert action.hypothesis_id == "RO-LOTTERY-PRICING"


def test_a_screened_question_is_not_repeated_without_new_grounds() -> None:
    """The second condition of the authorisation.

    `exploratory_done` is set from the recorded screens, and a question
    that has one is not re-run on the same data. Re-screening until
    something looks significant is multiple testing under another name.
    """
    action = _decide(
        candidates=[
            _candidate(
                hypothesis_id="RO-LOTTERY-PRICING",
                family="product_selection",
                exploratory_done=True,
            )
        ]
    )

    assert action.kind is ActionKind.NO_ACTION
    assert "already been screened" in action.reason


def test_a_supported_product_selection_screen_still_stops_at_the_human() -> None:
    """The authorisation grants no confirmatory trial.

    A positive COST_EDGE screen is exactly where a machine would be
    tempted to continue, and exactly where the quarterly budget and the
    p-values start to matter.
    """
    action = _decide(
        candidates=[
            _candidate(
                hypothesis_id="RO-LOTTERY-PRICING",
                family="product_selection",
                has_exploratory_support=True,
            )
        ]
    )

    assert action.kind is ActionKind.AWAIT_CONFIRMATION_APPROVAL
    assert "named human" in action.reason
    assert "research approve" in action.reason


def test_the_exploratory_cap_still_bounds_product_selection_screens() -> None:
    """The authorisation did not widen the research budget."""
    action = _decide(
        candidates=[_candidate(family="product_selection")],
        exploratory_screens_this_quarter=EXPLORATORY_SCREENS_PER_QUARTER,
    )

    assert action.kind is ActionKind.NO_ACTION
    assert "multiple testing under another name" in action.reason


def test_promotion_stays_locked_for_an_admitted_family() -> None:
    action = _decide(
        candidates=[_candidate(family="product_selection", has_exploratory_support=True)]
    )
    assert action.evidence["promotion_unlocked"] is False
