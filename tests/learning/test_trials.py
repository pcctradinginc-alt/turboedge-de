from __future__ import annotations

from datetime import UTC, datetime

import pytest

from turboedge.learning.trials import (
    TrialBudgetExceeded,
    TrialsConfig,
    effective_number_of_trials,
    new_trial_id,
)


def test_new_trial_id_has_expected_format(store) -> None:
    as_of = datetime(2026, 9, 10, tzinfo=UTC)
    trial_id = new_trial_id(store, "feature", "Add EWMA volatility feature", as_of=as_of)
    assert trial_id.startswith("TR-2026Q3-")
    assert len(trial_id) == len("TR-2026Q3-") + 6


def test_new_trial_id_persists_and_is_retrievable(store) -> None:
    as_of = datetime(2026, 9, 10, tzinfo=UTC)
    trial_id = new_trial_id(store, "feature", "Add EWMA volatility feature", as_of=as_of)
    persisted = store.get_research_trial(trial_id)
    assert persisted is not None
    assert persisted.kind == "feature"
    assert persisted.description == "Add EWMA volatility feature"
    assert persisted.quarter == "2026Q3"


def test_quarterly_budget_enforced(store) -> None:
    as_of = datetime(2026, 9, 10, tzinfo=UTC)
    cfg = TrialsConfig(quarterly_budget=2)
    new_trial_id(store, "feature", "one", config=cfg, as_of=as_of)
    new_trial_id(store, "feature", "two", config=cfg, as_of=as_of)
    with pytest.raises(TrialBudgetExceeded):
        new_trial_id(store, "feature", "three", config=cfg, as_of=as_of)


def test_quarterly_budget_force_override_still_creates_trial(store) -> None:
    as_of = datetime(2026, 9, 10, tzinfo=UTC)
    cfg = TrialsConfig(quarterly_budget=1)
    new_trial_id(store, "feature", "one", config=cfg, as_of=as_of)
    trial_id = new_trial_id(store, "feature", "two", config=cfg, as_of=as_of, force=True)
    assert store.get_research_trial(trial_id) is not None
    assert effective_number_of_trials(store, "2026Q3") == 2


def test_budget_is_scoped_per_quarter(store) -> None:
    cfg = TrialsConfig(quarterly_budget=1)
    q3 = datetime(2026, 9, 10, tzinfo=UTC)
    q4 = datetime(2026, 10, 10, tzinfo=UTC)
    new_trial_id(store, "feature", "q3-one", config=cfg, as_of=q3)
    # A new quarter has a fresh budget.
    new_trial_id(store, "feature", "q4-one", config=cfg, as_of=q4)
    assert effective_number_of_trials(store, "2026Q3") == 1
    assert effective_number_of_trials(store, "2026Q4") == 1
    assert effective_number_of_trials(store) == 2


def test_effective_number_of_trials_counts_all_by_default(store) -> None:
    as_of = datetime(2026, 9, 10, tzinfo=UTC)
    cfg = TrialsConfig(quarterly_budget=6)
    for i in range(3):
        new_trial_id(store, "feature", f"trial-{i}", config=cfg, as_of=as_of)
    assert effective_number_of_trials(store) == 3
