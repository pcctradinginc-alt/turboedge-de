from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np

from turboedge.learning.ledger import ForwardLedger, select_shadow_sample
from turboedge.storage.schemas import ExitReason, LedgerEntryStatus, LedgerLabel


def test_record_is_idempotent_per_entry_id(store, make_ledger_entry) -> None:
    ledger = ForwardLedger(store)
    entry = make_ledger_entry()

    n1 = ledger.record([entry])
    assert n1 == 1
    # Same (run_id, candidate_id, horizon_days) -> same entry_id -> no-op.
    n2 = ledger.record([entry])
    assert n2 == 0
    assert store.table_counts()["forward_ledger"] == 1


def test_entry_id_is_deterministic_hash_of_run_candidate_horizon(make_ledger_entry) -> None:
    e1 = make_ledger_entry(run_id="r", candidate_id="c", horizon_days=7)
    e2 = make_ledger_entry(run_id="r", candidate_id="c", horizon_days=7)
    e3 = make_ledger_entry(run_id="r", candidate_id="c", horizon_days=10)
    assert e1.entry_id == e2.entry_id
    assert e1.entry_id != e3.entry_id
    assert len(e1.entry_id) == 20


def test_record_roundtrip_preserves_fields(store, make_ledger_entry) -> None:
    ledger = ForwardLedger(store)
    entry = make_ledger_entry(alternatives=["DE000XXX0001", "DE000XXX0002"])
    ledger.record([entry])

    fetched = store.get_ledger_entry(entry.entry_id)
    assert fetched is not None
    assert fetched == entry


def test_due_for_labeling_filters_by_status_and_exit_due(store, make_ledger_entry) -> None:
    ledger = ForwardLedger(store)
    due = make_ledger_entry(candidate_id="due", exit_due=date(2026, 9, 5))
    not_due = make_ledger_entry(candidate_id="not-due", exit_due=date(2026, 9, 20))
    ledger.record([due, not_due])

    as_of = datetime(2026, 9, 10, tzinfo=UTC)
    result = ledger.due_for_labeling(as_of)
    assert [e.entry_id for e in result] == [due.entry_id]


def test_attach_label_transitions_status_and_is_append_only(store, make_ledger_entry) -> None:
    ledger = ForwardLedger(store)
    entry = make_ledger_entry()
    ledger.record([entry])

    label = LedgerLabel(
        entry_id=entry.entry_id,
        labeled_at=datetime(2026, 9, 10, tzinfo=UTC),
        exit_bid=5.10,
        exit_quote_timestamp=datetime(2026, 9, 10, 15, 30, tzinfo=UTC),
        financing_level_exit=18020.0,
        exit_reason=ExitReason.HORIZON,
        realized_selected_pnl=5.10 / entry.entry_ask - 1,
        underlying_pnl=0.01,
        median_turbo_pnl=None,
        best_turbo_pnl=None,
        ideal_turbo_pnl=None,
        mfe=0.06,
        mae=-0.01,
        ko_hit=False,
        time_to_ko_days=None,
        ambiguous_path=False,
    )
    ledger.attach_label(label)

    rows = ledger.entries(run_id=entry.run_id)
    assert len(rows) == 1
    fetched_entry, fetched_label = rows[0]
    assert fetched_entry.status == LedgerEntryStatus.LABELED
    assert fetched_label is not None
    assert fetched_label.exit_bid == 5.10

    # Attaching a second label for the same entry is refused (append-only).
    import pytest

    from turboedge.storage.duckdb import StoreError

    with pytest.raises(StoreError, match="already labeled"):
        ledger.attach_label(label)


def test_attach_label_expired_no_data_sets_that_status(store, make_ledger_entry) -> None:
    ledger = ForwardLedger(store)
    entry = make_ledger_entry()
    ledger.record([entry])

    label = LedgerLabel(
        entry_id=entry.entry_id,
        labeled_at=datetime(2026, 9, 10, tzinfo=UTC),
        exit_bid=None,
        exit_quote_timestamp=None,
        financing_level_exit=None,
        exit_reason=ExitReason.EXPIRED_NO_DATA,
        realized_selected_pnl=None,
        underlying_pnl=None,
        median_turbo_pnl=None,
        best_turbo_pnl=None,
        ideal_turbo_pnl=None,
        mfe=None,
        mae=None,
        ko_hit=False,
        time_to_ko_days=None,
        ambiguous_path=True,
    )
    ledger.attach_label(label)

    fetched = store.get_ledger_entry(entry.entry_id)
    assert fetched is not None
    assert fetched.status == LedgerEntryStatus.EXPIRED_NO_DATA


def test_entries_filter_by_is_shadow_and_category(store, make_ledger_entry) -> None:
    ledger = ForwardLedger(store)
    from turboedge.storage.schemas import Category

    shadow = make_ledger_entry(candidate_id="shadow-1", is_shadow=True, category=Category.REJECT)
    actionable = make_ledger_entry(
        candidate_id="actionable-1", is_shadow=False, category=Category.WATCH
    )
    ledger.record([shadow, actionable])

    shadow_rows = ledger.entries(is_shadow=True)
    assert [e.entry_id for e, _ in shadow_rows] == [shadow.entry_id]

    watch_rows = ledger.entries(category=Category.WATCH)
    assert [e.entry_id for e, _ in watch_rows] == [actionable.entry_id]


def test_select_shadow_sample_is_deterministic_given_rng_seed() -> None:
    candidates = [f"cand-{i}" for i in range(12)]

    def strata_fn(c: str) -> tuple[str, str, str]:
        idx = int(c.split("-")[1])
        return ("WATCH" if idx % 2 == 0 else "REJECT", "long", "5-10")

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    sample1 = select_shadow_sample(candidates, rng1, 2, strata_fn)
    sample2 = select_shadow_sample(candidates, rng2, 2, strata_fn)
    assert sample1 == sample2
    # 2 strata (WATCH/REJECT) x up to 2 per stratum = at most 4.
    assert len(sample1) == 4


def test_select_shadow_sample_takes_all_of_a_small_stratum() -> None:
    candidates = ["a", "b", "c"]

    def strata_fn(c: str) -> tuple[str, str, str]:
        return ("only-stratum", "long", "5-10")

    rng = np.random.default_rng(1)
    sample = select_shadow_sample(candidates, rng, per_stratum=10, strata_fn=strata_fn)
    assert sorted(sample) == ["a", "b", "c"]


def test_select_shadow_sample_empty_candidates_returns_empty() -> None:
    rng = np.random.default_rng(1)
    assert select_shadow_sample([], rng, 5, lambda c: ("x", "y", "z")) == []
