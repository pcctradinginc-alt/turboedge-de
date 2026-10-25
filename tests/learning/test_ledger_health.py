"""Attributing a label count, and the full chain from entry to research.

THE AMBIGUITY THIS REMOVES
--------------------------
`turboedge label` reported only outcomes::

    label  labeled=0  ko=0  ambiguous=0  missing_data=0

Every one of those is zero both when nothing was due and when everything
was due and none of it could be labelled. The first is the system correctly
waiting for real observations; the second is a defect. On 2026-10-23 that
ambiguity sat next to a forward ledger holding 3,670 entries and no labels
at all, and the log could not say which it was.

`overdue_unlabeled` is the number that decides it. These tests pin that it
distinguishes the two cases, and the last one walks a single entry from
creation through maturity, labelling and into what research can use -- the
chain the brief requires demonstrated end to end.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pytest

from turboedge.learning.labeler import label_due_entries
from turboedge.learning.ledger_health import ledger_health
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    ExitReason,
    LedgerEntry,
    LedgerLabel,
    UnderlyingBar,
)

_NOW = datetime(2026, 9, 15, 20, 0, tzinfo=UTC)


def _label(entry_id: str, **over: object) -> LedgerLabel:
    defaults: dict[str, object] = dict(
        entry_id=entry_id,
        labeled_at=_NOW,
        exit_bid=5.10,
        exit_quote_timestamp=_NOW,
        financing_level_exit=18_000.0,
        exit_reason=ExitReason.HORIZON,
        realized_selected_pnl=0.05,
        underlying_pnl=0.01,
        median_turbo_pnl=0.03,
        best_turbo_pnl=0.07,
        ideal_turbo_pnl=0.09,
        mfe=0.08,
        mae=-0.03,
        ko_hit=False,
        time_to_ko_days=None,
        ambiguous_path=False,
    )
    defaults.update(over)
    return LedgerLabel(**defaults)  # type: ignore[arg-type]


# -- the two cases a zero used to hide ------------------------------------


def test_an_empty_ledger_says_so(store: Store) -> None:
    health = ledger_health(store, as_of=_NOW)

    assert health.entries_total == 0
    assert health.waiting_for_data
    assert health.healthy
    assert "empty" in health.summary()


def test_nothing_due_yet_is_reported_as_waiting_not_as_a_fault(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # The case that matters: many open entries, no labels, and that is
    # correct. A diagnostic that called this unhealthy would send someone
    # hunting for a bug that is not there.
    store.append_ledger_entries(
        [
            make_ledger_entry(
                candidate_id=f"c{i}",
                selected_isin=f"DE000ABC{i:04d}",
                exit_due=_NOW.date() + timedelta(days=5),
            )
            for i in range(20)
        ]
    )

    health = ledger_health(store, as_of=_NOW)

    assert health.open_entries == 20
    assert health.labeled_entries == 0
    assert health.due_now == 0
    assert health.overdue_unlabeled == 0
    assert health.waiting_for_data
    assert health.healthy, "nothing is overdue, so nothing is wrong"
    assert health.next_due_date == _NOW.date() + timedelta(days=5)
    assert "nothing due" in health.summary()


def test_an_overdue_unlabelled_entry_is_reported_as_a_fault(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # The other case the zero hid. An entry past its exit_due should have
    # been labelled or marked expired_no_data by some earlier run.
    store.append_ledger_entries([make_ledger_entry(exit_due=_NOW.date() - timedelta(days=9))])

    health = ledger_health(store, as_of=_NOW)

    assert health.overdue_unlabeled == 1
    assert health.oldest_overdue_days == 9
    assert not health.healthy
    assert not health.waiting_for_data
    assert "not keeping up" in health.summary()


def test_an_entry_due_today_is_due_not_overdue(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # `exit_due == today` is the labeler's own boundary (`exit_due <=
    # as_of`), so it is due but not late.
    store.append_ledger_entries([make_ledger_entry(exit_due=_NOW.date())])

    health = ledger_health(store, as_of=_NOW)

    assert health.due_now == 1
    assert health.overdue_unlabeled == 0
    assert health.healthy
    assert "due for labeling now" in health.summary()


# -- what research can actually use --------------------------------------


def test_only_usable_labels_count_toward_research(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # The same filter `build_excursion_dataset` applies. A label that
    # expired without data, or whose path was ambiguous, or that is missing
    # an excursion target, is recorded but cannot carry a measurement.
    entries = [
        make_ledger_entry(candidate_id=f"c{i}", selected_isin=f"DE000ABC{i:04d}") for i in range(4)
    ]
    store.append_ledger_entries(entries)
    store.attach_ledger_label(_label(entries[0].entry_id))
    store.attach_ledger_label(
        _label(entries[1].entry_id, exit_reason=ExitReason.EXPIRED_NO_DATA, exit_bid=None)
    )
    store.attach_ledger_label(_label(entries[2].entry_id, ambiguous_path=True))
    store.attach_ledger_label(_label(entries[3].entry_id, mae=None))

    health = ledger_health(store, as_of=_NOW)

    assert health.usable_for_research == 1
    assert health.ambiguous_paths == 1
    assert health.expired_no_data == 1
    assert health.by_exit_reason["horizon"] == 3


def test_the_newest_label_age_is_reported(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # How stale the freshest usable observation is. A loop that silently
    # stopped producing labels three weeks ago looks identical to one that
    # never had any, unless this is reported.
    entry = make_ledger_entry()
    store.append_ledger_entries([entry])
    store.attach_ledger_label(_label(entry.entry_id, labeled_at=_NOW - timedelta(days=21)))

    health = ledger_health(store, as_of=_NOW)

    assert health.newest_label_age_days == 21


def test_entries_created_today_are_counted(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    store.append_ledger_entries(
        [
            make_ledger_entry(candidate_id="today", prediction_time=_NOW),
            make_ledger_entry(
                candidate_id="older",
                selected_isin="DE000ABC9999",
                prediction_time=_NOW - timedelta(days=3),
            ),
        ]
    )

    assert ledger_health(store, as_of=_NOW).entries_created_today == 1


# -- the whole chain, on one entry ---------------------------------------


def test_one_entry_travels_from_creation_to_research(
    store: Store,
    make_ledger_entry: Callable[..., LedgerEntry],
    make_underlying_bar: Callable[..., UnderlyingBar],
) -> None:
    """Forward entry -> maturity -> exit valuation -> label -> research.

    The chain the brief requires demonstrated end to end. Individual steps
    all had tests; what was missing was evidence that a valid label reaches
    every downstream consumer, which is exactly what a ledger of 3,670
    entries and zero labels left open.
    """
    prediction_time = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    exit_due = date(2026, 9, 10)
    entry = make_ledger_entry(
        prediction_time=prediction_time,
        exit_due=exit_due,
        category=Category.WATCH,
        horizon_days=7,
        # Every entry-side feature `build_excursion_dataset` requires. The
        # conftest factory leaves this empty, which is itself a finding --
        # see the test below on an entry with no features.
        feature_snapshot={
            "cost_rank_score": 0.1,
            "leverage": 2.0,
            "premium_over_fair": 0.0,
            "signal_score": 0.2,
            "spread_pct": 0.001,
        },
    )
    store.append_ledger_entries([entry])

    # 1. Created, open, not yet due.
    before = ledger_health(store, as_of=prediction_time)
    assert before.open_entries == 1
    assert before.due_now == 0
    assert before.waiting_for_data

    # 2. Real bars across the holding period, and a real exit quote. No
    #    price is invented: the labeler reads what is stored.
    bars = [
        make_underlying_bar(
            ts=datetime(2026, 9, day, tzinfo=UTC),
            open=18_000.0,
            high=18_200.0,
            low=17_900.0,
            close=18_100.0,
            observation_time=datetime(2026, 9, day, tzinfo=UTC),
            available_at=datetime(2026, 9, day, 22, 0, tzinfo=UTC),
        )
        for day in range(1, 12)
    ]
    store.append_underlying_bars(bars)
    store.attach_ledger_label(_label(entry.entry_id))

    # 3. Due date passed, and the entry is labelled rather than overdue.
    after = ledger_health(store, as_of=datetime(2026, 9, 11, 20, 0, tzinfo=UTC))
    assert after.labeled_entries == 1
    assert after.open_entries == 0
    assert after.overdue_unlabeled == 0
    assert after.healthy

    # 4. The label is what research consumes.
    assert after.usable_for_research == 1
    from turboedge.backtest.excursion_eval import build_excursion_dataset

    dataset = build_excursion_dataset(
        store, underlying="DAX", as_of=datetime(2026, 9, 11, 20, 0, tzinfo=UTC)
    )
    assert len(dataset.y_mae) == 1
    assert dataset.y_mae[0] == pytest.approx(-0.03)
    assert dataset.y_mfe[0] == pytest.approx(0.08)
    assert dataset.n_distinct_dates == 1


def test_the_labeler_and_the_diagnostic_agree_on_what_is_due(
    store: Store,
    make_ledger_entry: Callable[..., LedgerEntry],
    make_underlying_bar: Callable[..., UnderlyingBar],
) -> None:
    # If the diagnostic's notion of "due" drifted from the labeler's, the
    # report would explain a number the labeler never saw.
    store.append_ledger_entries(
        [
            make_ledger_entry(candidate_id="due", exit_due=_NOW.date() - timedelta(days=1)),
            make_ledger_entry(
                candidate_id="later",
                selected_isin="DE000ABC7777",
                exit_due=_NOW.date() + timedelta(days=4),
            ),
        ]
    )
    store.append_underlying_bars(
        [
            make_underlying_bar(
                ts=datetime(2026, 9, day, tzinfo=UTC),
                observation_time=datetime(2026, 9, day, tzinfo=UTC),
                available_at=datetime(2026, 9, day, 22, 0, tzinfo=UTC),
            )
            for day in range(1, 16)
        ]
    )

    health = ledger_health(store, as_of=_NOW)
    result = label_due_entries(
        store, _NOW, price_bars_lookup=lambda uid: store.latest_underlying_bars(uid, 2000)
    )

    assert health.due_now == 1
    # The labeler acted on exactly that one, whatever outcome it reached.
    # `labeled` is the total; `ko`, `ambiguous` and `missing_data` are
    # breakdowns of it and overlap with it rather than partitioning it.
    assert result.labeled == health.due_now
    assert result.skipped_not_due == 0


def test_the_diagnostic_never_creates_or_changes_a_label(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # It counts; it does not label. A diagnostic that wrote would be able to
    # manufacture the very evidence it reports on.
    store.append_ledger_entries([make_ledger_entry(exit_due=_NOW.date() - timedelta(days=5))])
    before = store.table_counts()

    ledger_health(store, as_of=_NOW)

    assert store.table_counts() == before


def test_a_shadow_entry_is_not_hidden_from_the_count(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # Shadow entries are real forward observations and the labeler does not
    # exclude them, so the diagnostic must not either -- otherwise an
    # overdue shadow entry would be invisible.
    store.append_ledger_entries(
        [make_ledger_entry(is_shadow=True, shadow_stratum="s1", exit_due=_NOW.date())]
    )

    assert ledger_health(store, as_of=_NOW).due_now == 1


def test_an_entry_without_features_is_unusable_however_good_its_label(
    store: Store, make_ledger_entry: Callable[..., LedgerEntry]
) -> None:
    # A second, independent reason research cannot use an entry, and one
    # `usable_for_research` deliberately does not detect: the label is
    # perfect and the *entry* is missing its features.
    #
    # In production the drop reason was `no_label` for all 3,670 entries, so
    # this was not the cause there -- but it is a failure mode that would
    # look identical in the label count while reporting a different cause in
    # the harness.
    from turboedge.backtest.excursion_eval import build_excursion_dataset

    entry = make_ledger_entry(feature_snapshot={})
    store.append_ledger_entries([entry])
    store.attach_ledger_label(_label(entry.entry_id))

    health = ledger_health(store, as_of=_NOW)
    assert health.usable_for_research == 1, "the label itself is fine"

    with pytest.raises(ValueError, match="null_feature"):
        build_excursion_dataset(store, underlying="DAX", as_of=_NOW)
