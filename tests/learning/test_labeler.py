from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from turboedge.learning.labeler import label_due_entries
from turboedge.learning.ledger import ForwardLedger
from turboedge.storage.schemas import Direction, ExitReason, LedgerEntryStatus, ProductType


def _utc(y: int, m: int, d: int, hh: int = 12, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def _bars(make_underlying_bar, rows: list[tuple[date, float, float, float, float]]):
    return [
        make_underlying_bar(
            ts=datetime.combine(d, datetime.min.time(), tzinfo=UTC),
            open=o,
            high=h,
            low=lo,
            close=c,
            observation_time=datetime.combine(d, datetime.min.time(), tzinfo=UTC),
            available_at=datetime.combine(d, datetime.min.time(), tzinfo=UTC),
            retrieved_at=datetime.combine(d, datetime.min.time(), tzinfo=UTC),
        )
        for (d, o, h, lo, c) in rows
    ]


def test_normal_horizon_exit_uses_snapshot_at_or_after_deadline(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    entry = make_ledger_entry(
        entry_ask=4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=17000.0,  # far from spot, never touched
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    # A snapshot right at the exit deadline: this is the real exit quote.
    store.append_product_snapshots(
        [
            make_product_snapshot(
                isin=entry.selected_isin,
                bid=5.10,
                ask=5.16,
                observation_time=_utc(2026, 9, 10, 15, 30),
                quote_timestamp=_utc(2026, 9, 10, 15, 30),
            )
        ]
    )
    bars = _bars(
        make_underlying_bar,
        [
            (date(2026, 9, 1), 18000, 18100, 17950, 18050),
            (date(2026, 9, 10), 18200, 18300, 18150, 18250),
        ],
    )

    result = label_due_entries(store, _utc(2026, 9, 11), price_bars_lookup=lambda u: bars)
    assert result.labeled == 1
    assert result.ko == 0
    assert result.ambiguous == 0
    assert result.missing_data == 0

    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.exit_reason == ExitReason.HORIZON
    assert label.exit_bid == 5.10
    assert label.realized_selected_pnl == pytest.approx(5.10 / 4.86 - 1)
    assert label.ko_hit is False
    assert label.ambiguous_path is False
    assert label.underlying_pnl is not None


def test_ko_detected_via_underlying_bars_barrier_touch(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    entry = make_ledger_entry(
        entry_ask=4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=17900.0,
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    # No product snapshot flags knocked_out, but the bars show low <= barrier
    # on 2026-09-05.
    bars = _bars(
        make_underlying_bar,
        [
            (date(2026, 9, 1), 18000, 18100, 17950, 18050),
            (date(2026, 9, 5), 17950, 17960, 17850, 17880),  # low breaches barrier
            (date(2026, 9, 10), 17800, 17850, 17700, 17820),
        ],
    )

    result = label_due_entries(store, _utc(2026, 9, 11), price_bars_lookup=lambda u: bars)
    assert result.labeled == 1
    assert result.ko == 1

    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.exit_reason == ExitReason.KO
    assert label.ko_hit is True
    assert label.ambiguous_path is False
    # turbo (default instrument lookup is None -> conservative 0 residual)
    assert label.exit_bid == 0.0
    assert label.realized_selected_pnl == pytest.approx(-1.0)
    assert label.time_to_ko_days == 4


def test_ko_via_gap_through_open(store, make_ledger_entry, make_underlying_bar) -> None:
    """A pure overnight gap through the barrier (open already below it) must
    still be detected, since a bar's low is always <= its open."""
    entry = make_ledger_entry(
        entry_ask=4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=17900.0,
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    bars = _bars(
        make_underlying_bar,
        [
            (date(2026, 9, 1), 18000, 18100, 17950, 18050),
            # Gap down: opens already below the barrier.
            (date(2026, 9, 4), 17800, 17820, 17750, 17790),
            (date(2026, 9, 10), 17700, 17750, 17650, 17720),
        ],
    )

    result = label_due_entries(store, _utc(2026, 9, 11), price_bars_lookup=lambda u: bars)
    assert result.ko == 1
    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.ko_hit is True


def test_mini_future_ko_uses_post_ko_snapshot_bid(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    entry = make_ledger_entry(
        entry_ask=4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=17900.0,
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    # Register the instrument as a mini_future.
    snap_for_instrument = make_product_snapshot(
        isin=entry.selected_isin,
        product_type=ProductType.MINI_FUTURE,
        bid=4.80,
        ask=4.86,
        observation_time=_utc(2026, 9, 1, 15, 30),
        quote_timestamp=_utc(2026, 9, 1, 15, 30),
    )
    store.append_product_snapshots([snap_for_instrument])
    store.upsert_instruments([snap_for_instrument])

    # A post-KO snapshot with a residual bid.
    post_ko = make_product_snapshot(
        isin=entry.selected_isin,
        product_type=ProductType.MINI_FUTURE,
        bid=0.20,
        ask=0.25,
        observation_time=_utc(2026, 9, 6, 9, 0),
        quote_timestamp=_utc(2026, 9, 6, 9, 0),
    )
    store.append_product_snapshots([post_ko])

    bars = _bars(
        make_underlying_bar,
        [
            (date(2026, 9, 1), 18000, 18100, 17950, 18050),
            (date(2026, 9, 5), 17950, 17960, 17850, 17880),
            (date(2026, 9, 10), 17800, 17850, 17700, 17820),
        ],
    )

    label_due_entries(store, _utc(2026, 9, 11), price_bars_lookup=lambda u: bars)
    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.exit_reason == ExitReason.KO
    assert label.exit_bid == 0.20
    assert label.realized_selected_pnl == pytest.approx(0.20 / 4.86 - 1)


def test_missing_quotes_falls_back_to_conservative_ambiguous(
    store, make_ledger_entry, make_underlying_bar
) -> None:
    """No product snapshot at all in the window, no barrier touch in bars:
    falls back to entry_bid * (1 - entry_spread), marked ambiguous."""
    entry = make_ledger_entry(
        entry_ask=4.86,
        entry_bid=4.80,
        entry_spread=(4.86 - 4.80) / 4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=15000.0,  # far away, never touched
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    bars = _bars(
        make_underlying_bar,
        [
            (date(2026, 9, 1), 18000, 18100, 17950, 18050),
            (date(2026, 9, 10), 18200, 18300, 18150, 18250),
        ],
    )

    result = label_due_entries(store, _utc(2026, 9, 13), price_bars_lookup=lambda u: bars)
    assert result.labeled == 1
    assert result.ambiguous == 1
    assert result.missing_data == 0

    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.exit_reason == ExitReason.NO_EXIT_QUOTE_CONSERVATIVE
    assert label.ambiguous_path is True
    expected_bid = 4.80 * (1 - entry.entry_spread)
    assert label.exit_bid == pytest.approx(expected_bid)
    assert label.exit_bid < 4.80  # never optimistic: strictly discounted


def test_expired_no_data_when_absolutely_nothing_available(store, make_ledger_entry) -> None:
    entry = make_ledger_entry(
        entry_ask=4.86,
        entry_bid=None,  # no entry bid at all either
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=None,
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    result = label_due_entries(store, _utc(2026, 9, 13), price_bars_lookup=lambda u: [])
    assert result.labeled == 0
    assert result.missing_data == 1

    fetched = store.get_ledger_entry(entry.entry_id)
    assert fetched is not None
    assert fetched.status == LedgerEntryStatus.EXPIRED_NO_DATA

    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.exit_reason == ExitReason.EXPIRED_NO_DATA
    assert label.realized_selected_pnl is None


def test_entry_not_yet_due_is_not_labeled(store, make_ledger_entry) -> None:
    entry = make_ledger_entry(exit_due=date(2026, 12, 31))
    ForwardLedger(store).record([entry])

    result = label_due_entries(store, _utc(2026, 9, 11), price_bars_lookup=lambda u: [])
    assert result.labeled == 0
    assert result.missing_data == 0
    fetched = store.get_ledger_entry(entry.entry_id)
    assert fetched is not None
    assert fetched.status == LedgerEntryStatus.OPEN


def test_fallback_exit_quote_within_grace_window_is_horizon_not_ambiguous(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    """No snapshot at/after the exact deadline, but one within the grace
    window -- still a real quote, so exit_reason is HORIZON, not ambiguous."""
    entry = make_ledger_entry(
        entry_ask=4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),  # a Thursday
        barrier_entry=15000.0,
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    # Last snapshot is BEFORE the deadline (exit_due 15:30), but still within
    # the +2 trading day grace window.
    store.append_product_snapshots(
        [
            make_product_snapshot(
                isin=entry.selected_isin,
                bid=5.00,
                ask=5.06,
                observation_time=_utc(2026, 9, 9, 10, 0),
                quote_timestamp=_utc(2026, 9, 9, 10, 0),
            )
        ]
    )
    bars = _bars(
        make_underlying_bar,
        [
            (date(2026, 9, 1), 18000, 18100, 17950, 18050),
            (date(2026, 9, 10), 18200, 18300, 18150, 18250),
        ],
    )

    label_due_entries(store, _utc(2026, 9, 13), price_bars_lookup=lambda u: bars)
    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.exit_reason == ExitReason.HORIZON
    assert label.ambiguous_path is False
    assert label.exit_bid == 5.00
