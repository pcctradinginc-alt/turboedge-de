from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from turboedge.learning.counterfactual import evaluate_counterfactual
from turboedge.learning.ledger import ForwardLedger


def _utc(y: int, m: int, d: int, hh: int = 12, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def _bars(make_underlying_bar):
    rows = [
        (date(2026, 9, 1), 18000, 18100, 17950, 18050),
        (date(2026, 9, 10), 18200, 18300, 18150, 18250),
    ]
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


def test_counterfactual_computes_median_and_best_across_alternatives(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    alt1 = "DE000ALT0001"
    alt2 = "DE000ALT0002"
    alt3 = "DE000ALT0003"
    entry = make_ledger_entry(
        entry_ask=4.86,
        prediction_time=_utc(2026, 9, 1, 15, 30),
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=15000.0,
        alternatives=[alt1, alt2, alt3],
    )
    ForwardLedger(store).record([entry])

    # Each alternative needs its own entry-time snapshot (ask/barrier) plus
    # its own exit-time snapshot (bid).
    entry_snaps = [
        make_product_snapshot(
            isin=isin,
            bid=entry_bid,
            ask=entry_ask,
            knockout_barrier=15000.0,
            observation_time=_utc(2026, 9, 1, 15, 30),
            quote_timestamp=_utc(2026, 9, 1, 15, 30),
        )
        for isin, entry_bid, entry_ask in [
            (alt1, 4.00, 4.06),
            (alt2, 5.00, 5.08),
            (alt3, 3.00, 3.05),
        ]
    ]
    exit_snaps = [
        make_product_snapshot(
            isin=isin,
            bid=exit_bid,
            ask=exit_bid + 0.05,
            knockout_barrier=15000.0,
            observation_time=_utc(2026, 9, 10, 15, 30),
            quote_timestamp=_utc(2026, 9, 10, 15, 30),
        )
        for isin, exit_bid in [
            (alt1, 4.20),  # return = 4.20/4.06 - 1 ~= 0.0345
            (alt2, 6.00),  # return = 6.00/5.08 - 1 ~= 0.1811 (best)
            (alt3, 2.50),  # return = 2.50/3.05 - 1 ~= -0.1803 (worst)
        ]
    ]
    store.append_product_snapshots(entry_snaps + exit_snaps)

    result = evaluate_counterfactual(
        store,
        entry,
        underlying_bars=_bars(make_underlying_bar),
        selected_realized_pnl=0.05,
    )

    assert result.n_evaluated == 3
    returns = sorted([4.20 / 4.06 - 1, 6.00 / 5.08 - 1, 2.50 / 3.05 - 1])
    expected_median = returns[1]
    expected_best = returns[2]
    assert result.median_turbo_pnl == pytest.approx(expected_median)
    assert result.best_turbo_pnl == pytest.approx(expected_best)
    assert result.product_selection_edge == pytest.approx(0.05 - expected_median)
    assert result.regret == pytest.approx(expected_best - 0.05)


def test_counterfactual_skips_alternatives_with_no_entry_snapshot(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    alt_with_data = "DE000ALT0001"
    alt_without_data = "DE000ALT0099"
    entry = make_ledger_entry(
        entry_ask=4.86,
        prediction_time=_utc(2026, 9, 1, 15, 30),
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=15000.0,
        alternatives=[alt_with_data, alt_without_data],
    )
    ForwardLedger(store).record([entry])

    store.append_product_snapshots(
        [
            make_product_snapshot(
                isin=alt_with_data,
                bid=4.00,
                ask=4.06,
                knockout_barrier=15000.0,
                observation_time=_utc(2026, 9, 1, 15, 30),
                quote_timestamp=_utc(2026, 9, 1, 15, 30),
            ),
            make_product_snapshot(
                isin=alt_with_data,
                bid=4.20,
                ask=4.25,
                knockout_barrier=15000.0,
                observation_time=_utc(2026, 9, 10, 15, 30),
                quote_timestamp=_utc(2026, 9, 10, 15, 30),
            ),
        ]
    )
    # alt_without_data has no snapshots at all -> must be skipped, not
    # forced to a fabricated value.

    result = evaluate_counterfactual(
        store,
        entry,
        underlying_bars=_bars(make_underlying_bar),
        selected_realized_pnl=0.05,
    )
    assert result.n_evaluated == 1
    # Entry is always at the ask (4.06), never the bid (CLAUDE.md rule 11).
    assert result.median_turbo_pnl == pytest.approx(4.20 / 4.06 - 1)


def test_counterfactual_no_alternatives_returns_none_results(
    store, make_ledger_entry, make_underlying_bar
) -> None:
    entry = make_ledger_entry(alternatives=[])
    ForwardLedger(store).record([entry])

    result = evaluate_counterfactual(
        store,
        entry,
        underlying_bars=_bars(make_underlying_bar),
        selected_realized_pnl=0.05,
    )
    assert result.n_evaluated == 0
    assert result.median_turbo_pnl is None
    assert result.best_turbo_pnl is None
    assert result.product_selection_edge is None
    assert result.regret is None


def test_counterfactual_edge_and_regret_none_without_selected_pnl(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    alt1 = "DE000ALT0001"
    entry = make_ledger_entry(
        entry_ask=4.86,
        prediction_time=_utc(2026, 9, 1, 15, 30),
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=15000.0,
        alternatives=[alt1],
    )
    ForwardLedger(store).record([entry])
    store.append_product_snapshots(
        [
            make_product_snapshot(
                isin=alt1,
                bid=4.00,
                ask=4.06,
                knockout_barrier=15000.0,
                observation_time=_utc(2026, 9, 1, 15, 30),
                quote_timestamp=_utc(2026, 9, 1, 15, 30),
            ),
            make_product_snapshot(
                isin=alt1,
                bid=4.20,
                ask=4.25,
                knockout_barrier=15000.0,
                observation_time=_utc(2026, 9, 10, 15, 30),
                quote_timestamp=_utc(2026, 9, 10, 15, 30),
            ),
        ]
    )

    result = evaluate_counterfactual(
        store,
        entry,
        underlying_bars=_bars(make_underlying_bar),
        selected_realized_pnl=None,
    )
    assert result.n_evaluated == 1
    assert result.median_turbo_pnl is not None
    assert result.product_selection_edge is None
    assert result.regret is None
