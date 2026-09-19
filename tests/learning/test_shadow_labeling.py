"""Tests for Phase F's shadow-portfolio labeling follow-up
(`learning/labeler.py`'s `label_due_shadow_positions`, wired into
`label_due_entries`, Master Spec §46).

CLAUDE.md rule 17 / the task's own hard requirement: a shadow position must
never be scored more favorably than an equivalent real forward-ledger entry
would have been. Since `label_due_shadow_positions` resolves every exit
through the exact same `resolve_exit_with_fallback` function real entries
use (not a re-implementation), the natural way to prove "same treatment" is
to label a real ledger entry and a shadow position sharing the same isin/
entry_ask/exit_due side by side and assert their realized returns match
exactly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from turboedge.learning.labeler import label_due_entries, label_due_shadow_positions
from turboedge.learning.ledger import ForwardLedger
from turboedge.storage.schemas import (
    Direction,
    ProductType,
    ShadowPortfolioKind,
    ShadowPosition,
)


def _utc(y: int, m: int, d: int, hh: int = 12, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def _bars(make_underlying_bar, rows):  # type: ignore[no-untyped-def]
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


def test_shadow_horizon_exit_matches_real_ledger_entry_exactly(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    """Same isin, same entry_ask, same exit_due -- a real HORIZON exit and
    the shadow position's own exit must resolve to the identical realized
    return (both go through `resolve_exit_with_fallback`)."""
    isin = "DE000ABC1234"
    entry = make_ledger_entry(
        selected_isin=isin,
        entry_ask=4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=17000.0,  # far from spot, never touched
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    entry_snapshot = make_product_snapshot(
        isin=isin,
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        knockout_barrier=17000.0,
        bid=4.80,
        ask=4.86,
        observation_time=_utc(2026, 9, 1, 15, 30),
        quote_timestamp=_utc(2026, 9, 1, 15, 30),
    )
    store.append_product_snapshots([entry_snapshot])
    store.upsert_instruments([entry_snapshot])
    store.append_product_snapshots(
        [
            make_product_snapshot(
                isin=isin,
                direction=Direction.LONG,
                product_type=ProductType.TURBO_OPEN_END,
                knockout_barrier=17000.0,
                bid=5.10,
                ask=5.16,
                observation_time=_utc(2026, 9, 10, 15, 30),
                quote_timestamp=_utc(2026, 9, 10, 15, 30),
            )
        ]
    )

    store.append_shadow_positions(
        [
            ShadowPosition(
                run_id="run-1",
                portfolio=ShadowPortfolioKind.TOP1,
                isin=isin,
                horizon_days=7,
                entry_ask=4.86,
                exit_due=date(2026, 9, 10),
                realized_net_return=None,
                created_at=_utc(2026, 9, 1, 15, 30),
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
    assert result.shadow_positions_labeled == 1

    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.realized_selected_pnl is not None

    shadow_rows = store.list_shadow_positions(run_id="run-1")
    assert len(shadow_rows) == 1
    assert shadow_rows[0].realized_net_return is not None
    assert shadow_rows[0].realized_net_return == label.realized_selected_pnl
    assert shadow_rows[0].realized_net_return == 5.10 / 4.86 - 1


def test_shadow_ko_matches_real_ledger_entry_exactly(
    store, make_ledger_entry, make_product_snapshot, make_underlying_bar
) -> None:
    """A barrier touch in the underlying bars must knock out the shadow
    position exactly like the real ledger entry -- same conservative 0
    residual (turbo), same -1.0 realized return."""
    isin = "DE000ABC1234"
    entry = make_ledger_entry(
        selected_isin=isin,
        entry_ask=4.86,
        entry_quote_timestamp=_utc(2026, 9, 1, 15, 30),
        entry_underlying_timestamp=_utc(2026, 9, 1, 15, 30),
        exit_due=date(2026, 9, 10),
        barrier_entry=17900.0,
        direction=Direction.LONG,
    )
    ForwardLedger(store).record([entry])

    entry_snapshot = make_product_snapshot(
        isin=isin,
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        knockout_barrier=17900.0,
        bid=4.80,
        ask=4.86,
        observation_time=_utc(2026, 9, 1, 15, 30),
        quote_timestamp=_utc(2026, 9, 1, 15, 30),
    )
    store.append_product_snapshots([entry_snapshot])
    store.upsert_instruments([entry_snapshot])

    store.append_shadow_positions(
        [
            ShadowPosition(
                run_id="run-2",
                portfolio=ShadowPortfolioKind.LOWEST_SPREAD,
                isin=isin,
                horizon_days=7,
                entry_ask=4.86,
                exit_due=date(2026, 9, 10),
                realized_net_return=None,
                created_at=_utc(2026, 9, 1, 15, 30),
            )
        ]
    )

    # No snapshot flags knocked_out, but the bars show low <= barrier on
    # 2026-09-05 -- same as test_ko_detected_via_underlying_bars_barrier_touch.
    bars = _bars(
        make_underlying_bar,
        [
            (date(2026, 9, 1), 18000, 18100, 17950, 18050),
            (date(2026, 9, 5), 17950, 17960, 17850, 17880),
            (date(2026, 9, 10), 17800, 17850, 17700, 17820),
        ],
    )

    result = label_due_entries(store, _utc(2026, 9, 11), price_bars_lookup=lambda u: bars)
    assert result.ko == 1
    assert result.shadow_positions_labeled == 1

    label = store.get_ledger_label(entry.entry_id)
    assert label is not None
    assert label.realized_selected_pnl == -1.0

    shadow_rows = store.list_shadow_positions(run_id="run-2")
    assert shadow_rows[0].realized_net_return == -1.0
    assert shadow_rows[0].realized_net_return == label.realized_selected_pnl


def test_shadow_position_with_no_resolvable_data_stays_unlabeled_never_optimistic(
    store, make_product_snapshot
) -> None:
    """`ShadowPosition` has no stored `entry_bid` (unlike `LedgerEntry`), so
    when the only known snapshot for the isin has no bid at all (bid_only)
    and nothing else exists, there is no conservative fallback value to
    compute -- the position must be left unlabeled (never assigned a
    fabricated, potentially-optimistic return, rule 29) rather than, say,
    defaulting to 0 return."""
    isin = "DE000ABC1234"
    entry_snapshot = make_product_snapshot(
        isin=isin,
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        knockout_barrier=17000.0,
        bid=None,
        ask=4.86,
        bid_only=True,
        observation_time=_utc(2026, 9, 1, 15, 30),
        quote_timestamp=_utc(2026, 9, 1, 15, 30),
    )
    store.append_product_snapshots([entry_snapshot])
    store.upsert_instruments([entry_snapshot])

    store.append_shadow_positions(
        [
            ShadowPosition(
                run_id="run-3",
                portfolio=ShadowPortfolioKind.MEDIAN_PRODUCT,
                isin=isin,
                horizon_days=7,
                entry_ask=4.86,
                exit_due=date(2026, 9, 10),
                realized_net_return=None,
                created_at=_utc(2026, 9, 1, 15, 30),
            )
        ]
    )

    labeled = label_due_shadow_positions(store, _utc(2026, 9, 13), price_bars_lookup=lambda u: [])
    assert labeled == 0

    shadow_rows = store.list_shadow_positions(run_id="run-3")
    assert shadow_rows[0].realized_net_return is None
    # Still due (unresolved), so a later run would try again rather than
    # silently treating it as final.
    still_due = store.shadow_positions_due_for_labeling(_utc(2026, 9, 13))
    assert any(p.isin == isin and p.run_id == "run-3" for p in still_due)


def test_not_yet_due_shadow_position_is_not_labeled(store, make_product_snapshot) -> None:
    isin = "DE000ABC1234"
    entry_snapshot = make_product_snapshot(
        isin=isin,
        direction=Direction.LONG,
        bid=4.80,
        ask=4.86,
        observation_time=_utc(2026, 9, 1, 15, 30),
        quote_timestamp=_utc(2026, 9, 1, 15, 30),
    )
    store.append_product_snapshots([entry_snapshot])
    store.upsert_instruments([entry_snapshot])

    store.append_shadow_positions(
        [
            ShadowPosition(
                run_id="run-4",
                portfolio=ShadowPortfolioKind.TOP1,
                isin=isin,
                horizon_days=7,
                entry_ask=4.86,
                exit_due=date(2026, 12, 31),  # far in the future
                realized_net_return=None,
                created_at=_utc(2026, 9, 1, 15, 30),
            )
        ]
    )

    labeled = label_due_shadow_positions(store, _utc(2026, 9, 11), price_bars_lookup=lambda u: [])
    assert labeled == 0
    shadow_rows = store.list_shadow_positions(run_id="run-4")
    assert shadow_rows[0].realized_net_return is None


def test_already_labeled_shadow_position_is_not_reprocessed(store, make_product_snapshot) -> None:
    isin = "DE000ABC1234"
    entry_snapshot = make_product_snapshot(
        isin=isin,
        direction=Direction.LONG,
        bid=4.80,
        ask=4.86,
        observation_time=_utc(2026, 9, 1, 15, 30),
        quote_timestamp=_utc(2026, 9, 1, 15, 30),
    )
    store.append_product_snapshots([entry_snapshot])
    store.upsert_instruments([entry_snapshot])

    store.append_shadow_positions(
        [
            ShadowPosition(
                run_id="run-5",
                portfolio=ShadowPortfolioKind.TOP1,
                isin=isin,
                horizon_days=7,
                entry_ask=4.86,
                exit_due=date(2026, 9, 10),
                realized_net_return=0.05,  # already labeled by a previous run
                created_at=_utc(2026, 9, 1, 15, 30),
            )
        ]
    )

    still_due = store.shadow_positions_due_for_labeling(_utc(2026, 9, 13))
    assert not any(p.run_id == "run-5" for p in still_due)

    labeled = label_due_shadow_positions(store, _utc(2026, 9, 13), price_bars_lookup=lambda u: [])
    assert labeled == 0
    shadow_rows = store.list_shadow_positions(run_id="run-5")
    assert shadow_rows[0].realized_net_return == 0.05  # untouched
