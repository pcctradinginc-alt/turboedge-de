"""Tests for reporting/monthly.py: honesty rules, groupings, persistence hooks."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from turboedge.learning.ledger import ForwardLedger
from turboedge.reporting.monthly import MonthlyReportConfig, build_monthly_report
from turboedge.storage.schemas import Category

_AS_OF = datetime(2026, 9, 1, 6, 30, tzinfo=UTC)
_MONTH = date(2026, 8, 1)


def test_month_with_no_matured_trades_is_a_status_report(store, make_ledger_entry) -> None:
    ledger = ForwardLedger(store)
    # Recorded (scanned) this month, but never matured/labeled: WATCH/REJECT
    # candidates from a normal scan run that found nothing ACTIONABLE.
    watch = make_ledger_entry(
        candidate_id="watch-1",
        category=Category.WATCH,
        prediction_time=datetime(2026, 8, 5, 8, 0, tzinfo=UTC),
        exit_due=date(2026, 8, 12),
    )
    reject = make_ledger_entry(
        candidate_id="reject-1",
        category=Category.REJECT,
        prediction_time=datetime(2026, 8, 5, 8, 0, tzinfo=UTC),
        exit_due=date(2026, 8, 12),
    )
    ledger.record([watch, reject])

    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)

    assert report.status_only is True
    assert report.n_scans_in_month == 1
    assert report.category_counts_in_month == {"WATCH": 1, "REJECT": 1}
    assert report.actionable.n == 0
    assert report.shadow.n == 0
    assert any("noch nicht gelabelt" in n for n in report.data_quality_notes)
    assert report.narrative  # a human-readable status line was produced


def test_month_with_completely_empty_ledger_still_reports(store) -> None:
    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    assert report.status_only is True
    assert report.n_scans_in_month == 0
    assert report.category_counts_in_month == {}
    assert report.actionable.n == 0
    assert report.actionable.reliable is False


def test_three_trades_are_marked_not_reliable_and_withhold_sharpe(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """CLAUDE.md/Spec §27.4 honesty rule: never a Sharpe ratio from 3 trades."""
    returns_bid = [5.10, 4.60, 5.30]  # +, -, + net returns off entry_ask=4.86
    for i, bid in enumerate(returns_bid):
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"cand-{i}",
                category=Category.ACTIONABLE,
                prediction_time=datetime(2026, 8, 1 + i, 8, 0, tzinfo=UTC),
                exit_due=date(2026, 8, 8 + i),
            ),
            label_overrides=dict(exit_bid=bid, realized_selected_pnl=bid / 4.86 - 1),
        )

    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    stats = report.actionable

    assert stats.n == 3
    assert stats.reliable is False
    assert stats.sharpe is None
    assert stats.sortino is None
    assert stats.psr is None
    assert stats.dsr is None
    # Mean/median/success-rate ARE still reported, with sample-size context.
    assert stats.mean_return is not None
    assert stats.success_rate == pytest.approx(2 / 3)
    assert stats.n_effective > 0.0
    assert any("nicht aussagekraeftig" in n for n in stats.notes)
    assert any(str(stats.n) in n for n in stats.notes)


def test_forty_trades_produce_full_metrics_with_hand_verified_wilson_interval(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """n=40, 20 successes (50%): Wilson 95% CI has a clean closed form since
    phat=0.5 makes the recentered point estimate exactly 0.5 regardless of
    n. Hand-derived expected interval (Wilson 1927 formula, z=1.959963985):

        half = z * sqrt((0.25 + z^2/(4n)) / n) / (1 + z^2/n)
             = 1.959963985 * sqrt((0.25 + 0.024009118) / 40) / 1.096036470
             ~= 0.147996

    giving [0.352004, 0.647996] -- independent of this repo's own
    implementation (computed here from the textbook formula directly).
    """
    entries_per_day = 2
    for i in range(40):
        day = 1 + (i // entries_per_day)
        pred = datetime(2026, 8, day, 8, 0, tzinfo=UTC)
        exit_due = date(2026, 8, day) + timedelta(days=7)
        success = i % 2 == 0
        exit_bid = 5.10 if success else 4.60
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"cand-{i}",
                category=Category.ACTIONABLE,
                prediction_time=pred,
                exit_due=exit_due,
            ),
            label_overrides=dict(exit_bid=exit_bid, realized_selected_pnl=exit_bid / 4.86 - 1),
        )

    cfg = MonthlyReportConfig(n_bootstrap=200)
    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF, cfg=cfg)
    stats = report.actionable

    assert stats.n == 40
    assert stats.reliable is True
    assert stats.success_rate == pytest.approx(0.5)
    ci = stats.success_rate_ci
    assert ci is not None
    assert ci.lower == pytest.approx(0.352004, abs=5e-4)
    assert ci.upper == pytest.approx(0.647996, abs=5e-4)
    # Full metrics are now populated (n=40 >= default min_trades_for_stats=10).
    assert stats.sharpe is not None
    assert stats.sortino is not None
    assert stats.psr is not None
    assert stats.dsr is not None
    assert stats.mean_return_ci is not None


def test_forty_trades_status_only_is_false(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    for i in range(40):
        day = 1 + (i // 2)
        pred = datetime(2026, 8, day, 8, 0, tzinfo=UTC)
        exit_due = date(2026, 8, day) + timedelta(days=7)
        exit_bid = 5.10 if i % 2 == 0 else 4.60
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"cand-{i}",
                category=Category.ACTIONABLE,
                prediction_time=pred,
                exit_due=exit_due,
            ),
            label_overrides=dict(exit_bid=exit_bid, realized_selected_pnl=exit_bid / 4.86 - 1),
        )
    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    assert report.status_only is False


def test_calibration_brier_and_ece_are_computed_from_p_profit_vs_outcome(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    # Two trades: p_profit=1.0 and outcome positive (perfectly calibrated,
    # contributes 0 squared error); p_profit=0.0 and outcome positive
    # (maximally miscalibrated, contributes (0-1)^2 = 1). Brier = mean = 0.5.
    record_labeled(
        entry_overrides=dict(
            candidate_id="cal-1",
            category=Category.ACTIONABLE,
            p_profit=1.0,
            prediction_time=datetime(2026, 8, 2, 8, 0, tzinfo=UTC),
            exit_due=date(2026, 8, 9),
        ),
        label_overrides=dict(exit_bid=5.10, realized_selected_pnl=5.10 / 4.86 - 1),
    )
    record_labeled(
        entry_overrides=dict(
            candidate_id="cal-2",
            category=Category.ACTIONABLE,
            p_profit=0.0,
            prediction_time=datetime(2026, 8, 3, 8, 0, tzinfo=UTC),
            exit_due=date(2026, 8, 10),
        ),
        label_overrides=dict(exit_bid=5.20, realized_selected_pnl=5.20 / 4.86 - 1),
    )

    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    stats = report.actionable
    assert stats.n == 2
    assert stats.brier == pytest.approx(0.5)
    assert stats.calibration_n == 2


def test_product_selection_edge_regret_and_issuer_drag(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    realized = 5.10 / 4.86 - 1
    record_labeled(
        entry_overrides=dict(
            candidate_id="pse-1",
            category=Category.ACTIONABLE,
            prediction_time=datetime(2026, 8, 4, 8, 0, tzinfo=UTC),
            exit_due=date(2026, 8, 11),
        ),
        label_overrides=dict(
            exit_bid=5.10,
            realized_selected_pnl=realized,
            median_turbo_pnl=0.02,
            best_turbo_pnl=0.09,
            ideal_turbo_pnl=0.07,
        ),
    )

    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    stats = report.actionable
    assert stats.product_selection_edge == pytest.approx(realized - 0.02)
    assert stats.product_selection_regret == pytest.approx(0.09 - realized)
    assert stats.issuer_drag == pytest.approx(realized - 0.07)
    assert stats.product_selection_edge_n == 1


def test_actionable_and_shadow_are_computed_separately(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    record_labeled(
        entry_overrides=dict(
            candidate_id="act-1",
            category=Category.ACTIONABLE,
            is_shadow=False,
            prediction_time=datetime(2026, 8, 4, 8, 0, tzinfo=UTC),
            exit_due=date(2026, 8, 11),
        ),
        label_overrides=dict(exit_bid=5.10, realized_selected_pnl=5.10 / 4.86 - 1),
    )
    record_labeled(
        entry_overrides=dict(
            candidate_id="shadow-1",
            category=Category.REJECT,
            is_shadow=True,
            shadow_stratum="REJECT|long|unknown",
            prediction_time=datetime(2026, 8, 5, 8, 0, tzinfo=UTC),
            exit_due=date(2026, 8, 12),
        ),
        label_overrides=dict(exit_bid=4.50, realized_selected_pnl=4.50 / 4.86 - 1),
    )

    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    assert report.actionable.n == 1
    assert report.shadow.n == 1
    assert report.actionable.mean_return == pytest.approx(5.10 / 4.86 - 1)
    assert report.shadow.mean_return == pytest.approx(4.50 / 4.86 - 1)


def test_groupings_cover_all_required_dimensions(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    record_labeled(
        entry_overrides=dict(
            candidate_id="grp-1",
            category=Category.ACTIONABLE,
            underlying="DAX",
            issuer="HSBC",
            horizon_days=7,
            prediction_time=datetime(2026, 8, 4, 8, 0, tzinfo=UTC),
            exit_due=date(2026, 8, 11),
            feature_snapshot={"leverage": 6.0},
        ),
        label_overrides=dict(exit_bid=5.10, realized_selected_pnl=5.10 / 4.86 - 1),
    )

    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    dims = {g.dimension for g in report.groupings_actionable}
    assert dims == {
        "underlying",
        "direction",
        "horizon",
        "issuer",
        "leverage_bucket",
        "signal_family",
    }
    underlying_grouping = next(
        g for g in report.groupings_actionable if g.dimension == "underlying"
    )
    assert "DAX" in underlying_grouping.buckets
    leverage_grouping = next(
        g for g in report.groupings_actionable if g.dimension == "leverage_bucket"
    )
    assert "5-10x" in leverage_grouping.buckets


def test_deterministic_given_same_seed(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    for i in range(12):
        exit_bid = 5.10 if i % 2 == 0 else 4.60
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"det-{i}",
                category=Category.ACTIONABLE,
                prediction_time=datetime(2026, 8, 1 + i, 8, 0, tzinfo=UTC),
                exit_due=date(2026, 8, 8 + i),
            ),
            label_overrides=dict(exit_bid=exit_bid, realized_selected_pnl=exit_bid / 4.86 - 1),
        )

    cfg = MonthlyReportConfig(bootstrap_seed=777, n_bootstrap=100)
    r1 = build_monthly_report(store, month=_MONTH, as_of=_AS_OF, cfg=cfg)
    r2 = build_monthly_report(store, month=_MONTH, as_of=_AS_OF, cfg=cfg)
    assert r1.actionable.mean_return_ci == r2.actionable.mean_return_ci
