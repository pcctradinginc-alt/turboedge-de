"""Regression tests for paired diagnostics."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest

from turboedge.backtest.ablation import compare
from turboedge.backtest.walkforward import WalkForwardResult
from turboedge.research.filter_evaluation import (
    evaluate_filters,
    evaluate_gate_replay,
    evaluate_lcb,
)


def _result(p: float, ret: float = 0.01) -> WalkForwardResult:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return cast(
        WalkForwardResult,
        SimpleNamespace(
            horizon_days=3,
            oos_predictions=[(start + timedelta(days=i), p, ret) for i in range(100)],
        ),
    )


def test_paired_gain_and_multiplicity() -> None:
    result = compare(_result(0.8), _result(0.4), tests=70)
    assert result.matched_rows == 100
    assert result.brier_gain == pytest.approx(0.32)
    assert result.gross_return_gain == pytest.approx(0.02)
    assert result.brier_interval is not None


def test_mismatched_outcome_fails_closed() -> None:
    with pytest.raises(ValueError, match="outcome mismatch"):
        compare(_result(0.8), _result(0.4, 0.02))


def test_empty_ledger_does_not_invent_economics() -> None:
    result = evaluate_filters([], as_of=datetime(2026, 11, 26, tzinfo=UTC))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["eligible_rows"] == 0
    assert result["effects"] == []
    assert result["causal_filter_ablation"] is False


def test_empty_lcb_has_no_coverage_claim() -> None:
    result = evaluate_lcb([], as_of=datetime(2026, 11, 26, tzinfo=UTC))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["formal_coverage_claim"] is False


def test_gate_replay_requires_archived_decisions() -> None:
    result = evaluate_gate_replay([], as_of=datetime(2026, 11, 26, tzinfo=UTC))
    assert result["status"] == "NO_ARCHIVED_GATE_INPUTS"
    assert result["promotion_allowed"] is False


def test_gate_replay_costs_and_missing_outcomes(make_ledger_entry) -> None:
    from turboedge.storage.schemas import Category, ExitReason, LedgerLabel

    cutoff = datetime(2026, 11, 26, tzinfo=UTC)
    entry = make_ledger_entry(
        prediction_time=cutoff - timedelta(days=10),
        exit_due=(cutoff - timedelta(days=5)).date(),
        category=Category.WATCH,
        is_shadow=True,
        feature_snapshot={
            "gate_archive_v1": 1.0,
            "gate_failure_count": 1.0,
            "gate_reason:lcb_ev_not_positive": 1.0,
        },
    )
    label = LedgerLabel(
        entry_id=entry.entry_id,
        labeled_at=cutoff - timedelta(days=1),
        exit_reason=ExitReason.HORIZON,
        realized_selected_pnl=0.1,
        ko_hit=False,
        ambiguous_path=False,
    )
    result = evaluate_gate_replay([(entry, label)], as_of=cutoff)
    assert result["status"] == "DESCRIPTIVE_ONLY"
    missing = evaluate_gate_replay([(entry, None)], as_of=cutoff)
    assert missing["status"] == "INCOMPLETE_OUTCOMES"
    multiple = entry.model_copy(
        update={"feature_snapshot": {**entry.feature_snapshot, "gate_failure_count": 2.0}}
    )
    assert evaluate_gate_replay([(multiple, label)], as_of=cutoff)["status"] == "NO_AFFECTED_ROWS"
