"""Regression tests for paired diagnostics."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest

from turboedge.backtest.ablation import compare
from turboedge.backtest.walkforward import WalkForwardResult
from turboedge.research.filter_evaluation import evaluate_filters


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
