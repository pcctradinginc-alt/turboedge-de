"""Tests for shared reporting helpers (`turboedge.reporting._common`)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pytest

from turboedge.reporting._common import (
    UNRESOLVED_SIGNAL_FAMILY,
    family_average_uniqueness_weights,
    signal_family_for,
    weighted_mean_return,
)
from turboedge.storage.schemas import LedgerEntry, LedgerLabel


def test_a_per_run_signal_id_is_not_treated_as_a_signal_family(
    make_ledger_entry: Callable[..., LedgerEntry],
) -> None:
    """A run identifier must not become a family (the 2026-09-26 tournament bug).

    `scan-all` writes `signal_id` as `<timestamp>-<hash>-<underlying>` and a
    `model_hash` for the *ensemble*, which is not in the registry. The old
    fallback returned that raw id, so every scan run became its own signal
    family: ~84 of them in one weekly report, Benjamini-Hochberg applied
    across 84 restatements of one strategy, and three "significant" families
    whose entries were all same-day positions from a single run.
    """
    entry_a = make_ledger_entry(
        signal_id="20260913T185445Z-ffe800dd60a3-NDX", model_hash="deadbeef"
    )
    entry_b = make_ledger_entry(
        signal_id="20260914T142925Z-68bdfc681105-NDX", model_hash="deadbeef"
    )

    family_a = signal_family_for(entry_a, {})
    family_b = signal_family_for(entry_b, {})

    assert family_a == UNRESOLVED_SIGNAL_FAMILY
    # Two different runs must land in ONE bucket, not two families.
    assert family_a == family_b


def test_registry_hash_still_wins(make_ledger_entry: Callable[..., LedgerEntry]) -> None:
    entry = make_ledger_entry(signal_id="20260913T185445Z-abc-NDX", model_hash="hash-1")
    assert signal_family_for(entry, {"hash-1": "tsmom"}) == "tsmom"


def test_version_suffix_fallback_still_works(
    make_ledger_entry: Callable[..., LedgerEntry],
) -> None:
    """The fix must not swallow the naming convention that did resolve correctly."""
    entry = make_ledger_entry(signal_id="tsmom_horizon_norm_v1", model_hash="not-registered")
    assert signal_family_for(entry, {}) == "tsmom_horizon_norm"


def _pair(
    make_ledger_entry: Callable[..., LedgerEntry],
    make_ledger_label: Callable[..., LedgerLabel],
    *,
    candidate_id: str,
    prediction_date: date,
    exit_due: date,
    underlying: str = "DAX",
    realized_pnl: float = 0.05,
) -> tuple[LedgerEntry, LedgerLabel]:
    entry = make_ledger_entry(
        candidate_id=candidate_id,
        prediction_time=datetime(
            prediction_date.year, prediction_date.month, prediction_date.day, 8, 0, tzinfo=UTC
        ),
        exit_due=exit_due,
        underlying=underlying,
    )
    label = make_ledger_label(entry_id=entry.entry_id, realized_selected_pnl=realized_pnl)
    return entry, label


def test_family_average_uniqueness_collapses_to_one_for_identical_windows(
    make_ledger_entry: Callable[..., LedgerEntry],
    make_ledger_label: Callable[..., LedgerLabel],
) -> None:
    """The docs/measured_results.md §6.9 case (2,672 rows, one shared label
    window, effective sample 1.00) at test scale: every entry shares the
    same ``(prediction_date, exit_due)`` window, so 50 rows collectively
    carry exactly one independent observation's worth of information."""
    pairs = [
        _pair(
            make_ledger_entry,
            make_ledger_label,
            candidate_id=f"same-{i}",
            prediction_date=date(2026, 8, 1),
            exit_due=date(2026, 8, 4),
        )
        for i in range(50)
    ]
    weights = family_average_uniqueness_weights(pairs)
    assert len(weights) == 50
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_family_average_uniqueness_approx_n_for_well_separated_windows(
    make_ledger_entry: Callable[..., LedgerEntry],
    make_ledger_label: Callable[..., LedgerLabel],
) -> None:
    """Non-overlapping label windows carry no shared information -- each
    entry keeps weight 1.0, so the effective sample equals the row count."""
    pairs = [
        _pair(
            make_ledger_entry,
            make_ledger_label,
            candidate_id=f"sep-{i}",
            prediction_date=date(2026, 1, 1) + timedelta(days=10 * i),
            exit_due=date(2026, 1, 1) + timedelta(days=10 * i + 2),
        )
        for i in range(20)
    ]
    weights = family_average_uniqueness_weights(pairs)
    assert sum(weights.values()) == pytest.approx(20.0, abs=1e-9)
    assert all(w == pytest.approx(1.0) for w in weights.values())


def test_family_average_uniqueness_groups_by_underlying(
    make_ledger_entry: Callable[..., LedgerEntry],
    make_ledger_label: Callable[..., LedgerLabel],
) -> None:
    """Same-day positions on two different underlyings must not dilute each
    other's weight -- concurrency is counted separately per underlying,
    matching ``average_uniqueness_weights``'s own grouping."""
    dax_pairs = [
        _pair(
            make_ledger_entry,
            make_ledger_label,
            candidate_id=f"dax-{i}",
            prediction_date=date(2026, 8, 1),
            exit_due=date(2026, 8, 4),
            underlying="DAX",
        )
        for i in range(10)
    ]
    ndx_pairs = [
        _pair(
            make_ledger_entry,
            make_ledger_label,
            candidate_id=f"ndx-{i}",
            prediction_date=date(2026, 8, 1),
            exit_due=date(2026, 8, 4),
            underlying="NDX",
        )
        for i in range(5)
    ]
    weights = family_average_uniqueness_weights(dax_pairs + ndx_pairs)
    dax_sum = sum(weights[e.entry_id] for e, _l in dax_pairs)
    ndx_sum = sum(weights[e.entry_id] for e, _l in ndx_pairs)
    assert dax_sum == pytest.approx(1.0, abs=1e-9)
    assert ndx_sum == pytest.approx(1.0, abs=1e-9)


def test_weighted_mean_return_matches_manual_computation(
    make_ledger_entry: Callable[..., LedgerEntry],
    make_ledger_label: Callable[..., LedgerLabel],
) -> None:
    pairs = [
        _pair(
            make_ledger_entry,
            make_ledger_label,
            candidate_id="a",
            prediction_date=date(2026, 1, 1),
            exit_due=date(2026, 1, 3),
            realized_pnl=0.10,
        ),
        _pair(
            make_ledger_entry,
            make_ledger_label,
            candidate_id="b",
            prediction_date=date(2026, 1, 1),
            exit_due=date(2026, 1, 3),
            realized_pnl=0.20,
        ),
    ]
    weights = family_average_uniqueness_weights(pairs)
    result = weighted_mean_return(pairs, weights)
    entry_a, _label_a = pairs[0]
    entry_b, _label_b = pairs[1]
    w_a, w_b = weights[entry_a.entry_id], weights[entry_b.entry_id]
    expected = (w_a * 0.10 + w_b * 0.20) / (w_a + w_b)
    assert result == pytest.approx(expected)


def test_weighted_mean_return_none_for_empty_pairs() -> None:
    assert weighted_mean_return([], {}) is None
