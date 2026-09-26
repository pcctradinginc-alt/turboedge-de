"""Tests for shared reporting helpers (`turboedge.reporting._common`)."""

from __future__ import annotations

from collections.abc import Callable

from turboedge.reporting._common import UNRESOLVED_SIGNAL_FAMILY, signal_family_for
from turboedge.storage.schemas import LedgerEntry


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
