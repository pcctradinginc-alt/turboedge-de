"""Tests for the frozen promotion criteria.

The whole safeguard behind evidence-gated automatic promotion is that a
candidate cannot change the bar it is measured against (GOVERNANCE.md §1.1a).
These tests exist to make that structural rather than conventional.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from turboedge.alpha.promotion import PromotionCriteria

_FROZEN = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)


def _criteria(**over: object) -> PromotionCriteria:
    defaults: dict[str, object] = {"criteria_version": "v1", "frozen_at": _FROZEN}
    defaults.update(over)
    return PromotionCriteria.model_validate(defaults)


def test_criteria_are_immutable() -> None:
    """A criteria set a caller can mutate in flight is not a gate.

    This is the single most important test in the file: automatic promotion is
    only defensible because the candidate cannot lower its own bar.
    """
    criteria = _criteria()

    with pytest.raises(ValidationError):
        criteria.min_absolute_net_ev_improvement = 0.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        criteria.block_on_single_regime_dependence = False  # type: ignore[misc]
    with pytest.raises(ValidationError):
        criteria.require_positive_lcb_net_ev = False  # type: ignore[misc]


def test_criteria_reject_unknown_fields() -> None:
    """A typo'd gate name must fail loudly rather than be silently ignored.

    `extra="forbid"` is what stops `block_on_single_regime_dependance=False`
    from looking like it disabled a gate while changing nothing.
    """
    with pytest.raises(ValidationError):
        PromotionCriteria.model_validate(
            {"criteria_version": "v1", "frozen_at": _FROZEN, "min_sharpe": 0.1}
        )


def test_thresholds_match_the_documented_ladder() -> None:
    """Every set number is GOVERNANCE.md §2's, not one chosen here.

    If the ladder changes, this test should fail and force the two to be
    reconciled deliberately rather than drifting apart.
    """
    criteria = _criteria()

    assert criteria.min_z_score_after_deflation == 1.645
    assert criteria.min_deflated_sharpe_ratio == 0.6
    assert criteria.min_probabilistic_sharpe_ratio == 0.95
    assert criteria.min_absolute_net_ev_improvement == 0.0010
    assert criteria.min_effective_sample == 100.0


def test_unmeasured_gates_are_unset_rather_than_defaulted() -> None:
    """An invented threshold in a promotion gate passes things; a missing one blocks.

    Seven risk and forward gates have no measured basis in this repository yet
    and must stay `None` until one exists.
    """
    criteria = _criteria()

    assert criteria.max_expected_shortfall is None
    assert criteria.max_ko_risk is None
    assert criteria.min_forward_shadow_days is None
    assert len(criteria.unset_gates) == 7


def test_criteria_report_themselves_as_incomplete() -> None:
    """Absence of a threshold must be visible, not read as permission."""
    assert _criteria().is_complete is False

    complete = _criteria(
        max_expected_shortfall=0.2,
        max_drawdown=0.2,
        max_ko_risk=0.1,
        max_alpha_correlation=0.7,
        max_issuer_concentration=0.5,
        min_data_quality=0.8,
        min_forward_shadow_days=60,
    )
    assert complete.is_complete is True
    assert complete.unset_gates == ()


def test_hard_blocks_default_to_blocking() -> None:
    """Leakage, governance violations, drift and single-regime dependence are
    not risks to be priced against a threshold -- they invalidate the evidence."""
    criteria = _criteria()

    assert criteria.block_on_unresolved_leakage
    assert criteria.block_on_governance_violation
    assert criteria.block_on_material_drift
    assert criteria.block_on_single_regime_dependence


def test_single_regime_block_exists_because_of_a_real_near_miss() -> None:
    """Trial 2026Q4-002 returned mean dLCB +0.0153 -- a PASS by its own frozen
    rule -- with DAX, NDX and XAU all negative and EURUSD alone at +0.1208.

    This gate is the one that would have caught it, so it defaults to on and a
    future edit that flips the default should have to explain itself here.
    """
    assert PromotionCriteria.model_fields["block_on_single_regime_dependence"].default is True


def test_criteria_carry_their_version_and_freeze_time() -> None:
    """A promotion must be auditable against the exact bar that applied."""
    criteria = _criteria(criteria_version="2026Q4-v1")

    assert criteria.criteria_version == "2026Q4-v1"
    assert criteria.frozen_at == _FROZEN
    assert "GOVERNANCE.md" in criteria.source


def test_no_evaluator_is_exported_in_phase_a() -> None:
    """Phase A ships the gate, not the thing that walks through it.

    Shipping both in one change would mean the first thing this object ever did
    was promote something.
    """
    import turboedge.alpha.promotion as module

    assert module.__all__ == ["PromotionCriteria"]
    defined_here = [
        name
        for name in dir(module)
        if not name.startswith("_")
        and callable(obj := getattr(module, name))
        and getattr(obj, "__module__", None) == module.__name__
    ]
    assert defined_here == ["PromotionCriteria"], (
        f"a definitions-only module gained something callable: {defined_here}"
    )
