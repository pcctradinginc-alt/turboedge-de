from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.pricing.integrity import IntegrityReport
from turboedge.ranking.gates import GateInput, GateThresholds, evaluate_gates
from turboedge.storage.schemas import Category, FieldReliability

_THRESHOLDS = GateThresholds(
    max_spread_pct=0.03,
    max_source_quote_age_s=30.0,
    max_quote_age_at_decision_s=120.0,
    min_leverage=2.0,
    max_leverage=20.0,
    min_distance_to_barrier_sigma=1.0,
)

_PASSING_INTEGRITY = IntegrityReport(passed=True, failures=[], warnings=[])


def _base_input(**overrides: object) -> GateInput:
    defaults: dict[str, object] = dict(
        integrity=_PASSING_INTEGRITY,
        bid_only=False,
        knocked_out=False,
        quote_age_s=10.0,
        spread_pct=0.01,
        leverage=5.0,
        distance_to_barrier_sigma=2.0,
        data_health_pass=True,
        lcb_ev=None,
        p_ko=None,
        cluster_risk_pass=None,
    )
    defaults.update(overrides)
    return GateInput(**defaults)  # type: ignore[arg-type]


def test_gate_thresholds_from_risk_config() -> None:
    @dataclass
    class FakeRiskConfig:
        max_spread_pct: float = 0.03
        max_source_quote_age_s: float = 30.0
        max_quote_age_at_decision_s: float = 120.0
        min_leverage: float = 2.0
        max_leverage: float = 20.0
        min_distance_to_barrier_sigma: float = 1.0
        # extra field a real RiskConfig also carries; from_risk_config must
        # ignore it rather than choke on it.
        default_financing_spread: float = 0.025

    th = GateThresholds.from_risk_config(FakeRiskConfig())
    assert th == _THRESHOLDS


def test_data_quality_on_integrity_failure() -> None:
    failing_integrity = IntegrityReport(passed=False, failures=["missing_ask"], warnings=[])
    category, reasons = evaluate_gates(_base_input(integrity=failing_integrity), _THRESHOLDS)
    assert category == Category.DATA_QUALITY
    assert "missing_ask" in reasons


def test_data_quality_on_data_health_fail() -> None:
    category, reasons = evaluate_gates(_base_input(data_health_pass=False), _THRESHOLDS)
    assert category == Category.DATA_QUALITY
    assert "data_health_fail" in reasons


def test_reject_bid_only() -> None:
    category, reasons = evaluate_gates(_base_input(bid_only=True), _THRESHOLDS)
    assert category == Category.REJECT
    assert "bid_only" in reasons


def test_reject_knocked_out() -> None:
    category, reasons = evaluate_gates(_base_input(knocked_out=True), _THRESHOLDS)
    assert category == Category.REJECT
    assert "knocked_out" in reasons


def test_reject_stale_quote_at_decision() -> None:
    category, reasons = evaluate_gates(_base_input(quote_age_s=999.0), _THRESHOLDS)
    assert category == Category.REJECT
    assert "quote_age_at_decision" in reasons
    assert "source_quote_stale" not in reasons


def test_reject_source_quote_stale() -> None:
    # Fresh at decision time (well under max_quote_age_at_decision_s) but the
    # source itself had already handed us a quote older than
    # max_source_quote_age_s -- a data-quality signal distinct from
    # quote_age_at_decision, and must not be conflated with it.
    category, reasons = evaluate_gates(
        _base_input(quote_age_s=45.0, source_quote_age_s=45.0), _THRESHOLDS
    )
    assert category == Category.REJECT
    assert "source_quote_stale" in reasons
    assert "quote_age_at_decision" not in reasons


def test_source_quote_age_none_defaults_to_no_source_staleness_check() -> None:
    # Backward compatibility: a caller that never populates source_quote_age_s
    # (default None) gets no source_quote_stale rejection, only the existing
    # decision-time check.
    category, reasons = evaluate_gates(_base_input(quote_age_s=10.0), _THRESHOLDS)
    assert "source_quote_stale" not in reasons
    assert category != Category.REJECT


def test_reject_missing_quote_timestamp() -> None:
    category, reasons = evaluate_gates(_base_input(quote_age_s=None), _THRESHOLDS)
    assert category == Category.REJECT
    assert "quote_timestamp_missing" in reasons


def test_reject_no_ask_quote_skips_spread_and_leverage_checks() -> None:
    category, reasons = evaluate_gates(_base_input(has_ask=False), _THRESHOLDS)
    assert category == Category.REJECT
    assert "no_ask_quote" in reasons
    # spread_pct/leverage are meaningless without an ask -- they must not be
    # evaluated (and thus must not add spurious extra REJECT reasons).
    assert "spread_too_high" not in reasons
    assert "leverage_out_of_range" not in reasons


def test_has_ask_defaults_true_for_backward_compatibility() -> None:
    category, reasons = evaluate_gates(_base_input(), _THRESHOLDS)
    assert "no_ask_quote" not in reasons
    assert category != Category.REJECT


def test_reject_no_live_quote_skips_spread_and_leverage_checks() -> None:
    category, reasons = evaluate_gates(_base_input(has_ask=False, no_live_quote=True), _THRESHOLDS)
    assert category == Category.REJECT
    assert "no_live_quote" in reasons
    assert "spread_too_high" not in reasons
    assert "leverage_out_of_range" not in reasons


def test_no_live_quote_takes_precedence_over_no_ask_quote_reason() -> None:
    # Both signal "no ask" (has_ask=False); when the source explicitly
    # reported no live quote at all, the more precise "no_live_quote" reason
    # is used instead of the generic "no_ask_quote".
    category, reasons = evaluate_gates(_base_input(has_ask=False, no_live_quote=True), _THRESHOLDS)
    assert category == Category.REJECT
    assert "no_live_quote" in reasons
    assert "no_ask_quote" not in reasons


def test_no_live_quote_defaults_false_for_backward_compatibility() -> None:
    category, reasons = evaluate_gates(_base_input(), _THRESHOLDS)
    assert "no_live_quote" not in reasons
    assert category != Category.REJECT


def test_reject_wide_spread() -> None:
    category, reasons = evaluate_gates(_base_input(spread_pct=0.10), _THRESHOLDS)
    assert category == Category.REJECT
    assert "spread_too_high" in reasons


def test_reject_leverage_out_of_range() -> None:
    category, reasons = evaluate_gates(_base_input(leverage=1.0), _THRESHOLDS)
    assert category == Category.REJECT
    assert "leverage_out_of_range" in reasons

    category, reasons = evaluate_gates(_base_input(leverage=50.0), _THRESHOLDS)
    assert category == Category.REJECT
    assert "leverage_out_of_range" in reasons


def test_reject_barrier_too_close() -> None:
    category, reasons = evaluate_gates(_base_input(distance_to_barrier_sigma=0.1), _THRESHOLDS)
    assert category == Category.REJECT
    assert "barrier_distance_too_small" in reasons


def test_watch_when_lcb_ev_missing() -> None:
    category, reasons = evaluate_gates(_base_input(), _THRESHOLDS)
    assert category == Category.WATCH
    assert "lcb_ev_not_evaluated" in reasons


def test_watch_when_lcb_ev_present_but_other_actionable_inputs_missing() -> None:
    category, reasons = evaluate_gates(_base_input(lcb_ev=1.5), _THRESHOLDS)
    assert category == Category.WATCH
    assert "p_ko_not_evaluated" in reasons
    assert "cluster_risk_not_confirmed" in reasons


def test_actionable_only_when_every_precondition_met() -> None:
    category, reasons = evaluate_gates(
        _base_input(lcb_ev=1.5, p_ko=0.1, cluster_risk_pass=True), _THRESHOLDS
    )
    assert category == Category.ACTIONABLE
    assert reasons


def test_never_actionable_without_lcb_ev_even_with_everything_else_present() -> None:
    category, _reasons = evaluate_gates(
        _base_input(lcb_ev=None, p_ko=0.1, cluster_risk_pass=True), _THRESHOLDS
    )
    assert category != Category.ACTIONABLE
    assert category == Category.WATCH


# ===========================================================================
# Phase B ("Produktstammdaten haerten"): ratio_reliability gate
# ===========================================================================


def test_actionable_when_ratio_reliability_not_populated_backward_compatible() -> None:
    # `ratio_reliability` defaults to `None` on `GateInput` -- a caller that
    # never heard of this dimension (every pre-existing test/caller) is
    # unaffected, same convention as `has_ask`/`no_live_quote` above.
    category, _reasons = evaluate_gates(
        _base_input(lcb_ev=1.5, p_ko=0.1, cluster_risk_pass=True), _THRESHOLDS
    )
    assert category == Category.ACTIONABLE


def test_ratio_unverified_prevents_actionable_but_allows_watch() -> None:
    # Kernsatz: a product whose Bezugsverhaeltnis is not reliably verified
    # must never become ACTIONABLE -- but it is not REJECTed or dropped
    # either; it still reaches WATCH (stays in the ledger/shadow sample).
    category, reasons = evaluate_gates(
        _base_input(
            lcb_ev=1.5,
            p_ko=0.1,
            cluster_risk_pass=True,
            ratio_reliability=FieldReliability.UNVERIFIED,
        ),
        _THRESHOLDS,
    )
    assert category == Category.WATCH
    assert "ratio_unverified" in reasons


def test_ratio_unverified_never_actionable_even_with_every_other_precondition_met() -> None:
    category, _reasons = evaluate_gates(
        _base_input(
            lcb_ev=1.5,
            p_ko=0.1,
            cluster_risk_pass=True,
            ratio_reliability=FieldReliability.UNVERIFIED,
        ),
        _THRESHOLDS,
    )
    assert category != Category.ACTIONABLE


@pytest.mark.parametrize(
    "reliability",
    [
        FieldReliability.SOURCE_REPORTED,
        FieldReliability.CROSS_SOURCE_VERIFIED,
        FieldReliability.DERIVED_VERIFIED,
    ],
)
def test_every_non_unverified_ratio_reliability_level_allows_actionable(
    reliability: FieldReliability,
) -> None:
    category, _reasons = evaluate_gates(
        _base_input(lcb_ev=1.5, p_ko=0.1, cluster_risk_pass=True, ratio_reliability=reliability),
        _THRESHOLDS,
    )
    assert category == Category.ACTIONABLE


@given(
    lcb_ev=st.one_of(st.none(), st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)),
    p_ko=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    cluster_risk_pass=st.one_of(st.none(), st.booleans()),
)
def test_property_never_actionable_when_lcb_ev_is_none(
    lcb_ev: float | None, p_ko: float | None, cluster_risk_pass: bool | None
) -> None:
    category, _reasons = evaluate_gates(
        _base_input(lcb_ev=lcb_ev, p_ko=p_ko, cluster_risk_pass=cluster_risk_pass), _THRESHOLDS
    )
    if lcb_ev is None:
        assert category != Category.ACTIONABLE


@given(
    lcb_ev=st.one_of(st.none(), st.floats(min_value=0.01, max_value=10.0, allow_nan=False)),
    p_ko=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    cluster_risk_pass=st.one_of(st.none(), st.booleans()),
)
def test_property_never_actionable_when_ratio_unverified(
    lcb_ev: float | None, p_ko: float | None, cluster_risk_pass: bool | None
) -> None:
    """Phase B: no combination of otherwise-passing EV/KO/cluster inputs can
    make an UNVERIFIED-ratio candidate ACTIONABLE -- this gate only ever
    makes ACTIONABLE harder to reach, never easier."""
    category, _reasons = evaluate_gates(
        _base_input(
            lcb_ev=lcb_ev,
            p_ko=p_ko,
            cluster_risk_pass=cluster_risk_pass,
            ratio_reliability=FieldReliability.UNVERIFIED,
        ),
        _THRESHOLDS,
    )
    assert category != Category.ACTIONABLE
