"""Unconditional tests for ``pipeline.scan::_maybe_send_trade_proposals``.

The entire point of this repository is to email the user a trade proposal
when a candidate clears every gate (Master Spec §34). Nothing in the suite
exercised that unconditionally before this file: the one test that reaches
for it, ``test_scan_ev.py::test_run_scan_with_rng_writes_forecasts_and_ledger_and_can_reach_actionable``,
branches on ``if candidate.category == Category.ACTIONABLE`` and currently
takes the ``else`` branch, so the email path has never actually run under
test. ``ranking/gates.py::evaluate_gates`` is already verified correct
elsewhere -- what is missing is coverage of the step *after* the gate: does
an ACTIONABLE candidate reliably become a sent email, get deduplicated
correctly, and get a warning (never a silent drop) if its pricing/evaluation
data has gone missing by delivery time.

Every test here calls ``_maybe_send_trade_proposals`` directly with
hand-built ``_PricedProduct`` / ``ProductHorizonEvaluation`` /
``CandidateEvaluation`` instances, so none of them depend on a fixture
happening to clear every gate -- they fail if the delivery path itself
breaks, regardless of what any given scan's gates produce.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from turboedge.pipeline.scan import _PricedProduct, _maybe_send_trade_proposals
from turboedge.pricing.integrity import IntegrityReport
from turboedge.ranking.ev import ProductHorizonEvaluation
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    ProductSnapshot,
)

_EVAL_TIME = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


def _clock() -> datetime:
    return _EVAL_TIME


def _make_priced(
    dax_product_factory: Callable[..., ProductSnapshot],
    *,
    isin: str,
    wkn: str,
    issuer: str = "BankA",
    ask: float = 60.05,
    bid: float = 60.03,
) -> _PricedProduct:
    """A minimal but internally-consistent ``_PricedProduct`` for ``isin``."""
    product = dax_product_factory(
        isin=isin,
        wkn=wkn,
        issuer=issuer,
        direction=Direction.LONG,
        financing_level=18000.0,
        ratio=0.01,
        quote_timestamp=_EVAL_TIME,
    ).model_copy(update={"ask": ask, "bid": bid})
    mid = (ask + bid) / 2
    costs = CostDecomposition(
        ask=ask,
        bid=bid,
        mid=mid,
        intrinsic=60.0,
        trading_spread_component=ask - mid,
        fair_gap_premium=0.01,
        financing_drag=0.02,
        issuer_margin=0.01,
        spread_pct=(ask - bid) / ask,
        gap_premium_pct=0.0002,
        financing_drag_pct=0.0003,
        issuer_margin_pct=0.0002,
    )
    return _PricedProduct(
        product=product,
        bid=bid,
        ask=ask,
        spot=24000.0,
        fx=1.0,
        integrity=IntegrityReport(passed=True),
        leverage_value=4.0,
        leverage_bucket_value="3-5x",
        spread_pct_value=(ask - bid) / ask,
        quote_age_s=5.0,
        source_quote_age_s=5.0,
        distance_pct=0.25,
        distance_sigma=2.0,
        realized_spread=0.02,
        used_default_spread=False,
        financing_spread_source="realized_history",
        costs=costs,
        financing_cost_pct={"5d": 0.0004},
        gap_premium_over_horizon_pct=0.0002,
        liquidity=0.9,
    )


def _make_evaluation(
    *,
    isin: str,
    horizon_days: int = 5,
    lcb_ev: float = 0.0469,
    p_ko: float = 0.10,
    score: float = 1.0,
) -> ProductHorizonEvaluation:
    """A minimal ``ProductHorizonEvaluation`` for ``isin`` at ``horizon_days``."""
    return ProductHorizonEvaluation(
        isin=isin,
        underlying_id="DAX",
        direction=Direction.LONG,
        horizon_days=horizon_days,
        mean_net_return=0.05,
        median_net_return=0.045,
        q05=-0.02,
        q95=0.10,
        p_profit=0.65,
        p_ko=p_ko,
        es95=-0.03,
        mfe_median=0.06,
        mae_median=-0.01,
        mc_standard_error=0.001,
        shrunk_mean=0.04,
        shrinkage_intensity=0.2,
        lcb_net_return=lcb_ev,
        utility=0.03,
        liquidity_factor=0.9,
        score=score,
        suggested_position_fraction=0.02,
        cluster_id="single_DAX",
        reasons=["all_gates_passed"],
    )


def _make_candidate(
    *,
    isin: str,
    horizon_days: int = 5,
    lcb_ev: float = 0.0469,
) -> CandidateEvaluation:
    """A minimal ``CandidateEvaluation`` for ``isin``.

    ``category`` is set to ``ACTIONABLE`` for realism, but
    ``_maybe_send_trade_proposals`` itself never re-checks it -- it trusts
    ``newly_actionable`` to already be the ACTIONABLE subset, which is
    exactly why testing it directly (bypassing the gate) makes this
    deterministic.
    """
    return CandidateEvaluation(
        run_id="run-1",
        candidate_id=f"{isin}-{horizon_days}d",
        isin=isin,
        issuer="BankA",
        underlying_id="DAX",
        direction=Direction.LONG,
        category=Category.ACTIONABLE,
        reasons=["all_gates_passed"],
        financing_cost_horizon_pct={f"{horizon_days}d": 0.0004},
        integrity_passed=True,
        lcb_ev=lcb_ev,
    )


def test_actionable_candidate_sends_one_email_with_mandatory_disclosures(
    store: Store,
    make_notifier: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """The entire point of this repository is this path: an ACTIONABLE
    candidate must produce exactly one email, and the body must carry the
    §34 mandatory disclosures -- P(KO) is conservative/uncalibrated, no
    model beats the null ("Nullmodell"), and this is a manual-execution-only
    research proposal -- plus the isin/wkn so the recipient knows which
    product it names. A rendering regression that silently drops any one of
    these must fail here, not in a real inbox."""
    isin = "DE000FAVLNG1"
    priced = {isin: _make_priced(dax_product_factory, isin=isin, wkn="TESTWK1")}
    evaluations_by_isin = {isin: _make_evaluation(isin=isin)}
    candidate = _make_candidate(isin=isin)
    notifier = make_notifier()
    warnings: list[str] = []

    results = _maybe_send_trade_proposals(
        store=store,
        notifier=notifier,
        underlying_id="DAX",
        newly_actionable=[candidate],
        priced=priced,
        evaluations_by_isin=evaluations_by_isin,
        warnings=warnings,
        clock=_clock,
    )

    assert len(results) == 1
    assert notifier.send_calls == 1
    body = notifier.sent_specs[0].body_text
    assert "P(KO)" in body
    assert "Nullmodell" in body
    assert "manual execution only" in body
    assert isin in body
    assert "TESTWK1" in body
    assert warnings == []


def test_dedup_sends_once_then_sends_again_on_material_change(
    store: Store,
    make_notifier: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """Sending the identical candidate twice must send exactly once -- but a
    materially changed proposal must not be swallowed by that same
    deduplication.

    ``_maybe_send_trade_proposals`` rounds ``score``/``lcb`` to 4 places and
    ``p_ko`` to 3 before handing them to ``notification_hash``, which then
    rounds every float *again* to its own default of 3 places -- so the
    real dedup resolution for score/lcb is 3 decimal places, not 4. Going
    from ``lcb_ev=0.0469`` (rounds to 0.047) to ``lcb_ev=0.09`` (rounds to
    0.09) is a change no rounding step can absorb, so it must produce a
    second email.
    """
    isin = "DE000FAVLNG1"
    priced = {isin: _make_priced(dax_product_factory, isin=isin, wkn="TESTWK1")}
    notifier = make_notifier()
    warnings: list[str] = []

    candidate = _make_candidate(isin=isin, lcb_ev=0.0469)
    evaluations_by_isin = {isin: _make_evaluation(isin=isin, lcb_ev=0.0469)}

    first = _maybe_send_trade_proposals(
        store=store,
        notifier=notifier,
        underlying_id="DAX",
        newly_actionable=[candidate],
        priced=priced,
        evaluations_by_isin=evaluations_by_isin,
        warnings=warnings,
        clock=_clock,
    )
    assert len(first) == 1
    assert notifier.send_calls == 1

    # Identical candidate again: must dedup to zero new sends.
    repeat = _maybe_send_trade_proposals(
        store=store,
        notifier=notifier,
        underlying_id="DAX",
        newly_actionable=[candidate],
        priced=priced,
        evaluations_by_isin=evaluations_by_isin,
        warnings=warnings,
        clock=_clock,
    )
    assert repeat == []
    assert notifier.send_calls == 1

    # Materially changed lcb_ev (0.047 -> 0.09 after rounding): must send again.
    changed_candidate = _make_candidate(isin=isin, lcb_ev=0.09)
    changed_evaluations = {isin: _make_evaluation(isin=isin, lcb_ev=0.09)}
    second = _maybe_send_trade_proposals(
        store=store,
        notifier=notifier,
        underlying_id="DAX",
        newly_actionable=[changed_candidate],
        priced=priced,
        evaluations_by_isin=changed_evaluations,
        warnings=warnings,
        clock=_clock,
    )
    assert len(second) == 1
    assert notifier.send_calls == 2


def test_candidate_missing_priced_or_evaluation_warns_not_silently_dropped(
    store: Store,
    make_notifier: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """Regression test for the fix landed 2026-09-26: a candidate that
    cleared every gate and then vanished from ``priced`` or
    ``evaluations_by_isin`` before delivery used to hit a bare ``continue``
    -- dropped with no warning, no log line, indistinguishable from "no
    candidate was actionable today". It must now record a distinct warning
    for each of the two missing-data cases, never send an email for either,
    and never raise."""
    isin = "DE000FAVLNG1"
    candidate = _make_candidate(isin=isin)
    evaluations_by_isin = {isin: _make_evaluation(isin=isin)}
    priced = {isin: _make_priced(dax_product_factory, isin=isin, wkn="TESTWK1")}

    # Missing from `priced`.
    notifier_a = make_notifier()
    warnings_a: list[str] = []
    results_a = _maybe_send_trade_proposals(
        store=store,
        notifier=notifier_a,
        underlying_id="DAX",
        newly_actionable=[candidate],
        priced={},
        evaluations_by_isin=evaluations_by_isin,
        warnings=warnings_a,
        clock=_clock,
    )
    assert results_a == []
    assert notifier_a.send_calls == 0
    assert "actionable_dropped_missing_priced" in warnings_a

    # Missing from `evaluations_by_isin`.
    notifier_b = make_notifier()
    warnings_b: list[str] = []
    results_b = _maybe_send_trade_proposals(
        store=store,
        notifier=notifier_b,
        underlying_id="DAX",
        newly_actionable=[candidate],
        priced=priced,
        evaluations_by_isin={},
        warnings=warnings_b,
        clock=_clock,
    )
    assert results_b == []
    assert notifier_b.send_calls == 0
    assert "actionable_dropped_missing_evaluation" in warnings_b


def test_notifier_none_sends_nothing_and_raises_nothing(
    store: Store,
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """The dry-run path (no Gmail credentials configured, ``notifier=None``)
    must stay safe: no email, no exception, no warning -- even though a
    perfectly valid ACTIONABLE candidate with complete pricing/evaluation
    data is offered to it."""
    isin = "DE000FAVLNG1"
    priced = {isin: _make_priced(dax_product_factory, isin=isin, wkn="TESTWK1")}
    evaluations_by_isin = {isin: _make_evaluation(isin=isin)}
    candidate = _make_candidate(isin=isin)
    warnings: list[str] = []

    results = _maybe_send_trade_proposals(
        store=store,
        notifier=None,
        underlying_id="DAX",
        newly_actionable=[candidate],
        priced=priced,
        evaluations_by_isin=evaluations_by_isin,
        warnings=warnings,
        clock=_clock,
    )

    assert results == []
    assert warnings == []


def test_multiple_actionable_candidates_each_get_their_own_email(
    store: Store,
    make_notifier: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """Two distinct ACTIONABLE candidates in the same scan must each
    produce their own, independently-addressed email -- one must never
    overwrite or suppress the other."""
    isin_a = "DE000FAVLNG1"
    isin_b = "DE000FAVLNG2"
    priced = {
        isin_a: _make_priced(dax_product_factory, isin=isin_a, wkn="WKNAAAA"),
        isin_b: _make_priced(dax_product_factory, isin=isin_b, wkn="WKNBBBB"),
    }
    evaluations_by_isin = {
        isin_a: _make_evaluation(isin=isin_a),
        isin_b: _make_evaluation(isin=isin_b),
    }
    candidates = [_make_candidate(isin=isin_a), _make_candidate(isin=isin_b)]
    notifier = make_notifier()
    warnings: list[str] = []

    results = _maybe_send_trade_proposals(
        store=store,
        notifier=notifier,
        underlying_id="DAX",
        newly_actionable=candidates,
        priced=priced,
        evaluations_by_isin=evaluations_by_isin,
        warnings=warnings,
        clock=_clock,
    )

    assert len(results) == 2
    assert notifier.send_calls == 2
    bodies = [spec.body_text for spec in notifier.sent_specs]
    assert any(isin_a in body and "WKNAAAA" in body for body in bodies)
    assert any(isin_b in body and "WKNBBBB" in body for body in bodies)
    assert warnings == []
