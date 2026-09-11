"""Candidate gating: ACTIONABLE / WATCH / REJECT / DATA_QUALITY.

Formula reference: Master Spec §19 ("Candidate Gates").

Build Contract, binding for this milestone: ``ACTIONABLE`` is never assigned
(gate ``lcb_ev is None -> kein ACTIONABLE``) because ``LCB_EV`` requires the
path model that only exists from Phase 3/4 onward. This module still
implements the full ACTIONABLE gate so no behavior needs to change once
``lcb_ev``/``p_ko``/``cluster_risk_pass`` are wired up by that later phase --
in this milestone every call simply arrives with ``lcb_ev=None`` and falls
through to WATCH.
"""

from __future__ import annotations

from dataclasses import dataclass

from turboedge.pricing.integrity import IntegrityReport
from turboedge.storage.schemas import Category


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Risk thresholds consumed by :func:`evaluate_gates` (``configs/risk.yaml``)."""

    max_spread_pct: float
    max_quote_age_s: float
    min_leverage: float
    max_leverage: float
    min_distance_to_barrier_sigma: float

    @classmethod
    def from_risk_config(cls, risk_config: object) -> GateThresholds:
        """Build from any object exposing the matching ``risk.yaml`` attributes.

        Duck-typed against ``turboedge.config.RiskConfig`` (or a plain
        namespace/mock with the same field names) so this module does not
        need to import ``turboedge.config`` directly.
        """
        return cls(
            max_spread_pct=risk_config.max_spread_pct,  # type: ignore[attr-defined]
            max_quote_age_s=risk_config.max_quote_age_s,  # type: ignore[attr-defined]
            min_leverage=risk_config.min_leverage,  # type: ignore[attr-defined]
            max_leverage=risk_config.max_leverage,  # type: ignore[attr-defined]
            min_distance_to_barrier_sigma=risk_config.min_distance_to_barrier_sigma,  # type: ignore[attr-defined]
        )


@dataclass(frozen=True, slots=True)
class GateInput:
    """Per-candidate inputs to :func:`evaluate_gates`."""

    integrity: IntegrityReport
    bid_only: bool
    knocked_out: bool
    quote_age_s: float | None
    spread_pct: float | None
    leverage: float | None
    distance_to_barrier_sigma: float | None
    data_health_pass: bool
    lcb_ev: float | None
    p_ko: float | None
    cluster_risk_pass: bool | None
    # False when the product has no ask quote at all (e.g. an issuer quoting
    # only bid outside trading hours). Defaults to True so every pre-existing
    # caller/test that never heard of this scenario is unaffected. When
    # False, ``spread_pct``/``leverage`` are not evaluated (both require an
    # ask) -- the product is rejected on "no_ask_quote" alone instead of
    # being penalized a second time for the spread/leverage fields it could
    # never have computed in the first place.
    has_ask: bool = True
    # True when the source has explicitly reported no live two-way market at
    # all for this product (bid AND ask both missing, source-reported
    # ``quote_presence is False`` -- e.g. Citi's closing-price-only rows).
    # Defaults to False for backward compatibility. When True, the product is
    # rejected on "no_live_quote" alone (implies ``has_ask=False`` behavior:
    # spread_pct/leverage are not evaluated) instead of the less precise
    # "no_ask_quote" -- this is not a data-quality violation (master data
    # stays plausible), it is "this source has nothing tradable to quote".
    no_live_quote: bool = False


def evaluate_gates(inp: GateInput, th: GateThresholds) -> tuple[Category, list[str]]:
    """Classify one candidate into a :class:`Category` with the reasons why.

    Order of evaluation (Master Spec §19, Build Contract):

    1. ``DATA_QUALITY`` if the integrity check failed, or ``data_health_pass``
       is False.
    2. ``REJECT`` if ``bid_only``, ``knocked_out``, the quote timestamp is
       missing or stale, there is no live quote at all (or no ask quote
       specifically), the spread is too wide, leverage is outside the
       configured band, or the barrier distance (in sigma units) is too
       small. A stale/missing quote, a missing ask and a source-reported
       absence of any live quote are tradability gates, not data-integrity
       failures (Build Contract Task 2 / Citi closing-price follow-up):
       ``pricing/integrity.check_product`` only *warns* about them (or does
       not flag them at all when ``no_live_quote`` applies), so they reach
       this REJECT branch rather than being pre-empted by the DATA_QUALITY
       branch above.
    3. ``ACTIONABLE`` only if every other gate passed *and* ``lcb_ev is not
       None and lcb_ev > 0 and p_ko is not None and cluster_risk_pass is
       True`` -- in this milestone ``lcb_ev`` is always ``None`` (no path
       model yet), so this branch is unreachable in practice and every
       surviving candidate falls through to ``WATCH``.
    4. ``WATCH`` otherwise, with a reason naming which ACTIONABLE
       precondition is still missing.
    """
    if not inp.integrity.passed or not inp.data_health_pass:
        reasons = list(inp.integrity.failures)
        if not inp.data_health_pass:
            reasons.append("data_health_fail")
        if not reasons:
            reasons.append("data_quality_fail")
        return Category.DATA_QUALITY, reasons

    reject_reasons: list[str] = []
    if inp.bid_only:
        reject_reasons.append("bid_only")
    if inp.knocked_out:
        reject_reasons.append("knocked_out")
    if inp.quote_age_s is None:
        reject_reasons.append("quote_timestamp_missing")
    elif inp.quote_age_s > th.max_quote_age_s:
        reject_reasons.append("quote_stale")
    if inp.no_live_quote:
        reject_reasons.append("no_live_quote")
    elif not inp.has_ask:
        reject_reasons.append("no_ask_quote")
    else:
        if inp.spread_pct is None or inp.spread_pct > th.max_spread_pct:
            reject_reasons.append("spread_too_high")
        if inp.leverage is None or not (th.min_leverage <= inp.leverage <= th.max_leverage):
            reject_reasons.append("leverage_out_of_range")
    if (
        inp.distance_to_barrier_sigma is None
        or inp.distance_to_barrier_sigma < th.min_distance_to_barrier_sigma
    ):
        reject_reasons.append("barrier_distance_too_small")
    if reject_reasons:
        return Category.REJECT, reject_reasons

    actionable = (
        inp.lcb_ev is not None
        and inp.lcb_ev > 0
        and inp.p_ko is not None
        and inp.cluster_risk_pass is True
    )
    if actionable:
        return Category.ACTIONABLE, ["all_gates_passed"]

    watch_reasons: list[str] = []
    if inp.lcb_ev is None:
        watch_reasons.append("lcb_ev_unavailable_phase_lt_4")
    elif not (inp.lcb_ev > 0):
        watch_reasons.append("lcb_ev_not_positive")
    if inp.p_ko is None:
        watch_reasons.append("p_ko_unavailable_phase_lt_4")
    if inp.cluster_risk_pass is not True:
        watch_reasons.append("cluster_risk_not_confirmed")
    if not watch_reasons:
        watch_reasons.append("watch_default")
    return Category.WATCH, watch_reasons
