"""Candidate gating: ACTIONABLE / WATCH / REJECT / DATA_QUALITY.

Formula reference: Master Spec §19 ("Candidate Gates").

ACTIONABLE is technically reachable: it is assigned when ``lcb_ev > 0``,
``p_ko`` is not None, ``cluster_risk_pass`` is True, and the product's
Bezugsverhaeltnis (``ratio_reliability``) is not ``UNVERIFIED`` (Phase B,
CLAUDE.md rule 29 -- see ``GateInput.ratio_reliability`` and
``evaluate_gates``'s docstring). In practice, no ACTIONABLE candidate is
produced today because no forecast model has a measured out-of-sample
advantage over the null model (see docs/measured_results.md). Thresholds
are not lowered to manufacture suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass

from turboedge.pricing.integrity import IntegrityReport
from turboedge.storage.schemas import Category, FieldReliability


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Risk thresholds consumed by :func:`evaluate_gates` (``configs/risk.yaml``).

    ``max_source_quote_age_s`` and ``max_quote_age_at_decision_s`` are two
    deliberately separate freshness gates (Build Contract freshness/duration
    review, 2026-09-14) -- see ``configs/risk.yaml`` for the measured
    distributions behind each value:

    - ``max_source_quote_age_s`` gates :attr:`GateInput.source_quote_age_s`
      (``quote_timestamp`` vs. the product's own ``retrieved_at`` -- how old
      the quote already was when the source handed it to us). A strict,
      source-data-quality bound: independent of how long the rest of this
      run's fetch takes, so it does not loosen as the pipeline grows slower.
    - ``max_quote_age_at_decision_s`` gates :attr:`GateInput.quote_age_s`
      (``quote_timestamp`` vs. the scan's evaluation time -- how old the
      quote is when we actually act on it). Must stay above the pipeline's
      own realistic fetch duration, or every candidate fetched early in a
      multi-source scan rejects purely for having been fetched early, not
      for anything the source did wrong.
    """

    max_spread_pct: float
    max_source_quote_age_s: float
    max_quote_age_at_decision_s: float
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
            max_source_quote_age_s=risk_config.max_source_quote_age_s,  # type: ignore[attr-defined]
            max_quote_age_at_decision_s=risk_config.max_quote_age_at_decision_s,  # type: ignore[attr-defined]
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
    # Age (seconds) of `quote_timestamp` relative to THIS snapshot's own
    # `retrieved_at` -- i.e. how stale the quote already was when the source
    # handed it to us, independent of `quote_age_s`'s much larger
    # decision-time age (see GateThresholds' docstring). `None` under the
    # same condition as `quote_age_s` being `None` (no `quote_timestamp` at
    # all) -- `quote_timestamp_missing` already covers that case, so this
    # field defaults to `None` for backward compatibility with every
    # pre-existing caller/test that never heard of source-side freshness.
    source_quote_age_s: float | None = None
    # Phase B ("Produktstammdaten haerten"): field-level reliability of this
    # candidate's Bezugsverhaeltnis (`ProductSnapshot.ratio_reliability`).
    # `None` is deliberately NOT treated as `UNVERIFIED` -- it means "this
    # call site never populated the field" (every pre-existing test/caller
    # that never heard of this dimension, same backward-compatibility
    # convention as `has_ask`/`no_live_quote` above), not "known and
    # unreliable". Every real pipeline candidate (`pipeline/scan.py`) always
    # passes the product's own `ratio_reliability`, which itself defaults to
    # `UNVERIFIED` on `ProductSnapshot` when unknown -- so in production this
    # is effectively always populated; `None` only ever occurs in synthetic
    # test input.
    ratio_reliability: FieldReliability | None = None
    # --- the two economic preconditions (audit 2026-10-24) --------------
    # Both follow the `cluster_risk_pass` convention exactly: `None` means
    # "this call site never evaluated it", and ACTIONABLE requires `is
    # True`. So an unsupplied value can only ever withhold ACTIONABLE,
    # never grant it -- fail-closed by construction rather than by a
    # caller remembering to pass the right thing.
    #
    # Whether at least one alpha source in `alpha/schemas.PRODUCTION_STATES`
    # supports this candidate. Until this existed, ACTIONABLE rested on a
    # positive *modelled* `lcb_ev` plus technical gates, and the scan
    # pipeline never read `alpha_sources` at all: the entire promotion
    # ladder, its gates and `PRODUCTION_STATES` itself were defined,
    # exported and consulted by nothing. A simulated edge is not a
    # confirmed one, and CLAUDE.md's standing rule is that a gate is never
    # lowered to manufacture a trade proposal.
    confirmed_alpha_pass: bool | None = None
    # The suggested capital fraction for this candidate. A proposal that
    # cannot be sized is not a proposal: "act on this, with no capital" is
    # incoherent, and it became reachable once `max_cluster_fraction`
    # started binding (it was inert until 2026-10-24, because nothing
    # supplied `cluster_fraction_used`). Also reachable from the KO cap and
    # from a degenerate return distribution, both of which already returned
    # 0.0 before. `None` means the call site did not supply it and is not
    # treated as zero -- the same backward-compatibility convention as
    # `ratio_reliability`.
    suggested_position_fraction: float | None = None
    # Whether total portfolio risk could be determined AND is within the
    # configured caps, counting positions already held. `False` when the
    # capital base is unknown (`risk.research_capital_eur` unset), because
    # a portfolio limit that cannot be evaluated is not a limit that has
    # been satisfied.
    portfolio_risk_pass: bool | None = None


def evaluate_gates(inp: GateInput, th: GateThresholds) -> tuple[Category, list[str]]:
    """Classify one candidate into a :class:`Category` with the reasons why.

    Order of evaluation (Master Spec §19, Build Contract):

    1. ``DATA_QUALITY`` if the integrity check failed, or ``data_health_pass``
       is False.
    2. ``REJECT`` if ``bid_only``, ``knocked_out``, the quote timestamp is
       missing or stale (BOTH at the source -- ``source_quote_age_s >
       max_source_quote_age_s``, reason ``source_quote_stale`` -- AND at
       decision time -- ``quote_age_s > max_quote_age_at_decision_s``,
       reason ``quote_age_at_decision``; see ``GateThresholds``' docstring
       for why these are separate checks with separate thresholds), there is
       no live quote at all (or no ask quote specifically), the spread is
       too wide, leverage is outside the configured band, or the barrier
       distance (in sigma units) is too small. A stale/missing quote, a
       missing ask and a source-reported absence of any live quote are
       tradability gates, not data-integrity failures (Build Contract Task 2
       / Citi closing-price follow-up): ``pricing/integrity.check_product``
       only *warns* about them (or does not flag them at all when
       ``no_live_quote`` applies), so they reach this REJECT branch rather
       than being pre-empted by the DATA_QUALITY branch above.
    3. ``ACTIONABLE`` only if every other gate passed *and* ``lcb_ev is not
       None and lcb_ev > 0 and p_ko is not None and cluster_risk_pass is
       True and ratio_reliability != UNVERIFIED and confirmed_alpha_pass is
       True and portfolio_risk_pass is True``.

       The last two were added by the 2026-10-24 audit, and together they
       are the difference between "the model likes this" and "acting on it
       is defensible". A positive modelled ``lcb_ev`` is necessary and was
       previously sufficient: nothing in the scan pipeline read
       ``alpha_sources``, so an unconfirmed, DORMANT, REJECTED or DISABLED
       alpha -- or none at all -- placed no obstacle in the way, and the
       portfolio-level caps could not be checked against positions already
       held because the capital base they are expressed in was unset. Both
       fields default to ``None`` and ACTIONABLE requires ``is True``, so
       an unevaluated precondition withholds ACTIONABLE instead of being
       read as satisfied. The last condition is
       Phase B ("Produktstammdaten haerten", CLAUDE.md rule 29): a product
       whose Bezugsverhaeltnis was never reported by the source, or was
       derived without surviving independent verification, must never
       become ACTIONABLE no matter how good its EV/KO/cluster numbers look
       -- it can still be WATCH (this gate never removes a candidate from
       the ledger or the shadow sample, it only ever makes ACTIONABLE
       *harder* to reach, never easier). In practice, no ACTIONABLE
       candidate is produced today because no forecast model has measured
       out-of-sample edge (see docs/measured_results.md), so every
       surviving candidate falls through to ``WATCH`` regardless.
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
    else:
        if (
            inp.source_quote_age_s is not None
            and inp.source_quote_age_s > th.max_source_quote_age_s
        ):
            reject_reasons.append("source_quote_stale")
        if inp.quote_age_s > th.max_quote_age_at_decision_s:
            reject_reasons.append("quote_age_at_decision")
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

    ratio_verified = inp.ratio_reliability != FieldReliability.UNVERIFIED
    actionable = (
        inp.lcb_ev is not None
        and inp.lcb_ev > 0
        and inp.p_ko is not None
        and inp.cluster_risk_pass is True
        and ratio_verified
        and inp.confirmed_alpha_pass is True
        and inp.portfolio_risk_pass is True
        and (inp.suggested_position_fraction is None or inp.suggested_position_fraction > 0.0)
    )
    if actionable:
        return Category.ACTIONABLE, ["all_gates_passed"]

    watch_reasons: list[str] = []
    if inp.lcb_ev is None:
        watch_reasons.append("lcb_ev_not_evaluated")
    elif not (inp.lcb_ev > 0):
        watch_reasons.append("lcb_ev_not_positive")
    if inp.p_ko is None:
        watch_reasons.append("p_ko_not_evaluated")
    if inp.cluster_risk_pass is not True:
        watch_reasons.append("cluster_risk_not_confirmed")
    if not ratio_verified:
        watch_reasons.append("ratio_unverified")
    if inp.confirmed_alpha_pass is not True:
        watch_reasons.append("no_confirmed_alpha_source")
    if inp.portfolio_risk_pass is not True:
        watch_reasons.append("portfolio_risk_not_confirmed")
    if inp.suggested_position_fraction is not None and inp.suggested_position_fraction <= 0.0:
        watch_reasons.append("suggested_size_zero")
    if not watch_reasons:
        watch_reasons.append("watch_default")
    return Category.WATCH, watch_reasons
