"""Scan pipeline: source health -> signal freeze -> quotes -> pricing -> gates.

Formula/order reference: Build Contract "Scan-Pipeline" (pipeline/scan.py
section) and Master Spec §3.1 ("Signal und Produkt strikt trennen") / §43
("Scanner Pipeline"). The step order below is binding, not incidental:

    1. source health (product adapters + reference sources), persisted
    2. underlying daily bars, persisted; only bars with
       available_at <= prediction_time are ever used (no look-ahead,
       CLAUDE.md rules 4/5)
    3. TSMOM baseline + EWMA volatility + gap distribution
    4. SignalSnapshot frozen and persisted -- BEFORE any product/quote fetch
    5. product quotes fetched via pipeline.universe.run_universe (per-source
       failures degrade gracefully, never abort the scan)
    6. integrity, consensus spot, cost decomposition, financing spread,
       gap premium, cross-issuer scores, liquidity -- per product, with a
       single product's failure downgrading only that product to
       DATA_QUALITY, never aborting the scan
    7. candidate gates (ACTIONABLE is technically reachable; no ACTIONABLE
       candidates are produced today due to lack of measured forecast edge) and ranking
    8. candidates persisted; optional deduplicated Gmail report

Every external dependency (product adapters, the underlying-price source, the
reference-rate source, healthchecks, the notifier, the clock) is injected as
a parameter so the whole pipeline is testable with fakes and needs no real
network access.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import structlog

from turboedge.adapters.base import HealthCheckResult, ProductFetchContext
from turboedge.adapters.registry import ProductSourceAdapter
from turboedge.config import TurboEdgeConfig, build_ev_config, config_hash
from turboedge.features.product import (
    distance_to_barrier,
    ewma_volatility,
    freshness_score,
    leverage_bucket,
    quote_age_seconds,
)
from turboedge.features.product import spread_pct as feature_spread_pct
from turboedge.learning.ledger import ForwardLedger, select_shadow_sample
from turboedge.learning.registry import ModelRegistry
from turboedge.models.ensemble import combine_forecasts
from turboedge.models.forecast import ForecastModel, HorizonForecast, build_default_models
from turboedge.models.protected_baseline import (
    TsmomConfig,
    TsmomResult,
    compute_tsmom,
    signal_version_hash,
)
from turboedge.monitoring.source_health import from_healthcheck
from turboedge.notifications.dedup import NotificationDeduplicator, notification_hash
from turboedge.notifications.gmail import (
    EmailMessageSpec,
    GmailNotifier,
    NotificationError,
    SendResult,
)
from turboedge.notifications.templates import (
    ScanReportContext,
    ScanReportRow,
    TradeProposalContext,
    render_scan_report,
    render_trade_proposal,
)
from turboedge.pipeline.universe import UniverseResult, run_universe
from turboedge.pricing.cross_issuer import (
    CrossIssuerInput,
    CrossIssuerScores,
    consensus_spot,
    cross_issuer_scores,
)
from turboedge.pricing.fair_value import dividend_yield_for_underlying, theoretical_fair_value
from turboedge.pricing.financing import (
    financing_cost_over_horizon,
    financing_spread_history,
    realized_financing_spread,
)
from turboedge.pricing.fx_resolution import FxResolution, fx_by_isin, resolve_fx_by_issuer
from turboedge.pricing.gap_premium import (
    GapDistribution,
    fair_gap_premium,
    gap_distribution_from_bars,
    gap_premium_over_horizon,
)
from turboedge.pricing.integrity import IntegrityReport, check_product
from turboedge.pricing.intrinsic import leverage as compute_leverage
from turboedge.pricing.issuer_margin import decompose_ask
from turboedge.provenance import data_snapshot_hash, git_commit, sha256_json
from turboedge.ranking.cluster import ClusterConfig, OpenClusterPosition
from turboedge.ranking.cluster import cluster_risk as compute_cluster_risk
from turboedge.ranking.cluster import cluster_risk_pass as compute_cluster_risk_pass
from turboedge.ranking.ev import ProductHorizonEvaluation, evaluate_product_horizons
from turboedge.ranking.gates import GateInput, GateThresholds, evaluate_gates
from turboedge.ranking.liquidity import liquidity_factor, quote_size_coverage, spread_quality
from turboedge.ranking.shrinkage import leverage_bucket_for
from turboedge.simulation.payoff import ProductTerms
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    ForecastRecord,
    HealthStatus,
    LedgerEntry,
    LedgerEntryStatus,
    ModelStatus,
    ProductSnapshot,
    ProductType,
    SignalSnapshot,
    SourceHealthRecord,
    UnderlyingBar,
)
from turboedge.universe.underlying_map import get_underlying_meta

logger = structlog.get_logger(__name__)

_VALID_HORIZONS: tuple[int, ...] = (3, 5, 7, 10, 14)
_MIN_BARS_FOR_VOLATILITY = 2
_BERLIN_TZ = ZoneInfo("Europe/Berlin")
_FRIDAY_WEEKDAY = 4
_CALENDAR_DAYS_PER_TRADING_DAY = 7.0 / 5.0
# Routine, budget-free trial id used for every ordinary scan's ledger entries
# (Master Spec §27.1's per-trial "TR-..." ids are for RESEARCH CHANGES --
# see learning/trials.py -- minting one per scan candidate would exhaust the
# quarterly adaptation budget instantly and mean something it does not: this
# scan did not change any feature/model/threshold, it just applied the
# already-approved current configuration).
_ROUTINE_TRIAL_ID = "TR-ROUTINE-SCAN"
# W4's measurement (Contract v3 coordinator note): no forecast model beats
# the null model out of sample yet (Brier worse in 20/20 index-horizon
# combinations tested). Every ACTIONABLE mail must carry this disclosure
# alongside the P(KO)-is-conservative one, for as long as that remains true.
_NO_MODEL_BEATS_NULL_DISCLOSURE = (
    "Bislang hat KEINE getestete Signalfamilie einen gemessenen "
    "out-of-sample-Vorteil (W4: die Kern-Prognosemodelle schlagen das "
    "Nullmodell in 20/20 getesteten Index/Horizont-Kombinationen nicht, "
    "Brier-Score jeweils schlechter; W9: 6 weitere Signalfamilien, 80 "
    "Walk-Forward-Zellen, 0 signifikant nach Benjamini-Hochberg-Korrektur, "
    "keine übersteht realistische Turbo-Kosten). Dieser Vorschlag beruht "
    "auf dem aktuellen Ensemble trotzdem, weil alle Gates (LCB(EV)>0, "
    "P(KO) bekannt, Cluster-Risiko im Limit) bestanden wurden -- er ist "
    "nicht durch einen erwiesenen Prognosevorteil gedeckt."
)


class PriceSource(Protocol):
    """Structural contract for the underlying-price dependency (YFinancePriceAdapter-compatible)."""

    def fetch_daily_bars(
        self, underlying_id: str, *, lookback_days: int | None = None
    ) -> list[UnderlyingBar]: ...


class EstrSource(Protocol):
    """Structural contract for the reference-rate dependency (EcbEstrAdapter-compatible)."""

    def get_estr(self) -> float: ...


# --------------------------------------------------------------------------
# Options / result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanOptions:
    """User-facing scan parameters (``turboedge scan`` CLI options)."""

    underlying_id: str
    direction: Direction | None = None
    horizon_days: int = 7
    top: int = 20
    email: bool = False

    def __post_init__(self) -> None:
        if self.horizon_days not in _VALID_HORIZONS:
            raise ValueError(
                f"horizon_days must be one of {_VALID_HORIZONS}, got {self.horizon_days!r}"
            )
        if self.top <= 0:
            raise ValueError(f"top must be > 0, got {self.top!r}")


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one :func:`run_scan` call."""

    run_id: str
    signal: SignalSnapshot | None
    candidates: list[CandidateEvaluation]
    counts: dict[Category, int]
    warnings: list[str]
    health: list[SourceHealthRecord]
    notification: SendResult | None
    # -- Contract v3 integration wave: EV pipeline outcome (empty/zero when
    # `run_scan(..., run_ev=False)`, i.e. the EV step never ran -- see run_scan).
    ledger_entries_written: int = 0
    forecasts_written: int = 0
    actionable_notifications: list[SendResult] = field(default_factory=list)
    new_cluster_positions: list[OpenClusterPosition] = field(default_factory=list)


# --------------------------------------------------------------------------
# small internal helpers
# --------------------------------------------------------------------------


def _add_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _candidate_id(isin: str, underlying_id: str, direction: Direction, horizon_days: int) -> str:
    """Stable candidate id: ``sha256(isin|underlying|direction|horizon)[:16]``."""
    payload = f"{isin}|{underlying_id}|{direction.value}|{horizon_days}d"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _is_friday_in_berlin(when: datetime) -> bool:
    return when.astimezone(_BERLIN_TZ).weekday() == _FRIDAY_WEEKDAY


def _safe_healthcheck(
    source_name: str, check: Callable[[], HealthCheckResult], now: datetime
) -> HealthCheckResult:
    """Run a healthcheck callable, converting any raised exception into a FAIL result.

    Every concrete adapter in this codebase already catches its own errors
    inside ``healthcheck()``; this is defense-in-depth for adapters built by
    other agents/milestones that might not.
    """
    try:
        return check()
    except Exception as exc:
        logger.error("healthcheck_raised", source=source_name, error=str(exc))
        return HealthCheckResult(
            source=source_name,
            status=HealthStatus.FAIL,
            ok=False,
            latency_ms=None,
            checked_at=now,
            message=f"healthcheck raised: {exc}",
        )


def _data_quality_candidate(
    *,
    run_id: str,
    product: ProductSnapshot,
    underlying_id: str,
    horizon_days: int,
    reasons: list[str],
) -> CandidateEvaluation:
    return CandidateEvaluation(
        run_id=run_id,
        candidate_id=_candidate_id(product.isin, underlying_id, product.direction, horizon_days),
        isin=product.isin,
        wkn=product.wkn,
        issuer=product.issuer,
        underlying_id=underlying_id,
        direction=product.direction,
        category=Category.DATA_QUALITY,
        reasons=reasons or ["data_quality_fail"],
        leverage=None,
        leverage_bucket=None,
        distance_to_barrier_pct=None,
        distance_to_barrier_sigma=None,
        costs=None,
        realized_financing_spread=None,
        financing_cost_horizon_pct={},
        cross_issuer_residual_zscore=None,
        issuer_markup_score=None,
        quote_dislocation_score=None,
        wrapper_edge=None,
        liquidity_factor=None,
        integrity_passed=False,
        lcb_ev=None,
        cost_rank_score=None,
    )


def _gate_only_candidate(
    *,
    run_id: str,
    product: ProductSnapshot,
    underlying_id: str,
    horizon_days: int,
    integrity: IntegrityReport,
    quote_age_s: float | None,
    distance_pct: float | None,
    distance_sigma: float | None,
    data_health_pass: bool,
    risk: object,
    has_ask: bool,
    no_live_quote: bool = False,
) -> CandidateEvaluation:
    """Build a candidate whose category comes purely from :func:`evaluate_gates`,
    with none of the ask-dependent pricing steps (decompose_ask, leverage,
    spread) run -- used for a product that has no ask at all (``has_ask=False``)
    and/or no live quote at all (``no_live_quote=True``, e.g. Citi's
    closing-price-only rows). Both scenarios skip the same pricing steps;
    ``evaluate_gates`` picks the more precise "no_live_quote" REJECT reason
    over "no_ask_quote" when both would otherwise apply.
    """
    thresholds = GateThresholds.from_risk_config(risk)
    gate_input = GateInput(
        integrity=integrity,
        bid_only=product.bid_only,
        knocked_out=product.knocked_out,
        quote_age_s=quote_age_s,
        spread_pct=None,
        leverage=None,
        distance_to_barrier_sigma=distance_sigma,
        data_health_pass=data_health_pass,
        lcb_ev=None,
        p_ko=None,
        cluster_risk_pass=None,
        has_ask=has_ask,
        no_live_quote=no_live_quote,
    )
    category, gate_reasons = evaluate_gates(gate_input, thresholds)
    return CandidateEvaluation(
        run_id=run_id,
        candidate_id=_candidate_id(product.isin, underlying_id, product.direction, horizon_days),
        isin=product.isin,
        wkn=product.wkn,
        issuer=product.issuer,
        underlying_id=underlying_id,
        direction=product.direction,
        category=category,
        reasons=list(gate_reasons),
        leverage=None,
        leverage_bucket=None,
        distance_to_barrier_pct=distance_pct,
        distance_to_barrier_sigma=distance_sigma,
        costs=None,
        realized_financing_spread=None,
        financing_cost_horizon_pct={},
        cross_issuer_residual_zscore=None,
        issuer_markup_score=None,
        quote_dislocation_score=None,
        wrapper_edge=None,
        liquidity_factor=None,
        integrity_passed=integrity.passed,
        lcb_ev=None,
        cost_rank_score=None,
    )


def _resolve_spot(
    product: ProductSnapshot,
    consensus_value: float | None,
    max_quote_age_s: float,
    spot_ref_max_deviation_pct: float,
    now: datetime,
    warnings: list[str],
) -> float | None:
    """Spot for pricing: cross-issuer consensus, with ``underlying_price_ref``
    only as a validated override.

    Build Contract BEFUND 1 (measured on a live BNP+Citi DAX scan,
    2026-09-11, ``docs/data_sources.md``): the previous rule used
    ``product.quote_timestamp`` (the product's own bid/ask timestamp) as a
    freshness proxy for ``underlying_price_ref``. That is wrong -- BNP's
    ``first.price`` batches/throttles independently of (and less frequently
    than) individual product bid/ask ticks, so a fresh quote timestamp does
    not imply a fresh reference price; three near-identical BNP short turbos
    (same financing level, barrier, ratio, leverage) showed
    ``issuer_margin_pct`` of -0.03%/-0.80%/-1.43% purely from this
    conflation. ``ProductSnapshot.underlying_price_ref_timestamp`` (parsed
    from BNP's own ``first.priceDate``, see ``adapters/issuer_feeds.py``) is
    the reference price's genuine own timestamp and is used here instead.

    ``underlying_price_ref`` is only used in place of the consensus when
    *all* of the following hold:

    1. it is present at all (structurally absent for Citi -- never guessed);
    2. it carries its own timestamp (``underlying_price_ref_timestamp``);
    3. that timestamp is fresh (age <= ``max_quote_age_s``, the same
       staleness bound applied to product quotes elsewhere in this module);
    4. a consensus is available *and* ``ref`` is within
       ``spot_ref_max_deviation_pct`` of it (relative deviation) -- catching
       exactly the batched/stale-reference scenario above, which a fresh
       *timestamp* alone cannot rule out if the underlying itself moved
       between reference-price updates.

    When a consensus is unavailable (too few/no other quotes to build one
    from -- e.g. a single-product universe), condition 4 cannot be evaluated;
    a ``ref`` that is otherwise fresh and independently timestamped is still
    used rather than discarding the only spot estimate available (CLAUDE.md
    rule 29 forbids imputing *missing* data, not falling back to the single
    genuine data point on hand when no second source exists to cross-check
    it against).

    Any rejection of a present-but-unusable ``ref`` (missing/stale own
    timestamp, or too far from an available consensus) appends the
    ``"spot_ref_rejected"`` warning; a structurally absent ``ref`` (e.g.
    every Citi product) is normal and never warned about.
    """
    ref = product.underlying_price_ref
    if ref is None:
        return consensus_value

    ref_ts = product.underlying_price_ref_timestamp
    if ref_ts is None:
        _add_warning(warnings, "spot_ref_rejected")
        return consensus_value

    age = quote_age_seconds(ref_ts, now)
    if age > max_quote_age_s:
        _add_warning(warnings, "spot_ref_rejected")
        return consensus_value

    if consensus_value is None:
        # Nothing to cross-check against; a fresh, independently-timestamped
        # ref is still the best available estimate.
        return ref

    deviation = (
        abs(ref - consensus_value) / abs(consensus_value) if consensus_value != 0 else float("inf")
    )
    if deviation > spot_ref_max_deviation_pct:
        _add_warning(warnings, "spot_ref_rejected")
        return consensus_value

    return ref


def _cost_rank_score(total_cost_pct: float, leverage_value: float | None) -> float | None:
    """ "Cost per exposure (h)" (Build Contract BEFUND 2): ``total_cost_pct``
    (round-trip spread + gap premium + financing + max(issuer margin, 0),
    all as a % of ask) divided by leverage.

    ``total_cost_pct`` alone is a % of capital employed (the ask), which is
    mechanically smaller for a higher-leverage product at equal underlying
    exposure cost -- a Hebel-2 and a Hebel-10 product with identical
    round-trip spread/financing/margin *as a % of underlying moved* would
    otherwise rank the Hebel-2 product as "cheaper" purely because 5x less
    capital sits behind an equivalent bet size. Dividing by leverage
    re-expresses cost as a % of UNDERLYING exposure: "how far the underlying
    has to move just to cover this product's costs over the horizon" --
    leverage-neutral, per the Spec's "kein pauschales Hebelziel". Two
    products with equal ``total_cost_pct / leverage_value`` therefore get an
    equal score regardless of how different their leverage is (the ranking
    invariance this formula is designed to guarantee).

    Returns ``None`` when ``leverage_value`` is ``None`` or non-positive
    (no ask -> leverage cannot be computed at all, or a degenerate/invalid
    value) rather than raising or dividing by zero.
    """
    if leverage_value is None or not (leverage_value > 0):
        return None
    return total_cost_pct / leverage_value


def _resolve_fx(
    product: ProductSnapshot, needs_fx: bool, fx_by_isin: Mapping[str, float]
) -> float | None:
    """FX (units of underlying currency per 1 unit product currency, EUR).

    Quanto products, EUR-underlyings (per underlying_map) and products whose
    reported ``underlying_currency`` already matches the product currency use
    ``fx=1``. Everything else needs a resolved fx (Befund 1: derived
    per-issuer from this fetch's own data, see ``pricing/fx_resolution.py``,
    looked up by ISIN); ``None`` means it could not be resolved for this
    product's issuer (never imputed, CLAUDE.md rule 29) -- the caller must
    treat that as ``fx_unresolved``, not a generic pricing failure.
    """
    if product.quanto is True:
        return 1.0
    if product.underlying_currency == product.currency:
        return 1.0
    if not needs_fx:
        return 1.0
    return fx_by_isin.get(product.isin)


# --------------------------------------------------------------------------
# per-product pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _PricedProduct:
    """Intermediate per-product pricing state, before cross-issuer scoring."""

    product: ProductSnapshot
    bid: float
    ask: float
    spot: float
    fx: float
    integrity: IntegrityReport
    leverage_value: float
    leverage_bucket_value: str
    spread_pct_value: float
    quote_age_s: float | None
    distance_pct: float | None
    distance_sigma: float | None
    realized_spread: float
    used_default_spread: bool
    financing_spread_source: str
    costs: CostDecomposition
    financing_cost_pct: dict[str, float]
    gap_premium_over_horizon_pct: float
    liquidity: float


def _evaluate_single_product(
    *,
    cfg: TurboEdgeConfig,
    store: Store,
    product: ProductSnapshot,
    underlying_id: str,
    consensus_value: float | None,
    needs_fx: bool,
    fx_by_isin: Mapping[str, float],
    sigma_t: float | None,
    gap_dist: GapDistribution | None,
    r: float,
    horizon_days: int,
    evaluation_time: datetime,
    next_night_is_weekend: bool,
    data_health_pass: bool,
    run_id: str,
    warnings: list[str],
) -> CandidateEvaluation | _PricedProduct:
    """Price and integrity-check one product. Never raises for ordinary data issues.

    Any :class:`ValueError` from the pricing formulas below (e.g. a
    non-positive ``financing_level``) is intentionally left to propagate --
    the caller (:func:`_process_products`) catches it and downgrades that one
    product to ``DATA_QUALITY`` without aborting the rest of the scan.
    """
    risk = cfg.risk

    fx = _resolve_fx(product, needs_fx, fx_by_isin)
    if fx is None:
        # Befund 1: FX could not be determined for this product's issuer
        # from the fetch's own data (neither the quanto nor the non-quanto
        # hypothesis clustered consistently, see pricing/fx_resolution.py).
        # A distinct reason, never folded into a pricing-defect bucket like
        # "bid_below_intrinsic" -- this product simply cannot be priced this
        # run, it is not evidence the product itself is mispriced.
        return _data_quality_candidate(
            run_id=run_id,
            product=product,
            underlying_id=underlying_id,
            horizon_days=horizon_days,
            reasons=["fx_unresolved"],
        )

    spot = _resolve_spot(
        product,
        consensus_value,
        risk.max_quote_age_s,
        risk.spot_ref_max_deviation_pct,
        evaluation_time,
        warnings,
    )
    if spot is None:
        return _data_quality_candidate(
            run_id=run_id,
            product=product,
            underlying_id=underlying_id,
            horizon_days=horizon_days,
            reasons=["spot_unavailable"],
        )

    integrity = check_product(
        product,
        consensus_value,
        evaluation_time,
        risk.max_quote_age_s,
        None,
        risk.integrity_tolerances.margin_warn_pct,
        ref_rate=r,
    )

    quote_age_s = (
        quote_age_seconds(product.quote_timestamp, evaluation_time)
        if product.quote_timestamp is not None
        else None
    )
    freshness = freshness_score(quote_age_s)

    # Distance-to-barrier needs only spot/barrier/sigma -- independent of
    # bid/ask -- so it is computed once, up front, and reused by every
    # return path below (including the no-ask REJECT branch).
    distance_pct: float | None = None
    distance_sigma: float | None = None
    if product.knockout_barrier is not None and sigma_t is not None:
        bd = distance_to_barrier(
            spot, product.knockout_barrier, product.direction, sigma_t, float(horizon_days)
        )
        distance_pct, distance_sigma = bd.pct, bd.sigma

    # Source explicitly reported no live two-way market at all for this
    # product (bid AND ask both missing, quote_presence is False -- e.g.
    # Citi's referencePriceMethod == "Closing Price" rows, see
    # adapters/issuer_feeds.py). That is "no tradable quote", not a
    # data-quality violation -- pricing/integrity.check_product does not
    # fail on `missing_bid` in this case, so `integrity.passed` here reflects
    # only genuine master-data problems (financing level, barrier, underlying
    # mapping, ...). Route through the gate evaluation for a REJECT
    # ("no_live_quote") rather than the DATA_QUALITY branch below.
    no_live_quote = product.bid is None and product.ask is None and product.quote_presence is False
    if no_live_quote and integrity.passed:
        return _gate_only_candidate(
            run_id=run_id,
            product=product,
            underlying_id=underlying_id,
            horizon_days=horizon_days,
            integrity=integrity,
            quote_age_s=quote_age_s,
            distance_pct=distance_pct,
            distance_sigma=distance_sigma,
            data_health_pass=data_health_pass,
            risk=risk,
            has_ask=False,
            no_live_quote=True,
        )

    if product.bid is None or product.financing_level is None:
        reasons = list(integrity.failures) or ["missing_pricing_inputs"]
        return _data_quality_candidate(
            run_id=run_id,
            product=product,
            underlying_id=underlying_id,
            horizon_days=horizon_days,
            reasons=reasons,
        )

    if product.ask is None:
        # Issuer is only quoting a bid (e.g. outside trading hours): a
        # tradability gate, not a data-integrity failure (Build Contract
        # Task 2 review finding) -- pricing/integrity.check_product no
        # longer fails on a missing ask, so `integrity` above reflects only
        # genuine data problems. Route through the same gate evaluation as
        # every other candidate (REJECT vs. DATA_QUALITY precedence stays
        # centralized in ranking/gates.py) but skip every ask-dependent
        # pricing step (decompose_ask, leverage, spread) entirely.
        return _gate_only_candidate(
            run_id=run_id,
            product=product,
            underlying_id=underlying_id,
            horizon_days=horizon_days,
            integrity=integrity,
            quote_age_s=quote_age_s,
            distance_pct=distance_pct,
            distance_sigma=distance_sigma,
            data_health_pass=data_health_pass,
            risk=risk,
            has_ask=False,
        )

    bid, ask, financing_level = product.bid, product.ask, product.financing_level

    if gap_dist is None:
        return _data_quality_candidate(
            run_id=run_id,
            product=product,
            underlying_id=underlying_id,
            horizon_days=horizon_days,
            reasons=["gap_distribution_unavailable"],
        )

    lev = compute_leverage(spot, ask, product.ratio, fx)
    lev_bucket = leverage_bucket(lev)
    spread_pct_value = feature_spread_pct(bid, ask)

    levels = store.financing_level_history(product.isin)
    realized_spread: float | None = None
    if len(levels) >= 2:
        observations = financing_spread_history(
            levels, r, product.direction, risk.financing_adjustment_jump_threshold_pct
        )
        realized_spread = realized_financing_spread(observations)
    used_default_spread = realized_spread is None

    # Financing spread priority (Contract v3, Punkt 3 / Build Contract
    # "Formeln"): (a) realized spread from >= 2 days of own financing-level
    # history [above]; (b) issuer-reported annualized funding rate (Citi
    # `items[].fundingRate`, ProductSnapshot.financing_rate) minus the
    # reference rate; (c) configs/risk.yaml's default_financing_spread. The
    # source actually used is always recorded (reasons list, below, and
    # CandidateEvaluation via `p.financing_spread_source`) -- never silent.
    if realized_spread is not None:
        spread_for_pricing = realized_spread
        financing_spread_source = "realized_history"
    elif product.financing_rate is not None:
        spread_for_pricing = product.financing_rate - r
        financing_spread_source = "issuer_funding_rate"
    else:
        spread_for_pricing = risk.default_financing_spread
        financing_spread_source = "financing_spread_default"

    fgp = fair_gap_premium(
        spot,
        financing_level,
        product.ratio,
        product.direction,
        gap_dist,
        next_night_is_weekend,
        fx,
    )

    costs = decompose_ask(
        bid,
        ask,
        spot,
        financing_level,
        product.ratio,
        product.direction,
        fgp,
        spread_for_pricing,
        r,
        fx,
        product_type=product.product_type,
        knockout_barrier=product.knockout_barrier,
        as_of=evaluation_time.date(),
        maturity=product.maturity,
        dividend_yield=dividend_yield_for_underlying(underlying_id),
    )

    is_classic = product.product_type == ProductType.TURBO_CLASSIC
    financing_cost_pct: dict[str, float] = {}
    for h in _VALID_HORIZONS:
        # A turbo_classic's financing cost is already inside its
        # present-valued fair value (theoretical_fair_value), not a
        # separate daily/horizon accrual -- same rationale as
        # decompose_ask's financing_drag=0 for classics (Build Contract W1).
        if is_classic:
            financing_cost_pct[f"{h}d"] = 0.0
            continue
        cost = financing_cost_over_horizon(
            financing_level, spread_for_pricing, r, float(h), product.ratio, product.direction, fx
        )
        financing_cost_pct[f"{h}d"] = cost / ask

    gap_premium_horizon_pct = (
        gap_premium_over_horizon(
            spot, financing_level, product.ratio, product.direction, gap_dist, horizon_days, fx
        )
        / ask
    )

    quote_size_cov = quote_size_coverage(product.ask_size, risk.required_notional_eur, ask)
    spread_qual = spread_quality(spread_pct_value, risk.max_spread_pct)
    quote_presence_value = (
        0.5 if product.quote_presence is None else (1.0 if product.quote_presence else 0.0)
    )
    liq = liquidity_factor(quote_presence_value, quote_size_cov, freshness, spread_qual)

    return _PricedProduct(
        product=product,
        bid=bid,
        ask=ask,
        spot=spot,
        fx=fx,
        integrity=integrity,
        leverage_value=lev,
        leverage_bucket_value=lev_bucket,
        spread_pct_value=spread_pct_value,
        quote_age_s=quote_age_s,
        distance_pct=distance_pct,
        distance_sigma=distance_sigma,
        realized_spread=spread_for_pricing,
        used_default_spread=used_default_spread,
        financing_spread_source=financing_spread_source,
        costs=costs,
        financing_cost_pct=financing_cost_pct,
        gap_premium_over_horizon_pct=gap_premium_horizon_pct,
        liquidity=liq,
    )


_CATEGORY_RANK: dict[Category, int] = {
    Category.ACTIONABLE: 0,
    Category.WATCH: 1,
    Category.REJECT: 2,
    Category.DATA_QUALITY: 3,
}


def _sort_candidates(
    candidates: Sequence[CandidateEvaluation], signal: SignalSnapshot | None
) -> list[CandidateEvaluation]:
    """Category first (WATCH, REJECT, DATA_QUALITY), then signal-aligned direction, then cost."""
    direction_hint = signal.direction_hint if signal is not None else None

    def sort_key(c: CandidateEvaluation) -> tuple[int, int, float]:
        direction_rank = 0 if direction_hint is None or c.direction == direction_hint else 1
        cost = c.cost_rank_score if c.cost_rank_score is not None else float("inf")
        return (_CATEGORY_RANK[c.category], direction_rank, cost)

    return sorted(candidates, key=sort_key)


def _process_products(
    *,
    cfg: TurboEdgeConfig,
    store: Store,
    run_id: str,
    products: Sequence[ProductSnapshot],
    underlying_id: str,
    signal: SignalSnapshot | None,
    sigma_t: float | None,
    gap_dist: GapDistribution | None,
    r: float,
    horizon_days: int,
    evaluation_time: datetime,
    next_night_is_weekend: bool,
    data_health_pass: bool,
    warnings: list[str],
) -> tuple[list[CandidateEvaluation], dict[str, _PricedProduct]]:
    risk = cfg.risk
    underlying_meta = get_underlying_meta(underlying_id)
    needs_fx = underlying_meta.currency != "EUR"

    # Befund 1 (2026-09-13 measurement session): fx is resolved per
    # (issuer, underlying) from this fetch's OWN product data (quanto vs.
    # non-quanto tested against each other, majority/consistency decides --
    # see pricing/fx_resolution.py) instead of a single yfinance daily-close
    # approximation applied uniformly to every issuer. A stale daily close
    # (or a wrongly-assumed-non-quanto issuer) previously made almost every
    # USD-underlying product look like it fails "bid below intrinsic" --
    # a systematic FX error masquerading as a per-product data defect.
    resolved_fx: dict[str, FxResolution] = {}
    fx_map: dict[str, float] = {}
    if needs_fx:
        resolved_fx = resolve_fx_by_issuer(products)
        fx_map = fx_by_isin(products, resolved_fx)
        unresolved_issuers = {p.issuer for p in products} - set(resolved_fx)
        if unresolved_issuers:
            _add_warning(warnings, "fx_unresolved_for_some_issuers")
        if not resolved_fx:
            _add_warning(warnings, "fx_unavailable_for_underlying")

    consensus_value: float | None
    if needs_fx and not fx_map:
        consensus_value = None
        _add_warning(warnings, "consensus_spot_unavailable_fx_unresolved")
    else:
        try:
            consensus_result = (
                consensus_spot(products, fx_by_isin=fx_map)
                if needs_fx
                else consensus_spot(products, fx=1.0)
            )
            consensus_value = consensus_result.value
            if consensus_result.used_bid_only_fallback:
                _add_warning(warnings, "consensus_from_bid_only")
        except ValueError:
            consensus_value = None
            _add_warning(warnings, "consensus_spot_unavailable")

    priced: dict[str, _PricedProduct] = {}
    finalized: dict[str, CandidateEvaluation] = {}

    for product in products:
        try:
            outcome = _evaluate_single_product(
                cfg=cfg,
                store=store,
                product=product,
                underlying_id=underlying_id,
                consensus_value=consensus_value,
                needs_fx=needs_fx,
                fx_by_isin=fx_map,
                sigma_t=sigma_t,
                gap_dist=gap_dist,
                r=r,
                horizon_days=horizon_days,
                evaluation_time=evaluation_time,
                next_night_is_weekend=next_night_is_weekend,
                data_health_pass=data_health_pass,
                run_id=run_id,
                warnings=warnings,
            )
        except Exception as exc:
            logger.warning("candidate_pricing_failed", isin=product.isin, error=str(exc))
            finalized[product.isin] = _data_quality_candidate(
                run_id=run_id,
                product=product,
                underlying_id=underlying_id,
                horizon_days=horizon_days,
                reasons=[f"pricing_error:{exc}"],
            )
            continue
        if isinstance(outcome, CandidateEvaluation):
            finalized[product.isin] = outcome
        else:
            priced[product.isin] = outcome

    rows = [
        CrossIssuerInput(
            isin=p.product.isin,
            issuer=p.product.issuer,
            underlying_id=underlying_id,
            direction=p.product.direction,
            leverage_bucket=p.leverage_bucket_value,
            normalized_spread=p.spread_pct_value,
            issuer_margin_pct=p.costs.issuer_margin_pct,
            financing_spread=p.realized_spread,
            quote_age_s=p.quote_age_s,
        )
        for p in priced.values()
    ]
    scores = cross_issuer_scores(rows)
    empty_scores = CrossIssuerScores(
        cross_issuer_residual_zscore=None,
        issuer_markup_score=None,
        quote_dislocation_score=None,
        wrapper_edge=None,
    )

    thresholds = GateThresholds.from_risk_config(risk)
    horizon_key = f"{horizon_days}d"

    for isin, p in priced.items():
        cs = scores.get(isin, empty_scores)
        gate_input = GateInput(
            integrity=p.integrity,
            bid_only=p.product.bid_only,
            knocked_out=p.product.knocked_out,
            quote_age_s=p.quote_age_s,
            spread_pct=p.spread_pct_value,
            leverage=p.leverage_value,
            distance_to_barrier_sigma=p.distance_sigma,
            data_health_pass=data_health_pass,
            lcb_ev=None,
            p_ko=None,
            cluster_risk_pass=None,
        )
        category, gate_reasons = evaluate_gates(gate_input, thresholds)
        reasons = list(gate_reasons)
        if (
            signal is not None
            and signal.direction_hint is not None
            and p.product.direction != signal.direction_hint
        ):
            reasons.append("counter_baseline_signal")

        round_trip_spread_pct = (p.ask - p.bid) / p.ask
        total_cost_pct = (
            round_trip_spread_pct
            + p.gap_premium_over_horizon_pct
            + p.financing_cost_pct[horizon_key]
            + max(p.costs.issuer_margin_pct, 0.0)
        )
        # See _cost_rank_score's docstring for the BEFUND 2 rationale
        # (leverage-normalized "Cost per exposure (h)"). `p.leverage_value`
        # is always a finite, positive float in this branch -- this loop
        # only ever processes priced products that reached leverage
        # computation successfully (the ask-less branch above returns with
        # cost_rank_score=None before reaching this code).
        cost_rank_score = _cost_rank_score(total_cost_pct, p.leverage_value)

        finalized[isin] = CandidateEvaluation(
            run_id=run_id,
            candidate_id=_candidate_id(isin, underlying_id, p.product.direction, horizon_days),
            isin=isin,
            wkn=p.product.wkn,
            issuer=p.product.issuer,
            underlying_id=underlying_id,
            direction=p.product.direction,
            category=category,
            reasons=reasons,
            leverage=p.leverage_value,
            leverage_bucket=p.leverage_bucket_value,
            distance_to_barrier_pct=p.distance_pct,
            distance_to_barrier_sigma=p.distance_sigma,
            costs=p.costs,
            realized_financing_spread=p.realized_spread,
            financing_cost_horizon_pct=p.financing_cost_pct,
            cross_issuer_residual_zscore=cs.cross_issuer_residual_zscore,
            issuer_markup_score=cs.issuer_markup_score,
            quote_dislocation_score=cs.quote_dislocation_score,
            wrapper_edge=cs.wrapper_edge,
            liquidity_factor=p.liquidity,
            integrity_passed=p.integrity.passed,
            lcb_ev=None,
            cost_rank_score=cost_rank_score,
            financing_spread_source=p.financing_spread_source,
        )

    return _sort_candidates(list(finalized.values()), signal), priced


# --------------------------------------------------------------------------
# notification
# --------------------------------------------------------------------------


def _notification_key_values(
    top_candidates: Sequence[CandidateEvaluation],
) -> dict[str, float | str | None]:
    values: dict[str, float | str | None] = {}
    for i, c in enumerate(top_candidates):
        values[f"c{i}_id"] = c.candidate_id
        values[f"c{i}_category"] = c.category.value
        values[f"c{i}_cost"] = c.cost_rank_score
    return values


def _maybe_send_email(
    *,
    store: Store,
    options: ScanOptions,
    notifier: GmailNotifier | None,
    run_id: str,
    underlying_id: str,
    signal: SignalSnapshot | None,
    candidates: Sequence[CandidateEvaluation],
    counts: dict[Category, int],
    warnings: list[str],
    clock: Callable[[], datetime],
    send_on: Sequence[str],
) -> SendResult | None:
    if not options.email or notifier is None:
        return None

    # Befund 3(b): a scan report mail is a targeted alert, not a daily
    # digest (Master Spec §34: "Kein taeglicher NO-TRADE-Spam") -- it must
    # only go out when at least one candidate actually landed in one of the
    # categories `configs/gmail.yaml`'s `send_on` names (e.g. an all-REJECT
    # scan with `send_on: ["ACTIONABLE"]` sends nothing). The existing
    # per-notification-hash dedup below still applies on top of this gate.
    active_categories = {cat.value for cat, count in counts.items() if count > 0}
    if not any(category in active_categories for category in send_on):
        return None

    top_candidates = list(candidates[: options.top])
    rows = [
        ScanReportRow(
            rank=i + 1,
            isin=c.isin,
            wkn=c.wkn,
            issuer=c.issuer,
            direction=c.direction.value,
            category=c.category.value,
            leverage=c.leverage,
            spread_pct=c.costs.spread_pct if c.costs is not None else None,
            distance_to_barrier_pct=c.distance_to_barrier_pct,
            issuer_margin_pct=c.costs.issuer_margin_pct if c.costs is not None else None,
            financing_cost_7d_pct=c.financing_cost_horizon_pct.get("7d"),
            liquidity_factor=c.liquidity_factor,
            cost_rank_score=c.cost_rank_score,
            reasons=c.reasons,
        )
        for i, c in enumerate(top_candidates)
    ]
    context = ScanReportContext(
        run_id=run_id,
        underlying_id=underlying_id,
        generated_at=clock(),
        signal_score=signal.score if signal is not None else None,
        direction_hint=(
            signal.direction_hint.value
            if signal is not None and signal.direction_hint is not None
            else None
        ),
        counts={cat.value: count for cat, count in counts.items() if count > 0},
        rows=rows,
        warnings=warnings,
    )
    subject, body = render_scan_report(context)

    key_values = _notification_key_values(top_candidates)
    n_hash = notification_hash(None, "SCAN_REPORT", key_values)
    dedup = NotificationDeduplicator(store)
    if not dedup.should_send(n_hash):
        return SendResult(sent=False, dry_run=False, recipients=[], message="skipped_duplicate")

    recipients = notifier.credentials.recipients if notifier.credentials is not None else []
    spec = EmailMessageSpec(subject=subject, body_text=body, to=recipients)
    try:
        result = notifier.send(spec)
    except NotificationError as exc:
        logger.error("scan_notification_failed", error=str(exc))
        _add_warning(warnings, f"notification_send_failed:{exc}")
        return SendResult(sent=False, dry_run=False, recipients=recipients, message=f"error: {exc}")

    dedup.mark_sent(n_hash, None, "SCAN_REPORT", subject, sent_at=clock())
    return result


# --------------------------------------------------------------------------
# EV pipeline (Contract v3 Abschnitt B): forecast -> paths -> EV -> gates ->
# ledger + shadow sample -> ACTIONABLE mail. Runs by default (``run_scan``'s
# ``run_ev=True`` default -- see that function's docstring); only skipped
# when a caller explicitly passes ``run_ev=False`` (test-only, to exercise
# the hard gates in isolation from this slower, stochastic layer).
# --------------------------------------------------------------------------


def default_forecast_models() -> list[ForecastModel]:
    """``models.forecast.build_default_models()`` plus, when present, W9's
    challenger models -- imported lazily so a missing/not-yet-built
    ``models/challengers.py`` degrades to "feature disabled" (logged) rather
    than blocking the scan (Contract v3: "wenn eine Datei noch fehlt,
    importiere sie lazy ... und behandle ImportError sauber")."""
    models: list[ForecastModel] = list(build_default_models())
    try:
        from turboedge.models import challengers as challengers_mod
    except ImportError:
        return models
    builder = getattr(challengers_mod, "build_challenger_models", None)
    if builder is None:
        return models
    try:
        extra = list(builder())
    except Exception as exc:  # pragma: no cover - defensive, W9 module not owned here
        logger.warning("challenger_models_build_failed", error=str(exc))
        return models
    logger.info("challenger_models_loaded", count=len(extra))
    models.extend(extra)
    return models


def _premium_uncertainty_term(premium_over_fair: float, mean_net_return: float) -> float:
    """Additional, analytic uncertainty term from halving the exit issuer
    markup (Contract v3, Punkt 1 "Aufschlag-Handhabung").

    A full second Monte Carlo simulation per candidate was judged not worth
    its cost here (Contract v3 Abschnitt B performance note already caps
    simulation volume elsewhere) -- instead this closed-form first-order
    approximation is used, derived from ``simulation/payoff.py``'s own exit
    formula ``exit_value = fair_value * (1 + premium_over_fair)``: halving
    the premium scales the alive-path exit value by
    ``(1 + premium/2) / (1 + premium)``, so the implied change in mean net
    return is ``(premium/2) * (mean_net_return + 1) / (1 + premium)``.
    Always returned as a non-negative magnitude; the caller *subtracts* it
    from the LCB (never adds), so this can only make gating more
    conservative, matching the direction every other documented
    simplification in this codebase already leans (fewer, not riskier,
    proposals on an approximation's account).
    """
    denom = 1.0 + premium_over_fair
    if abs(denom) < 1e-9:
        return abs(premium_over_fair) / 2.0
    return abs((premium_over_fair / 2.0) * (mean_net_return + 1.0) / denom)


@dataclass(frozen=True)
class EvPipelineResult:
    """Outcome of :func:`_run_ev_pipeline`."""

    candidates: list[CandidateEvaluation]
    ledger_entries: list[LedgerEntry]
    forecast_records: list[ForecastRecord]
    new_cluster_positions: list[OpenClusterPosition]
    newly_actionable: list[CandidateEvaluation]
    evaluations_by_isin: dict[str, ProductHorizonEvaluation] = field(default_factory=dict)


def _select_ev_pool(
    candidates: Sequence[CandidateEvaluation],
    priced: dict[str, _PricedProduct],
    *,
    rng: np.random.Generator,
    max_candidates_per_bucket: int,
    min_liquidity_factor: float,
    shadow_sample_per_stratum: int,
) -> tuple[set[str], set[str]]:
    """Return ``(prefiltered_isins, shadow_isins)`` (Contract v3 Abschnitt B
    "Performance"): ``prefiltered_isins`` are the cheapest
    ``max_candidates_per_bucket`` WATCH candidates per (direction,
    leverage_bucket) group among those with a usable ask and liquidity above
    ``min_liquidity_factor`` -- the hard, simulation-free gates are already
    enforced upstream (a WATCH category here means every REJECT/DATA_QUALITY
    gate already passed). ``shadow_isins`` are drawn from the FULL priced
    candidate pool (every category, Spec §25 selection-bias protection),
    minus whatever is already in ``prefiltered_isins``.
    """
    priced_candidates = [c for c in candidates if c.isin in priced]

    watch_eligible = [
        c
        for c in priced_candidates
        if c.category == Category.WATCH
        and priced[c.isin].liquidity >= min_liquidity_factor
        and priced[c.isin].leverage_value > 0.0
    ]
    groups: dict[tuple[str, str], list[CandidateEvaluation]] = {}
    for c in watch_eligible:
        bucket = leverage_bucket_for(priced[c.isin].leverage_value)
        groups.setdefault((c.direction.value, bucket), []).append(c)

    prefiltered: set[str] = set()
    for members in groups.values():
        ranked = sorted(members, key=lambda c: c.cost_rank_score or float("inf"))
        prefiltered.update(m.isin for m in ranked[:max_candidates_per_bucket])

    def _strata(c: CandidateEvaluation) -> tuple[str, str, str]:
        bucket = (
            leverage_bucket_for(priced[c.isin].leverage_value) if c.isin in priced else "unknown"
        )
        return (c.category.value, c.direction.value, bucket)

    shadow_sample = select_shadow_sample(priced_candidates, rng, shadow_sample_per_stratum, _strata)
    shadow_isins = {c.isin for c in shadow_sample} - prefiltered
    return prefiltered, shadow_isins


def _fit_forecast_ensemble(
    *,
    store: Store,
    models: Sequence[ForecastModel],
    bars: Sequence[UnderlyingBar],
    prediction_time: datetime,
    horizons: Sequence[int],
    default_new_model_weight: float,
    warnings: list[str],
) -> tuple[dict[int, HorizonForecast], list[ForecastRecord], dict[str, float]]:
    """Fit every model in ``models`` (skipping ones with insufficient history,
    logged, per Contract v3), register/update them in the model registry,
    and combine their per-horizon forecasts into one ensemble per horizon.

    Returns ``(ensemble_by_horizon, forecast_records, weights_used)``; all
    three are empty when not a single model could be fit (e.g. too little
    underlying history) -- callers must treat that as "no forecast this
    scan" (Contract v3 Verbindliche Entscheidung 5: "nur Underlyings
    scannen, für die ein Forecast UND Produkte vorliegen").
    """
    registry = ModelRegistry(store)
    fitted: list[tuple[ForecastModel, list[HorizonForecast]]] = []
    for model in models:
        try:
            model.fit(bars, prediction_time)
            forecasts = model.predict(bars, prediction_time, horizons=list(horizons))
        except Exception as exc:
            logger.warning(
                "forecast_model_fit_failed",
                model_id=getattr(model, "model_id", "?"),
                error=str(exc),
            )
            continue
        fitted.append((model, forecasts))

        existing = registry.get(model.model_id)
        if existing is None:
            status = (
                ModelStatus.PROTECTED if model.signal_family == "tsmom" else ModelStatus.CHALLENGER
            )
            registry.register(
                model.model_id,
                model.model_hash(),
                model.signal_family,
                params={},
                trial_id=None,
                status=status,
                initial_weight=default_new_model_weight,
            )
        else:
            registry.register(
                model.model_id,
                model.model_hash(),
                model.signal_family,
                params={},
                trial_id=None,
                status=existing.status,
                initial_weight=existing.weight,
            )

    if not fitted:
        _add_warning(warnings, "forecast_unavailable_insufficient_history")
        return {}, [], {}

    registry_weights = registry.weights()
    weight_map = {
        model.model_id: max(registry_weights.get(model.model_id, default_new_model_weight), 1e-9)
        for model, _ in fitted
    }

    ensemble_by_horizon: dict[int, HorizonForecast] = {}
    forecast_records: list[ForecastRecord] = []
    frozen_at = datetime.now(UTC)
    for h in horizons:
        per_model = [f for _model, flist in fitted for f in flist if f.horizon_days == h]
        if not per_model:
            continue
        ensemble = combine_forecasts(per_model, weight_map)
        ensemble_by_horizon[h] = ensemble
        for f in per_model:
            forecast_records.append(
                ForecastRecord(
                    run_id="",  # filled in by the caller (needs the enclosing run_id)
                    underlying_id=f.underlying_id,
                    horizon_days=h,
                    prediction_time=f.prediction_time,
                    frozen_at=frozen_at,
                    p_up=f.p_up,
                    mean=f.mean,
                    sigma=f.sigma,
                    quantiles=f.quantiles,
                    expected_shortfall_05=f.expected_shortfall_05,
                    uncertainty=f.uncertainty,
                    model_id=f.model_id,
                    model_hash=f.model_hash,
                    signal_family=f.signal_family,
                    n_train=f.n_train,
                    n_effective=f.n_effective,
                    component_weights={},
                    config_hash="",
                    git_commit=None,
                )
            )
        forecast_records.append(
            ForecastRecord(
                run_id="",
                underlying_id=ensemble.underlying_id,
                horizon_days=h,
                prediction_time=ensemble.prediction_time,
                frozen_at=frozen_at,
                p_up=ensemble.p_up,
                mean=ensemble.mean,
                sigma=ensemble.sigma,
                quantiles=ensemble.quantiles,
                expected_shortfall_05=ensemble.expected_shortfall_05,
                uncertainty=ensemble.uncertainty,
                model_id=ensemble.model_id,
                model_hash=ensemble.model_hash,
                signal_family=ensemble.signal_family,
                n_train=ensemble.n_train,
                n_effective=ensemble.n_effective,
                component_weights=dict(weight_map),
                config_hash="",
                git_commit=None,
            )
        )
    return ensemble_by_horizon, forecast_records, weight_map


_NOT_EVALUATED_PREFILTER_REASON = "not_evaluated_prefilter"


_STALE_PRE_EV_REASONS = frozenset(
    {"lcb_ev_not_evaluated", "p_ko_not_evaluated", "cluster_risk_not_confirmed"}
)


def _annotate_not_evaluated(
    candidates: Sequence[CandidateEvaluation],
    priced: dict[str, _PricedProduct],
    evaluated_isins: set[str],
) -> list[CandidateEvaluation]:
    """Mark WATCH candidates that were never simulated this run (Befund 4,
    2026-09-13; corrected against a live measurement in Befund 2, 2026-09-14
    -- see :func:`_run_ev_pipeline`'s call sites for why ``evaluated_isins``
    must be the actual simulation outcome, not the pre-simulation selection).

    A candidate is annotated when it already has a valid priced quote
    (``isin in priced``, i.e. every pre-EV gate passed) and category WATCH,
    but its ISIN is not in ``evaluated_isins`` -- :func:`_select_ev_pool`'s
    bucket-capacity cap excluded it, or it was selected but never produced a
    usable evaluation (missing barrier/financing_level, or a fair-value
    ``ValueError``) -- "never simulated", not "actively rejected". The
    category is left exactly as it already was (WATCH is the honest
    category here; it is never downgraded to REJECT for a reason that isn't
    a real gate failure). Its placeholder pre-EV reasons
    (:data:`_STALE_PRE_EV_REASONS` -- true-but-vague at the point
    ``evaluate_gates`` first ran, before this run's EV pipeline had decided
    which candidates it would even attempt) are replaced, not appended to,
    with the single specific :data:`_NOT_EVALUATED_PREFILTER_REASON` -- so
    this is visible/queryable rather than duplicating stale placeholder text
    alongside it.
    """
    annotated: list[CandidateEvaluation] = []
    for c in candidates:
        if c.category == Category.WATCH and c.isin in priced and c.isin not in evaluated_isins:
            kept = [r for r in c.reasons if r not in _STALE_PRE_EV_REASONS]
            annotated.append(
                c.model_copy(update={"reasons": [*kept, _NOT_EVALUATED_PREFILTER_REASON]})
            )
        else:
            annotated.append(c)
    return annotated


def _run_ev_pipeline(
    *,
    cfg: TurboEdgeConfig,
    store: Store,
    run_id: str,
    underlying_id: str,
    signal: SignalSnapshot | None,
    usable_bars: Sequence[UnderlyingBar],
    candidates: Sequence[CandidateEvaluation],
    priced: dict[str, _PricedProduct],
    prediction_time: datetime,
    evaluation_time: datetime,
    r: float,
    config_hash_value: str,
    git_commit_value: str | None,
    rng: np.random.Generator,
    forecast_models: Sequence[ForecastModel],
    cluster_id: str,
    cluster_open_positions: Sequence[OpenClusterPosition],
    warnings: list[str],
) -> EvPipelineResult:
    """Forecast -> paths -> EV -> gates -> ledger (+ shadow sample), the core
    of Contract v3 Abschnitt B. See the module docstring's step list; this
    function implements steps 1-4 for one underlying (step 5, the mail, is
    sent by the caller once it knows which candidates newly became
    ACTIONABLE this run -- see ``newly_actionable`` on the result)."""
    forecast_cfg = cfg.forecast
    ranking_cfg = cfg.ranking

    prefiltered_isins, shadow_isins = _select_ev_pool(
        candidates,
        priced,
        rng=rng,
        max_candidates_per_bucket=ranking_cfg.scan_filter.max_candidates_per_bucket,
        min_liquidity_factor=ranking_cfg.scan_filter.min_liquidity_factor,
        shadow_sample_per_stratum=cfg.learning.shadow_sample_per_stratum,
    )
    ev_pool_isins = prefiltered_isins | shadow_isins

    # Befund 4 (2026-09-13 measurement session) / Befund 2 (2026-09-14 live
    # measurement follow-up): a candidate that already passed every pre-EV
    # gate (category WATCH, a valid priced quote) but was never actually
    # simulated this run -- whether excluded by the bucket-capacity
    # prefilter above, or selected into `ev_pool_isins` but dropped later
    # (missing barrier/financing_level, or a fair-value ValueError, in the
    # "build ProductTerms" loop below) -- must not be indistinguishable from
    # a product that was genuinely rejected (stale, bid_only, leverage out
    # of range, barrier too close). Its category stays the honest WATCH it
    # already has. `_annotate_not_evaluated` is called once, right before
    # each return below, against `evaluated_isins` -- the set this function
    # actually produced a simulation result for (empty here, or
    # `set(best_by_isin)` after the loop) -- rather than against
    # `ev_pool_isins` eagerly: eager annotation against the *selection*
    # rather than the *outcome* mis-labeled exactly the ProductTerms/
    # fair-value dropouts above as if they had never even been attempted.
    evaluated_isins: set[str] = set()

    if not ev_pool_isins:
        _add_warning(warnings, "ev_pool_empty")
        return EvPipelineResult(
            candidates=_annotate_not_evaluated(candidates, priced, evaluated_isins),
            ledger_entries=[],
            forecast_records=[],
            new_cluster_positions=list(cluster_open_positions),
            newly_actionable=[],
        )

    ensemble_by_horizon, forecast_records, _weights = _fit_forecast_ensemble(
        store=store,
        models=forecast_models,
        bars=usable_bars,
        prediction_time=prediction_time,
        horizons=forecast_cfg.horizons,
        default_new_model_weight=forecast_cfg.default_new_model_weight,
        warnings=warnings,
    )
    if not ensemble_by_horizon:
        return EvPipelineResult(
            candidates=_annotate_not_evaluated(candidates, priced, evaluated_isins),
            ledger_entries=[],
            forecast_records=[],
            new_cluster_positions=list(cluster_open_positions),
            newly_actionable=[],
        )
    forecast_records = [
        r_.model_copy(
            update={
                "run_id": run_id,
                "config_hash": config_hash_value,
                "git_commit": git_commit_value,
            }
        )
        for r_ in forecast_records
    ]

    # -- build ProductTerms + premium_over_fair for every EV-pool candidate --
    terms_by_isin: dict[str, ProductTerms] = {}
    premium_by_isin: dict[str, float] = {}
    for isin in ev_pool_isins:
        p = priced.get(isin)
        if p is None or p.product.knockout_barrier is None or p.product.financing_level is None:
            continue
        try:
            fair_value = theoretical_fair_value(
                direction=p.product.direction,
                product_type=p.product.product_type,
                spot=p.spot,
                financing_level=p.product.financing_level,
                knockout_barrier=p.product.knockout_barrier,
                ratio=p.product.ratio,
                fx=p.fx,
                ref_rate=r,
                financing_spread=p.realized_spread,
                as_of=evaluation_time.date(),
                maturity=p.product.maturity,
                dividend_yield=dividend_yield_for_underlying(underlying_id),
            )
        except ValueError as exc:
            logger.warning("premium_over_fair_unavailable", isin=isin, error=str(exc))
            continue
        mid = (p.bid + p.ask) / 2.0
        premium_over_fair = (mid - fair_value) / mid if mid > 0 else 0.0
        premium_by_isin[isin] = premium_over_fair
        terms_by_isin[isin] = ProductTerms(
            isin=isin,
            direction=p.product.direction,
            product_type=p.product.product_type,
            financing_level=p.product.financing_level,
            knockout_barrier=p.product.knockout_barrier,
            ratio=p.product.ratio,
            fx=p.fx,
            entry_ask=p.ask,
            entry_bid=p.bid,
            maturity=p.product.maturity,
            financing_spread=p.realized_spread,
            ref_rate=r,
            exit_spread_pct=p.spread_pct_value,
            premium_over_fair=premium_over_fair,
        )

    if not terms_by_isin:
        _add_warning(warnings, "ev_pool_no_valid_product_terms")
        return EvPipelineResult(
            candidates=_annotate_not_evaluated(candidates, priced, evaluated_isins),
            ledger_entries=[],
            forecast_records=[],
            new_cluster_positions=list(cluster_open_positions),
            newly_actionable=[],
        )

    spot0 = statistics.median(priced[isin].spot for isin in terms_by_isin)
    horizons = sorted(ensemble_by_horizon)
    cluster_cfg: ClusterConfig = ranking_cfg.cluster
    baseline_cluster_risk = compute_cluster_risk(
        cluster_open_positions,
        OpenClusterPosition(
            underlying_id=underlying_id,
            cluster_id=cluster_id,
            capital_fraction=0.0,
            counts_as_active_position=False,
        ),
        cfg=cluster_cfg,
    )

    evaluations = evaluate_product_horizons(
        terms_by_isin,
        ensemble_by_horizon,
        usable_bars,
        underlying_id=underlying_id,
        spot0=spot0,
        start=prediction_time,
        as_of=evaluation_time.date(),
        cluster_id=cluster_id,
        rng=rng,
        horizons=horizons,
        cfg=build_ev_config(cfg),
        cluster_risk=baseline_cluster_risk,
        liquidity_factor_by_isin={isin: priced[isin].liquidity for isin in terms_by_isin},
        leverage_by_isin={isin: priced[isin].leverage_value for isin in terms_by_isin},
        fair_value_fn=theoretical_fair_value,
    )

    best_by_isin: dict[str, ProductHorizonEvaluation] = {}
    for ev in evaluations:
        current = best_by_isin.get(ev.isin)
        if current is None or ev.score > current.score:
            best_by_isin[ev.isin] = ev
    evaluated_isins = set(best_by_isin)
    candidates = _annotate_not_evaluated(candidates, priced, evaluated_isins)

    thresholds = GateThresholds.from_risk_config(cfg.risk)
    updated_candidates: dict[str, CandidateEvaluation] = {c.isin: c for c in candidates}
    ledger_entries: list[LedgerEntry] = []
    newly_actionable: list[CandidateEvaluation] = []
    running_cluster_positions = list(cluster_open_positions)

    for isin in sorted(best_by_isin, key=lambda i: best_by_isin[i].score, reverse=True):
        ev = best_by_isin[isin]
        p = priced[isin]
        original = updated_candidates[isin]
        premium_term = _premium_uncertainty_term(premium_by_isin.get(isin, 0.0), ev.mean_net_return)
        adjusted_lcb = ev.lcb_net_return - premium_term

        candidate_position = OpenClusterPosition(
            underlying_id=underlying_id,
            cluster_id=cluster_id,
            capital_fraction=ev.suggested_position_fraction,
            counts_as_active_position=True,
        )
        cluster_pass = compute_cluster_risk_pass(
            running_cluster_positions, candidate_position, cfg=cluster_cfg
        )

        gate_input = GateInput(
            integrity=p.integrity,
            bid_only=p.product.bid_only,
            knocked_out=p.product.knocked_out,
            quote_age_s=p.quote_age_s,
            spread_pct=p.spread_pct_value,
            leverage=p.leverage_value,
            distance_to_barrier_sigma=p.distance_sigma,
            data_health_pass=True,
            lcb_ev=adjusted_lcb,
            p_ko=ev.p_ko,
            cluster_risk_pass=cluster_pass,
        )
        category, gate_reasons = evaluate_gates(gate_input, thresholds)
        # Non-gate diagnostic annotations from the pre-EV pass
        # (counter-baseline-signal; financing_spread_source is carried on
        # `original`/`updated` as its own field, not a reason -- see
        # CandidateEvaluation.financing_spread_source) stay relevant; the
        # pre-EV *gate* reasons (e.g. "lcb_ev_not_evaluated") are now stale/
        # misleading now that real lcb_ev/p_ko/cluster_risk_pass exist and
        # are dropped in favor of the fresh `gate_reasons` below.
        carried_over = [
            reason for reason in original.reasons if reason == "counter_baseline_signal"
        ]
        reasons = [
            *carried_over,
            *gate_reasons,
            f"ev_horizon={ev.horizon_days}d",
            f"premium_over_fair={premium_by_isin.get(isin, 0.0):.4f}",
            f"premium_uncertainty_term={premium_term:.5f}",
            "p_ko_conservative_see_docs",
        ]
        updated = original.model_copy(
            update={"category": category, "reasons": reasons, "lcb_ev": adjusted_lcb}
        )
        updated_candidates[isin] = updated

        if category == Category.ACTIONABLE:
            running_cluster_positions.append(candidate_position)
            newly_actionable.append(updated)

        is_shadow = isin in shadow_isins
        lev_bucket = original.leverage_bucket or "unknown"
        stratum = f"{original.category.value}|{original.direction.value}|{lev_bucket}"
        exit_due = (
            evaluation_time
            + timedelta(days=round(ev.horizon_days * _CALENDAR_DAYS_PER_TRADING_DAY))
        ).date()
        entry_quote_ts = p.product.quote_timestamp or evaluation_time
        feature_snapshot = {
            "leverage": p.leverage_value,
            "spread_pct": p.spread_pct_value,
            "signal_score": signal.score if signal is not None else 0.0,
            "premium_over_fair": premium_by_isin.get(isin, 0.0),
            "cost_rank_score": original.cost_rank_score or 0.0,
        }
        ledger_entries.append(
            LedgerEntry(
                run_id=run_id,
                candidate_id=original.candidate_id,
                signal_id=signal.signal_id if signal is not None else f"{run_id}-{underlying_id}",
                signal_version_hash=(
                    signal.signal_version_hash if signal is not None else "unavailable"
                ),
                trial_id=_ROUTINE_TRIAL_ID,
                prediction_time=prediction_time,
                underlying=underlying_id,
                direction=p.product.direction,
                horizon_days=ev.horizon_days,
                regime_bucket=None,
                cluster_id=cluster_id,
                feature_hash=sha256_json(feature_snapshot),
                model_hash=ensemble_by_horizon[ev.horizon_days].model_hash,
                config_hash=config_hash_value,
                git_commit=git_commit_value,
                selected_wkn=p.product.wkn,
                selected_isin=isin,
                issuer=p.product.issuer,
                entry_bid=p.bid,
                entry_ask=p.ask,
                entry_spread=p.spread_pct_value,
                entry_quote_timestamp=entry_quote_ts,
                entry_underlying_timestamp=evaluation_time,
                financing_level_entry=p.product.financing_level,
                barrier_entry=p.product.knockout_barrier,
                ratio=p.product.ratio,
                fx=p.fx,
                predicted_return=ev.mean_net_return,
                p_profit=ev.p_profit,
                p_ko=ev.p_ko,
                expected_shortfall=ev.es95,
                lcb_ev=adjusted_lcb,
                uncertainty=ensemble_by_horizon[ev.horizon_days].uncertainty + premium_term,
                shrinkage_intensity=ev.shrinkage_intensity,
                category=category,
                is_shadow=is_shadow,
                shadow_stratum=stratum if is_shadow else None,
                suggested_position_fraction=ev.suggested_position_fraction,
                exit_due=exit_due,
                alternatives=_pick_alternatives(isin, p, priced),
                feature_snapshot=feature_snapshot,
                status=LedgerEntryStatus.OPEN,
            )
        )

    return EvPipelineResult(
        candidates=[updated_candidates[c.isin] for c in candidates],
        ledger_entries=ledger_entries,
        forecast_records=forecast_records,
        new_cluster_positions=running_cluster_positions,
        newly_actionable=newly_actionable,
        evaluations_by_isin=best_by_isin,
    )


def _pick_alternatives(
    isin: str, p: _PricedProduct, priced: dict[str, _PricedProduct], limit: int = 3
) -> list[str]:
    """Up to ``limit`` counterfactual ISINs (Master Spec §24): same
    direction and leverage bucket, a different issuer, ranked by how close
    their leverage is to ``isin``'s own."""
    same_group = [
        other_isin
        for other_isin, other in priced.items()
        if other_isin != isin
        and other.product.direction == p.product.direction
        and other.leverage_bucket_value == p.leverage_bucket_value
        and other.product.issuer != p.product.issuer
    ]
    same_group.sort(
        key=lambda other_isin: abs(priced[other_isin].leverage_value - p.leverage_value)
    )
    return same_group[:limit]


def _maybe_send_trade_proposals(
    *,
    store: Store,
    notifier: GmailNotifier | None,
    underlying_id: str,
    newly_actionable: Sequence[CandidateEvaluation],
    priced: dict[str, _PricedProduct],
    evaluations_by_isin: dict[str, ProductHorizonEvaluation],
    warnings: list[str],
    clock: Callable[[], datetime],
) -> list[SendResult]:
    """Send one §34 trade-proposal email per newly-ACTIONABLE candidate
    (Contract v3 Abschnitt C), deduplicated per candidate + rounded key
    metrics (a materially changed score/LCB/P(KO)/horizon produces a new
    hash and is sent again; an unchanged repeat is not)."""
    if notifier is None or not newly_actionable:
        return []
    dedup = NotificationDeduplicator(store)
    results: list[SendResult] = []
    for candidate in newly_actionable:
        p = priced.get(candidate.isin)
        ev = evaluations_by_isin.get(candidate.isin)
        if p is None or ev is None:
            continue
        context = TradeProposalContext(
            underlying_id=underlying_id,
            issuer=p.product.issuer,
            wkn=p.product.wkn,
            isin=candidate.isin,
            direction=p.product.direction.value,
            horizon_days=ev.horizon_days,
            ask=p.ask,
            bid=p.bid,
            spread_pct=p.spread_pct_value,
            leverage=p.leverage_value,
            distance_to_barrier_pct=p.distance_pct,
            distance_to_barrier_sigma=p.distance_sigma,
            p_profit=ev.p_profit,
            expected_net_return=ev.mean_net_return,
            lcb_net_return=candidate.lcb_ev if candidate.lcb_ev is not None else ev.lcb_net_return,
            p_ko=ev.p_ko,
            es95=ev.es95,
            spread_cost_pct=p.costs.spread_pct,
            financing_cost_pct=p.financing_cost_pct.get(f"{ev.horizon_days}d", 0.0),
            gap_premium_pct=p.costs.gap_premium_pct,
            issuer_margin_pct=p.costs.issuer_margin_pct,
            suggested_position_fraction=ev.suggested_position_fraction,
            reasons=candidate.reasons,
            no_model_beats_null_disclosure=_NO_MODEL_BEATS_NULL_DISCLOSURE,
        )
        subject, body = render_trade_proposal(context)
        key_values = {
            "score": round(ev.score, 4),
            "lcb": round(candidate.lcb_ev or 0.0, 4),
            "p_ko": round(ev.p_ko, 3),
            "horizon": ev.horizon_days,
        }
        n_hash = notification_hash(candidate.candidate_id, "ACTIONABLE", key_values)
        if not dedup.should_send(n_hash):
            continue
        recipients = notifier.credentials.recipients if notifier.credentials is not None else []
        spec = EmailMessageSpec(subject=subject, body_text=body, to=recipients)
        try:
            result = notifier.send(spec)
        except NotificationError as exc:
            logger.error("trade_proposal_send_failed", isin=candidate.isin, error=str(exc))
            _add_warning(warnings, f"trade_proposal_send_failed:{exc}")
            continue
        dedup.mark_sent(n_hash, candidate.candidate_id, "ACTIONABLE", subject, sent_at=clock())
        results.append(result)
    return results


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _run_scan_body(
    *,
    cfg: TurboEdgeConfig,
    store: Store,
    state_dir: str | Path,
    options: ScanOptions,
    product_adapters: Sequence[ProductSourceAdapter],
    price_adapter: PriceSource,
    estr_adapter: EstrSource,
    reference_healthchecks: Sequence[Callable[[], HealthCheckResult]],
    notifier: GmailNotifier | None,
    run_id: str,
    clock: Callable[[], datetime],
    prediction_time: datetime,
    config_hash_value: str,
    git_commit_value: str | None,
    rng: np.random.Generator,
    run_ev: bool,
    forecast_models: Sequence[ForecastModel] | None,
    cluster_id: str | None,
    cluster_open_positions: Sequence[OpenClusterPosition],
) -> ScanResult:
    warnings: list[str] = []
    underlying_id = options.underlying_id

    # 1) source health
    health_records: list[SourceHealthRecord] = []
    product_statuses: list[HealthStatus] = []
    for adapter in product_adapters:
        hc = _safe_healthcheck(adapter.name, adapter.healthcheck, prediction_time)
        product_statuses.append(hc.status)
        health_records.append(from_healthcheck(hc))
    for check in reference_healthchecks:
        hc = _safe_healthcheck("reference", check, prediction_time)
        health_records.append(from_healthcheck(hc))
    if health_records:
        store.append_source_health(health_records)
    data_health_pass = any(status != HealthStatus.FAIL for status in product_statuses)
    if not data_health_pass:
        _add_warning(warnings, "data_health_fail_all_product_sources")

    # 2) underlying bars
    try:
        fetched_bars = price_adapter.fetch_daily_bars(underlying_id, lookback_days=400)
        store.append_underlying_bars(fetched_bars)
    except Exception as exc:
        logger.warning("underlying_bars_fetch_failed", underlying_id=underlying_id, error=str(exc))
        _add_warning(warnings, "underlying_bars_fetch_failed_fallback_store")
        fetched_bars = store.latest_underlying_bars(underlying_id, 400)

    usable_bars = sorted(
        (bar for bar in fetched_bars if bar.available_at <= prediction_time),
        key=lambda bar: bar.ts,
    )
    assert all(bar.available_at <= prediction_time for bar in usable_bars)

    # 3) TSMOM + EWMA vol + gap distribution
    tsmom_cfg_src = cfg.models.tsmom_horizon_norm_v1
    tsmom_cfg = TsmomConfig(
        signal_id=tsmom_cfg_src.signal_id,
        lookbacks=tuple(tsmom_cfg_src.lookbacks),
        ewma_lambda=tsmom_cfg_src.ewma_lambda,
        clip=tsmom_cfg_src.clip,
        threshold=tsmom_cfg_src.threshold,
        version=tsmom_cfg_src.version,
    )

    closes: npt.NDArray[np.float64] = np.array([bar.close for bar in usable_bars], dtype=np.float64)
    sigma_t: float | None = None
    if closes.size >= _MIN_BARS_FOR_VOLATILITY:
        log_returns = np.diff(np.log(closes))
        sigma_series = ewma_volatility(log_returns, lam=tsmom_cfg.ewma_lambda)
        sigma_t = float(sigma_series[-1])

    gap_dist: GapDistribution | None = None
    if len(usable_bars) >= _MIN_BARS_FOR_VOLATILITY:
        gap_dist = gap_distribution_from_bars(usable_bars)

    tsmom_result: TsmomResult | None = None
    try:
        tsmom_result = compute_tsmom(closes, tsmom_cfg)
    except ValueError as exc:
        _add_warning(warnings, f"tsmom_unavailable:{exc}")

    # 4) SignalSnapshot -- persisted before any product/quote fetch
    signal: SignalSnapshot | None = None
    if tsmom_result is not None:
        signal = SignalSnapshot(
            signal_id=f"{run_id}-{underlying_id}",
            signal_version_hash=signal_version_hash(tsmom_cfg),
            underlying_id=underlying_id,
            prediction_time=prediction_time,
            frozen_at=clock(),
            score=tsmom_result.score,
            components=tsmom_result.components,
            direction_hint=tsmom_result.direction_hint,
            threshold=tsmom_cfg.threshold,
            config_hash=config_hash_value,
            git_commit=git_commit_value,
            data_snapshot_hash=data_snapshot_hash(usable_bars),
        )
        store.append_signal(signal)

    # 5) products (same run_id as the enclosing scan -- run_scan() already owns that
    # run's lifecycle in the `runs` table, so this sub-step must not start/finish it again)
    #
    # Befund 2 (2026-09-13 measurement session): the underlying's own
    # same-run daily close is already on hand from step 2 above (usable_bars)
    # -- threading it through as a ProductFetchContext lets a source like
    # gettex sanity-check its own internally-derived reference spot instead
    # of never receiving any cross-check at all (see
    # `adapters.base.ProductFetchContext`/`adapters.gettex` module docstring).
    fetch_context = (
        ProductFetchContext(daily_close_reference={underlying_id: usable_bars[-1].close})
        if usable_bars
        else None
    )
    universe_result: UniverseResult = run_universe(
        cfg,
        store,
        state_dir,
        product_adapters,
        [underlying_id],
        run_id=run_id,
        manage_run=False,
        context=fetch_context,
    )
    for source, err in universe_result.source_errors.items():
        _add_warning(warnings, f"product_source_failed:{source}:{err}")

    products: list[ProductSnapshot] = universe_result.products
    if options.direction is not None:
        products = [p for p in products if p.direction == options.direction]

    # 6+7) pricing + gates
    evaluation_time = clock()
    r = estr_adapter.get_estr()
    next_night_is_weekend = _is_friday_in_berlin(evaluation_time)

    candidates, priced = _process_products(
        cfg=cfg,
        store=store,
        run_id=run_id,
        products=products,
        underlying_id=underlying_id,
        signal=signal,
        sigma_t=sigma_t,
        gap_dist=gap_dist,
        r=r,
        horizon_days=options.horizon_days,
        evaluation_time=evaluation_time,
        next_night_is_weekend=next_night_is_weekend,
        data_health_pass=data_health_pass,
        warnings=warnings,
    )

    # 8.5) forecast -> paths -> EV -> gates -> ledger + shadow sample
    # (Contract v3 Abschnitt B). This is the default, production pipeline --
    # `run_ev` defaults to True in `run_scan()`, so both `turboedge scan`
    # and `turboedge scan-all` run it identically (Befund 1, 2026-09-14
    # measurement session: they used to diverge, `scan` silently staying on
    # the old WATCH-only Phase-1 behavior). `run_ev=False` is an explicit,
    # internal opt-out used only by tests that want to exercise the hard
    # gates (bid_only/stale/leverage/barrier/DATA_QUALITY) in isolation from
    # the slower, stochastic forecast/path-simulation layer -- no production
    # caller ever passes it.
    ledger_entries_written = 0
    forecasts_written = 0
    actionable_notifications: list[SendResult] = []
    new_cluster_positions: list[OpenClusterPosition] = list(cluster_open_positions)
    if run_ev and usable_bars:
        models = list(forecast_models) if forecast_models is not None else default_forecast_models()
        effective_cluster_id = cluster_id if cluster_id is not None else f"single_{underlying_id}"
        ev_result = _run_ev_pipeline(
            cfg=cfg,
            store=store,
            run_id=run_id,
            underlying_id=underlying_id,
            signal=signal,
            usable_bars=usable_bars,
            candidates=candidates,
            priced=priced,
            prediction_time=prediction_time,
            evaluation_time=evaluation_time,
            r=r,
            config_hash_value=config_hash_value,
            git_commit_value=git_commit_value,
            rng=rng,
            forecast_models=models,
            cluster_id=effective_cluster_id,
            cluster_open_positions=cluster_open_positions,
            warnings=warnings,
        )
        candidates = ev_result.candidates
        if ev_result.forecast_records:
            forecasts_written = store.append_forecasts(ev_result.forecast_records)
        if ev_result.ledger_entries:
            ledger_entries_written = ForwardLedger(store).record(ev_result.ledger_entries)
        new_cluster_positions = ev_result.new_cluster_positions
        actionable_notifications = _maybe_send_trade_proposals(
            store=store,
            notifier=notifier,
            underlying_id=underlying_id,
            newly_actionable=ev_result.newly_actionable,
            priced=priced,
            evaluations_by_isin=ev_result.evaluations_by_isin,
            warnings=warnings,
            clock=clock,
        )
    elif run_ev:
        _add_warning(warnings, "ev_skipped_no_underlying_bars")

    counts: dict[Category, int] = dict.fromkeys(Category, 0)
    for candidate in candidates:
        counts[candidate.category] += 1

    # 8) persist candidates + optional scan-report email
    store.append_candidates(candidates)

    notification = _maybe_send_email(
        store=store,
        options=options,
        notifier=notifier,
        run_id=run_id,
        underlying_id=underlying_id,
        signal=signal,
        candidates=candidates,
        counts=counts,
        warnings=warnings,
        clock=clock,
        send_on=cfg.gmail.send_on,
    )

    return ScanResult(
        run_id=run_id,
        signal=signal,
        candidates=candidates,
        counts=counts,
        warnings=warnings,
        health=health_records,
        notification=notification,
        ledger_entries_written=ledger_entries_written,
        forecasts_written=forecasts_written,
        actionable_notifications=actionable_notifications,
        new_cluster_positions=new_cluster_positions,
    )


def run_scan(
    cfg: TurboEdgeConfig,
    store: Store,
    state_dir: str | Path,
    *,
    options: ScanOptions,
    product_adapters: Sequence[ProductSourceAdapter],
    price_adapter: PriceSource,
    estr_adapter: EstrSource,
    reference_healthchecks: Sequence[Callable[[], HealthCheckResult]],
    notifier: GmailNotifier | None,
    run_id: str,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    rng: np.random.Generator | None = None,
    run_ev: bool = True,
    forecast_models: Sequence[ForecastModel] | None = None,
    cluster_id: str | None = None,
    cluster_open_positions: Sequence[OpenClusterPosition] = (),
) -> ScanResult:
    """Run one full scan for ``options.underlying_id``.

    See the module docstring for the binding step order. ``store.runs`` is
    updated with ``status="ok"`` on success or ``status="error"`` (with the
    exception message) if any step raises -- in the error case the original
    exception is re-raised after the run row is closed out, so callers (e.g.
    the CLI) can still map it to a non-zero exit code.

    ``run_ev`` (Contract v3 Abschnitt B integration wave) gates the
    forecast/EV/ledger/ACTIONABLE-mail step (§34): ``True`` (the default)
    runs it, exactly like ``turboedge scan-all`` (the production entry point
    ``pipeline.yml`` schedules) always has. Before 2026-09-14 this was
    instead keyed off ``rng is not None``, which left ``rng=None`` (the
    default, and every pre-existing caller -- including ``turboedge scan``)
    silently skipping the EV step and reporting every candidate's category
    from the WATCH-only cost/integrity gates alone (Befund 1, 2026-09-14
    measurement session: ``turboedge scan`` and ``turboedge scan-all``
    measurably disagreed on the same underlying because of it). ``rng`` is
    now purely about *which* random stream drives the path simulation, not
    *whether* it runs: pass ``None`` (the default) to auto-derive a seeded
    ``numpy.random.Generator`` from ``cfg.simulation.seed`` (CLAUDE.md rule
    16: never a bare unseeded source) -- what every CLI caller does -- or an
    explicit seeded generator (``scan-all``'s multi-underlying caller does
    this so one random stream threads across consecutive ``run_scan``
    calls; see ``pipeline/scan_all.py``). Pass ``run_ev=False`` only to
    exercise the hard gates in isolation from the slower, stochastic
    forecast/path layer (test-only -- no production caller does this).
    ``forecast_models`` defaults to
    :func:`models.forecast.build_default_models` plus any available W9
    challenger models; ``cluster_id``/``cluster_open_positions`` let a
    multi-underlying caller (``scan-all``) thread real cross-underlying
    correlation-cluster state through consecutive calls -- see
    ``pipeline/scan_all.py``.
    """
    prediction_time = clock()
    hash_value = config_hash(cfg)
    commit = git_commit()
    effective_rng = rng if rng is not None else np.random.default_rng(cfg.simulation.seed)

    store.start_run(
        run_id,
        command="scan",
        config_hash=hash_value,
        git_commit=commit,
        started_at=prediction_time,
    )

    try:
        result = _run_scan_body(
            cfg=cfg,
            store=store,
            state_dir=state_dir,
            options=options,
            product_adapters=product_adapters,
            price_adapter=price_adapter,
            estr_adapter=estr_adapter,
            reference_healthchecks=reference_healthchecks,
            notifier=notifier,
            run_id=run_id,
            clock=clock,
            prediction_time=prediction_time,
            config_hash_value=hash_value,
            git_commit_value=commit,
            rng=effective_rng,
            run_ev=run_ev,
            forecast_models=forecast_models,
            cluster_id=cluster_id,
            cluster_open_positions=cluster_open_positions,
        )
    except Exception as exc:
        store.finish_run(run_id, status="error", error=str(exc), finished_at=clock())
        raise
    store.finish_run(run_id, status="ok", finished_at=clock())
    return result


__all__ = [
    "EstrSource",
    "EvPipelineResult",
    "PriceSource",
    "ScanOptions",
    "ScanResult",
    "default_forecast_models",
    "run_scan",
]
