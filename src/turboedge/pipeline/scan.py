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
    7. candidate gates (ACTIONABLE is structurally unreachable this
       milestone: lcb_ev is always None) and ranking
    8. candidates persisted; optional deduplicated Gmail report

Every external dependency (product adapters, the underlying-price source, the
reference-rate source, healthchecks, the notifier, the clock) is injected as
a parameter so the whole pipeline is testable with fakes and needs no real
network access.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import structlog

from turboedge.adapters.base import HealthCheckResult
from turboedge.adapters.registry import ProductSourceAdapter
from turboedge.config import TurboEdgeConfig, config_hash
from turboedge.features.product import (
    distance_to_barrier,
    ewma_volatility,
    freshness_score,
    leverage_bucket,
    quote_age_seconds,
)
from turboedge.features.product import spread_pct as feature_spread_pct
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
from turboedge.notifications.templates import ScanReportContext, ScanReportRow, render_scan_report
from turboedge.pipeline.universe import UniverseResult, run_universe
from turboedge.pricing.cross_issuer import (
    CrossIssuerInput,
    CrossIssuerScores,
    consensus_spot,
    cross_issuer_scores,
)
from turboedge.pricing.financing import (
    financing_cost_over_horizon,
    financing_spread_history,
    realized_financing_spread,
)
from turboedge.pricing.gap_premium import (
    GapDistribution,
    fair_gap_premium,
    gap_distribution_from_bars,
    gap_premium_over_horizon,
)
from turboedge.pricing.integrity import IntegrityReport, check_product
from turboedge.pricing.intrinsic import leverage as compute_leverage
from turboedge.pricing.issuer_margin import decompose_ask
from turboedge.provenance import data_snapshot_hash, git_commit
from turboedge.ranking.gates import GateInput, GateThresholds, evaluate_gates
from turboedge.ranking.liquidity import liquidity_factor, quote_size_coverage, spread_quality
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    HealthStatus,
    ProductSnapshot,
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
    product: ProductSnapshot, needs_fx: bool, eurusd_close: float | None
) -> float | None:
    """FX (units of underlying currency per 1 unit product currency, EUR).

    Quanto products, EUR-underlyings (per underlying_map) and products whose
    reported ``underlying_currency`` already matches the product currency use
    ``fx=1``. Everything else needs the EURUSD daily close; ``None`` means it
    could not be resolved (never imputed, CLAUDE.md rule 29).
    """
    if product.quanto is True:
        return 1.0
    if product.underlying_currency == product.currency:
        return 1.0
    if not needs_fx:
        return 1.0
    return eurusd_close


# --------------------------------------------------------------------------
# per-product pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _PricedProduct:
    """Intermediate per-product pricing state, before cross-issuer scoring."""

    product: ProductSnapshot
    bid: float
    ask: float
    integrity: IntegrityReport
    leverage_value: float
    leverage_bucket_value: str
    spread_pct_value: float
    quote_age_s: float | None
    distance_pct: float | None
    distance_sigma: float | None
    realized_spread: float
    used_default_spread: bool
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
    eurusd_close: float | None,
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

    fx = _resolve_fx(product, needs_fx, eurusd_close)
    if fx is None:
        return _data_quality_candidate(
            run_id=run_id,
            product=product,
            underlying_id=underlying_id,
            horizon_days=horizon_days,
            reasons=["fx_unavailable"],
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
    spread_for_pricing = (
        realized_spread if realized_spread is not None else risk.default_financing_spread
    )

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
    )

    financing_cost_pct: dict[str, float] = {}
    for h in _VALID_HORIZONS:
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
        integrity=integrity,
        leverage_value=lev,
        leverage_bucket_value=lev_bucket,
        spread_pct_value=spread_pct_value,
        quote_age_s=quote_age_s,
        distance_pct=distance_pct,
        distance_sigma=distance_sigma,
        realized_spread=spread_for_pricing,
        used_default_spread=used_default_spread,
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
    price_adapter: PriceSource,
    evaluation_time: datetime,
    next_night_is_weekend: bool,
    data_health_pass: bool,
    warnings: list[str],
) -> list[CandidateEvaluation]:
    risk = cfg.risk
    underlying_meta = get_underlying_meta(underlying_id)
    needs_fx = underlying_meta.currency != "EUR"

    eurusd_close: float | None
    if needs_fx:
        try:
            fx_bars = price_adapter.fetch_daily_bars("EURUSD", lookback_days=5)
        except Exception as exc:
            logger.warning("fx_fetch_failed", underlying_id=underlying_id, error=str(exc))
            fx_bars = []
        if fx_bars:
            eurusd_close = fx_bars[-1].close
            _add_warning(warnings, "fx_daily_close_approximation")
        else:
            eurusd_close = None
            _add_warning(warnings, "fx_unavailable_for_underlying")
    else:
        eurusd_close = 1.0

    consensus_value: float | None
    if needs_fx and eurusd_close is None:
        consensus_value = None
        _add_warning(warnings, "consensus_spot_unavailable_fx_unresolved")
    else:
        group_fx = eurusd_close if needs_fx else 1.0
        assert group_fx is not None
        try:
            consensus_result = consensus_spot(products, fx=group_fx)
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
                eurusd_close=eurusd_close,
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
        if p.used_default_spread:
            reasons.append("financing_spread_default")
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
        )

    return _sort_candidates(list(finalized.values()), signal)


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
) -> SendResult | None:
    if not options.email or notifier is None:
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
    universe_result: UniverseResult = run_universe(
        cfg, store, state_dir, product_adapters, [underlying_id], run_id=run_id, manage_run=False
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

    candidates = _process_products(
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
        price_adapter=price_adapter,
        evaluation_time=evaluation_time,
        next_night_is_weekend=next_night_is_weekend,
        data_health_pass=data_health_pass,
        warnings=warnings,
    )

    counts: dict[Category, int] = dict.fromkeys(Category, 0)
    for candidate in candidates:
        counts[candidate.category] += 1

    # 8) persist candidates + optional email
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
    )

    return ScanResult(
        run_id=run_id,
        signal=signal,
        candidates=candidates,
        counts=counts,
        warnings=warnings,
        health=health_records,
        notification=notification,
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
) -> ScanResult:
    """Run one full scan for ``options.underlying_id``.

    See the module docstring for the binding step order. ``store.runs`` is
    updated with ``status="ok"`` on success or ``status="error"`` (with the
    exception message) if any step raises -- in the error case the original
    exception is re-raised after the run row is closed out, so callers (e.g.
    the CLI) can still map it to a non-zero exit code.
    """
    prediction_time = clock()
    hash_value = config_hash(cfg)
    commit = git_commit()

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
        )
    except Exception as exc:
        store.finish_run(run_id, status="error", error=str(exc), finished_at=clock())
        raise
    store.finish_run(run_id, status="ok", finished_at=clock())
    return result


__all__ = [
    "EstrSource",
    "PriceSource",
    "ScanOptions",
    "ScanResult",
    "run_scan",
]
