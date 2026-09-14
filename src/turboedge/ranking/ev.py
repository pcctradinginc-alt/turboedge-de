"""Product x horizon EV evaluation (Master Spec §16-§19, §15).

:func:`evaluate_product_horizons` is the single entry point: given every
candidate ``ProductTerms`` for one underlying (already filtered to that
underlying by the caller -- ``ProductTerms`` itself carries no
``underlying_id``), a per-horizon ``HorizonForecast`` and that underlying's
own bar history, it simulates the full turbo payoff (``simulation/
payoff.py``, W5) over three drift scenarios per horizon and returns one
:class:`ProductHorizonEvaluation` per ``(isin, horizon)`` pair with every
downstream ranking quantity (shrinkage, LCB, utility, score, suggested
sizing) already computed.

Performance (Build Contract v2 W7): simulated price paths are the expensive
part building them (``simulation/paths.py``: bootstrap + OHLC construction
over ``n_paths``) does not depend on any single product's terms, so this
module builds each ``(direction, horizon, scenario)`` :class:`PathSet`
**exactly once** and reuses it across every candidate product of that
direction -- not once per product. With up to a few thousand candidate
products for one underlying, this turns "N_products x N_horizons x
N_scenarios" path simulations into "N_directions (<=2) x N_horizons x
N_scenarios (2 or 3)" -- typically well under 30 total. See
:func:`evaluate_product_horizons`'s docstring "Performance" section for the
remaining, unavoidable per-product cost and its measured runtime.

Drift scenarios (Build Contract v2 W7 requirement 2) are **direction-aware**:
a forecast's central mean is the same regardless of which direction a
candidate trades, but "pessimistic" must mean *adverse for that candidate*.
For a LONG product, adverse is a lower (or negative) underlying drift; for a
SHORT product, adverse is a higher (or more positive) drift. Concretely, for
horizon ``h`` with ``forecast = forecast_by_horizon[h]``:

- ``central_drift = forecast.mean``
- LONG: ``pessimistic_drift = forecast.mean - z * forecast.uncertainty``,
  ``optimistic_drift = forecast.mean + z * forecast.uncertainty``
- SHORT: ``pessimistic_drift = forecast.mean + z * forecast.uncertainty``,
  ``optimistic_drift = forecast.mean - z * forecast.uncertainty``

This is why paths are keyed by direction: the *same* underlying gets a
different pessimistic-scenario path set depending on which side a candidate
trades.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field

from turboedge.models.forecast import HORIZONS, HorizonForecast
from turboedge.pricing.integrity import IntegrityReport
from turboedge.ranking.gates import GateInput
from turboedge.ranking.lcb import LcbConfig, lower_confidence_bound
from turboedge.ranking.shrinkage import (
    ShrinkageConfig,
    leverage_bucket_for,
    shrink_group_means,
    shrinkage_group_key,
)
from turboedge.ranking.sizing import SizingConfig
from turboedge.ranking.sizing import suggested_position_fraction as _suggested_fraction
from turboedge.ranking.utility import UtilityConfig
from turboedge.ranking.utility import score as _utility_score
from turboedge.simulation.paths import PathSet, simulate_paths
from turboedge.simulation.payoff import ProductTerms, simulate_product_payoff
from turboedge.storage.schemas import Direction, UnderlyingBar

logger = structlog.get_logger(__name__)

_SCENARIO_CENTRAL = "central"
_SCENARIO_PESSIMISTIC = "pessimistic"
_SCENARIO_OPTIMISTIC = "optimistic"


class EvConfig(BaseModel):
    """Own config for this module (wired into ``config.py``/YAML by the
    integration wave). Nests the other ``ranking/*`` modules' own configs so
    one object can be threaded through a scan run; each nested config keeps
    its own defaults/documentation in its own module.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Simulated paths per (direction, horizon, scenario) PathSet.
    n_paths: int = Field(default=2000, gt=0)
    #: Standard-normal multiplier for the pessimistic/optimistic drift
    #: scenarios (Build Contract v2 W7 requirement 2 default: 1.645, the
    #: one-sided ~95th percentile).
    z_pessimistic: float = Field(default=1.645, gt=0.0)
    z_optimistic: float = Field(default=1.645, gt=0.0)
    #: Whether to additionally simulate the optimistic scenario (Build
    #: Contract v2 W7 requirement 2: "optional optimistisch"). Off by
    #: default: the optimistic scenario is not currently consumed by
    #: LCB/utility (both are downside-only), so building it costs a third
    #: path simulation per (direction, horizon) for no effect yet -- left
    #: available for a future symmetric-risk metric.
    include_optimistic: bool = False
    #: ``simulation/paths.py`` method used to build every PathSet. Default
    #: matches ``simulate_paths``'s own default, ``vol_scaled_bootstrap``
    #: (W5 calibration study, 558 real DAX start dates: mean
    #: |simulated - realized P(KO)| at the trading-relevant 1.5-2 sigma
    #: barrier distance is 0.019 for ``vol_scaled_bootstrap`` vs 0.055 for
    #: ``block_bootstrap``, ~2.8x better) -- this module must not silently
    #: pin the older, worse-calibrated method as its own default.
    path_method: str = "vol_scaled_bootstrap"
    block_size: int = Field(default=5, gt=0)
    lookback_days: int = Field(default=750, gt=0)
    #: Liquidity factor substituted for a candidate missing from
    #: ``liquidity_factor_by_isin`` (near-zero, not zero, so it degrades the
    #: score rather than making it exactly zero/undefined -- consistent
    #: with ``ranking/liquidity.py``'s own ``_EPS`` floor).
    default_liquidity_factor: float = Field(default=1e-6, gt=0.0, le=1.0)

    shrinkage: ShrinkageConfig = Field(default_factory=ShrinkageConfig)
    lcb: LcbConfig = Field(default_factory=LcbConfig)
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)


@dataclass(frozen=True, slots=True)
class ProductHorizonEvaluation:
    """Everything computed for one ``(isin, horizon_days)`` candidate
    evaluation (Master Spec §16-§19, §15; Build Contract v2 W7).

    ``p_ko`` is **not a calibrated probability**: ``simulation/paths.py``'s
    own calibration study (558 real DAX start dates) found the simulated
    P(KO) remains conservatively biased (over-predicted) at the trading-
    relevant 1.5-2 sigma barrier distances even under the current default
    method (``vol_scaled_bootstrap``). This module applies no correction
    factor for that bias -- the effect is directional and conservative (it
    understates net EV and overstates knockout risk, so downstream gates
    propose fewer trades rather than riskier ones), so nothing here treats
    ``p_ko`` as a calibrated forecast probability.
    """

    isin: str
    underlying_id: str
    direction: Direction
    horizon_days: int
    mean_net_return: float
    median_net_return: float
    q05: float
    q95: float
    p_profit: float
    p_ko: float
    es95: float
    mfe_median: float
    mae_median: float
    mc_standard_error: float
    shrunk_mean: float
    shrinkage_intensity: float
    lcb_net_return: float
    utility: float
    liquidity_factor: float
    score: float
    suggested_position_fraction: float
    cluster_id: str
    reasons: list[str]


@dataclass(slots=True)
class _RawRecord:
    """Internal, mutable accumulator for one (isin, horizon) pair between
    the path-simulation pass and the shrinkage/scoring pass."""

    isin: str
    direction: Direction
    horizon_days: int
    leverage_bucket: str
    group_key: str
    mean_net_return: float
    median_net_return: float
    q05: float
    q95: float
    p_profit: float
    p_ko: float
    es95: float
    mfe_median: float
    mae_median: float
    mc_standard_error: float
    pessimistic_mean: float
    model_uncertainty: float
    suggested_position_fraction: float
    shrunk_mean: float = 0.0
    shrinkage_intensity: float = 0.0


def _approx_leverage(terms: ProductTerms, spot0: float) -> float:
    """Standard turbo leverage approximation ``(spot * ratio) / (ask * fx)``.

    Used *only* to bucket candidates for shrinkage grouping (Master Spec
    §17 leverage buckets) -- a coarser, better-sourced leverage typically
    already exists in ``CostDecomposition``/``pipeline/scan.py`` for gating
    purposes; this local approximation avoids a hard dependency on that
    pipeline for this module's own grouping needs.
    """
    if not (spot0 > 0.0) or not (terms.entry_ask > 0.0) or not (terms.fx > 0.0):
        return 1.0
    return (spot0 * terms.ratio) / (terms.entry_ask * terms.fx)


def _drift_for_scenario(
    direction: Direction, forecast: HorizonForecast, scenario: str, cfg: EvConfig
) -> float:
    if scenario == _SCENARIO_CENTRAL:
        return forecast.mean
    adverse_is_lower = direction == Direction.LONG
    if scenario == _SCENARIO_PESSIMISTIC:
        sign = -1.0 if adverse_is_lower else 1.0
        return forecast.mean + sign * cfg.z_pessimistic * forecast.uncertainty
    if scenario == _SCENARIO_OPTIMISTIC:
        sign = 1.0 if adverse_is_lower else -1.0
        return forecast.mean + sign * cfg.z_optimistic * forecast.uncertainty
    raise ValueError(f"unknown scenario {scenario!r}")  # pragma: no cover - internal invariant


def _build_paths(
    bars: Sequence[UnderlyingBar],
    forecast_by_horizon: Mapping[int, HorizonForecast],
    *,
    directions: set[Direction],
    horizons: Sequence[int],
    spot0: float,
    start: datetime,
    rng: np.random.Generator,
    cfg: EvConfig,
) -> dict[Direction, dict[int, dict[str, PathSet]]]:
    scenarios = [_SCENARIO_CENTRAL, _SCENARIO_PESSIMISTIC]
    if cfg.include_optimistic:
        scenarios.append(_SCENARIO_OPTIMISTIC)

    paths: dict[Direction, dict[int, dict[str, PathSet]]] = {}
    for direction in sorted(directions, key=lambda d: d.value):
        paths[direction] = {}
        for h in sorted(horizons):
            forecast = forecast_by_horizon[h]
            paths[direction][h] = {}
            for scenario in scenarios:
                drift = _drift_for_scenario(direction, forecast, scenario, cfg)
                paths[direction][h][scenario] = simulate_paths(
                    bars,
                    spot0=spot0,
                    start=start,
                    horizon_days=h,
                    n_paths=cfg.n_paths,
                    rng=rng,
                    drift_log_return=drift,
                    method=cfg.path_method,  # type: ignore[arg-type]
                    block_size=cfg.block_size,
                    lookback_days=cfg.lookback_days,
                )
    return paths


def evaluate_product_horizons(
    terms_by_isin: Mapping[str, ProductTerms],
    forecast_by_horizon: Mapping[int, HorizonForecast],
    bars: Sequence[UnderlyingBar],
    *,
    underlying_id: str,
    spot0: float,
    start: datetime,
    as_of: date,
    cluster_id: str,
    rng: np.random.Generator,
    horizons: Sequence[int] = HORIZONS,
    cfg: EvConfig | None = None,
    cluster_risk: float = 0.0,
    liquidity_factor_by_isin: Mapping[str, float] | None = None,
    leverage_by_isin: Mapping[str, float] | None = None,
    calibration_factor: float = 1.0,
    strategy_posterior_factor: float = 1.0,
    positive_memory_factor: float = 1.0,
    cluster_fraction_used: float = 0.0,
    total_risk_used: float = 0.0,
    fair_value_fn: Callable[..., float] | None = None,
) -> list[ProductHorizonEvaluation]:
    """Evaluate every ``(isin, horizon)`` pair for one underlying's
    candidate products (Master Spec §16-§19).

    ``terms_by_isin`` must all belong to the same ``underlying_id`` (the
    caller filters candidates by underlying before calling this function --
    ``ProductTerms`` carries no ``underlying_id`` field of its own). All of
    ``cluster_id``/``cluster_risk``/``cluster_fraction_used``/
    ``total_risk_used`` are therefore single scalars applying to the whole
    call, not per-candidate: they describe *this underlying's* correlation
    cluster's state, excluding the candidate currently being scored (its
    own marginal contribution is instead governed downstream by
    ``ranking/sizing.py``'s caps and ``ranking/gates.py``'s
    ``cluster_risk_pass`` via :func:`to_candidate_gate_input`, computed
    per-candidate by the caller with ``ranking/cluster.py``).

    ``calibration_factor``, ``strategy_posterior_factor`` and
    ``positive_memory_factor`` default to ``1.0`` (Build Contract v2 W7
    requirement 4): a later integration wave wires these to real per-
    candidate values; until then every score uses the neutral multiplier.

    Performance: builds each ``(direction, horizon, scenario)``
    :class:`~turboedge.simulation.paths.PathSet` exactly once (module
    docstring) and reuses it across every candidate. The remaining,
    unavoidable per-candidate cost is 2 (or 3, with
    ``cfg.include_optimistic``) calls to
    :func:`~turboedge.simulation.payoff.simulate_product_payoff` per
    horizon -- one per drift scenario -- since payoff evaluation is
    necessarily product-specific (financing level, barrier, spread, ...).
    Logs ``evaluate_product_horizons_timing`` (elapsed seconds, candidate
    count, path-build seconds) at INFO on completion; no ISIN/price data is
    logged (CLAUDE.md rule 18).

    Raises:
        ValueError: if ``horizons`` is empty, ``forecast_by_horizon`` is
            missing an entry for any requested horizon, ``spot0 <= 0``, or
            any candidate's ``ProductTerms.direction`` has no corresponding
            entry built (internal invariant).
    """
    if not horizons:
        raise ValueError("horizons must not be empty")
    missing = [h for h in horizons if h not in forecast_by_horizon]
    if missing:
        raise ValueError(f"forecast_by_horizon is missing entries for horizons {missing!r}")
    if not (spot0 > 0.0):
        raise ValueError(f"spot0 must be > 0, got {spot0!r}")

    c = cfg if cfg is not None else EvConfig()
    liquidity_map = liquidity_factor_by_isin or {}
    leverage_map = leverage_by_isin or {}

    if not terms_by_isin:
        return []

    t_start = time.perf_counter()
    directions_present = {t.direction for t in terms_by_isin.values()}
    paths = _build_paths(
        bars,
        forecast_by_horizon,
        directions=directions_present,
        horizons=horizons,
        spot0=spot0,
        start=start,
        rng=rng,
        cfg=c,
    )
    t_paths_built = time.perf_counter()

    records: list[_RawRecord] = []
    for isin, terms in terms_by_isin.items():
        for h in horizons:
            scenario_paths = paths[terms.direction][h]
            central_dist = simulate_product_payoff(
                terms,
                scenario_paths[_SCENARIO_CENTRAL],
                [h],
                as_of=as_of,
                fair_value_fn=fair_value_fn,
            )[h]
            pessimistic_dist = simulate_product_payoff(
                terms,
                scenario_paths[_SCENARIO_PESSIMISTIC],
                [h],
                as_of=as_of,
                fair_value_fn=fair_value_fn,
            )[h]

            leverage = leverage_map.get(isin)
            if leverage is None or not (leverage > 0.0):
                leverage = _approx_leverage(terms, spot0)
            bucket = leverage_bucket_for(leverage)
            group_key = f"{shrinkage_group_key(underlying_id, terms.direction, bucket)}|h{h}"

            # Sizing depends only on the central-scenario net-return sample,
            # P_KO and model uncertainty -- none of which depend on
            # shrinkage/LCB/utility -- so it is computed here, right after
            # central_dist, reusing its net_returns array directly rather
            # than storing the (large) array in _RawRecord or re-simulating
            # the payoff a second time later just to get it back.
            position_fraction = _suggested_fraction(
                central_dist.net_returns,
                central_dist.p_ko,
                forecast_by_horizon[h].uncertainty,
                cluster_fraction_used=cluster_fraction_used,
                total_risk_used=total_risk_used,
                cfg=c.sizing,
            )

            records.append(
                _RawRecord(
                    isin=isin,
                    direction=terms.direction,
                    horizon_days=h,
                    leverage_bucket=bucket,
                    group_key=group_key,
                    mean_net_return=central_dist.mean,
                    median_net_return=central_dist.median,
                    q05=central_dist.q05,
                    q95=central_dist.q95,
                    p_profit=central_dist.p_profit,
                    p_ko=central_dist.p_ko,
                    es95=central_dist.es95,
                    mfe_median=float(np.median(central_dist.mfe)),
                    mae_median=float(np.median(central_dist.mae)),
                    mc_standard_error=central_dist.mc_standard_error,
                    pessimistic_mean=pessimistic_dist.mean,
                    model_uncertainty=forecast_by_horizon[h].uncertainty,
                    suggested_position_fraction=position_fraction,
                )
            )

    # -- shrinkage: group, then shrink each group's raw means together -----
    groups: dict[str, list[_RawRecord]] = {}
    for rec in records:
        groups.setdefault(rec.group_key, []).append(rec)
    for members in groups.values():
        shrunk, intensity = shrink_group_means(
            [m.mean_net_return for m in members],
            [m.mc_standard_error for m in members],
            [m.model_uncertainty for m in members],
            cfg=c.shrinkage,
        )
        for member, shrunk_mean in zip(members, shrunk, strict=True):
            member.shrunk_mean = shrunk_mean
            member.shrinkage_intensity = intensity

    # -- LCB, utility, score, sizing -----------------------------------------
    evaluations: list[ProductHorizonEvaluation] = []
    for rec in records:
        lcb_net_return = lower_confidence_bound(
            rec.mean_net_return,
            rec.pessimistic_mean,
            rec.mc_standard_error,
            rec.shrunk_mean,
            z=c.lcb.z,
        )
        liquidity_factor = liquidity_map.get(rec.isin, c.default_liquidity_factor)
        utility, final_score = _utility_score(
            lcb_net_return,
            rec.es95,
            rec.p_ko,
            rec.model_uncertainty,
            cluster_risk,
            liquidity_factor,
            cfg=c.utility,
            calibration_factor=calibration_factor,
            strategy_posterior_factor=strategy_posterior_factor,
            positive_memory_factor=positive_memory_factor,
        )
        reasons = [
            f"shrinkage_group={rec.group_key}",
            f"shrinkage_intensity={rec.shrinkage_intensity:.4f}",
            f"leverage_bucket={rec.leverage_bucket}",
            f"path_method={c.path_method}",
        ]
        if rec.isin not in liquidity_map:
            reasons.append("liquidity_factor_defaulted")

        evaluations.append(
            ProductHorizonEvaluation(
                isin=rec.isin,
                underlying_id=underlying_id,
                direction=rec.direction,
                horizon_days=rec.horizon_days,
                mean_net_return=rec.mean_net_return,
                median_net_return=rec.median_net_return,
                q05=rec.q05,
                q95=rec.q95,
                p_profit=rec.p_profit,
                p_ko=rec.p_ko,
                es95=rec.es95,
                mfe_median=rec.mfe_median,
                mae_median=rec.mae_median,
                mc_standard_error=rec.mc_standard_error,
                shrunk_mean=rec.shrunk_mean,
                shrinkage_intensity=rec.shrinkage_intensity,
                lcb_net_return=lcb_net_return,
                utility=utility,
                liquidity_factor=liquidity_factor,
                score=final_score,
                suggested_position_fraction=rec.suggested_position_fraction,
                cluster_id=cluster_id,
                reasons=reasons,
            )
        )

    t_end = time.perf_counter()
    logger.info(
        "evaluate_product_horizons_timing",
        underlying_id=underlying_id,
        n_candidates=len(terms_by_isin),
        n_horizons=len(horizons),
        n_evaluations=len(evaluations),
        n_paths=c.n_paths,
        path_build_s=round(t_paths_built - t_start, 3),
        total_s=round(t_end - t_start, 3),
    )
    return evaluations


def to_candidate_gate_input(
    evaluation: ProductHorizonEvaluation,
    *,
    integrity: IntegrityReport,
    bid_only: bool,
    knocked_out: bool,
    quote_age_s: float | None,
    spread_pct: float | None,
    leverage: float | None,
    distance_to_barrier_sigma: float | None,
    data_health_pass: bool,
    cluster_risk_pass: bool,
    has_ask: bool = True,
    no_live_quote: bool = False,
    source_quote_age_s: float | None = None,
) -> GateInput:
    """Build a ``ranking/gates.py`` :class:`GateInput` from one
    :class:`ProductHorizonEvaluation` plus the market/data-quality facts
    that ``ranking/ev.py`` itself does not compute (Build Contract v2 W7
    requirement 7).

    ``ranking/gates.py`` is not modified: it already assigns ``ACTIONABLE``
    whenever ``lcb_ev > 0 and p_ko is not None and cluster_risk_pass is
    True`` (and every REJECT/DATA_QUALITY gate passes) -- this function only
    supplies ``lcb_ev=evaluation.lcb_net_return`` and
    ``p_ko=evaluation.p_ko`` from the evaluation, plus the market-quality
    fields the caller must supply separately (they come from
    ``ProductSnapshot``/``CandidateEvaluation``, not from this module).
    ``cluster_risk_pass`` must be computed by the caller via
    ``ranking/cluster.cluster_risk_pass`` (it depends on the full set of
    open positions and other already-selected candidates this scan, not
    just this one evaluation).
    """
    return GateInput(
        integrity=integrity,
        bid_only=bid_only,
        knocked_out=knocked_out,
        quote_age_s=quote_age_s,
        spread_pct=spread_pct,
        leverage=leverage,
        distance_to_barrier_sigma=distance_to_barrier_sigma,
        data_health_pass=data_health_pass,
        lcb_ev=evaluation.lcb_net_return,
        p_ko=evaluation.p_ko,
        cluster_risk_pass=cluster_risk_pass,
        has_ask=has_ask,
        no_live_quote=no_live_quote,
        source_quote_age_s=source_quote_age_s,
    )


__all__ = [
    "EvConfig",
    "ProductHorizonEvaluation",
    "evaluate_product_horizons",
    "to_candidate_gate_input",
]
