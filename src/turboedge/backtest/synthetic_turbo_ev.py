"""Harness for pre-registration 2026Q4-001 (``docs/preregistration_2026Q4_001.md``).

**Do not call :func:`run_synthetic_net_ev_trial` against real fetched price
history before 2026-10-01.** The pre-registration this module implements
forbids any measurement before that date -- producing the number early would
destroy the confirmatory status the whole document exists to protect. This
module is a harness: import and unit-test everything in it against
constructed synthetic bars (as this package's own test suite does), never
against ``adapters/fallback_prices.py``-fetched data, until the pre-
registered run date.

WHY this module exists (pre-registration §1/§3/§4): §6.11-§6.13 of
``docs/measured_results.md`` produced *retrospective* evidence that
``RegimeConditionalEmpiricalModel`` improves forecast CRPS, but chose its own
significance test (a moving-block bootstrap) after seeing the positive
result -- not confirmatory. Separately, §6.10 showed a better predicted
*distribution* is not automatically a better predicted *point forecast*, and
is not automatically economically valuable once priced through turbo costs,
KO risk and spread. This module answers the second question on a fixed,
frozen method, for one pre-registered primary hypothesis:

    H0: NetEV(regime_conditional) - NetEV(null) <= 0
    H1: NetEV(regime_conditional) - NetEV(null)  > 0

**One primary p-value** (one-sided, alpha=0.10, moving-block bootstrap over
prediction dates, block length 21). Everything else this module computes is
stability analysis (pre-registration §6): reported, never deflated as an
independent primary test, never promoted to primary after the fact -- kept
in a structurally separate :class:`StabilityAnalysis` field so a reader
cannot mistake one for the other.

Design, following the pre-registration exactly (§4):

- :func:`build_standardised_universe` depends on ``spot``/``config`` only --
  never a forecast -- so the identical ``dict[str, ProductTerms]`` object is
  reused for both arms at a given prediction date. Any measured difference
  in NetEV therefore attributes to the forecast, per the pre-registration's
  own reasoning, and to nothing else.
- ``entry_ask`` is the theoretical fair value (``pricing/fair_value.py::
  theoretical_fair_value`` -- **not** ``pricing/intrinsic.py``, which has no
  such function; this prompt's file pointer was wrong and is corrected here,
  noted in the trial report rather than picked silently).
- Walk-forward reuses ``backtest/walkforward.py::walk_forward_evaluate``
  (never a second walk-forward loop), called once per horizon with
  ``embargo=horizon`` (pre-registration §4, matching
  ``docs/measured_results.md`` §6.12's own reproducibility record) --
  *not* once for the whole horizon ladder with one shared embargo, which is
  what a single ``walk_forward_evaluate(..., horizons=HORIZONS, embargo=X)``
  call would give.
- ``ranking/ev.py::evaluate_product_horizons`` is reused for the actual
  payoff/knockout/cost pricing. Its ``ProductHorizonEvaluation.mean_net_return``
  (the *central*-scenario simulated mean net return -- i.e. paths built with
  drift = the model's own point forecast, per that module's own scenario
  design) is used as "the aggregate simulated net expected return... priced
  through the existing payoff/knock-out/cost machine" (pre-registration §1).

**Known interpretation flagged, not silently resolved** (see this trial's
report, not just this docstring): ``evaluate_product_horizons``'s central
scenario consumes only ``forecast.mean`` as path drift -- ``forecast.sigma``/
``uncertainty``/quantiles feed the *pessimistic* scenario (LCB/utility/
sizing), never the central-scenario mean-net-return number this module reads
as NetEV. So the primary statistic computed here is driven entirely by each
model's point-mean forecast difference, not by any improvement to the
*shape* of the predicted distribution that CRPS (docs/measured_results.md
§6.11-13) actually measures. This is the existing EV pipeline's own design,
reused unmodified per the brief ("reuse it rather than writing a second..."),
not a choice made here -- flagged because it bears directly on what a PASS
or FAIL in this trial can and cannot be read as evidence of.

**Second known interpretation flagged**: pre-registration §6 asks for
"sensitivity of the sign of the result to `spread` at 0.0025 and 0.01".
Under the pre-registration's own §4 formula (`entry_ask` = fair value,
`entry_bid = entry_ask * (1 - spread)`) and
``simulation/payoff.py::simulate_product_payoff``'s actual contract (every
net_return is computed from ``terms.entry_ask`` alone; ``entry_bid`` is
stored on ``ProductTerms`` but never read by the payoff engine), no value of
``spread`` can change any simulated NetEV in this harness. The sensitivity
check is implemented faithfully (see :func:`_spread_sensitivity_mean_delta`)
but is proven, not merely observed, to be a no-op given the current engine
contract -- see that function's docstring and this trial's report.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from turboedge.backtest.walkforward import walk_forward_evaluate
from turboedge.models.baselines import RegimeConditionalEmpiricalModel
from turboedge.models.directional import NullModel
from turboedge.models.forecast import HORIZONS, ForecastModel, HorizonForecast
from turboedge.pricing.fair_value import theoretical_fair_value
from turboedge.ranking.ev import EvConfig, evaluate_product_horizons
from turboedge.simulation.payoff import ProductTerms
from turboedge.storage.schemas import Direction, ProductType, UnderlyingBar

#: Walk-forward parameters, frozen identical to pre-registration §4 /
#: ``docs/measured_results.md`` §6.12 -- not part of ``SyntheticTurboConfig``
#: because the pre-registration treats them as fixed method, not a tunable
#: (§4: "No configuration tuning, no variant selection").
_MIN_TRAIN = 750
_STEP = 21

#: Primary-statistic bootstrap parameters (pre-registration §4/§6.12).
_ALPHA = 0.10
_BLOCK_LENGTH = 21
_N_BOOTSTRAP_RESAMPLES = 20_000

#: ``theoretical_fair_value`` requires an ``as_of`` date, but the
#: standardised universe is built exclusively from ``turbo_open_end``
#: products, whose fair value is plain intrinsic value and does not read
#: ``as_of``/``maturity`` at all (see ``pricing/fair_value.py``'s
#: ``theoretical_fair_value`` -- the non-``TURBO_CLASSIC`` branch returns
#: before either is touched). This sentinel documents that unused-ness
#: rather than threading a real prediction date through a function
#: ("depends only on spot and config") that must not depend on one.
_UNUSED_AS_OF = date(1970, 1, 1)

_SYNTHETIC_ISIN_PREFIX = "SYNTH"


@dataclass(frozen=True, slots=True)
class SyntheticTurboConfig:
    """Standardised-universe and simulation parameters (pre-registration §4)."""

    barrier_distances: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20)
    ratio: float = 0.01
    fx: float = 1.0
    spread: float = 0.005
    financing_spread: float = 0.02
    ref_rate: float = 0.02
    premium_over_fair: float = 0.0
    exit_spread_pct: float = 0.005
    n_paths: int = 2000
    seed: int = 20261001


def _synthetic_isin(direction: Direction, distance: float) -> str:
    """Deterministic, parseable ISIN stand-in for one standardised-grid product."""
    return f"{_SYNTHETIC_ISIN_PREFIX}-{direction.value.upper()}-{distance:.4f}"


def _parse_synthetic_isin(isin: str) -> tuple[Direction, float]:
    """Inverse of :func:`_synthetic_isin` -- recovers the grid cell's own direction/distance
    for the stability breakdown (pre-registration §6), without threading a second parallel
    structure alongside ``dict[str, ProductTerms]`` through the whole pipeline."""
    prefix, direction_str, distance_str = isin.split("-")
    if prefix != _SYNTHETIC_ISIN_PREFIX:
        raise ValueError(f"not a synthetic-universe isin: {isin!r}")
    return Direction(direction_str.lower()), float(distance_str)


def build_standardised_universe(
    spot: float, *, config: SyntheticTurboConfig
) -> dict[str, ProductTerms]:
    """The fixed standardised turbo grid for one prediction date's spot (pre-registration §4).

    Depends on ``spot``/``config`` only -- **never** a forecast -- so the
    identical returned ``dict`` can be (and, in
    :func:`run_synthetic_net_ev_trial`, is) passed unchanged to both arms'
    ``evaluate_product_horizons`` call for a given date: the only thing that
    differs between arms is the forecast, so any measured NetEV difference
    attributes to the forecast alone (§4: "these terms are identical across
    both arms at every date... any result therefore attributes to the
    forecast and to nothing else").

    One product per (barrier distance, direction): distances below spot are
    long, above spot are short (§4), ``financing_level == knockout_barrier``
    (open-end convention -- ``ProductType.TURBO_OPEN_END``, per
    ``storage/schemas.py``'s own convention comment), ``entry_ask`` is the
    theoretical fair value with no premium applied at entry (§4, verbatim:
    "entry_ask = theoretical fair value" -- distinct from
    ``premium_over_fair``, which this module carries forward at 0.0 and
    which only ever affects *exit* valuation inside
    ``simulation/payoff.py``), and ``entry_bid = entry_ask * (1 - spread)``.

    Raises:
        ValueError: if ``spot <= 0`` or any ``config.barrier_distances``
            entry is not within ``(0, 1)``.
    """
    if not (spot > 0.0):
        raise ValueError(f"spot must be > 0, got {spot!r}")

    universe: dict[str, ProductTerms] = {}
    for distance in config.barrier_distances:
        if not (0.0 < distance < 1.0):
            raise ValueError(f"barrier_distances must be within (0, 1), got {distance!r}")
        # Long: barrier below spot (adverse move down knocks out). Short:
        # barrier above spot (adverse move up knocks out) -- §4.
        for direction, sign in ((Direction.LONG, -1.0), (Direction.SHORT, 1.0)):
            barrier = spot * (1.0 + sign * distance)
            fair_value = theoretical_fair_value(
                direction=direction,
                product_type=ProductType.TURBO_OPEN_END,
                spot=spot,
                financing_level=barrier,
                knockout_barrier=barrier,
                ratio=config.ratio,
                fx=config.fx,
                ref_rate=config.ref_rate,
                financing_spread=config.financing_spread,
                as_of=_UNUSED_AS_OF,
                maturity=None,
                dividend_yield=0.0,
            )
            entry_ask = fair_value
            entry_bid = entry_ask * (1.0 - config.spread)
            isin = _synthetic_isin(direction, distance)
            universe[isin] = ProductTerms(
                isin=isin,
                direction=direction,
                product_type=ProductType.TURBO_OPEN_END,
                financing_level=barrier,
                knockout_barrier=barrier,
                ratio=config.ratio,
                fx=config.fx,
                entry_ask=entry_ask,
                entry_bid=entry_bid,
                maturity=None,
                financing_spread=config.financing_spread,
                ref_rate=config.ref_rate,
                exit_spread_pct=config.exit_spread_pct,
                premium_over_fair=config.premium_over_fair,
            )
    return universe


class ArmResult(BaseModel):
    """One model's ("arm's") results: the aggregate NetEV plus the per-date series behind it.

    ``dates``/``grid_mean_net_ev_by_date``/``n_cells_by_date`` are
    positionally aligned (same length, same order); ``n_cells_by_date``
    records how many ``(underlying, isin, horizon)`` grid cells contributed
    to that date's mean, which is diagnostic for how much the walk-forward's
    own horizon-tail truncation (longer horizons run out of test dates near
    the end of history first) thinned a given date's coverage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    dates: list[date]
    grid_mean_net_ev_by_date: list[float]
    n_cells_by_date: list[int]
    aggregate_mean_net_ev: float


class StabilityAnalysis(BaseModel):
    """Pre-registration §6: secondary stability analysis, reported, never primary, never
    deflated against a primary alpha, never promotable to primary after the fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delta_net_ev_by_underlying: dict[str, float]
    delta_net_ev_by_horizon: dict[int, float]
    delta_net_ev_by_barrier_distance: dict[float, float]
    share_cells_delta_positive: float = Field(ge=0.0, le=1.0)
    n_cells: int = Field(gt=0)
    #: Mean delta NetEV recomputed (pre-registration §6) at ``spread`` in
    #: {0.0025, 0.01} -- keys ``"spread_0.0025"``/``"spread_0.01"``. See
    #: :func:`_spread_sensitivity_mean_delta` for why, under the current
    #: payoff-engine contract, these are proven equal to the primary
    #: ``mean_delta_net_ev`` rather than independently re-simulated.
    spread_sensitivity_mean_delta: dict[str, float]
    #: Whether the sign of ``mean_delta_net_ev`` is unchanged at both probe
    #: spread values -- the reader-facing summary pre-registration §6 asks
    #: for ("the reader must be able to see that without re-running
    #: anything").
    spread_sensitivity_sign_stable: bool


class SyntheticEvResult(BaseModel):
    """Pre-registration 2026Q4-001 result: exactly one primary p-value (§4/§5) plus the
    stability analysis (§6), kept in a structurally separate nested field so a reader cannot
    mistake secondary evidence for the primary test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    null_arm: ArmResult
    regime_conditional_arm: ArmResult
    #: Primary statistic (§4): mean over dates of (regime_conditional - null).
    mean_delta_net_ev: float
    #: Primary, one-sided, moving-block-bootstrap p-value (§4).
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)
    block_length: int = Field(gt=0)
    n_bootstrap: int = Field(gt=0)
    bootstrap_seed: int
    n_dates: int = Field(gt=0)
    #: Pre-registration §5's decision table, verbatim:
    #: PASS = p<=alpha and mean_delta>0; FAIL_NULL_RESULT = p>alpha;
    #: FAIL_NEGATIVE_SIGNIFICANT = p<=alpha and mean_delta<=0 (§5 row 3: "a
    #: more informative negative than a null result").
    verdict: Literal["PASS", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT"]
    verdict_reason: str
    stability: StabilityAnalysis


@dataclass(frozen=True, slots=True)
class _CellRecord:
    """One ``(prediction date, underlying, isin, horizon)`` grid cell's NetEV, both arms."""

    date_key: date
    underlying_id: str
    isin: str
    direction: Direction
    barrier_distance: float
    horizon_days: int
    net_ev_null: float
    net_ev_regime_conditional: float

    @property
    def delta(self) -> float:
        return self.net_ev_regime_conditional - self.net_ev_null


def _collect_oos_forecasts_by_date(
    bars: Sequence[UnderlyingBar],
    model_factory: Callable[[], ForecastModel],
) -> dict[date, tuple[datetime, dict[int, HorizonForecast]]]:
    """Walk-forward OOS forecasts for every horizon, indexed by calendar date.

    Calls ``walk_forward_evaluate`` once per horizon in ``HORIZONS`` with
    ``embargo=h`` (pre-registration §4: "embargo=horizon", identical to
    ``docs/measured_results.md`` §6.12's own reproducibility record) --
    deliberately *not* one call across the whole horizon ladder with a
    single shared ``embargo``, which is what ``walk_forward_evaluate``
    itself would apply if given ``horizons=HORIZONS`` directly.

    The returned mapping's value pairs the full ``datetime`` (needed to look
    the source bar back up, for its ``close``/``available_at``) with a
    per-horizon forecast map; different horizons generally have slightly
    different date coverage (the walk-forward's own
    ``if i + h >= n: continue`` guard drops the last ``h`` bars' worth of
    dates for horizon ``h``), which the caller resolves by intersecting.
    """
    by_date: dict[date, tuple[datetime, dict[int, HorizonForecast]]] = {}
    for h in HORIZONS:
        [result] = walk_forward_evaluate(
            model_factory, bars, [h], min_train=_MIN_TRAIN, step=_STEP, embargo=h
        )
        for ts, forecast, _realized_return in result.oos_forecasts:
            d = ts.date()
            _, forecast_map = by_date.setdefault(d, (ts, {}))
            forecast_map[h] = forecast
    return by_date


def _run_trial_grid(
    bars_by_underlying: Mapping[str, Sequence[UnderlyingBar]],
    cfg: SyntheticTurboConfig,
) -> list[_CellRecord]:
    """Walk-forward both arms, then price the identical standardised grid (§4) at every
    common prediction date, for every underlying. No look-ahead: ``evaluate_product_horizons``'s
    own ``bars``/``start`` filtering (``simulation/paths.py::simulate_paths``) excludes any bar
    at or after ``start`` from the path-simulation bootstrap sample, and each model was fit
    only on bars up to that fold's training bar's ``available_at`` inside
    ``walk_forward_evaluate`` -- this function adds no additional data access of its own.
    """
    records: list[_CellRecord] = []
    for underlying_id, bars in bars_by_underlying.items():
        bars_list = list(bars)
        if not bars_list:
            raise ValueError(f"bars_by_underlying[{underlying_id!r}] must not be empty")
        bars_by_ts = {b.ts: b for b in bars_list}

        null_by_date = _collect_oos_forecasts_by_date(bars_list, NullModel)
        regime_by_date = _collect_oos_forecasts_by_date(bars_list, RegimeConditionalEmpiricalModel)
        common_dates = sorted(set(null_by_date) & set(regime_by_date))

        for d in common_dates:
            ts, null_forecast_map = null_by_date[d]
            _, regime_forecast_map = regime_by_date[d]
            common_horizons = sorted(set(null_forecast_map) & set(regime_forecast_map))
            if not common_horizons:
                continue
            bar = bars_by_ts.get(ts)
            if bar is None:  # pragma: no cover -- internal invariant
                raise AssertionError(
                    f"prediction ts {ts!r} not found in bars_by_underlying[{underlying_id!r}]"
                )

            # Same dict object handed to both arms (§4 identity requirement) --
            # built once per date, not once per arm.
            universe = build_standardised_universe(bar.close, config=cfg)
            cluster_id = f"synthetic::{underlying_id}"
            ev_cfg = EvConfig(n_paths=cfg.n_paths)

            null_evals = evaluate_product_horizons(
                universe,
                {h: null_forecast_map[h] for h in common_horizons},
                bars_list,
                underlying_id=underlying_id,
                spot0=bar.close,
                start=bar.available_at,
                as_of=d,
                cluster_id=cluster_id,
                # Fresh Generator, same seed, for both arms (§4: "identical
                # across arms") -- common random numbers, so any difference
                # in path draws is impossible and any NetEV difference
                # attributes to the forecast alone.
                rng=np.random.default_rng(cfg.seed),
                horizons=common_horizons,
                cfg=ev_cfg,
            )
            regime_evals = evaluate_product_horizons(
                universe,
                {h: regime_forecast_map[h] for h in common_horizons},
                bars_list,
                underlying_id=underlying_id,
                spot0=bar.close,
                start=bar.available_at,
                as_of=d,
                cluster_id=cluster_id,
                rng=np.random.default_rng(cfg.seed),
                horizons=common_horizons,
                cfg=ev_cfg,
            )

            null_by_key = {(e.isin, e.horizon_days): e for e in null_evals}
            regime_by_key = {(e.isin, e.horizon_days): e for e in regime_evals}
            # A candidate can be excluded per-arm by ev.py's own
            # implausible-magnitude guard; never silently defaulted here --
            # only cells priced by *both* arms enter the paired delta.
            common_keys = sorted(set(null_by_key) & set(regime_by_key))
            for isin, horizon_days in common_keys:
                direction, distance = _parse_synthetic_isin(isin)
                records.append(
                    _CellRecord(
                        date_key=d,
                        underlying_id=underlying_id,
                        isin=isin,
                        direction=direction,
                        barrier_distance=distance,
                        horizon_days=horizon_days,
                        net_ev_null=null_by_key[(isin, horizon_days)].mean_net_return,
                        net_ev_regime_conditional=regime_by_key[
                            (isin, horizon_days)
                        ].mean_net_return,
                    )
                )
    return records


def _moving_block_bootstrap_p_value(
    delta_by_date: npt.NDArray[np.float64],
    *,
    block_length: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    """One-sided moving-block bootstrap p-value for H0: mean(delta_by_date) <= 0.

    Models the approach in ``docs/measured_results.md`` §6.12: the observed
    series is re-centred so the null holds exactly (mean 0) in the
    resampling population, then resampled in circular blocks of
    ``block_length`` consecutive dates -- preserving whatever serial
    dependence exists within a block, which a plain i.i.d. bootstrap would
    destroy -- to build the null distribution of the block-mean statistic.
    The p-value is the one-sided fraction of null-world resample means at
    least as large as the actually observed mean, with add-one (Laplace)
    smoothing (as ``backtest/significance.py::bootstrap_p_value`` already
    does) so a p-value of exactly 0 is never reported from a finite number
    of resamples.

    Circular (wrap-around) block starts: a block starting near the end of
    the series wraps to its beginning rather than being excluded, which
    avoids under-sampling the tail (Politis & Romano's standard circular
    block bootstrap construction).

    Raises:
        ValueError: if ``delta_by_date`` is empty or ``block_length < 1``.
    """
    n = delta_by_date.shape[0]
    if n == 0:
        raise ValueError("delta_by_date must not be empty")
    if block_length < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length!r}")

    observed = float(np.mean(delta_by_date))
    centered = delta_by_date - observed  # H0 holds exactly in the resampling population
    n_blocks = -(-n // block_length)  # ceil division: enough blocks to cover length n
    block_offsets = np.arange(block_length)

    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + block_offsets[None, :]) % n
        boot_means[i] = float(np.mean(centered[idx.reshape(-1)[:n]]))

    extreme = int(np.sum(boot_means >= observed))
    return float((extreme + 1) / (n_resamples + 1))


def _verdict(
    p_value: float, mean_delta_net_ev: float, alpha: float
) -> tuple[Literal["PASS", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT"], str]:
    """Pre-registration §5's decision table, verbatim, stated before any result existed."""
    if p_value <= alpha and mean_delta_net_ev > 0.0:
        return (
            "PASS",
            f"p={p_value:.4f} <= alpha={alpha} and mean delta NetEV={mean_delta_net_ev:.6f} > 0: "
            "the distribution improvement carries economic information on standardised terms "
            "(pre-registration §5 row 1). Necessary condition met; forward real-product arm "
            "becomes the next trial. Still no promotion.",
        )
    if p_value > alpha:
        return (
            "FAIL_NULL_RESULT",
            f"p={p_value:.4f} > alpha={alpha}: statistically indistinguishable from no "
            "difference on standardised terms (pre-registration §5 row 2). The CRPS "
            "improvement is statistically interesting and economically inert on these terms.",
        )
    return (
        "FAIL_NEGATIVE_SIGNIFICANT",
        f"p={p_value:.4f} <= alpha={alpha} but mean delta NetEV={mean_delta_net_ev:.6f} <= 0: "
        "reported as evidence the better distribution is actively worse economically "
        "(pre-registration §5 row 3) -- a more informative negative than a null result.",
    )


def _spread_sensitivity_mean_delta(
    primary_mean_delta_net_ev: float, cfg: SyntheticTurboConfig
) -> dict[str, float]:
    """Pre-registration §6: "sensitivity of the sign of the result to `spread` at 0.0025 and
    0.01" -- so a result whose sign flips under a halved spread assumption is visible without
    re-running anything (§6, citing §6.10's leverage-amplification finding).

    Under the pre-registration's own §4 formula, ``spread`` only ever enters
    ``ProductTerms.entry_bid`` (``entry_ask`` is the theoretical fair value,
    unconditional on ``spread``); ``simulation/payoff.py::
    simulate_product_payoff`` computes every net_return from ``entry_ask``
    alone and never reads ``entry_bid``. So no value of ``spread`` can
    change any simulated NetEV in this harness, and therefore cannot change
    the sign of the primary result either -- this is a structural fact about
    the pre-registration's formula composed with the existing payoff
    engine's contract, not an empirical finding that needs 2x the (already
    expensive) path-simulation re-run to observe. It is verified directly
    below (entry_ask invariance under both probe ``spread`` values, on the
    real ``build_standardised_universe``) and raises loudly if that
    invariant is ever broken by a future change to either module, rather
    than silently reporting a stale number.
    """
    probe_spot = 100.0  # arbitrary and irrelevant: entry_ask's spread-independence holds for
    # any spot > 0, since `spread` never enters the fair-value computation at all.
    baseline_universe = build_standardised_universe(probe_spot, config=cfg)
    sensitivity: dict[str, float] = {}
    for probe_spread in (0.0025, 0.01):
        probe_cfg = replace(cfg, spread=probe_spread)
        probe_universe = build_standardised_universe(probe_spot, config=probe_cfg)
        for isin, probe_terms in probe_universe.items():
            baseline_terms = baseline_universe[isin]
            if probe_terms.entry_ask != baseline_terms.entry_ask:  # pragma: no cover
                raise AssertionError(
                    "spread changed entry_ask -- the payoff-engine spread-invariance this "
                    "stability shortcut relies on no longer holds; recompute "
                    "mean_delta_net_ev at this spread via a full _run_trial_grid rerun "
                    "instead of reusing the primary result"
                )
        sensitivity[f"spread_{probe_spread}"] = primary_mean_delta_net_ev
    return sensitivity


def _stability_analysis(
    records: Sequence[_CellRecord], primary_mean_delta_net_ev: float, cfg: SyntheticTurboConfig
) -> StabilityAnalysis:
    """Pre-registration §6: per-underlying/per-horizon/per-barrier-distance delta NetEV, the
    share of cells with a positive delta, and the spread sign-sensitivity -- all secondary."""
    if not records:
        raise ValueError("records must not be empty")

    by_underlying: dict[str, list[float]] = {}
    by_horizon: dict[int, list[float]] = {}
    by_distance: dict[float, list[float]] = {}
    n_positive = 0
    for rec in records:
        by_underlying.setdefault(rec.underlying_id, []).append(rec.delta)
        by_horizon.setdefault(rec.horizon_days, []).append(rec.delta)
        by_distance.setdefault(rec.barrier_distance, []).append(rec.delta)
        if rec.delta > 0.0:
            n_positive += 1

    spread_sensitivity = _spread_sensitivity_mean_delta(primary_mean_delta_net_ev, cfg)

    return StabilityAnalysis(
        delta_net_ev_by_underlying={k: float(np.mean(v)) for k, v in by_underlying.items()},
        delta_net_ev_by_horizon={k: float(np.mean(v)) for k, v in by_horizon.items()},
        delta_net_ev_by_barrier_distance={k: float(np.mean(v)) for k, v in by_distance.items()},
        share_cells_delta_positive=n_positive / len(records),
        n_cells=len(records),
        spread_sensitivity_mean_delta=spread_sensitivity,
        spread_sensitivity_sign_stable=all(
            (v > 0.0) == (primary_mean_delta_net_ev > 0.0) for v in spread_sensitivity.values()
        ),
    )


def run_synthetic_net_ev_trial(
    bars_by_underlying: Mapping[str, Sequence[UnderlyingBar]],
    *,
    config: SyntheticTurboConfig | None = None,
) -> SyntheticEvResult:
    """Run the pre-registration 2026Q4-001 historical-synthetic trial (§4/§5/§6).

    Takes bars as an argument rather than fetching them itself, so it is
    testable against constructed synthetic bars, and so the caller (which
    does the real fetch, per the pre-registration §4 data provenance
    requirement -- adapter, ``lookback_days``, fetch date, per-underlying
    bar counts, skip summary) is the one that records that provenance
    alongside this function's result, rather than this function silently
    reaching for a live adapter of its own.

    For each underlying in ``bars_by_underlying``: both frozen models
    (``models.directional.NullModel``, ``models.baselines.
    RegimeConditionalEmpiricalModel``) are walked forward per horizon
    (``PurgedWalkForwardSplit`` via ``walk_forward_evaluate``,
    ``min_train=750``, ``step=21``, ``embargo=horizon``); at every
    prediction date common to both arms and to at least one horizon, the
    identical standardised grid (:func:`build_standardised_universe`) is
    priced under each arm's forecast via ``evaluate_product_horizons``
    (common random numbers: a fresh ``Generator(seed)`` for each arm's
    call). The primary statistic is the mean over dates of the
    whole-grid-mean NetEV difference (regime_conditional - null); its
    one-sided moving-block-bootstrap p-value and the pre-registration §5
    verdict are returned alongside the §6 stability analysis.

    Raises:
        ValueError: if ``bars_by_underlying`` is empty, any underlying's
            bars are empty, or no ``(date, underlying, horizon)`` cell could
            be evaluated at all (e.g. insufficient history for
            ``min_train=750``) -- propagated from
            ``walk_forward_evaluate``/this function rather than returning a
            result built from zero data.
    """
    cfg = config if config is not None else SyntheticTurboConfig()
    if not bars_by_underlying:
        raise ValueError("bars_by_underlying must not be empty")

    cell_records = _run_trial_grid(bars_by_underlying, cfg)
    if not cell_records:
        raise ValueError(
            "no (date, underlying, horizon) cells were evaluated -- check that "
            f"bars_by_underlying has enough history for min_train={_MIN_TRAIN}, "
            f"step={_STEP} plus the longest horizon ({max(HORIZONS)}d)"
        )

    by_date: dict[date, list[_CellRecord]] = {}
    for rec in cell_records:
        by_date.setdefault(rec.date_key, []).append(rec)
    dates_sorted = sorted(by_date)

    null_means = [float(np.mean([r.net_ev_null for r in by_date[d]])) for d in dates_sorted]
    regime_means = [
        float(np.mean([r.net_ev_regime_conditional for r in by_date[d]])) for d in dates_sorted
    ]
    n_cells_by_date = [len(by_date[d]) for d in dates_sorted]
    delta_by_date = np.asarray(regime_means, dtype=np.float64) - np.asarray(
        null_means, dtype=np.float64
    )
    mean_delta_net_ev = float(np.mean(delta_by_date))

    bootstrap_rng = np.random.default_rng(cfg.seed)
    p_value = _moving_block_bootstrap_p_value(
        delta_by_date,
        block_length=_BLOCK_LENGTH,
        n_resamples=_N_BOOTSTRAP_RESAMPLES,
        rng=bootstrap_rng,
    )

    verdict, verdict_reason = _verdict(p_value, mean_delta_net_ev, _ALPHA)
    stability = _stability_analysis(cell_records, mean_delta_net_ev, cfg)

    null_arm = ArmResult(
        model_id=NullModel().model_id,
        dates=dates_sorted,
        grid_mean_net_ev_by_date=null_means,
        n_cells_by_date=n_cells_by_date,
        aggregate_mean_net_ev=float(np.mean(null_means)),
    )
    regime_arm = ArmResult(
        model_id=RegimeConditionalEmpiricalModel().model_id,
        dates=dates_sorted,
        grid_mean_net_ev_by_date=regime_means,
        n_cells_by_date=n_cells_by_date,
        aggregate_mean_net_ev=float(np.mean(regime_means)),
    )

    return SyntheticEvResult(
        null_arm=null_arm,
        regime_conditional_arm=regime_arm,
        mean_delta_net_ev=mean_delta_net_ev,
        p_value=p_value,
        alpha=_ALPHA,
        block_length=_BLOCK_LENGTH,
        n_bootstrap=_N_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=cfg.seed,
        n_dates=len(dates_sorted),
        verdict=verdict,
        verdict_reason=verdict_reason,
        stability=stability,
    )


__all__ = [
    "ArmResult",
    "StabilityAnalysis",
    "SyntheticEvResult",
    "SyntheticTurboConfig",
    "build_standardised_universe",
    "run_synthetic_net_ev_trial",
]
