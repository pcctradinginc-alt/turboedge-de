"""Harness for pre-registration 2026Q4-001 (``docs/preregistration_2026Q4_001.md``),
**as amended by §10 (Amendment B)**.

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

    H0: LCB_NetEV(regime_conditional) - LCB_NetEV(null) <= 0
    H1: LCB_NetEV(regime_conditional) - LCB_NetEV(null)  > 0

**One primary p-value** (one-sided, alpha=0.10, moving-block bootstrap over
prediction dates, block length 21). Everything else this module computes is
stability analysis (pre-registration §6): reported, never deflated as an
independent primary test, never promoted to primary after the fact -- kept
in a structurally separate :class:`StabilityAnalysis` field so a reader
cannot mistake one for the other.

Design, following the pre-registration exactly (§4, as amended by §10):

- :func:`build_standardised_universe` depends on ``spot``/``config`` only --
  never a forecast -- so the identical ``dict[str, ProductTerms]`` object is
  reused for both arms at a given prediction date. Any measured difference
  in NetEV therefore attributes to the forecast, per the pre-registration's
  own reasoning, and to nothing else.
- ``entry_ask`` is the theoretical fair value (``pricing/fair_value.py::
  theoretical_fair_value`` -- **not** ``pricing/intrinsic.py``, which has no
  such function; this prompt's file pointer was wrong and is corrected here,
  noted in the trial report rather than picked silently), *quote-straddled*
  by ``spread`` per Amendment B: ``entry_ask = fair_value * (1 + spread/2)``,
  ``entry_bid = fair_value * (1 - spread/2)`` -- not the original §4 formula
  (``entry_ask = fair_value`` outright, ``entry_bid = entry_ask * (1 -
  spread)``), which left ``entry_ask`` -- the only field
  ``simulation/payoff.py::simulate_product_payoff`` ever reads -- invariant
  to ``spread`` altogether (Amendment B, "Entry-side cost: §4 was too
  generous").
- Walk-forward reuses ``backtest/walkforward.py::walk_forward_evaluate``
  (never a second walk-forward loop), called once per horizon with
  ``embargo=horizon`` (pre-registration §4, matching
  ``docs/measured_results.md`` §6.12's own reproducibility record) --
  *not* once for the whole horizon ladder with one shared embargo, which is
  what a single ``walk_forward_evaluate(..., horizons=HORIZONS, embargo=X)``
  call would give. Per Amendment B, this walk-forward step is independent of
  ``spread`` (forecasting reads only ``bars``, never ``ProductTerms``), so
  :func:`_collect_forecasts_by_underlying` runs it **once** and its result is
  reused for the primary run and both §6 spread-sensitivity probes (see
  :func:`_spread_sensitivity_mean_delta_lcb_net_ev`).
- ``ranking/ev.py::evaluate_product_horizons`` is reused for the actual
  payoff/knockout/cost pricing. Its ``ProductHorizonEvaluation.lcb_net_return``
  is used as the primary statistic (Amendment B; see below), read once per
  arm, per date, per grid cell, and averaged first across the grid (per
  date) and then across dates.

**Amendment B, Change 1 -- why ``lcb_net_return`` and not ``mean_net_return``.**
The original §4 aggregation used ``ProductHorizonEvaluation.mean_net_return``
-- the *central*-scenario simulated mean net return, i.e. paths built with
drift = the model's own point forecast mean and nothing else
(``ranking/ev.py::_drift_for_scenario``, ``_SCENARIO_CENTRAL`` branch reads
only ``forecast.mean``). Separately, ``simulation/paths.py::simulate_paths``
has **no volatility parameter at all**; path dispersion comes entirely from
``bars``, which are identical across both arms of this trial. So
``mean_net_return`` cannot be moved by anything the two forecast models
disagree about *except* their point-mean drift -- it is structurally
incapable of reflecting a distributional (width/CRPS) improvement, which is
what the pre-registration's §1 hypothesis, as originally worded, claimed to
test.

``lcb_net_return`` reads both ``forecast.mean`` (via the central scenario)
**and** ``forecast.uncertainty`` (via the pessimistic scenario feeding
``ranking/lcb.py::lower_confidence_bound``), so it feels more of what the two
models actually disagree about than the mean alone -- and it is the quantity
``ranking/gates.py``'s ACTIONABLE gate actually uses
(``ranking/ev.py::to_candidate_gate_input``: ``lcb_ev=evaluation.
lcb_net_return``), so a PASS here answers a question this system's own
trading decision actually depends on.

**Scope limit, stated plainly, not overclaimed**: ``forecast.uncertainty`` is
the standard error of the forecast's own mean estimate -- **not** the
predictive width that ``docs/measured_results.md`` §6's CRPS scores measure.
This module's primary statistic therefore still cannot answer *"does the
improved predictive width carry economic value"*; only a future change that
extends ``simulation/paths.py::simulate_paths`` to accept a forecast
volatility could test that (Amendment B: "a separate, later trial and not a
patch to this one").

**Amendment B, Change 2 -- the quote now straddles fair value.** The original
§4 formula set ``entry_ask`` to the theoretical fair value outright, with
``spread`` only ever reducing ``entry_bid``. ``simulation/payoff.py::
simulate_product_payoff`` computes every net return from ``terms.entry_ask``
and never reads ``terms.entry_bid`` at all, so no value of ``spread`` could
change any simulated NetEV in the original harness -- making §6's
spread-sensitivity check a provable no-op. Amendment B replaces the formula
with a genuine two-sided quote around fair value:

    entry_ask = fair_value * (1 + spread / 2)
    entry_bid = fair_value * (1 - spread / 2)

``entry_ask`` -- the field every net return is computed from -- now
genuinely depends on ``spread``, so §6's sensitivity check is now a real
computation rather than a structural identity (see
:func:`_spread_sensitivity_mean_delta_lcb_net_ev`).

---

**2026Q4-002 extension** (``docs/preregistration_2026Q4_002.md``): the only
difference from 2026Q4-001 is §1 -- each arm's path dispersion is set to
*that arm's own forecast sigma* via ``simulation/paths.py::simulate_paths``'s
``target_sigma`` (commit ``b70c050``) instead of both arms inheriting
historical dispersion. :class:`SyntheticTurboConfig.sigma_source` selects the
mode: ``"historical"`` (default, unchanged -- reproduces 2026Q4-001 exactly,
bit-for-bit, by continuing to call ``evaluate_product_horizons`` exactly as
before), ``"forecast"`` (2026Q4-002 §1's primary mode: each arm gets its own
forecast's ``sigma`` as its path-dispersion target) and ``"null_forecast"``
(2026Q4-002 §2 point 3 / §6's falsification mode: **both** arms are forced to
the null model's own forecast sigma, so any surviving difference cannot be
attributed to width).

**Driving ``target_sigma`` without touching ``ranking/ev.py``.**
``ranking/ev.py::evaluate_product_horizons`` builds its ``PathSet``\\ s
internally (its private ``_build_paths``, which calls ``simulate_paths``) and
exposes no parameter for ``target_sigma`` at all -- and per this trial's own
constraint, ``ranking/ev.py`` is not modified to add one (that would be a
production path-engine change affecting every existing measurement, and
belongs in a separate, reviewed step, exactly as Amendment B of
2026Q4-001 said of ``target_sigma`` itself before it existed). Passing a
pre-built ``PathSet`` into ``evaluate_product_horizons`` is also not an
option: it has no such parameter either -- it always builds its own paths
from ``bars``/``forecast_by_horizon``.

The only remaining option the pre-registration itself names --
"construct the EV evaluation the way ``ranking/ev.py`` does but with the
sigma set" -- is what :func:`_evaluate_grid_with_target_sigma` does: it
mirrors ``evaluate_product_horizons``'s own payoff/shrinkage/LCB computation
(reusing its actual dependencies -- ``simulation/paths.py::simulate_paths``,
``simulation/payoff.py::simulate_product_payoff``,
``ranking/shrinkage.py::shrink_group_means``,
``ranking/lcb.py::lower_confidence_bound``, and ``ranking/ev.py::EvConfig``
itself for every default that is not sigma-specific -- ``z_pessimistic``,
``lcb.z``, ``shrinkage``, ``path_method``, ``block_size``, ``lookback_days``),
but builds each ``(direction, horizon, scenario)`` ``PathSet`` itself, one
level below ``evaluate_product_horizons``, with ``target_sigma`` threaded
through. Only the two trivial, direction-and-scenario-only formulas that
``evaluate_product_horizons`` computes inline (the pessimistic-scenario drift
sign, and the leverage approximation used purely for shrinkage-group
bucketing) are duplicated locally rather than imported across the module
boundary this harness must not cross -- see
:func:`_approx_leverage_for_grouping`'s docstring for why importing
``ranking/ev.py``'s own (private, underscore-prefixed) helpers instead was
rejected. This is used only for ``sigma_source in {"forecast",
"null_forecast"}``; ``"historical"`` keeps calling
``evaluate_product_horizons`` completely unchanged, so nothing about
2026Q4-001's measured behaviour can move.

**Unreachable targets (§3).** ``simulate_paths`` clamps and reports
(``target_sigma_met=False``) rather than raising when a target is below the
achievable floor (``_apply_volatility_scaling``'s own documented policy).
Clamped cells are priced and included exactly like any other cell -- never
dropped -- and counted per arm in :class:`StabilityAnalysis`
(``clamped_cell_count_null``/``clamped_cell_count_regime_conditional``),
alongside the primary statistic recomputed on reachable-only cells
(``reachable_only_mean_delta_lcb_net_ev``, §6: stability, never primary) and
the realised-vs-requested sigma per arm
(``mean_realized_sigma_by_arm``/``mean_target_sigma_by_arm``).

**The falsification arm (§2 point 3, §6) is a first-class, top-level result
field** (:class:`FalsificationArm`, ``SyntheticEvResult.falsification``), not
buried in stability analysis -- computed automatically whenever
``sigma_source="forecast"`` by rerunning the whole grid with
``sigma_source="null_forecast"`` (both arms forced to the null's own sigma)
against the identical, already-computed walk-forward forecasts. Whether it
"shows an effect" is a fixed, pre-committed numeric rule
(:func:`_falsification_shows_effect`): the *same* PASS bar as the primary
(``p <= alpha`` and ``mean_delta > 0``) applied to the falsification arm's own
delta series -- see that function's docstring for why this bar, not a
bespoke one, was chosen. The primary verdict is then one of **four** branches
(:func:`_verdict_with_falsification`, pre-registration §5) rather than
2026Q4-001's three: PASS's row is split into PASS (falsification shows no
effect) and CONFOUNDED (falsification shows an effect too).
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
from turboedge.ranking.lcb import lower_confidence_bound
from turboedge.ranking.shrinkage import leverage_bucket_for, shrink_group_means, shrinkage_group_key
from turboedge.simulation.paths import PathSet, simulate_paths
from turboedge.simulation.payoff import ProductTerms, simulate_product_payoff
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

#: Pre-registration §6: probe values for the spread-sensitivity stability
#: check ("sensitivity of the sign of the result to `spread` at 0.0025 and
#: 0.01").
_SPREAD_PROBES = (0.0025, 0.01)

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
    #: 2026Q4-002 §1: which dispersion each arm's simulated paths use.
    #: ``"historical"`` (default) reproduces 2026Q4-001 exactly -- both arms'
    #: dispersion comes entirely from ``bars`` via ``evaluate_product_horizons``,
    #: unchanged. ``"forecast"`` is the 2026Q4-002 primary mode: each arm's
    #: paths target *that arm's own* forecast sigma. ``"null_forecast"`` is
    #: the §2/§6 falsification mode: **both** arms are forced to the null
    #: model's own forecast sigma. See module docstring, "2026Q4-002
    #: extension".
    sigma_source: Literal["historical", "forecast", "null_forecast"] = "historical"


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
    """The fixed standardised turbo grid for one prediction date's spot (pre-registration §4,
    entry/exit quote per Amendment B §10).

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
    ``storage/schemas.py``'s own convention comment). The quote straddles
    theoretical fair value symmetrically around it (Amendment B §10,
    replacing §4's original "entry_ask = fair value outright" formula, which
    left ``entry_ask`` -- the only field ``simulation/payoff.py`` ever
    reads -- invariant to ``spread``):

        entry_ask = fair_value * (1 + spread / 2)
        entry_bid = fair_value * (1 - spread / 2)

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
            # Amendment B §10: a genuine two-sided quote straddling fair
            # value, replacing §4's "entry_ask = fair value" (which made
            # entry_ask, the only field simulate_product_payoff reads,
            # invariant to `spread`).
            entry_ask = fair_value * (1.0 + config.spread / 2.0)
            entry_bid = fair_value * (1.0 - config.spread / 2.0)
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
    """One model's ("arm's") results: the aggregate LCB NetEV plus the per-date series
    behind it (Amendment B §10 -- the primary statistic is ``lcb_net_return``, not
    ``mean_net_return``).

    ``dates``/``grid_mean_lcb_net_ev_by_date``/``n_cells_by_date`` are
    positionally aligned (same length, same order); ``n_cells_by_date``
    records how many ``(underlying, isin, horizon)`` grid cells contributed
    to that date's mean, which is diagnostic for how much the walk-forward's
    own horizon-tail truncation (longer horizons run out of test dates near
    the end of history first) thinned a given date's coverage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    dates: list[date]
    #: Per date, the mean across the standardised grid of each cell's
    #: ``ProductHorizonEvaluation.lcb_net_return`` for this arm.
    grid_mean_lcb_net_ev_by_date: list[float]
    n_cells_by_date: list[int]
    #: Mean over dates of ``grid_mean_lcb_net_ev_by_date``.
    aggregate_mean_lcb_net_ev: float


class StabilityAnalysis(BaseModel):
    """Pre-registration §6: secondary stability analysis, reported, never primary, never
    deflated against a primary alpha, never promotable to primary after the fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delta_net_ev_by_underlying: dict[str, float]
    delta_net_ev_by_horizon: dict[int, float]
    delta_net_ev_by_barrier_distance: dict[float, float]
    share_cells_delta_positive: float = Field(ge=0.0, le=1.0)
    n_cells: int = Field(gt=0)
    #: Mean delta LCB NetEV, genuinely re-simulated (Amendment B §10; see
    #: :func:`_spread_sensitivity_mean_delta_lcb_net_ev`) at ``spread`` in
    #: {0.0025, 0.01} -- keys ``"spread_0.0025"``/``"spread_0.01"``. Prior to
    #: Amendment B this reused the primary value unchanged, because
    #: ``spread`` could not affect ``entry_ask`` under the original §4
    #: formula; Amendment B's quote-straddle formula makes ``entry_ask``
    #: (and therefore every simulated net return) genuinely depend on
    #: ``spread``, so this is now a full grid re-run per probe value.
    spread_sensitivity_mean_delta: dict[str, float]
    #: Whether the sign of ``mean_delta_lcb_net_ev`` is unchanged at both
    #: probe spread values -- the reader-facing summary pre-registration §6
    #: asks for ("the reader must be able to see that without re-running
    #: anything").
    spread_sensitivity_sign_stable: bool
    #: 2026Q4-002 §3/§6: count of grid cells (out of ``n_cells``) whose
    #: requested ``target_sigma`` was unreachable and therefore clamped
    #: (``PathSet.target_sigma_met=False``), per arm. **These cells are
    #: still included** in every delta/primary computation above -- this is
    #: a count for the reader, not a record of anything dropped (§3:
    #: "unreachable cells are included at the clamped sigma and counted...
    #: not dropped"). Always 0 for ``sigma_source="historical"`` (no target
    #: is ever requested there, so nothing can be clamped).
    clamped_cell_count_null: int = 0
    clamped_cell_count_regime_conditional: int = 0
    #: 2026Q4-002 §6: the primary statistic (mean delta LCB NetEV, not a
    #: fresh p-value -- §6 asks for the *statistic* recomputed, not a second
    #: significance test) recomputed on only the cells where **neither** arm
    #: was clamped. ``None`` when every cell was clamped (no reachable cells
    #: to recompute from) or when ``sigma_source="historical"`` and no
    #: sigma-targeting information exists at all. Stability only, per §3:
    #: "never as the primary".
    reachable_only_mean_delta_lcb_net_ev: float | None = None
    #: 2026Q4-002 §6: "realised-vs-requested sigma per arm, so a reader can
    #: confirm the mechanism did what §1 claims" -- mean, across all cells
    #: that carry sigma-targeting diagnostics, of the *central-scenario*
    #: ``PathSet.realized_sigma``/``target_sigma`` for that arm. Keys
    #: ``"null"``/``"regime_conditional"``. Empty for
    #: ``sigma_source="historical"``.
    mean_realized_sigma_by_arm: dict[str, float] = Field(default_factory=dict)
    mean_target_sigma_by_arm: dict[str, float] = Field(default_factory=dict)


class FalsificationArm(BaseModel):
    """Pre-registration 2026Q4-002 §2 point 3 / §6: the comparison rerun with **both** arms
    forced to the null model's own forecast sigma (never each arm's own -- that is exactly the
    manipulation the primary "forecast" run applies, and this arm exists to remove it and see
    whether the effect survives).

    Kept as its own top-level field on :class:`SyntheticEvResult` -- **not** nested inside
    :class:`StabilityAnalysis` -- because the pre-registration frames it as the thing that lets
    a PASS be attacked (§2: "this arm exists specifically so a PASS can be attacked, and its
    outcome is reported whatever it says"), which means a reader must not have to go looking in
    secondary/stability analysis to find it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Same statistic as the primary (mean over dates of the whole-grid-mean
    #: LCB NetEV delta), computed with both arms forced to the null's own
    #: forecast sigma.
    mean_delta_lcb_net_ev: float
    #: Same one-sided moving-block-bootstrap procedure as the primary
    #: (identical alpha/block_length/n_resamples/seed), applied to this
    #: arm's own delta-by-date series.
    p_value: float = Field(ge=0.0, le=1.0)
    n_dates: int = Field(gt=0)
    #: See :func:`_falsification_shows_effect` for the fixed, pre-committed
    #: numeric rule this is computed from.
    shows_effect: bool
    shows_effect_reason: str


class SyntheticEvResult(BaseModel):
    """Pre-registration result: exactly one primary p-value (§4/§5, statistic per Amendment B
    §10) plus the stability analysis (§6), kept in a structurally separate nested field so a
    reader cannot mistake secondary evidence for the primary test.

    2026Q4-002 extension: ``verdict`` gains a fourth branch (``CONFOUNDED``) and
    ``falsification``/``sigma_source`` are new, both ``None``/``"historical"`` respectively for
    a 2026Q4-001-shaped (``sigma_source="historical"``) run -- see module docstring, "2026Q4-002
    extension".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    null_arm: ArmResult
    regime_conditional_arm: ArmResult
    #: Primary statistic (§4, as amended by §10): mean over dates of
    #: (regime_conditional - null) LCB NetEV -- **not** the mean-scenario
    #: NetEV (see module docstring, "Amendment B, Change 1").
    mean_delta_lcb_net_ev: float
    #: Primary, one-sided, moving-block-bootstrap p-value (§4).
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)
    block_length: int = Field(gt=0)
    n_bootstrap: int = Field(gt=0)
    bootstrap_seed: int
    n_dates: int = Field(gt=0)
    #: Pre-registration §5's decision table, verbatim:
    #: PASS = p<=alpha and mean_delta>0 and (2026Q4-002 only) falsification
    #: shows no effect; CONFOUNDED (2026Q4-002 only) = same, but falsification
    #: also shows an effect; FAIL_NULL_RESULT = p>alpha;
    #: FAIL_NEGATIVE_SIGNIFICANT = p<=alpha and mean_delta<=0 ("a more
    #: informative negative than a null result").
    verdict: Literal["PASS", "CONFOUNDED", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT"]
    verdict_reason: str
    stability: StabilityAnalysis
    #: 2026Q4-002 §2/§6: populated iff ``config.sigma_source == "forecast"`` --
    #: the only mode where "is the effect attributable to width" is a live
    #: question. ``None`` for ``"historical"`` (2026Q4-001 -- no sigma
    #: targeting exists to falsify) and ``"null_forecast"`` (this run *is*
    #: already the falsification-style comparison; nesting another
    #: falsification arm inside it is not meaningful).
    falsification: FalsificationArm | None = None
    #: Echoes ``config.sigma_source`` used to produce this result (§4
    #: provenance discipline: what was actually run should be readable off
    #: the result, not just inferred from the caller's own config object).
    sigma_source: Literal["historical", "forecast", "null_forecast"] = "historical"


@dataclass(frozen=True, slots=True)
class _CellRecord:
    """One ``(prediction date, underlying, isin, horizon)`` grid cell's LCB NetEV, both arms
    (Amendment B §10: ``ProductHorizonEvaluation.lcb_net_return``, not ``mean_net_return``).

    The ``*_realized_sigma``/``*_target_sigma``/``*_target_sigma_met`` fields (2026Q4-002 §1/§3/
    §6) default to ``None`` -- populated only for ``sigma_source in {"forecast",
    "null_forecast"}`` (see :func:`_price_trial_grid`) from the *central*-scenario
    ``PathSet``'s own diagnostics (the pessimistic scenario targets the same ``target_sigma``,
    per :func:`_evaluate_grid_with_target_sigma`, so the central scenario is representative
    without double-counting a cell for both scenarios). ``None`` throughout for
    ``sigma_source="historical"``, exactly like existing tests that construct a bare
    ``_CellRecord`` with only the original eight fields expect.
    """

    date_key: date
    underlying_id: str
    isin: str
    direction: Direction
    barrier_distance: float
    horizon_days: int
    lcb_net_ev_null: float
    lcb_net_ev_regime_conditional: float
    null_realized_sigma: float | None = None
    null_target_sigma: float | None = None
    null_target_sigma_met: bool | None = None
    regime_realized_sigma: float | None = None
    regime_target_sigma: float | None = None
    regime_target_sigma_met: bool | None = None

    @property
    def delta(self) -> float:
        return self.lcb_net_ev_regime_conditional - self.lcb_net_ev_null


@dataclass(frozen=True, slots=True)
class _UnderlyingForecasts:
    """One underlying's bars plus both arms' walk-forward OOS forecasts -- computed once
    (Amendment B §10: forecasting does not depend on ``spread``) and reused across the primary
    run and every §6 spread-sensitivity probe (see :func:`_price_trial_grid`)."""

    bars: list[UnderlyingBar]
    bars_by_ts: dict[datetime, UnderlyingBar]
    null_by_date: dict[date, tuple[datetime, dict[int, HorizonForecast]]]
    regime_by_date: dict[date, tuple[datetime, dict[int, HorizonForecast]]]
    common_dates: list[date]


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


def _collect_forecasts_by_underlying(
    bars_by_underlying: Mapping[str, Sequence[UnderlyingBar]],
) -> dict[str, _UnderlyingForecasts]:
    """Walk-forward both arms for every underlying, exactly once (Amendment B §10: the
    forecasts do not depend on ``spread``, so this result is reused for the primary run and
    both §6 spread-sensitivity probes rather than re-walked-forward per probe)."""
    result: dict[str, _UnderlyingForecasts] = {}
    for underlying_id, bars in bars_by_underlying.items():
        bars_list = list(bars)
        if not bars_list:
            raise ValueError(f"bars_by_underlying[{underlying_id!r}] must not be empty")
        bars_by_ts = {b.ts: b for b in bars_list}
        null_by_date = _collect_oos_forecasts_by_date(bars_list, NullModel)
        regime_by_date = _collect_oos_forecasts_by_date(bars_list, RegimeConditionalEmpiricalModel)
        common_dates = sorted(set(null_by_date) & set(regime_by_date))
        result[underlying_id] = _UnderlyingForecasts(
            bars=bars_list,
            bars_by_ts=bars_by_ts,
            null_by_date=null_by_date,
            regime_by_date=regime_by_date,
            common_dates=common_dates,
        )
    return result


#: Internal scenario keys for :func:`_evaluate_grid_with_target_sigma` --
#: mirrors ``ranking/ev.py``'s own ``_SCENARIO_CENTRAL``/``_SCENARIO_PESSIMISTIC``
#: constants (arbitrary internal dict keys; not part of any public contract).
_SIGMA_SCENARIO_CENTRAL = "central"
_SIGMA_SCENARIO_PESSIMISTIC = "pessimistic"

#: Mirrors ``ranking/ev.py::_MAX_PLAUSIBLE_UNDERLYING_MOVE`` exactly (see that
#: module's own 2026-09-24 incident comment for why this guard exists at
#: all). Duplicated, not imported, for the same reason as
#: :func:`_approx_leverage_for_grouping`.
_SIGMA_MAX_PLAUSIBLE_UNDERLYING_MOVE = 3.0


@dataclass(slots=True)
class _RawSigmaRecord:
    """Internal, mutable accumulator for one (isin, horizon) cell between the path-simulation
    pass and the shrinkage/LCB pass of :func:`_evaluate_grid_with_target_sigma` -- mirrors
    ``ranking/ev.py::_RawRecord``, trimmed to only what this trial's primary statistic and §6
    stability analysis read (``lcb_net_return`` and the sigma-targeting diagnostics)."""

    isin: str
    horizon_days: int
    group_key: str
    mean_net_return: float
    pessimistic_mean: float
    mc_standard_error: float
    model_uncertainty: float
    realized_sigma: float | None
    target_sigma: float | None
    target_sigma_met: bool | None
    shrunk_mean: float = 0.0


@dataclass(frozen=True, slots=True)
class _SigmaCellEval:
    """One (isin, horizon) cell's LCB NetEV plus the 2026Q4-002 §1/§3/§6 sigma-targeting
    diagnostics, as returned by :func:`_evaluate_grid_with_target_sigma`."""

    isin: str
    horizon_days: int
    lcb_net_return: float
    realized_sigma: float | None
    target_sigma: float | None
    target_sigma_met: bool | None


def _approx_leverage_for_grouping(terms: ProductTerms, spot0: float) -> float:
    """Standard turbo leverage approximation, used *only* to bucket cells for shrinkage
    grouping -- mirrors ``ranking/ev.py::_approx_leverage`` exactly.

    Duplicated rather than imported: ``ranking/ev.py`` is not modified by this trial (see
    module docstring), and importing its private, underscore-prefixed helper across the module
    boundary would create an undeclared dependency on an implementation detail of a module this
    harness is expressly forbidden from touching -- if ``ranking/ev.py`` ever renames or
    inlines ``_approx_leverage``, this trial's own tests (not a cross-module import failure
    somewhere else) are what should catch a divergence.
    """
    if not (spot0 > 0.0) or not (terms.entry_ask > 0.0) or not (terms.fx > 0.0):
        return 1.0
    return (spot0 * terms.ratio) / (terms.entry_ask * terms.fx)


def _evaluate_grid_with_target_sigma(
    terms_by_isin: Mapping[str, ProductTerms],
    forecast_by_horizon: Mapping[int, HorizonForecast],
    bars: Sequence[UnderlyingBar],
    *,
    underlying_id: str,
    spot0: float,
    start: datetime,
    as_of: date,
    rng: np.random.Generator,
    horizons: Sequence[int],
    n_paths: int,
    target_sigma_by_horizon: Mapping[int, float] | None,
) -> list[_SigmaCellEval]:
    """2026Q4-002 §1's sigma-targeting mode of the payoff/LCB pipeline -- see module docstring,
    "Driving target_sigma without touching ranking/ev.py", for why this exists as a parallel
    path rather than a parameter on ``evaluate_product_horizons``.

    Mirrors ``evaluate_product_horizons``'s own payoff/shrinkage/LCB computation using its
    actual dependencies (``simulate_paths``, ``simulate_product_payoff``, ``shrink_group_means``,
    ``lower_confidence_bound``) and its own default config (``EvConfig(n_paths=n_paths)`` --
    ``z_pessimistic``, ``lcb.z``, ``shrinkage``, ``path_method``, ``block_size``,
    ``lookback_days`` are all read from it, never hand-picked constants, so a future change to
    those defaults is felt here too), differing only in:

    - building each ``(direction, horizon, scenario)`` ``PathSet`` with
      ``target_sigma=target_sigma_by_horizon[h]`` (``None`` per horizon disables targeting for
      that horizon, matching ``simulate_paths``'s own no-op contract) instead of leaving
      dispersion to come from ``bars`` alone;
    - returning only what this trial needs (``lcb_net_return`` plus the sigma diagnostics) --
      utility/score/sizing/liquidity/gates are irrelevant to a standardised-universe trial that
      only ever reads ``lcb_net_return`` (see the existing ``_CellRecord``).

    Per-cell sigma diagnostics (``realized_sigma``/``target_sigma``/``target_sigma_met``) are
    taken from the **central**-scenario ``PathSet``: dispersion targeting runs before the drift
    tilt inside ``simulate_paths`` (``_apply_volatility_scaling`` precedes ``_apply_drift_tilt``),
    so the achievable dispersion does not depend on which scenario's drift is applied, and using
    one scenario avoids reporting a cell's clamp status twice.

    No look-ahead: identical to ``evaluate_product_horizons``'s own guarantee -- ``simulate_paths``
    excludes any bar at or after ``start`` from its bootstrap sample, and ``forecast_by_horizon``
    was already fit only on bars available before this date by the caller's walk-forward step.

    Raises:
        ValueError: if ``horizons`` is empty or ``spot0 <= 0`` (mirrors
            ``evaluate_product_horizons``'s own input validation).
    """
    if not horizons:
        raise ValueError("horizons must not be empty")
    if not (spot0 > 0.0):
        raise ValueError(f"spot0 must be > 0, got {spot0!r}")
    if not terms_by_isin:
        return []

    ev_cfg = EvConfig(n_paths=n_paths)
    directions_present = {t.direction for t in terms_by_isin.values()}

    paths: dict[Direction, dict[int, dict[str, PathSet]]] = {}
    for direction in sorted(directions_present, key=lambda d: d.value):
        paths[direction] = {}
        # Adverse-for-this-direction sign, mirroring ranking/ev.py::_drift_for_scenario
        # exactly: LONG's pessimistic scenario is a lower drift, SHORT's is a higher one.
        adverse_is_lower = direction == Direction.LONG
        pess_sign = -1.0 if adverse_is_lower else 1.0
        for h in sorted(horizons):
            forecast = forecast_by_horizon[h]
            target_sigma = None if target_sigma_by_horizon is None else target_sigma_by_horizon[h]
            central_drift = forecast.mean
            pessimistic_drift = (
                forecast.mean + pess_sign * ev_cfg.z_pessimistic * forecast.uncertainty
            )
            paths[direction][h] = {
                _SIGMA_SCENARIO_CENTRAL: simulate_paths(
                    bars,
                    spot0=spot0,
                    start=start,
                    horizon_days=h,
                    n_paths=ev_cfg.n_paths,
                    rng=rng,
                    drift_log_return=central_drift,
                    target_sigma=target_sigma,
                    method=ev_cfg.path_method,  # type: ignore[arg-type]
                    block_size=ev_cfg.block_size,
                    lookback_days=ev_cfg.lookback_days,
                ),
                _SIGMA_SCENARIO_PESSIMISTIC: simulate_paths(
                    bars,
                    spot0=spot0,
                    start=start,
                    horizon_days=h,
                    n_paths=ev_cfg.n_paths,
                    rng=rng,
                    drift_log_return=pessimistic_drift,
                    target_sigma=target_sigma,
                    method=ev_cfg.path_method,  # type: ignore[arg-type]
                    block_size=ev_cfg.block_size,
                    lookback_days=ev_cfg.lookback_days,
                ),
            }

    raw: list[_RawSigmaRecord] = []
    for isin, terms in terms_by_isin.items():
        for h in horizons:
            scenario_paths = paths[terms.direction][h]
            central_paths = scenario_paths[_SIGMA_SCENARIO_CENTRAL]
            central_dist = simulate_product_payoff(terms, central_paths, [h], as_of=as_of)[h]
            pessimistic_dist = simulate_product_payoff(
                terms, scenario_paths[_SIGMA_SCENARIO_PESSIMISTIC], [h], as_of=as_of
            )[h]

            leverage = _approx_leverage_for_grouping(terms, spot0)
            bucket = leverage_bucket_for(leverage)
            group_key = f"{shrinkage_group_key(underlying_id, terms.direction, bucket)}|h{h}"

            # Physical-impossibility guard, mirroring ranking/ev.py's own (see that module's
            # 2026-09-24 incident comment) -- never suppresses a plausible candidate, only
            # arithmetically impossible output.
            plausible_bound = _SIGMA_MAX_PLAUSIBLE_UNDERLYING_MOVE * leverage
            if abs(central_dist.mean) > plausible_bound or abs(pessimistic_dist.mean) > (
                plausible_bound
            ):
                continue

            raw.append(
                _RawSigmaRecord(
                    isin=isin,
                    horizon_days=h,
                    group_key=group_key,
                    mean_net_return=central_dist.mean,
                    pessimistic_mean=pessimistic_dist.mean,
                    mc_standard_error=central_dist.mc_standard_error,
                    model_uncertainty=forecast_by_horizon[h].uncertainty,
                    realized_sigma=central_paths.realized_sigma,
                    target_sigma=central_paths.target_sigma,
                    target_sigma_met=central_paths.target_sigma_met,
                )
            )

    groups: dict[str, list[_RawSigmaRecord]] = {}
    for rec in raw:
        groups.setdefault(rec.group_key, []).append(rec)
    for members in groups.values():
        shrunk, _intensity = shrink_group_means(
            [m.mean_net_return for m in members],
            [m.mc_standard_error for m in members],
            [m.model_uncertainty for m in members],
            cfg=ev_cfg.shrinkage,
        )
        for member, shrunk_mean in zip(members, shrunk, strict=True):
            member.shrunk_mean = shrunk_mean

    evaluations: list[_SigmaCellEval] = []
    for rec in raw:
        lcb_net_return = lower_confidence_bound(
            rec.mean_net_return,
            rec.pessimistic_mean,
            rec.mc_standard_error,
            rec.shrunk_mean,
            z=ev_cfg.lcb.z,
        )
        evaluations.append(
            _SigmaCellEval(
                isin=rec.isin,
                horizon_days=rec.horizon_days,
                lcb_net_return=lcb_net_return,
                realized_sigma=rec.realized_sigma,
                target_sigma=rec.target_sigma,
                target_sigma_met=rec.target_sigma_met,
            )
        )
    return evaluations


def _target_sigma_by_horizon(
    forecast_map: Mapping[int, HorizonForecast], horizons: Sequence[int]
) -> dict[int, float]:
    """2026Q4-002 §1: one arm's own forecast sigma, per horizon -- the target
    :func:`_evaluate_grid_with_target_sigma` is asked to hit for that arm."""
    return {h: forecast_map[h].sigma for h in horizons}


def _price_trial_grid(
    underlying_forecasts: Mapping[str, _UnderlyingForecasts],
    cfg: SyntheticTurboConfig,
) -> list[_CellRecord]:
    """Price the identical standardised grid (§4, quote per Amendment B §10) at every common
    prediction date, for every underlying, under both arms' already-computed forecasts.

    Takes pre-computed forecasts (:func:`_collect_forecasts_by_underlying`)
    rather than bars, so it can be called once for the primary ``cfg.spread``
    and again, cheaply (no re-walk-forward), for each §6 spread-sensitivity
    probe (:func:`_spread_sensitivity_mean_delta_lcb_net_ev`) -- only
    ``build_standardised_universe`` (cheap) and
    ``evaluate_product_horizons`` (the actual path-simulation cost) are
    repeated per call.

    No look-ahead: ``evaluate_product_horizons``'s own ``bars``/``start``
    filtering (``simulation/paths.py::simulate_paths``) excludes any bar at
    or after ``start`` from the path-simulation bootstrap sample, and each
    model was fit only on bars up to that fold's training bar's
    ``available_at`` inside ``walk_forward_evaluate`` -- this function adds
    no additional data access of its own.

    2026Q4-002 §1: for ``cfg.sigma_source in {"forecast", "null_forecast"}``,
    grid pricing runs through :func:`_evaluate_grid_with_target_sigma` instead
    of ``evaluate_product_horizons`` (module docstring, "Driving target_sigma
    without touching ranking/ev.py"). ``"historical"`` (default) is
    completely unchanged from 2026Q4-001 -- still ``evaluate_product_horizons``,
    still no ``target_sigma`` requested at all -- so this branch cannot move
    any existing measurement.
    """
    records: list[_CellRecord] = []
    for underlying_id, uf in underlying_forecasts.items():
        for d in uf.common_dates:
            ts, null_forecast_map = uf.null_by_date[d]
            _, regime_forecast_map = uf.regime_by_date[d]
            common_horizons = sorted(set(null_forecast_map) & set(regime_forecast_map))
            if not common_horizons:
                continue
            bar = uf.bars_by_ts.get(ts)
            if bar is None:  # pragma: no cover -- internal invariant
                raise AssertionError(
                    f"prediction ts {ts!r} not found in bars_by_underlying[{underlying_id!r}]"
                )

            # Same dict object handed to both arms (§4 identity requirement) --
            # built once per date, not once per arm.
            universe = build_standardised_universe(bar.close, config=cfg)
            cluster_id = f"synthetic::{underlying_id}"

            if cfg.sigma_source == "historical":
                ev_cfg = EvConfig(n_paths=cfg.n_paths)

                null_evals = evaluate_product_horizons(
                    universe,
                    {h: null_forecast_map[h] for h in common_horizons},
                    uf.bars,
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
                    uf.bars,
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
                            lcb_net_ev_null=null_by_key[(isin, horizon_days)].lcb_net_return,
                            lcb_net_ev_regime_conditional=regime_by_key[
                                (isin, horizon_days)
                            ].lcb_net_return,
                        )
                    )
                continue

            # cfg.sigma_source in {"forecast", "null_forecast"} (2026Q4-002 §1/§2).
            null_target = _target_sigma_by_horizon(null_forecast_map, common_horizons)
            if cfg.sigma_source == "forecast":
                # §1: each arm targets its own forecast's sigma.
                regime_target = _target_sigma_by_horizon(regime_forecast_map, common_horizons)
            else:
                # "null_forecast" -- §2 point 3/§6 falsification: BOTH arms
                # forced to the null's own sigma (regime is not allowed its
                # own, narrower, target here).
                regime_target = dict(null_target)

            null_sigma_evals = _evaluate_grid_with_target_sigma(
                universe,
                {h: null_forecast_map[h] for h in common_horizons},
                uf.bars,
                underlying_id=underlying_id,
                spot0=bar.close,
                start=bar.available_at,
                as_of=d,
                rng=np.random.default_rng(cfg.seed),  # same seed, both arms -- §4
                horizons=common_horizons,
                n_paths=cfg.n_paths,
                target_sigma_by_horizon=null_target,
            )
            regime_sigma_evals = _evaluate_grid_with_target_sigma(
                universe,
                {h: regime_forecast_map[h] for h in common_horizons},
                uf.bars,
                underlying_id=underlying_id,
                spot0=bar.close,
                start=bar.available_at,
                as_of=d,
                rng=np.random.default_rng(cfg.seed),
                horizons=common_horizons,
                n_paths=cfg.n_paths,
                target_sigma_by_horizon=regime_target,
            )

            null_sigma_by_key = {(e.isin, e.horizon_days): e for e in null_sigma_evals}
            regime_sigma_by_key = {(e.isin, e.horizon_days): e for e in regime_sigma_evals}
            common_sigma_keys = sorted(set(null_sigma_by_key) & set(regime_sigma_by_key))
            for isin, horizon_days in common_sigma_keys:
                direction, distance = _parse_synthetic_isin(isin)
                null_eval = null_sigma_by_key[(isin, horizon_days)]
                regime_eval = regime_sigma_by_key[(isin, horizon_days)]
                records.append(
                    _CellRecord(
                        date_key=d,
                        underlying_id=underlying_id,
                        isin=isin,
                        direction=direction,
                        barrier_distance=distance,
                        horizon_days=horizon_days,
                        lcb_net_ev_null=null_eval.lcb_net_return,
                        lcb_net_ev_regime_conditional=regime_eval.lcb_net_return,
                        null_realized_sigma=null_eval.realized_sigma,
                        null_target_sigma=null_eval.target_sigma,
                        null_target_sigma_met=null_eval.target_sigma_met,
                        regime_realized_sigma=regime_eval.realized_sigma,
                        regime_target_sigma=regime_eval.target_sigma,
                        regime_target_sigma_met=regime_eval.target_sigma_met,
                    )
                )
    return records


def _grid_means_by_date(
    records: Sequence[_CellRecord],
) -> tuple[list[date], list[float], list[float], list[int]]:
    """Group cell records by prediction date and average the LCB NetEV within each date, per
    arm -- shared by the primary run and every spread-sensitivity probe so both compute the
    per-date grid mean identically."""
    by_date: dict[date, list[_CellRecord]] = {}
    for rec in records:
        by_date.setdefault(rec.date_key, []).append(rec)
    dates_sorted = sorted(by_date)
    null_means = [float(np.mean([r.lcb_net_ev_null for r in by_date[d]])) for d in dates_sorted]
    regime_means = [
        float(np.mean([r.lcb_net_ev_regime_conditional for r in by_date[d]])) for d in dates_sorted
    ]
    n_cells_by_date = [len(by_date[d]) for d in dates_sorted]
    return dates_sorted, null_means, regime_means, n_cells_by_date


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
    p_value: float, mean_delta_lcb_net_ev: float, alpha: float
) -> tuple[Literal["PASS", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT"], str]:
    """Pre-registration §5's decision table, verbatim, stated before any result existed
    (statistic per Amendment B §10: LCB NetEV, not mean-scenario NetEV)."""
    if p_value <= alpha and mean_delta_lcb_net_ev > 0.0:
        return (
            "PASS",
            f"p={p_value:.4f} <= alpha={alpha} and mean delta LCB NetEV="
            f"{mean_delta_lcb_net_ev:.6f} > 0: the distribution improvement carries economic "
            "information on standardised terms (pre-registration §5 row 1). Necessary "
            "condition met; forward real-product arm becomes the next trial. Still no "
            "promotion.",
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
        f"p={p_value:.4f} <= alpha={alpha} but mean delta LCB NetEV="
        f"{mean_delta_lcb_net_ev:.6f} <= 0: reported as evidence the better distribution is "
        "actively worse economically (pre-registration §5 row 3) -- a more informative "
        "negative than a null result.",
    )


def _falsification_shows_effect(
    p_value: float, mean_delta_lcb_net_ev: float, alpha: float
) -> tuple[bool, str]:
    """2026Q4-002 §2 point 3 / §5: the fixed, pre-committed numeric definition of "the
    falsification arm shows an effect" -- written before any result of this trial exists (this
    module may not be run against real data before 2026-10-01; see the module-level warning),
    exactly the discipline the pre-registration itself demands of every other threshold.

    **Rule: the falsification arm is judged by exactly the same bar as the primary hypothesis.**
    It "shows an effect" iff its own mean delta LCB NetEV is positive **and** its own one-sided
    moving-block-bootstrap p-value (identical alpha, block length, resample count and
    methodology to the primary -- see :func:`_moving_block_bootstrap_p_value`) is ``<= alpha``.
    In other words: the falsification arm shows an effect iff it would itself PASS as a
    standalone trial.

    Reasoning, fixed here rather than chosen after seeing a number:

    1. **No new threshold.** There is exactly one pre-registered test procedure in this
       document (§4: one-sided, alpha=0.10, moving-block bootstrap, block length 21).
       Reusing it for the falsification arm avoids inventing a second, independently
       choosable significance standard whose looseness or strictness could itself be
       second-guessed once a result exists -- precisely the kind of post-hoc degree of
       freedom §5 forbids for every other threshold in this trial.
    2. **Symmetric by construction, and therefore legible.** "The falsification arm shows an
       effect" becomes "the same test that said PASS for the primary also says PASS here, with
       width neutralised" -- a sentence a reader can verify without decoding a bespoke
       threshold, which is the whole point of a check whose "outcome is reported whatever it
       says" (§2).
    3. **Matches §2's own language.** §2 asks whether "the effect survives" once both arms are
       forced to the same sigma. A p-value that no longer clears alpha, or a delta that is no
       longer positive, is not "the same effect surviving" under any reading of that sentence --
       both are an absence of effect by the primary's own definition of what an effect is.
    """
    shows_effect = p_value <= alpha and mean_delta_lcb_net_ev > 0.0
    verdict_word = "shows an effect" if shows_effect else "shows no effect"
    consequence = (
        "the primary result is not attributable to width alone (§2/§5: CONFOUNDED)"
        if shows_effect
        else "consistent with the primary result being attributable to width (§2/§5: eligible "
        "for PASS)"
    )
    reason = (
        "falsification arm (both arms forced to the null model's own forecast sigma): "
        f"p={p_value:.4f}, mean delta LCB NetEV={mean_delta_lcb_net_ev:.6f} against alpha="
        f"{alpha} -- {verdict_word} (same bar as the primary PASS criterion): {consequence}."
    )
    return shows_effect, reason


def _verdict_with_falsification(
    p_value: float,
    mean_delta_lcb_net_ev: float,
    alpha: float,
    falsification_shows_effect: bool,
) -> tuple[Literal["PASS", "CONFOUNDED", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT"], str]:
    """Pre-registration 2026Q4-002 §5's four-branch decision table, verbatim, stated before any
    result existed. Used only when ``cfg.sigma_source == "forecast"``
    (:func:`run_synthetic_net_ev_trial`); every other mode keeps using :func:`_verdict`'s
    three-branch table unchanged, since neither "historical" (no sigma-targeting exists to
    falsify) nor a standalone "null_forecast" run (it already *is* the falsification-style
    comparison) has a meaningful nested falsification check.

    Extends :func:`_verdict` by splitting its PASS row in two according to whether the §6
    falsification arm (:func:`_falsification_shows_effect`) also shows an effect; the other two
    rows (p > alpha; mean delta <= 0 at p <= alpha) are unchanged.
    """
    if p_value <= alpha and mean_delta_lcb_net_ev > 0.0:
        if falsification_shows_effect:
            return (
                "CONFOUNDED",
                f"p={p_value:.4f} <= alpha={alpha} and mean delta LCB NetEV="
                f"{mean_delta_lcb_net_ev:.6f} > 0, but the §6 falsification arm (both arms "
                "forced to the null's own forecast sigma) also shows an effect (pre-"
                "registration §5 row 2): the effect is not attributable to width. Reported as "
                "such, promoted to nothing.",
            )
        return (
            "PASS",
            f"p={p_value:.4f} <= alpha={alpha}, mean delta LCB NetEV="
            f"{mean_delta_lcb_net_ev:.6f} > 0, and the §6 falsification arm shows no effect "
            "(pre-registration §5 row 1): the forecast's width carries economic value on "
            "standardised terms. Necessary condition met; forward real-product arm becomes "
            "the next trial. Still no promotion.",
        )
    if p_value > alpha:
        return (
            "FAIL_NULL_RESULT",
            f"p={p_value:.4f} > alpha={alpha} (pre-registration §5 row 3): the distributional "
            "improvement is real, measurable and economically inert through both channels.",
        )
    return (
        "FAIL_NEGATIVE_SIGNIFICANT",
        f"p={p_value:.4f} <= alpha={alpha} but mean delta LCB NetEV="
        f"{mean_delta_lcb_net_ev:.6f} <= 0 (pre-registration §5 row 4): reported as evidence "
        "the sharper forecast is economically harmful.",
    )


def _spread_sensitivity_mean_delta_lcb_net_ev(
    underlying_forecasts: Mapping[str, _UnderlyingForecasts], cfg: SyntheticTurboConfig
) -> dict[str, float]:
    """Pre-registration §6: "sensitivity of the sign of the result to `spread` at 0.0025 and
    0.01" -- so a result whose sign flips under a halved spread assumption is visible without
    re-running anything (§6, citing §6.10's leverage-amplification finding).

    **Genuinely re-runs the trial grid at each probe spread** (Amendment B
    §10: the quote-straddle formula makes ``entry_ask`` -- the only field
    ``simulation/payoff.py::simulate_product_payoff`` reads -- depend on
    ``spread``, so the pre-Amendment-B shortcut of reusing the primary mean
    delta unchanged is no longer valid; it would silently misreport this
    check as a no-op it no longer is).

    This costs two extra grid-pricing passes (the same
    ``evaluate_product_horizons`` cost as the primary run, once per probe
    spread) -- the honest price of a check that can now actually fail. It
    does **not** cost two extra walk-forward passes: ``underlying_forecasts``
    is computed once by the caller (:func:`_collect_forecasts_by_underlying`)
    and reused here unchanged, because forecasting reads only ``bars``, never
    ``ProductTerms``/``spread``.

    Raises:
        ValueError: if a probe spread yields no evaluable cell at all
            (propagated as a clear error rather than a silently empty/zero
            sensitivity entry).
    """
    sensitivity: dict[str, float] = {}
    for probe_spread in _SPREAD_PROBES:
        probe_cfg = replace(cfg, spread=probe_spread)
        probe_records = _price_trial_grid(underlying_forecasts, probe_cfg)
        if not probe_records:
            raise ValueError(
                f"spread-sensitivity probe at spread={probe_spread} evaluated no cells -- "
                "cannot compute a stability figure from zero data"
            )
        _, null_means, regime_means, _ = _grid_means_by_date(probe_records)
        delta = np.asarray(regime_means, dtype=np.float64) - np.asarray(
            null_means, dtype=np.float64
        )
        sensitivity[f"spread_{probe_spread}"] = float(np.mean(delta))
    return sensitivity


def _stability_analysis(
    records: Sequence[_CellRecord],
    primary_mean_delta_lcb_net_ev: float,
    spread_sensitivity_mean_delta: Mapping[str, float],
) -> StabilityAnalysis:
    """Pre-registration §6: per-underlying/per-horizon/per-barrier-distance delta LCB NetEV,
    the share of cells with a positive delta, and the spread sign-sensitivity -- all secondary.

    Takes the already-computed ``spread_sensitivity_mean_delta`` (see
    :func:`_spread_sensitivity_mean_delta_lcb_net_ev`) rather than computing
    it itself, so this function stays a pure grouping/summary step over
    ``records`` and is independently testable against fabricated records
    without re-running the (expensive, Amendment-B-genuine) spread probes.

    2026Q4-002 §3/§6: also derives the clamped-cell counts, the reachable-only primary
    recompute and the realised-vs-requested sigma means, all straight from ``records``' own
    (optional, ``None``-defaulted) sigma-diagnostic fields -- no new required parameter, so
    every existing caller (including bare, hand-constructed ``_CellRecord``s from
    ``sigma_source="historical"`` runs, whose sigma fields are all ``None``) keeps working
    unchanged and simply gets 0 clamped cells and empty sigma dicts back.
    """
    if not records:
        raise ValueError("records must not be empty")

    by_underlying: dict[str, list[float]] = {}
    by_horizon: dict[int, list[float]] = {}
    by_distance: dict[float, list[float]] = {}
    n_positive = 0
    clamped_null = 0
    clamped_regime = 0
    for rec in records:
        by_underlying.setdefault(rec.underlying_id, []).append(rec.delta)
        by_horizon.setdefault(rec.horizon_days, []).append(rec.delta)
        by_distance.setdefault(rec.barrier_distance, []).append(rec.delta)
        if rec.delta > 0.0:
            n_positive += 1
        if rec.null_target_sigma_met is False:  # `is`, not `==`: None is not a clamp, only False is
            clamped_null += 1
        if rec.regime_target_sigma_met is False:
            clamped_regime += 1

    # §6: primary statistic recomputed on reachable-only cells -- a cell counts as "reachable"
    # unless *some* arm's target was known to be unreachable (`is False`; `None` means no target
    # was ever requested for that cell, e.g. sigma_source="historical", which is trivially
    # "reachable" since nothing was clamped).
    reachable = [
        r
        for r in records
        if r.null_target_sigma_met is not False and r.regime_target_sigma_met is not False
    ]
    reachable_only_mean_delta: float | None
    if reachable:
        _, reachable_null_means, reachable_regime_means, _ = _grid_means_by_date(reachable)
        reachable_only_mean_delta = float(
            np.mean(
                np.asarray(reachable_regime_means, dtype=np.float64)
                - np.asarray(reachable_null_means, dtype=np.float64)
            )
        )
    else:
        reachable_only_mean_delta = None

    def _mean_of_present(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return float(np.mean(present)) if present else None

    mean_realized_null = _mean_of_present([r.null_realized_sigma for r in records])
    mean_realized_regime = _mean_of_present([r.regime_realized_sigma for r in records])
    mean_target_null = _mean_of_present([r.null_target_sigma for r in records])
    mean_target_regime = _mean_of_present([r.regime_target_sigma for r in records])

    mean_realized_sigma_by_arm: dict[str, float] = {}
    if mean_realized_null is not None:
        mean_realized_sigma_by_arm["null"] = mean_realized_null
    if mean_realized_regime is not None:
        mean_realized_sigma_by_arm["regime_conditional"] = mean_realized_regime

    mean_target_sigma_by_arm: dict[str, float] = {}
    if mean_target_null is not None:
        mean_target_sigma_by_arm["null"] = mean_target_null
    if mean_target_regime is not None:
        mean_target_sigma_by_arm["regime_conditional"] = mean_target_regime

    return StabilityAnalysis(
        delta_net_ev_by_underlying={k: float(np.mean(v)) for k, v in by_underlying.items()},
        delta_net_ev_by_horizon={k: float(np.mean(v)) for k, v in by_horizon.items()},
        delta_net_ev_by_barrier_distance={k: float(np.mean(v)) for k, v in by_distance.items()},
        share_cells_delta_positive=n_positive / len(records),
        n_cells=len(records),
        spread_sensitivity_mean_delta=dict(spread_sensitivity_mean_delta),
        spread_sensitivity_sign_stable=all(
            (v > 0.0) == (primary_mean_delta_lcb_net_ev > 0.0)
            for v in spread_sensitivity_mean_delta.values()
        ),
        clamped_cell_count_null=clamped_null,
        clamped_cell_count_regime_conditional=clamped_regime,
        reachable_only_mean_delta_lcb_net_ev=reachable_only_mean_delta,
        mean_realized_sigma_by_arm=mean_realized_sigma_by_arm,
        mean_target_sigma_by_arm=mean_target_sigma_by_arm,
    )


def run_synthetic_net_ev_trial(
    bars_by_underlying: Mapping[str, Sequence[UnderlyingBar]],
    *,
    config: SyntheticTurboConfig | None = None,
) -> SyntheticEvResult:
    """Run the pre-registration 2026Q4-001 historical-synthetic trial (§4/§5/§6, as amended
    by §10).

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
    ``min_train=750``, ``step=21``, ``embargo=horizon``) **exactly once**
    (:func:`_collect_forecasts_by_underlying`); at every prediction date
    common to both arms and to at least one horizon, the identical
    standardised grid (:func:`build_standardised_universe`, quote per
    Amendment B §10) is priced under each arm's forecast via
    ``evaluate_product_horizons`` (common random numbers: a fresh
    ``Generator(seed)`` for each arm's call). The primary statistic is the
    mean over dates of the whole-grid-mean **LCB** NetEV difference
    (regime_conditional - null; Amendment B §10, not the mean-scenario
    NetEV); its one-sided moving-block-bootstrap p-value and the
    pre-registration §5 verdict are returned alongside the §6 stability
    analysis, whose spread-sensitivity entries are genuine re-runs of the
    grid at each probe spread reusing the same walk-forward forecasts.

    2026Q4-002 §1/§2/§6 (``cfg.sigma_source``): ``"historical"`` (default) is unchanged from
    2026Q4-001 in every particular above. ``"forecast"`` prices the grid with each arm's own
    forecast sigma as its path-dispersion target (:func:`_price_trial_grid`) and additionally
    reruns the whole grid with ``sigma_source="null_forecast"`` (both arms forced to the null's
    own sigma) against the same already-computed walk-forward forecasts, to build the §6
    falsification arm (:class:`FalsificationArm`) and feed the §5 four-branch verdict
    (:func:`_verdict_with_falsification`). A standalone ``"null_forecast"`` run prices the
    falsification comparison directly and is judged by the plain three-branch table
    (:func:`_verdict`), with no nested falsification arm of its own.

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

    underlying_forecasts = _collect_forecasts_by_underlying(bars_by_underlying)

    cell_records = _price_trial_grid(underlying_forecasts, cfg)
    if not cell_records:
        raise ValueError(
            "no (date, underlying, horizon) cells were evaluated -- check that "
            f"bars_by_underlying has enough history for min_train={_MIN_TRAIN}, "
            f"step={_STEP} plus the longest horizon ({max(HORIZONS)}d)"
        )

    dates_sorted, null_means, regime_means, n_cells_by_date = _grid_means_by_date(cell_records)
    delta_by_date = np.asarray(regime_means, dtype=np.float64) - np.asarray(
        null_means, dtype=np.float64
    )
    mean_delta_lcb_net_ev = float(np.mean(delta_by_date))

    bootstrap_rng = np.random.default_rng(cfg.seed)
    p_value = _moving_block_bootstrap_p_value(
        delta_by_date,
        block_length=_BLOCK_LENGTH,
        n_resamples=_N_BOOTSTRAP_RESAMPLES,
        rng=bootstrap_rng,
    )

    falsification: FalsificationArm | None = None
    if cfg.sigma_source == "forecast":
        # §2 point 3 / §6: rerun the identical grid with BOTH arms forced to the null's own
        # sigma, reusing the same walk-forward forecasts (no re-walk-forward needed -- forecasts
        # don't depend on sigma_source any more than they depend on spread).
        falsification_cfg = replace(cfg, sigma_source="null_forecast")
        falsification_records = _price_trial_grid(underlying_forecasts, falsification_cfg)
        if not falsification_records:
            raise ValueError(
                "the §2/§6 falsification arm (sigma_source='null_forecast') evaluated no "
                "cells -- cannot compute the CONFOUNDED check from zero data"
            )
        f_dates_sorted, f_null_means, f_regime_means, _ = _grid_means_by_date(falsification_records)
        f_delta_by_date = np.asarray(f_regime_means, dtype=np.float64) - np.asarray(
            f_null_means, dtype=np.float64
        )
        f_mean_delta = float(np.mean(f_delta_by_date))
        f_bootstrap_rng = np.random.default_rng(cfg.seed)
        f_p_value = _moving_block_bootstrap_p_value(
            f_delta_by_date,
            block_length=_BLOCK_LENGTH,
            n_resamples=_N_BOOTSTRAP_RESAMPLES,
            rng=f_bootstrap_rng,
        )
        shows_effect, shows_effect_reason = _falsification_shows_effect(
            f_p_value, f_mean_delta, _ALPHA
        )
        falsification = FalsificationArm(
            mean_delta_lcb_net_ev=f_mean_delta,
            p_value=f_p_value,
            n_dates=len(f_dates_sorted),
            shows_effect=shows_effect,
            shows_effect_reason=shows_effect_reason,
        )
        verdict, verdict_reason = _verdict_with_falsification(
            p_value, mean_delta_lcb_net_ev, _ALPHA, shows_effect
        )
    else:
        verdict, verdict_reason = _verdict(p_value, mean_delta_lcb_net_ev, _ALPHA)

    spread_sensitivity = _spread_sensitivity_mean_delta_lcb_net_ev(underlying_forecasts, cfg)
    stability = _stability_analysis(cell_records, mean_delta_lcb_net_ev, spread_sensitivity)

    null_arm = ArmResult(
        model_id=NullModel().model_id,
        dates=dates_sorted,
        grid_mean_lcb_net_ev_by_date=null_means,
        n_cells_by_date=n_cells_by_date,
        aggregate_mean_lcb_net_ev=float(np.mean(null_means)),
    )
    regime_arm = ArmResult(
        model_id=RegimeConditionalEmpiricalModel().model_id,
        dates=dates_sorted,
        grid_mean_lcb_net_ev_by_date=regime_means,
        n_cells_by_date=n_cells_by_date,
        aggregate_mean_lcb_net_ev=float(np.mean(regime_means)),
    )

    return SyntheticEvResult(
        null_arm=null_arm,
        regime_conditional_arm=regime_arm,
        mean_delta_lcb_net_ev=mean_delta_lcb_net_ev,
        p_value=p_value,
        alpha=_ALPHA,
        block_length=_BLOCK_LENGTH,
        n_bootstrap=_N_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=cfg.seed,
        n_dates=len(dates_sorted),
        verdict=verdict,
        verdict_reason=verdict_reason,
        stability=stability,
        falsification=falsification,
        sigma_source=cfg.sigma_source,
    )


__all__ = [
    "ArmResult",
    "FalsificationArm",
    "StabilityAnalysis",
    "SyntheticEvResult",
    "SyntheticTurboConfig",
    "build_standardised_universe",
    "run_synthetic_net_ev_trial",
]
