"""Full turbo/knockout product payoff simulation over simulated paths.

Formula reference: Master Spec §13 ("Produktbewertung"), §16
("Produkt-Simulation"), and the Build Contract's W5 interface.

Conventions (CLAUDE.md rules 11/12, documented per the Build Contract's
request to pin down the spread convention precisely):

- **Entry is always the ask** (``terms.entry_ask``, given directly -- it is
  an observed market quote, not something this module derives).
- **Exit is always the bid**, derived here from a *theoretical exit value*
  ``V`` (fair value with the issuer's persistent margin baked back in, see
  below). ``exit_spread_pct`` is documented on ``ProductTerms`` as the
  relative bid-ask spread **as a fraction of ask**, i.e. ``(ask-bid)/ask``.
  Applying that definition exactly would require solving for an implied ask
  from ``V`` first. Instead -- consistent with how a spread is quoted and
  applied everywhere else in this codebase (half-spread around a mid) --
  ``V`` is treated as the *mid* and the exit bid is
  ``V * (1 - exit_spread_pct/2)``, with an implied ask of
  ``V * (1 + exit_spread_pct/2)``. For the small spreads typical of these
  products (single-digit percent), the difference between "``exit_spread_pct``
  of ask" and "``exit_spread_pct`` of mid" is second-order
  (``O(exit_spread_pct**2)``) and is accepted here as a documented
  approximation, not a silent inconsistency.
- Net return is always the simple return ``exit_bid / entry_ask - 1``
  (Master Spec: "Renditen: ... Netto-Renditen von Produkten als einfache
  Rendite").

Exit value ``V`` at horizon ``h`` for a path that has *not* knocked out:
``V = fair_value_fn(spot=close[h-1], financing_level=F_h or fixed strike,
...) * (1 + premium_over_fair)`` -- ``premium_over_fair`` is carried forward
unchanged from entry (issuer margin persistence, Build Contract). For
``turbo_open_end``/``mini_future`` the financing level ``F`` is rolled
forward daily exactly as in ``pricing/financing.py``
(``F_h = F_0 * (1 + (r +/- s) * calendar_days/360)``, ``calendar_days = h *
7/5``); for ``turbo_classic`` the strike stays fixed and ``as_of`` is instead
advanced by the same ``calendar_days`` (rounded to the nearest whole day, for
date arithmetic), shrinking the fair-value formula's own time-to-maturity.

KO is an absorbing event (CLAUDE.md rule 15): once a path's first barrier
touch (:func:`turboedge.simulation.barrier.first_hit_index`) occurs within
the horizon, the path's net return for that (and every later) horizon is
fixed at the KO settlement, regardless of what the underlying does
afterwards. KO settlement never goes through the bid/ask spread above (it is
a formulaic issuer buy-back, not a market trade): ``net_return = ko_residual
/ entry_ask - 1``, with ``ko_residual`` exactly 0 for ``turbo_open_end`` /
``turbo_classic`` (Master Spec: barrier == financing level, no residual by
construction) and ``max(exec_price - F, 0) * ratio/fx`` (Long, mirrored for
Short) for ``mini_future``, where ``exec_price`` is the barrier itself, or
the (worse) opening price on a gap-through day, and ``F`` is rolled forward
to the knockout day using the same daily roll-forward formula.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import numpy.typing as npt

from turboedge.simulation.barrier import first_hit_index
from turboedge.simulation.paths import PathSet
from turboedge.storage.schemas import Direction, ProductType

_DAY_COUNT_BASIS = 360.0
_CALENDAR_DAYS_PER_TRADING_DAY = 7.0 / 5.0


@dataclass(frozen=True, slots=True)
class ProductTerms:
    """Static terms of one candidate certificate, as of the simulation's ``as_of``."""

    isin: str
    direction: Direction
    product_type: ProductType
    financing_level: float  # current F (open-end/mini-future) or fixed strike K (classic)
    knockout_barrier: float
    ratio: float
    fx: float
    entry_ask: float
    entry_bid: float
    maturity: date | None
    financing_spread: float  # annual, act/360
    ref_rate: float  # annual, act/360
    exit_spread_pct: float  # assumed relative bid-ask spread at exit; see module docstring
    premium_over_fair: float  # current (mid - theoretical_fair_value)/mid, carried to exit


@dataclass(frozen=True, slots=True)
class PayoffDistribution:
    """Simulated net-return distribution for one horizon (Master Spec §16)."""

    horizon_days: int
    net_returns: npt.NDArray[np.float64]
    ko: npt.NDArray[np.bool_]
    mfe: npt.NDArray[np.float64]
    mae: npt.NDArray[np.float64]
    mean: float
    median: float
    q05: float
    q25: float
    q75: float
    q95: float
    p_profit: float
    p_ko: float
    es95: float
    mc_standard_error: float


def _rolled_financing_level(
    f0: float,
    ref_rate: float,
    spread: float,
    direction: Direction,
    n_trading_days: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """``F(n) = F0 * (1 + (r +/- s) * calendar_days/360)``, ``calendar_days = n * 7/5``."""
    calendar_days = np.asarray(n_trading_days, dtype=np.float64) * _CALENDAR_DAYS_PER_TRADING_DAY
    rate_term = (ref_rate + spread) if direction == Direction.LONG else (ref_rate - spread)
    result: npt.NDArray[np.float64] = f0 * (1.0 + rate_term * calendar_days / _DAY_COUNT_BASIS)
    return result


def _financing_level_and_as_of_for_horizon(
    terms: ProductTerms, as_of: date, n_trading_days: int
) -> tuple[float, date]:
    """Financing level (or fixed strike) and valuation date to value the product at, after
    ``n_trading_days`` simulated trading days have elapsed.
    """
    calendar_days = n_trading_days * _CALENDAR_DAYS_PER_TRADING_DAY
    if terms.product_type == ProductType.TURBO_CLASSIC:
        as_of_n = as_of + timedelta(days=round(calendar_days))
        return terms.financing_level, as_of_n
    f_n = float(
        _rolled_financing_level(
            terms.financing_level,
            terms.ref_rate,
            terms.financing_spread,
            terms.direction,
            n_trading_days,
        )
    )
    return f_n, as_of


def _ko_residual(
    terms: ProductTerms,
    exec_price: npt.NDArray[np.float64],
    n_trading_days_at_ko: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if terms.product_type != ProductType.MINI_FUTURE:
        return np.zeros_like(exec_price)
    f_at_ko = np.asarray(
        _rolled_financing_level(
            terms.financing_level,
            terms.ref_rate,
            terms.financing_spread,
            terms.direction,
            n_trading_days_at_ko,
        )
    )
    moneyness = exec_price - f_at_ko if terms.direction == Direction.LONG else f_at_ko - exec_price
    return np.maximum(moneyness, 0.0) * terms.ratio / terms.fx


def _call_fair_value(
    fair_value_fn: Callable[..., float],
    terms: ProductTerms,
    spot: npt.NDArray[np.float64],
    financing_level: float,
    as_of_n: date,
) -> npt.NDArray[np.float64]:
    """Scalar-callable fallback: one Python-level ``fair_value_fn`` call per
    path. Only used when a caller supplies a custom scalar ``fair_value_fn``
    (e.g. a test stand-in) instead of a vectorized ``fair_value_array_fn`` --
    see :func:`simulate_product_payoff`'s "Performance" docstring section.
    Kept, unchanged, for that fallback and for backward compatibility; the
    production default path uses :func:`_call_fair_value_array` instead.
    """
    out = np.empty(spot.shape, dtype=np.float64)
    flat_spot = spot.reshape(-1)
    flat_out = out.reshape(-1)
    for i in range(flat_spot.size):
        value = fair_value_fn(
            direction=terms.direction,
            product_type=terms.product_type,
            spot=float(flat_spot[i]),
            financing_level=financing_level,
            knockout_barrier=terms.knockout_barrier,
            ratio=terms.ratio,
            fx=terms.fx,
            ref_rate=terms.ref_rate,
            financing_spread=terms.financing_spread,
            as_of=as_of_n,
            maturity=terms.maturity,
            dividend_yield=0.0,
        )
        if not np.isfinite(value):
            raise ValueError(f"fair_value_fn returned a non-finite value: {value!r}")
        flat_out[i] = value
    return out


def _call_fair_value_array(
    fair_value_array_fn: Callable[..., npt.NDArray[np.float64]],
    terms: ProductTerms,
    spot: npt.NDArray[np.float64],
    financing_level: npt.NDArray[np.float64] | float,
    as_of: date | npt.NDArray[np.object_],
) -> npt.NDArray[np.float64]:
    """Vectorized counterpart of :func:`_call_fair_value`: one call values
    every path (and, when ``spot``/``financing_level``/``as_of`` carry a
    day/horizon axis, every day or horizon) at once -- see
    :func:`simulate_product_payoff`'s "Performance" docstring section.
    """
    value = np.asarray(
        fair_value_array_fn(
            direction=terms.direction,
            product_type=terms.product_type,
            spot=spot,
            financing_level=financing_level,
            knockout_barrier=terms.knockout_barrier,
            ratio=terms.ratio,
            fx=terms.fx,
            ref_rate=terms.ref_rate,
            financing_spread=terms.financing_spread,
            as_of=as_of,
            maturity=terms.maturity,
            dividend_yield=0.0,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(value)):
        raise ValueError("fair_value_array_fn returned a non-finite value")
    return value


def _financing_and_as_of_arrays(
    terms: ProductTerms, as_of: date, n_trading_days: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64] | float, date | npt.NDArray[np.object_]]:
    """Vectorized (over a day/horizon axis) counterpart of
    :func:`_financing_level_and_as_of_for_horizon`: ``n_trading_days`` is an
    array of elapsed-trading-day counts (one per simulated day, or one per
    requested horizon), and this returns the matching array of financing
    levels / valuation dates -- reproducing that function's per-element
    formula exactly, via :func:`_rolled_financing_level` (already array-
    vectorized) for open-end/mini-future, or by advancing ``as_of`` per
    element for classic (a small, day/horizon-axis-sized Python loop over
    ``date`` arithmetic -- never sized to the path axis).
    """
    if terms.product_type == ProductType.TURBO_CLASSIC:
        calendar_days = n_trading_days * _CALENDAR_DAYS_PER_TRADING_DAY
        as_of_arr = np.array(
            [as_of + timedelta(days=round(float(cd))) for cd in calendar_days], dtype=object
        )
        return terms.financing_level, as_of_arr
    f_arr = _rolled_financing_level(
        terms.financing_level,
        terms.ref_rate,
        terms.financing_spread,
        terms.direction,
        n_trading_days,
    )
    return f_arr, as_of


def simulate_product_payoff(
    terms: ProductTerms,
    paths: PathSet,
    horizons: Sequence[int],
    *,
    as_of: date,
    fair_value_fn: Callable[..., float] | None = None,
    fair_value_array_fn: Callable[..., npt.NDArray[np.float64]] | None = None,
) -> dict[int, PayoffDistribution]:
    """Simulate the full turbo/knockout payoff over ``paths`` for each horizon in ``horizons``.

    ``fair_value_fn`` defaults (when both it and ``fair_value_array_fn`` are
    ``None``) to a lazy import of
    ``turboedge.pricing.fair_value.theoretical_fair_value_array`` -- lazy so
    this module has no hard import-time dependency on
    ``pricing/fair_value.py`` (Build Contract W5/W1 dependency note); tests
    should pass an explicit, simple stand-in instead of relying on the lazy
    import.

    Performance (Build Contract v2 W7 fix): this function is called once per
    ``(product, horizon)`` pair by ``ranking/ev.py`` -- a real scan's
    product x horizon x path count made a per-path Python loop over
    ``fair_value_fn`` the dominant cost, roughly an order of magnitude over
    budget. Two calling conventions are therefore supported:

    - ``fair_value_array_fn`` (vectorized: takes/returns numpy arrays, see
      :func:`turboedge.pricing.fair_value.theoretical_fair_value_array`) --
      used whenever provided, or resolved automatically via the lazy import
      above when neither callback is given (i.e. the default, production
      path). Every fair-value evaluation this function needs -- across all
      paths *and* across the whole day/horizon axis -- is then done in a
      small, fixed number of vectorized numpy calls, independent of
      ``n_paths``.
    - ``fair_value_fn`` (scalar: one call per element) -- used only as a
      fallback when a caller supplies it *without* also supplying
      ``fair_value_array_fn``. Kept for backward compatibility and for
      tests that want a trivial scalar stand-in without depending on
      ``pricing/fair_value.py``; this path still loops once per path in
      Python (the original, pre-vectorization behavior) and is not the
      hot path any real scan takes.

    Raises:
        ValueError: if ``terms.entry_ask <= 0``, ``terms.ratio <= 0``,
            ``terms.fx <= 0``, ``horizons`` is empty, any horizon exceeds
            ``paths``' simulated length, or ``fair_value_fn``/
            ``fair_value_array_fn`` returns a non-finite value.
    """
    if not (terms.entry_ask > 0):
        raise ValueError(f"terms.entry_ask must be > 0, got {terms.entry_ask!r}")
    if not (terms.ratio > 0):
        raise ValueError(f"terms.ratio must be > 0, got {terms.ratio!r}")
    if not (terms.fx > 0):
        raise ValueError(f"terms.fx must be > 0, got {terms.fx!r}")
    if not horizons:
        raise ValueError("horizons must not be empty")
    n_paths, h_max_available = paths.open.shape
    max_horizon = max(horizons)
    if max_horizon > h_max_available:
        raise ValueError(f"horizon {max_horizon} exceeds simulated path length {h_max_available}")

    if fair_value_fn is None and fair_value_array_fn is None:
        from turboedge.pricing.fair_value import theoretical_fair_value_array

        fair_value_array_fn = theoretical_fair_value_array
    use_array_path = fair_value_array_fn is not None

    first_idx = first_hit_index(paths, terms.knockout_barrier, terms.direction)  # (n_paths,)
    ko_mask_any = first_idx >= 0
    d_safe = np.clip(first_idx, 0, None)

    open_at_hit = paths.open[np.arange(n_paths), d_safe]
    if terms.direction == Direction.LONG:
        gap_through = open_at_hit <= terms.knockout_barrier
    else:
        gap_through = open_at_hit >= terms.knockout_barrier
    exec_price = np.where(gap_through, open_at_hit, terms.knockout_barrier)
    n_trading_days_at_ko = (d_safe + 1).astype(np.float64)
    ko_residual = _ko_residual(terms, exec_price, n_trading_days_at_ko)
    ko_residual = np.where(ko_mask_any, ko_residual, 0.0)
    net_return_ko = ko_residual / terms.entry_ask - 1.0

    # MFE/MAE: per-day product value evaluated at that day's high/low, running
    # best/worst since entry, truncated once a path has knocked out (Master
    # Spec §16: "MFE" / "MAE"). Computed once up to the longest requested
    # horizon (not the full simulated path length) and sliced per horizon below.
    day_idx = np.arange(1, max_horizon + 1)  # 1-based trading-day counts
    if use_array_path:
        assert fair_value_array_fn is not None
        day_idx_f = day_idx.astype(np.float64)
        f_or_k_days, as_of_days = _financing_and_as_of_arrays(terms, as_of, day_idx_f)
        value_high_all = _call_fair_value_array(
            fair_value_array_fn, terms, paths.high[:, :max_horizon], f_or_k_days, as_of_days
        ) * (1.0 + terms.premium_over_fair)
        value_low_all = _call_fair_value_array(
            fair_value_array_fn, terms, paths.low[:, :max_horizon], f_or_k_days, as_of_days
        ) * (1.0 + terms.premium_over_fair)
        if terms.direction == Direction.LONG:
            best_per_day, worst_per_day = value_high_all, value_low_all
        else:
            best_per_day, worst_per_day = value_low_all, value_high_all
    else:
        assert fair_value_fn is not None
        best_per_day = np.empty((n_paths, max_horizon), dtype=np.float64)
        worst_per_day = np.empty((n_paths, max_horizon), dtype=np.float64)
        for d in range(max_horizon):
            f_or_k, as_of_d = _financing_level_and_as_of_for_horizon(terms, as_of, int(day_idx[d]))
            value_high = _call_fair_value(
                fair_value_fn, terms, paths.high[:, d], f_or_k, as_of_d
            ) * (1.0 + terms.premium_over_fair)
            value_low = _call_fair_value(fair_value_fn, terms, paths.low[:, d], f_or_k, as_of_d) * (
                1.0 + terms.premium_over_fair
            )
            if terms.direction == Direction.LONG:
                best_per_day[:, d] = value_high
                worst_per_day[:, d] = value_low
            else:
                best_per_day[:, d] = value_low
                worst_per_day[:, d] = value_high

    day_valid = (first_idx[:, None] < 0) | (np.arange(max_horizon)[None, :] <= first_idx[:, None])
    best_masked = np.where(day_valid, best_per_day, -np.inf)
    worst_masked = np.where(day_valid, worst_per_day, np.inf)
    running_best = np.maximum.accumulate(best_masked, axis=1)
    running_worst = np.minimum.accumulate(worst_masked, axis=1)
    mfe_full = (running_best - terms.entry_ask) / terms.entry_ask
    mae_full = (running_worst - terms.entry_ask) / terms.entry_ask

    horizons_list = list(horizons)
    for h in horizons_list:
        if h <= 0:
            raise ValueError(f"horizon must be > 0, got {h!r}")

    # Exit value at each requested horizon (not yet knocked out): computed for
    # every horizon in one batch here (vectorized over paths *and* horizons)
    # when using the array path, then sliced per horizon in the loop below.
    net_return_alive_h: npt.NDArray[np.float64] | None = None
    if use_array_path:
        assert fair_value_array_fn is not None
        h_arr = np.asarray(horizons_list, dtype=np.float64)
        col_idx = np.asarray(horizons_list, dtype=np.int64) - 1
        spot_at_h = paths.close[:, col_idx]  # (n_paths, n_horizons)
        f_or_k_h, as_of_h_arr = _financing_and_as_of_arrays(terms, as_of, h_arr)
        fv_h = _call_fair_value_array(fair_value_array_fn, terms, spot_at_h, f_or_k_h, as_of_h_arr)
        exit_value_h = fv_h * (1.0 + terms.premium_over_fair)
        exit_bid_h = exit_value_h * (1.0 - terms.exit_spread_pct / 2.0)
        net_return_alive_h = exit_bid_h / terms.entry_ask - 1.0

    results: dict[int, PayoffDistribution] = {}
    for i, h in enumerate(horizons_list):
        ko_at_h = ko_mask_any & (first_idx <= h - 1)

        if net_return_alive_h is not None:
            net_return_alive = net_return_alive_h[:, i]
        else:
            assert fair_value_fn is not None
            spot_h = paths.close[:, h - 1]
            f_or_k, as_of_h = _financing_level_and_as_of_for_horizon(terms, as_of, h)
            fv = _call_fair_value(fair_value_fn, terms, spot_h, f_or_k, as_of_h)
            exit_value = fv * (1.0 + terms.premium_over_fair)
            exit_bid = exit_value * (1.0 - terms.exit_spread_pct / 2.0)
            net_return_alive = exit_bid / terms.entry_ask - 1.0

        net_returns = np.where(ko_at_h, net_return_ko, net_return_alive)
        if not np.all(np.isfinite(net_returns)):
            raise ValueError(f"non-finite net_returns computed for horizon {h}")

        mfe = mfe_full[:, h - 1]
        mae = mae_full[:, h - 1]

        # Single batched np.quantile call (one sort/partition of net_returns
        # shared across all four quantile points) instead of four separate
        # calls -- a measured ~2x contributor to this function's own
        # (already-vectorized) per-call cost at scan scale (profiling, W7).
        q05, q25, q75, q95 = (float(x) for x in np.quantile(net_returns, [0.05, 0.25, 0.75, 0.95]))
        below_q05 = net_returns <= q05
        es95 = float(np.mean(net_returns[below_q05])) if np.any(below_q05) else q05
        mc_se = float(np.std(net_returns, ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0

        results[h] = PayoffDistribution(
            horizon_days=h,
            net_returns=net_returns,
            ko=ko_at_h,
            mfe=mfe,
            mae=mae,
            mean=float(np.mean(net_returns)),
            median=float(np.median(net_returns)),
            q05=q05,
            q25=q25,
            q75=q75,
            q95=q95,
            p_profit=float(np.mean(net_returns > 0)),
            p_ko=float(np.mean(ko_at_h)),
            es95=es95,
            mc_standard_error=mc_se,
        )

    return results


__all__ = ["PayoffDistribution", "ProductTerms", "simulate_product_payoff"]
