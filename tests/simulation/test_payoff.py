from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from turboedge.pricing.fair_value import theoretical_fair_value
from turboedge.simulation.barrier import first_hit_index
from turboedge.simulation.paths import PathSet, simulate_paths
from turboedge.simulation.payoff import (
    PayoffDistribution,
    ProductTerms,
    _call_fair_value,
    _financing_level_and_as_of_for_horizon,
    _ko_residual,
    simulate_product_payoff,
)
from turboedge.storage.schemas import Direction, ProductType

from .conftest import synthetic_daily_bars


def simple_fair_value(
    *,
    direction: Direction,
    product_type: ProductType,
    spot: float,
    financing_level: float,
    knockout_barrier: float,
    ratio: float,
    fx: float,
    ref_rate: float,
    financing_spread: float,
    as_of: date,
    maturity: date | None,
    dividend_yield: float = 0.0,
) -> float:
    """Trivial intrinsic-only stand-in used by every test (no import dependency on W1)."""
    del knockout_barrier, ref_rate, financing_spread, as_of, maturity, dividend_yield
    moneyness = spot - financing_level if direction == Direction.LONG else financing_level - spot
    return max(moneyness, 0.0) * ratio / fx


def _flat_pathset(spot0: float, n_paths: int, horizon_days: int) -> PathSet:
    """A PathSet with zero volatility: every day's O=H=L=C=spot0."""
    flat = np.full((n_paths, horizon_days), spot0, dtype=np.float64)
    return PathSet(
        underlying_id="TEST",
        spot0=spot0,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        open=flat.copy(),
        high=flat.copy(),
        low=flat.copy(),
        close=flat.copy(),
        weekend_before=np.zeros(horizon_days, dtype=np.bool_),
        method="manual",
    )


def _default_terms(**overrides: object) -> ProductTerms:
    defaults: dict[str, object] = dict(
        isin="DE000TEST0001",
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=80.0,
        knockout_barrier=80.0,
        ratio=1.0,
        fx=1.0,
        entry_ask=20.5,
        entry_bid=20.0,
        maturity=None,
        financing_spread=0.01,
        ref_rate=0.03,
        exit_spread_pct=0.02,
        premium_over_fair=0.0,
    )
    defaults.update(overrides)
    return ProductTerms(**defaults)  # type: ignore[arg-type]


def _real_pathset(
    seed: int = 1, n_paths: int = 300, horizon_days: int = 14
) -> tuple[PathSet, list]:
    bars = synthetic_daily_bars(600, seed=seed)
    last = max(b.ts for b in bars)
    start = datetime(last.year, last.month, last.day, 20, 0, tzinfo=UTC)
    start += timedelta(days=1)
    rng = np.random.default_rng(seed)
    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=horizon_days,
        n_paths=n_paths,
        rng=rng,
        method="block_bootstrap",
    )
    return ps, bars


def test_simulate_product_payoff_shapes_and_types() -> None:
    ps, _ = _real_pathset()
    terms = _default_terms(financing_level=ps.spot0 * 0.85, knockout_barrier=ps.spot0 * 0.85)
    results = simulate_product_payoff(
        terms, ps, [3, 5, 7, 10, 14], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    assert set(results.keys()) == {3, 5, 7, 10, 14}
    n_paths = ps.open.shape[0]
    for h, dist in results.items():
        assert isinstance(dist, PayoffDistribution)
        assert dist.horizon_days == h
        assert dist.net_returns.shape == (n_paths,)
        assert dist.ko.shape == (n_paths,)
        assert dist.mfe.shape == (n_paths,)
        assert dist.mae.shape == (n_paths,)
        assert np.all(np.isfinite(dist.net_returns))
        for scalar in (
            dist.mean,
            dist.median,
            dist.q05,
            dist.q25,
            dist.q75,
            dist.q95,
            dist.p_profit,
            dist.p_ko,
            dist.es95,
            dist.mc_standard_error,
        ):
            assert isinstance(scalar, float)
            assert np.isfinite(scalar)
        assert 0.0 <= dist.p_profit <= 1.0
        assert 0.0 <= dist.p_ko <= 1.0
        assert dist.mc_standard_error >= 0.0


def test_simulate_product_payoff_determinism_with_seed() -> None:
    bars = synthetic_daily_bars(500, seed=5)
    last = max(b.ts for b in bars)
    start = datetime(last.year, last.month, last.day, 20, 0, tzinfo=UTC) + timedelta(days=1)
    terms = _default_terms(
        financing_level=bars[-1].close * 0.85, knockout_barrier=bars[-1].close * 0.85
    )

    psA = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=200,
        rng=np.random.default_rng(7),
    )
    psB = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=200,
        rng=np.random.default_rng(7),
    )
    resA = simulate_product_payoff(
        terms, psA, [5, 10], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    resB = simulate_product_payoff(
        terms, psB, [5, 10], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    for h in (5, 10):
        assert np.array_equal(resA[h].net_returns, resB[h].net_returns)


def test_gap_through_is_recognized_as_ko() -> None:
    ps = PathSet(
        underlying_id="TEST",
        spot0=100.0,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        open=np.array([[95.0, 90.0]]),  # already through the barrier at day-0 open
        high=np.array([[96.0, 91.0]]),
        low=np.array([[93.0, 88.0]]),
        close=np.array([[94.0, 89.0]]),
        weekend_before=np.array([False, False]),
        method="manual",
    )
    terms = _default_terms(
        product_type=ProductType.MINI_FUTURE,
        financing_level=90.0,
        knockout_barrier=100.0,
        entry_ask=10.5,
        ref_rate=0.0,
        financing_spread=0.0,  # F never rolls -> exact expected residual below
    )
    res = simulate_product_payoff(
        terms, ps, [1, 2], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    assert bool(res[1].ko[0]) is True
    assert bool(res[2].ko[0]) is True
    # Gap-through execution price is the (worse) opening price 95.0, not the
    # barrier 100.0: residual = max(95 - 90, 0) * 1/1 = 5.
    expected_net = 5.0 / 10.5 - 1.0
    assert res[1].net_returns[0] == pytest.approx(expected_net)
    assert res[2].net_returns[0] == pytest.approx(expected_net)


def test_long_short_symmetry() -> None:
    spot0 = 100.0
    horizon_days = 5
    n_paths = 4000
    bars = synthetic_daily_bars(500, seed=2, spot0=spot0)
    last = max(b.ts for b in bars)
    start = datetime(last.year, last.month, last.day, 20, 0, tzinfo=UTC) + timedelta(days=1)

    ps_long = simulate_paths(
        bars,
        spot0=spot0,
        start=start,
        horizon_days=horizon_days,
        n_paths=n_paths,
        rng=np.random.default_rng(3),
        drift_log_return=None,
    )
    # Mirror the same path set by additive reflection around spot0
    # (S' = 2*spot0 - S): with ref_rate=financing_spread=0 (F never rolls),
    # a Long with barrier/F=80 on S has *exactly* the same KO days and the
    # same max(S-F,0) intrinsic payoff, day for day, as a Short with
    # barrier/F'=2*spot0-80=120 on S' -- this is an exact per-path identity,
    # not merely a statistical one, so no simulation tolerance is needed
    # beyond floating-point.
    f_long = 80.0
    f_short = 2.0 * spot0 - f_long
    mirrored = PathSet(
        underlying_id=ps_long.underlying_id,
        spot0=spot0,
        start=ps_long.start,
        open=2.0 * spot0 - ps_long.open,
        high=2.0 * spot0 - ps_long.low,
        low=2.0 * spot0 - ps_long.high,
        close=2.0 * spot0 - ps_long.close,
        weekend_before=ps_long.weekend_before,
        method=ps_long.method,
    )
    entry_ask = spot0 - f_long + 0.5
    terms_long = _default_terms(
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=f_long,
        knockout_barrier=f_long,
        entry_ask=entry_ask,
        ref_rate=0.0,
        financing_spread=0.0,
    )
    terms_short = _default_terms(
        direction=Direction.SHORT,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=f_short,
        knockout_barrier=f_short,
        entry_ask=entry_ask,
        ref_rate=0.0,
        financing_spread=0.0,
    )
    res_long = simulate_product_payoff(
        terms_long, ps_long, [horizon_days], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    res_short = simulate_product_payoff(
        terms_short,
        mirrored,
        [horizon_days],
        as_of=date(2025, 1, 1),
        fair_value_fn=simple_fair_value,
    )
    assert res_long[horizon_days].ko.tolist() == res_short[horizon_days].ko.tolist()
    assert res_long[horizon_days].net_returns == pytest.approx(res_short[horizon_days].net_returns)
    assert res_long[horizon_days].mean == pytest.approx(res_short[horizon_days].mean)
    assert res_long[horizon_days].p_ko == pytest.approx(res_short[horizon_days].p_ko)


def test_open_end_financing_level_rolls_forward() -> None:
    spot0 = 100.0
    horizon_days = 10
    ps = _flat_pathset(spot0, n_paths=1, horizon_days=horizon_days)
    f0 = 70.0
    terms = _default_terms(
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=f0,
        knockout_barrier=f0,
        entry_ask=spot0 - f0,
        ref_rate=0.03,
        financing_spread=0.02,
        exit_spread_pct=0.0,
        premium_over_fair=0.0,
    )
    res = simulate_product_payoff(
        terms, ps, [1, horizon_days], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    calendar_days_h = horizon_days * 7.0 / 5.0
    f_h = f0 * (1.0 + (0.03 + 0.02) * calendar_days_h / 360.0)
    expected_value_h = spot0 - f_h
    expected_net_h = expected_value_h / terms.entry_ask - 1.0
    assert res[horizon_days].net_returns[0] == pytest.approx(expected_net_h)
    # F strictly increases with more elapsed time (Long: r+s > 0) -> value at
    # h=10 must be lower than at h=1.
    assert res[horizon_days].net_returns[0] < res[1].net_returns[0]


def test_classic_uses_fixed_strike_and_advances_as_of() -> None:
    spot0 = 100.0
    horizon_days = 5
    ps = _flat_pathset(spot0, n_paths=1, horizon_days=horizon_days)
    strike = 70.0
    as_of = date(2025, 1, 1)
    maturity = date(2025, 6, 1)

    calls: list[date] = []

    def recording_fv(
        *,
        direction: Direction,
        product_type: ProductType,
        spot: float,
        financing_level: float,
        knockout_barrier: float,
        ratio: float,
        fx: float,
        ref_rate: float,
        financing_spread: float,
        as_of: date,
        maturity: date | None,
        dividend_yield: float = 0.0,
    ) -> float:
        calls.append(as_of)
        assert financing_level == strike  # fixed strike, never rolled
        return simple_fair_value(
            direction=direction,
            product_type=product_type,
            spot=spot,
            financing_level=financing_level,
            knockout_barrier=knockout_barrier,
            ratio=ratio,
            fx=fx,
            ref_rate=ref_rate,
            financing_spread=financing_spread,
            as_of=as_of,
            maturity=maturity,
            dividend_yield=dividend_yield,
        )

    terms = _default_terms(
        product_type=ProductType.TURBO_CLASSIC,
        financing_level=strike,
        knockout_barrier=strike,
        maturity=maturity,
        entry_ask=spot0 - strike,
        exit_spread_pct=0.0,
        premium_over_fair=0.0,
    )
    simulate_product_payoff(terms, ps, [horizon_days], as_of=as_of, fair_value_fn=recording_fv)
    expected_as_of = as_of + __import__("datetime").timedelta(days=round(horizon_days * 7.0 / 5.0))
    assert expected_as_of in calls


def test_turbo_open_end_ko_residual_is_zero() -> None:
    ps = PathSet(
        underlying_id="TEST",
        spot0=100.0,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        open=np.array([[80.0]]),
        high=np.array([[81.0]]),
        low=np.array([[79.0]]),
        close=np.array([[79.5]]),
        weekend_before=np.array([False]),
        method="manual",
    )
    terms = _default_terms(
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=80.0,
        knockout_barrier=80.0,
        entry_ask=20.0,
    )
    res = simulate_product_payoff(
        terms, ps, [1], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    assert res[1].ko[0]
    assert res[1].net_returns[0] == pytest.approx(-1.0)  # 0 residual -> total loss


def test_turbo_classic_ko_residual_is_zero() -> None:
    ps = PathSet(
        underlying_id="TEST",
        spot0=100.0,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        open=np.array([[80.0]]),
        high=np.array([[81.0]]),
        low=np.array([[79.0]]),
        close=np.array([[79.5]]),
        weekend_before=np.array([False]),
        method="manual",
    )
    terms = _default_terms(
        product_type=ProductType.TURBO_CLASSIC,
        financing_level=80.0,
        knockout_barrier=80.0,
        maturity=date(2026, 1, 1),
        entry_ask=20.0,
    )
    res = simulate_product_payoff(
        terms, ps, [1], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    assert res[1].ko[0]
    assert res[1].net_returns[0] == pytest.approx(-1.0)


def test_mini_future_residual_positive_on_moderate_gap() -> None:
    ps = PathSet(
        underlying_id="TEST",
        spot0=100.0,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        open=np.array([[95.0]]),  # gaps through barrier=100, but well above F=90
        high=np.array([[96.0]]),
        low=np.array([[93.0]]),
        close=np.array([[94.0]]),
        weekend_before=np.array([False]),
        method="manual",
    )
    terms = _default_terms(
        product_type=ProductType.MINI_FUTURE,
        financing_level=90.0,
        knockout_barrier=100.0,
        entry_ask=10.5,
        ref_rate=0.0,
        financing_spread=0.0,
    )
    res = simulate_product_payoff(
        terms, ps, [1], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    assert res[1].ko[0]
    assert res[1].net_returns[0] > -1.0  # residual > 0, not a total loss


def test_entry_ask_exit_bid_zero_move_zero_drift_convention() -> None:
    spot0 = 100.0
    horizon_days = 5
    ps = _flat_pathset(spot0, n_paths=1, horizon_days=horizon_days)
    f0 = 80.0
    entry_ask = spot0 - f0  # premium_over_fair = 0 at entry
    terms = _default_terms(
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=f0,
        knockout_barrier=f0,
        entry_ask=entry_ask,
        ref_rate=0.03,
        financing_spread=0.01,
        exit_spread_pct=0.02,
        premium_over_fair=0.0,
    )
    res = simulate_product_payoff(
        terms, ps, [horizon_days], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    calendar_days = horizon_days * 7.0 / 5.0
    f_h = f0 * (1.0 + (0.03 + 0.01) * calendar_days / 360.0)
    v = spot0 - f_h
    exit_bid = v * (1.0 - 0.02 / 2.0)
    expected = exit_bid / entry_ask - 1.0
    assert res[horizon_days].net_returns[0] == pytest.approx(expected)
    # Sanity: this should be negative (spread + financing drag, no favorable move).
    assert expected < 0.0


def test_rejects_non_positive_entry_ask() -> None:
    ps = _flat_pathset(100.0, n_paths=1, horizon_days=3)
    terms = _default_terms(entry_ask=0.0)
    with pytest.raises(ValueError):
        simulate_product_payoff(
            terms, ps, [3], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
        )


def test_rejects_horizon_beyond_simulated_length() -> None:
    ps = _flat_pathset(100.0, n_paths=1, horizon_days=3)
    terms = _default_terms()
    with pytest.raises(ValueError):
        simulate_product_payoff(
            terms, ps, [5], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
        )


def test_rejects_empty_horizons() -> None:
    ps = _flat_pathset(100.0, n_paths=1, horizon_days=3)
    terms = _default_terms()
    with pytest.raises(ValueError):
        simulate_product_payoff(
            terms, ps, [], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
        )


def test_es95_is_mean_of_worst_5_percent() -> None:
    ps, _ = _real_pathset(seed=42, n_paths=5000, horizon_days=10)
    terms = _default_terms(financing_level=ps.spot0 * 0.85, knockout_barrier=ps.spot0 * 0.85)
    res = simulate_product_payoff(
        terms, ps, [10], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    dist = res[10]
    manual_es95 = float(np.mean(dist.net_returns[dist.net_returns <= dist.q05]))
    assert dist.es95 == pytest.approx(manual_es95)
    assert dist.es95 <= dist.q05 + 1e-9


def test_mc_standard_error_matches_formula() -> None:
    ps, _ = _real_pathset(seed=11, n_paths=1000, horizon_days=7)
    terms = _default_terms(financing_level=ps.spot0 * 0.85, knockout_barrier=ps.spot0 * 0.85)
    res = simulate_product_payoff(
        terms, ps, [7], as_of=date(2025, 1, 1), fair_value_fn=simple_fair_value
    )
    dist = res[7]
    expected_se = float(np.std(dist.net_returns, ddof=1) / np.sqrt(dist.net_returns.size))
    assert dist.mc_standard_error == pytest.approx(expected_se)


def test_lazy_default_fair_value_fn_resolves_to_theoretical_fair_value() -> None:
    """When both ``fair_value_fn`` and ``fair_value_array_fn`` are omitted,
    the lazy import must succeed and resolve to
    ``pricing.fair_value.theoretical_fair_value_array`` (the W7-vectorized
    default, kept lazy so this module has no hard import-time dependency on
    ``pricing/fair_value.py``). See
    ``test_vectorized_matches_reference_implementation`` below for the full
    numerical cross-check against the scalar ``theoretical_fair_value``.
    """
    ps = _flat_pathset(100.0, n_paths=2, horizon_days=1)
    terms = _default_terms(
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=80.0,
        knockout_barrier=80.0,
        entry_ask=20.0,
    )
    res = simulate_product_payoff(terms, ps, [1], as_of=date(2025, 1, 1))
    assert np.all(np.isfinite(res[1].net_returns))


def _simulate_product_payoff_reference(
    terms: ProductTerms,
    paths: PathSet,
    horizons: Sequence[int],
    *,
    as_of: date,
    fair_value_fn: Callable[..., float] = theoretical_fair_value,
) -> dict[int, PayoffDistribution]:
    """Pre-vectorization reference algorithm (Build Contract v2 W7
    performance fix).

    This is the exact per-path, per-day Python-loop assembly
    ``simulate_product_payoff`` used *before* it grew a vectorized
    (``fair_value_array_fn``) fast path -- reassembled here from the
    still-present, unchanged scalar building blocks
    (``_call_fair_value``, ``_financing_level_and_as_of_for_horizon``,
    ``_ko_residual``, ``first_hit_index``) purely to cross-check the new
    default (vectorized) path for exact result identity on the same
    seeds/parameters, not merely "similar" statistics. Not used by
    production code -- test-only.
    """
    n_paths, h_max_available = paths.open.shape
    max_horizon = max(horizons)
    assert max_horizon <= h_max_available

    first_idx = first_hit_index(paths, terms.knockout_barrier, terms.direction)
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

    day_idx = np.arange(1, max_horizon + 1)
    best_per_day = np.empty((n_paths, max_horizon), dtype=np.float64)
    worst_per_day = np.empty((n_paths, max_horizon), dtype=np.float64)
    for d in range(max_horizon):
        f_or_k, as_of_d = _financing_level_and_as_of_for_horizon(terms, as_of, int(day_idx[d]))
        value_high = _call_fair_value(fair_value_fn, terms, paths.high[:, d], f_or_k, as_of_d) * (
            1.0 + terms.premium_over_fair
        )
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

    results: dict[int, PayoffDistribution] = {}
    for h in horizons:
        ko_at_h = ko_mask_any & (first_idx <= h - 1)
        spot_h = paths.close[:, h - 1]
        f_or_k, as_of_h = _financing_level_and_as_of_for_horizon(terms, as_of, h)
        fv = _call_fair_value(fair_value_fn, terms, spot_h, f_or_k, as_of_h)
        exit_value = fv * (1.0 + terms.premium_over_fair)
        exit_bid = exit_value * (1.0 - terms.exit_spread_pct / 2.0)
        net_return_alive = exit_bid / terms.entry_ask - 1.0

        net_returns = np.where(ko_at_h, net_return_ko, net_return_alive)
        mfe = mfe_full[:, h - 1]
        mae = mae_full[:, h - 1]
        q05, q25, q75, q95 = (float(np.quantile(net_returns, q)) for q in (0.05, 0.25, 0.75, 0.95))
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


@pytest.mark.parametrize(
    ("product_type", "direction"),
    [
        (ProductType.TURBO_OPEN_END, Direction.LONG),
        (ProductType.TURBO_OPEN_END, Direction.SHORT),
        (ProductType.MINI_FUTURE, Direction.LONG),
        (ProductType.MINI_FUTURE, Direction.SHORT),
        (ProductType.TURBO_CLASSIC, Direction.LONG),
        (ProductType.TURBO_CLASSIC, Direction.SHORT),
    ],
)
def test_vectorized_matches_reference_implementation(
    product_type: ProductType, direction: Direction
) -> None:
    """Result-identity check (W7 performance fix): the new default
    (vectorized, ``theoretical_fair_value_array``) path must reproduce the
    pre-vectorization scalar-loop reference algorithm (using the real,
    production ``theoretical_fair_value``) elementwise -- net returns, KO
    flags, MFE/MAE -- and the derived summary statistics (mean, quantiles,
    ``p_ko``, ES95, MC standard error) within tight floating-point
    tolerance, on the same seeds/parameters. Not just "similar" -- the two
    algorithms must agree to numerical precision.
    """
    ps, _ = _real_pathset(seed=99, n_paths=2000, horizon_days=20)
    spot0 = ps.spot0
    as_of = date(2025, 1, 1)
    horizons = [3, 7, 12, 20]

    is_mini_future = product_type == ProductType.MINI_FUTURE
    if direction == Direction.LONG:
        financing_level = spot0 * 0.85
        knockout_barrier = spot0 * 0.90 if is_mini_future else financing_level
        entry_ask = spot0 - financing_level
    else:
        financing_level = spot0 * 1.15
        knockout_barrier = spot0 * 1.10 if is_mini_future else financing_level
        entry_ask = financing_level - spot0

    terms = _default_terms(
        direction=direction,
        product_type=product_type,
        financing_level=financing_level,
        knockout_barrier=knockout_barrier,
        maturity=date(2026, 6, 1) if product_type == ProductType.TURBO_CLASSIC else None,
        entry_ask=entry_ask,
    )

    fast = simulate_product_payoff(terms, ps, horizons, as_of=as_of)  # default: vectorized
    ref = _simulate_product_payoff_reference(terms, ps, horizons, as_of=as_of)

    assert set(fast.keys()) == set(ref.keys()) == set(horizons)
    for h in horizons:
        np.testing.assert_allclose(fast[h].net_returns, ref[h].net_returns, atol=1e-9, rtol=1e-9)
        assert fast[h].ko.tolist() == ref[h].ko.tolist()
        np.testing.assert_allclose(fast[h].mfe, ref[h].mfe, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(fast[h].mae, ref[h].mae, atol=1e-9, rtol=1e-9)
        for field in (
            "mean",
            "median",
            "q05",
            "q25",
            "q75",
            "q95",
            "p_profit",
            "p_ko",
            "es95",
            "mc_standard_error",
        ):
            assert getattr(fast[h], field) == pytest.approx(getattr(ref[h], field), abs=1e-9)
