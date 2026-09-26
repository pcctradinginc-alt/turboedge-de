from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from turboedge.simulation.bootstrap import DailyDraws
from turboedge.simulation.paths import PathSet, _apply_volatility_scaling, simulate_paths
from turboedge.storage.schemas import UnderlyingBar

from .conftest import make_bar, synthetic_daily_bars


def _start_after(bars: list[UnderlyingBar]) -> datetime:
    last = max(b.ts for b in bars)
    return datetime(last.year, last.month, last.day, 20, 0, tzinfo=UTC) + timedelta(days=1)


def _synthetic_bars_with_gaps(
    n_days: int,
    *,
    seed: int,
    gap_vol: float = 0.01,
    daily_vol: float = 0.005,
    range_vol: float = 0.002,
    spot0: float = 15000.0,
    start: date = date(2023, 1, 2),
) -> list[UnderlyingBar]:
    """Like ``synthetic_daily_bars`` but with a genuine, nonzero overnight gap
    each day (``synthetic_daily_bars`` sets every day's Open exactly equal to
    the previous day's Close, so its bootstrapped ``gap`` component has zero
    variance by construction -- fine for the existing tests, but it makes
    every ``target_sigma`` trivially "reachable" since the gap floor this
    module's docstring describes would then be zero. Used only for the
    volatility-scaling reachability tests, where a real, nonzero gap floor
    is the point.
    """
    rng = np.random.default_rng(seed)
    bars: list[UnderlyingBar] = []
    price = spot0
    d = start
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        o = price * float(np.exp(rng.normal(0.0, gap_vol)))
        c = o * float(np.exp(rng.normal(0.0, daily_vol)))
        hi = max(o, c) * float(np.exp(abs(rng.normal(0.0, range_vol))))
        lo = min(o, c) * float(np.exp(-abs(rng.normal(0.0, range_vol))))
        bars.append(make_bar(d, o, hi, lo, c))
        price = c
        d += timedelta(days=1)
    return bars


@pytest.mark.parametrize(
    "method",
    ["block_bootstrap", "regime_bootstrap", "monte_carlo", "vol_scaled_bootstrap"],
)
def test_simulate_paths_shapes_and_types(method: str) -> None:
    bars = synthetic_daily_bars(600, seed=1)
    start = _start_after(bars)
    rng = np.random.default_rng(0)
    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=14,
        n_paths=200,
        rng=rng,
        method=method,
    )
    assert isinstance(ps, PathSet)
    for arr in (ps.open, ps.high, ps.low, ps.close):
        assert arr.shape == (200, 14)
        assert arr.dtype == np.float64
        assert np.all(np.isfinite(arr))
        assert np.all(arr > 0)
    assert ps.weekend_before.shape == (14,)
    assert ps.weekend_before.dtype == np.bool_
    assert ps.method == method
    assert ps.spot0 == bars[-1].close


def test_simulate_paths_ohlc_consistency() -> None:
    bars = synthetic_daily_bars(600, seed=2)
    start = _start_after(bars)
    rng = np.random.default_rng(0)
    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=14,
        n_paths=500,
        rng=rng,
        method="block_bootstrap",
    )
    assert np.all(ps.high >= np.maximum(ps.open, ps.close) - 1e-8)
    assert np.all(ps.low <= np.minimum(ps.open, ps.close) + 1e-8)


def test_simulate_paths_determinism_with_seed() -> None:
    bars = synthetic_daily_bars(600, seed=3)
    start = _start_after(bars)
    psA = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=100,
        rng=np.random.default_rng(99),
        method="block_bootstrap",
    )
    psB = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=100,
        rng=np.random.default_rng(99),
        method="block_bootstrap",
    )
    assert np.array_equal(psA.close, psB.close)
    assert np.array_equal(psA.open, psB.open)


def test_simulate_paths_different_seeds_diverge() -> None:
    bars = synthetic_daily_bars(600, seed=3)
    start = _start_after(bars)
    psA = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=100,
        rng=np.random.default_rng(1),
        method="block_bootstrap",
    )
    psB = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=100,
        rng=np.random.default_rng(2),
        method="block_bootstrap",
    )
    assert not np.array_equal(psA.close, psB.close)


def test_simulate_paths_none_drift_removes_unconditional_mean() -> None:
    bars = synthetic_daily_bars(500, seed=4, drift=0.0003)  # historical upward drift
    start = _start_after(bars)
    rng = np.random.default_rng(5)
    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=20_000,
        rng=rng,
        drift_log_return=None,
        method="block_bootstrap",
    )
    total_log_return = np.log(ps.close[:, -1] / ps.spot0)
    assert total_log_return.mean() == pytest.approx(0.0, abs=1e-9)


def test_simulate_paths_drift_tilt_hits_target_mean() -> None:
    bars = synthetic_daily_bars(500, seed=6)
    start = _start_after(bars)
    rng = np.random.default_rng(7)
    target = 0.08
    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=20_000,
        rng=rng,
        drift_log_return=target,
        method="block_bootstrap",
    )
    total_log_return = np.log(ps.close[:, -1] / ps.spot0)
    assert total_log_return.mean() == pytest.approx(target, abs=1e-9)


def test_simulate_paths_drift_tilt_negative_target() -> None:
    bars = synthetic_daily_bars(500, seed=8)
    start = _start_after(bars)
    rng = np.random.default_rng(9)
    target = -0.05
    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=7,
        n_paths=20_000,
        rng=rng,
        drift_log_return=target,
        method="regime_bootstrap",
    )
    total_log_return = np.log(ps.close[:, -1] / ps.spot0)
    assert total_log_return.mean() == pytest.approx(target, abs=1e-9)


def test_simulate_paths_rejects_bad_inputs() -> None:
    bars = synthetic_daily_bars(100, seed=1)
    start = _start_after(bars)
    with pytest.raises(ValueError):
        simulate_paths(
            bars, spot0=0.0, start=start, horizon_days=5, n_paths=10, rng=np.random.default_rng(0)
        )
    with pytest.raises(ValueError):
        simulate_paths(
            bars, spot0=100.0, start=start, horizon_days=0, n_paths=10, rng=np.random.default_rng(0)
        )
    with pytest.raises(ValueError):
        simulate_paths(
            bars, spot0=100.0, start=start, horizon_days=5, n_paths=0, rng=np.random.default_rng(0)
        )


def test_simulate_paths_no_lookahead_ignores_bars_at_or_after_start() -> None:
    bars = synthetic_daily_bars(30, seed=1)
    start = _start_after(bars)
    # A "future" bar dated at/after start must never influence the bootstrap
    # sample (CLAUDE.md rule 4/5).
    future_bar = bars[-1]
    poisoned = future_bar.model_copy(update={"ts": start + timedelta(days=1)})
    rngA = np.random.default_rng(123)
    rngB = np.random.default_rng(123)
    psA = simulate_paths(
        bars,
        spot0=100.0,
        start=start,
        horizon_days=5,
        n_paths=50,
        rng=rngA,
        method="block_bootstrap",
    )
    psB = simulate_paths(
        [*bars, poisoned],
        spot0=100.0,
        start=start,
        horizon_days=5,
        n_paths=50,
        rng=rngB,
        method="block_bootstrap",
    )
    assert np.array_equal(psA.close, psB.close)


def test_simulate_paths_runtime_budget_10k_paths_14_days() -> None:
    """Generous runtime bound (not the tight '<1s' target, to keep CI robust
    across hardware) -- ``simulate_paths`` must be a fully vectorized numpy
    pipeline, not a per-path Python loop.
    """
    bars = synthetic_daily_bars(750, seed=1)
    start = _start_after(bars)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=14,
        n_paths=10_000,
        rng=rng,
        method="block_bootstrap",
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"simulate_paths took {elapsed:.3f}s for 10k paths x 14 days"


def test_simulate_paths_vol_scaled_bootstrap_runtime_budget() -> None:
    bars = synthetic_daily_bars(750, seed=1)
    start = _start_after(bars)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=14,
        n_paths=10_000,
        rng=rng,
        method="vol_scaled_bootstrap",
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"vol_scaled_bootstrap took {elapsed:.3f}s for 10k paths x 14 days"


# ---------------------------------------------------------------------------
# target_sigma / _apply_volatility_scaling (measured_results.md §6.14/§6.15)
# ---------------------------------------------------------------------------

_ALL_METHODS = ["block_bootstrap", "regime_bootstrap", "monte_carlo", "vol_scaled_bootstrap"]


def _synthetic_draws(seed: int, n_paths: int = 4000, horizon_days: int = 10) -> DailyDraws:
    """Hand-built ``DailyDraws`` for white-box testing of ``_apply_volatility_scaling``
    in isolation, independent of any particular bootstrap method."""
    rng = np.random.default_rng(seed)
    gap = rng.normal(0.0, 0.002, size=(n_paths, horizon_days))
    intraday = rng.normal(0.0003, 0.01, size=(n_paths, horizon_days))
    hi = np.abs(rng.normal(0.0, 0.003, size=(n_paths, horizon_days))) + np.maximum(intraday, 0.0)
    lo = -np.abs(rng.normal(0.0, 0.003, size=(n_paths, horizon_days))) + np.minimum(intraday, 0.0)
    return DailyDraws(gap=gap, intraday=intraday, hi=hi, lo=lo)


def test_apply_volatility_scaling_none_is_exact_noop() -> None:
    draws = _synthetic_draws(seed=1)
    scaled, realized_sigma, target_met = _apply_volatility_scaling(draws, None)
    assert scaled is draws  # bit-identical: the very same arrays, not merely equal-valued
    assert target_met is None
    expected = float(np.std(draws.gap.sum(axis=1) + draws.intraday.sum(axis=1)))
    assert realized_sigma == pytest.approx(expected, abs=1e-12)


def test_apply_volatility_scaling_is_not_a_noop_when_target_given() -> None:
    """Mandatory anti-triviality: fails if the function ever just returns its input."""
    draws = _synthetic_draws(seed=2)
    natural_sigma = float(np.std(draws.gap.sum(axis=1) + draws.intraday.sum(axis=1)))
    scaled, _, target_met = _apply_volatility_scaling(draws, natural_sigma * 2.0)
    assert target_met is True
    assert not np.array_equal(scaled.intraday, draws.intraday)
    assert not np.array_equal(scaled.hi, draws.hi)
    assert not np.array_equal(scaled.lo, draws.lo)


def test_apply_volatility_scaling_never_scales_gap() -> None:
    """Mandatory anti-triviality: fails if gap is scaled along with intraday."""
    draws = _synthetic_draws(seed=3)
    natural_sigma = float(np.std(draws.gap.sum(axis=1) + draws.intraday.sum(axis=1)))
    for target in (natural_sigma * 0.2, natural_sigma * 2.0, natural_sigma * 5.0, 1e-9):
        scaled, _, _ = _apply_volatility_scaling(draws, target)
        assert np.array_equal(scaled.gap, draws.gap), f"gap mutated for target={target}"


def test_apply_volatility_scaling_preserves_mean() -> None:
    draws = _synthetic_draws(seed=4)
    natural_sigma = float(np.std(draws.gap.sum(axis=1) + draws.intraday.sum(axis=1)))
    before_mean_total = float((draws.gap + draws.intraday).sum(axis=1).mean())
    for target in (natural_sigma * 0.3, natural_sigma * 3.0):
        scaled, _, _ = _apply_volatility_scaling(draws, target)
        after_mean_total = float((scaled.gap + scaled.intraday).sum(axis=1).mean())
        assert after_mean_total == pytest.approx(before_mean_total, abs=1e-9)
        # per-day mean is preserved too, not just the horizon total
        assert np.allclose(scaled.intraday.mean(axis=0), draws.intraday.mean(axis=0), atol=1e-12)


def test_apply_volatility_scaling_hits_reachable_target_exactly() -> None:
    draws = _synthetic_draws(seed=5)
    natural_sigma = float(np.std(draws.gap.sum(axis=1) + draws.intraday.sum(axis=1)))
    for factor in (0.5, 1.0, 1.5, 3.0):
        target = natural_sigma * factor
        _, realized_sigma, target_met = _apply_volatility_scaling(draws, target)
        assert target_met is True
        assert realized_sigma == pytest.approx(target, rel=1e-6)


def test_apply_volatility_scaling_unreachable_target_is_recorded_not_silent() -> None:
    draws = _synthetic_draws(seed=6)
    tiny_target = 1e-9  # far below the dispersion contributed by gap alone
    scaled, realized_sigma, target_met = _apply_volatility_scaling(draws, tiny_target)
    assert target_met is False
    # The achieved sigma must be visibly, not silently, different from what was asked.
    assert realized_sigma > tiny_target * 100
    # gap is still exactly untouched even in the unreachable/clamped case.
    assert np.array_equal(scaled.gap, draws.gap)

    # Independently recompute the exact reachability floor the docstring
    # describes (var(x) + 2f*cov(x,y) + f**2*var(y), minimized over f >= 0,
    # x=gap_total, y=intraday_total) and check realized_sigma landed there,
    # rather than assuming the naive var(gap)-only floor (which the exact
    # covariance-aware floor can sit fractionally below or at).
    gap_total = draws.gap.sum(axis=1)
    intraday_total = draws.intraday.sum(axis=1)
    var_gap = float(np.var(gap_total))
    var_intraday = float(np.var(intraday_total))
    cov_gi = float(np.cov(gap_total, intraday_total, bias=True)[0, 1])
    f_star = -cov_gi / var_intraday
    floor_var = (var_gap - cov_gi**2 / var_intraday) if f_star >= 0.0 else var_gap
    assert realized_sigma**2 == pytest.approx(floor_var, abs=1e-12)


def test_apply_volatility_scaling_ohlc_invariants_at_aggressive_factors() -> None:
    draws = _synthetic_draws(seed=7)
    natural_sigma = float(np.std(draws.gap.sum(axis=1) + draws.intraday.sum(axis=1)))
    for factor in (5.0, 0.2):
        scaled, _, _ = _apply_volatility_scaling(draws, natural_sigma * factor)
        assert np.all(scaled.hi >= np.maximum(scaled.intraday, 0.0) - 1e-9)
        assert np.all(scaled.lo <= np.minimum(scaled.intraday, 0.0) + 1e-9)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_simulate_paths_target_sigma_none_is_bit_identical(method: str) -> None:
    """The single most important test in this set: target_sigma=None must not
    change simulate_paths's output at all, for any method or seed."""
    bars = synthetic_daily_bars(600, seed=11)
    start = _start_after(bars)
    for seed in (0, 1, 42):
        psA = simulate_paths(
            bars,
            spot0=bars[-1].close,
            start=start,
            horizon_days=10,
            n_paths=300,
            rng=np.random.default_rng(seed),
            method=method,
        )
        psB = simulate_paths(
            bars,
            spot0=bars[-1].close,
            start=start,
            horizon_days=10,
            n_paths=300,
            rng=np.random.default_rng(seed),
            method=method,
            target_sigma=None,
        )
        assert np.array_equal(psA.open, psB.open)
        assert np.array_equal(psA.high, psB.high)
        assert np.array_equal(psA.low, psB.low)
        assert np.array_equal(psA.close, psB.close)
        assert psA.target_sigma_met is None
        assert psB.target_sigma_met is None


def test_simulate_paths_target_sigma_none_does_not_consume_extra_rng_state() -> None:
    """Confirms target_sigma=None isn't just numerically a no-op but structurally
    one too: it must not draw from ``rng`` at all, or determinism downstream of
    simulate_paths (e.g. a caller drawing more randomness afterwards) would
    silently shift relative to pre-existing behaviour."""
    bars = synthetic_daily_bars(600, seed=12)
    start = _start_after(bars)

    rngA = np.random.default_rng(7)
    simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=300,
        rng=rngA,
        method="vol_scaled_bootstrap",
    )
    next_drawA = rngA.standard_normal()

    rngB = np.random.default_rng(7)
    simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=300,
        rng=rngB,
        method="vol_scaled_bootstrap",
        target_sigma=None,
    )
    next_drawB = rngB.standard_normal()

    assert next_drawA == next_drawB


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_simulate_paths_target_sigma_achieved_when_reachable(method: str) -> None:
    bars = synthetic_daily_bars(700, seed=13)
    start = _start_after(bars)

    natural = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=4000,
        rng=np.random.default_rng(21),
        method=method,
    )
    assert natural.realized_sigma is not None
    target = natural.realized_sigma * 1.6

    scaled = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=4000,
        rng=np.random.default_rng(21),
        method=method,
        target_sigma=target,
    )
    assert scaled.target_sigma == pytest.approx(target)
    assert scaled.target_sigma_met is True
    assert scaled.realized_sigma == pytest.approx(target, rel=1e-6)

    total_log_return = np.log(scaled.close[:, -1] / scaled.spot0)
    assert total_log_return.std() == pytest.approx(target, rel=1e-3)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_simulate_paths_target_sigma_unreachable_is_detectable(method: str) -> None:
    # Needs bars with a genuine (nonzero-variance) overnight gap -- see
    # `_synthetic_bars_with_gaps`'s docstring for why `synthetic_daily_bars`
    # itself can't demonstrate an unreachable target (its gap is always 0).
    bars = _synthetic_bars_with_gaps(700, seed=14)
    start = _start_after(bars)
    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=4000,
        rng=np.random.default_rng(22),
        method=method,
        target_sigma=1e-9,
    )
    assert ps.target_sigma == pytest.approx(1e-9)
    assert ps.target_sigma_met is False
    # The caller can tell: realized_sigma is visibly, not silently, off-target.
    assert ps.realized_sigma is not None
    assert ps.realized_sigma > 1e-6


def test_simulate_paths_volatility_scaling_preserves_mean_alone() -> None:
    """Scaling alone (no drift tilt) must not move the mean total log return."""
    bars = synthetic_daily_bars(600, seed=15)
    start = _start_after(bars)

    baseline_draws_ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=6000,
        rng=np.random.default_rng(33),
        method="block_bootstrap",
        drift_log_return=0.0,
    )
    baseline_mean = np.log(baseline_draws_ps.close[:, -1] / baseline_draws_ps.spot0).mean()

    scaled_ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=6000,
        rng=np.random.default_rng(33),
        method="block_bootstrap",
        drift_log_return=0.0,
        target_sigma=(baseline_draws_ps.realized_sigma or 0.0) * 2.0,
    )
    scaled_mean = np.log(scaled_ps.close[:, -1] / scaled_ps.spot0).mean()
    assert scaled_mean == pytest.approx(baseline_mean, abs=1e-9)


def test_simulate_paths_scaling_and_drift_tilt_hit_both_targets() -> None:
    bars = synthetic_daily_bars(700, seed=16)
    start = _start_after(bars)

    natural = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=6000,
        rng=np.random.default_rng(44),
        method="vol_scaled_bootstrap",
    )
    assert natural.realized_sigma is not None
    target_sigma = natural.realized_sigma * 1.7
    target_drift = 0.04

    ps = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=6000,
        rng=np.random.default_rng(44),
        method="vol_scaled_bootstrap",
        drift_log_return=target_drift,
        target_sigma=target_sigma,
    )
    total_log_return = np.log(ps.close[:, -1] / ps.spot0)
    assert total_log_return.mean() == pytest.approx(target_drift, abs=1e-9)
    assert total_log_return.std() == pytest.approx(target_sigma, rel=1e-3)
    assert ps.target_sigma_met is True


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_simulate_paths_target_sigma_ohlc_invariants_aggressive_scale(method: str) -> None:
    bars = synthetic_daily_bars(700, seed=17)
    start = _start_after(bars)

    natural = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=2000,
        rng=np.random.default_rng(55),
        method=method,
    )
    assert natural.realized_sigma is not None

    for factor in (5.0, 0.2):
        ps = simulate_paths(
            bars,
            spot0=bars[-1].close,
            start=start,
            horizon_days=10,
            n_paths=2000,
            rng=np.random.default_rng(55),
            method=method,
            target_sigma=natural.realized_sigma * factor,
        )
        assert np.all(ps.high >= np.maximum(ps.open, ps.close) - 1e-6)
        assert np.all(ps.low <= np.minimum(ps.open, ps.close) + 1e-6)


def test_simulate_paths_target_sigma_determinism() -> None:
    bars = synthetic_daily_bars(600, seed=18)
    start = _start_after(bars)
    psA = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=500,
        rng=np.random.default_rng(66),
        method="vol_scaled_bootstrap",
        target_sigma=0.02,
    )
    psB = simulate_paths(
        bars,
        spot0=bars[-1].close,
        start=start,
        horizon_days=10,
        n_paths=500,
        rng=np.random.default_rng(66),
        method="vol_scaled_bootstrap",
        target_sigma=0.02,
    )
    assert np.array_equal(psA.close, psB.close)
    assert psA.realized_sigma == psB.realized_sigma
    assert psA.target_sigma_met == psB.target_sigma_met
