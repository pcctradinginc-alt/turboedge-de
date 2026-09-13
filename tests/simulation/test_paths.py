from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.simulation.paths import PathSet, simulate_paths
from turboedge.storage.schemas import UnderlyingBar

from .conftest import synthetic_daily_bars


def _start_after(bars: list[UnderlyingBar]) -> datetime:
    last = max(b.ts for b in bars)
    return datetime(last.year, last.month, last.day, 20, 0, tzinfo=UTC) + timedelta(days=1)


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
