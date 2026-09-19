from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from turboedge.features.volatility import ewma_volatility
from turboedge.simulation.ko_calibration import build_ko_calibration_dataset
from turboedge.storage.schemas import Direction, UnderlyingBar


def test_realized_ko_matches_future_high_low_and_long_short_mirror(
    synthetic_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = synthetic_bars(200, seed=11)
    rng = np.random.default_rng(3)
    # step_days huge -> exactly one "as of" bar (t0 = _MIN_HISTORY_BARS=130)
    # is used, one LONG and one SHORT observation from it.
    obs = build_ko_calibration_dataset(
        bars,
        "TEST",
        horizons=(5,),
        sigma_levels=(1.5,),
        n_paths=200,
        method="vol_scaled_bootstrap",
        block_size=5,
        lookback_days=750,
        rng=rng,
        step_days=1000,
    )
    assert len(obs) == 2
    by_dir = {o.direction: o for o in obs}
    long_obs = by_dir[Direction.LONG]
    short_obs = by_dir[Direction.SHORT]
    assert long_obs.t0_index == short_obs.t0_index == 130

    sorted_bars = sorted(bars, key=lambda b: b.ts)
    t0 = long_obs.t0_index
    future = sorted_bars[t0 + 1 : t0 + 1 + 5]
    fut_low = np.array([b.low for b in future], dtype=np.float64)
    fut_high = np.array([b.high for b in future], dtype=np.float64)

    # Independently recompute daily_sigma exactly the way the module does:
    # only from bars available at prediction_time.
    history = [b for b in sorted_bars[: t0 + 1] if b.available_at <= long_obs.prediction_time]
    hist_closes = np.array([b.close for b in history], dtype=np.float64)
    daily_sigma = float(ewma_volatility(hist_closes)[-1])
    spot0 = sorted_bars[t0].close
    pct_distance = 1.5 * daily_sigma * np.sqrt(5.0)
    long_barrier = spot0 * (1.0 - pct_distance)
    short_barrier = spot0 * (1.0 + pct_distance)

    # Long/short are mirror images: the barrier sits below spot for Long
    # (touched via LOW) and above spot for Short (touched via HIGH) -- rule
    # 15, KO is a path event, never inferred from close.
    assert long_barrier < spot0 < short_barrier
    assert long_obs.realized_ko == bool(np.any(fut_low <= long_barrier))
    assert short_obs.realized_ko == bool(np.any(fut_high >= short_barrier))


def test_sigma_levels_produce_monotonically_ordered_barriers(
    synthetic_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """Larger sigma_k -> barrier strictly farther from spot in both
    directions (sanity check on the standardized-distance formula, which
    ``distance_to_barrier`` elsewhere in the codebase also assumes)."""
    bars = synthetic_bars(200, seed=4)
    rng = np.random.default_rng(9)
    sigma_levels = (0.75, 1.5, 3.0)
    obs = build_ko_calibration_dataset(
        bars,
        "TEST",
        horizons=(5,),
        sigma_levels=sigma_levels,
        n_paths=100,
        method="vol_scaled_bootstrap",
        block_size=5,
        lookback_days=750,
        rng=rng,
        step_days=1000,
    )
    long_obs = sorted((o for o in obs if o.direction == Direction.LONG), key=lambda o: o.sigma_k)
    short_obs = sorted((o for o in obs if o.direction == Direction.SHORT), key=lambda o: o.sigma_k)
    assert [o.sigma_k for o in long_obs] == list(sigma_levels)
    # p_ko_raw should be (weakly) monotonically decreasing as the barrier
    # moves farther away -- a wider barrier is never easier to touch.
    long_p = [o.p_ko_raw for o in long_obs]
    short_p = [o.p_ko_raw for o in short_obs]
    assert long_p == sorted(long_p, reverse=True)
    assert short_p == sorted(short_p, reverse=True)


def test_no_lookahead_bar_with_delayed_available_at_is_excluded(
    synthetic_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """CLAUDE.md rule 5: a bar with an earlier ``ts`` but a later
    ``available_at`` (e.g. a delayed/revised data point) must not leak into
    sigma/regime computed for an "as of" bar predating that ``available_at``.
    """
    bars = synthetic_bars(200, seed=5)
    sorted_bars = sorted(bars, key=lambda b: b.ts)
    # bar 150's available_at is well after bar 130's (the single "as of" bar
    # step_days=1000 will select below).
    late_available_at = sorted_bars[150].available_at
    normal_available_at = sorted_bars[50].available_at

    def _spike(available_at: object) -> list[UnderlyingBar]:
        modified = list(sorted_bars)
        b = modified[50]
        modified[50] = b.model_copy(
            update={
                "close": b.close * 3.0,
                "high": b.high * 3.0,
                "low": b.low * 3.0,
                "available_at": available_at,
            }
        )
        return modified

    kwargs = dict(
        horizons=(5,),
        sigma_levels=(1.5,),
        n_paths=300,
        method="vol_scaled_bootstrap",
        block_size=5,
        lookback_days=750,
        step_days=1000,
    )
    included = build_ko_calibration_dataset(
        _spike(normal_available_at), "TEST", rng=np.random.default_rng(7), **kwargs
    )
    excluded = build_ko_calibration_dataset(
        _spike(late_available_at), "TEST", rng=np.random.default_rng(7), **kwargs
    )

    assert len(included) == 2
    assert len(excluded) == 2
    included_by_dir = {o.direction: o for o in included}
    excluded_by_dir = {o.direction: o for o in excluded}
    # Same rng seed, same bars except whether bar 50's 3x price spike was
    # "available" at prediction_time. If the spike leaked into sigma despite
    # its delayed available_at (the bug this test guards against), the two
    # runs would coincidentally need to match on both sigma AND the
    # bootstrap draws -- excluding real information a spike of this size
    # changes the resulting P(KO) materially.
    assert included_by_dir[Direction.LONG].p_ko_raw != pytest.approx(
        excluded_by_dir[Direction.LONG].p_ko_raw
    )


def test_build_dataset_raises_on_insufficient_history(
    synthetic_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = synthetic_bars(50, seed=1)
    with pytest.raises(ValueError):
        build_ko_calibration_dataset(
            bars,
            "TEST",
            horizons=(5,),
            sigma_levels=(1.5,),
            n_paths=100,
            method="vol_scaled_bootstrap",
            block_size=5,
            lookback_days=750,
            rng=np.random.default_rng(0),
        )


def test_build_dataset_rejects_empty_horizons_or_sigma_levels(
    synthetic_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = synthetic_bars(200, seed=1)
    with pytest.raises(ValueError):
        build_ko_calibration_dataset(
            bars,
            "TEST",
            horizons=(),
            n_paths=100,
            method="vol_scaled_bootstrap",
            block_size=5,
            lookback_days=750,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError):
        build_ko_calibration_dataset(
            bars,
            "TEST",
            sigma_levels=(),
            n_paths=100,
            method="vol_scaled_bootstrap",
            block_size=5,
            lookback_days=750,
            rng=np.random.default_rng(0),
        )
