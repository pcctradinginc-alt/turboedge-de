from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from turboedge.features.product import ewma_volatility
from turboedge.simulation.bootstrap import (
    RegimeBootstrapConfig,
    _block_indices,
    _regime_bucket_mask,
    _select,
    _standardized_components,
    block_bootstrap_daily,
    regime_bootstrap_daily,
    vol_scaled_bootstrap_daily,
)
from turboedge.simulation.overnight import DailyComponents, daily_components_from_bars
from turboedge.storage.schemas import UnderlyingBar

from .conftest import synthetic_daily_bars


def _components():
    bars = synthetic_daily_bars(400, seed=7)
    return daily_components_from_bars(bars)


def test_block_indices_shape_and_bounds() -> None:
    rng = np.random.default_rng(0)
    idx = _block_indices(n_hist=20, n_slots=14, block_size=5, n_paths=100, rng=rng)
    assert idx.shape == (100, 14)
    assert idx.min() >= 0
    assert idx.max() < 20


def test_block_indices_block_size_one_is_iid() -> None:
    rng = np.random.default_rng(0)
    idx = _block_indices(n_hist=10, n_slots=5, block_size=1, n_paths=3, rng=rng)
    assert idx.shape == (3, 5)


def test_block_indices_preserves_contiguity_within_a_block() -> None:
    rng = np.random.default_rng(0)
    idx = _block_indices(n_hist=1000, n_slots=6, block_size=6, n_paths=50, rng=rng)
    # Within a single block (n_slots == block_size), consecutive picks must be
    # consecutive historical indices (allowing for circular wraparound).
    diffs = np.diff(idx, axis=1) % 1000
    assert np.all(diffs == 1)


def test_block_bootstrap_daily_shapes() -> None:
    components = _components()
    horizon_days = 10
    weekend_before = np.array([False] * horizon_days)
    weekend_before[3] = True
    rng = np.random.default_rng(1)
    draws = block_bootstrap_daily(components, weekend_before, n_paths=64, rng=rng)
    for arr in (draws.gap, draws.intraday, draws.hi, draws.lo):
        assert arr.shape == (64, horizon_days)
        assert np.all(np.isfinite(arr))


def test_block_bootstrap_daily_weekend_positions_drawn_from_weekend_sample() -> None:
    components = _components()
    weekend_before = np.array([True, False, False, False, False])
    rng = np.random.default_rng(2)
    draws = block_bootstrap_daily(components, weekend_before, n_paths=2000, rng=rng)
    weekend_col = draws.gap[:, 0]
    weekday_col = draws.gap[:, 1]
    # Every value drawn into the weekend column must come from the
    # historical weekend gap sample, never the weekday sample (and vice
    # versa for an ordinary weekday column).
    weekend_hist = set(components.gap[components.weekend_before].tolist())
    weekday_hist = set(components.gap[~components.weekend_before].tolist())
    assert set(weekend_col.tolist()) <= weekend_hist
    assert set(weekday_col.tolist()) <= weekday_hist


def test_block_bootstrap_daily_determinism_with_seed() -> None:
    components = _components()
    horizon_days = 8
    weekend_before = np.array([False] * horizon_days)
    weekend_before[2] = True
    rngA = np.random.default_rng(42)
    rngB = np.random.default_rng(42)
    a = block_bootstrap_daily(components, weekend_before, n_paths=32, rng=rngA)
    b = block_bootstrap_daily(components, weekend_before, n_paths=32, rng=rngB)
    assert np.array_equal(a.gap, b.gap)
    assert np.array_equal(a.intraday, b.intraday)


def test_block_bootstrap_daily_raises_when_weekend_sample_empty() -> None:
    # A components series with no weekend days at all (contrived: strip them out).
    components = _components()
    weekday_mask = ~components.weekend_before
    weekday_only = _select(components, weekday_mask)
    weekend_before = np.array([True, False, False])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        block_bootstrap_daily(weekday_only, weekend_before, n_paths=4, rng=rng)


def test_regime_bootstrap_daily_shapes_and_fallback() -> None:
    components = _components()
    horizon_days = 10
    weekend_before = np.array([False] * horizon_days)
    weekend_before[5] = True
    rng = np.random.default_rng(3)
    cfg = RegimeBootstrapConfig(n_vol_buckets=3, min_bucket_days=250)
    draws = regime_bootstrap_daily(
        components, weekend_before, n_paths=64, rng=rng, regime_config=cfg
    )
    for arr in (draws.gap, draws.intraday, draws.hi, draws.lo):
        assert arr.shape == (64, horizon_days)
        assert np.all(np.isfinite(arr))


def test_regime_bootstrap_daily_uses_full_sample_when_bucket_too_small() -> None:
    # Only ~400 days of weekday history -> a 3-bucket split gives ~130 days
    # per bucket, well under the 250-day floor, so the fallback (all
    # weekday days) must kick in; with a huge min_bucket_days this is
    # trivially forced.
    components = _components()
    horizon_days = 6
    weekend_before = np.array([False] * horizon_days)
    rng = np.random.default_rng(4)
    cfg = RegimeBootstrapConfig(n_vol_buckets=3, min_bucket_days=10_000)
    draws = regime_bootstrap_daily(
        components, weekend_before, n_paths=32, rng=rng, regime_config=cfg
    )
    weekday_hist = set(components.gap[~components.weekend_before].tolist())
    assert set(draws.gap.reshape(-1).tolist()) <= weekday_hist


def test_regime_bootstrap_daily_determinism_with_seed() -> None:
    components = _components()
    horizon_days = 6
    weekend_before = np.array([False] * horizon_days)
    rngA = np.random.default_rng(11)
    rngB = np.random.default_rng(11)
    a = regime_bootstrap_daily(components, weekend_before, n_paths=32, rng=rngA)
    b = regime_bootstrap_daily(components, weekend_before, n_paths=32, rng=rngB)
    assert np.array_equal(a.gap, b.gap)


def test_regime_bootstrap_daily_bucket_actually_activates_on_realistic_data() -> None:
    """With production defaults (``min_bucket_days=150``) and a realistic
    ~750-trading-day lookback (``paths.py``'s own default), the regime
    bucket must actually restrict the weekday sample rather than silently
    falling back to "all" -- the previous ``min_bucket_days=250`` default
    did that on 100% of 558 real ^GDAXI dates tested (Build Contract v2 W5
    follow-up 2, ``scratchpad/w5_simulation_validation.md``). This directly
    guards against that regression recurring.
    """
    bars = synthetic_daily_bars(750, seed=13)
    components = daily_components_from_bars(bars)
    weekday_only = _select(components, ~components.weekend_before)
    n_weekday_total = weekday_only.gap.size

    cfg = RegimeBootstrapConfig()  # production defaults
    assert cfg.min_bucket_days < n_weekday_total  # sanity: threshold is achievable at all

    r = weekday_only.gap + weekday_only.intraday
    vol = ewma_volatility(r, cfg.ewma_lambda)
    bucket_mask = _regime_bucket_mask(vol, cfg.n_vol_buckets)
    bucket_size = int(bucket_mask.sum())

    assert bucket_size >= cfg.min_bucket_days, (
        f"regime bucket ({bucket_size} days) failed to clear min_bucket_days "
        f"({cfg.min_bucket_days}) on a realistic {n_weekday_total}-day weekday history -- "
        "activation should not be a coin flip at production settings"
    )
    assert bucket_size < n_weekday_total  # actually restrictive, not the full sample

    # And the public function must actually draw only from that restricted
    # subset, not silently fall back to the full weekday sample.
    horizon_days = 10
    weekend_before = np.array([False] * horizon_days)
    rng = np.random.default_rng(21)
    draws = regime_bootstrap_daily(
        components, weekend_before, n_paths=200, rng=rng, regime_config=cfg
    )
    restricted_weekday_values = set(weekday_only.gap[bucket_mask].tolist())
    drawn_values = set(draws.gap.reshape(-1).tolist())
    assert drawn_values <= restricted_weekday_values


def test_standardized_components_no_lookahead_lag() -> None:
    components = _components()
    standardized, sigma_current = _standardized_components(components, ewma_lambda=0.94)
    r = components.gap + components.intraday
    vol = ewma_volatility(r, 0.94)
    assert sigma_current == pytest.approx(float(vol[-1]))
    # Day 0 has no prior day to lag from -> standardized by its own (only
    # available) vol estimate; every later day must be standardized by the
    # PREVIOUS day's vol, never its own (no look-ahead, CLAUDE.md rule 4/5).
    assert standardized.gap[0] == pytest.approx(components.gap[0] / vol[0])
    assert standardized.gap[5] == pytest.approx(components.gap[5] / vol[4])
    assert standardized.intraday[10] == pytest.approx(components.intraday[10] / vol[9])


def test_standardized_components_rejects_too_short_series() -> None:
    components = _components()
    tiny = DailyComponents(
        dates=components.dates[:1],
        gap=components.gap[:1],
        weekend_before=components.weekend_before[:1],
        intraday=components.intraday[:1],
        hi=components.hi[:1],
        lo=components.lo[:1],
        n_discarded=0,
    )
    with pytest.raises(ValueError):
        _standardized_components(tiny, ewma_lambda=0.94)


def test_vol_scaled_bootstrap_daily_shapes_and_consistency() -> None:
    components = _components()
    horizon_days = 10
    weekend_before = np.array([False] * horizon_days)
    weekend_before[4] = True
    rng = np.random.default_rng(31)
    draws = vol_scaled_bootstrap_daily(components, weekend_before, n_paths=128, rng=rng)
    for arr in (draws.gap, draws.intraday, draws.hi, draws.lo):
        assert arr.shape == (128, horizon_days)
        assert np.all(np.isfinite(arr))
    # Day-level OHLC-consistency invariant must survive the standardize/rescale
    # round trip: hi >= max(intraday, 0), lo <= min(intraday, 0).
    assert np.all(draws.hi >= np.maximum(draws.intraday, 0.0) - 1e-9)
    assert np.all(draws.lo <= np.minimum(draws.intraday, 0.0) + 1e-9)


def test_vol_scaled_bootstrap_daily_determinism_with_seed() -> None:
    components = _components()
    horizon_days = 6
    weekend_before = np.array([False] * horizon_days)
    a = vol_scaled_bootstrap_daily(
        components, weekend_before, n_paths=32, rng=np.random.default_rng(41)
    )
    b = vol_scaled_bootstrap_daily(
        components, weekend_before, n_paths=32, rng=np.random.default_rng(41)
    )
    assert np.array_equal(a.gap, b.gap)
    assert np.array_equal(a.intraday, b.intraday)


def test_vol_scaled_bootstrap_daily_tracks_current_regime_not_blended_history() -> None:
    """The key differentiator vs. block_bootstrap: given a clean historical
    regime shift (high vol for the first 300 days, low vol for the most
    recent 100), the vol-scaled draws' dispersion should track the RECENT
    (current) regime, while a plain block bootstrap blends both.
    """
    rng_hist = np.random.default_rng(0)
    bars: list[UnderlyingBar] = []
    d = date(2023, 1, 2)
    price = 100.0
    for i in range(400):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        vol = 0.03 if i < 300 else 0.003
        ret = rng_hist.normal(0.0, vol)
        o = price
        c = price * float(np.exp(ret))
        hi = max(o, c) * float(np.exp(abs(rng_hist.normal(0.0, vol * 0.3))))
        lo = min(o, c) * float(np.exp(-abs(rng_hist.normal(0.0, vol * 0.3))))
        ts = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
        bars.append(
            UnderlyingBar(
                underlying_id="TEST",
                ts=ts,
                interval="1d",
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=1000.0,
                observation_time=ts,
                available_at=ts,
                retrieved_at=ts,
                source="test",
                parser_version="1",
                quality_score=1.0,
            )
        )
        price = c
        d += timedelta(days=1)

    components = daily_components_from_bars(bars)
    horizon_days = 5
    weekend_before = np.array([False] * horizon_days)

    n_paths = 20_000
    blocked = block_bootstrap_daily(
        components, weekend_before, n_paths, np.random.default_rng(1), block_size=5
    )
    scaled = vol_scaled_bootstrap_daily(
        components, weekend_before, n_paths, np.random.default_rng(1)
    )

    blocked_day0_std = float((blocked.gap[:, 0] + blocked.intraday[:, 0]).std())
    scaled_day0_std = float((scaled.gap[:, 0] + scaled.intraday[:, 0]).std())

    # block_bootstrap blends the whole (mostly high-vol) history -> closer to 0.03.
    # vol_scaled_bootstrap anchors to the current (low-vol) regime -> closer to 0.003.
    assert scaled_day0_std < blocked_day0_std / 3.0
    assert scaled_day0_std < 0.01
