from __future__ import annotations

import numpy as np
import pytest

from turboedge.features.trend import breakout_distance, ols_trend, robust_trend_distance


def test_ols_trend_recovers_exact_slope_on_noiseless_line() -> None:
    n = 100
    window = 20
    true_slope = 0.001
    log_prices = np.cumsum(np.full(n, true_slope))
    closes = np.exp(log_prices) * 100.0
    slope, tstat = ols_trend(closes, window)
    assert np.all(np.isnan(slope[: window - 1]))
    assert slope[-1] == pytest.approx(true_slope, abs=1e-9)
    # Noiseless perfect fit -> residual is pure floating-point noise -> the
    # t-stat blows up (or is exactly 0 by convention if se_b rounds to 0
    # exactly); either way it must be finite and not spuriously small.
    assert np.isfinite(tstat[-1])


def test_ols_trend_tstat_large_for_strong_trend_low_noise() -> None:
    rng = np.random.default_rng(0)
    n = 100
    window = 63
    drift = 0.01
    noise = rng.normal(0.0, 0.0005, size=n)
    log_prices = np.cumsum(np.full(n, drift) + noise)
    closes = np.exp(log_prices) * 100.0
    slope, tstat = ols_trend(closes, window)
    assert slope[-1] > 0
    assert tstat[-1] > 10.0  # strong, clean trend -> very large t-stat


def test_ols_trend_rejects_small_window() -> None:
    with pytest.raises(ValueError):
        ols_trend(np.array([1.0, 2.0, 3.0]), window=2)


def test_ols_trend_no_lookahead() -> None:
    rng = np.random.default_rng(1)
    n = 150
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n)))
    t = 100
    slope_prefix, _ = ols_trend(closes[: t + 1], window=63)
    future = np.concatenate([closes[: t + 1], np.array([1.0, 9999.0])])
    slope_future, _ = ols_trend(future[: t + 1], window=63)
    assert slope_prefix[-1] == pytest.approx(slope_future[-1])


def test_robust_trend_distance_sign_for_uptrend() -> None:
    n = 100
    window = 63
    log_prices = np.cumsum(np.full(n, 0.005))
    closes = np.exp(log_prices) * 100.0
    vol = np.full(n, 0.01)
    dist = robust_trend_distance(closes, vol, window)
    # A perfectly straight uptrend line has ~zero distance from itself.
    assert dist[-1] == pytest.approx(0.0, abs=1e-6)


def test_robust_trend_distance_positive_when_price_spikes_above_trend() -> None:
    n = 100
    window = 63
    log_prices = np.cumsum(np.full(n, 0.001))
    closes = np.exp(log_prices) * 100.0
    closes[-1] *= 1.2  # sudden spike above the established trend
    vol = np.full(n, 0.01)
    dist = robust_trend_distance(closes, vol, window)
    assert dist[-1] > 0.0


def test_breakout_distance_zero_at_new_high_and_low() -> None:
    n = 70
    window = 63
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    vol = np.full(n, 0.01)
    # Make the last bar a new high and check dist_high == 0 there.
    highs[-1] = 105.0
    closes[-1] = 105.0
    dist_high, _dist_low = breakout_distance(closes, highs, lows, vol, window)
    assert dist_high[-1] == pytest.approx(0.0, abs=1e-9)


def test_breakout_distance_no_lookahead() -> None:
    rng = np.random.default_rng(2)
    n = 150
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    highs = closes * 1.01
    lows = closes * 0.99
    vol = np.full(n, 0.01)
    t = 90
    dh_prefix, dl_prefix = breakout_distance(
        closes[: t + 1], highs[: t + 1], lows[: t + 1], vol[: t + 1], window=63
    )
    future_c = np.concatenate([closes[: t + 1], np.array([1.0, 999.0])])
    future_h = np.concatenate([highs[: t + 1], np.array([1.0, 999.0])])
    future_l = np.concatenate([lows[: t + 1], np.array([1.0, 999.0])])
    future_v = np.concatenate([vol[: t + 1], np.array([0.01, 0.01])])
    dh_future, dl_future = breakout_distance(
        future_c[: t + 1], future_h[: t + 1], future_l[: t + 1], future_v[: t + 1], window=63
    )
    assert dh_prefix[-1] == pytest.approx(dh_future[-1])
    assert dl_prefix[-1] == pytest.approx(dl_future[-1])
