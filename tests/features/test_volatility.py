from __future__ import annotations

import numpy as np
import pytest

from turboedge.features.volatility import (
    downside_semivariance,
    ewma_volatility,
    garman_klass_volatility,
    kurtosis,
    log_returns,
    parkinson_volatility,
    realized_volatility,
    skewness,
    volatility_of_volatility,
)


def test_log_returns_length_and_values() -> None:
    closes = np.array([100.0, 110.0, 99.0])
    rets = log_returns(closes)
    assert rets.shape == (2,)
    assert rets[0] == pytest.approx(np.log(1.1))
    assert rets[1] == pytest.approx(np.log(99.0 / 110.0))


def test_realized_volatility_matches_hand_computation() -> None:
    # 5 bars -> 4 returns; window=3 needs 3 returns to produce a value.
    closes = np.array([100.0, 101.0, 99.0, 103.0, 102.0])
    rv = realized_volatility(closes, window=3)
    assert np.all(np.isnan(rv[:3]))
    rets = log_returns(closes)
    expected_3 = np.std(rets[0:3], ddof=1)
    expected_4 = np.std(rets[1:4], ddof=1)
    assert rv[3] == pytest.approx(expected_3)
    assert rv[4] == pytest.approx(expected_4)


def test_realized_volatility_rejects_small_window() -> None:
    with pytest.raises(ValueError):
        realized_volatility(np.array([1.0, 2.0, 3.0]), window=1)


def test_ewma_volatility_warmup_masks_leading_values() -> None:
    closes = np.exp(np.cumsum(np.full(100, 0.001))) * 100.0
    vol = ewma_volatility(closes, warmup=20)
    # index 0: no return; indices 1..20 (first 20 return-indexed values) masked.
    assert np.all(np.isnan(vol[:21]))
    assert not np.isnan(vol[21])


def test_parkinson_volatility_zero_range_gives_zero_vol() -> None:
    n = 30
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    vol = parkinson_volatility(highs, lows, window=10)
    assert vol[-1] == pytest.approx(0.0)


def test_parkinson_volatility_positive_for_nonzero_range() -> None:
    n = 30
    rng = np.random.default_rng(1)
    lows = 100.0 + rng.uniform(0, 0.1, size=n)
    highs = lows + rng.uniform(0.5, 1.0, size=n)
    vol = parkinson_volatility(highs, lows, window=10)
    assert vol[-1] > 0.0


def test_garman_klass_volatility_zero_for_flat_bars() -> None:
    n = 30
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    vol = garman_klass_volatility(opens, highs, lows, closes, window=10)
    assert vol[-1] == pytest.approx(0.0)


def test_volatility_of_volatility_nonnegative() -> None:
    rng = np.random.default_rng(2)
    n = 200
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    vv = volatility_of_volatility(closes, vol_window=20, vol_of_vol_window=20)
    valid = vv[~np.isnan(vv)]
    assert valid.size > 0
    assert np.all(valid >= 0.0)


def test_skewness_and_kurtosis_shapes() -> None:
    rng = np.random.default_rng(3)
    n = 150
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    skew = skewness(closes, window=63)
    kurt = kurtosis(closes, window=63)
    assert skew.shape == (n,)
    assert kurt.shape == (n,)
    assert np.all(np.isnan(skew[:63]))
    assert not np.isnan(skew[-1])
    assert not np.isnan(kurt[-1])


def test_downside_semivariance_matches_hand_computation() -> None:
    closes = np.array([100.0, 99.0, 100.0, 98.0, 100.0])
    window = 3
    dsv = downside_semivariance(closes, window)
    rets = log_returns(closes)
    # dsv[3] uses rets[0:3] = ln(99/100), ln(100/99), ln(98/100)
    window_rets = rets[0:3]
    expected = np.mean(np.where(window_rets < 0, window_rets**2, 0.0))
    assert dsv[3] == pytest.approx(expected)


def test_no_lookahead_realized_volatility() -> None:
    rng = np.random.default_rng(4)
    n = 100
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    t = 60
    prefix_result = realized_volatility(closes[: t + 1], window=20)[-1]
    future = np.concatenate([closes[: t + 1], np.array([9999.0, 1.0, 5000.0])])
    result_with_future = realized_volatility(future[: t + 1], window=20)[-1]
    assert prefix_result == pytest.approx(result_with_future)
