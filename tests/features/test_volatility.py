from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from turboedge.features.volatility import (
    causal_ewma_sigma,
    downside_semivariance,
    ewma_volatility,
    garman_klass_volatility,
    kurtosis,
    log_returns,
    normalized_horizon_target,
    parkinson_volatility,
    realized_volatility,
    skewness,
    volatility_of_volatility,
)
from turboedge.storage.schemas import UnderlyingBar


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


# --- Phase D: causal_ewma_sigma / normalized_horizon_target ------------------------------------


def test_causal_ewma_sigma_matches_ewma_volatility_last_value(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(200, seed=6)
    as_of = bars[-1].available_at
    closes = np.array([b.close for b in bars])
    expected = ewma_volatility(closes)[-1]
    assert causal_ewma_sigma(bars, as_of) == pytest.approx(expected)


def test_causal_ewma_sigma_matches_prefix_when_as_of_is_earlier(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(200, seed=6)
    as_of = bars[120].available_at
    prefix_closes = np.array([b.close for b in bars[:121]])
    expected = ewma_volatility(prefix_closes)[-1]
    assert causal_ewma_sigma(bars, as_of) == pytest.approx(expected)


def test_causal_ewma_sigma_ignores_bars_with_later_available_at(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """A bar whose ``available_at`` is after ``as_of`` must not influence
    ``sigma_t`` even though it is present in the input sequence -- CLAUDE.md
    rule 5 / Phase D's causal-target requirement."""
    bars = make_bars(200, seed=7)
    as_of = bars[120].available_at
    baseline = causal_ewma_sigma(bars, as_of)

    tampered = list(bars)
    for i in range(121, len(tampered)):
        b = tampered[i]
        # Corrupt future closes wildly; since their available_at (== ts) is
        # still after as_of, this must not change the result.
        tampered[i] = b.model_copy(update={"close": b.close * 1000.0})
    tampered_result = causal_ewma_sigma(tampered, as_of)
    assert tampered_result == pytest.approx(baseline)

    # But a bar *reported* as available before as_of (its available_at
    # backdated) DOES legitimately change the result if its close differs --
    # confirming the gate is actually available_at, not ts or list position.
    corrupted_but_eligible = list(bars)
    b = corrupted_but_eligible[100]
    corrupted_but_eligible[100] = b.model_copy(update={"close": b.close * 5.0})
    changed_result = causal_ewma_sigma(corrupted_but_eligible, as_of)
    assert changed_result != pytest.approx(baseline)


def test_normalized_horizon_target_hand_example() -> None:
    closes = np.array([100.0, 101.0, 102.0, 100.0, 99.0])
    sigma = np.array([0.01, 0.01, 0.02, 0.02, np.nan])
    h = 2
    y = normalized_horizon_target(closes, sigma, h)
    # y[0] = ln(closes[2]/closes[0]) / (sigma[0]*sqrt(2))
    expected_0 = np.log(102.0 / 100.0) / (0.01 * np.sqrt(2))
    # y[1] = ln(closes[3]/closes[1]) / (sigma[1]*sqrt(2))
    expected_1 = np.log(100.0 / 101.0) / (0.01 * np.sqrt(2))
    # y[2] = ln(closes[4]/closes[2]) / (sigma[2]*sqrt(2))
    expected_2 = np.log(99.0 / 102.0) / (0.02 * np.sqrt(2))
    assert y[0] == pytest.approx(expected_0)
    assert y[1] == pytest.approx(expected_1)
    assert y[2] == pytest.approx(expected_2)
    # index 3, 4: t + h out of range -> NaN
    assert np.isnan(y[3])
    assert np.isnan(y[4])


def test_normalized_horizon_target_nan_when_sigma_missing_or_nonpositive() -> None:
    closes = np.array([100.0, 101.0, 102.0, 103.0])
    sigma = np.array([np.nan, 0.0, 0.01, 0.02])
    y = normalized_horizon_target(closes, sigma, horizon=1)
    assert np.isnan(y[0])  # sigma NaN
    assert np.isnan(y[1])  # sigma == 0 -- never divide by zero
    assert not np.isnan(y[2])


def test_normalized_horizon_target_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError):
        normalized_horizon_target(np.array([1.0, 2.0]), np.array([0.1, 0.1]), horizon=0)


def test_normalized_horizon_target_no_lookahead(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """Truncating the input to an earlier length never changes an
    already-computed value of the normalized target (same causal-prefix
    guarantee every other feature in this module carries)."""
    bars = make_bars(300, seed=8)
    closes = np.array([b.close for b in bars])
    sigma = ewma_volatility(closes)
    h = 5
    full = normalized_horizon_target(closes, sigma, h)
    t = 150
    prefix_closes = closes[: t + h + 1]
    prefix_sigma = ewma_volatility(prefix_closes)
    prefix = normalized_horizon_target(prefix_closes, prefix_sigma, h)
    assert prefix[t] == pytest.approx(full[t])
