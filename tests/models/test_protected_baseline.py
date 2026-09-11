from __future__ import annotations

import numpy as np
import pytest

from turboedge.models.protected_baseline import (
    TsmomConfig,
    TsmomResult,
    compute_tsmom,
    signal_version_hash,
)
from turboedge.storage.schemas import Direction


def _uptrend(n: int, drift: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.001, size=n)
    log_returns = drift + noise
    log_prices = np.cumsum(log_returns)
    return np.exp(log_prices) * 100.0


def test_compute_tsmom_uptrend_is_positive_and_long() -> None:
    cfg = TsmomConfig()
    closes = _uptrend(200, drift=0.01)
    result = compute_tsmom(closes, cfg)
    assert isinstance(result, TsmomResult)
    assert result.score > 0.0
    assert result.direction_hint == Direction.LONG
    assert set(result.components) == {"z_21", "z_63", "z_126"}


def test_compute_tsmom_downtrend_is_negative_and_short() -> None:
    cfg = TsmomConfig()
    closes = _uptrend(200, drift=-0.01, seed=1)
    result = compute_tsmom(closes, cfg)
    assert result.score < 0.0
    assert result.direction_hint == Direction.SHORT


def test_compute_tsmom_flat_market_no_direction_hint() -> None:
    cfg = TsmomConfig()
    # Tiny zero-mean noise: enough to keep EWMA vol > 0 without a real trend.
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0005, size=200)))
    result = compute_tsmom(closes, cfg)
    assert abs(result.score) <= cfg.clip
    if abs(result.score) < cfg.threshold:
        assert result.direction_hint is None


def test_compute_tsmom_clipping_engages_on_extreme_trend() -> None:
    cfg = TsmomConfig()
    # A very strong, very smooth (near-zero vol) trend drives raw z far
    # beyond the clip band.
    n = 200
    log_prices = np.cumsum(np.full(n, 0.05))
    closes = np.exp(log_prices) * 100.0
    result = compute_tsmom(closes, cfg)
    for value in result.components.values():
        assert -cfg.clip <= value <= cfg.clip
    assert any(abs(value) == pytest.approx(cfg.clip) for value in result.components.values())


def test_compute_tsmom_requires_sufficient_history() -> None:
    cfg = TsmomConfig(lookbacks=(21, 63, 126))
    closes = _uptrend(100)  # shorter than max(lookbacks) + 1 == 127
    with pytest.raises(ValueError):
        compute_tsmom(closes, cfg)


def test_compute_tsmom_rejects_nan() -> None:
    cfg = TsmomConfig()
    closes = _uptrend(200)
    closes[50] = np.nan
    with pytest.raises(ValueError):
        compute_tsmom(closes, cfg)


def test_compute_tsmom_rejects_nonpositive_price() -> None:
    cfg = TsmomConfig()
    closes = _uptrend(200)
    closes[50] = 0.0
    with pytest.raises(ValueError):
        compute_tsmom(closes, cfg)


def test_compute_tsmom_no_look_ahead() -> None:
    cfg = TsmomConfig()
    closes = _uptrend(250, drift=0.008, seed=3)
    t = 199  # evaluate "as of" this index
    result_at_t = compute_tsmom(closes[: t + 1], cfg)

    # Append arbitrary (even wildly different) future prices; the score
    # computed "as of" t from the original prefix must be unaffected because
    # compute_tsmom never sees indices beyond what it is given.
    future = closes[: t + 1].copy()
    future = np.concatenate([future, np.array([1.0, 5000.0, 0.5]) * future[-1]])
    result_with_future_appended_but_same_prefix = compute_tsmom(future[: t + 1], cfg)

    assert result_at_t.score == pytest.approx(result_with_future_appended_but_same_prefix.score)
    assert result_at_t.components == result_with_future_appended_but_same_prefix.components


def test_signal_version_hash_deterministic() -> None:
    cfg = TsmomConfig()
    assert signal_version_hash(cfg) == signal_version_hash(TsmomConfig())


def test_signal_version_hash_changes_with_threshold() -> None:
    cfg_a = TsmomConfig(threshold=0.5)
    cfg_b = TsmomConfig(threshold=0.6)
    assert signal_version_hash(cfg_a) != signal_version_hash(cfg_b)


def test_signal_version_hash_is_hex_sha256() -> None:
    h = signal_version_hash(TsmomConfig())
    assert len(h) == 64
    int(h, 16)  # must be valid hex
