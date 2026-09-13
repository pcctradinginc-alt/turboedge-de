from __future__ import annotations

import numpy as np
import pytest

from turboedge.ranking.sizing import (
    SizingConfig,
    kelly_fraction_empirical,
    suggested_position_fraction,
)


def test_kelly_fraction_zero_for_all_negative_returns() -> None:
    returns = np.full(500, -0.3)
    assert kelly_fraction_empirical(returns) == 0.0


def test_kelly_fraction_zero_for_empty_sample() -> None:
    assert kelly_fraction_empirical(np.array([])) == 0.0


def test_kelly_fraction_positive_for_favorable_coin_flip() -> None:
    # Classic 60/40 double-or-nothing bet: known full Kelly = 2p - 1 = 0.2.
    n = 200_000
    rng = np.random.default_rng(3)
    wins = rng.random(n) < 0.6
    returns = np.where(wins, 1.0, -1.0)
    f = kelly_fraction_empirical(returns, upper_bound=1.0)
    assert f == pytest.approx(0.2, abs=0.02)


def test_kelly_fraction_never_exceeds_upper_bound() -> None:
    # Huge positive edge, tiny downside -> derivative stays positive across
    # the whole search range -> capped at f_hi.
    returns = np.full(1000, 0.5)
    f = kelly_fraction_empirical(returns, upper_bound=2.0)
    assert f <= 2.0


def test_kelly_fraction_rejects_bad_upper_bound() -> None:
    with pytest.raises(ValueError):
        kelly_fraction_empirical(np.array([0.1, -0.1]), upper_bound=0.0)


def test_suggested_position_fraction_zero_when_all_returns_negative() -> None:
    returns = np.full(300, -0.5)
    f = suggested_position_fraction(returns, p_ko=0.2, uncertainty=0.02)
    assert f == 0.0


def test_suggested_position_fraction_never_exceeds_max_position_fraction() -> None:
    rng = np.random.default_rng(11)
    wins = rng.random(50_000) < 0.9
    returns = np.where(wins, 2.0, -0.9)
    cfg = SizingConfig(max_position_fraction=0.03)
    f = suggested_position_fraction(returns, p_ko=0.01, uncertainty=0.0, cfg=cfg)
    assert 0.0 <= f <= 0.03


def test_suggested_position_fraction_respects_cluster_and_total_risk_caps() -> None:
    rng = np.random.default_rng(11)
    wins = rng.random(50_000) < 0.9
    returns = np.where(wins, 2.0, -0.9)
    cfg = SizingConfig(max_position_fraction=1.0, max_cluster_fraction=0.10, max_total_risk=1.0)
    f = suggested_position_fraction(
        returns, p_ko=0.01, uncertainty=0.0, cluster_fraction_used=0.095, cfg=cfg
    )
    assert f <= 0.10 - 0.095 + 1e-9

    f_full = suggested_position_fraction(
        returns, p_ko=0.01, uncertainty=0.0, cluster_fraction_used=0.10, cfg=cfg
    )
    assert f_full == 0.0


def test_suggested_position_fraction_respects_ko_loss_cap() -> None:
    rng = np.random.default_rng(11)
    wins = rng.random(50_000) < 0.9
    returns = np.where(wins, 2.0, -0.9)
    cfg = SizingConfig(
        max_position_fraction=1.0,
        max_cluster_fraction=1.0,
        max_total_risk=1.0,
        max_ko_loss_contribution=0.01,
    )
    f = suggested_position_fraction(returns, p_ko=0.5, uncertainty=0.0, cfg=cfg)
    assert f * 0.5 <= 0.01 + 1e-9


def test_suggested_position_fraction_decreases_with_uncertainty() -> None:
    rng = np.random.default_rng(11)
    wins = rng.random(50_000) < 0.9
    returns = np.where(wins, 2.0, -0.9)
    cfg = SizingConfig(max_position_fraction=1.0, max_cluster_fraction=1.0, max_total_risk=1.0)
    low_unc = suggested_position_fraction(returns, p_ko=0.0, uncertainty=0.0, cfg=cfg)
    high_unc = suggested_position_fraction(returns, p_ko=0.0, uncertainty=0.5, cfg=cfg)
    assert high_unc < low_unc


def test_suggested_position_fraction_never_negative() -> None:
    f = suggested_position_fraction(np.array([-1.0, -1.0, -1.0]), p_ko=1.0, uncertainty=10.0)
    assert f >= 0.0


def test_suggested_position_fraction_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        suggested_position_fraction(np.array([0.1]), p_ko=1.5, uncertainty=0.0)
    with pytest.raises(ValueError):
        suggested_position_fraction(np.array([0.1]), p_ko=0.5, uncertainty=-0.1)
