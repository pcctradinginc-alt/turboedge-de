from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest
from scipy import stats

from turboedge.learning.posterior import (
    PosteriorConfig,
    StrategyPosterior,
    bootstrap_prior_from_backtest,
)


def test_prior_matches_configured_prior_variance() -> None:
    cfg = PosteriorConfig(mu0=0.0, kappa0=1.0, alpha0=2.0, prior_variance=0.05**2)
    posterior = StrategyPosterior.new("tsmom", 7, config=cfg)
    assert posterior.posterior_mean == pytest.approx(0.0)
    assert posterior.p_mu_positive() == pytest.approx(0.5, abs=1e-9)
    assert posterior.n == 0.0
    # E[sigma^2] = beta0 / (alpha0 - 1) must equal prior_variance.
    assert posterior.beta / (posterior.alpha - 1.0) == pytest.approx(cfg.prior_variance)


def test_update_matches_analytical_nig_formula() -> None:
    cfg = PosteriorConfig(mu0=0.0, kappa0=1.0, alpha0=2.0, prior_variance=0.05**2)
    posterior = StrategyPosterior.new("tsmom", 7, config=cfg)

    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
    posterior.update(returns)

    n = len(returns)
    xbar = returns.mean()
    ss = float(np.sum((returns - xbar) ** 2))
    kappa0, alpha0, beta0, mu0 = cfg.kappa0, cfg.alpha0, cfg.beta0, cfg.mu0

    expected_kappa = kappa0 + n
    expected_mu = (kappa0 * mu0 + n * xbar) / expected_kappa
    expected_alpha = alpha0 + n / 2
    expected_beta = beta0 + 0.5 * ss + (kappa0 * n * (xbar - mu0) ** 2) / (2 * expected_kappa)

    assert posterior.kappa == pytest.approx(expected_kappa)
    assert posterior.mu0 == pytest.approx(expected_mu)
    assert posterior.alpha == pytest.approx(expected_alpha)
    assert posterior.beta == pytest.approx(expected_beta)
    assert posterior.n == pytest.approx(n)


def test_p_mu_positive_matches_student_t_marginal_directly() -> None:
    cfg = PosteriorConfig()
    posterior = StrategyPosterior.new("tsmom", 7, config=cfg)
    posterior.update([0.02] * 50)

    df = 2.0 * posterior.alpha
    scale = math.sqrt(posterior.beta / (posterior.alpha * posterior.kappa))
    expected = 1.0 - stats.t(df=df, loc=posterior.mu0, scale=scale).cdf(0.0)

    assert posterior.p_mu_positive() == pytest.approx(expected)


def test_positive_returns_increase_p_mu_positive_above_half() -> None:
    posterior = StrategyPosterior.new("tsmom", 7)
    rng = np.random.default_rng(0)
    posterior.update(rng.normal(0.02, 0.05, 300))
    assert posterior.p_mu_positive() > 0.9


def test_negative_returns_decrease_p_mu_positive_below_half() -> None:
    posterior = StrategyPosterior.new("tsmom", 7)
    rng = np.random.default_rng(0)
    posterior.update(rng.normal(-0.02, 0.05, 300))
    assert posterior.p_mu_positive() < 0.1


def test_strategy_posterior_factor_shrinks_toward_half_for_small_n() -> None:
    cfg = PosteriorConfig(shrink_n0=20.0)
    small_n = StrategyPosterior.new("tsmom", 7, config=cfg)
    small_n.update([0.05, 0.05])  # n=2, strong signal but tiny sample

    large_n = StrategyPosterior.new("tsmom", 7, config=cfg)
    large_n.update([0.05] * 200)  # n=200, same strong signal

    factor_small = small_n.strategy_posterior_factor(cfg)
    factor_large = large_n.strategy_posterior_factor(cfg)

    assert 0.5 < factor_small < factor_large <= 1.0
    # With n=0 the factor is exactly 0.5 (no information).
    fresh = StrategyPosterior.new("tsmom", 7, config=cfg)
    assert fresh.strategy_posterior_factor(cfg) == pytest.approx(0.5)


def test_strategy_posterior_factor_always_within_unit_interval() -> None:
    rng = np.random.default_rng(3)
    for n in (0, 1, 5, 50, 5000):
        posterior = StrategyPosterior.new("tsmom", 7)
        if n:
            posterior.update(rng.normal(0.1, 0.2, n))
        factor = posterior.strategy_posterior_factor()
        assert 0.0 <= factor <= 1.0


def test_save_and_load_roundtrip(store) -> None:
    posterior = StrategyPosterior.new("tsmom", 7)
    posterior.update([0.01, 0.02, -0.01])
    posterior.save(store, updated_at=datetime(2026, 9, 10, tzinfo=UTC))

    loaded = StrategyPosterior.load(store, "tsmom", 7)
    assert loaded.mu0 == pytest.approx(posterior.mu0)
    assert loaded.kappa == pytest.approx(posterior.kappa)
    assert loaded.alpha == pytest.approx(posterior.alpha)
    assert loaded.beta == pytest.approx(posterior.beta)
    assert loaded.n == pytest.approx(posterior.n)


def test_load_missing_returns_fresh_prior(store) -> None:
    loaded = StrategyPosterior.load(store, "never_seen", 10)
    assert loaded.n == 0.0
    assert loaded.p_mu_positive() == pytest.approx(0.5, abs=1e-9)


def test_bootstrap_prior_downweights_synthetic_observations() -> None:
    cfg = PosteriorConfig(backtest_downweight=0.1)
    backtest_returns = [0.03] * 100
    posterior = bootstrap_prior_from_backtest("tsmom", 7, backtest_returns, config=cfg)
    assert posterior.n == pytest.approx(100 * 0.1)

    # Compare to an un-downweighted update: the downweighted posterior must
    # carry less evidence (higher posterior_std / less certain p_mu_positive
    # shift), demonstrating it does not dominate as strongly as raw data.
    undamped = StrategyPosterior.new("tsmom", 7, config=cfg)
    undamped.update(backtest_returns)
    assert posterior.posterior_std > undamped.posterior_std
