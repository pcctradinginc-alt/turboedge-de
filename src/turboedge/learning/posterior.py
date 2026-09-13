"""Bayesian strategy memory: Normal-Inverse-Gamma posterior over a strategy
family's expected net return, per horizon (Master Spec §21).

Model, for one ``(signal_family, horizon_days)``::

    mu, sigma^2 ~ NormalInverseGamma(mu0, kappa, alpha, beta)
    r_i | mu, sigma^2 ~ Normal(mu, sigma^2)   (net returns of realized trades)

This is the standard conjugate Normal/Inverse-Gamma pair (e.g. Murphy,
"Conjugate Bayesian analysis of the Gaussian distribution"): after observing
a batch of returns, the parameters update in closed form, and the marginal
posterior over ``mu`` alone (integrating out ``sigma^2``) is a
non-standardized Student-t distribution, which is what
:meth:`StrategyPosterior.p_mu_positive` uses to compute
``P(mu_strategy > 0 | data)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from turboedge.storage.duckdb import Store


class PosteriorConfig(BaseModel):
    """This module's own config (wired into config.py/YAML by the
    integration wave). Defaults describe a deliberately weak, wide prior
    (Build Contract v2 W6 requirement 5): centered at zero net return, one
    "virtual" prior observation (``kappa0=1``), with prior variance
    ``0.05**2`` (5% net-return standard deviation) driving ``beta0`` so
    ``E[sigma^2] = beta0/(alpha0-1) == prior_variance``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mu0: float = 0.0
    kappa0: float = Field(default=1.0, gt=0)
    alpha0: float = Field(default=2.0, gt=1.0)  # >1 so E[sigma^2] is defined
    prior_variance: float = Field(default=0.05**2, gt=0)
    # Shrinkage denominator for `strategy_posterior_factor`: at n == shrink_n0
    # posterior evidence and the neutral prior (0.5) are weighted equally;
    # as n -> 0 the factor -> 0.5 (no information), as n -> inf it -> p_mu_positive().
    shrink_n0: float = Field(default=20.0, gt=0)
    # Down-weighting applied when bootstrapping a prior from synthetic
    # backtest returns (Master Spec §29 "Bootstrap-Phase"): each synthetic
    # observation contributes this fraction of a real observation's weight
    # to the sufficient statistics, so real forward-ledger data collected
    # later is not dominated by potentially-biased synthetic simulation
    # (Spec §29 "Real-Product-Phase": "Produktselektionsmodelle sollen
    # bevorzugt auf echten Forward-Produktdaten lernen.").
    backtest_downweight: float = Field(default=0.1, gt=0.0, le=1.0)

    @property
    def beta0(self) -> float:
        return self.prior_variance * (self.alpha0 - 1.0)


@dataclass
class StrategyPosterior:
    """Normal-Inverse-Gamma posterior over one strategy family's expected
    net return at one horizon.

    ``mu0``/``kappa``/``alpha``/``beta`` are the *current* NIG hyper-
    parameters (initialized to the prior, then updated in place by
    :meth:`update`) -- the same four numbers persisted verbatim in the
    ``strategy_posteriors`` table.
    """

    signal_family: str
    horizon_days: int
    mu0: float
    kappa: float
    alpha: float
    beta: float
    n: float = 0.0

    @classmethod
    def new(
        cls,
        signal_family: str,
        horizon_days: int,
        *,
        config: PosteriorConfig | None = None,
    ) -> StrategyPosterior:
        """A fresh posterior at its prior (no observations yet)."""
        cfg = config if config is not None else PosteriorConfig()
        return cls(
            signal_family=signal_family,
            horizon_days=horizon_days,
            mu0=cfg.mu0,
            kappa=cfg.kappa0,
            alpha=cfg.alpha0,
            beta=cfg.beta0,
            n=0.0,
        )

    def update(self, returns: Sequence[float] | npt.NDArray[np.float64]) -> StrategyPosterior:
        """Conjugate NIG update given a batch of realized net returns.

        Mutates and returns ``self`` (chainable). A batch of zero returns is
        a no-op. Standard sufficient-statistics update (see module
        docstring); every call is a full closed-form batch update, so
        calling it once with N returns is equivalent to calling it N times
        with one return each.
        """
        x = np.asarray(returns, dtype=np.float64)
        m = x.size
        if m == 0:
            return self
        self._absorb(m, float(x.mean()), float(np.sum((x - x.mean()) ** 2)))
        return self

    def _absorb(self, m: float, xbar: float, sum_sq: float) -> None:
        """Fold ``m`` (possibly fractional, for down-weighted bootstrap
        observations) effective observations with mean ``xbar`` and sum of
        squared deviations ``sum_sq`` into the current posterior, in place."""
        if m <= 0:
            return
        kappa_n = self.kappa + m
        mu_n = (self.kappa * self.mu0 + m * xbar) / kappa_n
        alpha_n = self.alpha + m / 2.0
        beta_n = (
            self.beta + 0.5 * sum_sq + (self.kappa * m * (xbar - self.mu0) ** 2) / (2.0 * kappa_n)
        )
        self.mu0 = mu_n
        self.kappa = kappa_n
        self.alpha = alpha_n
        self.beta = beta_n
        self.n += m

    def _mu_marginal(self) -> Any:
        """Marginal posterior of ``mu`` (sigma^2 integrated out): a
        Student-t distribution with ``df = 2*alpha``, location ``mu0`` and
        scale ``sqrt(beta / (alpha * kappa))``."""
        df = 2.0 * self.alpha
        scale = math.sqrt(self.beta / (self.alpha * self.kappa))
        return stats.t(df=df, loc=self.mu0, scale=scale)

    def p_mu_positive(self) -> float:
        """``P(mu_strategy > 0 | data)`` via the Student-t marginal of mu."""
        return float(1.0 - self._mu_marginal().cdf(0.0))

    @property
    def posterior_mean(self) -> float:
        """Mean of the marginal posterior over ``mu`` (equals ``mu0``, the
        current NIG location parameter)."""
        return self.mu0

    @property
    def posterior_std(self) -> float:
        """Std of the marginal posterior over ``mu``. Only finite for
        ``alpha > 1`` (``df = 2*alpha > 2``); returns ``inf`` otherwise, as
        the Student-t variance is undefined for ``df <= 2``."""
        std = float(self._mu_marginal().std())
        return std if math.isfinite(std) else float("inf")

    def strategy_posterior_factor(self, config: PosteriorConfig | None = None) -> float:
        """Ensemble-weighting factor in ``[0, 1]``: ``P(mu > 0)`` shrunk
        toward the neutral prior 0.5 when the effective sample size ``n``
        is small, so a strategy with little forward evidence does not swing
        ensemble weights on noise (Master Spec §9.3, "Kleine effektive
        Stichprobe respektieren").

        ``factor = 0.5 + (p_mu_positive() - 0.5) * n / (n + shrink_n0)`` --
        a convex combination of 0.5 and ``p_mu_positive()``, so it is always
        within ``[0, 1]`` without needing an explicit clamp.
        """
        cfg = config if config is not None else PosteriorConfig()
        weight = self.n / (self.n + cfg.shrink_n0)
        factor = 0.5 + (self.p_mu_positive() - 0.5) * weight
        return factor

    # -- persistence ----------------------------------------------------------

    def save(self, store: Store, *, updated_at: datetime | None = None) -> None:
        """Upsert this posterior's current state into ``strategy_posteriors``."""
        store.upsert_strategy_posterior(
            self.signal_family,
            self.horizon_days,
            mu0=self.mu0,
            kappa=self.kappa,
            alpha=self.alpha,
            beta=self.beta,
            n=self.n,
            updated_at=updated_at if updated_at is not None else datetime.now(UTC),
        )

    @classmethod
    def load(
        cls,
        store: Store,
        signal_family: str,
        horizon_days: int,
        *,
        config: PosteriorConfig | None = None,
    ) -> StrategyPosterior:
        """Load a persisted posterior, or a fresh prior if none exists yet."""
        row = store.get_strategy_posterior(signal_family, horizon_days)
        if row is None:
            return cls.new(signal_family, horizon_days, config=config)
        mu0, kappa, alpha, beta, n, _updated_at = row
        return cls(
            signal_family=signal_family,
            horizon_days=horizon_days,
            mu0=mu0,
            kappa=kappa,
            alpha=alpha,
            beta=beta,
            n=n,
        )


def bootstrap_prior_from_backtest(
    signal_family: str,
    horizon_days: int,
    backtest_returns: Sequence[float] | npt.NDArray[np.float64],
    *,
    config: PosteriorConfig | None = None,
) -> StrategyPosterior:
    """Build an initial :class:`StrategyPosterior` from synthetic backtest
    returns (Master Spec §29 "Bootstrap-Phase"), down-weighted by
    ``config.backtest_downweight`` so it never dominates real forward-ledger
    data collected later (§29 "Real-Product-Phase").

    Each synthetic return contributes ``backtest_downweight`` of one real
    observation's weight to the NIG sufficient statistics (equivalent to
    scaling ``m`` in the conjugate update, not to literally duplicating or
    dropping rows), so the resulting posterior's effective sample size
    ``n`` is ``len(backtest_returns) * backtest_downweight`` rather than
    the raw count.
    """
    cfg = config if config is not None else PosteriorConfig()
    posterior = StrategyPosterior.new(signal_family, horizon_days, config=cfg)
    x = np.asarray(list(backtest_returns), dtype=np.float64)
    if x.size == 0:
        return posterior
    xbar = float(x.mean())
    sum_sq = float(np.sum((x - xbar) ** 2))
    m_eff = x.size * cfg.backtest_downweight
    # Sum of squared deviations scales with the down-weighted observation
    # count too, so the effective variance estimate is unaffected by the
    # down-weighting (only the *evidence weight* -- kappa/alpha growth -- is
    # reduced).
    posterior._absorb(m_eff, xbar, sum_sq * cfg.backtest_downweight)
    return posterior


__all__ = [
    "PosteriorConfig",
    "StrategyPosterior",
    "bootstrap_prior_from_backtest",
]
