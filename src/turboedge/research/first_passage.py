"""Analytic first-passage probabilities, as an independent reference.

WHAT THIS IS FOR
----------------
`simulation/payoff.py` derives `p_ko` by counting simulated paths whose
daily open or far extreme crossed the barrier
(`simulation/barrier.first_hit_index`). There is no independent check that
the resulting probability is numerically right, because there was nothing
to check it against: the repository had no analytic first-passage formula
on that path at all. `simulation/barrier.brownian_bridge_hit_probability`
exists but is used only by `positions/reevaluate.py` for an open position's
remaining risk, never by the simulation.

This module supplies the missing reference. It is **measurement
infrastructure, not a model**: nothing in the production path imports it,
and it changes no `p_ko` anywhere.

WHAT IT CANNOT DO, STATED FIRST
-------------------------------
An analytic Brownian result validates **numerical correctness under
Brownian assumptions**. It does not validate the production simulation
against real markets, and a disagreement between the two is not
automatically an error in either.

That distinction has teeth here. `simulation/bootstrap.py` samples each
simulated day's `(gap, intraday, hi, lo)` from **real historical days**, so
the daily low that `first_hit_index` tests against the barrier is an
empirical intraday extreme, not a Brownian one. Real ranges are not
Brownian ranges, so the production simulation *should* disagree with these
formulas -- and which one is closer to the truth about a real certificate
is an empirical question these formulas cannot settle.

So this module is used to answer one question exactly: given Brownian or
geometric-Brownian dynamics, is a path-counting estimate of the barrier-hit
probability numerically correct, and how much does discrete monitoring bias
it? Both are answerable to arbitrary precision, and neither is a claim
about markets.

THE MATHEMATICS
---------------
Classical, and deliberately so -- these are the results the 2026
first-passage literature builds on, and they are exactly checkable.

For `X_t = m t + sigma W_t` started at 0 with a barrier at `-a` (`a > 0`),
the first-passage probability over `[0, T]` is

    P(min X <= -a) = Phi((-a - mT) / (sigma sqrt(T)))
                     + exp(-2 m a / sigma^2) Phi((-a + mT) / (sigma sqrt(T)))

which reduces to `2 Phi(-a / (sigma sqrt(T)))` at zero drift -- the
reflection principle. For geometric Brownian motion the same formula
applies to `ln(S_t / S_0)` with `m = mu - sigma^2 / 2` and
`a = ln(S_0 / B)`.

Discrete monitoring at `n` equally spaced times can only ever see less than
the continuous path, so it **underestimates** the hit probability. The size
of that bias is what `discretisation_bias` measures, and the Brownian
bridge is the standard correction for it: between two observed points the
conditional hit probability is `exp(-2 a_0 a_1 / sigma_step^2)`, which is
exactly what `simulation/barrier.brownian_bridge_hit_probability` computes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from turboedge.storage.schemas import Direction

__all__ = [
    "BarrierSpec",
    "bm_hit_probability",
    "bridge_corrected_hit_probability",
    "discrete_monitoring_hit_probability",
    "gbm_hit_probability",
    "intraday_extreme_hit_probability",
]


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    """One barrier problem, in price terms.

    `mu` and `sigma` are per-unit-time log-return drift and volatility, and
    `horizon` is in the same unit. The benchmark uses trading days
    throughout, so `sigma` is a daily log-return volatility and `horizon` a
    number of days.
    """

    spot: float
    barrier: float
    mu: float
    sigma: float
    horizon: float
    direction: Direction

    def __post_init__(self) -> None:
        if not (self.spot > 0 and self.barrier > 0):
            raise ValueError(f"spot and barrier must be > 0, got {self.spot}, {self.barrier}")
        if not self.sigma > 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if not self.horizon > 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")

    @property
    def log_distance(self) -> float:
        """Log-distance to the barrier, positive while it is untouched."""
        if self.direction is Direction.LONG:
            return math.log(self.spot / self.barrier)
        return math.log(self.barrier / self.spot)

    @property
    def log_drift(self) -> float:
        """Drift of the log-price process, signed towards the barrier.

        For a Long certificate the barrier is below, so a positive drift
        moves away from it; for a Short it is above, so the sign flips.
        `mu - sigma^2/2` is the Ito correction that turns a geometric
        Brownian drift into a drift of the log-price.
        """
        m = self.mu - 0.5 * self.sigma**2
        return m if self.direction is Direction.LONG else -m


def bm_hit_probability(
    log_distance: float, log_drift: float, sigma: float, horizon: float
) -> float:
    """P(a Brownian motion with drift reaches `-log_distance` by `horizon`).

    The classical first-passage result. `log_distance > 0` is the starting
    distance from the barrier and `log_drift` is signed so that a positive
    value moves *away* from it.

    Exact, not an approximation: the only error is floating point.
    """
    if not sigma > 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not horizon > 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if log_distance <= 0.0:
        # Already at or past the barrier.
        return 1.0

    s = sigma * math.sqrt(horizon)
    a = log_distance
    m = log_drift
    first = float(norm.cdf((-a - m * horizon) / s))
    # exp(-2 m a / sigma^2) can overflow for a strong away-drift and a close
    # barrier; in that regime the second term's Phi is vanishingly small and
    # the product underflows to zero, so compute it in log space.
    # The second term is exp(-2 m a / sigma^2) * Phi((-a + mT) / s). For a
    # drift towards the barrier the exponential is large while the Phi is
    # tiny, and computing them separately overflows before they cancel.
    # Their product is a probability, so the sum of the logs is <= 0 up to
    # floating point -- adding in log space is both safe and exact.
    log_factor = -2.0 * m * a / sigma**2
    log_tail = float(norm.logcdf((-a + m * horizon) / s))
    second = math.exp(min(log_factor + log_tail, 0.0))
    return float(min(1.0, max(0.0, first + second)))


def gbm_hit_probability(spec: BarrierSpec) -> float:
    """P(geometric Brownian motion touches the barrier by the horizon).

    The same formula applied to `ln(S_t / S_0)`, which is Brownian with
    drift `mu - sigma^2/2`.
    """
    return bm_hit_probability(spec.log_distance, spec.log_drift, spec.sigma, spec.horizon)


def discrete_monitoring_hit_probability(
    spec: BarrierSpec, *, steps: int, paths: int, seed: int
) -> tuple[float, float]:
    """Hit probability seen by monitoring only `steps` equally spaced points.

    Returns `(probability, standard_error)`. This is what a path-counting
    simulation measures when it compares the barrier against observations
    at discrete times rather than the continuous path -- so the gap between
    this and :func:`gbm_hit_probability` is the discretisation bias, and it
    is always one-sided: discrete monitoring can only miss crossings, never
    invent them.

    Gaussian increments on purpose. The production simulator bootstraps
    real historical day shapes, which is a different -- and for real markets
    arguably better -- model; mixing the two would make the comparison
    measure two things at once.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if paths < 1:
        raise ValueError(f"paths must be >= 1, got {paths}")

    rng = np.random.default_rng(seed)
    dt = spec.horizon / steps
    increments = rng.normal(
        loc=spec.log_drift * dt,
        scale=spec.sigma * math.sqrt(dt),
        size=(paths, steps),
    )
    # Distance to the barrier along the path, in log space, starting
    # positive and hitting when it reaches zero.
    level = spec.log_distance + np.cumsum(increments, axis=1)
    hit = np.any(level <= 0.0, axis=1)
    p = float(np.mean(hit))
    se = math.sqrt(max(p * (1.0 - p), 0.0) / paths)
    return p, se


def bridge_corrected_hit_probability(
    spec: BarrierSpec, *, steps: int, paths: int, seed: int
) -> tuple[float, float]:
    """Discrete monitoring plus the Brownian-bridge between-observation term.

    Between two consecutive observations at log-distances `a0` and `a1`, both
    positive, the conditional probability of having touched the barrier in
    between is `exp(-2 a0 a1 / (sigma^2 dt))`. Accumulating the complement
    over the steps of a path gives that path's survival probability, and one
    minus the mean survival is the bridge-corrected hit probability.

    This is the same formula `simulation/barrier`'s
    `brownian_bridge_hit_probability` already implements, applied where the
    production simulation does not apply it: inside the barrier scan.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if paths < 1:
        raise ValueError(f"paths must be >= 1, got {paths}")

    rng = np.random.default_rng(seed)
    dt = spec.horizon / steps
    var_step = spec.sigma**2 * dt
    increments = rng.normal(
        loc=spec.log_drift * dt, scale=spec.sigma * math.sqrt(dt), size=(paths, steps)
    )
    level = spec.log_distance + np.cumsum(increments, axis=1)
    previous = np.concatenate(
        [np.full((paths, 1), spec.log_distance, dtype=np.float64), level[:, :-1]], axis=1
    )

    observed_hit = level <= 0.0
    # Only pairs where both endpoints are still on the safe side carry a
    # bridge term; a pair with an endpoint at or past the barrier is a
    # certain hit and is already counted by `observed_hit`.
    both_safe = (previous > 0.0) & (level > 0.0)
    with np.errstate(over="ignore", under="ignore"):
        bridge = np.where(both_safe, np.exp(-2.0 * previous * level / var_step), 0.0)
    survive_step = np.where(observed_hit, 0.0, 1.0 - bridge)
    survival = np.prod(survive_step, axis=1)
    p = float(np.mean(1.0 - survival))
    # The per-path survival is a product of probabilities rather than a
    # Bernoulli draw, so the standard error is that of the mean of those
    # values, not sqrt(p(1-p)/n).
    se = float(np.std(1.0 - survival, ddof=1) / math.sqrt(paths)) if paths > 1 else 0.0
    return p, se


def intraday_extreme_hit_probability(
    spec: BarrierSpec,
    *,
    steps: int,
    paths: int,
    seed: int,
    substeps: int = 20,
) -> tuple[float, float]:
    """Hit probability when each step's true extreme is observed too.

    This is what the production simulation's barrier scan actually sees:
    `simulation/barrier.first_hit_index` tests each day's open **and** its
    far extreme (low for Long, high for Short), not only the close. So its
    discretisation error is not the close-only error measured by
    :func:`discrete_monitoring_hit_probability` -- it is whatever remains
    once a genuine intraday extreme is visible.

    Modelled by simulating `substeps` Brownian increments inside each step
    and taking the minimum log-distance within the step as that step's
    observed extreme. Under Brownian dynamics that converges to the true
    continuous extreme as `substeps` grows, so the residual gap against
    :func:`gbm_hit_probability` measures the design's remaining bias rather
    than the sampling resolution.
    """
    if steps < 1 or paths < 1 or substeps < 1:
        raise ValueError("steps, paths and substeps must all be >= 1")

    rng = np.random.default_rng(seed)
    dt = spec.horizon / (steps * substeps)
    increments = rng.normal(
        loc=spec.log_drift * dt,
        scale=spec.sigma * math.sqrt(dt),
        size=(paths, steps * substeps),
    )
    level = spec.log_distance + np.cumsum(increments, axis=1)
    # Per step, the lowest log-distance reached inside it.
    per_step_min = level.reshape(paths, steps, substeps).min(axis=2)
    hit = np.any(per_step_min <= 0.0, axis=1)
    p = float(np.mean(hit))
    se = math.sqrt(max(p * (1.0 - p), 0.0) / paths)
    return p, se


def discretisation_bias(spec: BarrierSpec, *, steps: int, paths: int, seed: int) -> float:
    """How much discrete monitoring understates the hit probability.

    Always non-negative in expectation, because observing fewer points can
    only miss crossings.
    """
    discrete, _ = discrete_monitoring_hit_probability(spec, steps=steps, paths=paths, seed=seed)
    return gbm_hit_probability(spec) - discrete


def hit_probability_grid(
    specs: list[BarrierSpec], *, steps: int, paths: int, seed: int
) -> npt.NDArray[np.float64]:
    """Analytic, discrete and bridge-corrected probabilities for each spec.

    Returns an array of shape `(len(specs), 4)`: analytic, discrete,
    bridge-corrected, and the discrete estimate's standard error. One call
    so a benchmark table is one reproducible computation.
    """
    out = np.empty((len(specs), 4), dtype=np.float64)
    for i, spec in enumerate(specs):
        analytic = gbm_hit_probability(spec)
        discrete, se = discrete_monitoring_hit_probability(
            spec, steps=steps, paths=paths, seed=seed + i
        )
        bridged, _ = bridge_corrected_hit_probability(spec, steps=steps, paths=paths, seed=seed + i)
        out[i] = (analytic, discrete, bridged, se)
    return out
