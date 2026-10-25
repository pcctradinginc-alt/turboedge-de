"""Anytime-valid sequential testing, and where its assumptions fail here.

THE PROBLEM IT ADDRESSES
------------------------
`GOVERNANCE.md` fixes one pre-registered hypothesis and one p-value per
trial, run once, precisely because looking at a result repeatedly and
stopping when it looks good destroys a fixed-sample p-value. That rule is
sound and this module does not touch it. But it also means a trial cannot
accumulate evidence as forward observations arrive -- it has to wait for a
pre-declared sample and then look once.

A test martingale removes that constraint **if its assumptions hold**. For
a non-negative process `E_t` with `E[E_t] <= 1` under the null, Ville's
inequality gives

    P(sup_t E_t >= 1/alpha) <= alpha

so "reject when `E_t` reaches `1/alpha`" controls the type-I error at
`alpha` at *any* stopping time, including one chosen by looking at the
data. Optional stopping becomes free.

THE ASSUMPTION, WHICH IS THE WHOLE QUESTION
-------------------------------------------
The betting e-process used here is

    E_t = prod_{i<=t} (1 + lambda_i * X_i),   lambda_i predictable, in [0, c]

Under the null `E[X_i | F_{i-1}] <= 0` each factor has conditional
expectation at most 1, so `E_t` is a non-negative supermartingale and Ville
applies. That is a **conditional** mean condition on the filtration the
observations actually arrive in.

TurboEdge's forward observations do not satisfy it as given. Horizons
overlap -- `docs/measured_results.md` records that at 14 days nine of every
ten forward windows overlap -- so `X_i` shares its outcome window with its
predecessors and is not conditionally centred given the past. Claiming
anytime validity on overlapping observations would be asserting exactly the
premise that fails.

So this module ships the e-process **and** the measurement of what it does
when that premise is violated (`tests/research/test_sequential.py`), rather
than the e-process alone. Deciding it is safe to use would need
non-overlapping blocks, or a correction this module does not claim to
have.

NOT WIRED TO ANYTHING
---------------------
Measurement infrastructure. No confirmatory trial uses it, the quarterly
budget is untouched, and nothing here can promote an alpha.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "SequentialResult",
    "betting_e_process",
    "run_sequential_test",
    "ville_threshold",
]


def ville_threshold(alpha: float) -> float:
    """The e-value at which an anytime-valid test rejects.

    Ville's inequality: a non-negative supermartingale starting at 1
    exceeds `1/alpha` with probability at most `alpha`, whatever the
    stopping rule.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return 1.0 / alpha


@dataclass(frozen=True, slots=True)
class SequentialResult:
    """One run of the sequential test over a stream of observations."""

    e_values: npt.NDArray[np.float64]
    threshold: float
    #: Index of the first observation at which the e-value crossed the
    #: threshold, or `None` if it never did. This is the stopping time, and
    #: it is allowed to depend on the data -- that is the point.
    rejected_at: int | None

    @property
    def rejected(self) -> bool:
        return self.rejected_at is not None

    @property
    def final_e_value(self) -> float:
        return float(self.e_values[-1]) if self.e_values.size else 1.0

    @property
    def peak_e_value(self) -> float:
        return float(np.max(self.e_values)) if self.e_values.size else 1.0


def betting_e_process(
    observations: Sequence[float] | npt.NDArray[np.float64],
    *,
    lambda_cap: float = 0.5,
    warmup: int = 5,
) -> npt.NDArray[np.float64]:
    """The running e-process for `H0: E[X] <= 0` on bounded observations.

    `observations` must lie in `[-1, 1]`; the caller is responsible for the
    scaling, because clipping changes the hypothesis and should be a visible
    decision rather than a hidden one.

    The betting fraction `lambda_i` is **predictable**: it is computed from
    observations strictly before `i`. Using the current observation would
    break the supermartingale property and silently invalidate the whole
    construction, which is the most common way this kind of test goes
    wrong, so it is done in one place and documented here.

    During the first `warmup` observations the bet is zero, so the
    e-process stays at 1 rather than betting on an estimate drawn from one
    or two points.
    """
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"observations must be one-dimensional, got shape {x.shape}")
    if x.size and (np.max(np.abs(x)) > 1.0 + 1e-12):
        raise ValueError(
            "observations must lie in [-1, 1]; scale or clip them explicitly, because "
            "clipping changes the hypothesis being tested"
        )
    if not 0.0 <= lambda_cap <= 1.0:
        raise ValueError(f"lambda_cap must be in [0, 1], got {lambda_cap}")

    e_values = np.empty(x.size, dtype=np.float64)
    log_e = 0.0
    for i in range(x.size):
        past = x[:i]
        if i < warmup or past.size == 0:
            lam = 0.0
        else:
            mean = float(np.mean(past))
            var = float(np.var(past)) + 1e-12
            # The standard plug-in bet: proportional to the estimated edge
            # over its variance, clipped to keep every factor positive.
            lam = float(np.clip(mean / var, 0.0, lambda_cap))
        factor = 1.0 + lam * float(x[i])
        # lam <= 1 and |x| <= 1 keep the factor non-negative; the guard is
        # for floating point at the boundary only.
        log_e += math.log(max(factor, 1e-300))
        e_values[i] = math.exp(min(log_e, 700.0))
    return e_values


def run_sequential_test(
    observations: Sequence[float] | npt.NDArray[np.float64],
    *,
    alpha: float = 0.05,
    lambda_cap: float = 0.5,
    warmup: int = 5,
) -> SequentialResult:
    """Run the e-process and record the first threshold crossing, if any."""
    threshold = ville_threshold(alpha)
    e_values = betting_e_process(observations, lambda_cap=lambda_cap, warmup=warmup)
    crossed = np.flatnonzero(e_values >= threshold)
    return SequentialResult(
        e_values=e_values,
        threshold=threshold,
        rejected_at=int(crossed[0]) if crossed.size else None,
    )
