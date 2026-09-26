"""Multiple-testing-aware significance statistics.

Formula reference: Master Spec §27.4 ("Multiple Testing"); Bailey & Lopez de
Prado, "The Sharpe Ratio Efficient Frontier" (2012) for the Probabilistic
Sharpe Ratio, and "The Deflated Sharpe Ratio" (2014) for the Deflated Sharpe
Ratio; Benjamini & Hochberg (1995) for the FDR procedure.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015329

#: Smallest ``std / max|r|`` a return series must have before its Sharpe
#: ratio -- and therefore PSR/DSR -- carries any information.
#:
#: Measured 2026-09-20. ``np.std(ddof=1)`` over 20 identical values does not
#: return 0.0 but 2.2e-19 (floating-point cancellation), so the natural-
#: looking guard ``if std > 0`` never fires for a degenerate series. What got
#: through instead: ``sr_hat = 4.5e15``, ``skew``/``kurtosis`` both ``nan``,
#: and scipy's "Precision loss occurred in moment calculation due to
#: catastrophic cancellation" RuntimeWarning (visible in every pytest run of
#: this repository, via tests/reporting/test_weekly.py).
#:
#: The silent variant is worse than the loud one: a *nearly* identical series
#: (20 values differing in the 10th decimal) raises no warning at all, yet
#: yields ``sr_hat = 2.3e7`` and ``PSR = 0.9999999999994`` -- comfortably past
#: `reporting/weekly.py`'s ``ladder_min_psr = 0.95`` promotion gate. That is
#: not a rounding artifact but a governance hole: PSR/DSR exist to stop a
#: model being promoted on noise, and they were at their most confident
#: exactly where the data says least.
#:
#: This is reachable in production, not only in tests: `weekly.py` computes
#: PSR from realized forward-ledger returns once a family has
#: ``min_trades_for_comparison`` (10) trades, and ten turbo positions that all
#: knocked out -- or all closed at the same target -- have identical returns
#: by construction.
#:
#: 1e-6 sits far below anything real (a genuine return series has
#: ``std / max|r|`` of order 1e-1..1) and far above float64 cancellation noise
#: (~1e-16 relative). A series under it implies a Sharpe ratio above ~1e6,
#: which is never a measurement.
_MIN_RELATIVE_DISPERSION = 1e-6


def _has_usable_dispersion(r: npt.NDArray[np.float64]) -> bool:
    """Whether ``r`` varies enough for a Sharpe-based statistic to mean anything.

    Scale-relative on purpose: an absolute floor would reject legitimately
    small returns (a 0.01% daily series is not degenerate) while still
    accepting a degenerate series denominated in larger numbers.
    """
    if r.size < 2:
        return False
    scale = float(np.max(np.abs(r)))
    if not np.isfinite(scale) or scale == 0.0:
        # All-zero (or non-finite) returns: no dispersion, and no scale to
        # measure dispersion against.
        return False
    std = float(np.std(r, ddof=1))
    return np.isfinite(std) and std > _MIN_RELATIVE_DISPERSION * scale


def probabilistic_sharpe_ratio(
    returns: npt.NDArray[np.float64],
    benchmark_sr: float = 0.0,
    *,
    n_effective: float | None = None,
) -> float:
    """P(true Sharpe ratio > benchmark_sr), correcting for skew/kurtosis of ``returns``.

    Bailey & Lopez de Prado (2012):

        PSR = Phi((SR_hat - SR*) * sqrt(T-1) / sqrt(1 - g3*SR_hat + (g4-1)/4*SR_hat^2))

    where ``SR_hat`` is the (per-period, not annualized) sample Sharpe
    ratio, ``T`` the number of observations, ``g3`` the sample skewness and
    ``g4`` the sample kurtosis (non-excess; normal returns => 3). For
    (approximately) normal returns this reduces to the plain Z-test
    ``Phi((SR_hat - SR*) * sqrt(T-1))``.
    """
    r = np.asarray(returns, dtype=np.float64)
    # `n_effective` overrides the row count as T. Overlapping label windows
    # mean a row count is not an observation count: the forward ledger's 2,672
    # entries are worth 1.00 independent observations under average uniqueness
    # (docs/measured_results.md §6.9), and feeding 2,672 into sqrt(T-1) is what
    # let the 2026-09-26 tournament report same-day positions as significant.
    # Callers that know the effective sample pass it here rather than reshaping
    # the returns array to fake a length, which distorts the dispersion the
    # skew/kurtosis correction reads.
    n = r.shape[0] if n_effective is None else n_effective
    if n < 3:
        raise ValueError(f"returns must have at least 3 observations, got {n}")
    if not _has_usable_dispersion(r):
        # A (near-)constant series carries no information about the true
        # Sharpe ratio, so the honest answer is "undetermined" -- 0.5, the
        # value PSR takes when the observed and benchmark Sharpe coincide.
        # Deliberately not 1.0: the formula's limit for constant positive
        # returns is certainty, and reporting certainty from ten identical
        # knock-outs would let `weekly.py`'s ladder promote on noise (see
        # `_MIN_RELATIVE_DISPERSION`). Returning early also keeps the
        # degenerate input away from `stats.skew`/`stats.kurtosis`, whose
        # catastrophic-cancellation RuntimeWarning was the visible symptom
        # of this.
        return 0.5
    std = float(np.std(r, ddof=1))
    sr_hat = float(np.mean(r) / std)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # non-excess (normal == 3)
    denom_inner = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    denom = float(np.sqrt(max(denom_inner, 1e-12)))
    z = (sr_hat - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: npt.NDArray[np.float64],
    n_trials: int,
    *,
    skew: float | None = None,
    kurtosis: float | None = None,
    n_effective: float | None = None,
) -> float:
    """Probabilistic Sharpe Ratio benchmarked against the expected max Sharpe of ``n_trials`` runs.

    Bailey & Lopez de Prado (2014): with ``n_trials`` independent strategy
    trials of true zero skill, the *best* observed in-sample Sharpe ratio is
    expected (under normal-returns asymptotics) to be

        SR0 = sigma_SR * ((1-gamma)*Phi^-1(1 - 1/n_trials) + gamma*Phi^-1(1 - 1/(n_trials*e)))

    with ``gamma`` the Euler-Mascheroni constant and ``sigma_SR`` the
    standard error of the Sharpe-ratio estimator implied by this return
    series' own skew/kurtosis (Bailey & Lopez de Prado 2012, eq. 5); DSR is
    then the PSR of ``returns`` benchmarked against ``SR0`` instead of 0.
    With ``n_trials == 1``, ``SR0 == 0`` and DSR reduces exactly to PSR(0).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}")
    r = np.asarray(returns, dtype=np.float64)
    # See `probabilistic_sharpe_ratio` -- a row count is not an
    # observation count when label windows overlap.
    n = r.shape[0] if n_effective is None else n_effective
    if n < 3:
        raise ValueError(f"returns must have at least 3 observations, got {n}")
    if not _has_usable_dispersion(r):
        # Same reasoning as `probabilistic_sharpe_ratio`, and the guard must
        # live here too rather than being inherited from the PSR call at the
        # end: `sr_std` below is built from this series' own skew/kurtosis,
        # so a degenerate input would already have produced a `nan` `sr0`
        # benchmark before PSR ever saw it.
        return 0.5
    std = float(np.std(r, ddof=1))
    sr_hat = float(np.mean(r) / std)
    skew_ = float(stats.skew(r, bias=False)) if skew is None else skew
    kurt_ = float(stats.kurtosis(r, fisher=False, bias=False)) if kurtosis is None else kurtosis
    sr_std = float(
        np.sqrt(max(1.0 - skew_ * sr_hat + (kurt_ - 1.0) / 4.0 * sr_hat**2, 0.0) / (n - 1))
    )
    if n_trials == 1:
        sr0 = 0.0
    else:
        z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
        sr0 = sr_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)
    return probabilistic_sharpe_ratio(r, benchmark_sr=sr0, n_effective=n_effective)


def benjamini_hochberg(
    p_values: npt.NDArray[np.float64], alpha: float = 0.05
) -> npt.NDArray[np.bool_]:
    """Benjamini-Hochberg (1995) step-up FDR procedure; ``True`` where the null is rejected.

    Sorts p-values ascending, finds the largest rank ``k`` with
    ``p_(k) <= (k/m) * alpha``, and rejects all hypotheses ranked ``<= k``
    (ties in ``p`` broken by original index, deterministic). Output is
    realigned to the input's original order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-dimensional array")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must be within [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    thresholds = (ranks / m) * alpha
    passed = sorted_p <= thresholds
    reject_sorted = np.zeros(m, dtype=np.bool_)
    if np.any(passed):
        k = int(np.max(np.nonzero(passed)[0])) + 1  # largest rank satisfying the condition
        reject_sorted[:k] = True
    reject = np.zeros(m, dtype=np.bool_)
    reject[order] = reject_sorted
    return reject


def bootstrap_ci(
    values: npt.NDArray[np.float64],
    stat: Callable[[npt.NDArray[np.float64]], float],
    n: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap ``(1 - alpha)`` confidence interval for ``stat(values)``.

    Resamples ``values`` with replacement ``n`` times (via the caller-supplied,
    seedable ``rng`` -- CLAUDE.md rule "Determinismus"), applies ``stat`` to
    each resample, and returns the ``(alpha/2, 1 - alpha/2)`` percentiles of
    the resulting distribution.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("values must not be empty")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    stats_boot = np.empty(n, dtype=np.float64)
    size = v.shape[0]
    for i in range(n):
        sample = v[rng.integers(0, size, size=size)]
        stats_boot[i] = stat(sample)
    lo = float(np.percentile(stats_boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(stats_boot, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_p_value(
    values: npt.NDArray[np.float64],
    stat: Callable[[npt.NDArray[np.float64]], float],
    n: int,
    rng: np.random.Generator,
    *,
    null_value: float = 0.0,
) -> float:
    """Two-sided bootstrap hypothesis-test p-value for ``H0: stat(population) == null_value``.

    Standard bootstrap hypothesis test (Efron & Tibshirani 1993, §16.4): the
    observed sample is re-centered so the null holds exactly in the
    resampling population (``values - stat(values) + null_value``), then
    resampled with replacement ``n`` times via the same caller-supplied,
    seedable ``rng`` pattern as :func:`bootstrap_ci`. The p-value is the
    fraction of those null-world bootstrap statistics at least as extreme as
    the actually observed one, with add-one (Laplace) smoothing so a p-value
    of exactly 0 is never reported from a finite number of resamples.

    Used for e.g. a per-cell CRPS-difference test against a null model: pass
    ``values`` as the *paired* per-observation ``crps_challenger -
    crps_null`` differences and ``stat=np.mean``; a p-value near 0 means the
    observed mean difference is unlikely under "no true difference".
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("values must not be empty")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")

    observed = float(stat(v))
    centered = v - observed + null_value
    size = v.shape[0]
    stats_boot = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample = centered[rng.integers(0, size, size=size)]
        stats_boot[i] = stat(sample)
    extreme = np.sum(np.abs(stats_boot - null_value) >= abs(observed - null_value))
    return float((extreme + 1) / (n + 1))
