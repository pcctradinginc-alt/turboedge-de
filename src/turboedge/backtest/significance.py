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


def probabilistic_sharpe_ratio(
    returns: npt.NDArray[np.float64], benchmark_sr: float = 0.0
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
    n = r.shape[0]
    if n < 3:
        raise ValueError(f"returns must have at least 3 observations, got {n}")
    std = float(np.std(r, ddof=1))
    sr_hat = float(np.mean(r) / std) if std > 0 else 0.0
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
    n = r.shape[0]
    if n < 3:
        raise ValueError(f"returns must have at least 3 observations, got {n}")
    std = float(np.std(r, ddof=1))
    sr_hat = float(np.mean(r) / std) if std > 0 else 0.0
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
    return probabilistic_sharpe_ratio(r, benchmark_sr=sr0)


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
