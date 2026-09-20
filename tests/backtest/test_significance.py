from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from turboedge.backtest.significance import (
    benjamini_hochberg,
    bootstrap_ci,
    bootstrap_p_value,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


def test_psr_reduces_to_normal_z_test_for_normal_returns() -> None:
    # Literature check: for (near-)normal returns, PSR collapses to the plain
    # Z-test Phi((SR_hat - SR*) * sqrt(T-1)) (Bailey & Lopez de Prado 2012).
    rng = np.random.default_rng(0)
    n = 5000
    r = rng.normal(0.001, 0.01, size=n)
    std = np.std(r, ddof=1)
    sr_hat = np.mean(r) / std
    expected = stats.norm.cdf(sr_hat * np.sqrt(n - 1))
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert psr == pytest.approx(expected, abs=1e-3)


def test_psr_is_probability_in_unit_interval() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.02, size=300)
    psr = probabilistic_sharpe_ratio(r)
    assert 0.0 <= psr <= 1.0


def test_psr_higher_for_higher_sharpe() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    low = rng.normal(0.0001, 0.01, size=n)
    high = rng.normal(0.002, 0.01, size=n)
    assert probabilistic_sharpe_ratio(high) > probabilistic_sharpe_ratio(low)


def test_psr_requires_min_observations() -> None:
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(np.array([0.01, 0.02]))


def test_dsr_reduces_to_psr_with_one_trial() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, size=500)
    dsr = deflated_sharpe_ratio(r, n_trials=1)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert dsr == pytest.approx(psr)


def test_dsr_non_increasing_in_number_of_trials() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0.001, 0.01, size=500)
    dsr_1 = deflated_sharpe_ratio(r, n_trials=1)
    dsr_10 = deflated_sharpe_ratio(r, n_trials=10)
    dsr_1000 = deflated_sharpe_ratio(r, n_trials=1000)
    assert dsr_1 >= dsr_10 >= dsr_1000


def test_dsr_in_unit_interval() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.01, size=500)
    dsr = deflated_sharpe_ratio(r, n_trials=50)
    assert 0.0 <= dsr <= 1.0


def test_dsr_rejects_invalid_n_trials() -> None:
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 0.01, size=100)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(r, n_trials=0)


@pytest.mark.parametrize(
    ("label", "returns"),
    [
        # Ten turbo positions that all knocked out, or all closed at the same
        # target: identical returns by construction, and reachable in
        # production once a family has `min_trades_for_comparison` (10) trades.
        ("constant positive", [0.001] * 20),
        ("constant negative", [-1.0] * 20),
        ("all zero", [0.0] * 20),
        # The silent case: differs only in the 10th decimal, raises no
        # RuntimeWarning at all, and used to yield sr_hat = 2.3e7.
        ("near-constant", [0.001, 0.001, 0.001, 0.0010000001] * 5),
    ],
)
def test_psr_and_dsr_return_no_evidence_for_degenerate_series(
    label: str, returns: list[float]
) -> None:
    """A (near-)constant return series must yield 0.5, not near-certainty.

    ``np.std(ddof=1)`` over identical values returns ~2e-19, not 0.0, so the
    former ``if std > 0`` guard never fired. What came out instead was
    ``sr_hat`` of order 1e7..1e15 and ``PSR = 0.9999999999994`` -- past
    `reporting/weekly.py`'s ``ladder_min_psr = 0.95``. These statistics exist
    to stop promotion on noise; being most confident where the data says
    least inverted their purpose.

    0.5 is "undetermined" (observed Sharpe == benchmark), which blocks the
    ladder rather than clearing it.
    """
    r = np.array(returns, dtype=np.float64)
    assert probabilistic_sharpe_ratio(r) == 0.5
    assert deflated_sharpe_ratio(r, n_trials=10) == 0.5


def test_degenerate_series_raises_no_precision_warning() -> None:
    """The visible symptom must be gone, not merely suppressed.

    scipy's "Precision loss occurred in moment calculation due to
    catastrophic cancellation" warning fired on every pytest run of this
    repository via tests/reporting/test_weekly.py. It is gone because the
    degenerate input no longer reaches `stats.skew`/`stats.kurtosis` -- not
    because the warning is filtered anywhere.
    """
    r = np.array([0.001] * 20, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert probabilistic_sharpe_ratio(r) == 0.5
        assert deflated_sharpe_ratio(r, n_trials=10) == 0.5


def test_dispersion_guard_does_not_reject_small_but_real_returns() -> None:
    """The counterpart: the guard is scale-relative, so a genuinely small
    return series is still evaluated normally.

    Without this, an absolute floor could silently turn every low-volatility
    family into "no evidence" -- replacing one wrong answer with another.
    """
    rng = np.random.default_rng(11)
    tiny = rng.normal(1e-6, 1e-5, size=200)  # 0.001% mean, 0.001% sd
    psr = probabilistic_sharpe_ratio(tiny)
    assert psr != 0.5
    assert 0.0 <= psr <= 1.0


def test_benjamini_hochberg_textbook_example() -> None:
    # Classic worked example: 5 p-values, alpha=0.05.
    # sorted: 0.001, 0.008, 0.039, 0.041, 0.042; thresholds: .01,.02,.03,.04,.05
    # p_(1)=.001<=.01 True; p_(2)=.008<=.02 True; p_(3)=.039<=.03 False;
    # p_(4)=.041<=.04 False; p_(5)=.042<=.05 True -> largest k with True is 5
    # (step-up procedure: k is largest index satisfying the inequality, so
    # even though rank 3/4 fail, rank 5 passes and BH rejects ranks 1..5).
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    reject = benjamini_hochberg(p, alpha=0.05)
    assert reject.all()


def test_benjamini_hochberg_no_rejections_when_all_large() -> None:
    p = np.array([0.5, 0.6, 0.7, 0.9])
    reject = benjamini_hochberg(p, alpha=0.05)
    assert not reject.any()


def test_benjamini_hochberg_partial_rejection() -> None:
    # Only the smallest p-value clears its threshold; step-up procedure
    # rejects only the hypotheses at or below that rank.
    p = np.array([0.001, 0.5, 0.6, 0.7])
    reject = benjamini_hochberg(p, alpha=0.05)
    assert reject[0]
    assert not reject[1:].any()


def test_benjamini_hochberg_more_stringent_than_uncorrected() -> None:
    rng = np.random.default_rng(7)
    p = rng.uniform(0.0, 0.1, size=20)
    reject_bh = benjamini_hochberg(p, alpha=0.05)
    reject_uncorrected = p <= 0.05
    assert int(np.sum(reject_bh)) <= int(np.sum(reject_uncorrected))


def test_bootstrap_ci_contains_true_mean_with_known_distribution() -> None:
    rng_data = np.random.default_rng(8)
    values = rng_data.normal(0.0, 1.0, size=2000)
    rng = np.random.default_rng(42)
    lo, hi = bootstrap_ci(values, np.mean, n=2000, rng=rng)
    assert lo < np.mean(values) < hi
    assert lo < hi


def test_bootstrap_ci_deterministic_with_same_seed() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5])
    lo1, hi1 = bootstrap_ci(values, np.mean, n=500, rng=np.random.default_rng(123))
    lo2, hi2 = bootstrap_ci(values, np.mean, n=500, rng=np.random.default_rng(123))
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_ci_different_seeds_can_differ() -> None:
    values = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    lo1, hi1 = bootstrap_ci(values, np.mean, n=200, rng=np.random.default_rng(1))
    lo2, hi2 = bootstrap_ci(values, np.mean, n=200, rng=np.random.default_rng(2))
    assert (lo1, hi1) != (lo2, hi2)


def test_bootstrap_p_value_large_for_no_true_difference() -> None:
    # Paired differences centered at 0 with substantial noise: should not
    # look significant under the null.
    rng_data = np.random.default_rng(9)
    diffs = rng_data.normal(0.0, 1.0, size=500)
    rng = np.random.default_rng(43)
    p = bootstrap_p_value(diffs, np.mean, n=2000, rng=rng)
    assert p > 0.05


def test_bootstrap_p_value_small_for_clear_difference() -> None:
    # Paired differences clearly and consistently negative (e.g. a
    # challenger's per-cell CRPS reliably below the null's): p should be tiny.
    rng_data = np.random.default_rng(10)
    diffs = rng_data.normal(-1.0, 0.1, size=500)
    rng = np.random.default_rng(44)
    p = bootstrap_p_value(diffs, np.mean, n=2000, rng=rng)
    assert p < 0.01


def test_bootstrap_p_value_never_exactly_zero() -> None:
    # Add-one smoothing: even an enormous, unambiguous effect never reports
    # p == 0.0 from a finite number of resamples.
    diffs = np.full(200, -5.0)
    rng = np.random.default_rng(45)
    p = bootstrap_p_value(diffs, np.mean, n=500, rng=rng)
    assert p > 0.0


def test_bootstrap_p_value_deterministic_with_same_seed() -> None:
    values = np.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5])
    p1 = bootstrap_p_value(values, np.mean, n=500, rng=np.random.default_rng(123))
    p2 = bootstrap_p_value(values, np.mean, n=500, rng=np.random.default_rng(123))
    assert p1 == p2


def test_bootstrap_p_value_symmetric_in_sign() -> None:
    # A bootstrap test statistic (mean) should give (approximately) the same
    # p-value whether the effect is positive or negative, for symmetric data.
    rng_data = np.random.default_rng(11)
    diffs = rng_data.normal(0.3, 1.0, size=400)
    p_pos = bootstrap_p_value(diffs, np.mean, n=3000, rng=np.random.default_rng(46))
    p_neg = bootstrap_p_value(-diffs, np.mean, n=3000, rng=np.random.default_rng(46))
    assert p_pos == pytest.approx(p_neg, abs=0.03)
