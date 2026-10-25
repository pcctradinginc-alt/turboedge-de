"""The e-process, and the measurement that decides against using it here.

Ville's inequality makes optional stopping free **if** the e-process is a
non-negative supermartingale under the null, which needs
`E[X_i | F_{i-1}] <= 0` -- a *conditional* mean condition on the filtration
the observations arrive in.

TurboEdge's forward observations do not satisfy it. Horizons overlap, and
`docs/measured_results.md` records that at 14 days nine of every ten
windows do. These tests measure what that costs rather than assuming it is
a detail: the false-positive rate goes from 1.4% (independent, assumption
holds) to 49% (overlapping 14-day windows) against a nominal 5%.

They also establish that the implementation itself is sound -- the power
curve rises with sample size and the error rate stays controlled under
independence -- so the failure above is a property of the data model, not
of the code. That distinction is the whole point: a broken implementation
and an inapplicable method look identical from a single number.

Seeds are fixed and the repetition counts are small enough to run in CI.
The headline figures in `docs/mathematical_validation_2026.md` come from
larger runs; these assert the direction and magnitude, not the third
decimal.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboedge.research.sequential import (
    betting_e_process,
    run_sequential_test,
    ville_threshold,
)

ALPHA = 0.05
_TRIALS = 400


def _iid(rng: np.random.Generator, n: int, mean: float = 0.0, sd: float = 0.25) -> np.ndarray:
    return np.clip(rng.normal(mean, sd, n), -1.0, 1.0)


def _overlapping(
    rng: np.random.Generator, n: int, *, horizon: int = 14, mean: float = 0.0
) -> np.ndarray:
    """`n` overlapping `horizon`-day windows over one stream of daily shocks.

    Consecutive windows share `horizon - 1` of their shocks, which is
    exactly the dependence the forward ledger has.
    """
    daily = rng.normal(mean / horizon, 0.25 / np.sqrt(horizon), n + horizon)
    return np.clip(np.convolve(daily, np.ones(horizon), mode="valid")[:n], -1.0, 1.0)


def _rejection_rate(make: object, *, trials: int = _TRIALS, offset: int = 0) -> float:
    rejected = 0
    for seed in range(trials):
        x = make(np.random.default_rng(offset + seed))  # type: ignore[operator]
        if run_sequential_test(x, alpha=ALPHA).rejected:
            rejected += 1
    return rejected / trials


# -- the construction ----------------------------------------------------


def test_the_threshold_is_the_reciprocal_of_alpha() -> None:
    assert ville_threshold(0.05) == pytest.approx(20.0)
    assert ville_threshold(0.01) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        ville_threshold(0.0)
    with pytest.raises(ValueError):
        ville_threshold(1.0)


def test_the_e_process_starts_at_one_and_stays_non_negative() -> None:
    e = betting_e_process(_iid(np.random.default_rng(1), 200))

    assert e.size == 200
    assert np.all(e >= 0.0)
    # During warmup the bet is zero, so no evidence accrues.
    assert e[0] == pytest.approx(1.0)
    assert e[4] == pytest.approx(1.0)


def test_the_bet_uses_only_past_observations() -> None:
    # Using the current observation to size the bet would break the
    # supermartingale property and silently invalidate everything. Checked
    # behaviourally: changing observation `k` must not alter any e-value
    # before `k`.
    # A positive-drift series, so the bet is actually non-zero and the test
    # is not vacuous: with a negative past mean the bet is clipped to 0 and
    # the current observation cannot move the e-value at all.
    x = _iid(np.random.default_rng(2), 60, mean=0.2)
    baseline = betting_e_process(x)
    perturbed = x.copy()
    perturbed[30] = -1.0
    changed = betting_e_process(perturbed)

    assert np.allclose(baseline[:30], changed[:30]), (
        "an e-value changed before the observation that was perturbed, so the bet "
        "is not predictable and the supermartingale property does not hold"
    )
    assert not np.isclose(baseline[30], changed[30]), (
        "the perturbed observation did not move its own e-value, so this test "
        "would pass even if the bet were not predictable"
    )


def test_observations_outside_the_unit_interval_are_refused() -> None:
    # Clipping changes the hypothesis, so it must be the caller's visible
    # decision rather than a silent rescale here.
    with pytest.raises(ValueError, match="changes the hypothesis"):
        betting_e_process(np.array([0.0, 2.0, 0.0]))


# -- validity where the assumption holds ---------------------------------


def test_the_error_rate_is_controlled_on_independent_observations() -> None:
    rate = _rejection_rate(lambda rng: _iid(rng, 300), offset=10_000)

    assert rate <= ALPHA, f"nominal {ALPHA}, measured {rate}"


def test_the_power_rises_with_the_sample_size() -> None:
    # Establishes that the implementation works, so that the zero power on
    # blocked data below is a statement about sample size and not about the
    # code.
    small = _rejection_rate(lambda rng: _iid(rng, 30, mean=0.10), offset=20_000)
    large = _rejection_rate(lambda rng: _iid(rng, 300, mean=0.10), offset=30_000)

    assert small < 0.2
    assert large > 0.9


# -- invalidity on the data model this repository actually has -----------


def test_overlapping_horizons_inflate_the_error_rate_by_an_order_of_magnitude() -> None:
    # The finding. Measured at 4,000 repetitions: 49.2% against a nominal
    # 5%. Asserted loosely here so the test is cheap, but the direction and
    # scale are the point.
    rate = _rejection_rate(lambda rng: _overlapping(rng, 300), offset=40_000)

    assert rate > 4.0 * ALPHA, (
        f"overlapping windows measured {rate}; if this ever falls to the nominal "
        "level the dependence model has changed and the conclusion in "
        "docs/mathematical_validation_2026.md needs revisiting"
    )


def test_blocking_restores_validity_and_destroys_the_power() -> None:
    # Taking every `horizon`-th window removes the overlap, so the
    # assumption holds again -- at one fourteenth of the sample. The valid
    # version has no power at that size, which is why the method is not
    # usable here yet rather than simply wrong.
    horizon = 14
    valid_rate = _rejection_rate(
        lambda rng: _overlapping(rng, 300, horizon=horizon)[::horizon], offset=50_000
    )
    valid_power = _rejection_rate(
        lambda rng: _overlapping(rng, 300, horizon=horizon, mean=0.10)[::horizon],
        offset=60_000,
    )

    assert valid_rate <= ALPHA
    assert valid_power < 0.2, (
        "22 independent observations cannot support this test; ~100 are needed for "
        "usable power at this effect size, which is about five years of 14-day windows"
    )


# -- it must not be wired into anything ----------------------------------


def test_nothing_in_production_imports_the_sequential_test() -> None:
    import pathlib

    src = pathlib.Path("src/turboedge")
    importers = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if "research.sequential" in path.read_text(encoding="utf-8")
        and path.name != "sequential.py"
    ]

    assert importers == [], (
        f"research.sequential is imported by {importers}. It is not valid on "
        "overlapping forward observations, so wiring it into a confirmatory trial "
        "would replace a sound fixed-sample test with an unsound sequential one."
    )
