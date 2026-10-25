"""The analytic first-passage reference, checked against known truth.

Three kinds of check, and they are not interchangeable:

1. **Closed-form identities.** At zero drift the formula must reduce to the
   reflection principle exactly. These are equalities, not tolerances.
2. **Convergence.** A simulation with enough steps and paths must agree
   with the analytic value inside Monte Carlo error.
3. **Sign of the discretisation bias.** Observing fewer points can only
   miss crossings, so discrete monitoring must *understate* the hit
   probability. That direction is what makes the finding in
   `docs/mathematical_validation_2026.md` decisive, so it is pinned rather
   than assumed.

None of this says anything about real markets. The production simulator
bootstraps real historical day shapes, which is a different model; these
formulas are a reference for numerical correctness under Brownian
assumptions and nothing more.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from turboedge.research.first_passage import (
    BarrierSpec,
    bm_hit_probability,
    bridge_corrected_hit_probability,
    discrete_monitoring_hit_probability,
    gbm_hit_probability,
    hit_probability_grid,
    intraday_extreme_hit_probability,
)
from turboedge.storage.schemas import Direction


def _spec(
    *,
    distance: float = 0.05,
    sigma: float = 0.02,
    days: float = 10.0,
    mu: float = 0.0,
    direction: Direction = Direction.LONG,
) -> BarrierSpec:
    barrier = 100.0 * (1 - distance) if direction is Direction.LONG else 100.0 * (1 + distance)
    return BarrierSpec(
        spot=100.0, barrier=barrier, mu=mu, sigma=sigma, horizon=days, direction=direction
    )


# -- 1. closed-form identities -------------------------------------------


@pytest.mark.parametrize(
    ("a", "sigma", "horizon"),
    [(0.05, 0.02, 5.0), (0.10, 0.015, 10.0), (0.02, 0.03, 3.0), (0.3, 0.01, 1.0)],
)
def test_zero_drift_is_exactly_the_reflection_principle(
    a: float, sigma: float, horizon: float
) -> None:
    # P(min <= -a) = 2 Phi(-a / (sigma sqrt(T))). An equality, so compared
    # as one: a tolerance here would hide an algebra error.
    got = bm_hit_probability(a, 0.0, sigma, horizon)
    expected = 2.0 * float(norm.cdf(-a / (sigma * math.sqrt(horizon))))

    assert got == pytest.approx(expected, abs=1e-12)


def test_a_barrier_already_touched_is_certain() -> None:
    assert bm_hit_probability(0.0, 0.0, 0.02, 10.0) == 1.0
    assert bm_hit_probability(-0.01, 0.0, 0.02, 10.0) == 1.0


def test_a_distant_barrier_over_a_short_horizon_is_near_zero() -> None:
    assert gbm_hit_probability(_spec(distance=0.5, sigma=0.01, days=1.0)) < 1e-12


def test_the_probability_rises_with_horizon_volatility_and_proximity() -> None:
    base = gbm_hit_probability(_spec())

    assert gbm_hit_probability(_spec(days=20.0)) > base
    assert gbm_hit_probability(_spec(sigma=0.04)) > base
    assert gbm_hit_probability(_spec(distance=0.02)) > base


def test_drift_away_from_the_barrier_lowers_the_probability() -> None:
    towards = gbm_hit_probability(_spec(mu=-0.004))
    flat = gbm_hit_probability(_spec(mu=0.0))
    away = gbm_hit_probability(_spec(mu=+0.004))

    assert towards > flat > away


def test_long_and_short_are_symmetric_at_equal_log_distance() -> None:
    # With mu = sigma^2/2 the log-drift is exactly zero, so the two
    # directions face the same problem mirrored -- but only if the barrier
    # is placed at an equal *log* distance. A symmetric percentage offset is
    # not symmetric in logs: ln(100/95) = 0.05129 against ln(105/100) =
    # 0.04879, which is why a 5% down move and a 5% up move are not the same
    # first-passage problem for a turbo.
    mu = 0.5 * 0.02**2
    d = 0.05
    long_spec = BarrierSpec(
        spot=100.0,
        barrier=100.0 * math.exp(-d),
        mu=mu,
        sigma=0.02,
        horizon=10.0,
        direction=Direction.LONG,
    )
    short_spec = BarrierSpec(
        spot=100.0,
        barrier=100.0 * math.exp(+d),
        mu=mu,
        sigma=0.02,
        horizon=10.0,
        direction=Direction.SHORT,
    )
    assert long_spec.log_distance == pytest.approx(short_spec.log_distance, rel=1e-12)

    assert gbm_hit_probability(long_spec) == pytest.approx(
        gbm_hit_probability(short_spec), rel=1e-9
    )


def test_a_percentage_offset_is_not_log_symmetric() -> None:
    # The asymmetry the test above had to correct for, pinned so it is a
    # recorded property rather than a trap. A Long barrier 5% below is
    # further away in logs than a Short barrier 5% above, so at equal
    # volatility the Short is likelier to knock out.
    mu = 0.5 * 0.02**2
    long_p = gbm_hit_probability(_spec(mu=mu, direction=Direction.LONG))
    short_p = gbm_hit_probability(_spec(mu=mu, direction=Direction.SHORT))

    assert short_p > long_p


def test_the_ito_correction_is_applied() -> None:
    # `mu` is a geometric drift, so the log-price drifts at mu - sigma^2/2.
    # Omitting that correction would be a silent bias, largest at high
    # volatility, so it is pinned explicitly.
    spec = _spec(mu=0.0, sigma=0.05)

    assert spec.log_drift == pytest.approx(-0.5 * 0.05**2)


# -- 2. convergence against simulation -----------------------------------


@pytest.mark.parametrize("distance", [0.02, 0.05, 0.10])
def test_fine_monitoring_converges_to_the_analytic_value(distance: float) -> None:
    # Many sub-daily observations approximate the continuous path, so the
    # simulated rate must land inside a few standard errors of the formula.
    spec = _spec(distance=distance, sigma=0.02, days=10.0)
    analytic = gbm_hit_probability(spec)
    simulated, se = discrete_monitoring_hit_probability(spec, steps=4000, paths=40_000, seed=3)

    assert abs(simulated - analytic) < max(4.0 * se, 0.004), (
        f"analytic {analytic:.5f} vs {simulated:.5f} +/- {se:.5f}"
    )


def test_the_estimate_is_reproducible_for_a_given_seed() -> None:
    spec = _spec()
    first, _ = discrete_monitoring_hit_probability(spec, steps=10, paths=20_000, seed=42)
    again, _ = discrete_monitoring_hit_probability(spec, steps=10, paths=20_000, seed=42)
    other, _ = discrete_monitoring_hit_probability(spec, steps=10, paths=20_000, seed=43)

    assert first == again
    assert first != other


# -- 3. the discretisation bias, and its direction -----------------------


@pytest.mark.parametrize(
    ("distance", "sigma", "days"),
    [(0.02, 0.02, 3.0), (0.05, 0.02, 14.0), (0.10, 0.02, 14.0)],
)
def test_discrete_monitoring_understates_the_probability(
    distance: float, sigma: float, days: float
) -> None:
    # The direction that decides the whole question. Observing fewer points
    # can only miss crossings, never invent them -- so a daily scan is
    # biased DOWN, while `docs/measured_results.md` reports the production
    # P(KO) biased UP by 40-135% relative. The two point opposite ways,
    # which is why correcting the discretisation would worsen the
    # calibration rather than improve it.
    spec = _spec(distance=distance, sigma=sigma, days=days)
    analytic = gbm_hit_probability(spec)
    daily, se = discrete_monitoring_hit_probability(spec, steps=int(days), paths=200_000, seed=5)

    assert daily < analytic
    assert analytic - daily > 3.0 * se, "the bias must be larger than sampling noise"


def test_the_bridge_correction_removes_almost_all_of_the_bias() -> None:
    # Measured over the benchmark grid: mean residual 0.009pp against a
    # mean uncorrected bias of 4.5pp.
    spec = _spec(distance=0.02, sigma=0.02, days=3.0)
    analytic = gbm_hit_probability(spec)
    daily, _ = discrete_monitoring_hit_probability(spec, steps=3, paths=200_000, seed=9)
    bridged, _ = bridge_corrected_hit_probability(spec, steps=3, paths=200_000, seed=9)

    assert analytic - daily > 0.15, "this cell should show a large uncorrected bias"
    assert abs(analytic - bridged) < 0.005
    assert abs(analytic - bridged) < abs(analytic - daily) / 10.0


def test_observing_the_daily_extreme_removes_most_of_the_bias() -> None:
    # What `simulation/barrier.first_hit_index` already does: it tests each
    # day's open and far extreme, not only the close. Measured over the
    # grid, that alone cuts the mean bias from 6.07pp to 1.51pp, which is
    # the quantitative defence of the existing design.
    spec = _spec(distance=0.02, sigma=0.02, days=3.0)
    analytic = gbm_hit_probability(spec)
    close_only, _ = discrete_monitoring_hit_probability(spec, steps=3, paths=100_000, seed=13)
    with_extreme, _ = intraday_extreme_hit_probability(
        spec, steps=3, paths=100_000, seed=13, substeps=20
    )

    assert close_only < with_extreme < analytic, (
        "seeing the daily extreme must help, and must not overshoot the continuous value"
    )
    assert (analytic - with_extreme) < (analytic - close_only) / 2.0


# -- numerical stability --------------------------------------------------


@pytest.mark.parametrize("mu", [-0.5, -0.05, 0.0, 0.05, 0.5])
@pytest.mark.parametrize("distance", [0.001, 0.05, 0.5])
def test_no_overflow_or_nonsense_in_extreme_regimes(mu: float, distance: float) -> None:
    # The second term is exp(-2 m a / sigma^2) * Phi(...), which overflows
    # if the factors are computed separately before they cancel. Every
    # result must stay a probability.
    p = gbm_hit_probability(_spec(distance=distance, mu=mu, sigma=0.01, days=30.0))

    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_tiny_volatility_and_long_horizon_stay_finite() -> None:
    p = gbm_hit_probability(_spec(distance=0.2, sigma=1e-6, days=250.0))

    assert math.isfinite(p)
    assert p == pytest.approx(0.0, abs=1e-12)


def test_invalid_inputs_are_rejected_rather_than_guessed() -> None:
    with pytest.raises(ValueError):
        BarrierSpec(
            spot=0.0, barrier=90.0, mu=0.0, sigma=0.02, horizon=10.0, direction=Direction.LONG
        )
    with pytest.raises(ValueError):
        BarrierSpec(
            spot=100.0, barrier=90.0, mu=0.0, sigma=0.0, horizon=10.0, direction=Direction.LONG
        )
    with pytest.raises(ValueError):
        BarrierSpec(
            spot=100.0, barrier=90.0, mu=0.0, sigma=0.02, horizon=0.0, direction=Direction.LONG
        )
    with pytest.raises(ValueError):
        bm_hit_probability(0.05, 0.0, 0.02, 0.0)


# -- the grid, as a reproducible benchmark -------------------------------


def test_the_grid_returns_one_row_per_spec_and_is_reproducible() -> None:
    specs = [
        _spec(distance=d, sigma=s, days=t)
        for d in (0.02, 0.05)
        for s in (0.01, 0.02)
        for t in (3.0, 14.0)
    ]

    first = hit_probability_grid(specs, steps=5, paths=20_000, seed=17)
    again = hit_probability_grid(specs, steps=5, paths=20_000, seed=17)

    assert first.shape == (len(specs), 4)
    assert (first == again).all(), "the benchmark must be reproducible for a fixed seed"
    # analytic >= bridge-corrected >= discrete is not guaranteed cell by
    # cell at finite samples, but the discrete column must never exceed the
    # analytic one by more than sampling noise.
    for analytic, discrete, _bridged, se in first:
        assert discrete - analytic < 4.0 * max(se, 1e-9) + 1e-9


def test_nothing_in_production_imports_this_module() -> None:
    # Measurement infrastructure, not a model. If the production path ever
    # imports it, that is a model change and must be a deliberate one.
    import pathlib

    src = pathlib.Path("src/turboedge")
    importers = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if "research.first_passage" in path.read_text(encoding="utf-8")
        and path.name != "first_passage.py"
    ]

    assert importers == [], f"first_passage is imported by {importers}"
