"""Tests for the pre-registration 2026Q4-001 harness (``backtest/synthetic_turbo_ev.py``),
as amended by §10 (Amendment B).

Every bar series here is constructed locally (seeded numpy, no network, no
``adapters/fallback_prices.py``) -- this suite never calls
``run_synthetic_net_ev_trial`` against real fetched price history, per the
module's own "do not measure before 2026-10-01" constraint.

Coverage: standardised-universe construction (count, long/short split,
barrier placement, fair-value quote straddle per Amendment B), terms
identity across arms, ``spot0`` consistency with the universe, the
moving-block bootstrap on constructed series with a known mean, every
verdict branch (including the negative-delta one), populated-and-separate
stability fields, the two mandatory anti-triviality checks (now on
``lcb_net_return``, the Amendment B primary statistic), a pin that the
primary statistic really is the LCB and not the mean, and that ``spread``
now genuinely changes the result (it could not before Amendment B).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from turboedge.backtest import synthetic_turbo_ev as sev
from turboedge.backtest.synthetic_turbo_ev import (
    StabilityAnalysis,
    SyntheticEvResult,
    SyntheticTurboConfig,
    build_standardised_universe,
    run_synthetic_net_ev_trial,
)
from turboedge.models.baselines import RegimeConditionalEmpiricalModel
from turboedge.models.directional import NullModel
from turboedge.models.forecast import HORIZONS, HorizonForecast
from turboedge.pricing.fair_value import theoretical_fair_value
from turboedge.ranking.ev import EvConfig, evaluate_product_horizons
from turboedge.storage.schemas import Direction, UnderlyingBar

# ---------------------------------------------------------------------------
# Local, deterministic bar construction (no network) -- same shape as
# tests/ranking/test_ev.py::synthetic_daily_bars, kept local per that
# module's own "no cross-package test import" convention.
# ---------------------------------------------------------------------------


def _bars(
    n_days: int,
    *,
    underlying_id: str = "DAX",
    start: date = date(2015, 1, 1),
    spot0: float = 100.0,
    daily_vol: float = 0.01,
    drift: float = 0.0,
    seed: int = 0,
) -> list[UnderlyingBar]:
    rng = np.random.default_rng(seed)
    bars: list[UnderlyingBar] = []
    price = spot0
    d = start
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        ret = rng.normal(drift, daily_vol)
        o = price
        c = price * float(np.exp(ret))
        hi = max(o, c) * float(np.exp(abs(rng.normal(0.0, 0.003))))
        lo = min(o, c) * float(np.exp(-abs(rng.normal(0.0, 0.003))))
        ts = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
        bars.append(
            UnderlyingBar(
                underlying_id=underlying_id,
                ts=ts,
                interval="1d",
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=1000.0,
                observation_time=ts,
                available_at=ts,
                retrieved_at=ts,
                source="test",
                parser_version="1",
                quality_score=1.0,
            )
        )
        price = c
        d += timedelta(days=1)
    return bars


def _make_forecast(
    mean: float, *, horizon_days: int = 5, uncertainty: float = 0.01, sigma: float = 0.02
) -> HorizonForecast:
    return HorizonForecast(
        underlying_id="DAX",
        horizon_days=horizon_days,
        prediction_time=datetime(2020, 6, 1, 22, 0, tzinfo=UTC),
        p_up=0.5,
        mean=mean,
        sigma=sigma,
        quantiles={
            "q05": mean - 2 * sigma,
            "q25": mean - sigma,
            "q50": mean,
            "q75": mean + sigma,
            "q95": mean + 2 * sigma,
        },
        expected_shortfall_05=mean - 2.5 * sigma,
        uncertainty=uncertainty,
        model_id="rigged_test_model",
        model_hash="deadbeef",
        signal_family="rigged",
        n_train=1000,
        n_effective=1000.0,
    )


# ---------------------------------------------------------------------------
# build_standardised_universe
# ---------------------------------------------------------------------------


def test_universe_has_two_directions_per_barrier_distance() -> None:
    cfg = SyntheticTurboConfig()
    universe = build_standardised_universe(100.0, config=cfg)
    assert len(universe) == 2 * len(cfg.barrier_distances)
    n_long = sum(1 for t in universe.values() if t.direction == Direction.LONG)
    n_short = sum(1 for t in universe.values() if t.direction == Direction.SHORT)
    assert n_long == len(cfg.barrier_distances)
    assert n_short == len(cfg.barrier_distances)


def test_universe_long_barrier_below_short_barrier_above_spot() -> None:
    spot = 100.0
    universe = build_standardised_universe(spot, config=SyntheticTurboConfig())
    for terms in universe.values():
        if terms.direction == Direction.LONG:
            assert terms.knockout_barrier < spot
        else:
            assert terms.knockout_barrier > spot
        # open-end convention: financing_level == knockout_barrier
        assert terms.financing_level == terms.knockout_barrier


def test_universe_barrier_placement_matches_configured_distances() -> None:
    spot = 200.0
    cfg = SyntheticTurboConfig(barrier_distances=(0.02, 0.10))
    universe = build_standardised_universe(spot, config=cfg)
    barriers = sorted(t.knockout_barrier for t in universe.values())
    expected = sorted(
        [spot * 0.98, spot * 0.90, spot * 1.02, spot * 1.10],
    )
    for got, want in zip(barriers, expected, strict=True):
        assert got == pytest.approx(want)


def _fair_value_for(terms, spot: float):  # type: ignore[no-untyped-def]
    return theoretical_fair_value(
        direction=terms.direction,
        product_type=terms.product_type,
        spot=spot,
        financing_level=terms.financing_level,
        knockout_barrier=terms.knockout_barrier,
        ratio=terms.ratio,
        fx=terms.fx,
        ref_rate=terms.ref_rate,
        financing_spread=terms.financing_spread,
        as_of=date(2024, 3, 1),  # unused for TURBO_OPEN_END, any date works
        maturity=None,
        dividend_yield=0.0,
    )


def test_universe_quote_straddles_fair_value_per_amendment_b() -> None:
    """Amendment B §10: entry_ask = fair_value * (1 + spread/2), entry_bid =
    fair_value * (1 - spread/2) -- not the original §4 formula (entry_ask = fair_value
    outright), which left entry_ask, the only field simulate_product_payoff reads,
    invariant to spread."""
    spot = 137.5
    cfg = SyntheticTurboConfig(spread=0.02)
    universe = build_standardised_universe(spot, config=cfg)
    for terms in universe.values():
        fair_value = _fair_value_for(terms, spot)
        assert terms.entry_ask == pytest.approx(fair_value * 1.01)
        assert terms.entry_bid == pytest.approx(fair_value * 0.99)
        assert terms.entry_bid < fair_value < terms.entry_ask


def test_universe_spread_genuinely_changes_entry_ask() -> None:
    """Amendment B §10, Change 3's premise: unlike the original §4 formula, entry_ask now
    depends on spread. Directly pins the fact the spread-sensitivity check now relies on."""
    spot = 123.0
    universe_a = build_standardised_universe(spot, config=SyntheticTurboConfig(spread=0.0025))
    universe_b = build_standardised_universe(spot, config=SyntheticTurboConfig(spread=0.01))
    for isin, terms_a in universe_a.items():
        terms_b = universe_b[isin]
        assert terms_a.entry_ask != pytest.approx(terms_b.entry_ask)
        assert terms_a.entry_bid != pytest.approx(terms_b.entry_bid)


def test_universe_rejects_nonpositive_spot() -> None:
    with pytest.raises(ValueError):
        build_standardised_universe(0.0, config=SyntheticTurboConfig())
    with pytest.raises(ValueError):
        build_standardised_universe(-5.0, config=SyntheticTurboConfig())


def test_universe_rejects_out_of_range_barrier_distance() -> None:
    with pytest.raises(ValueError):
        build_standardised_universe(100.0, config=SyntheticTurboConfig(barrier_distances=(1.5,)))


def test_universe_is_a_pure_function_of_spot_and_config() -> None:
    """§4: depends only on spot and config, never a forecast -- two calls with the same
    inputs must be deeply equal (the property the whole trial's arm-comparability rests on)."""
    cfg = SyntheticTurboConfig()
    u1 = build_standardised_universe(123.45, config=cfg)
    u2 = build_standardised_universe(123.45, config=cfg)
    assert u1 == u2
    assert u1 is not u2  # different dict objects; content is what must match


def test_synthetic_isin_roundtrips_direction_and_distance() -> None:
    for direction in (Direction.LONG, Direction.SHORT):
        for distance in (0.02, 0.05, 0.10, 0.15, 0.20):
            isin = sev._synthetic_isin(direction, distance)
            got_direction, got_distance = sev._parse_synthetic_isin(isin)
            assert got_direction == direction
            assert got_distance == pytest.approx(distance)


# ---------------------------------------------------------------------------
# Terms identity across arms + spot0 consistency (§4/§6.10)
# ---------------------------------------------------------------------------


def test_both_arms_receive_the_identical_terms_object_at_every_date(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    real_fn = sev.evaluate_product_horizons

    def spy(terms_by_isin, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"terms": terms_by_isin, "spot0": kwargs["spot0"]})
        return real_fn(terms_by_isin, *args, **kwargs)

    monkeypatch.setattr(sev, "evaluate_product_horizons", spy)

    bars = {"DAX": _bars(780, seed=1, daily_vol=0.009)}
    cfg = SyntheticTurboConfig(n_paths=10)
    run_synthetic_net_ev_trial(bars, config=cfg)

    assert len(calls) >= 2
    assert len(calls) % 2 == 0  # calls come in (null, regime) pairs per date
    for i in range(0, len(calls), 2):
        assert calls[i]["terms"] is calls[i + 1]["terms"]
        assert calls[i]["spot0"] == calls[i + 1]["spot0"]


def test_spot0_passed_matches_the_spot_the_universe_was_built_from(monkeypatch) -> None:
    """§6.10: a spot0/universe mismatch produced 365 spurious gate crossings through
    leverage -- pin that spot0 always agrees with the universe actually priced.

    Exercises :func:`sev._price_trial_grid` directly (not the full trial) so every recorded
    call shares the one ``cfg`` this test rebuilds against: ``run_synthetic_net_ev_trial``'s
    own §6 spread-sensitivity probes (Amendment B §10, Change 3) legitimately call
    ``evaluate_product_horizons`` again under *different* ``cfg.spread`` values, which would
    otherwise make this test's single fixed ``cfg`` the wrong universe to rebuild a probe
    call's recorded terms against.
    """
    calls: list[dict[str, object]] = []
    real_fn = sev.evaluate_product_horizons

    def spy(terms_by_isin, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"terms": dict(terms_by_isin), "spot0": kwargs["spot0"]})
        return real_fn(terms_by_isin, *args, **kwargs)

    monkeypatch.setattr(sev, "evaluate_product_horizons", spy)

    cfg = SyntheticTurboConfig(n_paths=10)
    bars = {"DAX": _bars(780, seed=2, daily_vol=0.009)}
    underlying_forecasts = sev._collect_forecasts_by_underlying(bars)
    sev._price_trial_grid(underlying_forecasts, cfg)

    assert calls
    for call in calls:
        spot0 = call["spot0"]
        assert isinstance(spot0, float)
        expected_universe = build_standardised_universe(spot0, config=cfg)
        assert call["terms"] == expected_universe


# ---------------------------------------------------------------------------
# Amendment B §10, Change 1: the primary statistic is lcb_net_return
# ---------------------------------------------------------------------------


def test_primary_statistic_is_lcb_net_return_not_mean_net_return(monkeypatch) -> None:
    """Pins the Amendment B §10 primary-statistic switch: rig a fake
    ``evaluate_product_horizons`` whose ``mean_net_return`` is identical across arms (so
    reading it would collapse the delta to exactly 0) but whose ``lcb_net_return`` tracks
    each arm's real forecast mean (so it differs between arms, since regime_conditional and
    null disagree on this history). If this module is ever switched back to reading
    ``mean_net_return``, this test fails loudly instead of silently passing."""

    def fake_evaluate_product_horizons(terms_by_isin, forecast_by_horizon, bars, **kwargs):  # type: ignore[no-untyped-def]
        horizons = kwargs["horizons"]
        out = []
        for isin in terms_by_isin:
            for h in horizons:
                forecast = forecast_by_horizon[h]
                out.append(
                    SimpleNamespace(
                        isin=isin,
                        horizon_days=h,
                        lcb_net_return=forecast.mean * 10.0,
                        mean_net_return=0.12345,  # rigged identical across arms
                    )
                )
        return out

    monkeypatch.setattr(sev, "evaluate_product_horizons", fake_evaluate_product_horizons)

    bars = {"DAX": _bars(780, seed=21, daily_vol=0.009)}
    cfg = SyntheticTurboConfig(n_paths=10)
    result = run_synthetic_net_ev_trial(bars, config=cfg)

    assert abs(result.mean_delta_lcb_net_ev) > 1e-9


# ---------------------------------------------------------------------------
# Amendment B §10, Change 3: spread now genuinely changes the result
# ---------------------------------------------------------------------------


def test_spread_sensitivity_probes_genuinely_differ() -> None:
    """Before Amendment B, spread could not affect entry_ask under simulate_product_payoff's
    contract, so both probes were forced equal to the (unrelated) primary value. After
    Amendment B, entry_ask genuinely depends on spread, so the two probes must be an actual,
    independently re-simulated result, not a reused constant."""
    cfg = SyntheticTurboConfig(n_paths=30)
    bars = {"DAX": _bars(800, seed=42, daily_vol=0.009)}
    underlying_forecasts = sev._collect_forecasts_by_underlying(bars)

    sensitivity = sev._spread_sensitivity_mean_delta_lcb_net_ev(underlying_forecasts, cfg)

    assert set(sensitivity) == {"spread_0.0025", "spread_0.01"}
    assert sensitivity["spread_0.0025"] != pytest.approx(sensitivity["spread_0.01"])


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_p_value_known_mean_constant_series() -> None:
    """A constant series has an exactly-computable bootstrap p-value: every centred resample
    is identically 0, so it is never >= a strictly positive observed mean."""
    delta = np.full(100, 0.05)
    p = sev._moving_block_bootstrap_p_value(
        delta, block_length=21, n_resamples=1000, rng=np.random.default_rng(0)
    )
    assert p == pytest.approx(1.0 / 1001.0)


def test_bootstrap_p_value_small_for_a_clearly_positive_series() -> None:
    rng = np.random.default_rng(0)
    delta = rng.normal(0.02, 0.002, size=200)  # mean far from 0 relative to noise
    p = sev._moving_block_bootstrap_p_value(
        delta, block_length=21, n_resamples=2000, rng=np.random.default_rng(1)
    )
    assert p < 0.01


def test_bootstrap_p_value_large_for_a_zero_mean_series() -> None:
    rng = np.random.default_rng(3)
    delta = rng.normal(0.0, 0.01, size=200)
    p = sev._moving_block_bootstrap_p_value(
        delta, block_length=21, n_resamples=2000, rng=np.random.default_rng(4)
    )
    assert p > 0.10


def test_bootstrap_p_value_deterministic_for_same_seeded_rng() -> None:
    delta = np.random.default_rng(5).normal(0.01, 0.01, size=50)
    p1 = sev._moving_block_bootstrap_p_value(
        delta, block_length=10, n_resamples=500, rng=np.random.default_rng(42)
    )
    p2 = sev._moving_block_bootstrap_p_value(
        delta, block_length=10, n_resamples=500, rng=np.random.default_rng(42)
    )
    assert p1 == p2


def test_bootstrap_p_value_within_unit_interval() -> None:
    delta = np.random.default_rng(6).normal(0.0, 0.02, size=80)
    p = sev._moving_block_bootstrap_p_value(
        delta, block_length=21, n_resamples=500, rng=np.random.default_rng(7)
    )
    assert 0.0 < p <= 1.0


def test_bootstrap_p_value_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        sev._moving_block_bootstrap_p_value(
            np.array([]), block_length=21, n_resamples=10, rng=np.random.default_rng(0)
        )


def test_bootstrap_p_value_rejects_invalid_block_length() -> None:
    with pytest.raises(ValueError):
        sev._moving_block_bootstrap_p_value(
            np.array([1.0, 2.0]), block_length=0, n_resamples=10, rng=np.random.default_rng(0)
        )


# ---------------------------------------------------------------------------
# Verdict table (pre-registration §5) -- every branch
# ---------------------------------------------------------------------------


def test_verdict_pass_when_significant_and_positive() -> None:
    verdict, reason = sev._verdict(0.05, 0.01, 0.10)
    assert verdict == "PASS"
    assert "row 1" in reason


def test_verdict_fail_null_result_when_not_significant() -> None:
    verdict, reason = sev._verdict(0.50, 0.01, 0.10)
    assert verdict == "FAIL_NULL_RESULT"
    assert "row 2" in reason


def test_verdict_fail_null_result_even_if_delta_positive_but_not_significant() -> None:
    verdict, _ = sev._verdict(0.80, 0.05, 0.10)
    assert verdict == "FAIL_NULL_RESULT"


def test_verdict_fail_negative_significant_reported_as_worse() -> None:
    verdict, reason = sev._verdict(0.02, -0.01, 0.10)
    assert verdict == "FAIL_NEGATIVE_SIGNIFICANT"
    assert "row 3" in reason


def test_verdict_fail_negative_significant_at_exactly_zero_delta() -> None:
    verdict, _ = sev._verdict(0.02, 0.0, 0.10)
    assert verdict == "FAIL_NEGATIVE_SIGNIFICANT"


def test_verdict_boundary_p_equals_alpha_with_positive_delta_is_pass() -> None:
    verdict, _ = sev._verdict(0.10, 0.001, 0.10)
    assert verdict == "PASS"


# ---------------------------------------------------------------------------
# Stability analysis: populated, and structurally separate from the primary
# ---------------------------------------------------------------------------


def test_stability_analysis_groups_by_underlying_horizon_and_distance() -> None:
    records = [
        sev._CellRecord(
            date_key=date(2020, 1, 1),
            underlying_id="DAX",
            isin="SYNTH-LONG-0.0200",
            direction=Direction.LONG,
            barrier_distance=0.02,
            horizon_days=5,
            lcb_net_ev_null=0.0,
            lcb_net_ev_regime_conditional=0.01,
        ),
        sev._CellRecord(
            date_key=date(2020, 1, 1),
            underlying_id="DAX",
            isin="SYNTH-SHORT-0.0200",
            direction=Direction.SHORT,
            barrier_distance=0.02,
            horizon_days=5,
            lcb_net_ev_null=0.0,
            lcb_net_ev_regime_conditional=-0.01,
        ),
        sev._CellRecord(
            date_key=date(2020, 1, 2),
            underlying_id="NDX",
            isin="SYNTH-LONG-0.0500",
            direction=Direction.LONG,
            barrier_distance=0.05,
            horizon_days=10,
            lcb_net_ev_null=0.0,
            lcb_net_ev_regime_conditional=0.02,
        ),
    ]
    spread_sensitivity = {"spread_0.0025": 0.001, "spread_0.01": 0.002}
    stability = sev._stability_analysis(
        records,
        primary_mean_delta_lcb_net_ev=0.005,
        spread_sensitivity_mean_delta=spread_sensitivity,
    )

    assert stability.n_cells == 3
    assert stability.share_cells_delta_positive == pytest.approx(2.0 / 3.0)
    assert stability.delta_net_ev_by_underlying["DAX"] == pytest.approx(0.0)
    assert stability.delta_net_ev_by_underlying["NDX"] == pytest.approx(0.02)
    assert stability.delta_net_ev_by_horizon[5] == pytest.approx(0.0)
    assert stability.delta_net_ev_by_horizon[10] == pytest.approx(0.02)
    assert stability.delta_net_ev_by_barrier_distance[0.02] == pytest.approx(0.0)
    assert stability.delta_net_ev_by_barrier_distance[0.05] == pytest.approx(0.02)
    assert stability.spread_sensitivity_mean_delta == spread_sensitivity
    assert stability.spread_sensitivity_sign_stable  # both probes positive, primary positive


def test_stability_analysis_sign_stable_false_when_a_probe_flips_sign() -> None:
    records = [
        sev._CellRecord(
            date_key=date(2020, 1, 1),
            underlying_id="DAX",
            isin="SYNTH-LONG-0.0200",
            direction=Direction.LONG,
            barrier_distance=0.02,
            horizon_days=5,
            lcb_net_ev_null=0.0,
            lcb_net_ev_regime_conditional=0.01,
        ),
    ]
    spread_sensitivity = {"spread_0.0025": 0.001, "spread_0.01": -0.0005}
    stability = sev._stability_analysis(
        records,
        primary_mean_delta_lcb_net_ev=0.005,
        spread_sensitivity_mean_delta=spread_sensitivity,
    )
    assert not stability.spread_sensitivity_sign_stable


def test_stability_analysis_rejects_empty_records() -> None:
    with pytest.raises(ValueError):
        sev._stability_analysis([], 0.0, {"spread_0.0025": 0.0, "spread_0.01": 0.0})


@pytest.fixture(scope="module")
def small_trial_result() -> SyntheticEvResult:
    cfg = SyntheticTurboConfig(n_paths=25)
    bars = {"DAX": _bars(800, seed=42, daily_vol=0.009)}
    return run_synthetic_net_ev_trial(bars, config=cfg)


def test_full_trial_result_shape_and_types(small_trial_result: SyntheticEvResult) -> None:
    assert isinstance(small_trial_result, SyntheticEvResult)
    assert isinstance(small_trial_result.stability, StabilityAnalysis)
    assert small_trial_result.verdict in ("PASS", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT")
    assert 0.0 <= small_trial_result.p_value <= 1.0
    assert small_trial_result.alpha == pytest.approx(0.10)
    assert small_trial_result.block_length == 21
    assert small_trial_result.n_dates > 0
    assert small_trial_result.null_arm.dates == small_trial_result.regime_conditional_arm.dates
    assert len(small_trial_result.null_arm.dates) == small_trial_result.n_dates
    assert small_trial_result.null_arm.model_id == NullModel().model_id
    assert (
        small_trial_result.regime_conditional_arm.model_id
        == RegimeConditionalEmpiricalModel().model_id
    )
    assert small_trial_result.null_arm.dates == sorted(small_trial_result.null_arm.dates)


def test_full_trial_stability_is_populated_and_separate_from_primary(
    small_trial_result: SyntheticEvResult,
) -> None:
    stability = small_trial_result.stability
    assert stability.n_cells > 0
    assert set(stability.delta_net_ev_by_horizon).issubset(set(HORIZONS))
    assert set(stability.delta_net_ev_by_barrier_distance) == set(
        SyntheticTurboConfig().barrier_distances
    )
    assert set(stability.delta_net_ev_by_underlying) == {"DAX"}
    assert 0.0 <= stability.share_cells_delta_positive <= 1.0

    assert set(stability.spread_sensitivity_mean_delta) == {"spread_0.0025", "spread_0.01"}
    # Amendment B §10: the two probes are genuine, independent re-simulations -- they need not
    # equal the primary value (spread=0.005, between the two probes) or each other.
    for v in stability.spread_sensitivity_mean_delta.values():
        assert isinstance(v, float)
    assert stability.spread_sensitivity_sign_stable == all(
        (v > 0.0) == (small_trial_result.mean_delta_lcb_net_ev > 0.0)
        for v in stability.spread_sensitivity_mean_delta.values()
    )

    # Structural separation: the primary p-value/verdict live only on the top-level
    # result, never inside the (pydantic, extra="forbid") stability model.
    assert not hasattr(stability, "p_value")
    assert not hasattr(stability, "verdict")
    assert "p_value" not in StabilityAnalysis.model_fields
    assert "verdict" not in StabilityAnalysis.model_fields


def test_run_trial_rejects_empty_bars_mapping() -> None:
    with pytest.raises(ValueError):
        run_synthetic_net_ev_trial({}, config=SyntheticTurboConfig())


def test_run_trial_raises_on_insufficient_history() -> None:
    bars = {"DAX": _bars(100, seed=1)}
    with pytest.raises(ValueError):
        run_synthetic_net_ev_trial(bars, config=SyntheticTurboConfig(n_paths=10))


# ---------------------------------------------------------------------------
# Mandatory anti-triviality checks (Amendment B §10: on lcb_net_return)
# ---------------------------------------------------------------------------


def test_anti_triviality_genuinely_better_arm_passes() -> None:
    """(1) A rigged pair of forecasts where one arm is genuinely, unambiguously better on
    the priced grid must reach PASS -- otherwise a harness that always fails would pass
    this suite undetected.

    Restricted to the LONG half of the grid: the LCB NetEV this harness reads is driven,
    for a fixed uncertainty, by the forecast's drift (`HorizonForecast.mean`) in both the
    central and pessimistic scenarios, which helps LONG products and hurts SHORT products
    symmetrically for the same underlying (same forecast feeds both) -- a whole-grid average
    would partially cancel a pure drift rig by construction, which is a fact about the priced
    grid, not about the bootstrap/verdict machinery under test here.
    """
    bars = _bars(900, seed=11, daily_vol=0.008)
    cfg = SyntheticTurboConfig(n_paths=300)

    null_forecast = {5: _make_forecast(0.0, horizon_days=5)}
    better_forecast = {5: _make_forecast(0.04, horizon_days=5)}

    null_means: list[float] = []
    better_means: list[float] = []
    for i in range(800, 860, 6):  # 10 synthetic "dates" drawn from one long history
        bar = bars[i]
        grid = build_standardised_universe(bar.close, config=cfg)
        grid_long = {isin: t for isin, t in grid.items() if t.direction == Direction.LONG}
        ev_cfg = EvConfig(n_paths=cfg.n_paths)

        null_evals = evaluate_product_horizons(
            grid_long,
            null_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=bar.available_at,
            as_of=bar.ts.date(),
            cluster_id="anti-triviality",
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            cfg=ev_cfg,
        )
        better_evals = evaluate_product_horizons(
            grid_long,
            better_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=bar.available_at,
            as_of=bar.ts.date(),
            cluster_id="anti-triviality",
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            cfg=ev_cfg,
        )
        null_means.append(float(np.mean([e.lcb_net_return for e in null_evals])))
        better_means.append(float(np.mean([e.lcb_net_return for e in better_evals])))

    delta = np.asarray(better_means) - np.asarray(null_means)
    mean_delta = float(np.mean(delta))
    assert mean_delta > 0.0  # sanity: the rig actually produced an advantage

    p = sev._moving_block_bootstrap_p_value(
        delta, block_length=5, n_resamples=2000, rng=np.random.default_rng(0)
    )
    verdict, _ = sev._verdict(p, mean_delta, 0.10)
    assert verdict == "PASS"


def test_anti_triviality_identical_forecasts_yield_null_result() -> None:
    """(2) Two identical forecasts (same model, in effect) must yield delta LCB NetEV ~= 0
    and a non-significant p -- otherwise a harness that always finds a difference would pass
    this suite undetected."""
    bars = _bars(900, seed=12, daily_vol=0.008)
    cfg = SyntheticTurboConfig(n_paths=300)
    forecast = {5: _make_forecast(0.01, horizon_days=5)}

    deltas: list[float] = []
    for i in range(800, 860, 6):
        bar = bars[i]
        grid = build_standardised_universe(bar.close, config=cfg)
        ev_cfg = EvConfig(n_paths=cfg.n_paths)

        evals_a = evaluate_product_horizons(
            grid,
            forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=bar.available_at,
            as_of=bar.ts.date(),
            cluster_id="anti-triviality",
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            cfg=ev_cfg,
        )
        evals_b = evaluate_product_horizons(
            grid,
            forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=bar.available_at,
            as_of=bar.ts.date(),
            cluster_id="anti-triviality",
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            cfg=ev_cfg,
        )
        mean_a = float(np.mean([e.lcb_net_return for e in evals_a]))
        mean_b = float(np.mean([e.lcb_net_return for e in evals_b]))
        deltas.append(mean_b - mean_a)

    delta = np.asarray(deltas)
    # Identical forecast + identical (seeded) paths => bit-identical LCB NetEV per cell.
    assert np.allclose(delta, 0.0, atol=1e-9)

    p = sev._moving_block_bootstrap_p_value(
        delta, block_length=5, n_resamples=2000, rng=np.random.default_rng(0)
    )
    verdict, _ = sev._verdict(p, float(np.mean(delta)), 0.10)
    assert p > 0.10
    assert verdict == "FAIL_NULL_RESULT"


# ---------------------------------------------------------------------------
# 2026Q4-002 §1: sigma_source config, default reproduces 2026Q4-001 unchanged
# ---------------------------------------------------------------------------


def test_sigma_source_defaults_to_historical() -> None:
    assert SyntheticTurboConfig().sigma_source == "historical"


def test_default_call_reproduces_2026q4_001_shape(small_trial_result: SyntheticEvResult) -> None:
    """A default (``sigma_source="historical"``) call must keep behaving exactly like the
    2026Q4-001 harness: no falsification arm, no sigma-targeting diagnostics populated, and the
    verdict stays within 2026Q4-001's own three-value set."""
    assert small_trial_result.sigma_source == "historical"
    assert small_trial_result.falsification is None
    assert small_trial_result.verdict in (
        "PASS",
        "FAIL_NULL_RESULT",
        "FAIL_NEGATIVE_SIGNIFICANT",
    )
    stability = small_trial_result.stability
    assert stability.clamped_cell_count_null == 0
    assert stability.clamped_cell_count_regime_conditional == 0
    assert stability.mean_realized_sigma_by_arm == {}
    assert stability.mean_target_sigma_by_arm == {}
    # Historical mode never requests a target, so every cell is trivially "reachable" and the
    # reachable-only recompute must equal the primary exactly.
    assert stability.reachable_only_mean_delta_lcb_net_ev == pytest.approx(
        small_trial_result.mean_delta_lcb_net_ev
    )


# ---------------------------------------------------------------------------
# 2026Q4-002 §1: forecast mode actually targets each arm's own forecast sigma
# ---------------------------------------------------------------------------


def test_forecast_mode_sets_target_sigma_and_realized_matches_where_reachable() -> None:
    """§1: with ``target_sigma_by_horizon`` supplied, every returned cell's ``target_sigma``
    equals what was requested, and (for a target well above the gap-only floor, i.e. reachable)
    ``realized_sigma`` lands close to it with ``target_sigma_met=True``."""
    bars = _bars(800, seed=100, daily_vol=0.01)
    spot0 = bars[-1].close
    universe = build_standardised_universe(spot0, config=SyntheticTurboConfig(n_paths=2000))
    forecast = {5: _make_forecast(0.0, horizon_days=5, sigma=0.05, uncertainty=0.01)}

    evals = sev._evaluate_grid_with_target_sigma(
        universe,
        forecast,
        bars,
        underlying_id="DAX",
        spot0=spot0,
        start=bars[-1].available_at,
        as_of=bars[-1].ts.date(),
        rng=np.random.default_rng(123),
        horizons=[5],
        n_paths=2000,
        target_sigma_by_horizon={5: 0.05},
    )

    assert evals
    for e in evals:
        assert e.target_sigma == pytest.approx(0.05)
        assert e.target_sigma_met is True
        assert e.realized_sigma is not None
        assert e.realized_sigma == pytest.approx(0.05, rel=0.15)


def test_forecast_mode_none_target_disables_sigma_targeting() -> None:
    """``target_sigma_by_horizon=None`` must leave the sigma diagnostics at ``None`` (the
    ``simulate_paths`` no-op contract), matching ``sigma_source="historical"``'s intent."""
    bars = _bars(800, seed=101, daily_vol=0.01)
    spot0 = bars[-1].close
    universe = build_standardised_universe(spot0, config=SyntheticTurboConfig(n_paths=50))
    forecast = {5: _make_forecast(0.0, horizon_days=5, sigma=0.05, uncertainty=0.01)}

    evals = sev._evaluate_grid_with_target_sigma(
        universe,
        forecast,
        bars,
        underlying_id="DAX",
        spot0=spot0,
        start=bars[-1].available_at,
        as_of=bars[-1].ts.date(),
        rng=np.random.default_rng(123),
        horizons=[5],
        n_paths=50,
        target_sigma_by_horizon=None,
    )
    assert evals
    for e in evals:
        assert e.target_sigma is None
        assert e.target_sigma_met is None


# ---------------------------------------------------------------------------
# 2026Q4-002 §2/§6: the falsification mechanism itself can fire
# ---------------------------------------------------------------------------


def test_falsification_forces_both_arms_to_null_sigma_and_zeroes_a_pure_sigma_difference() -> None:
    """Explicit test bullet from the brief: "the falsification arm forces both arms to the
    same sigma and yields delta ~= 0 when the only difference between models is their sigma."

    Two forecasts share the same mean and uncertainty and differ *only* in sigma. Under the
    §1 "own sigma per arm" rule the two arms would generally be priced with different
    dispersion; under the §2/§6 falsification rule (both forced to the null's own sigma) the
    two calls become fully identical inputs (same mean, same uncertainty, same target_sigma,
    same rng seed) and must therefore produce bit-identical LCB NetEV per cell -- delta exactly
    0, not merely small.
    """
    bars = _bars(800, seed=102, daily_vol=0.01)
    spot0 = bars[-1].close
    universe = build_standardised_universe(spot0, config=SyntheticTurboConfig(n_paths=200))
    start = bars[-1].available_at
    as_of = bars[-1].ts.date()

    null_forecast = {5: _make_forecast(0.01, horizon_days=5, sigma=0.02, uncertainty=0.01)}
    regime_forecast = {5: _make_forecast(0.01, horizon_days=5, sigma=0.008, uncertainty=0.01)}
    null_sigma = {5: 0.02}
    falsification_target = dict(null_sigma)  # both forced to the null's own sigma

    null_evals = sev._evaluate_grid_with_target_sigma(
        universe,
        null_forecast,
        bars,
        underlying_id="DAX",
        spot0=spot0,
        start=start,
        as_of=as_of,
        rng=np.random.default_rng(20261002),
        horizons=[5],
        n_paths=200,
        target_sigma_by_horizon=falsification_target,
    )
    regime_evals = sev._evaluate_grid_with_target_sigma(
        universe,
        regime_forecast,
        bars,
        underlying_id="DAX",
        spot0=spot0,
        start=start,
        as_of=as_of,
        rng=np.random.default_rng(20261002),
        horizons=[5],
        n_paths=200,
        target_sigma_by_horizon=falsification_target,
    )

    null_by_key = {(e.isin, e.horizon_days): e.lcb_net_return for e in null_evals}
    regime_by_key = {(e.isin, e.horizon_days): e.lcb_net_return for e in regime_evals}
    assert set(null_by_key) == set(regime_by_key)
    deltas = np.array([regime_by_key[k] - null_by_key[k] for k in null_by_key], dtype=np.float64)
    assert np.allclose(deltas, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 2026Q4-002 §3/§6: clamped cells are counted, never dropped
# ---------------------------------------------------------------------------


def test_clamped_cells_counted_in_stability_not_dropped() -> None:
    records = [
        sev._CellRecord(
            date_key=date(2020, 1, 1),
            underlying_id="DAX",
            isin="SYNTH-LONG-0.0200",
            direction=Direction.LONG,
            barrier_distance=0.02,
            horizon_days=5,
            lcb_net_ev_null=0.0,
            lcb_net_ev_regime_conditional=0.01,
            null_realized_sigma=0.03,
            null_target_sigma=0.03,
            null_target_sigma_met=True,
            regime_realized_sigma=0.045,
            regime_target_sigma=0.01,
            regime_target_sigma_met=False,  # clamped: unreachable target
        ),
        sev._CellRecord(
            date_key=date(2020, 1, 1),
            underlying_id="DAX",
            isin="SYNTH-SHORT-0.0200",
            direction=Direction.SHORT,
            barrier_distance=0.02,
            horizon_days=5,
            lcb_net_ev_null=0.0,
            lcb_net_ev_regime_conditional=0.02,
            null_realized_sigma=0.03,
            null_target_sigma=0.03,
            null_target_sigma_met=True,
            regime_realized_sigma=0.02,
            regime_target_sigma=0.02,
            regime_target_sigma_met=True,
        ),
    ]
    spread_sensitivity = {"spread_0.0025": 0.001, "spread_0.01": 0.002}
    stability = sev._stability_analysis(
        records,
        primary_mean_delta_lcb_net_ev=0.015,
        spread_sensitivity_mean_delta=spread_sensitivity,
    )

    # Both cells are still in n_cells / share_cells_delta_positive -- clamping never drops a cell.
    assert stability.n_cells == 2
    assert stability.share_cells_delta_positive == pytest.approx(1.0)
    assert stability.clamped_cell_count_null == 0
    assert stability.clamped_cell_count_regime_conditional == 1

    # Reachable-only recompute excludes the clamped (LONG) cell -- only the SHORT cell's delta
    # (0.02) remains, not the mean of both (0.015).
    assert stability.reachable_only_mean_delta_lcb_net_ev == pytest.approx(0.02)

    assert stability.mean_realized_sigma_by_arm["null"] == pytest.approx(0.03)
    assert stability.mean_realized_sigma_by_arm["regime_conditional"] == pytest.approx(
        (0.045 + 0.02) / 2.0
    )
    assert stability.mean_target_sigma_by_arm["regime_conditional"] == pytest.approx(
        (0.01 + 0.02) / 2.0
    )


def test_clamped_cells_reachable_only_is_none_when_everything_is_clamped() -> None:
    records = [
        sev._CellRecord(
            date_key=date(2020, 1, 1),
            underlying_id="DAX",
            isin="SYNTH-LONG-0.0200",
            direction=Direction.LONG,
            barrier_distance=0.02,
            horizon_days=5,
            lcb_net_ev_null=0.0,
            lcb_net_ev_regime_conditional=0.01,
            regime_target_sigma_met=False,
        ),
    ]
    stability = sev._stability_analysis(
        records,
        primary_mean_delta_lcb_net_ev=0.01,
        spread_sensitivity_mean_delta={"spread_0.0025": 0.01, "spread_0.01": 0.01},
    )
    assert stability.reachable_only_mean_delta_lcb_net_ev is None


def test_clamped_cell_end_to_end_via_large_gap_risk_bars() -> None:
    """End-to-end (not just the stability-summary unit test above): a bar series with genuine,
    large overnight/weekend gaps gives ``_apply_volatility_scaling`` a high gap-only variance
    floor (``simulation/paths.py`` §3 -- gap is never scaled). Requesting a target far below
    that floor must clamp (``target_sigma_met=False``) rather than raise, and the cell must
    still come back in the result -- never silently dropped."""
    rng = np.random.default_rng(7)
    bars: list[UnderlyingBar] = []
    price = 100.0
    d = date(2015, 1, 1)
    for _ in range(800):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        gap = rng.normal(0.0, 0.05)  # large overnight/weekend gap risk
        o = price * float(np.exp(gap))
        ret = rng.normal(0.0, 0.001)  # tiny intraday move
        c = o * float(np.exp(ret))
        hi = max(o, c) * float(np.exp(abs(rng.normal(0.0, 0.0005))))
        lo = min(o, c) * float(np.exp(-abs(rng.normal(0.0, 0.0005))))
        ts = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
        bars.append(
            UnderlyingBar(
                underlying_id="DAX",
                ts=ts,
                interval="1d",
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=1000.0,
                observation_time=ts,
                available_at=ts,
                retrieved_at=ts,
                source="test",
                parser_version="1",
                quality_score=1.0,
            )
        )
        price = c
        d += timedelta(days=1)

    spot0 = bars[-1].close
    universe = build_standardised_universe(spot0, config=SyntheticTurboConfig(n_paths=500))
    forecast = {5: _make_forecast(0.0, horizon_days=5, sigma=0.0001, uncertainty=0.01)}

    evals = sev._evaluate_grid_with_target_sigma(
        universe,
        forecast,
        bars,
        underlying_id="DAX",
        spot0=spot0,
        start=bars[-1].available_at,
        as_of=bars[-1].ts.date(),
        rng=np.random.default_rng(5),
        horizons=[5],
        n_paths=500,
        target_sigma_by_horizon={5: 0.0001},
    )

    # Every cell of the standardised grid must still be present.
    assert len(evals) == len(universe)
    assert any(e.target_sigma_met is False for e in evals)
    for e in evals:
        assert e.target_sigma_met is False
        assert e.realized_sigma is not None
        # Clamped: the achieved sigma is far above the unreachable requested target.
        assert e.realized_sigma > e.target_sigma  # type: ignore[operator]


# ---------------------------------------------------------------------------
# 2026Q4-002 §2/§5: the falsification numeric rule and the four verdict branches
# ---------------------------------------------------------------------------


def test_falsification_shows_effect_when_significant_and_positive() -> None:
    shows_effect, reason = sev._falsification_shows_effect(0.02, 0.01, 0.10)
    assert shows_effect is True
    assert "shows an effect" in reason


def test_falsification_shows_no_effect_when_not_significant() -> None:
    shows_effect, _ = sev._falsification_shows_effect(0.50, 0.01, 0.10)
    assert shows_effect is False


def test_falsification_shows_no_effect_when_delta_not_positive() -> None:
    shows_effect, _ = sev._falsification_shows_effect(0.01, 0.0, 0.10)
    assert shows_effect is False
    shows_effect, _ = sev._falsification_shows_effect(0.01, -0.01, 0.10)
    assert shows_effect is False


def test_falsification_boundary_p_equals_alpha_is_an_effect() -> None:
    shows_effect, _ = sev._falsification_shows_effect(0.10, 0.001, 0.10)
    assert shows_effect is True


def test_verdict_with_falsification_pass_when_no_confound() -> None:
    verdict, reason = sev._verdict_with_falsification(0.05, 0.01, 0.10, False)
    assert verdict == "PASS"
    assert "row 1" in reason


def test_verdict_with_falsification_confounded_when_falsification_also_fires() -> None:
    verdict, reason = sev._verdict_with_falsification(0.05, 0.01, 0.10, True)
    assert verdict == "CONFOUNDED"
    assert "row 2" in reason


def test_verdict_with_falsification_fail_null_result_regardless_of_falsification() -> None:
    verdict, reason = sev._verdict_with_falsification(0.50, 0.01, 0.10, True)
    assert verdict == "FAIL_NULL_RESULT"
    assert "row 3" in reason
    verdict, _ = sev._verdict_with_falsification(0.50, 0.01, 0.10, False)
    assert verdict == "FAIL_NULL_RESULT"


def test_verdict_with_falsification_fail_negative_significant_regardless_of_falsification() -> None:
    verdict, reason = sev._verdict_with_falsification(0.02, -0.01, 0.10, True)
    assert verdict == "FAIL_NEGATIVE_SIGNIFICANT"
    assert "row 4" in reason
    verdict, _ = sev._verdict_with_falsification(0.02, -0.01, 0.10, False)
    assert verdict == "FAIL_NEGATIVE_SIGNIFICANT"


# ---------------------------------------------------------------------------
# 2026Q4-002 mandatory anti-triviality checks
# ---------------------------------------------------------------------------


def test_anti_triviality_narrower_sigma_genuinely_improves_payoff_passes() -> None:
    """(1) A rigged pair of forecasts, identical in mean and uncertainty and differing only in
    sigma (the challenger narrower), whose narrower dispersion genuinely improves the priced
    grid, must reach PASS: significant positive delta under "own sigma per arm", and the §6
    falsification arm (both forced to the null's sigma) shows no effect, since forcing sigma
    equal removes the only difference between the two forecasts.

    Restricted to the LONG half of the grid, same reasoning as the 2026Q4-001 anti-triviality
    test (a whole-grid average would partially cancel a symmetric effect by construction --
    a fact about the priced grid, not the machinery under test) -- and, empirically, to the
    farther barrier distances (10/15/20%): a direct probe of this rig (``dispersion's effect on
    a knock-out-truncated payoff is not monotonic in barrier distance`` -- narrowing sigma
    *hurts* the near-barrier cells here and *helps* the far-barrier ones, sensible for a
    convex, KO-truncated payoff where far-from-barrier cells behave more like the "wider
    dispersion raises a convex payoff's expectation" case) shows the near-barrier cells (2%, 5%)
    would swamp the far ones with the opposite sign at this vol/mean setting. Picking the
    subset where the rig's claimed direction (narrower is better) actually holds is exactly
    what "genuinely improves the payoff" requires this test to verify, not assume.
    """
    bars = _bars(900, seed=201, daily_vol=0.008)
    cfg = SyntheticTurboConfig(barrier_distances=(0.10, 0.15, 0.20), n_paths=300)

    null_forecast = {5: _make_forecast(0.0, horizon_days=5, sigma=0.05, uncertainty=0.01)}
    better_forecast = {5: _make_forecast(0.0, horizon_days=5, sigma=0.02, uncertainty=0.01)}
    null_sigma = {5: 0.05}
    better_sigma = {5: 0.02}
    falsification_sigma = dict(null_sigma)

    own_sigma_deltas: list[float] = []
    falsification_deltas: list[float] = []
    for i in range(800, 860, 6):  # 10 synthetic "dates" drawn from one long history
        bar = bars[i]
        grid = build_standardised_universe(bar.close, config=cfg)
        grid_long = {isin: t for isin, t in grid.items() if t.direction == Direction.LONG}
        start = bar.available_at
        as_of = bar.ts.date()

        null_own = sev._evaluate_grid_with_target_sigma(
            grid_long,
            null_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=null_sigma,
        )
        better_own = sev._evaluate_grid_with_target_sigma(
            grid_long,
            better_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=better_sigma,
        )
        own_sigma_deltas.append(
            float(np.mean([e.lcb_net_return for e in better_own]))
            - float(np.mean([e.lcb_net_return for e in null_own]))
        )

        null_f = sev._evaluate_grid_with_target_sigma(
            grid_long,
            null_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=falsification_sigma,
        )
        better_f = sev._evaluate_grid_with_target_sigma(
            grid_long,
            better_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=falsification_sigma,
        )
        falsification_deltas.append(
            float(np.mean([e.lcb_net_return for e in better_f]))
            - float(np.mean([e.lcb_net_return for e in null_f]))
        )

    own_delta = np.asarray(own_sigma_deltas)
    mean_own_delta = float(np.mean(own_delta))
    assert mean_own_delta > 0.0  # sanity: the rig actually produced an advantage

    p_own = sev._moving_block_bootstrap_p_value(
        own_delta, block_length=5, n_resamples=2000, rng=np.random.default_rng(0)
    )

    falsification_delta = np.asarray(falsification_deltas)
    # Both models are forced to identical (mean, uncertainty, target_sigma) inputs under the
    # same rng seed in the falsification arm -- bit-identical paths, hence exactly zero delta.
    assert np.allclose(falsification_delta, 0.0, atol=1e-9)
    p_falsification = sev._moving_block_bootstrap_p_value(
        falsification_delta, block_length=5, n_resamples=2000, rng=np.random.default_rng(1)
    )
    shows_effect, _ = sev._falsification_shows_effect(
        p_falsification, float(np.mean(falsification_delta)), 0.10
    )
    assert shows_effect is False

    verdict, _ = sev._verdict_with_falsification(p_own, mean_own_delta, 0.10, shows_effect)
    assert verdict == "PASS"


def test_anti_triviality_effect_surviving_falsification_is_confounded() -> None:
    """(2) This is the test that proves the falsification arm can actually fire (without it,
    §2's defence against the known bias is decorative): rig a pair of forecasts whose primary
    ("own sigma per arm") delta is driven by a real *mean* difference between the two models
    (the same channel 2026Q4-001 tested) as well as a sigma difference. Forcing both arms to
    the null's own sigma in the falsification arm neutralises only the sigma channel -- the mean
    difference is untouched, so the same significant, positive-signed effect survives, and the
    correct verdict is CONFOUNDED: the effect is not attributable to width.

    (A rig that differs *only* in sigma, with an identical mean and uncertainty, cannot produce
    CONFOUNDED under a correctly implemented falsification arm: forcing sigma equal in that case
    makes the two arms' inputs -- and, under the shared rng seed this harness always uses,
    their simulated paths -- fully identical, which is exactly
    ``test_anti_triviality_narrower_sigma_genuinely_improves_payoff_passes``'s PASS case above.
    A model pair that differs only in sigma is therefore the wrong rig to prove CONFOUNDED can
    fire at all; a mean difference that survives sigma-equalisation is the direct, defensible
    construction of a case where the primary's effect is genuinely not (solely) attributable to
    width, which is the property this test exists to demonstrate.)
    """
    bars = _bars(900, seed=202, daily_vol=0.008)
    cfg = SyntheticTurboConfig(n_paths=300)

    null_forecast = {5: _make_forecast(0.0, horizon_days=5, sigma=0.05, uncertainty=0.01)}
    better_forecast = {5: _make_forecast(0.04, horizon_days=5, sigma=0.02, uncertainty=0.01)}
    null_sigma = {5: 0.05}
    better_sigma = {5: 0.02}
    falsification_sigma = dict(null_sigma)

    own_sigma_deltas: list[float] = []
    falsification_deltas: list[float] = []
    for i in range(800, 860, 6):
        bar = bars[i]
        grid = build_standardised_universe(bar.close, config=cfg)
        grid_long = {isin: t for isin, t in grid.items() if t.direction == Direction.LONG}
        start = bar.available_at
        as_of = bar.ts.date()

        null_own = sev._evaluate_grid_with_target_sigma(
            grid_long,
            null_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=null_sigma,
        )
        better_own = sev._evaluate_grid_with_target_sigma(
            grid_long,
            better_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=better_sigma,
        )
        own_sigma_deltas.append(
            float(np.mean([e.lcb_net_return for e in better_own]))
            - float(np.mean([e.lcb_net_return for e in null_own]))
        )

        null_f = sev._evaluate_grid_with_target_sigma(
            grid_long,
            null_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=falsification_sigma,
        )
        better_f = sev._evaluate_grid_with_target_sigma(
            grid_long,
            better_forecast,
            bars,
            underlying_id="DAX",
            spot0=bar.close,
            start=start,
            as_of=as_of,
            rng=np.random.default_rng(cfg.seed),
            horizons=[5],
            n_paths=cfg.n_paths,
            target_sigma_by_horizon=falsification_sigma,
        )
        falsification_deltas.append(
            float(np.mean([e.lcb_net_return for e in better_f]))
            - float(np.mean([e.lcb_net_return for e in null_f]))
        )

    own_delta = np.asarray(own_sigma_deltas)
    mean_own_delta = float(np.mean(own_delta))
    assert mean_own_delta > 0.0
    p_own = sev._moving_block_bootstrap_p_value(
        own_delta, block_length=5, n_resamples=2000, rng=np.random.default_rng(0)
    )

    falsification_delta = np.asarray(falsification_deltas)
    mean_falsification_delta = float(np.mean(falsification_delta))
    # The mean-driven advantage is untouched by forcing sigma equal -- it survives, clearly
    # positive, not zeroed like the pure-sigma-difference case above.
    assert mean_falsification_delta > 0.0
    p_falsification = sev._moving_block_bootstrap_p_value(
        falsification_delta, block_length=5, n_resamples=2000, rng=np.random.default_rng(1)
    )
    shows_effect, _ = sev._falsification_shows_effect(
        p_falsification, mean_falsification_delta, 0.10
    )
    assert shows_effect is True

    verdict, _ = sev._verdict_with_falsification(p_own, mean_own_delta, 0.10, shows_effect)
    assert verdict == "CONFOUNDED"


# ---------------------------------------------------------------------------
# 2026Q4-002: full-trial integration -- forecast mode populates falsification
# ---------------------------------------------------------------------------


def test_full_trial_forecast_mode_populates_falsification_and_sigma_diagnostics() -> None:
    cfg = SyntheticTurboConfig(n_paths=25, sigma_source="forecast")
    bars = {"DAX": _bars(800, seed=42, daily_vol=0.009)}
    result = run_synthetic_net_ev_trial(bars, config=cfg)

    assert result.sigma_source == "forecast"
    assert result.falsification is not None
    assert 0.0 <= result.falsification.p_value <= 1.0
    assert result.falsification.n_dates > 0
    assert result.verdict in ("PASS", "CONFOUNDED", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT")
    if result.verdict == "PASS":
        assert result.falsification.shows_effect is False
    if result.verdict == "CONFOUNDED":
        assert result.falsification.shows_effect is True

    stability = result.stability
    assert set(stability.mean_realized_sigma_by_arm).issubset({"null", "regime_conditional"})
    assert set(stability.mean_target_sigma_by_arm).issubset({"null", "regime_conditional"})
    # Structural separation still holds under the new fields too.
    assert "falsification" not in StabilityAnalysis.model_fields


def test_full_trial_null_forecast_mode_has_no_nested_falsification() -> None:
    cfg = SyntheticTurboConfig(n_paths=25, sigma_source="null_forecast")
    bars = {"DAX": _bars(800, seed=42, daily_vol=0.009)}
    result = run_synthetic_net_ev_trial(bars, config=cfg)

    assert result.sigma_source == "null_forecast"
    assert result.falsification is None
    assert result.verdict in ("PASS", "FAIL_NULL_RESULT", "FAIL_NEGATIVE_SIGNIFICANT")
