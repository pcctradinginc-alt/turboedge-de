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
