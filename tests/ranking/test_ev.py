from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from turboedge.models.forecast import HorizonForecast
from turboedge.pricing.integrity import IntegrityReport
from turboedge.ranking.ev import (
    EvConfig,
    ProductHorizonEvaluation,
    evaluate_product_horizons,
    to_candidate_gate_input,
)
from turboedge.ranking.gates import GateThresholds, evaluate_gates
from turboedge.simulation.payoff import ProductTerms
from turboedge.storage.schemas import Category, Direction, ProductType, UnderlyingBar


def synthetic_daily_bars(
    n_days: int,
    *,
    start: date = date(2023, 1, 2),
    spot0: float = 100.0,
    daily_vol: float = 0.01,
    seed: int = 0,
) -> list[UnderlyingBar]:
    """Deterministic synthetic Mon-Fri daily OHLC bars (no network, seeded).

    Local copy of ``tests/simulation/conftest.py``'s helper -- kept
    independent here to avoid a cross-package test import (``tests/ranking``
    is not set up as a package importing from ``tests/simulation``).
    """
    rng = np.random.default_rng(seed)
    bars: list[UnderlyingBar] = []
    price = spot0
    d = start
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        ret = rng.normal(0.0, daily_vol)
        o = price
        c = price * float(np.exp(ret))
        hi = max(o, c) * float(np.exp(abs(rng.normal(0.0, 0.003))))
        lo = min(o, c) * float(np.exp(-abs(rng.normal(0.0, 0.003))))
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
    return bars


def simple_fair_value(
    *,
    direction: Direction,
    product_type: ProductType,
    spot: float,
    financing_level: float,
    knockout_barrier: float,
    ratio: float,
    fx: float,
    ref_rate: float,
    financing_spread: float,
    as_of: date,
    maturity: date | None,
    dividend_yield: float = 0.0,
) -> float:
    """Trivial intrinsic-only stand-in (no dependency on pricing/fair_value.py)."""
    del knockout_barrier, ref_rate, financing_spread, as_of, maturity, dividend_yield
    moneyness = spot - financing_level if direction == Direction.LONG else financing_level - spot
    return max(moneyness, 0.0) * ratio / fx


def make_forecast(
    horizon_days: int, mean: float, *, uncertainty: float = 0.01, sigma: float = 0.05
) -> HorizonForecast:
    return HorizonForecast(
        underlying_id="DAX",
        horizon_days=horizon_days,
        prediction_time=datetime(2024, 12, 2, 8, 0, tzinfo=UTC),
        p_up=0.5,
        mean=mean,
        sigma=sigma,
        quantiles={
            "q05": -2 * sigma,
            "q25": -0.5 * sigma,
            "q50": 0.0,
            "q75": 0.5 * sigma,
            "q95": 2 * sigma,
        },
        expected_shortfall_05=-2.5 * sigma,
        uncertainty=uncertainty,
        model_id="test_model",
        model_hash="hash",
        signal_family="test_family",
        n_train=500,
        n_effective=400.0,
    )


def make_terms(
    isin: str,
    *,
    direction: Direction = Direction.LONG,
    barrier: float = 70.0,
    financing: float = 70.0,
) -> ProductTerms:
    return ProductTerms(
        isin=isin,
        direction=direction,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=financing,
        knockout_barrier=barrier,
        ratio=1.0,
        fx=1.0,
        entry_ask=30.0,
        entry_bid=29.5,
        maturity=None,
        financing_spread=0.01,
        ref_rate=0.03,
        exit_spread_pct=0.02,
        premium_over_fair=0.0,
    )


BARS = synthetic_daily_bars(700, seed=1)
START = datetime(2024, 12, 1, 20, 0, tzinfo=UTC)
AS_OF = date(2024, 12, 1)
HORIZONS = (3, 5)
# block_bootstrap (not the production default vol_scaled_bootstrap) is used
# here only for test speed/simplicity; EvConfig's own default already
# matches simulate_paths's vol_scaled_bootstrap default (see ev.py).
FAST_CFG = EvConfig(n_paths=300, path_method="block_bootstrap")


def _forecasts(mean: float, *, uncertainty: float = 0.01) -> dict[int, HorizonForecast]:
    return {h: make_forecast(h, mean, uncertainty=uncertainty) for h in HORIZONS}


def test_ev_config_defaults_to_vol_scaled_bootstrap() -> None:
    # W5 calibration study (558 real DAX start dates): vol_scaled_bootstrap
    # is ~2.8x better calibrated for P(KO) than block_bootstrap at trading-
    # relevant barrier distances, and is simulate_paths's own default.
    # EvConfig must not silently pin the older, worse-calibrated method.
    assert EvConfig().path_method == "vol_scaled_bootstrap"


def test_evaluate_product_horizons_basic_shape() -> None:
    terms = {
        "DE000A1": make_terms("DE000A1", barrier=70.0),
        "DE000A2": make_terms("DE000A2", barrier=60.0),
    }
    evals = evaluate_product_horizons(
        terms,
        _forecasts(0.01),
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(1),
        horizons=HORIZONS,
        cfg=FAST_CFG,
        fair_value_fn=simple_fair_value,
    )
    assert len(evals) == len(terms) * len(HORIZONS)
    isins_horizons = {(e.isin, e.horizon_days) for e in evals}
    assert isins_horizons == {(isin, h) for isin in terms for h in HORIZONS}
    for e in evals:
        assert isinstance(e, ProductHorizonEvaluation)
        assert 0.0 <= e.p_ko <= 1.0
        assert 0.0 <= e.p_profit <= 1.0
        assert e.mc_standard_error >= 0.0
        assert 0.0 <= e.shrinkage_intensity <= 1.0
        assert e.suggested_position_fraction >= 0.0
        assert e.underlying_id == "DAX"


def test_lcb_never_exceeds_mean_net_return() -> None:
    terms = {"DE000A1": make_terms("DE000A1", barrier=70.0)}
    evals = evaluate_product_horizons(
        terms,
        _forecasts(0.02),
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(2),
        horizons=HORIZONS,
        cfg=FAST_CFG,
        fair_value_fn=simple_fair_value,
    )
    for e in evals:
        assert e.lcb_net_return <= e.mean_net_return + 1e-12


def test_higher_central_drift_increases_mean_net_return_for_long() -> None:
    terms = {"DE000A1": make_terms("DE000A1", direction=Direction.LONG, barrier=60.0)}
    low = evaluate_product_horizons(
        terms,
        _forecasts(0.0),
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(5),
        horizons=(5,),
        cfg=FAST_CFG,
        fair_value_fn=simple_fair_value,
    )[0]
    high = evaluate_product_horizons(
        terms,
        _forecasts(0.08),
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(5),
        horizons=(5,),
        cfg=FAST_CFG,
        fair_value_fn=simple_fair_value,
    )[0]
    assert high.mean_net_return > low.mean_net_return


def test_higher_central_drift_increases_mean_net_return_for_short() -> None:
    # For a SHORT product, a *lower* underlying drift is favorable, so the
    # mean net return should decrease as central drift rises.
    terms = {
        "DE000A1": make_terms("DE000A1", direction=Direction.SHORT, barrier=140.0, financing=140.0)
    }
    low = evaluate_product_horizons(
        terms,
        _forecasts(-0.08),
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(9),
        horizons=(5,),
        cfg=FAST_CFG,
        fair_value_fn=simple_fair_value,
    )[0]
    high = evaluate_product_horizons(
        terms,
        _forecasts(0.0),
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(9),
        horizons=(5,),
        cfg=FAST_CFG,
        fair_value_fn=simple_fair_value,
    )[0]
    assert low.mean_net_return > high.mean_net_return


def test_p_ko_increases_as_barrier_approaches_spot() -> None:
    terms = {
        "FAR": make_terms("FAR", direction=Direction.LONG, barrier=40.0, financing=40.0),
        "NEAR": make_terms("NEAR", direction=Direction.LONG, barrier=92.0, financing=92.0),
    }
    evals = evaluate_product_horizons(
        terms,
        {10: make_forecast(10, 0.0)},
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(3),
        horizons=(10,),
        cfg=FAST_CFG,
        fair_value_fn=simple_fair_value,
    )
    by_isin = {e.isin: e for e in evals}
    assert by_isin["NEAR"].p_ko > by_isin["FAR"].p_ko


def test_evaluate_product_horizons_is_deterministic_given_seed() -> None:
    terms = {"DE000A1": make_terms("DE000A1"), "DE000A2": make_terms("DE000A2", barrier=50.0)}

    def run() -> list[ProductHorizonEvaluation]:
        return evaluate_product_horizons(
            terms,
            _forecasts(0.01),
            BARS,
            underlying_id="DAX",
            spot0=100.0,
            start=START,
            as_of=AS_OF,
            cluster_id="corr_1",
            rng=np.random.default_rng(123),
            horizons=HORIZONS,
            cfg=FAST_CFG,
            fair_value_fn=simple_fair_value,
        )

    first = run()
    second = run()
    assert first == second


def test_evaluate_product_horizons_empty_terms_returns_empty_list() -> None:
    evals = evaluate_product_horizons(
        {},
        _forecasts(0.0),
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(1),
        horizons=HORIZONS,
        cfg=FAST_CFG,
    )
    assert evals == []


def test_evaluate_product_horizons_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError):
        evaluate_product_horizons(
            {"A": make_terms("A")},
            _forecasts(0.0),
            BARS,
            underlying_id="DAX",
            spot0=100.0,
            start=START,
            as_of=AS_OF,
            cluster_id="corr_1",
            rng=np.random.default_rng(1),
            horizons=(),
            cfg=FAST_CFG,
        )


def test_evaluate_product_horizons_rejects_missing_forecast() -> None:
    with pytest.raises(ValueError):
        evaluate_product_horizons(
            {"A": make_terms("A")},
            {3: make_forecast(3, 0.0)},
            BARS,
            underlying_id="DAX",
            spot0=100.0,
            start=START,
            as_of=AS_OF,
            cluster_id="corr_1",
            rng=np.random.default_rng(1),
            horizons=(3, 5),
            cfg=FAST_CFG,
        )


def test_to_candidate_gate_input_enables_actionable_when_all_gates_pass() -> None:
    evaluation = ProductHorizonEvaluation(
        isin="DE000A1",
        underlying_id="DAX",
        direction=Direction.LONG,
        horizon_days=5,
        mean_net_return=0.05,
        median_net_return=0.04,
        q05=-0.1,
        q95=0.2,
        p_profit=0.6,
        p_ko=0.05,
        es95=-0.12,
        mfe_median=0.05,
        mae_median=-0.03,
        mc_standard_error=0.001,
        shrunk_mean=0.045,
        shrinkage_intensity=0.2,
        lcb_net_return=0.01,
        utility=0.01,
        liquidity_factor=0.9,
        score=0.009,
        suggested_position_fraction=0.02,
        cluster_id="corr_1",
        reasons=["test"],
    )
    gate_input = to_candidate_gate_input(
        evaluation,
        integrity=IntegrityReport(passed=True),
        bid_only=False,
        knocked_out=False,
        quote_age_s=5.0,
        spread_pct=0.01,
        leverage=4.0,
        distance_to_barrier_sigma=2.0,
        data_health_pass=True,
        cluster_risk_pass=True,
    )
    assert gate_input.lcb_ev == evaluation.lcb_net_return
    assert gate_input.p_ko == evaluation.p_ko

    th = GateThresholds(
        max_spread_pct=0.03,
        max_quote_age_s=120,
        min_leverage=2.0,
        max_leverage=20.0,
        min_distance_to_barrier_sigma=1.0,
    )
    category, reasons = evaluate_gates(gate_input, th)
    assert category == Category.ACTIONABLE, reasons


def test_to_candidate_gate_input_stays_watch_when_cluster_risk_fails() -> None:
    evaluation = ProductHorizonEvaluation(
        isin="DE000A1",
        underlying_id="DAX",
        direction=Direction.LONG,
        horizon_days=5,
        mean_net_return=0.05,
        median_net_return=0.04,
        q05=-0.1,
        q95=0.2,
        p_profit=0.6,
        p_ko=0.05,
        es95=-0.12,
        mfe_median=0.05,
        mae_median=-0.03,
        mc_standard_error=0.001,
        shrunk_mean=0.045,
        shrinkage_intensity=0.2,
        lcb_net_return=0.01,
        utility=0.01,
        liquidity_factor=0.9,
        score=0.009,
        suggested_position_fraction=0.02,
        cluster_id="corr_1",
        reasons=["test"],
    )
    gate_input = to_candidate_gate_input(
        evaluation,
        integrity=IntegrityReport(passed=True),
        bid_only=False,
        knocked_out=False,
        quote_age_s=5.0,
        spread_pct=0.01,
        leverage=4.0,
        distance_to_barrier_sigma=2.0,
        data_health_pass=True,
        cluster_risk_pass=False,
    )
    th = GateThresholds(
        max_spread_pct=0.03,
        max_quote_age_s=120,
        min_leverage=2.0,
        max_leverage=20.0,
        min_distance_to_barrier_sigma=1.0,
    )
    category, reasons = evaluate_gates(gate_input, th)
    assert category == Category.WATCH
    assert "cluster_risk_not_confirmed" in reasons


@pytest.mark.slow
def test_evaluate_product_horizons_performance_proxy() -> None:
    """Performance proxy for the Build Contract v2 W7 target ("< 90s for one
    underlying's ~4000 candidate products x 5 horizons on a 2-core runner").

    Running the literal 4000 x 5 x (real) n_paths=2000 scale here would take
    on the order of many minutes (see Kurzbericht): ``simulate_product_
    payoff`` (``simulation/payoff.py``, W5/frozen for this workstream)
    evaluates ``fair_value_fn`` via a **Python-level loop over every
    simulated path, for every simulated day**, not a vectorized call -- a
    cost inherent to that module, independent of anything this module (path
    reuse across products) can optimize away. This test instead measures a
    smaller, still-realistic scale (300 products x 5 horizons, 300 paths)
    and asserts a generous wall-clock bound, then reports the observed
    per-(product, horizon) cost so it can be linearly extrapolated to the
    full contract scale in the Kurzbericht.
    """
    n_products = 300
    horizons = (3, 5, 7, 10, 14)
    terms = {
        f"DE000PERF{i:04d}": make_terms(f"DE000PERF{i:04d}", barrier=100.0 - (i % 30 + 5))
        for i in range(n_products)
    }
    forecasts = {h: make_forecast(h, 0.01) for h in horizons}
    cfg = EvConfig(n_paths=300)

    t0 = time.perf_counter()
    evals = evaluate_product_horizons(
        terms,
        forecasts,
        BARS,
        underlying_id="DAX",
        spot0=100.0,
        start=START,
        as_of=AS_OF,
        cluster_id="corr_1",
        rng=np.random.default_rng(42),
        horizons=horizons,
        cfg=cfg,
        fair_value_fn=simple_fair_value,
    )
    elapsed = time.perf_counter() - t0

    assert len(evals) == n_products * len(horizons)
    per_pair_ms = 1000.0 * elapsed / len(evals)
    print(
        f"\n[perf] evaluate_product_horizons: {len(evals)} (product, horizon) pairs "
        f"in {elapsed:.2f}s ({per_pair_ms:.3f} ms/pair, n_paths={cfg.n_paths})"
    )
    # Generous bound for CI variance; this scale is well below the full
    # contract target, see docstring.
    assert elapsed < 120.0
