from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from turboedge.backtest.walkforward import WalkForwardResult, walk_forward_evaluate
from turboedge.models.baselines import RegimeConditionalEmpiricalModel
from turboedge.models.directional import NullModel, NullModelConfig, TsmomForecastModel
from turboedge.models.forecast import HorizonForecast
from turboedge.storage.schemas import UnderlyingBar


def test_walk_forward_evaluate_null_model_produces_sane_results(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(600, seed=7)
    results = walk_forward_evaluate(
        lambda: NullModel(NullModelConfig(min_train_samples=20)),
        bars,
        horizons=[5, 10],
        min_train=200,
        step=50,
        embargo=10,
    )
    assert len(results) == 2
    for res in results:
        assert isinstance(res, WalkForwardResult)
        assert res.n_folds > 0
        assert 0.0 <= res.brier <= 1.0
        assert res.log_loss >= 0.0
        assert 0.0 <= res.ece <= 1.0
        assert 0.0 <= res.hit_rate <= 1.0
        assert 0.0 <= res.psr <= 1.0
        assert len(res.oos_predictions) > 0
        for _t, p_up, realized_return in res.oos_predictions:
            assert 0.0 <= p_up <= 1.0
            assert np.isfinite(realized_return)


def test_walk_forward_evaluate_tsmom_model_runs_end_to_end(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(900, seed=11)
    results = walk_forward_evaluate(
        TsmomForecastModel,
        bars,
        horizons=[5],
        min_train=350,
        step=150,
        embargo=5,
    )
    assert len(results) == 1
    res = results[0]
    assert res.model_id == "tsmom_forecast_v1"
    assert res.horizon_days == 5
    assert res.n_folds > 0


def test_walk_forward_evaluate_collects_full_distribution_alongside_p_up(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """Phase D: the harness must collect the full predictive distribution
    (oos_forecasts) and score it (crps/pinball/coverage), while p_up and the
    binary metrics (brier/log_loss/ece/hit_rate) remain exactly as before --
    this is required to stay a pure addition, never a replacement."""
    bars = make_bars(600, seed=9)
    results = walk_forward_evaluate(
        lambda: NullModel(NullModelConfig(min_train_samples=20)),
        bars,
        horizons=[5, 10],
        min_train=200,
        step=50,
        embargo=10,
    )
    for res in results:
        # -- diagnostic fields, unchanged in shape/semantics --
        assert 0.0 <= res.brier <= 1.0
        assert res.log_loss >= 0.0
        assert 0.0 <= res.ece <= 1.0
        assert len(res.oos_predictions) > 0
        for _t, p_up, realized_return in res.oos_predictions:
            assert 0.0 <= p_up <= 1.0
            assert np.isfinite(realized_return)

        # -- Phase D additions --
        assert len(res.oos_forecasts) == len(res.oos_predictions)
        for (t_a, p_up, ret_a), (t_b, fc, ret_b) in zip(
            res.oos_predictions, res.oos_forecasts, strict=True
        ):
            assert t_a == t_b
            assert ret_a == pytest.approx(ret_b)
            assert isinstance(fc, HorizonForecast)
            assert fc.p_up == pytest.approx(p_up)

        assert np.isfinite(res.crps)
        assert res.crps >= 0.0
        assert set(res.pinball_by_quantile) == {"q05", "q25", "q50", "q75", "q95"}
        for loss in res.pinball_by_quantile.values():
            assert loss >= 0.0
        assert 0.0 <= res.coverage_90 <= 1.0
        assert 0.0 <= res.coverage_50 <= 1.0
        # A well-specified in-family model's OOS coverage should land
        # reasonably close to its own nominal level (loose bound -- this is
        # a sanity check, not a strict calibration test).
        assert res.coverage_90 > 0.5
        assert res.coverage_50 > 0.2


def test_walk_forward_evaluate_new_baseline_model_produces_distribution(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    """One of the Phase D pre-registered baselines runs end-to-end through
    the same harness as the existing models."""
    bars = make_bars(700, seed=13)
    results = walk_forward_evaluate(
        RegimeConditionalEmpiricalModel,
        bars,
        horizons=[5],
        min_train=250,
        step=60,
        embargo=5,
    )
    assert len(results) == 1
    res = results[0]
    assert res.model_id == "regime_conditional_empirical_v1"
    assert np.isfinite(res.crps)
    assert len(res.oos_forecasts) > 0


def test_walk_forward_evaluate_raises_when_insufficient_history(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(50, seed=1)
    with pytest.raises(ValueError):
        walk_forward_evaluate(
            lambda: NullModel(NullModelConfig(min_train_samples=5)),
            bars,
            horizons=[5],
            min_train=1000,  # far more than available
            step=10,
            embargo=5,
        )
