from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from turboedge.backtest.walkforward import WalkForwardResult, walk_forward_evaluate
from turboedge.models.directional import NullModel, NullModelConfig, TsmomForecastModel
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
