"""The meta layer must not read anything unavailable at prediction time.

CLAUDE.md rules 4/5. These tests deliberately append future bars and assert
that nothing the controller produces changes -- the same construction used
for the W12 feature builders, applied one level up.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from turboedge.meta.controller import decide
from turboedge.meta.regimes import (
    classify_trend_regime,
    classify_volatility_regime,
    count_similar_history,
)


def test_regime_classification_ignores_future_bars(
    bars_factory: Callable[..., list], start: datetime
) -> None:
    history = bars_factory(300)
    at = start + timedelta(days=299)
    vol_before = classify_volatility_regime(history, at)
    trend_before = classify_trend_regime(history, at)

    # A violent future crash: if any of it leaked in, the regime would move.
    crash = bars_factory(60, seed=99, drift=-0.05)
    shifted = [
        b.model_copy(
            update={
                "ts": b.ts + timedelta(days=400),
                "available_at": b.available_at + timedelta(days=400),
            }
        )
        for b in crash
    ]
    for b in shifted:
        assert b.available_at > at

    assert classify_volatility_regime([*history, *shifted], at) == vol_before
    assert classify_trend_regime([*history, *shifted], at) == trend_before


def test_regime_history_count_ignores_future_bars(
    bars_factory: Callable[..., list], start: datetime
) -> None:
    history = bars_factory(300)
    at = start + timedelta(days=299)
    vol = classify_volatility_regime(history, at)
    trend = classify_trend_regime(history, at)
    before = count_similar_history(history, at, vol, trend)

    future = [
        b.model_copy(
            update={
                "ts": b.ts + timedelta(days=400),
                "available_at": b.available_at + timedelta(days=400),
            }
        )
        for b in bars_factory(100, seed=5)
    ]
    assert count_similar_history([*history, *future], at, vol, trend) == before


def test_whole_decision_is_unchanged_by_future_bars(
    bars_factory: Callable[..., list],
    forecast_factory: Callable[..., object],
    entry_factory: Callable[..., object],
    start: datetime,
) -> None:
    """The end-to-end guarantee, not just its parts."""
    history = bars_factory(300)
    at = start + timedelta(days=299)
    kwargs = dict(
        run_id="r",
        underlying_id="DAX",
        horizon_days=7,
        prediction_time=at,
        registry=[entry_factory("m1")],
        forecasts=[forecast_factory("m1")],
        walkforward=[],
        data_quality=1.0,
        stale_share=0.0,
        ratio_unverified_share=0.0,
        integrity_fail_share=0.0,
        now=start,
    )
    baseline = decide(bars=history, **kwargs)
    future = [
        b.model_copy(
            update={
                "ts": b.ts + timedelta(days=400),
                "available_at": b.available_at + timedelta(days=400),
            }
        )
        for b in bars_factory(100, seed=11, drift=0.05)
    ]
    with_future = decide(bars=[*history, *future], **kwargs)
    assert baseline.model_dump() == with_future.model_dump()
