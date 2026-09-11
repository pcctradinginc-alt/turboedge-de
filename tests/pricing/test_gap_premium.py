from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from turboedge.pricing.gap_premium import (
    GapDistribution,
    expected_gap_loss_per_night,
    fair_gap_premium,
    gap_distribution_from_bars,
    gap_premium_over_horizon,
)
from turboedge.storage.schemas import Direction, UnderlyingBar


def _bar(day_ts: datetime, open_: float, close: float) -> UnderlyingBar:
    return UnderlyingBar(
        underlying_id="DAX",
        ts=day_ts,
        interval="1d",
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume=1000.0,
        observation_time=day_ts,
        available_at=day_ts,
        retrieved_at=day_ts,
        source="test",
        parser_version="1",
        quality_score=1.0,
    )


def test_expected_gap_loss_per_night_zero_when_no_sample_breaches_f() -> None:
    # Spot far above F; even the worst (most negative) gap keeps S*e^g > F.
    spot, f, ratio = 24000.0, 10000.0, 0.01
    gaps = np.array([-0.01, -0.005, 0.0, 0.004, 0.01])
    loss = expected_gap_loss_per_night(spot, f, ratio, Direction.LONG, gaps)
    assert loss == pytest.approx(0.0)


def test_expected_gap_loss_per_night_synthetic_known_value() -> None:
    spot, f, ratio = 100.0, 100.0, 1.0
    # ln(0.99) and ln(1.01) roughly symmetric; only the down-gap breaches F for Long.
    gaps = np.array([math.log(0.99), math.log(1.01)])
    projected = spot * np.exp(gaps)
    expected = float(np.mean(np.maximum(f - projected, 0.0))) * ratio
    loss = expected_gap_loss_per_night(spot, f, ratio, Direction.LONG, gaps)
    assert loss == pytest.approx(expected)
    assert loss > 0.0


def test_expected_gap_loss_per_night_short_side() -> None:
    spot, f, ratio = 100.0, 100.0, 1.0
    gaps = np.array([math.log(0.99), math.log(1.01)])
    projected = spot * np.exp(gaps)
    expected = float(np.mean(np.maximum(projected - f, 0.0))) * ratio
    loss = expected_gap_loss_per_night(spot, f, ratio, Direction.SHORT, gaps)
    assert loss == pytest.approx(expected)


def test_expected_gap_loss_per_night_empty_sample_raises() -> None:
    with pytest.raises(ValueError):
        expected_gap_loss_per_night(100.0, 100.0, 1.0, Direction.LONG, np.array([]))


def test_gap_distribution_from_bars_separates_weekend() -> None:
    def d(day: int) -> datetime:
        return datetime(2026, 9, day, 22, 0, tzinfo=UTC)

    # Mon(7) Tue(8) Wed(9) Thu(10) Fri(11) -> Mon(14): weekday gaps Mon-Fri,
    # then a weekend gap Fri->Mon.
    bars = [
        _bar(d(7), open_=100.0, close=101.0),
        _bar(d(8), open_=101.2, close=102.0),
        _bar(d(9), open_=101.9, close=103.0),
        _bar(d(10), open_=103.3, close=104.0),
        _bar(d(11), open_=104.1, close=105.0),
        _bar(d(14), open_=106.0, close=107.0),  # Friday -> Monday: weekend gap
    ]
    dist = gap_distribution_from_bars(bars)
    assert isinstance(dist, GapDistribution)
    assert dist.weeknight_gaps.size == 4  # Mon->Tue, Tue->Wed, Wed->Thu, Thu->Fri
    assert dist.weekend_gaps.size == 1  # Fri->Mon
    assert dist.weekend_gaps[0] == pytest.approx(math.log(106.0 / 105.0))


def test_gap_distribution_from_bars_handles_unsorted_input() -> None:
    def d(day: int) -> datetime:
        return datetime(2026, 9, day, 22, 0, tzinfo=UTC)

    bars = [
        _bar(d(9), open_=102.0, close=103.0),
        _bar(d(7), open_=100.0, close=101.0),
        _bar(d(8), open_=101.1, close=102.0),
    ]
    dist = gap_distribution_from_bars(bars)
    assert dist.weeknight_gaps.size == 2


def test_fair_gap_premium_uses_weekend_sample_when_requested() -> None:
    spot, f, ratio = 100.0, 100.0, 1.0
    weeknight = np.array([0.0, 0.0])
    weekend = np.array([math.log(0.9), math.log(1.1)])
    dist = GapDistribution(weeknight_gaps=weeknight, weekend_gaps=weekend)

    weekday_premium = fair_gap_premium(
        spot, f, ratio, Direction.LONG, dist, next_night_is_weekend=False
    )
    weekend_premium = fair_gap_premium(
        spot, f, ratio, Direction.LONG, dist, next_night_is_weekend=True
    )

    assert weekday_premium == pytest.approx(0.0)
    assert weekend_premium > 0.0


def test_gap_premium_over_horizon_splits_weekend_and_weeknight() -> None:
    spot, f, ratio = 100.0, 100.0, 1.0
    weeknight = np.array([math.log(0.995)])
    weekend = np.array([math.log(0.95)])
    dist = GapDistribution(weeknight_gaps=weeknight, weekend_gaps=weekend)

    trading_days = 10  # -> 2 weekend nights, 8 weeknights
    total = gap_premium_over_horizon(spot, f, ratio, Direction.LONG, dist, trading_days)

    per_weeknight = expected_gap_loss_per_night(spot, f, ratio, Direction.LONG, weeknight)
    per_weekend = expected_gap_loss_per_night(spot, f, ratio, Direction.LONG, weekend)
    expected = 8 * per_weeknight + 2 * per_weekend
    assert total == pytest.approx(expected)


def test_gap_premium_over_horizon_zero_trading_days() -> None:
    dist = GapDistribution(weeknight_gaps=np.array([0.0]), weekend_gaps=np.array([0.0]))
    assert gap_premium_over_horizon(100.0, 100.0, 1.0, Direction.LONG, dist, 0) == 0.0


def test_gap_premium_over_horizon_rejects_negative_trading_days() -> None:
    dist = GapDistribution(weeknight_gaps=np.array([0.0]), weekend_gaps=np.array([0.0]))
    with pytest.raises(ValueError):
        gap_premium_over_horizon(100.0, 100.0, 1.0, Direction.LONG, dist, -1)
