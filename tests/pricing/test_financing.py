from __future__ import annotations

from datetime import UTC, datetime

import pytest

from turboedge.pricing.financing import (
    financing_cost_over_horizon,
    financing_spread_history,
    implied_financing_spread,
    realized_financing_spread,
)
from turboedge.storage.schemas import Direction


def _ts(day: int) -> datetime:
    return datetime(2026, 9, day, 12, 0, tzinfo=UTC)


def test_implied_financing_spread_long_roundtrip() -> None:
    f0, r, s, dt = 22000.0, 0.03, 0.025, 3.0
    f1 = f0 * (1.0 + (r + s) * dt / 360.0)
    recovered = implied_financing_spread(f0, f1, dt, r, Direction.LONG)
    assert recovered == pytest.approx(s, rel=1e-9)


def test_implied_financing_spread_short_roundtrip() -> None:
    f0, r, s, dt = 22000.0, 0.03, 0.018, 3.0
    # Short roll-forward is the inverse relation of the Long formula:
    # s = r - annualized_change  =>  annualized_change = r - s
    annualized_change = r - s
    f1 = f0 * (1.0 + annualized_change * dt / 360.0)
    recovered = implied_financing_spread(f0, f1, dt, r, Direction.SHORT)
    assert recovered == pytest.approx(s, rel=1e-9)


def test_implied_financing_spread_requires_positive_f_prev() -> None:
    with pytest.raises(ValueError):
        implied_financing_spread(0.0, 100.0, 1.0, 0.03, Direction.LONG)


def test_implied_financing_spread_requires_positive_days() -> None:
    with pytest.raises(ValueError):
        implied_financing_spread(100.0, 101.0, 0.0, 0.03, Direction.LONG)


def test_financing_spread_history_flags_dividend_jump() -> None:
    r, s, dt = 0.03, 0.02, 1.0
    f0 = 22000.0
    f1 = f0 * (1.0 + (r + s) * dt / 360.0)  # ordinary accrual, clean
    # A large, sudden drop in the financing level (dividend adjustment) that
    # does not correspond to a plausible funding spread.
    f2 = f1 * 0.85

    levels = [(_ts(1), f0), (_ts(2), f1), (_ts(3), f2)]
    observations = financing_spread_history(levels, r, Direction.LONG, max_daily_jump=0.08)

    assert len(observations) == 2
    assert observations[0].adjustment_suspected is False
    assert observations[0].spread == pytest.approx(s, rel=1e-6)
    assert observations[1].adjustment_suspected is True


def test_financing_spread_history_orders_unsorted_input() -> None:
    r = 0.03
    levels = [
        (_ts(2), 22100.0),
        (_ts(1), 22000.0),
    ]
    observations = financing_spread_history(levels, r, Direction.LONG, max_daily_jump=0.5)
    assert len(observations) == 1
    assert observations[0].start == _ts(1)
    assert observations[0].end == _ts(2)


def test_financing_spread_history_skips_nonpositive_gaps() -> None:
    r = 0.03
    levels = [(_ts(1), 22000.0), (_ts(1), 22001.0)]
    observations = financing_spread_history(levels, r, Direction.LONG, max_daily_jump=0.5)
    assert observations == []


def test_realized_financing_spread_is_median_of_clean_observations() -> None:
    r, dt = 0.03, 1.0
    f0 = 22000.0
    spreads = [0.02, 0.022, 0.025]
    levels: list[tuple[datetime, float]] = [(_ts(1), f0)]
    f = f0
    for i, s in enumerate(spreads, start=2):
        f = f * (1.0 + (r + s) * dt / 360.0)
        levels.append((_ts(i), f))
    observations = financing_spread_history(levels, r, Direction.LONG, max_daily_jump=0.5)
    result = realized_financing_spread(observations)
    assert result == pytest.approx(0.022, rel=1e-6)


def test_realized_financing_spread_none_without_clean_observations() -> None:
    r = 0.03
    f0 = 22000.0
    f1 = f0 * 3.0  # huge jump -> flagged
    levels = [(_ts(1), f0), (_ts(2), f1)]
    observations = financing_spread_history(levels, r, Direction.LONG, max_daily_jump=0.08)
    assert realized_financing_spread(observations) is None


def test_realized_financing_spread_none_for_empty_observations() -> None:
    assert realized_financing_spread([]) is None


def test_financing_cost_over_horizon_long() -> None:
    f, s, r, ratio = 22000.0, 0.02, 0.03, 0.01
    trading_days = 5
    calendar_days = trading_days * 7 / 5
    expected = f * (r + s) * calendar_days / 360.0 * ratio
    cost = financing_cost_over_horizon(f, s, r, trading_days, ratio, Direction.LONG)
    assert cost == pytest.approx(expected)


def test_financing_cost_over_horizon_short_can_be_negative() -> None:
    # spread below reference rate -> short holder earns carry (negative cost).
    f, s, r, ratio = 22000.0, 0.01, 0.03, 0.01
    cost = financing_cost_over_horizon(f, s, r, 5, ratio, Direction.SHORT)
    assert cost < 0.0


def test_financing_cost_over_horizon_rejects_negative_trading_days() -> None:
    with pytest.raises(ValueError):
        financing_cost_over_horizon(22000.0, 0.02, 0.03, -1, 0.01, Direction.LONG)
