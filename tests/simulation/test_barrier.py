from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from turboedge.simulation.barrier import (
    ambiguous_days,
    brownian_bridge_hit_probability,
    first_hit_index,
)
from turboedge.simulation.paths import PathSet
from turboedge.storage.schemas import Direction


def _pathset(open_, high, low, close) -> PathSet:
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return PathSet(
        underlying_id="TEST",
        spot0=float(open_[0, 0]),
        start=datetime(2025, 1, 1, tzinfo=UTC),
        open=open_,
        high=high,
        low=low,
        close=close,
        weekend_before=np.zeros(open_.shape[1], dtype=np.bool_),
        method="manual",
    )


def test_first_hit_index_long_low_touch() -> None:
    # Path 0: never touches. Path 1: low touches barrier on day 2 (idx 1).
    ps = _pathset(
        open_=[[100, 101], [100, 99]],
        high=[[101, 102], [100, 99.5]],
        low=[[99, 100], [98, 90]],
        close=[[100.5, 101.5], [99, 91]],
    )
    idx = first_hit_index(ps, barrier=95.0, direction=Direction.LONG)
    assert idx.tolist() == [-1, 1]


def test_first_hit_index_long_gap_through_at_open() -> None:
    # Open already at/below barrier on day 0 -> hit at day 0, regardless of low.
    ps = _pathset(
        open_=[[90.0, 91.0]],
        high=[[91.0, 92.0]],
        low=[[89.0, 90.5]],
        close=[[90.5, 91.5]],
    )
    idx = first_hit_index(ps, barrier=95.0, direction=Direction.LONG)
    assert idx.tolist() == [0]


def test_first_hit_index_short_high_touch_and_gap_through() -> None:
    ps = _pathset(
        open_=[[100, 101], [110, 109]],
        high=[[101, 106], [111, 109.5]],
        low=[[99, 100], [109, 108]],
        close=[[100.5, 105], [110.5, 109]],
    )
    idx = first_hit_index(ps, barrier=105.0, direction=Direction.SHORT)
    # Path 0: high touches 105 on day 1 (idx 1). Path 1: open already >= 105 on day 0.
    assert idx.tolist() == [1, 0]


def test_first_hit_index_returns_first_touch_not_last() -> None:
    ps = _pathset(
        open_=[[100, 90, 80]],
        high=[[101, 91, 81]],
        low=[[99, 89, 79]],
        close=[[100.5, 90.5, 80.5]],
    )
    idx = first_hit_index(ps, barrier=95.0, direction=Direction.LONG)
    assert idx.tolist() == [1]


def test_first_hit_index_long_short_symmetry() -> None:
    spot0 = 100.0
    barrier_long = 90.0
    barrier_short = 110.0
    open_up = [[100, 105, 111]]
    open_down = [[100, 95, 89]]
    long_ps = _pathset(
        open_=open_down,
        high=[[101, 96, 90]],
        low=[[99, 94, 88]],
        close=[[99.5, 94.5, 88.5]],
    )
    short_ps = _pathset(
        open_=open_up,
        high=[[101, 106, 112]],
        low=[[99, 104, 110]],
        close=[[100.5, 105.5, 111.5]],
    )
    idx_long = first_hit_index(long_ps, barrier_long, Direction.LONG)
    idx_short = first_hit_index(short_ps, barrier_short, Direction.SHORT)
    assert idx_long.tolist() == idx_short.tolist() == [2]
    assert spot0 == 100.0  # documents the mirrored setup


def test_brownian_bridge_hand_example() -> None:
    s0, s1, barrier, sigma = 100.0, 100.0, 90.0, 0.1
    expected = math.exp(-2.0 * math.log(s0 / barrier) * math.log(s1 / barrier) / sigma**2)
    got = brownian_bridge_hit_probability(s0, s1, barrier, sigma, Direction.LONG)
    assert got == pytest.approx(expected)
    assert 0.0 < got < 1.0


def test_brownian_bridge_certain_when_endpoint_already_through_long() -> None:
    assert brownian_bridge_hit_probability(100.0, 100.0, 105.0, 0.1, Direction.LONG) == 1.0
    assert brownian_bridge_hit_probability(100.0, 110.0, 105.0, 0.1, Direction.LONG) == 1.0


def test_brownian_bridge_certain_when_endpoint_already_through_short() -> None:
    assert brownian_bridge_hit_probability(100.0, 100.0, 95.0, 0.1, Direction.SHORT) == 1.0
    assert brownian_bridge_hit_probability(100.0, 90.0, 95.0, 0.1, Direction.SHORT) == 1.0


def test_brownian_bridge_short_side_symmetric_formula() -> None:
    s0, s1, barrier, sigma = 100.0, 102.0, 110.0, 0.08
    expected = math.exp(-2.0 * math.log(s0 / barrier) * math.log(s1 / barrier) / sigma**2)
    got = brownian_bridge_hit_probability(s0, s1, barrier, sigma, Direction.SHORT)
    assert got == pytest.approx(expected)


def test_brownian_bridge_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        brownian_bridge_hit_probability(0.0, 100.0, 90.0, 0.1, Direction.LONG)
    with pytest.raises(ValueError):
        brownian_bridge_hit_probability(100.0, 100.0, 90.0, 0.0, Direction.LONG)


def test_ambiguous_days_flags_day_touching_both_levels() -> None:
    ps = _pathset(
        open_=[[100.0]],
        high=[[105.0]],
        low=[[85.0]],
        close=[[95.0]],
    )
    flags = ambiguous_days(ps, barrier=90.0, other_level=100.0, direction=Direction.LONG)
    assert flags.tolist() == [[True]]


def test_ambiguous_days_false_when_only_one_level_touched() -> None:
    ps = _pathset(
        open_=[[100.0]],
        high=[[101.0]],
        low=[[95.0]],
        close=[[99.0]],
    )
    flags = ambiguous_days(ps, barrier=90.0, other_level=100.5, direction=Direction.LONG)
    assert flags.tolist() == [[False]]
