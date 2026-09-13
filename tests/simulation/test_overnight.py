from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from turboedge.simulation.overnight import DailyComponents, daily_components_from_bars

from .conftest import make_bar


def test_daily_components_basic_gap_and_intraday() -> None:
    # Mon(2) -> Tue(3) -> Wed(4): ordinary weeknight gaps.
    bars = [
        make_bar(date(2024, 1, 1), open_=100.0, high=101.0, low=99.5, close=100.5),  # Monday
        make_bar(date(2024, 1, 2), open_=100.6, high=102.0, low=100.0, close=101.5),  # Tuesday
        make_bar(date(2024, 1, 3), open_=101.4, high=103.0, low=101.0, close=102.5),  # Wednesday
    ]
    dc = daily_components_from_bars(bars)
    assert isinstance(dc, DailyComponents)
    assert dc.n_discarded == 0
    assert dc.gap.size == 2  # no gap for the first bar (needs a previous close)
    assert dc.gap[0] == pytest.approx(math.log(100.6 / 100.5))
    assert dc.intraday[0] == pytest.approx(math.log(101.5 / 100.6))
    assert dc.hi[0] == pytest.approx(math.log(102.0 / 100.6))
    assert dc.lo[0] == pytest.approx(math.log(100.0 / 100.6))
    assert not dc.weekend_before.any()
    assert dc.dates == [date(2024, 1, 2), date(2024, 1, 3)]


def test_daily_components_flags_weekend_gap() -> None:
    # Friday -> Monday: weekend gap.
    bars = [
        make_bar(date(2024, 1, 5), open_=100.0, high=101.0, low=99.0, close=100.5),  # Friday
        make_bar(date(2024, 1, 8), open_=101.0, high=102.0, low=100.5, close=101.5),  # Monday
    ]
    dc = daily_components_from_bars(bars)
    assert dc.gap.size == 1
    assert dc.weekend_before[0]
    assert dc.gap[0] == pytest.approx(math.log(101.0 / 100.5))


def test_daily_components_discards_inconsistent_bar() -> None:
    bars = [
        make_bar(date(2024, 1, 1), open_=100.0, high=101.0, low=99.5, close=100.5),
        # Broken bar: high below max(open, close).
        make_bar(date(2024, 1, 2), open_=100.6, high=100.55, low=100.0, close=101.5),
        make_bar(date(2024, 1, 3), open_=101.4, high=103.0, low=101.0, close=102.5),
    ]
    dc = daily_components_from_bars(bars)
    assert dc.n_discarded == 1
    # Only the Mon->Wed pair (bridging over the discarded Tuesday) is NOT
    # produced either, since the discarded bar is removed before pairing --
    # the remaining consistent bars are Mon and Wed, one gap between them.
    assert dc.gap.size == 1
    assert dc.dates == [date(2024, 1, 3)]


def test_daily_components_discards_low_above_open_close() -> None:
    bars = [
        make_bar(date(2024, 1, 1), open_=100.0, high=101.0, low=99.5, close=100.5),
        # Broken bar: low above min(open, close).
        make_bar(date(2024, 1, 2), open_=100.6, high=102.0, low=100.7, close=101.5),
    ]
    dc = daily_components_from_bars(bars)
    assert dc.n_discarded == 1
    assert dc.gap.size == 0


def test_daily_components_discards_non_positive_prices() -> None:
    bars = [
        make_bar(date(2024, 1, 1), open_=100.0, high=101.0, low=99.5, close=100.5),
        make_bar(date(2024, 1, 2), open_=0.0, high=1.0, low=0.0, close=0.5),
    ]
    dc = daily_components_from_bars(bars)
    assert dc.n_discarded == 1


def test_daily_components_handles_unsorted_input() -> None:
    bars = [
        make_bar(date(2024, 1, 3), open_=101.4, high=103.0, low=101.0, close=102.5),
        make_bar(date(2024, 1, 1), open_=100.0, high=101.0, low=99.5, close=100.5),
        make_bar(date(2024, 1, 2), open_=100.6, high=102.0, low=100.0, close=101.5),
    ]
    dc = daily_components_from_bars(bars)
    assert dc.n_discarded == 0
    assert dc.gap.size == 2
    assert dc.dates == [date(2024, 1, 2), date(2024, 1, 3)]


def test_daily_components_skips_duplicate_or_out_of_order_days() -> None:
    bars = [
        make_bar(date(2024, 1, 1), open_=100.0, high=101.0, low=99.5, close=100.5),
        make_bar(date(2024, 1, 1), open_=100.0, high=101.0, low=99.5, close=100.5),  # duplicate
        make_bar(date(2024, 1, 2), open_=100.6, high=102.0, low=100.0, close=101.5),
    ]
    dc = daily_components_from_bars(bars)
    assert dc.n_discarded == 1
    assert dc.gap.size == 1


def test_daily_components_field_shapes_stay_aligned() -> None:
    with pytest.raises(ValueError):
        DailyComponents(
            dates=[date(2024, 1, 1)],
            gap=np.array([0.0, 0.0]),
            weekend_before=np.array([False]),
            intraday=np.array([0.0]),
            hi=np.array([0.0]),
            lo=np.array([0.0]),
            n_discarded=0,
        )
