from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from turboedge.storage.schemas import UnderlyingBar


def make_bar(day: date, open_: float, high: float, low: float, close: float) -> UnderlyingBar:
    ts = datetime(day.year, day.month, day.day, 22, 0, tzinfo=UTC)
    return UnderlyingBar(
        underlying_id="DAX",
        ts=ts,
        interval="1d",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        observation_time=ts,
        available_at=ts,
        retrieved_at=ts,
        source="test",
        parser_version="1",
        quality_score=1.0,
    )


@pytest.fixture
def bar_factory() -> Callable[[date, float, float, float, float], UnderlyingBar]:
    return make_bar


def synthetic_daily_bars(
    n_days: int,
    *,
    start: date = date(2023, 1, 2),
    spot0: float = 15000.0,
    daily_vol: float = 0.01,
    drift: float = 0.0,
    range_vol: float = 0.003,
    seed: int = 0,
) -> list[UnderlyingBar]:
    """Deterministic synthetic Mon-Fri daily OHLC bars (no network, seeded).

    Weekends are skipped in the calendar (so consecutive bars naturally
    produce both weeknight and weekend gaps), and each day's High/Low are
    built to be consistent with its Open/Close by construction.
    """
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
        hi = max(o, c) * float(np.exp(abs(rng.normal(0.0, range_vol))))
        lo = min(o, c) * float(np.exp(-abs(rng.normal(0.0, range_vol))))
        bars.append(make_bar(d, o, hi, lo, c))
        price = c
        d += timedelta(days=1)
    return bars


@pytest.fixture
def synthetic_bars() -> Callable[..., list[UnderlyingBar]]:
    return synthetic_daily_bars
