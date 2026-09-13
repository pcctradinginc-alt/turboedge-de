from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.storage.schemas import UnderlyingBar


def _make_bars(
    n: int,
    *,
    seed: int = 0,
    drift: float = 0.0002,
    daily_vol: float = 0.01,
    underlying_id: str = "TEST",
    start: datetime = datetime(2010, 1, 1, tzinfo=UTC),
) -> list[UnderlyingBar]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, daily_vol, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    bars = []
    for i in range(n):
        ts = start + timedelta(days=i)
        c = float(closes[i])
        o = c * (1.0 + rng.normal(0.0, 0.001))
        h = max(o, c) * (1.0 + abs(rng.normal(0.0, 0.002)))
        lo = min(o, c) * (1.0 - abs(rng.normal(0.0, 0.002)))
        bars.append(
            UnderlyingBar(
                underlying_id=underlying_id,
                ts=ts,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=1000.0,
                observation_time=ts,
                available_at=ts,
                retrieved_at=ts,
                source="synthetic",
                parser_version="1",
                quality_score=0.9,
            )
        )
    return bars


@pytest.fixture
def make_bars() -> Callable[..., list[UnderlyingBar]]:
    """Factory for a chronological list of synthetic ``UnderlyingBar`` (no network)."""
    return _make_bars
