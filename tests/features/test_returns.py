from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.features.returns import bars_as_of, build_feature_frame, log_return_nd
from turboedge.storage.schemas import UnderlyingBar


def test_log_return_nd_basic() -> None:
    closes = np.array([100.0, 101.0, 102.0, 99.0])
    out = log_return_nd(closes, 2)
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(np.log(102.0 / 100.0))
    assert out[3] == pytest.approx(np.log(99.0 / 101.0))


def test_log_return_nd_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        log_return_nd(np.array([1.0, 2.0]), 0)


def test_bars_as_of_filters_and_sorts(make_bars: Callable[..., list[UnderlyingBar]]) -> None:
    bars = make_bars(20)
    cutoff = bars[10].available_at
    eligible = bars_as_of(list(reversed(bars)), cutoff)
    assert [b.ts for b in eligible] == [b.ts for b in bars[:11]]


def test_build_feature_frame_shape_and_names(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(200)
    dates, x, names = build_feature_frame(bars)
    assert len(dates) == 200
    assert x.shape == (200, len(names))
    assert len(set(names)) == len(names)  # no duplicate feature names
    # Early rows must be NaN (insufficient history), never imputed.
    assert np.isnan(x[0]).all()
    # Enough history at the end -> fully populated row.
    assert not np.isnan(x[-1]).any()


def test_build_feature_frame_rejects_empty() -> None:
    with pytest.raises(ValueError):
        build_feature_frame([])


def test_build_feature_frame_rejects_mixed_underlying(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars_a = make_bars(5, underlying_id="A")
    bars_b = make_bars(5, underlying_id="B", start=bars_a[-1].ts + timedelta(days=1))
    with pytest.raises(ValueError):
        build_feature_frame(bars_a + bars_b)


def test_build_feature_frame_rejects_unsorted(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(10)
    with pytest.raises(ValueError):
        build_feature_frame(list(reversed(bars)))


def test_build_feature_frame_no_lookahead(
    make_bars: Callable[..., list[UnderlyingBar]],
) -> None:
    bars = make_bars(300)
    t = 200
    _dates_prefix, x_prefix, names_prefix = build_feature_frame(bars[: t + 1])

    future_bars = make_bars(50, seed=999, start=bars[t].ts + timedelta(days=1))
    _dates_full, x_full, names_full = build_feature_frame(bars[: t + 1] + future_bars)

    assert names_prefix == names_full
    np.testing.assert_allclose(x_prefix[-1], x_full[t], rtol=0, atol=1e-12, equal_nan=True)


def test_build_feature_frame_single_bar_all_nan() -> None:
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    bar = UnderlyingBar(
        underlying_id="X",
        ts=ts,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        observation_time=ts,
        available_at=ts,
        retrieved_at=ts,
        source="synthetic",
        parser_version="1",
        quality_score=0.9,
    )
    _dates, x, _names = build_feature_frame([bar])
    assert x.shape[0] == 1
    assert np.isnan(x).all()
