from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.features.cross_asset import (
    align_auxiliary_series,
    own_level,
    own_level_diff_1d,
    own_log_return_1d,
)
from turboedge.storage.schemas import UnderlyingBar


def _bar(
    *, underlying_id: str, ts: datetime, available_at: datetime, close: float
) -> UnderlyingBar:
    return UnderlyingBar(
        underlying_id=underlying_id,
        ts=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        observation_time=available_at,
        available_at=available_at,
        retrieved_at=available_at,
        source="synthetic",
        parser_version="1",
        quality_score=0.9,
    )


def _day(d: int, hour: int = 0) -> datetime:
    return datetime(2020, 1, 1, hour, tzinfo=UTC) + timedelta(days=d)


# --- own_* transforms ---------------------------------------------------------------------


def test_own_log_return_1d() -> None:
    bars = [
        _bar(underlying_id="AUX", ts=_day(0), available_at=_day(0, 22), close=100.0),
        _bar(underlying_id="AUX", ts=_day(1), available_at=_day(1, 22), close=110.0),
        _bar(underlying_id="AUX", ts=_day(2), available_at=_day(2, 22), close=99.0),
    ]
    out = own_log_return_1d(bars)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(np.log(110.0 / 100.0))
    assert out[2] == pytest.approx(np.log(99.0 / 110.0))


def test_own_level_diff_1d() -> None:
    bars = [
        _bar(underlying_id="TNX", ts=_day(0), available_at=_day(0, 22), close=4.0),
        _bar(underlying_id="TNX", ts=_day(1), available_at=_day(1, 22), close=4.5),
        _bar(underlying_id="TNX", ts=_day(2), available_at=_day(2, 22), close=4.2),
    ]
    out = own_level_diff_1d(bars)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(-0.3)


def test_own_level() -> None:
    bars = [_bar(underlying_id="VIX", ts=_day(0), available_at=_day(0, 22), close=18.5)]
    out = own_level(bars)
    assert out.tolist() == pytest.approx([18.5])


# --- align_auxiliary_series: core no-look-ahead alignment ---------------------------------


def test_align_picks_most_recent_available_aux_value() -> None:
    # Primary (e.g. DAX) closes at 22:00 Europe/Berlin-equivalent each day (kept in UTC here).
    primary = [
        _bar(underlying_id="DAX", ts=_day(0), available_at=_day(0, 21), close=100.0),
        _bar(underlying_id="DAX", ts=_day(1), available_at=_day(1, 21), close=101.0),
        _bar(underlying_id="DAX", ts=_day(2), available_at=_day(2, 21), close=102.0),
    ]
    # Auxiliary (e.g. SPX) available slightly later each day (22:00), i.e. AFTER
    # the primary bar of the SAME day, but well before the NEXT day's primary bar.
    aux = [
        _bar(underlying_id="SPX", ts=_day(0), available_at=_day(0, 22), close=200.0),
        _bar(underlying_id="SPX", ts=_day(1), available_at=_day(1, 22), close=210.0),
        _bar(underlying_id="SPX", ts=_day(2), available_at=_day(2, 22), close=220.0),
    ]
    values = own_level(aux)
    out = align_auxiliary_series(primary, aux, values)
    # Day 0's primary prediction (as_of day0 21:00) is BEFORE aux day0 (22:00)
    # is available -> no aux data yet.
    assert np.isnan(out[0])
    # Day 1's primary prediction (day1 21:00) can see aux day0 (day0 22:00,
    # which is before day1 21:00) but NOT aux day1 (day1 22:00, after day1 21:00).
    assert out[1] == pytest.approx(200.0)
    # Day 2's primary prediction sees aux day1.
    assert out[2] == pytest.approx(210.0)


def test_align_ties_are_included() -> None:
    """available_at exactly equal to the primary's is included (<=, not <)."""
    ts0 = _day(0, 12)
    primary = [_bar(underlying_id="DAX", ts=_day(0), available_at=ts0, close=100.0)]
    aux = [_bar(underlying_id="SPX", ts=_day(0), available_at=ts0, close=200.0)]
    out = align_auxiliary_series(primary, aux, own_level(aux))
    assert out[0] == pytest.approx(200.0)


def test_align_no_aux_data_yet_is_nan() -> None:
    primary = [_bar(underlying_id="DAX", ts=_day(0), available_at=_day(0), close=100.0)]
    aux = [_bar(underlying_id="SPX", ts=_day(5), available_at=_day(5), close=200.0)]
    out = align_auxiliary_series(primary, aux, own_level(aux))
    assert np.isnan(out[0])


def test_align_empty_auxiliary_is_all_nan() -> None:
    primary = [_bar(underlying_id="DAX", ts=_day(0), available_at=_day(0), close=100.0)]
    out = align_auxiliary_series(primary, [], np.zeros(0, dtype=np.float64))
    assert out.shape == (1,)
    assert np.isnan(out[0])


def test_align_empty_primary_is_empty() -> None:
    aux = [_bar(underlying_id="SPX", ts=_day(0), available_at=_day(0), close=200.0)]
    out = align_auxiliary_series([], aux, own_level(aux))
    assert out.shape == (0,)


def test_align_raises_on_mismatched_lengths() -> None:
    aux = [_bar(underlying_id="SPX", ts=_day(0), available_at=_day(0), close=200.0)]
    with pytest.raises(ValueError):
        align_auxiliary_series([], aux, np.zeros(2, dtype=np.float64))


def test_align_unsorted_input_still_correct() -> None:
    """Neither primary nor auxiliary need be pre-sorted by the caller."""
    primary_sorted = [
        _bar(underlying_id="DAX", ts=_day(0), available_at=_day(0, 21), close=100.0),
        _bar(underlying_id="DAX", ts=_day(1), available_at=_day(1, 21), close=101.0),
    ]
    aux_sorted = [
        _bar(underlying_id="SPX", ts=_day(0), available_at=_day(0, 22), close=200.0),
        _bar(underlying_id="SPX", ts=_day(1), available_at=_day(1, 22), close=210.0),
    ]
    out_sorted = align_auxiliary_series(primary_sorted, aux_sorted, own_level(aux_sorted))

    primary_shuffled = [primary_sorted[1], primary_sorted[0]]
    aux_shuffled = [aux_sorted[1], aux_sorted[0]]
    aux_values_shuffled = np.array([210.0, 200.0])  # matches aux_shuffled order
    out_shuffled = align_auxiliary_series(primary_shuffled, aux_shuffled, aux_values_shuffled)
    # out_shuffled[0] corresponds to primary_sorted[1] (day 1), out_shuffled[1] to day 0.
    assert out_shuffled[0] == pytest.approx(out_sorted[1])
    assert np.isnan(out_shuffled[1]) and np.isnan(out_sorted[0])


# --- the specific no-look-ahead property models/challengers.py relies on -------------------


def test_appending_future_aux_bar_never_changes_past_alignment() -> None:
    """The whole point: models/challengers.py holds each auxiliary series' FULL
    history (including bars dated after any given as_of) for the entire
    walk-forward run, relying on align_auxiliary_series to never let a
    later-available aux entry leak into an earlier primary bar's feature."""
    primary = [
        _bar(underlying_id="DAX", ts=_day(0), available_at=_day(0, 21), close=100.0),
        _bar(underlying_id="DAX", ts=_day(1), available_at=_day(1, 21), close=101.0),
    ]
    aux_base = [
        _bar(underlying_id="SPX", ts=_day(0), available_at=_day(0, 22), close=200.0),
    ]
    out_base = align_auxiliary_series(primary, aux_base, own_level(aux_base))

    # Append a wildly different-valued aux bar dated AFTER both primary bars.
    aux_with_future = [
        *aux_base,
        _bar(underlying_id="SPX", ts=_day(5), available_at=_day(5), close=999_999.0),
    ]
    aux_values_with_future = np.array([200.0, 999_999.0])
    out_with_future = align_auxiliary_series(primary, aux_with_future, aux_values_with_future)

    assert np.isnan(out_base[0]) and np.isnan(out_with_future[0])
    assert out_base[1] == pytest.approx(out_with_future[1]) == pytest.approx(200.0)


def test_eu_prediction_uses_prior_day_us_close_not_same_day() -> None:
    """Direct test of the task's own example: 'an EU prediction in the morning
    may use the previous day's US close, but not the same day's', expressed
    via each side's own (fallback_prices-style) available_at convention: US
    closes materially AFTER the EU session (here: EU 21:00 UTC, US 22:00
    UTC, both same calendar day)."""
    eu = [
        _bar(underlying_id="DAX", ts=_day(0), available_at=_day(0, 21), close=100.0),
        _bar(underlying_id="DAX", ts=_day(1), available_at=_day(1, 21), close=101.0),
    ]
    us = [
        _bar(underlying_id="SPX", ts=_day(0), available_at=_day(0, 22), close=4000.0),
        _bar(underlying_id="SPX", ts=_day(1), available_at=_day(1, 22), close=4100.0),
    ]
    out = align_auxiliary_series(eu, us, own_level(us))
    # Day 1's EU prediction sees day 0's US close (4000), never day 1's (4100).
    assert out[1] == pytest.approx(4000.0)
