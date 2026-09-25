"""W12 leakage tests: deliberately constructed future information must raise.

CLAUDE.md rule 5 is `available_at <= prediction_time`. The failure mode this
guards against is silent: filtering on `observation_time` instead would hand
a model a volatility close that had not been published yet, inflating
out-of-sample results with no other symptom. Every test here builds exactly
that situation on purpose.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from turboedge.features.availability import (
    InformationLeakageError,
    assert_information_available_at_prediction,
    latest_per_series,
    select_available,
)
from turboedge.storage.schemas import ExternalObservation

PREDICTION_TIME = datetime(2026, 9, 25, 7, 40, tzinfo=UTC)


def make_obs(
    series_id: str = "VIX.CLOSE",
    *,
    observation_time: datetime,
    available_at: datetime,
    value: float = 15.0,
) -> ExternalObservation:
    return ExternalObservation(
        series_id=series_id,
        value=value,
        unit="index_points",
        frequency="daily",
        source_version="cboe_daily_prices_csv",
        observation_time=observation_time,
        available_at=available_at,
        retrieved_at=datetime(2026, 9, 25, 7, 39, tzinfo=UTC),
        source="cboe",
        parser_version="1",
        quality_score=1.0,
    )


def test_observation_available_before_prediction_is_accepted() -> None:
    obs = make_obs(
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 25, 0, 0, tzinfo=UTC),
    )
    assert_information_available_at_prediction([obs], PREDICTION_TIME)


def test_observation_available_after_prediction_raises() -> None:
    """The core leak: a close published after the moment we predict from.

    This is the real 2026-09-25 case -- CBOE's file at 17:00 UTC that day
    carried data only through 09-24, so the 09-25 close genuinely did not
    exist at 07:40 UTC.
    """
    obs = make_obs(
        observation_time=datetime(2026, 9, 25, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 26, 0, 0, tzinfo=UTC),
    )
    with pytest.raises(InformationLeakageError, match="leak future information"):
        assert_information_available_at_prediction([obs], PREDICTION_TIME)


def test_leak_is_caught_even_when_observation_time_looks_safe() -> None:
    """`observation_time` in the past does NOT make a row usable.

    This is the distinction the whole module exists for: a value observed on
    09-24 but only published on 09-26 (a correction, a delayed print) is not
    available at a 09-25 prediction, even though a naive
    `observation_time <= prediction_time` filter would accept it.
    """
    obs = make_obs(
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 26, 0, 0, tzinfo=UTC),
    )
    assert obs.observation_time < PREDICTION_TIME, "fixture must look safe on the wrong field"
    with pytest.raises(InformationLeakageError):
        assert_information_available_at_prediction([obs], PREDICTION_TIME)


def test_exactly_at_prediction_time_is_allowed() -> None:
    """`available_at == prediction_time` satisfies rule 5's `<=`."""
    obs = make_obs(observation_time=PREDICTION_TIME, available_at=PREDICTION_TIME)
    assert_information_available_at_prediction([obs], PREDICTION_TIME)


def test_one_leaking_row_among_many_still_raises() -> None:
    """A single future row in a long clean window must not pass unnoticed."""
    clean = [
        make_obs(
            observation_time=datetime(2026, 9, 1, 20, 15, tzinfo=UTC) + timedelta(days=i),
            available_at=datetime(2026, 9, 2, 0, 0, tzinfo=UTC) + timedelta(days=i),
        )
        for i in range(20)
    ]
    leaking = make_obs(
        series_id="VVIX.CLOSE",
        observation_time=datetime(2026, 9, 25, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 26, 0, 0, tzinfo=UTC),
    )
    with pytest.raises(InformationLeakageError, match=re.escape("VVIX.CLOSE")):
        assert_information_available_at_prediction([*clean, leaking], PREDICTION_TIME)


def test_naive_prediction_time_is_rejected() -> None:
    """A naive timestamp cannot be compared without assuming a timezone.

    Assuming one is how an off-by-one-day leak slips through unnoticed, so
    this raises rather than guessing UTC.
    """
    obs = make_obs(
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 25, 0, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        assert_information_available_at_prediction([obs], datetime(2026, 9, 25, 7, 40))


def test_select_available_filters_and_orders() -> None:
    past = make_obs(
        observation_time=datetime(2026, 9, 23, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 24, 0, 0, tzinfo=UTC),
        value=14.0,
    )
    recent = make_obs(
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 25, 0, 0, tzinfo=UTC),
        value=15.0,
    )
    future = make_obs(
        observation_time=datetime(2026, 9, 25, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 26, 0, 0, tzinfo=UTC),
        value=16.0,
    )
    got = select_available([future, recent, past], PREDICTION_TIME)
    assert [o.value for o in got] == [14.0, 15.0]
    # The filtered result must itself survive the assertion -- the two
    # functions are meant to be used together, and this pins that down.
    assert_information_available_at_prediction(got, PREDICTION_TIME)


def test_latest_per_series_prefers_the_later_publication_on_a_tie() -> None:
    """A correction published later wins over the original print.

    That is the value a model would genuinely have had at prediction time.
    """
    original = make_obs(
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 25, 0, 0, tzinfo=UTC),
        value=15.0,
    )
    correction = make_obs(
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 25, 6, 0, tzinfo=UTC),
        value=15.4,
    )
    other = make_obs(
        series_id="OVX.CLOSE",
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 25, 0, 0, tzinfo=UTC),
        value=55.0,
    )
    got = latest_per_series([original, correction, other], PREDICTION_TIME)
    assert set(got) == {"VIX.CLOSE", "OVX.CLOSE"}
    assert got["VIX.CLOSE"].value == 15.4
    assert got["OVX.CLOSE"].value == 55.0


def test_latest_per_series_ignores_unavailable_rows() -> None:
    available = make_obs(
        observation_time=datetime(2026, 9, 24, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 25, 0, 0, tzinfo=UTC),
        value=15.0,
    )
    future = make_obs(
        observation_time=datetime(2026, 9, 25, 20, 15, tzinfo=UTC),
        available_at=datetime(2026, 9, 26, 0, 0, tzinfo=UTC),
        value=99.0,
    )
    got = latest_per_series([available, future], PREDICTION_TIME)
    assert got["VIX.CLOSE"].value == 15.0
