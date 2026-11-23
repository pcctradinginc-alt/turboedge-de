"""Protect the wake-up guard against redundant or off-hours scans."""

from datetime import datetime

import pytest

from turboedge.monitoring.pipeline_wakeup import should_dispatch_scan




def _now(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def test_monday_missing_runs_during_market_hours_dispatches() -> None:
    assert should_dispatch_scan(
        [{"createdAt": "2026-11-21T16:21:41Z"}],
        now=_now("2026-11-23T13:35:00Z"),
    )


def test_recent_completed_run_suppresses_dispatch() -> None:
    assert not should_dispatch_scan(
        [{"createdAt": "2026-11-23T12:40:00Z", "status": "completed"}],
        now=_now("2026-11-23T13:35:00Z"),
    )


def test_recent_queued_run_suppresses_dispatch() -> None:
    assert not should_dispatch_scan(
        [{"createdAt": "2026-11-23T13:30:00Z", "status": "queued"}],
        now=_now("2026-11-23T13:35:00Z"),
    )


@pytest.mark.parametrize(
    "now",
    [
        "2026-11-23T19:15:00Z",  # after market hours (CET)
        "2026-11-22T13:00:00Z",  # Sunday
        "2026-07-20T06:55:00Z",  # before market hours (CEST)
    ],
)
def test_closed_market_does_not_dispatch(now: str) -> None:
    assert not should_dispatch_scan([], now=_now(now))


def test_naive_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        should_dispatch_scan([], now=datetime(2026, 11, 23, 13, 0))


def test_missing_run_timestamp_fails_closed() -> None:
    with pytest.raises(ValueError, match="createdAt"):
        should_dispatch_scan([{"status": "completed"}], now=_now("2026-11-23T13:35:00Z"))
