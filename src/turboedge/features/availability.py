"""Point-in-time availability enforcement for external information (W12).

CLAUDE.md rule 5 is ``available_at <= prediction_time``, and the distinction
from ``observation_time <= prediction_time`` is the whole point. For a daily
volatility index the two differ by about a day: the 2026-09-25 VIX close
does not exist while 2026-09-25 is still being traded, and CBOE's own file
was measured on 2026-09-25 17:00 UTC carrying data only through 2026-09-24.
Filtering on ``observation_time`` would therefore hand every model a close
that had not happened yet -- a leak that inflates out-of-sample results
without any other symptom, which is exactly why W12 routes every external
family through this one function instead of each adapter filtering its own
way.

Used by every external feature builder; the accompanying tests deliberately
construct leaking inputs and assert that they raise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from turboedge.storage.schemas import ExternalObservation


class InformationLeakageError(ValueError):
    """Raised when an observation would be used before it was available."""


def assert_information_available_at_prediction(
    observations: Iterable[ExternalObservation],
    prediction_time: datetime,
) -> None:
    """Raise if any observation's ``available_at`` is after ``prediction_time``.

    Deliberately raises rather than filtering: a builder that silently drops
    future rows hides the fact that it was handed them at all, and the same
    bug then reappears the next time someone assembles a feature matrix by
    hand. Filtering is `select_available`'s job, and it is a separate call so
    the choice is explicit at every site.

    Raises:
        InformationLeakageError: on the first violating observation, naming
            the series and both timestamps.
        ValueError: if ``prediction_time`` is not timezone-aware -- a naive
            timestamp cannot be compared against tz-aware ``available_at``
            without silently assuming a timezone, and assuming one here is
            how an off-by-one-day leak would slip through.
    """
    _require_aware(prediction_time, "prediction_time")
    for obs in observations:
        if obs.available_at > prediction_time:
            raise InformationLeakageError(
                f"series {obs.series_id!r} observation available_at="
                f"{obs.available_at.isoformat()} is after prediction_time="
                f"{prediction_time.isoformat()} "
                f"(observation_time={obs.observation_time.isoformat()}); "
                "using it would leak future information"
            )


def select_available(
    observations: Iterable[ExternalObservation],
    prediction_time: datetime,
) -> list[ExternalObservation]:
    """Return only the observations available at ``prediction_time``, in
    chronological order by ``observation_time``.

    The intended pairing: call this to build a feature's input window, then
    `assert_information_available_at_prediction` on the result as a
    self-check. The assertion is cheap and catches the case where a later
    refactor rebuilds the window from a different source.
    """
    _require_aware(prediction_time, "prediction_time")
    available = [o for o in observations if o.available_at <= prediction_time]
    available.sort(key=lambda o: (o.observation_time, o.series_id))
    return available


def latest_per_series(
    observations: Sequence[ExternalObservation],
    prediction_time: datetime,
) -> dict[str, ExternalObservation]:
    """Most recent available observation per ``series_id``.

    Ties on ``observation_time`` resolve to the later ``available_at`` --
    i.e. a correction published afterwards wins over the original print,
    which is the value a model would genuinely have had at
    ``prediction_time``.
    """
    latest: dict[str, ExternalObservation] = {}
    for obs in select_available(observations, prediction_time):
        current = latest.get(obs.series_id)
        if (
            current is None
            or obs.observation_time > current.observation_time
            or (
                obs.observation_time == current.observation_time
                and obs.available_at > current.available_at
            )
        ):
            latest[obs.series_id] = obs
    return latest


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")


__all__ = [
    "InformationLeakageError",
    "assert_information_available_at_prediction",
    "latest_per_series",
    "select_available",
]
