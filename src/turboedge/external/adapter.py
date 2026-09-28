"""The contract every external-series adapter implements.

One protocol rather than six bespoke classes, for the reason spec §17 gives:
the Data Factory's ingestion loop, raw archival, source-health reporting and
readiness evaluation must work the same way for a source added next year as
for the five added today. An adapter that needs the loop changed to
accommodate it has not been written to this contract.

An adapter's whole job is: fetch bytes, hand them over unmodified for
archival, and turn them into `ExternalObservation`s whose `available_at` it
can *justify*. It does not decide whether data are good enough, does not
write to the database, and does not know what research will do with them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol, runtime_checkable

from turboedge.external.schemas import AvailabilityPrecision, SeriesSpec
from turboedge.storage.schemas import ExternalObservation


@dataclass(frozen=True, slots=True)
class FetchedPayload:
    """One upstream response, exactly as received.

    `content` stays `bytes`, never a decoded string: the hash archived
    alongside it must be the hash of what the publisher actually sent, or the
    archive cannot prove what was parsed. `url` must already have any
    credential stripped -- `RawPayload` refuses to persist one, and finding
    that out at write time is too late.
    """

    source: str
    dataset: str
    url: str
    content: bytes
    http_status: int
    content_type: str
    retrieved_at: datetime
    request_fingerprint: str
    content_encoding: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Observations plus everything the readiness layer needs to judge them.

    `warnings` is not cosmetic: an unresolved parser warning blocks readiness
    (spec §38). An adapter that swallows an oddity to keep the pipeline green
    is removing the only signal that something upstream changed.
    """

    observations: list[ExternalObservation]
    warnings: tuple[str, ...] = ()
    #: Series the payload was expected to contain but did not.
    missing_series: tuple[str, ...] = ()


@runtime_checkable
class ExternalSeriesAdapter(Protocol):
    """Fetch and parse one external source's configured series."""

    @property
    def source_id(self) -> str:
        """Stable identifier, matching `ExternalSourceManifest.source_id`."""
        ...

    @property
    def parser_version(self) -> str:
        """Bumped whenever parsing changes semantics, so an old row can be
        told apart from a row produced by today's code."""
        ...

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        """Retrieve one series' raw payload.

        `since` is a hint for incremental retrieval (spec §75). An adapter
        whose endpoint always serves the full history may ignore it; it must
        never return *less* than `since` onwards.
        """
        ...

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        """Turn a payload into observations. Must not perform network I/O,
        so that an archived payload can be reparsed years later."""
        ...


def conservative_available_at(
    observation_day: date,
    *,
    lag_hours: float,
    publication_time: time = time(0, 0),
) -> datetime:
    """When a date-only release may first be used (spec §4).

    Never invents an intraday publication time. The rule is explicit and
    documented: take the start of the observation day, add the series'
    declared lag, and treat *that* as the earliest usable moment. Erring late
    costs a little statistical power; erring early invents a forecast the
    system could not have made.
    """
    if lag_hours < 0:
        raise ValueError(f"conservative lag must not be negative: {lag_hours}")
    base = datetime.combine(observation_day, publication_time, tzinfo=UTC)
    return base + timedelta(hours=lag_hours)


def resolve_available_at(
    spec: SeriesSpec,
    observation_day: date,
    *,
    exact: datetime | None = None,
) -> tuple[datetime, AvailabilityPrecision]:
    """The single place an adapter decides when an observation became usable.

    Centralised on purpose. Six adapters each inventing their own availability
    rule is six chances to quietly use tomorrow's number today, and the one
    that gets it wrong will be the one nobody reviews.

    An exact publisher timestamp always wins. Otherwise the series' declared
    precision decides, and a series that declared CONSERVATIVE_DATE must have
    supplied the lag (`SeriesSpec` enforces that at construction).
    """
    if exact is not None:
        if exact.tzinfo is None or exact.utcoffset() is None:
            raise ValueError(f"{spec.qualified_id}: exact release time must be tz-aware")
        return exact.astimezone(UTC), AvailabilityPrecision.EXACT_TIMESTAMP

    precision = spec.availability_precision
    if precision is AvailabilityPrecision.EXACT_TIMESTAMP:
        raise ValueError(
            f"{spec.qualified_id}: declared EXACT_TIMESTAMP but the adapter supplied "
            "no timestamp; refusing to substitute a made-up publication time"
        )
    if precision is AvailabilityPrecision.UNKNOWN:
        # Still returns a usable moment so the row can be stored, but the
        # precision travels with it and strict research will exclude it.
        return (
            datetime.combine(observation_day, time(0, 0), tzinfo=UTC) + timedelta(days=1),
            AvailabilityPrecision.UNKNOWN,
        )
    lag = spec.conservative_release_lag_hours
    if lag is None:
        # EXACT_DATE without a lag: usable once the publication day is over.
        return (
            datetime.combine(observation_day, time(0, 0), tzinfo=UTC) + timedelta(days=1),
            precision,
        )
    return conservative_available_at(observation_day, lag_hours=lag), precision


def resolve_forecast_available_at(
    spec: SeriesSpec, issued_at: datetime
) -> tuple[datetime, AvailabilityPrecision]:
    """When a forecast became usable -- the moment it was obtained.

    A forecast describes a period that has not happened yet, so the normal
    rule (observation day plus a publication lag) records it as knowable
    only after the thing it predicts, which is worse than useless: a
    point-in-time read would surface every forecast too late to have acted
    on it.

    The honest anchor is the moment TurboEdge actually held the value.
    That is conservative with respect to the publisher's own issue time,
    which is always earlier and is rarely stated, so the precision is
    CONSERVATIVE_DATE rather than EXACT_TIMESTAMP: the timestamp is exact,
    but it is *our* acquisition, not their release.
    """
    if not spec.forecast_series:
        raise ValueError(
            f"{spec.qualified_id}: resolve_forecast_available_at is only for a "
            "series declared forecast_series=True"
        )
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError(f"{spec.qualified_id}: issued_at must be timezone-aware")
    return issued_at.astimezone(UTC), AvailabilityPrecision.CONSERVATIVE_DATE


def deduplicate_observations(
    observations: Sequence[ExternalObservation],
) -> list[ExternalObservation]:
    """Collapse exact re-fetches, keep genuine revisions (spec §7).

    The key is `(source, series_id, observation_time, available_at)` -- the
    storage key. Two rows for the same period with *different* `available_at`
    are a first print and a revision, and merging them would delete the
    historical fact that the first number was all anyone had.
    """
    seen: dict[tuple[str, str, datetime, datetime], ExternalObservation] = {}
    for obs in observations:
        seen[(obs.source, obs.series_id, obs.observation_time, obs.available_at)] = obs
    return sorted(seen.values(), key=lambda o: (o.series_id, o.observation_time, o.available_at))
