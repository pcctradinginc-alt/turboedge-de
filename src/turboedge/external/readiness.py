"""Data readiness: when a dataset may be used for research, decided by rule.

The question this answers is the one the user should never have to remember
(spec §84): *"when can we finally use this dataset?"* The answer is computed
from the data's own properties -- how much independent evidence exists, over
how long, how well its availability is known, how complete and how fresh it
is -- and never from whether using it would have been profitable.

That last exclusion is the point. A readiness rule tuned until a dataset
"becomes ready" just as an alpha starts to look good is not a rule, it is a
result-dependent threshold, and this repository has an explicit governance
prohibition against those (GOVERNANCE.md §11, spec §55). So:

* `ReadinessPolicy` is frozen and versioned. Every `DataReadinessRecord`
  stores the `policy_version` that produced it, and changing a threshold
  requires a new version rather than an edit.
* Nothing here imports anything that can see a return, a P&L or a forecast.
* `effective_n` is used everywhere `n` is required. 100,000 quotes from one
  trading day are one day of evidence, and the tournament in this repository
  already produced false positives exactly once by counting rows instead
  (`docs/measured_results.md` §6.12).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from turboedge.external.schemas import (
    STRICT_PIT_PRECISIONS,
    AvailabilityPrecision,
    BackfillClass,
)
from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime


class ReadinessState(StrEnum):
    """Spec §32. Ordered by maturity, but not every state is reachable from
    every other -- see `READINESS_LEVEL` and the evaluator below."""

    DISABLED = "DISABLED"
    COLLECTING = "COLLECTING"
    SCHEMA_VALIDATED = "SCHEMA_VALIDATED"
    PIT_VALIDATED = "PIT_VALIDATED"
    EXPLORATORY_READY = "EXPLORATORY_READY"
    VALIDATION_READY = "VALIDATION_READY"
    CONFIRMATION_READY = "CONFIRMATION_READY"
    FORWARD_MATURE = "FORWARD_MATURE"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


#: Research maturity level (spec §41). LEVEL 5 (PRODUCTION_ELIGIBLE) is
#: deliberately absent: data alone can never create production eligibility,
#: which additionally requires a validated alpha, confirmation, positive
#: economic Turbo evidence, a forward shadow and the Evidence & Risk Gate.
READINESS_LEVEL: dict[ReadinessState, int] = {
    ReadinessState.DISABLED: -1,
    ReadinessState.BLOCKED: -1,
    ReadinessState.DEGRADED: 0,
    ReadinessState.COLLECTING: 0,
    ReadinessState.SCHEMA_VALIDATED: 0,
    ReadinessState.PIT_VALIDATED: 0,
    ReadinessState.EXPLORATORY_READY: 1,
    ReadinessState.VALIDATION_READY: 2,
    ReadinessState.CONFIRMATION_READY: 3,
    ReadinessState.FORWARD_MATURE: 4,
}


class RegimeCoverage(StrEnum):
    """Uncertainty metadata, not a gate (spec §42). Requiring every regime
    before any research could begin would stop research permanently, and
    manufacturing balanced regimes would be worse than having few."""

    LIMITED = "LIMITED"
    MODERATE = "MODERATE"
    BROAD = "BROAD"


class ReadinessProfile(BaseModel):
    """Thresholds for one observation frequency (spec §35).

    These are conservative initial defaults, **not** empirically optimised
    parameters. They were chosen from the sampling frequency alone, before
    any of the data existed, and must not be tuned against returns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_exploratory_observations: int = Field(gt=0)
    min_exploratory_span_days: int = Field(gt=0)
    min_validation_observations: int = Field(gt=0)
    min_confirmation_observations: int = Field(gt=0)
    min_forward_span_days: int = Field(gt=0)


#: Spec §35's defaults, verbatim. Keyed by series frequency.
DEFAULT_PROFILES: dict[str, ReadinessProfile] = {
    "daily": ReadinessProfile(
        min_exploratory_observations=180,
        min_exploratory_span_days=270,
        min_validation_observations=60,
        min_confirmation_observations=60,
        min_forward_span_days=90,
    ),
    "weekly": ReadinessProfile(
        min_exploratory_observations=52,
        min_exploratory_span_days=365,
        min_validation_observations=20,
        min_confirmation_observations=20,
        min_forward_span_days=140,
    ),
    "monthly": ReadinessProfile(
        min_exploratory_observations=48,
        min_exploratory_span_days=1460,
        min_validation_observations=12,
        min_confirmation_observations=12,
        min_forward_span_days=365,
    ),
    "quarterly": ReadinessProfile(
        min_exploratory_observations=32,
        min_exploratory_span_days=2920,
        min_validation_observations=8,
        min_confirmation_observations=8,
        min_forward_span_days=730,
    ),
}


class ReadinessPolicy(BaseModel):
    """The complete, frozen rule set. Versioned so a record can be replayed.

    `policy_version` is not decoration. A readiness state computed under one
    set of thresholds and compared against another is meaningless, and a
    threshold that can be edited in place is a threshold that can be lowered
    the week an alpha needs it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    profiles: dict[str, ReadinessProfile] = Field(default_factory=lambda: dict(DEFAULT_PROFILES))

    #: Fraction of expected observations that may be missing.
    max_missingness: float = Field(default=0.10, ge=0.0, le=1.0)
    #: A series whose newest observation is older than this is DEGRADED.
    max_staleness_days: int = Field(default=45, gt=0)
    #: Strict-PIT research needs a precision from `STRICT_PIT_PRECISIONS`.
    require_strict_pit_for_confirmation: bool = True

    def profile_for(self, frequency: str) -> ReadinessProfile:
        try:
            return self.profiles[frequency]
        except KeyError:
            raise KeyError(
                f"no readiness profile for frequency {frequency!r}; "
                f"configured: {sorted(self.profiles)}. Refusing to fall back to "
                "another frequency's thresholds, which would silently apply a "
                "daily bar to a quarterly series"
            ) from None


class SeriesEvidence(BaseModel):
    """What is actually known about one series' accumulated data (spec §36).

    Deliberately separate from the evaluator: this is measurement, that is
    policy. Keeping them apart is what makes it possible to re-evaluate old
    evidence under a new policy version without refetching anything.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    series_id: str
    frequency: str
    backfill_class: BackfillClass
    availability_precision: AvailabilityPrecision

    nominal_n: int = Field(ge=0)
    independent_dates: int = Field(ge=0)
    effective_n: float = Field(ge=0.0)

    first_observation: date | None = None
    last_observation: date | None = None
    collection_start: TzAwareDatetime | None = None

    #: Observations TurboEdge itself archived forward, i.e. not backfilled.
    forward_n: int = Field(default=0, ge=0)
    forward_effective_n: float = Field(default=0.0, ge=0.0)
    forward_first_observation: date | None = None

    event_count: int = Field(default=0, ge=0)
    independent_event_count: int = Field(default=0, ge=0)

    schema_valid: bool = False
    pit_integrity: bool = False
    provenance_complete: bool = False
    missingness: float = Field(default=1.0, ge=0.0, le=1.0)
    unresolved_warnings: tuple[str, ...] = ()

    regime_coverage: RegimeCoverage = RegimeCoverage.LIMITED

    @model_validator(mode="after")
    def _effective_cannot_exceed_nominal(self) -> Self:
        """Effective sample above nominal would mean the dependence
        correction invented evidence. That is a bug, not a rounding issue."""
        if self.effective_n > self.nominal_n:
            raise ValueError(
                f"{self.series_id}: effective_n {self.effective_n} exceeds nominal_n "
                f"{self.nominal_n}; a dependence correction may only ever reduce"
            )
        if self.forward_effective_n > self.forward_n:
            raise ValueError(
                f"{self.series_id}: forward_effective_n exceeds forward_n "
                f"({self.forward_effective_n} > {self.forward_n})"
            )
        return self

    def calendar_span_days(self, *, as_of: date | None = None) -> int:
        if self.first_observation is None:
            return 0
        end = self.last_observation or as_of or self.first_observation
        return max(0, (end - self.first_observation).days)

    def forward_span_days(self, *, as_of: date) -> int:
        if self.forward_first_observation is None:
            return 0
        return max(0, (as_of - self.forward_first_observation).days)


class DataReadinessRecord(BaseModel):
    """One evaluation of one series, persisted and auditable (spec §40)."""

    model_config = ConfigDict(extra="forbid")

    source: str
    series_id: str

    state: ReadinessState
    previous_state: ReadinessState | None = None

    evaluated_at: TzAwareDatetime
    first_reached_at: TzAwareDatetime | None = None

    collection_start: TzAwareDatetime | None = None
    research_usable_from: date | None = None
    forward_evidence_start: date | None = None

    nominal_n: int = Field(ge=0)
    effective_n: float = Field(ge=0.0)
    independent_dates: int = Field(ge=0)
    calendar_span_days: int = Field(ge=0)

    forward_n: int = Field(default=0, ge=0)
    forward_effective_n: float = Field(default=0.0, ge=0.0)
    forward_span_days: int = Field(default=0, ge=0)

    event_count: int = Field(default=0, ge=0)
    independent_event_count: int = Field(default=0, ge=0)

    pit_quality: bool = False
    availability_precision: AvailabilityPrecision
    completeness: float = Field(ge=0.0, le=1.0)
    freshness_days: int | None = None

    regime_coverage: RegimeCoverage = RegimeCoverage.LIMITED

    blocking_reasons: tuple[str, ...] = ()

    policy_version: str
    git_commit: str | None = None
    config_hash: str | None = None
    schema_version: str = SCHEMA_VERSION

    @property
    def level(self) -> int:
        return READINESS_LEVEL[self.state]

    @model_validator(mode="after")
    def _a_ready_series_states_no_blockers(self) -> Self:
        """A state at or above EXPLORATORY_READY with blocking reasons would
        be self-contradictory, and the contradiction would be invisible to
        anyone reading only the state."""
        if READINESS_LEVEL[self.state] >= 1 and self.blocking_reasons:
            raise ValueError(
                f"{self.series_id}: state {self.state} cannot carry blocking "
                f"reasons {list(self.blocking_reasons)}"
            )
        return self


def evaluate_readiness(
    evidence: SeriesEvidence,
    policy: ReadinessPolicy,
    *,
    now: datetime,
    enabled: bool = True,
    previous: DataReadinessRecord | None = None,
    git_commit: str | None = None,
    config_hash: str | None = None,
) -> DataReadinessRecord:
    """Decide one series' readiness from its evidence and a frozen policy.

    Deterministic: same evidence plus same policy gives the same state, on
    any machine, at any time. `now` is injected rather than read so that a
    replay of a historical record produces the historical answer.

    Order matters. Quality and point-in-time integrity are checked *before*
    sample size, because a large sample of rows whose availability is unknown
    is not weak evidence -- it is inadmissible evidence, and reporting it as
    "nearly ready" would invite exactly the wrong fix.
    """
    today = now.date()
    profile = policy.profile_for(evidence.frequency)
    reasons: list[str] = []

    if not enabled:
        return _record(
            evidence,
            policy,
            ReadinessState.DISABLED,
            now,
            previous,
            ("series is disabled in configuration",),
            git_commit,
            config_hash,
        )

    freshness_days = (today - evidence.last_observation).days if evidence.last_observation else None

    # -- hard blockers -----------------------------------------------------
    if not evidence.schema_valid:
        reasons.append("schema validation has not passed")
    if not evidence.provenance_complete:
        reasons.append("provenance is incomplete")
    if evidence.unresolved_warnings:
        reasons.append(f"unresolved parser warnings: {list(evidence.unresolved_warnings)}")
    if reasons:
        return _record(
            evidence,
            policy,
            ReadinessState.BLOCKED,
            now,
            previous,
            tuple(reasons),
            git_commit,
            config_hash,
            freshness_days=freshness_days,
        )

    if not evidence.pit_integrity:
        return _record(
            evidence,
            policy,
            ReadinessState.SCHEMA_VALIDATED,
            now,
            previous,
            ("point-in-time integrity has not been established",),
            git_commit,
            config_hash,
            freshness_days=freshness_days,
        )

    if freshness_days is not None and freshness_days > policy.max_staleness_days:
        return _record(
            evidence,
            policy,
            ReadinessState.DEGRADED,
            now,
            previous,
            (
                f"newest observation is {freshness_days} days old, above the "
                f"{policy.max_staleness_days}-day limit",
            ),
            git_commit,
            config_hash,
            freshness_days=freshness_days,
        )

    if evidence.missingness > policy.max_missingness:
        return _record(
            evidence,
            policy,
            ReadinessState.PIT_VALIDATED,
            now,
            previous,
            (f"missingness {evidence.missingness:.3f} exceeds {policy.max_missingness:.3f}",),
            git_commit,
            config_hash,
            freshness_days=freshness_days,
        )

    # -- sample sufficiency ------------------------------------------------
    # FORWARD_ONLY series may only count what TurboEdge archived itself: the
    # publisher's history is silently revised, so using it as if it had been
    # observed live is the "fake retrospective backtest" spec §34 forbids.
    forward_only = evidence.backfill_class is BackfillClass.FORWARD_ONLY
    usable_n = evidence.forward_effective_n if forward_only else evidence.effective_n
    span = (
        evidence.forward_span_days(as_of=today)
        if forward_only
        else evidence.calendar_span_days(as_of=today)
    )

    if usable_n < profile.min_exploratory_observations:
        reasons.append(
            f"effective sample {usable_n:.1f} below {profile.min_exploratory_observations}"
            + (" (forward-archived observations only)" if forward_only else "")
        )
    if span < profile.min_exploratory_span_days:
        reasons.append(
            f"calendar span {span}d below {profile.min_exploratory_span_days}d"
            + (" (forward archive only)" if forward_only else "")
        )
    if reasons:
        return _record(
            evidence,
            policy,
            ReadinessState.PIT_VALIDATED,
            now,
            previous,
            tuple(reasons),
            git_commit,
            config_hash,
            freshness_days=freshness_days,
        )

    state = ReadinessState.EXPLORATORY_READY

    if usable_n >= (profile.min_exploratory_observations + profile.min_validation_observations):
        state = ReadinessState.VALIDATION_READY

    strict_pit_ok = (
        evidence.availability_precision in STRICT_PIT_PRECISIONS
        or not policy.require_strict_pit_for_confirmation
    )
    if (
        state is ReadinessState.VALIDATION_READY
        and strict_pit_ok
        and usable_n
        >= (
            profile.min_exploratory_observations
            + profile.min_validation_observations
            + profile.min_confirmation_observations
        )
    ):
        state = ReadinessState.CONFIRMATION_READY

    if (
        state is ReadinessState.CONFIRMATION_READY
        and evidence.forward_span_days(as_of=today) >= profile.min_forward_span_days
        and evidence.forward_effective_n >= profile.min_confirmation_observations
    ):
        state = ReadinessState.FORWARD_MATURE

    return _record(
        evidence,
        policy,
        state,
        now,
        previous,
        (),
        git_commit,
        config_hash,
        freshness_days=freshness_days,
    )


def _record(
    evidence: SeriesEvidence,
    policy: ReadinessPolicy,
    state: ReadinessState,
    now: datetime,
    previous: DataReadinessRecord | None,
    reasons: tuple[str, ...],
    git_commit: str | None,
    config_hash: str | None,
    *,
    freshness_days: int | None = None,
) -> DataReadinessRecord:
    today = now.date()
    first_reached = None
    if previous is not None and previous.state is state:
        first_reached = previous.first_reached_at
    elif READINESS_LEVEL[state] >= 1:
        first_reached = now

    # A FORWARD_ONLY series' research window starts when TurboEdge itself
    # began archiving, never at the publisher's first historical value.
    if evidence.backfill_class is BackfillClass.FORWARD_ONLY:
        usable_from = evidence.forward_first_observation
    else:
        usable_from = evidence.first_observation

    return DataReadinessRecord(
        source=evidence.source,
        series_id=evidence.series_id,
        state=state,
        previous_state=previous.state if previous is not None else None,
        evaluated_at=now,
        first_reached_at=first_reached,
        collection_start=evidence.collection_start,
        research_usable_from=usable_from if READINESS_LEVEL[state] >= 1 else None,
        forward_evidence_start=evidence.forward_first_observation,
        nominal_n=evidence.nominal_n,
        effective_n=evidence.effective_n,
        independent_dates=evidence.independent_dates,
        calendar_span_days=evidence.calendar_span_days(as_of=today),
        forward_n=evidence.forward_n,
        forward_effective_n=evidence.forward_effective_n,
        forward_span_days=evidence.forward_span_days(as_of=today),
        event_count=evidence.event_count,
        independent_event_count=evidence.independent_event_count,
        pit_quality=evidence.pit_integrity,
        availability_precision=evidence.availability_precision,
        completeness=max(0.0, min(1.0, 1.0 - evidence.missingness)),
        freshness_days=freshness_days,
        regime_coverage=evidence.regime_coverage,
        blocking_reasons=reasons,
        policy_version=policy.policy_version,
        git_commit=git_commit,
        config_hash=config_hash,
    )


def next_evaluation_due(record: DataReadinessRecord, *, cadence: timedelta) -> datetime:
    """When this series should be looked at again. Purely informational."""
    return record.evaluated_at + cadence
