"""Schemas for the External Data Factory.

The repository already had adapters for Cboe and CFTC and 93,851 rows in
`external_observations` -- but no pipeline, no CLI command and no workflow
step ever wrote them. They came from ad-hoc research scripts. Rebuild the
database and that data is gone, unreproducibly. This package exists to make
external data a *factory* rather than an archaeological find.

Five timestamps are kept distinct and must never be conflated (spec §3):

    observation_time    the period the value describes
    source_release_time when the publisher released it, if known
    available_at        the earliest moment TurboEdge may use it
    vintage_time        which published revision this value is
    ingested_at         when TurboEdge downloaded it (`retrieved_at`)

`available_at` is the only one a research cutoff may compare against
(CLAUDE.md rule 5). The others exist so that this one can be justified
rather than asserted.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime

#: What a redacted credential looks like once `raw_archive.redact` has run.
REDACTED_VALUE = "REDACTED"

#: Query parameters whose values must never be persisted. Kept here rather
#: than in `raw_archive` so the model can enforce the rule even for a caller
#: that never went through the archive.
_CREDENTIAL_PARAM_RE = re.compile(
    r"(?i)\b(api_key|apikey|appid|app_id|access_token|token|password|secret)=([^&\s]*)"
)


class AvailabilityPrecision(StrEnum):
    """How well the moment a value became usable is actually known (spec §4).

    This is not a quality score. It is a statement about evidence: whether
    an auditable publication timestamp exists, or whether someone reasoned
    their way to a date. Confirmatory research may never silently consume
    UNKNOWN observations, which is the whole reason the distinction is
    stored rather than inferred at read time.
    """

    EXACT_TIMESTAMP = "EXACT_TIMESTAMP"
    EXACT_DATE = "EXACT_DATE"
    CONSERVATIVE_DATE = "CONSERVATIVE_DATE"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


#: Precisions a strict point-in-time study may consume. CONSERVATIVE_DATE is
#: included because a documented rule that errs late is defensible; INFERRED
#: and UNKNOWN are not, and a historical row without a recorded precision is
#: treated as UNKNOWN rather than as fine.
STRICT_PIT_PRECISIONS: frozenset[AvailabilityPrecision] = frozenset(
    {
        AvailabilityPrecision.EXACT_TIMESTAMP,
        AvailabilityPrecision.EXACT_DATE,
        AvailabilityPrecision.CONSERVATIVE_DATE,
    }
)


class BackfillClass(StrEnum):
    """Whether a series' history can honestly be used as if it were live.

    The distinction the repository has already paid for once: a value whose
    historical release timing cannot be reconstructed is not point-in-time
    evidence, however long the series is. Such a source starts forward
    archival immediately and earns confirmatory standing later, rather than
    borrowing it from a vendor's revised history.
    """

    #: Publisher exposes real vintages (ALFRED-style). History is PIT-safe.
    HISTORICAL_PIT_SAFE = "HISTORICAL_PIT_SAFE"
    #: History exists, release timing only reconstructable by a documented
    #: conservative rule. Usable for exploration, weaker for confirmation.
    HISTORICAL_CONSERVATIVE = "HISTORICAL_CONSERVATIVE"
    #: History exists but is silently revised with no vintage record. Only
    #: forward snapshots taken by TurboEdge itself count as evidence.
    FORWARD_ONLY = "FORWARD_ONLY"
    #: Nothing retrievable yet.
    UNKNOWN = "UNKNOWN"


class SourceStatus(StrEnum):
    """Operational state of one external source (spec §8)."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    AUTH_MISSING = "AUTH_MISSING"
    SCHEMA_CHANGED = "SCHEMA_CHANGED"
    STALE = "STALE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"
    DEFERRED = "DEFERRED"


#: Statuses in which automatic ingestion must not run. REVIEW_REQUIRED is
#: here because unclear licensing is not a reason to fetch and apologise
#: later (spec §8: "Do not guess licensing").
NON_INGESTING_STATUSES: frozenset[SourceStatus] = frozenset(
    {
        SourceStatus.AUTH_MISSING,
        SourceStatus.REVIEW_REQUIRED,
        SourceStatus.BLOCKED,
        SourceStatus.DEFERRED,
    }
)


class Criticality(StrEnum):
    """How much a source's failure matters. Drives notification, not gates."""

    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    EXPERIMENTAL = "EXPERIMENTAL"


class ExternalSourceManifest(BaseModel):
    """One external source, its access terms and its point-in-time honesty.

    `commercial_use_status` and `license_or_terms_reference` are required
    rather than optional: a source whose terms nobody looked up is exactly
    the source that should not be ingested automatically.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    display_name: str
    official_source: str
    homepage: str
    machine_endpoint: str
    access_method: str

    requires_auth: bool = False
    auth_environment_variable: str | None = None

    license_or_terms_reference: str
    commercial_use_status: str = "UNKNOWN"

    frequency: str
    expected_update_cadence: str

    supports_historical_data: bool
    supports_vintages: bool
    supports_exact_release_time: bool
    point_in_time_quality: AvailabilityPrecision
    backfill_class: BackfillClass

    enabled: bool = False
    criticality: Criticality = Criticality.EXPERIMENTAL

    last_successful_ingestion: TzAwareDatetime | None = None
    last_attempt: TzAwareDatetime | None = None
    status: SourceStatus = SourceStatus.REVIEW_REQUIRED
    status_note: str = ""

    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _auth_variable_present_when_required(self) -> Self:
        if self.requires_auth and not self.auth_environment_variable:
            raise ValueError(
                f"{self.source_id}: requires_auth is set but no "
                "auth_environment_variable is named, so nothing can check "
                "whether the credential is present"
            )
        return self

    @model_validator(mode="after")
    def _enabled_sources_have_settled_terms(self) -> Self:
        """An enabled source may not sit in a status that forbids ingestion.

        Catches the combination that would otherwise silently do nothing:
        `enabled: true` with `status: REVIEW_REQUIRED`, which reads like a
        live source and behaves like a disabled one.
        """
        if self.enabled and self.status in NON_INGESTING_STATUSES:
            raise ValueError(
                f"{self.source_id}: enabled=True is incompatible with status "
                f"{self.status}; resolve the status or leave the source disabled"
            )
        return self

    @property
    def may_ingest(self) -> bool:
        return self.enabled and self.status not in NON_INGESTING_STATUSES


class SeriesSpec(BaseModel):
    """One configured series (spec §9). Curated, never bulk-discovered."""

    model_config = ConfigDict(extra="forbid")

    source: str
    series_id: str
    name: str
    category: str
    unit: str
    frequency: str

    #: The publisher's own identifier, which is often not a legal column name
    #: and must not be conflated with our `series_id`.
    native_identifier: str

    availability_precision: AvailabilityPrecision
    backfill_class: BackfillClass

    #: Documented lag applied when the publisher gives a date but no time.
    #: `None` means the adapter supplies an exact timestamp itself.
    conservative_release_lag_hours: float | None = Field(default=None, ge=0.0)

    #: True when the publisher emits a value only when something happens,
    #: rather than on a calendar. The ECB main refinancing rate is the
    #: canonical case: 48 observations since 1999, one per rate decision
    #: that changed the rate. Treating such a series as a calendar series
    #: makes it look 99.3% incomplete when it is in fact complete, and its
    #: effective sample is the event count -- never the number of days the
    #: rate happened to stay put (spec §37).
    event_driven: bool = False

    enabled: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def _conservative_dates_declare_their_rule(self) -> Self:
        """A CONSERVATIVE_DATE series must say what the conservative rule is.

        Otherwise "conservative" is a label rather than a lag, and nothing
        downstream can check that the lag was actually applied.
        """
        if (
            self.availability_precision is AvailabilityPrecision.CONSERVATIVE_DATE
            and self.conservative_release_lag_hours is None
        ):
            raise ValueError(
                f"{self.series_id}: CONSERVATIVE_DATE requires an explicit "
                "conservative_release_lag_hours -- an undocumented conservative "
                "rule cannot be audited"
            )
        return self

    @property
    def qualified_id(self) -> str:
        return f"{self.source}.{self.series_id}"


class RawPayload(BaseModel):
    """One archived upstream response (spec §6A, §23).

    Kept so a parser bug is repairable without refetching history that the
    publisher may no longer serve. `request_fingerprint` must never contain a
    credential -- enforced below, because a persisted API key is the kind of
    mistake that is only noticed after the repository is public.
    """

    model_config = ConfigDict(extra="forbid")

    payload_id: str
    source: str
    dataset: str
    request_fingerprint: str
    retrieved_at: TzAwareDatetime
    http_status: int
    content_type: str
    content_encoding: str = ""
    byte_size: int = Field(ge=0)
    payload_hash: str
    stored_path: str
    parser_version: str
    git_commit: str | None = None
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _no_credentials_in_fingerprint(self) -> Self:
        """Reject a credential parameter that still carries a value.

        Checks the value, not the parameter name: a redacted
        `api_key=REDACTED` is exactly what a correctly handled request looks
        like, and rejecting it would push callers into stripping the
        parameter entirely, which loses the fact that a credential was used
        at all. What must never be stored is the secret itself.
        """
        haystack = f"{self.request_fingerprint} {self.stored_path}"
        for match in _CREDENTIAL_PARAM_RE.finditer(haystack):
            value = match.group(2)
            if value and value != REDACTED_VALUE:
                raise ValueError(
                    f"{self.source}: refusing to persist a request fingerprint whose "
                    f"{match.group(1)!r} still carries a value; redact it before archiving"
                )
        return self
