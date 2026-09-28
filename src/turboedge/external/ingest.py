"""The ingestion loop: fetch, archive, normalise, evaluate, hand off.

This is the piece that did not exist. The repository had working Cboe and
CFTC adapters and 93,851 rows in `external_observations`, but no pipeline,
CLI command or workflow step ever called them -- the rows came from ad-hoc
scripts run during research sessions. Nothing scheduled them, nothing
archived the payloads, and nothing would have reproduced them.

The order of operations is deliberate and each step is allowed to fail
without taking the rest with it:

1. **fetch** -- one adapter call per enabled series, rate-limited by host.
2. **archive** -- the raw bytes go to disk *before* parsing, so a parser
   that crashes still leaves the evidence behind.
3. **normalise** -- parse into `ExternalObservation`s and store them.
4. **evaluate** -- measure the accumulated evidence and apply the frozen
   readiness policy.
5. **hand off** -- emit any newly earned readiness trigger, exactly once.

Step 5 proposes; it never approves. A trigger records that a dataset became
usable and creates nothing past `PROPOSED`. It starts no research, consumes
no research budget and touches no CONFIRMATION or FORWARD partition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from turboedge.external.adapter import ExternalSeriesAdapter, deduplicate_observations
from turboedge.external.catalog import ExternalDataConfig
from turboedge.external.evidence import build_evidence
from turboedge.external.raw_archive import store_payload
from turboedge.external.readiness import DataReadinessRecord, evaluate_readiness
from turboedge.external.schemas import SeriesSpec, SourceStatus
from turboedge.external.triggers import ResearchTrigger, triggers_for_transition

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from turboedge.storage.duckdb import Store

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class SeriesResult:
    """What happened to one series in one ingestion run."""

    series_id: str
    source: str
    fetched: bool = False
    observations_written: int = 0
    payload_bytes: int = 0
    warnings: tuple[str, ...] = ()
    error: str | None = None
    readiness: DataReadinessRecord | None = None
    triggers: tuple[ResearchTrigger, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class IngestResult:
    """The whole run, per series, plus the sources that were skipped."""

    started_at: datetime
    finished_at: datetime
    series: list[SeriesResult] = field(default_factory=list)
    skipped_sources: dict[str, str] = field(default_factory=dict)

    @property
    def observations_written(self) -> int:
        return sum(s.observations_written for s in self.series)

    @property
    def failures(self) -> list[SeriesResult]:
        return [s for s in self.series if not s.ok]

    @property
    def new_triggers(self) -> list[ResearchTrigger]:
        return [t for s in self.series for t in s.triggers]


def run_external_ingest(
    cfg: ExternalDataConfig,
    store: Store,
    *,
    adapters: Mapping[str, ExternalSeriesAdapter],
    state_dir: Path,
    now: datetime | None = None,
    sources: Sequence[str] | None = None,
    credentials_present: Mapping[str, bool] | None = None,
) -> IngestResult:
    """Ingest every enabled series, then evaluate readiness for each.

    Args:
        adapters: `source_id -> adapter`. A configured source with no adapter
            is skipped and reported, never silently ignored.
        credentials_present: `source_id -> bool`, injected rather than read
            from the environment here so that the decision is testable and so
            that nothing in this module ever touches a credential value.

    One series failing does not abort the run. An external publisher being
    down for an afternoon is normal, and a run that gives up on the first
    timeout would lose the other nine series' data for that day -- which for
    a FORWARD_ONLY source is a permanent hole, not a retry.
    """
    started = now or datetime.now(UTC)
    result = IngestResult(started_at=started, finished_at=started)
    emitted = store.emitted_trigger_keys()

    wanted = set(sources) if sources is not None else None

    for manifest in cfg.sources.values():
        if wanted is not None and manifest.source_id not in wanted:
            continue

        # Record the manifest itself, so `external sources` can report on a
        # source that has never successfully ingested anything.
        store.upsert_external_source(manifest)

        if not manifest.may_ingest:
            result.skipped_sources[manifest.source_id] = (
                f"{manifest.status}: {manifest.status_note.strip() or 'not enabled'}"
            )
            continue
        if manifest.requires_auth:
            present = (credentials_present or {}).get(manifest.source_id, False)
            if not present:
                result.skipped_sources[manifest.source_id] = (
                    f"credential {manifest.auth_environment_variable} is not set"
                )
                store.upsert_external_source(
                    manifest.model_copy(
                        update={
                            "status": SourceStatus.AUTH_MISSING,
                            "last_attempt": started,
                        }
                    )
                )
                continue
        adapter = adapters.get(manifest.source_id)
        if adapter is None:
            result.skipped_sources[manifest.source_id] = "no adapter registered"
            continue

        specs = cfg.series_for(manifest.source_id)
        if not specs:
            result.skipped_sources[manifest.source_id] = "no series configured"
            continue

        any_success = False
        for spec in specs:
            outcome = _ingest_one(
                spec,
                adapter=adapter,
                store=store,
                state_dir=state_dir,
                cfg=cfg,
                policy_now=started,
                emitted=emitted,
            )
            any_success = any_success or outcome.fetched
            for trigger in outcome.triggers:
                emitted = emitted | {trigger.dedup_key}
            result.series.append(outcome)

        store.upsert_external_source(
            manifest.model_copy(
                update={
                    "last_attempt": started,
                    "last_successful_ingestion": (
                        started if any_success else manifest.last_successful_ingestion
                    ),
                    "status": SourceStatus.PASS if any_success else SourceStatus.FAIL,
                }
            )
        )

    result.finished_at = now or datetime.now(UTC)
    return result


def _ingest_one(
    spec: SeriesSpec,
    *,
    adapter: ExternalSeriesAdapter,
    store: Store,
    state_dir: Path,
    cfg: ExternalDataConfig,
    policy_now: datetime,
    emitted: frozenset[str] | set[str],
) -> SeriesResult:
    out = SeriesResult(series_id=spec.series_id, source=spec.source)
    try:
        payload = adapter.fetch(spec)
        out.fetched = True
        out.payload_bytes = len(payload.content)
        # Archive before parsing: a parser that raises must still leave the
        # evidence on disk, or the bug is unreproducible.
        record = store_payload(payload, state_dir=state_dir, parser_version=adapter.parser_version)
        store.append_raw_payload(record)

        parsed = adapter.parse(payload, spec)
        observations = deduplicate_observations(parsed.observations)
        out.observations_written = store.append_external_observations(observations)
        out.warnings = parsed.warnings
        if parsed.missing_series:
            out.warnings = (
                *out.warnings,
                f"payload did not contain: {list(parsed.missing_series)}",
            )
    except Exception as exc:
        out.error = f"{type(exc).__name__}: {exc}"
        log.warning(
            "external_ingest_series_failed",
            source=spec.source,
            series_id=spec.series_id,
            error=out.error,
        )

    out.readiness = _evaluate_and_persist(
        spec,
        store=store,
        cfg=cfg,
        now=policy_now,
        extra_warnings=out.warnings,
        fetch_error=out.error,
    )
    out.triggers = tuple(triggers_for_transition(out.readiness, already_emitted=frozenset(emitted)))
    for trigger in out.triggers:
        store.upsert_research_trigger(trigger)
        log.info(
            "external_data_ready",
            source=trigger.source,
            series_id=trigger.series_id,
            trigger=str(trigger.trigger_type),
            effective_n=trigger.effective_n,
        )
    return out


def _evaluate_and_persist(
    spec: SeriesSpec,
    *,
    store: Store,
    cfg: ExternalDataConfig,
    now: datetime,
    extra_warnings: Sequence[str],
    fetch_error: str | None,
) -> DataReadinessRecord:
    stored = [
        o
        for o in store.list_external_observations(series_id=spec.series_id)
        if o.source == spec.source
    ]
    previous = store.latest_readiness(spec.source, spec.series_id)
    warnings = list(extra_warnings)
    if fetch_error is not None:
        warnings.append(f"last fetch failed: {fetch_error}")

    evidence = build_evidence(
        spec,
        stored,
        # The first evaluation is what starts the forward clock. Leaving
        # this None until someone sets it by hand would mean a FORWARD_ONLY
        # series never accumulates any forward evidence at all: its
        # readiness is computed from self-archived data, and nothing counts
        # as self-archived until collection_start exists.
        collection_start=previous.collection_start if previous else now,
        parser_warnings=warnings,
    )
    record = evaluate_readiness(
        evidence,
        cfg.policy(),
        now=now,
        enabled=spec.enabled,
        previous=previous,
    )
    store.append_readiness_record(record)
    return record
