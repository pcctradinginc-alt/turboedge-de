"""Tests for the ingestion loop, against a real Store and a fake adapter."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.external.adapter import FetchedPayload, ParseResult
from turboedge.external.catalog import ExternalDataConfig
from turboedge.external.ingest import run_external_ingest
from turboedge.external.readiness import ReadinessState
from turboedge.external.schemas import (
    AvailabilityPrecision,
    BackfillClass,
    ExternalSourceManifest,
    SeriesSpec,
    SourceStatus,
)
from turboedge.external.triggers import TriggerType
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ExternalObservation

_NOW = datetime(2026, 9, 28, 6, 0, tzinfo=UTC)

_SPEC = SeriesSpec(
    source="demo",
    series_id="DEMO.X",
    name="x",
    category="c",
    unit="percent",
    frequency="daily",
    native_identifier="X",
    availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
    backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
    conservative_release_lag_hours=36,
)

_MANIFEST = ExternalSourceManifest(
    source_id="demo",
    display_name="Demo",
    official_source="Demo",
    homepage="https://example.org",
    machine_endpoint="https://example.org/api",
    access_method="REST",
    license_or_terms_reference="https://example.org/terms",
    commercial_use_status="PERMITTED_WITH_ATTRIBUTION",
    frequency="daily",
    expected_update_cadence="daily",
    supports_historical_data=True,
    supports_vintages=False,
    supports_exact_release_time=False,
    point_in_time_quality=AvailabilityPrecision.CONSERVATIVE_DATE,
    backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
    enabled=True,
    status=SourceStatus.PASS,
)


class FakeAdapter:
    """Serves `days` consecutive daily observations."""

    def __init__(self, days: int = 400, *, fail: bool = False, warn: bool = False) -> None:
        self._days = days
        self._fail = fail
        self._warn = warn
        self.fetch_calls = 0

    @property
    def source_id(self) -> str:
        return "demo"

    @property
    def parser_version(self) -> str:
        return "1"

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        self.fetch_calls += 1
        if self._fail:
            raise TimeoutError("upstream did not respond")
        return FetchedPayload(
            source="demo",
            dataset=spec.series_id,
            url="https://example.org/api/X",
            content=f"days={self._days}".encode(),
            http_status=200,
            content_type="application/json",
            retrieved_at=_NOW,
            request_fingerprint="GET /api/X",
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        # Ends yesterday, so the series is fresh: staleness is a separate
        # readiness rule with its own test and must not confound this one.
        start = _NOW.date() - timedelta(days=self._days)
        rows = []
        for i in range(self._days):
            day = start + timedelta(days=i)
            observation_time = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
            rows.append(
                ExternalObservation(
                    series_id=spec.series_id,
                    value=float(i),
                    unit=spec.unit,
                    frequency=spec.frequency,
                    source_version="v1",
                    observation_time=observation_time,
                    available_at=observation_time + timedelta(hours=36),
                    retrieved_at=_NOW,
                    source="demo",
                    parser_version="1",
                    quality_score=1.0,
                    availability_precision=str(AvailabilityPrecision.CONSERVATIVE_DATE),
                )
            )
        return ParseResult(
            observations=rows,
            warnings=("unexpected column FOO",) if self._warn else (),
        )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "t.duckdb") as s:
        s.init_schema()
        yield s


def _config(**over: object) -> ExternalDataConfig:
    manifest = _MANIFEST.model_copy(update=over) if over else _MANIFEST
    return ExternalDataConfig(sources={"demo": manifest}, series=[_SPEC])


def test_a_full_run_writes_observations_readiness_and_one_trigger(
    store: Store, tmp_path: Path
) -> None:
    result = run_external_ingest(
        _config(), store, adapters={"demo": FakeAdapter()}, state_dir=tmp_path, now=_NOW
    )

    assert result.observations_written == 400
    assert result.failures == []
    (series,) = result.series
    assert series.readiness is not None
    assert series.readiness.state is ReadinessState.CONFIRMATION_READY
    assert [t.trigger_type for t in result.new_triggers] == [
        TriggerType.EXPLORATORY_DATA_READY,
        TriggerType.VALIDATION_DATA_READY,
        TriggerType.CONFIRMATION_DATA_READY,
    ]


def test_a_second_run_emits_no_further_triggers(store: Store, tmp_path: Path) -> None:
    cfg = _config()
    run_external_ingest(cfg, store, adapters={"demo": FakeAdapter()}, state_dir=tmp_path, now=_NOW)
    again = run_external_ingest(
        cfg,
        store,
        adapters={"demo": FakeAdapter()},
        state_dir=tmp_path,
        now=_NOW + timedelta(days=1),
    )

    assert again.new_triggers == []
    assert len(store.list_research_triggers()) == 3


def test_the_raw_payload_is_archived_and_recorded(store: Store, tmp_path: Path) -> None:
    run_external_ingest(
        _config(), store, adapters={"demo": FakeAdapter()}, state_dir=tmp_path, now=_NOW
    )

    payloads = store.list_raw_payloads()
    assert len(payloads) == 1
    assert Path(payloads[0].stored_path).is_file()


def test_a_fetch_failure_is_recorded_without_aborting(store: Store, tmp_path: Path) -> None:
    result = run_external_ingest(
        _config(),
        store,
        adapters={"demo": FakeAdapter(fail=True)},
        state_dir=tmp_path,
        now=_NOW,
    )

    (series,) = result.series
    assert series.error is not None and "TimeoutError" in series.error
    assert series.readiness is not None
    # Readiness is still evaluated, and the failure is a blocking reason
    # rather than a silent gap.
    assert series.readiness.state is ReadinessState.BLOCKED
    assert any("last fetch failed" in r for r in series.readiness.blocking_reasons)


def test_a_parser_warning_blocks_readiness(store: Store, tmp_path: Path) -> None:
    result = run_external_ingest(
        _config(),
        store,
        adapters={"demo": FakeAdapter(warn=True)},
        state_dir=tmp_path,
        now=_NOW,
    )

    (series,) = result.series
    assert series.readiness is not None
    assert series.readiness.state is ReadinessState.BLOCKED
    assert result.new_triggers == []


def test_a_source_with_a_missing_credential_is_skipped_not_attempted(
    store: Store, tmp_path: Path
) -> None:
    adapter = FakeAdapter()
    cfg = _config(
        requires_auth=True,
        auth_environment_variable="DEMO_KEY",
        enabled=False,
        status=SourceStatus.AUTH_MISSING,
    )

    result = run_external_ingest(
        cfg, store, adapters={"demo": adapter}, state_dir=tmp_path, now=_NOW
    )

    assert adapter.fetch_calls == 0
    assert "demo" in result.skipped_sources
    assert result.observations_written == 0


def test_an_unresolved_licence_blocks_ingestion(store: Store, tmp_path: Path) -> None:
    adapter = FakeAdapter()
    cfg = _config(enabled=False, status=SourceStatus.REVIEW_REQUIRED)

    run_external_ingest(cfg, store, adapters={"demo": adapter}, state_dir=tmp_path, now=_NOW)

    assert adapter.fetch_calls == 0


def test_a_source_without_an_adapter_is_reported_not_ignored(store: Store, tmp_path: Path) -> None:
    result = run_external_ingest(_config(), store, adapters={}, state_dir=tmp_path, now=_NOW)

    assert result.skipped_sources["demo"] == "no adapter registered"


def test_the_manifest_is_persisted_even_when_the_source_is_skipped(
    store: Store, tmp_path: Path
) -> None:
    run_external_ingest(
        _config(enabled=False, status=SourceStatus.REVIEW_REQUIRED),
        store,
        adapters={},
        state_dir=tmp_path,
        now=_NOW,
    )

    stored = store.get_external_source("demo")
    assert stored is not None
    assert stored.status is SourceStatus.REVIEW_REQUIRED


def test_ingestion_is_idempotent_on_the_observation_key(store: Store, tmp_path: Path) -> None:
    cfg = _config()
    run_external_ingest(cfg, store, adapters={"demo": FakeAdapter()}, state_dir=tmp_path, now=_NOW)
    run_external_ingest(cfg, store, adapters={"demo": FakeAdapter()}, state_dir=tmp_path, now=_NOW)

    assert len(store.list_external_observations(series_id="DEMO.X")) == 400


def test_readiness_history_accumulates_rather_than_overwriting(
    store: Store, tmp_path: Path
) -> None:
    cfg = _config()
    run_external_ingest(cfg, store, adapters={"demo": FakeAdapter()}, state_dir=tmp_path, now=_NOW)
    run_external_ingest(
        cfg,
        store,
        adapters={"demo": FakeAdapter()},
        state_dir=tmp_path,
        now=_NOW + timedelta(days=1),
    )

    history = store.list_readiness_history("demo", "DEMO.X")
    assert len(history) == 2
    assert history[0].evaluated_at < history[1].evaluated_at


def test_sources_filter_limits_the_run(store: Store, tmp_path: Path) -> None:
    adapter = FakeAdapter()
    run_external_ingest(
        _config(),
        store,
        adapters={"demo": adapter},
        state_dir=tmp_path,
        now=_NOW,
        sources=["something_else"],
    )

    assert adapter.fetch_calls == 0
