"""Tests for the curated catalog, including the real shipped config."""

from __future__ import annotations

from pathlib import Path

import pytest

from turboedge.adapters.base import HttpClient
from turboedge.config import external_data_path
from turboedge.external.catalog import ExternalDataConfig, load_external_data_config
from turboedge.external.schemas import (
    NON_INGESTING_STATUSES,
    AvailabilityPrecision,
    BackfillClass,
    ExternalSourceManifest,
    SeriesSpec,
    SourceStatus,
)

_REPO_CONFIG = Path("configs/external_data.yaml")


def _manifest(**over: object) -> ExternalSourceManifest:
    defaults: dict[str, object] = dict(
        source_id="s",
        display_name="S",
        official_source="O",
        homepage="https://example.org",
        machine_endpoint="https://example.org/api",
        access_method="REST",
        license_or_terms_reference="https://example.org/terms",
        frequency="daily",
        expected_update_cadence="daily",
        supports_historical_data=True,
        supports_vintages=False,
        supports_exact_release_time=False,
        point_in_time_quality=AvailabilityPrecision.CONSERVATIVE_DATE,
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
        status=SourceStatus.PASS,
    )
    defaults.update(over)
    return ExternalSourceManifest(**defaults)  # type: ignore[arg-type]


def _spec(**over: object) -> SeriesSpec:
    defaults: dict[str, object] = dict(
        source="s",
        series_id="S.X",
        name="x",
        category="c",
        unit="u",
        frequency="daily",
        native_identifier="X",
        availability_precision=AvailabilityPrecision.EXACT_DATE,
        backfill_class=BackfillClass.HISTORICAL_CONSERVATIVE,
    )
    defaults.update(over)
    return SeriesSpec(**defaults)  # type: ignore[arg-type]


def test_the_shipped_catalog_loads() -> None:
    cfg = load_external_data_config(_REPO_CONFIG)

    assert cfg.sources
    assert cfg.series


def test_external_data_path_points_at_the_shipped_file() -> None:
    assert external_data_path("configs") == _REPO_CONFIG


def test_every_shipped_series_belongs_to_a_registered_source() -> None:
    cfg = load_external_data_config(_REPO_CONFIG)

    assert {s.source for s in cfg.series} <= set(cfg.sources)


def test_shipped_sources_requiring_auth_name_their_variable() -> None:
    cfg = load_external_data_config(_REPO_CONFIG)

    for manifest in cfg.sources.values():
        if manifest.requires_auth:
            assert manifest.auth_environment_variable
            # And must not be live until the credential exists.
            assert not manifest.may_ingest or manifest.status is SourceStatus.PASS


def test_shipped_enabled_sources_have_a_licence_reference() -> None:
    cfg = load_external_data_config(_REPO_CONFIG)

    for manifest in cfg.ingestible_sources():
        assert manifest.license_or_terms_reference.startswith("http")
        assert manifest.commercial_use_status != "UNKNOWN"


def test_shipped_forward_only_series_are_not_claimed_as_pit_safe() -> None:
    cfg = load_external_data_config(_REPO_CONFIG)

    for spec in cfg.series:
        if spec.backfill_class is BackfillClass.FORWARD_ONLY:
            assert spec.availability_precision in {
                AvailabilityPrecision.UNKNOWN,
                AvailabilityPrecision.INFERRED,
                AvailabilityPrecision.CONSERVATIVE_DATE,
            }


def test_a_series_pointing_at_an_unknown_source_is_refused() -> None:
    with pytest.raises(ValueError, match="unregistered source"):
        ExternalDataConfig(sources={"s": _manifest()}, series=[_spec(source="nope")])


def test_a_mismatched_source_key_is_refused() -> None:
    with pytest.raises(ValueError, match="does not match its source_id"):
        ExternalDataConfig(sources={"other": _manifest(source_id="s")}, series=[])


def test_duplicate_series_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate series"):
        ExternalDataConfig(sources={"s": _manifest()}, series=[_spec(), _spec()])


@pytest.mark.parametrize("status", sorted(NON_INGESTING_STATUSES))
def test_enabled_plus_a_blocking_status_is_refused(status: SourceStatus) -> None:
    # The combination that reads like a live source and behaves like a dead one.
    with pytest.raises(ValueError, match="incompatible with status"):
        _manifest(enabled=True, status=status)


def test_auth_without_a_named_variable_is_refused() -> None:
    with pytest.raises(ValueError, match="no auth_environment_variable"):
        _manifest(requires_auth=True)


def test_conservative_date_without_a_lag_is_refused() -> None:
    with pytest.raises(ValueError, match="conservative_release_lag_hours"):
        _spec(availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE)


def test_ingestible_sources_excludes_everything_unresolved() -> None:
    cfg = ExternalDataConfig(
        sources={
            "live": _manifest(source_id="live", enabled=True, status=SourceStatus.PASS),
            "noauth": _manifest(
                source_id="noauth",
                requires_auth=True,
                auth_environment_variable="X_KEY",
                status=SourceStatus.AUTH_MISSING,
            ),
            "unclear": _manifest(source_id="unclear", status=SourceStatus.REVIEW_REQUIRED),
        },
        series=[],
    )

    assert [m.source_id for m in cfg.ingestible_sources()] == ["live"]


def test_policy_merges_overrides_onto_the_defaults() -> None:
    cfg = load_external_data_config(_REPO_CONFIG)
    policy = cfg.policy()

    assert policy.profile_for("monthly").min_exploratory_observations == 48
    assert policy.profile_for("business_daily").min_exploratory_observations == 180


def test_every_blocked_wave_two_source_says_why_and_the_reasons_stay_distinct() -> None:
    # Two reasons that must not be conflated: a missing credential is a fact
    # about this machine, an unread licence is a fact about the publisher.
    # Collapsing them would hide which ones a key would fix.
    cfg = load_external_data_config(_REPO_CONFIG)

    credential_blocked = {"agsi", "alsi", "entsoe"}
    licence_blocked = {"portwatch", "kiel_trade"}

    for source_id in credential_blocked:
        manifest = cfg.sources[source_id]
        assert not manifest.enabled, source_id
        assert manifest.status is SourceStatus.AUTH_MISSING, source_id
        assert manifest.auth_environment_variable, source_id
        assert manifest.status_note.strip(), source_id

    for source_id in licence_blocked:
        manifest = cfg.sources[source_id]
        assert not manifest.enabled, source_id
        assert manifest.status is SourceStatus.REVIEW_REQUIRED, source_id
        # These need no key at all -- only a human to read the terms.
        assert not manifest.requires_auth, source_id
        assert manifest.status_note.strip(), source_id


def test_eia_is_live_on_a_ci_secret_and_says_its_series_are_unconfirmed() -> None:
    # EIA_API_KEY exists as a repository secret but not in the authoring
    # environment, so its series were configured from documentation and the
    # first CI run is their verification. That has to be stated, not implied
    # by silence -- otherwise a wrong facet looks like a quiet source.
    cfg = load_external_data_config(_REPO_CONFIG)
    manifest = cfg.sources["eia"]

    assert manifest.enabled
    assert manifest.status is SourceStatus.PASS
    assert "not present in the authoring environment" in manifest.status_note

    series = cfg.series_for("eia")
    assert series
    for spec in series:
        assert "NOT verified live" in spec.notes, spec.series_id


def test_agsi_and_alsi_share_one_credential() -> None:
    cfg = load_external_data_config(_REPO_CONFIG)

    assert (
        cfg.sources["agsi"].auth_environment_variable
        == cfg.sources["alsi"].auth_environment_variable
        == "GIE_API_KEY"
    )


def test_every_configured_series_has_an_adapter_registered() -> None:
    # A configured series whose source has no entry in the CLI's adapter
    # table is skipped with "no adapter registered" -- which reads exactly
    # like "not ready yet" in the readiness report. Wave 1 lost a source to
    # a mistyped class name this way.
    from turboedge.cli_external import _ADAPTER_CLASSES

    cfg = load_external_data_config(_REPO_CONFIG)
    missing = sorted({s.source for s in cfg.series} - set(_ADAPTER_CLASSES))

    assert missing == []


def test_every_adapter_in_the_table_can_actually_be_built() -> None:
    # The table maps source ids to module and class names as strings, so a
    # renamed class fails only at runtime. This catches it in CI instead.
    import importlib

    from turboedge.cli_external import _ADAPTER_CLASSES
    from turboedge.external.adapter import ExternalSeriesAdapter

    for source_id, (module_name, class_name) in _ADAPTER_CLASSES.items():
        module = importlib.import_module(module_name)
        factory = getattr(module, class_name, None)
        assert factory is not None, f"{source_id}: {module_name} has no {class_name}"
        instance = factory(HttpClient(user_agent="test"))
        assert isinstance(instance, ExternalSeriesAdapter), source_id
        assert instance.source_id == source_id
