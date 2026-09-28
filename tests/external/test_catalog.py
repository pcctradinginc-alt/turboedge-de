"""Tests for the curated catalog, including the real shipped config."""

from __future__ import annotations

from pathlib import Path

import pytest

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
