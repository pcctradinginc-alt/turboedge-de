"""The curated series catalog (spec §9).

"Do not hard-code hundreds of arbitrary series." The constraint is not about
disk. Every configured series is a hypothesis-in-waiting, and a catalog
assembled by downloading whatever an API offers turns the research programme
into a fishing expedition whose multiple-testing denominator nobody can
state. Each entry here is meant to be defensible on its own: a named
economic quantity, with a reason, whose availability semantics are known.

The catalog is also the only place that decides what is *enabled*. An
adapter existing does not make a source live; a source is live when someone
wrote down its licence reference, its credential requirement and its
point-in-time class, and set `enabled: true`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from turboedge.external.readiness import DEFAULT_PROFILES, ReadinessPolicy, ReadinessProfile
from turboedge.external.schemas import ExternalSourceManifest, SeriesSpec


class ExternalDataConfig(BaseModel):
    """Everything the Data Factory needs, loaded from `external_data.yaml`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: dict[str, ExternalSourceManifest]
    series: list[SeriesSpec]
    readiness_profiles: dict[str, ReadinessProfile] = {}
    readiness_policy_version: str = "1"
    max_missingness: float = 0.10
    max_staleness_days: int = 45
    max_staleness_periods: float = 2.5
    require_strict_pit_for_confirmation: bool = True

    @model_validator(mode="after")
    def _every_series_has_a_registered_source(self) -> Self:
        """A series pointing at an unregistered source would be silently
        skipped at ingestion time and silently absent from every readiness
        report -- a configuration typo that looks exactly like "not ready
        yet"."""
        unknown = sorted({s.source for s in self.series} - set(self.sources))
        if unknown:
            raise ValueError(
                f"series configured for unregistered source(s) {unknown}; "
                f"registered: {sorted(self.sources)}"
            )
        return self

    @model_validator(mode="after")
    def _source_ids_match_their_keys(self) -> Self:
        mismatched = [k for k, v in self.sources.items() if v.source_id != k]
        if mismatched:
            raise ValueError(
                f"source key does not match its source_id for {mismatched}; "
                "one of the two would end up in the database and the other in logs"
            )
        return self

    @model_validator(mode="after")
    def _series_ids_are_unique_per_source(self) -> Self:
        seen: set[str] = set()
        duplicates: list[str] = []
        for spec in self.series:
            if spec.qualified_id in seen:
                duplicates.append(spec.qualified_id)
            seen.add(spec.qualified_id)
        if duplicates:
            raise ValueError(
                f"duplicate series {sorted(set(duplicates))}; the storage key would "
                "collapse them into one and the second definition would win silently"
            )
        return self

    def policy(self) -> ReadinessPolicy:
        profiles = {**DEFAULT_PROFILES, **self.readiness_profiles}
        return ReadinessPolicy(
            policy_version=self.readiness_policy_version,
            profiles=profiles,
            max_missingness=self.max_missingness,
            max_staleness_days=self.max_staleness_days,
            max_staleness_periods=self.max_staleness_periods,
            require_strict_pit_for_confirmation=self.require_strict_pit_for_confirmation,
        )

    def series_for(self, source: str) -> list[SeriesSpec]:
        return [s for s in self.series if s.source == source and s.enabled]

    def ingestible_sources(self) -> list[ExternalSourceManifest]:
        """Sources that may actually be fetched right now.

        A source whose credential is missing, whose terms are unresolved or
        which is deferred is excluded here rather than failing later, so that
        "we did not fetch this" is a stated decision in one place instead of
        an exception in six adapters.
        """
        return [m for m in self.sources.values() if m.may_ingest]


def load_external_data_config(path: str | Path) -> ExternalDataConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return ExternalDataConfig.model_validate(raw)
