"""Negative Memory / Hypothesenfriedhof (Master Spec §23).

Failed research hypotheses are never deleted -- they are persisted so the
same idea is not blindly re-tried every quarter. Storage is a single JSON
file (not DuckDB) per Master Spec §23: ``state/registry/failed_hypotheses.json``.

File schema -- a JSON array of objects, each shaped like::

    {
        "feature": "rsi_14",              # str: what was tried
        "horizons": ["5d", "10d"],        # list[str]: horizons it was tested on
        "incremental_net_ev": -0.0012,    # float: OOS net-EV delta vs. baseline
        "effective_sample": 412,          # int: average-uniqueness-adjusted n
        "status": "dormant",              # "dormant" | "retest_allowed"
        "trial_id": "TR-2026Q3-a1b2c3",   # str | None: trial that tested it
        "recorded_at": "2026-09-10T12:00:00+00:00",  # ISO 8601 UTC
        "regime_change_note": null        # str | None: why a retest was allowed
    }

(Master Spec §23's own example omits ``trial_id``/``recorded_at``/
``regime_change_note``; they are additive fields this module writes for
traceability -- CLAUDE.md rule 33 -- and are always present, ``None`` when
not applicable, so every record has a stable, complete shape.)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_HypothesisStatus = Literal["dormant", "retest_allowed"]


class FailedHypothesesConfig(BaseModel):
    """This module's own config (wired into config.py/YAML by the
    integration wave)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = "registry/failed_hypotheses.json"


class FailedHypothesis(BaseModel):
    """One entry of the negative-memory registry (Master Spec §23)."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    horizons: list[str] = Field(default_factory=list)
    incremental_net_ev: float
    effective_sample: int = Field(ge=0)
    status: _HypothesisStatus = "dormant"
    trial_id: str | None = None
    recorded_at: datetime
    regime_change_note: str | None = None


def default_path(state_dir: Path, config: FailedHypothesesConfig | None = None) -> Path:
    cfg = config if config is not None else FailedHypothesesConfig()
    return state_dir / cfg.relative_path


def load(path: Path) -> list[FailedHypothesis]:
    """Read the negative-memory registry. Returns ``[]`` if the file does
    not exist yet (nothing failed yet is a valid, non-error state)."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array at the top level, got {type(raw)!r}")
    return [FailedHypothesis.model_validate(item) for item in raw]


def _write_all(path: Path, hypotheses: list[FailedHypothesis]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [h.model_dump(mode="json") for h in hypotheses]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path: Path, hypothesis: FailedHypothesis) -> list[FailedHypothesis]:
    """Append one failed hypothesis to the registry (never overwrites or
    removes existing entries -- CLAUDE.md rule 31: "Positive Muster
    verstärken, negative Muster nicht löschen."). Returns the full,
    updated list."""
    hypotheses = load(path)
    hypotheses.append(hypothesis)
    _write_all(path, hypotheses)
    return hypotheses


def allow_retest(
    path: Path,
    feature: str,
    regime_change_note: str,
    *,
    horizons: list[str] | None = None,
    now: datetime | None = None,
) -> list[FailedHypothesis]:
    """Flip matching dormant entries for ``feature`` to ``retest_allowed``.

    Master Spec §23: "Nur bei klar dokumentiertem Regimewechsel erneut
    testen." -- so this always requires a non-empty ``regime_change_note``
    documenting *why* a retest is now permitted, and it edits the status
    field of matching entries in place (does not delete/replace them,
    preserving the original record of what failed and when).

    Args:
        path: Path to the registry JSON file.
        feature: Match entries whose ``feature`` equals this exactly.
        regime_change_note: Required justification, stored on each matched
            entry.
        horizons: If given, only entries whose ``horizons`` list intersects
            this one are matched; otherwise all entries for ``feature``.
        now: Clock override for testing.

    Returns:
        The full, updated list of hypotheses.

    Raises:
        ValueError: If ``regime_change_note`` is blank, or no entry matches.
    """
    if not regime_change_note.strip():
        raise ValueError(
            "regime_change_note is required to allow a retest (Master Spec §23: "
            "'Nur bei klar dokumentiertem Regimewechsel erneut testen.')"
        )
    hypotheses = load(path)
    matched = False
    updated: list[FailedHypothesis] = []
    for h in hypotheses:
        matches = h.feature == feature and (
            horizons is None or bool(set(h.horizons) & set(horizons))
        )
        if matches:
            matched = True
            updated.append(
                h.model_copy(
                    update={
                        "status": "retest_allowed",
                        "regime_change_note": regime_change_note,
                    }
                )
            )
        else:
            updated.append(h)
    if not matched:
        raise ValueError(f"no failed_hypotheses entry found for feature={feature!r}")
    _write_all(path, updated)
    return updated


def is_dormant(path: Path, feature: str) -> bool:
    """True iff any entry for ``feature`` is currently ``dormant`` (i.e.
    should not be re-tried without going through :func:`allow_retest`
    first)."""
    return any(h.feature == feature and h.status == "dormant" for h in load(path))


def new_hypothesis(
    feature: str,
    horizons: list[str],
    incremental_net_ev: float,
    effective_sample: int,
    *,
    trial_id: str | None = None,
    now: datetime | None = None,
) -> FailedHypothesis:
    """Convenience constructor with ``status="dormant"`` and
    ``recorded_at`` defaulted to now -- the normal case when a trial's
    out-of-sample result fails the promotion ladder."""
    return FailedHypothesis(
        feature=feature,
        horizons=horizons,
        incremental_net_ev=incremental_net_ev,
        effective_sample=effective_sample,
        status="dormant",
        trial_id=trial_id,
        recorded_at=now if now is not None else datetime.now(UTC),
        regime_change_note=None,
    )


__all__ = [
    "FailedHypothesesConfig",
    "FailedHypothesis",
    "allow_retest",
    "append",
    "default_path",
    "is_dormant",
    "load",
    "new_hypothesis",
]
