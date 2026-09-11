"""Reproducibility helpers: git commit resolution and content hashing.

Every persisted prediction, signal and scan run must be traceable back to the
exact code and data that produced it (CLAUDE.md rule 33: "Every prediction
fully reproducibly stored"). This module provides the small set of primitives
that make that possible without pulling storage or config concerns in.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel


def git_commit() -> str | None:
    """Return the current git commit SHA, or ``None`` if it cannot be determined.

    Resolution order:

    1. The ``GITHUB_SHA`` environment variable (set by GitHub Actions runners).
    2. ``git rev-parse HEAD`` executed in the current working directory.

    Never raises. Any failure (git not installed, not inside a repository,
    detached filesystem, timeout) is treated as "unknown" so callers can still
    persist provenance records when git metadata happens to be unavailable
    (e.g. inside a stripped deployment artifact).
    """
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha.strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _default_json(value: Any) -> Any:
    """Fallback encoder for :func:`json.dumps` used by :func:`sha256_json`."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding: sorted keys, no incidental whitespace."""
    payload: Any = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_default_json)


def sha256_json(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON encoding of ``obj``.

    Used for ``config_hash``, ``data_snapshot_hash`` and any other
    content-addressable hash that must be stable across process and machine
    boundaries (same input -> same hash, independent of dict insertion order
    or object identity).
    """
    canonical = _canonical_json(obj)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def data_snapshot_hash(records: Sequence[BaseModel | dict[str, Any]]) -> str:
    """SHA-256 hash over an ordered sequence of records (pydantic models or dicts).

    Two calls with the same records in the same order always produce the same
    hash, regardless of the process or machine that computed it. This is what
    ``SignalSnapshot.data_snapshot_hash`` and the immutable Parquet archive
    (``storage/snapshots.py``) use to prove exactly which underlying data a
    prediction or run was frozen against.
    """
    encoded = [
        record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        for record in records
    ]
    return sha256_json(encoded)


def new_run_id() -> str:
    """Generate a new, roughly time-sortable, globally unique run identifier.

    Format: ``<UTC timestamp, second precision>-<uuid4 hex[:12]>``, e.g.
    ``20260910T193000Z-9f3a1c2b4d5e``. Lexicographic sort approximates
    chronological order, which is convenient when browsing
    ``state/snapshots/<table>/date=YYYY-MM-DD/<run_id>.parquet`` or the
    ``runs`` table.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"
