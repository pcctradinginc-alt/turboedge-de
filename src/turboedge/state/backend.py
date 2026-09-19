"""Durable, versioned storage backends for TurboEdge-DE's encrypted state
archives -- Phase H of the improvement programme ("dauerhafte Lernhistorie").

Why this module exists
-----------------------
Today the *entire* learning state (DuckDB with ``forward_ledger``,
``strategy_posteriors``, ``model_registry``, ``research_trials``, plus
``registry/``/``ledger/``/``trials/``) lives exclusively in a GitHub Actions
workflow artifact (``turboedge-state-enc``, 14-day retention -- see
``.github/actions/turboedge-state-pack/action.yml``). If the scheduled
pipeline does not produce a usable artifact for 14 days running -- measured
as a real risk, not a hypothetical one: on 2026-09-14 only one of three
scheduled scans fired, 37 minutes late, and public repos auto-disable
scheduled workflows after 60 days idle -- every posterior, model weight and
forward-ledger entry is gone. ``turboedge-state-backup-enc`` (90-day
retention, packed weekly) softens this but is not restored automatically and
is still bounded.

This module adds an optional, pluggable durable layer *underneath* that
mechanism -- it does not replace it. The three-case restore logic in
``.github/actions/turboedge-state/action.yml`` (A: genuine first run, empty
state is fine; B: prior runs exist but none of the last 50 carry a usable
artifact, abort loudly unless ``allow_fresh_state``; C: usable artifact
found, restore normally) is preserved exactly; a configured backend is
consulted only as an extra fallback *inside* case B, before the loud abort,
never instead of it. See ``docs/durable_state.md`` for the operator-facing
setup guide.

Contract every implementation must satisfy
-------------------------------------------
- ``put`` uploads the bytes already at ``local_path`` completely unchanged.
  Callers are responsible for encrypting *before* calling ``put`` (state
  pack already does -- see ``turboedge.state.archive``/``turboedge.state.
  crypto``); a backend must never see or store plaintext state.
- Every ``put`` creates a brand-new, immutable version; it never overwrites
  or deletes an existing one ("append-only"). A version becomes visible to
  ``list_versions``/``get`` only once it is fully and correctly written --
  an interrupted/failed ``put`` must leave zero new versions behind, never a
  half-written one that a later ``get`` could pick up.
- ``get`` re-verifies the SHA-256 of whatever it downloads/reads against
  what was recorded at ``put`` time and raises :class:`StateBackendError`
  loudly on any mismatch, instead of silently handing back corrupted bytes.
- An unreachable/misconfigured backend raises :class:`StateBackendError`
  from every method -- it never returns an empty result that could be
  mistaken for "no versions exist yet". Silently producing a fresh, empty
  state on a backend failure is exactly the failure mode this module exists
  to prevent; see the module-level warning in the module docstring above and
  ``docs/durable_state.md``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

_ENV_BACKEND_KIND = "TURBOEDGE_STATE_BACKEND"
_ENV_LOCAL_ROOT = "TURBOEDGE_STATE_LOCAL_ROOT"


class StateBackendError(Exception):
    """A backend ``put``/``get``/``list_versions``/``health`` failure, or a
    configuration problem (missing/invalid environment variables).

    Deliberately a single exception type across every backend and every
    failure mode (network error, auth failure, checksum mismatch, missing
    key/version, bad config) -- callers (the CLI, the restore composite
    action script) must treat all of these the same way: loudly, never by
    falling back to an empty state. The message distinguishes the cause.
    """


@dataclass(frozen=True)
class VersionInfo:
    """One immutable version of a key, as returned by ``list_versions``."""

    version_id: str
    created_at: datetime  # tz-aware UTC
    size: int
    sha256: str


@dataclass(frozen=True)
class HealthResult:
    """Result of a backend reachability check (``health()``).

    Unlike the other methods, ``health()`` itself does not raise for an
    ordinary "backend is down" condition -- that is exactly the question it
    answers -- it only raises :class:`StateBackendError` for a genuine
    configuration error (e.g. missing environment variables), since that is
    not something retrying or waiting will fix.
    """

    reachable: bool
    reason: str


@runtime_checkable
class StateBackend(Protocol):
    """What every durable state backend implements.

    ``key`` identifies a logical archive stream (e.g. ``"lean"`` for the
    routine packed state archive); each backend is free to choose its own
    on-disk/on-bucket layout underneath a key, as long as versions within a
    key are totally ordered by ``version_id`` (lexicographic sort ==
    chronological order -- see :func:`new_version_id`) and never mutated or
    removed by anything in this codebase.
    """

    def put(self, local_path: Path, key: str) -> str:
        """Upload the file at ``local_path`` as a new version of ``key``.

        Returns the new version's ``version_id``. Raises
        :class:`StateBackendError` on any failure; a failed ``put`` must
        never leave a partially-written version visible to ``get``/
        ``list_versions``.
        """
        ...

    def get(self, key: str, local_path: Path, version: str | None = None) -> str:
        """Download ``version`` of ``key`` (the latest, if ``version`` is
        ``None``) to ``local_path``, verifying its checksum first.

        Returns the ``version_id`` actually fetched. Writes ``local_path``
        atomically (never a partially-written file on failure). Raises
        :class:`StateBackendError` if the key/version does not exist, the
        backend is unreachable, or the checksum does not match -- never
        silently produces an empty or partial file.
        """
        ...

    def list_versions(self, key: str) -> list[VersionInfo]:
        """All versions of ``key``, oldest first (chronological order).

        Returns an empty list if the key has no versions yet (a backend
        that is merely *empty* is not an error); raises
        :class:`StateBackendError` if the backend itself is unreachable or
        misconfigured -- those two situations must never be conflated.
        """
        ...

    def health(self) -> HealthResult:
        """Check whether the backend is reachable right now, without
        raising for an ordinary "unreachable" result (see
        :class:`HealthResult`)."""
        ...


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's contents, streamed (safe for large archives)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_version_id() -> str:
    """A sortable, collision-resistant version identifier.

    ``<UTC timestamp to the microsecond>Z-<8 hex chars>``, e.g.
    ``20260919T142233123456Z-a1b2c3d4``. Every backend relies on plain
    lexicographic sort of these strings being equivalent to chronological
    order for ``list_versions``.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + "Z"
    return f"{ts}-{uuid4().hex[:8]}"


def parse_created_at(version_id: str, fallback: datetime) -> datetime:
    """Best-effort recovery of the timestamp encoded in a
    :func:`new_version_id` string; falls back to ``fallback`` (e.g. a
    filesystem mtime or an object store's own LastModified) for a
    version_id that was not produced by this scheme. Shared by every
    backend implementation (:mod:`turboedge.state.backend_s3` included)."""
    ts_part = version_id.split("-")[0].rstrip("Z")
    try:
        return datetime.strptime(ts_part, "%Y%m%dT%H%M%S%f").replace(tzinfo=UTC)
    except ValueError:
        return fallback


def _validate_key(key: str) -> None:
    if not key or key in {".", ".."} or "/" in key or "\\" in key:
        raise StateBackendError(f"invalid backend key: {key!r}")


class LocalStateBackend:
    """A backend rooted at a configurable local directory.

    Not itself "remote" -- but it is the same versioned/atomic/
    checksum-verified contract as the network backend, and is genuinely
    durable when ``root`` is something other than the ephemeral pipeline
    runner's own workspace (e.g. a mounted network share, an rclone/Dropbox
    sync target, or simply a path a human periodically copies elsewhere).
    Selected via ``TURBOEDGE_STATE_BACKEND=local`` +
    ``TURBOEDGE_STATE_LOCAL_ROOT`` (see :func:`backend_from_env`); also used
    directly by tests as a fast, hermetic stand-in for exercising the same
    contract the S3-compatible backend implements.

    On-disk layout under ``root``: ``<key>/<version_id>.bin`` (the archive
    bytes, written via temp-file + ``os.replace``) plus a sidecar
    ``<key>/<version_id>.sha256`` (``"<sha256>  <size>\\n"``) written only
    after the ``.bin`` is confirmed intact -- ``list_versions``/``get``
    require both files to exist before treating a version as valid, so an
    interruption between the two writes never surfaces a half version.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _key_dir(self, key: str) -> Path:
        _validate_key(key)
        return self._root / key

    def put(self, local_path: Path, key: str) -> str:
        local_path = Path(local_path)
        if not local_path.is_file():
            raise StateBackendError(f"put: no such file: {local_path}")

        expected_sha256 = sha256_file(local_path)
        size = local_path.stat().st_size
        version_id = new_version_id()
        key_dir = self._key_dir(key)
        key_dir.mkdir(parents=True, exist_ok=True)

        dest = key_dir / f"{version_id}.bin"
        tmp = key_dir / f".{version_id}.bin.tmp-{uuid4().hex}"
        try:
            shutil.copyfile(local_path, tmp)
            # Re-verify the copy before it becomes visible (the atomic
            # rename below) as a version -- a crash/interruption mid-copy
            # leaves only the still-hidden .tmp file behind.
            actual_sha256 = sha256_file(tmp)
            if actual_sha256 != expected_sha256:
                raise StateBackendError(
                    f"put: checksum mismatch copying {local_path} for key {key!r} "
                    f"(expected {expected_sha256}, got {actual_sha256}) -- no version recorded"
                )
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        meta_tmp = key_dir / f".{version_id}.sha256.tmp-{uuid4().hex}"
        meta_path = key_dir / f"{version_id}.sha256"
        try:
            meta_tmp.write_text(f"{expected_sha256}  {size}\n", encoding="utf-8")
            os.replace(meta_tmp, meta_path)
        finally:
            if meta_tmp.exists():
                meta_tmp.unlink(missing_ok=True)

        return version_id

    def get(self, key: str, local_path: Path, version: str | None = None) -> str:
        key_dir = self._key_dir(key)
        chosen = version
        if chosen is None:
            versions = self.list_versions(key)
            if not versions:
                raise StateBackendError(f"get: no versions found for key {key!r} in {key_dir}")
            chosen = versions[-1].version_id  # list_versions is chronological

        src = key_dir / f"{chosen}.bin"
        meta = key_dir / f"{chosen}.sha256"
        if not src.is_file() or not meta.is_file():
            raise StateBackendError(
                f"get: version {chosen!r} of key {key!r} not found (or incomplete) in {key_dir}"
            )

        expected_sha256 = meta.read_text(encoding="utf-8").split()[0]
        actual_sha256 = sha256_file(src)
        if actual_sha256 != expected_sha256:
            raise StateBackendError(
                f"get: checksum mismatch for key {key!r} version {chosen!r} -- expected "
                f"{expected_sha256}, got {actual_sha256}; refusing to hand back corrupted data"
            )

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_name(f".{local_path.name}.tmp-{uuid4().hex}")
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, local_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        return chosen

    def list_versions(self, key: str) -> list[VersionInfo]:
        key_dir = self._key_dir(key)
        if not key_dir.is_dir():
            return []

        out: list[VersionInfo] = []
        for bin_path in sorted(key_dir.glob("*.bin")):
            version_id = bin_path.stem
            meta_path = key_dir / f"{version_id}.sha256"
            if not meta_path.is_file():
                # An interrupted put's .bin without its .sha256 sidecar: not a valid version.
                continue
            line = meta_path.read_text(encoding="utf-8")
            sha256_val = line.split()[0] if line.split() else ""
            fallback = datetime.fromtimestamp(bin_path.stat().st_mtime, tz=UTC)
            out.append(
                VersionInfo(
                    version_id=version_id,
                    created_at=parse_created_at(version_id, fallback),
                    size=bin_path.stat().st_size,
                    sha256=sha256_val,
                )
            )
        out.sort(key=lambda v: v.version_id)
        return out

    def health(self) -> HealthResult:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            probe = self._root / f".health-{uuid4().hex}"
            probe.write_bytes(b"ok")
            probe.unlink()
            return HealthResult(
                reachable=True, reason=f"local backend root {self._root} is writable"
            )
        except OSError as exc:
            return HealthResult(reachable=False, reason=f"local backend root not writable: {exc}")


def backend_from_env() -> StateBackend | None:
    """Construct the durable backend configured via ``TURBOEDGE_STATE_BACKEND``.

    Returns ``None`` -- deliberately not an error -- when
    ``TURBOEDGE_STATE_BACKEND`` is unset/empty/``"none"``/``"off"``: the
    durable backend is entirely optional, and every caller (CLI, the
    restore composite action) must treat "not configured" as simply
    "nothing to fall back to" and keep going with the existing GitHub
    Actions artifact flow, exactly as before this module existed.

    Raises :class:`StateBackendError` for a recognized-but-misconfigured
    kind (e.g. ``"s3"`` without its required environment variables) or an
    unrecognized kind -- those are conscious configuration mistakes, not
    "backend absent", and must not be swallowed into a silent no-op.
    """
    kind = os.environ.get(_ENV_BACKEND_KIND, "").strip().lower()
    if kind in ("", "none", "off"):
        return None
    if kind == "local":
        root = os.environ.get(_ENV_LOCAL_ROOT, "").strip()
        if not root:
            raise StateBackendError(
                f"{_ENV_BACKEND_KIND}=local requires {_ENV_LOCAL_ROOT} to also be set"
            )
        return LocalStateBackend(Path(root))
    if kind == "s3":
        from turboedge.state.backend_s3 import S3CompatibleStateBackend

        return S3CompatibleStateBackend()
    raise StateBackendError(
        f"unknown {_ENV_BACKEND_KIND}={kind!r} (expected 's3', 'local', 'none'/'off', or unset)"
    )


__all__ = [
    "HealthResult",
    "LocalStateBackend",
    "StateBackend",
    "StateBackendError",
    "VersionInfo",
    "backend_from_env",
    "new_version_id",
    "parse_created_at",
    "sha256_file",
]
