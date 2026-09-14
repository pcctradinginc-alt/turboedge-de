"""Encrypted, integrity-checked tar.gz archives of the state directory.

``pack_state`` checkpoints DuckDB (opens/closes a :class:`~turboedge.storage.
duckdb.Store` and issues ``CHECKPOINT`` so the on-disk ``.duckdb`` file has no
pending WAL), tars up the relevant subtree of ``state_dir`` (the DuckDB file
plus ``snapshots/``, ``registry/``, ``ledger/``, ``trials/`` -- never
``imports/`` or temporary WAL/tmp sidecar files), embeds a ``MANIFEST.json``
(file list, SHA-256 per file, ``created_at``, ``schema_version``,
``git_commit``), and encrypts the whole tar.gz with
:func:`turboedge.state.crypto.encrypt_bytes`.

``unpack_state`` decrypts, verifies every file against the manifest, rejects
any archive member that could escape ``state_dir`` (absolute path, ``..``,
symlink/hardlink, or anything that isn't a plain file), and only then writes
-- to a temp directory, atomically renamed into place -- so a wrong
key/corrupted archive never touches the existing state directory.

Two archive "shapes" share this module, both via :data:`ArchiveConfig.
include_dirs`:

- :data:`LEAN_INCLUDE_DIRS` (``registry``, ``ledger``, ``trials`` -- no
  ``snapshots``): the small package packed several times a day by every
  scan/eod/monthly pipeline run (``turboedge state pack``'s default).
  ``state/snapshots/`` -- the immutable, ever-growing per-run Parquet
  archive (``storage/snapshots.py``) -- does not belong in an artifact that
  gets re-uploaded 5x/day; see ``turboedge-state-pack/action.yml``.
- :data:`FULL_INCLUDE_DIRS` (adds ``snapshots``): the periodic full backup
  packed weekly (``turboedge state pack --include-snapshots``), uploaded as
  its own longer-retained artifact.

:func:`unpack_state_subset` additively restores just the ``snapshots/``
subtree of a full archive onto a state_dir a lean ``unpack_state`` call
already populated, without touching (or requiring the presence of)
anything else -- used by pipeline.yml's weekly job so the accumulated
Parquet history from before daily packs stopped including it is never
silently dropped (CLAUDE.md rule 33), even though it no longer gets
re-verified/re-replaced as a whole on every run.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from turboedge.provenance import git_commit
from turboedge.state.crypto import decrypt_bytes, encrypt_bytes
from turboedge.storage.duckdb import Store

ARCHIVE_SCHEMA_VERSION = 1
MANIFEST_NAME = "MANIFEST.json"
DEFAULT_DB_FILENAME = "turboedge.duckdb"

_DEFAULT_INCLUDE_DIRS: tuple[str, ...] = ("snapshots", "registry", "ledger", "trials")
#: The full archive shape (default `ArchiveConfig.include_dirs`, and the
#: `--include-snapshots` `state pack` shape): everything, snapshots included.
FULL_INCLUDE_DIRS: tuple[str, ...] = _DEFAULT_INCLUDE_DIRS
#: The lean archive shape (`state pack`'s default): everything except
#: `snapshots/` -- see the module docstring.
LEAN_INCLUDE_DIRS: tuple[str, ...] = tuple(d for d in _DEFAULT_INCLUDE_DIRS if d != "snapshots")
#: Manifest-path prefix (posix, trailing slash) identifying the immutable
#: Parquet archive subtree -- what `unpack_state_subset` restores.
SNAPSHOTS_PREFIX = "snapshots/"
# Directory names never included even if nested under one of the include
# dirs above (currently none are, but this stays defensive).
_EXCLUDE_DIR_NAMES: frozenset[str] = frozenset({"imports"})
# DuckDB WAL and any *.tmp sidecar files are transient/inconsistent by
# definition; CHECKPOINT (see `_checkpoint` below) removes the WAL before
# packing, but this exclusion is kept as defense in depth.
_EXCLUDE_SUFFIXES: tuple[str, ...] = (".wal", ".tmp")


class ArchiveConfig(BaseModel):
    """What :func:`pack_state`/:func:`unpack_state` treat as "the state
    directory" -- the DuckDB file plus the additive-record subdirectories.
    Defaults match the current layout (``state/`` per
    :func:`turboedge.config.default_state_dir`); overridable for tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    db_filename: str = DEFAULT_DB_FILENAME
    include_dirs: tuple[str, ...] = _DEFAULT_INCLUDE_DIRS


class StateArchiveError(Exception):
    """Archive-level failure: missing/unsafe member, manifest/hash mismatch,
    or a structurally invalid archive (not raised for crypto failures --
    those are :class:`~turboedge.state.crypto.StateCryptoError`)."""


@dataclass(frozen=True)
class ManifestFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class PackResult:
    out_path: Path
    file_count: int
    total_bytes: int
    archive_bytes: int


@dataclass(frozen=True)
class UnpackResult:
    state_dir: Path
    file_count: int
    total_bytes: int


def _iter_include_files(state_dir: Path, config: ArchiveConfig) -> list[Path]:
    """Every file `pack_state` includes, in a deterministic (sorted) order."""
    files: list[Path] = []

    db_path = state_dir / config.db_filename
    if db_path.is_file():
        files.append(db_path)

    for dirname in config.include_dirs:
        base = state_dir / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in _EXCLUDE_SUFFIXES:
                continue
            rel_parts = path.relative_to(state_dir).parts
            if any(part in _EXCLUDE_DIR_NAMES for part in rel_parts):
                continue
            files.append(path)

    return files


def _checkpoint(db_path: Path) -> None:
    """Flush DuckDB's WAL into the main file via ``CHECKPOINT``.

    ``Store`` (owned by another workstream's file, ``storage/duckdb.py``,
    outside this module's assignment) does not expose its connection through
    a public method, so this reaches the private ``_conn`` attribute rather
    than adding a new public method to a shared file. A missing db file is
    fine -- ``Store`` creates an empty one, which is then simply an empty
    (but valid, checkpointed) database in the archive.
    """
    with Store(db_path) as store:
        store.init_schema()
        store._conn.execute("CHECKPOINT")


def pack_state(
    state_dir: Path,
    out_path: Path,
    passphrase: str,
    *,
    config: ArchiveConfig | None = None,
) -> PackResult:
    """Checkpoint DuckDB, tar.gz + manifest the state directory, encrypt it.

    Writes ``out_path`` atomically (temp file in the same directory, then
    ``os.replace``). Raises :class:`~turboedge.state.crypto.StateCryptoError`
    if ``passphrase`` is shorter than
    :data:`turboedge.state.crypto.MIN_PASSPHRASE_LEN`.
    """
    cfg = config or ArchiveConfig()
    state_dir = Path(state_dir)
    out_path = Path(out_path)

    db_path = state_dir / cfg.db_filename
    has_any_state = db_path.exists() or any((state_dir / d).is_dir() for d in cfg.include_dirs)
    if has_any_state:
        _checkpoint(db_path)

    files = _iter_include_files(state_dir, cfg)

    buf = io.BytesIO()
    manifest_files: list[ManifestFile] = []
    total_bytes = 0
    now = datetime.now(UTC)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in files:
            data = path.read_bytes()
            digest = sha256(data).hexdigest()
            arcname = path.relative_to(state_dir).as_posix()

            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            info.mtime = int(path.stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))

            manifest_files.append(ManifestFile(path=arcname, sha256=digest, size=len(data)))
            total_bytes += len(data)

        manifest: dict[str, Any] = {
            "created_at": now.isoformat(),
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "git_commit": git_commit(),
            "files": [{"path": f.path, "sha256": f.sha256, "size": f.size} for f in manifest_files],
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        manifest_info = tarfile.TarInfo(name=MANIFEST_NAME)
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = int(now.timestamp())
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

    encrypted = encrypt_bytes(buf.getvalue(), passphrase)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp-{uuid4().hex}")
    try:
        tmp_path.write_bytes(encrypted)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return PackResult(
        out_path=out_path,
        file_count=len(manifest_files),
        total_bytes=total_bytes,
        archive_bytes=len(encrypted),
    )


def _validate_member(member: tarfile.TarInfo) -> None:
    """Raise :class:`StateArchiveError` for any member that could escape
    ``state_dir`` on extraction, or that isn't a plain file/directory."""
    name = member.name
    if not name or name.startswith("/") or name.startswith("\\"):
        raise StateArchiveError(f"unsafe archive member (absolute path): {name!r}")

    normalized = os.path.normpath(name)
    if normalized == ".." or normalized.startswith(f"..{os.sep}") or normalized.startswith("/"):
        raise StateArchiveError(f"unsafe archive member (path traversal): {name!r}")

    if member.issym() or member.islnk():
        raise StateArchiveError(f"unsafe archive member (symlink/hardlink not allowed): {name!r}")

    if not (member.isfile() or member.isdir()):
        raise StateArchiveError(f"unsafe archive member (not a regular file/dir): {name!r}")


def unpack_state(in_path: Path, state_dir: Path, passphrase: str) -> UnpackResult:
    """Decrypt, verify and extract a state archive into ``state_dir``.

    ``state_dir`` is replaced atomically: every member is validated and
    hash-checked against the embedded manifest *before* anything is written
    to disk, then extracted into a fresh temp directory, then swapped into
    place with two ``os.replace`` calls (old state_dir -> backup, temp ->
    state_dir). A wrong passphrase, tampered ciphertext, unsafe member, or
    manifest/hash mismatch raises before any of that -- the existing
    ``state_dir`` (if any) is left untouched.

    Raises:
        turboedge.state.crypto.StateCryptoError: wrong passphrase, or the
            archive's ciphertext is corrupted/tampered with.
        StateArchiveError: the archive is structurally invalid (missing
            manifest, unsafe member, manifest/hash/size mismatch).
    """
    in_path = Path(in_path)
    state_dir = Path(state_dir)

    encrypted = in_path.read_bytes()
    tar_bytes = decrypt_bytes(encrypted, passphrase)  # StateCryptoError propagates as-is

    manifest: dict[str, Any] | None = None
    entries: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            _validate_member(member)
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                raise StateArchiveError(f"could not read archive member: {member.name!r}")
            data = extracted.read()
            if member.name == MANIFEST_NAME:
                manifest = json.loads(data.decode("utf-8"))
            else:
                entries[member.name] = data

    if manifest is None:
        raise StateArchiveError(f"archive is missing {MANIFEST_NAME}; not a valid state archive")

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise StateArchiveError(f"{MANIFEST_NAME} is missing/malformed 'files' list")

    manifest_paths: set[str] = set()
    for entry in manifest_files:
        path = entry["path"]
        expected_hash = entry["sha256"]
        expected_size = entry["size"]
        manifest_paths.add(path)

        entry_data = entries.get(path)
        if entry_data is None:
            raise StateArchiveError(f"manifest lists {path!r} but the archive does not contain it")
        if len(entry_data) != expected_size:
            raise StateArchiveError(
                f"size mismatch for {path!r}: manifest says {expected_size} bytes, "
                f"archive has {len(entry_data)}"
            )
        actual_hash = sha256(entry_data).hexdigest()
        if actual_hash != expected_hash:
            raise StateArchiveError(
                f"checksum mismatch for {path!r}: manifest sha256={expected_hash}, "
                f"actual={actual_hash} -- archive is corrupted or was tampered with"
            )

    extra_paths = set(entries) - manifest_paths
    if extra_paths:
        raise StateArchiveError(
            f"archive contains file(s) not listed in {MANIFEST_NAME}: {sorted(extra_paths)}"
        )

    tmp_dir = state_dir.parent / f".{state_dir.name}.unpack-tmp-{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        for path, data in entries.items():
            dest = tmp_dir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

        state_dir.parent.mkdir(parents=True, exist_ok=True)
        if state_dir.exists():
            backup_dir = state_dir.parent / f".{state_dir.name}.replaced-{uuid4().hex}"
            os.replace(state_dir, backup_dir)
            os.replace(tmp_dir, state_dir)
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            os.replace(tmp_dir, state_dir)
    finally:
        # A no-op once the rename above succeeded (tmp_dir no longer exists
        # at this path); cleans up on any failure before that point.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return UnpackResult(
        state_dir=state_dir,
        file_count=len(entries),
        total_bytes=sum(len(d) for d in entries.values()),
    )


def unpack_state_subset(
    in_path: Path,
    state_dir: Path,
    passphrase: str,
    *,
    prefixes: tuple[str, ...],
) -> UnpackResult:
    """Decrypt, verify and ADDITIVELY extract only the archive members whose
    manifest path starts with one of ``prefixes`` (e.g. ``(SNAPSHOTS_PREFIX,
    )``) into ``state_dir``, leaving every other file already there (and
    every non-matching manifest entry in ``in_path``) untouched.

    Unlike :func:`unpack_state`, this is a partial merge, not a full,
    atomic replace of ``state_dir`` -- there is no all-or-nothing swap of
    the whole directory, since the whole point is to layer one subtree of a
    (typically older, periodic) full archive onto a state_dir a more recent
    lean :func:`unpack_state` call already populated, without rolling back
    anything that unpack restored. ``state_dir`` need not already exist (it
    is created if missing, e.g. for a from-scratch restore of just the
    Parquet archive). Every extracted file is still individually verified
    against the manifest's sha256/size before being written, and each
    write is atomic (temp file in the same directory, then ``os.replace``)
    -- a wrong key, tampered ciphertext, unsafe member, or a hash/size
    mismatch on any MATCHING entry raises before anything is written; a
    matching entry always wins over a same-named file already on disk.

    Non-matching manifest entries are validated for archive-safety
    (:func:`_validate_member`, same as :func:`unpack_state`) but never
    read, verified against sha256/size, or written -- this is a deliberate
    partial restore, not a general-purpose alternative to
    :func:`unpack_state`.

    Raises:
        turboedge.state.crypto.StateCryptoError: wrong passphrase, or the
            archive's ciphertext is corrupted/tampered with.
        StateArchiveError: the archive is structurally invalid, or a
            matching entry fails its manifest hash/size check.
    """
    in_path = Path(in_path)
    state_dir = Path(state_dir)

    encrypted = in_path.read_bytes()
    tar_bytes = decrypt_bytes(encrypted, passphrase)  # StateCryptoError propagates as-is

    manifest: dict[str, Any] | None = None
    matching_entries: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            _validate_member(member)
            if not member.isfile():
                continue
            is_manifest = member.name == MANIFEST_NAME
            if not is_manifest and not any(member.name.startswith(p) for p in prefixes):
                continue  # not the manifest, and outside every requested prefix
            extracted = tar.extractfile(member)
            if extracted is None:
                raise StateArchiveError(f"could not read archive member: {member.name!r}")
            data = extracted.read()
            if is_manifest:
                manifest = json.loads(data.decode("utf-8"))
            else:
                matching_entries[member.name] = data

    if manifest is None:
        raise StateArchiveError(f"archive is missing {MANIFEST_NAME}; not a valid state archive")

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise StateArchiveError(f"{MANIFEST_NAME} is missing/malformed 'files' list")

    manifest_by_path = {entry["path"]: entry for entry in manifest_files}

    verified: dict[str, bytes] = {}
    for path, data in matching_entries.items():
        entry = manifest_by_path.get(path)
        if entry is None:
            raise StateArchiveError(
                f"archive contains {path!r} but it is not listed in {MANIFEST_NAME}"
            )
        expected_size = entry["size"]
        expected_hash = entry["sha256"]
        if len(data) != expected_size:
            raise StateArchiveError(
                f"size mismatch for {path!r}: manifest says {expected_size} bytes, "
                f"archive has {len(data)}"
            )
        actual_hash = sha256(data).hexdigest()
        if actual_hash != expected_hash:
            raise StateArchiveError(
                f"checksum mismatch for {path!r}: manifest sha256={expected_hash}, "
                f"actual={actual_hash} -- archive is corrupted or was tampered with"
            )
        verified[path] = data

    state_dir.mkdir(parents=True, exist_ok=True)
    for path, data in verified.items():
        dest = state_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_name(f".{dest.name}.tmp-{uuid4().hex}")
        try:
            tmp_dest.write_bytes(data)
            os.replace(tmp_dest, dest)
        finally:
            tmp_dest.unlink(missing_ok=True)

    return UnpackResult(
        state_dir=state_dir,
        file_count=len(verified),
        total_bytes=sum(len(d) for d in verified.values()),
    )


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "FULL_INCLUDE_DIRS",
    "LEAN_INCLUDE_DIRS",
    "MANIFEST_NAME",
    "SNAPSHOTS_PREFIX",
    "ArchiveConfig",
    "ManifestFile",
    "PackResult",
    "StateArchiveError",
    "UnpackResult",
    "pack_state",
    "unpack_state",
    "unpack_state_subset",
]
