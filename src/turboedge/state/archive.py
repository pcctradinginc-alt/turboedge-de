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


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "ArchiveConfig",
    "ManifestFile",
    "PackResult",
    "StateArchiveError",
    "UnpackResult",
    "pack_state",
    "unpack_state",
]
