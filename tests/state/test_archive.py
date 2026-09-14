from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest

from turboedge.state.archive import (
    FULL_INCLUDE_DIRS,
    LEAN_INCLUDE_DIRS,
    MANIFEST_NAME,
    SNAPSHOTS_PREFIX,
    ArchiveConfig,
    StateArchiveError,
    pack_state,
    pack_state_files,
    unpack_state,
    unpack_state_subset,
)
from turboedge.state.crypto import (
    MIN_PASSPHRASE_LEN,
    StateCryptoError,
    decrypt_bytes,
    encrypt_bytes,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ProductSnapshot

_PASSPHRASE = "correct-horse-battery-staple-24"


def _make_state_dir(tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]) -> Path:
    """A realistic state_dir: a real DuckDB (with one product_snapshots row),
    plus files under snapshots/, registry/, ledger/, trials/, and a couple of
    things that must be EXCLUDED (imports/, a WAL sidecar)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    with Store(state_dir / "turboedge.duckdb") as store:
        store.init_schema()
        store.append_product_snapshots([make_product_snapshot()])

    snapshot_dir = state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "run1.parquet").write_bytes(b"fake-parquet-bytes")
    (state_dir / "registry").mkdir()
    (state_dir / "registry" / "models.json").write_text('{"champion": "tsmom"}')
    (state_dir / "ledger").mkdir()
    (state_dir / "ledger" / "entries.json").write_text("[]")
    (state_dir / "trials").mkdir()
    (state_dir / "trials" / "trial1.json").write_text("{}")

    # Must be excluded:
    (state_dir / "imports" / "products").mkdir(parents=True)
    (state_dir / "imports" / "products" / "manual.csv").write_text("isin,bid,ask\n")
    (state_dir / "turboedge.duckdb.wal").write_bytes(b"stale-wal-bytes")

    return state_dir


def test_pack_unpack_roundtrip(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    archive_path = tmp_path / "out" / "state.tar.enc"

    result = pack_state(state_dir, archive_path, _PASSPHRASE)
    assert archive_path.is_file()
    assert result.file_count >= 4  # db + one file in each of the 4 subdirs

    restore_dir = tmp_path / "restored"
    unpack_result = unpack_state(archive_path, restore_dir, _PASSPHRASE)
    assert unpack_result.file_count == result.file_count

    # DuckDB content survived.
    with Store(restore_dir / "turboedge.duckdb") as store:
        store.init_schema()
        assert store.table_counts()["product_snapshots"] == 1

    # Other included files survived.
    assert (restore_dir / "registry" / "models.json").read_text() == '{"champion": "tsmom"}'
    assert (restore_dir / "ledger" / "entries.json").read_text() == "[]"
    assert (restore_dir / "trials" / "trial1.json").read_text() == "{}"
    assert (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run1.parquet"
    ).read_bytes() == b"fake-parquet-bytes"

    # Excluded content did NOT survive.
    assert not (restore_dir / "imports").exists()
    assert not (restore_dir / "turboedge.duckdb.wal").exists()


def test_pack_excludes_imports_and_wal(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    archive_path = tmp_path / "state.tar.enc"
    pack_state(state_dir, archive_path, _PASSPHRASE)

    tar_bytes = _decrypt_to_tar_bytes(archive_path)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        names = tar.getnames()
    assert not any("imports" in n for n in names)
    assert not any(n.endswith(".wal") for n in names)
    assert "turboedge.duckdb" in names
    assert MANIFEST_NAME in names


def test_manifest_contains_sha256_and_metadata(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    archive_path = tmp_path / "state.tar.enc"
    pack_state(state_dir, archive_path, _PASSPHRASE)

    tar_bytes = _decrypt_to_tar_bytes(archive_path)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        manifest_member = tar.extractfile(MANIFEST_NAME)
        assert manifest_member is not None
        manifest = json.loads(manifest_member.read())

    assert manifest["schema_version"] == 1
    assert "created_at" in manifest
    assert isinstance(manifest["files"], list)
    assert len(manifest["files"]) >= 4
    for entry in manifest["files"]:
        assert set(entry) == {"path", "sha256", "size"}
        assert len(entry["sha256"]) == 64  # hex sha256


def test_unpack_wrong_key_raises_and_leaves_state_dir_untouched(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    archive_path = tmp_path / "state.tar.enc"
    pack_state(state_dir, archive_path, _PASSPHRASE)

    restore_dir = tmp_path / "restored"
    restore_dir.mkdir()
    (restore_dir / "sentinel.txt").write_text("pre-existing content")

    with pytest.raises(StateCryptoError):
        unpack_state(archive_path, restore_dir, "b" * MIN_PASSPHRASE_LEN)

    # Existing directory must be untouched by a failed unpack.
    assert (restore_dir / "sentinel.txt").read_text() == "pre-existing content"


def test_unpack_manifest_hash_mismatch_rejected(tmp_path: Path) -> None:
    tampered_path = tmp_path / "tampered.tar.enc"
    _build_tampered_archive(
        tampered_path,
        files={"registry/models.json": b'{"champion": "tsmom"}'},
        manifest_override={
            "files": [
                {
                    "path": "registry/models.json",
                    "sha256": "0" * 64,  # wrong hash
                    "size": len(b'{"champion": "tsmom"}'),
                }
            ]
        },
    )

    with pytest.raises(StateArchiveError, match="checksum mismatch"):
        unpack_state(tampered_path, tmp_path / "restored2", _PASSPHRASE)


def test_unpack_path_traversal_rejected(tmp_path: Path) -> None:
    evil_path = tmp_path / "evil.tar.enc"
    payload = b"pwned"
    _build_tampered_archive(
        evil_path,
        files={"../../etc/evil": payload},
        manifest_override={
            "files": [
                {"path": "../../etc/evil", "sha256": _sha256_hex(payload), "size": len(payload)}
            ]
        },
    )

    with pytest.raises(StateArchiveError):
        unpack_state(evil_path, tmp_path / "restored3", _PASSPHRASE)


def test_unpack_absolute_path_rejected(tmp_path: Path) -> None:
    evil_path = tmp_path / "evil_abs.tar.enc"
    payload = b"pwned"
    _build_tampered_archive(
        evil_path,
        files={"/etc/evil": payload},
        manifest_override={
            "files": [{"path": "/etc/evil", "sha256": _sha256_hex(payload), "size": len(payload)}]
        },
    )

    with pytest.raises(StateArchiveError):
        unpack_state(evil_path, tmp_path / "restored4", _PASSPHRASE)


def test_unpack_symlink_member_rejected(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        link_info = tarfile.TarInfo(name="registry/evil_link")
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = "/etc/passwd"
        tar.addfile(link_info)

        manifest_bytes = json.dumps(
            {"schema_version": 1, "files": [], "created_at": "x", "git_commit": None}
        ).encode("utf-8")
        manifest_info = tarfile.TarInfo(name=MANIFEST_NAME)
        manifest_info.size = len(manifest_bytes)
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

    evil_path = tmp_path / "evil_symlink.tar.enc"
    evil_path.write_bytes(encrypt_bytes(buf.getvalue(), _PASSPHRASE))

    with pytest.raises(StateArchiveError):
        unpack_state(evil_path, tmp_path / "restored5", _PASSPHRASE)


def test_unpack_missing_manifest_rejected(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="turboedge.duckdb")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))

    evil_path = tmp_path / "no_manifest.tar.enc"
    evil_path.write_bytes(encrypt_bytes(buf.getvalue(), _PASSPHRASE))

    with pytest.raises(StateArchiveError, match=MANIFEST_NAME):
        unpack_state(evil_path, tmp_path / "restored6", _PASSPHRASE)


# -- lean vs. full archive shapes (state pack --include-snapshots) --------


def test_lean_config_excludes_snapshots_dir(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    archive_path = tmp_path / "lean.tar.enc"
    result = pack_state(
        state_dir, archive_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=LEAN_INCLUDE_DIRS)
    )

    tar_bytes = _decrypt_to_tar_bytes(archive_path)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        names = tar.getnames()
    assert not any(n.startswith(SNAPSHOTS_PREFIX) for n in names)
    assert "registry/models.json" in names
    assert "ledger/entries.json" in names
    assert "trials/trial1.json" in names
    assert "turboedge.duckdb" in names
    assert result.file_count == 4  # db + registry + ledger + trials, no snapshots

    restore_dir = tmp_path / "restored_lean"
    unpack_result = unpack_state(archive_path, restore_dir, _PASSPHRASE)
    assert unpack_result.file_count == 4
    assert not (restore_dir / "snapshots").exists()
    # Everything else still round-trips.
    assert (restore_dir / "registry" / "models.json").read_text() == '{"champion": "tsmom"}'


def test_full_config_includes_snapshots_dir(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    archive_path = tmp_path / "full.tar.enc"
    result = pack_state(
        state_dir, archive_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=FULL_INCLUDE_DIRS)
    )

    tar_bytes = _decrypt_to_tar_bytes(archive_path)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        names = tar.getnames()
    assert any(n.startswith(SNAPSHOTS_PREFIX) for n in names)
    assert result.file_count == 5  # db + snapshots + registry + ledger + trials


# -- unpack_state_subset (additive merge, e.g. weekly restoring snapshots/) -


def test_unpack_state_subset_merges_onto_existing_lean_restore(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    lean_path = tmp_path / "lean.tar.enc"
    full_path = tmp_path / "full.tar.enc"
    pack_state(
        state_dir, lean_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=LEAN_INCLUDE_DIRS)
    )
    pack_state(
        state_dir, full_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=FULL_INCLUDE_DIRS)
    )

    restore_dir = tmp_path / "restored"
    unpack_state(lean_path, restore_dir, _PASSPHRASE)
    assert not (restore_dir / "snapshots").exists()
    # A lean restore's registry content is the baseline the subset merge
    # below must NOT disturb.
    assert (restore_dir / "registry" / "models.json").read_text() == '{"champion": "tsmom"}'

    subset_result = unpack_state_subset(
        full_path, restore_dir, _PASSPHRASE, prefixes=(SNAPSHOTS_PREFIX,)
    )

    assert subset_result.file_count == 1
    assert (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run1.parquet"
    ).read_bytes() == b"fake-parquet-bytes"
    # Untouched: db, registry, ledger, trials survive exactly as the lean
    # restore left them.
    assert (restore_dir / "registry" / "models.json").read_text() == '{"champion": "tsmom"}'
    assert (restore_dir / "ledger" / "entries.json").read_text() == "[]"
    with Store(restore_dir / "turboedge.duckdb") as store:
        store.init_schema()
        assert store.table_counts()["product_snapshots"] == 1


def test_unpack_state_subset_does_not_overwrite_newer_state_dir_files(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    """A stale full archive's non-matching content (e.g. an older db) must
    never leak into state_dir via a subset restore -- only prefix-matching
    entries are ever written."""
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    full_path = tmp_path / "full.tar.enc"
    pack_state(
        state_dir, full_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=FULL_INCLUDE_DIRS)
    )

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "turboedge.duckdb").write_bytes(b"sentinel-current-db")

    unpack_state_subset(full_path, target_dir, _PASSPHRASE, prefixes=(SNAPSHOTS_PREFIX,))

    assert (target_dir / "turboedge.duckdb").read_bytes() == b"sentinel-current-db"
    assert (
        target_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run1.parquet"
    ).read_bytes() == b"fake-parquet-bytes"


def test_unpack_state_subset_creates_state_dir_if_missing(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    full_path = tmp_path / "full.tar.enc"
    pack_state(
        state_dir, full_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=FULL_INCLUDE_DIRS)
    )

    fresh_dir = tmp_path / "brand_new"
    assert not fresh_dir.exists()

    result = unpack_state_subset(full_path, fresh_dir, _PASSPHRASE, prefixes=(SNAPSHOTS_PREFIX,))

    assert result.file_count == 1
    assert fresh_dir.is_dir()


def test_unpack_state_subset_no_matching_prefix_is_a_noop(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    lean_path = tmp_path / "lean.tar.enc"
    pack_state(
        state_dir, lean_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=LEAN_INCLUDE_DIRS)
    )

    target_dir = tmp_path / "target"
    target_dir.mkdir()

    result = unpack_state_subset(lean_path, target_dir, _PASSPHRASE, prefixes=(SNAPSHOTS_PREFIX,))

    assert result.file_count == 0
    assert not (target_dir / "snapshots").exists()


def test_unpack_state_subset_wrong_key_raises(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state_dir = _make_state_dir(tmp_path, make_product_snapshot)
    full_path = tmp_path / "full.tar.enc"
    pack_state(
        state_dir, full_path, _PASSPHRASE, config=ArchiveConfig(include_dirs=FULL_INCLUDE_DIRS)
    )

    target_dir = tmp_path / "target"
    target_dir.mkdir()

    with pytest.raises(StateCryptoError):
        unpack_state_subset(
            full_path, target_dir, "b" * MIN_PASSPHRASE_LEN, prefixes=(SNAPSHOTS_PREFIX,)
        )


def test_unpack_state_subset_tampered_hash_rejected(tmp_path: Path) -> None:
    payload = b"tampered-parquet-bytes"
    tampered_path = tmp_path / "tampered.tar.enc"
    _build_tampered_archive(
        tampered_path,
        files={"snapshots/product_snapshots/date=2026-09-10/run1.parquet": payload},
        manifest_override={
            "files": [
                {
                    "path": "snapshots/product_snapshots/date=2026-09-10/run1.parquet",
                    "sha256": "0" * 64,
                    "size": len(payload),
                }
            ]
        },
    )

    with pytest.raises(StateArchiveError, match="checksum mismatch"):
        unpack_state_subset(
            tampered_path, tmp_path / "target", _PASSPHRASE, prefixes=(SNAPSHOTS_PREFIX,)
        )


def test_unpack_state_subset_path_traversal_rejected(tmp_path: Path) -> None:
    evil_path = tmp_path / "evil.tar.enc"
    payload = b"pwned"
    _build_tampered_archive(
        evil_path,
        files={"../../etc/evil": payload},
        manifest_override={
            "files": [
                {"path": "../../etc/evil", "sha256": _sha256_hex(payload), "size": len(payload)}
            ]
        },
    )

    with pytest.raises(StateArchiveError):
        unpack_state_subset(
            evil_path, tmp_path / "target", _PASSPHRASE, prefixes=(SNAPSHOTS_PREFIX,)
        )


def test_pack_on_empty_state_dir_produces_minimal_archive(tmp_path: Path) -> None:
    """A `state pack` run before any pipeline command has ever touched the
    state dir must not blow up. With nothing at all on disk yet, it must not
    fabricate an empty DuckDB file either -- zero files packed (manifest
    only), and unpacking that archive back out must not error."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    archive_path = tmp_path / "state.tar.enc"

    result = pack_state(state_dir, archive_path, _PASSPHRASE)
    assert archive_path.is_file()
    assert result.file_count == 0

    restore_dir = tmp_path / "restored_empty"
    unpack_result = unpack_state(archive_path, restore_dir, _PASSPHRASE)
    assert unpack_result.file_count == 0
    assert restore_dir.is_dir()


# -- pack_state_files (incremental per-scan snapshot artifact) ----------


def test_pack_state_files_packs_only_the_given_files(tmp_path: Path) -> None:
    """Two files already exist under state/snapshots/ (mimicking older
    history plus this run's new output) -- pack_state_files with only the
    "new" one selected must not pull the other in, proving the incremental
    artifact never silently re-includes accumulated history."""
    state_dir = tmp_path / "state"
    old_dir = state_dir / "snapshots" / "product_snapshots" / "date=2026-09-09"
    old_dir.mkdir(parents=True)
    old_file = old_dir / "run-old.parquet"
    old_file.write_bytes(b"old-bytes")

    new_dir = state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    new_dir.mkdir(parents=True)
    new_file = new_dir / "run-new.parquet"
    new_file.write_bytes(b"new-bytes")

    out_path = tmp_path / "out" / "snapshot-batch.tar.enc"
    result = pack_state_files(state_dir, out_path, _PASSPHRASE, [new_file])

    assert result.file_count == 1
    assert result.total_bytes == len(b"new-bytes")
    assert out_path.is_file()

    restore_dir = tmp_path / "restored"
    unpack_result = unpack_state(out_path, restore_dir, _PASSPHRASE)
    assert unpack_result.file_count == 1
    restored_path = (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run-new.parquet"
    )
    assert restored_path.read_bytes() == b"new-bytes"
    # The old file must NOT have been carried into this archive.
    assert not (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-09" / "run-old.parquet"
    ).exists()


def test_pack_state_files_multiple_files_deterministic_order(tmp_path: Path) -> None:
    """Files passed out of order still end up in a deterministic (sorted)
    manifest order, regardless of the caller's argument order -- exact tar
    bytes are NOT compared here since gzip/tar embed a wall-clock mtime per
    call, which legitimately differs between two separate pack_state_files
    invocations a few milliseconds apart."""
    state_dir = tmp_path / "state"
    snap_dir = state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    snap_dir.mkdir(parents=True)
    file_b = snap_dir / "run-b.parquet"
    file_a = snap_dir / "run-a.parquet"
    file_b.write_bytes(b"bbb")
    file_a.write_bytes(b"aaa")

    out_path = tmp_path / "out.tar.enc"
    # Passed out of order -- the manifest/tar order must still be deterministic.
    result1 = pack_state_files(state_dir, out_path, _PASSPHRASE, [file_b, file_a])

    out_path2 = tmp_path / "out2.tar.enc"
    result2 = pack_state_files(state_dir, out_path2, _PASSPHRASE, [file_a, file_b])

    assert result1.file_count == result2.file_count == 2

    def _manifest_paths(archive_path: Path) -> list[str]:
        tar_bytes = decrypt_bytes(archive_path.read_bytes(), _PASSPHRASE)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            member = tar.extractfile(MANIFEST_NAME)
            assert member is not None
            manifest = json.loads(member.read())
        return [f["path"] for f in manifest["files"]]

    order1 = _manifest_paths(out_path)
    order2 = _manifest_paths(out_path2)
    assert order1 == order2 == sorted(order1)


def test_pack_state_files_empty_list_produces_empty_archive(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    out_path = tmp_path / "empty.tar.enc"

    result = pack_state_files(state_dir, out_path, _PASSPHRASE, [])

    assert result.file_count == 0
    assert out_path.is_file()
    restore_dir = tmp_path / "restored-empty"
    unpack_result = unpack_state(out_path, restore_dir, _PASSPHRASE)
    assert unpack_result.file_count == 0


def test_pack_state_files_rejects_path_outside_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    outside = tmp_path / "elsewhere" / "not-in-state.parquet"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x")

    with pytest.raises(ValueError, match="not inside state_dir"):
        pack_state_files(state_dir, tmp_path / "out.tar.enc", _PASSPHRASE, [outside])


def test_pack_state_files_deduplicates_repeated_paths(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    snap_dir = state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    snap_dir.mkdir(parents=True)
    f = snap_dir / "run-1.parquet"
    f.write_bytes(b"once")

    out_path = tmp_path / "out.tar.enc"
    result = pack_state_files(state_dir, out_path, _PASSPHRASE, [f, f])

    assert result.file_count == 1


# -- helpers ------------------------------------------------------------


def _decrypt_to_tar_bytes(archive_path: Path) -> bytes:
    return decrypt_bytes(archive_path.read_bytes(), _PASSPHRASE)


def _sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def _build_tampered_archive(
    out_path: Path, *, files: dict[str, bytes], manifest_override: dict[str, object]
) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        manifest = {
            "created_at": "2026-09-11T00:00:00+00:00",
            "schema_version": 1,
            "git_commit": None,
            "files": [],
        }
        manifest.update(manifest_override)
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        manifest_info = tarfile.TarInfo(name=MANIFEST_NAME)
        manifest_info.size = len(manifest_bytes)
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

    out_path.write_bytes(encrypt_bytes(buf.getvalue(), _PASSPHRASE))
