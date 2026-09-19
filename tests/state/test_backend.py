from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from turboedge.state.backend import (
    LocalStateBackend,
    StateBackendError,
    backend_from_env,
)


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_put_get_roundtrip_exact_bytes(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    src = _write(tmp_path / "archive.enc", b"encrypted-state-archive-bytes \x00\x01\x02")

    version_id = backend.put(src, "lean")

    out = tmp_path / "restored.enc"
    fetched_version = backend.get("lean", out, version=None)

    assert fetched_version == version_id
    assert out.read_bytes() == src.read_bytes()


def test_put_never_overwrites_previous_version(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    src1 = _write(tmp_path / "v1.enc", b"version one payload")
    src2 = _write(tmp_path / "v2.enc", b"version two payload, different bytes")

    v1 = backend.put(src1, "lean")
    v2 = backend.put(src2, "lean")

    assert v1 != v2
    versions = backend.list_versions("lean")
    assert [v.version_id for v in versions] == [v1, v2]

    # Both old and new content are still separately retrievable.
    out1 = tmp_path / "out1.enc"
    out2 = tmp_path / "out2.enc"
    backend.get("lean", out1, version=v1)
    backend.get("lean", out2, version=v2)
    assert out1.read_bytes() == b"version one payload"
    assert out2.read_bytes() == b"version two payload, different bytes"


def test_get_without_version_picks_the_latest(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    backend.put(_write(tmp_path / "v1.enc", b"old"), "lean")
    v2 = backend.put(_write(tmp_path / "v2.enc", b"new"), "lean")

    out = tmp_path / "out.enc"
    fetched = backend.get("lean", out)

    assert fetched == v2
    assert out.read_bytes() == b"new"


def test_list_versions_chronological(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    ids = [
        backend.put(_write(tmp_path / f"v{i}.enc", f"payload {i}".encode()), "lean")
        for i in range(5)
    ]

    versions = backend.list_versions("lean")

    assert [v.version_id for v in versions] == ids
    # created_at must also be non-decreasing (list_versions sorts by
    # version_id, which is lexicographically == chronologically ordered).
    timestamps = [v.created_at for v in versions]
    assert timestamps == sorted(timestamps)


def test_list_versions_empty_key_returns_empty_list_not_error(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    assert backend.list_versions("never-written") == []


def test_get_tampered_checksum_raises_loudly(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    version_id = backend.put(_write(tmp_path / "src.enc", b"authentic bytes"), "lean")

    # Corrupt the stored version bytes directly, bypassing put()'s own
    # checksum verification -- simulates bit-rot / an out-of-band edit.
    key_dir = tmp_path / "backend-root" / "lean"
    (key_dir / f"{version_id}.bin").write_bytes(b"TAMPERED bytes, different length!!")

    with pytest.raises(StateBackendError, match="checksum mismatch"):
        backend.get("lean", tmp_path / "out.enc", version=version_id)

    # No partial/corrupted file left behind for the caller to mistakenly use.
    assert not (tmp_path / "out.enc").exists()


def test_get_missing_version_raises(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    backend.put(_write(tmp_path / "src.enc", b"data"), "lean")

    with pytest.raises(StateBackendError, match="not found"):
        backend.get("lean", tmp_path / "out.enc", version="does-not-exist")


def test_get_no_versions_at_all_raises_not_empty_state(tmp_path: Path) -> None:
    """An unreachable/empty backend must raise, never silently hand back an
    empty file that could be mistaken for legitimate (if empty) state."""
    backend = LocalStateBackend(tmp_path / "backend-root")

    with pytest.raises(StateBackendError, match="no versions found"):
        backend.get("never-written", tmp_path / "out.enc")

    assert not (tmp_path / "out.enc").exists()


def test_put_missing_source_file_raises(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    with pytest.raises(StateBackendError, match="no such file"):
        backend.put(tmp_path / "does-not-exist.enc", "lean")


def test_interrupted_put_leaves_no_valid_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a copy that gets interrupted partway through (crash, disk
    full, ...): the temp file ends up truncated/wrong, so put()'s own
    re-verification must catch it and the failed attempt must not become a
    version list_versions/get can see."""
    backend = LocalStateBackend(tmp_path / "backend-root")
    src = _write(tmp_path / "src.enc", b"a" * 10_000)

    def _truncating_copyfile(source: str, dest: str) -> str:
        # Write only the first half of the bytes -- an interrupted copy.
        data = Path(source).read_bytes()
        Path(dest).write_bytes(data[: len(data) // 2])
        return dest

    monkeypatch.setattr(shutil, "copyfile", _truncating_copyfile)

    with pytest.raises(StateBackendError, match="checksum mismatch"):
        backend.put(src, "lean")

    key_dir = tmp_path / "backend-root" / "lean"
    # Nothing valid (or even a leftover .bin/.tmp) survives the failure.
    assert not key_dir.exists() or list(key_dir.glob("*.bin")) == []
    assert backend.list_versions("lean") == []


def test_health_reports_writable_root(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    result = backend.health()
    assert result.reachable is True


def test_health_reports_unwritable_root(tmp_path: Path) -> None:
    # A regular file where a directory is expected: mkdir(parents=True)
    # underneath it must fail with an OSError, exercised without needing
    # filesystem permission tricks (which are unreliable across platforms/CI).
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    backend = LocalStateBackend(blocker / "nested-root")

    result = backend.health()

    assert result.reachable is False
    assert "not writable" in result.reason


def test_invalid_key_rejected(tmp_path: Path) -> None:
    backend = LocalStateBackend(tmp_path / "backend-root")
    src = _write(tmp_path / "src.enc", b"data")
    with pytest.raises(StateBackendError, match="invalid backend key"):
        backend.put(src, "../escape")


class TestBackendFromEnv:
    def test_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TURBOEDGE_STATE_BACKEND", raising=False)
        assert backend_from_env() is None

    @pytest.mark.parametrize("value", ["", "none", "off", "None", "OFF"])
    def test_explicit_none_like_values_return_none(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("TURBOEDGE_STATE_BACKEND", value)
        assert backend_from_env() is None

    def test_unknown_kind_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TURBOEDGE_STATE_BACKEND", "dropbox")
        with pytest.raises(StateBackendError, match="unknown"):
            backend_from_env()

    def test_local_without_root_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TURBOEDGE_STATE_BACKEND", "local")
        monkeypatch.delenv("TURBOEDGE_STATE_LOCAL_ROOT", raising=False)
        with pytest.raises(StateBackendError, match="TURBOEDGE_STATE_LOCAL_ROOT"):
            backend_from_env()

    def test_local_with_root_returns_local_backend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TURBOEDGE_STATE_BACKEND", "local")
        monkeypatch.setenv("TURBOEDGE_STATE_LOCAL_ROOT", str(tmp_path / "durable"))
        backend = backend_from_env()
        assert isinstance(backend, LocalStateBackend)

    def test_s3_missing_credentials_raises_without_touching_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kind='s3' with no TURBOEDGE_STATE_S3_* vars must fail fast in
        config validation -- long before any boto3 client/network call is
        attempted, and regardless of whether boto3 is even installed."""
        monkeypatch.setenv("TURBOEDGE_STATE_BACKEND", "s3")
        for var in (
            "TURBOEDGE_STATE_S3_BUCKET",
            "TURBOEDGE_STATE_S3_ACCESS_KEY_ID",
            "TURBOEDGE_STATE_S3_SECRET_ACCESS_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(StateBackendError, match="missing required environment"):
            backend_from_env()
