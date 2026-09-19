"""Tests for the S3-compatible remote backend, entirely against an in-memory
fake S3 client -- `S3CompatibleStateBackend._boto3_client` construction is
monkeypatched out, so no real `boto3.session.Session()` is ever created and
no network call is ever attempted, satisfying the requirement that the
remote backend is never actually contacted in tests."""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from turboedge.state import backend_s3
from turboedge.state.backend import StateBackendError
from turboedge.state.backend_s3 import S3CompatibleStateBackend, S3Config


class _FakePaginator:
    def __init__(self, client: _FakeS3Client) -> None:
        self._client = client

    def paginate(self, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:
        yield self._client.list_objects_v2(Bucket=Bucket, Prefix=Prefix, MaxKeys=10_000)


class _FakeS3Client:
    """Enough of the boto3 S3 client surface for S3CompatibleStateBackend,
    backed by an in-memory dict -- never touches the network."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.fail_put = False
        self.corrupt_next_head = False

    def put_object(
        self,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentMD5: str | None = None,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.fail_put:
            raise RuntimeError("simulated network failure during put_object")
        expected_md5 = base64.b64encode(hashlib.md5(Body, usedforsecurity=False).digest()).decode()
        if ContentMD5 != expected_md5:
            raise RuntimeError("simulated S3 400 BadDigest: Content-MD5 does not match body")
        self.objects[Key] = {
            "body": bytes(Body),
            "metadata": dict(Metadata or {}),
            "last_modified": datetime.now(UTC),
        }
        return {}

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise RuntimeError(f"simulated 404 NoSuchKey: {Key}")
        obj = self.objects[Key]
        if self.corrupt_next_head:
            self.corrupt_next_head = False
            return {"Metadata": {"turboedge-sha256": "0" * 64}, "ContentLength": len(obj["body"])}
        return {"Metadata": obj["metadata"], "ContentLength": len(obj["body"])}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise RuntimeError(f"simulated 404 NoSuchKey: {Key}")
        obj = self.objects[Key]
        return {"Body": io.BytesIO(obj["body"]), "Metadata": obj["metadata"]}

    def list_objects_v2(self, Bucket: str, Prefix: str = "", MaxKeys: int = 1000) -> dict[str, Any]:
        keys = sorted(k for k in self.objects if k.startswith(Prefix))[:MaxKeys]
        contents = [
            {
                "Key": k,
                "Size": len(self.objects[k]["body"]),
                "LastModified": self.objects[k]["last_modified"],
            }
            for k in keys
        ]
        return {"Contents": contents}

    def get_paginator(self, operation_name: str) -> _FakePaginator:
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeS3Client:
    client = _FakeS3Client()
    monkeypatch.setattr(backend_s3, "_boto3_client", lambda config: client)
    return client


def _config() -> S3Config:
    return S3Config(
        bucket="turboedge-test-bucket",
        access_key_id="AKIA-fake",
        secret_access_key="fake-secret",
        endpoint_url="https://fake.example.invalid",
        region="auto",
        prefix="turboedge-state",
    )


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_put_get_roundtrip_exact_bytes(fake_client: _FakeS3Client, tmp_path: Path) -> None:
    backend = S3CompatibleStateBackend(config=_config())
    src = _write(tmp_path / "archive.enc", b"encrypted payload \x00\x01\x02" * 100)

    version_id = backend.put(src, "lean")
    out = tmp_path / "restored.enc"
    fetched_version = backend.get("lean", out, version=None)

    assert fetched_version == version_id
    assert out.read_bytes() == src.read_bytes()


def test_put_never_overwrites_creates_new_object_each_time(
    fake_client: _FakeS3Client, tmp_path: Path
) -> None:
    backend = S3CompatibleStateBackend(config=_config())
    v1 = backend.put(_write(tmp_path / "v1.enc", b"first"), "lean")
    v2 = backend.put(_write(tmp_path / "v2.enc", b"second"), "lean")

    assert v1 != v2
    assert len(fake_client.objects) == 2  # two distinct object keys, both still present


def test_list_versions_chronological(fake_client: _FakeS3Client, tmp_path: Path) -> None:
    backend = S3CompatibleStateBackend(config=_config())
    ids = [
        backend.put(_write(tmp_path / f"v{i}.enc", f"payload-{i}".encode()), "lean")
        for i in range(4)
    ]

    versions = backend.list_versions("lean")

    assert [v.version_id for v in versions] == ids
    assert [v.size for v in versions] == [len(f"payload-{i}".encode()) for i in range(4)]
    for v in versions:
        assert len(v.sha256) == 64


def test_list_versions_empty_key_returns_empty_list(fake_client: _FakeS3Client) -> None:
    backend = S3CompatibleStateBackend(config=_config())
    assert backend.list_versions("never-written") == []


def test_get_checksum_mismatch_raises_loudly(fake_client: _FakeS3Client, tmp_path: Path) -> None:
    backend = S3CompatibleStateBackend(config=_config())
    version_id = backend.put(_write(tmp_path / "src.enc", b"authentic bytes"), "lean")

    # Corrupt the stored object body directly in the fake store -- simulates
    # bit-rot on the remote side after a successful upload.
    object_key = f"turboedge-state/lean/{version_id}.enc"
    fake_client.objects[object_key]["body"] = b"TAMPERED, wrong bytes entirely"

    with pytest.raises(StateBackendError, match="checksum mismatch"):
        backend.get("lean", tmp_path / "out.enc", version=version_id)

    assert not (tmp_path / "out.enc").exists()


def test_put_upload_failure_leaves_no_version(fake_client: _FakeS3Client, tmp_path: Path) -> None:
    """An aborted/failed upload (network drop, etc.) must not leave anything
    a later list_versions/get can pick up as valid."""
    fake_client.fail_put = True
    backend = S3CompatibleStateBackend(config=_config())
    src = _write(tmp_path / "src.enc", b"data that never arrives")

    with pytest.raises(StateBackendError, match="upload"):
        backend.put(src, "lean")

    assert backend.list_versions("lean") == []
    assert fake_client.objects == {}


def test_put_post_upload_verification_mismatch_raises(
    fake_client: _FakeS3Client, tmp_path: Path
) -> None:
    """Even if put_object itself reports success, a mismatching HEAD
    (wrong metadata landed -- e.g. a misbehaving proxy) must fail loudly
    rather than being treated as a good version."""
    fake_client.corrupt_next_head = True
    backend = S3CompatibleStateBackend(config=_config())
    src = _write(tmp_path / "src.enc", b"some payload")

    with pytest.raises(StateBackendError, match="verification failed"):
        backend.put(src, "lean")


def test_get_no_versions_raises_not_empty_state(fake_client: _FakeS3Client, tmp_path: Path) -> None:
    backend = S3CompatibleStateBackend(config=_config())
    with pytest.raises(StateBackendError, match="no versions found"):
        backend.get("never-written", tmp_path / "out.enc")
    assert not (tmp_path / "out.enc").exists()


def test_health_reachable(fake_client: _FakeS3Client) -> None:
    backend = S3CompatibleStateBackend(config=_config())
    result = backend.health()
    assert result.reachable is True


def test_health_unreachable_returns_result_not_exception(
    fake_client: _FakeS3Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**kwargs: object) -> None:
        raise RuntimeError("simulated connection refused")

    monkeypatch.setattr(fake_client, "list_objects_v2", _boom)
    backend = S3CompatibleStateBackend(config=_config())

    result = backend.health()

    assert result.reachable is False
    assert "unreachable" in result.reason or "connection refused" in result.reason


def test_list_versions_propagates_backend_error_on_network_failure(
    fake_client: _FakeS3Client, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unreachable backend must raise from list_versions/get -- it must
    never be treated as 'zero versions exist' (which would look identical
    to a legitimately fresh backend and risk a silent fresh-state reset)."""

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("simulated DNS failure")

    monkeypatch.setattr(fake_client, "get_paginator", _boom)
    backend = S3CompatibleStateBackend(config=_config())

    with pytest.raises(StateBackendError, match="listing"):
        backend.list_versions("lean")


def test_config_from_env_never_imports_boto3_until_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """config_from_env() is pure env-var validation; it must not attempt
    any network or boto3 import at all."""
    monkeypatch.setenv("TURBOEDGE_STATE_S3_BUCKET", "b")
    monkeypatch.setenv("TURBOEDGE_STATE_S3_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("TURBOEDGE_STATE_S3_SECRET_ACCESS_KEY", "sk")
    monkeypatch.delenv("TURBOEDGE_STATE_S3_ENDPOINT_URL", raising=False)

    config = backend_s3.config_from_env()

    assert config.bucket == "b"
    assert config.region == backend_s3.DEFAULT_REGION
    assert config.prefix == backend_s3.DEFAULT_PREFIX
    assert config.endpoint_url is None
