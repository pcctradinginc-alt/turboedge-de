from __future__ import annotations

import subprocess
from datetime import UTC, datetime

import pytest

from turboedge.provenance import data_snapshot_hash, git_commit, new_run_id, sha256_json
from turboedge.storage.schemas import HealthStatus, SourceHealthRecord


def test_sha256_json_is_deterministic_regardless_of_key_order() -> None:
    a = sha256_json({"b": 1, "a": 2})
    b = sha256_json({"a": 2, "b": 1})
    assert a == b
    assert len(a) == 64


def test_sha256_json_differs_for_different_content() -> None:
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_sha256_json_handles_pydantic_models() -> None:
    record = SourceHealthRecord(
        source="ecb",
        checked_at=datetime(2026, 9, 10, tzinfo=UTC),
        availability=1.0,
        freshness=1.0,
        missingness=0.0,
        schema_consistency=1.0,
        cross_source_agreement=None,
        score=1.0,
        status=HealthStatus.PASS,
        message="ok",
    )
    digest = sha256_json(record)
    assert len(digest) == 64


def test_data_snapshot_hash_stable_for_same_ordered_records() -> None:
    records = [{"x": 1}, {"x": 2}]
    assert data_snapshot_hash(records) == data_snapshot_hash(list(records))


def test_data_snapshot_hash_sensitive_to_order() -> None:
    a = data_snapshot_hash([{"x": 1}, {"x": 2}])
    b = data_snapshot_hash([{"x": 2}, {"x": 1}])
    assert a != b


def test_new_run_id_unique_and_sortable_prefix() -> None:
    first = new_run_id()
    second = new_run_id()
    assert first != second
    assert "T" in first
    date_part, _, suffix = first.partition("-")
    assert len(date_part) == 16  # YYYYMMDDTHHMMSSZ
    assert len(suffix) == 12


def test_git_commit_prefers_github_sha_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "deadbeef" * 5)
    assert git_commit() == "deadbeef" * 5


def test_git_commit_falls_back_to_git_rev_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    class _FakeCompletedProcess:
        returncode = 0
        stdout = "abc123\n"

    def _fake_run(*args: object, **kwargs: object) -> _FakeCompletedProcess:
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert git_commit() == "abc123"


def test_git_commit_returns_none_when_git_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def _fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert git_commit() is None
