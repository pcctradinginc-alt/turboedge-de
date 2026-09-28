"""Tests for the raw payload archive: content addressing and redaction."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from turboedge.external.adapter import FetchedPayload
from turboedge.external.raw_archive import (
    archive_size_bytes,
    payload_hash,
    read_payload,
    redact,
    store_payload,
)
from turboedge.external.schemas import RawPayload

_NOW = datetime(2026, 9, 28, 6, 0, tzinfo=UTC)


def _payload(content: bytes = b"a,b\n1,2\n", **over: object) -> FetchedPayload:
    defaults: dict[str, object] = dict(
        source="ecb",
        dataset="EST",
        url="https://data-api.ecb.europa.eu/service/data/EST/B.X.WT",
        content=content,
        http_status=200,
        content_type="application/json",
        retrieved_at=_NOW,
        request_fingerprint="GET /service/data/EST/B.X.WT",
    )
    defaults.update(over)
    return FetchedPayload(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    [
        "https://api.stlouisfed.org/fred/series?api_key=abc123&series_id=INDPRO",
        "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData?appId=SECRET&statsDataId=1",
        "GET /x?token=tok123&y=1",
        "https://h/x?API_KEY=Mixed&z=2",
    ],
)
def test_credentials_are_redacted(raw: str) -> None:
    out = redact(raw)

    assert "REDACTED" in out
    for leaked in ("abc123", "SECRET", "tok123", "Mixed"):
        assert leaked not in out


def test_redaction_keeps_the_rest_of_the_request_readable() -> None:
    out = redact("https://api.stlouisfed.org/fred/series?api_key=abc&series_id=INDPRO")

    assert "series_id=INDPRO" in out


def test_storing_the_same_payload_twice_is_idempotent(tmp_path: Path) -> None:
    first = store_payload(_payload(), state_dir=tmp_path, parser_version="1")
    second = store_payload(_payload(), state_dir=tmp_path, parser_version="1")

    assert first.payload_id == second.payload_id
    assert first.stored_path == second.stored_path
    assert len(list((tmp_path / "raw").rglob("*.*"))) == 1


def test_different_bytes_are_archived_side_by_side(tmp_path: Path) -> None:
    # A genuine revision must not overwrite the original.
    store_payload(_payload(b"first"), state_dir=tmp_path, parser_version="1")
    store_payload(_payload(b"revised"), state_dir=tmp_path, parser_version="1")

    assert len(list((tmp_path / "raw").rglob("*.*"))) == 2


def test_round_trip_preserves_the_bytes_exactly(tmp_path: Path) -> None:
    content = b'{"x": 1}' * 2000  # large enough to be gzipped
    record = store_payload(_payload(content), state_dir=tmp_path, parser_version="1")

    assert record.content_encoding == "gzip"
    assert record.byte_size == len(content)
    assert read_payload(record) == content


def test_small_payloads_are_stored_uncompressed(tmp_path: Path) -> None:
    record = store_payload(_payload(b"tiny"), state_dir=tmp_path, parser_version="1")

    assert record.content_encoding != "gzip"
    assert read_payload(record) == b"tiny"


def test_a_tampered_archive_file_is_refused(tmp_path: Path) -> None:
    record = store_payload(_payload(b"original"), state_dir=tmp_path, parser_version="1")
    Path(record.stored_path).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="refusing to reparse"):
        read_payload(record)


def test_a_missing_archive_file_is_reported_clearly(tmp_path: Path) -> None:
    record = store_payload(_payload(), state_dir=tmp_path, parser_version="1")
    Path(record.stored_path).unlink()

    with pytest.raises(FileNotFoundError, match="archived payload missing"):
        read_payload(record)


def test_a_credential_reaching_the_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="refusing to persist"):
        RawPayload(
            payload_id="p",
            source="fred",
            dataset="INDPRO",
            request_fingerprint="GET /fred/series?api_key=leaked",
            retrieved_at=_NOW,
            http_status=200,
            content_type="application/json",
            byte_size=1,
            payload_hash="h",
            stored_path="/tmp/x.json",
            parser_version="1",
        )


def test_store_payload_redacts_before_the_model_can_reject(tmp_path: Path) -> None:
    record = store_payload(
        _payload(request_fingerprint="GET /fred/series?api_key=leaked&series_id=INDPRO"),
        state_dir=tmp_path,
        parser_version="1",
    )

    assert "leaked" not in record.request_fingerprint
    assert "series_id=INDPRO" in record.request_fingerprint


def test_partial_files_are_never_left_at_a_content_addressed_path(tmp_path: Path) -> None:
    store_payload(_payload(), state_dir=tmp_path, parser_version="1")

    assert not list((tmp_path / "raw").rglob("*.partial"))


def test_archive_size_is_reportable(tmp_path: Path) -> None:
    assert archive_size_bytes(tmp_path) == 0
    store_payload(_payload(b"x" * 100), state_dir=tmp_path, parser_version="1")

    assert archive_size_bytes(tmp_path) > 0


def test_hash_is_of_the_original_bytes(tmp_path: Path) -> None:
    content = b"y" * 9000
    record = store_payload(_payload(content), state_dir=tmp_path, parser_version="1")

    assert record.payload_hash == payload_hash(content)
