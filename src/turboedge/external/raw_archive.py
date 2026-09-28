"""Permanent archive of raw upstream responses (spec §6A, §23, §24).

A parser bug found in 2028 is repairable only if the bytes that were parsed
in 2026 still exist. Publishers do not guarantee that a historical endpoint
keeps serving what it served last year -- several of the sources here serve
"the current file" and nothing else -- so refetching is not a recovery plan.

Two properties matter and both are structural rather than procedural:

* **Content-addressed.** The file name is the hash of the bytes. Refetching
  an unchanged file writes the same path and the same row, so ingestion is
  idempotent (spec §7) without anyone tracking what was already downloaded.
  A genuine revision has different bytes, therefore a different hash, and is
  archived beside the original rather than over it.
* **Credential-free.** A stored URL with an `api_key=` in it is a leaked
  credential with a long half-life, and this repository is public. The
  redaction happens here, on the way in, and `RawPayload` refuses anything
  that slipped through.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

from turboedge.external.adapter import FetchedPayload
from turboedge.external.schemas import _CREDENTIAL_PARAM_RE, RawPayload
from turboedge.provenance import git_commit

#: The redaction pattern is the model's own, imported rather than
#: redefined: `RawPayload` rejects what this misses, so a second list here
#: would guarantee the two drift apart -- which is precisely how
#: `securityToken=` and `x-key=` stayed unredacted until 2026-09-28.
_SECRET_RE = _CREDENTIAL_PARAM_RE

#: Payloads at or above this size are gzipped on disk. Below it the gzip
#: header is a meaningful fraction of the file and costs more than it saves.
_COMPRESS_THRESHOLD_BYTES = 4096

_EXTENSION_BY_TYPE: dict[str, str] = {
    "application/json": "json",
    "application/vnd.sdmx.data+json": "json",
    "text/csv": "csv",
    "text/plain": "txt",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/zip": "zip",
}


def redact(text: str) -> str:
    """Replace every credential value in a URL or fingerprint with `REDACTED`.

    Deliberately blunt: it redacts on parameter *name*, so a new source with
    an unusual credential parameter is a one-line addition to `_SECRET_PARAMS`
    rather than a leak. Over-redacting a harmless `key=` parameter costs
    nothing; under-redacting costs a rotated credential and a rewritten
    history.
    """
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}=REDACTED", text)


def payload_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extension_for(content_type: str) -> str:
    base = content_type.split(";", 1)[0].strip().lower()
    return _EXTENSION_BY_TYPE.get(base, "bin")


def archive_path(
    state_dir: Path, *, source: str, day: str, digest: str, content_type: str, gzipped: bool
) -> Path:
    suffix = extension_for(content_type)
    name = f"{digest[:16]}.{suffix}" + (".gz" if gzipped else "")
    return Path(state_dir) / "raw" / source / f"date={day}" / name


def store_payload(
    payload: FetchedPayload,
    *,
    state_dir: Path,
    parser_version: str,
) -> RawPayload:
    """Write one payload to the archive and return its manifest row.

    Writing the same payload twice is a no-op on disk (same content, same
    path) and produces an identical `payload_id`, so the caller can persist
    the row unconditionally.

    `byte_size` records the size of the *original* bytes, not of the stored
    file, because that is what a later reader needs to check the payload came
    back intact.
    """
    digest = payload_hash(payload.content)
    day = payload.retrieved_at.date().isoformat()
    gzipped = len(payload.content) >= _COMPRESS_THRESHOLD_BYTES

    path = archive_path(
        state_dir,
        source=payload.source,
        day=day,
        digest=digest,
        content_type=payload.content_type,
        gzipped=gzipped,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Written via a temporary sibling then renamed: a crash mid-write
        # must not leave a truncated file at a content-addressed path, where
        # its name would assert a hash its contents no longer have.
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_bytes(gzip.compress(payload.content) if gzipped else payload.content)
        tmp.replace(path)

    return RawPayload(
        payload_id=f"{payload.source}:{payload.dataset}:{digest[:16]}",
        source=payload.source,
        dataset=payload.dataset,
        request_fingerprint=redact(payload.request_fingerprint),
        retrieved_at=payload.retrieved_at,
        http_status=payload.http_status,
        content_type=payload.content_type,
        content_encoding="gzip" if gzipped else payload.content_encoding,
        byte_size=len(payload.content),
        payload_hash=digest,
        stored_path=str(path),
        parser_version=parser_version,
        git_commit=git_commit(),
    )


def read_payload(record: RawPayload) -> bytes:
    """Read an archived payload back, verifying it is what it claims to be.

    The hash check is not paranoia about disk corruption. It is what makes a
    reprocessing run (spec §24) trustworthy: reparsing bytes that silently
    changed would produce a "corrected" history that never existed.
    """
    path = Path(record.stored_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"archived payload missing: {path} (recorded for {record.payload_id})"
        )
    raw = path.read_bytes()
    content = gzip.decompress(raw) if path.suffix == ".gz" else raw
    actual = payload_hash(content)
    if actual != record.payload_hash:
        raise ValueError(
            f"{record.payload_id}: archived bytes hash to {actual[:16]} but the manifest "
            f"records {record.payload_hash[:16]}; refusing to reparse a payload that "
            "is not the one that was fetched"
        )
    return content


def archive_size_bytes(state_dir: Path) -> int:
    root = Path(state_dir) / "raw"
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
