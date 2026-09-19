"""S3-compatible remote :class:`~turboedge.state.backend.StateBackend`.

Works with AWS S3 *and* any S3-compatible object store -- Cloudflare R2,
Backblaze B2, Wasabi, MinIO, DigitalOcean Spaces, etc. -- via the standard
``endpoint_url`` override, so a user is never locked into one vendor to keep
years of learning history durable (that vendor-neutrality is a deliberate
requirement of Phase H, not an accident). AWS S3 itself works too: leave
``TURBOEDGE_STATE_S3_ENDPOINT_URL`` unset and set a real AWS region.

Optional dependency
--------------------
This module needs ``boto3`` (the ``state-remote`` extra: ``uv sync --extra
state-remote``). The rest of the package has no hard dependency on it --
:func:`turboedge.state.backend.backend_from_env` only imports this module
when ``TURBOEDGE_STATE_BACKEND=s3`` is actually set, and this module itself
only imports ``boto3`` inside :meth:`S3CompatibleStateBackend.__init__`, so
merely having ``TURBOEDGE_STATE_BACKEND`` unset never requires it installed.

Credentials and configuration come *exclusively* from environment
variables (never a config file, never a CLI flag, never anything that could
land in the repo or a log line) -- see ``docs/durable_state.md``:

- ``TURBOEDGE_STATE_S3_BUCKET`` (required)
- ``TURBOEDGE_STATE_S3_ACCESS_KEY_ID`` (required)
- ``TURBOEDGE_STATE_S3_SECRET_ACCESS_KEY`` (required)
- ``TURBOEDGE_STATE_S3_ENDPOINT_URL`` (optional; omit for AWS S3 proper)
- ``TURBOEDGE_STATE_S3_REGION`` (optional; default ``"auto"``, the
  Cloudflare R2 convention -- AWS S3 needs a real region string here)
- ``TURBOEDGE_STATE_S3_PREFIX`` (optional; default ``"turboedge-state"``)

Encryption: this backend only ever transports whatever bytes are already at
``local_path`` -- the caller (``turboedge state pack`` followed by
``turboedge state backend put``) is responsible for encrypting first (see
``turboedge.state.archive``/``turboedge.state.crypto``). This module never
sees plaintext state and performs no encryption/decryption of its own.

Integrity: every object is uploaded with an S3 ``Content-MD5`` header, which
makes S3 itself reject (400) a body that arrives corrupted/truncated --
that is what actually prevents an interrupted upload from ever landing as a
valid-looking object, on top of the SHA-256 stored as object metadata and
re-verified on every ``get``/``put``.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from turboedge.state.backend import (
    HealthResult,
    StateBackendError,
    VersionInfo,
    new_version_id,
    parse_created_at,
)

_ENV_BUCKET = "TURBOEDGE_STATE_S3_BUCKET"
_ENV_ACCESS_KEY = "TURBOEDGE_STATE_S3_ACCESS_KEY_ID"
_ENV_SECRET_KEY = "TURBOEDGE_STATE_S3_SECRET_ACCESS_KEY"
_ENV_ENDPOINT = "TURBOEDGE_STATE_S3_ENDPOINT_URL"
_ENV_REGION = "TURBOEDGE_STATE_S3_REGION"
_ENV_PREFIX = "TURBOEDGE_STATE_S3_PREFIX"

DEFAULT_REGION = "auto"
DEFAULT_PREFIX = "turboedge-state"

_META_SHA256 = "turboedge-sha256"
_META_SIZE = "turboedge-size"

_INSTALL_HINT = "uv sync --extra state-remote"


@dataclass(frozen=True)
class S3Config:
    bucket: str
    access_key_id: str
    secret_access_key: str
    endpoint_url: str | None
    region: str
    prefix: str


def config_from_env() -> S3Config:
    """Build :class:`S3Config` from ``TURBOEDGE_STATE_S3_*`` env vars.

    Raises :class:`StateBackendError` (never falls back to a default) if
    any of the three required variables is missing/blank -- credentials are
    never invented or reused from elsewhere.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    access_key = os.environ.get(_ENV_ACCESS_KEY, "").strip()
    secret_key = os.environ.get(_ENV_SECRET_KEY, "").strip()

    required = (
        (_ENV_BUCKET, bucket),
        (_ENV_ACCESS_KEY, access_key),
        (_ENV_SECRET_KEY, secret_key),
    )
    missing = [name for name, val in required if not val]
    if missing:
        raise StateBackendError(
            "S3-compatible remote state backend (TURBOEDGE_STATE_BACKEND=s3) is missing "
            f"required environment variable(s): {', '.join(missing)}. See docs/durable_state.md."
        )

    endpoint = os.environ.get(_ENV_ENDPOINT, "").strip() or None
    region = os.environ.get(_ENV_REGION, "").strip() or DEFAULT_REGION
    prefix = os.environ.get(_ENV_PREFIX, "").strip() or DEFAULT_PREFIX
    return S3Config(
        bucket=bucket,
        access_key_id=access_key,
        secret_access_key=secret_key,
        endpoint_url=endpoint,
        region=region,
        prefix=prefix,
    )


def _boto3_client(config: S3Config) -> Any:
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:
        raise StateBackendError(
            "the S3-compatible remote state backend needs the optional 'boto3' dependency "
            f"-- install it with: {_INSTALL_HINT}"
        ) from exc

    session = boto3.session.Session()
    return session.client(
        "s3",
        region_name=config.region,
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        config=BotoConfig(
            signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}
        ),
    )


class S3CompatibleStateBackend:
    """Durable, versioned, checksum-verified backend on an S3-compatible bucket.

    Object layout: ``<prefix>/<key>/<version_id>.enc`` -- ``put`` always
    writes a brand-new object key (never overwrites), so this is
    append-only by construction; there is no server-side delete/overwrite
    path anywhere in this class. ``version_id`` (see
    :func:`turboedge.state.backend.new_version_id`) is chosen client-side
    *before* the upload, so a failed ``put_object`` call simply means that
    object key was never created -- nothing to clean up, and nothing for a
    concurrent ``list_versions``/``get`` to ever see.
    """

    def __init__(self, config: S3Config | None = None) -> None:
        self._config = config or config_from_env()
        self._client = _boto3_client(self._config)

    def _object_key(self, key: str, version_id: str) -> str:
        if not key or key in {".", ".."} or "/" in key or "\\" in key:
            raise StateBackendError(f"invalid backend key: {key!r}")
        return f"{self._config.prefix}/{key}/{version_id}.enc"

    def _prefix_for(self, key: str) -> str:
        return f"{self._config.prefix}/{key}/"

    def put(self, local_path: Path, key: str) -> str:
        local_path = Path(local_path)
        if not local_path.is_file():
            raise StateBackendError(f"put: no such file: {local_path}")

        data = local_path.read_bytes()
        data_sha256 = hashlib.sha256(data).hexdigest()
        data_md5 = hashlib.md5(data, usedforsecurity=False).digest()
        content_md5 = base64.b64encode(data_md5).decode("ascii")
        version_id = new_version_id()
        object_key = self._object_key(key, version_id)

        try:
            self._client.put_object(
                Bucket=self._config.bucket,
                Key=object_key,
                Body=data,
                ContentMD5=content_md5,
                Metadata={_META_SHA256: data_sha256, _META_SIZE: str(len(data))},
            )
        except Exception as exc:
            raise StateBackendError(
                f"put: upload of key {key!r} version {version_id} to "
                f"s3://{self._config.bucket}/{object_key} failed: {exc}"
            ) from exc

        # Trust, but verify: a 200 from put_object plus a passing
        # Content-MD5 check already makes silent truncation very unlikely,
        # but re-reading the object's own recorded metadata closes the loop
        # end-to-end (e.g. against a misbehaving proxy/mirror) rather than
        # taking the upload's own success response on faith.
        try:
            head = self._client.head_object(Bucket=self._config.bucket, Key=object_key)
        except Exception as exc:
            raise StateBackendError(
                f"put: uploaded key {key!r} version {version_id} but post-upload verification "
                f"(HEAD) failed: {exc} -- treat this version as unverified/invalid"
            ) from exc

        remote_sha256 = (head.get("Metadata") or {}).get(_META_SHA256)
        remote_size = head.get("ContentLength")
        if remote_sha256 != data_sha256 or remote_size != len(data):
            raise StateBackendError(
                f"put: post-upload verification failed for key {key!r} version {version_id} "
                f"-- expected sha256={data_sha256} size={len(data)}, got "
                f"sha256={remote_sha256!r} size={remote_size!r}; treat this version as invalid"
            )
        return version_id

    def get(self, key: str, local_path: Path, version: str | None = None) -> str:
        chosen = version
        if chosen is None:
            versions = self.list_versions(key)
            if not versions:
                raise StateBackendError(
                    f"get: no versions found for key {key!r} in bucket {self._config.bucket}"
                )
            chosen = versions[-1].version_id

        object_key = self._object_key(key, chosen)
        try:
            resp = self._client.get_object(Bucket=self._config.bucket, Key=object_key)
            data = resp["Body"].read()
        except Exception as exc:
            raise StateBackendError(
                f"get: download of key {key!r} version {chosen!r} from "
                f"s3://{self._config.bucket}/{object_key} failed: {exc}"
            ) from exc

        expected_sha256 = (resp.get("Metadata") or {}).get(_META_SHA256)
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise StateBackendError(
                f"get: checksum mismatch for key {key!r} version {chosen!r} -- expected "
                f"{expected_sha256!r}, got {actual_sha256!r}; refusing to hand back corrupted data"
            )

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_name(f".{local_path.name}.tmp-{uuid4().hex}")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, local_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        return chosen

    def list_versions(self, key: str) -> list[VersionInfo]:
        prefix = self._prefix_for(key)
        out: list[VersionInfo] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
                for obj in page.get("Contents", None) or []:
                    object_key = str(obj["Key"])
                    version_id = object_key[len(prefix) :]
                    if version_id.endswith(".enc"):
                        version_id = version_id[: -len(".enc")]
                    try:
                        head = self._client.head_object(Bucket=self._config.bucket, Key=object_key)
                    except Exception as exc:
                        raise StateBackendError(
                            f"list_versions: could not read metadata for {object_key!r}: {exc}"
                        ) from exc
                    sha256_val = str((head.get("Metadata") or {}).get(_META_SHA256, ""))
                    last_modified = obj.get("LastModified") or datetime.now(UTC)
                    out.append(
                        VersionInfo(
                            version_id=version_id,
                            created_at=parse_created_at(version_id, last_modified),
                            size=int(obj["Size"]),
                            sha256=sha256_val,
                        )
                    )
        except StateBackendError:
            raise
        except Exception as exc:
            raise StateBackendError(
                f"list_versions: listing s3://{self._config.bucket}/{prefix} failed: {exc}"
            ) from exc

        out.sort(key=lambda v: v.version_id)
        return out

    def health(self) -> HealthResult:
        try:
            self._client.list_objects_v2(
                Bucket=self._config.bucket, Prefix=self._config.prefix, MaxKeys=1
            )
            return HealthResult(reachable=True, reason=f"reachable (bucket={self._config.bucket})")
        except Exception as exc:
            return HealthResult(reachable=False, reason=f"unreachable or unauthorized: {exc}")


__all__ = [
    "DEFAULT_PREFIX",
    "DEFAULT_REGION",
    "S3CompatibleStateBackend",
    "S3Config",
    "config_from_env",
]
