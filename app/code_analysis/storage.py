"""Tenant-scoped artifact storage with a local development implementation."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile

from app.core.config import get_settings


_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,1023}$")


def _validate_key(key: str) -> str:
    if (
        not _SAFE_KEY.fullmatch(key)
        or key.startswith("/")
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError("Invalid artifact key")
    return key


def owner_artifact_prefix(tenant_id: str, user_id: str) -> str:
    """Return a non-identifying, path-safe owner prefix.

    Tenant claims are never interpolated directly into filesystem/S3 paths.
    Authorization remains database/RLS based; this prefix only partitions
    objects and avoids leaking user identifiers in storage listings.
    """
    tenant = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:32]
    user = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
    return f"tenants/{tenant}/users/{user}"


class ArtifactStore:
    def put(self, key: str, data: bytes) -> tuple[str, str, int]:
        raise NotImplementedError

    def get(self, key: str, *, max_bytes: int | None = None) -> bytes:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    """Filesystem implementation used only for local development/tests."""
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / _validate_key(key)).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError("Invalid artifact key")
        return candidate

    def put(self, key: str, data: bytes) -> tuple[str, str, int]:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A crash must not leave a partially written artifact that still has a
        # valid database reference.
        fd, temporary = tempfile.mkstemp(prefix=".upload-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return key, hashlib.sha256(data).hexdigest(), len(data)

    def get(self, key: str, *, max_bytes: int | None = None) -> bytes:
        path = self._path(key)
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise ValueError("Artifact exceeds the allowed read size")
        with path.open("rb") as handle:
            data = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("Artifact exceeds the allowed read size")
        return data

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3ArtifactStore(ArtifactStore):
    """S3-compatible implementation; boto3 stays optional for local API use."""
    def __init__(self) -> None:
        settings = get_settings()
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as exc:  # pragma: no cover - deployment configuration
            raise RuntimeError("S3 artifact storage requires boto3") from exc
        self._client_error = ClientError
        self.bucket = settings.code_analysis_s3_bucket
        self.client = boto3.client("s3", endpoint_url=settings.code_analysis_s3_endpoint or None,
            aws_access_key_id=settings.code_analysis_s3_access_key or None,
            aws_secret_access_key=settings.code_analysis_s3_secret_key or None)
        self.encryption = settings.code_analysis_s3_server_side_encryption
        self.kms_key_id = settings.code_analysis_s3_kms_key_id
        self._ready = False

    def _ensure_bucket(self) -> None:
        if self._ready: return
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except self._client_error as exc:
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if status not in {404} and code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            self.client.create_bucket(Bucket=self.bucket)
        self._ready = True

    def put(self, key: str, data: bytes) -> tuple[str, str, int]:
        self._ensure_bucket()
        key = _validate_key(key)
        digest = hashlib.sha256(data).hexdigest()
        options: dict[str, str] = {}
        if self.encryption:
            options["ServerSideEncryption"] = self.encryption
        if self.encryption == "aws:kms" and self.kms_key_id:
            options["SSEKMSKeyId"] = self.kms_key_id
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            Metadata={"sha256": digest},
            **options,
        )
        return key, digest, len(data)

    def get(self, key: str, *, max_bytes: int | None = None) -> bytes:
        self._ensure_bucket()
        key = _validate_key(key)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        size = int(response.get("ContentLength") or 0)
        if max_bytes is not None and size > max_bytes:
            body.close()
            raise ValueError("Artifact exceeds the allowed read size")
        try:
            data = body.read() if max_bytes is None else body.read(max_bytes + 1)
        finally:
            body.close()
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("Artifact exceeds the allowed read size")
        return data

    def delete(self, key: str) -> None:
        self._ensure_bucket()
        key = _validate_key(key)
        self.client.delete_object(Bucket=self.bucket, Key=key)


def get_artifact_store() -> ArtifactStore:
    settings = get_settings()
    return S3ArtifactStore() if settings.code_analysis_storage_backend == "s3" else LocalArtifactStore(settings.code_analysis_storage_dir)
