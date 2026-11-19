"""Versioned RAG release manifest with published-only retrieval policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings


class RagRelease(BaseModel):
    version: str
    source: str
    status: Literal["canary", "published", "rolled_back"]
    approved_by: str | None = None


class RagReleaseManifest(BaseModel):
    releases: list[RagRelease] = Field(default_factory=list)

    def published(self) -> RagRelease | None:
        return next((item for item in reversed(self.releases) if item.status == "published"), None)


def load_release_manifest(path: str | None = None) -> RagReleaseManifest:
    target = Path(path or get_settings().rag_release_manifest_path)
    if not target.exists():
        return RagReleaseManifest(releases=[RagRelease(version=get_settings().rag_index_version, source="local", status="published", approved_by="bootstrap")])
    return RagReleaseManifest.model_validate(json.loads(target.read_text(encoding="utf-8")))


def published_rag_version() -> str:
    release = load_release_manifest().published()
    if release is None:
        raise RuntimeError("No published RAG release is available.")
    return release.version
