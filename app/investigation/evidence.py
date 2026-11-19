"""Canonical Evidence construction helpers shared by chat and report adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.investigation.contracts import AttributionEvidence
from app.investigation.fingerprint import canonicalize


def scope_fingerprint(scope: dict[str, Any]) -> str:
    canonical = json.dumps(canonicalize(scope), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_attribution_evidence(
    *, evidence_id: str, evidence_type: str, metric_id: str, claim: str,
    values: dict[str, Any], source_action_id: str, source_ref: str,
    scope: dict[str, Any], method_id: str, method_version: str,
    authorized_metric_provenance: str, unit: str = "",
    trace_id: str | None = None,
    comparison: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> AttributionEvidence:
    """Validate the traceable evidence envelope before it enters a surface."""
    return AttributionEvidence(
        evidence_id=evidence_id, trace_id=trace_id,
        evidence_type=evidence_type, metric_id=metric_id,
        claim=claim, values=values, unit=unit, source_action_id=source_action_id,
        source_ref=source_ref, scope_fingerprint=scope_fingerprint(scope),
        method_id=method_id, method_version=method_version,
        authorized_metric_provenance=authorized_metric_provenance,
        comparison=comparison or {}, data_quality=data_quality or {},
        limitations=limitations or [],
    )
