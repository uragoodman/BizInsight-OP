"""Authoritative metric authorization boundary.

Semantic retrieval is advisory.  Only this module may turn a canonical metric
reference plus verified provenance into an executable metric authorization.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ProvenanceType = Literal["exact", "semantic_mapping", "explorer_selection", "registered_relation"]


class AuthorizedMetric(BaseModel):
    """A server-authorized metric that QueryPlanBuilder is allowed to consume."""

    model_config = ConfigDict(frozen=True)

    metric_id: str = Field(min_length=1)
    provenance_type: ProvenanceType
    provenance_id: str = Field(min_length=1)
    # Split fingerprints are the authoritative fields.  ``scope_fingerprint``
    # stays as a read-only compatibility field for older graph payloads.
    authorization_fingerprint: str | None = Field(default=None, min_length=1)
    query_scope_fingerprint: str | None = Field(default=None, min_length=1)
    scope_fingerprint: str = Field(min_length=1)
    catalog_digest: str = Field(min_length=1)

    def model_post_init(self, __context: Any) -> None:
        if self.query_scope_fingerprint is None:
            object.__setattr__(self, "query_scope_fingerprint", self.scope_fingerprint)


def canonical_scope_payload(scope: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize scope values before hashing them for mapping invalidation."""

    raw = dict(scope or {})
    # Only stable authorization/scope inputs belong in this fingerprint.
    # Derived values (query_scope_fingerprint), mapping confirmation lists,
    # resource limits, and UI-only fields must not make an AuthorizedMetric
    # stale merely because they are present at one boundary and omitted at
    # another.  The catalog digest and the explicit authz versions cover
    # metric-definition and policy changes separately.
    stable_keys = {
        "tenant_id", "user_id", "scope_type", "permissions", "role",
        "seller_id", "seller_ids", "start_date", "end_date",
        "default_start_date", "default_end_date", "metric_name",
        "allowed_metrics", "allowed_dimensions", "dimensions",
        "timezone", "granularity", "order_status", "currency",
        "filters", "authorization_fingerprint", "permission_version",
        "policy_set_version", "authz_ref", "actions",
        "seller_scope_type", "include_unattributed",
    }
    raw = {key: value for key, value in raw.items() if key in stable_keys}
    seller_ids = raw.get("seller_ids")
    if seller_ids is None and raw.get("seller_id") is not None:
        seller_ids = [raw.get("seller_id")]
    if seller_ids is not None:
        raw["seller_ids"] = sorted(str(value) for value in seller_ids if value is not None)
        raw.pop("seller_id", None)

    for key in ("dimensions", "allowed_metrics", "permissions"):
        value = raw.get(key)
        if isinstance(value, (list, tuple, set)):
            raw[key] = sorted(str(item) for item in value)

    for key in ("start_date", "end_date", "default_start_date", "default_end_date"):
        value = raw.get(key)
        if isinstance(value, (date, datetime)):
            raw[key] = value.isoformat()
        elif value is not None:
            raw[key] = str(value)

    # Optional identity fields are intentionally omitted when they are not
    # present.  Authorization itself is checked separately against the
    # trusted TenantContext; a metric selection created before identity fields
    # were attached must not become stale merely because ``None`` was serialized.
    return {str(key): raw[key] for key in sorted(raw) if raw[key] is not None}


def scope_fingerprint(scope: dict[str, Any] | None) -> str:
    canonical = json.dumps(
        canonical_scope_payload(scope), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _authorization_scope(scope: str, scope_data: dict[str, Any] | None) -> dict[str, Any]:
    """Build the same scope payload for authorization and re-validation."""
    payload = dict(scope_data or {})
    payload.setdefault("scope_type", scope)
    return payload


def _compact_exact(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def server_exact_mentions(question: str, *, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find longest non-overlapping exact mentions with Python-codepoint spans.

    A normalized compact view carries the original code-point index for each
    character, so full-width/case/whitespace normalization cannot make the
    server trust an LLM-provided span.
    """
    compact_chars: list[str] = []
    original_indexes: list[int] = []
    for index, char in enumerate(question):
        normalized = unicodedata.normalize("NFKC", char).casefold()
        if normalized.isspace():
            continue
        for normalized_char in normalized:
            compact_chars.append(normalized_char)
            original_indexes.append(index)
    compact_question = "".join(compact_chars)
    matches: list[tuple[int, int, int, str, str, int, int]] = []
    for item in catalog:
        metric_id = str(item.get("metric_id") or "")
        phrases = [
            metric_id,
            str(item.get("display_name") or ""),
            *[str(v) for v in item.get("exact_aliases", [])],
        ]
        for phrase in phrases:
            compact_phrase = _compact_exact(phrase)
            if not compact_phrase:
                continue
            cursor = 0
            while True:
                start = compact_question.find(compact_phrase, cursor)
                if start < 0:
                    break
                end = start + len(compact_phrase)
                original_start = original_indexes[start]
                original_end = original_indexes[end - 1] + 1
                if re.fullmatch(r"[a-z0-9_]+", compact_phrase):
                    # Check the original text so whitespace remains a token
                    # boundary.  Checking the whitespace-stripped string
                    # would reject ``GMV mean`` as the fake token ``gmvmean``.
                    left = question[original_start - 1].casefold() if original_start else ""
                    right = question[original_end].casefold() if original_end < len(question) else ""
                    if re.fullmatch(r"[a-z0-9_]", left) or re.fullmatch(r"[a-z0-9_]", right):
                        cursor = start + 1
                        continue
                matches.append((
                    len(compact_phrase), start, end, metric_id, phrase,
                    original_start, original_end,
                ))
                cursor = start + 1
    occupied: list[tuple[int, int]] = []
    result: list[dict[str, Any]] = []
    for _, start, end, metric_id, _phrase, original_start, original_end in sorted(
        matches, key=lambda item: (-item[0], item[1], item[3])
    ):
        if any(start < right and end > left for left, right in occupied):
            continue
        occupied.append((start, end))
        result.append({
            "mention_id": f"mention:{uuid.uuid4().hex}",
            "raw_text": question[original_start:original_end],
            "start": original_start,
            "end": original_end,
            "metric_id": metric_id,
            "status": "exact",
            "provenance": "exact",
        })
    return sorted(result, key=lambda item: int(item["start"]))


def authorize_metric(
    metric_id: str,
    *,
    provenance_type: ProvenanceType,
    provenance_id: str,
    scope: str,
    scope_data: dict[str, Any] | None,
    catalog_digest: str | None = None,
) -> AuthorizedMetric:
    """Resolve and validate one metric at the authoritative boundary."""

    from app.metrics.registry import get_registry

    registry = get_registry()
    canonical = registry.resolve(metric_id)
    registry.get_implementation(canonical, scope)  # fail closed on scope
    trusted_scope = dict(scope_data or {})
    allowed_metrics = trusted_scope.get("allowed_metrics")
    if allowed_metrics is not None and canonical not in {str(item) for item in allowed_metrics}:
        raise ValueError("metric is outside the server-authorized metric scope")
    selected_metric = trusted_scope.get("metric_name")
    # The explorer's selected metric is a verified default context, not a
    # restriction that prevents the user from asking another metric already
    # allowed by the current scope. It is authoritative only when the
    # provenance explicitly claims the explorer selection.
    if selected_metric and provenance_type == "explorer_selection":
        if registry.resolve(str(selected_metric)) != canonical:
            raise ValueError("metric does not match the server-validated explorer selection")
    if provenance_type == "semantic_mapping":
        confirmed_ids = {
            str(item)
            for item in trusted_scope.get("confirmed_mapping_ids", [])
        }
        if provenance_id not in confirmed_ids:
            raise ValueError("semantic metric mapping is not a server-confirmed mapping")
    return AuthorizedMetric(
        metric_id=canonical,
        provenance_type=provenance_type,
        provenance_id=provenance_id,
        authorization_fingerprint=(scope_data or {}).get("authorization_fingerprint"),
        query_scope_fingerprint=scope_fingerprint(_authorization_scope(scope, scope_data)),
        scope_fingerprint=scope_fingerprint(_authorization_scope(scope, scope_data)),
        catalog_digest=catalog_digest or registry.catalog_digest,
    )


def validate_authorized_metric(
    metric: AuthorizedMetric,
    *,
    scope: str,
    scope_data: dict[str, Any] | None,
) -> AuthorizedMetric:
    """Re-check an authorization immediately before QueryPlan creation."""

    from app.metrics.registry import get_registry

    registry = get_registry()
    if metric.catalog_digest != registry.catalog_digest:
        raise ValueError("metric authorization catalog digest is stale")
    current_scope_fingerprint = scope_fingerprint(_authorization_scope(scope, scope_data))
    if (metric.query_scope_fingerprint or metric.scope_fingerprint) != current_scope_fingerprint:
        raise ValueError("metric authorization scope fingerprint is stale")
    canonical = registry.resolve(metric.metric_id)
    registry.get_implementation(canonical, scope)
    if canonical != metric.metric_id:
        raise ValueError("metric authorization is not canonical")
    trusted_scope = dict(scope_data or {})
    allowed_metrics = trusted_scope.get("allowed_metrics")
    if allowed_metrics is not None and canonical not in {str(item) for item in allowed_metrics}:
        raise ValueError("metric is outside the server-authorized metric scope")
    selected_metric = trusted_scope.get("metric_name")
    if selected_metric and metric.provenance_type == "explorer_selection" and registry.resolve(str(selected_metric)) != canonical:
        raise ValueError("metric does not match the server-validated explorer selection")
    if metric.provenance_type == "semantic_mapping":
        confirmed_ids = {str(item) for item in trusted_scope.get("confirmed_mapping_ids", [])}
        if metric.provenance_id not in confirmed_ids:
            raise ValueError("semantic metric mapping is not a server-confirmed mapping")
    return metric
