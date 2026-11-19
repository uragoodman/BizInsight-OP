"""Deterministic, provenance-preserving evidence projection for Critic prompts."""

from __future__ import annotations

import json
import math
from typing import Any

from app.context.orchestrator import (
    OrchestratedItem,
    Source,
    TrustTier,
    _is_current_evidence,
)


CRITIC_EVIDENCE_SUMMARY_VERSION = "critic-evidence-summary-v1"
_OMITTED_KEYS = frozenset({
    "question", "original_query", "prompt", "instructions", "raw_sql", "sql",
    "request_id", "client_request_id", "gateway_request_id",
})
_QUERY_RESULT_FIELDS = (
    "metric_result", "trend_result", "compare_result", "breakdown_result",
    "multi_dimension_breakdown_result", "contribution_result",
    "statistics_result", "association_result",
)
_PARSED_FIELDS = (
    "query_type", "metric_name", "metric_date", "current_date", "baseline_date",
    "current_start", "current_end", "baseline_start", "baseline_end", "start_date",
    "end_date", "left_date", "right_date", "granularity", "dimension", "dimensions", "limit",
    "statistics", "tool_name", "association_pair_id", "filters", "authorized_metric",
)


def _json_value(value: Any, key: str = "") -> Any:
    """Keep structured facts, removing request text and non-JSON numeric values."""
    if key.casefold() in _OMITTED_KEYS:
        return None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if str(child_key).casefold() in _OMITTED_KEYS:
                continue
            projected = _json_value(child_value, str(child_key))
            if projected is not None:
                result[str(child_key)] = projected
        return result
    if isinstance(value, (list, tuple)):
        return [projected for item in value if (projected := _json_value(item)) is not None]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    return str(value)


def _project_parsed(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    projected = {
        key: _json_value(value[key], key)
        for key in _PARSED_FIELDS
        if key in value and value[key] is not None
    }
    if not projected:
        return None
    authorized = projected.get("authorized_metric")
    if isinstance(authorized, dict):
        projected["authorized_metric"] = {
            key: authorized[key]
            for key in ("metric_id", "definition_version")
            if key in authorized and authorized[key] is not None
        }
    return projected


def _project_evidence_content(row: dict[str, Any]) -> dict[str, Any]:
    content = row.get("content")
    if not isinstance(content, dict):
        return {"content": _json_value(content)}

    source_type = str(row.get("source_type") or "")
    if source_type in {"mcp", "diagnostic_query", "investigation_action"}:
        result = content.get("result")
        projected_result: Any = _json_value(result)
        if isinstance(result, dict):
            envelope: dict[str, Any] = {}
            for key in ("agent", "status"):
                if result.get(key) is not None:
                    envelope[key] = _json_value(result[key], key)
            if result.get("reason") is not None:
                envelope["reason"] = _json_value(result["reason"], "reason")
            parsed = _project_parsed(result.get("parsed"))
            if parsed:
                envelope["parsed"] = parsed
            present_payloads = [key for key in _QUERY_RESULT_FIELDS if key in result and result[key] is not None]
            if present_payloads:
                for key in present_payloads:
                    envelope[key] = _json_value(result[key], key)
            elif not parsed:
                # Unknown tool schemas stay visible in full. If they do not fit,
                # the packer rejects them instead of silently dropping facts.
                envelope = _json_value(result)
            projected_result = envelope
        projected: dict[str, Any] = {}
        for key in ("metric_id", "query_type", "operation"):
            if content.get(key) is not None:
                projected[key] = _json_value(content[key], key)
        if projected_result is not None:
            projected["result"] = projected_result
        return projected or _json_value(content)

    if source_type == "root_cause":
        # The narrative duplicates the answer under review. Keep structured
        # contribution data and its provenance as the authoritative facts.
        projected = {
            key: _json_value(content[key], key)
            for key in ("metric_id", "contribution", "evidence_count")
            if key in content and content[key] is not None
        }
        return projected or _json_value(content)

    return _json_value(content)


def _collect_evidence_ids(value: Any, target: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in {
                "evidence_id", "evidence_ids", "input_evidence_ids",
                "verified_evidence_ids", "source_evidence_ids",
                "contribution_evidence_ids",
            }:
                values = child if isinstance(child, (list, tuple, set)) else [child]
                target.update(str(item) for item in values if isinstance(item, str) and item)
            _collect_evidence_ids(child, target)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_evidence_ids(child, target)


def _project_action(item: Any) -> Any:
    if not isinstance(item, dict):
        return _json_value(item)
    fields = (
        "action_id", "step_id", "kind", "state", "requirement_basis",
        "depends_on", "action_type", "status", "required", "may_be_ignored",
        "target_metric", "metric_id", "dimension", "failure_code",
        "verified_evidence_ids", "evidence_ids", "validation_status",
        "validation_checks", "validation_result",
        "limitations", "stop_reason",
    )
    projected = {
        key: _json_value(item[key], key)
        for key in fields
        if key in item and item[key] is not None and key != "validation_checks"
    }
    checks = item.get("validation_checks")
    if isinstance(checks, list):
        projected["validation_checks"] = [
            {
                key: _json_value(check[key], key)
                for key in ("name", "status", "conflicting_evidence_ids")
                if key in check and check[key] is not None
            }
            for check in checks if isinstance(check, dict)
        ]
    return projected


def _project_decomposition(item: Any) -> Any:
    if not isinstance(item, dict):
        return _json_value(item)
    values = item.get("values") if isinstance(item.get("values"), dict) else {}
    value_fields = (
        "baseline_value", "current_value", "total_delta", "factor_metrics",
        "factor_identities", "input_evidence_ids", "baseline_factor_values",
        "current_factor_values", "display_method", "interaction_delta", "reconciliation_ratio",
        "residual_value", "residual_type", "residual_explainable",
        "residual_tolerance",
    )
    projected_values = {
        key: _json_value(values[key], key)
        for key in value_fields
        if key in values and values[key] is not None
    }
    fields = (
        "evidence_id", "evidence_type", "metric_id", "comparison", "unit",
        "scope_fingerprint", "role",
    )
    result = {
        key: _json_value(item[key], key)
        for key in fields
        if key in item and item[key] is not None
    }
    if projected_values:
        result["values"] = projected_values
    return result


def _project_investigation(context: Any) -> dict[str, Any] | None:
    if not isinstance(context, dict) or not context:
        return None
    fields = (
        "schema_version", "target", "plan_steps", "validated_numeric_decompositions",
        "verified_evidence_ids", "investigation_limitations", "terminal_reason",
        "global_stop", "business_causality", "conflicting_evidence_ids",
        "contribution_ratio_semantics", "operation_intent", "task_type",
        "target_metric_id", "method_id", "scope", "baseline_date", "current_date",
        "decomposition_review_status", "limitations", "stop_reason",
    )
    projection: dict[str, Any] = {"schema_version": CRITIC_EVIDENCE_SUMMARY_VERSION}
    for key in fields:
        if key not in context or context[key] is None:
            continue
        if key == "plan_steps" and isinstance(context[key], list):
            projection[key] = [_project_action(item) for item in context[key]]
        elif key == "validated_numeric_decompositions" and isinstance(context[key], list):
            projection[key] = [_project_decomposition(item) for item in context[key]]
        elif key in {"terminal_reason", "global_stop"} and isinstance(context[key], dict):
            projection[key] = {
                field: _json_value(context[key][field], field)
                for field in (
                    "code", "initiator", "is_terminal", "may_be_ignored",
                    "independent_of_optional_actions", "required_actions_complete",
                )
                if field in context[key] and context[key][field] is not None
            }
        else:
            projection[key] = _json_value(context[key], key)
    actions = context.get("action_results")
    if isinstance(actions, list):
        projection["action_results"] = [_project_action(item) for item in actions]
    # Numeric evidence is represented once below as evidence ledger entries;
    # keeping the duplicate nested copies here caused most of the token growth.
    projection["numeric_evidence_refs"] = sorted({
        str(item.get("evidence_id"))
        for field in ("verified_numeric_evidence", "verified_contribution_evidence")
        for item in (context.get(field) or [])
        if isinstance(item, dict) and item.get("evidence_id")
    })
    return projection


def _claim_ledger(claims: Any) -> tuple[dict[str, Any], list[str], set[str]]:
    entries: list[dict[str, Any]] = []
    claim_ids: list[str] = []
    evidence_ids: set[str] = set()
    for index, claim in enumerate(claims if isinstance(claims, list) else [], start=1):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or f"claim_{index:03d}")
        refs = list(dict.fromkeys(
            str(item) for item in (claim.get("source_evidence_ids") or [])
            if isinstance(item, str) and item
        ))
        evidence_ids.update(refs)
        claim_ids.append(claim_id)
        entries.append({
            "claim_id": claim_id,
            "claim_type": str(claim.get("claim_type") or "unknown"),
            "claim": str(claim.get("value") or ""),
            "evidence_ids": refs,
            "binding_status": "bound" if refs else "unbound",
        })
    return {"claims": entries}, claim_ids, evidence_ids


def _coverage_matrix(
    state: dict[str, Any],
    evidence_facts: list[dict[str, Any]],
    missing_evidence_ids: list[str],
    claim_ids: list[str],
) -> dict[str, Any]:
    metric_ids: set[str] = set()
    dimensions: set[str] = set()
    periods: set[str] = set()
    trends: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    for entry in evidence_facts:
        metric_id = entry.get("metric_id")
        if isinstance(metric_id, str) and metric_id:
            metric_ids.add(metric_id)
        facts = entry.get("facts") or {}
        if not isinstance(facts, dict):
            continue
        result = facts.get("result") if isinstance(facts.get("result"), dict) else facts
        parsed = result.get("parsed") or {}
        if isinstance(parsed, dict):
            for field in ("start_date", "end_date", "baseline_date", "current_date", "left_date", "right_date"):
                if parsed.get(field):
                    periods.add(str(parsed[field]))
            if parsed.get("dimension"):
                dimensions.add(str(parsed["dimension"]))
            dimensions.update(str(item) for item in parsed.get("dimensions") or [] if item)
        for key in _QUERY_RESULT_FIELDS:
            payload = result.get(key)
            if not isinstance(payload, dict):
                continue
            if payload.get("metric_name"):
                metric_ids.add(str(payload["metric_name"]))
            for field in ("start_date", "end_date", "baseline_date", "current_date"):
                if payload.get(field):
                    periods.add(str(payload[field]))
            if payload.get("dimension"):
                dimensions.add(str(payload["dimension"]))
            dimensions.update(str(item) for item in payload.get("dimensions") or [] if item)
            if key == "trend_result":
                trends.append({
                    "evidence_id": entry["evidence_id"],
                    "metric_id": payload.get("metric_name") or entry.get("metric_id"),
                    "start_date": payload.get("start_date"),
                    "end_date": payload.get("end_date"),
                    "point_count": len(payload.get("points") or []),
                })
            if key in {
                "contribution_result", "breakdown_result",
                "multi_dimension_breakdown_result",
            }:
                contributions.append({
                    "evidence_id": entry["evidence_id"],
                    "metric_id": payload.get("metric_name") or entry.get("metric_id"),
                    "dimension": payload.get("dimension"),
                    "dimensions": payload.get("dimensions"),
                    "component_count": len(
                        payload.get("components") or payload.get("current_items") or payload.get("items") or []
                    ),
                })
        facts_values = entry.get("facts") or {}
        contribution = facts_values.get("contribution") if isinstance(facts_values, dict) else None
        if isinstance(contribution, list):
            for item in contribution:
                if isinstance(item, dict) and item.get("dimension"):
                    dimensions.add(str(item["dimension"]))
                    contributions.append({
                        "evidence_id": entry["evidence_id"],
                        "metric_id": entry.get("metric_id"),
                        "dimension": item.get("dimension"),
                        "component_count": len(item.get("components") or []),
                    })

    presentation = state.get("analysis_presentation") or {}
    analysis = state.get("analysis_result") or {}
    completeness = state.get("result_completeness") or {}
    limitations: list[str] = []
    for values in (
        (presentation.get("limitations") if isinstance(presentation, dict) else []),
        (analysis.get("limitations") if isinstance(analysis, dict) else []),
        (completeness.get("missing_dimensions") if isinstance(completeness, dict) else []),
    ):
        if isinstance(values, list):
            limitations.extend(str(value) for value in values if value)
    return {
        "schema_version": CRITIC_EVIDENCE_SUMMARY_VERSION,
        "metrics": sorted(metric_ids),
        "dimensions": sorted(dimensions),
        "periods": sorted(periods),
        "trends": trends,
        "contributions": contributions,
        "missing": list(dict.fromkeys(limitations)),
        "missing_evidence_ids": missing_evidence_ids,
        "claim_ids": claim_ids,
        "result_completeness_status": completeness.get("status") if isinstance(completeness, dict) else None,
    }


def build_critic_evidence_items(
    state: dict[str, Any],
    investigation_context: dict[str, Any] | None,
) -> tuple[list[OrchestratedItem], dict[str, Any]]:
    """Build compact Critic-only items without mutating or replacing source evidence."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in state.get("evidence_bundles") or []:
        if not _is_current_evidence(row):
            continue
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id:
            continue
        if evidence_id in rows_by_id:
            duplicate_count += 1
            continue
        rows_by_id[evidence_id] = row

    claim_payload, claim_ids, claim_evidence_ids = _claim_ledger(state.get("claims"))
    required_ids = set(claim_evidence_ids)
    _collect_evidence_ids(investigation_context, required_ids)
    _collect_evidence_ids(state.get("analysis_presentation"), required_ids)
    _collect_evidence_ids(state.get("result_completeness"), required_ids)
    # Required evidence may itself cite input evidence. Follow those references
    # so a derived fact keeps its supporting ledger in the Critic prompt.
    if not required_ids and rows_by_id:
        required_ids.update(rows_by_id)
    pending = list(required_ids)
    visited: set[str] = set()
    while pending:
        evidence_id = pending.pop()
        if evidence_id in visited:
            continue
        visited.add(evidence_id)
        row = rows_by_id.get(evidence_id)
        if not row:
            continue
        nested: set[str] = set()
        _collect_evidence_ids(row.get("content"), nested)
        pending.extend(nested - required_ids)
        required_ids.update(nested)
    required_ids &= set(rows_by_id)
    missing_ids = sorted(visited - set(rows_by_id))
    for claim in claim_payload["claims"]:
        refs = set(claim["evidence_ids"])
        claim["binding_status"] = (
            "unbound" if not refs else
            "bound" if refs <= set(rows_by_id) else
            "evidence_unavailable"
        )
    evidence_facts: list[dict[str, Any]] = []
    items: list[OrchestratedItem] = []
    scope = str(state.get("current_query_scope_fingerprint") or "")
    version = str(state.get("data_version_manifest_hash") or "current")
    for evidence_id, row in rows_by_id.items():
        facts = _project_evidence_content(row)
        raw_content = row.get("content") if isinstance(row.get("content"), dict) else {}
        raw_result = raw_content.get("result") if isinstance(raw_content, dict) else {}
        raw_parsed = raw_result.get("parsed") if isinstance(raw_result, dict) else {}
        evidence_facts.append({
            "evidence_id": evidence_id,
            "source_type": str(row.get("source_type") or "unknown"),
            "source_ref": str(row.get("source_ref") or ""),
            "scope_fingerprint": str(row.get("scope_fingerprint") or scope),
            "version": str(row.get("version") or version),
            "metric_id": str(
                raw_content.get("metric_id")
                or (raw_result.get("metric_id") if isinstance(raw_result, dict) else "")
                or (raw_result.get("metric_name") if isinstance(raw_result, dict) else "")
                or (raw_parsed.get("metric_name") if isinstance(raw_parsed, dict) else "")
                or ""
            ),
            "facts": facts,
            "created_at": str(row.get("created_at") or row.get("observed_at") or ""),
        })

    coverage = _coverage_matrix(state, evidence_facts, missing_ids, claim_ids)
    refs = set(required_ids)
    for claim in claim_payload["claims"]:
        refs.update(claim["evidence_ids"])
    claim_content = {
        "schema_version": CRITIC_EVIDENCE_SUMMARY_VERSION,
        **claim_payload,
    }
    items.append(OrchestratedItem(
        item_id="critic.claim_evidence",
        source=Source.TOOL_EVIDENCE,
        trust_tier=TrustTier.CURRENT_EVIDENCE,
        content=json.dumps(claim_content, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        source_refs=("answer_claims",),
        scope_fingerprint=scope,
        version=CRITIC_EVIDENCE_SUMMARY_VERSION,
        required=True,
        priority=100,
        evidence_ids=tuple(sorted(refs)),
    ))
    items.append(OrchestratedItem(
        item_id="critic.coverage",
        source=Source.TOOL_EVIDENCE,
        trust_tier=TrustTier.CURRENT_EVIDENCE,
        content=json.dumps(coverage, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        source_refs=("critic_coverage",),
        scope_fingerprint=scope,
        version=CRITIC_EVIDENCE_SUMMARY_VERSION,
        required=True,
        priority=95,
        evidence_ids=tuple(sorted(refs)),
    ))

    projected_investigation = _project_investigation(investigation_context)
    if projected_investigation:
        investigation_ids: set[str] = set()
        _collect_evidence_ids(investigation_context, investigation_ids)
        items.append(OrchestratedItem(
            item_id="evidence.investigation_context",
            source=Source.TOOL_EVIDENCE,
            trust_tier=TrustTier.CURRENT_EVIDENCE,
            content=json.dumps(projected_investigation, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            source_refs=("investigation_context",),
            scope_fingerprint=scope,
            version=CRITIC_EVIDENCE_SUMMARY_VERSION,
            required=True,
            priority=90,
            evidence_ids=tuple(sorted(investigation_ids)),
        ))

    for entry in evidence_facts:
        evidence_id = entry["evidence_id"]
        evidence_ids = (evidence_id,)
        items.append(OrchestratedItem(
            item_id=f"evidence.{evidence_id}",
            source=Source.TOOL_EVIDENCE,
            trust_tier=TrustTier.CURRENT_EVIDENCE,
            content=json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            source_refs=(entry["source_ref"] or evidence_id,),
            scope_fingerprint=entry["scope_fingerprint"],
            version=entry["version"],
            required=evidence_id in required_ids,
            priority=85 if evidence_id in required_ids else 40,
            evidence_ids=evidence_ids,
            created_at=entry["created_at"],
        ))

    metadata = {
        "schema_version": CRITIC_EVIDENCE_SUMMARY_VERSION,
        "required_evidence_ids": sorted(required_ids),
        "optional_evidence_ids": sorted(set(rows_by_id) - required_ids),
        "missing_evidence_ids": missing_ids,
        "claim_count": len(claim_ids),
        "unbound_claim_ids": [
            entry["claim_id"] for entry in claim_payload["claims"] if not entry["evidence_ids"]
        ],
        "unavailable_claim_ids": [
            entry["claim_id"]
            for entry in claim_payload["claims"]
            if entry["binding_status"] == "evidence_unavailable"
        ],
        "evidence_count": len(rows_by_id),
        "deduplicated_evidence_count": duplicate_count,
        "coverage_counts": {
            "metrics": len(coverage["metrics"]),
            "dimensions": len(coverage["dimensions"]),
            "trends": len(coverage["trends"]),
            "contributions": len(coverage["contributions"]),
            "missing": len(coverage["missing"]),
        },
    }
    return items, metadata
