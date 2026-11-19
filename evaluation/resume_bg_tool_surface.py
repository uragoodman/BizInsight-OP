"""Frozen, case-scoped tool surface shared by the formal B and G arms.

This is an evaluation contract, not an authorization grant. Every selected
operation still passes the existing Registry, tenant, permission and MCP
gateway checks at execution time.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


_TOOL_TEMPLATES: dict[str, dict[str, Any]] = {
    "query_value": {
        "description": "Read one registered metric value for one explicit date.",
        "query_type": "metric_value",
        "parameters": {"type": "object", "properties": {"metric_id": {"type": "string"}, "metric_date": {"type": "string", "format": "date"}}, "required": ["metric_id", "metric_date"], "additionalProperties": False},
    },
    "metric_compare": {
        "description": "Compare one registered metric at explicit baseline and current dates; delta is current minus baseline.",
        "query_type": "metric_compare",
        "parameters": {"type": "object", "properties": {"metric_id": {"type": "string"}, "baseline_date": {"type": "string", "format": "date"}, "current_date": {"type": "string", "format": "date"}}, "required": ["metric_id", "baseline_date", "current_date"], "additionalProperties": False},
    },
    "query_trend": {
        "description": "Read registered daily metric observations over an inclusive date range.",
        "query_type": "metric_trend",
        "parameters": {"type": "object", "properties": {"metric_id": {"type": "string"}, "start_date": {"type": "string", "format": "date"}, "end_date": {"type": "string", "format": "date"}}, "required": ["metric_id", "start_date", "end_date"], "additionalProperties": False},
    },
    "registered_numeric_decomposition": {
        "description": "Explain a registered numeric change using registered factors; this is not business-causal proof.",
        "query_type": "registered_numeric_decomposition",
        "parameters": {"type": "object", "properties": {"metric_id": {"type": "string"}, "baseline_date": {"type": "string", "format": "date"}, "current_date": {"type": "string", "format": "date"}}, "required": ["metric_id", "baseline_date", "current_date"], "additionalProperties": False},
    },
    "metric_contribution": {
        "description": "Compare registered statistical contribution by an authorized dimension for two dates.",
        "query_type": "metric_contribution",
        "parameters": {"type": "object", "properties": {"metric_id": {"type": "string"}, "baseline_date": {"type": "string", "format": "date"}, "current_date": {"type": "string", "format": "date"}, "dimension": {"type": "string"}}, "required": ["metric_id", "baseline_date", "current_date", "dimension"], "additionalProperties": False},
    },
    "period_metric_contribution": {
        "description": "Compare registered statistical contribution over two complete natural periods.",
        "query_type": "metric_contribution",
        "parameters": {"type": "object", "properties": {"metric_id": {"type": "string"}, "baseline_start": {"type": "string", "format": "date"}, "baseline_end": {"type": "string", "format": "date"}, "current_start": {"type": "string", "format": "date"}, "current_end": {"type": "string", "format": "date"}, "granularity": {"type": "string", "enum": ["week", "month"]}, "dimension": {"type": "string"}}, "required": ["metric_id", "baseline_start", "baseline_end", "current_start", "current_end", "granularity", "dimension"], "additionalProperties": False},
    },
}

_SAFE_FAILURE_SEMANTICS = {
    "capability_gap": "Return a structured limitation; do not substitute another metric, dimension or operation.",
    "data_insufficient": "Retain the observed missing-data result and stop the affected operation; do not treat missing as zero.",
    "authorization_denied": "Stop before execution and preserve the authorization refusal.",
    "execution_failed": "Preserve the execution failure category; do not report success.",
}


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_evaluation_tool_surface(case: Mapping[str, Any], tenant_context: Any) -> dict[str, Any]:
    """Build a deterministic surface from current Registry capability and IAM.

    The target metric and requested dimension come from the frozen case, but
    allowed operations/dimensions are derived again from Registry and the
    actual test identity. This function does not authorize execution.
    """
    from app.auth.context import build_authorization_context
    from app.metrics.registry import get_registry

    metric_id = str(case.get("target_metric") or "")
    scope = str(case.get("scope") or "platform")
    registry = get_registry()
    definition = registry.get(metric_id)
    implementation = registry.get_implementation(metric_id, scope)
    authz = build_authorization_context(tenant_context, metric_scope_type=scope)
    authz.require_action("analyze")
    authz.require_metrics([metric_id])
    capability = definition.attribution.for_scope(scope)
    dimensions = sorted(set(capability.dimensions) & set(authz.allowed_dimensions))
    requested_dimension = case.get("dimension")
    registered_dimensions = set(capability.dimensions)
    if not requested_dimension:
        requested_dimension_status = None
    elif requested_dimension not in registered_dimensions:
        requested_dimension_status = "capability_gap"
    elif requested_dimension not in set(authz.allowed_dimensions):
        requested_dimension_status = "authorization_denied"
    else:
        requested_dimension_status = "authorized"

    registered_decompositions = _registered_decomposition_contracts(
        metric_id, scope, definition, authz, registry,
    )
    capability_map = {
        "query_value": "value" in implementation.capabilities,
        "metric_compare": "compare" in implementation.capabilities,
        "query_trend": "trend" in implementation.capabilities,
        "registered_numeric_decomposition": bool(registered_decompositions),
        "metric_contribution": bool(capability.dimensions),
        "period_metric_contribution": "period" in implementation.capabilities and bool(capability.dimensions),
    }
    tools: list[dict[str, Any]] = []
    for name, template in _TOOL_TEMPLATES.items():
        if not capability_map[name]:
            continue
        item = {"name": name, **json.loads(json.dumps(template))}
        if name == "metric_contribution" or name == "period_metric_contribution":
            item["parameters"]["properties"]["dimension"]["enum"] = dimensions
        tools.append(item)

    surface = {
        "schema_version": "evaluation-tool-surface-v1",
        "scope": scope,
        "target_metric_id": metric_id,
        "tools": tools,
        "allowed_operation_names": [item["name"] for item in tools],
        "allowed_query_types": sorted({str(item["query_type"]) for item in tools}),
        "authorized_dimensions": dimensions,
        # Numeric decomposition legitimately reads the target's registered
        # factor/source metrics. Keep this allowlist method-bound and derived
        # from the same Registry + identity permissions as the target; it is
        # not a blanket expansion to other metrics.
        "registered_decompositions": registered_decompositions,
        "requested_dimension": requested_dimension,
        "requested_dimension_status": requested_dimension_status,
        "failure_semantics": dict(_SAFE_FAILURE_SEMANTICS),
        "runtime_authorization_required": True,
        "registry_capability_digest": getattr(registry, "catalog_digest", None),
    }
    surface["surface_sha256"] = canonical_json_hash(surface)
    return surface


def _registered_decomposition_contracts(metric_id, scope, definition, authz, registry):
    """Return only executable, identity-authorized Registry method inputs."""
    capability = definition.attribution.for_scope(scope)
    contracts: list[dict[str, Any]] = []
    for method_id in capability.method_ids:
        inputs = {metric_id}
        if method_id == "product_shapley_v1":
            specs = capability.factor_specs.get(method_id, ())
            if len(specs) < 2:
                continue
            for spec in specs:
                inputs.add(str(spec.factor_id))
                inputs.update(str(item) for item in spec.source_metric_ids)
        elif method_id == "symmetric_ratio_v1":
            aggregation = definition.period_aggregation
            if aggregation is None:
                continue
            numerator = registry.get(aggregation.numerator_metric)
            denominator = registry.get(aggregation.denominator_metric)
            if numerator.aggregation != "sum" or denominator.aggregation != "sum":
                continue
            inputs.update((numerator.metric_id, denominator.metric_id))
        else:
            # The evaluation surface only describes methods for which the
            # governed chat workflow has an executable deterministic path.
            continue
        try:
            authz.require_metrics(sorted(inputs))
            for input_metric in inputs:
                registry.get_implementation(input_metric, scope)
        except Exception:  # noqa: BLE001 - unavailable/unauthorized means omit
            continue
        contracts.append({
            "target_metric_id": metric_id,
            "method_id": method_id,
            "input_metric_ids": sorted(inputs),
        })
    return contracts


def validate_evaluation_tool_surface(surface: Mapping[str, Any]) -> bool:
    supplied = dict(surface)
    digest = supplied.pop("surface_sha256", None)
    return bool(digest and digest == canonical_json_hash(supplied))


def surface_prompt_block(surface: Mapping[str, Any]) -> str:
    if not validate_evaluation_tool_surface(surface):
        raise ValueError("EvaluationToolSurface hash is invalid")
    return (
        "\n\n冻结的 B/G 共用 EvaluationToolSurface（完全相同的工具定义、参数和错误语义；"
        "该描述不是授权，服务端仍执行全部权限校验）：\n"
        + json.dumps(dict(surface), ensure_ascii=False, sort_keys=True)
    )


def validate_planned_operation(surface: Mapping[str, Any], *, metric_id: str | None,
                               query_type: str | None, dimension: str | None = None,
                               dimensions: list[str] | None = None,
                               operation_intent: str | None = None,
                               registered_decomposition: Mapping[str, Any] | None = None) -> str | None:
    """Validate a G-side registry plan against the same menu shown to B."""
    if not validate_evaluation_tool_surface(surface):
        return "evaluation_tool_surface_hash_invalid"
    names = set(surface.get("allowed_operation_names") or [])
    if dimensions:
        # The frozen B/G surface does not yet define multi-axis parameters or
        # failure semantics, so a generic breakdown allowlist must not widen
        # into the newly added MCP tool implicitly.
        return "evaluation_tool_surface_multi_dimension_operation_unavailable"
    if operation_intent == "registered_numeric_decomposition":
        if "registered_numeric_decomposition" not in names:
            return "evaluation_tool_surface_operation_unavailable"
        request = dict(registered_decomposition or {})
        contracts = [
            item for item in (surface.get("registered_decompositions") or [])
            if isinstance(item, Mapping)
            and item.get("target_metric_id") == request.get("target_metric_id")
            and item.get("method_id") == request.get("method_id")
        ]
        expected_inputs = set(map(str, contracts[0].get("input_metric_ids") or [])) if len(contracts) == 1 else set()
        actual_inputs = set(map(str, request.get("input_metric_ids") or []))
        if (
            request.get("status") != "ready"
            or request.get("target_metric_id") != surface.get("target_metric_id")
            or not expected_inputs
            or actual_inputs != expected_inputs
        ):
            return "evaluation_tool_surface_decomposition_contract_mismatch"
        if metric_id and metric_id not in expected_inputs:
            return "evaluation_tool_surface_metric_mismatch"
        # The initial planner carries a value-plan for the target; subsequent
        # required trend reads may target only this method's registered inputs.
        if metric_id != request.get("target_metric_id") and query_type != "metric_trend":
            return "evaluation_tool_surface_decomposition_input_operation_mismatch"
        return None
    if metric_id and metric_id != surface.get("target_metric_id"):
        return "evaluation_tool_surface_metric_mismatch"
    mapping = {
        "metric_value": "query_value", "metric_compare": "metric_compare",
        "metric_trend": "query_trend", "metric_contribution": "metric_contribution",
        "metric_breakdown": "metric_contribution", "metric_statistics": "query_trend",
    }
    name = mapping.get(str(query_type or ""))
    if name not in names:
        return "evaluation_tool_surface_operation_unavailable"
    if dimension:
        if dimension not in set(surface.get("authorized_dimensions") or []):
            return "evaluation_tool_surface_dimension_unavailable"
    return None
