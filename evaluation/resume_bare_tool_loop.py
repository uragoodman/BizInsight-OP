"""Minimal B-arm tool loop for a development-only B/G comparison.

The model chooses a registered operation and writes the final answer. Every
fact query crosses ``query_business_data`` with a server-created
``AuthorizedMetric``. Numeric decomposition methods and dependencies are
resolved from the same Registry contract and calculated with the same shared
attribution functions as the governed workflow.
"""
from __future__ import annotations

import json
import copy
import calendar
import hashlib
import math
import re
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from app.agents.contracts import MetricQueryPlan
from app.core.execution_context import activate_state, reset_state
from app.metrics.authorization import authorize_metric, server_exact_mentions
from app.metrics.registry import get_registry
from app.metrics.comparison_units import (
    comparison_table_unit_projection,
    ensure_comparison_unit_note,
)
from app.mcp.client import query_business_data
from app.tenancy import TenantContext, activate_tenant_context


class PublicToolExecutionFailure(RuntimeError):
    """Safe diagnostic wrapper for a failed public query execution."""

    def __init__(
        self, *, tool_name: str | None, request_id: str, plan: MetricQueryPlan,
        cause: Exception | None = None,
        failure_context: Mapping[str, Any] | None = None,
    ):
        super().__init__("Public query execution failed; raw exception text is intentionally withheld.")
        self.tool_name = tool_name
        self.request_id = request_id
        self.plan_summary = {
            "query_type": plan.query_type,
            "metric_name": plan.metric_name,
            "dimension": plan.dimension,
            "baseline_date": plan.baseline_date,
            "current_date": plan.current_date,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
        }
        # The public gateway normally returns a typed execution_failed result
        # instead of raising. Preserve that already-sanitized classification;
        # reducing it to RuntimeError would lose the service and failure code.
        context = dict(failure_context or {})
        self.cause_type = context.get("cause_type") or (type(cause).__name__ if cause else "PublicQueryFailure")
        self.service = context.get("service") or getattr(cause, "service", None)
        self.code = context.get("code") or getattr(cause, "code", None)
        self.failure_category = context.get("category")
        self.failure_context = {
            key: context[key]
            for key in ("category", "service", "code", "cause_type", "request_id", "retryable")
            if key in context
        }
        self.mcp_attempt_count = getattr(cause, "mcp_attempt_count", None)
        self.limitation_code = getattr(cause, "limitation_code", None)


class RegisteredCapabilityRefusal(PermissionError):
    """The request is known but outside the metric's registered capability."""

    limitation_code = "CAPABILITY_GAP"


def _exact_metrics(question: str) -> set[str]:
    catalog = get_registry().planning_catalog(scope="platform", surface="chat")
    return {str(item["metric_id"]) for item in server_exact_mentions(question, catalog=catalog)}


def _comparison_roles(question: str) -> tuple[str, str]:
    from app.context.compiler import _comparison_date_roles

    roles = _comparison_date_roles(question)
    if not roles or len(roles) != 2:
        raise ValueError("B-arm comparison tools require two explicit dates in the user question.")
    baseline, current = str(roles[0]), str(roles[1])
    if baseline == current:
        raise ValueError("B-arm comparison dates must be distinct.")
    return baseline, current


def _trend_range(question: str) -> tuple[str, str, str | None, str | None]:
    """Return query start/end and optional baseline/current roles.

    A single explicit date is a point-in-time trend request, not a comparison.
    Two dates retain their textual task roles; they are never chronologically
    sorted here.
    """
    dates = re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", question)
    unique = list(dict.fromkeys(dates))
    if len(unique) == 1:
        return unique[0], unique[0], None, None
    if len(unique) == 2:
        return unique[0], unique[1], unique[0], unique[1]
    raise ValueError("B-arm trend tools require one or two explicit dates in the user question.")


def _registered_decomposition_contract(metric_id: str, *, scope: str) -> tuple[str, list[str], list[Any]]:
    """Resolve a supported method and its exact inputs from the metric Registry."""
    registry = get_registry()
    definition = registry.get(metric_id)
    capability = definition.attribution.for_scope(scope)
    specs = list(capability.factor_specs.get("product_shapley_v1", ()))
    if "product_shapley_v1" in capability.method_ids and len(specs) == 2:
        sources = sorted({source for spec in specs for source in spec.source_metric_ids})
        return "product_shapley_v1", sources, specs
    period = definition.period_aggregation
    if "symmetric_ratio_v1" in capability.method_ids and period is not None:
        numerator = registry.get(period.numerator_metric)
        denominator = registry.get(period.denominator_metric)
        if numerator.aggregation == "sum" and denominator.aggregation == "sum":
            return "symmetric_ratio_v1", [period.numerator_metric, period.denominator_metric], []
    raise RegisteredCapabilityRefusal(
        definition.attribution.limitation or "No registered numeric decomposition method exists in this scope."
    )


def _adapt_registered_comparison_units(
    final: dict[str, Any], tool_calls: list[dict[str, Any]],
) -> list[str]:
    """Project registered comparison units into the B arm's visible output.

    Only successful public compare results whose metric matches the actual
    tool arguments are eligible. This adapter changes labels, not values,
    dates, evidence, or the model-authored conclusion.
    """
    verified_metrics: list[str] = []
    for call in tool_calls:
        if not isinstance(call, Mapping) or call.get("tool") != "metric_compare":
            continue
        response = call.get("response")
        body = response.get("result") if isinstance(response, Mapping) else None
        compare = body.get("compare_result") if isinstance(body, Mapping) else None
        args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        if not isinstance(compare, Mapping) or compare.get("status") != "ok":
            continue
        metric_id = str(compare.get("metric_name") or "")
        planned_metric_id = str(args.get("metric_id") or args.get("metric_name") or "")
        if not metric_id or metric_id != planned_metric_id:
            continue
        if metric_id not in verified_metrics:
            verified_metrics.append(metric_id)

    if not verified_metrics:
        return []

    operations: list[str] = []
    answer = str(final.get("answer") or "")
    for metric_id in verified_metrics:
        answer, added = ensure_comparison_unit_note(answer, metric_id)
        if added:
            operations.append("registered_comparison_unit_text_supplement")
    final["answer"] = answer

    presentation = final.get("presentation")
    if isinstance(presentation, Mapping):
        projected = dict(presentation)
        for metric_id in verified_metrics:
            projected = comparison_table_unit_projection(projected, metric_id=metric_id)
        final["presentation"] = projected
        if any("columns" in projected or "notes" in projected for _ in verified_metrics):
            operations.append("registered_comparison_unit_table_labels")
    return operations


def _comparison_month_roles(question: str, arguments: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Resolve two explicitly named Chinese year-months and reject model drift."""
    explicit_days = re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", question)
    found = [(int(year), int(month)) for year, month in re.findall(r"(20\d{2})\s*年\s*(1[0-2]|0?[1-9])\s*月", question)]
    unique: list[tuple[int, int]] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    if len(explicit_days) >= 4:
        expected = tuple(explicit_days[:4])
        actual = tuple(str(arguments.get(key) or "") for key in (
            "baseline_start", "baseline_end", "current_start", "current_end",
        ))
        if actual != expected:
            raise ValueError("Period contribution ranges must match the explicit ranges in the original question.")
        return expected
    if len(unique) != 2:
        raise ValueError("Period contribution requires exactly two explicit year-months in the original question.")
    # The user-facing question states baseline first, current second. Preserve
    # those roles; chronological sorting must not reverse them.
    (by, bm), (cy, cm) = unique
    expected = (
        f"{by:04d}-{bm:02d}-01", f"{by:04d}-{bm:02d}-{calendar.monthrange(by, bm)[1]:02d}",
        f"{cy:04d}-{cm:02d}-01", f"{cy:04d}-{cm:02d}-{calendar.monthrange(cy, cm)[1]:02d}",
    )
    actual = tuple(str(arguments.get(key) or "") for key in (
        "baseline_start", "baseline_end", "current_start", "current_end",
    ))
    if actual != expected:
        raise ValueError("Period contribution ranges must match the two complete months named in the question.")
    return expected


def _authorized_plan(
    *, question: str, tool: str, arguments: Mapping[str, Any], context: TenantContext,
    run_id: str, sequence: int,
) -> MetricQueryPlan:
    from datetime import date
    from app.auth.context import build_authorization_context
    from app.mcp.tools import TOOL_REGISTRY

    metric_id = str(arguments.get("metric_id") or "")
    if metric_id not in _exact_metrics(question):
        raise PermissionError("Requested metric is not an exact metric mention in the original question.")
    scope_data = {
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        "permissions": context.role,
        "scope_type": "platform",
    }
    authorized = authorize_metric(
        metric_id,
        provenance_type="exact",
        provenance_id=f"bare-loop:{run_id}:{sequence}",
        scope="platform",
        scope_data=scope_data,
    )
    # Dependencies are not user-selected targets. They are resolved from the
    # Registry below and checked against the same identity's IAM context,
    # without manufacturing a permission from the target mention.
    authz = build_authorization_context(context, metric_scope_type="platform")
    authz.require_action("analyze")
    authz.require_metrics([metric_id])

    if tool == "query_value":
        dates = list(dict.fromkeys(re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", question)))
        if len(dates) != 1 or str(arguments.get("metric_date") or "") != dates[0]:
            raise ValueError("Value date must match the single explicit date in the original question.")
        implementation = get_registry().get_implementation(metric_id, "platform")
        if "value" not in implementation.capabilities:
            raise RegisteredCapabilityRefusal("Metric does not register platform value capability.")
        return MetricQueryPlan(
            query_type="metric_value", metric_name=metric_id,
            authorized_metric=authorized, metric_date=dates[0], tool_name=None,
        )

    if tool == "metric_compare":
        baseline, current = _comparison_roles(question)
        if arguments.get("baseline_date") != baseline or arguments.get("current_date") != current:
            raise ValueError("Compare dates must match the original baseline/current roles.")
        implementation = get_registry().get_implementation(metric_id, "platform")
        if "compare" not in implementation.capabilities:
            raise RegisteredCapabilityRefusal("Metric does not register platform compare capability.")
        authz.require_tool("compare_metric")
        return MetricQueryPlan(
            query_type="metric_compare", metric_name=metric_id,
            authorized_metric=authorized, baseline_date=baseline,
            current_date=current, tool_name="compare_metric",
        )

    if tool == "query_trend":
        start, end, baseline, current = _trend_range(question)
        implementation = get_registry().get_implementation(metric_id, "platform")
        if "trend" not in implementation.capabilities:
            raise PermissionError("Metric does not register platform trend capability.")
        if arguments.get("start_date") != start or arguments.get("end_date") != end:
            raise ValueError("Trend dates must match the explicit date roles in the original question.")
        return MetricQueryPlan(
            query_type="metric_trend", metric_name=metric_id,
            authorized_metric=authorized, start_date=start, end_date=end,
            granularity="day", tool_name="query_metric_trend",
        )

    if tool == "registered_numeric_decomposition":
        baseline, current = _comparison_roles(question)
        method_id, input_metric_ids, factor_specs = _registered_decomposition_contract(metric_id, scope="platform")
        required_metric_ids = {metric_id, *input_metric_ids}
        if method_id == "symmetric_ratio_v1":
            required_metric_ids.update(input_metric_ids)
        else:
            required_metric_ids.update(
                source for spec in factor_specs for source in spec.source_metric_ids
            )
        authz.require_metrics(sorted(required_metric_ids))
        if arguments.get("baseline_date") != baseline or arguments.get("current_date") != current:
            raise ValueError("Decomposition dates must match the original baseline/current roles.")
        implementation = get_registry().get_implementation(metric_id, "platform")
        if "trend" not in implementation.capabilities:
            raise RegisteredCapabilityRefusal("Registered decomposition target has no executable trend input.")
        authz.require_tool("query_metric_trend")
        for dependency in sorted(required_metric_ids - {metric_id}):
            dep_impl = get_registry().get_implementation(dependency, "platform")
            if "trend" not in dep_impl.capabilities:
                raise RegisteredCapabilityRefusal(
                    f"Registered decomposition input {dependency} has no executable trend capability."
                )
        return MetricQueryPlan(
            query_type="metric_trend", metric_name=metric_id,
            authorized_metric=authorized, start_date=baseline, end_date=current,
            granularity="day", tool_name="query_metric_trend",
        )

    if tool == "metric_contribution":
        baseline, current = _comparison_roles(question)
        dimension = str(arguments.get("dimension") or "")
        if arguments.get("baseline_date") != baseline or arguments.get("current_date") != current:
            raise ValueError("Contribution dates must match the original question's baseline/current roles.")
        attribution = get_registry().get(metric_id).attribution.for_scope("platform")
        if dimension not in attribution.dimensions:
            raise RegisteredCapabilityRefusal("Dimension is not in the metric's registered attribution contract.")
        authz.require_dimensions([dimension])
        tool_name = "compare_breakdown_by_dimension"
        authz.require_tool(tool_name)
        if tool_name not in TOOL_REGISTRY:
            raise PermissionError("Registered contribution tool is unavailable.")
        return MetricQueryPlan(
            query_type="metric_contribution", metric_name=metric_id,
            authorized_metric=authorized, current_date=current,
            baseline_date=baseline, dimension=dimension, limit=100,
            tool_name=tool_name,
        )
    if tool == "period_metric_contribution":
        from datetime import date
        from app.mcp.tools import TOOL_REGISTRY

        if str(arguments.get("granularity") or "") != "month":
            raise ValueError("This B-arm period contribution route supports only a registered natural month.")
        baseline_start, baseline_end, current_start, current_end = _comparison_month_roles(question, arguments)
        dimension = str(arguments.get("dimension") or "")
        attribution = get_registry().get(metric_id).attribution.for_scope("platform")
        if dimension not in attribution.dimensions or "symmetric_mix_within_v1" not in attribution.method_ids:
            raise RegisteredCapabilityRefusal("Metric has no registered period contribution contract for this dimension.")
        authz.require_dimensions([dimension])
        if metric_id == "freight_rate" and dimension == "month_position":
            tool_name = "compare_freight_rate_ratio_facts_by_dimension_period"
        elif metric_id == "avg_order_value" and dimension == "month_position":
            tool_name = "compare_avg_order_value_ratio_facts_by_dimension_period"
        else:
            tool_name = "compare_breakdown_by_dimension_period"
        authz.require_tool(tool_name)
        if tool_name not in TOOL_REGISTRY:
            raise RegisteredCapabilityRefusal("Registered period-contribution tool is unavailable.")
        return MetricQueryPlan(
            query_type="metric_contribution", metric_name=metric_id,
            authorized_metric=authorized,
            baseline_start=baseline_start, baseline_end=baseline_end,
            current_start=current_start, current_end=current_end,
            granularity="month", dimension=dimension, limit=10, tool_name=tool_name,
        )
    raise ValueError("Unknown B-arm tool.")


def execute_authorized_public_tool(
    *, question: str, tool: str, arguments: Mapping[str, Any], context: TenantContext,
    run_id: str, sequence: int, request_id: str | None = None,
) -> dict[str, Any]:
    """Execute one model-selected tool through the same public query gateway as G."""
    if tool == "registered_numeric_decomposition":
        return _execute_registered_numeric_decomposition(
            question=question, arguments=arguments, context=context,
            run_id=run_id, sequence=sequence, request_id=request_id,
        )
    plan = _authorized_plan(
        question=question, tool=tool, arguments=arguments,
        context=context, run_id=run_id, sequence=sequence,
    )
    request_id = request_id or f"{run_id}:tool:{sequence}:{uuid4().hex[:10]}"
    try:
        response = query_business_data(
            "metric_query", question,
            tenant_id=context.tenant_id, role=context.role,
            request_id=request_id, idempotency_key=request_id,
            plan=plan,
        )
    except Exception as exc:
        raise PublicToolExecutionFailure(
            tool_name=plan.tool_name, request_id=request_id, plan=plan, cause=exc,
        ) from exc
    result = response.get("result") or {}
    if result.get("status") not in {"ok", "not_found", "insufficient_data"}:
        failure_context = result.get("failure_context")
        if result.get("status") == "execution_failed" and isinstance(failure_context, Mapping):
            raise PublicToolExecutionFailure(
                tool_name=plan.tool_name, request_id=request_id, plan=plan,
                failure_context=failure_context,
            )
        raise RuntimeError(f"Public query gateway refused B-arm tool: {result.get('reason') or result.get('status')}")
    return response


def _execute_registered_numeric_decomposition(
    *, question: str, arguments: Mapping[str, Any], context: TenantContext,
    run_id: str, sequence: int, request_id: str | None,
) -> dict[str, Any]:
    """Query registered sufficient statistics through the authorized gateway,
    then run the shared deterministic method and reconciliation checks.
    """
    from app.analysis.attribution_methods import (
        product_shapley_attribution, resolve_product_factor_values,
        symmetric_ratio_attribution, bind_registered_ratio_factor_identities,
    )
    from app.auth.context import build_authorization_context
    from app.metrics.authorization import authorize_metric

    metric_id = str(arguments.get("metric_id") or "")
    plan = _authorized_plan(
        question=question, tool="registered_numeric_decomposition", arguments=arguments,
        context=context, run_id=run_id, sequence=sequence,
    )
    baseline, current = _comparison_roles(question)
    method_id, source_ids, factor_specs = _registered_decomposition_contract(metric_id, scope="platform")
    required_ids = {metric_id, *source_ids}
    if method_id == "product_shapley_v1":
        required_ids.update(source for spec in factor_specs for source in spec.source_metric_ids)
    authz = build_authorization_context(context, metric_scope_type="platform")
    authz.require_action("analyze")
    authz.require_metrics(sorted(required_ids))
    authz.require_tool("query_metric_trend")

    scope_data = {
        "tenant_id": context.tenant_id, "user_id": context.user_id,
        "permissions": context.role, "scope_type": "platform",
    }
    child_envelopes: dict[str, dict[str, Any]] = {}
    child_ids: dict[str, str] = {}
    for dependency in sorted(required_ids):
        dep_plan = plan.model_copy(update={
            "metric_name": dependency,
            "authorized_metric": (
                plan.authorized_metric if dependency == metric_id else authorize_metric(
                    dependency, provenance_type="registered_relation",
                    provenance_id=f"registry-relation:{metric_id}:{dependency}",
                    scope="platform", scope_data=scope_data,
                )
            ),
            "tool_name": "query_metric_trend",
        })
        child_request_id = f"{request_id or f'{run_id}:tool:{sequence}'}:input:{dependency}"
        try:
            envelope = query_business_data(
                "metric_query", question, tenant_id=context.tenant_id, role=context.role,
                request_id=child_request_id, idempotency_key=child_request_id, plan=dep_plan,
            )
        except Exception as exc:
            raise PublicToolExecutionFailure(
                tool_name="query_metric_trend", request_id=child_request_id,
                plan=dep_plan, cause=exc,
            ) from exc
        result = envelope.get("result") or {}
        if result.get("status") != "ok":
            raise RegisteredDataInsufficient(
                f"Registered numeric decomposition input {dependency} returned {result.get('status') or 'unknown'}."
            )
        trend = result.get("trend_result") or {}
        points = trend.get("points") or []
        by_day: dict[str, float] = {}
        for point in points:
            if not isinstance(point, Mapping):
                continue
            bucket = str(point.get("bucket") or point.get("date") or "")
            value = point.get("value")
            if bucket and value is not None:
                number = float(value)
                if not math.isfinite(number) or bucket in by_day:
                    raise ValueError("Trend evidence has a non-finite or duplicate date bucket.")
                by_day[bucket] = number
        if baseline not in by_day or current not in by_day:
            raise RegisteredDataInsufficient(
                f"Registered numeric decomposition input {dependency} does not cover both requested dates."
            )
        child_envelopes[dependency] = envelope
        child_ids[dependency] = child_request_id

    registry = get_registry()
    target_definition = registry.get(metric_id)
    target_values = {
        metric: {
            str(point.get("bucket") or point.get("date")): float(point["value"])
            for point in ((envelope.get("result") or {}).get("trend_result") or {}).get("points", [])
            if isinstance(point, Mapping) and point.get("value") is not None
        }
        for metric, envelope in child_envelopes.items()
    }
    if method_id == "product_shapley_v1":
        totals = {
            metric: (values[baseline], values[current])
            for metric, values in target_values.items() if baseline in values and current in values
        }
        baseline_factors, current_factors = resolve_product_factor_values(factor_specs, totals)
        attribution = product_shapley_attribution(
            metric_id=metric_id, baseline_factors=baseline_factors,
            current_factors=current_factors, unit=target_definition.unit,
        )
        if not (
            math.isclose(attribution.baseline_value, target_values[metric_id][baseline], rel_tol=1e-8, abs_tol=0.03)
            and math.isclose(attribution.current_value, target_values[metric_id][current], rel_tol=1e-8, abs_tol=0.03)
        ):
            raise ValueError("Registered product factors do not reconcile to the target metric trend.")
    else:
        aggregation = target_definition.period_aggregation
        if aggregation is None:
            raise RegisteredCapabilityRefusal("Ratio decomposition contract is incomplete.")
        numerator = target_values[aggregation.numerator_metric]
        denominator = target_values[aggregation.denominator_metric]
        attribution = symmetric_ratio_attribution(
            metric_id=metric_id,
            baseline_numerator=numerator[baseline], current_numerator=numerator[current],
            baseline_denominator=denominator[baseline], current_denominator=denominator[current],
            multiplier=aggregation.multiplier, unit=target_definition.unit,
        )
        attribution = bind_registered_ratio_factor_identities(attribution, scope="platform")
        target_series = target_values.get(metric_id, {})
        for day, expected in ((baseline, attribution.baseline_value), (current, attribution.current_value)):
            if day in target_series and not math.isclose(target_series[day], expected, rel_tol=1e-8, abs_tol=0.03):
                raise ValueError("Registered ratio dependencies do not reconcile to the target metric trend.")

    request_id = request_id or f"{run_id}:tool:{sequence}:decomposition"
    source_evidence_ids = [f"{request_id}:source:{metric}" for metric in sorted(child_envelopes)]
    attribution = attribution.model_copy(update={"evidence_ids": source_evidence_ids})
    return {
        "task_type": "metric_query", "question": question,
        "server": "registry-numeric-decomposition", "tool_name": "registered_numeric_decomposition",
        "arguments": _safe_tool_arguments(arguments), "request_id": request_id,
        "idempotency_key": request_id, "data": [],
        "source_queries": [
            {"metric_id": metric, "request_id": child_ids[metric], "evidence_id": f"{request_id}:source:{metric}",
             "envelope": child_envelopes[metric]}
            for metric in sorted(child_envelopes)
        ],
        "result": {
            "status": "ok", "parsed": plan.model_dump(mode="json"),
            "operation_intent": "registered_numeric_decomposition",
            "target_metric_id": metric_id, "baseline_date": baseline, "current_date": current,
            "method_id": method_id, "metric_unit": target_definition.unit,
            "decomposition_result": attribution.model_dump(mode="json"),
            "source_evidence_ids": source_evidence_ids,
        },
    }


class RegisteredDataInsufficient(RuntimeError):
    limitation_code = "DATA_INSUFFICIENT"


def _safe_tool_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only non-sensitive, typed fields needed to diagnose a tool call."""
    allowed = {"metric_id", "metric_date", "start_date", "end_date", "baseline_date", "current_date", "dimension", "limit",
               "baseline_start", "baseline_end", "current_start", "current_end", "granularity"}
    result: dict[str, Any] = {}
    for key in allowed:
        value = arguments.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            result[key] = value[:80] if isinstance(value, str) else value
    return result


def _registered_decomposition_card(response: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a fail-closed display card from a successful registered result."""
    from app.metrics.registry import get_registry

    result = response.get("result") or {}
    if (
        result.get("status") != "ok"
        or result.get("operation_intent") != "registered_numeric_decomposition"
    ):
        return None
    parsed = result.get("parsed") or {}
    metric_id = str(result.get("target_metric_id") or "")
    if not metric_id or str(parsed.get("metric_name") or "") != metric_id:
        raise ValueError("registered decomposition target does not match its authorized plan")
    baseline_date = str(result.get("baseline_date") or "")
    current_date = str(result.get("current_date") or "")
    parsed_baseline = str(parsed.get("baseline_date") or "")
    parsed_current = str(parsed.get("current_date") or "")
    parsed_start = str(parsed.get("start_date") or "")
    parsed_end = str(parsed.get("end_date") or "")
    if (
        not baseline_date or not current_date
        or (parsed_baseline and parsed_baseline != baseline_date)
        or (parsed_current and parsed_current != current_date)
        or (
            parsed_start and parsed_end
            and {parsed_start, parsed_end} != {baseline_date, current_date}
        )
    ):
        raise ValueError("registered decomposition dates do not match the authorized plan")
    authorized = parsed.get("authorized_metric") or {}
    if str(authorized.get("metric_id") or "") != metric_id:
        raise ValueError("registered decomposition plan is missing its target authorization")
    registry = get_registry()
    definition = registry.get(metric_id)
    capability = definition.attribution.for_scope("platform")
    method_id = str(result.get("method_id") or "")
    if method_id not in capability.method_ids:
        raise ValueError("registered decomposition method is unavailable for this target and scope")
    from app.investigation.contracts import AttributionResult

    decomposition = AttributionResult.model_validate(result.get("decomposition_result") or {})
    if decomposition.metric_id != metric_id or decomposition.method_id != method_id:
        raise ValueError("registered decomposition result identity conflicts with its plan")
    if str(result.get("metric_unit") or "") != definition.unit:
        raise ValueError("registered decomposition unit conflicts with the metric registry")
    components = {item.component_id: item for item in decomposition.display_components}
    metric_unit = str(result.get("metric_unit") or definition.unit)
    # Decomposition components explain an absolute change in the target metric.
    # For a percent-valued target, that unit is percentage points, even when
    # the low-level formula payload labels the target level itself as "%".
    change_unit = "percentage_points" if metric_unit == "%" else metric_unit
    factors: list[dict[str, Any]] = []
    if method_id == "symmetric_ratio_v1":
        from app.analysis.attribution_methods import bind_registered_ratio_factor_identities

        expected = bind_registered_ratio_factor_identities(decomposition, scope="platform")
        # Rendering requires the producer's explicit mapping. Do not silently
        # reconstruct a missing map at this consumer boundary.
        if not decomposition.factor_identities:
            raise ValueError("registered ratio result is missing factor identity metadata")
        if {
            item.role_id: item.metric_id for item in decomposition.factor_identities
        } != {item.role_id: item.metric_id for item in expected.factor_identities}:
            raise ValueError("registered ratio result factor mapping conflicts with the registry")
        identities = {item.role_id: item for item in decomposition.factor_identities}
        for role_id in ("numerator_effect", "denominator_effect"):
            component = components.get(role_id)
            identity = identities.get(role_id)
            if component is None or identity is None:
                raise ValueError("registered ratio result does not contain both mapped effect roles")
            registered_factor = registry.get(identity.metric_id)
            if identity.display_name != (registered_factor.display_name or identity.metric_id):
                raise ValueError("registered ratio display name conflicts with the metric registry")
            factors.append({
                "metric_id": identity.metric_id,
                "display_name": identity.display_name,
                "contribution": component.value,
                "unit": change_unit,
                "role_id": role_id,
            })
    elif method_id == "product_shapley_v1":
        specs = list(capability.factor_specs.get(method_id, ()))
        registered_ids = [item.factor_id for item in specs]
        if len(registered_ids) != 2 or set(components) != set(registered_ids):
            raise ValueError("registered product result does not match its factor contract")
        for factor_id in registered_ids:
            factor_definition = registry.get(factor_id)
            component = components[factor_id]
            factors.append({
                "metric_id": factor_id,
                "display_name": factor_definition.display_name or factor_id,
                "contribution": component.value,
                "unit": change_unit,
                "role_id": factor_id,
            })
    else:
        raise ValueError("no deterministic display contract exists for this decomposition method")
    if set(decomposition.evidence_ids) != set(result.get("source_evidence_ids") or []):
        raise ValueError("registered decomposition evidence references do not match source evidence")
    return {
        "metric_id": metric_id,
        "display_name": definition.display_name or metric_id,
        "unit": metric_unit,
        "metric_unit": metric_unit,
        "change_unit": change_unit,
        "component_unit": change_unit,
        "baseline_value": decomposition.baseline_value,
        "current_value": decomposition.current_value,
        "delta_value": decomposition.total_delta,
        "baseline_date": baseline_date,
        "current_date": current_date,
        "method_id": method_id,
        "factors": factors,
        "evidence_ids": list(decomposition.evidence_ids),
        "analysis_nature": "registered_numeric_decomposition",
        "business_cause_confirmed": False,
    }


def _registered_decomposition_text(card: Mapping[str, Any]) -> str:
    """Deterministic user-facing supplement from the validated Registry card."""
    metric_name = str(card.get("display_name") or card.get("metric_id") or "目标指标")
    factors = [item for item in card.get("factors", []) if isinstance(item, Mapping)]
    rendered = []
    for item in factors:
        factor_name = str(item.get("display_name") or item.get("metric_id") or "")
        value = item.get("contribution")
        if not factor_name or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        unit = str(item.get("unit") or "").strip()
        rendered.append(f"{factor_name} {float(value):+.2f}{(' ' + unit) if unit else ''}")
    if not rendered:
        return ""
    return (
        f"已验证的注册数值分解（{metric_name}）：" + "；".join(rendered)
        + "。该分解仅解释数值变化，不确认业务原因。"
    )


def _safe_tool_error(exc: Exception) -> dict[str, str]:
    """Record an error class and known public service code, never raw text."""
    detail = {"exception_type": type(exc).__name__}
    limitation_code = getattr(exc, "limitation_code", None)
    if isinstance(limitation_code, str) and limitation_code in {
        "CAPABILITY_GAP", "AUTHORIZATION_BLOCKED", "DATA_INSUFFICIENT",
    }:
        detail["limitation_code"] = limitation_code
    service = getattr(exc, "service", None)
    code = getattr(exc, "code", None)
    if isinstance(service, str) and service in {"mcp", "postgresql"}:
        detail["service"] = service
    safe_failure_codes = {
        "timeout", "circuit_open", "external_call_failed", "connection_error",
        "tool_timeout", "database_timeout", "database_execution_failed",
        "capability_not_supported", "dimension_not_allowed", "permission_denied",
        "execution_failed",
    }
    if isinstance(code, str) and code in safe_failure_codes:
        detail["service_error_code"] = code
    failure_category = getattr(exc, "failure_category", None)
    if isinstance(failure_category, str) and failure_category in {
        "tool_timeout", "database_error", "capability_gap", "authorization_denied", "execution_error",
    }:
        detail["failure_category"] = failure_category
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        detail["cause_type"] = type(cause).__name__
        chain = []
        seen: set[int] = set()
        current = cause
        while current is not None and id(current) not in seen and len(chain) < 6:
            seen.add(id(current))
            item = {"type": type(current).__name__}
            service = getattr(current, "service", None)
            code = getattr(current, "code", None)
            if isinstance(service, str) and service in {"mcp", "postgresql"}:
                item["service"] = service
            if isinstance(code, str) and code in {
                "timeout", "circuit_open", "external_call_failed", "connection_error",
            }:
                item["code"] = code
            chain.append(item)
            current = current.__cause__
        detail["safe_cause_chain"] = chain
    attempt_count = getattr(exc, "mcp_attempt_count", None)
    if isinstance(attempt_count, int) and attempt_count >= 0:
        detail["mcp_attempt_count"] = attempt_count
    resolved_tool = getattr(exc, "tool_name", None)
    if isinstance(resolved_tool, str) and resolved_tool in {
        "query_metric_trend", "compare_breakdown_by_dimension",
        "compare_breakdown_by_dimension_period",
        "compare_freight_rate_ratio_facts_by_dimension_period",
        "compare_avg_order_value_ratio_facts_by_dimension_period",
    }:
        detail["resolved_tool_name"] = resolved_tool
    safe_cause_type = getattr(exc, "cause_type", None)
    if isinstance(safe_cause_type, str):
        detail["cause_type"] = safe_cause_type
    return detail


_SYSTEM_PROMPT = """你是一个基础工具调用循环中的经营数据分析助手。只根据原问题与工具结果回答，不补造数值或原因。
可用工具：
1. query_trend(metric_id, start_date, end_date)：查询一日观测或基线日到当前日的趋势。
2. registered_numeric_decomposition(metric_id, baseline_date, current_date)：按注册契约查询目标和因子，并执行已注册数值分解；只解释数值，不证明业务因果。比率分解请使用结果的 factor_identities，将 numerator_effect / denominator_effect 标为注册指标名称；不得只显示角色名，也不得按数值或顺序猜测。
3. metric_contribution(metric_id, baseline_date, current_date, dimension)：查询注册维度上的单日对单日统计贡献。
4. period_metric_contribution(metric_id, baseline_start, baseline_end, current_start, current_end, granularity, dimension)：查询注册维度上的完整自然周/月贡献。
先按需要调用工具，再返回FINAL。不得把统计贡献称为业务因果。
所有工具参数必须沿用问题中的目标指标、日期角色和维度。若服务端返回能力或数据限制，不要换指标或维度。
仅输出JSON：
工具调用：{"action":"TOOL_CALL","tool":"query_value|metric_compare|query_trend|registered_numeric_decomposition|metric_contribution|period_metric_contribution","arguments":{...}}
最终答案：{"action":"FINAL","answer":"...","claims":[{"text":"...","evidence_ids":["..."]}],"presentation":{"metric_id":"...","baseline_date":"...","current_date":"...","baseline_value":0,"current_value":0,"delta":0,"unit":"...","dimension":"...","items":[{"key":"...","delta":0,"evidence_ids":["..."]}]}}
"""


def _evaluation_system_prompt(surface: Mapping[str, Any]) -> str:
    """Build B's model-visible menu from the exact shared evaluation surface."""
    from evaluation.resume_bg_tool_surface import surface_prompt_block, validate_evaluation_tool_surface

    if not validate_evaluation_tool_surface(surface):
        raise ValueError("B-arm EvaluationToolSurface hash is invalid")
    return (
        "你是一个公开数据评测中的经营分析助手。只根据原问题与工具结果回答，不补造数值或原因。\n"
        "本次只能选择下方冻结的 EvaluationToolSurface 中列出的操作，遵守其参数 schema 与错误语义；"
        "服务端仍独立执行指标注册、身份授权和参数校验。能力／数据限制是终止结果，不能换指标、维度或操作。\n"
        "数值分解和统计贡献只解释注册数值关系，不证明业务因果。先按需要调用工具，再返回 FINAL。\n"
        "仅输出 JSON：工具调用：{\"action\":\"TOOL_CALL\",\"tool\":\"<surface 中的 name>\",\"arguments\":{...}}；"
        "最终答案：{\"action\":\"FINAL\",\"answer\":\"...\",\"claims\":[{\"text\":\"...\",\"evidence_ids\":[...]}],"
        "\"presentation\":{...}}。不得将统计贡献称为业务因果。\n"
        + surface_prompt_block(surface)
    )


def _capture_model_evidence_references(
    model_final: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Record, without enriching, the evidence IDs the B model actually emitted.

    This is observability for the B/G evaluation, not a claim-binding feature:
    it neither invents IDs nor changes the model final, the displayed answer,
    tool selection, or stopping behavior.  The governing arm remains the only
    arm that creates verified claim-evidence bindings.
    """
    def references(value: Any) -> list[str]:
        found: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"evidence_ids", "source_evidence_ids"} and isinstance(child, list):
                    found.extend(str(item) for item in child if item)
                elif key == "evidence_id" and isinstance(child, str) and child:
                    found.append(child)
                else:
                    found.extend(references(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(references(child))
        return found

    authored = model_final if isinstance(model_final, Mapping) else {}
    claims = authored.get("claims")
    claim_rows: list[dict[str, Any]] = []
    if isinstance(claims, list):
        for index, claim in enumerate(claims):
            row: dict[str, Any] = {
                "claim_index": index,
                "claim_is_mapping": isinstance(claim, Mapping),
                "evidence_ids_field_present": isinstance(claim, Mapping) and "evidence_ids" in claim,
                "evidence_ids": [],
            }
            if isinstance(claim, Mapping):
                raw_ids = claim.get("evidence_ids")
                if isinstance(raw_ids, list):
                    row["evidence_ids"] = [str(item) for item in raw_ids if item]
                text = claim.get("text")
                if text is not None:
                    # The final itself is already retained.  A digest makes it
                    # possible to link this row to that unmodified final without
                    # duplicating user-facing text in a second trace field.
                    row["claim_text_sha256"] = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            claim_rows.append(row)

    presentation = authored.get("presentation")
    return {
        "schema_version": "bare-model-evidence-reference-capture-v1",
        "source": "model_authored_final" if model_final is not None else "deterministic_terminal",
        "model_final_present": isinstance(model_final, Mapping),
        "claims_field_present": isinstance(claims, list),
        "claims": claim_rows,
        "presentation_field_present": isinstance(presentation, Mapping),
        "presentation_evidence_ids": references(presentation),
    }


def run_bare_tool_loop(
    question: str,
    *,
    tenant_context: TenantContext,
    run_id: str,
    max_rounds: int = 8,
    max_tool_calls: int = 8,
    max_latency_ms: int = 240_000,
    max_task_tokens: int = 60_000,
    max_cost_cny: float = 5.0,
    llm_call_config: Any | None = None,
    model_fn: Callable[[str, str], Any] | None = None,
    tool_fn: Callable[..., dict[str, Any]] = execute_authorized_public_tool,
    evaluation_tool_surface: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an ungoverned tool-selection loop with the same guarded query tools."""
    from app.llm.client import llm_extract
    from app.metrics.registry import get_registry

    exact = sorted(_exact_metrics(question))
    if not exact:
        raise ValueError("The B-arm tool loop requires an exact registered metric mention.")
    definition = get_registry().get(exact[0])
    attribution = definition.attribution.for_scope("platform")
    from app.auth.context import build_authorization_context
    tool_authz = build_authorization_context(
        tenant_context, metric_scope_type="platform",
    )
    tool_authz.require_action("analyze")
    tool_authz.require_metrics([exact[0]])
    available_dimensions = set(attribution.dimensions) & set(tool_authz.allowed_dimensions)
    try:
        trend_start, trend_end, baseline_date, current_date = _trend_range(question)
    except ValueError:
        trend_start = trend_end = baseline_date = current_date = "<explicit date from question>"
    tools: list[dict[str, Any]] = [
        {"tool": "query_value", "arguments": {"metric_id": exact[0], "metric_date": "<one explicit date>"}},
        {"tool": "metric_compare", "arguments": {"metric_id": exact[0], "baseline_date": "<explicit baseline>", "current_date": "<explicit current>"}},
        {"tool": "query_trend", "arguments": {"metric_id": exact[0], "start_date": trend_start, "end_date": trend_end}},
    ]
    try:
        _method_id, _inputs, _specs = _registered_decomposition_contract(exact[0], scope="platform")
        if baseline_date and current_date:
            tools.append({"tool": "registered_numeric_decomposition", "arguments": {
                "metric_id": exact[0], "baseline_date": baseline_date, "current_date": current_date,
            }})
    except RegisteredCapabilityRefusal:
        pass
    for dimension in sorted(available_dimensions):
        tools.append({"tool": "metric_contribution", "arguments": {
            "metric_id": exact[0], "baseline_date": "<explicit baseline date>",
            "current_date": "<explicit current date>", "dimension": dimension,
        }})
    if "symmetric_mix_within_v1" in attribution.method_ids:
        for dimension in sorted(available_dimensions):
            tools.append({"tool": "period_metric_contribution", "arguments": {
                "metric_id": exact[0], "baseline_start": "<complete baseline natural month>",
                "baseline_end": "<complete baseline natural month>", "current_start": "<complete current natural month>",
                "current_end": "<complete current natural month>", "granularity": "month", "dimension": dimension,
            }})
    if isinstance(max_tool_calls, bool) or int(max_tool_calls) < 0:
        raise ValueError("max_tool_calls must be a non-negative integer")
    if isinstance(max_latency_ms, bool) or int(max_latency_ms) <= 0:
        raise ValueError("max_latency_ms must be a positive integer")
    if isinstance(max_task_tokens, bool) or int(max_task_tokens) <= 0:
        raise ValueError("max_task_tokens must be a positive integer")
    if isinstance(max_cost_cny, bool) or float(max_cost_cny) <= 0:
        raise ValueError("max_cost_cny must be positive")
    started_monotonic = time.monotonic()
    deadline_seconds = int(max_latency_ms) / 1000.0
    state: dict[str, Any] = {
        "trace_id": run_id,
        "current_agent": "bare_tool_loop",
        "user_message": question,
        "tenant_id": tenant_context.tenant_id,
        "user_id": tenant_context.user_id,
        "user_role": tenant_context.role,
        "scope_type": tenant_context.scope_type,
        "llm_calls": [], "errors": [], "llm_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0},
        "runtime_budget": {
            "max_tokens": int(max_task_tokens), "max_cost": float(max_cost_cny),
            "max_latency_ms": int(max_latency_ms), "max_tool_calls": int(max_tool_calls),
        },
        "tool_calls": [], "evidence_bundles": [], "mcp_call_attempts": [],
    }
    if evaluation_tool_surface is not None:
        state["evaluation_tool_surface"] = copy.deepcopy(dict(evaluation_tool_surface))
    from app.core.config import get_settings
    from app.llm.call_config import LLMCallConfig
    call_config = llm_call_config or LLMCallConfig.provider_ab_experiment(
        get_settings(), provider_id="primary", max_completion_tokens=4096,
    )
    if not isinstance(call_config, LLMCallConfig):
        raise ValueError("B-loop requires the same validated internal LLM call configuration as G.")
    state["llm_call_config"] = call_config.to_state()
    turns: list[dict[str, Any]] = []
    seen: set[str] = set()
    final: dict[str, Any] | None = None
    model_authored_final: dict[str, Any] | None = None
    output_adaptations: list[str] = []
    allowed_evaluation_tools: set[str] | None = None
    if evaluation_tool_surface is not None:
        from evaluation.resume_bg_tool_surface import validate_evaluation_tool_surface
        if not validate_evaluation_tool_surface(evaluation_tool_surface):
            raise ValueError("B-arm EvaluationToolSurface hash is invalid")
        if evaluation_tool_surface.get("target_metric_id") != exact[0]:
            raise ValueError("B-arm EvaluationToolSurface target metric mismatch")
        tools = [
            {"tool": row["name"], "description": row["description"], "parameters": row["parameters"]}
            for row in evaluation_tool_surface.get("tools", [])
        ]
        allowed_evaluation_tools = set(evaluation_tool_surface.get("allowed_operation_names") or [])
    user = json.dumps({"question": question, "available_tools": tools}, ensure_ascii=False)
    invoke_model = model_fn or llm_extract
    system_prompt = (
        _evaluation_system_prompt(evaluation_tool_surface)
        if evaluation_tool_surface is not None else _SYSTEM_PROMPT
    )
    token = activate_state(state)
    try:
        with activate_tenant_context(tenant_context):
            for index in range(1, max_rounds + 1):
                if time.monotonic() - started_monotonic >= deadline_seconds:
                    state["terminal_code"] = "TIMEOUT"
                    state["presentation"] = {"status": "partial", "limitation_code": "TIMEOUT"}
                    state.setdefault("errors", []).append({"code": "workflow_deadline_exceeded", "boundary": "before_model_call"})
                    break
                action = invoke_model(system_prompt, user)
                if time.monotonic() - started_monotonic >= deadline_seconds:
                    state["terminal_code"] = "TIMEOUT"
                    state["presentation"] = {"status": "partial", "limitation_code": "TIMEOUT"}
                    state.setdefault("errors", []).append({"code": "workflow_deadline_exceeded", "boundary": "after_model_call"})
                    break
                if isinstance(action, str):
                    action = json.loads(action)
                if not isinstance(action, dict):
                    # llm_extract deliberately returns None for provider
                    # failures. Do not misreport that transport failure as a
                    # malformed model action; retain the provider exception
                    # from this run's actual logical call record.
                    latest_call = next((
                        item for item in reversed(state.get("llm_calls") or [])
                        if item.get("trace_id") == run_id
                    ), None)
                    if isinstance(latest_call, Mapping) and latest_call.get("outcome") == "provider_error":
                        attempts = latest_call.get("attempts") or []
                        latest_attempt = attempts[-1] if attempts and isinstance(attempts[-1], Mapping) else {}
                        exception_type = latest_attempt.get("exception_type")
                        failure_kind = "timeout" if exception_type == "APITimeoutError" else "provider_error"
                        state["terminal_code"] = "EXECUTION_FAILED"
                        state["presentation"] = {
                            "status": "failed", "limitation_code": "EXECUTION_FAILED",
                            "failure_kind": failure_kind,
                        }
                        state.setdefault("errors", []).append({
                            "code": "llm_provider_error",
                            "failure_kind": failure_kind,
                            "exception_type": exception_type,
                            "operation": latest_call.get("operation"),
                            "call_id": latest_call.get("call_id"),
                            "attempt_id": latest_attempt.get("attempt_id"),
                        })
                        # A transport failure after successful tool calls is a
                        # failed task, but it still needs a visible terminal
                        # response. Do not turn retained evidence into an
                        # analysis claim or mark the run completed.
                        final = {
                            "action": "FINAL",
                            "answer": "模型请求超时，分析未完成；此前成功执行的查询结果已保留，但未形成最终结论。"
                            if failure_kind == "timeout"
                            else "模型服务调用失败，分析未完成；此前成功执行的查询结果已保留，但未形成最终结论。",
                            "claims": [],
                            "presentation": state["presentation"],
                        }
                    elif isinstance(latest_call, Mapping) and latest_call.get("outcome") == "empty_response":
                        attempts = latest_call.get("attempts") or []
                        latest_attempt = attempts[-1] if attempts and isinstance(attempts[-1], Mapping) else {}
                        state["terminal_code"] = "EXECUTION_FAILED"
                        state["presentation"] = {
                            "status": "failed",
                            "limitation_code": "EXECUTION_FAILED",
                            "failure_kind": "empty_model_response",
                        }
                        state.setdefault("errors", []).append({
                            "code": "llm_empty_response",
                            "failure_kind": "empty_model_response",
                            "operation": latest_call.get("operation"),
                            "call_id": latest_call.get("call_id"),
                            "attempt_id": latest_attempt.get("attempt_id"),
                            "sdk_request_sent": latest_attempt.get("sdk_request_sent"),
                            "usage_status": latest_attempt.get("usage_status"),
                        })
                        # Keep a visible, deterministic failure notice while
                        # retaining all completed tool evidence. This is not a
                        # substitute analysis and must never score as success.
                        final = {
                            "action": "FINAL",
                            "answer": "数据查询已执行，但模型未返回可用的最终回答；本次分析未完成。",
                            "claims": [],
                            "presentation": state["presentation"],
                        }
                    else:
                        state.setdefault("errors", []).append({"code": "invalid_baseline_action"})
                    break
                if action.get("action") == "FINAL":
                    # Preserve the exact model-authored object before any registered
                    # display adapter mutates the user-facing result below.
                    model_authored_final = copy.deepcopy(action)
                    final = copy.deepcopy(action)
                    break
                if action.get("action") != "TOOL_CALL" or action.get("tool") not in {
                    "query_value", "metric_compare", "query_trend", "registered_numeric_decomposition",
                    "metric_contribution", "period_metric_contribution",
                }:
                    state.setdefault("errors", []).append({"code": "invalid_baseline_action"})
                    break
                args = action.get("arguments")
                if not isinstance(args, dict):
                    state.setdefault("errors", []).append({"code": "invalid_tool_arguments"})
                    break
                request_id = f"{run_id}:tool:{index}:{uuid4().hex[:10]}"
                if allowed_evaluation_tools is not None and action.get("tool") not in allowed_evaluation_tools:
                    requested_tool = str(action.get("tool") or "")[:80]
                    surface_sha256 = str((evaluation_tool_surface or {}).get("surface_sha256") or "")
                    available_names = list((evaluation_tool_surface or {}).get("allowed_operation_names") or [])
                    call = {
                        "tool": requested_tool,
                        "tool_name": requested_tool,
                        "planned_metric_id": args.get("metric_id") or args.get("metric_name"),
                        "planned_operation": requested_tool,
                        "arguments": _safe_tool_arguments(args),
                        "request_id": request_id,
                        "idempotency_key": request_id,
                        "status": "unsupported",
                        "call_status": "refused_before_execution",
                        "response": {
                            "server": "evaluation-tool-surface",
                            "tool_name": requested_tool,
                            "arguments": _safe_tool_arguments(args),
                            "request_id": request_id,
                            "result": {
                                "status": "unsupported",
                                "reason_code": "CAPABILITY_GAP",
                                "reason": "Requested operation is absent from the frozen authorized tool surface.",
                                "parsed": {},
                            },
                            "available_tools": available_names,
                            "evaluation_tool_surface_sha256": surface_sha256,
                        },
                    }
                    state["tool_calls"].append(call)
                    state.setdefault("decision_records", []).append({
                        "decision_id": f"{request_id}:surface-denial",
                        "request_id": request_id,
                        "limitation_code": "CAPABILITY_GAP",
                        "metric_id": str(args.get("metric_id") or args.get("metric_name") or ""),
                        "operation": requested_tool,
                        "dimension": args.get("dimension"),
                        "scope": "platform",
                        "comparison": {
                            key: args[key] for key in (
                                "baseline_date", "current_date", "baseline_start", "baseline_end",
                                "current_start", "current_end", "granularity",
                            ) if key in args
                        },
                        "evaluation_tool_surface_checked": True,
                        "evaluation_tool_surface_sha256": surface_sha256,
                        "requested_tool": requested_tool,
                        "allowed_tools": available_names,
                        "authorization_checked": False,
                        "sql_executed": False,
                        "source": "evaluation_tool_surface",
                    })
                    state.setdefault("errors", []).append({
                        "code": "evaluation_tool_surface_violation",
                        "requested_tool": requested_tool,
                        "surface_sha256": surface_sha256,
                        "request_id": request_id,
                    })
                    state["terminal_code"] = "CAPABILITY_GAP"
                    state["presentation"] = {
                        "status": "partial",
                        "limitation_code": "CAPABILITY_GAP",
                        "metric_id": str(args.get("metric_id") or args.get("metric_name") or ""),
                        "scope": "platform",
                        **({"dimension": args.get("dimension")} if args.get("dimension") else {}),
                        **{key: args[key] for key in (
                            "baseline_date", "current_date", "baseline_start", "baseline_end",
                            "current_start", "current_end", "granularity",
                        ) if key in args},
                    }
                    answer = "当前执行工具面不支持所请求的指标操作，未执行数据查询。"
                    final = {
                        "action": "FINAL",
                        "answer": answer,
                        "claims": [],
                        "presentation": state["presentation"],
                    }
                    break
                signature = json.dumps({"tool": action["tool"], "arguments": args}, sort_keys=True, ensure_ascii=False)
                if signature in seen:
                    state.setdefault("errors", []).append({"code": "duplicate_tool_call"})
                    break
                seen.add(signature)
                if len(state["tool_calls"]) >= int(max_tool_calls):
                    state["terminal_code"] = "BUDGET_EXHAUSTED"
                    state["presentation"] = {"status": "partial", "limitation_code": "BUDGET_EXHAUSTED"}
                    state.setdefault("errors", []).append({"code": "tool_call_budget_exhausted"})
                    break
                if time.monotonic() - started_monotonic >= deadline_seconds:
                    state["terminal_code"] = "TIMEOUT"
                    state["presentation"] = {"status": "partial", "limitation_code": "TIMEOUT"}
                    state.setdefault("errors", []).append({"code": "workflow_deadline_exceeded", "boundary": "before_tool_call"})
                    break
                call = {
                    "tool": action["tool"], "arguments": _safe_tool_arguments(args),
                    "request_id": request_id, "call_status": "started",
                }
                state["tool_calls"].append(call)
                try:
                    response = tool_fn(
                        question=question, tool=str(action["tool"]), arguments=args,
                        context=tenant_context, run_id=run_id, sequence=index,
                        request_id=request_id,
                    )
                    if evaluation_tool_surface is not None:
                        response["available_tools"] = list(evaluation_tool_surface.get("allowed_operation_names") or [])
                        response["evaluation_tool_surface_sha256"] = evaluation_tool_surface.get("surface_sha256")
                    if action["tool"] == "registered_numeric_decomposition":
                        card = _registered_decomposition_card(response)
                        if card is not None:
                            response.setdefault("result", {})["presentation_card"] = card
                            state.setdefault("registered_decomposition_cards", []).append(card)
                    call.update({
                        "tool_name": response.get("tool_name"),
                        "status": (response.get("result") or {}).get("status"),
                        "call_status": "succeeded",
                        "response": response,
                        # Preserve the exact request-scoped authorization
                        # snapshot emitted by the public query gateway on the
                        # invocation itself. The companion audit must be able
                        # to bind the decision to this request without
                        # inferring it from a run-level identity or borrowing
                        # a snapshot from the nested response later.
                        "authorization_snapshot": copy.deepcopy(response.get("authorization_snapshot"))
                        if isinstance(response.get("authorization_snapshot"), Mapping) else None,
                    })
                    evidence_id = f"{run_id}:evidence:{len(turns) + 1}"
                    state["evidence_bundles"].append({"evidence_id": evidence_id, "request_id": call["request_id"], "content": response})
                    turns.append({"call": call, "evidence_id": evidence_id})
                    user += "\nTOOL_RESULT " + json.dumps({"evidence_id": evidence_id, "result": response}, ensure_ascii=False, default=str)
                    result_status = str((response.get("result") or {}).get("status") or "")
                    if result_status in {"not_found", "insufficient_data"}:
                        state["terminal_code"] = "DATA_INSUFFICIENT"
                        state["presentation"] = {
                            "status": "partial", "metric_id": str(args.get("metric_id") or ""),
                            "scope": "platform",
                            "limitation_code": "DATA_INSUFFICIENT",
                            **({"baseline_date": args.get("baseline_date"), "current_date": args.get("current_date")}
                               if args.get("baseline_date") else {}),
                            **({"current_date": args.get("start_date")}
                               if not args.get("baseline_date") and args.get("start_date") == args.get("end_date") else {}),
                        }
                        state.setdefault("decision_records", []).append({
                            "decision_id": f"{request_id}:data-limit",
                            "request_id": request_id,
                            "limitation_code": "DATA_INSUFFICIENT",
                            "metric_id": str(args.get("metric_id") or ""),
                            "operation": str(action["tool"]),
                            "scope": "platform",
                            "authorization_checked": True,
                            "registry_capability_checked": False,
                            "sql_executed": True,
                            "evidence_id": evidence_id,
                            "source": "public_query_result",
                        })
                        final = {
                            "action": "FINAL",
                            "answer": "该请求的必要数据缺失，无法完成比较或分解；缺失数据不代表数值为零。",
                            "claims": [{"text": "目标数据缺失，因此未形成有效数值结论。", "evidence_ids": [evidence_id]}],
                            "presentation": state["presentation"],
                        }
                        break
                    if time.monotonic() - started_monotonic >= deadline_seconds:
                        state["terminal_code"] = "TIMEOUT"
                        state["presentation"] = {"status": "partial", "limitation_code": "TIMEOUT"}
                        state.setdefault("errors", []).append({"code": "workflow_deadline_exceeded", "boundary": "after_tool_call"})
                        break
                except Exception as exc:
                    safe_error = _safe_tool_error(exc)
                    limitation_code = getattr(exc, "limitation_code", None)
                    controlled_limitation = limitation_code in {
                        "CAPABILITY_GAP", "AUTHORIZATION_BLOCKED", "DATA_INSUFFICIENT",
                    }
                    call.update({"call_status": "refused" if controlled_limitation else "failed", **safe_error})
                    plan_summary = getattr(exc, "plan_summary", None)
                    if isinstance(plan_summary, dict):
                        call["resolved_plan"] = plan_summary
                    if controlled_limitation:
                        decision = {
                            "decision_id": f"{request_id}:decision",
                            "request_id": request_id,
                            "limitation_code": limitation_code,
                            "metric_id": str(args.get("metric_id") or ""),
                            "operation": str(action["tool"]),
                            "dimension": args.get("dimension"),
                            "scope": "platform",
                            "authorization_checked": True,
                            "registry_capability_checked": limitation_code == "CAPABILITY_GAP",
                            "sql_executed": False,
                            "source": "registered_capability" if limitation_code == "CAPABILITY_GAP" else "authorized_query_boundary",
                        }
                        state.setdefault("decision_records", []).append(decision)
                        state["terminal_code"] = limitation_code
                        answer = (
                            "当前注册能力不支持所请求的分析，未执行数据查询。"
                            if limitation_code == "CAPABILITY_GAP"
                            else "当前授权范围不允许所请求的分析，未执行数据查询。"
                            if limitation_code == "AUTHORIZATION_BLOCKED"
                            else "所需数据不完整，无法完成该项分析；未将缺失数据解释为零。"
                        )
                        state["presentation"] = {
                            "status": "unsupported" if limitation_code in {"CAPABILITY_GAP", "AUTHORIZATION_BLOCKED"} else "partial",
                            "metric_id": str(args.get("metric_id") or ""),
                            "scope": "platform",
                            "limitation_code": limitation_code,
                            **({
                                "baseline_start": args.get("baseline_start"),
                                "baseline_end": args.get("baseline_end"),
                                "current_start": args.get("current_start"),
                                "current_end": args.get("current_end"),
                            } if args.get("baseline_start") else {}),
                            **({"baseline_date": args.get("baseline_date"), "current_date": args.get("current_date")}
                               if args.get("baseline_date") else {}),
                        }
                        final = {"action": "FINAL", "answer": answer, "claims": [], "presentation": state["presentation"]}
                        break
                    state.setdefault("errors", []).append({
                        "code": "tool_execution_error", "tool": action["tool"],
                        "tool_name": action["tool"], "arguments": _safe_tool_arguments(args),
                        "request_id": getattr(exc, "request_id", request_id),
                        **safe_error,
                        **({"resolved_plan": plan_summary} if isinstance(plan_summary, dict) else {}),
                    })
                    break
    finally:
        reset_state(token)
    if final is not None and state.get("registered_decomposition_cards"):
        # The model-authored prose remains untouched. The visible structured
        # card is projected from the validated tool result and Registry.
        presentation = final.get("presentation")
        if not isinstance(presentation, dict):
            presentation = {}
        card = state["registered_decomposition_cards"][-1]
        # The validated Registry card is authoritative for the structured
        # target/value fields. Keep unrelated model-authored presentation data,
        # but never publish a contradictory delta or period beside the card.
        projected_fields = {
            "metric_id": card.get("metric_id"),
            "baseline_date": card.get("baseline_date"),
            "current_date": card.get("current_date"),
            "baseline_value": card.get("baseline_value"),
            "current_value": card.get("current_value"),
            "delta": card.get("delta_value"),
            "delta_value": card.get("delta_value"),
            "unit": card.get("unit"),
            "metric_unit": card.get("metric_unit"),
            "change_unit": card.get("change_unit"),
            "component_unit": card.get("component_unit"),
        }
        final["presentation"] = {
            **presentation,
            **projected_fields,
            "metric_decomposition": card,
        }
        # A successful, evidence-backed registered decomposition is complete
        # for this bare-tool task once the model emits FINAL. Keep this status
        # separate from the independent task scorer's evidence/text checks.
        if final.get("action") == "FINAL" and not state.get("errors"):
            final["presentation"]["status"] = "completed"
            state["terminal_code"] = "ENOUGH_EVIDENCE"
        output_adaptations.append("registered_decomposition_card_projection")
        card = state["registered_decomposition_cards"][-1]
        answer = str(final.get("answer") or "")
        # Registry IDs are useful for machine binding, but they are not a
        # user-facing factor label. Require the registered display name in
        # prose; otherwise readers of a localized answer would see only keys
        # such as ``avg_order_value``. The structured card remains the
        # authoritative factor mapping.
        missing_factor_label = any(
            not str(item.get("display_name") or "").strip()
            or str(item.get("display_name") or "").strip() not in answer
            for item in card.get("factors", []) if isinstance(item, dict)
        )
        if missing_factor_label:
            supplement = _registered_decomposition_text(card)
            if supplement:
                final["answer"] = (answer.rstrip() + "\n\n" + supplement).strip()
                output_adaptations.append("registered_factor_label_text_supplement")
    if final is not None:
        output_adaptations.extend(
            _adapt_registered_comparison_units(final, state.get("tool_calls") or [])
        )
    state["model_authored_final"] = copy.deepcopy(model_authored_final)
    state["model_evidence_reference_capture"] = _capture_model_evidence_references(model_authored_final)
    state["final_display"] = copy.deepcopy(final)
    state["final_output_adaptation"] = {
        "adapter_id": "registered_display_adapter_v1",
        "source": "model_final" if model_authored_final is not None else "deterministic_terminal",
        "applied": bool(output_adaptations),
        "operations": output_adaptations,
    }
    state["final"] = final
    state["tool_evidence"] = turns
    state["llm_call_summary"] = {
        "logical_calls": len(state.get("llm_calls") or []),
        "sdk_attempts": sum(len(item.get("attempts") or []) for item in state.get("llm_calls") or []),
        "usage": state.get("llm_token_summary"),
    }
    return state
