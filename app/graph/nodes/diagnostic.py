"""Governed Root Cause diagnostic loop (Phases 3-4).

The model chooses a typed action; this module owns deterministic validation,
tool execution, evidence creation, and terminal-state construction.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from app.analysis.attribution_methods import DataReconciliationError
from app.agents.contracts import MetricQueryPlan, stable_evidence_id
from app.agents.diagnostic_contracts import (
    AbstainAction,
    CallCandidateAction,
    CallToolAction,
    DiagnosticAction,
    DiagnosticBudget,
    DiagnosticComparison,
    DiagnosticFactQuery,
    DiagnosticPeriod,
    DiagnosticPlan,
    DiagnosticStep,
    ExecuteDecompositionAction,
    FinishAction,
    Observation,
    ParallelFactsAction,
    PolicyDecision,
    ReplanAction,
)
from app.core.config import get_settings
from app.core.time_expression import extract_iso_dates
from app.graph.state import AgentState
from app.mcp.client import query_business_data, query_scoped_business_data
from app.mcp.tools import TOOL_REGISTRY


class _DecompositionDataUnavailable(ValueError):
    """Registered calculation could not run because required facts are absent."""


_ACTION_ADAPTER = TypeAdapter(DiagnosticAction)
_LABELS = {
    "customer_state": "客户州", "customer_city": "客户城市", "seller_state": "卖家州",
    "product_category": "品类", "payment_type": "支付方式",
}
# Objective-text markers used when an LLM-planned step does not declare its
# evidence category.  Order matters: "相关经营指标趋势" must classify as
# related_metric, not trend.
_KIND_TEXT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dimension_contribution", ("维度", "拆解", "贡献", "dimension", "contribution")),
    ("related_metric", ("相关", "经营指标", "related", "order_count", "avg_order_value", "delivery_delay_rate")),
    ("trend", ("趋势", "trend")),
)
# Server-enforced minimum evidence skeleton for an anomaly diagnosis.
_SKELETON: tuple[tuple[str, str, str], ...] = (
    ("dimension_contribution", "req-dimension", "服务端要求：用 compare_breakdown_by_dimension 做维度贡献拆解"),
    ("related_metric", "req-related", "服务端要求：查询相关经营指标趋势（order_count、avg_order_value、delivery_delay_rate）"),
    ("trend", "req-trend", "服务端要求：查询目标指标趋势"),
)


def _step_kind(step: Any) -> str | None:
    declared = getattr(step, "kind", None)
    if declared:
        return declared
    text = str(getattr(step, "objective", "") or "").casefold()
    for kind, markers in _KIND_TEXT_MARKERS:
        if any(marker in text for marker in markers):
            return kind
    return None


def _target_metric(state: AgentState) -> str:
    """Deterministic target metric for the diagnosis under way.

    Priority: the first metric the user names (the question's subject) — a
    later mention ("订单量明明在涨") must not steal the target, and neither
    may the initial data query's own choice of metric to fetch.
    """
    stored = state.get("diagnostic_target_metric")
    if stored:
        return str(stored)
    # In attribution questions the first metric mention can be a proposed
    # driver ("order count changes explain GMV decline?"). Bind the target to
    # the metric nearest the primary directional claim before falling back to
    # first-mention ordering.
    try:
        from app.graph.nodes.premise import _claimed_direction, _metric_for_directional_claim

        question = str(state.get("user_message", ""))
        direction = _claimed_direction(question)
        if direction:
            claimed_metric = _metric_for_directional_claim(question, direction)
            if claimed_metric:
                state["diagnostic_target_metric"] = claimed_metric
                return claimed_metric
    except Exception:  # noqa: BLE001 - keep registry fallback below available
        pass
    try:
        from app.metrics.registry import get_registry
        haystack = str(state.get("user_message", "")).casefold()
        best: tuple[int, str] | None = None
        for metric in get_registry().list():
            for candidate in (metric.metric_id, *metric.aliases):
                if not candidate:
                    continue
                index = haystack.find(str(candidate).casefold())
                if index >= 0 and (best is None or index < best[0]):
                    best = (index, metric.metric_id)
        if best is not None:
            state["diagnostic_target_metric"] = best[1]
            return best[1]
    except Exception:  # noqa: BLE001 - registry unavailable → next fallback
        pass
    tool_result = (state.get("tool_result") or {}).get("result") or {}
    for payload in (tool_result.get("metric_result"), tool_result.get("trend_result")):
        if isinstance(payload, dict) and payload.get("metric_name"):
            state["diagnostic_target_metric"] = str(payload["metric_name"])
            return str(payload["metric_name"])
    return "gmv"


def _related_metrics(target: str, *, scope_name: str = "platform") -> list[str]:
    settings = get_settings()
    try:
        mapping = json.loads(settings.diagnostic_related_metrics_json)
    except Exception:  # noqa: BLE001 - malformed config → registry factors only
        mapping = {}
    related: list[str] = [str(item) for item in (mapping.get(target) or []) if str(item) != target]
    try:
        from app.metrics.registry import get_registry
        definition = get_registry().get(target)
        capability = definition.attribution.for_scope(scope_name)
        for spec in capability.factor_specs.get("product_shapley_v1", ()):
            for factor in (spec.factor_id, *spec.source_metric_ids):
                if factor != target and factor not in related:
                    related.append(factor)
        if "symmetric_ratio_v1" in capability.method_ids and definition.period_aggregation:
            for factor in (
                definition.period_aggregation.numerator_metric,
                definition.period_aggregation.denominator_metric,
            ):
                if factor != target and factor not in related:
                    related.append(factor)
    except Exception:  # noqa: BLE001 - missing metadata cannot expand data access
        pass
    return related[:4]


def _authorized_related_metrics(state: AgentState, *, scope_name: str) -> list[str]:
    """Return related metrics with both current policy and executable provenance.

    A selected target may open a registered relationship investigation when
    the current trusted policy also permits the related metric. The server
    issues a scoped ``registered_relation`` credential; the model cannot name
    arbitrary metrics or create this provenance.
    """
    from app.auth.context import AuthorizationDenied, build_authorization_context
    from app.metrics.registry import get_registry
    from app.tenancy import current_tenant_context

    tenant = current_tenant_context()
    if tenant is None:
        return []
    scope = state.get("controlled_analysis_scope") or {}
    try:
        authz = build_authorization_context(
            tenant,
            allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
            allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
            metric_scope_type=scope_name,
        )
        if "query_metric_trend" not in authz.allowed_tools:
            return []
        authorized_metric_ids = {
            str(item.get("metric_id"))
            for item in state.get("authorized_metrics", [])
            if isinstance(item, dict) and item.get("metric_id")
        }
        result: list[str] = []
        target_metric = _target_metric(state)
        if not scope and target_metric not in authorized_metric_ids:
            return []
        scope_data = {
            **scope,
            "tenant_id": state.get("tenant_id") or tenant.tenant_id,
            "user_id": state.get("user_id") or tenant.user_id,
            "permissions": state.get("user_role") or tenant.role,
            "scope_type": scope_name,
        }
        for metric_id in _related_metrics(target_metric, scope_name=scope_name):
            if scope and metric_id not in set(scope.get("allowed_metrics") or []):
                continue
            try:
                authz.require_metrics([metric_id])
                implementation = get_registry().get(metric_id).implementations.get(scope_name)
            except (AuthorizationDenied, KeyError, ValueError):
                continue
            if implementation is not None and "trend" in implementation.capabilities:
                if metric_id not in authorized_metric_ids:
                    # Locked explorer scopes require a separately authorized
                    # metric provenance; a registry relationship cannot widen
                    # their explicit metric allowlist.
                    if scope:
                        continue
                    from app.metrics.authorization import authorize_metric
                    try:
                        credential = authorize_metric(
                            metric_id,
                            provenance_type="registered_relation",
                            provenance_id=f"registry-relation:{target_metric}:{metric_id}",
                            scope=scope_name,
                            scope_data=scope_data,
                        )
                    except (KeyError, ValueError):
                        continue
                    state.setdefault("authorized_metrics", []).append(
                        credential.model_dump(mode="json")
                    )
                    authorized_metric_ids.add(metric_id)
                result.append(metric_id)
        return result
    except Exception:  # noqa: BLE001 - authorization uncertainty fails closed
        return []


def _first_dimension(
    target: str,
    scope_name: str = "platform",
    *,
    state: AgentState | None = None,
) -> str | None:
    """Choose a registered dimension that is executable for this identity.

    Registry order alone is not an authorization decision. In particular, a
    metric can register customer/location dimensions that the current role is
    not permitted to query. Defaults therefore come from the intersection of
    the metric contract and the live authorization snapshot.
    """
    from app.metrics.registry import get_registry
    from app.auth.context import AuthorizationDenied, build_authorization_context
    from app.tenancy import current_tenant_context

    definition = get_registry().get(target)
    capability = definition.attribution.for_scope(scope_name)
    registered = set(capability.dimensions)
    tenant = current_tenant_context()
    if tenant is None:
        return None
    scope = (state or {}).get("controlled_analysis_scope") or {}
    try:
        authz = build_authorization_context(
            tenant,
            allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
            allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
            metric_scope_type=scope_name,
        )
        authz.require_action("analyze")
        authz.require_metrics([target])
    except AuthorizationDenied:
        return None
    authorized = registered & set(authz.allowed_dimensions)
    if scope_name == "seller":
        authorized &= {"product_category", "month_position"}
    if scope:
        authorized &= set(scope.get("allowed_dimensions") or [])
    if not authorized:
        return None
    try:
        from app.graph.nodes.root_cause import _get_priority_dimensions

        preferred = sorted(set(_get_priority_dimensions(target)) & authorized)
        if preferred:
            return preferred[0]
    except Exception:  # noqa: BLE001 - stable registry order remains deterministic
        pass
    return sorted(authorized)[0]


def _requested_dimension_policy(state: AgentState, target: str | None = None) -> dict[str, Any]:
    """Resolve explicit dimension intent against registry, query tool and authz."""
    from app.graph.nodes.root_cause import infer_requested_dimensions
    from app.metrics.registry import get_registry
    from app.auth.context import AuthorizationDenied, build_authorization_context
    from app.tenancy import current_tenant_context

    requested = list(state.get("requested_dimensions") or infer_requested_dimensions(
        str(state.get("user_message") or "")
    ))
    if not requested:
        return {"requested": [], "registered": [], "available": [], "unsupported": [], "unauthorized": [], "unexecutable": []}

    metric_id = target or _target_metric(state)
    scope = state.get("controlled_analysis_scope") or {}
    scope_name = "seller" if scope else "platform"

    from app.context.support_resolver import resolve_support

    def _support(dimension: str, authz=None):
        return resolve_support(
            metric_id=metric_id,
            dimension=dimension,
            operation="contribution",
            scope=scope,
            authz=authz,
        )

    def _registered(results: dict[str, Any]) -> set[str]:
        return {
            dimension for dimension, result in results.items()
            if result.schema_registered == "ok" and result.metric_compatible == "ok"
        }

    def _unsupported(results: dict[str, Any]) -> list[str]:
        return [
            dimension for dimension in requested
            if results[dimension].reason_code in {
                "dimension_not_registered", "dimension_not_supported_by_metric",
            }
        ]

    tenant = current_tenant_context()
    if tenant is None:
        results = {dimension: _support(dimension) for dimension in requested}
        registered = _registered(results)
        return {
            "requested": requested, "registered": sorted(registered), "available": [],
            "unsupported": _unsupported(results),
            "unauthorized": [d for d in requested if d in registered],
            "unexecutable": [],
        }
    try:
        authz = build_authorization_context(
            tenant,
            allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
            allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
            metric_scope_type=scope_name,
        )
        try:
            authz.require_metrics([metric_id])
        except AuthorizationDenied:
            results = {dimension: _support(dimension, authz) for dimension in requested}
            registered = _registered(results)
            return {
                "requested": requested, "registered": sorted(registered), "available": [],
                "unsupported": _unsupported(results),
                "unauthorized": [d for d in requested if d in registered],
                "unexecutable": [],
            }
        results = {dimension: _support(dimension, authz) for dimension in requested}
        registered = _registered(results)
        unauthorized = [
            dimension for dimension in requested
            if results[dimension].user_permitted == "denied"
        ]
        authorized = registered - set(unauthorized)
        if scope_name == "seller":
            authorized &= {"product_category", "month_position"}
        try:
            definition = get_registry().get(metric_id)
            attribution = definition.attribution.for_scope(scope_name)
            methods = set(attribution.method_ids)
        except Exception:  # noqa: BLE001 - missing attribution fails closed
            methods = set()
        ratio_tool = (
            "compare_freight_rate_ratio_facts_by_dimension_period"
            if metric_id == "freight_rate" and "symmetric_mix_within_v1" in methods
            else "compare_avg_order_value_ratio_facts_by_dimension_period"
            if metric_id == "avg_order_value" and "symmetric_mix_within_v1" in methods
            else "compare_breakdown_by_dimension"
        )
        has_tool = bool(methods and ratio_tool in authz.allowed_tools)
        if has_tool and (scope or ratio_tool != "compare_breakdown_by_dimension"):
            start, end = _diagnostic_dates(state)
            if scope:
                start = str(scope.get("default_start_date") or start)
                end = str(scope.get("default_end_date") or end)
            has_tool = _in_scope_comparison_windows(
                date.fromisoformat(start), date.fromisoformat(end),
                seller_scoped=bool(scope),
                granularity=str(scope.get("granularity") or "") or None,
            ) is not None
        available = sorted(authorized) if has_tool else []
        unexecutable = [
            dimension for dimension in requested
            if dimension in authorized and dimension not in available
        ]
        return {
            "requested": requested, "registered": sorted(registered), "available": available,
            "unsupported": _unsupported(results), "unauthorized": unauthorized,
            "unexecutable": unexecutable,
        }
    except Exception:  # noqa: BLE001 - authorization/capability uncertainty fails closed
        results = {dimension: _support(dimension) for dimension in requested}
        registered = _registered(results)
        return {
            "requested": requested, "registered": sorted(registered), "available": [],
            "unsupported": _unsupported(results),
            "unauthorized": [d for d in requested if d in registered],
            "unexecutable": [],
        }


def _apply_requested_dimension_policy(state: AgentState, policy: dict[str, Any]) -> list[str]:
    requested = list(policy.get("requested") or [])
    state["requested_dimensions"] = requested
    unavailable = sorted(set(
        policy.get("unsupported", [])
        + policy.get("unauthorized", [])
        + policy.get("unexecutable", [])
    ))
    state["unavailable_requested_dimensions"] = unavailable
    if not requested or not unavailable:
        return unavailable

    if policy.get("unauthorized"):
        stop_code = "AUTHORIZATION_BLOCKED"
        reason = "当前授权范围不允许按指定维度查询；系统未替换为其他维度。"
    elif policy.get("unsupported"):
        stop_code = "CAPABILITY_GAP"
        reason = "该指标的注册归因契约不支持用户指定的维度；系统未替换为其他维度。"
    else:
        stop_code = "CAPABILITY_GAP"
        reason = "当前指标、日期范围或已授权查询工具无法执行用户指定的维度；系统未替换为其他维度。"
    for dimension in unavailable:
        gap = {
            "code": stop_code,
            "kind": "requested_dimension",
            "metric_id": _target_metric(state),
            "requested_dimension": dimension,
            "supported_dimensions": list(policy.get("available") or []),
            "reason": reason,
        }
        gaps = state.setdefault("diagnostic_capability_gaps", [])
        if gap not in gaps:
            gaps.append(gap)
    detail = f"{reason} 未完成维度归因：{', '.join(unavailable)}。"
    if policy.get("available"):
        detail += f"可选维度：{', '.join(policy['available'])}。"
    _set_investigation_stop_reason(state, stop_code, detail)
    return unavailable


def _evidence_kinds_seen(state: AgentState) -> set[str]:
    decomposition_request = state.get("registered_decomposition_request") or {}
    if decomposition_request.get("status") == "ready":
        required_inputs = [str(item) for item in decomposition_request.get("input_metric_ids", [])]
        if required_inputs and all(_trusted_daily_series(state, metric_id) is not None for metric_id in required_inputs):
            return {"trend"}
        return set()
    kinds: set[str] = set()
    for item in state.get("root_cause_evidence", []):
        purpose = str(item.get("purpose", ""))
        if purpose == "trend_baseline":
            kinds.add("trend")
        elif purpose.startswith("dimension_contribution:"):
            kinds.add("dimension_contribution")
        elif purpose.startswith("cross_metric:"):
            kinds.add("related_metric")
    return kinds


def _queried_related_metrics(state: AgentState) -> set[str]:
    return {
        str(item.get("purpose", "")).split(":", 1)[1]
        for item in state.get("root_cause_evidence", [])
        if str(item.get("purpose", "")).startswith("cross_metric:")
    }


def _required_evidence_kinds_for(state: AgentState) -> set[str]:
    """Return only evidence-floor categories executable in this scope.

    A universal trend + dimension + related-metric floor can otherwise stop an
    investigation before it reaches another supported action (for example, a
    registered order-level association when the root metric has no dimension
    breakdown). Capability and authorization are evaluated from server-owned
    registry/context state; the model cannot make an unavailable category
    executable by naming it in a plan.
    """
    from app.metrics.registry import get_registry
    from app.tenancy import current_tenant_context

    if (state.get("registered_decomposition_request") or {}).get("status") == "ready":
        return {"trend"}

    scope = state.get("controlled_analysis_scope") or {}
    tenant = current_tenant_context()
    if tenant is None:
        # In a locked scope, missing identity means no query category can be
        # asserted as authorized. Do not fall back to the legacy evidence
        # floor: an unsupported required step could stop the investigation
        # before candidate generation. Candidate/tool authorization remains
        # independently fail-closed.
        return {"trend"} if not scope else set()
    scope_name = "seller" if scope else "platform"
    metric_id = _target_metric(state)
    try:
        definition = get_registry().get(metric_id)
        implementation = definition.implementations.get(scope_name)
        if implementation is None:
            return set()
        metric_capability = definition.attribution.for_scope(scope_name)
        from app.auth.context import AuthorizationDenied, build_authorization_context

        allowed_metric_ids = scope.get("allowed_metrics") if scope else None
        allowed_dimensions = scope.get("allowed_dimensions") if scope else None
        authz = build_authorization_context(
            tenant,
            allowed_metric_ids=allowed_metric_ids,
            allowed_dimensions=allowed_dimensions,
            metric_scope_type=scope_name,
        )
        try:
            authz.require_metrics([metric_id])
        except AuthorizationDenied:
            return set()

        supported: set[str] = set()
        if "trend" in implementation.capabilities and "query_metric_trend" in authz.allowed_tools:
            supported.add("trend")
        if state.get("unavailable_requested_dimensions"):
            return supported

        dimension_ids = set(metric_capability.dimensions) & set(authz.allowed_dimensions)
        requested_dimensions = set(state.get("requested_dimensions") or [])
        if requested_dimensions:
            dimension_ids &= requested_dimensions
        if scope_name == "seller":
            dimension_ids &= {"product_category", "month_position"}
        dimension_tool = (
            "compare_freight_rate_ratio_facts_by_dimension_period"
            if metric_id == "freight_rate" and "symmetric_mix_within_v1" in metric_capability.method_ids
            else "compare_avg_order_value_ratio_facts_by_dimension_period"
            if metric_id == "avg_order_value" and "symmetric_mix_within_v1" in metric_capability.method_ids
            else "compare_breakdown_by_dimension"
        )
        start, end = _diagnostic_dates(state)
        if scope:
            start = str(scope.get("default_start_date") or start)
            end = str(scope.get("default_end_date") or end)
        comparison_supported = (not scope) or (_in_scope_comparison_windows(
            date.fromisoformat(start), date.fromisoformat(end),
            seller_scoped=bool(scope),
            granularity=str(scope.get("granularity") or "") or None,
        ) is not None)
        if dimension_tool != "compare_breakdown_by_dimension":
            # Grouped ratio tools accept natural week/month periods, whereas
            # additive tools also support point-day comparisons.
            comparison_supported = _in_scope_comparison_windows(
                date.fromisoformat(start), date.fromisoformat(end), seller_scoped=bool(scope),
                granularity=str(scope.get("granularity") or "") or None,
            ) is not None
        if (
            dimension_ids
            and metric_capability.method_ids
            and dimension_tool in authz.allowed_tools
            and comparison_supported
        ):
            supported.add("dimension_contribution")

        if _authorized_related_metrics(state, scope_name=scope_name):
            supported.add("related_metric")
        return supported
    except Exception:  # noqa: BLE001 - uncertainty must not require an unsafe query
        return set()


def _ensure_required_skeleton(
    plan: DiagnosticPlan,
    *,
    supported_kinds: set[str] | None = None,
) -> tuple[DiagnosticPlan, list[str]]:
    """Server-side minimal diagnostic skeleton (review fix #1).

    Every step whose evidence category belongs to the required skeleton is
    upgraded to ``required=True`` — whether the LLM declared it or the server
    appends it — so the deterministic executor guarantees the evidence floor
    and terminal actions are gated on it.  Steps that merely *claim* a
    category never satisfy the skeleton without being executed.
    """
    required_kinds = set(supported_kinds) if supported_kinds is not None else {
        "trend", "dimension_contribution", "related_metric",
    }
    steps: list[DiagnosticStep] = []
    appended: list[str] = []
    covered: set[str] = set()
    existing_ids: set[str] = set()
    for step in plan.steps:
        kind = _step_kind(step)
        if kind in {"trend", "dimension_contribution", "related_metric"}:
            if kind in required_kinds:
                covered.add(kind)
                # Upgrading must also pin the detected kind — otherwise the
                # executor falls back to "trend" for every kind-less step.
                if not step.required or not step.kind:
                    step = step.model_copy(update={"required": True, "kind": kind})
            elif step.required:
                # Do not turn an unsupported model-declared requirement into a
                # dead-end. Keep it visible in the plan, but not as a blocker.
                step = step.model_copy(update={"required": False, "kind": kind})
        existing_ids.add(step.step_id)
        steps.append(step)
    for kind, step_id, objective in _SKELETON:
        if kind not in required_kinds:
            continue
        if kind in covered:
            continue
        unique_id = step_id
        suffix = 2
        while unique_id in existing_ids:
            unique_id = f"{step_id}-{suffix}"
            suffix += 1
        steps.append(DiagnosticStep(
            step_id=unique_id, objective=objective,
            allowed_actions=["CALL_TOOL", "FINISH", "ABSTAIN"],
            completion_condition=f"已有 {kind} 证据", kind=kind, required=True,
        ))
        existing_ids.add(unique_id)
        appended.append(unique_id)
        covered.add(kind)
    if not appended:
        # Kind-steps may still have been upgraded to required above; always
        # return the updated plan, never the pre-upgrade original.
        return plan.model_copy(update={"steps": steps}), []
    return plan.model_copy(update={"steps": steps}), appended


def _question_for_query(query: Any) -> str:
    """Create a deterministic display question; execution uses ``plan``."""
    if query.period and query.tool_name == "query_metric_statistics":
        return f"查询 {query.metric_name} 在该时间范围的分布描述统计"
    if query.period:
        return f"查询 {query.metric_name} 趋势"
    if query.comparison:
        comparison = query.comparison
        if comparison.current_start:
            label = _LABELS.get(query.dimension or "", query.dimension or "维度")
            return (
                f"比较 {comparison.baseline_start} 至 {comparison.baseline_end} 与 "
                f"{comparison.current_start} 至 {comparison.current_end} 的 "
                f"{query.metric_name} 按 {label} 贡献度"
            )
        label = _LABELS.get(query.dimension or "", query.dimension or "维度")
        return f"拆解 {query.comparison.current_date} 与 {query.comparison.baseline_date} 的 {query.metric_name} 按 {label} 贡献度"
    return f"查询 {query.metric_name}"


def _dates(question: str) -> tuple[str, str]:
    from app.context.compiler import _comparison_date_roles

    roles = _comparison_date_roles(question)
    if roles is not None:
        # A trend needs chronological bounds, while comparisons retain their
        # explicit baseline/current roles separately below.
        return tuple(sorted(roles))
    # Use the same CJK-safe extractor as the context and premise layers so an
    # ISO date adjacent to Chinese text cannot fall through to the legacy date.
    matches = extract_iso_dates(question)
    end = matches[0] if matches else "2025-03-31"
    start = (date.fromisoformat(end) - timedelta(days=6)).isoformat()
    return start, end


def _diagnostic_comparison_dates(state: AgentState) -> tuple[str, str] | None:
    """Consume the input compiler's roles; only legacy states parse the text."""
    from app.context.compiler import _comparison_date_roles

    slots = (state.get("context_compilation") or {}).get("known_slots") or {}
    baseline = slots.get("baseline_date")
    current = slots.get("current_date")
    if baseline is not None or current is not None:
        baseline = baseline.get("value") if isinstance(baseline, dict) else baseline
        current = current.get("value") if isinstance(current, dict) else current
        # Incomplete/invalid compiled roles must not be replaced by guesses.
        try:
            baseline_day, current_day = date.fromisoformat(baseline), date.fromisoformat(current)
        except (TypeError, ValueError):
            raise ValueError("Diagnostic comparison date roles are invalid.") from None
        if baseline_day == current_day:
            raise ValueError("Diagnostic comparison dates must be distinct.")
        return baseline_day.isoformat(), current_day.isoformat()
    return _comparison_date_roles(str(state.get("user_message") or ""))


def _diagnostic_comparison_periods(state: AgentState) -> dict[str, str] | None:
    """Reuse the server-resolved period pair for later diagnostic queries.

    A monthly question such as "August vs July" has no ISO date roles. The
    initial data planner has already resolved and authorized those periods;
    dropping them here used to fall through to a synthetic default date.
    """

    def field(value: Any, name: str) -> Any:
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    def validate(
        baseline_start: Any,
        baseline_end: Any,
        current_start: Any,
        current_end: Any,
        granularity: Any,
        *,
        required: bool = False,
    ) -> dict[str, str] | None:
        values = (baseline_start, baseline_end, current_start, current_end)
        if not any(value is not None for value in values):
            if required:
                raise ValueError("Diagnostic comparison period boundaries are missing.")
            return None
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("Diagnostic comparison period boundaries are incomplete.")
        if granularity not in {"week", "month"}:
            raise ValueError("Diagnostic comparison period granularity is invalid.")
        try:
            baseline_start_date, baseline_end_date, current_start_date, current_end_date = (
                date.fromisoformat(value) for value in values
            )
        except ValueError:
            raise ValueError("Diagnostic comparison period dates are invalid.") from None
        if (
            baseline_end_date < baseline_start_date
            or current_end_date < current_start_date
            or current_start_date <= baseline_end_date
        ):
            raise ValueError("Diagnostic comparison periods must be ordered and non-overlapping.")
        return {
            "baseline_start": baseline_start_date.isoformat(),
            "baseline_end": baseline_end_date.isoformat(),
            "current_start": current_start_date.isoformat(),
            "current_end": current_end_date.isoformat(),
            "granularity": str(granularity),
        }

    target = _target_metric(state)
    batch = state.get("metric_query_batch_plan")
    plans = field(batch, "plans") or []
    for plan in plans:
        if field(plan, "query_type") != "metric_compare":
            continue
        metric = str(field(plan, "metric_name") or "")
        authorized = field(plan, "authorized_metric")
        authorized_metric = str(field(authorized, "metric_id") or "")
        if metric != target or authorized_metric != target:
            continue
        period = validate(
            field(plan, "baseline_start"), field(plan, "baseline_end"),
            field(plan, "current_start"), field(plan, "current_end"),
            field(plan, "granularity"),
        )
        if period is not None:
            return period

    compilation = state.get("context_compilation") or {}
    contract = field(compilation, "request_contract") or {}
    contract_metric = str(field(contract, "metric_id") or "")
    comparison = field(contract, "comparison") or {}
    if (
        field(comparison, "kind") == "period_pair"
        and (not contract_metric or contract_metric == target)
    ):
        baseline = field(comparison, "baseline") or {}
        current = field(comparison, "current") or {}
        period = validate(
            field(baseline, "start_date"), field(baseline, "end_date"),
            field(current, "start_date"), field(current, "end_date"),
            field(comparison, "granularity"),
            required=True,
        )
        if period is not None:
            return period

    operation_intent = field(compilation, "operation_intent") or {}
    intent_periods = field(operation_intent, "comparison_periods") or {}
    if intent_periods:
        baseline = field(intent_periods, "baseline") or {}
        current = field(intent_periods, "current") or {}
        return validate(
            field(baseline, "start_date"), field(baseline, "end_date"),
            field(current, "start_date"), field(current, "end_date"),
            field(intent_periods, "granularity"),
            required=True,
        )
    return None


def _diagnostic_dates(state: AgentState) -> tuple[str, str]:
    roles = _diagnostic_comparison_dates(state)
    if roles:
        return tuple(sorted(roles))
    periods = _diagnostic_comparison_periods(state)
    if periods:
        return periods["baseline_start"], periods["current_end"]
    return _dates(str(state.get("user_message") or ""))


def _registered_decomposition_dates(state: AgentState) -> tuple[str, str] | None:
    request = state.get("registered_decomposition_request") or {}
    if request.get("status") != "ready":
        return None
    baseline = str(request.get("baseline_date") or "")
    current = str(request.get("current_date") or "")
    try:
        baseline_date, current_date = date.fromisoformat(baseline), date.fromisoformat(current)
    except ValueError:
        return None
    if current_date <= baseline_date:
        return None
    return baseline, current


def _default_plan(state: AgentState) -> DiagnosticPlan:
    metric = _target_metric(state)
    return DiagnosticPlan(
        goal=f"解释 {metric} 的异常变化并判断是否有足够证据归因。",
        hypotheses=[f"{metric} 的变化可能与结构维度变化或相关经营指标变化有关。"],
        steps=[{
            "step_id": "trend",
            "objective": f"确认 {metric} 的变化趋势",
            "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
            "completion_condition": "已有趋势证据",
        }],
        stop_conditions=["证据足够支持结论", "没有显著原因", "预算耗尽"],
        budget=DiagnosticBudget(
            max_rounds=min(6, int(state.get("runtime_budget", {}).get("max_graph_steps", 32))),
            max_tool_calls=min(8, int(state.get("runtime_budget", {}).get("max_tool_calls", 8))),
            max_tokens=int(state.get("runtime_budget", {}).get("max_tokens", 12000) or 12000),
            max_cost=float(state.get("runtime_budget", {}).get("max_cost", 1.0) or 1.0),
        ),
    )


# Canonical step fields used to repair LLM plans that declare only a kind.
_KIND_STEP_DEFAULTS: dict[str, dict[str, Any]] = {
    "trend": {"step_id": "plan-trend", "objective": "查询目标指标趋势", "completion_condition": "已有目标指标趋势证据"},
    "dimension_contribution": {"step_id": "plan-dimension", "objective": "对目标指标做维度贡献拆解", "completion_condition": "已有维度贡献证据"},
    "related_metric": {"step_id": "plan-related", "objective": "查询相关经营指标趋势", "completion_condition": "已有相关经营指标证据"},
}


def _sanitize_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Repair common LLM plan slips deterministically (review fix #1).

    Steps that declare only a ``kind`` are completed from canonical fields;
    duplicate step ids get suffixed.  Field validation stays strict after this.
    """
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return payload
    seen: set[str] = set()
    repaired: list[Any] = []
    for index, step in enumerate(steps):
        if isinstance(step, dict):
            base = _KIND_STEP_DEFAULTS.get(str(step.get("kind") or ""), {})
            if not step.get("step_id"):
                step = {**step, "step_id": str(base.get("step_id") or f"step-{index + 1}")}
            if not step.get("objective"):
                step = {**step, "objective": str(base.get("objective") or "探索步骤")}
            if not step.get("allowed_actions"):
                step = {**step, "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"]}
            if not step.get("completion_condition"):
                step = {**step, "completion_condition": str(base.get("completion_condition") or "证据已观察")}
            unique = str(step["step_id"])
            suffix = 2
            while unique in seen:
                unique = f"{step['step_id']}-{suffix}"
                suffix += 1
            step = {**step, "step_id": unique}
        seen.add(str(step.get("step_id") if isinstance(step, dict) else index))
        repaired.append(step)
    return {**payload, "steps": repaired}


def _normalize_plan_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Review-fix: common degenerate field types → canonical shapes.

    The schema stays strict; this only repairs representations the models
    actually emit (a scalar where an array belongs, budget as a bare int).
    """
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("stop_conditions"), str):
        payload = {**payload, "stop_conditions": [payload["stop_conditions"]]}
    if isinstance(payload.get("hypotheses"), str):
        payload = {**payload, "hypotheses": [payload["hypotheses"]]}
    if isinstance(payload.get("steps"), dict):
        payload = {**payload, "steps": [payload["steps"]]}
    if isinstance(payload.get("budget"), (int, float)):
        payload = {**payload, "budget": {"max_rounds": int(payload["budget"])}}
    return payload


def _classify_validation_error(exc: Exception) -> str:
    """Failure taxonomy (review fix #5): schema errors are split so the eval
    can separate 'model cannot do the task' from 'model formatted wrong'."""
    text = str(exc)
    if "missing" in text and "Field required" in text:
        return "schema_missing_field"
    if "Input should be a valid" in text or "type=model_type" in text or "type=literal_error" in text:
        return "schema_type_error"
    return "schema_other"


def _normalize_action_fields(payload: Any, state: AgentState) -> Any:
    """Repair representation defects at the LLM/action boundary only.

    Internal action contracts remain strict. A FINISH response may borrow only
    evidence IDs already persisted in state; it cannot invent new evidence.
    """
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("action"), dict):
        payload = payload["action"]
    if str(payload.get("action_type") or "").upper() == "FINISH":
        if not payload.get("evidence_ids"):
            payload = {
                **payload,
                "evidence_ids": [
                    str(item.get("evidence_id"))
                    for item in state.get("evidence_bundles", [])
                    if item.get("evidence_id")
                ],
            }
        if not payload.get("conclusion"):
            payload = {**payload, "conclusion": _findings_block(state)}
    return payload


def _parse_llm_payload(payload: Any, model_type: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("plan"), dict):
        payload = payload["plan"]
    payload = _normalize_plan_fields(payload)
    if isinstance(payload, dict) and model_type is DiagnosticPlan:
        payload = _sanitize_plan_payload(payload)
    return model_type.model_validate(payload)


def _step_resume_signature(step: Any) -> tuple[Any, ...]:
    """Plan meaning used for safe progress reuse; step_id is only a label."""
    return (
        step.kind,
        bool(step.required),
        step.objective.strip(),
        step.completion_condition.strip(),
        tuple(sorted(step.allowed_actions)),
    )


def _step_has_compatible_valid_evidence(
    state: AgentState,
    step: Any,
    valid_evidence_ids: set[str],
) -> bool:
    """Require a validated, in-scope action result that satisfies this step kind."""
    raw = state.get("investigation_state") or {}
    target = raw.get("target") or {}
    target_metric = str(target.get("metric_id") or "")
    target_scope = str(target.get("scope_fingerprint") or "")
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in raw.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    for result in raw.get("action_results", []):
        if not isinstance(result, dict) or result.get("status") != "completed":
            continue
        if (result.get("validation_result") or {}).get("status") != "valid":
            continue
        action_type = str(result.get("action_type") or "")
        metric_id = str(result.get("metric_id") or "")
        for evidence_id in result.get("evidence_ids", []):
            eid = str(evidence_id)
            if eid not in valid_evidence_ids:
                continue
            evidence = evidence_by_id.get(eid)
            if (
                evidence is None
                or evidence.get("source_action_id") != result.get("candidate_id")
                or evidence.get("scope_fingerprint") != target_scope
            ):
                continue
            evidence_type = str(evidence.get("evidence_type") or "")
            values = evidence.get("values") or {}
            if step.kind == "trend":
                period = ((values.get("result") or {}).get("period")
                          or (values.get("result") or {}))
                if (
                    action_type == "ANALYZE_TREND"
                    and metric_id == target_metric
                    and evidence_type == "trend"
                    and str(period.get("end_date") or "") == str(target.get("current_end") or "")
                ):
                    return True
            elif step.kind == "related_metric":
                period = ((values.get("result") or {}).get("period")
                          or (values.get("result") or {}))
                if (
                    action_type == "ANALYZE_TREND"
                    and metric_id
                    and evidence_type == "trend"
                    and str(period.get("end_date") or "") == str(target.get("current_end") or "")
                ):
                    return True
            elif step.kind == "dimension_contribution":
                comparison = evidence.get("comparison") or {}
                baseline = comparison.get("baseline_period") or {}
                current = comparison.get("current_period") or {}
                if (
                    action_type in {"DECOMPOSE_METRIC", "BREAKDOWN_DIMENSION"}
                    and metric_id == target_metric
                    and evidence_type in {"metric_decomposition", "dimension_contribution"}
                    and baseline.get("start_date") == target.get("baseline_start")
                    and baseline.get("end_date") == target.get("baseline_end")
                    and current.get("start_date") == target.get("current_start")
                    and current.get("end_date") == target.get("current_end")
                ):
                    return True
    return False


def _branch_is_trusted_for_action_metric(branch: Any, metric_id: str, allowed_dimensions: set[str]) -> bool:
    """Keep candidate branches inside the current metric and dimension grants.

    A root branch can be trusted without a parent evidence link only when it
    does not name a dimension. Previously that root exception also admitted
    a named but unauthorized dimension, which allowed candidate generation to
    propose it before the tool gateway refused it.
    """
    if getattr(branch, "metric_id", None) != metric_id:
        return False
    # The contract defaults dimension to an empty string for root branches.
    # Treat that representation like an omitted dimension; otherwise the root
    # target is filtered out before candidate generation.
    dimension = str(getattr(branch, "dimension", None) or "").strip()
    if dimension == "__metric_factor__":
        return True
    if dimension:
        return dimension in allowed_dimensions
    return (
        getattr(branch, "parent_evidence_id", None) is None
        and getattr(branch, "depth", None) == 0
    )


def _reconcile_diagnostic_step_state(
    state: AgentState,
    previous_plan: DiagnosticPlan | None,
    next_plan: DiagnosticPlan,
    previous_step_state: dict[str, str],
) -> dict[str, str]:
    """Reuse only semantically identical completed steps backed by valid evidence."""
    reconciled = {step.step_id: "pending" for step in next_plan.steps}
    raw = state.get("investigation_state") or {}
    if (
        previous_plan is None
        # Plan-level intent can contain filter/range constraints not carried
        # as dedicated fields by DiagnosticStep. A changed goal is therefore
        # not eligible for reuse, even if a step label happens to match.
        or previous_plan.goal.strip() != next_plan.goal.strip()
        or state.get("diagnostic_investigation_invalidated")
        or not isinstance(raw, dict)
        or not raw.get("target")
        or _conflicting_evidence_ids(raw)
        or any(
            isinstance(item, dict)
            and item.get("failure_code") in {"INVALID_DATA", "AUTHORIZATION_BLOCKED", "EXECUTION_FAILED"}
            for item in raw.get("action_results", [])
        )
    ):
        return reconciled

    valid_ids = set(_completed_valid_evidence_ids(state))
    if not valid_ids:
        return reconciled
    old_completed = [
        step for step in previous_plan.steps
        if previous_step_state.get(step.step_id) == "completed"
    ]
    used_old_ids: set[str] = set()
    for new_step in next_plan.steps:
        for old_step in old_completed:
            if old_step.step_id in used_old_ids:
                continue
            if _step_resume_signature(old_step) != _step_resume_signature(new_step):
                continue
            if not _step_has_compatible_valid_evidence(state, old_step, valid_ids):
                continue
            reconciled[new_step.step_id] = "completed"
            used_old_ids.add(old_step.step_id)
            break
    return reconciled


def _diagnostic_context_user_message(state: AgentState, *, system_prompt: str = "") -> str:
    """Attach the WP1 diagnostic bundle outside the original user instruction.

    The bundle carries only task progress, controlled definitions and a
    current-evidence manifest.  It is reference material, never authorization,
    candidate identity or a numeric fact.  ``system_prompt`` is passed to the
    packer so the Bundle budget covers the actual planning/decision prompt.
    """
    instruction = str(state.get("user_message") or "")
    from app.context.orchestrator import ContextOrchestrator
    from app.llm.routing import model_for

    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "diagnostic", instruction=instruction,
        system_prompt=system_prompt, model=model_for("extract")[0],
    )
    if not bundle.text:
        return instruction
    return (
        f"{instruction}\n\n"
        "[受控上下文参考，不得作为授权、候选或数值依据]\n"
        f"{bundle.text}"
    )


def diagnostic_plan(state: AgentState) -> AgentState:
    from app.llm.client import llm_agent_action

    if state.get("seller_scope_payload"):
        from app.graph.nodes.seller_scope_guard import reject_unsupported_seller_scope

        return reject_unsupported_seller_scope(state)

    decomposition_request = state.get("registered_decomposition_request") or {}
    compilation = state.get("context_compilation") or {}
    request_contract = compilation.get("request_contract") or {}
    if decomposition_request.get("status") == "ready":
        dates = _registered_decomposition_dates(state)
        if dates is None:
            _set_investigation_stop_reason(
                state, "CAPABILITY_GAP", "Registered numeric decomposition requires a valid ordered date pair.",
            )
            state["diagnostic_target_metric"] = str(decomposition_request.get("target_metric_id") or "")
            _ensure_graph_investigation_state(state)
            plan = _default_plan(state).model_copy(update={
                "goal": "Complete the requested registered numeric decomposition, without inferring business causality.",
                "hypotheses": ["No business-causal hypothesis is requested or asserted."],
                "steps": [{
                    "step_id": "registered-decomposition-inputs",
                    "kind": "trend", "objective": "Obtain the registered decomposition inputs.",
                    "allowed_actions": ["CALL_TOOL", "PARALLEL_FACTS", "FINISH", "ABSTAIN"],
                    "completion_condition": "All registered inputs are complete.", "required": True,
                }],
            })
            state["diagnostic_plan"] = plan.model_dump(mode="json")
            state["diagnostic_step_state"] = {}
            state["diagnostic_fallback"] = True
            state["decision_driver"] = "registered_numeric_decomposition"
            state["current_agent"] = "diagnostic_plan"
            return state
        target_metric = str(decomposition_request.get("target_metric_id") or "")
        state["diagnostic_target_metric"] = target_metric
        _ensure_graph_investigation_state(state)
        plan = DiagnosticPlan(
            goal=f"Execute the Registry method {decomposition_request.get('method_id')} for {target_metric}; report numeric contributions only.",
            # DiagnosticPlan's schema requires at least one hypothesis entry.
            # This is an explicit boundary marker, not a causal hypothesis.
            hypotheses=["No business-causal hypothesis is requested or asserted."],
            steps=[{
                "step_id": "registered-decomposition-inputs",
                "kind": "trend",
                "objective": "Collect complete authorized daily series for the target and registered decomposition inputs.",
                "allowed_actions": ["CALL_TOOL", "PARALLEL_FACTS", "FINISH", "ABSTAIN"],
                "completion_condition": "Every registered input series is complete for the requested baseline/current span.",
                "required": True,
            }],
            stop_conditions=["Registered inputs are complete and reconcile", "Authorization, capability, data, or execution validation fails"],
            budget=DiagnosticBudget(
                max_rounds=min(4, int(state.get("runtime_budget", {}).get("max_graph_steps", 32))),
                max_tool_calls=min(8, int(state.get("runtime_budget", {}).get("max_tool_calls", 8))),
                max_tokens=int(state.get("runtime_budget", {}).get("max_tokens", 12000) or 12000),
                max_cost=float(state.get("runtime_budget", {}).get("max_cost", 1.0) or 1.0),
            ),
        )
        state["diagnostic_plan"] = plan.model_dump(mode="json")
        state["diagnostic_step_state"] = {"registered-decomposition-inputs": "pending"}
        state["diagnostic_fallback"] = True
        state["decision_driver"] = "registered_numeric_decomposition"
        state["current_agent"] = "diagnostic_plan"
        state.setdefault("node_history", []).append({
            "node": "diagnostic_plan",
            "summary": f"Registry decomposition plan: target={target_metric}; method={decomposition_request.get('method_id')}; factors={decomposition_request.get('factor_metric_ids')}; dependencies={decomposition_request.get('input_metric_ids')}; causal research excluded.",
        })
        return state

    if request_contract.get("required_evidence_sequence") == ["trend", "dimension_contribution"]:
        # This user request names two ordered, non-causal evidence steps. Keep
        # their order server-owned so a planner response cannot collapse or
        # reorder the requested workflow.
        from app.context.request_contract import validate_compiled_request_contract

        valid_contract, contract_reason = validate_compiled_request_contract(
            compilation,
            state_contract_hash=state.get("request_contract_hash"),
            require_state_anchor=True,
        )
        target_metric = str(request_contract.get("metric_id") or "")
        dimension = str(request_contract.get("dimension") or "")
        if not valid_contract or not target_metric or not dimension:
            reason = contract_reason or "request_contract_required_fields_missing"
            from app.context.request_contract import record_contract_violation
            record_contract_violation(state, "diagnostic_plan", reason)
            _set_investigation_stop_reason(
                state, "CAPABILITY_GAP", "The ordered statistical-analysis request contract is invalid; no query was run.",
            )
        state["diagnostic_target_metric"] = target_metric
        state["requested_dimensions"] = [dimension] if dimension else []
        _ensure_graph_investigation_state(state)
        policy = _requested_dimension_policy(state, target_metric or None)
        unavailable = _apply_requested_dimension_policy(state, policy)
        supported = _required_evidence_kinds_for(state)
        required_kinds = {"trend", "dimension_contribution"}
        if not unavailable and not required_kinds <= supported:
            _set_investigation_stop_reason(
                state, "CAPABILITY_GAP",
                "The registered and authorized capability set cannot execute the complete requested trend→contribution sequence.",
            )
        budget = state.get("runtime_budget") or {}
        plan = DiagnosticPlan(
            goal=(
                f"Verify the registered {target_metric} trend, then calculate the requested "
                f"{dimension} statistical contribution; do not infer business causality."
            ),
            hypotheses=["No business-causal hypothesis is requested or asserted."],
            steps=[
                {
                    "step_id": "requested-trend",
                    "kind": "trend",
                    "objective": f"Query the authorized {target_metric} trend for the requested comparison span.",
                    "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                    "completion_condition": "A validated trend result covers the requested span.",
                    "required": True,
                },
                {
                    "step_id": "requested-dimension-contribution",
                    "kind": "dimension_contribution",
                    "objective": f"Query {target_metric} statistical contribution by the requested {dimension} dimension.",
                    "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                    "completion_condition": "A validated contribution result covers the same comparison span and dimension.",
                    "required": True,
                },
            ],
            stop_conditions=[
                "Both requested evidence steps are validated for the same scope and comparison span.",
                "Stop on capability, authorization, data, integrity, execution, or budget limits.",
            ],
            budget=DiagnosticBudget(
                max_rounds=max(2, min(8, int(budget.get("max_graph_steps", 32)))),
                max_tool_calls=min(8, int(budget.get("max_tool_calls", 8))),
                max_tokens=int(budget.get("max_tokens", 12000) or 12000),
                max_cost=float(budget.get("max_cost", 1.0) or 1.0),
            ),
        )
        state["diagnostic_plan"] = plan.model_dump(mode="json")
        state["diagnostic_step_state"] = {
            "requested-trend": "pending",
            "requested-dimension-contribution": "pending",
        }
        state["diagnostic_fallback"] = True
        state["decision_driver"] = "sealed_request_sequence"
        state["current_agent"] = "diagnostic_plan"
        state.setdefault("node_history", []).append({
            "node": "diagnostic_plan",
            "summary": "Server plan preserves the sealed trend→dimension-contribution sequence; business causality is outside the request.",
        })
        return state

    previous_plan: DiagnosticPlan | None = None
    try:
        if state.get("diagnostic_plan"):
            previous_plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
    except Exception:  # malformed old plans cannot transfer completion state
        previous_plan = None
    previous_step_state = dict(state.get("diagnostic_step_state") or {})

    request_policy = _requested_dimension_policy(state)
    unavailable_dimensions = _apply_requested_dimension_policy(state, request_policy)
    supported_kinds = _required_evidence_kinds_for(state)
    evidence_kind_names = {
        "trend": "目标指标趋势",
        "dimension_contribution": "维度贡献拆解",
        "related_metric": "已授权相关指标趋势",
    }
    required_evidence_text = "、".join(
        evidence_kind_names[kind]
        for kind in ("trend", "dimension_contribution", "related_metric")
        if kind in supported_kinds
    ) or "无（当前范围没有可强制采集的证据类别）"
    prompt = (
        "你是根因诊断计划器。只输出 JSON 对象 plan，字段为 goal、hypotheses、steps、"
        "stop_conditions、budget。类型约束：hypotheses 和 stop_conditions 必须是字符串数组，"
        "steps 必须是对象数组；hypotheses 每项是简短字符串，禁止输出 {text,status} 对象。"
        "正确示例：\"hypotheses\":[\"订单量变化可能与目标指标变化有关\"]。"
        "stop_conditions 示例：\"stop_conditions\":[\"已定位主要根因\",\"关键假设被证伪\","
        "\"达到预算上限\"]。约束：hypotheses 最多 4 条，且只能围绕可用工具验证的假设"
        "（维度贡献拆解、指标趋势、相关经营指标）；steps 只做 2-4 个，每个 step 建议声明 "
        "kind 字段（trend | dimension_contribution | related_metric），"
        "allowed_actions 建议包含 CALL_TOOL 与 FINISH、ABSTAIN。"
        f"注意：服务端只会强制采集当前注册能力及授权范围支持的证据类别：{required_evidence_text}。"
        "不支持或未授权的证据类别不会成为必需步骤；你的计划应聚焦于可验证假设与额外探索。"
        "budget 必须是对象，形如 {\"max_rounds\": 5, \"max_tool_calls\": 6, \"max_tokens\": 12000, "
        "\"max_cost\": 1.0}。step 完整字段示例：{\"step_id\":\"trend\",\"objective\":\"确认趋势\","
        "\"kind\":\"trend\",\"allowed_actions\":[\"CALL_TOOL\",\"FINISH\",\"ABSTAIN\"],"
        "\"completion_condition\":\"已有趋势证据\"}。所有动作必须可由后续强类型校验。"
    )
    plan = None
    failure_class = "llm_unavailable"
    last_exc: Exception | None = None
    failures: list[dict[str, Any]] = state.setdefault("diagnostic_llm_failures", [])
    diagnostic_user_message = _diagnostic_context_user_message(state, system_prompt=prompt)
    for _attempt in range(2):  # one retry: reasoning models occasionally emit non-schema JSON
        last_exc = None
        try:
            raw = llm_agent_action(prompt, diagnostic_user_message)
            if not raw:
                failure_class = "empty_response"
                plan = None
            else:
                plan = _parse_llm_payload(raw, DiagnosticPlan)
                failure_class = "ok"
        except Exception as exc:
            failure_class = _classify_validation_error(exc)
            last_exc = exc
            print(f"[DEBUG/diagnostic] plan validation failed ({failure_class}): {type(exc).__name__}: {exc}")
            plan = None
        failures.append({"stage": "plan", "attempt": _attempt + 1, "class": failure_class, "recovered": False})
        failures[-1]["message"] = str(last_exc) if last_exc else ""
        if plan is not None:
            if len(failures) > 1 and failures[-2]["stage"] == "plan":
                failures[-2]["recovered"] = True
            break
    if plan is None:
        failures[-1]["recovered"] = False
        # Provider/schema failure falls back to a deterministic plan, but
        # stays on the same governed investigation path (never the legacy
        # branch_evidence → root_cause route).
        state["diagnostic_target_metric"] = _target_metric(state)
        _ensure_graph_investigation_state(state)
        fallback_plan, _ = _ensure_required_skeleton(
            _default_plan(state),
            supported_kinds=supported_kinds,
        )
        state["diagnostic_plan"] = fallback_plan.model_dump(mode="json")
        state["diagnostic_step_state"] = _reconcile_diagnostic_step_state(
            state, previous_plan, fallback_plan, previous_step_state,
        )
        state["hypotheses"] = [{"text": item, "status": "pending"} for item in fallback_plan.hypotheses]
        state["diagnostic_fallback"] = True
        state["decision_driver"] = "workflow_fallback"
        state["node_history"].append({"node": "diagnostic_plan", "summary": f"LLM plan unusable ({failure_class}); deterministic plan uses governed workflow."})
        return state
    state["diagnostic_target_metric"] = _target_metric(state)
    _ensure_graph_investigation_state(state)
    plan, appended = _ensure_required_skeleton(
        plan,
        supported_kinds=supported_kinds,
    )
    # Keep enough planned calls for required steps plus one adaptive action,
    # but derive that floor from the actual validated plan rather than assuming
    # every metric supports trend, dimension breakdown, and related metrics.
    runtime_tools = int(state.get("runtime_budget", {}).get("max_tool_calls", 8))
    required_step_count = sum(1 for step in plan.steps if step.required)
    required_tool_need = required_step_count + (1 if required_step_count else 0)
    floor_rounds = min(8, len(plan.steps) + 1)
    floor_tools = min(max(0, runtime_tools), required_tool_need)
    if plan.budget.max_rounds < floor_rounds or plan.budget.max_tool_calls < floor_tools:
        plan = plan.model_copy(update={"budget": plan.budget.model_copy(update={
            "max_rounds": max(plan.budget.max_rounds, floor_rounds),
            "max_tool_calls": max(plan.budget.max_tool_calls, floor_tools),
        })})
    state["diagnostic_plan"] = plan.model_dump(mode="json")
    state["diagnostic_step_state"] = _reconcile_diagnostic_step_state(
        state, previous_plan, plan, previous_step_state,
    )
    state["hypotheses"] = [{"text": item, "status": "pending"} for item in plan.hypotheses]
    state["decision_driver"] = "llm"
    state["diagnostic_fallback"] = False
    summary = f"Plan with {len(plan.steps)} steps."
    if appended:
        summary += f" Server appended required steps: {', '.join(appended)}."
    state["node_history"].append({"node": "diagnostic_plan", "summary": summary})
    return state


def _fallback_action(state: AgentState) -> AbstainAction:
    return AbstainAction(action_type="ABSTAIN", reason="budget_exhausted")


def _pending_required_steps(plan: DiagnosticPlan, state: AgentState) -> list[DiagnosticStep]:
    step_state = state.get("diagnostic_step_state", {})
    return [
        step for step in plan.steps
        if step.required and step_state.get(step.step_id, "pending") == "pending"
    ]


def _required_action_for(state: AgentState, step: DiagnosticStep) -> Any | None:
    """Deterministically build the next safe action for a required step.

    Review fix #4/#6: the system — not the model — produces the action that
    guarantees a required evidence category gets observed.  Returns None when
    the evidence already exists (the caller then completes the step).
    """
    target = _target_metric(state)
    start, end = _diagnostic_dates(state)
    scope = state.get("controlled_analysis_scope") or {}
    decomposition_request = state.get("registered_decomposition_request") or {}
    if decomposition_request.get("status") == "ready":
        dates = _registered_decomposition_dates(state)
        if dates is None:
            _set_investigation_stop_reason(state, "CAPABILITY_GAP", "The registered decomposition period is invalid.")
            return None
        baseline_date, current_date = dates
        input_ids = list(dict.fromkeys(str(item) for item in decomposition_request.get("input_metric_ids", [])))
        if not input_ids or target not in input_ids:
            _set_investigation_stop_reason(state, "CAPABILITY_GAP", "The registered method has no complete target/input metric contract.")
            return None
        if all(_trusted_daily_series(state, metric_id) is not None for metric_id in input_ids):
            return None
        from app.auth.context import AuthorizationDenied, build_authorization_context
        from app.tenancy import current_tenant_context

        tenant_context = current_tenant_context()
        try:
            if tenant_context is None:
                raise AuthorizationDenied("missing_context", "trusted tenant context is required")
            authz = build_authorization_context(
                tenant_context,
                allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
                allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
                metric_scope_type="seller" if scope else "platform",
            )
            authz.require_action("analyze")
            authz.require_tool("query_metric_trend")
            authz.require_metrics(input_ids)
            if scope and not set(input_ids) <= set(scope.get("allowed_metrics") or []):
                raise AuthorizationDenied("scope_rejected", "a registered dependency is outside the locked scope")
        except AuthorizationDenied as exc:
            _set_investigation_stop_reason(
                state, "AUTHORIZATION_BLOCKED", f"Registered decomposition input authorization was denied: {exc.code}.",
            )
            return None
        return ParallelFactsAction(
            action_type="PARALLEL_FACTS",
            facts=[
                DiagnosticFactQuery(
                    tool_name="query_metric_trend", metric_name=metric_id,
                    period=DiagnosticPeriod(start_date=baseline_date, end_date=current_date, granularity="day"),
                )
                for metric_id in input_ids
            ],
        )
    if scope:
        start = str(scope.get("default_start_date") or start)
        end = str(scope.get("default_end_date") or end)
    kind = _step_kind(step) or "trend"
    denied = set(state.get("diagnostic_denied_metrics") or [])
    if kind == "trend":
        if "trend" in _evidence_kinds_seen(state) or target in denied:
            return None
        return CallToolAction(
            action_type="CALL_TOOL", tool_name="query_metric_trend", metric_name=target,
            period=DiagnosticPeriod(start_date=start, end_date=end, granularity="day"),
        )
    if kind == "dimension_contribution":
        if "dimension_contribution" in _evidence_kinds_seen(state) or target in denied:
            return None
        requested_dimensions = list(state.get("requested_dimensions") or [])
        if requested_dimensions:
            policy = _requested_dimension_policy(state, target)
            unavailable = _apply_requested_dimension_policy(state, policy)
            if unavailable:
                return None
            selected_dimension = requested_dimensions[0]
        else:
            selected_dimension = _first_dimension(
                target, "seller" if scope else "platform", state=state,
            )
            if not selected_dimension:
                return None
        explicit_dates = _diagnostic_comparison_dates(state) if not scope else None
        windows = _in_scope_comparison_windows(
            date.fromisoformat(start), date.fromisoformat(end), seller_scoped=bool(scope),
            granularity=str(scope.get("granularity") or "") or None,
        )
        if explicit_dates is not None:
            baseline_start = baseline_end = date.fromisoformat(explicit_dates[0])
            current_start = current_end = date.fromisoformat(explicit_dates[1])
            granularity = "day"
        elif windows is None:
            if scope:
                detail = "当前锁定范围内没有两个不重叠且受查询契约支持的比较周期。"
                state.setdefault("diagnostic_capability_gaps", []).append({
                    "metric_id": target, "kind": "period_attribution", "detail": detail,
                })
                _set_investigation_stop_reason(state, "CAPABILITY_GAP", detail)
                return None
            # The platform tool also supports an exact current-day versus
            # previous-day contribution query. Keep both dates inside the
            # implicit seven-day analysis window.
            current_start = current_end = date.fromisoformat(end)
            baseline_start = baseline_end = current_start - timedelta(days=1)
            granularity = "day"
        else:
            baseline_start, baseline_end, current_start, current_end, granularity = windows
        from app.metrics.registry import get_registry
        capability = get_registry().get(target).attribution.for_scope("seller" if scope else "platform")
        ratio_tool = {
            "freight_rate": "compare_freight_rate_ratio_facts_by_dimension_period",
            "avg_order_value": "compare_avg_order_value_ratio_facts_by_dimension_period",
        }.get(target) if "symmetric_mix_within_v1" in capability.method_ids else None
        return CallToolAction(
            action_type="CALL_TOOL", tool_name=ratio_tool or "compare_breakdown_by_dimension", metric_name=target,
            dimension=selected_dimension,
            comparison=DiagnosticComparison(
                **({
                    "current_date": current_end.isoformat(),
                    "baseline_date": baseline_end.isoformat(),
                } if granularity == "day" and not ratio_tool else {
                    "current_start": current_start.isoformat(), "current_end": current_end.isoformat(),
                    "baseline_start": baseline_start.isoformat(), "baseline_end": baseline_end.isoformat(),
                    "granularity": granularity,
                }),
            ),
        )
    # related_metric: one PARALLEL_FACTS batch covering every related metric
    # not yet observed (and not previously denied), trimmed to the remaining
    # tool-call budget.
    related_done = _queried_related_metrics(state) | set(state.get("diagnostic_denied_metrics") or [])
    scope_name = "seller" if scope else "platform"
    pending_related = [
        metric for metric in _authorized_related_metrics(state, scope_name=scope_name)
        if metric not in related_done
    ]
    if not pending_related:
        return None
    remaining_budget = int(state.get("runtime_budget", {}).get("max_tool_calls", 8)) - len(state.get("tool_calls", [])) - 1
    facts = [
        DiagnosticFactQuery(
            tool_name="query_metric_trend", metric_name=metric,
            period=DiagnosticPeriod(start_date=start, end_date=end, granularity="day"),
        )
        for metric in pending_related[: max(1, remaining_budget)]
    ]
    return ParallelFactsAction(action_type="PARALLEL_FACTS", facts=facts)


def validated_dimension_contribution_cards(state: AgentState) -> list[dict[str, Any]]:
    """Project the same validated cards for finalization, answers and review."""
    records = validated_dimension_contribution_evidence(state)
    cards: list[dict[str, Any]] = []
    for record in records:
        cards.append({
            "dimension": record["dimension"],
            "metric_name": record["metric_id"],
            "unit": record["unit"],
            "aggregation": "registered_additive_contribution",
            "baseline_value": record["baseline_value"],
            "current_value": record["current_value"],
            "baseline_date": record["baseline_date"],
            "current_date": record["current_date"],
            "baseline_period": record["baseline_period"],
            "current_period": record["current_period"],
            "total_delta": record["total_delta"],
            "residual_value": record.get("residual_value"),
            "residual_type": record.get("residual_type") or "unknown",
            "items": record["items"],
            "evidence_ids": record["evidence_ids"],
            "analysis_nature": "descriptive_contribution",
            "method": record["method_id"],
        })
    return cards


def _validated_statistical_contribution_terminal(
    state: AgentState, *, allow_completed_stop: bool = False,
) -> FinishAction | None:
    """Finish a user-requested statistical-contribution task at its evidence floor.

    A dimensional statistical contribution is complete once its required trend
    and contribution evidence have both passed the normal action, scope,
    period, registry, and reconciliation checks. Optional causal exploration is
    not part of this operation. Any global stop, failed action, or execution
    error keeps its existing terminal semantics.
    """
    compiled_intent = (state.get("context_compilation") or {}).get("operation_intent") or {}
    if not isinstance(compiled_intent, dict) or compiled_intent.get("kind") != "dimensional_statistical_contribution":
        return None
    if state.get("errors"):
        return None

    investigation = _ensure_graph_investigation_state(state)
    if investigation is None:
        return None
    if investigation.stop_reason is not None and not (
        allow_completed_stop and investigation.stop_reason.code == "ENOUGH_EVIDENCE"
    ):
        return None
    raw_investigation = investigation.model_dump(mode="json")
    if _conflicting_evidence_ids(raw_investigation):
        return None
    if any(result.status == "failed" for result in investigation.action_results):
        return None
    if any(result.failure_code in {"INVALID_DATA", "AUTHORIZATION_BLOCKED", "EXECUTION_FAILED"}
           for result in investigation.action_results):
        return None

    try:
        plan = DiagnosticPlan.model_validate(state.get("diagnostic_plan") or {})
    except Exception:  # noqa: BLE001 - malformed plans cannot lower the evidence floor
        return None
    required_kinds = {
        _step_kind(step) for step in plan.steps if step.required
    }
    if not {"trend", "dimension_contribution"} <= required_kinds:
        return None
    step_states = state.get("diagnostic_step_state") or {}
    if any(
        step.required and step_states.get(step.step_id, "pending") != "completed"
        for step in plan.steps
    ):
        return None

    target = raw_investigation.get("target") or {}
    metric_id = str(target.get("metric_id") or "")
    if not metric_id or metric_id != _target_metric(state):
        return None
    valid_ids = set(_completed_valid_evidence_ids(state))
    trusted_bundle_ids = {
        str(item.get("evidence_id"))
        for item in (state.get("evidence_bundles") or [])
        if isinstance(item, dict) and item.get("trusted") is True and item.get("evidence_id")
    }
    contribution_records = validated_dimension_contribution_evidence(state)
    if not contribution_records:
        return None
    requested_dimensions = [str(item) for item in (state.get("requested_dimensions") or []) if item]
    if requested_dimensions and not set(requested_dimensions) <= {
        str(item.get("dimension") or "") for item in contribution_records
    }:
        return None

    # Recheck the current authorization snapshot before using the evidence to
    # end the task. This does not authorize new tools or queries.
    from app.auth.context import AuthorizationDenied, build_authorization_context
    from app.metrics.registry import get_registry
    from app.tenancy import current_tenant_context

    tenant = current_tenant_context()
    if tenant is None:
        return None
    scope = state.get("controlled_analysis_scope") or {}
    scope_name = "seller" if scope else "platform"
    try:
        authorization = build_authorization_context(
            tenant,
            allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
            allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
            metric_scope_type=scope_name,
        )
        authorization.require_action("analyze")
        authorization.require_metrics([metric_id])
        dimensions = [str(item.get("dimension") or "") for item in contribution_records]
        authorization.require_dimensions(dimensions)
        if scope and metric_id not in set(scope.get("allowed_metrics") or []):
            return None
        if scope and not set(dimensions) <= set(scope.get("allowed_dimensions") or []):
            return None
        for dimension in dimensions:
            capability = get_registry().get(metric_id).attribution.for_scope(scope_name)
            if dimension not in capability.dimensions:
                return None
    except (AuthorizationDenied, KeyError, ValueError):
        return None

    try:
        baseline_start = str(target.get("baseline_start") or "")
        baseline_end = str(target.get("baseline_end") or "")
        current_start = str(target.get("current_start") or "")
        current_end = str(target.get("current_end") or "")
        series_start, series_end = _diagnostic_dates(state)
        series = _trusted_daily_series(state, metric_id)
        if not all((baseline_start, baseline_end, current_start, current_end)) or series is None:
            return None
        if _period_series_total(series, date.fromisoformat(series_start), date.fromisoformat(series_end)) is None:
            return None
        if any(
            str(item.get("metric_id") or "") != metric_id
            or item.get("evidence_id") not in valid_ids
            or item.get("evidence_id") not in trusted_bundle_ids
            or item.get("analysis_nature") != "descriptive_contribution"
            or item.get("baseline_period") != {"start_date": baseline_start, "end_date": baseline_end}
            or item.get("current_period") != {"start_date": current_start, "end_date": current_end}
            for item in contribution_records
        ):
            return None
        if any(
            _period_series_total(series, date.fromisoformat(start), date.fromisoformat(end)) is None
            for start, end in ((baseline_start, baseline_end), (current_start, current_end))
        ):
            return None
        trend_action_ids = {
            str(result.candidate_id)
            for result in investigation.action_results
            if result.action_type == "ANALYZE_TREND"
            and result.metric_id == metric_id
            and result.status == "completed"
            and result.validation_result.status == "valid"
        }
        trend_evidence_ids = [
            evidence.evidence_id
            for evidence in investigation.evidence
            if evidence.evidence_id in valid_ids
            and evidence.evidence_id in trusted_bundle_ids
            and evidence.source_action_id in trend_action_ids
            and evidence.evidence_type == "trend"
            and evidence.metric_id == metric_id
            and evidence.authorized_metric_provenance == "tool_gateway_policy_validated"
            and evidence.scope_fingerprint == investigation.target.scope_fingerprint
            and (
                evidence.comparison.get("period") or {}
            ).get("start_date") == series_start
            and (
                evidence.comparison.get("period") or {}
            ).get("end_date") == series_end
            and (
                evidence.comparison.get("period") or {}
            ).get("granularity") == "day"
            and (not evidence.trace_id or not state.get("trace_id") or evidence.trace_id == state.get("trace_id"))
        ]
        if not trend_evidence_ids:
            return None
    except (TypeError, ValueError, KeyError):
        return None

    evidence_ids = list(dict.fromkeys([
        *trend_evidence_ids,
        *(str(evidence_id) for item in contribution_records for evidence_id in item.get("evidence_ids", [])),
    ]))
    if not evidence_ids or not set(evidence_ids) <= valid_ids or not set(evidence_ids) <= trusted_bundle_ids:
        return None
    dimension_text = ", ".join(dict.fromkeys(str(item["dimension"]) for item in contribution_records))
    return FinishAction(
        action_type="FINISH",
        conclusion=(
            f"The requested {metric_id} trend and {dimension_text} statistical contribution are validated "
            "for the authorized comparison scope and period. This is a descriptive contribution result; "
            "the underlying business cause remains unconfirmed."
        ),
        evidence_ids=evidence_ids,
        residual_uncertainty="Optional deeper investigation was not required to complete the requested statistical contribution; business causality remains unconfirmed.",
    )


def diagnostic_decide(state: AgentState) -> AgentState:
    state["diagnostic_round"] = int(state.get("diagnostic_round", 0)) + 1
    state.pop("diagnostic_selected_candidate_id", None)
    state.pop("diagnostic_decision_source", None)
    state.pop("diagnostic_fallback_reason", None)
    state.pop("diagnostic_backend_terminal", None)
    plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
    state["diagnostic_required_action"] = False
    investigation_raw = state.get("investigation_state") or {}
    stop_raw = investigation_raw.get("stop_reason") if isinstance(investigation_raw, dict) else None
    hard_stop = str((stop_raw or {}).get("code") or "") if isinstance(stop_raw, dict) else ""
    if hard_stop in {"AUTHORIZATION_BLOCKED", "CAPABILITY_GAP", "DATA_INSUFFICIENT", "USER_SCOPE_LIMIT", "BUDGET_EXHAUSTED", "INVALID_DATA", "EXECUTION_FAILED"}:
        reason = "budget_exhausted" if hard_stop == "BUDGET_EXHAUSTED" else "insufficient_evidence"
        known = _completed_valid_evidence_ids(state)
        action: Any = AbstainAction(
            action_type="ABSTAIN", reason=reason, supporting_evidence_ids=known[:50],
        )
        state["termination_initiator"] = "budget" if reason == "budget_exhausted" else "policy"
    elif state["diagnostic_round"] > plan.budget.max_rounds:
        action: Any = _fallback_action(state)
        state["termination_initiator"] = "budget"
    else:
        action = None
        # Review fix #1/#6: while a server-required step is pending, the system
        # — not the model — produces the next action.  The LLM is consulted
        # only once the required evidence floor is covered (or cannot be).
        for step in _pending_required_steps(plan, state):
            candidate = _required_action_for(state, step)
            if candidate is not None:
                action = candidate
                state["diagnostic_required_action"] = True
                state["node_history"].append({
                    "node": "diagnostic_decide",
                    "summary": f"CALL_TOOL(required:{step.step_id})",
                })
                state["diagnostic_action"] = action.model_dump(mode="json")
                return state
            if _step_kind(step) in _evidence_kinds_seen(state):
                # Only evidence of the required category completes a step.
                state["diagnostic_step_state"][step.step_id] = "completed"
                continue
            investigation_raw = state.get("investigation_state") or {}
            stop_raw = investigation_raw.get("stop_reason") if isinstance(investigation_raw, dict) else None
            stop_code = str((stop_raw or {}).get("code") or "") if isinstance(stop_raw, dict) else ""
            if not stop_code:
                _set_investigation_stop_reason(
                    state, "CAPABILITY_GAP",
                    f"必需分析步骤 {step.step_id} 无法生成受支持的查询，且没有对应证据。",
                )
                investigation_raw = state.get("investigation_state") or {}
                stop_raw = investigation_raw.get("stop_reason") if isinstance(investigation_raw, dict) else None
                stop_code = str((stop_raw or {}).get("code") or "") if isinstance(stop_raw, dict) else "CAPABILITY_GAP"
            stop_reason = (
                "budget_exhausted" if stop_code == "BUDGET_EXHAUSTED"
                else "insufficient_evidence"
            )
            safe_evidence = investigation_raw.get("evidence", []) if isinstance(investigation_raw, dict) else []
            scope_fingerprint = (investigation_raw.get("target") or {}).get("scope_fingerprint") if isinstance(investigation_raw, dict) else None
            known = [
                item.get("evidence_id") for item in safe_evidence
                if isinstance(item, dict) and item.get("evidence_id")
                and item.get("scope_fingerprint") == scope_fingerprint
            ]
            action = AbstainAction(
                action_type="ABSTAIN", reason=stop_reason,
                supporting_evidence_ids=known[:50],
            )
            state["termination_initiator"] = "budget" if stop_reason == "budget_exhausted" else "policy"
            break
        if action is None:
            action = _validated_statistical_contribution_terminal(state)
            if action is not None:
                state["diagnostic_backend_terminal"] = "validated_statistical_contribution"
                state["diagnostic_decision_source"] = "backend"
                state["termination_initiator"] = "agent"
                state["node_history"].append({
                    "node": "diagnostic_decide",
                    "summary": "FINISH(validated_statistical_contribution)",
                })
                state["diagnostic_action"] = action.model_dump(mode="json")
                return state
            action = _llm_decide_action(state)
            if isinstance(action, AbstainAction):
                investigation_raw = state.get("investigation_state") or {}
                stop_raw = investigation_raw.get("stop_reason") if isinstance(investigation_raw, dict) else None
                stop_code = str((stop_raw or {}).get("code") or "") if isinstance(stop_raw, dict) else ""
                unresolved_data_gaps = [
                    step for step in plan.steps
                    if step.required
                    and state.get("diagnostic_step_state", {}).get(step.step_id) == "data_insufficient"
                ]
                if stop_code == "NO_HIGH_VALUE_ACTION" and unresolved_data_gaps:
                    from app.investigation.contracts import StopReason

                    investigation = _ensure_graph_investigation_state(state)
                    if investigation is not None:
                        detail = next((
                            str(item.get("detail"))
                            for item in reversed(state.get("diagnostic_candidate_data_gaps", []))
                            if isinstance(item, dict) and item.get("detail")
                        ), "Required evidence could not be obtained from the available data.")
                        investigation.stop_reason = StopReason(
                            code="DATA_INSUFFICIENT", detail=detail, initiator="backend",
                        )
                        state["investigation_state"] = investigation.model_dump(mode="json")
    state["diagnostic_action"] = action.model_dump(mode="json")
    state["node_history"].append({"node": "diagnostic_decide", "summary": action.action_type})
    return state


_MATERIALITY_PCT = 30.0


def _materiality_flag(points: list[Any], pct: float) -> str:
    """Use a fixed floor plus historical volatility for small real changes.

    A fixed 30% cutoff incorrectly labels a 20–25% move as noise when the
    preceding baseline is flat. The 10% floor is conservative for this
    seven-point fixture; volatile histories retain a higher adaptive cutoff.
    """
    baseline_changes: list[float] = []
    for previous, current in zip(points[:-2], points[1:-1]):
        if not isinstance(previous, dict) or not isinstance(current, dict):
            continue
        previous_value, current_value = previous.get("value"), current.get("value")
        if previous_value is None or current_value is None or not previous_value:
            continue
        baseline_changes.append(abs((float(current_value) - float(previous_value)) / abs(float(previous_value))) * 100)
    baseline_volatility = sorted(baseline_changes)[len(baseline_changes) // 2] if baseline_changes else 0.0
    threshold = max(10.0, baseline_volatility * 3.0, _MATERIALITY_PCT if baseline_volatility else 10.0)
    return "显著" if abs(pct) >= threshold else "不显著"


def _trend_change(points: list[Any]) -> tuple[Any, Any] | None:
    """Last observed value vs the previous one, skipping null points."""
    values = [point.get("value") for point in points if isinstance(point, dict) and point.get("value") is not None]
    if len(values) >= 2:
        return values[-1], values[-2]
    if len(values) == 1:
        return values[-1], None
    return None


def _capability_gap_text(item: dict[str, Any]) -> str | None:
    if item.get("kind") == "requested_dimension":
        dimension = str(item.get("requested_dimension") or "指定")
        label = _LABELS.get(dimension, dimension)
        metric = str(item.get("metric_id") or "目标指标")
        reason = str(item.get("reason") or "该维度当前不可用")
        supported = [
            _LABELS.get(str(value), str(value))
            for value in item.get("supported_dimensions", [])
        ]
        choices = f"可选维度：{'、'.join(supported)}。" if supported else ""
        return f"无法按你指定的{label}维度完成{metric}归因：{reason} {choices}请重新选择维度后再分析。"
    reason = str(item.get("reason") or item.get("detail") or item.get("message") or "").strip()
    if not reason:
        return None
    label = str(item.get("metric_id") or item.get("kind") or "当前分析")
    return f"能力缺口({label}): {reason}"


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _observed_values_summary_with_evidence(
    state: AgentState,
    *,
    allowed_evidence_ids: set[str] | None,
) -> tuple[str, list[str]]:
    """Render only values backed by the finalizer's accepted evidence set."""

    parts: list[str] = []
    included_ids: list[str] = []
    records = state.get("root_cause_evidence") or []
    if not isinstance(records, list):
        return "", []
    for item in records:
        if not isinstance(item, dict):
            continue
        typed = item.get("investigation_evidence")
        typed = typed if isinstance(typed, dict) else {}
        evidence_id = str(typed.get("evidence_id") or item.get("query_id") or "")
        if allowed_evidence_ids is not None and (
            not evidence_id or evidence_id not in allowed_evidence_ids
        ):
            continue

        purpose = str(item.get("purpose", ""))
        metric = str(item.get("metric_id") or "metric")
        if purpose.startswith("metric_decomposition:"):
            values = typed.get("values", {})
            if not isinstance(values, dict):
                continue
            baseline = values.get("baseline_value")
            current = values.get("current_value")
            delta = values.get("total_delta")
            if all(_finite_number(value) for value in (baseline, current, delta)):
                parts.append(
                    f"{metric}: baseline={baseline}, current={current}, delta={delta}"
                )
                if evidence_id:
                    included_ids.append(evidence_id)
            continue
        if not (purpose == "trend_baseline" or purpose.startswith("cross_metric:")):
            continue

        envelope = item.get("result")
        envelope = envelope if isinstance(envelope, dict) else {}
        inner = envelope.get("result", envelope)
        inner = inner if isinstance(inner, dict) else {}
        trend = inner.get("trend_result")
        if not isinstance(trend, dict):
            continue
        points = trend.get("points") or []
        if not isinstance(points, list):
            continue
        rendered: list[str] = []
        for point in points:
            if not isinstance(point, dict) or not _finite_number(point.get("value")):
                continue
            when = point.get("date") or point.get("period") or point.get("bucket") or "?"
            unit = str(point.get("unit") or "")
            rendered.append(f"{when}={point.get('value')}{unit}")
        if rendered:
            parts.append(f"{metric}: " + ", ".join(rendered[-5:]))
            if evidence_id:
                included_ids.append(evidence_id)
        elif (
            allowed_evidence_ids is None
            or trend.get("status") in {"ok", "no_data"}
            or inner.get("status") in {"ok", "no_data"}
            or envelope.get("status") in {"ok", "no_data"}
        ):
            parts.append(f"{metric}: no numeric observations")
            if evidence_id:
                included_ids.append(evidence_id)
    summary = ("Observed values: " + "; ".join(parts) + ".") if parts else ""
    return summary, list(dict.fromkeys(included_ids))


def _observed_values_summary(state: AgentState) -> str:
    """Compatibility wrapper used by focused diagnostic summaries."""

    return _observed_values_summary_with_evidence(
        state, allowed_evidence_ids=None,
    )[0]


def _findings_block(state: AgentState) -> str:
    """Deterministic evidence summary (review follow-up #1).

    The system — never the model — extracts values, computes the day-over-day
    change, and flags materiality (|change| >= 30%).  The terminal judgment
    reads these lines instead of raw MCP JSON, removing the extraction and
    arithmetic burden from the model.
    """
    lines: list[str] = ["[确定性证据摘要 — 数值由系统计算，禁止编造]"]
    missing_data = False
    dimension_line: str | None = None
    for item in state.get("root_cause_evidence", []):
        purpose = str(item.get("purpose", ""))
        envelope = item.get("result", {}) or {}
        inner = envelope.get("result", envelope) or {}
        metric = str(item.get("metric_id") or "")
        if purpose == "trend_baseline" or purpose.startswith("cross_metric:"):
            points = (inner.get("trend_result") or {}).get("points") or []
            last_point = points[-1] if points else None
            marker = "（目标指标）" if purpose == "trend_baseline" else ""
            if not points or (isinstance(last_point, dict) and last_point.get("value") is None):
                lines.append(f"- {metric}{marker}: 数据不完整（最新观测缺失，无法评估当日变化）")
                missing_data = True
                continue
            change = _trend_change(points)
            if change is None or change[1] is None:
                lines.append(f"- {metric}{marker}: 数据不完整（有效点不足，无法计算变化）")
                missing_data = True
                continue
            last, base = change
            pct = (last - base) / abs(base) * 100 if base else 0.0
            flag = _materiality_flag(points, pct)
            lines.append(f"- {metric}{marker}: {base} → {last}（单日 {pct:+.1f}%）→ {flag}")
        elif purpose.startswith("metric_decomposition:"):
            typed_raw = item.get("investigation_evidence")
            typed_values = typed_raw.get("values", {}) if isinstance(typed_raw, dict) else {}
            delta = typed_values.get("total_delta")
            baseline = typed_values.get("baseline_value")
            current = typed_values.get("current_value")
            display = typed_values.get("display_components")
            factors = typed_values.get("factor_metrics")
            if (
                not all(_finite_number(value) for value in (delta, baseline, current))
                or not isinstance(display, list) or not display
            ):
                lines.append(f"- 指标分解({metric}): Evidence 缺少可验证的基期、当前期或分解分量")
                missing_data = True
                continue
            factor_names = [value for value in factors if isinstance(value, str)] if isinstance(factors, list) else []
            component_lines: list[str] = []
            for index, component in enumerate(display):
                if not isinstance(component, dict) or not _finite_number(component.get("value")):
                    continue
                component_id = str(component.get("component_id") or "")
                label = component_id
                if component_id in factor_names:
                    label = component_id
                elif component_id in {"numerator_effect", "denominator_effect"} and index < len(factor_names):
                    label = factor_names[index]
                value = float(component["value"])
                share = value / float(delta) * 100 if abs(float(delta)) > 1e-12 else None
                share_text = f"（占净变化 {share:+.1f}%）" if share is not None else ""
                component_lines.append(f"{label}={value:+.4g}{share_text}")
            if not component_lines:
                lines.append(f"- 指标分解({metric}): 没有可显示的有效分解分量")
                missing_data = True
                continue
            raw_components = typed_values.get("raw_components")
            interaction = next((
                component.get("value") for component in raw_components
                if isinstance(component, dict)
                and component.get("component_type") == "interaction_effect"
                and _finite_number(component.get("value"))
            ), None) if isinstance(raw_components, list) else None
            interaction_text = f"；原始交互项={float(interaction):+.4g}" if interaction is not None else ""
            display_method = str(typed_values.get("display_method") or "raw")
            method_id = str(typed_raw.get("method_id") or "registered_method") if isinstance(typed_raw, dict) else "registered_method"
            lines.append(
                f"- 指标分解({metric}/{method_id}): "
                f"{float(baseline):.4g} → {float(current):.4g}（净变化 {float(delta):+.4g}）；"
                f"{', '.join(component_lines)}{interaction_text}；展示口径={display_method}。"
                "这是统计贡献分解，不代表因果关系。"
            )
        elif purpose.startswith("dimension_contribution:"):
            dimension = purpose.split(":", 1)[1]
            typed_raw = item.get("investigation_evidence")
            typed_values = typed_raw.get("values", {}) if isinstance(typed_raw, dict) else {}
            if typed_values.get("attribution_status") == "ok":
                query_spec = typed_values.get("query", {})
                dimension = str(query_spec.get("dimension") or dimension)
                components = typed_values.get("components", [])
                total_delta = abs(float(typed_values.get("total_delta") or 0.0))
                best = max(
                    (components_item for components_item in components if isinstance(components_item, dict)),
                    key=lambda component: abs(float(component.get("delta_value") or 0.0)),
                    default=None,
                )
                if best is not None and total_delta > 1e-12:
                    ratio = abs(float(best.get("delta_value") or 0.0)) / total_delta
                    flag = "显著" if ratio >= 0.3 else "不显著"
                    dimension_line = (
                        f"- 维度贡献({dimension}): 最大单项 {best.get('dimension_value')} "
                        f"占总变化 {ratio * 100:.1f}% → {flag}"
                    )
                    continue
            contribution = inner.get("contribution_result") or inner
            current_items = contribution.get("current_items") or []
            baseline_items = {str(b.get("dimension_value")): b.get("value") for b in (contribution.get("baseline_items") or []) if isinstance(b, dict)}
            total_delta = abs(float(contribution.get("total_delta") or 0.0))
            best_ratio, best_value = 0.0, ""
            for entry in current_items:
                if not isinstance(entry, dict):
                    continue
                base_value = baseline_items.get(str(entry.get("dimension_value"))) or 0.0
                delta = abs(float(entry.get("value") or 0.0) - float(base_value))
                ratio = delta / total_delta if total_delta else 0.0
                if ratio >= best_ratio:
                    best_ratio, best_value = ratio, str(entry.get("dimension_value"))
            if best_ratio:
                flag = "显著" if best_ratio >= 0.3 else "不显著"
                dimension_line = f"- 维度贡献({dimension}): 最大单项 {best_value} 占总变化 {best_ratio * 100:.1f}% → {flag}"
            else:
                dimension_line = f"- 维度贡献({dimension}): 无可解析数据"
    for item in state.get("root_cause_evidence", []):
        purpose = str(item.get("purpose", ""))
        if not purpose.startswith(("distribution_analysis:", "anomaly_check:", "data_quality_check:")):
            continue
        typed = item.get("investigation_evidence")
        values = typed.get("values", {}) if isinstance(typed, dict) else {}
        analysis = values.get("series_analysis") or {}
        metric = str(item.get("metric_id") or "metric")
        status = str(analysis.get("status") or "unknown")
        if purpose.startswith("distribution_analysis:"):
            if status != "ok":
                lines.append(f"- 时间序列分布({metric}): 无可用描述统计（status={status}）")
                missing_data = True
                continue
            stats = analysis.get("statistics") or {}
            summary = ", ".join(
                f"{key}={stats[key]}"
                for key in ("mean", "median", "min", "max", "std")
                if isinstance(stats.get(key), (int, float))
            )
            lines.append(
                f"- 时间序列分布({metric}): n={analysis.get('count')}; "
                f"{summary or '描述统计不可用'}; min/max bucket="
                f"{analysis.get('min_bucket')}/{analysis.get('max_bucket')}"
            )
        elif purpose.startswith("anomaly_check:"):
            if status != "ok":
                lines.append(f"- 时间序列异常({metric}): 数据不足或粒度不支持（status={status}），未作异常判断")
                missing_data = True
                continue
            findings = analysis.get("anomalies") or []
            samples = ", ".join(
                f"{row.get('metric_date')} {row.get('direction')}={row.get('value')}"
                for row in findings[:5] if isinstance(row, dict)
            )
            lines.append(
                f"- 时间序列异常({metric}): 检出 {len(findings)} 个统计异常候选点"
                + (f"（{samples}）" if samples else "；当前规则未检出异常点")
                + "；这不表示业务原因或因果关系"
            )
        else:
            if status != "ok":
                lines.append(f"- 数据质量({metric}): 检查数据不足（status={status}）")
                missing_data = True
                continue
            lines.append(
                f"- 数据质量({metric}/{analysis.get('check_id')}): "
                f"{analysis.get('affected_bucket_count')}/{analysis.get('bucket_count')} 个时间桶命中检查"
            )
    if dimension_line:
        lines.append(dimension_line)
    if missing_data:
        lines.append("- 注意：存在数据不完整的指标，无法据此下因果结论")
    for gap in state.get("diagnostic_capability_gaps", []):
        if isinstance(gap, dict):
            text = _capability_gap_text(gap)
            if text:
                lines.append(f"- {text}")
    return "\n".join(lines)


# Shared terminal-judgment rules: used verbatim by the governed loop AND by
# the single-call baseline script, so both arms are scored on the same task.
TERMINAL_JUDGMENT_RULES = (
    "你是受控调查中的决策器。系统已计算证据数值与显著性；你不得自行计算、编造或改变口径。\n"
    "决策顺序：\n"
    "1) 先检查当前候选集与 investigation.open_branches。若存在未解决分支且有对应、未重复的合法候选，"
    "必须用 CALL_CANDIDATE 选择最能推进高优先级分支的候选；即使已有显著因子，也只说明找到了调查线索，"
    "不代表其下游驱动已查清，因此不能仅凭显著性提前 FINISH 或 ABSTAIN。\n"
    "2) 只有不存在值得继续的未解决分支/合法候选，或调查预算/权限/数据硬停止时，才作终止判断："
    "有显著证据时 FINISH，found_factor 填证据支持的指标 ID（维度归因时填维度/分组）；"
    "所有证据均不显著或数据不足时 ABSTAIN(insufficient_evidence)。存在多个显著因子时，"
    "优先选择与目标指标机制上更直接相关的证据。\n"
    "3) 若候选集非空，禁止用 FINISH/ABSTAIN 绕过未解决且有候选支撑的分支。\n"
    "除非输入包含经过校验的随机实验设计与可用实验数据，否则不得声称某指标或维度导致了目标变化；"
    "只描述观测到的统计关联或贡献，并明确它不证明因果。只输出一个 JSON action，必须用 action_type 判别。FINISH 形如："
    "{\"action_type\":\"FINISH\",\"found_factor\":\"<candidate_metric>\","
    "\"conclusion\":\"<引用摘要数值的统计关联或贡献解释>\","
    "\"evidence_ids\":[\"<已有证据ID>\"]}；"
    "ABSTAIN 形如：{\"action_type\":\"ABSTAIN\",\"reason\":\"insufficient_evidence\"}。\n"
)


def _conflicting_evidence_ids(investigation: dict[str, Any]) -> set[str]:
    return {
        str(evidence_id)
        for result in investigation.get("action_results", [])
        if isinstance(result, dict)
        for check in (result.get("validation_result") or {}).get("checks", [])
        if isinstance(check, dict) and check.get("status") == "failed"
        for evidence_id in check.get("conflicting_evidence_ids", [])
    }


def _completed_valid_evidence_ids(state: AgentState) -> list[str]:
    """Return only evidence linked to completed, validated actions in scope."""
    raw = state.get("investigation_state") or {}
    if not isinstance(raw, dict):
        return []
    target = raw.get("target") or {}
    scope_fingerprint = target.get("scope_fingerprint") if isinstance(target, dict) else None
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in raw.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    valid: list[str] = []
    conflicting_ids = _conflicting_evidence_ids(raw)
    for result in raw.get("action_results", []):
        if not isinstance(result, dict) or result.get("status") != "completed":
            continue
        validation = result.get("validation_result") or {}
        if not isinstance(validation, dict) or validation.get("status") != "valid":
            continue
        for evidence_id in result.get("evidence_ids", []):
            evidence = evidence_by_id.get(str(evidence_id))
            if (
                evidence is not None
                and str(evidence_id) not in conflicting_ids
                and evidence.get("source_action_id") == result.get("candidate_id")
                and evidence.get("scope_fingerprint") == scope_fingerprint
                and (not evidence.get("trace_id") or not state.get("trace_id")
                     or evidence.get("trace_id") == state.get("trace_id"))
                and str(evidence_id) not in valid
            ):
                valid.append(str(evidence_id))
    return valid[:50]


def validated_dimension_contribution_evidence(state: AgentState) -> list[dict[str, Any]]:
    """Return additive dimension results whose action, scope, period and totals validate.

    This is the shared trust boundary for answer projection and Critic context. It
    deliberately reads the server investigation ledger rather than the LLM summary
    or an answer card, so neither can promote unvalidated numbers into facts.
    """
    investigation = state.get("investigation_state") or {}
    if not isinstance(investigation, dict):
        return []
    target = investigation.get("target") or {}
    if not isinstance(target, dict):
        return []
    target_metric = str(target.get("metric_id") or "")
    scope_fingerprint = str(target.get("scope_fingerprint") or "")
    if not target_metric or not scope_fingerprint:
        return []

    valid_ids = set(_completed_valid_evidence_ids(state))
    conflicts = _conflicting_evidence_ids(investigation)
    actions = {
        str(item.get("candidate_id")): item
        for item in investigation.get("action_results", [])
        if isinstance(item, dict) and item.get("candidate_id")
    }
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in investigation.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    trusted_bundle_ids = {
        str(item.get("evidence_id"))
        for item in (state.get("evidence_bundles") or [])
        if isinstance(item, dict) and item.get("trusted") is True and item.get("evidence_id")
    }

    def finite_number(value: Any) -> bool:
        import math
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    def unit_for(metric_id: str, evidence: dict[str, Any]) -> str:
        # Registry metadata is the canonical unit source. Evidence may carry a
        # unit too, but an omitted unit is never inferred from the numeric value.
        try:
            from app.metrics.registry import MetricRegistry
            definition = MetricRegistry().get(metric_id)
            scope = "seller" if (target.get("canonical_scope") or {}).get("seller_id") else "platform"
            capability = definition.attribution.for_scope(scope)
            dimension = str(((evidence.get("values") or {}).get("query") or {}).get("dimension") or "")
            if (
                scope not in definition.implementations
                or "additive_change_v1" not in capability.method_ids
                or dimension not in capability.dimensions
            ):
                return ""
            registered_unit = str(definition.unit or "")
            evidence_unit = str(evidence.get("unit") or "")
            if evidence_unit and evidence_unit != registered_unit:
                return ""
            return registered_unit
        except Exception:  # noqa: BLE001 - an unknown registry unit blocks projection
            return str(evidence.get("unit") or "")

    records: list[dict[str, Any]] = []
    for evidence_id in sorted(valid_ids):
        evidence = evidence_by_id.get(evidence_id)
        if (
            not evidence
            or evidence_id not in trusted_bundle_ids
            or evidence_id in conflicts
            or evidence.get("evidence_type") != "dimension_contribution"
            or str(evidence.get("metric_id") or "") != target_metric
            or evidence.get("scope_fingerprint") != scope_fingerprint
        ):
            continue
        candidate_id = str(evidence.get("source_action_id") or "")
        action = actions.get(candidate_id) or {}
        if (
            not candidate_id
            or action.get("status") != "completed"
            or (action.get("validation_result") or {}).get("status") != "valid"
            or action.get("action_type") not in {"BREAKDOWN_DIMENSION", "DECOMPOSE_METRIC"}
        ):
            continue

        values = evidence.get("values") or {}
        query = values.get("query") or {}
        result = values.get("result") or {}
        comparison = (evidence.get("comparison") or {}).get("comparison") or query.get("comparison") or {}
        dimension = str(query.get("dimension") or result.get("dimension") or "")
        method = str(evidence.get("method_id") or "")
        if (
            not dimension
            or method != "additive_change_v1"
            or action.get("dimension") != dimension
            or not method
            or action.get("method_id") != method
            or query.get("metric_name") != target_metric
            or result.get("metric_name") != target_metric
            or result.get("dimension") != dimension
            or result.get("status") != "ok"
            or values.get("attribution_status") != "ok"
            or (evidence.get("data_quality") or {}).get("status") not in {"ok", "valid"}
        ):
            continue

        # Compare role-specific dates; do not sort dates or infer current from
        # left/right. Single-day and explicit period contracts are both accepted.
        current_date = comparison.get("current_date") or result.get("current_date")
        baseline_date = comparison.get("baseline_date") or result.get("baseline_date")
        current_start = comparison.get("current_start") or (comparison.get("current_period") or {}).get("start_date")
        current_end = comparison.get("current_end") or (comparison.get("current_period") or {}).get("end_date")
        baseline_start = comparison.get("baseline_start") or (comparison.get("baseline_period") or {}).get("start_date")
        baseline_end = comparison.get("baseline_end") or (comparison.get("baseline_period") or {}).get("end_date")
        expected_current = (target.get("current_start"), target.get("current_end"))
        expected_baseline = (target.get("baseline_start"), target.get("baseline_end"))
        if current_date and baseline_date:
            if expected_current[0] != expected_current[1] or expected_baseline[0] != expected_baseline[1]:
                continue
            if (str(current_date), str(current_date)) != tuple(map(str, expected_current)):
                continue
            if (str(baseline_date), str(baseline_date)) != tuple(map(str, expected_baseline)):
                continue
        elif not all((current_start, current_end, baseline_start, baseline_end)) or (
            str(current_start), str(current_end)
        ) != tuple(map(str, expected_current)) or (
            str(baseline_start), str(baseline_end)
        ) != tuple(map(str, expected_baseline)):
            continue

        baseline = values.get("baseline_value", result.get("baseline_total"))
        current = values.get("current_value", result.get("current_total"))
        delta = values.get("total_delta", result.get("total_delta"))
        unit = unit_for(target_metric, evidence)
        components = values.get("components") or []
        if not unit or not all(finite_number(item) for item in (baseline, current, delta)) or not components:
            continue
        tolerance = max(0.03, abs(float(delta)) * 1e-9)
        if abs((float(current) - float(baseline)) - float(delta)) > tolerance:
            continue
        normalized_components: list[dict[str, Any]] = []
        for component in components:
            if not isinstance(component, dict):
                normalized_components = []
                break
            group_delta = component.get("delta_value")
            group_current = component.get("current_value")
            group_baseline = component.get("baseline_value")
            if not all(finite_number(item) for item in (group_delta, group_current, group_baseline)):
                normalized_components = []
                break
            if abs((float(group_current) - float(group_baseline)) - float(group_delta)) > tolerance:
                normalized_components = []
                break
            normalized_components.append({
                "dimension_value": component.get("dimension_value"),
                "current_value": float(group_current),
                "baseline_value": float(group_baseline),
                "delta_value": float(group_delta),
                "contribution_ratio": component.get("contribution_ratio"),
            })
        if not normalized_components:
            continue
        residual = values.get("residual_value")
        component_delta_sum = sum(item["delta_value"] for item in normalized_components)
        if finite_number(residual):
            if abs(component_delta_sum + float(residual) - float(delta)) > tolerance:
                continue
        elif abs(component_delta_sum - float(delta)) > tolerance:
            continue

        records.append({
            "evidence_id": evidence_id,
            "metric_id": target_metric,
            "dimension": dimension,
            "method_id": method,
            "unit": unit,
            "baseline_value": float(baseline),
            "current_value": float(current),
            "total_delta": float(delta),
            "baseline_date": str(baseline_date or baseline_start),
            "current_date": str(current_date or current_start),
            "baseline_period": {"start_date": str(baseline_start or baseline_date), "end_date": str(baseline_end or baseline_date)},
            "current_period": {"start_date": str(current_start or current_date), "end_date": str(current_end or current_date)},
            "items": normalized_components,
            "residual_value": float(residual) if finite_number(residual) else None,
            "residual_type": values.get("residual_type"),
            "evidence_ids": [evidence_id],
            "scope_fingerprint": scope_fingerprint,
            "source_action_id": candidate_id,
            "analysis_nature": "descriptive_contribution",
        })
    return records


def _optional_data_gap_is_scoped_to_valid_decomposition(
    state: AgentState,
    action_result: Any,
    *,
    allow_budget_stop_context: bool = False,
) -> tuple[bool, str | None]:
    """Whether this failed follow-up is local to a branch already numerically explained.

    Do not infer optionality from ``required=False`` alone. The required plan
    floor must be complete, the failed action must be attached to a factor
    branch created by a validated in-scope decomposition, and no conflicting
    evidence or execution error may be present.
    """
    if (
        getattr(action_result, "failure_code", None) != "DATA_INSUFFICIENT"
        or getattr(action_result, "status", None) != "failed"
        or state.get("diagnostic_required_action")
    ):
        return False, None

    try:
        plan = DiagnosticPlan.model_validate(state.get("diagnostic_plan") or {})
    except Exception:  # noqa: BLE001 - legacy/malformed plans cannot relax stops
        return False, None
    step_states = state.get("diagnostic_step_state") or {}
    if any(
        step.required and step_states.get(step.step_id, "pending") != "completed"
        for step in plan.steps
    ):
        return False, None

    investigation = state.get("investigation_state") or {}
    if not isinstance(investigation, dict):
        return False, None
    stop = investigation.get("stop_reason") or {}
    if not isinstance(stop, dict):
        return False, None
    stop_code = stop.get("code")
    budget_context = False
    budget_contribution_gap = False
    budget_gap_marker = None
    if stop_code == "DATA_INSUFFICIENT":
        # Preserve the original stop-classification path. An optional local
        # gap is only scoped after all checks below also validate its lineage.
        if state.get("errors"):
            return False, None
    elif stop_code == "BUDGET_EXHAUSTED" and allow_budget_stop_context:
        progress = investigation.get("progress") or {}
        results = investigation.get("action_results") or []
        last_result = results[-1] if results else None
        action_id = str(getattr(action_result, "action_id", "") or "")
        candidate_id = str(getattr(action_result, "candidate_id", "") or "")
        candidate = next((
            item for item in investigation.get("candidate_actions", [])
            if isinstance(item, dict) and str(item.get("candidate_id") or "") == candidate_id
        ), None)
        candidate_action = (candidate or {}).get("action") or {}
        # Critic may preserve a local optional-gap classification after an
        # independent budget stop only when the persisted stop and latest
        # action jointly prove that exact transition. The gap label alone is
        # never sufficient.
        budget_context = bool(
            stop.get("initiator") == "budget"
            and investigation.get("remaining_budget") == 0
            and progress.get("remaining_budget") == 0
            and progress.get("investigation_progress") == "budget_exhausted"
            and isinstance(last_result, dict)
            and str(last_result.get("action_id") or "") == action_id
            and last_result.get("status") == "failed"
            and last_result.get("failure_code") == "DATA_INSUFFICIENT"
            and last_result.get("action_type") == getattr(action_result, "action_type", None)
            and last_result.get("metric_id") == getattr(action_result, "metric_id", None)
            and last_result.get("candidate_id") == getattr(action_result, "candidate_id", None)
            and last_result.get("branch_id") == getattr(action_result, "branch_id", None)
            and isinstance(candidate, dict)
            and candidate.get("branch_id") == getattr(action_result, "branch_id", None)
            and candidate_action.get("action_type") == getattr(action_result, "action_type", None)
            and candidate_action.get("metric_id") == getattr(action_result, "metric_id", None)
            and action_id
        )
        errors = state.get("errors") or []
        if any(
            not isinstance(error, dict)
            or error.get("code") not in {"token_budget_exceeded", "budget_exhausted"}
            for error in errors
        ):
            return False, None
        matching_gaps = [
            gap for gap in (state.get("diagnostic_candidate_data_gaps") or [])
            if isinstance(gap, dict) and str(gap.get("action_id") or "") == action_id
        ]
        if len(matching_gaps) != 1:
            return False, None
        budget_gap_marker = matching_gaps[0]
        action_id_match = any(
            isinstance(item, dict)
            and str(item.get("action_id") or "") == action_id
            and item.get("status") == "failed"
            and item.get("failure_code") == "DATA_INSUFFICIENT"
            and item.get("candidate_id") == getattr(action_result, "candidate_id", None)
            and item.get("action_type") == getattr(action_result, "action_type", None)
            and item.get("metric_id") == getattr(action_result, "metric_id", None)
            and item.get("branch_id") == getattr(action_result, "branch_id", None)
            for item in investigation.get("action_results", [])
        )
        matching_plan_candidate = (
            isinstance(candidate, dict)
            and candidate.get("branch_id") == getattr(action_result, "branch_id", None)
            and candidate_action.get("action_type") == getattr(action_result, "action_type", None)
            and candidate_action.get("metric_id") == getattr(action_result, "metric_id", None)
        )
        old_scoped_budget_context = bool(
            budget_context
            and budget_gap_marker.get("failure_scope") == "optional_deep_check"
            and budget_gap_marker.get("global_stop_code_at_recording") == "BUDGET_EXHAUSTED"
            and budget_gap_marker.get("global_stop_initiator_at_recording") == "budget"
            and budget_gap_marker.get("candidate_id") == getattr(action_result, "candidate_id", None)
            and budget_gap_marker.get("action_type") == getattr(action_result, "action_type", None)
            and budget_gap_marker.get("metric_id") == getattr(action_result, "metric_id", None)
        )
        # A terminal budget stop may be recorded after the local optional check
        # (and therefore may not be the last action or have exhausted action
        # count). Preserve that independent terminal reason while still
        # classifying the prior check locally, but only with an exact action /
        # candidate match; the gap marker alone never grants this classification.
        budget_contribution_gap = bool(
            stop.get("initiator") == "budget"
            and getattr(action_result, "action_type", None) == "CHECK_ANOMALY"
            and action_id_match
            and budget_gap_marker.get("action_type") == "CHECK_ANOMALY"
            and budget_gap_marker.get("metric_id") == getattr(action_result, "metric_id", None)
            and budget_gap_marker.get("candidate_id") == getattr(action_result, "candidate_id", None)
        )
        if not (old_scoped_budget_context or budget_contribution_gap):
            return False, None
        budget_context = old_scoped_budget_context or budget_contribution_gap
    else:
        return False, None
    if _conflicting_evidence_ids(investigation):
        return False, None
    if any(
        isinstance(item, dict)
        and item.get("failure_code") in {
            "INVALID_DATA", "AUTHORIZATION_BLOCKED", "EXECUTION_FAILED",
        }
        for item in investigation.get("action_results", [])
    ):
        return False, None

    target = investigation.get("target") or {}
    target_metric = str(target.get("metric_id") or "")
    failed_metric = str(getattr(action_result, "metric_id", "") or "")
    failed_branch_id = str(getattr(action_result, "branch_id", "") or "")
    if not target_metric or not failed_metric or not failed_branch_id:
        return False, None
    if budget_context and not budget_contribution_gap and str(budget_gap_marker.get("related_decomposition_evidence_id") or "") == "":
        return False, None
    valid_ids = set(_completed_valid_evidence_ids(state))
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in investigation.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    branches = investigation.get("branches", [])
    validated_contribution_ids = {
        item["evidence_id"]: item
        for item in validated_dimension_contribution_evidence(state)
    }
    target_period = {
        "current_period": {
            "start_date": target.get("current_start"),
            "end_date": target.get("current_end"),
        },
        "baseline_period": {
            "start_date": target.get("baseline_start"),
            "end_date": target.get("baseline_end"),
        },
    }
    for evidence_id in valid_ids:
        evidence = evidence_by_id.get(evidence_id)
        contribution = validated_contribution_ids.get(evidence_id)
        if (
            evidence is not None
            and contribution is not None
            and evidence.get("evidence_type") == "dimension_contribution"
            and failed_metric == target_metric
            and failed_metric == contribution.get("metric_id")
        ):
            if getattr(action_result, "action_type", None) == "CHECK_ANOMALY":
                from app.investigation.engine import MIN_ANOMALY_VALID_BUCKETS
                expected_start = str(target.get("baseline_start") or "")
                expected_end = str(target.get("current_end") or "")
                linked_series = []
                for series_id in valid_ids:
                    series = evidence_by_id.get(series_id)
                    if (
                        not series
                        or series.get("evidence_type") != "trend"
                        or str(series.get("metric_id") or "") != failed_metric
                        or series.get("scope_fingerprint") != target.get("scope_fingerprint")
                    ):
                        continue
                    series_values = series.get("values") or {}
                    period = (series_values.get("query") or {}).get("period") or {}
                    points = ((series_values.get("result") or {}).get("points") or [])
                    numeric_count = sum(
                        1 for point in points
                        if isinstance(point, dict)
                        and isinstance(point.get("value"), (int, float))
                        and not isinstance(point.get("value"), bool)
                    )
                    if (
                        str(period.get("start_date") or "") == expected_start
                        and str(period.get("end_date") or "") == expected_end
                        and numeric_count < MIN_ANOMALY_VALID_BUCKETS
                    ):
                        linked_series.append((series_id, numeric_count))
                if not linked_series:
                    continue
            linked_dimension_branch = any(
                isinstance(branch, dict)
                and branch.get("branch_id") == failed_branch_id
                and branch.get("parent_evidence_id") == evidence_id
                and branch.get("branch_type") == "dimension_contribution"
                and branch.get("metric_id") == failed_metric
                and branch.get("dimension") == contribution.get("dimension")
                for branch in branches
            )
            if linked_dimension_branch:
                if budget_context and not budget_contribution_gap and budget_gap_marker.get("related_decomposition_evidence_id") != evidence_id:
                    continue
                return True, evidence_id
        if (
            evidence is None
            or evidence.get("evidence_type") != "metric_decomposition"
            or str(evidence.get("metric_id") or "") != target_metric
            or evidence.get("scope_fingerprint") != target.get("scope_fingerprint")
        ):
            continue
        values = evidence.get("values") or {}
        quality = evidence.get("data_quality") or {}
        comparison = evidence.get("comparison") or {}
        if (
            failed_metric not in (values.get("factor_metrics") or [])
            or quality.get("status") != "valid"
            or quality.get("input_series_complete") is not True
            or any(
                (comparison.get(period) or {}).get(field) != target_period[period][field]
                for period in target_period
                for field in ("start_date", "end_date")
            )
        ):
            continue
        linked_factor_branch = any(
            isinstance(branch, dict)
            and branch.get("branch_id") == failed_branch_id
            and branch.get("parent_evidence_id") == evidence_id
            and branch.get("branch_type") == "metric_effect"
            and branch.get("metric_id") == failed_metric
            for branch in branches
        )
        if linked_factor_branch:
            if budget_context and budget_gap_marker.get("related_decomposition_evidence_id") != evidence_id:
                continue
            return True, evidence_id
    return False, None


def _record_candidate_data_gap(
    state: AgentState,
    investigation: Any,
    action_result: Any,
    *,
    fallback_candidate: Any,
    failed_query: Any | None = None,
    failure_detail: str,
) -> bool:
    """Persist one insufficient action and apply its bounded stop semantics."""
    from app.investigation.engine import continue_after_candidate_data_gap

    state["investigation_state"] = investigation.model_dump(mode="json")
    scoped_optional_gap, decomposition_evidence_id = (
        _optional_data_gap_is_scoped_to_valid_decomposition(state, action_result)
    )
    if scoped_optional_gap and investigation.remaining_budget <= 0:
        # The failed check remains a local action result. The run still stops
        # for its independent exhausted budget, never by reclassifying the
        # already-valid decomposition as globally data-insufficient.
        from app.investigation.contracts import StopReason

        investigation.stop_reason = StopReason(
            code="BUDGET_EXHAUSTED",
            detail=(
                "Investigation action budget exhausted after an optional check; "
                "its data gap does not invalidate the linked decomposition."
            ),
            initiator="budget",
        )
        investigation.progress = investigation.progress.model_copy(update={
            "remaining_budget": investigation.remaining_budget,
            "investigation_progress": "budget_exhausted",
        })
        may_continue = False
    else:
        # With remaining budget the shared transition releases only this
        # recorded candidate failure. Candidate rebuilding excludes its
        # executed fingerprint and terminates when no alternatives remain.
        may_continue = continue_after_candidate_data_gap(investigation, action_result)

    gap = {
        "action_id": action_result.action_id,
        "candidate_id": action_result.candidate_id or fallback_candidate.candidate_id,
        "action_type": action_result.action_type or fallback_candidate.action.action_type,
        "metric_id": action_result.metric_id or fallback_candidate.action.metric_id,
        "dimension": action_result.dimension or getattr(fallback_candidate.action, "dimension", None),
        "detail": failure_detail,
        "continuing_with_alternatives": may_continue,
    }
    if scoped_optional_gap:
        gap.update({
            "failure_scope": "optional_deep_check",
            "related_decomposition_evidence_id": decomposition_evidence_id,
        })
        if investigation.stop_reason and investigation.stop_reason.code == "BUDGET_EXHAUSTED":
            gap.update({
                "global_stop_code_at_recording": investigation.stop_reason.code,
                "global_stop_initiator_at_recording": investigation.stop_reason.initiator,
            })
    gaps = state.setdefault("diagnostic_candidate_data_gaps", [])
    if gap not in gaps:
        gaps.append(gap)

    if state.get("diagnostic_required_action"):
        failed_kind = (
            "trend" if getattr(failed_query, "period", None) is not None
            else "dimension_contribution"
            if getattr(failed_query, "comparison", None) is not None
            else None
        )
        if failed_kind:
            step_state = state.setdefault("diagnostic_step_state", {})
            for step in DiagnosticPlan.model_validate(state["diagnostic_plan"]).steps:
                if (
                    step.required
                    and _step_kind(step) == failed_kind
                    and step_state.get(step.step_id, "pending") == "pending"
                ):
                    step_state[step.step_id] = "data_insufficient"
                    gap["required_step_id"] = step.step_id
                    break
    return may_continue


def _validated_metric_decomposition_terminal(
    state: AgentState,
    *,
    allow_partial_projection: bool = False,
    allow_completed_projection: bool = False,
) -> FinishAction | None:
    """Build a bounded numeric conclusion from a fully validated registered decomposition."""
    errors = [item for item in (state.get("errors") or []) if isinstance(item, dict)]
    if errors and not allow_partial_projection:
        return None
    if any(item.get("code") != "token_budget_exceeded" for item in errors):
        return None
    investigation = _ensure_graph_investigation_state(state)
    if investigation is None:
        return None
    stop = investigation.stop_reason
    if stop is not None:
        allowed_partial_stop = allow_partial_projection and stop.code in {"DATA_INSUFFICIENT", "BUDGET_EXHAUSTED"}
        allowed_completed_stop = allow_completed_projection and stop.code == "ENOUGH_EVIDENCE"
        if not (allowed_partial_stop or allowed_completed_stop):
            return None
        try:
            plan = DiagnosticPlan.model_validate(state.get("diagnostic_plan") or {})
        except Exception:  # noqa: BLE001 - malformed plans cannot relax the evidence floor
            return None
        step_states = state.get("diagnostic_step_state") or {}
        if any(
            step.required and step_states.get(step.step_id, "pending") != "completed"
            for step in plan.steps
        ):
            return None
        if _conflicting_evidence_ids(investigation.model_dump(mode="json")):
            return None
        if any(
            result.failure_code in {"INVALID_DATA", "AUTHORIZATION_BLOCKED", "EXECUTION_FAILED"}
            for result in investigation.action_results
        ):
            return None
        if stop.code == "DATA_INSUFFICIENT" and not any(
            isinstance(item, dict) and item.get("failure_scope") == "optional_deep_check"
            for item in (state.get("diagnostic_candidate_data_gaps") or [])
        ):
            return None
    raw = state.get("investigation_state") or {}
    if not isinstance(raw, dict):
        return None
    from app.auth.context import AuthorizationDenied, build_authorization_context
    from app.metrics.registry import get_registry
    from app.tenancy import current_tenant_context

    context = current_tenant_context()
    if context is None:
        return None
    scope = state.get("controlled_analysis_scope") or {}
    scope_name = "seller" if scope else "platform"
    try:
        authz = build_authorization_context(
            context,
            allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
            allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
            metric_scope_type=scope_name,
        )
        authz.require_action("analyze")
        target = investigation.target
        requested_metric = str(state.get("diagnostic_target_metric") or "").strip().casefold()
        if requested_metric and requested_metric != target.metric_id.casefold():
            return None
        definition = get_registry().get(target.metric_id)
        capability = definition.attribution.for_scope(scope_name)
        valid_ids = set(_completed_valid_evidence_ids(state))
        evidence_by_id = {item.evidence_id: item for item in investigation.evidence}
        for result in reversed(investigation.action_results):
            if (
                result.action_type != "DECOMPOSE_METRIC"
                or result.metric_id != target.metric_id
                or result.status != "completed"
                or result.validation_result.status != "valid"
                or not result.method_id
            ):
                continue
            specs = capability.factor_specs.get(result.method_id, ())
            factors = [item.factor_id for item in specs]
            ratio_contract = definition.period_aggregation if result.method_id == "symmetric_ratio_v1" else None
            if ratio_contract is not None:
                factors = [ratio_contract.numerator_metric, ratio_contract.denominator_metric]
            if result.method_id not in capability.method_ids or len(factors) < 2:
                continue
            source_metric_ids = sorted({
                source_id for item in specs for source_id in item.source_metric_ids
            })
            if ratio_contract is not None:
                source_metric_ids = list(factors)
            required_input_metric_ids = set(source_metric_ids)
            if ratio_contract is not None:
                required_input_metric_ids.add(target.metric_id)
            for factor_id in factors:
                if get_registry().get(factor_id).period_aggregation is not None:
                    required_input_metric_ids.add(factor_id)
            authz.require_metrics([target.metric_id, *factors, *source_metric_ids])
            if scope and not {target.metric_id, *factors, *source_metric_ids} <= set(scope.get("allowed_metrics") or []):
                continue
            evidence = next((
                evidence_by_id[evidence_id]
                for evidence_id in result.evidence_ids
                if evidence_id in valid_ids and evidence_id in evidence_by_id
            ), None)
            if evidence is None:
                continue
            values = evidence.values
            if (
                evidence.evidence_type != "metric_decomposition"
                or evidence.metric_id != target.metric_id
                or evidence.scope_fingerprint != target.scope_fingerprint
                or evidence.source_action_id != result.candidate_id
                or evidence.method_id != result.method_id
                or evidence.authorized_metric_provenance != "tool_gateway_policy_validated"
                or values.get("factor_metrics") != factors
                or values.get("residual_explainable") is not True
            ):
                continue
            if evidence.trace_id and state.get("trace_id") and evidence.trace_id != state.get("trace_id"):
                continue
            if (
                evidence.comparison.get("baseline_period") != {
                    "start_date": target.baseline_start.isoformat(),
                    "end_date": target.baseline_end.isoformat(),
                }
                or evidence.comparison.get("current_period") != {
                    "start_date": target.current_start.isoformat(),
                    "end_date": target.current_end.isoformat(),
                }
            ):
                continue
            input_ids = values.get("input_evidence_ids")
            if not isinstance(input_ids, list) or not input_ids:
                continue
            input_evidence = [evidence_by_id.get(str(item)) for item in input_ids]
            if (
                any(item is None or item.evidence_id not in valid_ids for item in input_evidence)
                or len(input_evidence) != len(required_input_metric_ids)
            ):
                continue
            decomposition_dates = _registered_decomposition_dates(state)
            if decomposition_dates:
                input_start, input_end = decomposition_dates
            else:
                input_start, input_end = _diagnostic_dates(state)
            if scope and not decomposition_dates:
                input_start = str(scope.get("default_start_date") or input_start)
                input_end = str(scope.get("default_end_date") or input_end)
            if any(
                item is None
                or item.evidence_type != "trend"
                or item.scope_fingerprint != target.scope_fingerprint
                or item.authorized_metric_provenance != "tool_gateway_policy_validated"
                or (item.comparison.get("period") or {}).get("start_date") != input_start
                or (item.comparison.get("period") or {}).get("end_date") != input_end
                or (item.comparison.get("period") or {}).get("granularity") != "day"
                for item in input_evidence
            ):
                continue
            if {
                item.metric_id for item in input_evidence
                if item is not None
            } != required_input_metric_ids:
                continue
            if any(_trusted_daily_series(state, metric_id) is None for metric_id in required_input_metric_ids):
                continue
            baseline, current, delta = (
                values.get("baseline_value"), values.get("current_value"), values.get("total_delta"),
            )
            components = values.get("display_components")
            if (
                any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in (baseline, current, delta))
                or not isinstance(components, list) or len(components) != len(factors)
            ):
                continue
            component_by_id = {
                str(item.get("component_id")): item.get("value")
                for item in components if isinstance(item, dict)
            }
            if ratio_contract is not None:
                if set(component_by_id) != {"numerator_effect", "denominator_effect"}:
                    continue
                component_by_id = {
                    factors[0]: component_by_id["numerator_effect"],
                    factors[1]: component_by_id["denominator_effect"],
                }
            if set(component_by_id) != set(factors) or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in component_by_id.values()
            ):
                continue
            if result.method_id == "product_shapley_v1":
                baseline_factors = values.get("baseline_factor_values")
                current_factors = values.get("current_factor_values")
                if (
                    not isinstance(baseline_factors, dict)
                    or not isinstance(current_factors, dict)
                    or set(baseline_factors) != set(factors)
                    or set(current_factors) != set(factors)
                    or any(
                        not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        for value in [*baseline_factors.values(), *current_factors.values()]
                    )
                    or not math.isclose(math.prod(float(baseline_factors[item]) for item in factors), float(baseline), rel_tol=1e-8, abs_tol=0.01)
                    or not math.isclose(math.prod(float(current_factors[item]) for item in factors), float(current), rel_tol=1e-8, abs_tol=0.01)
                ):
                    continue
            elif ratio_contract is not None:
                # Only additive sufficient statistics may establish a ratio;
                # daily distinct counts cannot be summed across periods.
                if any(get_registry().get(factor).aggregation != "sum" for factor in factors):
                    continue
                expected_values = []
                for period_start, period_end in (
                    (target.baseline_start, target.baseline_end),
                    (target.current_start, target.current_end),
                ):
                    numerator = _period_series_total(_trusted_daily_series(state, factors[0]), period_start, period_end)
                    denominator = _period_series_total(_trusted_daily_series(state, factors[1]), period_start, period_end)
                    if numerator is None or denominator is None or denominator <= 0:
                        break
                    expected_values.append(numerator / denominator * ratio_contract.multiplier)
                if len(expected_values) != 2 or not all(math.isclose(actual, float(observed), rel_tol=1e-8, abs_tol=0.01)
                    for actual, observed in zip(expected_values, (baseline, current))):
                    continue
            if not math.isclose(sum(float(value) for value in component_by_id.values()), float(delta), rel_tol=1e-8, abs_tol=0.03):
                continue
            registry = get_registry()
            rendered = []
            change_unit = "个百分点" if definition.unit == "%" else definition.unit
            for factor_id in factors:
                factor_name = registry.get(factor_id).display_name or factor_id
                rendered.append(f"{factor_name}分解贡献 {float(component_by_id[factor_id]):,.2f}{change_unit}")
            period = (
                f"基线 {target.baseline_start.isoformat()} 至 {target.baseline_end.isoformat()}，"
                f"当前 {target.current_start.isoformat()} 至 {target.current_end.isoformat()}"
            )
            factor_id = _single_factor_explaining_delta(component_by_id, float(delta))
            if factor_id is not None:
                factor_name = registry.get(factor_id).display_name or factor_id
                direction = "数值上升" if float(delta) > 0 else "数值下降"
                decomposition_statement = (
                    f"在该指标定义和期间内，{factor_name}变化解释了{definition.display_name or target.metric_id}的{direction}"
                )
            else:
                decomposition_statement = (
                    "该结果说明指标口径下的数值变化由注册因子分解"
                )
            conclusion = (
                f"按已注册的 {result.method_id} 对{definition.display_name or target.metric_id}作数值分解（{period}）："
                f"指标从 {float(baseline):,.2f}{definition.unit} 变为 {float(current):,.2f}{definition.unit}，"
                f"变化 {float(delta):+,.2f}{change_unit}；" + "；".join(rendered)
                + f"。各分解项与指标总变化通过对账。{decomposition_statement}；"
                "这不说明因子变化背后的业务原因。"
            )
            return FinishAction(
                action_type="FINISH", conclusion=conclusion,
                evidence_ids=[evidence.evidence_id],
                residual_uncertainty="注册分解因子的业务变化原因尚未确认。",
                found_factor=factor_id,
            )
    except (AuthorizationDenied, KeyError, TypeError, ValueError):
        return None
    return None


def _metric_decomposition_projection(
    state: AgentState,
    *,
    allow_partial_projection: bool,
    allow_completed_projection: bool = False,
) -> dict[str, Any] | None:
    """Return structured numeric facts only after the terminal validator accepts them."""
    raw = state.get("investigation_state") or {}
    if not isinstance(raw, dict) or not any(
        isinstance(item, dict)
        and item.get("action_type") == "DECOMPOSE_METRIC"
        and item.get("status") == "completed"
        for item in raw.get("action_results", [])
    ):
        return None
    terminal = _validated_metric_decomposition_terminal(
        state,
        allow_partial_projection=allow_partial_projection,
        allow_completed_projection=allow_completed_projection,
    )
    if terminal is None or len(terminal.evidence_ids) != 1:
        return None
    evidence = next((
        item for item in raw.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id") == terminal.evidence_ids[0]
    ), None)
    target = raw.get("target") or {}
    if evidence is None or not isinstance(target, dict):
        return None
    values = evidence.get("values") or {}
    metric_id = str(target.get("metric_id") or "")
    target_scope_fingerprint = str(target.get("scope_fingerprint") or "")
    selected_comparison = evidence.get("comparison") or {}
    for other in raw.get("evidence", []):
        if not isinstance(other, dict) or other.get("evidence_id") == evidence.get("evidence_id"):
            continue
        if (
            other.get("evidence_type") != "metric_decomposition"
            or str(other.get("metric_id") or "") != metric_id
            or str(other.get("scope_fingerprint") or "") != target_scope_fingerprint
            or any(
                (other.get("comparison") or {}).get(period_key)
                != selected_comparison.get(period_key)
                for period_key in ("baseline_period", "current_period")
            )
        ):
            continue
        other_values = other.get("values") or {}
        for key in ("baseline_value", "current_value", "total_delta"):
            left, right = values.get(key), other_values.get(key)
            if (
                not isinstance(left, (int, float)) or isinstance(left, bool)
                or not isinstance(right, (int, float)) or isinstance(right, bool)
                or not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=0.01)
            ):
                return None
    factors = values.get("factor_metrics")
    components = values.get("display_components")
    if not metric_id or not isinstance(factors, list) or not isinstance(components, list):
        return None
    from app.metrics.registry import get_registry

    registry = get_registry()
    definition = registry.get(metric_id)
    component_by_id = {
        str(item.get("component_id")): item
        for item in components if isinstance(item, dict)
    }
    if evidence.get("method_id") == "symmetric_ratio_v1":
        aggregation = definition.period_aggregation
        identities = values.get("factor_identities")
        identity_items = identities if isinstance(identities, list) else []
        identity_by_role = {
            str(item.get("role_id")): str(item.get("metric_id"))
            for item in identity_items if isinstance(item, dict)
        }
        expected = (
            {
                "numerator_effect": aggregation.numerator_metric,
                "denominator_effect": aggregation.denominator_metric,
            }
            if aggregation is not None and aggregation.kind == "weighted_ratio" else {}
        )
        # Legacy typed evidence has no explicit identity list. It remains
        # renderable only when the registry's exact ordered factors and the
        # canonical component role IDs jointly form an unambiguous mapping.
        # Public tool results use AttributionResult.factor_identities and are
        # required to carry that mapping at their boundary.
        identities_valid = (
            identity_by_role == expected and len(identity_items) == 2
            if identity_items else True
        )
        if (
            not expected or not identities_valid
            or factors != [expected["numerator_effect"], expected["denominator_effect"]]
            or not {"numerator_effect", "denominator_effect"} <= set(component_by_id)
        ):
            return None
        registered_role_map = identity_by_role or expected
        component_by_id = {
            registered_role_map["numerator_effect"]: component_by_id["numerator_effect"],
            registered_role_map["denominator_effect"]: component_by_id["denominator_effect"],
        }
    rendered_factors = []
    for factor_id in factors:
        component = component_by_id.get(str(factor_id))
        value = component.get("value") if isinstance(component, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        factor_definition = registry.get(str(factor_id))
        rendered_factors.append({
            "metric_id": str(factor_id),
            "display_name": factor_definition.display_name or str(factor_id),
            "contribution": float(value),
            "unit": str(component.get("unit") or definition.unit),
        })
    baseline = values.get("baseline_value")
    current = values.get("current_value")
    delta = values.get("total_delta")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in (baseline, current, delta)):
        return None
    change_unit = rendered_factors[0]["unit"] if rendered_factors else definition.unit
    if definition.unit == "%":
        change_unit = "percentage_points"
    return {
        "metric_id": metric_id,
        "display_name": definition.display_name or metric_id,
        "unit": definition.unit,
        "baseline_value": float(baseline),
        "current_value": float(current),
        "delta_value": float(delta),
        "change_unit": change_unit,
        "baseline_period": dict(evidence.get("comparison", {}).get("baseline_period") or {}),
        "current_period": dict(evidence.get("comparison", {}).get("current_period") or {}),
        "method_id": str(evidence.get("method_id") or ""),
        "factors": rendered_factors,
        "evidence_ids": [str(evidence.get("evidence_id"))],
        "analysis_nature": "registered_numeric_decomposition",
        "business_cause_confirmed": False,
    }


def _single_factor_explaining_delta(
    components: dict[str, float],
    total_delta: float,
) -> str | None:
    """Return a factor only when it alone accounts for a non-negligible delta.

    Callers must first validate that these values came from the registered,
    authorized decomposition for the active scope and period. The tolerance is
    in the metric's displayed units and scales slightly with larger changes.
    """
    if not math.isfinite(total_delta):
        return None
    tolerance = max(0.03, abs(total_delta) * 1e-9)
    if abs(total_delta) <= tolerance or not components:
        return None
    if any(not math.isfinite(float(value)) for value in components.values()):
        return None
    candidates = [
        factor_id
        for factor_id, value in components.items()
        if abs(float(value) - total_delta) <= tolerance
        and all(
            other_id == factor_id or abs(float(other_value)) <= tolerance
            for other_id, other_value in components.items()
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _llm_decide_action(state: AgentState) -> Any:
    from app.llm.client import llm_agent_action
    from app.investigation.contracts import branch_share_for_priority

    plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
    if int(state.get("diagnostic_round", 0)) > plan.budget.max_rounds:
        state["termination_initiator"] = "budget"
        return _fallback_action(state)
    if state.get("diagnostic_fallback"):
        terminal = _validated_metric_decomposition_terminal(state)
        if terminal is not None:
            state["diagnostic_backend_terminal"] = "validated_metric_decomposition"
            return terminal
    findings = _findings_block(state)
    candidate_catalog = _candidate_tool_catalog(state)
    investigation_raw = state.get("investigation_state") or {}
    stop = investigation_raw.get("stop_reason") if isinstance(investigation_raw, dict) else None
    hard_stop = str((stop or {}).get("code") or "") if isinstance(stop, dict) else ""
    if hard_stop in {
        "AUTHORIZATION_BLOCKED", "CAPABILITY_GAP", "DATA_INSUFFICIENT",
        "USER_SCOPE_LIMIT", "BUDGET_EXHAUSTED", "NO_HIGH_VALUE_ACTION",
    }:
        return AbstainAction(
            action_type="ABSTAIN",
            reason="insufficient_evidence",
            supporting_evidence_ids=[],
        )
    decomposition_request = state.get("registered_decomposition_request") or {}
    if decomposition_request.get("status") == "ready":
        # A registered numeric method is an executable contract, not an
        # exploratory hypothesis for the model to reinterpret. Select only
        # the current server-created candidate that matches that contract;
        # policy_execute still rechecks identity, scope, tool, and factors.
        from app.investigation.contracts import DecomposeMetricAction
        from app.investigation.engine import select_candidate

        investigation = _ensure_graph_investigation_state(state)
        request_target = str(decomposition_request.get("target_metric_id") or "")
        request_method = str(decomposition_request.get("method_id") or "")
        matching = [
            item for item in (investigation.candidate_actions if investigation else [])
            if item.candidate_id in candidate_catalog
            and isinstance(item.action, DecomposeMetricAction)
            and item.action.metric_id == request_target
            and item.action.method_id == request_method
        ]
        candidate_set = investigation.candidate_set if investigation else None
        if len(matching) == 1 and candidate_set is not None:
            candidate = select_candidate(
                investigation, matching[0].candidate_id,
                candidate_set_id=candidate_set.candidate_set_id,
            )
            if candidate is not None:
                state["diagnostic_selected_candidate_id"] = candidate.candidate_id
                state["diagnostic_decision_source"] = "backend"
                state["investigation_state"] = investigation.model_dump(mode="json")
                return candidate_catalog[candidate.candidate_id]
        _set_investigation_stop_reason(
            state, "CAPABILITY_GAP",
            "The registered decomposition candidate is unavailable or ambiguous after server-side validation.",
        )
        state["diagnostic_decision_source"] = "backend"
        return AbstainAction(
            action_type="ABSTAIN", reason="insufficient_evidence",
            supporting_evidence_ids=_completed_valid_evidence_ids(state),
        )
    candidate_prompt_items = []
    investigation = _ensure_graph_investigation_state(state)
    typed_candidates = {
        item.candidate_id: item
        for item in (investigation.candidate_actions if investigation else [])
    }
    for candidate_id, action in candidate_catalog.items():
        candidate = typed_candidates.get(candidate_id)
        features = candidate.features if candidate is not None else None
        selection_context = ({
            "branch_id": candidate.branch_id,
            "target_branch_share": features.unresolved_target_share,
            "relevance_score": features.relevance_score,
            "estimated_query_cost": features.estimated_query_cost,
            "data_availability": features.data_availability,
            "already_explored": features.already_explored,
            "evidence_gap": features.evidence_gap,
            "backend_priority_score": features.rule_score,
        } if features is not None else {})
        if isinstance(action, ExecuteDecompositionAction):
            resolved = candidate.action if candidate else None
            candidate_prompt_items.append({
                "candidate_id": candidate_id,
                "action_type": "DECOMPOSE_METRIC",
                "metric_id": getattr(resolved, "metric_id", None),
                "method_id": getattr(resolved, "method_id", None),
                "factor_metric_ids": getattr(resolved, "factor_metric_ids", []),
                "execution": "server-side deterministic computation from existing authorized evidence",
                "selection_context": selection_context,
            })
        else:
            candidate_prompt_items.append({
                "candidate_id": candidate_id, "action_type": "CALL_TOOL",
                "tool_name": action.tool_name, "metric_id": action.metric_name,
                "dimension": action.dimension,
                "period": action.period.model_dump(mode="json") if action.period else None,
                "comparison": action.comparison.model_dump(mode="json") if action.comparison else None,
                "selection_context": selection_context,
            })
    candidate_prompt = json.dumps(candidate_prompt_items, ensure_ascii=False)
    active_candidate_set = investigation.candidate_set if investigation else None
    candidate_set_id = active_candidate_set.candidate_set_id if active_candidate_set else ""
    open_branches = sorted(
        (
            branch for branch in investigation.branches
            if branch.status in {"open", "investigating"}
        ),
        key=lambda branch: (
            -branch_share_for_priority(branch),
            branch.depth,
            branch.branch_id,
        ),
    ) if investigation is not None else []
    decision_context = {
        "plan": state["diagnostic_plan"],
        "observations": state.get("observations", []),
        "policy": state.get("policy_decision", {}),
        "investigation": ({
            "target_metric_id": investigation.target.metric_id,
            "current_focus_branch_id": investigation.current_focus_branch_id,
            "open_branches": [branch.model_dump(mode="json") for branch in open_branches[:8]],
            "omitted_open_branch_count": max(0, len(open_branches) - 8),
            "remaining_action_budget": investigation.remaining_budget,
            "progress": investigation.progress.model_dump(mode="json"),
        } if investigation is not None else {}),
    }
    prompt = (
        TERMINAL_JUDGMENT_RULES
        + f"当前候选集 ID：{candidate_set_id}。如需追加数据，只能输出 CALL_CANDIDATE，"
        "并同时原样返回 candidate_set_id 与 candidate_id；"
        "不得自行构造 tool_name、metric_id、dimension、日期、筛选或范围。"
        "候选为空时只能 FINISH、ABSTAIN 或 REPLAN。候选如下：\n"
        "Use selection_context and investigation.open_branches to choose an action that advances the largest supported unresolved branch without repeating work. Backend priority scores are heuristic ordering signals, not measured effects or predicted results.\n"
        + candidate_prompt + "\n"
        "FINISH 的 evidence_ids 必须引用下面观察里出现过的真实证据 ID。\n"
        + findings
        + "\n[计划与原始观察]\n"
        + json.dumps(decision_context, ensure_ascii=False)
    )
    action: Any = None
    last_exc: Exception | None = None
    diagnostic_user_message = _diagnostic_context_user_message(state, system_prompt=prompt)
    for _attempt in range(2):  # one retry: an invalid action must not silently become an abstention
        try:
            raw = llm_agent_action(prompt, diagnostic_user_message)
            if isinstance(raw, dict) and isinstance(raw.get("action"), dict):
                raw = raw["action"]
            if isinstance(raw, dict) and raw.get("action_type") in {"CALL_TOOL", "PARALLEL_FACTS"}:
                raise ValueError(
                    "Model-generated tool queries are not accepted; select a current server candidate by ID."
                )
            if isinstance(raw, dict) and raw.get("action_type") == "CALL_CANDIDATE":
                candidate_action = CallCandidateAction.model_validate(raw)
                from app.investigation.engine import select_candidate
                investigation = _ensure_graph_investigation_state(state)
                selected = select_candidate(
                    investigation, candidate_action.candidate_id,
                    candidate_set_id=candidate_action.candidate_set_id,
                ) if investigation else None
                action = candidate_catalog.get(candidate_action.candidate_id) if selected else None
                if action is not None:
                    state["diagnostic_selected_candidate_id"] = candidate_action.candidate_id
                    state["diagnostic_decision_source"] = "agent"
                    state["investigation_state"] = investigation.model_dump(mode="json")
                else:
                    raise ValueError("candidate_set_id/candidate_id did not resolve to the current server candidate set")
            if action is not None:
                break
            raw = _normalize_action_fields(raw, state)
            action = _ACTION_ADAPTER.validate_python(raw) if raw else None
        except Exception as exc:
            last_exc = exc
            action = None
        if action is not None:
            break
        if _attempt == 0:
            prompt = (
                "你上一次输出的 action 没有通过强类型校验，错误如下：\n"
                f"{last_exc}\n"
                f"当前候选集 ID：{candidate_set_id}。请依据本轮开放分支和下面重新附上的候选清单重试。"
                "若候选集非空且仍有未解决分支，必须输出 CALL_CANDIDATE 并原样复制 candidate_set_id/candidate_id；"
                "不得此时 FINISH 或 ABSTAIN。CALL_TOOL 与 PARALLEL_FACTS 都不是合法的模型查询选择格式。"
                "只有没有待查候选或存在硬停止时才可终止；FINISH 必须引用已有 evidence_ids，ABSTAIN 仅写 reason。\n"
                "[当前候选清单]\n" + candidate_prompt + "\n"
                "[确定性证据摘要]\n" + findings + "\n"
                + json.dumps(decision_context, ensure_ascii=False)
            )
    if action is None:
        failure_class = _classify_validation_error(last_exc) if last_exc else "empty_response"
        state.setdefault("diagnostic_llm_failures", []).append(
            {"stage": "decide", "attempt": 2, "class": failure_class, "recovered": False,
             "message": str(last_exc) if last_exc else ""})
        from app.investigation.engine import select_candidate

        investigation = _ensure_graph_investigation_state(state)
        fallback_candidate = None
        if investigation is not None and investigation.candidate_set is not None:
            candidate_ids = list(investigation.candidate_set.candidate_ids)
            candidate_ids.sort(key=lambda candidate_id: not isinstance(
                candidate_catalog.get(candidate_id), ExecuteDecompositionAction,
            ))
            for candidate_id in candidate_ids:
                selected = select_candidate(
                    investigation, candidate_id,
                    candidate_set_id=investigation.candidate_set.candidate_set_id,
                )
                if selected is not None and candidate_id in candidate_catalog:
                    fallback_candidate = selected
                    break
        if fallback_candidate is not None:
            state["diagnostic_selected_candidate_id"] = fallback_candidate.candidate_id
            state["diagnostic_decision_source"] = "workflow_fallback"
            state["diagnostic_fallback_reason"] = failure_class
            investigation.observations.append({
                "event": "workflow_fallback_selected_candidate",
                "candidate_set_id": (
                    investigation.candidate_set.candidate_set_id
                    if investigation.candidate_set else None
                ),
                "candidate_id": fallback_candidate.candidate_id,
                "reason": failure_class,
            })
            state["investigation_state"] = investigation.model_dump(mode="json")
            action = candidate_catalog[fallback_candidate.candidate_id]
        else:
            terminal_decomposition = _validated_metric_decomposition_terminal(state)
            if terminal_decomposition is not None:
                action = terminal_decomposition
                state["diagnostic_backend_terminal"] = "validated_metric_decomposition"
            else:
                # Candidate exhaustion is a normal terminal state. Provider
                # failures remain in diagnostic_llm_failures; actual execution
                # faults continue to be recorded in errors by their executor.
                action = AbstainAction(
                    action_type="ABSTAIN", reason="insufficient_evidence",
                    supporting_evidence_ids=_completed_valid_evidence_ids(state),
                )
            state["diagnostic_decision_source"] = "workflow_fallback"
            state["termination_initiator"] = "policy"
    if isinstance(action, FinishAction) or (
        isinstance(action, AbstainAction) and action.reason == "insufficient_evidence"
    ):
        investigation = _ensure_graph_investigation_state(state)
        deterministic_decomposition_finish = (
            state.get("diagnostic_backend_terminal") == "validated_metric_decomposition"
            and isinstance(action, FinishAction)
            and _validated_metric_decomposition_terminal(state) is not None
        )
        deterministic_statistical_contribution_finish = (
            state.get("diagnostic_backend_terminal") == "validated_statistical_contribution"
            and isinstance(action, FinishAction)
            and _validated_statistical_contribution_terminal(state) is not None
        )
        if investigation is not None and not (
            deterministic_decomposition_finish or deterministic_statistical_contribution_finish
        ):
            requested_stop = "FINISH" if isinstance(action, FinishAction) else "ABSTAIN"
            replacement = _audit_soft_finish(
                investigation, candidate_catalog, requested_stop=requested_stop,
            )
            if replacement is not None:
                action = replacement
                state["investigation_state"] = investigation.model_dump(mode="json")
                if isinstance(action, CallCandidateAction):
                    # `_audit_soft_finish` has already selected the candidate
                    # in the server-owned InvestigationState. Resolve its
                    # opaque handle here, just as the normal LLM path does;
                    # passing CALL_CANDIDATE to policy_execute would be
                    # (correctly) rejected as an unresolved client-style
                    # handle and could burn decision rounds without executing
                    # the still-authorized candidate.
                    selected_candidate_id = action.candidate_id
                    resolved_action = candidate_catalog.get(selected_candidate_id)
                    if resolved_action is None:
                        from app.investigation.contracts import StopReason

                        action = AbstainAction(
                            action_type="ABSTAIN", reason="insufficient_evidence",
                            supporting_evidence_ids=[],
                        )
                        investigation.stop_reason = StopReason(
                            code="AUTHORIZATION_BLOCKED",
                            detail="Server-selected continuation candidate is no longer in the current catalog.",
                            initiator="policy",
                        )
                        state["investigation_state"] = investigation.model_dump(mode="json")
                        state["termination_initiator"] = "policy"
                    else:
                        action = resolved_action
                        state["diagnostic_selected_candidate_id"] = selected_candidate_id
                        state["diagnostic_decision_source"] = "backend"
    if isinstance(action, ParallelFactsAction):
        # Batch model-generated queries predate candidate IDs and can expand
        # the action surface; deterministic required batches remain server-owned.
        state.setdefault("errors", []).append({
            "code": "diagnostic_candidate_required",
            "message": "Model-selected parallel queries are not accepted; select a server candidate.",
            "agent_name": "diagnostic_decide",
        })
        state["diagnostic_decision_source"] = "workflow_fallback"
        return _fallback_action(state)
    return action


def _audit_soft_finish(
    investigation: Any,
    candidate_catalog: dict[str, Any],
    *,
    requested_stop: str = "FINISH",
) -> Any | None:
    """Reject premature chat terminal actions and continue via legal candidates."""
    from app.investigation.contracts import StopReason
    from app.investigation.engine import select_candidate
    from app.investigation.stop_policy import soft_stop_blockers

    blockers = soft_stop_blockers(investigation)
    if not blockers:
        return None
    investigation.observations.append({
        "event": "soft_stop_rejected",
        "requested_stop": requested_stop,
        "blockers": blockers,
    })
    available = [
        item for item in investigation.candidate_actions
        if item.candidate_id in candidate_catalog
    ]
    if available and investigation.candidate_set is not None:
        selected = select_candidate(
            investigation, available[0].candidate_id,
            candidate_set_id=(investigation.candidate_set.candidate_set_id if investigation.candidate_set else None),
        )
        if selected is not None:
            return CallCandidateAction(
                action_type="CALL_CANDIDATE",
                candidate_set_id=(investigation.candidate_set.candidate_set_id if investigation.candidate_set else ""),
                candidate_id=selected.candidate_id,
            )
    investigation.stop_reason = StopReason(
        code="DATA_INSUFFICIENT",
        detail=f"Agent requested {requested_stop}, but the evidence/branch completeness audit did not pass.",
        initiator="backend",
    )
    completed_evidence_ids = {
        evidence_id for result in investigation.action_results
        if result.status == "completed" for evidence_id in result.evidence_ids
    }
    return AbstainAction(
        action_type="ABSTAIN", reason="insufficient_evidence",
        supporting_evidence_ids=[
            evidence.evidence_id for evidence in investigation.evidence
            if evidence.evidence_id in completed_evidence_ids
        ],
    )


def _same_query(left: Any, right: Any) -> bool:
    return left.tool_name == right.tool_name and _query_arguments(left) == _query_arguments(right)


def _ensure_graph_investigation_state(state: AgentState) -> Any | None:
    """Create/reload the shared investigation state used by both adapters."""
    from app.investigation.contracts import (
        AnalysisTarget, AttributionEvidence, InvestigationState, StopReason, UnresolvedBranch,
        branch_share_for_priority,
    )
    from app.investigation.engine import ensure_root_branch
    from app.investigation.evidence import scope_fingerprint as fingerprint_scope
    from app.metrics.registry import get_registry

    scope = state.get("controlled_analysis_scope") or {}
    decomposition_scope_violation = False
    scope_snapshot = dict(scope) if scope else {
        "tenant_id": state.get("tenant_id"), "user_id": state.get("user_id"),
    }
    decomposition_request = state.get("registered_decomposition_request") or {}
    decomposition_dates = _registered_decomposition_dates(state)
    if decomposition_request.get("status") == "ready" and decomposition_dates:
        start, end = decomposition_dates
    else:
        start, end = _diagnostic_dates(state)
    if scope and decomposition_dates:
        try:
            locked_start = date.fromisoformat(str(scope.get("default_start_date") or ""))
            locked_end = date.fromisoformat(str(scope.get("default_end_date") or ""))
            requested_start, requested_end = map(date.fromisoformat, decomposition_dates)
            if requested_start < locked_start or requested_end > locked_end:
                decomposition_scope_violation = True
        except ValueError:
            decomposition_scope_violation = True
    elif scope:
        start = str(scope.get("default_start_date") or start)
        end = str(scope.get("default_end_date") or end)
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    explicit_dates = (
        decomposition_dates if decomposition_request.get("status") == "ready"
        else (_diagnostic_comparison_dates(state) if not scope else None)
    )
    comparison_windows = None
    if explicit_dates is None:
        comparison_windows = _in_scope_comparison_windows(
            start_date, end_date, seller_scoped=bool(scope),
            granularity=str(scope.get("granularity") or "") or None,
        )
    if explicit_dates is not None:
        baseline_start = baseline_end = date.fromisoformat(explicit_dates[0])
        current_start = current_end = date.fromisoformat(explicit_dates[1])
    elif comparison_windows:
        baseline_start, baseline_end = comparison_windows[:2]
        current_start, current_end = comparison_windows[2:4]
    elif end_date > start_date:
        # Align the immutable target with the point comparison used by the
        # required dimension step. Attribution for this fallback is separately
        # gated on validated evidence for those exact two dates.
        baseline_start = baseline_end = end_date - timedelta(days=1)
        current_start = current_end = end_date
    else:
        baseline_start = baseline_end = start_date
        current_start = current_end = end_date
    try:
        registry = get_registry()
        metric = registry.resolve(_target_metric(state))
        definition = registry.get(metric)
        target = AnalysisTarget(
            metric_id=metric,
            current_start=current_start,
            current_end=current_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            canonical_scope=scope_snapshot,
            scope_fingerprint=fingerprint_scope(scope_snapshot),
            requested_analysis_type=("contribution" if decomposition_request.get("status") == "ready" else "diagnostic"),
            metric_definition_version=definition.version,
        )
    except Exception:
        logging.getLogger(__name__).exception("Could not resolve diagnostic investigation target")
        return None

    raw = state.get("investigation_state")
    if raw is None:
        investigation = InvestigationState(target=target)
    else:
        try:
            investigation = InvestigationState.model_validate(raw)
        except Exception:
            detail = "Saved investigation state failed contract validation; execution is stopped without resetting its budget."
            state["diagnostic_investigation_invalidated"] = "invalid_state"
            errors = state.setdefault("errors", [])
            if not any(item.get("code") == "diagnostic_investigation_state_invalid" for item in errors if isinstance(item, dict)):
                errors.append({
                    "code": "diagnostic_investigation_state_invalid",
                    "message": detail,
                    "agent_name": "diagnostic_state",
                })
            if isinstance(raw, dict):
                invalid_state = dict(raw)
                invalid_state["stop_reason"] = {
                    "code": "CAPABILITY_GAP", "detail": detail, "initiator": "backend",
                }
                state["investigation_state"] = invalid_state
            return None
        if investigation.target != target:
            # A resumed turn is bound to its original target and scope. Never
            # replace it with a fresh state: that would discard fingerprints
            # and restore the action budget after a scope/intent change.
            detail = "Investigation target or locked scope changed during execution; create a new turn to continue."
            investigation.stop_reason = StopReason(
                code="AUTHORIZATION_BLOCKED", detail=detail, initiator="policy",
            )
            state["investigation_state"] = investigation.model_dump(mode="json")
            state["diagnostic_investigation_invalidated"] = "target_changed"
            errors = state.setdefault("errors", [])
            if not any(item.get("code") == "diagnostic_investigation_target_changed" for item in errors if isinstance(item, dict)):
                errors.append({
                    "code": "diagnostic_investigation_target_changed",
                    "message": detail,
                    "agent_name": "diagnostic_state",
                })
            return None
    count = int(state.get("diagnostic_investigation_action_count", 0))
    if not raw:
        investigation.remaining_budget = max(0, 6 - count)
        root_branch = ensure_root_branch(investigation)
        migrated_evidence_ids: set[str] = set()
        # One-time adapter for executions/checkpoints created before the shared
        # InvestigationState was persisted. Only typed, scope-matching evidence
        # can seed branches; free-form observations remain non-authoritative.
        for record in state.get("root_cause_evidence", []):
            if not isinstance(record, dict) or not str(record.get("purpose", "")).startswith("dimension_contribution:"):
                continue
            try:
                typed = AttributionEvidence.model_validate(record.get("investigation_evidence"))
            except Exception:
                continue
            if typed.metric_id != metric or typed.scope_fingerprint != target.scope_fingerprint:
                continue
            parent_dimension = str(record.get("purpose", "")).split(":", 1)[1]
            comparison = typed.comparison.get("comparison") or {}
            existing_filters = dict(comparison.get("filters") or {})
            if parent_dimension != "product_category" or existing_filters:
                continue
            investigation.evidence.append(typed)
            migrated_evidence_ids.add(typed.evidence_id)
            total_delta = typed.values.get("total_delta")
            if isinstance(total_delta, (int, float)) and investigation.root_effect_value is None:
                investigation.root_effect_value = float(total_delta)
            denominator = abs(float(total_delta)) if isinstance(total_delta, (int, float)) else 0.0
            for component in typed.values.get("components", []):
                if not isinstance(component, dict):
                    continue
                group_value = component.get("dimension_value")
                effect = component.get("delta_value", component.get("value"))
                if group_value is None or str(group_value).strip().casefold() in {"", "other"} or not isinstance(effect, (int, float)):
                    continue
                root_share = abs(float(effect)) / denominator if denominator > 1e-12 else None
                investigation.branches.append(UnresolvedBranch(
                    branch_id=f"branch_{typed.evidence_id}_{str(group_value)[:32]}",
                    # Legacy root_cause_evidence does not carry the new shared
                    # ActionResult ledger. Keep its parent link evidence-only;
                    # new runtime branches require both validated provenance refs.
                    parent_branch_id=None,
                    parent_evidence_id=typed.evidence_id,
                    branch_type="dimension_contribution",
                    metric_id=metric,
                    dimension=parent_dimension,
                    dimension_value=str(group_value),
                    effect_value=float(effect),
                    effect_unit=typed.unit or None,
                    root_attributed_effect_value=float(effect),
                    root_effect_share=root_share,
                    effect_share_of_root=root_share,
                    depth=1,
                    inherited_filters={parent_dimension: str(group_value)},
                ))
        migrated_children = [
            branch for branch in investigation.branches
            if branch.parent_evidence_id in migrated_evidence_ids and branch.status == "open"
        ]
        if migrated_children:
            # The typed legacy dimension evidence is itself the completed root
            # breakdown; resume from its generated child branch, not the root.
            investigation.branches = [
                branch.model_copy(update={"status": "explained"})
                if branch.branch_id == root_branch.branch_id else branch
                for branch in investigation.branches
            ]
            investigation.current_focus_branch_id = max(
                migrated_children,
                key=lambda branch: (branch_share_for_priority(branch), -branch.depth, branch.branch_id),
            ).branch_id
    elif not investigation.branches and investigation.stop_reason is None:
        # Older shared-state checkpoints may predate explicit root nodes.
        # Add only the missing structural root; preserve their evidence and budget.
        ensure_root_branch(investigation)
    if decomposition_scope_violation and investigation.stop_reason is None:
        investigation.stop_reason = StopReason(
            code="USER_SCOPE_LIMIT",
            detail="Requested decomposition dates exceed or could not be validated against the locked analysis period.",
            initiator="policy",
        )
    state["investigation_state"] = investigation.model_dump(mode="json")
    return investigation


def _trusted_daily_series(state: AgentState, metric_id: str) -> dict[date, float] | None:
    """Return only complete, same-scope daily facts collected through MCP."""
    from math import isfinite

    from app.investigation.contracts import AttributionEvidence
    from app.investigation.series_analysis import normalize_points

    investigation = _ensure_graph_investigation_state(state)
    if investigation is None:
        return None
    scope = state.get("controlled_analysis_scope") or {}
    decomposition_request = state.get("registered_decomposition_request") or {}
    decomposition_dates = _registered_decomposition_dates(state)
    if decomposition_request.get("status") == "ready" and decomposition_dates:
        source_start, source_end = decomposition_dates
    else:
        source_start, source_end = _diagnostic_dates(state)
    if scope and not decomposition_dates:
        source_start = str(scope.get("default_start_date") or source_start)
        source_end = str(scope.get("default_end_date") or source_end)
    first, last = date.fromisoformat(source_start), date.fromisoformat(source_end)
    expected_days = (last - first).days + 1
    if expected_days <= 0 or expected_days > 3660:
        return None

    for record in reversed(state.get("root_cause_evidence", [])):
        if not isinstance(record, dict) or str(record.get("metric_id")) != metric_id:
            continue
        try:
            evidence = AttributionEvidence.model_validate(record.get("investigation_evidence"))
        except Exception:
            continue
        if (
            evidence.metric_id != metric_id
            or evidence.scope_fingerprint != investigation.target.scope_fingerprint
            or evidence.authorized_metric_provenance != "tool_gateway_policy_validated"
            or evidence.evidence_type != "trend"
        ):
            continue
        period = evidence.comparison.get("period") or {}
        if (
            period.get("start_date") != first.isoformat()
            or period.get("end_date") != last.isoformat()
            or period.get("granularity") != "day"
        ):
            continue
        payload = evidence.values.get("result")
        if not isinstance(payload, dict) or payload.get("status") not in {"ok", None}:
            continue
        points = normalize_points(payload)
        if len(points) != expected_days:
            continue
        values: dict[date, float] = {}
        valid = True
        for point in points:
            try:
                bucket = date.fromisoformat(point["bucket"])
                value = point["value"]
                if value is None or not isfinite(float(value)) or bucket in values:
                    valid = False
                    break
                values[bucket] = float(value)
            except (KeyError, TypeError, ValueError, OverflowError):
                valid = False
                break
        if valid and len(values) == expected_days and first in values and last in values:
            return values
    return None


def _period_series_total(series: dict[date, float], start: date, end: date) -> float | None:
    buckets = [day for day in series if start <= day <= end]
    expected = (end - start).days + 1
    if expected <= 0 or len(buckets) != expected:
        return None
    return sum(series[day] for day in buckets)


def _registered_input_queries_complete(state: AgentState, metric_ids: list[str]) -> bool:
    """Whether every registered input was actually queried for this request's exact dates.

    A successful query that returns null/missing buckets is evidence of a data
    limitation. A planned action, a failed query, or an unrelated period is not.
    """
    request = state.get("registered_decomposition_request") or {}
    dates = _registered_decomposition_dates(state)
    if request.get("status") != "ready" or not dates:
        return False
    baseline, current = (str(item) for item in dates)
    expected = {baseline, current}
    calls_by_evidence = {
        str(call.get("evidence_id")): call
        for call in state.get("tool_calls", [])
        if isinstance(call, dict) and call.get("evidence_id")
    }
    queried: set[str] = set()
    for record in state.get("root_cause_evidence", []):
        if not isinstance(record, dict) or record.get("metric_id") not in metric_ids:
            continue
        evidence_id = str(record.get("query_id") or "")
        call = calls_by_evidence.get(evidence_id)
        if not call or call.get("status") != "ok" or call.get("tool_name") != "registry_trend":
            continue
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        execution = result.get("result") or {}
        if execution.get("status") != "ok":
            continue
        parsed = execution.get("parsed") or {}
        trend = execution.get("trend_result") or {}
        if (
            parsed.get("query_type") != "metric_trend"
            or parsed.get("metric_name") != record.get("metric_id")
            or parsed.get("start_date") != baseline
            or parsed.get("end_date") != current
            or trend.get("start_date") != baseline
            or trend.get("end_date") != current
            or trend.get("status") not in {"ok", "not_found"}
        ):
            continue
        points = trend.get("points")
        if not isinstance(points, list):
            # A valid not_found response is still a completed query for this span.
            if trend.get("status") == "not_found":
                queried.add(str(record.get("metric_id")))
            continue
        returned_dates = {
            str(point.get("bucket")) for point in points if isinstance(point, dict)
        }
        if expected <= returned_dates:
            queried.add(str(record.get("metric_id")))
    return set(metric_ids) <= queried


def _available_chat_decompositions(state: AgentState, metric_id: str, scope_name: str) -> dict[str, tuple[str, ...]]:
    """Return deterministic methods only when their authorized inputs are complete."""
    from app.metrics.registry import get_registry

    try:
        definition = get_registry().get(metric_id)
        capability = definition.attribution.for_scope(scope_name)
        investigation = _ensure_graph_investigation_state(state)
        if investigation is None:
            return {}
        target = investigation.target
        # A decomposition needs a genuine before/after comparison. When the
        # requested scope cannot supply two disjoint supported periods,
        # _ensure_graph_investigation_state keeps a fallback target for other
        # diagnostic actions; it must not turn that fallback into attribution.
        if target.baseline_end >= target.current_start:
            return {}
        if (
            target.baseline_start == target.baseline_end
            and target.current_start == target.current_end
        ):
            # A point-to-point fallback is legal for decomposition only when a
            # validated dimension query has already established those exact
            # comparison dates in this scope. A seven-day trend by itself does
            # not authorize choosing an arbitrary baseline.
            validated_ids = {
                evidence_id
                for result in investigation.action_results
                if result.status == "completed"
                and result.validation_result.status == "valid"
                for evidence_id in result.evidence_ids
            }
            point_comparison_seen = any(
                evidence.evidence_id in validated_ids
                and evidence.evidence_type == "dimension_contribution"
                and evidence.metric_id == metric_id
                and evidence.scope_fingerprint == target.scope_fingerprint
                and evidence.authorized_metric_provenance == "tool_gateway_policy_validated"
                and (evidence.comparison.get("comparison") or {}).get("baseline_date")
                    == target.baseline_start.isoformat()
                and (evidence.comparison.get("comparison") or {}).get("current_date")
                    == target.current_start.isoformat()
                for evidence in investigation.evidence
            )
            # A server-confirmed premise can establish the same point comparison.
            # Require matching trusted target observations, not just state labels.
            premise = state.get("premise_check") or {}
            series = _trusted_daily_series(state, metric_id)
            premise_comparison_seen = (
                premise.get("status") in {"confirmed", "confirmed_by_user"}
                and premise.get("metric_name") == metric_id
                and premise.get("baseline_date") == target.baseline_start.isoformat()
                and premise.get("current_date") == target.current_start.isoformat()
                and series is not None
                and series.get(target.baseline_start) == premise.get("baseline_value")
                and series.get(target.current_start) == premise.get("current_value")
            )
            registered_request = state.get("registered_decomposition_request") or {}
            requested_pair = _registered_decomposition_dates(state)
            explicit_registered_pair = (
                registered_request.get("status") == "ready"
                and registered_request.get("target_metric_id") == metric_id
                and requested_pair == (
                    target.baseline_start.isoformat(), target.current_start.isoformat(),
                )
                and registered_request.get("method_id") in capability.method_ids
                and registered_request.get("baseline_date") == target.baseline_start.isoformat()
                and registered_request.get("current_date") == target.current_start.isoformat()
            )
            if not (explicit_registered_pair or point_comparison_seen or premise_comparison_seen):
                return {}
        available: dict[str, tuple[str, ...]] = {}
        factor_specs = capability.factor_specs.get("product_shapley_v1", ())
        if "product_shapley_v1" in capability.method_ids and len(factor_specs) == 2:
            source_ids = {
                source_id for spec in factor_specs for source_id in spec.source_metric_ids
            }
            verified_ratio_factor_ids = {
                spec.factor_id for spec in factor_specs
                if get_registry().get(spec.factor_id).period_aggregation is not None
            }
            required_series_ids = source_ids | verified_ratio_factor_ids
            if required_series_ids and all(
                _trusted_daily_series(state, source_id) is not None
                for source_id in required_series_ids
            ):
                available["product_shapley_v1"] = tuple(spec.factor_id for spec in factor_specs)
        period_aggregation = definition.period_aggregation
        if "symmetric_ratio_v1" in capability.method_ids and period_aggregation is not None:
            numerator = get_registry().get(period_aggregation.numerator_metric)
            denominator = get_registry().get(period_aggregation.denominator_metric)
            # Daily distinct counts are not additive sufficient statistics.
            if numerator.aggregation == "sum" and denominator.aggregation == "sum":
                numerator_series = _trusted_daily_series(state, numerator.metric_id)
                denominator_series = _trusted_daily_series(state, denominator.metric_id)
                if numerator_series is not None and denominator_series is not None:
                    available["symmetric_ratio_v1"] = (numerator.metric_id, denominator.metric_id)
        return available
    except Exception:  # noqa: BLE001 - capability discovery fails closed
        return {}


def _execute_chat_decomposition(state: AgentState, candidate: Any) -> Any:
    """Build a typed decomposition from complete authorized daily evidence only."""
    from app.analysis.attribution_methods import (
        AttributionDataError,
        bind_registered_ratio_factor_identities,
        product_shapley_attribution,
        resolve_product_factor_values,
        symmetric_ratio_attribution,
    )
    from app.investigation.evidence import build_attribution_evidence
    from app.metrics.registry import get_registry

    investigation = _ensure_graph_investigation_state(state)
    if investigation is None:
        raise ValueError("investigation target is unavailable")
    target = investigation.target
    scope_name = "seller" if state.get("controlled_analysis_scope") else "platform"
    definition = get_registry().get(candidate.action.metric_id)
    capability = definition.attribution.for_scope(scope_name)
    method_id = candidate.action.method_id
    if method_id not in capability.method_ids:
        raise ValueError("decomposition method is not registered for this scope")
    factor_ids = list(candidate.action.factor_metric_ids)
    input_metric_ids = set(factor_ids)
    factor_specs = capability.factor_specs.get(method_id, ())
    if method_id == "product_shapley_v1":
        expected_factor_ids = [item.factor_id for item in factor_specs]
        if len(factor_specs) != 2 or factor_ids != expected_factor_ids:
            raise ValueError("product factors do not match the registered factor contract")
        source_metric_ids = sorted({
            source_id for spec in factor_specs for source_id in spec.source_metric_ids
        })
    else:
        source_metric_ids = []
    authz = state.get("authorization_context")
    if authz is not None:
        authz.require_metrics([*factor_ids, *source_metric_ids])

    def totals(metric_id: str) -> tuple[float, float]:
        series = _trusted_daily_series(state, metric_id)
        if series is None:
            raise _DecompositionDataUnavailable(f"Complete authorized daily trend is unavailable for {metric_id}.")
        baseline = _period_series_total(series, target.baseline_start, target.baseline_end)
        current = _period_series_total(series, target.current_start, target.current_end)
        if baseline is None or current is None:
            raise _DecompositionDataUnavailable(f"{metric_id} trend does not cover both target periods.")
        return baseline, current

    if method_id == "product_shapley_v1":
        source_metric_ids = {
            source_id for spec in factor_specs for source_id in spec.source_metric_ids
        }
        input_metric_ids = set(source_metric_ids)
        for factor_id in factor_ids:
            factor_definition = get_registry().get(factor_id)
            aggregation = factor_definition.period_aggregation
            if aggregation is None:
                continue
            factor_series = _trusted_daily_series(state, factor_id)
            numerator_series = _trusted_daily_series(state, aggregation.numerator_metric)
            denominator_series = _trusted_daily_series(state, aggregation.denominator_metric)
            if factor_series is None or numerator_series is None or denominator_series is None:
                raise _DecompositionDataUnavailable(
                    f"Complete daily evidence is unavailable for registered ratio factor {factor_id}."
                )
            input_metric_ids.add(factor_id)
            for period_start, period_end in (
                (target.baseline_start, target.baseline_end),
                (target.current_start, target.current_end),
            ):
                day = period_start
                while day <= period_end:
                    numerator = numerator_series.get(day)
                    denominator = denominator_series.get(day)
                    observed_factor = factor_series.get(day)
                    if numerator is None or denominator is None or observed_factor is None:
                        raise _DecompositionDataUnavailable(
                            f"Registered ratio factor {factor_id} is missing a period bucket."
                        )
                    if denominator <= 0:
                        raise _DecompositionDataUnavailable(
                            f"Registered ratio factor {factor_id} has a non-positive denominator."
                        )
                    expected_factor = float(numerator) / float(denominator) * aggregation.multiplier
                    if not math.isclose(
                        float(observed_factor), expected_factor,
                        rel_tol=1e-8, abs_tol=0.01,
                    ):
                        raise DataReconciliationError(
                            f"Observed {factor_id} does not reconcile to its registered numerator/denominator."
                        )
                    day += timedelta(days=1)
        period_totals = {
            source_id: totals(source_id) for source_id in source_metric_ids
        }
        try:
            baseline_factors, current_factors = resolve_product_factor_values(
                factor_specs, period_totals,
            )
        except AttributionDataError as exc:
            raise _DecompositionDataUnavailable(str(exc)) from exc
        target_baseline, target_current = totals(candidate.action.metric_id)
        result = product_shapley_attribution(
            metric_id=candidate.action.metric_id,
            baseline_factors=baseline_factors, current_factors=current_factors,
            unit=("percentage_points" if definition.unit == "%" else definition.unit),
        )
        if not (
            math.isclose(result.baseline_value, target_baseline, rel_tol=1e-8, abs_tol=0.01)
            and math.isclose(result.current_value, target_current, rel_tol=1e-8, abs_tol=0.01)
        ):
            raise ValueError("注册因子乘积与目标指标趋势总量不一致")
    elif method_id == "symmetric_ratio_v1":
        period = definition.period_aggregation
        if period is None or factor_ids != [period.numerator_metric, period.denominator_metric]:
            raise ValueError("ratio factors do not match the registered metric definition")
        if any(get_registry().get(item).aggregation != "sum" for item in factor_ids):
            raise ValueError("当前因子不是可加总充分统计量，聊天归因不支持按日求和")
        n0, n1 = totals(factor_ids[0])
        d0, d1 = totals(factor_ids[1])
        input_metric_ids.add(candidate.action.metric_id)
        target_series = _trusted_daily_series(state, candidate.action.metric_id)
        if target_series is None:
            raise _DecompositionDataUnavailable("Complete target ratio trend is unavailable.")
        # The target's ratio observations must agree with the same registered
        # sufficient statistics, not merely share a direction of change.
        for start, end, numerator, denominator in (
            (target.baseline_start, target.baseline_end, n0, d0),
            (target.current_start, target.current_end, n1, d1),
        ):
            if start == end and denominator > 0 and not math.isclose(
                float(target_series[start]), numerator / denominator * period.multiplier,
                rel_tol=1e-8, abs_tol=0.01,
            ):
                raise DataReconciliationError("Registered ratio factors disagree with the target trend.")
        result = symmetric_ratio_attribution(
            metric_id=candidate.action.metric_id,
            baseline_numerator=n0, current_numerator=n1,
            baseline_denominator=d0, current_denominator=d1,
            multiplier=period.multiplier,
            unit=("percentage_points" if definition.unit == "%" else definition.unit),
        )
        result = bind_registered_ratio_factor_identities(result, scope=scope_name)
    else:
        raise ValueError("unsupported deterministic decomposition method")

    evidence_id = stable_evidence_id(
        "metric_decomposition", method_id,
        {"metric": candidate.action.metric_id, "factors": factor_ids,
         "baseline": result.baseline_value, "current": result.current_value,
         "scope": target.scope_fingerprint},
    )
    validated_trend_ids = {
        evidence_id
        for action_result in investigation.action_results
        if action_result.status == "completed"
        and action_result.validation_result.status == "valid"
        for evidence_id in action_result.evidence_ids
    }
    decomposition_dates = _registered_decomposition_dates(state)
    if decomposition_dates:
        input_start, input_end = decomposition_dates
    else:
        input_start, input_end = _diagnostic_dates(state)
    if state.get("controlled_analysis_scope") and not decomposition_dates:
        input_start = str(state["controlled_analysis_scope"].get("default_start_date") or input_start)
        input_end = str(state["controlled_analysis_scope"].get("default_end_date") or input_end)
    input_evidence_ids = [
        item.evidence_id for item in investigation.evidence
        if item.evidence_id in validated_trend_ids
        and item.evidence_type == "trend"
        and item.metric_id in input_metric_ids
        and item.scope_fingerprint == target.scope_fingerprint
        and item.authorized_metric_provenance == "tool_gateway_policy_validated"
        and (item.comparison.get("period") or {}).get("start_date") == input_start
        and (item.comparison.get("period") or {}).get("end_date") == input_end
        and (item.comparison.get("period") or {}).get("granularity") == "day"
    ]
    typed = build_attribution_evidence(
        evidence_id=evidence_id, evidence_type="metric_decomposition",
        trace_id=str(state.get("trace_id") or "") or None,
        metric_id=candidate.action.metric_id,
        claim=f"{definition.display_name} 变化按注册方法 {method_id} 作确定性统计分解。",
        values={
            "baseline_value": result.baseline_value,
            "current_value": result.current_value,
            "total_delta": result.total_delta,
            "factor_metrics": factor_ids,
            "factor_identities": [
                item.model_dump(mode="json") for item in result.factor_identities
            ],
            "input_evidence_ids": input_evidence_ids,
            **({
                "baseline_factor_values": baseline_factors,
                "current_factor_values": current_factors,
            } if method_id == "product_shapley_v1" else {}),
            "raw_components": [item.model_dump(mode="json") for item in result.raw_components],
            "display_method": result.display_method,
            "display_components": [item.model_dump(mode="json") for item in result.display_components],
            "interaction_delta": next(
                (item.value for item in result.raw_components if item.component_type == "interaction_effect"), 0.0,
            ),
            "reconciliation_ratio": result.reconciliation_ratio,
            "residual_value": result.residual_value,
            "residual_type": result.residual_type,
            "residual_explainable": result.residual_explainable,
            "residual_tolerance": result.residual_tolerance,
            "residual_note": result.residual_note,
            "components": [item.model_dump(mode="json") for item in result.display_components],
        },
        source_action_id=candidate.candidate_id,
        source_ref="authorized_complete_daily_trend_evidence",
        scope=target.canonical_scope, method_id=result.method_id,
        method_version=result.method_version,
        authorized_metric_provenance="tool_gateway_policy_validated",
        comparison={
            "current_period": {"start_date": target.current_start.isoformat(), "end_date": target.current_end.isoformat()},
            "baseline_period": {"start_date": target.baseline_start.isoformat(), "end_date": target.baseline_end.isoformat()},
        },
        data_quality={"status": "valid", "input_series_complete": True},
        unit=definition.unit,
        limitations=["分解描述注册指标的统计变化，不代表因果效应。"],
    )
    return typed


def _record_diagnostic_capability_gap(state: AgentState, metric_id: str, reason: str) -> None:
    gaps = state.setdefault("diagnostic_capability_gaps", [])
    item = {"metric_id": metric_id, "reason": reason}
    if item not in gaps:
        gaps.append(item)


def _set_investigation_stop_reason(state: AgentState, code: str, detail: str) -> None:
    from app.investigation.contracts import StopReason

    investigation = _ensure_graph_investigation_state(state)
    if investigation is None:
        raw = state.get("investigation_state")
        if isinstance(raw, dict):
            raw["stop_reason"] = {
                "code": code,
                "detail": detail,
                "initiator": "policy" if code == "AUTHORIZATION_BLOCKED" else "backend",
            }
            state["investigation_state"] = raw
        return
    if investigation.stop_reason is None:
        investigation.stop_reason = StopReason(
            code=code,
            detail=detail,
            initiator="policy" if code == "AUTHORIZATION_BLOCKED" else "backend",
        )
    state["investigation_state"] = investigation.model_dump(mode="json")


def _record_terminal_investigation_reason(
    state: AgentState,
    action: Any,
    *,
    terminal_evidence_valid: bool,
) -> dict[str, str] | None:
    """Persist the chat terminal decision into the investigation ledger.

    LangGraph has its own terminal action, but Trace/evaluation consumers read
    ``InvestigationState.stop_reason``. Keep these representations aligned;
    preserve an already-recorded hard execution/policy stop when the Agent
    subsequently abstains.
    """
    raw = state.get("investigation_state")
    if not isinstance(raw, dict):
        return None

    current = raw.get("stop_reason")
    current_code = str(current.get("code") or "") if isinstance(current, dict) else ""
    guardrail_codes = {
        str(item.get("code") or "")
        for item in state.get("diagnostic_guardrails", [])
        if isinstance(item, dict)
    }
    terminal_guardrail_failure = not terminal_evidence_valid or bool(guardrail_codes & {
        "target_metric_cannot_be_root_cause",
        "finish_cites_untrusted_or_unvalidated_evidence",
    })
    if terminal_guardrail_failure and current_code not in {
        "AUTHORIZATION_BLOCKED", "EXECUTION_FAILED",
    }:
        code, detail, initiator = (
            "INVALID_DATA",
            "Terminal decision referenced evidence that did not pass provenance and validation checks.",
            "policy",
        )
    elif current_code and current_code != "COMPLETED":
        return current
    elif getattr(action, "action_type", None) == "FINISH":
        code, detail, initiator = (
            "ENOUGH_EVIDENCE",
            "The Agent finalized a conclusion using validated in-scope evidence.",
            "agent",
        )
    elif getattr(action, "reason", None) == "budget_exhausted":
        code, detail, initiator = (
            "BUDGET_EXHAUSTED",
            "The Agent abstained because the investigation budget was exhausted.",
            "budget",
        )
    elif getattr(action, "reason", None) == "ambiguous_question":
        code, detail, initiator = (
            "NO_HIGH_VALUE_ACTION",
            "The question remains ambiguous and needs user clarification before another safe action can be selected.",
            "agent",
        )
    else:
        plan = state.get("diagnostic_plan") or {}
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        step_states = state.get("diagnostic_step_state", {})
        has_required_data_gap = any(
            isinstance(step, dict)
            and step.get("required")
            and step_states.get(str(step.get("step_id") or "")) == "data_insufficient"
            for step in steps
        ) or any(
            isinstance(item, dict) and item.get("required_step_id")
            for item in state.get("diagnostic_candidate_data_gaps", [])
        )
        if has_required_data_gap:
            code, detail, initiator = (
                "DATA_INSUFFICIENT",
                "Required diagnostic evidence could not be obtained from the available data.",
                "backend",
            )
        elif state.get("diagnostic_capability_gaps"):
            code, detail, initiator = (
                "CAPABILITY_GAP",
                "No supported analysis capability can answer the remaining diagnostic question.",
                "backend",
            )
        elif state.get("errors"):
            code, detail, initiator = (
                "EXECUTION_FAILED",
                "The diagnostic execution ended with an error.",
                "backend",
            )
        else:
            code, detail, initiator = (
                "NO_HIGH_VALUE_ACTION",
                "The Agent abstained because it found no further supported action with sufficient evidence.",
                "agent",
            )

    terminal = {"code": code, "detail": detail, "initiator": initiator}
    raw["stop_reason"] = terminal
    state["investigation_state"] = raw
    return terminal


def _candidate_tool_catalog(state: AgentState) -> dict[str, CallToolAction]:
    """Create the bounded extra-exploration allowlist from trusted registry data."""
    from app.investigation.candidates import candidate_id_for, generate_candidate_actions
    from app.investigation.contracts import CandidateAction, branch_share_for_priority
    from app.investigation.engine import prepare_candidates
    from app.metrics.registry import get_registry
    from app.auth.context import AuthorizationDenied, build_authorization_context
    from app.tenancy import current_tenant_context

    if int(state.get("diagnostic_investigation_action_count", 0)) >= 6:
        return {}

    target_metric = _target_metric(state)
    request_policy = _requested_dimension_policy(state, target_metric)
    unavailable_dimensions = _apply_requested_dimension_policy(state, request_policy)
    if unavailable_dimensions:
        return {}
    requested_dimensions = set(request_policy.get("requested") or [])
    scope = state.get("controlled_analysis_scope") or {}
    scope_name = "seller" if scope else "platform"
    decomposition_request = state.get("registered_decomposition_request") or {}
    decomposition_dates = _registered_decomposition_dates(state)
    if decomposition_request.get("status") == "ready" and decomposition_dates:
        start, end = decomposition_dates
    else:
        start, end = _diagnostic_dates(state)
    if scope and not decomposition_dates:
        start = str(scope.get("default_start_date") or start)
        end = str(scope.get("default_end_date") or end)
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    comparison_windows = _in_scope_comparison_windows(
        start_date, end_date, seller_scoped=bool(scope),
        granularity=str(scope.get("granularity") or "") or None,
    )
    try:
        tenant_context = current_tenant_context()
        if tenant_context is None:
            _set_investigation_stop_reason(
                state, "AUTHORIZATION_BLOCKED", "缺少可信租户授权上下文，已禁止生成调查候选。",
            )
            return {}
        authz = build_authorization_context(
            tenant_context,
            allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
            allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
            metric_scope_type=scope_name,
        )
        authz.require_action("analyze")
        registry = get_registry()
        metric = registry.resolve(target_metric)
        definition = registry.get(metric)
        authz.require_metrics([metric])
        implementation = definition.implementations.get(scope_name)
        if implementation is None:
            _record_diagnostic_capability_gap(state, metric, f"当前范围没有已注册的 {scope_name} 查询实现。")
            _set_investigation_stop_reason(
                state, "CAPABILITY_GAP", f"当前范围没有已注册的 {scope_name} 查询实现。",
            )
            return {}
        if scope and metric not in set(scope.get("allowed_metrics") or []):
            _set_investigation_stop_reason(
                state, "AUTHORIZATION_BLOCKED", "目标指标不在当前锁定的授权指标范围内。",
            )
            return {}
        capability = definition.attribution.for_scope(scope_name)
        investigation = _ensure_graph_investigation_state(state)
        if investigation is None or investigation.target.metric_id != metric:
            return {}
        target = investigation.target
        authorized_metric_ids = {
            str(item.get("metric_id"))
            for item in state.get("authorized_metrics", [])
            if isinstance(item, dict) and item.get("metric_id")
        }
        # A policy allow-list is not an executable MetricQueryPlan credential.
        # Child-metric drilldowns need their own authoritative provenance;
        # never infer one from the root metric's authorization.
        blocked_branch_ids = {
            branch.branch_id for branch in investigation.branches
            if branch.metric_id != metric
            and branch.status in {"open", "investigating"}
            and branch.metric_id not in authorized_metric_ids
        }
        if blocked_branch_ids:
            investigation.branches = [
                branch.model_copy(update={"status": "blocked"})
                if branch.branch_id in blocked_branch_ids else branch
                for branch in investigation.branches
            ]
            for branch_id in sorted(blocked_branch_ids):
                rejection = {
                    "candidate_id": branch_id,
                    "reason": "child metric has no server-authorized AuthorizedMetric provenance",
                }
                if rejection not in investigation.policy_rejections:
                    investigation.policy_rejections.append(rejection)
        open_branches = [
            branch for branch in investigation.branches
            if branch.status in {"open", "investigating"}
            and branch.metric_id in set(authz.allowed_metric_ids)
            and (not scope or branch.metric_id in set(scope.get("allowed_metrics") or []))
        ]
        active_branch = next(
            (branch for branch in open_branches
             if branch.branch_id == investigation.current_focus_branch_id),
            None,
        ) or max(open_branches, key=lambda item: (branch_share_for_priority(item), -item.depth), default=None)
        action_metric = active_branch.metric_id if active_branch else metric
        action_definition = registry.get(action_metric)
        action_implementation = action_definition.implementations.get(scope_name)
        action_capability = action_definition.attribution.for_scope(scope_name)
        if action_implementation is None:
            _record_diagnostic_capability_gap(
                state, action_metric, f"当前范围没有已注册的 {scope_name} 查询实现。",
            )
            _set_investigation_stop_reason(
                state, "CAPABILITY_GAP", f"子指标 {action_metric} 没有当前范围查询实现。",
            )
            return {}
        try:
            authz.require_metrics([action_metric])
        except AuthorizationDenied:
            _set_investigation_stop_reason(
                state, "AUTHORIZATION_BLOCKED", f"子指标 {action_metric} 未通过当前授权复核。",
            )
            return {}
        if not capability.method_ids:
            _record_diagnostic_capability_gap(
                state, metric,
                definition.attribution.limitation or "当前范围没有已验证的归因方法。",
            )
        allowed_dimensions = set(action_capability.dimensions)
        if requested_dimensions:
            allowed_dimensions &= requested_dimensions
        # The scoped MCP supports category breakdowns and month-position
        # breakdowns; only category→month-position currently supports drill filters.
        if scope_name == "seller":
            allowed_dimensions &= {"product_category", "month_position"}
        if scope:
            # An explicit empty dimension grant means no dimensions are
            # authorized; never interpret it as "use registry defaults".
            allowed_dimensions &= set(scope.get("allowed_dimensions") or [])
            if action_capability.dimensions and not allowed_dimensions:
                _record_diagnostic_capability_gap(
                    state, metric, "当前授权范围不允许任何已声明的归因维度。",
                )
        allowed_dimensions &= set(authz.allowed_dimensions)
        has_open_factor_branch = any(
            branch.status in {"open", "investigating"} and branch.metric_id != metric
            for branch in investigation.branches
        )
        focused_on_factor = bool(
            active_branch is not None and active_branch.branch_type == "metric_effect"
        )
        decomposition_specs = (
            {} if has_open_factor_branch or focused_on_factor
            else _available_chat_decompositions(state, metric, scope_name)
        )
        authorized_decomposition_specs: dict[str, tuple[str, ...]] = {}
        for method_id, factor_ids in decomposition_specs.items():
            try:
                factor_specs = action_capability.factor_specs.get(method_id, ())
                source_metric_ids = {
                    source_id for spec in factor_specs for source_id in spec.source_metric_ids
                }
                required_metric_ids = {metric, *factor_ids, *source_metric_ids}
                authz.require_metrics(sorted(required_metric_ids))
                if scope and not required_metric_ids <= set(scope.get("allowed_metrics") or []):
                    if decomposition_request.get("status") == "ready" and method_id == decomposition_request.get("method_id"):
                        _set_investigation_stop_reason(
                            state, "AUTHORIZATION_BLOCKED", "A registered decomposition dependency is outside the locked metric scope.",
                        )
                    continue
                authorized_decomposition_specs[method_id] = factor_ids
            except AuthorizationDenied:
                if decomposition_request.get("status") == "ready" and method_id == decomposition_request.get("method_id"):
                    _set_investigation_stop_reason(
                        state, "AUTHORIZATION_BLOCKED", "A registered decomposition dependency was denied by current metric authorization.",
                    )
                continue
        if decomposition_request.get("status") == "ready":
            registered_inputs = list(dict.fromkeys(
                str(item) for item in decomposition_request.get("input_metric_ids", [])
            ))
            missing_inputs = [
                item for item in registered_inputs
                if _trusted_daily_series(state, item) is None
            ]
            if missing_inputs and _registered_input_queries_complete(state, registered_inputs):
                dates = _registered_decomposition_dates(state)
                baseline, current = dates or (None, None)
                evidence_ids = [
                    str(record.get("query_id"))
                    for record in state.get("root_cause_evidence", [])
                    if isinstance(record, dict)
                    and record.get("metric_id") in missing_inputs
                    and record.get("query_id")
                ]
                detail = (
                    "Registered numeric decomposition inputs were queried for the requested period, "
                    f"but complete daily evidence is unavailable for: {', '.join(missing_inputs)}."
                )
                step_id = "registered-decomposition-inputs"
                state.setdefault("diagnostic_step_state", {})[step_id] = "data_insufficient"
                gap = {
                    "operation_intent": "registered_numeric_decomposition",
                    "failure_scope": "required_registered_decomposition_inputs",
                    "required_step_id": step_id,
                    "failure_code": "DATA_INSUFFICIENT",
                    "failure_detail": detail,
                    "target_metric_id": decomposition_request.get("target_metric_id"),
                    "missing_input_metric_ids": missing_inputs,
                    "queried_input_metric_ids": registered_inputs,
                    "period": {
                        "baseline_date": str(baseline) if baseline else None,
                        "current_date": str(current) if current else None,
                    },
                    "evidence_ids": evidence_ids,
                }
                gaps = state.setdefault("diagnostic_candidate_data_gaps", [])
                if gap not in gaps:
                    gaps.append(gap)
                _set_investigation_stop_reason(state, "DATA_INSUFFICIENT", detail)
                return {}
        supported_actions = {"ANALYZE_TREND"} if "trend" in action_implementation.capabilities else set()
        if "query_metric_trend" not in authz.allowed_tools:
            supported_actions.discard("ANALYZE_TREND")
        if "statistics" in action_implementation.capabilities and "query_metric_statistics" in authz.allowed_tools:
            supported_actions.add("ANALYZE_DISTRIBUTION")
        entity_distribution_metrics: tuple[str, ...] = ()
        if (
            scope_name == "seller"
            and action_metric == "seller_avg_freight_per_order"
            and action_metric in set((scope or {}).get("allowed_metrics") or [])
            and "query_scoped_order_freight_distribution" in authz.allowed_tools
        ):
            supported_actions.add("ANALYZE_ENTITY_DISTRIBUTION")
            entity_distribution_metrics = (action_metric,)
        association_pairs_by_metric: dict[str, tuple[str, ...]] = {}
        if (
            scope_name == "seller"
            and action_metric in set((scope or {}).get("allowed_metrics") or [])
            and "query_scoped_registered_association" in authz.allowed_tools
        ):
            from app.analysis.association_catalog import registered_pair_ids_for_metric

            registered_pairs = registered_pair_ids_for_metric(action_metric)
            if registered_pairs:
                try:
                    authz.require_metrics([action_metric])
                except AuthorizationDenied:
                    registered_pairs = ()
            if registered_pairs:
                supported_actions.add("ANALYZE_ASSOCIATION")
                association_pairs_by_metric[action_metric] = registered_pairs
        if "trend" in action_implementation.capabilities and "query_metric_trend" in authz.allowed_tools:
            supported_actions.add("CHECK_ANOMALY")
            supported_actions.add("CHECK_DATA_QUALITY")
        ratio_breakdown_tool = {
            "freight_rate": "compare_freight_rate_ratio_facts_by_dimension_period",
            "avg_order_value": "compare_avg_order_value_ratio_facts_by_dimension_period",
        }.get(action_metric, "")
        if (
            "additive_change_v1" in action_capability.method_ids
            and "compare_breakdown_by_dimension" in TOOL_REGISTRY
            and "compare_breakdown_by_dimension" in authz.allowed_tools
        ) or (
            "symmetric_mix_within_v1" in action_capability.method_ids
            and ratio_breakdown_tool in TOOL_REGISTRY
            and ratio_breakdown_tool in authz.allowed_tools
        ):
            supported_actions.add("BREAKDOWN_DIMENSION")
        if (
            "additive_change_v1" in action_capability.method_ids
            and "compare_breakdown_by_dimension" in TOOL_REGISTRY
            and "compare_breakdown_by_dimension" in authz.allowed_tools
        ):
            # Root branches use BREAKDOWN_DIMENSION; once evidence opens a
            # child branch, the same authorized query contract is exposed as
            # DRILL_DOWN. This is scope-independent: platform metrics must not
            # lose their registered breakdown capability at child branches.
            supported_actions.add("DRILL_DOWN")
        if authorized_decomposition_specs:
            supported_actions.add("DECOMPOSE_METRIC")
        trusted_branches = [
            branch for branch in investigation.branches
            if _branch_is_trusted_for_action_metric(branch, action_metric, allowed_dimensions)
        ]
        candidates = generate_candidate_actions(
            target, allowed_methods=action_capability.method_ids,
            allowed_dimensions=allowed_dimensions,
            branches=trusted_branches,
            focus_branch_id=investigation.current_focus_branch_id,
            executed_candidate_ids=[item.candidate_id for item in investigation.action_results],
            data_available_methods=action_capability.method_ids,
            supported_action_types=supported_actions,
            factor_metrics_by_method=authorized_decomposition_specs,
            quality_check_ids=(
                ("zero_value_buckets", "negative_value_buckets", "missing_value_buckets")
                if "CHECK_DATA_QUALITY" in supported_actions else ()
            ),
            entity_distribution_metric_ids=entity_distribution_metrics,
            association_pair_ids_by_metric=association_pairs_by_metric,
            # Authorization is rechecked in shared preparation, so generate the
            # ranked legal set first and apply Top-K only after that filter.
            top_k=None,
        )
        if decomposition_request.get("status") == "ready":
            requested_method = str(decomposition_request.get("method_id") or "")
            requested_target = str(decomposition_request.get("target_metric_id") or "")
            candidates = [
                candidate for candidate in candidates
                if candidate.action.action_type == "DECOMPOSE_METRIC"
                and candidate.action.metric_id == requested_target
                and candidate.action.method_id == requested_method
            ]
            if not candidates:
                if not state.get("diagnostic_capability_gaps"):
                    _record_diagnostic_capability_gap(
                        state, requested_target,
                        "The requested registered decomposition method is unavailable for the validated target inputs and scope.",
                    )
                _set_investigation_stop_reason(
                    state, "CAPABILITY_GAP", "The registered decomposition candidate is unavailable under the current method, input, and scope contract.",
                )
                state["investigation_state"] = investigation.model_dump(mode="json")
                return {}
        catalog: dict[str, CallToolAction | ExecuteDecompositionAction] = {}
        candidate_models: dict[str, CandidateAction] = {}
        for candidate in candidates:
            action = candidate.action
            if action.action_type == "DECOMPOSE_METRIC":
                query: CallToolAction | ExecuteDecompositionAction = ExecuteDecompositionAction(
                    action_type="DECOMPOSE_METRIC", candidate_id=candidate.candidate_id,
                )
            elif action.action_type == "ANALYZE_TREND":
                query = CallToolAction(
                    action_type="CALL_TOOL", tool_name="query_metric_trend", metric_name=action.metric_id,
                    period=DiagnosticPeriod(
                        start_date=str(scope.get("default_start_date") or start),
                        end_date=str(scope.get("default_end_date") or end),
                        granularity=str(scope.get("granularity") or "day"),
                    ),
                )
            elif action.action_type == "ANALYZE_DISTRIBUTION":
                query = CallToolAction(
                    action_type="CALL_TOOL", tool_name="query_metric_statistics", metric_name=action.metric_id,
                    period=DiagnosticPeriod(
                        start_date=str(scope.get("default_start_date") or start),
                        end_date=str(scope.get("default_end_date") or end),
                        granularity=str(scope.get("granularity") or "day"),
                    ),
                    statistics=["count", "mean", "std", "min", "q1", "median", "q3", "max"],
                )
            elif action.action_type == "ANALYZE_ENTITY_DISTRIBUTION":
                query = CallToolAction(
                    action_type="CALL_TOOL",
                    tool_name="query_scoped_order_freight_distribution",
                    metric_name=action.metric_id,
                    period=DiagnosticPeriod(
                        start_date=str(scope.get("default_start_date") or start),
                        end_date=str(scope.get("default_end_date") or end),
                        granularity="day",
                    ),
                )
            elif action.action_type == "ANALYZE_ASSOCIATION":
                query = CallToolAction(
                    action_type="CALL_TOOL",
                    tool_name="query_scoped_registered_association",
                    metric_name=action.metric_id,
                    period=DiagnosticPeriod(
                        start_date=str(scope.get("default_start_date") or start),
                        end_date=str(scope.get("default_end_date") or end),
                        granularity="day",
                    ),
                )
                query = query.model_copy(update={
                    "statistics": [],
                    "association_pair_id": action.pair_id,
                })
            elif action.action_type in {"CHECK_ANOMALY", "CHECK_DATA_QUALITY"}:
                query = CallToolAction(
                    action_type="CALL_TOOL", tool_name="query_metric_trend", metric_name=action.metric_id,
                    period=DiagnosticPeriod(
                        start_date=str(scope.get("default_start_date") or start),
                        end_date=str(scope.get("default_end_date") or end),
                        granularity=str(scope.get("granularity") or "day"),
                    ),
                )
            elif action.action_type in {"BREAKDOWN_DIMENSION", "DRILL_DOWN"}:
                if comparison_windows is None:
                    continue
                baseline_start, baseline_end, current_start, current_end, comparison_granularity = comparison_windows
                if action.method_id == "symmetric_mix_within_v1":
                    tool_name = ratio_breakdown_tool
                elif action.method_id == "additive_change_v1":
                    tool_name = "compare_breakdown_by_dimension"
                else:
                    continue
                query = CallToolAction(
                    action_type="CALL_TOOL", tool_name=tool_name,
                    metric_name=action.metric_id, dimension=action.dimension,
                    # Top-10 is a display default. The registered contribution
                    # reconciliation needs all groups, within the tool's cap.
                    limit=100,
                    comparison=DiagnosticComparison(
                        current_start=current_start.isoformat(), current_end=current_end.isoformat(),
                        baseline_start=baseline_start.isoformat(), baseline_end=baseline_end.isoformat(),
                        granularity=comparison_granularity,
                        filters=action.inherited_filters,
                        parent_evidence_id=action.parent_evidence_id,
                    ),
                )
            else:
                continue
            candidate_id = candidate_id_for(action)
            seen = False
            if isinstance(query, CallToolAction):
                args = _query_arguments(query)
                seen = any(call.get("tool_name") == query.tool_name and call.get("arguments") == args for call in state.get("tool_calls", []))
            reusable_series_action = action.action_type in {"CHECK_ANOMALY", "CHECK_DATA_QUALITY"}
            if not seen or reusable_series_action:
                catalog[candidate_id] = query
                candidate_models[candidate_id] = candidate
        permitted = prepare_candidates(
            investigation,
            candidate_models.values(),
            current_period={"start": target.current_start.isoformat(), "end": target.current_end.isoformat()},
            baseline_period={"start": target.baseline_start.isoformat(), "end": target.baseline_end.isoformat()},
            authorization_blocked=lambda item: (
                "metric is outside the locked authorized metric set"
                if item.action.metric_id not in set(authz.allowed_metric_ids)
                else (
                    "dimension is outside the locked authorized dimension set"
                    if getattr(item.action, "dimension", None) is not None
                    and item.action.dimension not in set(authz.allowed_dimensions)
                    else False
                )
            ),
            authorization_fingerprint=authz.authz_digest,
        )
        state["investigation_state"] = investigation.model_dump(mode="json")
        return {item.candidate_id: catalog[item.candidate_id] for item in permitted if item.candidate_id in catalog}
    except AuthorizationDenied as exc:
        logging.getLogger(__name__).warning("Diagnostic candidate authorization denied: %s", exc.code)
        _set_investigation_stop_reason(
            state, "AUTHORIZATION_BLOCKED", "当前授权策略不允许生成这组调查候选。",
        )
        return {}
    except Exception:
        logging.getLogger(__name__).exception("Failed to build governed diagnostic candidate catalog")
        _set_investigation_stop_reason(
            state, "CAPABILITY_GAP", "受控调查候选生成失败，已停止继续查询。",
        )
        return {}


def _in_scope_comparison_windows(
    start: date, end: date, *, seller_scoped: bool,
    granularity: str | None = None,
) -> tuple[date, date, date, date, Literal["week", "month"]] | None:
    """Choose two non-overlapping comparison windows wholly inside user scope.

    Honor a selected week/month granularity for seller scopes. When no
    granularity is selected, seller ranges may be split into two in-scope
    windows. Platform MCP period-contribution accepts complete natural periods.
    """
    if end <= start:
        return None
    from calendar import monthrange

    if seller_scoped and granularity == "month":
        current_end = end.replace(day=monthrange(end.year, end.month)[1])
        if current_end > end:
            current_end = end.replace(day=1) - timedelta(days=1)
        current_start = current_end.replace(day=1)
        baseline_end = current_start - timedelta(days=1)
        baseline_start = baseline_end.replace(day=1)
        if current_end.day == monthrange(current_end.year, current_end.month)[1] and baseline_start >= start:
            return baseline_start, baseline_end, current_start, current_end, "month"
        return None
    if seller_scoped and granularity == "week":
        current_end = end - timedelta(days=(end.weekday() + 1) % 7)
        current_start = current_end - timedelta(days=6)
        baseline_end = current_start - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=6)
        if baseline_start >= start:
            return baseline_start, baseline_end, current_start, current_end, "week"
        return None
    if seller_scoped and granularity == "day":
        # The contribution query accepts natural week/month periods only.
        # Never relabel arbitrary day-range halves as monthly periods.
        return None

    if granularity != "week":
        month_start = end.replace(day=1)
        current_month_end = end.replace(day=monthrange(end.year, end.month)[1])
        if current_month_end > end:
            current_month_end = end
        previous_month_end = month_start - timedelta(days=1)
        previous_month_start = previous_month_end.replace(day=1)
        if previous_month_start >= start and current_month_end <= end:
            # Only full natural months are legal for the governed period MCP tool.
            if current_month_end == end and end.day == monthrange(end.year, end.month)[1]:
                return previous_month_start, previous_month_end, month_start, current_month_end, "month"
    if granularity == "month":
        return None

    # Fall back to the latest pair of complete Monday-Sunday weeks.
    current_week_end = end - timedelta(days=(end.weekday() + 1) % 7)
    current_week_start = current_week_end - timedelta(days=6)
    baseline_week_end = current_week_start - timedelta(days=1)
    baseline_week_start = baseline_week_end - timedelta(days=6)
    if baseline_week_start >= start:
        return baseline_week_start, baseline_week_end, current_week_start, current_week_end, "week"
    return None


def _plan_for_query(query: Any, state: AgentState | None = None) -> MetricQueryPlan:
    fields: dict[str, Any] = {"metric_name": query.metric_name, "tool_name": query.tool_name}
    if state is not None:
        authorized = next((
            item for item in state.get("authorized_metrics", [])
            if isinstance(item, dict) and item.get("metric_id") == query.metric_name
        ), None)
        # Required actions are generated by the server from the sealed request
        # contract. When this route bypasses data_query, authorize that exact
        # requested metric here using the live trusted identity; never infer a
        # credential from an LLM action or from registry membership alone.
        if authorized is None and state.get("diagnostic_required_action"):
            authorized = _authorize_required_contract_metric(query, state)
        # Model-selected diagnostic factors still need provenance from the
        # authoritative planner/confirmed mapping/explorer-selection path.
        if authorized is not None:
            fields["authorized_metric"] = (
                authorized if isinstance(authorized, dict)
                else authorized.model_dump(mode="json")
            )
    if query.period:
        if query.tool_name == "query_scoped_order_freight_distribution":
            fields.update(
                query_type="metric_entity_statistics",
                start_date=query.period.start_date,
                end_date=query.period.end_date,
            )
        elif query.tool_name == "query_scoped_registered_association":
            fields.update(
                query_type="metric_association",
                start_date=query.period.start_date,
                end_date=query.period.end_date,
                association_pair_id=getattr(query, "association_pair_id", None),
            )
        elif query.tool_name == "query_metric_trend":
            fields.update(query_type="metric_trend", start_date=query.period.start_date, end_date=query.period.end_date, granularity=query.period.granularity)
        elif query.tool_name == "query_metric_statistics":
            fields.update(
                query_type="metric_statistics", start_date=query.period.start_date,
                end_date=query.period.end_date, granularity=query.period.granularity,
                statistics=list(query.statistics),
            )
        else:
            fields.update(query_type="metric_value", metric_date=query.period.end_date)
    elif query.comparison:
        if query.tool_name in {
            "compare_breakdown_by_dimension",
            "compare_freight_rate_ratio_facts_by_dimension_period",
            "compare_avg_order_value_ratio_facts_by_dimension_period",
        }:
            comparison = query.comparison
            fields.update(
                query_type="metric_contribution", current_date=comparison.current_date,
                baseline_date=comparison.baseline_date, current_start=comparison.current_start,
                current_end=comparison.current_end, baseline_start=comparison.baseline_start,
                baseline_end=comparison.baseline_end, granularity=comparison.granularity,
                dimension=query.dimension, limit=query.limit, filters=dict(comparison.filters),
            )
        else:
            fields.update(query_type="metric_compare", left_date=query.comparison.current_date, right_date=query.comparison.baseline_date)
    return MetricQueryPlan.model_validate(fields)


def _authorize_required_contract_metric(query: Any, state: AgentState) -> Any | None:
    """Issue a query credential only for a matching server-required contract step."""
    from app.auth.context import AuthorizationDenied, build_authorization_context
    from app.context.request_contract import validate_compiled_request_contract
    from app.metrics.authorization import authorize_metric
    from app.tenancy import current_tenant_context

    compilation = state.get("context_compilation")
    valid, _reason = validate_compiled_request_contract(
        compilation,
        state_contract_hash=state.get("request_contract_hash"),
        require_state_anchor=True,
    )
    if not valid or not isinstance(compilation, dict):
        return None
    contract = compilation.get("request_contract")
    if not isinstance(contract, dict) or contract.get("metric_id") != query.metric_name:
        return None
    sequence = contract.get("required_evidence_sequence") or []
    if not sequence:
        return None
    if query.tool_name == "query_metric_trend" and "trend" not in sequence:
        return None
    if query.tool_name in {
        "compare_breakdown_by_dimension",
        "compare_breakdown_by_dimension_period",
        "compare_freight_rate_ratio_facts_by_dimension_period",
        "compare_avg_order_value_ratio_facts_by_dimension_period",
    } and "dimension_contribution" not in sequence:
        return None

    tenant = current_tenant_context()
    if tenant is None:
        return None
    scope = state.get("controlled_analysis_scope") or {}
    scope_name = "seller" if scope else str(state.get("scope_type") or "platform")
    allowed_metrics = scope.get("allowed_metrics") if scope else None
    allowed_dimensions = scope.get("allowed_dimensions") if scope else None
    try:
        authz = build_authorization_context(
            tenant,
            allowed_metric_ids=allowed_metrics,
            allowed_dimensions=allowed_dimensions,
            metric_scope_type=scope_name,
        )
        authz.require_action("analyze")
        authz.require_metrics([query.metric_name])
        authz.require_tool(query.tool_name)
        if query.dimension:
            authz.require_dimensions([query.dimension])
        scope_data = {
            **scope,
            "tenant_id": state.get("tenant_id") or tenant.tenant_id,
            "user_id": state.get("user_id") or tenant.user_id,
            "permissions": state.get("user_role") or tenant.role,
            "scope_type": scope_name,
        }
        credential = authorize_metric(
            query.metric_name,
            provenance_type="exact",
            provenance_id=f"request-contract:{contract['contract_hash']}",
            scope=scope_name,
            scope_data=scope_data,
        )
    except (AuthorizationDenied, KeyError, ValueError):
        return None

    rows = list(state.get("authorized_metrics") or [])
    rows.append(credential.model_dump(mode="json"))
    state["authorized_metrics"] = rows
    return credential


def _query_arguments(query: Any) -> dict[str, Any]:
    """Typed query identity used for audit records and duplicate detection."""
    arguments: dict[str, Any] = {"metric_name": query.metric_name}
    if query.dimension:
        arguments["dimension"] = query.dimension
    if query.period is not None:
        arguments["period"] = query.period.model_dump(mode="json")
    statistics = getattr(query, "statistics", None)
    if statistics:
        arguments["statistics"] = list(statistics)
    if getattr(query, "association_pair_id", None):
        arguments["association_pair_id"] = query.association_pair_id
    if query.comparison is not None:
        comparison = query.comparison.model_dump(mode="json")
        # Parent Evidence is lineage, not query semantics. Keeping it out of
        # the fingerprint prevents a different evidence ID from re-running an
        # otherwise identical filtered database query.
        comparison.pop("parent_evidence_id", None)
        arguments["comparison"] = comparison
    return arguments


def _policy_for_query(query: Any, state: AgentState) -> PolicyDecision:
    if query.tool_name not in TOOL_REGISTRY:
        return PolicyDecision(decision="deny", reason="Tool is not registered.", action_type="CALL_TOOL")
    scope = state.get("controlled_analysis_scope") or {}
    try:
        from app.metrics.registry import get_registry
        registry = get_registry()
        metric = registry.resolve(query.metric_name)
        attribution_dimensions = set(registry.get(metric).attribution.for_scope(
            "seller" if scope else "platform"
        ).dimensions)
        if query.dimension and not registry.is_dimension_allowed(metric, query.dimension) and query.dimension not in attribution_dimensions:
            return PolicyDecision(decision="deny", reason="Dimension is not allowed for this metric.", action_type="CALL_TOOL")
        scope_dimensions = set(scope.get("allowed_dimensions") or [])
        if query.dimension and scope_dimensions and query.dimension not in scope_dimensions:
            return PolicyDecision(decision="deny", reason="Dimension is outside the current controlled scope.", action_type="CALL_TOOL")
        if scope and metric not in set(scope.get("allowed_metrics", [])):
            # Server-produced required actions query the governed skeleton's
            # own metric set (a signed scope extension): tool, registry and
            # dimension checks above still apply.  Model-chosen actions are
            # never extended this way.
            if not state.get("diagnostic_required_action"):
                return PolicyDecision(decision="deny", reason="Metric is outside the controlled scope.", action_type="CALL_TOOL")
    except Exception as exc:
        return PolicyDecision(decision="deny", reason=f"Metric validation failed: {exc}", action_type="CALL_TOOL")
    return PolicyDecision(decision="allow", reason="Tool, metric, dimension and scope validated.", action_type="CALL_TOOL")


def _execute_query(query: Any, state: AgentState) -> dict[str, Any]:
    plan = _plan_for_query(query, state)
    request_id = _diagnostic_query_request_id(query, state)
    if state.get("seller_scope_payload"):
        return {
            "task_type": "metric_query", "question": _question_for_query(query),
            "server": "seller-scope-guard", "tool_name": None, "request_id": request_id,
            "result": {
                "status": "unsupported", "parsed": plan.model_dump(mode="json"),
                "reason_code": "seller_scope_operation_unsupported",
                "reason": "Diagnostic follow-up queries do not have a seller-scoped handler.",
            },
        }
    scope = state.get("controlled_analysis_scope")
    if scope:
        raw_boundary = state.get("authorization_boundary")
        authorization_boundary = (
            raw_boundary if isinstance(raw_boundary, dict) and raw_boundary else None
        )
        return query_scoped_business_data(
            f"{_question_for_query(query)} {state.get('user_message', '')}".strip(),
            scope,
            request_id=request_id,
            plan=plan,
            manifest_id=state.get("data_version_manifest_id"),
            authorization_boundary=authorization_boundary,
            current_query_scope_fingerprint=(
                state.get("current_query_scope_fingerprint")
                or (authorization_boundary or {}).get("query_scope_fingerprint")
                or ""
            ),
        )
    return query_business_data(
        "metric_query", f"{_question_for_query(query)} {state.get('user_message', '')}".strip(),
        tenant_id=state.get("tenant_id"), role=state.get("user_role"), plan=plan,
        request_id=request_id, idempotency_key=request_id,
    )


def _diagnostic_query_request_id(query: Any, state: AgentState) -> str:
    """Use the same identity at execution and exception-recording boundaries."""
    identity = stable_evidence_id("diagnostic_query", query.tool_name, _query_arguments(query))
    return f"{state.get('trace_id') or state.get('thread_id') or 'diagnostic'}:{identity}"


def policy_execute(state: AgentState) -> AgentState:
    raw = state.get("diagnostic_action", {})
    action = _ACTION_ADAPTER.validate_python(raw)
    plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
    pending_steps = [
        step for step in plan.steps
        if state.get("diagnostic_step_state", {}).get(step.step_id, "pending") == "pending"
    ]
    if isinstance(action, ExecuteDecompositionAction):
        from app.auth.context import AuthorizationDenied, build_authorization_context
        from app.investigation.contracts import DecomposeMetricAction, InvestigationState
        from app.tenancy import current_tenant_context

        investigation = _ensure_graph_investigation_state(state)
        selected_id = str(state.get("diagnostic_selected_candidate_id") or "")
        selected = next((
            item for item in investigation.candidate_actions
            if item.candidate_id == action.candidate_id == selected_id
        ), None) if investigation else None
        if selected is None or not isinstance(selected.action, DecomposeMetricAction):
            state["policy_decision"] = PolicyDecision(
                decision="deny", reason="Decomposition must resolve to a currently selected server candidate.",
                action_type="DECOMPOSE_METRIC",
            ).model_dump(mode="json")
            state["termination_initiator"] = "policy"
            return state
        scope = state.get("controlled_analysis_scope") or {}
        scope_name = "seller" if scope else "platform"
        try:
            tenant_context = current_tenant_context()
            if tenant_context is None:
                raise AuthorizationDenied("missing_context", "trusted authorization context is required")
            authz = build_authorization_context(
                tenant_context,
                allowed_metric_ids=scope.get("allowed_metrics") if scope else None,
                allowed_dimensions=scope.get("allowed_dimensions") if scope else None,
                metric_scope_type=scope_name,
            )
            authz.require_action("analyze")
            authz.require_metrics([selected.action.metric_id, *selected.action.factor_metric_ids])
            if scope and not set([selected.action.metric_id, *selected.action.factor_metric_ids]) <= set(scope.get("allowed_metrics") or []):
                raise AuthorizationDenied("scope_rejected", "a required decomposition metric is outside the locked metric scope")
            state["diagnostic_local_attribution_evidence"] = _execute_chat_decomposition(state, selected)
        except AuthorizationDenied as exc:
            state["policy_decision"] = PolicyDecision(
                decision="deny", reason=f"Decomposition authorization denied: {exc.code}.",
                action_type="DECOMPOSE_METRIC",
            ).model_dump(mode="json")
            state["termination_initiator"] = "policy"
            _set_investigation_stop_reason(state, "AUTHORIZATION_BLOCKED", "比率/乘积因子未通过当前权限复核。")
            return state
        except _DecompositionDataUnavailable as exc:
            _set_investigation_stop_reason(state, "DATA_INSUFFICIENT", str(exc))
            state["policy_decision"] = PolicyDecision(
                decision="deny", reason="Registered decomposition inputs are incomplete.",
                action_type="DECOMPOSE_METRIC",
            ).model_dump(mode="json")
            state["termination_initiator"] = "backend"
            return state
        except DataReconciliationError as exc:
            _set_investigation_stop_reason(state, "INVALID_DATA", str(exc))
            input_metric_ids = {
                selected.action.metric_id,
                *selected.action.factor_metric_ids,
            }
            input_evidence_ids = {
                str(item.get("query_id"))
                for item in state.get("root_cause_evidence", [])
                if isinstance(item, dict)
                and item.get("metric_id") in input_metric_ids
                and item.get("query_id")
                and str((item.get("result") or {}).get("status") or "") == "ok"
            }
            for bundle in state.get("evidence_bundles", []):
                if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
                    continue
                content = bundle.get("content") or {}
                result = content.get("result") or {}
                parsed = result.get("parsed") or {}
                evidence_id = bundle.get("evidence_id")
                if parsed.get("metric_name") in input_metric_ids and evidence_id:
                    input_evidence_ids.add(str(evidence_id))
            input_evidence_ids = sorted(input_evidence_ids)
            failures = state.setdefault("diagnostic_data_reconciliation_failures", [])
            failures.append({
                "code": "DATA_RECONCILIATION_FAILED",
                "metric_id": selected.action.metric_id,
                "method_id": selected.action.method_id,
                "input_metric_ids": sorted(input_metric_ids),
                "detail": str(exc),
                "input_evidence_ids": input_evidence_ids,
            })
            state["diagnostic_action"] = AbstainAction(
                action_type="ABSTAIN",
                reason="insufficient_evidence",
                supporting_evidence_ids=_completed_valid_evidence_ids(state),
            ).model_dump(mode="json")
            state["policy_decision"] = PolicyDecision(
                decision="deny", reason="Registered factors do not reconcile to the target metric.",
                action_type="DECOMPOSE_METRIC",
            ).model_dump(mode="json")
            state["termination_initiator"] = "backend"
            return state
        except Exception as exc:  # noqa: BLE001 - retain unexpected computation failures
            logging.getLogger(__name__).exception("Chat decomposition execution failed")
            _set_investigation_stop_reason(
                state, "EXECUTION_FAILED", f"Registered decomposition execution failed: {type(exc).__name__}.",
            )
            state.setdefault("errors", []).append({
                "code": "diagnostic_decomposition_execution_failed",
                "message": str(exc), "agent_name": "policy_execute",
            })
            state["diagnostic_action"] = AbstainAction(
                action_type="ABSTAIN",
                reason="insufficient_evidence",
                supporting_evidence_ids=_completed_valid_evidence_ids(state),
            ).model_dump(mode="json")
            state["policy_decision"] = PolicyDecision(
                decision="deny", reason="Registered decomposition execution failed.",
                action_type="DECOMPOSE_METRIC",
            ).model_dump(mode="json")
            state["termination_initiator"] = "backend"
            return state
        state["policy_decision"] = PolicyDecision(
            decision="allow", reason="Selected registered decomposition was re-authorized and computed from validated evidence.",
            action_type="DECOMPOSE_METRIC",
        ).model_dump(mode="json")
        return state
    if isinstance(action, CallCandidateAction):
        state["policy_decision"] = PolicyDecision(
            decision="deny", reason="Unresolved candidate handle cannot reach execution.",
            action_type="CALL_CANDIDATE",
        ).model_dump(mode="json")
        state["termination_initiator"] = "policy"
        return state
    if isinstance(action, CallToolAction) and not state.get("diagnostic_required_action"):
        candidate_catalog = _candidate_tool_catalog(state)
        if not any(
            isinstance(candidate, CallToolAction) and _same_query(action, candidate)
            for candidate in candidate_catalog.values()
        ):
            state["policy_decision"] = PolicyDecision(
                decision="deny",
                reason="Model-selected query is not present in the server candidate catalog.",
                action_type="CALL_TOOL",
            ).model_dump(mode="json")
            state["termination_initiator"] = "policy"
            return state
    # Server-produced required actions are legal by construction; the step
    # allowlist governs model-chosen actions only.
    if not state.get("diagnostic_required_action") and pending_steps and action.action_type not in set(pending_steps[0].allowed_actions):
        state.setdefault("errors", []).append({
            "code": "diagnostic_action_not_allowed",
            "message": f"Action {action.action_type} is not allowed for step {pending_steps[0].step_id}.",
            "agent_name": "policy_execute",
        })
        state["diagnostic_action"] = AbstainAction(
            action_type="ABSTAIN", reason="insufficient_evidence",
        ).model_dump(mode="json")
        state["termination_initiator"] = "policy"
        state["policy_decision"] = PolicyDecision(
            decision="deny", reason="Action is outside the current step allowlist.", action_type=action.action_type,
        ).model_dump(mode="json")
        return state
    if isinstance(action, ReplanAction):
        old_steps = {item.step_id: item for item in DiagnosticPlan.model_validate(state["diagnostic_plan"]).steps}
        observed_step_ids = {
            step_id for step_id, status in state.get("diagnostic_step_state", {}).items()
            if status == "completed"
        }
        new_ids = {step.step_id for step in action.steps}
        changed_observed = any(
            step_id in observed_step_ids and old_steps.get(step_id) != next(
                (step for step in action.steps if step.step_id == step_id), None
            )
            for step_id in observed_step_ids
        )
        if observed_step_ids - new_ids:
            decision = PolicyDecision(decision="deny", reason="REPLAN cannot remove an observed step.", action_type="REPLAN")
        elif changed_observed:
            decision = PolicyDecision(decision="deny", reason="REPLAN cannot modify an observed step.", action_type="REPLAN")
        else:
            current = DiagnosticPlan.model_validate(state["diagnostic_plan"])
            # A REPLAN may not drop the required evidence skeleton either.
            updated, _appended = _ensure_required_skeleton(
                current.model_copy(update={
                    "hypotheses": action.hypotheses or current.hypotheses,
                    "steps": action.steps,
                    "stop_conditions": action.stop_conditions,
                }),
                supported_kinds=_required_evidence_kinds_for(state),
            )
            state["diagnostic_plan"] = updated.model_dump(mode="json")
            state["diagnostic_step_state"] = {step.step_id: state.get("diagnostic_step_state", {}).get(step.step_id, "pending") for step in updated.steps}
            decision = PolicyDecision(decision="allow", reason="Plan updated without changing observed steps.", action_type="REPLAN")
        state["policy_decision"] = decision.model_dump(mode="json")
        return state
    if isinstance(action, AbstainAction) and action.reason == "insufficient_evidence":
        # Review fix #3: an insufficient-evidence ABSTAIN may not preempt the
        # required evidence floor unless the server has recorded a hard stop
        # (for example, an unsupported comparison period or denied scope).
        investigation_raw = state.get("investigation_state") or {}
        stop_raw = investigation_raw.get("stop_reason") if isinstance(investigation_raw, dict) else None
        hard_stop = str((stop_raw or {}).get("code") or "") if isinstance(stop_raw, dict) else ""
        hard_stop_allows_abstain = hard_stop in {
            "AUTHORIZATION_BLOCKED", "CAPABILITY_GAP", "DATA_INSUFFICIENT",
            "USER_SCOPE_LIMIT", "BUDGET_EXHAUSTED",
        }
        required_pending = [step.step_id for step in _pending_required_steps(plan, state)]
        if required_pending and not hard_stop_allows_abstain:
            state["policy_decision"] = PolicyDecision(
                decision="deny",
                reason=f"ABSTAIN refused: required diagnostic steps incomplete: {', '.join(required_pending)}.",
                action_type="ABSTAIN",
            ).model_dump(mode="json")
            return state
        has_dimension_evidence = any(
            str(item.get("purpose", "")).startswith("dimension_contribution:")
            for item in state.get("root_cause_evidence", [])
        )
        plan_promises_dimension = any(
            _step_kind(step) == "dimension_contribution"
            or "拆解" in step.objective or "dimension" in step.objective.lower()
            for step in plan.steps
        )
        if plan_promises_dimension and not has_dimension_evidence and not hard_stop_allows_abstain:
            state["policy_decision"] = PolicyDecision(
                decision="deny",
                reason="ABSTAIN refused: a planned dimension hypothesis is still unverified.",
                action_type="ABSTAIN",
            ).model_dump(mode="json")
            return state
    if action.action_type in {"FINISH", "ABSTAIN"}:
        decision = PolicyDecision(decision="allow", reason="Non-data action requires no external call.", action_type=action.action_type)
        state["policy_decision"] = decision.model_dump(mode="json")
        return state
    queries = action.facts if isinstance(action, ParallelFactsAction) else [action]
    action_count = int(state.get("diagnostic_investigation_action_count", 0))
    if action_count + len(queries) > 6:
        state["termination_initiator"] = "budget"
        state["policy_decision"] = PolicyDecision(
            decision="deny", reason="Attribution investigation action budget (6) exceeded.",
            action_type=action.action_type,
        ).model_dump(mode="json")
        return state
    investigation = _ensure_graph_investigation_state(state)
    selected_candidate_id = str(state.get("diagnostic_selected_candidate_id") or "")
    selected_candidate = next((
        candidate for candidate in investigation.candidate_actions
        if candidate.candidate_id == selected_candidate_id
    ), None) if investigation else None
    reusable_series_analysis = selected_candidate is not None and getattr(
        selected_candidate.action, "action_type", ""
    ) in {"CHECK_ANOMALY", "CHECK_DATA_QUALITY"}
    cached_results: dict[int, dict[str, Any]] = {}
    if reusable_series_analysis and len(queries) == 1:
        cached = _cached_trend_result(state, queries[0])
        if cached is not None:
            cached_results[0] = cached
    external_query_count = len(queries) - len(cached_results)
    if len(state.get("tool_calls", [])) + external_query_count > int(state.get("runtime_budget", {}).get("max_tool_calls", 8)):
        state["termination_initiator"] = "budget"
        state["policy_decision"] = PolicyDecision(decision="deny", reason="Tool-call budget exceeded.", action_type=action.action_type).model_dump(mode="json")
        state.setdefault("errors", []).append({"code": "tool_call_budget_exceeded", "message": "Diagnostic tool-call budget exceeded.", "agent_name": "policy_execute"})
        return state
    # Deterministic approval blocks redundant external calls: a fact already
    # observed must not be queried twice (the critic flags repeats; the policy
    # layer prevents them from spending budget in the first place).
    seen = {(str(item.get("tool_name")), repr(item.get("arguments") or {})) for item in state.get("tool_calls", [])}
    for index, query in enumerate(queries):
        if (query.tool_name, repr(_query_arguments(query))) in seen and index not in cached_results:
            state["policy_decision"] = PolicyDecision(
                decision="deny",
                reason="Duplicate query already observed; choose a different fact to verify.",
                action_type=action.action_type,
            ).model_dump(mode="json")
            return state
    results = []
    for index, query in enumerate(queries):
        decision = _policy_for_query(query, state)
        if decision.decision == "deny":
            if state.get("diagnostic_required_action"):
                # The executor must not re-issue a denied required query.
                state.setdefault("diagnostic_denied_metrics", []).append(str(query.metric_name))
            state["policy_decision"] = decision.model_dump(mode="json")
            state["termination_initiator"] = "policy"
            return state
        try:
            if index in cached_results:
                query_result = cached_results[index]
            else:
                query_result = _execute_query(query, state)
        except Exception as exc:
            # Also preserve unexpected exceptions from scoped/legacy gateways;
            # do not leak raw exception messages or lose the failed request ID.
            from app.mcp.client import classify_query_failure
            from app.core.failure_context import public_failure_message

            request_id = _diagnostic_query_request_id(query, state)
            context = classify_query_failure(exc, request_id=request_id, default_service="mcp")
            message = public_failure_message(context)
            query_result = {
                "status": "failed", "error": message,
                "tool_name": query.tool_name, "arguments": _query_arguments(query),
                "request_id": request_id,
                "failure_context": context.model_dump(mode="json"),
                "result": {
                    "status": "unsupported" if context.category in {"authorization_denied", "capability_gap"} else "execution_failed",
                    "reason": message, "failure_context": context.model_dump(mode="json"),
                },
            }
        results.append(query_result)
        # ParallelFactsAction is executed sequentially at this boundary. Once
        # a fact fails with a terminal execution condition, later facts cannot
        # repair that request and would only issue redundant tool/SQL calls.
        # Keep the failure result and leave the remaining facts unexecuted.
        result_envelope = (
            query_result.get("result")
            if isinstance(query_result.get("result"), dict)
            else query_result
        )
        failure_context = (
            result_envelope.get("failure_context")
            if isinstance(result_envelope, dict)
            else None
        )
        failure_category = (
            str(failure_context.get("category") or "")
            if isinstance(failure_context, dict)
            else ""
        )
        if (
            str(result_envelope.get("status") or "").casefold()
            in {"failed", "timeout", "error", "execution_failed", "unsupported"}
            or failure_category in {
                "database_error", "tool_timeout", "execution_error", "capability_gap",
            }
        ):
            break
    state["diagnostic_tool_results"] = results
    state["diagnostic_investigation_action_count"] = int(
        state.get("diagnostic_investigation_action_count", 0)
    ) + len(results)
    unsupported_result = any(
        str(
            (item.get("result") if isinstance(item.get("result"), dict) else item)
            .get("status") or ""
        ).casefold() == "unsupported"
        for item in results
    )
    state["policy_decision"] = PolicyDecision(
        decision="allow",
        reason=(
            "Queries were authorized and dispatched; an unsupported result leaves the required evidence incomplete."
            if unsupported_result
            else "Queries were authorized and dispatched; result usability is determined after observation."
        ),
        action_type=action.action_type,
    ).model_dump(mode="json")
    return state


def _record_result(state: AgentState, query: Any, result: dict[str, Any], *, derived_action_id: str | None = None) -> str:
    tool_name = result.get("tool_name") or query.tool_name
    arguments = _query_arguments(query)
    cached_source_id = result.get("cached_source_evidence_id")
    content = {
        # Keep the public gateway request identity inside the evidence payload.
        # Companion auditing and field-level answer binding must resolve this
        # exact evidence back to this exact invocation; an evidence ID alone
        # cannot establish that association.
        **({"request_id": str(result.get("request_id"))} if result.get("request_id") else {}),
        "metric_id": str(getattr(query, "metric_name", "") or "") or None,
        "operation": str(getattr(query, "query_type", "") or "") or None,
        "query": _question_for_query(query),
        "result": result.get("result", result),
        **({"derived_from_evidence_id": cached_source_id} if cached_source_id else {}),
        **({"derived_action_id": derived_action_id} if derived_action_id else {}),
    }
    evidence_id = stable_evidence_id(
        "investigation" if cached_source_id else "mcp",
        str(state.get("diagnostic_selected_candidate_id") or tool_name) if cached_source_id else tool_name,
        content,
    )
    if not cached_source_id:
        payload = result.get("result", {})
        authorization_snapshot = result.get("authorization_snapshot")
        state.setdefault("tool_calls", []).append({
            "server": result.get("server", "mcp"), "tool_name": tool_name,
            "arguments": arguments,
            "status": (
                payload.get("status")
                if payload.get("status") in {"ok", "not_found", "unsupported"}
                else "failed"
            ),
            "request_id": result.get("request_id", ""),
            "evidence_id": evidence_id,
            # Keep the gateway's request-scoped authorization decision on the
            # exact diagnostic invocation. The companion audit must not have
            # to infer it from identity or from a different query result.
            # Preserve the snapshot as returned so a request-ID mismatch stays
            # detectable instead of being silently rewritten here.
            "authorization_snapshot": dict(authorization_snapshot)
            if isinstance(authorization_snapshot, dict) else None,
        })
    state.setdefault("evidence_bundles", []).append({
        "evidence_id": evidence_id,
        "source_type": "investigation_action" if cached_source_id else "mcp",
        "source_ref": str(cached_source_id or tool_name),
        "content": content, "trusted": False,
        "validation_status": "pending_typed_evidence_validation",
    })
    return evidence_id


def _cached_trend_result(state: AgentState, query: Any) -> dict[str, Any] | None:
    """Reuse only an exact, trusted trend response for local deterministic analysis."""
    if query.tool_name != "query_metric_trend":
        return None
    arguments = _query_arguments(query)
    investigation = _ensure_graph_investigation_state(state)
    if investigation is None:
        return None
    source_evidence_by_id = {
        str(item.get("query_id")): item.get("investigation_evidence")
        for item in state.get("root_cause_evidence", [])
        if isinstance(item, dict)
    }
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in state.get("evidence_bundles", [])
        if isinstance(item, dict) and item.get("trusted") is True
    }
    for call in reversed(state.get("tool_calls", [])):
        if call.get("tool_name") not in {"query_metric_trend", "registry_trend"} or call.get("arguments") != arguments:
            continue
        evidence = evidence_by_id.get(str(call.get("evidence_id") or ""))
        if evidence is None:
            continue
        typed_source = source_evidence_by_id.get(str(evidence.get("evidence_id") or ""))
        if not isinstance(typed_source, dict):
            continue
        if (
            typed_source.get("metric_id") != query.metric_name
            or typed_source.get("scope_fingerprint") != investigation.target.scope_fingerprint
            or typed_source.get("authorized_metric_provenance") != "tool_gateway_policy_validated"
        ):
            continue
        content = evidence.get("content") or {}
        result = content.get("result") if isinstance(content, dict) else None
        if not isinstance(result, dict):
            continue
        return {
            "result": result,
            "tool_name": "query_metric_trend",
            "server": "derived-from-authorized-evidence",
            "cached_source_evidence_id": evidence["evidence_id"],
        }
    return None


def _typed_contribution_values(result: dict[str, Any], query: Any) -> dict[str, Any]:
    """Normalize only server-returned contribution facts for follow-up search."""
    from app.analysis.contribution import calculate_change_contribution
    from app.analysis.attribution_methods import AttributionDataError

    envelope = result.get("result") if isinstance(result.get("result"), dict) else result
    payload = None
    if isinstance(envelope, dict):
        payload = envelope.get("contribution_result") or envelope.get("breakdown_result")
    if not isinstance(payload, dict):
        payload = envelope if isinstance(envelope, dict) else {}
    current_rows = payload.get("current_items") or []
    baseline_rows = payload.get("baseline_items") or []
    current = {
        str(row.get("dimension_value")): float(row.get("value") or 0)
        for row in current_rows if isinstance(row, dict) and row.get("dimension_value") is not None
    }
    baseline = {
        str(row.get("dimension_value")): float(row.get("value") or 0)
        for row in baseline_rows if isinstance(row, dict) and row.get("dimension_value") is not None
    }
    total_delta = payload.get("total_delta")
    if not isinstance(total_delta, (int, float)):
        current_total, baseline_total = payload.get("current_total"), payload.get("baseline_total")
        if isinstance(current_total, (int, float)) and isinstance(baseline_total, (int, float)):
            total_delta = float(current_total) - float(baseline_total)
    inherited = query.comparison.filters if query.comparison else {}
    if not isinstance(total_delta, (int, float)):
        return {
            "components": [], "total_delta": None,
            "attribution_status": "missing_total_delta",
            "residual_value": None, "residual_type": "unknown",
            "residual_explainable": False,
            "residual_tolerance": 0.03,
            "residual_note": "缺少整体变化量，无法计算覆盖或舍入残差。",
            "query_filters": dict(inherited),
        }


    comparison = query.comparison
    current_raw = (
        comparison.current_end or comparison.current_date
        if comparison else None
    )
    baseline_raw = (
        comparison.baseline_end or comparison.baseline_date
        if comparison else None
    )
    current_date = date.fromisoformat(current_raw) if current_raw else date.min
    baseline_date = date.fromisoformat(baseline_raw) if baseline_raw else date.min
    try:
        finding = calculate_change_contribution(
            metric_name=str(query.metric_name), dimension=str(query.dimension or "dimension"),
            current_date=current_date, baseline_date=baseline_date,
            current_items=current_rows, baseline_items=baseline_rows,
            overall_total_delta=float(total_delta),
            current_total=float(payload["current_total"]) if isinstance(payload.get("current_total"), (int, float)) else None,
            baseline_total=float(payload["baseline_total"]) if isinstance(payload.get("baseline_total"), (int, float)) else None,
        )
        components = [
            {"dimension_value": item.dimension_value, "delta_value": item.delta_value,
             "current_value": item.current_value, "baseline_value": item.baseline_value,
             "contribution_ratio": item.contribution_ratio}
            for item in finding.items
        ]
        explained = sum(float(item["delta_value"]) for item in components)
        residual_value = round(float(finding.total_delta) - explained, 2)
        return {
            "components": components,
            "total_delta": float(finding.total_delta),
            # Preserve the period totals from the server result so the shared
            # investigation engine can reconcile this breakdown with trend
            # evidence for the same metric/scope/window.
            "baseline_value": (
                float(payload["baseline_total"])
                if isinstance(payload.get("baseline_total"), (int, float)) else None
            ),
            "current_value": (
                float(payload["current_total"])
                if isinstance(payload.get("current_total"), (int, float)) else None
            ),
            "explained_share": 1.0 if abs(float(finding.total_delta)) <= 1e-12 else min(1.0, abs(explained / float(finding.total_delta))),
            "attribution_status": "ok",
            "method_id": "additive_change_v1",
            "residual_value": residual_value,
            "residual_type": "rounding" if abs(residual_value) <= 0.03 else "unknown",
            "residual_explainable": abs(residual_value) <= 0.03,
            "residual_tolerance": 0.03,
            "residual_note": "展示到两位小数后的组贡献与整体变化的舍入差异。",
            "query_filters": dict(inherited),
        }
    except (AttributionDataError, KeyError, TypeError, ValueError) as exc:
        return {
            "components": [], "total_delta": float(total_delta),
            "attribution_status": "insufficient_data",
            "attribution_error": type(exc).__name__,
            "residual_value": float(total_delta), "residual_type": "unknown",
            "residual_explainable": False,
            "residual_tolerance": 0.03,
            "residual_note": "分组贡献无法通过数据校验，整体变化尚未归属到可验证分组。",
            "query_filters": dict(inherited),
        }


def _typed_ratio_contribution_values(result: dict[str, Any], query: Any) -> dict[str, Any]:
    """Calculate grouped ratio effects from complete server-returned sufficient facts."""
    from app.analysis.attribution_methods import (
        AttributionDataError, symmetric_mix_within_attribution,
    )
    from app.metrics.registry import get_registry

    envelope = result.get("result") if isinstance(result.get("result"), dict) else result
    payload = envelope.get("contribution_result") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        payload = envelope if isinstance(envelope, dict) else {}
    comparison = query.comparison
    if (
        str(payload.get("status", "")).casefold() != "ok"
        or payload.get("complete") is not True
        or payload.get("metric_name") != query.metric_name
        or payload.get("dimension") != query.dimension
        or query.metric_name not in {"freight_rate", "avg_order_value"}
        or query.dimension != "month_position"
        or comparison is None
    ):
        return {
            "components": [], "total_delta": None,
            "attribution_status": "incomplete_ratio_facts",
            "residual_value": None, "residual_type": "coverage",
            "residual_explainable": False,
            "residual_tolerance": 0.03,
            "residual_note": "比率分组事实不完整，覆盖残差无法量化。",
        }
    expected_periods = {
        "current_period": (comparison.current_start, comparison.current_end),
        "baseline_period": (comparison.baseline_start, comparison.baseline_end),
    }
    for name, expected in expected_periods.items():
        actual = payload.get(name)
        if not isinstance(actual, dict) or (actual.get("start_date"), actual.get("end_date")) != expected:
            return {
                "components": [], "total_delta": None,
                "attribution_status": "period_mismatch",
                "residual_value": None, "residual_type": "unknown",
                "residual_explainable": False,
                "residual_tolerance": 0.03,
                "residual_note": "比较区间不一致，残差不可解释。",
            }
    baseline_groups, current_groups = payload.get("baseline_groups"), payload.get("current_groups")
    if not isinstance(baseline_groups, dict) or not isinstance(current_groups, dict):
        return {
            "components": [], "total_delta": None,
            "attribution_status": "missing_ratio_groups",
            "residual_value": None, "residual_type": "coverage",
            "residual_explainable": False,
            "residual_tolerance": 0.03,
            "residual_note": "缺少比率分组充分统计量，覆盖残差无法量化。",
        }
    try:
        definition = get_registry().get(query.metric_name)
        aggregation = definition.period_aggregation
        capability = definition.attribution.for_scope("platform")
        if (
            aggregation is None
            or (aggregation.numerator_metric, aggregation.denominator_metric)
            not in {("freight_value", "gmv"), ("gmv", "order_count")}
            or "symmetric_mix_within_v1" not in capability.method_ids
            or query.dimension not in capability.dimensions
        ):
            raise AttributionDataError("ratio factors or grouped attribution are not registered for this scope")
        attribution = symmetric_mix_within_attribution(
            metric_id=query.metric_name, baseline_groups=baseline_groups,
            current_groups=current_groups, multiplier=aggregation.multiplier,
            unit=definition.unit,
        )
        totals = (
            (payload.get("baseline_numerator_total"), sum(float(row["numerator"]) for row in baseline_groups.values())),
            (payload.get("current_numerator_total"), sum(float(row["numerator"]) for row in current_groups.values())),
            (payload.get("baseline_denominator_total"), sum(float(row["denominator"]) for row in baseline_groups.values())),
            (payload.get("current_denominator_total"), sum(float(row["denominator"]) for row in current_groups.values())),
        )
        import math
        if any(
            actual is None or not math.isfinite(float(actual))
            or abs(float(actual) - expected) > 0.03
            for actual, expected in totals
        ):
            raise AttributionDataError("grouped ratio totals failed reconciliation")
    except (AttributionDataError, KeyError, TypeError, ValueError, OverflowError):
        return {
            "components": [], "total_delta": None,
            "attribution_status": "invalid_ratio_facts",
            "residual_value": None, "residual_type": "unknown",
            "residual_explainable": False,
            "residual_tolerance": 0.03,
            "residual_note": "比率分组事实未通过确定性计算与总量对账。",
        }

    effects_by_group: dict[str, float] = {}
    for item in attribution.raw_components:
        group = item.component_id.rsplit(":", 1)[0]
        effects_by_group[group] = effects_by_group.get(group, 0.0) + item.value
    components = []
    for group in sorted(set(baseline_groups) | set(current_groups)):
        base, current = baseline_groups.get(group), current_groups.get(group)
        components.append({
            "dimension_value": group,
            "delta_value": effects_by_group.get(group, 0.0),
            "baseline_value": (
                float(base["numerator"]) / float(base["denominator"]) * aggregation.multiplier
                if base and float(base["denominator"]) > 0 else None
            ),
            "current_value": (
                float(current["numerator"]) / float(current["denominator"]) * aggregation.multiplier
                if current and float(current["denominator"]) > 0 else None
            ),
            "component_type": "grouped_ratio_contribution",
        })
    return {
        "components": components,
        "complete": True,
        "dimension": query.dimension,
        "raw_components": [item.model_dump(mode="json") for item in attribution.raw_components],
        "baseline_groups": baseline_groups,
        "current_groups": current_groups,
        "multiplier": aggregation.multiplier,
        "baseline_value": attribution.baseline_value,
        "current_value": attribution.current_value,
        "total_delta": attribution.total_delta,
        "explained_share": attribution.reconciliation_ratio,
        "attribution_status": "ok",
        "method_id": attribution.method_id,
        "method_version": attribution.method_version,
        "residual_value": attribution.residual_value,
        "residual_type": attribution.residual_type,
        "residual_explainable": attribution.residual_explainable,
        "residual_tolerance": attribution.residual_tolerance,
        "residual_note": attribution.residual_note,
        "query_filters": dict(comparison.filters),
    }


def _record_graph_investigation_results(
    state: AgentState,
    action: Any,
    rows: list[tuple[Any, dict[str, Any], Any]],
) -> set[str]:
    """Apply the same shared Evidence/action/budget transition used by reports."""
    from app.investigation.candidates import candidate_id_for
    from app.investigation.contracts import (
        AnalyzeTrendAction, BreakdownDimensionAction, CandidateAction,
        CandidateFeatures, ComparePeriodAction, InvestigationState,
    )
    from app.investigation.engine import continue_after_candidate_data_gap, record_action_result

    investigation = _ensure_graph_investigation_state(state)
    if investigation is None:
        return set()
    selected_id = str(state.get("diagnostic_selected_candidate_id") or "")
    accepted_evidence_ids: set[str] = set()
    for index, (query, result, evidence) in enumerate(rows):
        selected = next((item for item in investigation.candidate_actions if item.candidate_id == selected_id), None)
        if selected is not None and len(rows) == 1:
            candidate = selected
        elif getattr(query, "period", None) is not None:
            typed_action = AnalyzeTrendAction(action_type="ANALYZE_TREND", metric_id=query.metric_name)
            candidate = CandidateAction(
                candidate_id=candidate_id_for(typed_action), action=typed_action,
                features=CandidateFeatures(rule_score=0.65, evidence_gap=f"trend:{query.metric_name}"),
            )
        elif getattr(query, "comparison", None) is not None:
            comparison = query.comparison
            typed_action = BreakdownDimensionAction(
                action_type="DRILL_DOWN" if comparison.parent_evidence_id else "BREAKDOWN_DIMENSION",
                metric_id=query.metric_name,
                dimension=query.dimension or "unknown",
                parent_evidence_id=comparison.parent_evidence_id,
                inherited_filters=dict(comparison.filters or {}),
                method_id="additive_change_v1",
            )
            candidate = CandidateAction(
                candidate_id=candidate_id_for(typed_action), action=typed_action,
                features=CandidateFeatures(rule_score=0.8, evidence_gap=f"breakdown:{query.metric_name}:{query.dimension}"),
            )
        else:
            typed_action = ComparePeriodAction(action_type="COMPARE_PERIOD", metric_id=query.metric_name)
            candidate = CandidateAction(
                candidate_id=candidate_id_for(typed_action), action=typed_action,
                features=CandidateFeatures(rule_score=0.95, evidence_gap=f"comparison:{query.metric_name}"),
            )
        # The typed observation was initially keyed by its evidence ID. Bind
        # it to the authoritative server candidate selected/executed this
        # round so the shared engine and Trace can verify action provenance.
        evidence = evidence.model_copy(update={"source_action_id": candidate.candidate_id})
        status_payload = result.get("result") if isinstance(result.get("result"), dict) else result
        status = str((status_payload or {}).get("status", "unknown")).casefold()
        failure_context = (
            status_payload.get("failure_context")
            if isinstance(status_payload, dict)
            and isinstance(status_payload.get("failure_context"), dict)
            else None
        )
        from app.core.failure_context import public_failure_message
        # A tool may successfully execute while reporting an incomplete data
        # result (for example, the requested period has missing observations).
        # Missing contribution fields in that typed limitation are not a
        # contradiction: reserve INVALID_DATA for a nominally successful
        # result whose attribution payload fails its deterministic checks.
        attribution_invalid = (
            status == "ok"
            and getattr(query, "comparison", None) is not None
            and evidence.values.get("attribution_status") != "ok"
        )
        evidence_rows = [] if status != "ok" or attribution_invalid else [evidence]
        failure_code = (
            "INVALID_DATA" if attribution_invalid
            else "AUTHORIZATION_BLOCKED" if (
                failure_context and failure_context.get("category") == "authorization_denied"
            )
            else "CAPABILITY_GAP" if (
                status == "unsupported"
                or (failure_context and failure_context.get("category") == "capability_gap")
            )
            else "EXECUTION_FAILED" if status in {
                "failed", "timeout", "error", "execution_failed",
            } or (failure_context and failure_context.get("category") in {
                "tool_timeout", "database_error", "execution_error",
            })
            else "DATA_INSUFFICIENT"
        )
        accepted = record_action_result(
            investigation,
            candidate,
            evidence_rows,
            current_period={"start": investigation.target.current_start.isoformat(), "end": investigation.target.current_end.isoformat()},
            baseline_period={"start": investigation.target.baseline_start.isoformat(), "end": investigation.target.baseline_end.isoformat()},
            decision_source=state.get("diagnostic_decision_source", "unknown"),
            failure_code=failure_code,
            failure_detail=(
                "分组贡献未通过确定性归因数据完整性校验。"
                if attribution_invalid
                else public_failure_message(failure_context)
                if failure_context
                else f"调查动作 {query.tool_name} 返回状态 {status}，未产生可用证据。"
            ),
            failure_context=failure_context,
        )
        if accepted:
            accepted_evidence_ids.add(evidence.evidence_id)
        if not accepted:
            action_result = investigation.action_results[-1] if investigation.action_results else None
            failure_code = (
                action_result.failure_code if action_result is not None
                else (investigation.stop_reason.code if investigation.stop_reason else "DATA_INSUFFICIENT")
            )
            failure_detail = (
                action_result.failure_detail if action_result is not None
                else (investigation.stop_reason.detail if investigation.stop_reason else "Evidence 未通过调查引擎校验。")
            )
            if failure_code == "DATA_INSUFFICIENT":
                _record_candidate_data_gap(
                    state,
                    investigation,
                    action_result,
                    fallback_candidate=candidate,
                    failed_query=query,
                    failure_detail=failure_detail,
                )
                continue
            state.setdefault("errors", []).append({
                "code": f"investigation_{failure_code.casefold()}",
                "message": failure_detail,
                "agent_name": "observe",
                **({"failure_context": failure_context} if failure_context else {}),
                **({"request_id": failure_context.get("request_id")} if failure_context else {}),
            })
            break
    state["investigation_state"] = investigation.model_dump(mode="json")
    return accepted_evidence_ids


def observe(state: AgentState) -> AgentState:
    action = _ACTION_ADAPTER.validate_python(state.get("diagnostic_action", {}))
    results = state.pop("diagnostic_tool_results", [])
    if isinstance(action, ExecuteDecompositionAction):
        from app.investigation.engine import record_action_result
        from app.investigation.contracts import InvestigationState

        typed = state.pop("diagnostic_local_attribution_evidence", None)
        investigation = _ensure_graph_investigation_state(state)
        selected_id = str(state.get("diagnostic_selected_candidate_id") or "")
        candidate = next((
            item for item in investigation.candidate_actions
            if item.candidate_id == selected_id == action.candidate_id
        ), None) if investigation else None
        if (
            typed is None
            and investigation is not None
            and (state.get("policy_decision") or {}).get("decision") == "deny"
        ):
            # A typed policy refusal already carries the authoritative stop
            # code (for example INVALID_DATA or DATA_INSUFFICIENT). Do not
            # relabel that refusal as a missing execution result.
            state["investigation_state"] = investigation.model_dump(mode="json")
            state["node_history"].append({
                "node": "observe", "summary": "registered decomposition denied by policy",
            })
        elif typed is None or candidate is None or investigation is None:
            state.setdefault("errors", []).append({
                "code": "diagnostic_decomposition_result_missing",
                "message": "服务端分解动作没有产生可入账的授权证据。",
                "agent_name": "observe",
            })
            _set_investigation_stop_reason(state, "CAPABILITY_GAP", "分解未产生可验证证据。")
        else:
            evidence_id = typed.evidence_id
            target = investigation.target
            accepted = record_action_result(
                investigation, candidate, [typed],
                current_period={"start": target.current_start.isoformat(), "end": target.current_end.isoformat()},
                baseline_period={"start": target.baseline_start.isoformat(), "end": target.baseline_end.isoformat()},
                decision_source=state.get("diagnostic_decision_source", "unknown"),
            )
            state["diagnostic_investigation_action_count"] = int(
                state.get("diagnostic_investigation_action_count", 0)
            ) + 1
            if not accepted:
                state.setdefault("errors", []).append({
                    "code": "diagnostic_decomposition_evidence_rejected",
                    "message": "共享调查引擎拒绝了分解证据的范围、指标或方法校验。",
                    "agent_name": "observe",
                })
                state["investigation_state"] = investigation.model_dump(mode="json")
                return state
            state.setdefault("root_cause_evidence", []).append({
                "purpose": f"metric_decomposition:{typed.metric_id}",
                "metric_id": typed.metric_id,
                "query_id": evidence_id,
                "result": {"status": "ok", "result": typed.values},
                "investigation_evidence": typed.model_dump(mode="json"),
            })
            state.setdefault("evidence_bundles", []).append({
                "evidence_id": evidence_id,
                "source_type": "deterministic_attribution",
                "source_ref": typed.source_ref,
                "content": {"result": typed.values, "claim": typed.claim},
                "trusted": True,
            })
            state["investigation_state"] = investigation.model_dump(mode="json")
            state.setdefault("observations", []).append(Observation(
                observation_id=f"obs_{len(state.get('observations', [])) + 1}",
                action_type="DECOMPOSE_METRIC",
                summary=f"已完成 {typed.metric_id} 的确定性归因分解。",
                evidence_ids=[evidence_id],
                payload={"method_id": typed.method_id, "method_version": typed.method_version},
            ).model_dump(mode="json"))
            state["node_history"].append({
                "node": "observe", "summary": f"deterministic decomposition evidence={evidence_id}",
            })
    elif isinstance(action, (CallToolAction, ParallelFactsAction)):
        target = str(state.get("diagnostic_target_metric") or "")
        queries = action.facts if isinstance(action, ParallelFactsAction) else [action]
        evidence_ids = []
        investigation_evidence = []
        pending_root_evidence: list[dict[str, Any]] = []
        evidence_kind_by_id: dict[str, str] = {}
        investigation = _ensure_graph_investigation_state(state)
        selected_id = str(state.get("diagnostic_selected_candidate_id") or "")
        selected_candidate = next((
            candidate for candidate in investigation.candidate_actions
            if candidate.candidate_id == selected_id
        ), None) if investigation else None
        selected_analysis_action = selected_candidate.action if selected_candidate else None
        for query, result in zip(queries, results):
            action_type = getattr(selected_analysis_action, "action_type", "")
            # Reusing a query does not make different derived checks the same
            # evidence. Keep the raw bundle and typed result on one identity.
            eid = _record_result(state, query, result, derived_action_id=(
                selected_id if action_type in {"CHECK_ANOMALY", "CHECK_DATA_QUALITY"} else None
            ))
            evidence_ids.append(eid)
            envelope = result.get("result") if isinstance(result.get("result"), dict) else result
            envelope_status = str(
                envelope.get("status", result.get("status", "unknown"))
            ).casefold()
            from app.investigation.series_analysis import (
                check_series_quality, detect_series_anomalies_for_payload,
                extract_query_payload,
            )
            query_payload = extract_query_payload(envelope if isinstance(envelope, dict) else {})
            # Some MCP responses keep the execution status on the outer result
            # envelope while placing points under ``trend_result``. Carry the
            # authoritative status into typed evidence so downstream validators
            # can distinguish a valid empty/partial response from a failed call.
            typed_query_payload = dict(query_payload)
            typed_query_payload.setdefault("status", envelope_status)
            derived_analysis: dict[str, Any] | None = None
            if action_type == "ANALYZE_DISTRIBUTION":
                derived_analysis = {
                    "status": query_payload.get("status", "unknown"),
                    "distribution_basis": "time_bucket_values",
                    "count": query_payload.get("count"),
                    "statistics": query_payload.get("statistics", {}),
                    "min_bucket": query_payload.get("min_bucket"),
                    "max_bucket": query_payload.get("max_bucket"),
                }
            elif action_type == "ANALYZE_ENTITY_DISTRIBUTION":
                derived_analysis = {
                    "status": query_payload.get("status", "unknown"),
                    "distribution_basis": query_payload.get("basis"),
                    "entity": query_payload.get("entity"),
                    "sample_size": query_payload.get("sample_size"),
                    "statistics": query_payload.get("statistics", {}),
                    "zero_freight_orders": query_payload.get("zero_freight_orders"),
                    "excluded_missing_freight_orders": query_payload.get("excluded_missing_freight_orders"),
                    "population_definition": query_payload.get("population_definition"),
                }
            elif action_type == "ANALYZE_ASSOCIATION":
                derived_analysis = dict(query_payload)
            elif action_type == "CHECK_DATA_QUALITY":
                check_id = getattr(selected_analysis_action, "check_id", "")
                derived_analysis = check_series_quality(query_payload, check_id)
            elif action_type == "CHECK_ANOMALY":
                scope = state.get("controlled_analysis_scope") or {}
                period = query.period
                try:
                    from app.metrics.registry import get_registry
                    metric_definition = get_registry().get(str(query.metric_name))
                    derived_analysis = detect_series_anomalies_for_payload(
                        query_payload,
                        metric_id=str(query.metric_name),
                        unit=metric_definition.unit,
                        start_date=period.start_date,
                        end_date=period.end_date,
                        granularity=period.granularity,
                        definition_version=metric_definition.version,
                        source_ref=str(query_payload.get("source_table") or query.tool_name),
                    ) if period else {"status": "invalid_period", "anomalies": []}
                except Exception as exc:
                    derived_analysis = {
                        "status": "invalid_series",
                        "error_code": type(exc).__name__,
                        "anomalies": [],
                    }
            if action_type == "ANALYZE_DISTRIBUTION":
                purpose = f"distribution_analysis:{query.metric_name}"
                method_id = "descriptive_distribution_v1"
                evidence_type = "distribution_summary"
            elif action_type == "ANALYZE_ENTITY_DISTRIBUTION":
                purpose = f"entity_distribution:{query.metric_name}"
                method_id = "order_entity_distribution_v1"
                evidence_type = "entity_distribution_summary"
            elif action_type == "ANALYZE_ASSOCIATION":
                purpose = f"registered_association:{getattr(query, 'association_pair_id', '')}"
                from app.analysis.association_catalog import get_association_pair
                pair = get_association_pair(str(getattr(query, "association_pair_id", "")))
                if pair is None or pair.target_metric_id != query.metric_name:
                    raise ValueError("Association Evidence references an unregistered pair or metric")
                method_id = pair.method_id
                evidence_type = "registered_association"
            elif action_type == "CHECK_ANOMALY":
                purpose = f"anomaly_check:{query.metric_name}"
                method_id = "rolling_series_anomaly_v1"
                evidence_type = "time_series_anomaly"
            elif action_type == "CHECK_DATA_QUALITY":
                purpose = f"data_quality_check:{query.metric_name}"
                method_id = "time_series_quality_check_v1"
                evidence_type = "time_series_quality"
            elif query.period:
                # A trend on the target metric is the baseline; a trend on any
                # other metric is cross-metric (related) evidence.
                purpose = (
                    "trend_baseline"
                    if not target or query.metric_name == target
                    else f"cross_metric:{query.metric_name}"
                )
                method_id = "observed_trend_v1"
                evidence_type = "trend"
            elif query.tool_name in {
            "compare_breakdown_by_dimension",
            "compare_freight_rate_ratio_facts_by_dimension_period",
            "compare_avg_order_value_ratio_facts_by_dimension_period",
        }:
                purpose = f"dimension_contribution:{query.dimension}"
                if query.tool_name in {
                    "compare_freight_rate_ratio_facts_by_dimension_period",
                    "compare_avg_order_value_ratio_facts_by_dimension_period",
                }:
                    method_id = "symmetric_mix_within_v1"
                    evidence_type = "grouped_ratio_attribution"
                else:
                    method_id = "additive_change_v1"
                    evidence_type = "dimension_contribution"
            else:
                purpose = f"cross_metric:{query.metric_name}"
                method_id = "observed_metric_v1"
                evidence_type = "metric_observation"
            evidence_kind_by_id[eid] = (
                "trend" if purpose == "trend_baseline" or purpose.startswith(
                    ("distribution_analysis:", "entity_distribution:", "anomaly_check:", "data_quality_check:")
                )
                else "dimension_contribution" if purpose.startswith("dimension_contribution:")
                else "related_metric"
            )
            from app.investigation.evidence import build_attribution_evidence
            raw_scope = state.get("controlled_analysis_scope") or {
                "tenant_id": state.get("tenant_id"),
                "user_id": state.get("user_id"),
            }
            typed_evidence = build_attribution_evidence(
                evidence_id=eid,
                evidence_type=evidence_type,
                trace_id=str(state.get("trace_id") or "") or None,
                metric_id=str(query.metric_name),
                claim=(
                    "描述统计针对所选粒度下的时间桶指标值，不是订单或商品明细的分布。"
                    if action_type == "ANALYZE_DISTRIBUTION"
                    else "受控查询返回的观测数据；结论需受指标口径和数据质量限制。"
                ),
                values={
                    "query": _query_arguments(query),
                    "result": typed_query_payload,
                    **({"series_analysis": derived_analysis} if derived_analysis is not None else {}),
                    **(_typed_ratio_contribution_values(result, query)
                       if query.tool_name in {
                           "compare_freight_rate_ratio_facts_by_dimension_period",
                           "compare_avg_order_value_ratio_facts_by_dimension_period",
                       } else {}),
                    **(_typed_contribution_values(result, query)
                       if query.tool_name == "compare_breakdown_by_dimension" else {}),
                },
                source_action_id=eid,
                source_ref=str(query.tool_name),
                scope=raw_scope,
                method_id=method_id,
                method_version="1.0",
                authorized_metric_provenance="tool_gateway_policy_validated",
                comparison={
                    "period": query.period.model_dump(mode="json") if query.period else None,
                    "comparison": query.comparison.model_dump(mode="json") if query.comparison else None,
                },
                data_quality={"status": result.get("result", {}).get("status", result.get("status", "unknown"))},
                limitations=[
                    "统计关联和贡献证据不单独证明因果关系。",
                    *(["均值、分位数、最值等统计量针对查询返回的时间桶指标值；不代表底层订单/商品明细分布。"]
                      if action_type == "ANALYZE_DISTRIBUTION" else []),
                ],
            )
            pending_root_evidence.append({
                "purpose": purpose,
                "metric_id": query.metric_name, "query_id": eid, "result": result,
                "investigation_evidence": typed_evidence.model_dump(mode="json"),
            })
            investigation_evidence.append((query, result, typed_evidence))
        accepted_evidence_ids = _record_graph_investigation_results(state, action, investigation_evidence)
        # Raw MCP payloads are retained for audit, but become trusted Evidence
        # only after the shared engine accepts their typed scope/method/result.
        for bundle in state.get("evidence_bundles", []):
            if isinstance(bundle, dict) and bundle.get("evidence_id") in accepted_evidence_ids:
                bundle["trusted"] = True
                bundle["validation_status"] = "accepted_by_investigation_engine"
        for record in pending_root_evidence:
            if record.get("query_id") in accepted_evidence_ids:
                state.setdefault("root_cause_evidence", []).append(record)
        observed_kinds = {
            evidence_kind_by_id[evidence_id]
            for evidence_id in accepted_evidence_ids
            if evidence_id in evidence_kind_by_id
        }
        obs = Observation(
            observation_id=f"obs_{len(state.get('observations', [])) + 1}",
            action_type=action.action_type,
            summary=(
                f"已验证并入账 {len(accepted_evidence_ids)} 条证据；"
                f"隔离 {len(investigation_evidence) - len(accepted_evidence_ids)} 条未通过校验的结果。"
            ),
            evidence_ids=sorted(accepted_evidence_ids),
            payload={
                "result_count": len(results),
                "accepted_evidence_count": len(accepted_evidence_ids),
                "rejected_evidence_count": len(investigation_evidence) - len(accepted_evidence_ids),
            },
        )
        state.setdefault("observations", []).append(obs.model_dump(mode="json"))
        # Review fix #2: a step is completed only when the observation actually
        # matches its evidence category — never "first pending step wins".
        plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
        step_state = state.setdefault("diagnostic_step_state", {})
        remaining_kinds = set(observed_kinds)
        for step in plan.steps:
            kind = _step_kind(step)
            if step_state.get(step.step_id, "pending") not in {"pending", "data_insufficient"}:
                continue
            if kind is not None and kind in remaining_kinds:
                step_state[step.step_id] = "completed"
                remaining_kinds.discard(kind)
        for step in plan.steps:  # legacy kind-less steps match leftover evidence
            kind = _step_kind(step)
            if kind is None and remaining_kinds and step_state.get(step.step_id, "pending") == "pending":
                step_state[step.step_id] = "completed"
                remaining_kinds.clear()
        if observed_kinds:
            state["diagnostic_no_progress_rounds"] = 0
        else:
            state["diagnostic_no_progress_rounds"] = int(state.get("diagnostic_no_progress_rounds", 0)) + 1
    investigation_count = int(state.get("diagnostic_investigation_action_count", 0))
    if investigation_count > 0:
        from app.core.execution_context import enqueue_sse_event
        from app.schemas.events import SseInvestigationProgressEvent

        action_label = action.action_type
        if isinstance(action, CallToolAction):
            action_label = f"{action.metric_name}:{action.dimension or action.tool_name}"
        enqueue_sse_event(SseInvestigationProgressEvent(
            completed_actions=min(6, investigation_count),
            remaining_budget=max(0, 6 - investigation_count),
            status="in_progress" if investigation_count < 6 else "stalled",
            current_action=action_label,
        ))
    state["node_history"].append({"node": "observe", "summary": f"observations={len(state.get('observations', []))}"})
    return state


def _record_agent_run(state: AgentState, status: str) -> None:
    """Assemble the shared runtime AgentRun record (AGENT_PLAN Phase 10)."""
    from app.agents.runtime import make_agent_run

    usage = state.get("llm_usage", {}) or {}
    initiator = state.get("termination_initiator", "unknown") if state.get("termination_initiator") in {"agent", "budget", "policy", "critic"} else "unknown"
    run = make_agent_run(
        run_id=str(state.get("trace_id") or "unknown"),
        domain="diagnostic",
        decision_driver="llm" if state.get("decision_driver") == "llm" else "workflow_fallback",
        status=status,
        termination_initiator=initiator,
        termination_reason=str((state.get("finish_decision", {}) or {}).get("reason") or (state.get("finish_decision", {}) or {}).get("residual_uncertainty") or ""),
        rounds=int(state.get("diagnostic_round", 0) or 0),
        tool_calls=len(state.get("tool_calls", []) or []),
        usage=usage,
        plan=dict(state.get("diagnostic_plan", {}) or {}),
        trace=state.get("node_history", []),
    )
    state["agent_run"] = run.model_dump(mode="json")


def _validated_share_text(
    *, share: Any, effect: Any, root_delta: Any, component_values: list[Any],
) -> str:
    """Render an absolute magnitude share without implying signed ownership."""
    if (
        not isinstance(share, (int, float)) or isinstance(share, bool)
        or not math.isfinite(float(share))
    ):
        return ""
    pieces = [f"绝对贡献幅度相当于根目标净变化绝对值的 {float(share) * 100:.1f}%"]
    if isinstance(effect, (int, float)) and not isinstance(effect, bool) and math.isfinite(float(effect)):
        if float(effect) > 0:
            pieces.append("该项为正向贡献")
        elif float(effect) < 0:
            pieces.append("该项为负向贡献")
        if (
            isinstance(root_delta, (int, float)) and not isinstance(root_delta, bool)
            and math.isfinite(float(root_delta)) and float(root_delta) != 0
            and float(effect) != 0
            and math.copysign(1.0, float(effect)) != math.copysign(1.0, float(root_delta))
        ):
            opposing_component_exists = any(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) != 0
                and math.copysign(1.0, float(value)) != math.copysign(1.0, float(effect))
                for value in component_values
            )
            if opposing_component_exists:
                pieces.append("与另一已验证贡献方向相反，形成数值抵消")
            else:
                pieces.append("方向与根目标净变化相反")
    return "（" + "；".join(pieces) + "）"


def _english_contribution_share_text(share: Any, effect: Any) -> str:
    if not isinstance(share, (int, float)) or isinstance(share, bool) or not math.isfinite(float(share)):
        return ""
    direction = (
        "positive" if isinstance(effect, (int, float)) and not isinstance(effect, bool) and effect > 0
        else "negative" if isinstance(effect, (int, float)) and not isinstance(effect, bool) and effect < 0
        else "zero"
    )
    opposing = "; this is an opposing contribution to the net change" if float(share) < 0 else ""
    return (
        f" (absolute contribution magnitude is {abs(float(share)) * 100:.1f}% of the "
        f"absolute net change; this item is a {direction} contribution{opposing})"
    )


def _render_unresolved_branch(
    branch: dict[str, Any], *, dimension_labels: dict[str, str],
    validated_evidence_ids: set[str], evidence_by_id: dict[str, dict[str, Any]],
    target_metric_id: str, target_metric_name: str, root_delta: Any,
    numeric_decomposition_valid: bool,
) -> str:
    """Render branch semantics by type; never treat every branch as a dimension."""
    def display_contribution_unit(unit: Any) -> str:
        canonical = str(unit or "").strip()
        return "个百分点" if canonical == "percentage_points" else canonical

    branch_type = str(branch.get("branch_type") or "")
    if not branch_type:
        # Match the legacy checkpoint migration in UnresolvedBranch. Never
        # surface the internal factor sentinel as a user-facing dimension.
        if str(branch.get("dimension") or "") == "__metric_factor__":
            branch_type = "metric_effect"
        elif not branch.get("dimension") and not branch.get("parent_branch_id"):
            branch_type = "root_target"
        else:
            branch_type = "dimension_contribution"
    status = str(branch.get("status") or "open")
    blocked = status == "blocked"
    parent_id = str(branch.get("parent_evidence_id") or "")
    evidence_valid = bool(parent_id and parent_id in validated_evidence_ids)
    evidence = evidence_by_id.get(parent_id) or {}
    values = evidence.get("values") or {}
    effect = branch.get("effect_value")
    if branch_type == "metric_effect":
        factor_metrics = values.get("factor_metrics") or []
        components = values.get("display_components") or values.get("components") or []
        component_by_id = {
            str(item.get("component_id")): item for item in components if isinstance(item, dict)
        }
        method_id = str(evidence.get("method_id") or values.get("method_id") or "")
        factor_id = str(branch.get("metric_id") or "")
        if method_id == "symmetric_ratio_v1":
            try:
                from app.metrics.registry import get_registry
                aggregation = get_registry().get(target_metric_id).period_aggregation
                expected = {
                    "numerator_effect": aggregation.numerator_metric,
                    "denominator_effect": aggregation.denominator_metric,
                } if aggregation is not None and aggregation.kind == "weighted_ratio" else {}
                identities = values.get("factor_identities") or []
                mapped = {
                    str(item.get("role_id")): str(item.get("metric_id"))
                    for item in identities if isinstance(item, dict)
                }
                if identities and (len(identities) != 2 or mapped != expected):
                    evidence_valid = False
                role = next((role for role, metric in expected.items() if metric == factor_id), None)
                component = component_by_id.get(role or "")
                evidence_valid = bool(
                    evidence_valid and evidence.get("evidence_type") == "metric_decomposition"
                    and str(evidence.get("metric_id") or "") == target_metric_id
                    and factor_metrics == [expected.get("numerator_effect"), expected.get("denominator_effect")]
                    and role and component is not None
                )
            except Exception:  # malformed registry or evidence fails closed
                evidence_valid = False
        else:
            component = component_by_id.get(factor_id)
            evidence_valid = bool(
                evidence_valid and evidence.get("evidence_type") == "metric_decomposition"
                and str(evidence.get("metric_id") or "") == target_metric_id
                and factor_id in factor_metrics and component is not None
            )
        component_value = component.get("value") if isinstance(component, dict) else None
        evidence_valid = bool(
            evidence_valid
            and isinstance(effect, (int, float)) and not isinstance(effect, bool)
            and isinstance(component_value, (int, float)) and not isinstance(component_value, bool)
            and math.isclose(float(effect), float(component_value), rel_tol=1e-9, abs_tol=0.01)
        )
    elif branch_type == "dimension_contribution":
        dimension = str(branch.get("dimension") or "")
        group = str(branch.get("dimension_value") or "")
        components = values.get("components") or []
        matching = [
            item for item in components if isinstance(item, dict)
            and str(item.get("dimension_value") or "") == group
        ]
        matched = matching[0] if len(matching) == 1 else None
        evidence_valid = bool(
            evidence_valid and evidence.get("evidence_type") == "dimension_contribution"
            and str(evidence.get("metric_id") or "") == str(branch.get("metric_id") or "")
            and str(values.get("dimension") or "") == dimension
            and matched is not None
            and isinstance(effect, (int, float)) and not isinstance(effect, bool)
            and isinstance(matched.get("delta_value"), (int, float))
            and math.isclose(float(effect), float(matched["delta_value"]), rel_tol=1e-9, abs_tol=0.01)
        )
    raw_components = values.get("components") or values.get("display_components") or []
    component_values = [
        item.get("delta_value", item.get("value"))
        for item in raw_components if isinstance(item, dict)
    ]
    share = branch.get("root_effect_share")
    if share is None:
        share = branch.get("effect_share_of_root")
    share_text = _validated_share_text(
        share=share if evidence_valid else None,
        effect=effect if evidence_valid else None,
        root_delta=root_delta,
        component_values=component_values if evidence_valid else [],
    )
    factor_id = str(branch.get("metric_id") or "")
    try:
        from app.metrics.registry import get_registry
        factor_name = get_registry().get(factor_id).display_name or factor_id
    except Exception:  # malformed/legacy branch labels remain visible but untrusted
        factor_name = factor_id or "未知指标"
    state_label = "受数据覆盖限制" if blocked else "尚未完成深入调查"

    if branch_type == "root_target":
        if numeric_decomposition_valid and factor_id == target_metric_id:
            follow_up = "根目标后续调查受数据覆盖限制；" if blocked else ""
            return f"分析目标「{target_metric_name}」的注册数值分解已验证；{follow_up}业务原因仍未确认。"
        return f"分析目标「{target_metric_name}」：目标指标调查{state_label}。"
    if branch_type == "metric_effect":
        if evidence_valid:
            effect = branch.get("effect_value")
            # The decomposition component carries the unit of the absolute
            # contribution (e.g. percentage_points); evidence.unit describes
            # the target metric's level and can therefore be "%" instead.
            effect_unit = (
                str(component.get("unit") or "").strip()
                if isinstance(component, dict) else ""
            ) or str(branch.get("effect_unit") or "").strip()
            effect_unit = display_contribution_unit(effect_unit)
            effect_text = (
                f"数值贡献 {float(effect):+.2f} {effect_unit}".rstrip()
                if isinstance(effect, (int, float)) and not isinstance(effect, bool)
                else "数值贡献已验证"
            )
            return (
                f"指标因素「{factor_name}」对「{target_metric_name}」变化的{effect_text}{share_text}；"
                f"该因素背后的业务原因仍未确认。"
            )
        return f"指标因素「{factor_name}」的后续调查{state_label}，尚无可引用的有效贡献证据。"
    if branch_type == "dimension_contribution":
        dimension_id = str(branch.get("dimension") or "")
        dimension_name = dimension_labels.get(dimension_id, dimension_id or "维度")
        group_name = str(branch.get("dimension_value") or "未细分")
        if evidence_valid:
            effect = branch.get("effect_value")
            effect_unit = display_contribution_unit(branch.get("effect_unit"))
            effect_text = (
                f"贡献 {float(effect):+.2f} {effect_unit}".rstrip()
                if isinstance(effect, (int, float)) and not isinstance(effect, bool)
                else "贡献结果已验证"
            )
            return f"维度贡献「{dimension_name}={group_name}」：{effect_text}{share_text}；{state_label}。"
        return f"维度调查「{dimension_name}={group_name}」：{state_label}，未采用未验证证据。"
    if branch_type == "business_driver":
        label = str(branch.get("label") or factor_name)
        return f"业务原因线索「{label}」：{state_label}；尚未确认因果关系。"
    if branch_type == "data_quality":
        label = str(branch.get("label") or "数据质量检查")
        return f"数据质量检查「{label}」：{state_label}。"
    return f"调查分支「{branch.get('label') or branch_type}」：{state_label}。"


def diagnostic_finalize(state: AgentState) -> AgentState:
    token_budget_exhausted = any(
        isinstance(error, dict) and error.get("code") == "token_budget_exceeded"
        for error in (state.get("errors") or [])
    )
    raw_action = state.get("diagnostic_action")
    if token_budget_exhausted and not raw_action:
        action = AbstainAction(action_type="ABSTAIN", reason="budget_exhausted")
    else:
        action = _ACTION_ADAPTER.validate_python(raw_action or {})
    investigation_before_finalize = state.get("investigation_state") or {}
    stop_before_finalize = (
        investigation_before_finalize.get("stop_reason")
        if isinstance(investigation_before_finalize, dict)
        else None
    )
    if (
        isinstance(stop_before_finalize, dict)
        and str(stop_before_finalize.get("code") or "") in {
            "INVALID_DATA", "AUTHORIZATION_BLOCKED", "EXECUTION_FAILED",
            "CAPABILITY_GAP", "DATA_INSUFFICIENT", "USER_SCOPE_LIMIT",
            "BUDGET_EXHAUSTED",
        }
        and not isinstance(action, AbstainAction)
    ):
        # Observe can route a hard-stopped investigation directly to
        # finalization while diagnostic_action still contains the attempted
        # CALL_TOOL/PARALLEL_FACTS/FINISH action. It is not a valid terminal
        # decision after a hard stop; abstain while retaining the structured
        # stop reason and only already-validated evidence.
        action = AbstainAction(
            action_type="ABSTAIN",
            reason="insufficient_evidence",
            supporting_evidence_ids=_completed_valid_evidence_ids(state),
        )
        state["diagnostic_action"] = action.model_dump(mode="json")
    known = {
        item.get("evidence_id")
        for item in state.get("evidence_bundles", [])
        if isinstance(item, dict) and item.get("trusted") is True and item.get("evidence_id")
    }
    # A target metric is the symptom being explained, never a root cause of
    # itself.  Providers may still return FINISH(found_factor=target_metric)
    # after correctly observing that every related metric is flat.  Convert
    # that structurally invalid diagnosis to a truthful abstention before it
    # reaches analysis_result or the evaluator.
    target_metric = str(state.get("diagnostic_target_metric") or "").strip().casefold()
    registered_numeric_decomposition = (
        (state.get("registered_decomposition_request") or {}).get("status") == "ready"
    )
    found_factor = str(getattr(action, "found_factor", "") or "").strip().casefold()
    capability_limitations = [
        text
        for item in state.get("diagnostic_capability_gaps", [])
        if isinstance(item, dict)
        if (text := _capability_gap_text(item))
    ]
    from app.investigation.causal_readiness import assess_causal_readiness
    causal_readiness = assess_causal_readiness(state.get("root_cause_evidence", []))
    state["causal_readiness"] = causal_readiness
    investigation = state.get("investigation_state") or {}
    investigation_summary = None
    investigation_limitations: list[str] = []
    unvalidated_evidence_ids: set[str] = set()
    validated_evidence_ids: set[str] = set()
    if isinstance(investigation, dict):
        target_data = investigation.get("target") or {}
        target_scope_fingerprint = (
            target_data.get("scope_fingerprint")
            if isinstance(target_data, dict)
            else getattr(target_data, "scope_fingerprint", None)
        )
        investigation_target_metric = str(
            (target_data.get("metric_id") if isinstance(target_data, dict)
             else getattr(target_data, "metric_id", "")) or ""
        ).strip().casefold()
        requested_target_metric = str(state.get("diagnostic_target_metric") or "").strip().casefold()
        target_metric_mismatch = bool(
            investigation_target_metric and requested_target_metric
            and investigation_target_metric != requested_target_metric
        )
        typed_evidence_by_id: dict[str, dict[str, Any]] = {}
        duplicate_typed_evidence_ids: set[str] = set()
        for evidence in investigation.get("evidence", []):
            if not isinstance(evidence, dict) or not evidence.get("evidence_id"):
                continue
            evidence_id = str(evidence["evidence_id"])
            if evidence_id in typed_evidence_by_id:
                duplicate_typed_evidence_ids.add(evidence_id)
            else:
                typed_evidence_by_id[evidence_id] = evidence
        for result in investigation.get("action_results", []):
            if not isinstance(result, dict):
                continue
            validation = result.get("validation_result") or {}
            validation_status = (
                str(validation.get("status") or "unknown")
                if isinstance(validation, dict)
                else str(getattr(validation, "status", "unknown") or "unknown")
            )
            result_evidence_ids = {
                str(value) for value in (result.get("evidence_ids") or []) if value
            }
            result_is_valid = result.get("status") == "completed" and validation_status == "valid"
            for evidence_id in result_evidence_ids & known:
                evidence = typed_evidence_by_id.get(evidence_id)
                provenance_matches = (
                    evidence_id not in duplicate_typed_evidence_ids
                    and evidence is not None
                    and str(evidence.get("source_action_id") or "") == str(result.get("candidate_id") or "")
                    and str(evidence.get("scope_fingerprint") or "") == str(target_scope_fingerprint or "")
                    and (
                        not result.get("metric_id")
                        or str(evidence.get("metric_id") or "") == str(result.get("metric_id"))
                    )
                    and (
                        not result.get("method_id")
                        or str(evidence.get("method_id") or "") == str(result.get("method_id"))
                    )
                    and (
                        not evidence.get("trace_id")
                        or not state.get("trace_id")
                        or str(evidence.get("trace_id")) == str(state.get("trace_id"))
                    )
                )
                if result_is_valid and provenance_matches:
                    validated_evidence_ids.add(evidence_id)
                else:
                    unvalidated_evidence_ids.add(evidence_id)
        # A trusted-looking bundle is still not a conclusion source unless it
        # is linked to a completed, base-validated action and typed Evidence
        # with matching action, metric, scope, method and (when present) Trace.
        # This also fails closed for legacy checkpoints missing typed state.
        unvalidated_evidence_ids.update(_conflicting_evidence_ids(investigation))
        unvalidated_evidence_ids.update(known - validated_evidence_ids)
        validated_evidence_ids -= unvalidated_evidence_ids
        if target_metric_mismatch:
            validated_evidence_ids.clear()
            unvalidated_evidence_ids.update(known)
            state.setdefault("diagnostic_guardrails", []).append({
                "code": "investigation_root_target_mismatch",
                "requested_metric_id": requested_target_metric,
                "investigation_metric_id": investigation_target_metric,
            })
            investigation_limitations.append(
                "调查账本的根指标与本轮请求不一致，已停止使用该调查结果形成结论。"
            )
        if unvalidated_evidence_ids:
            investigation_limitations.append(
                "部分证据未通过完整性校验，或未关联到已完成且校验通过的调查动作，因此未用于确定性调查结论。"
            )
        stop = investigation.get("stop_reason") or {}
        if isinstance(stop, dict) and stop.get("code") == "BUDGET_EXHAUSTED":
            investigation_limitations.append(
                "后续调查因预算耗尽而停止；该全局停止原因仍然有效，不改变此前独立校验通过的数值贡献。"
            )
        progress = investigation.get("progress") or {}
        branches = [
            branch for branch in investigation.get("branches", [])
            if isinstance(branch, dict)
            and branch.get("status") in {"open", "investigating", "low_priority", "blocked", "skipped"}
        ]
        if stop or branches or progress:
            investigation_summary = {
                "stop_reason": stop,
                "progress": progress,
                "calculation_progress": investigation.get("calculation_progress") or {},
                "remaining_budget": investigation.get("remaining_budget"),
                "action_result_count": len(investigation.get("action_results", [])),
                "action_results": [
                    {
                        key: item.get(key)
                        for key in (
                            "action_id", "candidate_id", "action_type", "metric_id",
                            "dimension", "method_id", "status", "failure_code",
                            "failure_detail", "failure_context", "calculation_progress", "validation_result",
                            "evidence_ids",
                        )
                        if key in item
                    }
                    for item in investigation.get("action_results", [])[:6]
                    if isinstance(item, dict)
                ],
                "unresolved_branches": branches,
            }
        stop_labels = {
            "ENOUGH_EVIDENCE": "当前证据达到本轮调查停止条件",
            "NO_HIGH_VALUE_ACTION": "当前没有剩余的高价值合法调查动作",
            "CAPABILITY_GAP": "指标或查询能力不足",
            "DATA_INSUFFICIENT": "数据或样本不足",
            "BUDGET_EXHAUSTED": "调查动作预算已用尽",
            "USER_SCOPE_LIMIT": "已达到用户范围或下钻深度限制",
            "AUTHORIZATION_BLOCKED": "授权校验阻止了后续调查",
            "INVALID_DATA": "\u8bc1\u636e\u6570\u636e\u4e0d\u4e00\u81f4",
            "EXECUTION_FAILED": "调查查询执行失败",
            "COMPLETED": "候选调查动作已处理完毕",
        }
        if stop:
            code = str(stop.get("code") or "")
            detail = str(stop.get("detail") or "").strip()
            label = stop_labels.get(code, code or "未知")
            failure_context = stop.get("failure_context")
            failure_note = ""
            if isinstance(failure_context, dict):
                category = str(failure_context.get("category") or "")
                service = str(failure_context.get("service") or "")
                failure_code = str(failure_context.get("code") or "")
                if category or service or failure_code:
                    failure_note = (
                        f"失败分类：{category or 'unknown'}；"
                        f"执行服务：{service or 'unknown'}；"
                        f"错误码：{failure_code or 'unknown'}。"
                    )
            investigation_limitations.append(
                f"调查停止状态：{label}。"
                + (f"详情：{detail}" if detail else "")
                + (f" {failure_note}" if failure_note else "")
            )
        if token_budget_exhausted:
            investigation_limitations.append(
                "Token budget exhausted; verified evidence was retained and further planning, model requests, and data queries were stopped."
            )
        if branches:
            dimension_labels = {
                "product_category": "商品类别", "seller_state": "卖家州/地区",
                "customer_state": "顾客州/地区", "customer_city": "顾客城市",
                "payment_type": "支付方式", "month_position": "月内时间位置",
            }
            branch_numeric_decomposition = _metric_decomposition_projection(
                state, allow_partial_projection=True,
            )
            branch_target_id = str(target_data.get("metric_id") or investigation_target_metric or "")
            try:
                from app.metrics.registry import get_registry
                branch_target_name = get_registry().get(branch_target_id).display_name or branch_target_id
            except Exception:  # no display metadata must not change investigation state
                branch_target_name = branch_target_id or "目标指标"
            root_delta = (
                investigation.get("root_effect_value")
                if isinstance(investigation, dict) else None
            )
            if root_delta is None and branch_numeric_decomposition:
                root_delta = branch_numeric_decomposition.get("delta_value")
            branch_texts = []
            for branch in branches[:8]:
                rendered = _render_unresolved_branch(
                    branch,
                    dimension_labels=dimension_labels,
                    validated_evidence_ids=validated_evidence_ids,
                    evidence_by_id=typed_evidence_by_id,
                    target_metric_id=branch_target_id,
                    target_metric_name=branch_target_name,
                    root_delta=root_delta,
                    numeric_decomposition_valid=branch_numeric_decomposition is not None,
                )
                if rendered:
                    branch_texts.append(rendered)
            if branch_texts:
                investigation_limitations.append("调查分支状态：" + "；".join(branch_texts) + "。")
    if isinstance(investigation, dict):
        try:
            from app.investigation.contracts import ActionResult
            from app.investigation.engine import MIN_ANOMALY_VALID_BUCKETS
            valid_ids = set(_completed_valid_evidence_ids(state))
            evidence_by_id = {
                str(item.get("evidence_id")): item
                for item in investigation.get("evidence", [])
                if isinstance(item, dict) and item.get("evidence_id")
            }
            target = investigation.get("target") or {}
            for raw_action in investigation.get("action_results", []):
                if not isinstance(raw_action, dict) or raw_action.get("failure_code") != "DATA_INSUFFICIENT":
                    continue
                try:
                    scoped_optional, linked_evidence_id = _optional_data_gap_is_scoped_to_valid_decomposition(
                        state, ActionResult.model_validate(raw_action), allow_budget_stop_context=True,
                    )
                except Exception:  # malformed or unlinked records remain blocking
                    scoped_optional, linked_evidence_id = False, None
                if not scoped_optional or not linked_evidence_id:
                    continue
                detail = str(raw_action.get("failure_detail") or "required data was unavailable")
                action_name = str(raw_action.get("action_type") or "optional check")
                metric_name = str(raw_action.get("metric_id") or "")
                count_text = ""
                if action_name == "CHECK_ANOMALY":
                    target_start = str(target.get("baseline_start") or "")
                    target_end = str(target.get("current_end") or "")
                    matching_series = []
                    for eid in valid_ids:
                        evidence = evidence_by_id.get(eid) or {}
                        if (
                            evidence.get("evidence_type") != "trend"
                            or str(evidence.get("metric_id") or "") != metric_name
                            or evidence.get("scope_fingerprint") != target.get("scope_fingerprint")
                        ):
                            continue
                        values = evidence.get("values") or {}
                        period = ((values.get("query") or {}).get("period") or {})
                        if str(period.get("start_date") or "") != target_start or str(period.get("end_date") or "") != target_end:
                            continue
                        points = ((values.get("result") or {}).get("points") or [])
                        count = sum(
                            1 for point in points if isinstance(point, dict)
                            and isinstance(point.get("value"), (int, float))
                            and not isinstance(point.get("value"), bool)
                        )
                        matching_series.append(count)
                    if len(matching_series) == 1:
                        count_text = (
                            f"Observed {matching_series[0]} valid buckets; the public minimum is "
                            f"{MIN_ANOMALY_VALID_BUCKETS}. "
                        )
                investigation_limitations.append(
                    f"Optional deep check {action_name} {metric_name} was not completed: "
                    f"{count_text}{detail}. This local limitation does not invalidate the separately validated "
                    f"numeric evidence {linked_evidence_id}."
                )
        except Exception:  # noqa: BLE001 - do not turn malformed state into an optional limitation
            pass
    if registered_numeric_decomposition:
        common_limitations = [
            "数值分解只解释已验证比较期的统计变化，不证明业务因果。",
            *investigation_limitations,
            *capability_limitations,
        ]
    else:
        common_limitations = [
            "维度统计贡献和相关性不构成已确认的业务因果。",
            f"业务因果尚未确认：{causal_readiness['reason']}",
            *investigation_limitations,
            *capability_limitations,
        ]
    if token_budget_exhausted:
        independent_errors = [
            error for error in (state.get("errors") or [])
            if isinstance(error, dict) and error.get("code") != "token_budget_exceeded"
        ]
        action = AbstainAction(
            action_type="ABSTAIN",
            reason="insufficient_evidence" if independent_errors else "budget_exhausted",
            supporting_evidence_ids=sorted(validated_evidence_ids),
        )
        state["diagnostic_action"] = action.model_dump(mode="json")
    if isinstance(action, FinishAction) and target_metric and found_factor == target_metric:
        action = AbstainAction(
            action_type="ABSTAIN",
            reason="insufficient_evidence",
            supporting_evidence_ids=[item for item in action.evidence_ids if item in known],
        )
        state["diagnostic_action"] = action.model_dump(mode="json")
        state.setdefault("diagnostic_guardrails", []).append({
            "code": "target_metric_cannot_be_root_cause",
            "target_metric": target_metric,
        })
    if isinstance(action, FinishAction):
        cited_unvalidated_ids = (set(action.evidence_ids) & known) - validated_evidence_ids
        if cited_unvalidated_ids:
            action = AbstainAction(
                action_type="ABSTAIN",
                reason="insufficient_evidence",
                supporting_evidence_ids=[
                    item for item in action.evidence_ids
                    if item in validated_evidence_ids
                ],
            )
            state["diagnostic_action"] = action.model_dump(mode="json")
            state.setdefault("diagnostic_guardrails", []).append({
                "code": "finish_cites_untrusted_or_unvalidated_evidence",
                "evidence_ids": sorted(cited_unvalidated_ids),
            })
            investigation_limitations.append(
                "最终结论引用了未被可信证据链确认的查询结果，系统已停止给出确定性归因。"
            )
    terminal_evidence_valid = (
        isinstance(action, FinishAction) and set(action.evidence_ids) <= known
    ) or (
        isinstance(action, AbstainAction) and set(action.supporting_evidence_ids) <= known
    )
    terminal_stop_reason = _record_terminal_investigation_reason(
        state, action, terminal_evidence_valid=terminal_evidence_valid,
    )
    if terminal_stop_reason is not None:
        if investigation_summary is None:
            investigation_raw = state.get("investigation_state") or {}
            investigation_summary = {
                "stop_reason": terminal_stop_reason,
                "progress": investigation_raw.get("progress", {}) if isinstance(investigation_raw, dict) else {},
                "calculation_progress": investigation_raw.get("calculation_progress", {}) if isinstance(investigation_raw, dict) else {},
                "remaining_budget": investigation_raw.get("remaining_budget") if isinstance(investigation_raw, dict) else None,
                "action_result_count": len(investigation_raw.get("action_results", [])) if isinstance(investigation_raw, dict) else 0,
                "action_results": [],
                "unresolved_branches": [],
            }
        else:
            investigation_summary["stop_reason"] = terminal_stop_reason
    if isinstance(action, FinishAction) and set(action.evidence_ids) <= known:
        numeric_decomposition = _metric_decomposition_projection(
            state,
            allow_partial_projection=False,
            allow_completed_projection=True,
        )
        if numeric_decomposition and not set(numeric_decomposition["evidence_ids"]) <= set(action.evidence_ids):
            numeric_decomposition = None
        state["finish_decision"] = action.model_dump(mode="json")
        state["analysis_result"] = {
            "summary": action.conclusion,
            "task_type": "metric_query" if registered_numeric_decomposition else "anomaly_analysis",
            "abstain": False,
            "evidence_ids": action.evidence_ids,
            "contribution": validated_dimension_contribution_cards(state),
            "limitations": common_limitations + ([action.residual_uncertainty] if action.residual_uncertainty else []),
            "causal_readiness": causal_readiness,
            "investigation_summary": investigation_summary,
            **({"metric_decomposition": numeric_decomposition} if numeric_decomposition else {}),
        }
        state["termination_initiator"] = "agent"
    elif isinstance(action, AbstainAction) and set(action.supporting_evidence_ids) <= known:
        state["finish_decision"] = action.model_dump(mode="json")
        pending_required: list[str] = []
        try:
            abstain_plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
            step_state = state.get("diagnostic_step_state", {})
            pending_required = [
                step.step_id for step in abstain_plan.steps
                if step.required and step_state.get(step.step_id, "pending") in {"pending", "data_insufficient"}
            ]
        except Exception:  # noqa: BLE001 - plan absent in unit-fixture states
            pass
        # Preserve a validated descriptive breakdown even when unresolved
        # branches prevent a full business-cause conclusion.
        partial_contributions: list[dict] = []
        for item in state.get("root_cause_evidence", []):
            evidence = item.get("investigation_evidence") or {}
            evidence_id = str(evidence.get("evidence_id") or item.get("query_id") or "")
            if evidence_id not in validated_evidence_ids:
                continue
            values = evidence.get("values") or {}
            if evidence.get("evidence_type") != "dimension_contribution" or values.get("attribution_status") != "ok":
                continue
            components = [component for component in (values.get("components") or []) if isinstance(component, dict)]
            if not components:
                continue
            dimension = str((values.get("query") or {}).get("dimension") or "")
            result = values.get("result") or {}
            comparison = (evidence.get("comparison") or {}).get("comparison") or {}
            current_period = result.get("current_period") or {}
            baseline_period = result.get("baseline_period") or {}
            ranked = sorted(components, key=lambda component: abs(float(component.get("delta_value") or 0)), reverse=True)
            partial_contributions.append({
                "dimension": dimension,
                "metric_name": str(evidence.get("metric_id") or ""),
                "current_date": (
                    result.get("current_date") or current_period.get("end_date")
                    or comparison.get("current_date") or comparison.get("current_end") or "unknown"
                ),
                "baseline_date": (
                    result.get("baseline_date") or baseline_period.get("end_date")
                    or comparison.get("baseline_date") or comparison.get("baseline_end") or "unknown"
                ),
                "total_delta": values.get("total_delta"),
                "items": [{
                    "dimension_value": component.get("dimension_value"),
                    "current_value": component.get("current_value"),
                    "baseline_value": component.get("baseline_value"),
                    "delta_value": component.get("delta_value"),
                    "contribution_ratio": component.get("contribution_ratio"),
                } for component in ranked],
                "evidence_ids": [evidence_id],
                "analysis_nature": "descriptive_contribution",
                "method": str(evidence.get("method_id") or "additive_change_v1"),
            })
        partial_summary = "Current evidence does not establish the business root cause; the underlying business mechanism remains unresolved."
        if partial_contributions:
            top_card = partial_contributions[0]
            top_item = max(top_card["items"], key=lambda value: abs(float(value.get("delta_value") or 0)))
            share = top_item.get("contribution_ratio")
            item_delta = top_item.get("delta_value")
            share_text = _english_contribution_share_text(share, item_delta)
            partial_summary = (
                f"Within the examined {top_card['dimension']} dimension, {top_item.get('dimension_value')} "
                f"has the largest measured contribution{share_text}. This descriptive attribution does not prove a business cause; the underlying mechanism remains unresolved."
            )
            if all(float(item.get("delta_value") or 0) == 0 for item in top_card["items"]):
                partial_summary = (
                    f"Within the examined {top_card['dimension']} dimension, all returned group changes are zero. "
                    "No nonzero contribution was observed; the underlying business mechanism remains unresolved."
                )
        observed_summary, observed_evidence_ids = _observed_values_summary_with_evidence(
            state, allowed_evidence_ids=validated_evidence_ids,
        )
        if observed_summary:
            partial_summary = f"{partial_summary} {observed_summary}"
        numeric_decomposition = _metric_decomposition_projection(
            state,
            allow_partial_projection=True,
        )
        if numeric_decomposition:
            if partial_contributions:
                partial_summary += (
                    " A separate registered numeric decomposition is also validated; "
                    "the business root cause remains unconfirmed."
                )
            else:
                partial_summary = (
                    "The registered metric decomposition is validated, but later investigation stopped at a recorded limitation; "
                    "the business root cause remains unconfirmed."
                )
        # Every number in the abstention summary/card must be referenced by
        # evidence that passed the final scope/action/provenance checks. The
        # model's proposed support list is not itself a validation result.
        safe_supporting_ids = [
            str(evidence_id)
            for evidence_id in action.supporting_evidence_ids
            if str(evidence_id) in validated_evidence_ids
        ]
        projected_evidence_ids = [
            str(evidence_id)
            for card in partial_contributions
            for evidence_id in (card.get("evidence_ids") or [])
        ]
        projected_evidence_ids.extend(observed_evidence_ids)
        if numeric_decomposition:
            projected_evidence_ids.extend(
                str(evidence_id)
                for evidence_id in (numeric_decomposition.get("evidence_ids") or [])
            )
        action = action.model_copy(update={
            "supporting_evidence_ids": list(dict.fromkeys([
                *safe_supporting_ids,
                *(
                    evidence_id for evidence_id in projected_evidence_ids
                    if evidence_id in validated_evidence_ids
                ),
            ])),
        })
        state["finish_decision"] = action.model_dump(mode="json")
        state["analysis_result"] = {
            "summary": partial_summary,
            "task_type": "metric_query" if registered_numeric_decomposition else "anomaly_analysis", "abstain": True,
            "evidence_ids": action.supporting_evidence_ids,
            "contribution": partial_contributions,
            "stop_reason": action.reason,
            "limitations": common_limitations,
            "causal_readiness": causal_readiness,
            "investigation_summary": investigation_summary,
            # Budget-exhausted abstains must name what was never verified.
            "pending_required_steps": pending_required,
            **({"metric_decomposition": numeric_decomposition} if numeric_decomposition else {}),
        }
        state["termination_initiator"] = "agent"
    else:
        state.setdefault("errors", []).append({"code": "diagnostic_evidence_reference_invalid", "message": "Terminal action references unknown evidence.", "agent_name": "diagnostic_finalize"})
        state["analysis_result"] = {
            "summary": "证据引用无效或未通过可信校验，无法安全完成归因。",
            "task_type": "metric_query" if registered_numeric_decomposition else "anomaly_analysis", "abstain": True,
            "stop_reason": "policy", "evidence_ids": [],
            "limitations": common_limitations,
            "causal_readiness": causal_readiness,
            "investigation_summary": investigation_summary,
        }
        state["termination_initiator"] = "policy"
    state["pending_query"] = None
    state["next_agent"] = "generate_answer"
    state["current_agent"] = "diagnostic_finalize"
    analysis = state.get("analysis_result", {}) or {}
    _record_agent_run(state, "abstained" if analysis.get("abstain") else ("failed" if state.get("errors") else "finished"))
    investigation_count = int(state.get("diagnostic_investigation_action_count", 0))
    if investigation_count > 0:
        from app.core.execution_context import enqueue_sse_event
        from app.schemas.events import SseInvestigationProgressEvent

        # Keep the terminal SSE event aligned with the authoritative investigation
        # ledger. ``analysis.stop_reason`` is a legacy, user-facing free-text value
        # (for example ``budget_exhausted``), while the ledger stores the stable
        # machine-readable terminal code used by Trace and recovery.
        summary = analysis.get("investigation_summary") or {}
        structured_stop = summary.get("stop_reason") if isinstance(summary, dict) else None
        stop_code = structured_stop.get("code") if isinstance(structured_stop, dict) else None
        stop_reason = str(stop_code or analysis.get("stop_reason") or "completed")
        enqueue_sse_event(SseInvestigationProgressEvent(
            completed_actions=min(6, investigation_count),
            remaining_budget=max(0, 6 - investigation_count),
            status="abstained" if analysis.get("abstain") else "sufficient",
            stop_reason=stop_reason,
        ))
    state["node_history"].append({"node": "diagnostic_finalize", "summary": "Diagnostic loop finalized."})
    return state
