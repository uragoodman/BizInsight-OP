from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Any
from time import perf_counter

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.agents.data_query_agent import (
    build_metric_query_plan,
)
from app.mcp.schemas import DataQueryAgentResult, McpToolInvocation, MetricQueryPlan
from app.core.failure_context import ExecutionFailureContext
from app.metrics.authorization import validate_authorized_metric
from app.schemas.explorer import ControlledAnalysisScope
from app.data.repositories.explorer import breakdown_scoped_month_position, breakdown_scoped_payment_type, breakdown_scoped_product_category, breakdown_scoped_time_dimension, compare_scoped_metric, compare_scoped_order_count_allocation, query_scoped_metric, query_scoped_statistics, query_scoped_trend
from app.agents.data_query_agent import apply_trusted_time_range
from app.agents.data_query_agent import _extract_dimension
from app.core.time_expression import resolve_expression
from app.mcp.server import SERVER_NAME
from app.mcp.tools import get_supported_tools
from app.tenancy import current_tenant_context
from app.data.repositories.tenant_metrics import SELLER_METRICS, query_seller_metric, query_seller_trend
from app.core.config import get_settings
from app.core.execution_context import active_state
from app.resilience import ExternalServiceError, execute_with_resilience
from app.security import ensure_tool_allowed
from app.mcp.remote_client import call_remote_mcp_tool
from uuid import uuid4


# Keep contribution queries complete up to the tool's public 100-group cap.
# The previous top-10 default silently folded valid groups into ``Other``.
DEFAULT_CONTRIBUTION_GROUP_LIMIT = 100

# Authorization evidence is scoped to one synchronous gateway invocation.
# ContextVar keeps concurrent requests isolated and avoids a module-global
# "current request" slot.  The public wrapper below attaches the captured
# decision to the exact response returned for this call.
_ACTIVE_AUTHORIZATION_CAPTURE: ContextVar[dict[str, Any] | None] = ContextVar(
    "mcp_authorization_capture", default=None,
)


def _capture_authorization_snapshot(
    *, request_id: str, plan: MetricQueryPlan | None, decision: str,
    reason_code: str, authorization_checked: bool, authz: Any = None,
    boundary: Any = None, scope_data: dict[str, Any] | None = None,
) -> None:
    capture = _ACTIVE_AUTHORIZATION_CAPTURE.get()
    if capture is None:
        return
    authorized = plan.authorized_metric.model_dump(mode="json") if plan and plan.authorized_metric else None
    identity = {
        "tenant_id": (scope_data or {}).get("tenant_id"),
        "user_id": (scope_data or {}).get("user_id"),
        "role": (scope_data or {}).get("permissions"),
        "scope_type": (scope_data or {}).get("scope_type"),
    }
    identity_sha256 = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest() if any(identity.values()) else None
    plan_fields = {}
    if plan is not None:
        plan_fields = {
            key: getattr(plan, key, None)
            for key in (
                "metric_date", "start_date", "end_date", "baseline_date", "current_date",
                "baseline_start", "baseline_end", "current_start", "current_end",
                "granularity", "dimension", "dimensions",
            )
            if getattr(plan, key, None) is not None
            and (key != "dimensions" or getattr(plan, key, None))
        }
    authorization_fingerprint = getattr(authz, "authz_digest", None)
    permission_version = getattr(authz, "permission_version", None)
    policy_set_version = getattr(authz, "policy_set_version", None)
    query_scope_fingerprint = (
        (scope_data or {}).get("query_scope_fingerprint")
        or (authorized or {}).get("query_scope_fingerprint")
        or (authorized or {}).get("scope_fingerprint")
    )
    # The runtime snapshot records the server's current authorization result,
    # while the original AuthorizedMetric retains provenance and catalog scope.
    # Do not mutate the plan object or invent an authorization grant.
    if authorized is not None:
        authorized = {
            **authorized,
            "authorization_fingerprint": authorization_fingerprint or authorized.get("authorization_fingerprint"),
            "permission_version": permission_version,
            "policy_set_version": policy_set_version,
        }
    capture.clear()
    capture.update({
        "schema_version": "per-tool-authorization-snapshot-v1",
        "request_id": request_id,
        "decision": decision,
        "decision_source": "query_business_data.pre_execution_authorization",
        "reason_code": reason_code,
        "authorization_checked": authorization_checked,
        "metric_id": plan.metric_name if plan else None,
        "authorized_metric_id": (authorized or {}).get("metric_id"),
        "authorized_metric": authorized,
        "scope": (scope_data or {}).get("scope_type") or "platform",
        "operation": plan.query_type if plan else None,
        "dimension": plan.dimension if plan else None,
        "dimensions": list(plan.dimensions) if plan and plan.dimensions else None,
        "period": plan_fields,
        "authorization_fingerprint": authorization_fingerprint,
        "permission_version": permission_version,
        "policy_set_version": policy_set_version,
        "authorization_boundary_scope_sha256": (
            hashlib.sha256(str(getattr(boundary, "query_scope_fingerprint", "")).encode("utf-8")).hexdigest()
            if getattr(boundary, "query_scope_fingerprint", None) else None
        ),
        "authorization_boundary_match": (
            bool(
                getattr(boundary, "permission_version", None) == permission_version
                and getattr(boundary, "policy_set_version", None) == policy_set_version
                and getattr(boundary, "authorization_fingerprint", None) == authorization_fingerprint
                and getattr(boundary, "query_scope_fingerprint", None) == (scope_data or {}).get("query_scope_fingerprint")
            ) if boundary is not None else None
        ),
        "authorization_boundary": {
            "provided": boundary is not None,
            "permission_version": getattr(boundary, "permission_version", None) if boundary is not None else permission_version,
            "policy_set_version": getattr(boundary, "policy_set_version", None) if boundary is not None else policy_set_version,
            "authorization_fingerprint": (
                getattr(boundary, "authorization_fingerprint", None)
                if boundary is not None else authorization_fingerprint
            ),
            "query_scope_fingerprint": (
                getattr(boundary, "query_scope_fingerprint", None)
                if boundary is not None else query_scope_fingerprint
            ),
        },
        "query_scope_fingerprint": query_scope_fingerprint,
        "identity_sha256": identity_sha256,
    })


def _server_workdir() -> str:
    """
    返回启动 MCP Server 子进程时使用的项目根目录。

    stdio client 会拉起一个独立的 Python 进程，如果不显式指定 cwd，
    在不同入口下运行时可能会出现导入路径不一致的问题。
    """

    return str(Path(__file__).resolve().parents[2])


def _server_parameters() -> StdioServerParameters:
    """构造本地 MCP Server 的 stdio 启动参数。"""

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        env=os.environ.copy(),
        cwd=_server_workdir(),
    )


def _extract_result_payload(raw_result: Any) -> dict[str, Any]:
    """
    把 MCP 工具返回值整理成普通字典。

    FastMCP 会优先把结构化结果放在 `structuredContent`。如果没有结构化结果，
    这里再退回去解析文本内容。这样下游 workflow 总能拿到统一的 dict。
    """

    structured = getattr(raw_result, "structuredContent", None)
    if structured is not None:
        return dict(structured)

    content = getattr(raw_result, "content", None) or []
    if content:
        first_block = content[0]
        text = getattr(first_block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw_text": text}
    return {}


async def _call_local_mcp_tool(tool_name: str, arguments: dict[str, object]) -> dict[str, Any]:
    """
    通过真实的 MCP stdio client 调用本地工具。

    这一步是真正把 workflow 和 MCP server 接起来的关键：
    - workflow 并不是直接 import 某个 tool 函数
    - 而是走 MCP 协议去调用
    """

    server = _server_parameters()
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.list_tools()
            result = await session.call_tool(tool_name, arguments)
            return _extract_result_payload(result)


def _safe_exception_chain(exc: BaseException) -> list[dict[str, str]]:
    """Describe exception classes and public dependency codes without messages."""
    rows: list[dict[str, str]] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(rows) < 6:
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
        rows.append(item)
        current = current.__cause__
    return rows


def classify_query_failure(
    exc: BaseException,
    *,
    request_id: str | None,
    default_service: str = "metric_registry",
) -> ExecutionFailureContext:
    """Classify an execution failure without persisting exception text."""
    from app.metrics.exceptions import (
        CapabilityNotSupported, DimensionNotAllowed, MetricNotFound, ScopeNotAllowed,
    )

    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    for item in chain:
        if isinstance(item, ScopeNotAllowed):
            return ExecutionFailureContext(
                category="authorization_denied", service="metric_registry",
                code="scope_not_allowed", cause_type=type(item).__name__,
                request_id=request_id, retryable=False,
            )
        if isinstance(item, PermissionError):
            return ExecutionFailureContext(
                category="authorization_denied",
                service=default_service if default_service in {
                    "metric_registry", "mcp", "remote_mcp",
                } else "unknown",
                code="permission_denied", cause_type=type(item).__name__,
                request_id=request_id, retryable=False,
            )
        if isinstance(item, (CapabilityNotSupported, DimensionNotAllowed, MetricNotFound)):
            code = (
                "dimension_not_allowed" if isinstance(item, DimensionNotAllowed)
                else "metric_not_found" if isinstance(item, MetricNotFound)
                else "capability_not_supported"
            )
            return ExecutionFailureContext(
                category="capability_gap", service="metric_registry",
                code=code, cause_type=type(item).__name__,
                request_id=request_id, retryable=False,
            )

    for item in chain:
        if isinstance(item, ExternalServiceError):
            service = item.service if item.service in {"mcp", "postgresql"} else default_service
            if service == "postgresql":
                category = "database_error"
                code = "database_timeout" if item.code == "timeout" else "database_execution_failed"
            elif item.code == "timeout":
                category, code = "tool_timeout", "tool_timeout"
            else:
                category, code = "execution_error", item.code
            return ExecutionFailureContext(
                category=category, service=service, code=code,
                cause_type=type(item.__cause__ or item).__name__,
                request_id=request_id, retryable=item.retryable,
            )

    try:
        from sqlalchemy.exc import SQLAlchemyError
    except ImportError:  # pragma: no cover - SQLAlchemy is a runtime dependency
        SQLAlchemyError = ()  # type: ignore[assignment,misc]
    for item in chain:
        if SQLAlchemyError and isinstance(item, SQLAlchemyError):
            is_timeout = "timeout" in type(item).__name__.casefold()
            return ExecutionFailureContext(
                category="database_error", service="postgresql",
                code="database_timeout" if is_timeout else "database_execution_failed",
                cause_type=type(item).__name__, request_id=request_id, retryable=None,
            )
    for item in chain:
        if isinstance(item, (TimeoutError, asyncio.TimeoutError)):
            service = default_service if default_service in {"metric_registry", "remote_mcp"} else "mcp"
            return ExecutionFailureContext(
                category="tool_timeout", service=service, code="tool_timeout",
                cause_type=type(item).__name__, request_id=request_id, retryable=None,
            )
    return ExecutionFailureContext(
        category="execution_error",
        service=default_service if default_service in {
            "metric_registry", "mcp", "postgresql", "remote_mcp",
        } else "unknown",
        code="execution_failed", cause_type=type(exc).__name__,
        request_id=request_id, retryable=None,
    )


def call_local_mcp_tool(
    tool_name: str,
    arguments: dict[str, object],
    *,
    tenant_id: str | None = None,
    role: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """给当前同步 workflow 使用的同步包装函数。"""
    context = current_tenant_context()
    ensure_tool_allowed(
        "data_query",
        tool_name,
        tenant_id=tenant_id,
        role=role,
        scope_type=context.scope_type if context else None,
    )

    async def bounded_call() -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                _call_local_mcp_tool(tool_name, arguments),
                timeout=get_settings().mcp_timeout_seconds,
            )
        except TimeoutError as exc:
            raise ExternalServiceError("mcp", "timeout", "MCP call timed out.") from exc

    def run_bounded_call() -> dict[str, Any]:
        # The public workflow is synchronous, but it may be hosted by a caller
        # that already owns an event loop. asyncio.run() on that same thread
        # fails before scheduling the coroutine, leaving it unawaited. Bridge
        # only this sync/async boundary through a worker and copy request
        # ContextVars; create the coroutine inside the worker so failed
        # submission cannot orphan it.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(bounded_call())
        request_context = copy_context()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-sync-bridge") as executor:
            return executor.submit(
                request_context.run,
                lambda: asyncio.run(bounded_call()),
            ).result()

    state = active_state() or {}
    call_config = state.get("llm_call_config") or {}
    experiment_scoped = call_config.get("mode") == "provider_ab_experiment"
    # The paired experiment explicitly disables automatic tool retries. Normal
    # business requests retain execute_with_resilience's established default.
    max_attempts = 1 if experiment_scoped else 2
    attempt_records = state.setdefault("mcp_call_attempts", []) if experiment_scoped else None
    attempt_number = 0

    def observed_call() -> dict[str, Any]:
        nonlocal attempt_number
        attempt_number += 1
        started = perf_counter()
        try:
            result = run_bounded_call()
        except Exception as exc:
            if attempt_records is not None:
                attempt_records.append({
                    "request_id": request_id,
                    "tool_name": tool_name,
                    "attempt_number": attempt_number,
                    "attempt_limit": max_attempts,
                    "outcome": "exception",
                    "exception_chain": _safe_exception_chain(exc),
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                })
            try:
                setattr(exc, "mcp_attempt_count", attempt_number)
            except Exception:
                pass
            raise
        if attempt_records is not None:
            attempt_records.append({
                "request_id": request_id,
                "tool_name": tool_name,
                "attempt_number": attempt_number,
                "attempt_limit": max_attempts,
                "outcome": "response_received",
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            })
        return result

    try:
        return execute_with_resilience(observed_call, "mcp", attempts=max_attempts)
    except Exception as exc:
        try:
            setattr(exc, "mcp_attempt_count", attempt_number)
        except Exception:
            pass
        raise


def _build_arguments_from_plan(plan: Any) -> dict[str, object]:
    """
    根据查询计划构造 MCP 工具入参。

    这里之所以单独抽函数，是因为不同查询类型的参数结构不同：
    - 单点查询：只要一个日期
    - 趋势查询：要指标名 + 时间范围
    - 对比查询：要指标名 + 左右两个日期
    - 拆解查询：要指标名 + 日期 + 维度 + 限制条数
    """

    if plan.query_type == "metric_breakdown" and plan.dimensions:
        return {
            "metric_name": plan.metric_name,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "dimensions": list(plan.dimensions),
            "limit": plan.limit or 100,
        }
    if plan.query_type == "metric_value":
        return {"metric_date": plan.metric_date}
    if plan.query_type == "metric_trend":
        return {
            "metric_name": plan.metric_name,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "granularity": plan.granularity or "day",
        }
    if plan.query_type == "metric_period":
        return {
            "metric_name": plan.metric_name,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "granularity": "month",
        }
    if plan.query_type == "metric_statistics":
        return {
            "metric_name": plan.metric_name,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "granularity": plan.granularity or "day",
            "statistics": list(plan.statistics),
        }
    if plan.query_type == "metric_entity_statistics":
        return {
            "metric_name": plan.metric_name,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
        }
    if plan.query_type == "metric_association":
        return {
            "metric_name": plan.metric_name,
            "association_pair_id": plan.association_pair_id,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
        }
    if plan.query_type == "metric_compare":
        if (
            plan.current_start and plan.current_end
            and plan.baseline_start and plan.baseline_end
        ):
            return {
                "metric_name": plan.metric_name,
                "current_start": plan.current_start,
                "current_end": plan.current_end,
                "baseline_start": plan.baseline_start,
                "baseline_end": plan.baseline_end,
                "granularity": plan.granularity or "month",
            }
        # CompareMetricArguments is a legacy wire contract. Internally the
        # old repository contract treats left as baseline and right as current.
        return {
            "metric_name": plan.metric_name,
            "left_date": plan.baseline_date or plan.left_date,
            "right_date": plan.current_date or plan.right_date,
        }
    if plan.query_type == "metric_contribution":
        raise ValueError("Metric contributions require a validated dedicated dispatch route.")
    return {
        "metric_name": plan.metric_name,
        "metric_date": plan.metric_date,
        "dimension": plan.dimension,
        "limit": plan.limit or 10,
    }


def _platform_contribution_route(
    plan: MetricQueryPlan,
) -> tuple[str | None, dict[str, object] | None, str | None]:
    """Validate and normalize a platform contribution plan to its MCP tool.

    Contributions are comparison operations with their own point-pair and
    period contracts. They must never fall through to the Registry's
    single-point ``breakdown`` adapter.
    """
    if not plan.dimension:
        return None, None, "Metric contribution requires an explicit dimension."
    effective_limit = (
        plan.limit if plan.limit is not None else DEFAULT_CONTRIBUTION_GROUP_LIMIT
    )

    # ``left_date`` / ``right_date`` are the comparison contract fields. The
    # current contribution contract is explicit about current/baseline dates;
    # do not silently assign new meaning to compare-only fields here.
    if plan.left_date is not None or plan.right_date is not None:
        return None, None, "Contribution plans must use current/baseline date fields, not left/right comparison fields."

    if any(value is not None for value in (plan.metric_date, plan.start_date, plan.end_date)):
        return None, None, "Contribution plan mixes point/range query fields with contribution dates."

    point_values = (plan.current_date, plan.baseline_date)
    period_values = (
        plan.current_start, plan.current_end,
        plan.baseline_start, plan.baseline_end,
    )
    has_point = any(value is not None for value in point_values)
    has_period = any(value is not None for value in period_values)

    if has_point and has_period:
        return None, None, "Contribution plan mixes point dates and period ranges."
    if has_point:
        if not all(point_values):
            return None, None, "Point contribution requires both current_date and baseline_date."
        if plan.tool_name not in (None, "compare_breakdown_by_dimension"):
            return None, None, "Point contribution tool does not match its date shape."
        tool_name = "compare_breakdown_by_dimension"
        try:
            date.fromisoformat(str(plan.current_date))
            date.fromisoformat(str(plan.baseline_date))
        except ValueError:
            return None, None, "Point contribution dates must be valid ISO calendar dates."
        arguments: dict[str, object] = {
            "metric_name": plan.metric_name,
            "current_date": plan.current_date,
            "baseline_date": plan.baseline_date,
            "dimension": plan.dimension,
            "limit": effective_limit,
        }
    elif has_period:
        if not all(period_values):
            return None, None, "Period contribution requires complete current and baseline ranges."
        if plan.granularity not in {"week", "month"}:
            return None, None, "Period contribution requires an explicit week or month granularity."
        allowed_period_tools = {
            None,
            "compare_breakdown_by_dimension",
            "compare_breakdown_by_dimension_period",
            "compare_freight_rate_ratio_facts_by_dimension_period",
            "compare_avg_order_value_ratio_facts_by_dimension_period",
        }
        if plan.tool_name not in allowed_period_tools:
            return None, None, "Period contribution tool does not match its date shape."
        try:
            parsed = [date.fromisoformat(str(value)) for value in period_values]
        except ValueError:
            return None, None, "Period contribution dates must be valid ISO calendar dates."
        current_start, current_end, baseline_start, baseline_end = parsed
        if current_start > current_end or baseline_start > baseline_end:
            return None, None, "Period contribution range start must not follow its end."
        selected_period_tool = plan.tool_name
        if selected_period_tool in {None, "compare_breakdown_by_dimension_period"}:
            # Resolve registered ratio period contributions from the metric's
            # declared numerator/denominator contract. The generic breakdown
            # handler cannot reproduce a weighted ratio contribution.
            try:
                from app.metrics.registry import get_registry

                definition = get_registry().get(str(plan.metric_name))
                attribution = definition.attribution.for_scope("platform")
                aggregation = definition.period_aggregation
                if (
                    plan.dimension == "month_position"
                    and "symmetric_mix_within_v1" in attribution.method_ids
                    and "month_position" in attribution.dimensions
                    and aggregation is not None
                ):
                    ratio_contract = (aggregation.numerator_metric, aggregation.denominator_metric)
                    selected_period_tool = {
                        ("freight_value", "gmv"): "compare_freight_rate_ratio_facts_by_dimension_period",
                        ("gmv", "order_count"): "compare_avg_order_value_ratio_facts_by_dimension_period",
                    }.get(ratio_contract, selected_period_tool)
            except Exception:
                # The common capability validation below will refuse the
                # request if Registry metadata is incomplete or unavailable.
                pass
        if selected_period_tool in {
            "compare_freight_rate_ratio_facts_by_dimension_period",
            "compare_avg_order_value_ratio_facts_by_dimension_period",
        }:
            tool_name = selected_period_tool
        else:
            # Legacy planners may mark a contribution with the point tool
            # name before the period shape is known. The date shape, not that
            # hint, selects the period contract; never send range fields to a
            # one-day contribution tool.
            tool_name = "compare_breakdown_by_dimension_period"
        arguments = {
            "metric_name": plan.metric_name,
            "current_start": plan.current_start,
            "current_end": plan.current_end,
            "baseline_start": plan.baseline_start,
            "baseline_end": plan.baseline_end,
            "granularity": plan.granularity,
            "dimension": plan.dimension,
            "limit": effective_limit,
        }
    else:
        return None, None, "Contribution requires either a complete point pair or two complete period ranges."

    limit = effective_limit
    if not 1 <= limit <= 100:
        return None, None, "Contribution limit must be between 1 and 100."

    try:
        from app.metrics.registry import get_registry

        registry = get_registry()
        implementation = registry.get_implementation(str(plan.metric_name), "platform")
        required = {"breakdown", "compare"}
        if not required.issubset(implementation.capabilities):
            return None, None, "Metric does not register both breakdown and compare capabilities required for contribution."
        dimension_allowed = registry.is_dimension_allowed(
            str(plan.metric_name), plan.dimension, "platform",
        )
        # The two registered ratio-facts tools use their own attribution
        # contract for month_position; that dimension is intentionally not a
        # generic breakdown dimension for these metrics. The dedicated branch
        # below revalidates the metric, dimension, and method before dispatch.
        ratio_facts_dimension = bool(
            has_period
            and plan.dimension == "month_position"
            and tool_name in {
                "compare_freight_rate_ratio_facts_by_dimension_period",
                "compare_avg_order_value_ratio_facts_by_dimension_period",
            }
        )
        if not dimension_allowed and not ratio_facts_dimension:
            return None, None, "Contribution dimension is not registered for this metric and scope."
    except Exception as exc:
        return None, None, f"Contribution capability validation failed: {type(exc).__name__}."

    return tool_name, arguments, None


def _adapt_legacy_compare_result(plan: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the old compare result (left=baseline, right=current).

    The public workflow contract remains current/baseline-oriented, including
    its legacy ``left_*`` aliases. The old repository's delta and ratio are
    already calculated as right-minus-left and relative to left, so the
    adapter swaps operands and dates while preserving those calculations.
    """
    baseline_date = plan.baseline_date or payload.get("left_date") or plan.left_date
    current_date = plan.current_date or payload.get("right_date") or plan.right_date
    baseline_value = payload.get("left_value")
    current_value = payload.get("right_value")
    return {
        **payload,
        "current_date": current_date,
        "baseline_date": baseline_date,
        "current_value": current_value,
        "baseline_value": baseline_value,
        "left_date": current_date,
        "right_date": baseline_date,
        "left_value": current_value,
        "right_value": baseline_value,
    }


def _govern_multi_dimension_result(
    payload: dict[str, Any], *, authorization_context: Any,
) -> dict[str, Any]:
    """Apply the standard result policy using distinct customers as cell size.

    The multidimensional repository returns aggregates plus distinct customer
    counts. Tag each cell with that count for the shared k-threshold policy,
    then remove suppressed cells entirely so neither values nor small-cell
    counts reach the answer model.
    """

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return payload
    rows = payload.get("items")
    if not isinstance(rows, list):
        return payload

    tagged_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_size = row.get("customer_count")
        if (
            not isinstance(sample_size, (int, float))
            or isinstance(sample_size, bool)
            or sample_size < 0
        ):
            sample_size = 0
        tagged_rows.append({**row, "sample_size": sample_size})

    from app.security.result_policy import apply_result_policy

    governed = apply_result_policy(
        {
            **payload,
            "items": tagged_rows,
            "dimension_count": len(payload.get("dimensions") or []),
            "classification": payload.get("classification") or "internal",
        },
        authorization_context=authorization_context,
    ).payload
    safe_rows = governed.get("items") if isinstance(governed, dict) else None
    safe_rows = safe_rows if isinstance(safe_rows, list) else []
    visible: list[dict[str, Any]] = []
    suppressed_count = int(payload.get("suppressed_cell_count") or 0)
    for row in safe_rows:
        if not isinstance(row, dict):
            continue
        if row.get("suppressed") is True:
            suppressed_count += 1
            continue
        visible.append({key: value for key, value in row.items() if key != "sample_size"})
    governed["items"] = visible
    governed["suppressed_cell_count"] = suppressed_count
    return governed


def _result_from_payload(
    plan: Any,
    payload: dict[str, Any],
    *,
    legacy_compare_result: bool = False,
) -> DataQueryAgentResult:
    """
    把不同 tool 的返回值装回统一的 DataQueryAgentResult。

    这样 workflow 下游虽然能感知“这是趋势/对比/拆解”，但整体仍然消费一个稳定
    的结果对象，而不是为每个工具写一套完全独立的处理分支。
    """

    kwargs: dict[str, Any] = {
        "status": payload["status"],
        "parsed": plan,
    }
    if plan.query_type in {"metric_value", "metric_period"}:
        kwargs["metric_result"] = payload
    elif plan.query_type == "metric_trend":
        kwargs["trend_result"] = payload
    elif plan.query_type == "metric_compare":
        if legacy_compare_result:
            payload = _adapt_legacy_compare_result(plan, payload)
        current_date = plan.current_date or payload.get("current_date") or payload.get("left_date")
        baseline_date = plan.baseline_date or payload.get("baseline_date") or payload.get("right_date")
        current_value = payload.get("current_value", payload.get("left_value"))
        baseline_value = payload.get("baseline_value", payload.get("right_value"))
        kwargs["compare_result"] = {
            **payload,
            "current_date": current_date,
            "baseline_date": baseline_date,
            "current_value": current_value,
            "baseline_value": baseline_value,
        }
    elif plan.query_type == "metric_contribution":
        kwargs["contribution_result"] = payload
    elif plan.query_type in {"metric_statistics", "metric_entity_statistics"}:
        kwargs["statistics_result"] = payload
    elif plan.query_type == "metric_association":
        kwargs["association_result"] = payload
    elif plan.dimensions:
        kwargs["multi_dimension_breakdown_result"] = payload
    else:
        kwargs["breakdown_result"] = payload
    return DataQueryAgentResult(**kwargs)


def _build_analysis_context_note(
    default_start: date, default_end: date,
    periods: list[dict[str, object]],
) -> str:
    """比较用户在聊天中指定的分析时间与面板默认范围，有差异时生成说明。"""
    for p in periods:
        name = p.get("name", "")
        start = p.get("start_date")
        end = p.get("end_date")
        if start and end:
            if start < default_start or end > default_end:
                period_label = {
                    "current": "对比当前期", "baseline": "对比基线期",
                    "trend": "趋势查询", "single": "本次查询",
                }.get(name, "本次查询")
                return (
                    f"当前分析面板默认范围为 {default_start.isoformat()} 至 {default_end.isoformat()}；"
                    f"{period_label} {start.isoformat()} 至 {end.isoformat()} 超出该范围，"
                    f"已按聊天内容查询。"
                )
    return ""


_MONTH_REFERENCE_PATTERN = re.compile(r"(?<!\d)(1[0-2]|[1-9])\s*月")


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _previous_month_bounds(value: date) -> tuple[date, date]:
    if value.month == 1:
        return _month_bounds(value.year - 1, 12)
    return _month_bounds(value.year, value.month - 1)


def _resolve_scoped_compare_periods(
    scope: ControlledAnalysisScope,
    question: str,
    plan: MetricQueryPlan,
) -> tuple[date, date, date, date] | None:
    """Resolve a comparison only from dates already contained in Scope.

    Relative month words are deliberately anchored to the selected explorer
    range, not to the computer clock or the entire Olist dataset.  For
    ``10月与上月`` inside a Sep-Oct scope this returns Oct versus Sep.
    """
    range_fields = (plan.current_start, plan.current_end, plan.baseline_start, plan.baseline_end)
    if any(range_fields):
        if not all(range_fields):
            return None
        try:
            current_start = date.fromisoformat(plan.current_start)
            current_end = date.fromisoformat(plan.current_end)
            baseline_start = date.fromisoformat(plan.baseline_start)
            baseline_end = date.fromisoformat(plan.baseline_end)
        except (TypeError, ValueError):
            return None
        if current_end < current_start or baseline_end < baseline_start or baseline_end >= current_start:
            return None
        if (
            current_start < scope.default_start_date
            or current_end > scope.default_end_date
            or baseline_start < scope.default_start_date
            or baseline_end > scope.default_end_date
        ):
            return None
        return current_start, current_end, baseline_start, baseline_end
    if plan.current_date and plan.baseline_date:
        try:
            current = date.fromisoformat(plan.current_date)
            baseline = date.fromisoformat(plan.baseline_date)
        except ValueError:
            return None
        if current < scope.default_start_date or current > scope.default_end_date:
            return None
        if baseline < scope.default_start_date or baseline > scope.default_end_date:
            # Baseline may be outside a one-day current scope but must still
            # remain inside the server-locked date interval.
            return None
        return current, current, baseline, baseline
    # Backward compatibility for callers that still provide the old compare
    # fields: the repository wire contract is left=baseline, right=current.
    if plan.left_date and plan.right_date:
        try:
            baseline = date.fromisoformat(plan.left_date)
            current = date.fromisoformat(plan.right_date)
        except ValueError:
            return None
        return current, current, baseline, baseline

    requested_months = [int(value) for value in _MONTH_REFERENCE_PATTERN.findall(question)]
    current_start: date
    current_end: date
    if requested_months:
        requested_month = requested_months[0]
        candidates = []
        for year in range(scope.default_start_date.year, scope.default_end_date.year + 1):
            start, end = _month_bounds(year, requested_month)
            candidates.append((start, end))
        if not candidates:
            return None
        current_start, current_end = candidates[-1]
    else:
        current_start, current_end = _month_bounds(scope.default_end_date.year, scope.default_end_date.month)

    baseline_start, baseline_end = _previous_month_bounds(current_start)
    return current_start, current_end, baseline_start, baseline_end


# === Period 解析与 Policy 验证 ===

# ResolvedPeriod:
#   name: "current" / "baseline" / "trend" / "single"
#   start_date / end_date: date or None
#   points: int or None (for trend)


def _resolve_periods(
    plan: MetricQueryPlan,
    scope: ControlledAnalysisScope,
    question: str,
) -> list[dict[str, object]]:
    """将查询计划统一解析为 Period 列表，供 policy 验证 + note 生成共用。"""
    if plan.query_type in {"metric_trend", "metric_statistics", "metric_entity_statistics", "metric_association"}:
        start = plan.start_date or scope.default_start_date
        end = plan.end_date or scope.default_end_date
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)
        return [{
            "name": "trend", "start_date": start, "end_date": end,
            "points": None,
        }]
    if plan.query_type == "metric_compare":
        periods_tuple = _resolve_scoped_compare_periods(scope, question, plan)
        if periods_tuple is None:
            return []
        current_start, current_end, baseline_start, baseline_end = periods_tuple
        return [
            {"name": "current", "start_date": current_start, "end_date": current_end},
            {"name": "baseline", "start_date": baseline_start, "end_date": baseline_end},
        ]
    if plan.query_type == "metric_contribution":
        periods_tuple = _resolve_scoped_compare_periods(scope, question, plan)
        if periods_tuple is None:
            return []
        current_start, current_end, baseline_start, baseline_end = periods_tuple
        return [
            {"name": "current", "start_date": current_start, "end_date": current_end},
            {"name": "baseline", "start_date": baseline_start, "end_date": baseline_end},
        ]
    # metric_value / metric_breakdown / other: single period from plan or default
    if plan.query_type == "metric_breakdown":
        return [{"name": "single", "start_date": scope.default_start_date, "end_date": scope.default_end_date}]
    if plan.metric_date:
        d = date.fromisoformat(plan.metric_date)
        return [{"name": "single", "start_date": d, "end_date": d}]
    return [{"name": "single", "start_date": scope.default_start_date, "end_date": scope.default_end_date}]


def validate_periods(
    periods: list[dict[str, object]],
    scope: ControlledAnalysisScope,
) -> str | None:
    """返回 None 表示通过，返回字符串表示拒绝原因。"""
    for p in periods:
        start = p.get("start_date")
        end = p.get("end_date")
        if start and end:
            days = (end - start).days
            if days > scope.max_date_span_days:
                return f"查询时间范围 {days} 天超过允许上限 {scope.max_date_span_days} 天。"
        points = p.get("points")
        if points is not None and points > scope.max_trend_points:
            return f"数据点 {points} 个超过允许上限 {scope.max_trend_points} 个。"
    return None


def _query_scoped_authentication_denial(
    question: str,
    scope_payload: dict[str, object],
    *,
    request_id: str | None,
    plan: MetricQueryPlan | None,
    reason: str,
) -> dict[str, Any]:
    try:
        scope_data = ControlledAnalysisScope.model_validate(scope_payload).model_dump(mode="json")
    except Exception:
        scope_data = {}
    return {
        "task_type": "metric_query", "question": question,
        "server": "scoped-mcp", "tool_name": None,
        "arguments": {"scope": scope_data},
        "request_id": request_id or str(uuid4()),
        "result": DataQueryAgentResult(
            status="unsupported", parsed=plan or MetricQueryPlan(), reason=reason,
        ).model_dump(),
    }


def query_scoped_business_data(
    question: str,
    scope_payload: dict[str, object],
    *,
    request_id: str | None = None,
    plan: MetricQueryPlan | None = None,
    manifest_id: str | None = None,
    authorization_boundary=None,
    current_query_scope_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Refresh a session-backed identity, then enter the scoped MCP handler.

    The API's initial authentication snapshot can become stale while the
    graph is planning. Session- and API-key-backed identities are reloaded
    immediately before MCP authorization and data access, so a role/grant/
    credential revoke between HTTP ingress and this boundary cannot rely on
    that snapshot.
    """
    from app.tenancy import activate_tenant_context, current_tenant_context

    context = current_tenant_context()
    if context is None or not (context.auth_session_id or context.auth_credential_id):
        return _query_scoped_business_data_with_context(
            question, scope_payload, request_id=request_id, plan=plan,
            manifest_id=manifest_id, authorization_boundary=authorization_boundary,
            current_query_scope_fingerprint=current_query_scope_fingerprint,
        )

    try:
        from app.iam.service import authenticate_api_credential, authenticate_session

        if context.auth_session_id:
            refreshed = authenticate_session(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                session_id=context.auth_session_id,
            )
        else:
            refreshed = authenticate_api_credential(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                credential_id=str(context.auth_credential_id),
            )
        if refreshed is None:
            raise PermissionError("Authenticated session or API credential is revoked, expired, or no longer authorized.")
        if (
            refreshed.tenant_id != context.tenant_id
            or refreshed.user_id != context.user_id
            or refreshed.scope_type != context.scope_type
        ):
            raise PermissionError("Authenticated identity changed before MCP execution.")
    except Exception as exc:
        return _query_scoped_authentication_denial(
            question, scope_payload, request_id=request_id, plan=plan,
            reason=f"MCP authentication refresh failed closed: {exc}",
        )

    with activate_tenant_context(refreshed):
        return _query_scoped_business_data_with_context(
            question, scope_payload, request_id=request_id, plan=plan,
            manifest_id=manifest_id, authorization_boundary=authorization_boundary,
            current_query_scope_fingerprint=current_query_scope_fingerprint,
        )


def _query_scoped_business_data_with_context(
    question: str,
    scope_payload: dict[str, object],
    *,
    request_id: str | None = None,
    plan: MetricQueryPlan | None = None,
    manifest_id: str | None = None,
    authorization_boundary=None,
    current_query_scope_fingerprint: str | None = None,
) -> dict[str, Any]:
    """执行受控指标计划，不允许自然语言修改可信数据浏览范围。"""
    if plan is not None and plan.dimensions:
        return {
            "task_type": "metric_query", "question": question,
            "server": "scoped-mcp", "tool_name": None,
            "available_tools": [
                "query_scoped_metric", "query_scoped_metric_compare",
                "query_scoped_trend", "breakdown_scoped_product_category",
                "breakdown_scoped_month_position",
            ],
            "request_id": request_id or str(uuid4()),
            "result": DataQueryAgentResult(
                status="unsupported", parsed=plan,
                reason="多维交叉查询当前不支持受控店铺/浏览范围；为避免扩大数据范围，已停止执行。",
            ).model_dump(),
        }
    if "seller_scope_intent" in scope_payload:
        from app.scope.seller_scope import verify_seller_scope_payload

        try:
            tenant = current_tenant_context()
            if tenant is None:
                raise PermissionError("Trusted tenant context is required")
            intent = verify_seller_scope_payload(scope_payload, tenant=tenant)
        except Exception as exc:
            return {
                "task_type": "metric_query", "question": question,
                "server": "scoped-mcp", "tool_name": None,
                "request_id": request_id or str(uuid4()),
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan or MetricQueryPlan(),
                    reason=f"Invalid structured seller scope intent: {exc}",
                ).model_dump(),
            }
        return _query_scoped_seller_breakdown_with_context(
            question, intent, request_id=request_id,
            plan=plan, authorization_boundary=authorization_boundary,
            current_query_scope_fingerprint=current_query_scope_fingerprint,
        )
    if plan is not None and plan.query_type == "metric_period":
        return {
            "task_type": "metric_query", "question": question,
            "server": "scoped-mcp", "tool_name": None,
            "request_id": request_id or str(uuid4()),
            "result": DataQueryAgentResult(
                status="unsupported", parsed=plan,
                reason="Exact period aggregation is not declared for this locked seller scope.",
            ).model_dump(),
        }
    scope = ControlledAnalysisScope.model_validate(scope_payload)
    # Legacy Explorer scopes are server-issued single-seller scopes. Adapt a
    # seller breakdown request to that same explicit single scope; text cannot
    # promote it to selected/all or change the locked seller/date range.
    if plan is not None and plan.query_type == "metric_breakdown" and plan.dimension == "seller":
        from app.scope.seller_scope import SellerScopeIntent

        if plan.metric_name not in scope.allowed_metrics:
            return {
                "task_type": "metric_query", "question": question,
                "server": "scoped-mcp", "tool_name": None,
                "request_id": request_id or str(uuid4()),
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan,
                    reason="Metric is outside the server-locked Explorer scope.",
                ).model_dump(),
            }
        if "seller" not in scope.allowed_dimensions:
            return {
                "task_type": "metric_query", "question": question,
                "server": "scoped-mcp", "tool_name": None,
                "request_id": request_id or str(uuid4()),
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan,
                    reason="Seller dimension is outside the server-locked Explorer scope.",
                ).model_dump(),
            }

        seller_boundary = authorization_boundary
        seller_current_scope_fingerprint = current_query_scope_fingerprint
        if authorization_boundary is not None:
            # A normal Explorer turn arrives with a boundary for the locked
            # Explorer scope, while this controlled adapter narrows it to one
            # seller breakdown. Validate the incoming boundary against that
            # original server-issued scope before signing a new local seller
            # boundary. A boundary for any other scope continues to the seller
            # handler, which checks it against the resolved seller scope.
            try:
                from app.auth.context import build_authorization_context
                from app.auth.enforcement import (
                    assert_authorization_boundary,
                    coerce_authorization_boundary,
                )
                from app.scope.canonical import CanonicalScope

                tenant = current_tenant_context()
                if tenant is None:
                    raise PermissionError("Trusted tenant context is required to validate Explorer boundary")
                legacy_authz = build_authorization_context(
                    tenant,
                    allowed_metric_ids=scope.allowed_metrics,
                    allowed_dimensions=scope.allowed_dimensions,
                    metric_scope_type="seller",
                )
                legacy_scope = CanonicalScope.resolve(
                    legacy_authz,
                    seller_ids=(scope.seller_id,),
                    start_date=scope.default_start_date,
                    end_date=scope.default_end_date,
                    metric_ids=scope.allowed_metrics,
                    dimensions=scope.allowed_dimensions,
                    granularity=scope.granularity,
                )
                incoming_boundary = coerce_authorization_boundary(authorization_boundary)
                if incoming_boundary.query_scope_fingerprint == legacy_scope.query_scope_fingerprint:
                    if (
                        current_query_scope_fingerprint
                        and current_query_scope_fingerprint != legacy_scope.query_scope_fingerprint
                    ):
                        raise ValueError("Current query fingerprint does not match the locked Explorer scope")
                    assert_authorization_boundary(
                        legacy_authz, incoming_boundary,
                        current_scope_fingerprint=legacy_scope.query_scope_fingerprint,
                    )
                    # The already-validated Explorer boundary is not a seller
                    # boundary. The trusted seller handler signs and checks a
                    # narrower boundary for this call.
                    seller_boundary = None
                    seller_current_scope_fingerprint = None
            except Exception as exc:
                return {
                    "task_type": "metric_query", "question": question,
                    "server": "scoped-mcp", "tool_name": None,
                    "request_id": request_id or str(uuid4()),
                    "result": DataQueryAgentResult(
                        status="unsupported", parsed=plan,
                        reason=f"Explorer authorization boundary is stale or invalid: {exc}",
                    ).model_dump(),
                }

        intent = SellerScopeIntent(
            seller_scope_type="single",
            seller_ids=(scope.seller_id,),
            start_date=scope.default_start_date,
            end_date=scope.default_end_date,
            metric_ids=(plan.metric_name or scope.metric_name,),
            dimensions=("seller",),
            granularity=scope.granularity,
        )
        return _query_scoped_seller_breakdown_with_context(
            question, intent, request_id=request_id, plan=plan,
            authorization_boundary=seller_boundary,
            current_query_scope_fingerprint=seller_current_scope_fingerprint,
        )
    active_authz = None
    if plan is None:
        return {
            "task_type": "metric_query", "question": question,
            "server": "scoped-mcp", "tool_name": None,
            "arguments": {"scope": scope.model_dump(mode="json")},
            "request_id": request_id or str(uuid4()),
            "result": DataQueryAgentResult(
                status="unsupported", parsed=MetricQueryPlan(),
                reason="MCP 执行必须接收服务端生成的 AuthorizedMetric。",
            ).model_dump(),
        }
    if plan is not None:
        if plan.authorized_metric is None:
            return {
                "task_type": "metric_query", "question": question,
                "server": "scoped-mcp", "tool_name": None,
                "arguments": {"scope": scope.model_dump(mode="json")},
                "request_id": request_id or str(uuid4()),
                "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason="QueryPlan 缺少服务端 AuthorizedMetric，已阻止执行。").model_dump(),
            }
        try:
            if plan.metric_name != plan.authorized_metric.metric_id:
                raise ValueError("QueryPlan 指标与 AuthorizedMetric 不一致，已阻止执行。")
            trusted_context = current_tenant_context()
            from app.auth.context import AuthorizationDenied, build_authorization_context

            if trusted_context is None:
                raise AuthorizationDenied(
                    "AUTH_UNKNOWN",
                    "Trusted tenant context is required before data access.",
                )
            authz = build_authorization_context(
                trusted_context,
                allowed_metric_ids=scope.allowed_metrics,
                allowed_dimensions=scope.allowed_dimensions,
                metric_scope_type="seller",
            )
            active_authz = authz
            authz.require_action("analyze")
            authz.require_sellers([scope.seller_id])
            authz.require_metrics([plan.metric_name])
            if plan.dimension:
                authz.require_dimensions([plan.dimension])
            authz.require_date_range(
                scope.seller_id,
                scope.default_start_date,
                scope.default_end_date,
            )
            if authorization_boundary is not None:
                from app.auth.enforcement import assert_authorization_boundary, coerce_authorization_boundary

                boundary = coerce_authorization_boundary(authorization_boundary)
                assert_authorization_boundary(
                    authz,
                    boundary,
                    current_scope_fingerprint=current_query_scope_fingerprint or "",
                )
            validation_scope = {
                **scope.model_dump(mode="json"),
            }
            # New lifecycle-authorized plans include the same server-derived
            # authz fields in their fingerprint.  Keep old test fixtures and
            # legacy plans compatible only when they do not carry that field.
            if plan.authorized_metric.authorization_fingerprint:
                validation_scope.update({
                    # These identity/policy inputs are part of the server
                    # scope fingerprint used by lifecycle-aware planning.
                    # Reconstruct them from trusted context, never the client.
                    "tenant_id": trusted_context.tenant_id,
                    "user_id": trusted_context.user_id,
                    "permissions": trusted_context.role,
                    "scope_type": "seller",
                    "authorization_fingerprint": authz.authz_digest,
                    "permission_version": authz.permission_version,
                    "policy_set_version": authz.policy_set_version,
                    "authz_ref": authz.authz_ref,
                })
            validate_authorized_metric(
                plan.authorized_metric,
                scope="seller",
                scope_data=validation_scope,
            )
        except Exception as exc:
            return {
                "task_type": "metric_query", "question": question,
                "server": "scoped-mcp", "tool_name": None,
                "arguments": {"scope": scope.model_dump(mode="json")},
                "request_id": request_id or str(uuid4()),
                "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason=str(exc)).model_dump(),
            }
    scope_time_bounds = (scope.default_start_date, scope.default_end_date)
    time_scope_error: str | None = None
    planned_date_values = (
        plan.start_date, plan.end_date, plan.metric_date,
        plan.left_date, plan.right_date, plan.current_date, plan.baseline_date,
        plan.current_start, plan.current_end, plan.baseline_start, plan.baseline_end,
    )
    for raw_date in planned_date_values:
        if not raw_date:
            continue
        try:
            planned_date = date.fromisoformat(str(raw_date))
        except ValueError:
            time_scope_error = "查询计划包含无效日期，已阻止执行。"
            break
        if planned_date < scope.default_start_date or planned_date > scope.default_end_date:
            time_scope_error = (
                f"请求日期 {planned_date.isoformat()} 超出当前会话锁定范围 "
                f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}；"
                "请先在数据浏览中调整范围。"
            )
            break
    try:
        requested_time = resolve_expression(question, data_bounds=scope_time_bounds)
        if requested_time is not None and requested_time.truncated and time_scope_error is None:
            time_scope_error = (
                f"请求时间范围 {requested_time.requested_start.isoformat()} 至 "
                f"{requested_time.requested_end.isoformat()} 超出当前会话锁定范围 "
                f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}；"
                "请先在数据浏览中调整范围。"
            )
    except ValueError as exc:
        if time_scope_error is None:
            time_scope_error = f"无法在当前会话锁定范围内解析请求的时间：{exc}"
    plan = apply_trusted_time_range(plan, question, data_bounds=scope_time_bounds)
    # A fully specified comparison plan is already the canonical time
    # interpretation.  Re-resolving its natural-language question here would
    # unnecessarily consult the global dataset clock (and can block scoped
    # execution when that unrelated service is unavailable).
    has_explicit_comparison_periods = (
        plan.query_type == "metric_compare"
        and bool(
            (plan.current_date and plan.baseline_date)
            or (plan.left_date and plan.right_date)
        )
    ) or (
        plan.query_type == "metric_contribution"
        and bool(
            plan.current_start and plan.current_end
            and plan.baseline_start and plan.baseline_end
        )
    )
    if has_explicit_comparison_periods:
        resolved_time = None
    else:
        try:
            resolved_time = resolve_expression(question, data_bounds=scope_time_bounds)
        except ValueError:
            resolved_time = None
    # 锁定范围内未写日期，默认使用完整锁定区间。通用规划器通常依赖日期
    # 才能识别趋势，因此保留用户的分析意图，但绝不接受新的日期边界。
    if plan.query_type == "metric_value" and ("趋势" in question or "trend" in question.lower()):
        plan = plan.model_copy(update={"query_type": "metric_trend"})
    request_id = request_id or str(uuid4())

    def unsupported(reason: str) -> dict[str, Any]:
        labels = {
            "product_category": "\u5546\u54c1\u54c1\u7c7b",
            "payment_type": "\u652f\u4ed8\u65b9\u5f0f",
            "month_position": "\u6708\u5185\u65e5\u671f\u4f4d\u7f6e",
        }
        allowed_text = ", ".join(labels.get(item, item) for item in scope.allowed_dimensions)
        if allowed_text:
            reason = f"{reason} \u5f53\u524d\u5141\u8bb8\u7684\u7ef4\u5ea6\uff1a{allowed_text}\u3002"
        return {
            "task_type": "metric_query", "question": question, "server": "scoped-mcp",
            "available_tools": ["query_scoped_metric", "query_scoped_metric_compare", "query_scoped_trend", "breakdown_scoped_product_category", "breakdown_scoped_month_position"],
            "tool_name": None, "arguments": {"scope": scope.model_dump(mode="json")}, "request_id": request_id,
            "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason=reason).model_dump(),
        }

    def contribution_failure(reason: str) -> dict[str, Any]:
        """Return a typed failure when scoped contribution facts do not reconcile."""
        return {
            "task_type": "metric_query", "question": question, "server": "scoped-mcp",
            "available_tools": [
                "compare_breakdown_by_dimension",
                "compare_breakdown_by_dimension_period",
            ],
            "tool_name": None,
            "arguments": {"scope": scope.model_dump(mode="json")},
            "request_id": request_id,
            "result": DataQueryAgentResult(
                status="execution_failed", parsed=plan, reason=reason,
            ).model_dump(),
        }
    if time_scope_error:
        return unsupported(time_scope_error)
    inferred_dimension = _extract_dimension(question)
    allowed_dimensions = set(scope.allowed_dimensions)
    if plan.dimension and plan.dimension not in allowed_dimensions:
        return unsupported("请求的维度不在当前锁定分析范围内，不能改用其他维度代替。")
    if inferred_dimension and inferred_dimension not in allowed_dimensions:
        dimension_tokens = ("拆解", "分布", "按", "商品", "品类", "城市", "地区", "支付")
        if any(token in question for token in dimension_tokens) or " by " in f" {question.casefold()} ":
            return unsupported("请求的维度不在当前锁定分析范围内，不能改用其他维度代替。")
    if plan.query_type == "metric_value" and inferred_dimension in set(scope.allowed_dimensions):
        # 未写日期的派生维度默认在完整锁定范围内拆解。
        plan = plan.model_copy(update={"query_type": "metric_breakdown", "dimension": inferred_dimension})
    elif plan.query_type == "metric_value" and any(token in question for token in ("拆解", "分布", "品类")):
        plan = plan.model_copy(update={"query_type": "metric_breakdown", "dimension": "product_category"})
    metric_name = plan.metric_name or scope.metric_name
    versioned_gmv = bool(manifest_id and metric_name == "gmv")
    # 当锁定范围覆盖整个自然月且问题含月份位置关键词时，以 scope 日期为准
    # 覆盖全局 resolve_expression 解析出的错误月份（如"本月"被解析为 10 月而非 8 月）
    _is_single_month = scope.default_start_date.day == 1 and scope.default_start_date.month == scope.default_end_date.month
    if _is_single_month and any(t in question for t in ("月初", "月中", "月末", "月内分布", "月份分布", "本月")):
        plan = plan.model_copy(update={
            "start_date": scope.default_start_date.isoformat(),
            "end_date": scope.default_end_date.isoformat(),
            "metric_date": scope.default_start_date.isoformat(),
        })
    if metric_name not in scope.allowed_metrics:
        return unsupported("当前会话只允许查询锁定范围内的关联指标。")
    mentioned_sellers = set(re.findall(r"\b[a-f0-9]{32}\b", question.lower()))
    if mentioned_sellers and mentioned_sellers != {scope.seller_id.lower()}:
        return unsupported("请求店铺与当前锁定范围不一致；请先修改数据浏览范围。")
    if plan.query_type == "metric_contribution":
        pass  # handled in routing below, after period validation


    # === Period 解析：统一转为 Period 列表 ===
    resolved_periods = _resolve_periods(plan, scope, question)

    # === QueryPolicy 验证：检查跨度/点数 ===
    policy_rejection = validate_periods(resolved_periods, scope)
    if policy_rejection:
        return unsupported(policy_rejection)

    # 初始化聊天文字日期变量，供后续比较
    trend_start = trend_end = chat_metric_date = None
    if plan.query_type == "metric_entity_statistics":
        if metric_name != "seller_avg_freight_per_order":
            return unsupported("当前仅支持按订单实体统计店铺每单运费分布。")
        trend_start = date.fromisoformat(plan.start_date) if plan.start_date else scope.default_start_date
        trend_end = date.fromisoformat(plan.end_date) if plan.end_date else scope.default_end_date
        from app.data.repositories.explorer import query_scoped_order_freight_distribution

        payload = query_scoped_order_freight_distribution(
            scope, start_date=trend_start, end_date=trend_end,
        )
        effective_plan = plan.model_copy(update={
            "metric_name": metric_name,
            "start_date": trend_start.isoformat(),
            "end_date": trend_end.isoformat(),
        })
        tool_name = "query_scoped_order_freight_distribution"
    elif plan.query_type == "metric_association":
        pair_id = plan.association_pair_id
        from app.analysis.association_catalog import is_registered_pair_for_metric
        if not pair_id or not is_registered_pair_for_metric(pair_id, str(metric_name)):
            return unsupported("当前指标范围没有注册该关联分析组合。")
        trend_start = date.fromisoformat(plan.start_date) if plan.start_date else scope.default_start_date
        trend_end = date.fromisoformat(plan.end_date) if plan.end_date else scope.default_end_date
        from app.data.repositories.explorer import query_scoped_delivery_review_association

        payload = query_scoped_delivery_review_association(
            scope, start_date=trend_start, end_date=trend_end, pair_id=pair_id,
        )
        effective_plan = plan.model_copy(update={
            "metric_name": metric_name,
            "start_date": trend_start.isoformat(),
            "end_date": trend_end.isoformat(),
        })
        tool_name = "query_scoped_registered_association"
    elif plan.query_type == "metric_statistics":
        trend_start = plan.start_date or scope.default_start_date
        trend_end = plan.end_date or scope.default_end_date
        if isinstance(trend_start, str):
            trend_start = date.fromisoformat(trend_start)
        if isinstance(trend_end, str):
            trend_end = date.fromisoformat(trend_end)
        payload = query_scoped_statistics(
            scope,
            metric_name,
            start_date=trend_start,
            end_date=trend_end,
            granularity=plan.granularity or scope.granularity,
            requested=plan.statistics,
        )
        effective_plan = plan.model_copy(update={
            "metric_name": metric_name,
            "start_date": trend_start.isoformat(),
            "end_date": trend_end.isoformat(),
            "granularity": plan.granularity or scope.granularity,
        })
        tool_name = "query_scoped_metric_statistics"
    elif plan.query_type == "metric_breakdown":
        if versioned_gmv:
            return unsupported("Versioned GMV pilot deliberately supports value, trend, and compare only.")
        if plan.dimension == "payment_type":
            payload = breakdown_scoped_payment_type(scope, metric_name)
            effective_plan = plan.model_copy(update={"metric_name": metric_name, "metric_date": None})
            tool_name = "breakdown_scoped_payment_type"
        elif plan.dimension in {"month_position", "day_of_week", "hour_of_day", "quarter_of_year", "half_of_year"}:
            payload = breakdown_scoped_time_dimension(scope, metric_name, plan.dimension)
            effective_plan = plan.model_copy(update={"metric_name": metric_name, "metric_date": None})
            tool_name = f"breakdown_scoped_{plan.dimension}"
        elif plan.dimension not in (None, "product_category"):
            return unsupported("当前锁定范围只支持商品品类及已开放的派生时间维度拆解。")
        else:
            payload = breakdown_scoped_product_category(scope, metric_name, plan.limit or 10)
            effective_plan = plan.model_copy(update={"metric_name": metric_name, "dimension": "product_category", "metric_date": None})
            tool_name = "breakdown_scoped_product_category"
    elif plan.query_type == "metric_compare":
        periods = _resolve_scoped_compare_periods(scope, question, plan)
        if periods is None:
            return unsupported("无法确定对比周期；请检查问题是否包含可识别的月份对比。")
        current_start, current_end, baseline_start, baseline_end = periods
        if versioned_gmv:
            from app.data_versions.service import query_gmv_manifest
            current = query_gmv_manifest(current_tenant_context(), manifest_id, scope.seller_id, current_start, current_end)
            baseline = query_gmv_manifest(current_tenant_context(), manifest_id, scope.seller_id, baseline_start, baseline_end)
            payload = {"metric_name": "gmv", "left_value": current["value"], "right_value": baseline["value"], "delta_value": current["value"] - baseline["value"], "delta_ratio": ((current["value"] - baseline["value"]) / baseline["value"] if baseline["value"] else None), "manifest_id": manifest_id, "manifest_hash": current["manifest_hash"], "source_table": current["source_table"], "status": "ok"}
        else:
            payload = compare_scoped_metric(
            scope, metric_name,
            current_start=current_start, current_end=current_end,
            baseline_start=baseline_start, baseline_end=baseline_end,
            )
        effective_plan = plan.model_copy(update={
            "metric_name": metric_name,
            "left_date": current_start.isoformat(),
            "right_date": baseline_start.isoformat(),
            "start_date": current_start.isoformat(),
            "end_date": current_end.isoformat(),
            "granularity": "month",
        })
        tool_name = "query_scoped_metric_compare"
    elif plan.query_type == "metric_contribution":
        # Contribution is permitted only when the Registry explicitly
        # declares additive attribution for this seller-scoped metric. The
        # unified resolver keeps ordinary implementation metadata separate
        # from attribution metadata, so order_count's seller value/trend
        # declaration cannot be mistaken for contribution support.
        from app.context.support_resolver import resolve_support

        requested_dimension = plan.dimension or "product_category"
        support = resolve_support(
            metric_id=metric_name,
            dimension=requested_dimension,
            operation="contribution",
            scope=scope,
            authz=active_authz,
        )
        if not support.executable:
            messages = {
                "contribution_not_supported": "当前店铺范围未声明该指标的可加总贡献分析能力。",
                "contribution_not_additive": "该指标是比率/均值类指标，不能使用加法贡献方法。",
                "contribution_distinct_not_additive": "该指标按该维度分组的去重口径不可加总，不能使用加法贡献方法。",
                "dimension_not_supported_by_metric": "当前店铺范围未声明该指标在所请求维度上的可加总贡献能力。",
                "dimension_outside_locked_scope": "请求的维度不在当前锁定范围内。",
                "dimension_not_permitted": "当前账号无权按该维度查看。",
                "dimension_not_registered": "该维度未在指标体系中定义。",
                "support_unknown": "无法验证指标归因能力，已阻止贡献查询。",
            }
            return unsupported(
                messages.get(
                    support.reason_code or "",
                    "当前店铺范围不支持该贡献分析。",
                )
            )
        filters = dict(plan.filters or {})
        if filters and (set(filters) != {"product_category"} or plan.dimension != "month_position"):
            return unsupported("当前锁定范围仅支持使用已验证的商品品类证据筛选月内时段拆解。")
        if plan.dimension not in (None, "product_category", "month_position"):
            return unsupported("当前锁定范围只支持商品品类拆解，以及证据约束的品类内月内时段拆解。")

        periods = _resolve_scoped_compare_periods(scope, question, plan)
        if periods is None:
            return unsupported("无法确定两段对比周期；请检查问题是否包含可识别的月份对比。")
        current_start, current_end, baseline_start, baseline_end = periods

        current_scope = scope.model_copy(update={"default_start_date": current_start, "default_end_date": current_end})
        baseline_scope = scope.model_copy(update={"default_start_date": baseline_start, "default_end_date": baseline_end})
        limit = plan.limit or 10

        current_items = baseline_items = []
        filtered_month_position = plan.dimension == "month_position" and bool(filters)
        if filtered_month_position:
            category_filter = filters["product_category"]
            current_breakdown = breakdown_scoped_month_position(
                current_scope, metric_name, product_category_filter=category_filter,
            )
            baseline_breakdown = breakdown_scoped_month_position(
                baseline_scope, metric_name, product_category_filter=category_filter,
            )
            output_dimension = "month_position"
        elif plan.dimension == "month_position":
            current_breakdown = breakdown_scoped_month_position(current_scope, metric_name)
            baseline_breakdown = breakdown_scoped_month_position(baseline_scope, metric_name)
            output_dimension = "month_position"
        else:
            output_dimension = "product_category"

        if metric_name == "order_count" and output_dimension == "product_category":
            # A seller order can touch several categories. Allocate each order
            # equally across the categories present inside this seller's locked
            # rows so group values reconcile with the distinct-order total.
            payload = compare_scoped_order_count_allocation(
                scope,
                current_start, current_end, baseline_start, baseline_end,
                limit=limit,
            )
            contribution_arguments = {
                "dimension": output_dimension,
                "baseline_date": baseline_start.isoformat(),
                "current_date": current_start.isoformat(),
                "baseline_start": baseline_start.isoformat(),
                "baseline_end": baseline_end.isoformat(),
                "current_start": current_start.isoformat(),
                "current_end": current_end.isoformat(),
            }
            day_pair = (
                current_start == current_end and baseline_start == baseline_end
            )
            tool_name = "compare_breakdown_by_dimension" if day_pair else "compare_breakdown_by_dimension_period"
            effective_plan = plan.model_copy(update={
                "metric_name": metric_name, "dimension": output_dimension,
                "current_date": current_start.isoformat(), "baseline_date": baseline_start.isoformat(),
                "start_date": current_start.isoformat(), "end_date": current_end.isoformat(),
                "limit": limit,
            })
        else:
            if output_dimension == "product_category":
                current_breakdown = breakdown_scoped_product_category(current_scope, metric_name, limit=limit)
                baseline_breakdown = breakdown_scoped_product_category(baseline_scope, metric_name, limit=limit)
            current_items = current_breakdown.get("items", [])
            baseline_items = baseline_breakdown.get("items", [])

            def checked_total(value: object) -> float | None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                number = float(value)
                return number if math.isfinite(number) else None

            if filtered_month_position:
                # population_total is calculated by the same repository query
                # and category predicate as the month_position groups. A
                # whole-seller compare would have a different population.
                current_total = checked_total(current_breakdown.get("population_total"))
                baseline_total = checked_total(baseline_breakdown.get("population_total"))
                if current_total is None or baseline_total is None:
                    return contribution_failure(
                        "过滤后的月内时段查询没有提供同口径的独立总值，已停止贡献投影。",
                    )
            else:
                comp = compare_scoped_metric(
                    scope, metric_name,
                    current_start=current_start, current_end=current_end,
                    baseline_start=baseline_start, baseline_end=baseline_end,
                )
                # compare_scoped_metric uses left=current, right=baseline.
                current_total = checked_total(comp.get("left_value", comp.get("current_value")))
                baseline_total = checked_total(comp.get("right_value", comp.get("baseline_value")))
                if current_total is None or baseline_total is None or comp.get("status") != "ok":
                    return contribution_failure(
                        "受控比较没有返回可核验的基线值和当前值，已停止贡献投影。",
                    )

            total_delta = current_total - baseline_total
            if not math.isfinite(total_delta):
                return contribution_failure("贡献总变化不是有限数值，已停止投影。")

            # month_position has a complete, fixed set of buckets. Reconcile
            # these grouped facts to totals from exactly the same population.
            if plan.dimension == "month_position":
                for label, items, expected_total in (
                    ("当前期", current_items, current_total),
                    ("基线期", baseline_items, baseline_total),
                ):
                    try:
                        grouped_total = sum(
                            float(item["value"])
                            for item in items
                            if isinstance(item, dict)
                            and not isinstance(item.get("value"), bool)
                            and isinstance(item.get("value"), (int, float))
                            and math.isfinite(float(item["value"]))
                        )
                        if len(items) != sum(
                            1 for item in items
                            if isinstance(item, dict)
                            and not isinstance(item.get("value"), bool)
                            and isinstance(item.get("value"), (int, float))
                            and math.isfinite(float(item["value"]))
                        ):
                            raise ValueError("invalid group value")
                    except (TypeError, ValueError, OverflowError):
                        return contribution_failure(
                            f"{label}月内时段分组包含无效数值，已停止贡献投影。",
                        )
                    if abs(grouped_total - expected_total) > 0.01:
                        return contribution_failure(
                            f"{label}月内时段分组与同口径总值无法对账，已停止贡献投影。",
                        )

            point_pair = current_start == current_end and baseline_start == baseline_end
            if point_pair:
                tool_name = "compare_breakdown_by_dimension"
                contribution_arguments = {
                    "metric_name": metric_name,
                    "current_date": current_start.isoformat(),
                    "baseline_date": baseline_start.isoformat(),
                    "dimension": output_dimension,
                    "limit": limit,
                }
                effective_plan = plan.model_copy(update={
                    "metric_name": metric_name, "dimension": output_dimension,
                    "current_date": current_start.isoformat(),
                    "baseline_date": baseline_start.isoformat(),
                    "current_start": None, "current_end": None,
                    "baseline_start": None, "baseline_end": None,
                    "limit": limit,
                })
                current_period = baseline_period = None
            else:
                tool_name = "compare_breakdown_by_dimension_period"
                contribution_arguments = {
                    "metric_name": metric_name,
                    "current_start": current_start.isoformat(),
                    "current_end": current_end.isoformat(),
                    "baseline_start": baseline_start.isoformat(),
                    "baseline_end": baseline_end.isoformat(),
                    "granularity": plan.granularity or "month",
                    "dimension": output_dimension,
                    "limit": limit,
                }
                effective_plan = plan.model_copy(update={
                    "metric_name": metric_name, "dimension": output_dimension,
                    "current_date": None, "baseline_date": None,
                    "current_start": current_start.isoformat(),
                    "current_end": current_end.isoformat(),
                    "baseline_start": baseline_start.isoformat(),
                    "baseline_end": baseline_end.isoformat(),
                    "granularity": plan.granularity or "month",
                    "limit": limit,
                })
                current_period = {
                    "start_date": current_start.isoformat(),
                    "end_date": current_end.isoformat(),
                    "granularity": plan.granularity or "month",
                }
                baseline_period = {
                    "start_date": baseline_start.isoformat(),
                    "end_date": baseline_end.isoformat(),
                    "granularity": plan.granularity or "month",
                }

            payload = {
                "metric_name": metric_name, "dimension": output_dimension,
                "current_date": current_start.isoformat(), "baseline_date": baseline_start.isoformat(),
                "current_items": current_items, "baseline_items": baseline_items,
                "current_total": current_total, "baseline_total": baseline_total,
                "total_delta": total_delta,
                "method": "additive_change_v1",
                **({"current_period": current_period, "baseline_period": baseline_period}
                   if current_period and baseline_period else {}),
                "source_table": current_breakdown.get("source_table", "scoped_breakdown"),
                "status": "ok",
            }
    elif plan.query_type == "metric_trend":
        # 直接使用用户指定日期；未指定时 query_scoped_trend 内部用 scope 默认范围
        trend_start = plan.start_date or scope.default_start_date
        trend_end = plan.end_date or scope.default_end_date
        if isinstance(trend_start, str):
            trend_start = date.fromisoformat(trend_start)
        if isinstance(trend_end, str):
            trend_end = date.fromisoformat(trend_end)
        if versioned_gmv:
            from app.data_versions.service import query_gmv_manifest_trend
            payload = query_gmv_manifest_trend(current_tenant_context(), manifest_id, scope.seller_id, trend_start, trend_end)
        else:
            trend_granularity = plan.granularity or scope.granularity
            trend_scope = scope.model_copy(update={"granularity": trend_granularity})
            payload = query_scoped_trend(trend_scope, metric_name, start_date=trend_start, end_date=trend_end)
        effective_plan = plan.model_copy(update={"metric_name": metric_name, "start_date": trend_start.isoformat(), "end_date": trend_end.isoformat(), "granularity": plan.granularity or scope.granularity})
        tool_name = "query_scoped_trend"
    else:
        # 直接使用用户指定日期（可能是None）；query_scoped_metric 会用 scope 默认范围汇总
        chat_metric_date = plan.metric_date
        if versioned_gmv:
            from app.data_versions.service import query_gmv_manifest
            metric_day = date.fromisoformat(chat_metric_date) if chat_metric_date else None
            payload = query_gmv_manifest(current_tenant_context(), manifest_id, scope.seller_id, metric_day or scope.default_start_date, metric_day or scope.default_end_date)
        else:
            payload = (query_scoped_metric(scope, metric_name, metric_date=chat_metric_date)
                       if chat_metric_date is not None else query_scoped_metric(scope, metric_name))
        effective_plan = plan.model_copy(update={"metric_name": metric_name, "metric_date": chat_metric_date or None})
        tool_name = "query_scoped_metric"
    # Normalize the common envelope before Pydantic validation.  Some fixed
    # seller-extension handlers intentionally return only metric_name/points;
    # the workflow contract still requires the trusted period and granularity
    # on every trend result.  Likewise, a point handler may omit metric_date
    # when it represents the entire locked range.
    if effective_plan.query_type == "metric_trend":
        payload = {
            **payload,
            "metric_name": payload.get("metric_name") or metric_name,
            "start_date": (
                effective_plan.start_date.isoformat()
                if hasattr(effective_plan.start_date, "isoformat")
                else effective_plan.start_date or scope.default_start_date.isoformat()
            ),
            "end_date": (
                effective_plan.end_date.isoformat()
                if hasattr(effective_plan.end_date, "isoformat")
                else effective_plan.end_date or scope.default_end_date.isoformat()
            ),
            "granularity": effective_plan.granularity or scope.granularity,
        }
    elif effective_plan.query_type == "metric_value":
        payload = {
            **payload,
            "metric_name": payload.get("metric_name") or metric_name,
            "metric_date": payload.get("metric_date") or (
                effective_plan.metric_date
                or f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}"
            ),
        }

    # 记录聊天文字与面板默认范围是否有差异，供 answer 节点展示
    analysis_context_note = _build_analysis_context_note(
        scope.default_start_date, scope.default_end_date, resolved_periods,
    )
    from app.security.result_policy import apply_result_policy
    governed_payload = apply_result_policy(payload, authorization_context=active_authz).payload
    response = {
        "task_type": "metric_query", "question": question, "server": "scoped-mcp", "tool_name": tool_name,
        "available_tools": [
            "query_scoped_metric", "query_scoped_metric_compare", "query_scoped_trend",
            "breakdown_scoped_product_category", "breakdown_scoped_month_position",
            "compare_breakdown_by_dimension", "compare_breakdown_by_dimension_period",
        ],
        "arguments": {"scope": scope.model_dump(mode="json"), "metric_name": metric_name,
                      **(contribution_arguments if effective_plan.query_type == "metric_contribution" else {}),
                      "manifest_id": manifest_id, "time_resolution": (resolved_time.as_dict() if resolved_time else {
                          "raw_expression": question,
                          "start_date": effective_plan.start_date,
                          "end_date": effective_plan.end_date,
                          "granularity": effective_plan.granularity or scope.granularity,
                          "periods": [
                              {
                                  "name": item.get("name"),
                                  "start_date": item["start_date"].isoformat() if item.get("start_date") else None,
                                  "end_date": item["end_date"].isoformat() if item.get("end_date") else None,
                              }
                              for item in resolved_periods
                          ],
                      })}, "request_id": request_id,
        "result": _result_from_payload(effective_plan, governed_payload).model_dump(),
        "analysis_context_note": analysis_context_note,
    }
    if effective_plan.query_type == "metric_contribution":
        response["execution_metadata"] = {
            "controlled_handlers": (
                ["breakdown_scoped_month_position"]
                if plan.dimension == "month_position"
                else ["breakdown_scoped_product_category"]
            ) + ([] if filtered_month_position else ["compare_scoped_metric"]),
            "filter_applied": dict(plan.filters or {}),
        }
    return response


def _query_scoped_seller_breakdown_with_context(
    question: str,
    intent_payload,
    *,
    request_id: str | None,
    plan: MetricQueryPlan | None,
    authorization_boundary=None,
    current_query_scope_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Authorize a seller-breakdown intent and resolve it against trusted identity."""
    request_id = request_id or str(uuid4())
    if plan is None or plan.authorized_metric is None:
        reason = "Seller breakdown requires a server-authorized metric plan."
        return {
            "task_type": "metric_query", "question": question, "server": "scoped-mcp",
            "tool_name": None, "request_id": request_id,
            "result": DataQueryAgentResult(status="unsupported", parsed=plan or MetricQueryPlan(), reason=reason).model_dump(),
        }
    if plan.query_type != "metric_breakdown" or plan.dimension != "seller":
        return {
            "task_type": "metric_query", "question": question, "server": "scoped-mcp",
            "tool_name": None, "request_id": request_id,
            "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason="Seller scope intent only supports a seller breakdown plan.").model_dump(),
        }
    tenant = current_tenant_context()
    if tenant is None:
        return {
            "task_type": "metric_query", "question": question, "server": "scoped-mcp",
            "tool_name": None, "request_id": request_id,
            "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason="Trusted tenant context is required.").model_dump(),
        }
    try:
        from app.auth.context import build_authorization_context
        from app.auth.enforcement import (
            AuthorizationBoundary, assert_authorization_boundary,
            coerce_authorization_boundary,
        )
        from app.metrics.authorization import authorize_metric
        from app.scope.seller_scope import resolve_seller_scope

        auth_scope = "seller" if tenant.scope_type == "seller" else "platform"
        authz = build_authorization_context(tenant, metric_scope_type=auth_scope)
        authz.require_action("analyze")
        if plan.metric_name != plan.authorized_metric.metric_id:
            raise ValueError("QueryPlan metric does not match AuthorizedMetric")
        scope = resolve_seller_scope(intent_payload, tenant=tenant, authz=authz)
        authz.require_metrics((plan.metric_name,))
        validation_scope = scope.as_query_dict() | {
            "tenant_id": tenant.tenant_id,
            "user_id": tenant.user_id,
            "permissions": tenant.role,
            "scope_type": auth_scope,
            "metric_name": plan.metric_name,
            "allowed_metrics": list(scope.metric_ids),
            "allowed_dimensions": list(scope.dimensions),
            "authorization_fingerprint": authz.authz_digest,
            "permission_version": authz.permission_version,
            "policy_set_version": authz.policy_set_version,
            "authz_ref": authz.authz_ref,
        }
        if (
            authorization_boundary is not None
            and current_query_scope_fingerprint
            and current_query_scope_fingerprint != scope.query_scope_fingerprint
        ):
            raise ValueError("Current query scope fingerprint does not match the resolved seller scope")
        # A previous AuthorizedMetric carries provenance, not authority for
        # this newly resolved seller scope. Mint a fresh credential over the
        # exact server-derived seller scope, then validate it at this boundary.
        provenance = plan.authorized_metric
        if provenance.provenance_type == "semantic_mapping":
            validation_scope["confirmed_mapping_ids"] = [provenance.provenance_id]
        seller_metric = authorize_metric(
            plan.metric_name,
            provenance_type=provenance.provenance_type,
            provenance_id=provenance.provenance_id,
            scope=auth_scope,
            scope_data=validation_scope,
        )
        validate_authorized_metric(
            seller_metric, scope=auth_scope, scope_data=validation_scope,
        )
        # If the upstream planner supplied a boundary, it must already bind
        # this seller scope. Legacy single-scope fingerprints are not reused;
        # this trusted entry signs a fresh boundary after resolving the seller
        # scope and the current authorization snapshot.
        boundary = (
            coerce_authorization_boundary(authorization_boundary)
            if authorization_boundary is not None
            else AuthorizationBoundary(
                permission_version=authz.permission_version,
                policy_set_version=authz.policy_set_version,
                authorization_fingerprint=authz.authz_digest,
                query_scope_fingerprint=scope.query_scope_fingerprint,
            )
        )
        assert_authorization_boundary(
            authz, boundary,
            current_scope_fingerprint=scope.query_scope_fingerprint,
        )
        from app.data.repositories.explorer import breakdown_by_seller
        payload = breakdown_by_seller(scope, plan.metric_name, limit=plan.limit or 10)
    except Exception as exc:
        return {
            "task_type": "metric_query", "question": question, "server": "scoped-mcp",
            "tool_name": None, "arguments": {
                "seller_scope_type": getattr(intent_payload, "seller_scope_type", None),
            },
            "request_id": request_id,
            "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason=str(exc)).model_dump(),
        }
    return {
        "task_type": "metric_query", "question": question, "server": "scoped-mcp",
        "available_tools": ["breakdown_scoped_seller"],
        "tool_name": "breakdown_scoped_seller",
        "arguments": {
            "metric_name": plan.metric_name,
            "dimension": "seller",
            "seller_scope_type": scope.seller_scope_type,
            "seller_ids": list(scope.seller_ids) if scope.seller_scope_type != "all" else None,
            "include_unattributed": scope.include_unattributed,
            "query_scope_fingerprint": scope.query_scope_fingerprint,
        },
            "seller_boundary": {
                "permission_version": boundary.permission_version,
                "policy_set_version": boundary.policy_set_version,
                "authorization_fingerprint": boundary.authorization_fingerprint,
                "query_scope_fingerprint": boundary.query_scope_fingerprint,
            },
        "seller_scope_fingerprint": scope.query_scope_fingerprint,
        "request_id": request_id,
        "result": _result_from_payload(
            plan.model_copy(update={"authorized_metric": seller_metric}),
            {
                **payload,
                "metric_date": scope.start_date.isoformat(),
                "start_date": scope.start_date.isoformat(),
                "end_date": scope.end_date.isoformat(),
                "items": [
                    {
                        "dimension_value": item["seller_id"],
                        "seller_id": item["seller_id"],
                        "value": item["value"],
                        "share": None,
                        "order_count": item["order_count"],
                    }
                    for item in payload["items"]
                ],
            },
        ).model_dump(),
    }


def is_platform_registry_value_plan(plan: MetricQueryPlan, tenant_context=None) -> bool:
    """Whether an authorized platform point-value plan has the generic route.

    This is only a dispatch classification. MetricRegistry.execute remains
    responsible for checking the registered scope capability and handler.
    """
    return bool(
        plan.query_type == "metric_value"
        and plan.metric_name
        and plan.metric_date
        and tenant_context is not None
        and tenant_context.scope_type == "platform"
    )


def _query_business_data_impl(
    task_type: str,
    question: str,
    *,
    tenant_id: str | None = None,
    role: str | None = None,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    plan: MetricQueryPlan | None = None,
    authorization_boundary=None,
    current_query_scope_fingerprint: str | None = None,
) -> dict[str, Any]:
    """
    通过 MCP 层执行查数，并返回适合写入 trace 的统一结构。

    顶层保留这些字段：
    - `server`
    - `tool_name`
    - `arguments`
    - `result`

    这样 trace 能清楚记录：到底调用了哪个 MCP 工具，传了什么参数，拿到了什么结果。
    返回一个status

    处理流程可以理解成三步：
    1. 先判断这个任务是不是指标查询任务
    2. 如果是，再把自然语言问题解析成查询计划
    3. 如果计划能映射到具体工具，就发起 MCP 调用并包装返回结果
    """

    # 条件 1：只有 metric_query 类型的任务才需要走查数流程。
    # 其他任务在当前 MVP 里不做指标查询，因此直接返回 unsupported。
    request_id = request_id or str(uuid4())
    idempotency_key = idempotency_key or request_id
    if task_type != "metric_query":
        # 如果这个task_type不属于指标查询，那么就
        # 这里先构造一个“被跳过”的统一结果，方便上游记录 trace。
        skipped = DataQueryAgentResult(
            status="unsupported",
            parsed=build_metric_query_plan(question),
            # 说明为什么跳过：当前任务类型不需要查指标数据。
            reason=f"Task type {task_type} does not require metric querying in the current MVP.",
        )
        return {
            "task_type": task_type,
            "question": question,
            "server": SERVER_NAME,
            "available_tools": get_supported_tools(),
            "tool_name": None,
            "arguments": {},
            "data": [],
            "result": skipped.model_dump(),
        }

    # 条件 2：先把自然语言问题转换成查询计划。
    # 优先使用 LLM 提取结构化计划；不可用时降级到规则解析。
    if plan is None:
        return {
            "task_type": task_type, "question": question, "server": SERVER_NAME,
            "tool_name": None, "arguments": {}, "data": [], "request_id": request_id,
            "idempotency_key": idempotency_key,
            "result": DataQueryAgentResult(
                status="unsupported", parsed=MetricQueryPlan(),
                reason="MCP 执行必须接收服务端生成的 AuthorizedMetric。",
            ).model_dump(),
        }
    if plan is not None:
        if plan.authorized_metric is None:
            return {
                "task_type": task_type, "question": question, "server": SERVER_NAME,
                "available_tools": get_supported_tools(),
                "tool_name": None, "arguments": {}, "data": [],
                "request_id": request_id, "idempotency_key": idempotency_key,
                "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason="QueryPlan 缺少服务端 AuthorizedMetric，已阻止执行。").model_dump(),
            }
        if plan.metric_name != plan.authorized_metric.metric_id:
            return {
                "task_type": task_type, "question": question, "server": SERVER_NAME,
                "available_tools": get_supported_tools(),
                "tool_name": None, "arguments": {}, "data": [],
                "request_id": request_id, "idempotency_key": idempotency_key,
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan,
                    reason="QueryPlan 指标与 AuthorizedMetric 不一致，已阻止执行。",
                ).model_dump(),
            }
        context = current_tenant_context()
        scope_data = {
            "tenant_id": tenant_id or (context.tenant_id if context else None),
            "user_id": context.user_id if context else None,
            "permissions": role or (context.role if context else None),
            "scope_type": "platform",
        }
        try:
            if context is None:
                raise ValueError("AUTH_UNKNOWN: trusted tenant context is required")
            # Dimension registration and dimension authorization are separate
            # checks. Contribution routing validates the metric's registered
            # operation/dimension contract below; the query boundary must also
            # enforce the same identity-derived dimension grant used by the
            # diagnostic planner, even when no optional authorization-boundary
            # token was supplied.
            from app.auth.context import build_authorization_context

            current_authz = build_authorization_context(
                context,
                allowed_metric_ids=[plan.metric_name] if plan.metric_name else None,
                metric_scope_type="platform",
            )
            if plan.dimension:
                current_authz.require_dimensions([plan.dimension])
            if plan.dimensions:
                current_authz.require_multi_dimension_dimensions(plan.dimensions)
                current_authz.require_action("analyze")
                current_authz.require_metrics([plan.metric_name] if plan.metric_name else [])
                current_authz.require_tool("query_metric_by_dimensions")
            if authorization_boundary is not None:
                from app.auth.enforcement import assert_authorization_boundary, coerce_authorization_boundary
                authorization_boundary = coerce_authorization_boundary(authorization_boundary)
                assert_authorization_boundary(
                    current_authz,
                    authorization_boundary,
                    current_scope_fingerprint=current_query_scope_fingerprint or "",
                )
            validate_authorized_metric(plan.authorized_metric, scope="platform", scope_data=scope_data)
            scope_data.update({
                "authorization_fingerprint": current_authz.authz_digest,
                "permission_version": current_authz.permission_version,
                "policy_set_version": current_authz.policy_set_version,
                "query_scope_fingerprint": current_query_scope_fingerprint,
            })
            _capture_authorization_snapshot(
                request_id=request_id, plan=plan, decision="allowed",
                reason_code="authorization_checks_passed", authorization_checked=True,
                authz=current_authz, boundary=authorization_boundary, scope_data=scope_data,
            )
        except Exception as exc:
            from app.auth.context import AuthorizationDenied
            decision = "denied" if isinstance(exc, (AuthorizationDenied, PermissionError)) else "unknown"
            code = str(getattr(exc, "code", "")) or (
                "authorized_metric_invalid" if isinstance(exc, ValueError) else "authorization_context_error"
            )
            _capture_authorization_snapshot(
                request_id=request_id, plan=plan, decision=decision,
                reason_code=code, authorization_checked=context is not None,
                authz=locals().get("current_authz"), boundary=authorization_boundary,
                scope_data=scope_data,
            )
            return {
                "task_type": task_type, "question": question, "server": SERVER_NAME,
                "available_tools": get_supported_tools(),
                "tool_name": None, "arguments": {}, "data": [],
                "request_id": request_id, "idempotency_key": idempotency_key,
                "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason=str(exc)).model_dump(),
            }
    # Formal B/G smoke may provide one frozen task-relevant operation menu.
    # This is an extra constraint after the normal authorization checks above.
    from app.core.execution_context import active_state
    evaluation_state = active_state() or {}
    evaluation_surface = evaluation_state.get("evaluation_tool_surface") if isinstance(evaluation_state, dict) else None
    if isinstance(evaluation_surface, dict):
        from evaluation.resume_bg_tool_surface import validate_planned_operation
        decomposition_request = (
            evaluation_state.get("registered_decomposition_request")
            if isinstance(evaluation_state, dict) else None
        )
        decomposition_is_active = (
            isinstance(decomposition_request, dict)
            and decomposition_request.get("status") == "ready"
        )
        surface_mismatch = validate_planned_operation(
            evaluation_surface, metric_id=plan.metric_name,
            query_type=plan.query_type, dimension=plan.dimension,
            dimensions=list(plan.dimensions),
            operation_intent=(
                "registered_numeric_decomposition" if decomposition_is_active else None
            ),
            registered_decomposition=(decomposition_request if decomposition_is_active else None),
        )
        if surface_mismatch:
            evaluation_state["terminal_code"] = "CAPABILITY_GAP"
            result = DataQueryAgentResult(
                status="unsupported", parsed=plan,
                reason="计划超出本次冻结的共同 B/G 工具面；已在数据执行前停止。",
            ).model_dump()
            return {
                "task_type": task_type, "question": question, "server": "evaluation-tool-surface-guard",
                "available_tools": list(evaluation_surface.get("allowed_operation_names") or []),
                "tool_name": None, "arguments": {}, "data": [],
                "request_id": request_id, "idempotency_key": idempotency_key,
                "evaluation_tool_surface_sha256": evaluation_surface.get("surface_sha256"),
                "result": result,
            }
    contribution_route: tuple[str, dict[str, object]] | None = None
    if plan.query_type == "metric_contribution":
        # Validate the contribution shape before any natural-language date
        # resolution. Invalid/mixed plans must not consult time-bound metadata
        # or reach Registry/MCP data handlers.
        tool_name, contribution_arguments, rejection = _platform_contribution_route(plan)
        if rejection:
            return {
                "task_type": task_type,
                "question": question,
                "server": "metric-registry-planner",
                "available_tools": get_supported_tools(),
                "tool_name": None,
                "arguments": {},
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "data": [],
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan, reason=rejection,
                ).model_dump(),
            }
        assert tool_name is not None and contribution_arguments is not None
        contribution_route = (tool_name, contribution_arguments)
    # Comparison/contribution questions commonly mention the current date
    # before the baseline date (for example ``2018-08-20 与 2018-08-19``).
    # Their typed plan carries separate left/right or current/baseline fields;
    # the generic range parser must not reject that valid descending order.
    # Typed period/comparison plans are already resolved by the server-side
    # planner.  Avoid an unrelated global data-bounds lookup for natural text
    # such as "compare two monthly periods" when the plan carries exact dates.
    has_explicit_comparison_periods = plan.query_type == "metric_contribution" or (
        plan.query_type == "metric_compare"
        and bool(
            (plan.current_date and plan.baseline_date)
            or (plan.left_date and plan.right_date)
            or (plan.current_start and plan.current_end and plan.baseline_start and plan.baseline_end)
        )
    )
    if has_explicit_comparison_periods:
        resolved_time = None
    else:
        try:
            resolved_time = resolve_expression(question)
        except ValueError:
            resolved_time = None

    # A seller request never crosses the stdio MCP process boundary because that
    # boundary has no authenticated per-request context. The trusted ContextVar
    # was established by the API/workflow and repositories set PostgreSQL RLS
    # transaction variables before any query. Client arguments cannot select a
    # different tenant.
    tenant_context = current_tenant_context()
    if tenant_context is not None and tenant_context.scope_type == "seller":
        if plan.dimensions:
            return {
                "task_type": task_type, "question": question, "server": "tenant-mcp",
                "available_tools": ["query_seller_metric", "query_seller_metric_trend"],
                "tool_name": None, "arguments": {}, "data": [], "request_id": request_id,
                "idempotency_key": idempotency_key,
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan,
                    reason="多维交叉查询当前只支持平台数据范围；店铺范围不会回退到全局查询。",
                ).model_dump(),
            }
        if plan.metric_name not in SELLER_METRICS:
            return {
                "task_type": task_type, "question": question, "server": "tenant-mcp",
                "available_tools": ["query_seller_metric", "query_seller_metric_trend"],
                "tool_name": None, "arguments": {}, "data": [], "request_id": request_id,
                "idempotency_key": idempotency_key,
                "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason="Metric or dimension is not valid for seller scope.").model_dump(),
            }
        if plan.query_type == "metric_value" and plan.metric_date:
            payload = query_seller_metric(tenant_context, plan.metric_name, plan.metric_date)
            tool_name = "query_seller_metric"
            arguments = {"metric_name": plan.metric_name, "metric_date": plan.metric_date}
        elif plan.query_type == "metric_trend" and plan.start_date and plan.end_date:
            payload = query_seller_trend(tenant_context, plan.metric_name, plan.start_date, plan.end_date)
            tool_name = "query_seller_metric_trend"
            arguments = {"metric_name": plan.metric_name, "start_date": plan.start_date, "end_date": plan.end_date}
        elif plan.query_type == "metric_statistics" and plan.start_date and plan.end_date:
            from app.metrics.statistics import summarize_points

            trend_payload = query_seller_trend(tenant_context, plan.metric_name, plan.start_date, plan.end_date)
            payload = {
                **trend_payload,
                **summarize_points(trend_payload.get("points", []), requested=plan.statistics),
            }
            tool_name = "query_seller_metric_statistics"
            arguments = {
                "metric_name": plan.metric_name,
                "start_date": plan.start_date,
                "end_date": plan.end_date,
                "statistics": list(plan.statistics),
            }
        else:
            return {
                "task_type": task_type, "question": question, "server": "tenant-mcp",
                "available_tools": ["query_seller_metric", "query_seller_metric_trend", "query_seller_metric_statistics"],
                "tool_name": None, "arguments": {}, "data": [], "request_id": request_id,
                "idempotency_key": idempotency_key,
                "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason="Seller scope supports metric value and trend only.").model_dump(),
            }
        payload["status"] = "ok"
        if plan.query_type == "metric_value":
            payload["metric_date"] = plan.metric_date
        else:
            payload["start_date"], payload["end_date"], payload["granularity"] = plan.start_date, plan.end_date, "day"
        from app.auth.context import build_authorization_context
        from app.security.result_policy import apply_result_policy
        seller_authz = build_authorization_context(
            tenant_context,
            allowed_metric_ids=[plan.metric_name],
            metric_scope_type="seller",
        )
        payload = apply_result_policy(payload, authorization_context=seller_authz).payload
        agent_result = _result_from_payload(plan, payload)
        return {"task_type": task_type, "question": question, "server": "tenant-mcp", "available_tools": ["query_seller_metric", "query_seller_metric_trend", "query_seller_metric_statistics"], "tool_name": tool_name, "arguments": arguments, "request_id": request_id or str(uuid4()), "idempotency_key": idempotency_key or str(uuid4()), "data": [], "result": agent_result.model_dump()}

    # 条件 3：如果计划里没有 tool_name，说明当前问题还不能映射到任何已支持的 MCP 工具。
    # 这通常意味着：
    # - 没识别出指标名
    # - 没识别出日期
    # - 问题类型不在支持范围内
    tenant_context = current_tenant_context()
    # Registry-backed platform point queries do not need a parallel legacy
    # query_<metric> MCP name. Registry.execute still checks the actual scoped
    # capability and installed handler before executing anything.
    platform_registry_value = is_platform_registry_value_plan(plan, tenant_context)
    if plan.tool_name is None and not platform_registry_value and contribution_route is None:
        if plan.dimensions:
            reason = "多维交叉查询需要明确的日期或日期范围。"
        elif plan.metric_name and plan.query_type == "metric_value" and not plan.metric_date:
            reason = "请提供明确日期，例如“查询 2018-08-10 的 GMV”。当前不能把“某一天”自动替换为任意日期。"
            if "商户" in question or "卖家" in question:
                reason += " 查询指定商户还需先进入该商户的数据范围；当前不支持在问题中任选商户。"
        elif plan.metric_name is None:
            reason = "未识别出可查询的指标。可查询 GMV、订单量、客单价、复购率等。"
        else:
            reason = "当前问题缺少执行查询所需的日期、对比范围或拆解维度。"
        # 同样返回一个统一结构，但状态标记为 unsupported。
        unsupported = DataQueryAgentResult(
            status="unsupported",
            parsed=plan,
            # 说明为什么不能继续执行：没有匹配到支持的工具。
            reason=reason,
        )
        return {
            "task_type": task_type,
            "question": question,
            "server": SERVER_NAME,
            "available_tools": get_supported_tools(),
            "tool_name": None,
            "arguments": {},
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "data": [],
            "result": unsupported.model_dump(),
        }

    # 条件 4：走到这里，说明任务类型正确，而且查询计划也成功映射到了具体工具。
    # 接下来先把 plan 转成该工具所需的参数结构。
    arguments = (
        contribution_route[1]
        if contribution_route is not None
        else _build_arguments_from_plan(plan)
    )

    # Point-pair and period contributions have dedicated MCP contracts. Never
    # route either through the Registry's single-day breakdown operation.
    if contribution_route is not None:
        tool_name, arguments = contribution_route
        ratio_tools = {
            "compare_freight_rate_ratio_facts_by_dimension_period": "freight_rate",
            "compare_avg_order_value_ratio_facts_by_dimension_period": "avg_order_value",
        }
        if tool_name in ratio_tools:
            expected_metric = ratio_tools[tool_name]
            if plan.metric_name != expected_metric or plan.dimension != "month_position":
                return {
                    "task_type": task_type, "question": question, "server": SERVER_NAME,
                    "available_tools": get_supported_tools(),
                    "tool_name": None, "arguments": {}, "data": [],
                    "request_id": request_id, "idempotency_key": idempotency_key,
                    "result": DataQueryAgentResult(
                        status="unsupported", parsed=plan,
                        reason=f"Registered ratio-facts execution is restricted to {expected_metric} by month_position.",
                    ).model_dump(),
                }
            try:
                context = current_tenant_context()
                if context is None:
                    raise ValueError("trusted authorization context is required")
                from app.auth.context import build_authorization_context

                active_authz = build_authorization_context(
                    context,
                    allowed_metric_ids=[expected_metric],
                    allowed_dimensions=["month_position"],
                    metric_scope_type="platform",
                )
                active_authz.require_action("analyze")
                active_authz.require_metrics([expected_metric])
                active_authz.require_dimensions(["month_position"])
                active_authz.require_tool(tool_name)
                from app.metrics.registry import get_registry
                definition = get_registry().get(expected_metric)
                capability = definition.attribution.for_scope("platform")
                aggregation = definition.period_aggregation
                if (
                    "symmetric_mix_within_v1" not in capability.method_ids
                    or "month_position" not in capability.dimensions
                    or aggregation is None
                    or (aggregation.numerator_metric, aggregation.denominator_metric) not in {
                        ("freight_value", "gmv"), ("gmv", "order_count"),
                    }
                ):
                    raise ValueError("ratio attribution capability is not registered for this scope")
            except Exception as exc:
                return {
                    "task_type": task_type, "question": question, "server": SERVER_NAME,
                    "available_tools": get_supported_tools(),
                    "tool_name": None, "arguments": {}, "data": [],
                    "request_id": request_id, "idempotency_key": idempotency_key,
                    "result": DataQueryAgentResult(status="unsupported", parsed=plan, reason=str(exc)).model_dump(),
                }
            arguments.pop("limit", None)
        settings = get_settings()
        server_name = "remote-mcp" if settings.mcp_transport == "remote_http" else SERVER_NAME
        try:
            if settings.mcp_transport == "remote_http":
                context = current_tenant_context()
                ensure_tool_allowed(
                    "data_query", tool_name, tenant_id=tenant_id, role=role,
                    scope_type=context.scope_type if context else None,
                )
                payload = call_remote_mcp_tool(
                    tool_name, arguments, request_id=request_id,
                    idempotency_key=idempotency_key,
                )
            else:
                payload = call_local_mcp_tool(
                    tool_name, arguments, tenant_id=tenant_id, role=role,
                    request_id=request_id,
                )
        except Exception as exc:
            # Contribution tools bypass the Registry/fallback block below.
            # Return the same typed failure contract instead of letting the
            # diagnostic executor reduce the exception to an error string.
            failure_context = classify_query_failure(
                exc, request_id=request_id,
                default_service="remote_mcp" if settings.mcp_transport == "remote_http" else "mcp",
            )
            from app.core.failure_context import public_failure_message

            failed_status = (
                "unsupported" if failure_context.category in {"capability_gap", "authorization_denied"}
                else "execution_failed"
            )
            return {
                "task_type": task_type, "question": question,
                "server": server_name, "available_tools": get_supported_tools(),
                "tool_name": tool_name, "arguments": arguments,
                "request_id": request_id, "idempotency_key": idempotency_key,
                "data": [], "result": DataQueryAgentResult(
                    status=failed_status, parsed=plan,
                    reason=public_failure_message(failure_context),
                    failure_context=failure_context,
                ).model_dump(),
            }
        return {
            "task_type": task_type, "question": question,
            "server": server_name, "available_tools": get_supported_tools(),
            "tool_name": tool_name, "arguments": arguments,
            "request_id": request_id, "idempotency_key": idempotency_key,
            "data": [], "result": _result_from_payload(plan, payload).model_dump(),
        }

    # Defensive fail-closed guard: a contribution plan must never become a
    # generic Registry breakdown or legacy fallback query.
    if plan.query_type == "metric_contribution":
        return {
            "task_type": task_type, "question": question, "server": "metric-registry-planner",
            "available_tools": get_supported_tools(), "tool_name": None,
            "arguments": {}, "request_id": request_id,
            "idempotency_key": idempotency_key, "data": [],
            "result": DataQueryAgentResult(
                status="unsupported", parsed=plan,
                reason="No validated contribution dispatch route is available.",
            ).model_dump(),
        }

    if plan.query_type == "metric_breakdown" and plan.dimension == "seller":
        return {
            "task_type": task_type, "question": question, "server": "metric-registry-planner",
            "available_tools": get_supported_tools(), "tool_name": None,
            "arguments": {}, "request_id": request_id,
            "idempotency_key": idempotency_key or request_id, "data": [],
            "result": DataQueryAgentResult(
                status="unsupported", parsed=plan,
                reason="Seller breakdown requires a server-resolved seller_scope_intent and the controlled scope path.",
            ).model_dump(),
        }

    if plan.dimensions:
        if (
            plan.query_type != "metric_breakdown"
            or plan.tool_name != "query_metric_by_dimensions"
            or not plan.start_date
            or not plan.end_date
        ):
            return {
                "task_type": task_type, "question": question,
                "server": "metric-registry-planner",
                "available_tools": get_supported_tools(), "tool_name": None,
                "arguments": {}, "request_id": request_id,
                "idempotency_key": idempotency_key, "data": [],
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan,
                    reason="多维交叉查询需要明确的指标和起止日期。",
                ).model_dump(),
            }
        context = current_tenant_context()
        if context is None or context.scope_type != "platform":
            return {
                "task_type": task_type, "question": question,
                "server": "metric-registry-planner",
                "available_tools": get_supported_tools(), "tool_name": None,
                "arguments": {}, "request_id": request_id,
                "idempotency_key": idempotency_key, "data": [],
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan,
                    reason="多维交叉查询当前只支持已授权的平台数据范围。",
                ).model_dump(),
            }
        maximum_dimensions = int(get_settings().result_policy_max_derived_dimensions)
        if len(plan.dimensions) > maximum_dimensions:
            return {
                "task_type": task_type, "question": question,
                "server": "metric-registry-planner",
                "available_tools": get_supported_tools(), "tool_name": None,
                "arguments": {}, "request_id": request_id,
                "idempotency_key": idempotency_key, "data": [],
                "result": DataQueryAgentResult(
                    status="unsupported", parsed=plan,
                    reason=(
                        "交叉维度数量超过当前结果隐私策略允许的上限"
                        f"（{maximum_dimensions}）。"
                    ),
                ).model_dump(),
            }
        tool_name = "query_metric_by_dimensions"
        arguments = _build_arguments_from_plan(plan)
        settings = get_settings()
        server_name = "remote-mcp" if settings.mcp_transport == "remote_http" else SERVER_NAME
        try:
            if settings.mcp_transport == "remote_http":
                ensure_tool_allowed(
                    "data_query", tool_name, tenant_id=tenant_id, role=role,
                    scope_type=context.scope_type,
                )
                payload = call_remote_mcp_tool(
                    tool_name, arguments, request_id=request_id,
                    idempotency_key=idempotency_key,
                )
            else:
                payload = call_local_mcp_tool(
                    tool_name, arguments, tenant_id=tenant_id, role=role,
                    request_id=request_id,
                )
            payload = _govern_multi_dimension_result(
                payload, authorization_context=current_authz,
            )
        except Exception as exc:
            failure_context = classify_query_failure(
                exc, request_id=request_id,
                default_service="remote_mcp" if settings.mcp_transport == "remote_http" else "mcp",
            )
            from app.core.failure_context import public_failure_message

            failed_status = (
                "unsupported" if failure_context.category in {"capability_gap", "authorization_denied"}
                else "execution_failed"
            )
            return {
                "task_type": task_type, "question": question,
                "server": server_name, "available_tools": get_supported_tools(),
                "tool_name": tool_name, "arguments": arguments,
                "request_id": request_id, "idempotency_key": idempotency_key,
                "data": [], "result": DataQueryAgentResult(
                    status=failed_status, parsed=plan,
                    reason=public_failure_message(failure_context),
                    failure_context=failure_context,
                ).model_dump(),
            }
        return {
            "task_type": task_type, "question": question,
            "server": server_name, "available_tools": get_supported_tools(),
            "tool_name": tool_name, "arguments": arguments,
            "request_id": request_id, "idempotency_key": idempotency_key,
            "data": [], "result": _result_from_payload(plan, payload).model_dump(),
        }

    # ------ Registry dispatch (primary execution path) ------
    if plan.metric_name:
        try:
            from app.metrics.registry import get_registry
            import logging

            logger = logging.getLogger(__name__)
            reg = get_registry()

            op_map = {
                "metric_value": "value",
                "metric_period": "period",
                "metric_trend": "trend",
                "metric_compare": "compare",
                "metric_breakdown": "breakdown",
                "metric_statistics": "statistics",
            }
            operation = op_map.get(plan.query_type, "value")
            period_comparison = bool(
                plan.query_type == "metric_compare"
                and plan.current_start and plan.current_end
                and plan.baseline_start and plan.baseline_end
            )
            if period_comparison:
                implementation = reg.get_implementation(plan.metric_name, "platform")
                if "period" not in implementation.capabilities or "compare" not in implementation.capabilities:
                    raise ValueError("Metric does not support registered period comparison")
                current_period = reg.execute(
                    plan.metric_name, operation="period",
                    start_date=date.fromisoformat(plan.current_start),
                    end_date=date.fromisoformat(plan.current_end),
                )
                baseline_period = reg.execute(
                    plan.metric_name, operation="period",
                    start_date=date.fromisoformat(plan.baseline_start),
                    end_date=date.fromisoformat(plan.baseline_end),
                )
                current_value = current_period.get("value")
                baseline_value = baseline_period.get("value")
                delta_value = (
                    round(float(current_value) - float(baseline_value), 2)
                    if current_value is not None and baseline_value is not None else None
                )
                delta_ratio = (
                    round(delta_value * 100.0 / float(baseline_value), 2)
                    if delta_value is not None and baseline_value not in (None, 0) else None
                )
                registry_result = {
                    "left_value": baseline_value,
                    "right_value": current_value,
                    "delta_value": delta_value,
                    "delta_ratio": delta_ratio,
                    "source_table": current_period.get("source_table") or baseline_period.get("source_table", ""),
                    "status": "ok" if current_value is not None and baseline_value is not None else "not_found",
                }
            else:
                registry_result = reg.execute(
                    plan.metric_name,
                    operation=operation,
                    metric_date=(
                        date.fromisoformat(plan.metric_date)
                        if plan.metric_date else None
                    ),
                    start_date=(
                        date.fromisoformat(plan.start_date)
                        if plan.start_date else None
                    ),
                    end_date=(
                        date.fromisoformat(plan.end_date)
                        if plan.end_date else None
                    ),
                    left_date=(
                        date.fromisoformat(plan.baseline_date or plan.left_date)
                        if (plan.baseline_date or plan.left_date) else None
                    ),
                    right_date=(
                        date.fromisoformat(plan.current_date or plan.right_date)
                        if (plan.current_date or plan.right_date) else None
                    ),
                    granularity=plan.granularity or "day",
                    dimension=plan.dimension,
                    limit=plan.limit or 10,
                    # Statistics uses the same registered implementation.
                    **({"statistics": tuple(plan.statistics)} if plan.query_type == "metric_statistics" else {}),
                )

            # Units come from the registered definition for the same scope as
            # this registry execution. Do not infer them from values or names.
            metric_definition = reg.get(plan.metric_name)
            metric_unit = metric_definition.unit

            # Wrap registry result into unified DataQueryAgentResult
            payload = {
                "metric_name": plan.metric_name,
                "metric_date": (
                    f"{plan.start_date} 至 {plan.end_date}"
                    if plan.query_type == "metric_period" and plan.start_date and plan.end_date
                    else plan.metric_date or ""
                ),
                "value": registry_result.get("value"),
                "source_table": registry_result.get("source_table", ""),
                "status": registry_result.get("status", "ok"),
            }
            if plan.query_type in {"metric_value", "metric_period", "metric_trend"}:
                payload["unit"] = metric_unit
            if plan.query_type == "metric_period":
                payload["period_start_date"] = plan.start_date
                payload["period_end_date"] = plan.end_date
            if registry_result.get("source_lineage") is not None:
                payload["source_lineage"] = registry_result["source_lineage"]
            if plan.query_type in {"metric_value", "metric_period"}:
                result_payload = payload
            elif plan.query_type == "metric_trend":
                result_payload = {
                    **payload,
                    "start_date": plan.start_date or "",
                    "end_date": plan.end_date or "",
                    "granularity": plan.granularity or "day",
                    "points": registry_result.get("points", []),
                }
            elif plan.query_type in ("metric_compare", "metric_contribution"):
                current_date = plan.current_date or plan.right_date or plan.current_start or ""
                baseline_date = plan.baseline_date or plan.left_date or plan.baseline_start or ""
                current_value = registry_result.get("right_value")
                baseline_value = registry_result.get("left_value")
                result_payload = {
                    **payload,
                    "current_date": current_date,
                    "baseline_date": baseline_date,
                    "current_value": current_value,
                    "baseline_value": baseline_value,
                    # The workflow's legacy fields are kept current-oriented;
                    # the old repository left/right contract is converted here.
                    "left_date": current_date,
                    "left_end_date": plan.current_end if period_comparison else None,
                    "right_date": baseline_date,
                    "right_end_date": plan.baseline_end if period_comparison else None,
                    "left_value": current_value,
                    "right_value": baseline_value,
                    "delta_value": registry_result.get("delta_value"),
                    "delta_ratio": registry_result.get("delta_ratio"),
                }
            elif plan.query_type == "metric_breakdown":
                result_payload = {
                    **payload,
                    "dimension": plan.dimension or "",
                    "items": registry_result.get("items", []),
                }
            elif plan.query_type == "metric_statistics":
                result_payload = {
                    **payload,
                    "start_date": plan.start_date or "",
                    "end_date": plan.end_date or "",
                    "granularity": plan.granularity or "day",
                    "statistics": registry_result.get("statistics", {}),
                    "count": registry_result.get("count", 0),
                    "min_bucket": registry_result.get("min_bucket"),
                    "max_bucket": registry_result.get("max_bucket"),
                }
            else:
                result_payload = payload

            agent_result = _result_from_payload(plan, result_payload)
            invocation = McpToolInvocation(
                server="metric-registry",
                tool_name=f"registry_{operation}",
                arguments=arguments,
                result=agent_result.model_dump(),
            )
            return {
                "task_type": task_type, "question": question,
                "server": invocation.server,
                "available_tools": get_supported_tools(),
                "tool_name": invocation.tool_name,
                "arguments": invocation.arguments,
                "request_id": request_id,
                "idempotency_key": idempotency_key or request_id or str(uuid4()),
                "data": [],
                "time_resolution": resolved_time.as_dict() if resolved_time else None,
                "result": invocation.result,
            }
        except Exception as exc:
            from app.metrics.exceptions import (
                CapabilityNotSupported, DimensionNotAllowed, MetricNotFound, ScopeNotAllowed,
            )

            failure_context = classify_query_failure(
                exc, request_id=request_id, default_service="metric_registry",
            )
            if isinstance(exc, (CapabilityNotSupported, DimensionNotAllowed, MetricNotFound, ScopeNotAllowed)):
                # A registry capability refusal is a terminal product boundary,
                # not a reason to retry the same item through a legacy tool.
                return {
                    "task_type": task_type,
                    "question": question,
                    "server": "metric-registry",
                    "available_tools": get_supported_tools(),
                    "tool_name": f"registry_{operation}",
                    "arguments": arguments,
                    "request_id": request_id,
                    "idempotency_key": idempotency_key or request_id or str(uuid4()),
                    "data": [],
                    "result": DataQueryAgentResult(
                        status="unsupported", parsed=plan,
                        reason=(
                            "当前授权不允许执行该项查询。"
                            if failure_context.category == "authorization_denied"
                            else "注册的指标能力不支持该项查询。"
                        ),
                        failure_context=failure_context,
                    ).model_dump(),
                }
            known_execution_failure = (
                isinstance(exc, (TimeoutError, asyncio.TimeoutError, ExternalServiceError))
            )
            try:
                from sqlalchemy.exc import SQLAlchemyError
                known_execution_failure = known_execution_failure or isinstance(exc, SQLAlchemyError)
            except ImportError:  # pragma: no cover - SQLAlchemy is a runtime dependency
                pass
            if platform_registry_value or known_execution_failure:
                # A timeout/database failure is terminal for this exact query.
                # Falling back would repeat the same read through a second path
                # and can obscure the original failure.
                return {
                    "task_type": task_type,
                    "question": question,
                    "server": "metric-registry",
                    "available_tools": get_supported_tools(),
                    "tool_name": f"registry_{operation}",
                    "arguments": arguments,
                    "request_id": request_id,
                    "idempotency_key": idempotency_key,
                    "data": [],
                    "result": DataQueryAgentResult(
                        status="execution_failed", parsed=plan,
                        reason=(
                            "数据工具查询超时，未取得该项结果。"
                            if failure_context.category == "tool_timeout"
                            else "数据库查询执行失败，未取得该项结果。"
                            if failure_context.category == "database_error"
                            else "数据查询执行失败，未取得该项结果。"
                        ),
                        failure_context=failure_context,
                    ).model_dump(),
                }
            # Registry is the primary path; log the bypass for observability.
            logger.warning(
                "Registry dispatch failed for %s/%s, falling back to MCP: %s",
                plan.metric_name, plan.query_type, exc,
            )

    # 真正通过 MCP 协议调用本地工具（Registry 不可用时的回退路径）。
    # 这里不是直接 import tool 函数执行，而是通过 stdio client 与 MCP server 通信。
    request_id = request_id or str(uuid4())
    idempotency_key = idempotency_key or str(uuid4())
    settings = get_settings()
    try:
        if settings.mcp_transport == "remote_http":
            context = current_tenant_context()
            ensure_tool_allowed(
                "data_query",
                plan.tool_name,
                tenant_id=tenant_id,
                role=role,
                scope_type=context.scope_type if context else None,
            )
            payload = call_remote_mcp_tool(plan.tool_name, arguments, request_id=request_id, idempotency_key=idempotency_key)
            server_name = "remote-mcp"
        else:
            payload = call_local_mcp_tool(
                plan.tool_name, arguments, tenant_id=tenant_id, role=role,
                request_id=request_id,
            )
            server_name = SERVER_NAME
    except Exception as exc:
        failure_context = classify_query_failure(
            exc,
            request_id=request_id,
            default_service="remote_mcp" if settings.mcp_transport == "remote_http" else "mcp",
        )
        from app.core.failure_context import public_failure_message
        failed_result = DataQueryAgentResult(
            status="execution_failed",
            parsed=plan,
            reason=public_failure_message(failure_context),
            failure_context=failure_context,
        )
        return {
            "task_type": task_type,
            "question": question,
            "server": "remote-mcp" if settings.mcp_transport == "remote_http" else SERVER_NAME,
            "available_tools": get_supported_tools(),
            "tool_name": plan.tool_name,
            "arguments": arguments,
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "data": [],
            "result": failed_result.model_dump(),
        }

    # 把不同工具返回的 payload 装回统一的 DataQueryAgentResult。
    # 这样上游和下游都可以只消费一种结果结构。
    agent_result = _result_from_payload(
        plan,
        payload,
        legacy_compare_result=(
            plan.query_type == "metric_compare"
            and plan.tool_name == "compare_metric"
        ),
    )

    # 记录一次完整的工具调用，包括：
    # - 连接的 server
    # - 调用的 tool 名称
    # - 传入的 arguments
    # - 工具执行后的 result
    invocation = McpToolInvocation(
        server=server_name,
        tool_name=plan.tool_name,
        arguments=arguments,
        result=agent_result.model_dump(),
    )

    # 最终返回的是适合写入 trace 的统一字典结构。
    # 其中 available_tools 便于调试时查看当前 server 支持哪些工具。
    return {
        "task_type": task_type,
        "question": question,
        "server": invocation.server,
        "available_tools": get_supported_tools(),
        "tool_name": invocation.tool_name,
        "arguments": invocation.arguments,
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "data": [],
        "time_resolution": resolved_time.as_dict() if resolved_time else None,
        "result": invocation.result,
    }


def query_business_data(
    task_type: str,
    question: str,
    *,
    tenant_id: str | None = None,
    role: str | None = None,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    plan: MetricQueryPlan | None = None,
    authorization_boundary=None,
    current_query_scope_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Run the existing query gateway and attach its request-bound auth decision."""
    requested_id = request_id or str(uuid4())
    capture: dict[str, Any] = {}
    token = _ACTIVE_AUTHORIZATION_CAPTURE.set(capture)
    try:
        response = _query_business_data_impl(
            task_type, question, tenant_id=tenant_id, role=role,
            request_id=requested_id, idempotency_key=idempotency_key,
            plan=plan, authorization_boundary=authorization_boundary,
            current_query_scope_fingerprint=current_query_scope_fingerprint,
        )
    finally:
        _ACTIVE_AUTHORIZATION_CAPTURE.reset(token)
    if not isinstance(response, dict):
        return response
    # The ID passed into the gateway is the authorization decision's primary
    # key. Preserve a distinct returned ID if an older tool adapter emits one.
    response_id = str(response.get("request_id") or requested_id)
    snapshot = dict(capture) if capture else {
        "schema_version": "per-tool-authorization-snapshot-v1",
        "request_id": requested_id,
        "decision": "denied" if plan is not None and plan.authorized_metric is None else "unknown",
        "decision_source": "required_authorization_missing" if plan is not None and plan.authorized_metric is None else "authorization_not_evaluated",
        "reason_code": "missing_authorized_metric" if plan is not None and plan.authorized_metric is None else "authorization_not_evaluated",
        "authorization_checked": False,
        "metric_id": plan.metric_name if plan else None,
        "authorized_metric_id": None,
        "authorized_metric": None,
        "scope": "platform",
        "operation": plan.query_type if plan else None,
        "dimension": plan.dimension if plan else None,
        "period": {},
        "identity_sha256": None,
    }
    snapshot["request_id"] = requested_id
    snapshot["response_request_id"] = response_id
    snapshot["tool"] = response.get("tool_name")
    snapshot["tool_arguments"] = response.get("arguments") if isinstance(response.get("arguments"), dict) else {}
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    snapshot["tool_result_status"] = result.get("status")
    snapshot["execution_attempted"] = response.get("tool_name") is not None
    response["authorization_snapshot"] = snapshot
    return response
