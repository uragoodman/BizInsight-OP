"""Data Query 节点 — MCP 工具调用。

V1 阶段内部复用现有 MCP client（query_business_data），
Target 阶段将升级为独立 LLM Agent（显式选择工具与参数）。

MECE 写入字段：tool_result, tool_calls, current_agent, node_history
"""

from app.core.time_expression import data_time_bounds
from app.graph.state import AgentState
from app.agents.contracts import EvidenceBundle, ToolInvocation, stable_evidence_id
from app.core.config import get_settings
from app.core.exceptions import sanitize_exception
from app.observability import record_event
from app.security import ensure_safe_untrusted_content
from time import perf_counter
from app.mcp.client import query_business_data, query_scoped_business_data
from app.agents.data_query_agent import build_query_plan, plan_confirmed_metric_queries
from app.graph.nodes.premise import has_directional_premise, premise_query_hint
from app.mcp.schemas import MetricMention, MetricPlanningDecision, MetricQueryBatchPlan, SemanticResolution
from app.metrics.authorization import AuthorizedMetric
from app.metrics.semantic_mapping_repository import create_pending_mappings, load_confirmed_mappings, load_skipped_mappings
from app.metrics.registry import get_registry
from app.context.orchestrator import ContextOrchestrator
from app.llm.routing import model_for
from app.tenancy import TenantContext, current_tenant_context
from app.auth.context import build_authorization_context
from app.auth.enforcement import AuthorizationBoundary, assert_authorization_boundary
from app.scope.canonical import CanonicalScope
from app.schemas.explorer import ControlledAnalysisScope
from app.scope.seller_scope import resolve_seller_scope, verify_seller_scope_payload
from uuid import uuid4


def _contribution_tool_result_mismatch(contract: dict, result: object) -> str | None:
    """Check the actual gateway envelope against the sealed request contract."""
    if not isinstance(result, dict):
        return "contract_violation_tool_result_shape"
    body = result.get("result") if isinstance(result.get("result"), dict) else {}
    if body.get("status") != "ok":
        # A capability/data refusal is a valid fail-closed outcome; it is not a
        # successful substitute query. Its envelope remains auditable.
        return None
    arguments = result.get("arguments") if isinstance(result.get("arguments"), dict) else {}
    if arguments.get("dimension") != contract.get("dimension"):
        return "contract_violation_tool_dimension"
    if arguments.get("metric_name") != contract.get("metric_id"):
        return "contract_violation_tool_metric"
    comparison = contract.get("comparison") or {}
    if comparison.get("kind") == "day_pair":
        if arguments.get("baseline_date") != comparison.get("baseline_date"):
            return "contract_violation_tool_baseline_date"
        if arguments.get("current_date") != comparison.get("current_date"):
            return "contract_violation_tool_current_date"
    else:
        baseline = comparison.get("baseline") or {}
        current = comparison.get("current") or {}
        expected = {
            "baseline_start": baseline.get("start_date"),
            "baseline_end": baseline.get("end_date"),
            "current_start": current.get("start_date"),
            "current_end": current.get("end_date"),
        }
        if any(arguments.get(key) != value for key, value in expected.items()):
            return "contract_violation_tool_period"
    return None


def _seller_scope_planning_context(payload: dict) -> tuple[str, dict, TenantContext]:
    """Build planner constraints from a verified seller scope and live identity.

    Seller IDs and the signed token stay out of planner input. The planner only
    receives the server-authorized metric/dimension/date envelope; the MCP
    handler verifies and resolves the signed payload again before SQL.
    """

    tenant = current_tenant_context()
    if tenant is None:
        raise ValueError("Trusted tenant context is required for seller scope planning")
    intent = verify_seller_scope_payload(payload, tenant=tenant)
    auth_scope = "seller" if tenant.scope_type == "seller" else "platform"
    authz = build_authorization_context(tenant, metric_scope_type=auth_scope)
    canonical = resolve_seller_scope(intent, tenant=tenant, authz=authz)
    scope_data = {
        "tenant_id": tenant.tenant_id,
        "user_id": tenant.user_id,
        "permissions": tenant.role,
        "scope_type": auth_scope,
        "seller_scope_type": canonical.seller_scope_type,
        "include_unattributed": canonical.include_unattributed,
        "start_date": canonical.start_date.isoformat(),
        "end_date": canonical.end_date.isoformat(),
        "default_start_date": canonical.start_date.isoformat(),
        "default_end_date": canonical.end_date.isoformat(),
        "granularity": canonical.granularity,
        "allowed_metrics": list(canonical.metric_ids),
        "allowed_dimensions": list(canonical.dimensions),
        "authorization_fingerprint": authz.authz_digest,
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authz_ref": authz.authz_ref,
        "query_scope_fingerprint": canonical.query_scope_fingerprint,
    }
    return auth_scope, scope_data, tenant


def _record_planning_capability_limit(state, decision, *, request_id, scope):
    """Audit a server-verified capability refusal without querying facts."""
    from app.tenancy import current_tenant_context
    from hashlib import sha256

    context = current_tenant_context()
    if (
        context is None
        or decision.reason_code not in {"unsupported_dimension", "period_contribution_not_registered"}
        or len(decision.confirmed_metrics) != 1
    ):
        return
    metric_id = decision.confirmed_metrics[0]
    authz = build_authorization_context(context, metric_scope_type=scope)
    authz.require_action("analyze")
    authz.require_metrics([metric_id])
    definition = get_registry().get(metric_id)
    implementation = definition.implementations.get(scope)
    dimension_missing = (
        implementation is None
        or not decision.dimension
        or decision.dimension not in implementation.dimensions
    )
    period_method_missing = False
    contract = (state.get("context_compilation") or {}).get("request_contract") or {}
    comparison = contract.get("comparison") or {} if isinstance(contract, dict) else {}
    if (
        decision.reason_code == "period_contribution_not_registered"
        and isinstance(contract, dict)
        and contract.get("operation_intent") == "dimensional_statistical_contribution"
        and isinstance(comparison, dict)
        and comparison.get("kind") == "period_pair"
    ):
        from app.analysis.contribution_capabilities import registered_period_contribution_method

        period_method_missing = registered_period_contribution_method(
            definition,
            metric_id=metric_id,
            dimension=str(decision.dimension or ""),
            scope=scope,
            granularity=str(comparison.get("granularity") or "") or None,
        ) is None
    if not dimension_missing and not period_method_missing:
        return
    # These dimensions have a derived query implementation; absence from the
    # static list alone is not proof of an unsupported capability.
    if dimension_missing and decision.dimension in {"month_position", "day_of_week", "hour_of_day", "quarter_of_year", "half_of_year"}:
        return
    state["terminal_code"] = "CAPABILITY_GAP"
    event_id = f"{request_id}:capability-limit"
    record = {
        "decision_id": event_id, "request_id": request_id,
        "limitation_code": "CAPABILITY_GAP", "metric_id": metric_id,
        "dimension": decision.dimension, "operation": decision.query_type,
        "scope": scope, "authorization_checked": True,
        "registry_capability_checked": True, "sql_executed": False,
        "question_sha256": sha256(str(state.get("user_message") or "").encode()).hexdigest(),
    }
    state.setdefault("decision_records", []).append(record)
    state.setdefault("control_events", []).append({
        "event_id": event_id,
        "event_type": "capability_preflight_stop",
        "request_id": request_id,
        "metric_id": metric_id,
        "dimension": decision.dimension,
        "operation": decision.query_type,
        "scope": scope,
        "authorization_checked": True,
        "registry_capability_checked": True,
        "sql_executed": False,
        "decision_record_id": event_id,
    })


def _validate_lifecycle_boundary(state: AgentState, controlled_scope: dict) -> dict[str, object]:
    """Build fresh trusted authorization inputs and validate any saved boundary."""

    raw_boundary = state.get("authorization_boundary") or {}
    from app.tenancy import current_tenant_context

    context = current_tenant_context()
    if context is None:
        if raw_boundary:
            raise ValueError("Trusted tenant context is required to validate the authorization boundary")
        # Legacy/unit callers without authenticated request context continue
        # through their existing guarded path. Production HTTP requests always
        # activate the immutable TenantContext before invoking the graph.
        return {}
    controlled = ControlledAnalysisScope.model_validate(controlled_scope)
    authz = build_authorization_context(
        context,
        allowed_metric_ids=controlled.allowed_metrics,
        allowed_dimensions=controlled.allowed_dimensions,
        metric_scope_type="seller",
    )
    canonical = CanonicalScope.resolve(
        authz,
        seller_ids=[controlled.seller_id],
        start_date=controlled.default_start_date,
        end_date=controlled.default_end_date,
        metric_ids=controlled.allowed_metrics,
        dimensions=controlled.allowed_dimensions,
        granularity=controlled.granularity,
    )
    if raw_boundary:
        boundary = AuthorizationBoundary(
            permission_version=str(raw_boundary.get("permission_version") or ""),
            policy_set_version=str(raw_boundary.get("policy_set_version") or ""),
            authorization_fingerprint=str(raw_boundary.get("authorization_fingerprint") or ""),
            query_scope_fingerprint=str(raw_boundary.get("query_scope_fingerprint") or ""),
        )
        assert_authorization_boundary(
            authz,
            boundary,
            current_scope_fingerprint=canonical.query_scope_fingerprint,
        )
    current = {
        "authorization_fingerprint": authz.authz_digest,
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authz_ref": authz.authz_ref,
        "query_scope_fingerprint": canonical.query_scope_fingerprint,
    }
    state["current_query_scope_fingerprint"] = canonical.query_scope_fingerprint
    return current


def _capture_seller_boundary(state: AgentState, result: dict) -> bool:
    """Accept only the seller-scope boundary returned by the controlled MCP handler."""
    boundary = result.get("seller_boundary")
    fingerprint = result.get("seller_scope_fingerprint")
    expected = {
        "permission_version", "policy_set_version", "authorization_fingerprint",
        "query_scope_fingerprint",
    }
    if not isinstance(boundary, dict) or set(boundary) != expected:
        return False
    if not all(isinstance(boundary.get(key), str) and boundary[key] for key in expected):
        return False
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    if boundary["query_scope_fingerprint"] != fingerprint:
        return False
    state["seller_authorization_boundary"] = dict(boundary)
    state["seller_scope_fingerprint"] = fingerprint
    return True


def _persist_semantic_candidates(state: AgentState, batch_plans, *, planner_scope: str) -> object:
    """Persist pending semantic candidates without making them executable."""
    decision = batch_plans.decision
    candidates = [candidate for candidate in decision.ambiguous_candidates if candidate.candidate_no is not None]
    if not candidates:
        return batch_plans
    resolution_id = f"resolution:{state.get('trace_id', '')}:{uuid4().hex[:12]}"
    mention_id = f"mention:{uuid4().hex}"
    scope_data = {
        **(state.get("controlled_analysis_scope") or {}),
        "tenant_id": state.get("tenant_id"),
        "user_id": state.get("user_id"),
        "permissions": state.get("user_role"),
    }
    try:
        context = TenantContext(
            tenant_id=str(state.get("tenant_id") or "platform"),
            user_id=str(state.get("user_id") or "anonymous"),
            role=state.get("user_role", "admin"),
            scope_type="seller" if planner_scope == "seller" else "platform",
            seller_id=(state.get("controlled_analysis_scope") or {}).get("seller_id") if planner_scope == "seller" else None,
        )
        rows = create_pending_mappings(
            context,
            resolution_id=resolution_id,
            mention_id=mention_id,
            candidates=[item.model_dump(mode="json") for item in candidates],
            scope=scope_data,
            catalog_digest=get_registry().catalog_digest,
            trace_id=str(state.get("trace_id") or ""),
            session_id=state.get("chat_session_id"),
            turn_id=state.get("chat_turn_id"),
        )
    except Exception as exc:
        # Database/migration outages are a safe clarification, never a reason
        # to execute a candidate or fall back to string replacement.
        state.setdefault("errors", []).append({
            "code": "semantic_mapping_persistence_failed",
            "message": "语义候选未能持久化，已停止查询。",
            "public_message": "语义候选暂时无法保存，请稍后重试。",
            "retryable": True,
            "agent_name": "data_query",
            "detail": type(exc).__name__,
        })
        return batch_plans
    mapping_by_candidate = {int(row["candidate_no"]): str(row["mapping_id"]) for row in rows}
    updated_candidates = [
        item.model_copy(update={"mapping_id": mapping_by_candidate.get(item.candidate_no)})
        for item in decision.ambiguous_candidates
    ]
    semantic = SemanticResolution(
        resolution_id=resolution_id,
        mentions=[MetricMention(
            mention_id=mention_id,
            raw_text=decision.semantic_interpretation.source_text if decision.semantic_interpretation else state.get("user_message", "")[:120],
            status="candidate_pending",
            mapping_ids=list(mapping_by_candidate.values()),
        )],
        catalog_digest=get_registry().catalog_digest,
        scope_fingerprint=__import__("app.metrics.authorization", fromlist=["scope_fingerprint"]).scope_fingerprint(scope_data),
        requires_confirmation=True,
    )
    updated = decision.model_copy(update={"ambiguous_candidates": updated_candidates, "semantic_resolution": semantic})
    state["semantic_mapping_ids"] = list(mapping_by_candidate.values())
    return batch_plans.model_copy(update={"decision": updated})


def _historical_query_hint(state: dict, question: str, pending_query: dict | None = None) -> str:
    """短追问时注入历史上下文，帮助 planner 识别上轮指标。

    不修改问题语义，仅对 <15 字的短追问追加只读历史提示。
    提示文本不包含 evidence_id，不影响后续 evidence 绑定。
    """
    if pending_query:
        return pending_query.get("question", question)
    result = question
    if len(question) < 15:
        historical_claims = state.get("historical_claims_context", [])
        historical_hypotheses = state.get("historical_hypothesis_context", {})
        hints = []
        if historical_claims:
            vals = [c.get("value", "")[:60] for c in historical_claims[:2] if c.get("value")]
            if vals:
                hints.append("上轮确认: " + "; ".join(vals))
        if historical_hypotheses:
            mids = list(historical_hypotheses.keys())[:2]
            if mids:
                hints.append("上轮分析指标: " + ", ".join(mids))
        if hints:
            result = f"{question}\n[历史上下文: {' | '.join(hints)}]"
    return result


def _followup_authorized_metrics(state: AgentState, question: str) -> list[AuthorizedMetric]:
    """Reuse a same-scope predecessor only when the compiler resolved it.

    The compiler's relation and slot provenance distinguish an elliptical
    continuation from an explicit metric switch. The lifecycle loader has
    already restricted the predecessor to the same scope fingerprint; this
    function additionally requires the selected metric to match a persisted,
    server-authorized metric that remains allowed in the current scope.
    """
    context = state.get("previous_turn_context") or {}
    compilation = state.get("context_compilation") or {}
    slot = (compilation.get("known_slots") or {}).get("metric") or {}
    metric_id = str(slot.get("value") or "")
    current_scope_fingerprint = str(state.get("current_query_scope_fingerprint") or "")
    previous_scope_fingerprint = str(context.get("scope_fingerprint") or "")
    if (
        not context
        or compilation.get("relation") != "continue"
        or compilation.get("needs_clarification")
        or slot.get("source") != "previous_turn_context"
        or not metric_id
        or not compilation.get("source_turn_id")
        or compilation.get("source_turn_id") != context.get("source_turn_id")
        or not current_scope_fingerprint
        or current_scope_fingerprint != previous_scope_fingerprint
    ):
        return []
    rows = context.get("authorized_metrics") or []
    result: list[AuthorizedMetric] = []
    allowed = set((state.get("controlled_analysis_scope") or {}).get("allowed_metrics") or [])
    for row in rows:
        try:
            metric = AuthorizedMetric.model_validate(row)
        except Exception:
            continue
        if metric.metric_id != metric_id or (allowed and metric.metric_id not in allowed):
            continue
        result.append(metric)
    return result


def data_query(state: AgentState) -> AgentState:
    """执行业务数据查询，调用 MCP 工具层获取指标数据。

    参数：
    - `state`：当前工作流状态，至少包含 `user_message`、`task_type`。

    返回值：
    - `AgentState`：写入 MCP 工具调用结果后的状态。
    """

    # 打点入口：本节点是业务数据边界。先读取任务类型、原始问题、挂起查询和
    # 工具预算，再决定是否允许调用 MCP；所有真正的工具调用都应在本节点留下 tool_calls。
    task_type = state.get("task_type", "metric_query")
    if state.get("seller_scope_payload") and task_type != "metric_query":
        from app.graph.nodes.seller_scope_guard import reject_unsupported_seller_scope

        return reject_unsupported_seller_scope(state)
    question = state["user_message"]
    compilation = state.get("context_compilation") or {}
    request_contract = compilation.get("request_contract") if isinstance(compilation, dict) else None
    contribution_contract = (
        request_contract if isinstance(request_contract, dict)
        and request_contract.get("operation_intent") == "dimensional_statistical_contribution"
        else None
    )
    from app.context.request_contract import (
        contribution_execution_reservation, contribution_execution_state_mismatch,
        contribution_plan_mismatch, record_contract_observation,
        record_contract_violation, validate_compiled_request_contract,
    )
    if isinstance(request_contract, dict):
        valid_contract, contract_reason = validate_compiled_request_contract(
            compilation,
            state_contract_hash=state.get("request_contract_hash"),
            require_state_anchor=True,
        )
        record_contract_observation(state, "data_query_entry", {
            "contract_valid": valid_contract,
            "validation_reason": contract_reason,
        })
        expected_clarification = bool(
            compilation.get("needs_clarification")
            and request_contract.get("parse_status") == "clarification_required"
            and contract_reason == request_contract.get("uncertainty_reason")
        )
        if not valid_contract and not expected_clarification:
            record_contract_violation(state, "data_query", contract_reason or "request_contract_invalid")
            state["query_results"] = []
            state["tool_result"] = {
                "server": "request-contract-guard", "tool_name": None,
                "arguments": {}, "request_id": str(state.get("trace_id") or "") + ":contract",
                "result": {"status": "unsupported", "reason_code": contract_reason,
                           "reason": "请求契约校验失败，已在数据执行前停止。", "parsed": {}},
            }
            state["current_agent"] = "data_query"
            return state
        if valid_contract and contribution_contract:
            contract_hash = str(contribution_contract.get("contract_hash") or "")
            prior = [
                item for item in (state.get("request_contract_query_reservations") or [])
                if isinstance(item, dict) and item.get("contract_hash") == contract_hash
            ]
            if prior:
                reason = "contract_violation_duplicate_contribution_execution"
                record_contract_violation(state, "data_query", reason)
                record_contract_observation(state, "duplicate_contribution_execution", {
                    "reservation_count": len(prior),
                    "execution_signatures": [item.get("execution_signature") for item in prior],
                })
                refusal_id = str(state.get("trace_id") or "") + ":duplicate-contribution"
                state["query_results"] = []
                state["tool_result"] = {
                    "server": "request-contract-guard", "tool_name": None,
                    "arguments": {}, "request_id": refusal_id,
                    "result": {"status": "unsupported", "reason_code": reason,
                               "reason": "同一请求契约的贡献查询已执行或已预留，不会重复查询。", "parsed": {}},
                }
                state["current_agent"] = "data_query"
                return state
    elif state.get("context_compilation") and state.get("context_compilation", {}).get("operation_intent", {}).get("kind") == "dimensional_statistical_contribution":
        record_contract_violation(state, "data_query", "request_contract_missing")
        state["query_results"] = []
        state["tool_result"] = {
            "server": "request-contract-guard", "tool_name": None,
            "arguments": {}, "request_id": str(state.get("trace_id") or "") + ":contract",
            "result": {"status": "unsupported", "reason_code": "request_contract_missing",
                       "reason": "贡献请求缺少编译契约，已停止。", "parsed": {}},
        }
        state["current_agent"] = "data_query"
        return state
    followup = compilation.get("result_followup") if isinstance(compilation, dict) else None
    if isinstance(followup, dict) and followup.get("kind"):
        from app.context.result_context import trusted_result_context

        result_context = trusted_result_context(
            state.get("previous_turn_context") or {}, state,
        )
        if result_context is None:
            reason = str(followup.get("unavailable_reason") or "previous_result_context_unavailable")
            decision = MetricPlanningDecision(
                status="clarification_required",
                reason="上一轮结果不在当前范围或已不可用，未执行新的数据查询。",
                reason_code="scope_rejected",
            )
            state["metric_planning_decision"] = decision.model_dump(mode="json")
            state["metric_query_batch_plan"] = {
                "decision": decision.model_dump(mode="json"), "plans": [],
            }
            state["query_results"] = []
            state["tool_result"] = {
                "server": "result-context", "tool_name": None, "arguments": {},
                "request_id": str(state.get("trace_id") or "") + ":result-context",
                "result": {
                    "status": "clarification_required", "reason_code": reason,
                    "reason": "上一轮结果不可复用，未执行新的数据查询。", "parsed": {},
                },
            }
            state["current_agent"] = "data_query"
            return state
        query_type = (
            str(result_context.get("query_type") or "metric_value")
            if result_context.get("result_status") in {"not_found", "insufficient_data"}
            else "metric_compare" if result_context.get("comparison")
            else "metric_breakdown" if result_context.get("rows")
            else "metric_value"
        )
        decision = MetricPlanningDecision(
            status="confirmed",
            confirmed_metrics=[str(result_context.get("metric_id") or "")],
            query_type=query_type,
            operation_intent="ordinary_metric_query",
            dimension=result_context.get("dimension"),
            reason="本轮是同一范围的结果追问，复用上一轮服务端结果快照；未执行新的业务查询。",
            reason_code="exact_match",
        )
        state["result_followup"] = {
            **followup,
            "source_turn_id": result_context.get("source_turn_id"),
            "source_trace_id": result_context.get("source_trace_id"),
        }
        state["result_followup_context"] = result_context
        state["metric_planning_decision"] = decision.model_dump(mode="json")
        state["metric_query_batch_plan"] = {
            "decision": decision.model_dump(mode="json"), "plans": [],
        }
        state["query_results"] = []
        state["tool_calls"] = []
        state["tool_result"] = {}
        state.setdefault("node_history", []).append({
            "node": "data_query",
            "summary": "Skipped business query: answered from same-scope result context.",
        })
        state["current_agent"] = "data_query"
        return state
    pending_query = state.get("pending_query")
    tool_limit = int(state.get("runtime_budget", {}).get("max_tool_calls", get_settings().max_tool_calls))
    # 预算检查必须发生在调用 MCP 之前，避免重试或异常恢复造成无限工具调用。
    if len(state.get("tool_calls", [])) >= tool_limit:
        state.setdefault("errors", []).append({
            "code": "tool_call_budget_exceeded", "message": "Tool call budget exceeded.",
            "retryable": False, "agent_name": "data_query",
        })
        return state

    # MCP 的数据工具只处理 metric_query。异常任务通常已含有指标和日期，
    # 因而可直接复用问题；报告若没有具体指标，则先取最近一天 GMV 作为草案
    # 的可追溯数据锚点。任务类型仍保留在 state 中，供后续条件边使用。
    compiled_question = str(
        (state.get("context_compilation") or {}).get("standalone_question") or question
    )
    query_question = pending_query["question"] if pending_query else compiled_question
    # 第一次异常分析请求先向指标注册表请求趋势窗口，尽量让正常路径只进行一次数据查询。
    # premise_check 节点仍保留旧计划或不支持计划下的兜底逻辑，因此这里不是唯一安全边界。
    if (
        task_type == "anomaly_analysis" and not pending_query
        and not contribution_contract and has_directional_premise(question)
    ):
        query_question = premise_query_hint(question) or query_question
    if task_type == "report_generation" and not pending_query:
        # 报告兜底查询必须使用与确定性时间解析器相同的数据边界。
        # daily_metrics 可能比 Explorer 的已交付订单数据覆盖更晚日期，否则生成的日期
        # 会被受控查询层拒绝，导致报告草稿无法得到可追溯的数据锚点。
        _, latest_date = data_time_bounds()
        query_question = f"查询 {latest_date.isoformat()} 的 GMV"

    # Phase 7: 短追问时注入历史上下文，帮助 planner 识别上轮指标
    query_question = _historical_query_hint(state, query_question, pending_query)

    # WP1: the planner receives only conversation/task/registry context.  Tool
    # evidence is intentionally absent here; it is refreshed after execution.
    data_query_context = ContextOrchestrator().build_specialist_bundle(
        state,
        "data_query",
        instruction=query_question,
        model=model_for("extract")[0],
    )

    started = perf_counter()
    request_id = str(state.get("trace_id", "")) + ":" + str(len(state.get("tool_calls", [])) + 1)
    # 受控范围完整性检查：用户启用了浏览锁定但 scope 未生成时，不允许退回全局 MCP
    query_mode = "unrestricted"
    if state.get("use_current_explorer_context") and not state.get("controlled_analysis_scope"):
        state.setdefault("errors", []).append({
            "code": "controlled_scope_missing",
            "message": "已启用数据浏览范围锁定，但受控范围未生成（resolve_explorer_context 失败）。请重新确认筛选条件。",
            "retryable": False,
            "agent_name": "data_query",
        })
        record_event("mcp.call", {
            "trace_id": state.get("trace_id"), "thread_id": state.get("thread_id"),
            "tool_name": None, "duration_ms": 0,
            "status": "skipped_controlled_scope_missing",
        })
        state["current_agent"] = "data_query"
        state.setdefault("node_history", []).append({
            "node": "data_query",
            "summary": "Skipped MCP call: controlled_scope_missing",
        })
        return state

    # Registry 驱动的规划器只有在 triage 已经把请求路由到 data_query 后才运行。
    # 它只确认指标语义并生成不可直接执行的计划；真正的处理器和可信 Scope 仍由下面的
    # query_scoped_business_data/query_business_data 选择。
    batch_plans = None
    if not pending_query:
        controlled_scope = state.get("controlled_analysis_scope")
        seller_planner_tenant = None
        if state.get("seller_scope_payload"):
            try:
                planner_scope, planner_scope_data, seller_planner_tenant = _seller_scope_planning_context(
                    state["seller_scope_payload"],
                )
            except (PermissionError, ValueError):
                from app.graph.nodes.seller_scope_guard import reject_unsupported_seller_scope

                return reject_unsupported_seller_scope(state)
            allowed = set(planner_scope_data["allowed_metrics"])
        else:
            planner_scope = "seller" if controlled_scope else "platform"
            configured_allowed = controlled_scope.get("allowed_metrics", []) if controlled_scope else []
            allowed = set(configured_allowed) if configured_allowed else None
            planner_scope_data = {
                **(controlled_scope or {}),
                "tenant_id": state.get("tenant_id"),
                "user_id": state.get("user_id"),
                "permissions": state.get("user_role"),
                "scope_type": planner_scope,
            }
        if controlled_scope and not seller_planner_tenant:
            current_authorization = _validate_lifecycle_boundary(state, controlled_scope)
            planner_scope_data.update(current_authorization)
        confirmed_mappings: list[dict] = []
        skipped_mappings: list[dict] = []
        if state.get("semantic_mapping_ids"):
            try:
                context = seller_planner_tenant or TenantContext(
                    tenant_id=str(state.get("tenant_id") or "platform"),
                    user_id=str(state.get("user_id") or "anonymous"),
                    role=state.get("user_role", "admin"),
                    scope_type="seller" if planner_scope == "seller" else "platform",
                    seller_id=(controlled_scope or {}).get("seller_id") if planner_scope == "seller" else None,
                )
                confirmed_mappings = load_confirmed_mappings(
                    context,
                    list(state.get("semantic_mapping_ids") or []),
                    scope=planner_scope_data,
                    catalog_digest=get_registry().catalog_digest,
                    session_id=state.get("chat_session_id"),
                )
            except Exception as exc:
                state.setdefault("errors", []).append({
                    "code": "semantic_mapping_validation_failed",
                    "message": "无法验证语义确认，已停止查询。",
                    "public_message": "语义确认暂时无法验证，请重新确认。",
                    "retryable": True,
                    "agent_name": "data_query",
                    "detail": type(exc).__name__,
                })
        if state.get("skipped_semantic_mapping_ids"):
            try:
                context = seller_planner_tenant or TenantContext(
                    tenant_id=str(state.get("tenant_id") or "platform"),
                    user_id=str(state.get("user_id") or "anonymous"),
                    role=state.get("user_role", "admin"),
                    scope_type="seller" if planner_scope == "seller" else "platform",
                    seller_id=(controlled_scope or {}).get("seller_id") if planner_scope == "seller" else None,
                )
                skipped_mappings = load_skipped_mappings(
                    context,
                    list(state.get("skipped_semantic_mapping_ids") or []),
                    scope=planner_scope_data,
                    catalog_digest=get_registry().catalog_digest,
                    session_id=state.get("chat_session_id"),
                )
            except Exception as exc:
                state.setdefault("errors", []).append({
                    "code": "semantic_skip_validation_failed",
                    "message": "无法验证用户跳过的语义候选，已停止查询。",
                    "public_message": "跳过选择暂时无法验证，请重新提交。",
                    "retryable": True,
                    "agent_name": "data_query",
                    "detail": type(exc).__name__,
                })
        batch_plans = plan_confirmed_metric_queries(
            query_question,
            scope=planner_scope,
            allowed_metrics=allowed,
            metric_confirmation=state.get("metric_confirmation"),
            semantic_mapping_ids=state.get("semantic_mapping_ids"),
            scope_data=planner_scope_data,
            default_metric=(controlled_scope or {}).get("metric_name") if controlled_scope else None,
            explorer_selection_id=state.get("explorer_selection_id"),
            confirmed_mappings=confirmed_mappings,
            followup_authorized_metrics=_followup_authorized_metrics(state, query_question),
            operation_intent=state.get("context_compilation", {}).get("operation_intent"),
            request_contract=request_contract,
            comparison_periods=(
                (state.get("context_compilation", {}).get("operation_intent") or {}).get("comparison_periods")
                if isinstance(state.get("context_compilation", {}).get("operation_intent"), dict)
                else None
            ),
            comparison_dates={
                role: str(slot.get("value"))
                for role, slot in (state.get("context_compilation", {}).get("known_slots") or {}).items()
                if role in {"baseline_date", "current_date"} and isinstance(slot, dict) and slot.get("value")
            },
            evaluation_tool_surface=state.get("evaluation_tool_surface"),
            context_items=ContextOrchestrator().packer_items(data_query_context),
        )
        if any(
            isinstance(error, dict) and error.get("code") == "token_budget_exceeded"
            for error in (state.get("errors") or [])
        ):
            # The planner response that exhausted the request budget is kept in
            # the LLM call ledger, but no SQL/tool execution follows it.
            state["metric_planning_decision"] = batch_plans.decision.model_dump(mode="json")
            state["metric_query_batch_plan"] = batch_plans.model_dump(mode="json")
            state["current_agent"] = "data_query"
            state.setdefault("node_history", []).append({
                "node": "data_query",
                "summary": "Stopped before database execution: LLM token budget exhausted.",
            })
            return state
        evaluation_surface = state.get("evaluation_tool_surface")
        if isinstance(evaluation_surface, dict) and batch_plans.plans:
            from evaluation.resume_bg_tool_surface import validate_planned_operation
            operation_intent = str(batch_plans.decision.operation_intent or "")
            mismatch = None
            registered_decomposition = batch_plans.decision.registered_decomposition
            if hasattr(registered_decomposition, "model_dump"):
                registered_decomposition = registered_decomposition.model_dump(mode="json")
            for planned in batch_plans.plans:
                mismatch = validate_planned_operation(
                    evaluation_surface,
                    metric_id=planned.metric_name,
                    query_type=planned.query_type,
                    dimension=planned.dimension,
                    dimensions=list(planned.dimensions),
                    operation_intent=operation_intent,
                    registered_decomposition=(
                        registered_decomposition
                        if isinstance(registered_decomposition, dict) else None
                    ),
                )
                if mismatch:
                    break
            if mismatch:
                state.setdefault("errors", []).append({
                    "code": "evaluation_tool_surface_violation",
                    "reason_code": mismatch,
                    "surface_sha256": evaluation_surface.get("surface_sha256"),
                    "agent_name": "data_query",
                    "retryable": False,
                })
                state["terminal_code"] = "CAPABILITY_GAP"
                state["metric_planning_decision"] = batch_plans.decision.model_dump(mode="json")
                state["metric_query_batch_plan"] = batch_plans.model_dump(mode="json")
                state["query_results"] = []
                state["tool_result"] = {
                    "server": "evaluation-tool-surface-guard", "tool_name": None,
                    "arguments": {}, "request_id": request_id,
                    "result": {"status": "unsupported", "reason_code": mismatch,
                               "reason": "计划超出冻结的评测工具面，已在查询前停止。", "parsed": {}},
                }
                state["current_agent"] = "data_query"
                return state
        if skipped_mappings:
            skipped_labels = []
            for row in skipped_mappings:
                metric_id = str(row.get("metric_id") or "")
                try:
                    label = get_registry().get(metric_id).display_name if metric_id else str(row.get("mention_id"))
                except Exception:
                    label = metric_id or str(row.get("mention_id"))
                skipped_labels.append(label)
            skipped_resolution = batch_plans.decision.semantic_resolution
            if skipped_resolution is None:
                skipped_resolution = SemanticResolution(
                    resolution_id=str(skipped_mappings[0].get("resolution_id") or "skipped-resolution"),
                    mentions=[], catalog_digest=get_registry().catalog_digest,
                    requires_confirmation=False,
                )
            skipped_mentions = [MetricMention(
                mention_id=str(row.get("mention_id") or row.get("mapping_id")),
                raw_text=str(row.get("metric_id") or row.get("mention_id") or ""),
                status="skipped_by_user", provenance="user_skip",
                metric_id=str(row.get("metric_id")) if row.get("metric_id") else None,
                mapping_ids=[str(row.get("mapping_id"))],
            ) for row in skipped_mappings]
            batch_plans = batch_plans.model_copy(update={
                "decision": batch_plans.decision.model_copy(update={
                    "skipped_metrics": skipped_labels,
                    "semantic_resolution": skipped_resolution.model_copy(update={"mentions": skipped_mentions}),
                }),
            })
        if batch_plans.decision.status != "confirmed":
            batch_plans = _persist_semantic_candidates(
                state, batch_plans, planner_scope=planner_scope,
            )
        state["metric_planning_decision"] = batch_plans.decision.model_dump(mode="json")
        state["metric_query_batch_plan"] = batch_plans.model_dump(mode="json")
        if contribution_contract:
            contract_mismatch = contribution_plan_mismatch(
                contribution_contract,
                batch_plans.decision,
                [plan.model_dump(mode="json") for plan in batch_plans.plans],
            )
            record_contract_observation(state, "planner", {
                "planner_operation_intent": batch_plans.decision.operation_intent,
                "query_type": batch_plans.decision.query_type,
                "dimension": batch_plans.decision.dimension,
                "plan_count": len(batch_plans.plans),
                "plan_contract_mismatch": contract_mismatch,
            })
            state["metric_planning_request_contract_hash"] = contribution_contract.get("contract_hash")
            if contract_mismatch:
                record_contract_violation(state, "planner", contract_mismatch)
                state["query_results"] = []
                state["tool_result"] = {
                    "server": "request-contract-guard", "tool_name": None,
                    "arguments": {}, "request_id": request_id,
                    "result": {"status": "unsupported", "reason_code": contract_mismatch,
                               "reason": "规划结果偏离明确的贡献请求，已在执行前停止。", "parsed": {}},
                }
                state["current_agent"] = "data_query"
                return state
        state["authorized_metrics"] = [
            plan.authorized_metric.model_dump(mode="json")
            for plan in batch_plans.plans
            if plan.authorized_metric is not None
        ]
        registered_decomposition = batch_plans.decision.registered_decomposition
        if (
            batch_plans.decision.operation_intent == "registered_numeric_decomposition"
            and registered_decomposition is not None
            and registered_decomposition.status == "ready"
            and batch_plans.decision.status == "confirmed"
        ):
            # The Registry method may require registered dependency metrics
            # that were not named as user goals. Create their server-side
            # relation authorizations only after the planner checked the
            # active scope and Registry implementation. The tool gateway and
            # tenant authorization still validate each dependency again.
            from app.metrics.authorization import authorize_metric

            scope_kind = "seller" if planner_scope == "seller" else "platform"
            authorized_by_id = {
                str(item.get("metric_id")): item
                for item in state.get("authorized_metrics", [])
                if isinstance(item, dict) and item.get("metric_id")
            }
            for dependency_id in registered_decomposition.input_metric_ids:
                if dependency_id in authorized_by_id:
                    continue
                try:
                    dependency_authorization = authorize_metric(
                        dependency_id,
                        provenance_type="registered_relation",
                        provenance_id=(
                            f"registered-decomposition:{registered_decomposition.method_id}:"
                            f"{registered_decomposition.target_metric_id}:{dependency_id}"
                        ),
                        scope=scope_kind,
                        scope_data=planner_scope_data,
                    )
                except (ValueError, KeyError) as exc:
                    state["metric_planning_decision"] = batch_plans.decision.model_copy(update={
                        "status": "unsupported",
                        "reason": f"Registered decomposition dependency authorization failed: {dependency_id}.",
                        "reason_code": "scope_rejected",
                        "registered_decomposition": registered_decomposition.model_copy(update={
                            "status": "authorization_blocked", "reason": str(exc),
                        }),
                    }).model_dump(mode="json")
                    state["tool_result"] = {
                        "server": "metric-registry-planner",
                        "tool_name": "metric_clarification",
                        "arguments": {"target_metric_id": registered_decomposition.target_metric_id},
                        "request_id": request_id,
                        "result": {"status": "unsupported", "parsed": {},
                                   "reason": "A registered decomposition dependency is not authorized."},
                    }
                    state["query_results"] = []
                    state["current_agent"] = "data_query"
                    state.setdefault("node_history", []).append({
                        "node": "data_query", "summary": "Registered decomposition stopped before data access because a required dependency was not authorized.",
                    })
                    return state
                authorized_by_id[dependency_id] = dependency_authorization.model_dump(mode="json")
            state["authorized_metrics"] = list(authorized_by_id.values())
            # The target plan exists only to carry its normal AuthorizedMetric
            # provenance. No ordinary compare/value query is dispatched here.
            state["registered_decomposition_request"] = registered_decomposition.model_dump(mode="json")
            state["diagnostic_target_metric"] = registered_decomposition.target_metric_id
            state["tool_result"] = {
                "server": "metric-registry-planner",
                "tool_name": "registered_numeric_decomposition",
                "arguments": {
                    "target_metric_id": registered_decomposition.target_metric_id,
                    "method_id": registered_decomposition.method_id,
                    "factor_metric_ids": list(registered_decomposition.factor_metric_ids),
                    "baseline_date": registered_decomposition.baseline_date,
                    "current_date": registered_decomposition.current_date,
                },
                "result": {
                    "status": "planned",
                    "parsed": batch_plans.decision.model_dump(mode="json"),
                    "request_id": request_id,
                },
            }
            state["query_results"] = []
            state["current_agent"] = "data_query"
            state.setdefault("node_history", []).append({
                "node": "data_query",
                "summary": (
                    "Preserved registered numeric decomposition intent; routed to the governed decomposition executor "
                    f"for target={registered_decomposition.target_metric_id}, method={registered_decomposition.method_id}."
                ),
            })
            return state
        if batch_plans.decision.status != "confirmed":
            clarification = batch_plans.decision.model_dump(mode="json")
            result_status = (
                "unsupported"
                if batch_plans.decision.status == "unsupported"
                else "clarification_required"
            )
            outcome_field = (
                "capability_boundary"
                if batch_plans.decision.reason_code == "period_contribution_not_registered"
                else "clarification"
            )
            result = {
                "server": "metric-registry-planner", "tool_name": "metric_clarification",
                "arguments": {"scope": planner_scope}, "request_id": request_id,
                "result": {
                    "status": result_status, "parsed": {}, outcome_field: clarification,
                    "reason": clarification.get("reason", "请确认要分析的指标。"),
                },
            }
            state["tool_result"] = result
            state["query_results"] = []
            if result_status == "unsupported":
                _record_planning_capability_limit(
                    state, batch_plans.decision, request_id=request_id, scope=planner_scope,
                )
            # A verified capability gap is a preflight control outcome, not a
            # data-tool invocation.  Keep the planner result for answer
            # generation, but do not add a synthetic tool_call that would
            # contaminate tool-validity and authorization metrics.
            capability_stop_recorded = any(
                isinstance(item, dict)
                and item.get("request_id") == request_id
                and item.get("limitation_code") == "CAPABILITY_GAP"
                and item.get("sql_executed") is False
                for item in state.get("decision_records", [])
            )
            if not capability_stop_recorded:
                state.setdefault("tool_calls", []).append(ToolInvocation(
                    server=result["server"], tool_name="metric_clarification", arguments=result["arguments"],
                    request_id=request_id, idempotency_key=request_id, status="unsupported",
                ).model_dump())
            state["current_agent"] = "data_query"
            state.setdefault("node_history", []).append({
                "node": "data_query", "summary": "Metric clarification required; no data tool was called.",
            })
            return state
    if pending_query and not batch_plans:
        # Root-cause follow-ups may change the date or operation, but they may
        # only reuse an AuthorizedMetric produced by the preceding query.  A
        # bare metric ID from pending state is not sufficient provenance.
        pending_metric = str(pending_query.get("metric_id") or "")
        authorized_rows = state.get("authorized_metrics") or []
        authorized_row = next(
            (item for item in authorized_rows if str(item.get("metric_id")) == pending_metric),
            None,
        )
        if authorized_row is None:
            state.setdefault("errors", []).append({
                "code": "missing_followup_authorization",
                "message": "后续查询缺少上一轮的 AuthorizedMetric，已拒绝执行。",
                "retryable": False,
                "agent_name": "data_query",
            })
            state["query_results"] = []
            return state
        try:
            authorized = AuthorizedMetric.model_validate(authorized_row)
            followup_decision = MetricPlanningDecision(
                status="confirmed",
                confirmed_metrics=[authorized.metric_id],
                query_type=str(pending_query.get("query_type") or "metric_value"),
                dimension=pending_query.get("dimension"),
                reason="复用上一轮已验证的授权指标。",
                reason_code="exact_match",
            )
            batch_plans = MetricQueryBatchPlan(
                decision=followup_decision,
                plans=build_query_plan(query_question, followup_decision, metrics=[authorized]),
            )
        except Exception as exc:
            state.setdefault("errors", []).append({
                "code": "invalid_followup_authorization",
                "message": "后续查询的授权指标无效，已拒绝执行。",
                "retryable": False,
                "agent_name": "data_query",
                "detail": type(exc).__name__,
            })
            state["query_results"] = []
            return state
    # 使用 trace 范围内的 request_id 作为幂等键，保证同一轮调用可在 Trace 中关联。
    # Registry 分发会原样返回该值；只有没有提供 request_id 的旧 MCP 兜底实现才可能自行生成。
    query_kwargs = {"task_type": "metric_query", "question": query_question, "idempotency_key": request_id}
    # 保留旧版/离线测试网关的原始函数签名，便于测试替换查询实现而不破坏节点逻辑。
    if state.get("tenant_id") or state.get("user_role"):
        query_kwargs.update({"tenant_id": state.get("tenant_id"), "role": state.get("user_role"), "request_id": request_id})
    # 已启用的数据浏览上下文已由 resolve_explorer_context 在服务端重新查库并锁定。
    # 普通查数直接复用该可信快照，避免 MCP 再按自然语言解析成不同的店铺/日期范围。
    if batch_plans and batch_plans.plans:
        # Defensive invariant: no plan reaches an MCP gateway without the
        # server-created authorization object, even if a planner/plugin was
        # replaced in a test or an older checkpoint is resumed.
        unauthorized = [plan for plan in batch_plans.plans if plan.authorized_metric is None]
        if unauthorized:
            state.setdefault("errors", []).append({
                "code": "missing_authorized_metric",
                "message": "QueryPlan 缺少服务端授权指标，已阻止 MCP 执行。",
                "retryable": False,
                "agent_name": "data_query",
            })
            state["query_results"] = []
            state["tool_result"] = {
                "server": "metric-registry-planner",
                "tool_name": "metric_clarification",
                "arguments": {}, "request_id": request_id,
                "result": {"status": "unsupported", "parsed": {}, "reason": "缺少 authoritative provenance。"},
            }
            return state
        contribution_reservation = None
        if contribution_contract:
            # Reserve only after the planner and server authorization checks,
            # but before crossing the tool boundary. A timeout or exception is
            # still an attempted execution and must not be retried by re-entry.
            planned = batch_plans.plans[0]
            item_request_id = f"{request_id}:1"
            contribution_reservation = contribution_execution_reservation(
                contribution_contract, planned, state, request_id=item_request_id,
            )
            state.setdefault("request_contract_query_reservations", []).append(contribution_reservation)
            record_contract_observation(state, "contribution_execution_reserved", {
                "execution_signature": contribution_reservation["execution_signature"],
                "tool_name": contribution_reservation["tool_name"],
                "metric_id": contribution_reservation["metric_id"],
                "dimension": contribution_reservation["dimension"],
                "comparison": contribution_reservation["comparison"],
                "scope_sha256": contribution_reservation["scope_sha256"],
            })
        results: list[dict] = []
        for index, planned in enumerate(batch_plans.plans, start=1):
            item_request_id = f"{request_id}:{index}"
            try:
                if state.get("seller_scope_payload") and state.get("controlled_analysis_scope"):
                    item = {
                        "server": "scoped-mcp", "tool_name": None,
                        "request_id": item_request_id,
                        "result": {
                            "status": "unsupported",
                            "parsed": planned.model_dump(mode="json"),
                            "reason": "Signed seller scope cannot be combined with an Explorer scope.",
                            "reason_code": "seller_scope_explorer_scope_conflict",
                        },
                    }
                elif state.get("seller_scope_payload"):
                    item = query_scoped_business_data(
                        query_question,
                        state["seller_scope_payload"],
                        request_id=item_request_id,
                        plan=planned,
                        manifest_id=state.get("data_version_manifest_id") or None,
                    )
                elif state.get("controlled_analysis_scope"):
                    scoped_kwargs = {
                        "request_id": item_request_id,
                        "plan": planned,
                        "manifest_id": state.get("data_version_manifest_id") or None,
                    }
                    if state.get("authorization_boundary"):
                        scoped_kwargs.update({
                            "authorization_boundary": state.get("authorization_boundary"),
                            "current_query_scope_fingerprint": state.get("current_query_scope_fingerprint") or "",
                        })
                    try:
                        item = query_scoped_business_data(query_question, state["controlled_analysis_scope"], **scoped_kwargs)
                    except TypeError as exc:
                        # Older gateways may not accept ``plan``. Only that exact
                        # compatibility error may use the legacy call signature.
                        if "plan" not in str(exc):
                            raise
                        item = query_scoped_business_data(query_question, state["controlled_analysis_scope"], request_id=item_request_id, manifest_id=state.get("data_version_manifest_id") or None)
                else:
                    item_kwargs = dict(query_kwargs)
                    item_kwargs["request_id"] = item_request_id
                    item_kwargs["plan"] = planned
                    item = query_business_data(**item_kwargs)
            except Exception as exc:
                # Preserve earlier successful items and make the failed plan
                # terminal. Do not re-run the batch or hide the exception as
                # an unsupported capability.
                if len(batch_plans.plans) == 1:
                    raise
                public_reason, diagnostic_detail = sanitize_exception(exc)
                item = {
                    "server": "metric-registry",
                    "tool_name": "registry_query",
                    "arguments": {"metric_name": planned.metric_name, "query_type": planned.query_type},
                    "request_id": item_request_id,
                    "idempotency_key": item_request_id,
                    "result": {
                        "status": "execution_failed",
                        "parsed": planned.model_dump(mode="json"),
                        "reason": public_reason,
                        "diagnostic_detail": diagnostic_detail,
                    },
                }
            # Normalize each per-plan gateway outcome before it enters
            # query_results. The top-level compatibility envelope is
            # normalized below too, but consumers also attach evidence to
            # these per-plan rows; leaving a tuple/list here would make a
            # malformed gateway response crash after it had already been
            # converted to a controlled refusal.
            if not isinstance(item, dict):
                item = {
                    "server": "invalid-tool-gateway",
                    "tool_name": None,
                    "arguments": {},
                    "request_id": item_request_id,
                    "idempotency_key": item_request_id,
                    "result": {
                        "status": "unsupported",
                        "parsed": planned.model_dump(mode="json"),
                        "reason": (
                            "数据查询网关返回了无效结果类型："
                            f"{type(item).__name__}。"
                        ),
                        "reason_code": "invalid_tool_result_shape",
                    },
                }
            else:
                item.setdefault("request_id", item_request_id)
                item.setdefault("idempotency_key", item_request_id)
                if not isinstance(item.get("result"), dict):
                    raw_inner = item.get("result")
                    item["result"] = {
                        "status": "unsupported",
                        "parsed": planned.model_dump(mode="json"),
                        "reason": (
                            "数据查询结果格式异常："
                            f"{type(raw_inner).__name__}，期望 dict。"
                        ),
                        "reason_code": "invalid_tool_result_shape",
                    }
                if item.get("tool_name") == "breakdown_scoped_seller" and item.get("result", {}).get("status") == "ok":
                    if not _capture_seller_boundary(state, item):
                        item["result"] = {
                            "status": "unsupported",
                            "parsed": planned.model_dump(mode="json"),
                            "reason": "Seller MCP result omitted a valid seller authorization boundary.",
                            "reason_code": "seller_boundary_missing_or_invalid",
                        }
            results.append(item)
            if len(batch_plans.plans) > 1 and isinstance(item, dict):
                item["_batch_item"] = True
            # 先验证网关返回值，再把第一项复制出来作为兼容性的顶层 result。
            # batch_results 中本身包含第一项，如果直接把第一项对象挂回自己，会形成循环引用，
            # 最终导致审计序列化失败。
            result = dict(results[0]) if results and isinstance(results[0], dict) else (results[0] if results else {
                "server": "metric-registry-planner", "tool_name": None,
                "arguments": {}, "request_id": request_id,
                "result": {"status": "unsupported", "parsed": {}, "reason": "No query result returned."},
            })
        if isinstance(result, dict):
            result["batch_results"] = results
            result["batch_count"] = len(results)
        state["query_results"] = results
        query_mode = (
            "seller-scoped" if state.get("seller_scope_payload")
            else "scoped" if state.get("controlled_analysis_scope") else "unrestricted"
        )
    elif state.get("seller_scope_payload") and state.get("controlled_analysis_scope"):
        query_mode = "scoped"
        result = {
            "server": "scoped-mcp", "tool_name": None,
            "request_id": request_id,
            "result": {
                "status": "unsupported", "parsed": {},
                "reason": "Signed seller scope cannot be combined with an Explorer scope.",
                "reason_code": "seller_scope_explorer_scope_conflict",
            },
        }
    elif state.get("seller_scope_payload"):
        query_mode = "seller-scoped"
        result = query_scoped_business_data(
            query_question,
            state["seller_scope_payload"],
            request_id=request_id,
            manifest_id=state.get("data_version_manifest_id") or None,
        )
    elif state.get("controlled_analysis_scope"):
        query_mode = "scoped"
        scoped_kwargs = {
            "request_id": request_id,
            "manifest_id": state.get("data_version_manifest_id") or None,
        }
        if state.get("authorization_boundary"):
            scoped_kwargs.update({
                "authorization_boundary": state.get("authorization_boundary"),
                "current_query_scope_fingerprint": state.get("current_query_scope_fingerprint") or "",
            })
        result = query_scoped_business_data(query_question, state["controlled_analysis_scope"], **scoped_kwargs)
    else:
        query_mode = "unrestricted"
        result = query_business_data(**query_kwargs)
    # 所有工具网关按合同都应返回 dict。遇到旧实现或外部网关返回 tuple/list 时，
    # 要把异常结果转换成可观测的 unsupported，而不是让它流到后续 .get() 触发 500。
    if not isinstance(result, dict):
        result = {
            "server": "invalid-tool-gateway", "tool_name": None, "arguments": {}, "request_id": request_id,
            "result": {
                "status": "unsupported", "parsed": {},
                "reason": f"数据查询网关返回了无效结果类型：{type(result).__name__}。",
            },
        }
    if result.get("tool_name") == "breakdown_scoped_seller" and result.get("result", {}).get("status") == "ok":
        if not _capture_seller_boundary(state, result):
            result["result"] = {
                "status": "unsupported", "parsed": {},
                "reason": "Seller MCP result omitted a valid seller authorization boundary.",
                "reason_code": "seller_boundary_missing_or_invalid",
            }
    ensure_safe_untrusted_content(result, "mcp_tool_result")
    # 内层 result["result"] 也必须是 dict，否则下游 .get() 会因 tuple/list 抛出
    # AttributeError。与最外层防护不同，这里不需要重建整个 result，覆盖内层即可。
    inner_result = result.get("result")
    if not isinstance(inner_result, dict):
        result["result"] = {
            "status": "unsupported", "parsed": {},
            "reason": f"数据查询结果格式异常：{type(inner_result).__name__}，期望 dict。",
        }
    if contribution_contract and batch_plans and batch_plans.plans:
        tool_mismatch = _contribution_tool_result_mismatch(contribution_contract, result)
        record_contract_observation(state, "tool_result", {
            "tool_name": result.get("tool_name"),
            "arguments": result.get("arguments") or {},
            "status": (result.get("result") or {}).get("status"),
            "mismatch": tool_mismatch,
        })
        if tool_mismatch:
            record_contract_violation(state, "tool", tool_mismatch)
            result["result"] = {
                **(result.get("result") or {}),
                "status": "unsupported",
                "reason_code": tool_mismatch,
                "reason": "工具执行结果与已确认请求契约不一致，未作为请求结果接受。",
            }
        reservation = contribution_reservation if "contribution_reservation" in locals() else None
        if isinstance(reservation, dict):
            status = str((result.get("result") or {}).get("status") or "unknown")
            reservation["status"] = "completed" if status in {"ok", "partial", "not_found"} else "attempted_failed"
            reservation["result_status"] = status
            reservation["result_request_id"] = result.get("request_id")
    # Tool gateways intentionally return a structured ``unsupported`` object
    # instead of throwing.  Authorization/scope invalidation is different
    # from an ordinary capability clarification: it must become a terminal
    # abstention so the lifecycle cannot report a stale execution as
    # successfully completed.
    inner_result = result.get("result", {})
    # Preserve a typed execution failure as a terminal workflow error.  The
    # result/evidence envelope remains available for audit, while the final
    # answer status cannot be inferred as completed from the absence of rows.
    from app.core.failure_context import ExecutionFailureContext, public_failure_message
    failure_items = state.get("query_results") or [result]
    for failure_item in failure_items:
        if not isinstance(failure_item, dict):
            continue
        item_result = failure_item.get("result")
        if not isinstance(item_result, dict):
            continue
        item_status = item_result.get("status")
        if item_status not in {"execution_failed", "unsupported"}:
            continue
        raw_context = item_result.get("failure_context")
        if item_status == "unsupported" and not raw_context:
            continue
        try:
            failure_context = ExecutionFailureContext.model_validate(raw_context)
        except Exception:
            # A malformed/missing failure contract is itself an execution
            # error; never silently treat it as an ordinary empty result.
            failure_context = ExecutionFailureContext(
                category="execution_error", service="unknown",
                code="failure_context_invalid", cause_type="InvalidFailureContext",
                request_id=str(failure_item.get("request_id") or request_id),
                retryable=None,
            )
        error_code = {
            "tool_timeout": "tool_timeout",
            "database_error": "database_error",
            "capability_gap": "CAPABILITY_GAP",
            "authorization_denied": "AUTH_DENIED",
            "execution_error": "execution_failed",
        }[failure_context.category]
        if item_status == "unsupported" and failure_context.category not in {
            "authorization_denied", "execution_error", "database_error", "tool_timeout",
        }:
            continue
        if not any(
            isinstance(existing, dict)
            and existing.get("code") == error_code
            and existing.get("request_id") == failure_context.request_id
            for existing in (state.get("errors") or [])
        ):
            state.setdefault("errors", []).append({
                "code": error_code,
                "message": public_failure_message(failure_context),
                "public_message": public_failure_message(failure_context),
                "retryable": failure_context.retryable,
                "failure_context": failure_context.model_dump(mode="json"),
                "request_id": failure_context.request_id,
                "agent_name": "data_query",
            })
    if isinstance(inner_result, dict) and inner_result.get("status") == "unsupported":
        reason_text = str(inner_result.get("reason") or "")
        lowered_reason = reason_text.lower()
        if any(token in lowered_reason for token in ("stale", "authorization", "permission", "scope fingerprint", "越权", "权限")):
            if "scope" in lowered_reason or "范围" in reason_text:
                error_code = "SCOPE_STALE"
            elif "denied" in lowered_reason or "权限" in reason_text:
                error_code = "AUTH_DENIED"
            else:
                error_code = "AUTH_STALE"
            state.setdefault("errors", []).append({
                "code": error_code,
                "message": reason_text or "授权或查询范围已失效，已停止执行。",
                "public_message": "当前授权或查询范围已失效，未继续访问数据。请刷新范围后重试。",
                "retryable": True,
                "agent_name": "data_query",
            })
    record_event("mcp.call", {
        "trace_id": state.get("trace_id"), "thread_id": state.get("thread_id"),
        "tool_name": result.get("tool_name"), "duration_ms": round((perf_counter() - started) * 1000, 3),
        "status": result.get("result", {}).get("status", "unknown"),
        "query_mode": query_mode,
        "scoped_seller": state.get("controlled_analysis_scope", {}).get("seller_id") if query_mode == "scoped" else None,
        "scoped_dates": f"{state.get('controlled_analysis_scope', {}).get('start_date')}~{state.get('controlled_analysis_scope', {}).get('end_date')}" if query_mode == "scoped" else None,
    })
    result["requested_task_type"] = task_type
    result["query_question"] = query_question
    evaluation_surface = state.get("evaluation_tool_surface")
    if isinstance(evaluation_surface, dict):
        # Normalize only the evaluation-visible surface metadata. The actual
        # gateway call and its authorization result remain untouched.
        result["available_tools"] = list(evaluation_surface.get("allowed_operation_names") or [])
        result["evaluation_tool_surface_sha256"] = evaluation_surface.get("surface_sha256")
    if pending_query:
        state.setdefault("root_cause_evidence", []).append({
            "purpose": pending_query["purpose"],
            "question": query_question,
            "metric_id": pending_query.get("metric_id"),
            "query_id": pending_query.get("query_id", ""),
            "result": result,
        })
        # 显式写 None 才能让 LangGraph 的 state 合并清除待执行计划。
        state["pending_query"] = None
        state["next_agent"] = "root_cause"
        state.setdefault("handoffs", []).append({
            "from": "data_query",
            "to": "root_cause",
            "reason": f"Returned {pending_query['purpose']} MCP evidence.",
        })
    else:
        state["tool_result"] = result
    tool_name = result.get("tool_name") or "unsupported"
    first_plan = batch_plans.plans[0] if batch_plans and batch_plans.plans else None
    # Bind the evidence envelope and the persisted invocation to the exact
    # request ID that crossed the gateway. Older adapters may return a second
    # internal response ID; preserve it separately rather than letting it
    # detach authorization and evidence from this invocation.
    response_request_id = result.get("request_id")
    first_query_result = (state.get("query_results") or [{}])[0]
    tool_request_id = str(
        first_query_result.get("request_id")
        if isinstance(first_query_result, dict) and first_query_result.get("request_id")
        else response_request_id or request_id
    )
    if response_request_id and str(response_request_id) != tool_request_id:
        result["gateway_request_id"] = str(response_request_id)
    result["request_id"] = tool_request_id
    authorization_snapshot = result.get("authorization_snapshot")
    if isinstance(authorization_snapshot, dict):
        authorization_snapshot = dict(authorization_snapshot)
        snapshot_request_id = str(authorization_snapshot.get("request_id") or "")
        if snapshot_request_id and snapshot_request_id != tool_request_id:
            state.setdefault("errors", []).append({
                "code": "authorization_snapshot_request_mismatch",
                "message": "授权快照与工具调用的请求 ID 不一致，已保留为不可验证。",
                "retryable": False,
                "agent_name": "data_query",
                "request_id": tool_request_id,
            })
            authorization_snapshot["decision"] = "unknown"
            authorization_snapshot["reason_code"] = "authorization_snapshot_request_mismatch"
        authorization_snapshot["request_id"] = tool_request_id
    evidence_content = {
        "question": query_question,
        "request_id": tool_request_id,
        "result": result.get("result", {}),
    }
    if first_plan is not None:
        evidence_content.update({
            "request_id": result.get("request_id"),
            "metric_id": first_plan.metric_name,
            "query_type": first_plan.query_type,
        })
    if pending_query:
        evidence_content["query_id"] = pending_query.get("query_id", "")
    first_evidence_id = stable_evidence_id("mcp", tool_name, evidence_content)
    state.setdefault("evidence_bundles", []).append(EvidenceBundle(
        evidence_id=first_evidence_id,
        source_type="mcp", source_ref=tool_name, content=evidence_content,
    ).model_dump(mode="json"))
    invocation = ToolInvocation(
        server=result.get("server", "local-mcp"), tool_name=tool_name,
        planned_metric_id=first_plan.metric_name if first_plan is not None else None,
        planned_operation=first_plan.query_type if first_plan is not None else None,
        arguments=result.get("arguments", {}),
        request_id=tool_request_id,
        gateway_request_id=str(response_request_id) if response_request_id and str(response_request_id) != tool_request_id else None,
        idempotency_key=result.get("idempotency_key") or tool_request_id,
        status=result.get("result", {}).get("status", "failed") if result.get("result", {}).get("status") in {"ok", "not_found", "unsupported"} else "failed",
        evidence_id=first_evidence_id,
        authorization_snapshot=authorization_snapshot if isinstance(authorization_snapshot, dict) else None,
    )
    if state.get("query_results"):
        state["query_results"][0]["evidence_id"] = first_evidence_id
    state["current_agent"] = "data_query"
    state.setdefault("_revision_counts", {})
    state["_revision_counts"]["data_query"] = state["_revision_counts"].get("data_query", 0) + 1
    state.setdefault("node_history", []).append(
        {
            "node": "data_query",
            "summary": f"MCP tool={result.get('tool_name', 'none')} "
            f"status={result.get('result', {}).get('status', 'unknown')} "
            f"mode={query_mode}",
        }
    )
    state.setdefault("tool_calls", []).append(
        {
            **invocation.model_dump(),
        }
    )
    # 批量查询保留一个兼容旧消费者的顶层 tool_result；每个额外指标仍然单独记录
    # tool invocation 和 evidence，保证批量结果也能逐项追溯。
    for batch_index, batch_item in enumerate(state.get("query_results", [])[1:], start=2):
        if not isinstance(batch_item, dict):
            continue
        item_result = batch_item.get("result", {})
        item_tool = batch_item.get("tool_name") or "registry_query"
        planned_item = batch_plans.plans[batch_index - 1] if batch_plans and len(batch_plans.plans) >= batch_index else None
        item_request_id = str(batch_item.get("request_id") or f"{request_id}:{batch_index}")
        item_response_request_id = batch_item.get("gateway_request_id")
        item_authorization_snapshot = batch_item.get("authorization_snapshot")
        if isinstance(item_authorization_snapshot, dict):
            item_authorization_snapshot = dict(item_authorization_snapshot)
            snapshot_request_id = str(item_authorization_snapshot.get("request_id") or "")
            if snapshot_request_id and snapshot_request_id != item_request_id:
                state.setdefault("errors", []).append({
                    "code": "authorization_snapshot_request_mismatch",
                    "message": "授权快照与批量工具调用的请求 ID 不一致，已保留为不可验证。",
                    "retryable": False,
                    "agent_name": "data_query",
                    "request_id": item_request_id,
                })
                item_authorization_snapshot["decision"] = "unknown"
                item_authorization_snapshot["reason_code"] = "authorization_snapshot_request_mismatch"
            item_authorization_snapshot["request_id"] = item_request_id
        state["tool_calls"].append(ToolInvocation(
            server=batch_item.get("server", "metric-registry"), tool_name=item_tool,
            planned_metric_id=planned_item.metric_name if planned_item is not None else None,
            planned_operation=planned_item.query_type if planned_item is not None else None,
            arguments=batch_item.get("arguments", {}),
            request_id=item_request_id,
            gateway_request_id=str(item_response_request_id) if item_response_request_id else None,
            idempotency_key=batch_item.get("idempotency_key", item_request_id),
            status=item_result.get("status", "failed") if isinstance(item_result, dict) and item_result.get("status") in {"ok", "not_found", "unsupported"} else "failed",
            evidence_id=None,
            authorization_snapshot=item_authorization_snapshot if isinstance(item_authorization_snapshot, dict) else None,
        ).model_dump())
        item_content = {"question": query_question, "metric_index": batch_index, "request_id": item_request_id, "result": item_result}
        if planned_item is not None:
            item_content.update({
                "request_id": batch_item.get("request_id"),
                "metric_id": planned_item.metric_name,
                "query_type": planned_item.query_type,
            })
        item_evidence_id = stable_evidence_id("mcp", item_tool, item_content)
        state.setdefault("evidence_bundles", []).append(EvidenceBundle(
            evidence_id=item_evidence_id,
            source_type="mcp", source_ref=item_tool, content=item_content,
        ).model_dump(mode="json"))
        batch_item["evidence_id"] = item_evidence_id
        state["tool_calls"][-1]["evidence_id"] = item_evidence_id
    if len(state.get("query_results", [])) > 1:
        item_statuses = [
            str((item.get("result") or {}).get("status") or "execution_failed")
            for item in state["query_results"] if isinstance(item, dict)
        ]
        if all(status == "ok" for status in item_statuses):
            state["metric_batch_execution"] = {"status": "all_success"}
        elif any(status not in {"ok", "not_found", "unsupported"} for status in item_statuses):
            state["metric_batch_execution"] = {"status": "execution_failed"}
        else:
            state["metric_batch_execution"] = {"status": "partial_unavailable"}
        result["batch_status"] = state["metric_batch_execution"]["status"]
    # An executed required query with an explicit missing-data result is a
    # terminal data limitation, not an ordinary completed comparison. Preserve
    # the query/evidence envelope and its exact metric, operation and period.
    # Do not infer this from an empty list or from execution/authorization
    # errors; only the typed result status activates this branch.
    missing_rows = [
        item for item in (state.get("query_results") or [])
        if isinstance(item, dict)
        and isinstance(item.get("result"), dict)
        and item["result"].get("status") in {"not_found", "insufficient_data"}
    ]
    if not missing_rows and isinstance(result.get("result"), dict) and result["result"].get("status") in {
        "not_found", "insufficient_data",
    }:
        missing_rows = [result]
    if missing_rows:
        state["terminal_code"] = "DATA_INSUFFICIENT"
        state["termination_initiator"] = "policy"
        existing_decisions = state.setdefault("decision_records", [])
        for missing_index, missing in enumerate(missing_rows):
            planned = (
                batch_plans.plans[missing_index]
                if batch_plans is not None and missing_index < len(batch_plans.plans)
                else first_plan
            )
            arguments = missing.get("arguments") if isinstance(missing.get("arguments"), dict) else {}
            parsed = (missing.get("result") or {}).get("parsed")
            parsed = parsed if isinstance(parsed, dict) else {}
            metric_id = str(
                (planned.metric_name if planned is not None else None)
                or arguments.get("metric_id") or arguments.get("metric_name")
                or parsed.get("metric_name") or ""
            )
            operation = str(
                (planned.query_type if planned is not None else None)
                or parsed.get("query_type") or ""
            )
            comparison = {}
            if planned is not None:
                plan_dump = planned.model_dump(mode="json")
                for key in (
                    "metric_date", "start_date", "end_date", "baseline_date", "current_date",
                    "baseline_start", "baseline_end", "current_start", "current_end", "granularity",
                ):
                    if plan_dump.get(key) is not None:
                        comparison[key] = plan_dump[key]
            for key in (
                "metric_date", "start_date", "end_date", "baseline_date", "current_date",
                "baseline_start", "baseline_end", "current_start", "current_end", "granularity",
            ):
                if key not in comparison and arguments.get(key) is not None:
                    comparison[key] = arguments[key]
            record = {
                "decision_id": f"{missing.get('request_id') or request_id}:data-limit",
                "request_id": missing.get("request_id") or request_id,
                "limitation_code": "DATA_INSUFFICIENT",
                "metric_id": metric_id,
                "operation": operation,
                "scope": "seller" if state.get("controlled_analysis_scope") else "platform",
                "comparison": comparison,
                "dimension": arguments.get("dimension") or (parsed.get("dimension") if isinstance(parsed, dict) else None),
                "authorization_checked": True,
                "registry_capability_checked": True,
                "sql_executed": True,
                "evidence_id": missing.get("evidence_id"),
                "source": "typed_query_result",
            }
            if not any(
                isinstance(previous, dict)
                and previous.get("request_id") == record["request_id"]
                and previous.get("limitation_code") == "DATA_INSUFFICIENT"
                for previous in existing_decisions
            ):
                existing_decisions.append(record)
    if contribution_contract:
        route_mismatch = contribution_execution_state_mismatch(contribution_contract, state)
        record_contract_observation(state, "contribution_route_precondition", {
            "validated_for_answer_route": route_mismatch is None,
            "mismatch": route_mismatch,
        })
        if route_mismatch:
            record_contract_violation(state, "data_query_route", route_mismatch)
    return state
