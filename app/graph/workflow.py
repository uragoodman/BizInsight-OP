"""LangGraph 聊天工作流编排 — Target 多 Agent 版。

阅读顺序建议：
1. ``build_chat_graph``：注册哪些节点、节点之间怎样连线；
2. ``_route_*``：节点执行完成后，根据 state 的哪些字段选择下一站；
3. ``run_chat_workflow`` / ``stream_chat_workflow``：HTTP API 如何启动图；
4. ``init_state`` / ``persist_trace``：运行前准备默认状态、结束后保存现场。

每个节点都经 ``instrument_node`` 包装。因此节点实际执行顺序是：
运行时预算检查 → 原节点函数 → 节点输出契约校验 → 记录耗时/错误。
若包装层或节点把结构化错误写入 ``state[\"errors\"]``，下面的路由函数会
尽快把流程交给 critic；critic 最终仍会尝试进入 ``persist_trace`` 留下现场。

图结构（查询/报告/受控诊断 + 审核返工；旧 root_cause 节点不再注册）：

START → init_state → triage
                        │
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
  metric_definition  data_query     data_query
  (all types)      (metric_query)  (anomaly_analysis)
         │              │              │
         ↓              ↓              ↓
      data_query   generate_answer  premise_check
  (report_gen)                    ↓
         ↓                     diagnostic_plan
       report                     ↓
  (report_gen)             governed investigation
         │
         └──────────┬──────────────┘
                    ↓
             generate_answer
                    ↓
                  critic
                    │
          ┌─────────┼──────────────┐
          ↓         ↓              ↓
      persist_  human_review  needs_revision
      _trace       │         (→ target node)
          ↓    ┌────┴────┐        ↓
         END triage persist   specialist /
               (reset) _trace  gen_answer /
                       ↓        triage
                      END
"""

from uuid import uuid4
from queue import Empty, Queue
from threading import Thread

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.answer import generate_answer
from app.graph.nodes.critic import critic
from app.graph.nodes.data_query import data_query
from app.graph.nodes.metric_definition import metric_definition
from app.graph.nodes.report import report
from app.graph.nodes.seller_scope_guard import has_seller_scope, seller_scope_guard
from app.graph.nodes.seller_catalog import seller_catalog
from app.graph.nodes.resolve_explorer_context import resolve_explorer_context
from app.graph.nodes.triage import triage
from app.graph.terminal_status import result_terminal_status
from app.context.compiler import compile_turn_context, context_clarification
from app.context.orchestrator import ContextOrchestrator
from app.graph.nodes.semantic_parse import semantic_parse
from app.graph.nodes.intent_router import intent_router
from app.graph.nodes.complex_plan import complex_plan
from app.graph.nodes.complex_clarification import complex_clarification
from app.graph.nodes.human_review import human_review
from app.graph.nodes.diagnostic import diagnostic_plan, diagnostic_decide, policy_execute, observe, diagnostic_finalize
from app.graph.nodes.premise import premise_check, premise_confirmation
from app.memory.service import load_session_memory, persist_memory_p0, retrieve_analysis_memories
from app.chat_history.repository import ChatHistoryRepository
from app.context.result_context import build_result_context
from app.graph.state import AgentState
from app.graph.critic_review_mode import CriticLLMReviewMode
from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
from app.llm.call_config import LLMCallConfig
from app.agents.contracts import CriticRevisionTarget
from app.graph.runtime import get_checkpointer, instrument_node
from app.harness.storage import write_trace
from app.security import ensure_safe_prompt
from app.core.config import get_settings
from app.metrics.authorization import scope_fingerprint
from app.core import execution_context
from app.core.exceptions import sanitize_exception
from app.schemas.events import SSE_EVENT_ADAPTER, SseCompletedEvent, SseFailedEvent, SseHeartbeatEvent, SseNodeCompletedEvent, SseNodeError, SsePlanEvent, SseStartedEvent, SseSuspendedEvent
from app.schemas.analysis_presentation import AnalysisPresentation
import logging
LOGGER = logging.getLogger(__name__)


from app.tenancy import TenantContext, activate_tenant_context
from app.observability import span
from app.rag.release import published_rag_version

CRITIC_REVISION_ROUTES = {target: target.value for target in CriticRevisionTarget}

_CONTEXT_ORCHESTRATOR = ContextOrchestrator()


def _with_evidence_refresh(node):
    """Mark current-turn evidence available after a tool-executing node.

    The wrapper runs for every return path, including early failures, so a
    later agent can distinguish "no evidence yet" from "tool execution done".
    """
    def wrapped(state: AgentState) -> AgentState:
        state = node(state)
        _CONTEXT_ORCHESTRATOR.refresh_evidence(state)
        return state

    return wrapped


# ============================================================
# 路由函数（纯 state 读取，不引入 graph 级别的耦合）
# ============================================================

def _has_token_budget_error(state: AgentState) -> bool:
    return any(
        isinstance(error, dict) and error.get("code") == "token_budget_exceeded"
        for error in (state.get("errors") or [])
    )


def _budget_stop_route(state: AgentState) -> str:
    """End diagnosis without entering another planner, query or Critic."""
    return "diagnostic_finalize" if state.get("task_type") == "anomaly_analysis" else "generate_answer"


def _route_after_triage(state: AgentState) -> str:
    """Keep seller turns out of generic memory/RAG specialist paths."""

    if state.get("triage_unresolved"):
        return "context_clarification"
    if has_seller_scope(state):
        if state.get("task_type") != "metric_query":
            return "seller_scope_unsupported"
        if (state.get("context_compilation") or {}).get("needs_clarification"):
            return "context_clarification"
        return "data_query"
    return "retrieve_analysis_memories"


def _route_after_triage_for_shadow(state: AgentState) -> str:
    """Run legacy triage first for shadow audit and enrolled Phase 3 canaries."""

    mode = get_settings().complex_route_mode
    if mode == "shadow":
        return "intent_router"
    if mode == "active" and state.get("complex_route_canary_selected") is True:
        # A canary route may replace ordinary triage decisions, but never a
        # graph-level stop condition emitted by the existing path.
        if _has_token_budget_error(state):
            return _budget_stop_route(state)
        if state.get("errors"):
            return _route_after_triage(state)
        return "intent_router"
    return _route_after_triage(state)


def _route_from_triage(state: AgentState) -> str:
    """根据 triage 写入的 ``next_agent`` 选择第一位 Specialist。

    输入：triage 已写入 ``task_type``、``next_agent``；
    输出：LangGraph 中下一个节点的名字。
    注意：这里不执行 Agent，只做纯读取/分支，方便单独测试路由规则。
    """
    if _has_token_budget_error(state):
        return _budget_stop_route(state)
    if state.get("triage_unresolved"):
        return "context_clarification"
    if has_seller_scope(state) and state.get("task_type") != "metric_query":
        return "seller_scope_unsupported"
    if state.get("errors"):
        # 上游失败时不继续调用 Specialist，先交给 critic 汇总错误并结束/处理。
        return "critic"
    if (state.get("context_compilation") or {}).get("needs_clarification"):
        return "context_clarification"
    request_contract = ((state.get("context_compilation") or {}).get("request_contract") or {})
    if request_contract.get("required_evidence_sequence") == ["trend", "dimension_contribution"]:
        # An explicitly ordered trend→contribution request enters the bounded
        # governed evidence loop directly; it must not be collapsed into the
        # single-query contribution shortcut.
        return "premise_check"
    # dict.get(key, default)：triage 未写 next_agent 时，保守地走指标定义路径。
    return state.get("next_agent", "metric_definition")


def _route_after_explorer_context(state: AgentState) -> str:
    """浏览上下文校验失败时直接交给 critic，成功后才进入 triage。"""
    if _has_token_budget_error(state):
        return _budget_stop_route(state)
    return "critic" if state.get("errors") else "triage"


def _route_after_intent_router(state: AgentState) -> str:
    """Shadow preserves triage; active canaries use only a sealed selected route."""

    mode = get_settings().complex_route_mode
    if mode == "shadow":
        if not state.get("task_type"):
            return "triage"
        return _route_after_triage(state)
    if mode != "active":
        return "triage"
    if _has_token_budget_error(state):
        return _budget_stop_route(state)
    if state.get("errors"):
        return "critic"
    canary = state.get("complex_route_canary_observation")
    if not isinstance(canary, dict) or canary.get("eligible") is not True:
        # Direct entry into the active node without a sampled allowlisted
        # request remains closed. The normal graph sends non-enrolled traffic
        # straight through legacy triage without visiting this node.
        return "complex_clarification"
    action = canary.get("action")
    if action == "complex_plan":
        from app.context.semantic_contract_v2 import SealedRouteDecision

        return (
            "complex_plan"
            if isinstance(state.get("complex_route_decision"), SealedRouteDecision)
            else "complex_clarification"
        )
    if action == "legacy_triage":
        return _route_after_triage(state)
    return "complex_clarification"


def _route_after_semantic_parse(state: AgentState) -> str:
    """Run triage first for shadow and active canary baseline capture."""

    return (
        "triage"
        if get_settings().complex_route_mode in {"shadow", "active"}
        else "intent_router"
    )


def _route_after_complex_plan(state: AgentState) -> str:
    if state.get("errors"):
        return "persist_trace"
    clarification = state.get("complex_clarification") or {}
    if clarification.get("reason_code") in {
        "scope_required", "scope_rejected", "plan_creation_rejected",
        "needs_context", "semantic_router_not_ready",
    }:
        return "complex_clarification"
    if state.get("complex_plan"):
        return "persist_trace"
    return "complex_clarification"


def _route_after_metric_definition(state: AgentState) -> str:
    """metric_definition 执行后，根据 task_type 决定下一步。

    - report_generation → data_query（先拉数据再交给 report 整合）
    - 其他 → generate_answer
    """
    if _has_token_budget_error(state):
        return _budget_stop_route(state)
    if has_seller_scope(state) and state.get("task_type") == "report_generation":
        return "seller_scope_unsupported"
    if state.get("errors"):
        return "critic"
    if state.get("task_type") == "report_generation":
        # 报告同时需要“口径上下文”和“实际业务数据”，因此先补 data_query。
        return "data_query"
    # 单纯定义问题已有 RAG 上下文，可以直接组织用户答案。
    return "generate_answer"


def _route_after_data_query(state: AgentState) -> str:
    """data_query 执行后，根据 task_type 决定下一步。

    - anomaly_analysis → premise_check，再进入受治理的诊断路径
    - report_generation → report（需要 tool_result + metric_context 写报告）
    - 其他 → generate_answer
    """
    if _has_token_budget_error(state):
        return _budget_stop_route(state)
    if has_seller_scope(state) and (
        state.get("task_type") != "metric_query"
        or (state.get("registered_decomposition_request") or {}).get("status") == "ready"
        or ((state.get("context_compilation") or {}).get("request_contract") or {}).get("required_evidence_sequence")
    ):
        return "seller_scope_unsupported"
    if state.get("errors"):
        return "critic"

    # 指标澄清是一次安全收敛的规划结果，不是 Root Cause 的输入。
    # 例如“应该如何做异常分析”可能被分类为 anomaly_analysis，但它没有具体指标。
    # 如果直接交给 Root Cause，Root Cause 会误以为已有假设链，可能在 Critic 介入
    # 之前就失败。因此这里先转到 critic，让审核路径处理缺少指标或上下文的问题。
    raw_result = state.get("tool_result", {}).get("result", {})
    if isinstance(raw_result, dict) and raw_result.get("status") == "clarification_required":
        return "critic"

    # A dimensional statistical-contribution request asks for one bounded,
    # registered comparison. Do not enter the open-ended diagnostic loop unless
    # the sealed contract explicitly asks for business-causal investigation.
    # Keep anomaly_analysis as a Critic-applicability label, but validate that
    # exactly one contract-bound query and evidence result exists before answer.
    compilation = state.get("context_compilation") or {}
    contract = compilation.get("request_contract") or {}
    if contract.get("operation_intent") == "dimensional_statistical_contribution":
        from app.context.request_contract import (
            contribution_execution_state_mismatch, validate_compiled_request_contract,
        )

        valid_contract, _reason = validate_compiled_request_contract(
            compilation,
            state_contract_hash=state.get("request_contract_hash"),
            require_state_anchor=True,
        )
        if valid_contract:
            if contribution_execution_state_mismatch(contract, state) is None:
                return "generate_answer"
            return "persist_trace"

    task_type = state.get("task_type", "")
    decomposition_request = state.get("registered_decomposition_request") or {}
    if decomposition_request.get("status") == "ready":
        return "diagnostic_plan"
    if task_type == "anomaly_analysis":
        return "premise_check"
    if task_type == "report_generation":
        # 报告节点把 RAG 口径、数据结果和模板整合成报告草案。
        return "report"
    return "generate_answer"


def _route_after_generate_answer(state: AgentState) -> str:
    """Persist terminal failures instead of sending them into another review loop."""

    if state.get("errors"):
        return "persist_trace"
    return "critic"


def _route_after_premise_check(state: AgentState) -> str:
    if has_seller_scope(state):
        return "seller_scope_unsupported"
    if _has_token_budget_error(state):
        return _budget_stop_route(state)
    if state.get("errors"):
        return "critic"
    if state.get("premise_check", {}).get("status") == "mismatch":
        return "premise_confirmation"
    if state.get("premise_check", {}).get("status") in {"unavailable", "cancelled"}:
        return "persist_trace"
    return "diagnostic_plan"


def _route_after_premise_confirmation(_state: AgentState) -> str:
    """Close a confirmed correction from the already-validated trend evidence.

    The original anomaly request was premised on the opposite direction. A
    confirmation resolves that mismatch; investigating a new direction needs a
    new explicit request so it cannot silently widen into a root-cause search.
    """
    return "persist_trace"


def _route_after_diagnostic_plan(state: AgentState) -> str:
    # LLM failure changes the selector to a deterministic plan, not the
    # execution architecture. Keep all diagnosis on the governed loop.
    if has_seller_scope(state):
        return "seller_scope_unsupported"
    if _has_token_budget_error(state):
        return "diagnostic_finalize"
    if state.get("errors"):
        return "diagnostic_finalize"
    return "diagnostic_decide"


def _route_after_diagnostic_decide(state: AgentState) -> str:
    if has_seller_scope(state):
        return "seller_scope_unsupported"
    if state.get("errors"):
        return "diagnostic_finalize"
    action = state.get("diagnostic_action", {}) or {}
    investigation = state.get("investigation_state", {}) or {}
    stop = investigation.get("stop_reason", {}) or {}
    if action.get("action_type") == "ABSTAIN" and stop.get("code") in {
        "AUTHORIZATION_BLOCKED", "CAPABILITY_GAP", "DATA_INSUFFICIENT",
        "USER_SCOPE_LIMIT", "BUDGET_EXHAUSTED", "NO_HIGH_VALUE_ACTION",
    }:
        return "diagnostic_finalize"
    return "policy_execute"


def _route_after_policy(state: AgentState) -> str:
    if has_seller_scope(state):
        return "seller_scope_unsupported"
    decision = state.get("policy_decision", {})
    action_type = state.get("diagnostic_action", {}).get("action_type")
    if state.get("errors"):
        return "diagnostic_finalize"
    if decision.get("decision") == "deny":
        return "diagnostic_decide"
    if action_type in {"FINISH", "ABSTAIN"}:
        return "diagnostic_finalize"
    if action_type == "REPLAN":
        return "diagnostic_decide"
    return "observe"


def _route_after_observe(state: AgentState) -> str:
    if has_seller_scope(state):
        return "seller_scope_unsupported"
    if _has_token_budget_error(state) or state.get("errors"):
        return "diagnostic_finalize"
    return "diagnostic_decide"


def _route_from_critic(state: AgentState) -> str:
    """根据 critic_verdict + critic_revision_target 做定向返工。

    critic 节点自身已处理 revision 上限，
    此函数只负责按裁决路由：

    - approved → persist_trace
    - human_review → human_review（人工审核中断点）
    - needs_revision → 目标节点（revision_target）
    - rejected → persist_trace；其他无效状态 → triage
    """

    # critic 负责写 verdict；缺失时按 needs_revision 处理，避免静默放行答案。
    verdict = state.get("critic_verdict", "needs_revision")
    if state.get("errors"):
        # 保留失败现场比继续返工更重要：直接去落盘，供 /traces 和调试使用。
        return "persist_trace"
    if (state.get("critic_result") or {}).get("data_review", {}).get("status") == "incomplete":
        # Persist a clear abstention; retrying the identical comparison cannot
        # supply the missing operand.
        return "persist_trace"
    if (state.get("critic_result") or {}).get("data_review", {}).get("batch_status") in {
        "partial_unavailable", "execution_failed",
    }:
        # Partial batches are terminal. Re-querying them would repeat already
        # successful metrics and cannot cure a missing/unsupported item.
        return "persist_trace"
    if verdict == "approved":
        return "persist_trace"
    if verdict == "rejected":
        # Rejection is terminal. Only an explicit needs_revision decision may
        # restart a bounded revision; otherwise failures repeat the whole run.
        return "persist_trace"
    if verdict == "human_review":
        return "human_review"

    target = state.get("critic_revision_target")

    # A statistical contribution contract does not authorize expanding into
    # causal investigation. Permit answer-only revision; if Critic asks for a
    # new diagnostic/query path, persist the bounded result and its limitation.
    contract = (((state.get("context_compilation") or {}).get("request_contract")) or {})
    if contract.get("operation_intent") == "dimensional_statistical_contribution":
        return "generate_answer" if target == "generate_answer" else "persist_trace"

    # Persisted/legacy state may still carry retired specialist names.
    # Normalize them into the governed diagnostic path; never execute the old route.
    target = {"root_cause": "diagnostic_plan", "branch_evidence": "diagnostic_plan"}.get(target, target)

    try:
        normalized_target = CriticRevisionTarget(target)
    except (TypeError, ValueError):
        return "triage"
    return CRITIC_REVISION_ROUTES[normalized_target]


def _route_after_human_review(state: AgentState) -> str:
    """人工审核结果只能通向持久化，被驳回的报告绝不会被隐式重新生成。"""
    return "persist_trace"


# ============================================================
# Graph 构建
# ============================================================

def build_chat_graph(
    checkpointed: bool = False,
    checkpointer=None,
    *,
    critic_llm_review_mode: CriticLLMReviewMode | str = CriticLLMReviewMode.FULL,
    critic_llm_thinking_mode: CriticLLMThinkingMode | str = CriticLLMThinkingMode.PROVIDER_DEFAULT,
) -> StateGraph:
    """构建并编译 Target 版 LangGraph 聊天工作流。

    主要流程：
        1. 用 StateGraph(AgentState) 声明图骨架——此刻只描述「共享状态 + 节点 + 边」，不运行任何 Agent；
        2. 注册全部节点：一律经 instrument_node 外壳包装，保证预算、契约校验、异常转 AgentError、耗时记录；
        3. 定义无条件边：init_state → resolve_explorer_context → load_session_memory → triage 等固定走向；
        4. 定义条件边：各 _route_* 函数按 state 字段决定下一站（含受治理诊断循环与 critic 返工路由）；
        5. 编译返回图实例：按 checkpointed 参数决定是否启用 checkpoint（供 HITL 中断恢复）。

    协作边设计：
    - triage → 条件路由：正常时到首个 Specialist，有 error 时走 critic 快速路径。
    - metric_definition → report_generation 去 data_query 拉数据，其他直接 generate_answer。
    - anomaly_analysis 在 premise_check 后进入唯一的 governed diagnostic 循环。
    - report → generate_answer。
    - generate_answer → critic（审核 final answer，而非中间产物）。
    - critic → 条件路由：approved→persist_trace，human_review→人工审核，needs_revision→定向返工到目标节点。
    """

    # StateGraph 只描述“共享状态 + 节点 + 边”；此时不会运行任何 Agent。
    review_mode = CriticLLMReviewMode.parse(critic_llm_review_mode)
    thinking_mode = CriticLLMThinkingMode.parse(critic_llm_thinking_mode)

    def init_state_with_review_mode(state: AgentState) -> AgentState:
        state["critic_llm_review_mode"] = review_mode.value
        state["critic_llm_thinking_config"] = thinking_mode.trace_summary()
        return init_state(state)

    def critic_with_review_mode(state: AgentState) -> AgentState:
        state["critic_llm_review_mode"] = review_mode.value
        state["critic_llm_thinking_config"] = thinking_mode.trace_summary()
        return critic(
            state,
            llm_review_mode=review_mode,
            llm_thinking_mode=thinking_mode,
        )

    graph = StateGraph(AgentState)

    # ---- 注册节点 ----
    # instrument_node 是统一外壳：它负责预算、契约校验、异常转 AgentError、耗时记录。
    # 因而不要在此处直接注册原始 node，否则该节点会绕过这些运行时保障。
    graph.add_node("init_state", instrument_node("init_state", init_state_with_review_mode))
    graph.add_node("resolve_explorer_context", instrument_node("resolve_explorer_context", resolve_explorer_context))
    graph.add_node("load_session_memory", instrument_node("load_session_memory", load_session_memory))
    graph.add_node("context_compiler", instrument_node("context_compiler", compile_turn_context))
    graph.add_node("semantic_parse", instrument_node("semantic_parse", semantic_parse))
    graph.add_node("intent_router", instrument_node("intent_router", intent_router))
    graph.add_node("complex_plan", instrument_node("complex_plan", complex_plan))
    graph.add_node("complex_clarification", instrument_node("complex_clarification", complex_clarification))
    graph.add_node("context_clarification", instrument_node("context_clarification", context_clarification))
    graph.add_node("triage", instrument_node("triage", triage))
    graph.add_node("retrieve_analysis_memories", instrument_node("retrieve_analysis_memories", retrieve_analysis_memories))
    graph.add_node("metric_definition", instrument_node("metric_definition", metric_definition))
    graph.add_node("seller_catalog", instrument_node("seller_catalog", seller_catalog))
    graph.add_node("seller_scope_unsupported", instrument_node("seller_scope_unsupported", seller_scope_guard))
    graph.add_node("data_query", instrument_node("data_query", _with_evidence_refresh(data_query)))
    graph.add_node("premise_check", instrument_node("premise_check", premise_check))
    graph.add_node("premise_confirmation", instrument_node("premise_confirmation", premise_confirmation))
    graph.add_node("diagnostic_plan", instrument_node("diagnostic_plan", diagnostic_plan))
    graph.add_node("diagnostic_decide", instrument_node("diagnostic_decide", diagnostic_decide))
    graph.add_node("policy_execute", instrument_node("policy_execute", policy_execute))
    graph.add_node("observe", instrument_node("observe", _with_evidence_refresh(observe)))
    graph.add_node("diagnostic_finalize", instrument_node("diagnostic_finalize", diagnostic_finalize))
    graph.add_node("report", instrument_node("report", report))
    graph.add_node("generate_answer", instrument_node("generate_answer", generate_answer))
    graph.add_node("critic", instrument_node("critic", critic_with_review_mode))
    graph.add_node("human_review", instrument_node("human_review", human_review))
    graph.add_node("persist_trace", instrument_node("persist_trace", persist_trace))
    graph.add_node("persist_memory_p0", instrument_node("persist_memory_p0", persist_memory_p0))

    # ---- 边定义 ----
    # add_edge 是无条件跳转；add_conditional_edges 会在“源节点完成后”调用对应
    # _route_* 函数，再按其返回的字符串选择目标节点。
    graph.add_edge(START, "init_state")
    # 浏览上下文节点在未启用时无副作用；启用时重新查库，不信任浏览器结果。
    graph.add_edge("init_state", "resolve_explorer_context")
    graph.add_conditional_edges("resolve_explorer_context", _route_after_explorer_context, {"triage": "load_session_memory", "critic": "critic", "diagnostic_finalize": "diagnostic_finalize", "generate_answer": "generate_answer"})
    graph.add_edge("load_session_memory", "context_compiler")
    graph.add_edge("context_compiler", "semantic_parse")
    graph.add_conditional_edges(
        "semantic_parse",
        _route_after_semantic_parse,
        {"triage": "triage", "intent_router": "intent_router"},
    )
    graph.add_conditional_edges(
        "intent_router",
        _route_after_intent_router,
        {
            "triage": "triage",
            "retrieve_analysis_memories": "retrieve_analysis_memories",
            "data_query": "data_query",
            "context_clarification": "context_clarification",
            "seller_scope_unsupported": "seller_scope_unsupported",
            "complex_plan": "complex_plan",
            "complex_clarification": "complex_clarification",
            "critic": "critic",
            "diagnostic_finalize": "diagnostic_finalize",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "complex_plan",
        _route_after_complex_plan,
        {
            "complex_clarification": "complex_clarification",
            "persist_trace": "persist_trace",
        },
    )
    graph.add_edge("complex_clarification", "persist_trace")

    # Seller-scoped turns bypass generic analysis memory retrieval. No memory
    # path may reinterpret the signed scope or supply cross-seller evidence.
    graph.add_conditional_edges(
        "triage", _route_after_triage_for_shadow,
        {
            "intent_router": "intent_router",
            "retrieve_analysis_memories": "retrieve_analysis_memories",
            "data_query": "data_query",
            "context_clarification": "context_clarification",
            "seller_scope_unsupported": "seller_scope_unsupported",
            "diagnostic_finalize": "diagnostic_finalize",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "retrieve_analysis_memories",
        _route_from_triage,
        {"metric_definition": "metric_definition", "data_query": "data_query", "seller_catalog": "seller_catalog", "premise_check": "premise_check", "context_clarification": "context_clarification", "critic": "critic", "diagnostic_finalize": "diagnostic_finalize", "generate_answer": "generate_answer", "seller_scope_unsupported": "seller_scope_unsupported"},
    )
    graph.add_edge("seller_scope_unsupported", "persist_trace")
    graph.add_edge("context_clarification", "persist_trace")
    # Store catalog output is deterministic and already tenant-filtered; it does not need
    # an LLM answer/critic pass and cannot alter the active analysis scope.
    graph.add_edge("seller_catalog", "persist_trace")

    # metric_definition → data_query（report_gen）或 generate_answer
    graph.add_conditional_edges(
        "metric_definition",
        _route_after_metric_definition,
        {"data_query": "data_query", "generate_answer": "generate_answer", "critic": "critic", "diagnostic_finalize": "diagnostic_finalize", "seller_scope_unsupported": "seller_scope_unsupported"},
    )

    # data_query → premise_check（anomaly）/ report（report_gen）/ generate_answer
    graph.add_conditional_edges(
        "data_query",
        _route_after_data_query,
        {
            "premise_check": "premise_check",
            "diagnostic_plan": "diagnostic_plan",
            "report": "report",
            "generate_answer": "generate_answer",
            "critic": "critic",
            "diagnostic_finalize": "diagnostic_finalize",
            "seller_scope_unsupported": "seller_scope_unsupported",
        },
    )
    graph.add_conditional_edges(
        "premise_check",
        _route_after_premise_check,
        {
            "premise_confirmation": "premise_confirmation",
            "persist_trace": "persist_trace",
            "diagnostic_plan": "diagnostic_plan",
            "critic": "critic",
            "diagnostic_finalize": "diagnostic_finalize",
            "generate_answer": "generate_answer",
            "seller_scope_unsupported": "seller_scope_unsupported",
        },
    )
    graph.add_conditional_edges(
        "premise_confirmation",
        _route_after_premise_confirmation,
        {"persist_trace": "persist_trace"},
    )
    graph.add_conditional_edges(
        "diagnostic_plan", _route_after_diagnostic_plan,
        {"diagnostic_decide": "diagnostic_decide", "diagnostic_finalize": "diagnostic_finalize", "seller_scope_unsupported": "seller_scope_unsupported"},
    )
    graph.add_conditional_edges(
        "diagnostic_decide", _route_after_diagnostic_decide,
        {"policy_execute": "policy_execute", "diagnostic_finalize": "diagnostic_finalize", "seller_scope_unsupported": "seller_scope_unsupported"},
    )
    graph.add_conditional_edges(
        "policy_execute", _route_after_policy,
        {"diagnostic_decide": "diagnostic_decide", "diagnostic_finalize": "diagnostic_finalize", "observe": "observe", "seller_scope_unsupported": "seller_scope_unsupported"},
    )
    graph.add_conditional_edges("observe", _route_after_observe, {"diagnostic_decide": "diagnostic_decide", "diagnostic_finalize": "diagnostic_finalize", "seller_scope_unsupported": "seller_scope_unsupported"})
    graph.add_edge("diagnostic_finalize", "generate_answer")
    graph.add_edge("report", "generate_answer")

    # generate_answer → critic（审核最终 answer）
    graph.add_conditional_edges(
        "generate_answer",
        _route_after_generate_answer,
        {"persist_trace": "persist_trace", "critic": "critic"},
    )

    # critic → 条件路由：approved→落盘 / needs_revision→定向返工 / rejected→triage
    graph.add_conditional_edges(
        "critic",
        _route_from_critic,
        {
            "persist_trace": "persist_trace",
            "metric_definition": "metric_definition",
            "data_query": "data_query",
            "diagnostic_plan": "diagnostic_plan",
            "report": "report",
            "generate_answer": "generate_answer",
            "human_review": "human_review",
            "triage": "triage",
        },
    )
    graph.add_conditional_edges(
        "human_review", _route_after_human_review,
        {"persist_trace": "persist_trace"},
    )

    # Trace 成功落盘后才允许写 Memory，让每条长期结论都能回指已存在的 Trace。
    graph.add_edge("persist_trace", "persist_memory_p0")
    graph.add_edge("persist_memory_p0", END)

    # 编译后才得到可 invoke/stream 的图实例。启用 checkpoint 时，thread_id 可用于
    # 人工审核中断后的恢复；未启用时则适配离线评测和普通单次调用。
    return graph.compile(checkpointer=checkpointer or (get_checkpointer() if checkpointed else None))


# ============================================================
# API 入口
# ============================================================

def chat_context_scope_fingerprint(context: TenantContext, authorization_boundary: dict | None = None) -> str:
    """Fingerprint the server-issued identity and grants for conversation continuity.

    Explorer requests use their canonical query-scope fingerprint. Ordinary chat
    turns still need a continuity boundary, but it is derived only from the
    trusted TenantContext and changes when identity, role, or seller grants do.
    """
    boundary = authorization_boundary or {}
    if boundary.get("query_scope_fingerprint"):
        return str(boundary["query_scope_fingerprint"])
    return scope_fingerprint({
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        "scope_type": context.scope_type,
        "role": context.role,
        "seller_ids": [
            {"seller_id": grant.seller_id, "start_date": grant.start_date, "end_date": grant.end_date}
            for grant in context.seller_grants
        ] or list(context.allowed_seller_ids),
        "seller_id": context.seller_id,
    })


# Backward-compatible internal import used by older tests and call sites.
_chat_context_scope_fingerprint = chat_context_scope_fingerprint


def _turn_context_slots(result: dict) -> dict:
    slots = dict((result.get("context_compilation") or {}).get("known_slots") or {})
    tool_result = result.get("tool_result") or {}
    batch = tool_result.get("batch_results") or [tool_result]
    for item in batch:
        metric_result = ((item or {}).get("result") or {}).get("metric_result") or {}
        period_start = metric_result.get("period_start_date")
        period_end = metric_result.get("period_end_date")
        if period_start and period_end and "period" not in slots:
            slots["period"] = {
                "value": {"start_date": str(period_start), "end_date": str(period_end)},
                "source": "current_query_evidence", "confidence": 1.0,
            }
        metric_date = metric_result.get("metric_date")
        if metric_date and not (period_start and period_end) and "date" not in slots:
            slots["date"] = {
                "value": str(metric_date), "source": "current_query_evidence",
                "confidence": 1.0,
            }
            break
    return slots


def _pending_metric_clarification(decision: dict | None) -> dict | None:
    """Return only planner outcomes that need another user decision.

    A capability ``unsupported`` outcome is a completed, fail-closed answer.
    Putting it in the SSE ``clarification`` field makes the API lifecycle treat
    it as an unfinished user interaction, even though no input can change the
    registered capability.
    """
    if not isinstance(decision, dict):
        return None
    if decision.get("status") in {"clarification_required", "confirmation_required"}:
        return decision
    return None


def _persist_chat_turn(context: TenantContext, session_id: str | None, turn_id: str | None,
                       question: str, result: dict, scope: dict | None,
                       task_id: str | None = None, *, persistence_turn_id: str | None = None,
                       authorization_boundary: dict | None = None,
                       context_scope_fingerprint: str | None = None) -> bool:
    """把一次成功的聊天轮次写入会话历史（best-effort，不阻塞主流程）。

    函数作用：把用户问题与最终答案存档到会话历史表，供前端多轮会话回放。
    输入：context 租户上下文；session_id/turn_id 会话与轮次标识；question 用户问题；
         result 图执行结果；scope 分析范围；task_id 会话任务 ID。
    输出：无。历史库不可用时绝不掩盖答案——只记日志，主流程照常返回。
    主要流程：
        1. 前置校验：缺会话/轮次标识、图有错误、或没有有效 answer 时直接放弃存档；
        2. 调用 ChatHistoryRepository.save_turn 写入一行历史；
        3. 写入异常只记日志不向上抛，保证历史是辅助能力而非主流程故障点。
    """
    if not session_id or not (persistence_turn_id or turn_id):
        return True
    # Durable scoped requests are committed only by LifecycleService at the
    # API boundary. Keeping this best-effort writer for them created a second
    # writer and could leave the final_execution_id unset or overwrite status.
    # The skip is keyed on a bound lifecycle execution, not merely on the
    # presence of a persistence turn id: direct (non-durable) callers may pass
    # a server-assigned turn id and still rely on this best-effort history
    # write, as the WP2 real-chain verifier does.
    if persistence_turn_id and (authorization_boundary or {}).get("execution_id"):
        return True
    try:
        errors = result.get("errors") or []
        planning = result.get("metric_planning_decision") or {}
        from app.chat_history.safe_projection import (
            safe_analysis_presentation, safe_clarification,
        )

        clarification = safe_clarification(
            result.get("clarification")
            or result.get("complex_clarification")
            or _pending_metric_clarification(planning)
        )
        presentation = safe_analysis_presentation(
            result.get("presentation") or result.get("analysis_presentation")
        )
        clarification_status = (clarification or {}).get("status")
        requested_status = result.get("_status")
        if requested_status in {"queued", "running", "suspended", "interrupted"}:
            # Legacy chat_turns has no nonterminal `suspended` turn state. Keep
            # the graph checkpoint resumable while marking the completed HTTP
            # turn terminal; interaction_state carries the pending user action.
            status = "interrupted" if requested_status == "suspended" else requested_status
        elif errors:
            status = result_terminal_status(result)
        elif clarification_status == "complex_scope_required":
            # A scope prompt is an unfinished interaction, not a completed
            # analysis. Keep the legacy turn terminal while preserving the
            # awaiting_scope state so history can restore its scope action.
            status = "interrupted"
        elif clarification and clarification_status != "unsupported":
            # The streaming adapter exposes every non-unsupported clarification
            # as a suspended interaction. Keep the legacy history row aligned:
            # it must not claim a completed analysis when it is waiting for
            # user input (for example, an incomplete complex-analysis request).
            status = "abstained"
        elif _pending_metric_clarification(planning) is not None:
            status = "abstained"
        elif (
            planning.get("status") == "unsupported"
            or (presentation or {}).get("status") == "unsupported"
        ):
            # Unsupported capability requests are terminal safe refusals. They
            # must not be persisted as successful completed chat turns merely
            # because they require no follow-up input.
            status = "abstained"
        else:
            status = "completed"
        interaction_state = "none"
        if planning.get("status") in {"clarification_required", "confirmation_required"}:
            interaction_state = "awaiting_metric_confirmation"
        elif clarification and clarification_status != "unsupported":
            interaction_state = "awaiting_human_input"
        if requested_status == "suspended":
            interaction_state = "awaiting_human_input"
        if clarification_status == "complex_scope_required" and not errors:
            interaction_state = "awaiting_scope"
        authz_boundary = authorization_boundary or {}
        ChatHistoryRepository(context).save_turn(
            session_id, persistence_turn_id or turn_id, question, str(result.get("answer") or "") or None,
            str(result.get("task_type") or "unknown"), str(result.get("trace_id") or ""), scope, task_id,
            client_turn_id=turn_id,
            status=status,
            interaction_state=interaction_state,
            error_code=(errors[-1].get("code") if errors else None),
            error_detail=(errors[-1] if errors else None),
            result_snapshot={
                "task_type": result.get("task_type"),
                "answer": result.get("answer"),
                "clarification": clarification,
                **({"presentation": presentation} if presentation is not None else {}),
                "result_context": build_result_context(result),
                "turn_context": result.get("turn_context") or {
                    "authorized_metrics": result.get("authorized_metrics") or [],
                    "metric_planning_decision": result.get("metric_planning_decision") or {},
                    "context_slots": _turn_context_slots(result),
                    "scope": scope or {},
                },
            },
            context_snapshot={
                **({"scope": scope} if scope else {}),
                **({"turn_resolution": result["turn_resolution"]} if result.get("turn_resolution") else {}),
            },
            permission_version=authz_boundary.get("permission_version"),
            policy_set_version=authz_boundary.get("policy_set_version"),
            authz_ref=authz_boundary.get("authz_ref"),
            authz_digest=authz_boundary.get("authorization_fingerprint"),
            scope_fingerprint=authz_boundary.get("query_scope_fingerprint") or context_scope_fingerprint or chat_context_scope_fingerprint(context, authz_boundary),
        )
        return True
    except Exception:
        LOGGER.exception("Unable to persist chat history")
        return False


def _previous_turn_context(
    context: TenantContext,
    session_id: str | None,
    exclude_turn_id: str | None,
    scope_fingerprint: str | None,
    source_turn_id: str | None = None,
) -> dict:
    """Load one authorized same-scope predecessor, never its answer as authority."""
    if not session_id or not scope_fingerprint:
        return {}
    try:
        return ChatHistoryRepository(context).latest_completed_context(
            session_id,
            exclude_turn_id=exclude_turn_id,
            source_turn_id=source_turn_id,
            scope_fingerprint=scope_fingerprint,
        ) or {}
    except Exception:
        LOGGER.exception("Unable to load previous turn context")
        return {}


def _task_progress_context(
    context: TenantContext,
    task_id: str | None,
    scope_fingerprint: str,
    *,
    trusted_query_scope: bool,
) -> dict:
    """Load only a task snapshot whose recorded scope matches this request."""
    if not getattr(get_settings(), "task_continuity_enabled", True):
        return {}
    if not trusted_query_scope or not task_id or not scope_fingerprint:
        return {}
    try:
        from app.tasks.service import get_task

        task = get_task(context, task_id)
        progress = dict((task or {}).get("task_state") or {})
        if progress.get("scope_fingerprint") != scope_fingerprint:
            return {}
        progress.setdefault("task_id", task_id)
        progress.setdefault("task_state_version", int((task or {}).get("task_state_version") or 0))
        return progress
    except Exception:  # noqa: BLE001 - task summary is optional historical context
        LOGGER.exception("Unable to load task progress context")
        return {}


def _premise_confirmation(message: str) -> bool:
    text = message.casefold().strip()
    if text in {"确认", "继续", "好的", "好", "可以", "是", "yes", "ok", "continue", "proceed"}:
        return True
    return any(token in text for token in ("按实际", "按真实", "分析实际上升", "分析实际下降"))


def _pending_premise(
    graph,
    config: dict,
    context: TenantContext,
    session_id: str | None,
    scope_fingerprint: str,
    authorization_boundary: dict | None = None,
) -> dict | None:
    try:
        checkpoint = graph.get_state(config)
    except Exception:
        return None
    saved = checkpoint.values if checkpoint else {}
    if saved.get("awaiting_user_confirmation") and saved.get("premise_check", {}).get("status") == "mismatch":
        saved_boundary = saved.get("authorization_boundary") or {}
        current_boundary = authorization_boundary or {}
        identity_matches = (
            saved.get("tenant_id") == context.tenant_id
            and saved.get("user_id") == context.user_id
            and saved.get("chat_session_id") == session_id
            and str(saved.get("current_query_scope_fingerprint") or "") == scope_fingerprint
        )
        boundary_matches = all(
            str(saved_boundary.get(key) or "") == str(current_boundary.get(key) or "")
            for key in (
                "permission_version", "policy_set_version",
                "authorization_fingerprint", "query_scope_fingerprint",
            )
        )
        if not identity_matches or not boundary_matches:
            # Never run a stale checkpoint as a new request: LangGraph may
            # otherwise resume the interrupted node under a different session
            # or authorization snapshot.
            raise PermissionError("Pending premise checkpoint is bound to another session or scope.")
        return saved
    return None


def _premise_interaction_reason(state: dict) -> str | None:
    if (
        state.get("awaiting_user_confirmation")
        and (state.get("premise_check") or {}).get("status") == "mismatch"
    ):
        return "premise_direction_mismatch"
    return None


def _notify_result_context(result: dict, callback=None) -> None:
    """Hand the minimal result projection to the durable commit boundary."""

    if not callable(callback):
        return
    try:
        context = build_result_context(result)
        if context:
            callback(context)
    except Exception:  # noqa: BLE001 - result-context persistence is auxiliary
        LOGGER.exception("Unable to build result context")


def run_chat_workflow(message: str, thread_id: str | None = None, tenant_context: TenantContext | None = None, explorer_context: dict | None = None, use_current_explorer_context: bool = False, memory_session_id: str | None = None, memory_enabled: bool = True, metric_confirmation: list[str] | None = None, chat_session_id: str | None = None, chat_turn_id: str | None = None, task_id: str | None = None, semantic_mapping_ids: list[str] | None = None, skipped_semantic_mapping_ids: list[str] | None = None, persistence_turn_id: str | None = None, authorization_boundary: dict | None = None, trace_id: str | None = None, inherited_from_turn_id: str | None = None, *, turn_relation_decision: dict | None = None, seller_scope_payload: dict | None = None, on_result_context=None, _critic_llm_review_mode: CriticLLMReviewMode | str = CriticLLMReviewMode.FULL, _critic_llm_thinking_mode: CriticLLMThinkingMode | str = CriticLLMThinkingMode.PROVIDER_DEFAULT, _llm_call_config: LLMCallConfig | None = None, _evaluation_tool_surface: dict | None = None) -> dict[str, object]:
    """同步执行完整工作流，供普通 ``POST /chat`` 使用。

    函数作用：同步跑完整张图并返回聚合结果，供前端一次性拿到答案。
    输入：message 用户问题；thread_id/explorer_context/metric_confirmation 等可选上下文参数。
    输出：dict —— task_type、answer、trace_id；指标未确认时含 clarification；有卡片数据时含 presentation。
    主要流程：
        1. ensure_safe_prompt 校验输入安全性（防 prompt 注入）；
        2. 构建带 checkpoint 的图，生成 trace_id 并确定可恢复的 thread_id；
        3. 注入租户上下文后 graph.invoke 同步执行——节点选择完全由图的条件边决定；
        4. best-effort 保存本轮会话历史（失败不影响返回）；
        5. 组装返回体：clarification 仅当指标规划 status 非 confirmed 时返回，presentation 仅当存在时返回。
    """
    # 校验输入安全性，防止 prompt 注入。
    ensure_safe_prompt(message)
    # 构建带 checkpoint 的图；未指定 thread_id 时用 trace_id 充当可恢复会话键。
    review_mode = CriticLLMReviewMode.parse(_critic_llm_review_mode)
    thinking_mode = CriticLLMThinkingMode.parse(_critic_llm_thinking_mode)
    call_config = _llm_call_config or LLMCallConfig.normal(get_settings())
    if not isinstance(call_config, LLMCallConfig):
        raise ValueError("_llm_call_config must be a validated internal LLMCallConfig.")
    if _evaluation_tool_surface is not None:
        from evaluation.resume_bg_tool_surface import validate_evaluation_tool_surface
        if not validate_evaluation_tool_surface(_evaluation_tool_surface):
            raise ValueError("_evaluation_tool_surface failed its frozen hash check.")
    graph = build_chat_graph(
        checkpointed=True,
        critic_llm_review_mode=review_mode,
        critic_llm_thinking_mode=thinking_mode,
    )
    context = tenant_context or TenantContext(tenant_id="platform", user_id=get_settings().user_id, role="admin", scope_type="platform")
    context_scope_fp = chat_context_scope_fingerprint(context, authorization_boundary)
    # 每次后端调用都生成一个新的 trace_id，用于标识本次运行和本次观测记录。
    trace_id = trace_id or str(uuid4())
    # 调用方传了 thread_id 就复用该 checkpoint 线程；没有传就用本次 trace_id 临时创建线程。
    resolved_thread_id = thread_id or trace_id
    # LangGraph 只从 configurable.thread_id 查找和保存 checkpoint。
    config = {"configurable": {"thread_id": resolved_thread_id}}
    # 只有调用方明确传入 thread_id 时才检查旧线程是否停在待确认状态。
    saved_premise = _pending_premise(
        graph, config, context, chat_session_id, context_scope_fp,
        authorization_boundary,
    ) if thread_id else None
    # 旧线程存在待确认前提，且当前问题满足确认语义时，才走 resume 分支。
    resuming_premise = bool(saved_premise and _premise_confirmation(message))
    if saved_premise and not resuming_premise:
        # 用户没有确认前提时，直接返回原来的澄清现场，不重复执行整张图。
        result = dict(saved_premise)
        _persist_chat_turn(context, chat_session_id, chat_turn_id, message, result, explorer_context, task_id, persistence_turn_id=persistence_turn_id, authorization_boundary=authorization_boundary, context_scope_fingerprint=context_scope_fp)
        return {
            "task_type": result.get("task_type", "anomaly_analysis"), "answer": result.get("answer", ""),
            "trace_id": result.get("trace_id", resolved_thread_id), "clarification": None, "presentation": None,
        }
    # resume 使用 Command，只把新确认消息交给中断点；普通新请求则构造完整初始 state。
    graph_input: object = Command(resume={"action": "confirm", "message": message}) if resuming_premise else {
        "user_message": message, "trace_id": trace_id, "thread_id": resolved_thread_id,
        "explorer_context": explorer_context, "use_current_explorer_context": use_current_explorer_context,
        "seller_scope_payload": seller_scope_payload,
        "memory_session_id": memory_session_id, "memory_enabled": memory_enabled,
        "chat_session_id": chat_session_id, "chat_turn_id": chat_turn_id,
        "task_id": task_id,
        "metric_confirmation": metric_confirmation or [],
        "semantic_mapping_ids": semantic_mapping_ids or [],
        "skipped_semantic_mapping_ids": skipped_semantic_mapping_ids or [],
        "authorization_boundary": authorization_boundary or {},
        "current_query_scope_fingerprint": context_scope_fp,
        "task_progress_context": _task_progress_context(
            context, task_id, context_scope_fp,
            trusted_query_scope=bool((authorization_boundary or {}).get("query_scope_fingerprint")),
        ),
        "turn_relation_decision": turn_relation_decision or {},
        "lifecycle_execution_id": (authorization_boundary or {}).get("execution_id") or "",
        "llm_call_config": call_config.to_state(),
        **({"evaluation_tool_surface": _evaluation_tool_surface} if _evaluation_tool_surface is not None else {}),
        "llm_call_config_summary": call_config.trace_summary(),
        "critic_llm_thinking_config": thinking_mode.trace_summary(),
        "persistence_turn_id": persistence_turn_id or chat_turn_id or "",
        "previous_turn_context": _previous_turn_context(
            context, chat_session_id, persistence_turn_id or chat_turn_id,
            context_scope_fp,
            inherited_from_turn_id,
        ),
        **context.as_state(), "audit_actor": context.user_id,
    }
    if resuming_premise:
        # 恢复时沿用 checkpoint 中原来的 trace_id，保证同一条分析链路不会被拆成两条 Trace。
        trace_id = str(saved_premise.get("trace_id") or trace_id)
        _persist_chat_turn(
            context, chat_session_id, chat_turn_id, message,
            {"_status": "running", "task_type": "pending", "trace_id": trace_id},
            explorer_context, task_id, persistence_turn_id=persistence_turn_id, authorization_boundary=authorization_boundary, context_scope_fingerprint=context_scope_fp,
        )
    # trace_id 标识本次运行；thread_id 标识可恢复的 LangGraph 会话，调用方可复用它。
    if not resuming_premise:
        _persist_chat_turn(
            context, chat_session_id, chat_turn_id, message,
            {"_status": "running", "task_type": "pending", "trace_id": trace_id},
            explorer_context, task_id, persistence_turn_id=persistence_turn_id, authorization_boundary=authorization_boundary, context_scope_fingerprint=context_scope_fp,
        )
    with activate_tenant_context(context), span("graph.run", {"trace_id": trace_id, "thread_id": resolved_thread_id, "tenant_id": context.tenant_id, "gen_ai.operation.name": "invoke_agent_graph"}):
        # 同步执行整张图；节点选择完全由图的条件边决定。
        # graph.invoke 执行节点；每个节点的中间状态由 checkpointer 按 resolved_thread_id 保存。
        try:
            result = graph.invoke(graph_input, config=config, durability=get_settings().checkpoint_durability)
            suspended = bool(result.get("__interrupt__"))
            if suspended:
                # 节点通过 interrupt 暂停时，LangGraph 会把节点已经写入的字段保存到 checkpoint。
                # 这里重新读取合并后的 state，保证 API 返回修正后的答案，而不是只返回
                # 一个不包含业务结果的 interrupt 外壳。
                result = graph.get_state(config).values
                result["_status"] = "suspended"
        finally:
            execution_context.release_turn_budget(trace_id)
    # 尽量保存本轮会话历史（失败不影响返回）。
    # LangGraph 完成后再写聊天历史；写入失败不会改变已经生成的图结果。
    committed = _persist_chat_turn(context, chat_session_id, chat_turn_id, message, result, explorer_context, task_id, persistence_turn_id=persistence_turn_id, authorization_boundary=authorization_boundary, context_scope_fingerprint=context_scope_fp)
    if chat_session_id and chat_turn_id and persistence_turn_id is None and not committed:
        raise RuntimeError("chat_turn_persistence_failed")
    _notify_result_context(result, on_result_context)
    # 组装返回体：只返回前端需要的聚合字段。
    return {
        "task_type": result.get("task_type", "unknown"),
        "answer": result.get("answer", ""),
        "trace_id": result["trace_id"],
        **({"_status": "suspended"} if result.get("_status") == "suspended" else {}),
        "error": bool(result.get("errors")),
        "errors": result.get("errors") or [],
        "clarification": (
            result.get("complex_clarification")
            or _pending_metric_clarification(result.get("metric_planning_decision"))
        ),
        "complex_plan": result.get("complex_plan") or None,
        **({"clarification": {"status": "awaiting_human_review"}} if result.get("_status") == "suspended" else {}),
        "presentation": AnalysisPresentation.model_validate(result["analysis_presentation"]).model_dump(mode="json") if result.get("analysis_presentation") else None,
        "turn_context": {
            "authorized_metrics": result.get("authorized_metrics") or [],
            "metric_planning_decision": result.get("metric_planning_decision") or {},
            "context_slots": _turn_context_slots(result),
            "scope": explorer_context or {},
        },
        # Kept for the durable lifecycle commit; callers must not expose this
        # internal context object as user-facing answer text.
        "turn_resolution": result.get("turn_resolution") or {},
    }


def stream_chat_workflow(message: str, thread_id: str | None = None, tenant_context: TenantContext | None = None, explorer_context: dict | None = None, use_current_explorer_context: bool = False, memory_session_id: str | None = None, memory_enabled: bool = True, metric_confirmation: list[str] | None = None, chat_session_id: str | None = None, chat_turn_id: str | None = None, task_id: str | None = None, trace_id: str | None = None, emit_started: bool = True, semantic_mapping_ids: list[str] | None = None, skipped_semantic_mapping_ids: list[str] | None = None, persistence_turn_id: str | None = None, authorization_boundary: dict | None = None, inherited_from_turn_id: str | None = None, *, turn_relation_decision: dict | None = None, on_turn_resolution=None, on_result_context=None, seller_scope_payload: dict | None = None):
    """以 SSE 友好的生成器方式运行图，并推送节点级进度。

    函数作用：把图的节点级进度以 SSE 事件流推给前端，供流式聊天使用。
    输入：同 run_chat_workflow（额外无参，返回生成器而非 dict）。
    输出：生成器，逐个 yield 序列化的 SSE 事件 dict —— started → node_* / heartbeat → completed / failed。
    主要流程：
        1. 校验输入、构建带 checkpoint 的图，确定 trace_id/thread_id；
        2. 启动后台线程 run_graph 执行图：stream_mode="updates" 每完成一个节点就入队节点事件，
           triage 节点完成后额外推送一次执行计划；
        3. 主线程从队列读事件并 yield：超时 2 秒无新事件且未见终态时发心跳保活；
        4. 终态（completed/failed）后丢弃后续事件；队列哨兵 None 结束循环；
        5. 流结束仍未见到终态时，合成一个 failed 事件兜底，避免前端永远等待。
    """
    ensure_safe_prompt(message)
    graph = build_chat_graph(checkpointed=True)
    context = tenant_context or TenantContext(tenant_id="platform", user_id=get_settings().user_id, role="admin", scope_type="platform")
    context_scope_fp = chat_context_scope_fingerprint(context, authorization_boundary)
    trace_id = trace_id or str(uuid4())
    resolved_thread_id = thread_id or trace_id
    config = {"configurable": {"thread_id": resolved_thread_id}}
    saved_premise = _pending_premise(
        graph, config, context, chat_session_id, context_scope_fp,
        authorization_boundary,
    ) if thread_id else None
    resuming_premise = bool(saved_premise and _premise_confirmation(message))
    if saved_premise and not resuming_premise:
        if emit_started:
            yield SseStartedEvent(trace_id=str(saved_premise.get("trace_id") or trace_id), sequence_number=0).model_dump(mode="json")
        yield SseCompletedEvent(
            task_type=saved_premise.get("task_type", "anomaly_analysis"),
            answer=saved_premise.get("answer", ""), trace_id=str(saved_premise.get("trace_id") or trace_id),
            error=False, error_detail=None, clarification=None, presentation=None,
        ).model_dump(mode="json")
        return
    if resuming_premise:
        trace_id = str(saved_premise.get("trace_id") or trace_id)
        initial_state: object = Command(resume={"action": "confirm", "message": message})
    else:
        initial_state = {
        "user_message": message, "trace_id": trace_id, "thread_id": resolved_thread_id,
        "explorer_context": explorer_context, "use_current_explorer_context": use_current_explorer_context,
        "seller_scope_payload": seller_scope_payload,
        "memory_session_id": memory_session_id, "memory_enabled": memory_enabled,
        "chat_session_id": chat_session_id, "chat_turn_id": chat_turn_id,
        "task_id": task_id,
        "metric_confirmation": metric_confirmation or [],
        "semantic_mapping_ids": semantic_mapping_ids or [],
        "skipped_semantic_mapping_ids": skipped_semantic_mapping_ids or [],
        "authorization_boundary": authorization_boundary or {},
        "current_query_scope_fingerprint": context_scope_fp,
        "task_progress_context": _task_progress_context(
            context, task_id, context_scope_fp,
            trusted_query_scope=bool((authorization_boundary or {}).get("query_scope_fingerprint")),
        ),
        "turn_relation_decision": turn_relation_decision or {},
        "lifecycle_execution_id": (authorization_boundary or {}).get("execution_id") or "",
        "persistence_turn_id": persistence_turn_id or chat_turn_id or "",
        "previous_turn_context": _previous_turn_context(
            context, chat_session_id, persistence_turn_id or chat_turn_id,
            context_scope_fp,
            inherited_from_turn_id,
        ),
        **context.as_state(), "audit_actor": context.user_id,
        }
    # 用队列在后台线程与 HTTP 生成器之间传递事件，隔离 LangGraph 同步执行与 SSE 输出。
    _persist_chat_turn(
        context, chat_session_id, chat_turn_id, message,
        {"_status": "running", "task_type": "pending", "trace_id": trace_id},
        explorer_context, task_id, persistence_turn_id=persistence_turn_id, authorization_boundary=authorization_boundary, context_scope_fingerprint=context_scope_fp,
    )
    events: Queue[dict | None] = Queue()

    def run_graph() -> None:
        """在独立线程中执行工作流，每完成一个节点就向队列放入事件。"""
        execution_context.set_graph_event_queue(events)
        try:
            with activate_tenant_context(context), span("graph.stream", {"trace_id": trace_id, "thread_id": resolved_thread_id, "tenant_id": context.tenant_id}):
                # stream_mode="updates"：每次只给出“刚完成节点”写入的 state 增量，
                # 因此适合向前端展示节点进度，而不是把完整 state 重复传输。
                for update in graph.stream(initial_state, config=config, stream_mode="updates", durability=get_settings().checkpoint_durability):
                    for node_name, node_state in update.items():
                        # LangGraph emits interrupts as a control update whose value
                        # is a tuple of Interrupt objects, not a node state mapping.
                        # The checkpoint below is authoritative for the suspended
                        # state; do not treat this control event as node progress.
                        if node_name == "__interrupt__":
                            continue
                        if not isinstance(node_state, dict):
                            raise TypeError(f"Unexpected update payload for graph node {node_name!r}.")
                        print(f"[DEBUG/workflow] node_completed={node_name} task_type={node_state.get('task_type')} answer_len={len(node_state.get('answer',''))} errors={bool(node_state.get('errors'))}")
                        # 节点完成事件只携带安全错误摘要，不含 diagnostic_message
                        err_list = node_state.get("errors") or []
                        err_safe = SseNodeError(
                            code=err_list[-1]["code"],
                            public_message=err_list[-1].get("public_message", err_list[-1]["message"]),
                            agent_name=err_list[-1].get("agent_name"),
                        ) if err_list else None
                        events.put(SseNodeCompletedEvent(
                            node=node_name,
                            task_type=node_state.get("task_type"),
                            error=err_safe,
                        ).model_dump(mode="json"))
                        # triage 节点完成后发送执行计划：让前端知道预期要跑哪些 Agent
                        if node_name == "triage":
                            plan = node_state.get("execution_plan") or {}
                            agents = plan.get("agents", []) if isinstance(plan, dict) else []
                            events.put(SsePlanEvent(
                                task_type=node_state.get("task_type"),
                                agents=agents,
                            ).model_dump(mode="json"))
                # 流结束后从 checkpoint 取最终合并状态，不能用最后一个 update 代替它。
                snapshot = graph.get_state(config)
                result = snapshot.values
                if snapshot.next:
                    result["_status"] = "suspended"
                    # Interrupts do not reach the graph's normal persist_trace
                    # node. Preserve the suspended checkpoint under its trace id
                    # so the trace shown to the user can be inspected later.
                    try:
                        write_trace(result.get("trace_id", trace_id), dict(result))
                    except Exception as exc:  # noqa: BLE001 - tracing must not change routing
                        LOGGER.warning(
                            "Unable to persist suspended workflow trace (%s)",
                            type(exc).__name__,
                        )
            committed = _persist_chat_turn(context, chat_session_id, chat_turn_id, message, result, explorer_context, task_id, persistence_turn_id=persistence_turn_id, authorization_boundary=authorization_boundary, context_scope_fingerprint=context_scope_fp)
            if chat_session_id and chat_turn_id and persistence_turn_id is None and not committed:
                events.put(SseFailedEvent(detail="chat_turn_persistence_failed").model_dump(mode="json"))
                return
            if snapshot.next:
                events.put(SseSuspendedEvent(
                    task_type=result.get("task_type", "unknown"),
                    answer=result.get("answer", ""),
                    trace_id=result.get("trace_id", trace_id),
                    interaction_state="awaiting_human_input",
                    interaction_reason_code=_premise_interaction_reason(result),
                    scope_fingerprint=context_scope_fp,
                    clarification=(
                        result.get("metric_planning_decision")
                        if isinstance(result.get("metric_planning_decision"), dict)
                        and result.get("metric_planning_decision", {}).get("status") != "confirmed"
                        else None
                    ),
                ).model_dump(mode="json"))
                return
            # 工作流正常结束，推送完成事件
            errors = result.get("errors") or []
            if not isinstance(errors, list):
                errors = []
            if callable(on_turn_resolution) and isinstance(result.get("turn_resolution"), dict):
                on_turn_resolution(result["turn_resolution"])
            last_err = errors[-1] if errors and isinstance(errors[-1], dict) else None
            error_code = str(
                (last_err or {}).get("code") or result.get("error_code") or ""
            ) or None
            has_terminal_error = bool(errors or result.get("error") or error_code)
            error_detail = (
                (last_err or {}).get("public_message") or (last_err or {}).get("message")
                if has_terminal_error else None
            )
            if has_terminal_error and not error_detail:
                error_detail = "分析过程未能形成可验证结论。"
            planning = result.get("metric_planning_decision") or {}
            pending_clarification = (
                result.get("complex_clarification")
                or _pending_metric_clarification(planning)
            )
            if pending_clarification is not None:
                from app.chat_history.safe_projection import safe_clarification

                pending_clarification = safe_clarification(pending_clarification)
            has_pending_clarification = pending_clarification is not None
            terminal_status = None
            if not has_pending_clarification:
                terminal_status = result_terminal_status(
                    result,
                    default_error_status="failed" if has_terminal_error else "completed",
                )
                # The completed SSE event is terminal by construction; do not
                # serialize an intermediate graph status into its narrower
                # terminal-status enum.
                if terminal_status not in {
                    "completed", "abstained", "failed", "cancelled", "interrupted",
                }:
                    terminal_status = "failed"
                if terminal_status == "completed":
                    terminal_status = None
            _notify_result_context(result, on_result_context)
            events.put(SseCompletedEvent(
                task_type=result.get("task_type", "unknown"),
                answer=result.get("answer", ""),
                trace_id=result.get("trace_id", trace_id),
                error=has_terminal_error,
                error_detail=error_detail,
                terminal_status=terminal_status,
                error_code=error_code,
                clarification=pending_clarification,
                presentation=AnalysisPresentation.model_validate(result["analysis_presentation"]) if result.get("analysis_presentation") else None,
                complex_plan=result.get("complex_plan") or None,
            ).model_dump(mode="json"))
            # 在 run_graph finally 中清除队列引用
        except Exception as exc:
            # 工作流异常终止，推送失败事件
            # 使用 sanitize_exception 分离公开消息与诊断消息，避免敏感信息泄漏到 SSE 前端
            public_msg, diag_msg = sanitize_exception(exc)
            LOGGER.warning("Workflow diagnostic: %s", diag_msg)
            events.put(SseFailedEvent(detail=public_msg).model_dump(mode="json"))
        finally:
            events.put(None)  # 哨兵值，通知生成器结束
            execution_context.set_graph_event_queue(None)
            execution_context.release_turn_budget(trace_id)

    Thread(target=run_graph, name=f"agent-stream-{trace_id}", daemon=True).start()
    terminal_seen = False
    seq = 0
    if emit_started:
        yield SseStartedEvent(trace_id=trace_id, sequence_number=seq).model_dump(mode="json")
        seq += 1
    # 逐一从队列取出事件，分配连续序号，并确保终态唯一
    while True:
        try:
            raw = events.get(timeout=2)
        except Empty:
            if not terminal_seen:
                yield SseHeartbeatEvent(trace_id=trace_id, sequence_number=seq).model_dump(mode="json")
                seq += 1
            continue
        if raw is None:
            break
        # 终态守卫：终态之后的所有事件丢弃
        if terminal_seen:
            LOGGER.warning("SSE state violation: %s after terminal", raw.get("event"))
            continue
        # 终态检测
        if raw["event"] in ("completed", "failed", "suspended"):
            terminal_seen = True
        # 通过 TypeAdapter 校验后分配序号
        try:
            event_model = SSE_EVENT_ADAPTER.validate_python(raw)
        except Exception:
            LOGGER.warning("SSE event validation failed: %s", raw.get("event"))
            continue
        event_model = event_model.model_copy(update={"sequence_number": seq})
        yield event_model.model_dump(mode="json")
        seq += 1
    # 流结束但未见终态时合成 failed
    if not terminal_seen:
        LOGGER.error("SSE stream ended without terminal event")
        synthetic = SseFailedEvent(detail="stream_terminated_without_terminal", sequence_number=seq)
        yield synthetic.model_dump(mode="json")


def resume_human_review(thread_id: str, decision: dict[str, str], tenant_context: TenantContext | None = None) -> dict[str, str]:
    """从检查点恢复中断的工作流，注入人工审核决定后继续执行。

    关键安全约束：
    - 检查点所属租户必须与当前请求 tenant_context 一致（防止跨租户恢复）
    - 已完成的节点不会重放（LangGraph 从中断点继续）

    参数：
    - thread_id：工作流线程 ID（与检查点 key 对应）
    - decision：审核决定字典（含 action, actor, reason, answer）
    - tenant_context：当前请求的租户上下文
    """
    from langgraph.types import Command

    graph = build_chat_graph(checkpointed=True)
    context = tenant_context or TenantContext(tenant_id="platform", user_id=get_settings().user_id, role="admin", scope_type="platform")
    config = {"configurable": {"thread_id": thread_id}}
    # 从检查点读取已保存的状态，定位中断前的执行现场。
    checkpoint = graph.get_state(config)
    saved = checkpoint.values if checkpoint else {}
    # 安全约束：检查点必须属于当前请求租户，防止跨租户恢复。
    if saved.get("tenant_id") != context.tenant_id:
        raise PermissionError("Checkpoint belongs to another tenant")
    # HITL resume is a security boundary as well: the checkpoint's original
    # authorization snapshot must still match the current policy before the
    # graph can touch data or tools again.
    raw_boundary = saved.get("authorization_boundary") or {}
    lifecycle_service = None
    lifecycle_execution = None
    review_interaction_id = None
    interaction_index: str | None = None
    if raw_boundary:
        from app.auth.context import build_authorization_context
        from app.auth.enforcement import AuthorizationBoundary, assert_authorization_boundary

        controlled = saved.get("controlled_analysis_scope") or {}
        current_authz = build_authorization_context(
            context,
            allowed_metric_ids=controlled.get("allowed_metrics") if controlled else None,
            allowed_dimensions=controlled.get("allowed_dimensions") if controlled else None,
            metric_scope_type="seller" if controlled else None,
        )
        assert_authorization_boundary(
            current_authz,
            AuthorizationBoundary(
                permission_version=str(raw_boundary.get("permission_version") or ""),
                policy_set_version=str(raw_boundary.get("policy_set_version") or ""),
                authorization_fingerprint=str(raw_boundary.get("authorization_fingerprint") or ""),
                query_scope_fingerprint=str(raw_boundary.get("query_scope_fingerprint") or ""),
            ),
            current_scope_fingerprint=str(
                saved.get("current_query_scope_fingerprint")
                or raw_boundary.get("query_scope_fingerprint")
                or ""
            ),
        )
        from app.chat_lifecycle.service import LifecycleService

        lifecycle_service = LifecycleService(context, current_authz)
        lifecycle_execution = lifecycle_service.execution_for_thread(thread_id)
        if lifecycle_execution:
            pending_nodes = list(getattr(checkpoint, "next", ()) or ())
            last_decision = saved.get("human_decision") or {}
            last_index = last_decision.get("interaction_index")
            completed_checkpoint = not pending_nodes and last_index is not None
            # A completed checkpoint points at the graph's final node (usually
            # persist_trace), while the durable interaction is still keyed to
            # the interrupt node that accepted the decision.
            node_name = str(pending_nodes[0] if pending_nodes else "human_review")
            if pending_nodes:
                interaction_index = str(int(last_index) + 1 if last_index is not None else 0)
            elif last_index is not None:
                # Retrying the same HTTP request after a successful resume must
                # replay its durable response instead of invoking the node.
                interaction_index = str(last_index)
            else:
                raise RuntimeError("human_review_checkpoint_not_pending")
            claim, review_interaction_id, replay = lifecycle_service.claim_review_interaction(
                lifecycle_execution, node_name=node_name,
                interrupt_index=interaction_index, decision=dict(decision),
                force_reclaim=completed_checkpoint,
            )
            if claim == "replay":
                return {key: str(value or "") for key, value in (replay or {}).items()}
            if claim in {"processing", "claimed"} and completed_checkpoint:
                # The graph checkpoint proves the interrupt node and all
                # downstream work completed, but the PostgreSQL Turn Commit
                # may have failed. Rebuild that commit from the checkpoint
                # without invoking the graph or repeating node side effects.
                recovered_response = {
                    "task_type": str(saved.get("task_type", "unknown")),
                    "answer": str(saved.get("answer", "")),
                    "trace_id": str(saved.get("trace_id", "")),
                }
                recovered_errors = saved.get("errors") or []
                recovered_status = result_terminal_status(saved)
                committed = lifecycle_service.finalize(
                    lifecycle_execution,
                    status=recovered_status,
                    answer=recovered_response["answer"], result=saved,
                    error_code=(
                        str(recovered_errors[-1].get("code"))
                        if recovered_errors and isinstance(recovered_errors[-1], dict) else None
                    ),
                    interaction_state="none",
                    task_id=str(saved.get("task_id") or "") or None,
                    task_status=recovered_status,
                    relation_decision=saved.get("turn_relation_decision") or {},
                    final_node="human_review",
                    review_interaction_id=review_interaction_id,
                    review_interaction_result=recovered_response,
                )
                if not committed:
                    raise RuntimeError("human_review_turn_recovery_not_persisted")
                return recovered_response
            if claim != "claimed":
                raise RuntimeError(f"human_review_{claim}")
            decision = {**decision, "_interaction_index": interaction_index}
        elif not getattr(checkpoint, "next", ()) and saved.get("human_decision"):
            # Legacy checkpoints have no lifecycle row, but a completed review
            # still replays its saved result instead of re-running side effects.
            return {
                "task_type": str(saved.get("task_type", "unknown")),
                "answer": str(saved.get("answer", "")),
                "trace_id": str(saved.get("trace_id", "")),
            }
    if not raw_boundary and not getattr(checkpoint, "next", ()) and saved.get("human_decision"):
        return {
            "task_type": str(saved.get("task_type", "unknown")),
            "answer": str(saved.get("answer", "")),
            "trace_id": str(saved.get("trace_id", "")),
        }
    if not getattr(checkpoint, "next", ()) and not lifecycle_execution:
        raise RuntimeError("human_review_checkpoint_not_pending")
    with activate_tenant_context(context):
        # 以人工审核决定恢复执行；LangGraph 从中断点继续，不重放已完成节点。
        import threading

        stop_lease_renewal = threading.Event()
        claim_token = getattr(lifecycle_service, "_review_claim_tokens", {}).get(review_interaction_id) \
            if lifecycle_service and review_interaction_id else None
        renew_lease = getattr(lifecycle_service, "renew_review_interaction", None) if lifecycle_service else None

        def keep_review_lease() -> None:
            while not stop_lease_renewal.wait(25):
                try:
                    renewed = bool(claim_token and renew_lease and renew_lease(review_interaction_id, claim_token))
                except Exception:  # noqa: BLE001 - lease renewal failure is resolved by expiry/retry
                    renewed = False
                if not renewed:
                    return

        renewal_thread = None
        if claim_token and renew_lease:
            renewal_thread = threading.Thread(target=keep_review_lease, daemon=True)
            renewal_thread.start()
        try:
            result = graph.invoke(Command(resume=decision), config=config, durability=get_settings().checkpoint_durability)
        finally:
            stop_lease_renewal.set()
            if renewal_thread:
                renewal_thread.join(timeout=1)
            execution_context.release_turn_budget(str(saved.get("trace_id") or ""))
    suspended = bool(result.get("__interrupt__"))
    if suspended:
        result = graph.get_state(config).values
    response = {
        "task_type": str(result.get("task_type", "unknown")),
        "answer": str(result.get("answer", "")),
        "trace_id": str(result.get("trace_id", "")),
    }
    if lifecycle_service and lifecycle_execution:
        errors = result.get("errors") or []
        committed = lifecycle_service.finalize(
            lifecycle_execution,
            status="suspended" if suspended else result_terminal_status(result),
            answer=response["answer"], result=result,
            error_code=str(errors[-1].get("code")) if errors and isinstance(errors[-1], dict) else None,
            interaction_state="awaiting_human_input" if suspended else "none",
            task_id=str(saved.get("task_id") or "") or None,
            task_status=None if suspended else result_terminal_status(result),
            relation_decision=saved.get("turn_relation_decision") or {},
            final_node="human_review",
            review_interaction_id=review_interaction_id,
            review_interaction_result=response,
        )
        if not committed:
            raise RuntimeError("human_review_turn_commit_not_persisted")
    return response


# ============================================================
# Graph 辅助节点
# ============================================================

def init_state(state: AgentState) -> AgentState:
    """补齐所有节点共享的可变容器与运行时预算。

    ``setdefault`` 不会覆盖调用方显式传入的值，故测试或上层 API 可以按请求指定预算；
    未指定时才使用 Settings 中的默认值。之后各节点可安全 append 到这些列表。
    """
    # 补齐所有节点共享的可变容器，之后各节点可安全 append。
    state.setdefault("handoffs", [])
    state.setdefault("node_history", [])
    state.setdefault("triage_unresolved", False)
    state.setdefault("tool_calls", [])
    state.setdefault("critic_decisions", [])
    state.setdefault("critic_llm_review_mode", CriticLLMReviewMode.FULL.value)
    default_llm_config = LLMCallConfig.normal(get_settings())
    state.setdefault("llm_call_config", default_llm_config.to_state())
    state.setdefault("llm_call_config_summary", default_llm_config.trace_summary())
    state.setdefault("llm_calls", [])
    state.setdefault("llm_token_summary", {
        "request_status": "no_model_request", "logical_call_count": 0,
        "sdk_attempt_count": 0, "local_send_refusal_count": 0,
        "successful_response_count": 0,
        "provider_exception_count": 0, "parse_failure_count": 0,
        "known_input_tokens_subtotal": 0, "known_output_tokens_subtotal": 0,
        "known_tokens_subtotal": 0, "input_tokens": 0, "output_tokens": 0,
        "total_tokens": 0, "usage_complete": True, "unknown_usage_attempt_count": 0,
    })
    state.setdefault("messages", [])
    state.setdefault("errors", [])
    state.setdefault("evidence_bundles", [])
    state.setdefault("claims", [])
    state.setdefault("pre_critic_answer_snapshots", [])
    state.setdefault("metric_confirmation", [])
    state.setdefault("semantic_mapping_ids", [])
    state.setdefault("skipped_semantic_mapping_ids", [])
    state.setdefault("query_results", [])
    state.setdefault("premise_check", {})
    state.setdefault("awaiting_user_confirmation", False)
    state.setdefault("diagnostic_plan", {})
    state.setdefault("diagnostic_step_state", {})
    state.setdefault("diagnostic_action", {})
    state.setdefault("diagnostic_tool_results", [])
    state.setdefault("diagnostic_round", 0)
    state.setdefault("policy_decision", {})
    state.setdefault("hypotheses", [])
    state.setdefault("observations", [])
    state.setdefault("finish_decision", {})
    state.setdefault("diagnostic_fallback", False)
    state.setdefault("decision_driver", "workflow_fallback")
    state.setdefault("termination_initiator", "policy")
    state.setdefault("diagnostic_target_metric", "")
    state.setdefault("diagnostic_no_progress_rounds", 0)
    state.setdefault("diagnostic_required_action", False)
    state.setdefault("diagnostic_denied_metrics", [])
    state.setdefault("diagnostic_llm_failures", [])
    state.setdefault("agent_run", {})
    state.setdefault("use_current_explorer_context", False)
    state.setdefault("memory_enabled", True)
    state.setdefault("memory_context", "")
    state.setdefault("session_memory_context", "")
    state.setdefault("memory_session_loaded", False)
    state.setdefault("analysis_memories", [])
    state.setdefault("memory_errors", [])
    state.setdefault("previous_turn_context", {})
    state.setdefault("semantic_contract", {})
    state.setdefault("dialogue_context_resolution", {})
    state.setdefault("routing_context", {})
    state["complex_route_canary_selected"] = False
    state["complex_route_canary_observation"] = {}
    state.setdefault("semantic_parse_status", "not_run")
    state.setdefault("semantic_parse_observation", {"mode": "off", "status": "not_run"})
    state.setdefault("semantic_interpreter_observation", {"mode": "off", "status": "not_run"})
    state.setdefault("complex_route_decision", {})
    state.setdefault("route_decision_status", "not_run")
    state.setdefault("complex_plan", {})
    state.setdefault("complex_clarification", {})
    state.setdefault("result_followup", {})
    state.setdefault("result_followup_context", {})
    state.setdefault("chat_session_id", None)
    state.setdefault("chat_turn_id", None)
    state.setdefault("task_id", None)
    state.setdefault("node_timings", [])
    state.setdefault("context_packing_events", [])
    state.setdefault("context_packing_summary", {"calls": 0, "omitted_items": 0, "truncated_items": 0, "required_overflows": 0, "input_tokens": 0, "optional_candidates": 0, "optional_included": 0})
    state.setdefault("triage_context", {})
    state.setdefault("context_bundles", {})
    state.setdefault("context_source_availability", {})
    state.setdefault("context_orchestration_events", [])
    state.setdefault("context_orchestration_summary", {
        "bundles": 0, "items": 0, "included": 0, "omitted": 0,
        "required_overflows": 0, "input_tokens": 0,
    })
    state.setdefault("evidence_refresh", {})
    state.setdefault("turn_resolution", {})
    state.setdefault("turn_relation_decision", {})
    state.setdefault("preference_context", {})
    state.setdefault("task_state_version", 0)
    state.setdefault("rag_review", {})
    state.setdefault("critic_investigation_context", None)
    state.setdefault("runtime_steps", 0)
    state.setdefault("runtime_budget", {})
    # 预算未显式指定时使用 Settings 默认值（setdefault 不会覆盖调用方传入值）。
    state["runtime_budget"].setdefault("max_graph_steps", get_settings().max_graph_steps)
    state["runtime_budget"].setdefault("max_tool_calls", get_settings().max_tool_calls)
    state["runtime_budget"].setdefault("max_latency_ms", get_settings().max_latency_ms)
    state["runtime_budget"].setdefault("max_tokens", get_settings().max_tokens)
    state["runtime_budget"].setdefault("global_max_tokens", state["runtime_budget"]["max_tokens"])
    state["runtime_budget"].setdefault("max_cost", get_settings().max_cost)
    state.setdefault("budget_warnings", [])
    # 这些字段是本次图运行的基线元数据，供 trace、审计和 RAG release 回放使用。
    state["current_agent"] = "init_state"
    state.setdefault("data_version_manifest_id", "")
    state.setdefault("data_version_manifest_hash", "")
    # 以下字段会写入 Trace 信封，用来标识一代可比较的工作流证据。
    # 如果编排逻辑、提示词或 state 合同发生不兼容变化，应有意识地升级这些版本号，
    # 避免把不同版本的运行结果直接混在一起比较。
    state["trace_generation"] = "trace-v2"
    state["trace_kind"] = "agent_chat"
    state["trace_schema_version"] = "2.0"
    state["workflow_version"] = "governed-agent-v1"
    state["prompt_version"] = "v1"
    state["rag_release_version"] = published_rag_version()
    state["permission_policy_version"] = "v1"
    return state


def persist_trace(state: AgentState) -> AgentState:
    """将最终 state（包括错误和节点历史）持久化为 trace。

    这是普通图的最后一站：先写入自己的历史记录，再快照整个 state。
    ``write_trace`` 会按配置写文件或数据库，``/traces/{trace_id}`` 读取的正是该快照。
    """
    # 记录本节点自身的执行历史，再整体落盘。
    state["node_history"].append(
        {"node": "persist_trace", "summary": "Trace persisted to storage."}
    )
    # P4 is a best-effort sidecar over the completed state.  It writes only
    # sampled, contract-anchored executions and must never affect trace or
    # workflow completion.
    try:
        from app.context.shadow_audit import record_completed_shadow_audit

        record_completed_shadow_audit(state)
    except Exception:  # noqa: BLE001 - the audit sidecar cannot change the flow
        pass
    # 把完整 state（含错误与节点历史）写为 trace 文件或数据库记录。
    write_trace(state["trace_id"], dict(state))
    return state
