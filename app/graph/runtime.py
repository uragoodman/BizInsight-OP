"""图节点的运行时保护层：预算控制、节点契约校验、异常隔离与可观测性记录。

该文件不参与图的路由逻辑，而是作为每个节点"外面包的一层"，确保：
1. 所有节点在统一的预算（时间、步骤数）约束下运行
2. 每个节点输出的 state 经过 Pydantic 契约校验，不干净的数据不会污染下游
3. 节点内的任何未捕获异常都会被转成结构化错误继续图流，不会崩溃整个图
4. 每次节点执行的耗时、状态、错误都会通过 OpenTelemetry 记录
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import sqlite3
from pathlib import Path
from time import perf_counter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt

from app.agents.contracts import AgentError, AnswerOutput, AnalysisOutput, CriticDecision, DataQueryOutput, MetricDefinitionOutput, TriageDecision
from app.core.config import get_settings
from app.graph.state import AGENTSTATE_KEYS, AgentState
from app.observability import record_event
from app.core import execution_context
from app.core.execution_context import activate_state, reset_state
from app.core.exceptions import sanitize_exception
from app.schemas.events import SseNodeStartedEvent


Node = Callable[[AgentState], AgentState]
_CHECKPOINTER = None
_CHECKPOINTER_CONTEXT = None
LOGGER = logging.getLogger(__name__)


def get_checkpointer():
    """获取（或创建）全局唯一的 SQLite 检查点实例。

    检查点是 LangGraph 支持 HITL（Human-In-The-Loop）和中断恢复的基础设施：
    - 每个 thread_id 的执行历史被持久化到 SQLite 文件中
    - 当 critic 裁决为 human_review 且生产策略要求人工审批时，图在此中断
    - 外部调用 `Command(resume=...)` 后可从中断点继续，而不需要重新跑整图
    - 当前工作流中的所有节点（triage → specialist → critic）都会经过检查点保存

    返回：
    - SqliteSaver：SQLite 持久化的检查点实例
    """
    global _CHECKPOINTER, _CHECKPOINTER_CONTEXT
    if _CHECKPOINTER is None:
        settings = get_settings()
        if settings.checkpoint_backend == "postgres":
            # 生产租户不能共享本地文件检查点。PostgreSQL saver 将图状态存入数据库，
            # workflow.py 恢复人工审核时还会再次校验 checkpoint 所属租户，避免跨租户恢复。
            from langgraph.checkpoint.postgres import PostgresSaver
            connection_url = settings.postgres_url.replace("postgresql+psycopg://", "postgresql://")
            _CHECKPOINTER_CONTEXT = PostgresSaver.from_conn_string(connection_url)
            _CHECKPOINTER = _CHECKPOINTER_CONTEXT.__enter__()
            _CHECKPOINTER.setup()
        else:
            if settings.environment.lower() in {"production", "prod"}:
                raise RuntimeError("Production tenant deployment requires CHECKPOINT_BACKEND=postgres")
            path = Path(settings.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            _CHECKPOINTER = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
            _CHECKPOINTER.setup()
    return _CHECKPOINTER


def _validate_boundary(node_name: str, state: AgentState) -> None:
    """校验当前节点的输出是否满足预期的契约（节点契约）。

    每个节点在执行完毕后，这里会用对应的 Pydantic 模型校验关键字段是否存在、
    类型是否正确。如果校验失败，Pydantic 会抛出 ValidationError，
    被 _instrument_node 的 except 捕获，转为结构化错误。

    这保证了：
    - triage 一定输出了 task_type 和 next_agent
    - data_query 一定输出了 tool_calls 和 tool_result
    - critic 一定输出了 critic_verdict 和 critic_revision_target
    - 等等

    参数：
    - node_name：当前节点名（triage / critic / metric_definition / data_query / root_cause / report / generate_answer）
    - state：节点执行完毕后的完整 state
    """
    if node_name == "triage":
        # Triage 节点必须输出 task_type、next_agent 和 handoff 理由
        TriageDecision(
            task_type=state["task_type"],
            next_agent=state["next_agent"],
            reason=state.get("handoffs", [{}])[-1].get("reason", "classified"),
        )
    elif node_name == "critic":
        # Critic 节点必须输出裁决结论、修订目标（可能为空）和问题列表
        decision = state.get("critic_decisions", [{}])[-1]
        CriticDecision(
            verdict=state["critic_verdict"],
            revision_target=state.get("critic_revision_target"),
            issues=decision.get("llm_issues", []),
        )
    elif node_name == "metric_definition":
        # 指标定义节点必须输出 RAG 上下文和引用列表
        MetricDefinitionOutput(metric_context=state.get("metric_context", ""), citations=state.get("rag_citations", []))
    elif node_name == "data_query":
        result_envelope = state.get("tool_result", {})
        result_payload = result_envelope.get("result", {}) if isinstance(result_envelope, dict) else {}
        # A rejected capability is a control-plane outcome, not an attempted
        # data-tool invocation.  It deliberately has no ToolInvocation:
        # manufacturing one would make the governance audit report a false
        # invalid/unauthorized call.  Validate the linked record instead.
        if (
            isinstance(result_envelope, dict)
            and result_envelope.get("server") == "metric-registry-planner"
            and result_envelope.get("tool_name") == "metric_clarification"
            and isinstance(result_payload, dict)
            and result_payload.get("status") == "unsupported"
        ):
            request_id = str(result_envelope.get("request_id") or "")
            control_events = [
                item for item in (state.get("control_events") or [])
                if isinstance(item, dict)
                and item.get("event_type") == "capability_preflight_stop"
                and item.get("request_id") == request_id
            ]
            records = [
                item for item in (state.get("decision_records") or [])
                if isinstance(item, dict)
                and item.get("request_id") == request_id
                and item.get("limitation_code") == "CAPABILITY_GAP"
            ]
            if len(control_events) == 1 and len(records) == 1 and not state.get("tool_calls"):
                if (
                    control_events[0].get("sql_executed") is False
                    and records[0].get("sql_executed") is False
                    and state.get("terminal_code") == "CAPABILITY_GAP"
                ):
                    return
            # Otherwise fall through to the normal invocation contract: a
            # malformed pseudo-stop must not silently bypass validation.
        if (
            isinstance(result_envelope, dict)
            and result_envelope.get("server") == "request-contract-guard"
            and result_envelope.get("tool_name") is None
            and isinstance(result_payload, dict)
            and result_payload.get("status") == "unsupported"
        ):
            contract = (
                (state.get("context_compilation") or {}).get("request_contract") or {}
            )
            reason_code = str(result_payload.get("reason_code") or "")
            contract_hash = str(contract.get("contract_hash") or "")
            recorded_failure = any(
                isinstance(item, dict)
                and item.get("reason") == reason_code
                and item.get("contract_hash") == contract_hash
                for item in (state.get("request_contract_failures") or [])
            )
            if (
                reason_code
                and contract_hash
                and recorded_failure
                and not state.get("tool_calls")
                and not state.get("query_results")
            ):
                # A correlated contract refusal is a valid pre-execution stop,
                # not a missing invocation. Keep it fail-closed without
                # manufacturing a ToolInvocation or hiding the reason behind
                # an IndexError.
                return
        if (
            isinstance(result_envelope, dict)
            and result_envelope.get("tool_name") == "registered_numeric_decomposition"
            and isinstance(result_payload, dict)
            and result_payload.get("status") == "planned"
        ):
            # This is a server-generated operation plan, not a completed tool
            # invocation. Validate its typed authorization carrier without
            # manufacturing a ToolInvocation before any data action runs.
            from app.mcp.schemas import MetricPlanningDecision

            decision = MetricPlanningDecision.model_validate(result_payload.get("parsed") or {})
            request = state.get("registered_decomposition_request") or {}
            request_id = str(result_payload.get("request_id") or "")
            if (
                not request_id
                or decision.operation_intent != "registered_numeric_decomposition"
                or decision.registered_decomposition is None
                or decision.registered_decomposition.status != "ready"
                or request.get("status") != "ready"
                or decision.registered_decomposition.model_dump(mode="json") != request
            ):
                raise ValueError("Registered numeric decomposition planning envelope is incomplete or inconsistent.")
            return
        if state.get("result_followup_context"):
            # A validated result follow-up is answered from the previous turn's
            # projection; no ToolInvocation or fresh MCP result exists to bind.
            return
        invocations = state.get("tool_calls") or []
        if not invocations:
            raise ValueError("Data query result has no correlated tool invocation.")
        invocation = invocations[-1]
        DataQueryOutput(invocation=invocation, result=result_envelope)
    elif node_name in {"root_cause", "report"} and state.get("analysis_result"):
        # 归因/报告节点必须输出分析摘要和任务标记
        AnalysisOutput(summary=state["analysis_result"].get("summary", ""), task_type=state["analysis_result"].get("task_type", state.get("task_type", "metric_definition")))
    elif node_name == "generate_answer":
        # 回答生成节点必须输出最终文本和声明列表
        AnswerOutput(answer=state.get("answer", ""), claims=state.get("claims", []))


def instrument_node(node_name: str, node: Node) -> Node:
    """包装一个图节点，加上统一的运行时保护层。

    在 workflow.py 中，每个节点在注册时都经过此函数包装：
        graph.add_node("triage", instrument_node("triage", triage))

    包装层做的事情（按执行顺序）：
    1. 总延迟预算检查：从图开始执行到当前节点，总耗时超过阈值则拒绝
    2. 总步骤预算检查：步骤数超过阈值则拒绝
    3. 激活执行上下文（execution_context）供 LLM client 等使用
    4. 真正执行节点逻辑
    5. 节点契约校验（_validate_boundary）
    6. 自动清理执行上下文
    7. 任何未捕获异常都被转成 AgentError，不会崩溃图
    8. 记录节点耗时到 node_timings，并通过 OpenTelemetry 发送事件

    参数：
    - node_name：节点名，用于错误报告、契约选择和可观测性标签
    - node：原始节点函数

    返回：
    - 包装后的节点函数。构图阶段调用 instrument_node 时只创建并返回 wrapped，
      不执行任何预算检查或业务节点；LangGraph 日后运行到该节点才调用 wrapped(state)。
    """

    def wrapped(state: AgentState) -> AgentState:
        """所有图节点的实际运行入口（运行阶段才会执行）。

        设计思路：业务节点只关心“读取哪些 state、写入哪些 state”；本函数集中处理
        每个节点都必须遵守的横切规则：预算、执行上下文、输出契约、异常隔离、计时与
        可观测性。这样新增节点只要经过 ``instrument_node`` 注册，就自动拥有同一套
        运行时行为，不必复制 try/except、计时和 Pydantic 校验代码。

        返回约定：无论成功、预算超限还是业务异常，通常都返回 AgentState。失败信息写入
        ``state[\"errors\"]``，由 workflow 中的 ``_route_*`` 函数决定后续转 critic
        还是持久化；只有 LangGraph 的 ``GraphInterrupt`` 必须原样抛出，以保留 HITL 中断。
        """
        # ``started`` 只用于本节点耗时；它与 runtime_started_at（整张图的起点）不同。
        started = perf_counter()
        # 通过 ContextVar 向事件队列发送节点开始信号，使前端可以实时高亮当前节点
        execution_context.mark_node_started(node_name)
        execution_context.enqueue_sse_event(SseNodeStartedEvent(node=node_name))
        # init_state 会设置 runtime_budget；get(..., {}) 使单测/直接调用节点仍可运行。
        budget = state.get("runtime_budget", {})

        # 仅第一个进入 wrapped 的节点写下“整张图的计时起点”。后续节点复用该值，
        # 所以单步调试、LLM 和 MCP 的等待时间都会累计进图级延迟预算。
        if "runtime_started_at" not in state:
            state["runtime_started_at"] = started

        # 每进入一个节点外壳就算一步（包括 critic、persist_trace）。它是防循环保险，
        # 尤其保护 root_cause → data_query 的补充证据循环和 critic 的返工循环。
        steps = int(state.get("runtime_steps", 0)) + 1
        state["runtime_steps"] = steps

        # 入口日志描述“准备执行哪个节点”和当前已有的 state；不代表业务节点已成功运行。
        print(f"[DEBUG] >>> NODE={node_name} step={steps} task_type={state.get('task_type')} keys_in={list(state.keys())}")

        # persist_trace 是故障取证的最后安全网。工作流已经决定结束后，即使预算耗尽，
        # 也不能因为再次拦截而丢失导致失败的证据。这里只跳过节点执行前的预算拦截，
        # persist_trace 自身的耗时仍会在下面的 finally 中记录。
        is_trace_persistence = node_name == "persist_trace"

        # ---- 预算检查 1：总执行延迟 ----
        max_latency_ms = int(budget.get("max_latency_ms", get_settings().max_latency_ms))
        elapsed_ms = (started - float(state["runtime_started_at"])) * 1000
        if not is_trace_persistence and elapsed_ms > max_latency_ms:
            # 不抛异常而是写结构化错误：图可以据此转向 critic/persist_trace，
            # 前端也可从 errors[-1]["code"] 得到明确的失败原因。
            state.setdefault("errors", []).append(
                AgentError(code="graph_latency_budget_exceeded", message="Graph latency budget exceeded.", agent_name=node_name).model_dump()
            )
            # 让后续 critic 明确知道此执行不应被当作正常审核通过。
            state["critic_verdict"] = "rejected"
            return state

        # ---- 预算检查 2：步骤数上限 ----
        if not is_trace_persistence and steps > int(budget.get("max_graph_steps", get_settings().max_graph_steps)):
            # 同样返回可路由的错误 state，避免无限循环演变为整个服务卡死。
            state.setdefault("errors", []).append(
                AgentError(code="graph_step_budget_exceeded", message="Graph step budget exceeded.", agent_name=node_name).model_dump()
            )
            state["critic_verdict"] = "rejected"
            return state

        try:
            # 将“当前这次节点运行”的 state 绑定到 ContextVar。深层的 LLM/RAG/MCP
            # 调用不必层层传 state，也能记录 token/cost、trace_id 等运行信息。
            token = activate_state(state)
            try:
                # 现在才进入传入的原始业务函数，例如 triage(state) 或 data_query(state)。
                # 原节点可原地修改 state，也可返回一个新的 state；下方统一使用 result 校验。
                result = node(state)
                # 在交给下一条图边前检查“该节点承诺产出的关键字段”。例如 data_query
                # 必须有合法 tool_calls/tool_result；校验失败会进入下面的 except 路径。
                _validate_boundary(node_name, result)
                unexpected = set(result.keys()) - AGENTSTATE_KEYS - {"_prev_keys"}
                if unexpected:
                    LOGGER.warning("Node %s wrote undeclared state keys: %s", node_name, unexpected)
                return result
            finally:
                # 必须与 activate_state 成对出现：否则下一节点可能误读上一个节点的 state。
                # finally 即使 node()/校验抛异常或上面 return，也仍会执行。
                reset_state(token)

        except GraphInterrupt:
            # 人工审核的 interrupt 不是错误；LangGraph 要靠它保存检查点并暂停图。
            raise
        except Exception as exc:
            public_msg, diag_msg = sanitize_exception(exc)
            failure_context = None
            error_code = "agent_execution_failed"
            request_id = None
            retryable = False
            is_database_error = False
            try:
                from sqlalchemy.exc import SQLAlchemyError

                if isinstance(exc, SQLAlchemyError):
                    is_database_error = True
                    from app.core.failure_context import ExecutionFailureContext, public_failure_message

                    failure_context = ExecutionFailureContext(
                        category="database_error",
                        service="postgresql",
                        code="database_execution_failed",
                        cause_type=type(exc).__name__,
                        # No tool request exists when failure happens during
                        # node preparation; do not fabricate a tool request ID.
                        request_id=None,
                        retryable=None,
                    ).model_dump(mode="json")
                    public_msg = public_failure_message(failure_context)
                    # SQLAlchemy's rendered exception may contain the SQL and
                    # driver message. Keep diagnostics classified and redacted.
                    diag_msg = f"Database operation failed ({type(exc).__name__})."
                    error_code = "database_error"
            except ImportError:  # pragma: no cover - SQLAlchemy is a runtime dependency
                pass
            LOGGER.warning("Node %s: %s", node_name, diag_msg)
            if get_settings().environment == "development" and not is_database_error:
                LOGGER.exception("Agent node failed: %s", node_name)
            state.setdefault("errors", []).append(
                AgentError(
                    code=error_code,
                    message=public_msg,
                    public_message=public_msg,
                    diagnostic_message=diag_msg,
                    retryable=retryable,
                    agent_name=node_name,
                    cause_type=type(exc).__name__,
                    failure_context=failure_context,
                    request_id=None,
                ).model_dump()
            )
            state["critic_verdict"] = "rejected"
            return state

        finally:
            execution_context.mark_node_finished()
            # 这是最外层 finally，因此三种路径都会记录耗时：正常返回、预算提前返回、
            # 异常转 AgentError。它测量的是 wrapped 的完整开销，不只是 node(state)。
            duration_ms = round((perf_counter() - started) * 1000, 3)
            state.setdefault("node_timings", []).append({"node": node_name, "duration_ms": duration_ms})
            keys_out = list(state.keys())
            # 调试辅助：与上次记录的 key 集合比较，只能粗略提示“新增了哪些 key”，
            # 不能反映已有 key 的值被更新；精确内容请查看最终 trace 的 state_snapshot。
            wrote = [k for k in keys_out if k not in (state.get("_prev_keys") or [])]
            print(f"[DEBUG] <<< NODE={node_name} duration={duration_ms}ms keys_out={len(keys_out)} wrote={wrote} errors={bool(state.get('errors'))}")
            state["_prev_keys"] = keys_out

            # 除本地 print 外，还把节点结果发送给 OpenTelemetry/metrics。status 取决于
            # 整个 state 是否已有 errors，因此它可能反映上游遗留错误，而非本节点新错误。
            record_event("agent.run", {
                "trace_id": state.get("trace_id"), "thread_id": state.get("thread_id"),
                "agent_name": node_name, "task_type": state.get("task_type"),
                "duration_ms": duration_ms, "status": "error" if state.get("errors") else "ok",
            })
            if state.get("errors"):
                record_event("agent.error", {
                    "trace_id": state.get("trace_id"), "agent_name": node_name,
                    "error_code": state["errors"][-1].get("code", "unknown"),
                })

    return wrapped
