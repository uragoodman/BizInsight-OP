"""每个图节点的执行上下文与 LLM 预算控制。

本模块通过 ContextVar 实现图节点级别的执行状态传递，
使 LLM client 等深层调用方无需显式接收 state 即可读取
当前节点的运行时上下文并累积 token/cost 消耗。

工作流程：
1. runtime.py 的 instrument_node() 在节点执行前调用
   `activate_state(state)` → 将当前 AgentState 写入 ContextVar
2. LLM client 内部的 `_record_usage()` → 调用 `consume_llm_usage()`
   → 从 ContextVar 读取 state → 累加 token/cost → 超限时抛异常
3. 节点执行完毕后，instrument_node() 的 finally 块调用
   `reset_state(token)` → 清理 ContextVar

节点跟踪（用于 SSE 进度推送）：
- 外部（workflow.py）将事件队列引用注入 ContextVar
- runtime.py 的 instrument_node 在节点开始/结束时通过队列写入 `node_started`
- 前端自行维护耗时计数，不需要跨线程读取
"""

from __future__ import annotations

from contextvars import ContextVar
from queue import Queue
from time import perf_counter
from typing import Any


# ContextVar 是 Python 标准库提供的协程/线程局部变量，
# 比 threading.local 更适合 async context，且 automatic copy-on-write。
# 存储的是当前正在执行的节点的 AgentState 引用。
_ACTIVE_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "active_agent_state", default=None
)


class TokenBudgetExceeded(RuntimeError):
    """A completed model response exceeded the request's configured budget."""


def activate_state(state: dict[str, Any]):
    """将给定的 AgentState 绑定到当前执行上下文。

    在 instrument_node() 中调用，使 LLM client、MCP client、
    RAG retriever 等深层调用方能通过 `_ACTIVE_STATE.get()` 读取
    当前节点的 state（如 trace_id、runtime_budget、llm_usage 等）。

    参数：
    - state：当前节点的 AgentState 字典引用

    返回：
    - Token：用于后续 reset_state() 恢复上下文的令牌
    """
    return _ACTIVE_STATE.set(state)


def reset_state(token) -> None:
    """清除当前执行上下文，恢复到节点执行前的状态。

    在 instrument_node() 的 finally 块中调用，确保 ContextVar
    不会泄漏到下一个节点或跨请求污染。

    参数：
    - token：activate_state() 返回的令牌
    """
    _ACTIVE_STATE.reset(token)


def active_state() -> dict[str, Any] | None:
    """Return the current graph state for non-critical telemetry only."""
    return _ACTIVE_STATE.get()


def token_budget_is_exhausted(state: dict[str, Any] | None = None) -> bool:
    """Return whether this request has no token budget left for another call.

    A configured limit of zero means unlimited. Reaching the limit exactly is
    allowed for the response that consumed it, but closes the gate for the
    next model request. A previously recorded budget error remains terminal.
    """
    current = state if state is not None else _ACTIVE_STATE.get()
    if current is None:
        return False
    errors = current.get("errors") or []
    if any(isinstance(error, dict) and error.get("code") == "token_budget_exceeded" for error in errors):
        return True
    budget = current.get("runtime_budget") or {}
    try:
        limit = int(budget.get("max_tokens", 0) or 0)
        usage = current.get("llm_usage") or {}
        consumed = (
            int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("output_tokens", 0) or 0)
            + int(usage.get("preflight_reserved_tokens", 0) or 0)
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return limit > 0 and consumed >= limit


def turn_soft_limit_reached(state: dict[str, Any] | None = None) -> bool:
    """Return whether current token usage has reached the request soft limit."""

    current = state if state is not None else _ACTIVE_STATE.get()
    if current is None:
        return False
    runtime_budget = current.get("runtime_budget") or {}
    try:
        soft_limit = int(runtime_budget.get("soft_tokens", 0) or 0)
        usage = current.get("llm_usage") or {}
        used_tokens = int(usage.get("input_tokens", 0) or 0) + int(
            usage.get("output_tokens", 0) or 0
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return soft_limit > 0 and used_tokens >= soft_limit


def release_turn_budget(_trace_id: str = "") -> None:
    """Compatibility hook for workflow teardown.

    Turn budgets are currently stored on the graph state and have no
    process-local reservation to release. Workflow callers keep this hook so
    a future shared budget implementation can add cleanup without changing
    the graph lifecycle.
    """
    return None


def record_token_budget_exceeded(state: dict[str, Any] | None = None) -> None:
    """Record the terminal token-budget marker once for this request."""
    current = state if state is not None else _ACTIVE_STATE.get()
    if current is None:
        return
    errors = current.setdefault("errors", [])
    if any(isinstance(error, dict) and error.get("code") == "token_budget_exceeded" for error in errors):
        return
    errors.append({
        "code": "token_budget_exceeded",
        "message": "LLM token budget exceeded.",
        "retryable": False,
        "agent_name": current.get("current_agent"),
    })


def consume_llm_usage(input_tokens: int, output_tokens: int, cost: float) -> None:
    """在每次 LLM 调用完成后累加 token 消耗，并检查预算。

    由 LLM client 的 `_record_usage()` 调用。从 ContextVar 中读取
    当前节点的 state，累加 llm_usage，然后检查是否超过两个预算上限：
    - max_tokens：总 token 消耗上限（input + output）
    - max_cost：总成本消耗上限（美元）

    如果超出任一预算：
    - 向 state.errors 追加一条结构化错误
    - 抛出 RuntimeError → LLM client 捕获后返回空字符串/None
    - 图流不会崩溃，但当前节点终止并进入 critic 的 error 快速路径

    参数：
    - input_tokens：本次 LLM 调用的输入 token 数
    - output_tokens：本次 LLM 调用的输出 token 数
    - cost：本次调用的估算成本（美元）
    """
    state = _ACTIVE_STATE.get()
    if state is None:
        return  # 无活跃上下文时不执行预算检查（如测试环境）
    # 累加 usage
    usage = state.setdefault("llm_usage", {
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    })
    usage["input_tokens"] += input_tokens
    usage["output_tokens"] += output_tokens
    usage["estimated_cost"] = round(usage["estimated_cost"] + cost, 8)
    budget = state.get("runtime_budget", {})
    # 检查 Token 预算
    token_limit = int(budget.get("max_tokens", 0))
    cost_limit = float(budget.get("max_cost", 0))
    if token_limit and (
        usage["input_tokens"] + usage["output_tokens"]
        + int(usage.get("preflight_reserved_tokens", 0) or 0)
    ) > token_limit:
        record_token_budget_exceeded(state)
        raise TokenBudgetExceeded("LLM token budget exceeded.")
    # 检查成本预算
    if cost_limit and usage["estimated_cost"] > cost_limit:
        state.setdefault("errors", []).append({
            "code": "cost_budget_exceeded",
            "message": "LLM cost budget exceeded.",
            "retryable": False,
            "agent_name": state.get("current_agent"),
        })
        raise RuntimeError("LLM cost budget exceeded.")


# ============================================================
# 图节点实时状态跟踪（用于 SSE 进度推送）
# ============================================================

_GRAPH_EVENT_QUEUE: ContextVar[Queue | None] = ContextVar(
    "graph_event_queue", default=None
)
_GRAPH_CURRENT_NODE: ContextVar[str | None] = ContextVar(
    "graph_current_node", default=None
)
_GRAPH_CURRENT_NODE_AT: ContextVar[float | None] = ContextVar(
    "graph_current_node_started_at", default=None
)


from app.schemas.events import SSE_EVENT_ADAPTER, SseBaseEvent


def current_graph_node() -> str | None:
    """Return the node currently executing in the active graph context."""
    return _GRAPH_CURRENT_NODE.get()


def set_graph_event_queue(queue: Queue | None) -> None:
    """由 workflow.py 的 run_graph 在启动图前调用，注入事件队列引用。"""
    _GRAPH_EVENT_QUEUE.set(queue)


def enqueue_sse_event(event: SseBaseEvent) -> None:
    """通过 TypeAdapter 校验后向队列发送事件。仅当队列已设置时生效。

    所有 SSE 事件必须经过此函数入队，禁止直接向队列写裸 dict。
    """
    validated = SSE_EVENT_ADAPTER.validate_python(event.model_dump(mode="json"))
    q = _GRAPH_EVENT_QUEUE.get()
    if q is not None:
        q.put(validated.model_dump(mode="json"))


def mark_node_started(node_name: str) -> None:
    """记录节点开始执行的时间与名称。由 runtime.py 的 instrument_node 调用。"""
    _GRAPH_CURRENT_NODE.set(node_name)
    _GRAPH_CURRENT_NODE_AT.set(perf_counter())


def mark_node_finished() -> None:
    """节点执行完毕（无论成功/失败），清除当前节点。"""
    _GRAPH_CURRENT_NODE.set(None)
    _GRAPH_CURRENT_NODE_AT.set(None)
