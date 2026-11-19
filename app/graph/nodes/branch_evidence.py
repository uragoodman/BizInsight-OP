"""Branch Evidence 节点 — P1.1 受控并行。

把 root_cause 的串行证据循环（root_cause ↔ data_query 一对一）改为一次性把
趋势 + 维度贡献等独立事实查询并发执行：
- 每个分支只写自己的 ``branch_results.<purpose>``，不共享 dict 原地覆盖。
- 分支失败只产出带原因的 ``status=unavailable`` 证据，主链仍由 Critic 判断是否足以回答。
- ``root_cause_queue`` 置空，root_cause 进入后直接评估全部证据并 finalize，不再串行循环。
- 分支耗时写入 ``branch_timings`` 供 trace 评估并行收益。

并发安全：MCP 每次调用独立派生 stdio 子进程，因此用有界线程池限流，
并用 ``contextvars.copy_context()`` 把租户上下文传播进工作线程。
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from uuid import uuid4
from typing import Any

from app.agents.contracts import EvidenceBundle, stable_evidence_id
from app.graph.state import AgentState
from app.graph.nodes.root_cause import _build_follow_up_queue
from app.mcp.client import query_business_data, query_scoped_business_data
from app.observability import record_event

# 并发上限：避免为每次 MCP 调用派生过多 stdio 子进程。
MAX_BRANCH_WORKERS = 4


def _execute_query(state: AgentState, pending_query: dict, request_id: str) -> tuple[dict | None, float, str | None]:
    """执行单个 pending_query（与 data_query 同路径）。

    返回值：``(result, duration_ms, error)``——成功时 error 为 None，失败时 result 为 None。
    """
    question = str(pending_query.get("question") or "")
    started = perf_counter()
    try:
        if state.get("seller_scope_payload"):
            return ({
                "server": "seller-scope-guard", "tool_name": None, "arguments": {}, "request_id": request_id,
                "result": {
                    "status": "unsupported", "parsed": {},
                    "reason_code": "seller_scope_operation_unsupported",
                    "reason": "Branch evidence queries do not have a seller-scoped handler.",
                },
            }, 0.0, None)
        if state.get("controlled_analysis_scope"):
            result = query_scoped_business_data(question, state["controlled_analysis_scope"], request_id=request_id, manifest_id=state.get("data_version_manifest_id") or None)
        else:
            kwargs: dict[str, Any] = {"task_type": "metric_query", "question": question, "idempotency_key": request_id}
            if state.get("tenant_id") or state.get("user_role"):
                kwargs.update({"tenant_id": state.get("tenant_id"), "role": state.get("user_role"), "request_id": request_id})
            result = query_business_data(**kwargs)
        if not isinstance(result, dict):
            result = {"server": "invalid-tool-gateway", "tool_name": None, "arguments": {}, "request_id": request_id,
                      "result": {"status": "unsupported", "parsed": {}, "reason": "invalid result type"}}
        duration = round((perf_counter() - started) * 1000, 3)
        return result, duration, None
    except Exception as exc:  # noqa: BLE001 - 分支失败隔离，不阻断主链
        duration = round((perf_counter() - started) * 1000, 3)
        return None, duration, str(exc)


def branch_evidence(state: AgentState) -> AgentState:
    """并发执行趋势 + 维度贡献等独立事实查询，写入 branch_results 与 root_cause_evidence。

    只对 anomaly_analysis 生效；已执行过（branch_results 非空）或非异常任务时透传。
    每个分支只写自己的 ``branch_results.<purpose>``；失败分支产出 status=unavailable 证据。
    """
    if state.get("seller_scope_payload"):
        from app.graph.nodes.seller_scope_guard import reject_unsupported_seller_scope

        return reject_unsupported_seller_scope(state)
    if state.get("task_type") != "anomaly_analysis" or state.get("branch_results") is not None:
        return state

    question = str(state.get("user_message") or "")
    queue = _build_follow_up_queue(question)
    state["branch_results"] = {}
    if not queue:
        # 无可补查分支：直接交给 root_cause 用既有 evidence 收尾。
        state["root_cause_queue"] = []
        return state

    started = perf_counter()
    branch_results: dict[str, dict[str, Any]] = {}
    root_cause_evidence: list[dict[str, Any]] = list(state.get("root_cause_evidence", []))
    state.setdefault("evidence_bundles", [])

    def run_item(item: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        request_id = f"{state.get('trace_id', '')}:{item.get('query_id', uuid4().hex[:8])}"
        try:
            # 每个线程各自 copy 一份 context：共享同一个 Context 会导致
            # "cannot enter context" 重入错误（ctx.run 不可被多线程同时进入）。
            ctx = contextvars.copy_context()
            result, duration, error = ctx.run(_execute_query, state, item, request_id)
        except Exception as exc:  # noqa: BLE001 - 分支任务本身异常也转为 unavailable，不阻断主链
            result, duration, error = None, 0.0, f"{type(exc).__name__}:{exc}"
        if error is not None:
            unavailable = {
                "server": "branch", "tool_name": None, "arguments": {}, "request_id": request_id,
                "result": {"status": "unavailable", "parsed": {}, "reason": f"branch_error:{error}"},
            }
            return unavailable, {"duration_ms": duration, "status": "unavailable"}
        status = result.get("result", {}).get("status", "unknown")
        return result, {"duration_ms": duration, "status": status}

    with ThreadPoolExecutor(max_workers=min(MAX_BRANCH_WORKERS, len(queue))) as pool:
        futures = {pool.submit(run_item, item): item for item in queue}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result, meta = future.result()
            except Exception as exc:  # noqa: BLE001 - 兜底：分支抛异常也转 unavailable 证据
                unavailable = {
                    "server": "branch", "tool_name": None, "arguments": {}, "request_id": "",
                    "result": {"status": "unavailable", "parsed": {}, "reason": f"branch_error:{type(exc).__name__}:{exc}"},
                }
                result, meta = unavailable, {"duration_ms": 0, "status": "unavailable"}
            purpose = item["purpose"]
            branch_results[purpose] = {"query_id": item["query_id"], "result": result, **meta}
            root_cause_evidence.append({
                "purpose": purpose,
                "question": item.get("question", ""),
                "metric_id": item.get("metric_id"),
                "query_id": item.get("query_id", ""),
                "result": result,
            })
            state["evidence_bundles"].append(EvidenceBundle(
                evidence_id=stable_evidence_id("mcp", result.get("tool_name", "branch"), {"query_id": item["query_id"], "result": result.get("result", {})}),
                source_type="mcp", source_ref=result.get("tool_name") or "branch",
                content={"query_id": item["query_id"], "result": result.get("result", {})},
            ).model_dump(mode="json"))
            record_event("branch.evidence", {
                "trace_id": state.get("trace_id"), "purpose": purpose,
                "duration_ms": meta["duration_ms"], "status": meta["status"],
            })

    state["branch_results"] = branch_results
    state["root_cause_evidence"] = root_cause_evidence
    # 证据已齐：置空队列，root_cause 进入后直接评估全部证据，不再串行循环。
    state["root_cause_queue"] = []
    state["branch_timings"] = {
        "total_ms": round((perf_counter() - started) * 1000, 3),
        "branches": {name: meta["duration_ms"] for name, meta in branch_results.items()},
    }
    state["next_agent"] = "root_cause"
    state["current_agent"] = "branch_evidence"
    unavailable = sum(1 for meta in branch_results.values() if meta["status"] == "unavailable")
    state.setdefault("node_history", []).append({
        "node": "branch_evidence",
        "summary": (
            f"Ran {len(queue)} independent evidence branches concurrently "
            f"({len(branch_results) - unavailable} resolved, {unavailable} unavailable)."
        ),
    })
    return state
