"""Triage Agent — LLM task classification with fail-closed routing.

The classifier selects only a registered task type. If it is unavailable or
returns an invalid value, the workflow asks for clarification instead of
guessing a route from the user's words.

条件路由输出：
- `task_type`：LLM 判定的任务类型。
- `next_agent`：根据路由表映射的下游 Agent 节点名。
- `handoffs`：记录 triage → specialist 的 handoff 原因。

MECE 写入字段：task_type, current_agent, next_agent, handoffs, node_history
"""

from app.graph.state import AgentState
from app.agents.planner import build_execution_plan
from app.context.orchestrator import ContextOrchestrator
from app.llm.client import llm_classify
from app.llm.routing import model_for
from app.llm.triage_prompt import ROUTING_TABLE, SYSTEM_PROMPT

TASK_TYPES = list(ROUTING_TABLE.keys())


def triage(state: AgentState) -> AgentState:
    """判断用户问题类型，决定下游 Specialist Agent。

    优先调用 LLM 分类；分类不可用或输出不在注册集合中时 fail closed 到澄清。
    并将 handoff 信息写入 state，供 graph 条件边使用。

    参数：
    - `state`：当前工作流状态，至少包含 `user_message`。

    返回值：
    - `AgentState`：写入 `task_type`、`next_agent`、`handoffs` 后的状态。
    """

    # 打点入口：triage 只负责“识别任务类型和选择下游节点”，不应在这里执行
    # MCP 查询或修改受控分析范围。memory/historical context 仅作为不可信参考。
    compilation = state.get("context_compilation") or {}
    from app.context.request_contract import (
        record_contract_observation, record_contract_violation,
        validate_compiled_request_contract,
    )
    contract_valid, contract_reason = validate_compiled_request_contract(compilation)
    contract = compilation.get("request_contract") or {}
    expected_clarification = bool(
        compilation.get("needs_clarification")
        and contract.get("parse_status") == "clarification_required"
        and contract_reason == contract.get("uncertainty_reason")
    )
    if not contract_valid and not expected_clarification:
        record_contract_violation(state, "triage", contract_reason or "request_contract_invalid")
    record_contract_observation(state, "triage", {
        "contract_valid": contract_valid or expected_clarification,
        "validation_reason": contract_reason,
    })
    question = str(compilation.get("standalone_question") or state["user_message"])
    # WP1: TriageContext is the only context handed to routing.  It contains
    # conversation/task/preference candidates only; analysis memory, RAG and
    # tool evidence are deliberately excluded by the orchestrator policy.
    instruction = state["user_message"]
    if question != state["user_message"]:
        instruction = f"{state['user_message']}\n解析后: {question}"
    triage_context = ContextOrchestrator().build_triage_context(
        state,
        instruction=instruction,
        system_prompt=SYSTEM_PROMPT,
        model=model_for("classify")[0],
    )
    packed_text = triage_context.bundle.text
    # Ambiguous historical references are stopped before either retrieval or data tools.
    if compilation.get("needs_clarification"):
        task_type = (
            "metric_definition"
            if compilation.get("intent") == "retrieve_metric_definition"
            else "metric_query"
        )
    elif (
        isinstance(compilation.get("result_followup"), dict)
        and not compilation.get("needs_clarification")
    ):
        # The compiler has already validated and bound this follow-up to a
        # trusted same-scope result snapshot. Route it deterministically; a
        # second probabilistic task classification would only add cost and
        # could send a report-action request into the full report workflow.
        task_type = "metric_query"
    elif (compilation.get("operation_intent") or {}).get("kind") == "registered_numeric_decomposition":
        # This remains a metric query task (so Critic applicability is
        # unchanged), while the separate operation intent tells the planner
        # to use the Registry-validated decomposition path.
        task_type = "metric_query"
    elif (compilation.get("operation_intent") or {}).get("kind") in {
        "dimensional_statistical_contribution",
        "business_causal_investigation",
    }:
        # Preserve the normal optional Critic applicability label. The sealed
        # operation contract, not this broad task family, controls the edge
        # after data_query: one verified contribution goes to answer; explicit
        # business-causal intent alone enters the multi-step diagnostic loop.
        task_type = "anomaly_analysis"
    else:
        try:
            task_type = llm_classify(
                SYSTEM_PROMPT,
                packed_text,
                TASK_TYPES,
                operation="classify.task_type",
                interaction_policy="nonstream_fast",
            )
        except Exception:  # noqa: BLE001 - classifier failure must not guess a route
            task_type = ""

    if not task_type or task_type not in TASK_TYPES:
        reason_code = (
            "task_type_classifier_unavailable"
            if not task_type
            else "task_type_classifier_invalid_output"
        )
        clarification_context = dict(compilation)
        clarification_context["needs_clarification"] = True
        clarification_context["clarification_reason"] = "task_type_unresolved"
        missing_slots = list(clarification_context.get("missing_slots") or [])
        if "task_type" not in missing_slots:
            missing_slots.append("task_type")
        clarification_context["missing_slots"] = missing_slots
        state["context_compilation"] = clarification_context
        state["triage_unresolved"] = True
        state["triage_unresolved_reason"] = reason_code
        state["task_type"] = "clarification"
        state["next_agent"] = "context_clarification"
        state["current_agent"] = "triage"
        state.setdefault("node_history", []).append({
            "node": "triage",
            "summary": f"task classifier did not produce a registered route; fail closed ({reason_code})",
        })
        state.setdefault("handoffs", []).append({
            "from": "triage",
            "to": "context_clarification",
            "reason": reason_code,
        })
        return state

    # A registered task type means legacy triage resolved its route.  Keep this
    # explicit because Phase 2 shadow audit requires a boolean baseline and
    # must distinguish a normal route from classifier fail-closed handling.
    state["triage_unresolved"] = False
    state.pop("triage_unresolved_reason", None)

    # 打点出口：下面四个字段决定条件边实际走向，也是面试/debug 时最重要的路由快照。
    state["task_type"] = task_type
    state["next_agent"] = ROUTING_TABLE[task_type]
    state["execution_plan"] = build_execution_plan(task_type).model_dump(mode="json")
    compilation = state.get("context_compilation") or {}
    if (
        isinstance(compilation.get("result_followup"), dict)
        and not compilation.get("needs_clarification")
    ):
        # A validated same-scope result follow-up is answered from the prior
        # result snapshot. Do not let a generic phrase such as “证据是什么”
        # route it through metric-definition RAG or a fresh data query.
        task_type = "metric_query"
        state["task_type"] = task_type
        state["next_agent"] = "data_query"
        state["execution_plan"] = build_execution_plan(task_type).model_dump(mode="json")
    state["request_contract_hash"] = contract.get("contract_hash")
    if contract.get("required_evidence_sequence"):
        # Seed the governed diagnostic from the compiler's sealed slots. The
        # diagnostic path must not recover metric/dimension by reparsing text.
        state["diagnostic_target_metric"] = contract.get("metric_id")
        state["requested_dimensions"] = [contract["dimension"]] if contract.get("dimension") else []
    state["current_agent"] = "triage"
    state.setdefault("node_history", []).append(
        {
            "node": "triage",
            "summary": f"LLM routed to task_type={task_type}, next={ROUTING_TABLE[task_type]}",
        }
    )
    state.setdefault("handoffs", []).append(
        {
            "from": "triage",
            "to": ROUTING_TABLE[task_type],
            "reason": f"Triage classified as {task_type}",
        }
    )
    return state
