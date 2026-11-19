"""Render a unified, non-executing complex-analysis clarification card."""

from __future__ import annotations

from app.graph.state import AgentState


def complex_clarification(state: AgentState) -> AgentState:
    decision = state.get("complex_route_decision") or {}
    evidence = decision.get("evidence") or {}
    existing = state.get("complex_clarification") or {}
    existing_reason = str(existing.get("reason_code") or "")
    evidence_reason = str(evidence.get("clarification_reason") or "")
    # Preserve a specific deterministic router/compiler reason when an earlier
    # generic clarification placeholder was already written to state.
    reason = (
        evidence_reason
        if evidence_reason and existing_reason in {"", "needs_context"}
        else existing_reason or evidence_reason or "needs_context"
    )

    if reason == "semantic_router_not_ready":
        clarification = {
            "status": "complex_analysis_unavailable",
            "reason_code": "semantic_router_not_ready",
        }
        answer = "复杂分析功能当前尚未启用；本轮未创建或执行复杂分析计划。"
    elif reason in {
        "semantic_interpreter_failed",
        "router_validation_failed",
        "sealed_decision_rejected",
        "canary_not_authorized",
    }:
        clarification = {
            "status": "complex_analysis_unavailable",
            "reason_code": reason,
        }
        answer = "语义路由或灰度校验未通过，本轮未创建或执行复杂分析计划。"
    elif reason in {
        "customer_rfm_segmentation:missing_required_arguments",
        "customer_rfm_segmentation:explicit_method_missing_required_arguments",
        "customer_rfm_segmentation:insufficient_basis",
    }:
        clarification = {
            "status": "needs_context",
            "reason_code": reason,
            "missing_evidence": ["two_distinct_rfm_basis"],
        }
        answer = "请从购买频次、金额、最近消费时间中选择至少两个不同的客户分层依据。"
    elif reason == "dimension_required":
        clarification = {
            "status": "needs_context",
            "reason_code": reason,
        }
        answer = "请明确要按哪个维度展示统计贡献。"
    elif reason == "multiple_capabilities":
        candidates = existing.get("candidate_templates")
        if isinstance(candidates, list) and len(candidates) >= 2:
            clarification = {
                "status": "complex_template_required",
                "reason_code": "multiple_capabilities",
                "candidate_templates": candidates,
            }
            labels = [
                str(item.get("display_name"))
                for item in candidates
                if isinstance(item, dict) and item.get("display_name")
            ]
            answer = (
                f"这项分析可以使用{ '或'.join(labels) }。请明确选择一种；选择后系统会先生成待确认计划。"
                if len(labels) >= 2
                else "这项请求匹配多个分析模板，请补充分析方式后重试。"
            )
        else:
            clarification = {"status": "needs_context", "reason_code": "multiple_capabilities"}
            answer = "这项请求匹配多个分析模板，请补充分析方式后重试；系统尚未创建计划。"
    elif reason == "template_confirmation_rejected":
        clarification = {
            "status": "complex_template_confirmation_rejected",
            "reason_code": "template_confirmation_rejected",
        }
        answer = "所选分析模板未注册、当前不可见，或该请求已明确排除该分析；未创建计划。"
    elif reason == "scope_required":
        clarification = {
            "status": "complex_scope_required",
            "reason_code": "scope_required",
            "reason": "复杂分析需要服务端验证的店铺和日期范围。",
        }
        answer = "请先选择店铺、开始日期和结束日期，再确认复杂分析计划。"
    elif reason == "comparison_dimension_not_supported":
        clarification = {
            "status": "needs_context",
            "reason_code": "comparison_dimension_not_supported",
        }
        answer = "当前查询不能同时按商品品类分组并按支付方式比较。请先选择其中一个维度；本轮没有执行数据查询。"
    elif reason == "dimension_not_supported_by_metric":
        clarification = {
            "status": "needs_context",
            "reason_code": "dimension_not_supported_by_metric",
        }
        answer = "当前指标暂不支持请求的统计贡献维度。请改选该指标支持的统计贡献维度；本轮没有执行数据查询。"
    elif reason == "scope_rejected":
        clarification = {
            "status": "complex_scope_rejected",
            "reason_code": "scope_rejected",
        }
        answer = "当前账号无法访问所选店铺或日期范围，未创建计划。"
    elif reason == "plan_creation_rejected":
        clarification = {
            "status": "complex_plan_rejected",
            "reason_code": "plan_creation_rejected",
        }
        answer = "计划校验未通过，未创建计划。请检查分析范围或重新发起请求。"
    else:
        clarification = {
            "status": "needs_context",
            "reason_code": "needs_context",
        }
        answer = "请补充分析对象或分析操作，以便继续处理请求。"

    state["complex_clarification"] = clarification
    state["task_type"] = "complex_analysis"
    state["answer"] = answer
    state.setdefault("node_history", []).append(
        {"node": "complex_clarification", "summary": f"Clarification reason={reason}."}
    )
    return state
