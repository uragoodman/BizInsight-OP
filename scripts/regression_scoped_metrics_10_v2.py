"""Calculate and execute a second ten-case scoped-metric regression set.

The script first obtains a server-side baseline through the controlled MCP
gateway, then submits the same question through the real chat lifecycle and
compares the resulting trend points.
"""

from __future__ import annotations

import json
import math
from uuid import uuid4

from app.auth.context import build_authorization_context
from app.chat_lifecycle.integration import begin_chat_lifecycle, finalize_chat_lifecycle
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.graph.workflow import run_chat_workflow
from app.harness.storage import read_trace
from app.mcp.client import query_scoped_business_data
from app.mcp.schemas import MetricQueryPlan
from app.metrics.authorization import authorize_metric
from app.schemas.explorer import ExplorerQuery
from app.tenancy import TenantContext, activate_tenant_context
from app.metrics.registry import get_registry


SELLERS = [
    "7c67e1448b00f6e969d365cea6b010ab",
    "8b321bb669392f5163d04c59e235e066",
    "1025f0e2d44d7041d6cf58b6550e0bfa",
    "620c87c171fb2a6dd6e8bb4dec959fc6",
    "a1043bafd471dff536d0c462352beb48",
    "cca3071e3e9bb7d12640c9fbe2301306",
    "f8db351d8c4c4c22c6835c19a46f01b0",
    "e9779976487b77c6d4ac45f75ec7afe9",
    "fa1c13f2614d7b5c4749cbc52fecda94",
    "7d13fca15225358621be4086e1eb0964",
]

CASES = [
    ("seller_avg_early_delivery_days", "按月分析店铺平均提前送达天数，哪个月最高，哪个月最低？"),
    ("seller_avg_delivery_estimate_abs_error_days", "按月分析店铺预计送达误差绝对值，哪个月最高？"),
    ("seller_p90_order_to_delivery_days", "按月分析店铺 P90 下单至签收时长，哪个月最长？"),
    ("seller_missing_delivery_timestamp_rate", "按月分析店铺签收时间缺失率，哪个月最高？"),
    ("seller_historical_returning_customer_count", "按月统计店铺历史回流客户数，哪个月最多？"),
    ("seller_historical_returning_customer_rate", "按月统计店铺历史回流客户率，哪个月最高？"),
    ("seller_new_customer_count", "按月统计店铺新客数，哪个月最多？"),
    ("seller_top1_product_gross_share", "按月分析店铺 Top 1 商品销售额占比，哪个月最高？"),
    ("seller_top3_category_gross_share", "按月分析店铺 Top 3 品类销售额占比，哪个月最高？"),
    ("seller_new_product_gross_share", "按月分析店铺新品销售额占比，哪个月最高？"),
]

START_DATE = "2017-09-01"
END_DATE = "2018-08-31"


def _points(payload: dict) -> list[dict]:
    result = payload.get("result") or {}
    trend = result.get("trend_result") or {}
    return list(trend.get("points") or [])


def _summary(metric: str, points: list[dict]) -> dict[str, object]:
    values = [(str(point.get("bucket")), float(point.get("value"))) for point in points]
    highest = max(value for _, value in values)
    lowest = min(value for _, value in values)
    return {
        "metric": metric,
        "points": [{"month": month, "value": value} for month, value in values],
        "highest": [month for month, value in values if math.isclose(value, highest, abs_tol=1e-9)],
        "highest_value": highest,
        "lowest": [month for month, value in values if math.isclose(value, lowest, abs_tol=1e-9)],
        "lowest_value": lowest,
    }


def main() -> None:
    context = TenantContext(
        tenant_id="platform",
        user_id=f"regression-v2-{uuid4().hex[:8]}",
        role="admin",
        scope_type="platform",
    )
    output: list[dict[str, object]] = []
    with activate_tenant_context(context):
        for case_no, ((metric, question), seller_id) in enumerate(zip(CASES, SELLERS), start=1):
            explorer_payload = {
                "seller_id": seller_id,
                "metric_name": metric,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "granularity": "month",
            }
            snapshot = preview_explorer(context, ExplorerQuery.model_validate(explorer_payload))
            scope = controlled_scope(snapshot)
            scope_data = scope.model_dump(mode="json")
            authz = build_authorization_context(
                context,
                allowed_metric_ids=scope.allowed_metrics,
                allowed_dimensions=scope.allowed_dimensions,
                metric_scope_type="seller",
            )
            authorized = authorize_metric(
                metric,
                provenance_type="explorer_selection",
                provenance_id=f"baseline-v2-{case_no}",
                scope="seller",
                scope_data={
                    **scope_data,
                    "tenant_id": context.tenant_id,
                    "user_id": context.user_id,
                    "permissions": context.role,
                    "scope_type": "seller",
                    "authorization_fingerprint": authz.authz_digest,
                    "permission_version": authz.permission_version,
                    "policy_set_version": authz.policy_set_version,
                    "authz_ref": authz.authz_ref,
                },
            )
            baseline_plan = MetricQueryPlan(
                query_type="metric_trend",
                metric_name=metric,
                granularity="month",
                authorized_metric=authorized,
            )
            baseline = query_scoped_business_data(question, scope_data, plan=baseline_plan)
            baseline_points = _points(baseline)
            baseline_summary = _summary(metric, baseline_points) if baseline_points else {
                "metric": metric,
                "points": [],
                "error": (baseline.get("result") or {}).get("reason"),
            }

            session_id = f"regression-v2-{uuid4().hex}"
            trace_id = str(uuid4())
            binding = begin_chat_lifecycle(
                context,
                session_id=session_id,
                client_turn_id=f"regression-v2-case-{case_no}-{uuid4().hex}",
                question=question,
                explorer_payload=explorer_payload,
                thread_id=f"regression-v2-thread-{uuid4().hex}",
                trace_id=trace_id,
            )
            if binding is None or binding.execution is None:
                output.append({"case": case_no, "metric": metric, "status": "lifecycle_not_created", "baseline": baseline_summary})
                continue

            try:
                answer = run_chat_workflow(
                    question,
                    thread_id=binding.execution.thread_id,
                    tenant_context=context,
                    explorer_context=explorer_payload,
                    use_current_explorer_context=True,
                    chat_session_id=session_id,
                    chat_turn_id=binding.turn.turn_id,
                    persistence_turn_id=binding.persistence_turn_id,
                    authorization_boundary=binding.authorization_boundary,
                    trace_id=binding.execution.trace_id,
                )
                trace = read_trace(binding.execution.trace_id, context) or {}
                state = trace.get("state_snapshot") or trace
                actual_result = state.get("tool_result") or {}
                actual_points = _points(actual_result)
                decision = answer.get("turn_context", {}).get("metric_planning_decision", {})
                expected = [(str(point.get("bucket")), float(point.get("value"))) for point in baseline_points]
                actual = [(str(point.get("bucket")), float(point.get("value"))) for point in actual_points]
                passed = (
                    not answer.get("error")
                    and not answer.get("clarification")
                    and not answer.get("errors")
                    and decision.get("status") == "confirmed"
                    and len(actual) == len(expected)
                    and all(
                        actual_item[0] == expected_item[0]
                        and math.isclose(actual_item[1], expected_item[1], rel_tol=1e-9, abs_tol=1e-8)
                        for actual_item, expected_item in zip(actual, expected)
                    )
                )
                output.append({
                    "case": case_no,
                    "seller_id": seller_id,
                    "metric": metric,
                    "question": question,
                    "status": "passed" if passed else "failed",
                    "trace_id": binding.execution.trace_id,
                    "decision": decision.get("status"),
                    "tool": actual_result.get("tool_name"),
                    "baseline": baseline_summary,
                    "project_answer": state.get("answer") or state.get("final_answer"),
                    "actual_points": [{"month": month, "value": value} for month, value in actual],
                    "errors": answer.get("errors") or state.get("errors") or [],
                })
                finalize_chat_lifecycle(binding, result=answer, status="completed" if passed else "failed")
            except Exception as exc:
                finalize_chat_lifecycle(
                    binding,
                    result={"answer": ""},
                    status="failed",
                    error_code=type(exc).__name__,
                    error_detail={"message": str(exc)},
                )
                output.append({
                    "case": case_no,
                    "seller_id": seller_id,
                    "metric": metric,
                    "question": question,
                    "status": "exception",
                    "trace_id": binding.execution.trace_id,
                    "baseline": baseline_summary,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                })

    print(json.dumps({
        "passed": sum(item.get("status") == "passed" for item in output),
        "total": len(output),
        "results": output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
