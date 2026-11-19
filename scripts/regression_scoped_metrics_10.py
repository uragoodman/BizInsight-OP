"""Run the ten scoped metric cases through the real lifecycle and MCP path.

This is intentionally a small executable regression harness, not a production
fixture. It computes a trusted baseline through the same scoped MCP gateway,
then runs the full begin -> graph -> finalize lifecycle and compares the trace
tool payload with that baseline.
"""

from __future__ import annotations

import json
import math
from uuid import uuid4

from app.chat_lifecycle.integration import begin_chat_lifecycle, finalize_chat_lifecycle
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.graph.workflow import run_chat_workflow
from app.harness.storage import read_trace
from app.mcp.client import query_scoped_business_data
from app.mcp.schemas import MetricQueryPlan
from app.metrics.authorization import authorize_metric
from app.auth.context import build_authorization_context
from app.schemas.explorer import ExplorerQuery
from app.tenancy import TenantContext, activate_tenant_context


SELLERS = [
    "6560211a19b47992c3666cc44a7e94c0",
    "4a3ca9315b744ce9f8e9374361493884",
    "cc419e0650a3c5ba77189a1882b7556a",
    "1f50f920176fa81dab994f9023523100",
    "da8622b14eb17ae2831f4ac5b9dab84a",
    "955fee9216a65b617aa5c0531780ce60",
    "7a67c85e85bb2ce8582c35f2203ad736",
    "ea8482cd71df3c1969d7b9473ff13abc",
    "4869f7a5dfa277a7dca6462dcf3b52b2",
    "3d871de0142ce09b7081e2b9d1733cb1",
]

CASES = [
    ("seller_avg_freight_per_order", "month", "\u6309\u6708\u5206\u6790\u5e97\u94fa\u5e73\u5747\u6bcf\u5355\u8fd0\u8d39\uff0c\u54ea\u4e2a\u6708\u6700\u9ad8\uff0c\u54ea\u4e2a\u6708\u6700\u4f4e\uff1f"),
    ("seller_items_per_order", "month", "\u6309\u6708\u5206\u6790\u5e97\u94fa\u5e73\u5747\u6bcf\u5355\u5546\u54c1\u6570\uff0c\u6700\u9ad8\u548c\u6700\u4f4e\u5206\u522b\u662f\u54ea\u4e2a\u6708\uff1f"),
    ("seller_freight_to_gmv_rate", "quarter", "\u6309\u5b63\u5ea6\u5206\u6790\u5e97\u94fa\u8fd0\u8d39\u5360\u6210\u4ea4\u989d\u7684\u6bd4\u4f8b\uff0c\u54ea\u4e2a\u5b63\u5ea6\u6700\u9ad8\uff1f"),
    ("seller_avg_item_price", "month", "\u6309\u6708\u5206\u6790\u5e97\u94fa\u5e73\u5747\u5546\u54c1\u552e\u4ef7\u7684\u53d8\u5316\uff0c\u6700\u9ad8\u548c\u6700\u4f4e\u6708\u4efd\u662f\u4ec0\u4e48\uff1f"),
    ("seller_active_product_count", "month", "\u6bcf\u6708\u6709\u9500\u552e\u7684\u5546\u54c1\u6570\u662f\u591a\u5c11\uff0c\u54ea\u4e2a\u6708\u6700\u591a\uff1f"),
    ("seller_avg_payment_installments", "month", "\u6309\u6708\u7edf\u8ba1\u5e97\u94fa\u5e73\u5747\u5206\u671f\u6570\uff0c\u54ea\u4e2a\u6708\u6700\u9ad8\uff1f"),
    ("seller_credit_card_payment_share", "month", "\u6309\u6708\u7edf\u8ba1\u4fe1\u7528\u5361\u652f\u4ed8\u91d1\u989d\u5360\u6bd4\uff0c\u54ea\u4e2a\u6708\u6700\u9ad8\u6700\u4f4e\uff1f"),
    ("seller_new_customer_rate", "month", "\u6309\u6708\u5206\u6790\u5e97\u94fa\u65b0\u5ba2\u5360\u6bd4\u7684\u53d8\u5316\uff0c\u6700\u9ad8\u662f\u54ea\u4e2a\u6708\uff1f"),
    ("seller_avg_late_delivery_days", "month", "\u6309\u6708\u7edf\u8ba1\u5e97\u94fa\u5e73\u5747\u5ef6\u8bef\u9001\u8fbe\u5929\u6570\uff0c\u54ea\u4e2a\u6708\u6700\u9ad8\uff1f"),
    ("seller_order_bad_review_rate", "month", "\u6309\u6708\u5206\u6790\u5e97\u94fa\u8ba2\u5355\u5dee\u8bc4\u7387\uff0c\u54ea\u4e2a\u6708\u6700\u9ad8\uff1f"),
]


def _points(payload: dict) -> list[dict]:
    return list(((payload.get("result") or {}).get("trend_result") or {}).get("points") or [])


def main() -> None:
    context = TenantContext(
        tenant_id="platform", user_id=f"regression-{uuid4().hex[:8]}",
        role="admin", scope_type="platform",
    )
    results: list[dict[str, object]] = []
    with activate_tenant_context(context):
        for index, ((metric, grain, question), seller) in enumerate(zip(CASES, SELLERS), start=1):
            case_context = context
            payload = {
                "seller_id": seller, "metric_name": metric,
                "start_date": "2017-09-01", "end_date": "2018-08-31",
                "granularity": grain,
            }
            snapshot = preview_explorer(case_context, ExplorerQuery.model_validate(payload))
            scope = controlled_scope(snapshot)
            scope_data = scope.model_dump(mode="json")
            baseline_authz = build_authorization_context(
                case_context,
                allowed_metric_ids=scope.allowed_metrics,
                allowed_dimensions=scope.allowed_dimensions,
                metric_scope_type="seller",
            )
            baseline_scope_data = {
                **scope_data,
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "permissions": context.role,
                "scope_type": "seller",
                "authorization_fingerprint": baseline_authz.authz_digest,
                "permission_version": baseline_authz.permission_version,
                "policy_set_version": baseline_authz.policy_set_version,
                "authz_ref": baseline_authz.authz_ref,
            }
            authorized = authorize_metric(
                metric, provenance_type="explorer_selection",
                provenance_id=f"baseline-{index}", scope="seller",
                scope_data=baseline_scope_data,
            )
            baseline_plan = MetricQueryPlan(
                query_type="metric_trend", metric_name=metric, granularity=grain,
                authorized_metric=authorized,
            )
            baseline = query_scoped_business_data(question, scope_data, plan=baseline_plan)

            session_id = f"regression-10-{uuid4().hex}"
            client_turn_id = f"regression-case-{index}-{uuid4().hex}"
            trace_id = str(uuid4())
            binding = begin_chat_lifecycle(
                case_context, session_id=session_id, client_turn_id=client_turn_id,
                question=question, explorer_payload=payload,
                thread_id=f"regression-thread-{uuid4().hex}", trace_id=trace_id,
            )
            if binding is None or binding.execution is None:
                results.append({"case": index, "metric": metric, "status": "lifecycle_not_created"})
                continue
            try:
                answer = run_chat_workflow(
                    question, thread_id=binding.execution.thread_id,
                    tenant_context=case_context, explorer_context=payload,
                    use_current_explorer_context=True, chat_session_id=session_id,
                    chat_turn_id=binding.turn.turn_id, persistence_turn_id=binding.persistence_turn_id,
                    authorization_boundary=binding.authorization_boundary,
                    trace_id=binding.execution.trace_id,
                )
                trace = read_trace(binding.execution.trace_id, case_context) or {}
                state = trace.get("state_snapshot") or trace
                actual_result = state.get("tool_result") or {}
                actual_points = _points(actual_result)
                expected_points = _points(baseline)
                decision = answer.get("turn_context", {}).get("metric_planning_decision", {})
                passed = not answer.get("error") and not answer.get("clarification") and not answer.get("errors") and decision.get("status") == "confirmed" and len(actual_points) == len(expected_points)
                mismatch = None
                if passed:
                    actual = [(str(x.get("bucket") or x.get("metric_date")), float(x.get("value"))) for x in actual_points]
                    expected = [(str(x.get("bucket") or x.get("metric_date")), float(x.get("value"))) for x in expected_points]
                    passed = all(
                        actual_item[0] == expected_item[0]
                        and math.isclose(actual_item[1], expected_item[1], rel_tol=1e-9, abs_tol=1e-8)
                        for actual_item, expected_item in zip(actual, expected)
                    )
                    if not passed:
                        mismatch = {"actual": actual[:3], "expected": expected[:3]}
                results.append({
                    "case": index, "seller": seller, "metric": metric, "grain": grain,
                    "status": "passed" if passed else "failed",
                    "trace_id": binding.execution.trace_id,
                    "tool": (actual_result.get("tool_name") or state.get("tool_result", {}).get("tool_name")),
                    "decision": decision.get("status"),
                    "expected_points": len(expected_points),
                    "actual_points": len(actual_points),
                    "baseline_status": ((baseline.get("result") or {}).get("status")),
                    "baseline_reason": ((baseline.get("result") or {}).get("reason")),
                    "mismatch": mismatch,
                    "error": (answer.get("errors") or state.get("errors") or [])[-1:] if not passed else [],
                })
                finalize_chat_lifecycle(binding, result=answer, status="completed" if passed else "failed")
            except Exception as exc:
                finalize_chat_lifecycle(binding, result={"answer": ""}, status="failed", error_code=type(exc).__name__, error_detail={"message": str(exc)})
                results.append({"case": index, "seller": seller, "metric": metric, "status": "exception", "trace_id": binding.execution.trace_id, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"passed": sum(item.get("status") == "passed" for item in results), "total": len(results), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
