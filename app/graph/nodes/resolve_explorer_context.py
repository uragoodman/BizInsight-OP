"""将数据浏览器筛选条件转换为可追溯、可信的工作流上下文。"""

import hashlib
import json

from app.agents.contracts import EvidenceBundle, stable_evidence_id
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.graph.state import AgentState
from app.schemas.explorer import ExplorerQuery
from app.tenancy import current_tenant_context
from app.core.config import get_settings
from app.metrics.authorization import scope_fingerprint


def resolve_explorer_context(state: AgentState) -> AgentState:
    """用户启用浏览上下文时，在图内重新查库并登记为 evidence。

    三种状态：
    - 未启用 → 写入 node_history 记录"scope disabled"，直接返回
    - 已启用但缺筛选条件 → 写入结构化 errors，返回
    - 已启用且条件完整 → 后端重验、生成 controlled_analysis_scope
    """
    state.setdefault("node_history", [])
    enabled = state.get("use_current_explorer_context", False)
    if not enabled:
        state["node_history"].append({
            "node": "resolve_explorer_context",
            "summary": "Explorer scope disabled; using unrestricted query path.",
        })
        state["current_agent"] = "resolve_explorer_context"
        return state

    payload = state.get("explorer_context")
    if not payload:
        state.setdefault("errors", []).append({
            "code": "controlled_scope_missing",
            "message": "已启用数据浏览范围锁定，但缺少筛选条件。请先选择店铺、时间和指标。",
            "retryable": False,
            "agent_name": "resolve_explorer_context",
        })
        state["current_agent"] = "resolve_explorer_context"
        state.setdefault("node_history", []).append({
            "node": "resolve_explorer_context",
            "summary": "Explorer scope enabled but missing filter payload.",
        })
        return state

    # Identity and grants must come from the authenticated request context,
    # never from checkpoint/graph state. In particular, ``as_state`` does not
    # serialize the complete seller-grant set (nor should it).
    context = current_tenant_context()
    if context is None:
        state.setdefault("errors", []).append({
            "code": "tenant_context_missing",
            "message": "无法验证当前用户授权上下文，已停止受控数据查询。请重新登录后重试。",
            "retryable": False,
            "agent_name": "resolve_explorer_context",
        })
        state["current_agent"] = "resolve_explorer_context"
        state.setdefault("node_history", []).append({
            "node": "resolve_explorer_context",
            "summary": "Explorer scope rejected because trusted tenant context is missing.",
        })
        return state

    # Check that the graph is still operating for the identity under which the
    # request entered. These state fields are consistency checks only; they do
    # not construct or expand authorization.
    identity_matches = (
        state.get("tenant_id") == context.tenant_id
        and state.get("user_id") == context.user_id
        and state.get("scope_type") == context.scope_type
        and state.get("user_role") == context.role
    )
    if not identity_matches:
        state.setdefault("errors", []).append({
            "code": "tenant_context_mismatch",
            "message": "当前会话授权上下文已变化，已停止受控数据查询。请重新确认范围后重试。",
            "retryable": False,
            "agent_name": "resolve_explorer_context",
        })
        state["current_agent"] = "resolve_explorer_context"
        state.setdefault("node_history", []).append({
            "node": "resolve_explorer_context",
            "summary": "Explorer scope rejected because graph identity differs from trusted tenant context.",
        })
        return state

    snapshot = preview_explorer(context, ExplorerQuery.model_validate(payload))
    state["explorer_context"] = snapshot["context"]
    state["explorer_snapshot"] = snapshot
    state["controlled_analysis_scope"] = controlled_scope(snapshot).model_dump(mode="json")
    if get_settings().data_versioning_enabled and str(snapshot["context"].get("metric_name")) == "gmv":
        from app.data_versions.service import resolve_gmv_manifest

        manifest = resolve_gmv_manifest(context, state["controlled_analysis_scope"], scope_fingerprint(state["controlled_analysis_scope"]))
        state["data_version_manifest_id"] = manifest["manifest_id"]
        state["data_version_manifest_hash"] = manifest["manifest_hash"]
    state["current_agent"] = "resolve_explorer_context"

    evidence_content = {"context": snapshot["context"], "summary": snapshot["summary"], "lineage": snapshot["lineage"],
                        "data_version_manifest_id": state.get("data_version_manifest_id"),
                        "data_version_manifest_hash": state.get("data_version_manifest_hash")}
    state.setdefault("evidence_bundles", []).append(EvidenceBundle(
        evidence_id=stable_evidence_id("explorer", str(snapshot["context"].get("seller_id")), evidence_content),
        source_type="explorer", source_ref="data-explorer", content=evidence_content,
    ).model_dump(mode="json"))

    scope_config = state["controlled_analysis_scope"]
    # 基于规范化 Scope 字段计算 SHA-256 指纹，用于追溯和判断范围一致
    verified_scope_fp = scope_fingerprint(scope_config)
    selection_id = "explorer:" + hashlib.sha256(
        f"{state.get('trace_id', '')}:{verified_scope_fp}".encode("utf-8")
    ).hexdigest()
    state["explorer_selection_id"] = selection_id
    state.setdefault("node_history", []).append({
        "node": "resolve_explorer_context",
        "summary": (
            f"Locked scope: seller={snapshot['context'].get('seller_id')}, "
            f"metric={snapshot['context'].get('metric_name')}, "
            f"dates={snapshot['context'].get('start_date')}~{snapshot['context'].get('end_date')}, "
            f"grain={snapshot['context'].get('granularity')}, "
            f"fp={verified_scope_fp[:12]}; selection={selection_id[:20]}; manifest={state.get('data_version_manifest_hash') or 'legacy'}"
        ),
    })
    state.setdefault("handoffs", []).append({
        "from": "data_explorer", "to": "triage",
        "reason": f"Server-side explorer snapshot attached: {snapshot['context'].get('seller_id')} {snapshot['context'].get('start_date')}~{snapshot['context'].get('end_date')}",
    })
    return state
