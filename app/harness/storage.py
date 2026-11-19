"""Trace 存储工具。

该文件负责把 LangGraph 工作流执行过程中产生的事件和最终状态
持久化到本地文件系统，属于 Harness 层的基础组件。

当前设计：
- 每条 trace 保存为一个 JSON 文件
- 文件名使用 `trace_id`
- trace 目录由配置项 `trace_dir` 控制
- trace 结构同时保留 `state_snapshot`（最终状态）和 `events`（节点级事件）

V1 升级要点：
- 从"最终 state 落盘"升级为"事件 + 最终状态"一起落盘
- 新增 graph_path 字段记录节点执行路径
- 保持向后兼容：旧 write_trace 调用方（单次调用）不受影响

使用场景：
- 调试每次请求到底经过了哪些步骤
- 后续做评测、审计、问题回放
- 不依赖额外数据库，开发阶段足够轻量
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.security import redact_sensitive
from app.tenancy import TenantContext
from app.data.database import engine
from sqlalchemy import text


# Trace 信封独立于业务 state 做版本化：这让质量仪表盘只对「相同分代的运行」
# 计算可比较指标，同时让旧的无版本 Trace 仍能作为 legacy 证据被读取。
TRACE_SCHEMA_VERSION = "2.0"
TRACE_GENERATION = "trace-v2"
DEFAULT_WORKFLOW_VERSION = "governed-agent-v1"
DEFAULT_PROMPT_VERSION = "v1"


def _trace_version(payload: dict[str, Any]) -> dict[str, str]:
    """从业务 state 提取非敏感版本维度，用于质量证据分桶。

    函数作用：把 state 里的 trace_generation/workflow_version 等版本字段整理成
    一个扁平字典，随 trace 一起落盘，供质量仪表盘按「版本分代」分组统计。
    输入：payload —— 完整的图执行 state。
    输出：dict[str,str] —— 六个版本维度；缺失的维度用默认值或 "unknown" 兜底。
    难理解点：这里刻意只取非敏感版本信息，不包含任何业务数据或回答内容。
    """
    return {
        "trace_kind": str(payload.get("trace_kind") or ("agent_chat" if "critic_verdict" in payload else "unclassified")),
        "generation": str(payload.get("trace_generation") or TRACE_GENERATION),
        "schema_version": str(payload.get("trace_schema_version") or TRACE_SCHEMA_VERSION),
        "workflow_version": str(payload.get("workflow_version") or DEFAULT_WORKFLOW_VERSION),
        "prompt_version": str(payload.get("prompt_version") or DEFAULT_PROMPT_VERSION),
        "rag_release_version": str(payload.get("rag_release_version") or "unknown"),
        "permission_policy_version": str(payload.get("permission_policy_version") or "unknown"),
    }


def _trace_dir() -> Path:
    """返回 trace 目录，并在目录不存在时自动创建。

    这里把"trace 保存到哪里"的规则集中封装起来，
    避免项目里其他位置直接拼接目录路径。
    """

    path = Path(get_settings().trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_graph_path(node_history: list[dict[str, Any]]) -> list[str]:
    """从节点执行历史中提取图执行路径。

    参数：
    - `node_history`：节点执行历史列表，每个元素包含 `node` 字段。

    返回值：
    - `list[str]`：按执行顺序排列的节点名列表。
    """

    return [entry.get("node", "unknown") for entry in (node_history or [])]


def write_trace(trace_id: str, payload: dict[str, Any]) -> None:
    """将一次 LangGraph 工作流的执行状态和事件持久化为 trace 文件。

    参数：
    - `trace_id`：本次请求的唯一标识，用来生成文件名。
    - `payload`：要保存的完整状态，一般为 graph invoke 后的完整 state。

    写入内容：
    - `graph_path`：节点执行路径（节点名列表）。
    - `state_snapshot`：完整 state 快照。
    - `events`：节点级执行历史。

    使用场景：
    - workflow 中的 persist_trace 节点调用于 graph 末尾。
    - Tests 和调试通过 trace_id 读取 trace 文件。

    主要流程：
        1. 组装 trace 信封：记录时间、版本维度、节点执行路径、脱敏后的 state 快照与事件列表；
        2. 按 trace_backend 分流——postgres 走 RLS upsert（租户隔离），文件后端写本地 JSON；
        3. 文件分支强制生产环境必须使用 postgres，否则直接报错；
        4. 序列化统一带 default=str，保证 Date/Decimal 等诊断字段不阻塞落盘。
    """

    node_history = payload.get("node_history", [])
    snapshot = redact_sensitive(payload)
    # Preference values are runtime personalisation input, not trace evidence.
    # Keep source-need/eligibility metadata for audit, but do not persist the
    # recalled value or candidate content in the trace snapshot.
    if isinstance(snapshot, dict) and isinstance(snapshot.get("preference_context"), dict):
        preference = {
            key: value for key, value in snapshot["preference_context"].items()
            if key not in {"items"}
        }
        if isinstance(preference.get("effects"), list):
            preference["effects"] = [
                {
                    key: value for key, value in effect.items()
                    if key not in {"fields", "value", "effective_fields"}
                }
                for effect in preference["effects"] if isinstance(effect, dict)
            ]
        plan = preference.get("candidate_analysis_plan")
        if isinstance(plan, dict):
            preference["candidate_analysis_plan"] = {
                key: value for key, value in plan.items() if key != "steps"
            } | {
                "step_fields": [
                    item.get("field") for item in (plan.get("steps") or [])
                    if isinstance(item, dict) and item.get("field")
                ],
            }
        snapshot["preference_context"] = preference
    trace = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "trace_version": _trace_version(payload),
        "graph_path": _extract_graph_path(node_history),
        "state_snapshot": snapshot,
        "events": node_history,
    }
    settings = get_settings()
    if settings.trace_backend == "postgres":
        tenant_id = str(payload.get("tenant_id") or "platform")
        user_id = str(payload.get("user_id") or "system")
        # RLS 关键点：set_config 在当前连接会话内设置行级安全上下文，
        # 使随后的 INSERT/UPSERT 只能落在该租户的行上，实现多租户隔离。
        with engine.begin() as connection:
            connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_id})
            connection.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": user_id})
            connection.execute(text("SELECT set_config('app.scope_type', 'seller', true)"))
            connection.execute(text("""INSERT INTO governance.tenant_traces(tenant_id,user_id,trace_id,evaluation_run_id,trace_json)
                VALUES(:tenant,:user,:trace,:run,CAST(:payload AS jsonb))
                ON CONFLICT(tenant_id,trace_id) DO UPDATE SET
                  user_id=EXCLUDED.user_id, evaluation_run_id=EXCLUDED.evaluation_run_id, trace_json=EXCLUDED.trace_json"""), {
                    "tenant": tenant_id, "user": user_id, "trace": trace_id,
                    "run": payload.get("evaluation_run_id"),
                    "payload": json.dumps(trace, ensure_ascii=False, default=str),
                })
        return
    if settings.environment.lower() in {"production", "prod"}:
        # 生产环境必须用 postgres 后端，本地文件在多实例/多租户下无法隔离。
        raise RuntimeError("Production tenant deployment requires TRACE_BACKEND=postgres")
    target = _trace_dir() / f"{trace_id}.json"
    # 诊断字段可能含 Date/Decimal，default=str 保证不因序列化问题而丢失 trace。
    target.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_trace(trace_id: str, tenant_context: TenantContext | None = None) -> dict[str, Any] | None:
    """按 trace_id 读取单条 trace。

    参数：
    - `trace_id`：trace 唯一标识。

    返回值：
    - `dict[str, Any] | None`：trace 内容，文件不存在时返回 None。

    主要流程：
        1. postgres 后端：必须携带租户上下文，设置 RLS 后按 trace_id 查询，无结果返回 None；
        2. 文件后端：目标文件不存在直接返回 None，存在则读取并解析 JSON；
        3. 租户校验：非平台作用域下，Trace 归属租户与请求租户不一致时视为不可见。
    """

    if get_settings().trace_backend == "postgres":
        if tenant_context is None: return None
        # 设置 RLS 上下文后按 trace_id 查询，保证只读本租户的数据。
        with engine.begin() as connection:
            connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_context.tenant_id})
            connection.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": tenant_context.user_id})
            connection.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": tenant_context.scope_type})
            row=connection.execute(text("SELECT trace_json FROM governance.tenant_traces WHERE trace_id=:trace"), {"trace":trace_id}).scalar_one_or_none()
        return row
    target = _trace_dir() / f"{trace_id}.json"
    # 文件不存在即视为无此 trace。
    if not target.exists():
        return None
    trace = json.loads(target.read_text(encoding="utf-8"))
    if tenant_context is not None:
        snapshot = trace.get("state_snapshot", {})
        owner = snapshot.get("tenant_id", "platform")
        trace_user = snapshot.get("user_id")
        # 非平台作用域下，归属租户不一致时视为不可见。
        if tenant_context.scope_type != "platform" and (owner != tenant_context.tenant_id or (trace_user and trace_user != tenant_context.user_id)):
            return None
    return trace


def list_traces(tenant_context: TenantContext | None = None) -> list[dict[str, str]]:
    """列出当前所有 trace 的摘要信息。

    返回值：
    - `list[dict[str, str]]`：每个元素包含 `trace_id` 和 `file`。

    主要流程：
        1. postgres 后端：必须携带租户上下文，设置 RLS 后按创建时间倒序列出该租户的 trace_id；
        2. 文件后端：遍历目录内全部 *.json 文件，按文件名排序；
        3. 租户过滤：非平台作用域时，逐个读取文件头判断归属租户，只保留本租户的 Trace；
           读取失败（损坏/并发写入）的文件直接跳过，不影响整体列表。
    """

    if get_settings().trace_backend == "postgres":
        if tenant_context is None: return []
        # 设置 RLS 上下文后按创建时间倒序列出本租户的 trace_id。
        with engine.begin() as connection:
            connection.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_context.tenant_id})
            connection.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": tenant_context.user_id})
            connection.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": tenant_context.scope_type})
            rows=connection.execute(text("SELECT trace_id FROM governance.tenant_traces ORDER BY created_at DESC")).all()
        return [{"trace_id":row[0],"file":"postgres"} for row in rows]
    items = []
    for path in sorted(_trace_dir().glob("*.json")):
        if tenant_context is not None and tenant_context.scope_type != "platform":
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8")).get("state_snapshot", {})
                owner = snapshot.get("tenant_id", "platform")
                trace_user = snapshot.get("user_id")
            except (OSError, json.JSONDecodeError):
                # 损坏/并发写入的文件直接跳过，不影响整体列表。
                continue
            if owner != tenant_context.tenant_id or (trace_user and trace_user != tenant_context.user_id):
                # 只保留本租户的 Trace。
                continue
        items.append({"trace_id": path.stem, "file": str(path)})
    return items
