"""带版本号的 Agent 间契约模型（Pydantic）。

本模块定义了所有图节点间通信的结构化契约。每个契约类都：
- 继承 ContractModel（严格 extra="forbid" + schema 版本校验）
- 通过 Pydantic 校验确保节点输出不会带着脏数据跨越边界
- 在 runtime.py 的 _validate_boundary() 中被实例化校验

契约校验的作用：
- 下游节点可以安全地假设上游的字段一定存在且类型正确
- 不需要在每个节点重复 if/else 防御式编程
- 版本升级时可以通过 migrate_compatible_payload 平滑兼容

SIGNATURE 声明 — 注册节点读取 & 写入 state 的 spec：
- triage          → TaskType, next_agent: AgentName
- metric_definition → metric_context, rag_citations
- data_query      → tool_calls, tool_result
- root_cause      → analysis_result, pending_query
- report          → analysis_result, report_id, report_risks
- generate_answer → answer, claims
- critic          → critic_verdict, critic_revision_target, critic_decisions
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.mcp.schemas import MetricQueryPlan as McpMetricQueryPlan


CONTRACT_SCHEMA_VERSION = "1.0"


def stable_evidence_id(source_type: str, source_ref: str, content: dict[str, Any]) -> str:
    """生成可复现的证据 ID，使答案可回溯到来源。

    对 (来源类型, 引用键, 内容) 三元组做 SHA256 哈希，
    前缀 "ev_" 便于在日志和 trace 中识别这是证据 ID。

    使用场景：
    - RAG 检索结果存入 evidence_bundles 时生成 ID
    - MCP 查询结果作为证据链时生成 ID
    - 最终 answer 的 claims 通过 source_evidence_ids 回溯

    参数：
    - source_type：证据来源类型（"rag"/"mcp"/"root_cause"/"report"/"explorer"/"result_context"）
    - source_ref：来源引用（如文档路径、工具名）
    - content：证据内容字典

    返回：
    - str：格式为 "ev_xxxxxxxxxxxxxxxx" 的稳定 ID
    """
    raw = json.dumps(
        {"source_type": source_type, "source_ref": source_ref, "content": content},
        sort_keys=True, default=str,
    )
    return f"ev_{sha256(raw.encode('utf-8')).hexdigest()[:16]}"


# ============================================================
# 基础模型
# ============================================================

class ContractModel(BaseModel):
    """严格基础模型：禁止额外字段 + schema 版本自动兼容。

    所有 Agent 间的契约模型都应继承此类而非直接继承 BaseModel。
    - extra="forbid"：任何不在模型中定义的字段都会触发 ValidationError
    - migrate_compatible_payload：自动将旧版本(0.x / 无版本)升级到当前版本
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: str = CONTRACT_SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def migrate_compatible_payload(cls, value: Any) -> Any:
        """兼容旧版本或未标版本的 payload，拒绝不兼容的大版本。"""
        if not isinstance(value, dict):
            return value
        version = value.get("schema_version")
        if version is None or str(version).startswith("0."):
            return {**value, "schema_version": CONTRACT_SCHEMA_VERSION}
        if not str(version).startswith("1."):
            raise ValueError(f"Unsupported contract schema version: {version}")
        return value


# ============================================================
# 枚举类型
# ============================================================

class TaskType(str, Enum):
    """triage 节点可判别出的任务类型。"""
    CLARIFICATION = "clarification"          # 分类未能安全确定，转入澄清节点
    METRIC_DEFINITION = "metric_definition"  # 指标口径/定义问询
    METRIC_QUERY = "metric_query"            # 单点/趋势/对比数据查询
    ANOMALY_ANALYSIS = "anomaly_analysis"    # 异常归因分析
    REPORT_GENERATION = "report_generation"  # 周期性报告生成
    SELLER_CATALOG = "seller_catalog"        # 查询当前身份可访问的店铺目录


class AgentName(str, Enum):
    """图节点名称枚举，与 workflow.py 中注册的节点名一一对应。"""
    TRIAGE = "triage"
    CONTEXT_CLARIFICATION = "context_clarification"
    METRIC_DEFINITION = "metric_definition"
    DATA_QUERY = "data_query"
    ROOT_CAUSE = "root_cause"
    REPORT = "report"
    SELLER_CATALOG = "seller_catalog"
    GENERATE_ANSWER = "generate_answer"
    CRITIC = "critic"


class CriticRevisionTarget(str, Enum):
    """Whitelisted graph nodes that Critic may request for revision."""

    TRIAGE = "triage"
    METRIC_DEFINITION = "metric_definition"
    DATA_QUERY = "data_query"
    DIAGNOSTIC_PLAN = "diagnostic_plan"
    REPORT = "report"
    GENERATE_ANSWER = "generate_answer"


# ============================================================
# 节点契约（每个节点输出时由 runtime._validate_boundary 校验）
# ============================================================

class TriageDecision(ContractModel):
    """triage 节点的输出契约：必须包含任务类型、下游目标、分类理由。"""
    task_type: TaskType
    next_agent: AgentName
    reason: str = Field(min_length=1)


class MetricQueryPlan(McpMetricQueryPlan):
    """带版本号的 Data Query 查询计划，位于 Agent ↔ MCP 契约边界。

    继承自 MCP 层的 schemas.MetricQueryPlan，增加了 schema_version 校验。
    """
    model_config = ConfigDict(extra="forbid")
    schema_version: str = CONTRACT_SCHEMA_VERSION


class ToolInvocation(ContractModel):
    """单次 MCP 工具调用的完整记录。

    每次 data_query 节点执行后都会产生一条 ToolInvocation，
    追加到 state.tool_calls 列表中。
    """
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    idempotency_key: str = Field(default_factory=lambda: str(uuid4()))
    server: str                        # "local-mcp"、"remote-mcp" 或 "tenant-mcp"
    tool_name: str                     # 调用的工具名（如 "query_gmv"）
    planned_metric_id: Optional[str] = None
    planned_operation: Optional[str] = None
    arguments: dict[str, Any]          # 传给工具的参数字典
    status: Literal["ok", "not_found", "unsupported", "failed"]  # 调用结果状态
    duration_ms: Optional[float] = Field(default=None, ge=0)  # 工具执行耗时（毫秒）
    evidence_id: Optional[str] = None
    authorization_snapshot: Optional[dict[str, Any]] = None
    gateway_request_id: Optional[str] = None


class EvidenceBundle(ContractModel):
    """一条带稳定 ID 的可追溯证据。

    所有 RAG 检索结果、MCP 查询结果、归因分析结论都通过此类
    存入 state.evidence_bundles，供最终答案的 claims 溯源。
    """
    evidence_id: str                              # stable_evidence_id 生成的唯一 ID
    source_type: Literal["rag", "mcp", "root_cause", "report", "explorer", "result_context"]  # 证据来源
    source_ref: str                               # 来源引用（文档路径、工具名等）
    content: dict[str, Any]                       # 证据内容
    trusted: bool = True                          # 是否来自受信源（RAG 可信文档、MCP 工具结果）
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Claim(ContractModel):
    """最终答案中一条可追溯声明的结构化表示。

    generate_answer 节点将答案拆解为若干 Claim，每条都可回溯到
    具体的 evidence_id。Critic 可逐一核实 claim 是否有证据支撑。
    """
    claim_type: Literal["definition", "metric", "analysis", "recommendation"]
    value: str = Field(min_length=1)              # 声明文本
    source_evidence_ids: list[str] = Field(default_factory=list)  # 支撑证据 ID 列表


class CriticDecision(ContractModel):
    """critic 节点的输出契约：裁决结论 + 修订目标 + 问题清单。

    revision_target 只有在 verdict="needs_revision" 时才必须非空，
    workflow.py 的路由函数据此决定返工目标节点。
    """
    verdict: Literal["approved", "needs_revision", "rejected", "human_review"]
    revision_target: Optional[CriticRevisionTarget] = None      # 返工目标节点
    issues: list[str] = Field(default_factory=list)         # 本次审核发现的问题
    residual_issues: list[str] = Field(default_factory=list) # 先前审核未解决的问题


class AgentError(ContractModel):
    """节点执行中捕获的结构化错误。

    与 Python 异常不同，AgentError 写入 state.errors 后图流继续：
    - 有 error 的节点执行完后，路由函数将其导向 critic 的 error 快速路径
    - critic 看到 error 时直接裁决为 rejected，不执行正常审核
    """
    code: str                    # 错误码（"agent_execution_failed"、"token_budget_exceeded" 等）
    message: str                 # 人类可读的错误描述
    retryable: bool = False      # 是否可重试（true 时图可能将该节点重新入队）
    agent_name: Optional[str] = None  # 发生错误的节点名
    cause_type: Optional[str] = None  # 原始异常类型名（用于分类和统计）
    public_message: Optional[str] = None      # 可向用户展示的安全中文描述，用于 SSE 事件
    diagnostic_message: Optional[str] = None  # 经脱敏后的诊断摘要，仅写入 Trace/日志
    failure_context: Optional[dict[str, Any]] = None  # 已分类的工具/数据库失败上下文
    request_id: Optional[str] = None           # 关联本次失败的逻辑请求


class FinalResponse(ContractModel):
    """工作流最终返回给 API 层的输出契约。"""
    task_type: TaskType
    answer: str                  # 最终答案文本
    trace_id: str                # 本次执行的追踪 ID
    claims: list[Claim] = Field(default_factory=list)  # 可追溯声明列表


# ============================================================
# 节点输出契约（供 _validate_boundary 校验各节点的输出结构）
# ============================================================

class MetricDefinitionOutput(ContractModel):
    """metric_definition 节点的输出契约。"""
    metric_context: str                              # RAG 检索拼接的上下文文本
    citations: list[dict[str, Any]] = Field(default_factory=list)  # 引用源列表


class DataQueryOutput(ContractModel):
    """data_query 节点的输出契约。"""
    invocation: ToolInvocation    # 工具调用记录
    result: dict[str, Any] = Field(default_factory=dict)  # 工具返回的原始结果


class AnalysisOutput(ContractModel):
    """root_cause / report 节点的输出契约。"""
    summary: str                  # 分析摘要
    task_type: TaskType           # 任务类型标记


class AnswerOutput(ContractModel):
    """generate_answer 节点的输出契约。"""
    answer: str                   # 最终答案文本
    claims: list[Claim] = Field(default_factory=list)  # 可追溯声明列表


# ============================================================
# 执行计划（供未来的受控规划器使用）
# ============================================================

class ExecutionPlan(ContractModel):
    """经过校验的、白名单内的执行计划。

    当前 triage 节点使用简化的单步路由；未来规划器拆解复杂
    多步查询时，执行计划将被本模型校验后才进入图流。
    """

    agents: list[AgentName] = Field(min_length=1)  # 执行计划中的 Agent 序列
    max_steps: int = Field(default=16, ge=1, le=64)       # 最大步骤数
    max_tool_calls: int = Field(default=8, ge=0, le=32)   # 最大 MCP 工具调用次数
    dependencies: dict[AgentName, list[AgentName]] = Field(default_factory=dict)  # Agent 间依赖关系

    @model_validator(mode="after")
    def validate_topology(self) -> "ExecutionPlan":
        """校验依赖关系：依赖的 Agent 必须在执行计划中、且排在当前 Agent 之前。"""
        positions = {agent: index for index, agent in enumerate(self.agents)}
        for agent, dependencies in self.dependencies.items():
            if agent not in positions or any(
                dep not in positions or positions[dep] >= positions[agent]
                for dep in dependencies
            ):
                raise ValueError(
                    "Execution plan dependencies must reference earlier registered agents."
                )
        return self
