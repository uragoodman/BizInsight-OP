"""聊天与人工审核的请求/响应 Schema。

定义了 /chat、/chat/stream、/chat/{thread_id}/review 三个接口的
请求体和响应体模型。所有 Field 的 constraints（min_length、pattern）
在 FastAPI 层面即被校验，脏数据不会进入图引擎。
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from app.schemas.analysis_presentation import AnalysisPresentation
from app.schemas.explorer import ExplorerQuery, SignedSellerScopePayload


class ChatRequest(BaseModel):
    """聊天请求体：用户自然语言问题 + 可选的持久化线程 ID。

    thread_id 用于 LangGraph 检查点：同一 thread_id 的多次请求共享
    同一个对话历史和 state 快照。
    """
    message: str = Field(..., min_length=1, description="用户自然语言查询")
    thread_id: str | None = Field(
        default=None, min_length=1,
        description="可选：持久化工作流线程 ID（用于跨请求检查点恢复）",
    )
    use_current_explorer_context: bool = False
    explorer_context: ExplorerQuery | None = None
    seller_scope_payload: SignedSellerScopePayload | None = None
    memory_session_id: str | None = Field(
        default=None, min_length=1, max_length=128,
        description=(
            "Deprecated compatibility field for non-durable memory-only callers. "
            "Durable chat uses chat_session_id as its sole conversation identity."
        ),
    )
    memory_enabled: bool = Field(
        default=True,
        description=(
            "Deprecated compatibility override for legacy memory-only callers. "
            "Durable chat (chat_session_id) ignores this field: analysis-memory "
            "writes follow the server-side Critic/scope contract and "
            "MEM0_ENABLED governs personalization."
        ),
    )
    chat_session_id: str | None = Field(default=None, min_length=1, max_length=128)
    chat_turn_id: str | None = Field(default=None, min_length=1, max_length=128)
    # Explicit user retry creates a new Turn while retaining a provenance link
    # to the previous user-visible turn.  System repair must use a new
    # Execution on the same Turn instead.
    inherited_from_turn_id: str | None = Field(default=None, min_length=1, max_length=128)
    task_id: str | None = Field(default=None, min_length=8, max_length=64)
    # Sent only after the UI has shown a clarification card. These are metric
    # IDs, not tools or query parameters; the backend still validates them
    # against the current Registry catalog and trusted Scope.
    metric_confirmation: list[str] | None = Field(default=None, max_length=100)
    # New authoritative confirmation path.  The server resolves these IDs to
    # mappings bound to user/session/scope/catalog; clients never submit the
    # executable metric as the source of truth.
    semantic_mapping_ids: list[str] | None = Field(default=None, max_length=100)
    # Mapping IDs the user explicitly chose to skip.  The server resolves
    # these IDs to the persisted mention and records the decision; the client
    # never supplies executable metric IDs as a substitute.
    skipped_semantic_mapping_ids: list[str] | None = Field(default=None, max_length=100)
    @model_validator(mode="after")
    def seller_scope_is_exclusive(self) -> "ChatRequest":
        if self.seller_scope_payload is not None and (
            self.explorer_context is not None or self.use_current_explorer_context
        ):
            raise ValueError("seller_scope_payload cannot be combined with Explorer scope")
        return self


class SemanticMappingConfirmationRequest(BaseModel):
    mapping_id: str = Field(..., min_length=8, max_length=128)
    chat_session_id: str | None = Field(default=None, min_length=1, max_length=128)
    explorer_context: ExplorerQuery | None = None


class SemanticMappingSkipRequest(BaseModel):
    mapping_id: str = Field(..., min_length=8, max_length=128)
    chat_session_id: str | None = Field(default=None, min_length=1, max_length=128)
    explorer_context: ExplorerQuery | None = None


class ChatResponse(BaseModel):
    """聊天响应体：任务类型、最终答案、追踪 ID、结构化展示结论。"""
    task_type: str    # triage 判定的任务类型（metric_query / anomaly_analysis / …）
    answer: str       # 工作流生成的最终答案文本
    trace_id: str     # 本次执行的唯一追踪 ID，可用于 /traces 查询
    error: bool = False
    errors: list[dict] = Field(default_factory=list)
    clarification: dict | None = None
    presentation: AnalysisPresentation | None = None
    complex_plan: dict | None = None
    durable_turn_id: str | None = None
    execution_id: str | None = None
    commit_version: int | None = None
    interaction_state: Literal[
        "none", "awaiting_human_input", "awaiting_metric_confirmation", "awaiting_scope"
    ] = "none"


class HumanReviewRequest(BaseModel):
    """人工审核决定：批准 / 编辑后批准 / 驳回。

    当 critic 裁决为 human_review 且生产策略要求人工审批时，
    前端通过此 schema 提交审核决定。
    """
    action: str = Field(..., pattern="^(approve|edit|reject)$")  # approve/edit/reject
    actor: str = Field(default="unknown", min_length=1)          # 审核人标识
    reason: str = ""                                              # 审核备注/驳回原因
    answer: str | None = None                                    # edit 时提交的编辑后答案
