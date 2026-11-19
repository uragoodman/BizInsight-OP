"""SSE 事件契约模型 — 前后端之间的事件 Schema 单一来源。

每类事件由一个 Pydantic 模型定义，``event`` 字面量字段充当判别器。
后端序列化前经过 ``TypeAdapter`` 校验；前端据此生成 TypeScript 判别联合。
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

from app.schemas.analysis_presentation import AnalysisPresentation


class SseBaseEvent(BaseModel):
    """所有 SSE 事件的基类，携带单调递增的序号。"""
    sequence_number: int = 0


class SseStartedEvent(SseBaseEvent):
    event: Literal["started"] = "started"
    trace_id: str


class SseNodeStartedEvent(SseBaseEvent):
    event: Literal["node_started"] = "node_started"
    node: str


class SseNodeError(BaseModel):
    """节点完成事件中携带的安全错误摘要（不含 diagnostic_message）。"""
    code: str
    public_message: str
    agent_name: Optional[str] = None


class SseNodeCompletedEvent(SseBaseEvent):
    event: Literal["node_completed"] = "node_completed"
    node: str
    task_type: Optional[str] = None
    error: Optional[SseNodeError] = None


class SsePlanEvent(SseBaseEvent):
    event: Literal["plan"] = "plan"
    task_type: Optional[str] = None
    agents: List[str] = Field(default_factory=list)


class SseInvestigationProgressEvent(SseBaseEvent):
    """Safe, compact progress for bounded attribution investigations."""
    event: Literal["investigation_progress"] = "investigation_progress"
    completed_actions: int = Field(default=0, ge=0, le=6)
    remaining_budget: int = Field(default=6, ge=0, le=6)
    status: Literal["in_progress", "sufficient", "stalled", "blocked", "abstained"]
    current_action: Optional[str] = None
    stop_reason: Optional[str] = None


class SseHeartbeatEvent(SseBaseEvent):
    event: Literal["heartbeat"] = "heartbeat"
    trace_id: str


class SseCompletedEvent(SseBaseEvent):
    event: Literal["completed"] = "completed"
    task_type: str = "unknown"
    answer: str = ""
    trace_id: str = ""
    error: bool = False
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
    terminal_status: Optional[
        Literal["completed", "abstained", "failed", "cancelled", "interrupted"]
    ] = None
    clarification: Dict[str, Any] | None = None
    presentation: AnalysisPresentation | None = None
    complex_plan: Dict[str, Any] | None = None
    durable_turn_id: str | None = None
    execution_id: str | None = None
    commit_version: int | None = None


class SseFailedEvent(SseBaseEvent):
    event: Literal["failed"] = "failed"
    detail: str


class SseSuspendedEvent(SseBaseEvent):
    event: Literal["suspended"] = "suspended"
    task_type: str = "unknown"
    answer: str = ""
    trace_id: str = ""
    interaction_state: Literal["awaiting_human_input", "awaiting_metric_confirmation"]
    interaction_reason_code: Literal["premise_direction_mismatch"] | None = None
    scope_fingerprint: str | None = None
    clarification: Dict[str, Any] | None = None
    # The chat API can suspend a completed graph result to wait for explicit
    # confirmation of a sealed complex plan. Preserve that actionable plan in
    # the terminal event so the client can render its confirmation controls.
    complex_plan: Dict[str, Any] | None = None


# 可判别联合 —— 用于 TypeAdapter 运行时校验
SseEvent = Annotated[
    Union[
        SseStartedEvent,
        SseNodeStartedEvent,
        SseNodeCompletedEvent,
        SsePlanEvent,
        SseInvestigationProgressEvent,
        SseHeartbeatEvent,
        SseCompletedEvent,
        SseSuspendedEvent,
        SseFailedEvent,
    ],
    Field(discriminator="event"),
]
SSE_EVENT_ADAPTER: TypeAdapter = TypeAdapter(SseEvent)
