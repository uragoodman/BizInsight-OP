"""记忆层的最小契约。

这些对象不复制 AgentState、原始数据或完整 Scope；Scope 始终需要由
resolve_explorer_context 在每一轮重新签发。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


MemoryStatus = Literal["active", "pending_review", "superseded", "stale", "revoked", "rejected"]


class SessionMemory(BaseModel):
    memory_session_id: str = Field(min_length=1, max_length=128)
    session_summary: str = Field(max_length=1000)
    open_questions: list[str] = Field(default_factory=list, max_length=5)
    last_scope_fingerprint: str = Field(min_length=64, max_length=64)
    last_seller_id: str = Field(min_length=1, max_length=64)
    last_start_date: date
    last_end_date: date
    last_trace_id: str = Field(min_length=1, max_length=128)
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    # Phase 7: version == 2 时 snapshot_data 非空
    version: int = Field(default=1, ge=1, le=2)
    snapshot_data: dict | None = None

    @field_validator("open_questions")
    @classmethod
    def limit_open_question_length(cls, value: list[str]) -> list[str]:
        return [item.strip()[:240] for item in value if item.strip()]


class AnalysisMemory(BaseModel):
    memory_id: str = Field(min_length=1, max_length=128)
    seller_id: str = Field(min_length=1, max_length=64)
    start_date: date
    end_date: date
    scope_fingerprint: str = Field(min_length=64, max_length=64)
    metric_tags: list[str] = Field(default_factory=list, max_length=8)
    task_type: str = Field(min_length=1, max_length=64)
    insight_text: str = Field(min_length=1, max_length=500)
    limitations: str = Field(default="", max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    source_trace_id: str = Field(min_length=1, max_length=128)
    status: MemoryStatus = "active"
    created_at: datetime | None = None
    # P3.1 语义召回：嵌入版本随模型/算法递增，用于审计与全量重嵌入。
    embedding_model: str | None = None
    embedding_version: str | None = None
    # P3.2 生命周期：适用范围窗口、取代关系、数据/口径版本与审核轨迹。
    valid_from: date | None = None
    valid_to: date | None = None
    superseded_by: str | None = None
    source_data_version: str | None = None
    metric_definition_version: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    normalized_claim_hash: str | None = Field(default=None, min_length=64, max_length=64)
    review_reason: str = Field(default="", max_length=1000)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        unique = list(dict.fromkeys(value))
        if not all(item.startswith("ev_") and len(item) > 3 for item in unique):
            raise ValueError("evidence_ids must use existing ev_xxx EvidenceBundle IDs")
        return unique


class MemoryProposal(BaseModel):
    """P3.2 记忆取代候选：模型只能"提出"取代关系，服务端校验范围与证据后才写状态。

    - `supersedes_memory_id`：被取代的旧记忆（同一 tenant/user、Scope 高重叠、同指标）。
    - 候选必须经 Critic/人工审核后才把旧记忆标为 superseded。
    """
    supersedes_memory_id: str = Field(min_length=1, max_length=128)
    insight_text: str = Field(min_length=1, max_length=500)
    scope_fingerprint: str = Field(min_length=64, max_length=64)
    metric_tags: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    limitations: str = Field(default="", max_length=500)


class MemoryReviewRequest(BaseModel):
    action: Literal["supersede", "approve_independent", "reject"]
    actor: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1000)
    old_memory_id: str | None = Field(default=None, max_length=128)


class MemoryConsolidationRequest(BaseModel):
    """An operator may request a proposal, never an automatic memory mutation."""

    scope_fingerprint: str = Field(min_length=64, max_length=64)
    actor: str = Field(min_length=1, max_length=128)


class MemoryConsolidationProposal(BaseModel):
    insight_text: str = Field(min_length=1, max_length=500)
    limitations: str = Field(default="", max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)

    @field_validator("evidence_ids")
    @classmethod
    def validate_proposal_evidence_ids(cls, value: list[str]) -> list[str]:
        unique = list(dict.fromkeys(value))
        if not all(item.startswith("ev_") and len(item) > 3 for item in unique):
            raise ValueError("evidence_ids must use existing ev_xxx EvidenceBundle IDs")
        return unique


class MemoryStaleRequest(BaseModel):
    scope_fingerprint: str = Field(min_length=64, max_length=64)
    task_type: str = Field(min_length=1, max_length=64)
    source_data_version: str = Field(min_length=64, max_length=64)
    actor: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="source data version replaced", min_length=1, max_length=1000)
