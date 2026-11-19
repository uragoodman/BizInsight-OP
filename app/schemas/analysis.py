from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PeriodicReportRequest(BaseModel):
    start_date: date
    end_date: date
    granularity: Literal["week", "month"]
    scope_type: Literal["platform", "seller"] = "platform"
    seller_id: str | None = Field(default=None, max_length=64)
    metric_names: list[str] = Field(default_factory=lambda: ["gmv", "order_count", "avg_order_value"], min_length=1)
    include_charts: bool = True

    @model_validator(mode="after")
    def valid_range(self) -> "PeriodicReportRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be greater than or equal to start_date")
        if self.scope_type == "seller" and not self.seller_id:
            raise ValueError("seller_id is required for seller-scoped reports")
        if self.scope_type == "platform" and self.seller_id is not None:
            raise ValueError("seller_id is only allowed for seller-scoped reports")
        return self


class ReportReviewRequest(BaseModel):
    version: int = Field(ge=1)
    action: Literal["approve", "edit", "reject"]
    actor: str = Field(min_length=1)
    reason: str = ""
    edited_content: str | None = None

    @model_validator(mode="after")
    def edit_requires_content(self) -> "ReportReviewRequest":
        if self.action == "edit" and not self.edited_content:
            raise ValueError("edited_content is required for edit")
        return self


class ReportPublishRequest(BaseModel):
    version: int = Field(ge=1)
    actor: str = Field(min_length=1)


class MetricTargetRequest(BaseModel):
    period_start: date
    period_end: date
    target_value: float
    version: str = Field(min_length=1)
    effective_at: date
    created_by: str = Field(min_length=1)
    scope: str = "global"
    seller_id: str | None = Field(default=None, max_length=64)


class InsightBriefRequest(BaseModel):
    """洞察生成请求：角度、受众、风格、条数、字数区间、建议开关、重点指标与下钻维度。"""
    angle: Literal["overview", "contribution", "anomaly", "target", "risk"] = "overview"
    audience: Literal["executive", "operations", "technical"] = "operations"
    style: Literal["formal", "concise", "conversational"] = "concise"
    max_insights: int = Field(default=5, ge=1, le=10)
    min_words: int = Field(default=40, ge=1)
    max_words: int = Field(default=200, ge=1)
    include_recommendations: bool = True
    focus_metrics: list[str] = Field(default_factory=list)
    dimension: str | None = None

    @model_validator(mode="after")
    def words_range(self) -> "InsightBriefRequest":
        if self.max_words < self.min_words:
            raise ValueError("max_words must be >= min_words")
        return self


class InsightReviewRequest(BaseModel):
    """洞察人工审核请求：approve / reject / edit（编辑时须带 edited_insights）。"""
    insight_version: int = Field(ge=1)
    action: Literal["approve", "reject", "edit"]
    actor: str = Field(min_length=1)
    reason: str = ""
    edited_insights: list[dict] | None = None

    @model_validator(mode="after")
    def edit_requires_content(self) -> "InsightReviewRequest":
        if self.action == "edit" and not self.edited_insights:
            raise ValueError("edited_insights is required for edit")
        return self
