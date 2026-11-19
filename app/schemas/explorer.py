"""数据浏览器的结构化筛选条件与稳定展示响应。

三层设计：
- SecurityScope: seller_id, allowed_*, tenant — 硬约束，不能突破
- AnalysisContext: default_start/end_date — 软默认，用户文本可覆盖
- QueryPolicy: max_date_span_days, max_trend_points, ... — 资源约束
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ExplorerMetric = str
ExplorerGranularity = Literal["day", "week", "month", "quarter", "year"]
CONTROLLED_ANALYSIS_DEFAULT_DIMENSIONS = (
    "product_category", "payment_type", "month_position",
    "day_of_week", "hour_of_day", "quarter_of_year", "half_of_year",
)


class ExplorerQuery(BaseModel):
    """前端只提交筛选条件；后端必须据此重新查库，不能相信浏览器传来的数据行。"""

    seller_id: str | None = Field(default=None, max_length=64)
    metric_name: ExplorerMetric = "gmv"
    start_date: date
    end_date: date
    granularity: ExplorerGranularity = "day"

    @model_validator(mode="after")
    def validate_range(self) -> "ExplorerQuery":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        if (self.end_date - self.start_date).days > 366 * 3:
            raise ValueError("Explorer date range must not exceed three calendar years")
        return self


class SellerScopeResolutionRequest(BaseModel):
    """User-selected seller range; the API reauthorizes it before issuing a token."""

    model_config = {"extra": "forbid", "frozen": True}

    seller_scope_type: Literal["single", "selected", "all"]
    seller_ids: tuple[str, ...] = ()
    start_date: date
    end_date: date
    metric_ids: tuple[str, ...] = ("gmv",)
    granularity: ExplorerGranularity = "day"
    include_unattributed: bool = False

    @model_validator(mode="after")
    def validate_scope_shape(self) -> "SellerScopeResolutionRequest":
        seller_ids = tuple(sorted({str(value).strip() for value in self.seller_ids if str(value).strip()}))
        object.__setattr__(self, "seller_ids", seller_ids)
        metric_ids = tuple(sorted({str(value).strip() for value in self.metric_ids if str(value).strip()}))
        object.__setattr__(self, "metric_ids", metric_ids)
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        if (self.end_date - self.start_date).days > 366 * 3:
            raise ValueError("Seller scope date range must not exceed 1098 days")
        if self.seller_scope_type == "all" and seller_ids:
            raise ValueError("ALL requires an explicit marker and no seller_ids")
        if self.seller_scope_type == "single" and len(seller_ids) != 1:
            raise ValueError("single scope requires exactly one seller_id")
        if self.seller_scope_type == "selected" and not seller_ids:
            raise ValueError("selected scope requires seller_ids")
        if self.seller_scope_type == "selected" and len(seller_ids) > 100:
            raise ValueError("selected scope cannot contain more than 100 sellers")
        if self.include_unattributed and self.seller_scope_type != "all":
            raise ValueError("include_unattributed is only valid for ALL scope")
        if not metric_ids or any(metric not in {"gmv", "order_count"} for metric in metric_ids):
            raise ValueError("Seller scope supports only gmv and order_count")
        return self


class SellerScopeProvenance(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    source: Literal["explorer_selector"]
    token: str
    expires_at: str


class SignedSellerScopePayload(BaseModel):
    """Server-issued execution payload returned by the scope resolver."""

    model_config = {"extra": "forbid", "frozen": True}

    seller_scope_intent: dict[str, Any]
    provenance: SellerScopeProvenance


class ControlledAnalysisScope(BaseModel):
    """由服务端根据数据浏览选择签发的 Agent 受控范围。

    三层职责分离：
    - SecurityScope（安全边界 — 不可突破）：
        seller_id, allowed_metrics, allowed_dimensions
        这些字段决定「用户能不能看这个数据」，图节点和 MCP 层见到不符直接拒绝。
    - AnalysisContext（默认上下文 — 用户文本可覆盖）：
        default_start_date, default_end_date
        当用户没有在聊天中明确指定时间时作为分析周期；文本中的日期优先。
    - QueryPolicy（资源约束 — 防滥用）：
        max_date_span_days, max_trend_points, max_rows, timeout_seconds
        阻止极端范围查询，不影响正常分析灵活性。
    """

    # === SecurityScope（安全边界 — 不可突破）===
    seller_id: str
    allowed_metrics: tuple[str, ...] = ()
    allowed_dimensions: tuple[
        Literal[
            "seller", "product_category", "payment_type", "month_position",
            "day_of_week", "hour_of_day", "quarter_of_year", "half_of_year",
        ],
        ...,
    ] = CONTROLLED_ANALYSIS_DEFAULT_DIMENSIONS

    # === AnalysisContext（默认上下文 — 用户文本可覆盖）===
    default_start_date: date
    default_end_date: date
    metric_name: ExplorerMetric
    granularity: ExplorerGranularity

    # === QueryPolicy（资源约束 — 防滥用）===
    max_date_span_days: int = 3650
    max_trend_points: int = 120
    max_rows: int = 10_000
    timeout_seconds: int = 30

    @model_validator(mode="after")
    def ensure_allowed_metrics(self) -> "ControlledAnalysisScope":
        """Lazily populate allowed_metrics from the registry if empty."""
        if not self.allowed_metrics:
            from app.metrics.registry import get_registry

            reg = get_registry()
            explorer_ids = [
                m.metric_id
                for m in reg.list()
                if (
                    "explorer" in m.surfaces
                    and reg.supports(m.metric_id, "value", scope="seller")
                )
            ]
            object.__setattr__(self, "allowed_metrics", tuple(explorer_ids))
        return self


class TimeRangePresetRequest(BaseModel):
    """前端只提交结构化预设，日期始终由服务端解析。"""
    kind: Literal["month", "previous_month", "recent_months", "year", "half", "quarter"]
    year: int | None = Field(default=None, ge=2010, le=2100)
    value: int | None = Field(default=None, ge=1, le=36)
