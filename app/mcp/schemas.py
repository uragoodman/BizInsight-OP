"""MCP 层使用的结构化数据模型。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from app.metrics.authorization import AuthorizedMetric
from app.core.failure_context import ExecutionFailureContext


def _validate_metric_name(v: str) -> str:
    """Runtime validation via the metric registry — no import cycle."""
    from app.metrics.registry import get_registry

    try:
        return get_registry().resolve(v)
    except Exception as exc:
        raise ValueError(f"Unknown metric name: {v!r}") from exc
    return v


MetricName = Annotated[str, AfterValidator(_validate_metric_name)]

QueryType = Literal["metric_value", "metric_period", "metric_trend", "metric_compare", "metric_breakdown", "metric_contribution", "metric_statistics", "metric_entity_statistics", "metric_association"]

StatisticName = Literal["count", "mean", "average", "std", "min", "q1", "median", "q3", "max"]

TimeGranularity = Literal["day", "week", "month", "quarter", "year"]

DimensionName = Literal[
    "seller_id",
    "seller",
    "seller_state",
    "customer_state",
    "customer_city",
    "product_category",
    "payment_type",
    "month_position",
    "day_of_week",
    "hour_of_day",
    "quarter_of_year",
    "half_of_year",
]


class MetricToolArguments(BaseModel):
    """单日单指标查询工具共用的入参结构。"""

    metric_date: str = Field(..., description="指标日期，格式为 YYYY-MM-DD。")


class MetricTrendArguments(BaseModel):
    """趋势查询工具的入参结构。"""

    metric_name: MetricName = Field(..., description="要查询趋势的指标名。")
    start_date: str = Field(..., description="趋势查询开始日期，格式为 YYYY-MM-DD。")
    end_date: str = Field(..., description="趋势查询结束日期，格式为 YYYY-MM-DD。")
    granularity: TimeGranularity = Field(default="day", description="趋势聚合粒度。")


class MetricStatisticsArguments(MetricTrendArguments):
    """描述性统计查询工具的入参结构。"""

    statistics: list[StatisticName] = Field(
        default_factory=list,
        description="需要返回的统计量；为空时返回完整描述性统计摘要。",
    )


class MetricCompareArguments(BaseModel):
    """对比查询工具的入参结构。"""

    metric_name: MetricName = Field(..., description="要进行对比的指标名。")
    left_date: str = Field(..., description="对比左侧日期，格式为 YYYY-MM-DD。")
    right_date: str = Field(..., description="对比右侧日期，格式为 YYYY-MM-DD。")


class MetricBreakdownArguments(BaseModel):
    """维度拆解查询工具的入参结构。"""

    metric_name: MetricName = Field(..., description="要拆解的指标名。")
    metric_date: str = Field(..., description="拆解针对的指标日期，格式为 YYYY-MM-DD。")
    dimension: DimensionName = Field(..., description="拆解维度名。")
    limit: int = Field(default=10, ge=1, le=100, description="返回维度项数量上限。")


class MetricMultiDimensionBreakdownArguments(BaseModel):
    """多维指标交叉拆解工具的入参结构。"""

    metric_name: MetricName = Field(..., description="要交叉拆解的指标名。")
    start_date: date = Field(..., description="查询开始日期，包含该日。")
    end_date: date = Field(..., description="查询结束日期，包含该日。")
    dimensions: list[DimensionName] = Field(
        ..., min_length=2, max_length=4,
        description="两个到四个已注册的分组维度；必须唯一。",
    )
    limit: int = Field(default=100, ge=1, le=100, description="最多返回的交叉分组数。")

    @field_validator("dimensions")
    @classmethod
    def reject_duplicate_dimensions(cls, value: list[DimensionName]) -> list[DimensionName]:
        if len(value) != len(set(value)):
            raise ValueError("dimensions must be unique")
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> "MetricMultiDimensionBreakdownArguments":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be greater than or equal to start_date")
        return self


class MetricContributionArguments(BaseModel):
    metric_name: MetricName
    current_date: str
    baseline_date: str
    dimension: DimensionName
    limit: int = Field(default=10, ge=1, le=100)


class MetricPeriodContributionArguments(BaseModel):
    metric_name: MetricName
    current_start: str
    current_end: str
    baseline_start: str
    baseline_end: str
    granularity: Literal["week", "month"]
    dimension: DimensionName
    limit: int = Field(default=10, ge=1, le=100)


class FreightRateRatioFactsArguments(BaseModel):
    """Fixed, non-additive fact contract for platform freight-rate attribution."""

    metric_name: Literal["freight_rate"]
    current_start: str
    current_end: str
    baseline_start: str
    baseline_end: str
    granularity: Literal["week", "month"]
    dimension: Literal["month_position"]


class RegisteredRatioFactsArguments(BaseModel):
    """Complete grouped facts for a Registry-authorized platform ratio."""

    metric_name: MetricName
    current_start: str
    current_end: str
    baseline_start: str
    baseline_end: str
    granularity: Literal["week", "month"]
    dimension: Literal["month_position"]


class MetricBatchPeriodArguments(BaseModel):
    metric_names: list[MetricName] = Field(min_length=1, max_length=15)
    start_date: str
    end_date: str
    granularity: Literal["week", "month"]

    @field_validator("metric_names")
    @classmethod
    def reject_duplicate_metrics(cls, value: list[MetricName]) -> list[MetricName]:
        if len(value) != len(set(value)):
            raise ValueError("metric_names must be unique")
        return value


class MetricQueryPlan(BaseModel):
    """Data Query Agent 交给 MCP 层的结构化执行计划。"""

    query_type: QueryType = Field(default="metric_value", description="当前问题被判定出的查询类型。")
    metric_name: Optional[MetricName] = Field(default=None, description="从用户问题中解析出的标准化指标名。")
    metric_date: Optional[str] = Field(default=None, description="单点查询或拆解查询使用的指标日期。")
    current_date: Optional[str] = None
    baseline_date: Optional[str] = None
    current_start: Optional[str] = None
    current_end: Optional[str] = None
    baseline_start: Optional[str] = None
    baseline_end: Optional[str] = None
    filters: dict[str, str] = Field(default_factory=dict, max_length=3)
    start_date: Optional[str] = Field(default=None, description="趋势查询开始日期。")
    end_date: Optional[str] = Field(default=None, description="趋势查询结束日期。")
    left_date: Optional[str] = Field(default=None, description="对比查询左侧日期。")
    right_date: Optional[str] = Field(default=None, description="对比查询右侧日期。")
    granularity: Optional[TimeGranularity] = Field(default=None, description="趋势查询使用的时间粒度。")
    dimension: Optional[DimensionName] = Field(default=None, description="维度拆解查询使用的维度名。")
    dimensions: list[DimensionName] = Field(
        default_factory=list,
        description="多维交叉拆解使用的维度名；仅适用于两个或更多维度。",
    )
    limit: Optional[int] = Field(default=None, description="维度拆解查询返回数量上限。")
    statistics: list[StatisticName] = Field(default_factory=list, description="需要返回的描述性统计量。")
    tool_name: Optional[str] = Field(default=None, description="根据解析结果映射出的 MCP 工具名。")
    association_pair_id: str | None = Field(default=None, min_length=1, max_length=100)
    # Query execution must be created from this server-authorized object.  The
    # legacy metric_name remains for wire compatibility and display only.
    authorized_metric: AuthorizedMetric | None = Field(
        default=None,
        description="服务端授权后的指标；缺失时不得进入新的 MCP 执行路径。",
    )

    @model_validator(mode="after")
    def validate_multi_dimension_breakdown(self) -> "MetricQueryPlan":
        if self.dimensions:
            if not 2 <= len(self.dimensions) <= 4:
                raise ValueError("dimensions must contain between 2 and 4 entries")
            if len(self.dimensions) != len(set(self.dimensions)):
                raise ValueError("dimensions must be unique")
            if self.dimension is not None:
                raise ValueError("multi-dimension plans must not also set dimension")
            if self.query_type != "metric_breakdown":
                raise ValueError("multiple dimensions require metric_breakdown")
        return self


class MetricConfirmationCandidate(BaseModel):
    """A non-executable candidate displayed when semantic confirmation fails."""

    # This is display/audit metadata only.  The LLM-facing semantic protocol
    # selects candidate_no; the server resolves the real metric ID from the
    # immutable candidate set.  ``None`` is required for no_match.
    metric_id: MetricName | None = None
    display_name: str
    candidate_no: int | None = Field(default=None, ge=0)
    mapping_id: str | None = None
    relation: Literal["equivalent", "related_but_different", "no_match"] | None = None
    reason: str = ""
    evidence: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_relation_candidate(self) -> "MetricConfirmationCandidate":
        if self.relation == "no_match" and (self.candidate_no is not None or self.metric_id is not None):
            raise ValueError("no_match candidates must not carry candidate_no or metric_id")
        if self.relation in {"equivalent", "related_but_different"}:
            if self.candidate_no is None or self.metric_id is None:
                raise ValueError("matched semantic candidates require candidate_no and metric_id")
        return self


class MetricMention(BaseModel):
    """Server-auditable metric expression extracted from the user question."""

    mention_id: str
    raw_text: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    status: Literal[
        "exact", "candidate_pending", "confirmed", "unsupported",
        "skipped_by_user", "invalid", "ambiguous",
    ]
    provenance: Literal[
        "exact", "semantic_mapping", "explorer_selection", "user_skip", "none",
    ] = "none"
    metric_id: str | None = None
    mapping_ids: list[str] = Field(default_factory=list)


class SemanticResolution(BaseModel):
    """Non-executable semantic interpretation and its pending mappings."""

    resolution_id: str
    mentions: list[MetricMention] = Field(default_factory=list)
    catalog_digest: str = ""
    scope_fingerprint: str = ""
    requires_confirmation: bool = False
    expires_at: datetime | None = None


class MetricSemanticInterpretation(BaseModel):
    """审计用的自然语言指标解释，不包含任何可执行查询细节。"""

    source_text: str = ""
    metric_id: MetricName
    display_name: str = ""
    dimension: Optional[DimensionName] = None
    dimension_label: str = ""
    interpretation: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = ""
    requires_confirmation: bool = False


class RegisteredNumericDecomposition(BaseModel):
    """A server-resolved numeric decomposition request, not a causal claim."""

    status: Literal["ready", "unsupported", "authorization_blocked", "dates_required"]
    target_metric_id: str | None = None
    method_id: str | None = None
    factor_metric_ids: list[str] = Field(default_factory=list)
    input_metric_ids: list[str] = Field(default_factory=list)
    baseline_date: str | None = None
    current_date: str | None = None
    reason: str = ""
    causal_boundary: Literal["numeric_explanation_only"] = "numeric_explanation_only"


class MetricPlanningDecision(BaseModel):
    """Single LLM extraction result, validated against the Registry catalog."""

    status: Literal["confirmed", "clarification_required", "unsupported"]
    confirmed_metrics: list[MetricName] = Field(default_factory=list)
    ambiguous_candidates: list[MetricConfirmationCandidate] = Field(default_factory=list)
    query_type: QueryType = "metric_value"
    operation_intent: Literal[
        "ordinary_metric_query",
        "registered_numeric_decomposition",
        "dimensional_statistical_contribution",
        "business_causal_investigation",
    ] = "ordinary_metric_query"
    registered_decomposition: RegisteredNumericDecomposition | None = None
    dimension: Optional[DimensionName] = None
    dimensions: list[DimensionName] = Field(default_factory=list)
    granularity: Optional[TimeGranularity] = None
    statistics: list[StatisticName] = Field(default_factory=list, description="需要返回的描述性统计量。")
    reason: str = ""
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason_code: Literal[
        "exact_match",
        "semantic_auto_resolved",
        "semantic_confirmation_required",
        "unsupported_metric",
        "unsupported_dimension",
        "ambiguous_metric",
        "invalid_llm_output",
        "contract_violation",
        "scope_rejected",
        "provider_unavailable",
    ] | None = None
    # 用户明确提到、但当前范围不可用的指标名称；仅用于解释，不参与查询。
    unavailable_metrics: list[str] = Field(default_factory=list)
    skipped_metrics: list[str] = Field(default_factory=list)
    semantic_interpretation: MetricSemanticInterpretation | None = None
    semantic_resolution: SemanticResolution | None = None

    @field_validator("confirmed_metrics")
    @classmethod
    def unique_confirmed_metrics(cls, value: list[MetricName]) -> list[MetricName]:
        if len(value) != len(set(value)):
            raise ValueError("confirmed_metrics must be unique")
        return value

    @model_validator(mode="after")
    def validate_multi_dimension_decision(self) -> "MetricPlanningDecision":
        if self.dimensions:
            if not 2 <= len(self.dimensions) <= 4:
                raise ValueError("dimensions must contain between 2 and 4 entries")
            if len(self.dimensions) != len(set(self.dimensions)):
                raise ValueError("dimensions must be unique")
            if self.dimension is not None:
                raise ValueError("multi-dimension decisions must not also set dimension")
            if self.query_type != "metric_breakdown":
                raise ValueError("multiple dimensions require metric_breakdown")
        return self


class MetricQueryBatchPlan(BaseModel):
    """One confirmed intent with one plan per explicitly requested metric."""

    decision: MetricPlanningDecision
    plans: list[MetricQueryPlan] = Field(default_factory=list)

    @field_validator("plans")
    @classmethod
    def plans_match_confirmation(cls, value: list[MetricQueryPlan]) -> list[MetricQueryPlan]:
        names = [plan.metric_name for plan in value if plan.metric_name]
        if len(names) != len(set(names)):
            raise ValueError("batch plans must not duplicate metrics")
        return value


class MetricToolResult(BaseModel):
    """单日单指标查询工具返回的结构化结果。"""

    metric_name: MetricName
    metric_date: str
    period_start_date: str | None = None
    period_end_date: str | None = None
    value: Optional[float]
    unit: str = ""
    source_table: str
    status: Literal["ok", "not_found"]


class TrendPoint(BaseModel):
    """趋势查询返回的单个时间点。"""

    bucket: str = Field(..., description="聚合后的时间桶标识，例如某日或某周。")
    value: Optional[float] = Field(..., description="该时间点的指标值。")


class MetricTrendResult(BaseModel):
    """趋势查询工具返回的结构化结果。"""

    metric_name: MetricName
    start_date: str
    end_date: str
    granularity: TimeGranularity
    unit: str = ""
    points: list[TrendPoint]
    source_table: str
    status: Literal["ok", "not_found"]


class MetricCompareResult(BaseModel):
    """对比查询工具返回的结构化结果。"""

    metric_name: MetricName
    # Canonical task roles are carried alongside legacy left/right fields.
    # Internally left=current and right=baseline only at the adapter boundary.
    baseline_date: Optional[str] = None
    current_date: Optional[str] = None
    left_date: str
    left_end_date: Optional[str] = None
    right_date: str
    right_end_date: Optional[str] = None
    left_value: Optional[float]
    right_value: Optional[float]
    current_value: Optional[float] = None
    baseline_value: Optional[float] = None
    delta_value: Optional[float]
    delta_ratio: Optional[float]
    source_table: str
    source_lineage: Optional[str] = None
    status: Literal["ok", "not_found"]


class BreakdownItem(BaseModel):
    """维度拆解结果中的单个维度项。"""

    dimension_value: str = Field(..., description="维度值，例如某个州、城市或 seller。")
    value: Optional[float] = Field(..., description="该维度值对应的指标值。")
    share: Optional[float] = Field(default=None, description="该维度值对整体的占比。")
    seller_id: str | None = None
    order_count: int | None = None


class MetricBreakdownResult(BaseModel):
    """维度拆解查询工具返回的结构化结果。"""

    metric_name: MetricName
    metric_date: str
    dimension: DimensionName
    items: list[BreakdownItem]
    source_table: str
    status: Literal["ok", "not_found"]
    start_date: str | None = None
    end_date: str | None = None
    total: dict[str, object] | None = None
    unattributed: dict[str, object] | None = None
    non_additive: bool = False
    semantics: str | None = None
    seller_scope_type: Literal["single", "selected", "all"] | None = None
    seller_ids: list[str] | None = None
    all_sellers: bool = False
    query_scope_fingerprint: str | None = None
    group_order_count_non_additive: bool = False
    group_order_count_semantics: str | None = None
    message: str | None = None
    total_includes_unattributed: bool = False


class DataQueryAgentResult(BaseModel):
    """MCP 层与 workflow 下游节点之间共享的稳定结果结构。"""

    agent: Literal["DataQueryAgent"] = "DataQueryAgent"
    status: Literal["ok", "not_found", "unsupported", "insufficient_data", "execution_failed"]
    parsed: MetricQueryPlan
    metric_result: Optional[MetricToolResult] = None
    trend_result: Optional[MetricTrendResult] = None
    compare_result: Optional[MetricCompareResult] = None
    breakdown_result: Optional[MetricBreakdownResult] = None
    multi_dimension_breakdown_result: Optional[dict] = None
    contribution_result: Optional[dict] = None
    statistics_result: Optional[dict] = None
    association_result: Optional[dict] = None
    reason: Optional[str] = None
    failure_context: Optional[ExecutionFailureContext] = None


class McpToolInvocation(BaseModel):
    """用于 trace 记录的单次 MCP 工具调用信息。"""

    server: str
    tool_name: Optional[str] = None
    arguments: dict[str, object] = Field(default_factory=dict)
    result: dict[str, object]
