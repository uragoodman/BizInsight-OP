"""分析结论展示契约 — 前后端之间结构化分析结果的事实标准。

后端在 generate_answer 节点构建此模型的实例，通过
``SseCompletedEvent.presentation`` 和 ``ChatResponse.presentation``
传递给前端。前端 ``AnalysisResultCard.vue`` 据此渲染结论卡。

不耦合于 events.py 或 chat.py，保持单一职责。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisScope(BaseModel):
    seller_id: str | None = None
    start_date: str = ""
    end_date: str = ""
    granularity: str = ""


class SemanticInterpretationCard(BaseModel):
    """前端展示的受控语义映射与口径说明。"""

    source_text: str = ""
    metric_id: str = ""
    display_name: str = ""
    dimension: str | None = None
    dimension_label: str = ""
    dimensions: list[str] = Field(default_factory=list)
    interpretation: str = ""
    confidence: float = 0.0
    reason: str = ""
    requires_confirmation: bool = False


class FactCard(BaseModel):
    """单指标值或趋势结论的事实卡。"""
    metric_name: str
    display_name: str = ""
    value: float | None = None
    unit: str = ""
    date_or_range: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ChangeCard(BaseModel):
    """比较查询的变化卡：当前期 vs 基线期。"""
    metric_name: str
    display_name: str = ""
    current_value: float | None = None
    baseline_value: float | None = None
    delta_value: float | None = None
    unit: str | None = Field(default=None, description="Unit of current_value and baseline_value.")
    change_unit: str | None = Field(default=None, description="Unit of the absolute delta_value.")
    delta_ratio: float | None = Field(default=None, description="Relative change in percent: delta_value / baseline_value * 100; null when unavailable or baseline is zero.")
    delta_ratio_unit: str | None = Field(default=None, description="Unit of delta_ratio; percent for the compare operation contract.")
    current_period: str = ""
    baseline_period: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ContributionEffect(BaseModel):
    component_type: str
    value: float
    unit: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ContributionItem(BaseModel):
    """贡献项的单个品类/维度行。"""
    dimension_value: str
    delta_value: float
    contribution_ratio: float | None = None
    current_value: float | None = None
    baseline_value: float | None = None
    components: list[ContributionEffect] = Field(default_factory=list)
    # Exact field-path to source-evidence mapping. Parent card references are
    # not sufficient to claim that each displayed numeric atom is grounded.
    field_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)


class ContributionCard(BaseModel):
    """维度贡献度分析卡。"""
    dimension: str
    metric_name: str = ""
    metric_definition: str = ""
    definition_version: str = ""
    unit: str = ""
    contribution_unit: str | None = Field(
        default=None,
        description="Unit of absolute contribution components; may differ from the metric level unit.",
    )
    aggregation: str = ""
    current_date: str = ""
    baseline_date: str = ""
    # Nullable for compatibility with older contribution cards that only
    # carried per-dimension values and a total delta.
    baseline_value: float | None = None
    current_value: float | None = None
    total_delta: float | None = None
    current_period: dict | None = None
    baseline_period: dict | None = None
    residual_value: float | None = None
    residual_type: Literal["coverage", "rounding", "model", "unknown"] = "unknown"
    residual_explainable: bool = False
    residual_tolerance: float | None = None
    residual_note: str = ""
    analysis_nature: Literal["descriptive_contribution"] = "descriptive_contribution"
    method: str = "additive_change_v1"
    items: list[ContributionItem] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    field_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)


class MetricDecompositionFactor(BaseModel):
    metric_id: str
    display_name: str = ""
    contribution: float
    unit: str = ""


class MetricDecompositionCard(BaseModel):
    """Validated numerical decomposition; it does not assert business causality."""

    metric_id: str
    display_name: str = ""
    unit: str = ""
    baseline_value: float
    current_value: float
    delta_value: float
    change_unit: str = ""
    baseline_period: dict = Field(default_factory=dict)
    current_period: dict = Field(default_factory=dict)
    method_id: str = ""
    factors: list[MetricDecompositionFactor] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    analysis_nature: Literal["registered_numeric_decomposition"] = "registered_numeric_decomposition"
    business_cause_confirmed: Literal[False] = False


class RecommendationCard(BaseModel):
    """证据约束的建议卡。"""
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class FunnelStage(BaseModel):
    """单个漏斗阶段。"""
    label: str
    count: int
    rate: float | None = None


class FunnelCard(BaseModel):
    """某日期的漏斗数据。"""
    date: str
    stages: list[FunnelStage] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class JointClusterMetric(BaseModel):
    """联合分析中的单个指标值。"""
    metric_name: str
    display_name: str
    value: float | None = None
    unit: str = ""
    comparison: ChangeCard | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class JointAnalysisCluster(BaseModel):
    """一个受控联合分析问题——履约×评价×销量。"""
    question: str = ""
    seller_id: str = ""
    period: str = ""
    grain: str = "order_id + seller_id"
    allocation_policy: str = "is_single_seller_order = true"
    metrics: list[JointClusterMetric] = Field(default_factory=list)
    assessment: str = ""
    sample_size: int = 0
    sample_note: str = ""
    association_limitation: str = "关联不等于因果。多卖家订单已排除。"
    evidence_ids: list[str] = Field(default_factory=list)


class ReviewSample(BaseModel):
    """单条脱敏评论文本样本。"""
    order_id_abbr: str
    score: int
    comment_excerpt: str
    has_title: bool = False
    purchase_date: str = ""


class ReviewSamplesCard(BaseModel):
    """评论文本抽样卡片。"""
    summary: str = ""
    comment_rate: float | None = None
    avg_length: float | None = None
    samples: list[ReviewSample] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class RiskDimension(BaseModel):
    """风险评分的单个维度分解。"""
    name: str
    value: float
    weight: float
    contribution: float


class RiskAssessment(BaseModel):
    """跨域风险评分卡。"""
    score: float = Field(ge=0, le=100)
    level: Literal["low", "medium", "high"] = "low"
    dimensions: list[RiskDimension] = Field(default_factory=list)
    sample_size: int = 0
    evidence_ids: list[str] = Field(default_factory=list)


# Phase 8: 白名单复杂计算模板结果
class CohortRow(BaseModel):
    cohort_label: str
    customer_count: int
    retention: list[dict] = Field(default_factory=list)


class CohortResult(BaseModel):
    cohorts: list[CohortRow] = Field(default_factory=list)
    granularity: str = "month"
    total_customers: int = 0


class RFMSegment(BaseModel):
    segment_name: str
    customer_count: int
    pct: float
    avg_value: float


class RFMResult(BaseModel):
    segments: list[RFMSegment] = Field(default_factory=list)
    total_customers: int = 0


class ParetoItem(BaseModel):
    rank: int
    name: str
    value: float
    cumulative_pct: float


class ParetoResult(BaseModel):
    dimension: str = ""
    items: list[ParetoItem] = Field(default_factory=list)
    p80_cutoff: dict | None = None


class CorrelationResult(BaseModel):
    sample_size: int = 0
    delayed: dict = Field(default_factory=dict)
    on_time: dict = Field(default_factory=dict)
    lift: float | None = None
    chi_square: dict | None = None


class AnalysisTemplateCard(BaseModel):
    """白名单分析模板的结果卡片。"""
    template_name: str
    display_name: str = ""
    params: dict = Field(default_factory=dict)
    result: CohortResult | RFMResult | ParetoResult | CorrelationResult | dict
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    """被引用的证据摘要，不携带完整 content。"""
    evidence_id: str
    source_type: str
    summary: str = ""


class HypothesisStepCard(BaseModel):
    """假设验证链的一步 — 展示层映射。"""
    step: int
    hypothesis: str
    purpose: str
    evidence_ids: list[str] = Field(default_factory=list)
    verdict: Literal["supported", "refuted", "inconclusive"] = "inconclusive"
    confidence: float | None = None
    rationale: str = ""
    next_action: str = ""


class AnalysisPresentation(BaseModel):
    """前端渲染分析结论卡的全部结构化数据。"""
    status: Literal["completed", "partial", "clarification_required", "unsupported", "failed"] = "completed"
    scope: AnalysisScope | None = None
    semantic_interpretation: SemanticInterpretationCard | None = None
    facts: list[FactCard] = Field(default_factory=list)
    changes: list[ChangeCard] = Field(default_factory=list)
    contributions: list[ContributionCard] = Field(default_factory=list)
    metric_decomposition: MetricDecompositionCard | None = None
    funnel: list[FunnelCard] = Field(default_factory=list)
    joint_clusters: list[JointAnalysisCluster] = Field(default_factory=list)
    review_samples: ReviewSamplesCard | None = None
    risk: RiskAssessment | None = None
    template_analyses: list[AnalysisTemplateCard] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationCard] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    hypothesis_chains_by_metric: dict[str, list[HypothesisStepCard]] = Field(default_factory=dict)
    trace_id: str = ""
    plan_preferences: dict = Field(default_factory=dict)
    execution_governance: dict = Field(default_factory=dict)
