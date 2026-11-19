"""Strict Phase 6 business-analysis contracts."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.runtime import AgentRun


class AnalysisContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    contract_version: Literal["1.0"] = "1.0"


class PeriodSpec(AnalysisContract):
    start_date: date
    end_date: date
    granularity: Literal["day", "week", "month"] = "day"
    timezone: str = "UTC"

    @model_validator(mode="after")
    def validate_range(self) -> "PeriodSpec":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be greater than or equal to start_date")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return self


class MetricPoint(AnalysisContract):
    bucket: date
    value: float | None


class MetricSeries(AnalysisContract):
    metric_name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    period: PeriodSpec
    points: list[MetricPoint]
    source_refs: list[str] = Field(min_length=1)
    definition_version: str = Field(min_length=1)
    freshness_at: datetime | None = None
    component_totals: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_points(self) -> "MetricSeries":
        buckets = [point.bucket for point in self.points]
        if buckets != sorted(buckets):
            raise ValueError("points must be ordered by bucket")
        if len(buckets) != len(set(buckets)):
            raise ValueError("points must not contain duplicate buckets")
        if any(bucket < self.period.start_date or bucket > self.period.end_date for bucket in buckets):
            raise ValueError("point bucket must be within the declared period")
        if self.freshness_at is not None and self.freshness_at.tzinfo is None:
            raise ValueError("freshness_at must be timezone-aware")
        return self


class AnomalyFinding(AnalysisContract):
    metric_name: str
    metric_date: date
    value: float
    baseline: float
    score: float
    threshold: float = Field(gt=0)
    direction: Literal["up", "down"]
    method: Literal["mad", "iqr", "change_threshold"]
    sufficient_data: bool
    duration_days: int = Field(default=1, ge=1)
    evidence_ids: list[str] = Field(min_length=1)


class ContributionItem(AnalysisContract):
    dimension_value: str
    current_value: float
    baseline_value: float
    delta_value: float
    contribution_ratio: float | None


class ContributionFinding(AnalysisContract):
    metric_name: str
    dimension: str
    current_date: date
    baseline_date: date
    current_period: PeriodSpec | None = None
    baseline_period: PeriodSpec | None = None
    total_delta: float
    items: list[ContributionItem]
    evidence_ids: list[str] = Field(min_length=1)
    residual_value: float = 0.0
    residual_type: Literal["coverage", "rounding", "model", "unknown"] = "unknown"
    residual_explainable: bool = False
    residual_tolerance: float = Field(default=0.0, ge=0.0)
    residual_note: str = ""

    @model_validator(mode="after")
    def validate_reconciliation(self) -> "ContributionFinding":
        explained = round(sum(item.delta_value for item in self.items), 2)
        if abs(explained - self.total_delta) > 0.02:
            raise ValueError("contribution items must reconcile to total_delta")
        return self


class PeriodComparison(AnalysisContract):
    comparison_type: Literal["period_over_period", "year_over_year", "target"]
    current_period: PeriodSpec
    baseline_period: PeriodSpec | None = None
    current_value: float | None
    baseline_value: float | None
    delta_value: float | None
    delta_ratio: float | None
    attainment_rate: float | None = None
    target_version: str | None = None
    target_effective_at: date | None = None
    status: Literal["ok", "not_found", "zero_baseline"]


class ChartArtifact(AnalysisContract):
    chart_type: Literal["trend", "contribution"]
    metric_name: str | None = None
    unit: str | None = None
    data_points: list[dict[str, object]] = Field(default_factory=list)
    path: str | None = None
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Literal["ok", "chart_unavailable"]
    reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact(self) -> "ChartArtifact":
        if self.status == "ok" and (
            not self.content_hash
            or not self.evidence_ids
            or (not self.path and not self.data_points)
        ):
            raise ValueError("ok chart requires content_hash, evidence_ids, and path or data_points")
        if self.status == "chart_unavailable" and not self.reason:
            raise ValueError("unavailable chart requires a reason")
        return self


class ReportMetric(AnalysisContract):
    metric_name: str
    value: float | None
    unit: str
    period_over_period: PeriodComparison | None = None
    year_over_year: PeriodComparison | None = None
    target_comparison: PeriodComparison | None = None
    evidence_ids: list[str] = Field(min_length=1)


class ReportScope(AnalysisContract):
    scope_type: Literal["platform", "seller"] = "platform"
    seller_id: str | None = None
    seller_label: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "ReportScope":
        if self.scope_type == "seller" and not self.seller_id:
            raise ValueError("seller scope requires seller_id")
        if self.scope_type == "platform" and (self.seller_id or self.seller_label):
            raise ValueError("platform scope cannot carry seller metadata")
        return self


class PeriodicReport(AnalysisContract):
    report_id: str
    report_type: Literal["weekly", "monthly"]
    period: PeriodSpec
    scope: ReportScope = Field(default_factory=ReportScope)
    title: str
    summary: str
    metrics: list[ReportMetric] = Field(min_length=1)
    anomalies: list[AnomalyFinding] = Field(default_factory=list)
    contributions: list[ContributionFinding] = Field(default_factory=list)
    charts: list[ChartArtifact] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_freshness_at: datetime | None = None
    lineage_refs: list[str] = Field(min_length=1)
    evidence_records: dict[str, dict[str, object]] = Field(min_length=1)
    # Immutable, report-time dimension facts used by later insight attribution.
    # Older report versions deserialize with an empty map and must not live-query.
    attribution_breakdowns: dict[str, dict[str, object]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_report_metadata(self) -> "PeriodicReport":
        if self.data_freshness_at is None or self.data_freshness_at.tzinfo is None:
            raise ValueError("report data_freshness_at must be timezone-aware")
        return self


class AlertEvent(AnalysisContract):
    event_id: str
    idempotency_key: str
    event_type: Literal[
        "MetricWindowClosed", "AnomalyDetected", "ReportDrafted", "ReportApproved", "ReportPublished", "DiagnosisProposed"
    ]
    severity: Literal["info", "warning", "critical"] = "info"
    payload: dict[str, object]
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReportApproval(AnalysisContract):
    report_id: str
    report_version: int = Field(ge=1)
    action: Literal["approve", "edit", "reject"]
    actor: str = Field(min_length=1)
    reason: str = ""
    original_hash: str
    reviewed_hash: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================
# Phase 5: 假设验证链
# ============================================================


class HypothesisVerdict(str, Enum):
    """受控维度贡献假设裁决，不能宣称因果根因确认。"""
    SUPPORTED = "supported"       # 数据支持该维度有较大解释力
    REFUTED = "refuted"           # 该维度无显著贡献
    INCONCLUSIVE = "inconclusive" # 解释力有限，需更多证据


class HypothesisStep(AnalysisContract):
    """受控假设验证链的一步 — 假设在查询前创建，证据在查询后评价。

    query_id 是连接 root_cause → data_query → _evaluate_step 的精确键。
    每次构建队列时生成，data_query 原样回写至 root_cause_evidence 和
    EvidenceBundle.content，_evaluate_step 按 query_id 精确关联。
    """
    step: int
    metric_id: str = ""
    hypothesis: str                # 查询前创建（中性表述，如"品类结构变化能解释XXX观察到的变化"）
    purpose: str                   # "dimension_contribution:product_category"
    query_id: str = Field(default_factory=lambda: f"rcq_{uuid4().hex[:8]}")
    evidence_ids: list[str] = Field(default_factory=list)   # 原始 MCP evidence（按 query_id 匹配）
    derived_evidence_ids: list[str] = Field(default_factory=list)  # calculate_change_contribution 派生 hash
    verdict: HypothesisVerdict = HypothesisVerdict.INCONCLUSIVE
    confidence: float | None = None  # 贡献解释度（ratio = abs(top.contribution_ratio)）
    rationale: str = ""              # 裁决理由，简练可读
    next_action: Literal["continue", "cross_check", "stop"] = "continue"


# ============================================================
# AI 洞察层（report insights）
#
# 确定性报告事实层保持不可变；AI 解读按报告版本懒加载为独立洞察制品。
# LLM 只消费筛选后的结构化事实与证据 ID，输出必须可引用；每条事实性洞察
# 须通过服务端证据白名单、格式约束与 Critic 校验后才能进入可审核链路。
# ============================================================


class InsightAngle(str, Enum):
    """AI 洞察的分析角度。角度决定确定性层筛选哪些证据交给 LLM。"""
    OVERVIEW = "overview"          # 整体指标概览与趋势
    CONTRIBUTION = "contribution"  # 维度下钻归因（变化贡献度）
    ANOMALY = "anomaly"            # 统计异常聚焦
    TARGET = "target"              # 目标达成与差距
    RISK = "risk"                  # 风险与建议


class InsightStatementType(str, Enum):
    """洞察语句类型：事实/观察必须引用证据，建议与推测可选。"""
    FACTUAL = "factual"
    OBSERVATION = "observation"
    RECOMMENDATION = "recommendation"
    HYPOTHESIS = "hypothesis"


class InsightStyle(str, Enum):
    """洞察文风：决定句式与术语密度，随 Brief 进入提示词。"""
    FORMAL = "formal"                  # 正式：书面语，术语规范
    CONCISE = "concise"                # 简洁：短句，结论优先
    CONVERSATIONAL = "conversational"  # 口语：直白，少行话


class InsightAudience(str, Enum):
    """洞察受众：决定口吻与信息深度，随 Brief 进入提示词并纳入 Critic 范围复核。"""
    EXECUTIVE = "executive"    # 管理层：结论优先 + 建议，避免过程堆砌
    OPERATIONS = "operations"  # 运营：归因细节 + 可执行动作
    TECHNICAL = "technical"    # 技术：口径/血缘/方法说明，措辞严谨


class InsightBrief(AnalysisContract):
    """一次洞察生成的约束规格：角度、受众、风格、条数、字数区间、建议开关。"""
    angle: InsightAngle = InsightAngle.OVERVIEW
    audience: InsightAudience = InsightAudience.OPERATIONS
    style: InsightStyle = InsightStyle.CONCISE
    max_insights: int = Field(default=5, ge=1, le=10)
    min_words: int = Field(default=40, ge=1)
    max_words: int = Field(default=200, ge=1)
    include_recommendations: bool = True
    focus_metrics: list[str] = Field(default_factory=list)
    dimension: str | None = None

    @model_validator(mode="after")
    def validate_words(self) -> "InsightBrief":
        if self.max_words < self.min_words:
            raise ValueError("max_words must be >= min_words")
        return self


class Insight(AnalysisContract):
    """一条可审计的 AI 洞察。事实性结论必须引用报告证据。"""
    title: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    statement_type: InsightStatementType = InsightStatementType.OBSERVATION
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    recommendation: str | None = None


class InsightCriticVerdict(str, Enum):
    """Critic 对洞察的裁决：approved 放行 / rejected 驳回；不可用时 verdict 为空。"""
    APPROVED = "approved"
    REJECTED = "rejected"


class FactorContribution(AnalysisContract):
    """乘法指标分解中的一个因子贡献（如订单量变化对 GMV 变化的贡献）。"""
    factor_metric: str
    delta_value: float
    contribution_ratio: float | None = None


class MetricDecomposition(AnalysisContract):
    """可对账的指标分解证据（乘积、比例等）。

    用于"变化归因"角度：把总指标变化拆成各因子变化的贡献 + 交互项。
    与 ContributionFinding 同属可引用的数据证据。
    """
    metric_name: str
    method_id: str = "product_shapley_v1"
    method_version: str = "1.0"
    factor_metrics: list[str] = Field(min_length=2)
    current_value: float
    baseline_value: float
    total_delta: float
    factors: list[FactorContribution] = Field(min_length=2)
    interaction_delta: float
    display_factors: list[FactorContribution] = Field(default_factory=list)
    display_method: Literal["raw", "shapley", "symmetric_ratio"] = "raw"
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reconciliation(self) -> "MetricDecomposition":
        explained = round(sum(item.delta_value for item in self.factors) + self.interaction_delta, 2)
        if abs(explained - self.total_delta) > 0.02:
            raise ValueError("decomposition factors must reconcile to total_delta")
        return self


class InsightArtifact(AnalysisContract):
    """版本绑定的洞察制品，附着在某个 report_version 的事实层之上。

    状态机：
    - draft：通过服务端白名单 + Critic 复核，等待人工审核。
    - pending_review：通过白名单，但 Critic 不可用，处于待复核（不视为已通过）。
    - rejected：白名单/格式校验未通过，或被 Critic / 人工驳回（仍保存审计）。
    - approved：人工审核通过，可在报告发布时被 pin。
    - published：该洞察被报告发布动作 pin 为发布物的一部分。
    """
    report_id: str
    report_version: int = Field(ge=1)
    insight_version: int = Field(default=0, ge=0)
    status: Literal["generating", "draft", "rejected", "approved", "published", "pending_review"]
    brief: InsightBrief
    selected_evidence_ids: list[str] = Field(default_factory=list)
    input_snapshot_hash: str = Field(min_length=1)
    insights: list[Insight] = Field(default_factory=list)
    model: str = Field(default="")
    prompt_version: str = Field(min_length=1)
    critic_verdict: InsightCriticVerdict | None = None
    critic_result: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    attribution_evidence: dict[str, dict[str, object]] = Field(default_factory=dict)
    investigation_summary: dict[str, object] | None = None
    # AGENT_PLAN Phase 8: which driver produced the attribution evidence.
    evidence_decision_driver: Literal["llm", "workflow_fallback"] = "workflow_fallback"
    agent_run: AgentRun | None = None
    actor: str = "system"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_reason: str = ""
    # 数据版本谱系：洞察继承其报告在生成时钉定的密封 manifest。
    # 未版本化路径生成的洞察保持 NULL，replay 诚实返回 unavailable。
    data_version_manifest_id: str | None = None
    data_version_manifest_hash: str | None = None
    # 观测字段：供质量仪表盘统计（异步生成后填充）。
    duration_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
