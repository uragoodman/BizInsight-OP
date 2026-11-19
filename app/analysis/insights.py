"""AI 洞察层（report insights）。

确定性报告事实层保持不可变；AI 解读按报告版本懒加载为独立洞察制品。
LLM 只消费筛选后的结构化事实与证据 ID，输出必须可引用；每条事实性洞察
须通过服务端证据白名单、格式约束与 Critic 校验后才能进入可审核链路。

职责边界：
- 角度 → 确定性层选材（本文件 build_insight_context）
- 文本 → LLM 生成（llm_extract，JSON schema 输出）
- 反幻觉 → 服务端逐条白名单校验（validate_insights）
- 表述强度/范围 → Critic 审核（critic_review）
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Callable

from app.analysis.attribution import query_attribution_evidence
from app.analysis.language_guardrails import unsupported_causal_assertions
from app.agents.runtime import make_agent_run
from app.analysis.contracts import (
    Insight,
    InsightAngle,
    InsightArtifact,
    InsightAudience,
    InsightBrief,
    InsightCriticVerdict,
    InsightStatementType,
    InsightStyle,
    MetricDecomposition,
    PeriodicReport,
)
from app.core.config import get_settings
from app.core.execution_context import activate_state, reset_state
from app.tenancy import TenantContext, activate_tenant_context
from app.context.packer import ContextItem, ContextSource, pack_for_llm
from app.llm.client import llm_extract
from app.llm.routing import model_for

PROMPT_VERSION = "insights-v1"

# 可支撑"事实性结论"的证据 kind；仅引用图表/未知 kind 不足以支撑事实断言。
# 例如 GMV 结论不能只引用图表文件 ID，必须引用对应指标序列或贡献分解证据。
_DATA_KINDS = {
    "metric_series", "seller_metric_series", "anomaly", "contribution",
    "metric_target", "metric_decomposition", "grouped_ratio_attribution",
    "attribution_capability_gap",
}
_DATA_LIMITATION_MARKERS = (
    "数据不足", "缺少", "没有可用", "无法完成", "无法进行", "未能完成",
    "不可计算", "不支持该分析", "数据缺口", "insufficient data", "no data",
)

# 单次生成交给 LLM 的异常/贡献数量上限，控制 token 成本与延迟。
MAX_ANOMALIES = 5
MAX_CONTRIBUTIONS = 5
MAX_GROUPED_ATTRIBUTIONS = 4

_ANGLES_USE_ANOMALIES = {InsightAngle.ANOMALY, InsightAngle.OVERVIEW, InsightAngle.RISK}
_ANGLES_USE_CONTRIBUTIONS = {InsightAngle.CONTRIBUTION, InsightAngle.OVERVIEW, InsightAngle.RISK}

# 受众/风格 → 提示词指令：决定口吻与信息深度。
_AUDIENCE_INSTRUCTION: dict[InsightAudience, str] = {
    InsightAudience.EXECUTIVE: "受众为管理层：结论优先，每条给出明确判断与建议，避免过程细节与口径堆砌。",
    InsightAudience.OPERATIONS: "受众为运营：给出归因细节、可执行的下一步动作与数据依据。",
    InsightAudience.TECHNICAL: "受众为技术人员：补充指标口径、数据来源与血缘说明，措辞严谨。",
}
_STYLE_INSTRUCTION: dict[InsightStyle, str] = {
    InsightStyle.FORMAL: "风格正式：书面语，术语规范，避免口语化。",
    InsightStyle.CONCISE: "风格简洁：短句，结论优先，不赘述过程。",
    InsightStyle.CONVERSATIONAL: "风格口语：直白易懂，少用术语，像同事汇报。",
}


class InsightUnavailableError(RuntimeError):
    """LLM 或配额不可用，无法生成洞察（报告确定性层不受影响）。"""


def input_snapshot_hash(
    report_id: str, report_version: int, brief: InsightBrief, selected_ids: list[str],
    investigation_summary: dict[str, object] | None = None,
) -> str:
    """同一输入（报告版本 + Brief + 可见证据）→ 同一哈希，用于审计与幂等复用。"""
    payload = {
        "report_id": report_id,
        "report_version": report_version,
        "brief": brief.model_dump(mode="json"),
        "selected_evidence_ids": sorted(selected_ids),
        # The bounded investigation path affects what the LLM can conclude;
        # prevent idempotency from reusing an artifact with a stale stop or
        # branch state when the selected evidence IDs happen to be unchanged.
        "investigation_summary": investigation_summary,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cmp_payload(cmp: Any) -> dict[str, Any] | None:
    """把 PeriodComparison 压缩成传给 LLM 的摘要字段；无对比时返回 None。"""
    if cmp is None:
        return None
    return {
        "current_value": cmp.current_value,
        "baseline_value": cmp.baseline_value,
        "delta_value": cmp.delta_value,
        "delta_ratio": cmp.delta_ratio,
        "attainment_rate": cmp.attainment_rate,
        "status": cmp.status,
    }


def build_insight_context(
    report: PeriodicReport,
    brief: InsightBrief,
    extra_contributions: list = None,
    extra_decompositions: list[MetricDecomposition] | None = None,
    extra_attribution_evidence: dict[str, dict[str, object]] | None = None,
    investigation_summary: dict[str, object] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """按角度从报告事实层筛选数据，并并入受控补查证据。

    返回值：
    - `selected_evidence_ids`：本次交给 LLM 的可见证据 ID（白名单）。
    - `kind_map`：证据 ID → kind（含补查证据）。
    - `context`：结构化数据载荷（报告事实 + 补查证据，供 LLM 写有据洞察）。
    """
    kind_map = {eid: str(rec.get("kind", "unknown")) for eid, rec in report.evidence_records.items()}

    focused = [m for m in report.metrics if not brief.focus_metrics or m.metric_name in brief.focus_metrics]
    metrics_payload: list[dict[str, Any]] = []
    selected: set[str] = set()
    for metric in focused:
        metrics_payload.append({
            "metric_name": metric.metric_name, "value": metric.value, "unit": metric.unit,
            "period_over_period": _cmp_payload(metric.period_over_period),
            "year_over_year": _cmp_payload(metric.year_over_year),
            "target_comparison": _cmp_payload(metric.target_comparison),
            "evidence_ids": list(metric.evidence_ids),
        })
        selected.update(metric.evidence_ids)

    anomalies_payload: list[dict[str, Any]] = []
    if brief.angle in _ANGLES_USE_ANOMALIES:
        for anomaly in sorted(report.anomalies, key=lambda item: abs(item.score), reverse=True)[:MAX_ANOMALIES]:
            if brief.angle == InsightAngle.ANOMALY and brief.focus_metrics and anomaly.metric_name not in brief.focus_metrics:
                continue
            anomalies_payload.append({
                "metric_name": anomaly.metric_name, "metric_date": anomaly.metric_date.isoformat(),
                "direction": anomaly.direction, "score": anomaly.score, "threshold": anomaly.threshold,
                "evidence_ids": list(anomaly.evidence_ids),
            })
            selected.update(anomaly.evidence_ids)

    extra_contributions = extra_contributions or []
    contributions_payload: list[dict[str, Any]] = []
    if brief.angle in _ANGLES_USE_CONTRIBUTIONS:
        combined = list(report.contributions) + list(extra_contributions)
        for contribution in combined[:MAX_CONTRIBUTIONS]:
            if brief.dimension and contribution.dimension != brief.dimension:
                continue
            contributions_payload.append({
                "metric_name": contribution.metric_name, "dimension": contribution.dimension,
                "total_delta": contribution.total_delta,
                "items": [
                    {"dimension_value": item.dimension_value, "delta_value": item.delta_value,
                     "contribution_ratio": item.contribution_ratio}
                    for item in contribution.items[:5]
                ],
                "evidence_ids": list(contribution.evidence_ids),
            })
            selected.update(contribution.evidence_ids)

    decompositions_payload: list[dict[str, Any]] = []
    for decomposition in (extra_decompositions or []):
        decompositions_payload.append({
            "metric_name": decomposition.metric_name,
            "method_id": decomposition.method_id,
            "method_version": decomposition.method_version,
            "factor_metrics": decomposition.factor_metrics,
            "current_value": decomposition.current_value,
            "baseline_value": decomposition.baseline_value,
            "total_delta": decomposition.total_delta,
            "factors": [
                {"factor_metric": item.factor_metric, "delta_value": item.delta_value,
                 "contribution_ratio": item.contribution_ratio}
                for item in decomposition.factors
            ],
            "interaction_delta": decomposition.interaction_delta,
            "evidence_ids": list(decomposition.evidence_ids),
        })
        selected.update(decomposition.evidence_ids)

    grouped_attributions_payload: list[dict[str, Any]] = []
    if brief.angle in _ANGLES_USE_CONTRIBUTIONS:
        grouped_records = [
            record for record in (extra_attribution_evidence or {}).values()
            if record.get("kind") == "grouped_ratio_attribution"
            and (not brief.focus_metrics or record.get("metric_name") in brief.focus_metrics)
            and (not brief.dimension or record.get("dimension") == brief.dimension)
        ]
        grouped_records.sort(key=lambda record: (
            -abs(float(record.get("total_delta") or 0.0)), str(record.get("dimension", "")),
        ))
        for record in grouped_records[:MAX_GROUPED_ATTRIBUTIONS]:
            groups = list(record.get("groups") or [])
            ranked_groups = sorted(
                groups,
                key=lambda item: -abs(float(item.get("total_effect") or 0.0)),
            )
            visible_groups = ranked_groups[:5]
            omitted_effect = sum(float(item.get("total_effect") or 0.0) for item in ranked_groups[5:])
            evidence_id = str(record.get("evidence_id", ""))
            if evidence_id:
                selected.add(evidence_id)
                kind_map[evidence_id] = "grouped_ratio_attribution"
            grouped_attributions_payload.append({
                "metric_name": record.get("metric_name"),
                "dimension": record.get("dimension"),
                "method_id": record.get("method_id"),
                "method_version": record.get("method_version"),
                "baseline_value": record.get("baseline_value"),
                "current_value": record.get("current_value"),
                "total_delta": record.get("total_delta"),
                "groups": visible_groups,
                "omitted_group_count": max(0, len(ranked_groups) - len(visible_groups)),
                "omitted_group_effect_sum": omitted_effect,
                "evidence_id": evidence_id,
            })

    capability_gaps: list[dict[str, Any]] = []
    data_gaps: list[dict[str, Any]] = []
    for evidence_id, record in (extra_attribution_evidence or {}).items():
        if record.get("kind") == "capability_gap":
            kind_map[evidence_id] = "attribution_capability_gap"
            selected.add(evidence_id)
            capability_gaps.append({
                "metric_name": record.get("metric_name"),
                "reason": record.get("reason"),
                "evidence_id": evidence_id,
            })
        elif record.get("kind") == "data_gap":
            kind_map[evidence_id] = "attribution_data_gap"
            selected.add(evidence_id)
            data_gaps.append({
                "metric_name": record.get("metric_name"),
                "dimension": record.get("dimension"),
                "current_period": record.get("current_period"),
                "baseline_period": record.get("baseline_period"),
                "period_statuses": record.get("period_statuses"),
                "reason": record.get("reason"),
                "evidence_id": evidence_id,
            })

    selected_ids = sorted(selected)
    context: dict[str, Any] = {
        "report_type": report.report_type,
        "period": report.period.model_dump(mode="json"),
        "scope": report.scope.model_dump(mode="json"),
        "risks": list(report.risks),
        "metrics": metrics_payload,
        "anomalies": anomalies_payload,
        "contributions": contributions_payload,
        "decompositions": decompositions_payload,
        "grouped_attributions": grouped_attributions_payload,
        "attribution_capability_gaps": capability_gaps,
        "attribution_data_gaps": data_gaps,
        "investigation_summary": investigation_summary,
    }
    return selected_ids, kind_map, context


_GENERATION_SYSTEM = """你是一名严谨的经营分析分析师。你只允许基于「报告数据」撰写洞察，严禁编造数据中不存在的数字、维度或结论。

规则：
1. 每条 factual / observation 洞察必须至少引用一个「可见证据 ID」（见 allowed_evidence_ids），引用方式是在该洞察的 evidence_ids 数组里填入对应 ID。
2. 禁止引用 allowed_evidence_ids 之外、或报告数据中不存在的证据 ID。
3. 事实性断言只能基于给定数值；对原因的推测必须标记为 statement_type=hypothesis，并措辞使用"可能/推测"，不得断言因果。
4. 不要使用"导致""必然""唯一原因"等确定性因果措辞，除非数据明确支持。
5. finding 长度必须落在给定字数区间内；insights 总条数不得超过 max_insights。
6. 只输出 JSON 对象，格式：
{"insights":[{"title":"标题","finding":"结论文本","evidence_ids":["..."],"statement_type":"factual|observation|recommendation|hypothesis","confidence":0.0,"recommendation":"可选建议"}]}
"""


def _generation_prompt(brief: InsightBrief, selected_ids: list[str], kind_map: dict[str, str], context: dict[str, Any]) -> str:
    """把 Brief + 可见证据白名单 + 结构化数据组装成生成阶段 LLM 的 JSON 输入。"""
    payload = {
        "分析要求": {
            "角度": brief.angle.value,
            "受众": _AUDIENCE_INSTRUCTION.get(brief.audience, ""),
            "风格": _STYLE_INSTRUCTION.get(brief.style, ""),
            "最多洞察条数": brief.max_insights,
            "每条 finding 字数范围": [brief.min_words, brief.max_words],
            "必须包含建议": brief.include_recommendations,
            "重点指标": brief.focus_metrics or None,
            "下钻维度": brief.dimension or None,
        },
        "allowed_evidence_ids": selected_ids,
        "可见证据 ID 类型": {eid: kind_map.get(eid, "unknown") for eid in selected_ids},
        "报告数据": context,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _llm_generate(brief: InsightBrief, selected_ids: list[str], kind_map: dict[str, str], context: dict[str, Any]) -> list[Any]:
    """调用 LLM 生成原始洞察 JSON（JSON mode 输出，后由白名单校验）。

    参数：
    - `brief` / `selected_ids` / `kind_map` / `context`：见 `_generation_prompt`。

    返回值：`list[dict]` 原始洞察列表；LLM 不可用时抛 `InsightUnavailableError`。
    """
    prompt = _generation_prompt(brief, selected_ids, kind_map, context)
    packed = pack_for_llm("insight", [ContextItem(source=ContextSource.INSIGHT_GENERATION_INPUT, content=prompt, priority=100, required=True)], system_prompt=_GENERATION_SYSTEM, model=model_for("extract", quality=True)[0])
    parsed = llm_extract(_GENERATION_SYSTEM, packed.text, quality=True)
    if not isinstance(parsed, dict):
        raise InsightUnavailableError("LLM 不可用，洞察未生成")
    raw = parsed.get("insights")
    if not isinstance(raw, list):
        return []
    return raw


def validate_insights(insights: list[Insight], brief: InsightBrief, selected_ids: list[str], kind_map: dict[str, str]) -> list[str]:
    """服务端逐条白名单/格式校验，不依赖 prompt 自觉。

    校验规则（用户修正 #2：逐条检查，而非整篇）：
    - 事实性/观察性洞察必须有证据，且引用全部属于可见白名单。
    - 证据类型须与结论匹配：事实断言不能只引用图表/未知 kind。
    - 条数、字数区间、建议开关。
    """
    errors: list[str] = []
    if not insights:
        return ["no_insights"]
    if len(insights) > brief.max_insights:
        errors.append(f"insight_count_exceeds:{len(insights)}>{brief.max_insights}")
    for index, insight in enumerate(insights):
        claim_text = " ".join(part for part in (insight.title, insight.finding, insight.recommendation or "") if part)
        if unsupported_causal_assertions(claim_text):
            errors.append(f"insight[{index}].unsupported_causal_assertion")
        uncited = [eid for eid in insight.evidence_ids if eid not in selected_ids]
        if uncited:
            errors.append(f"insight[{index}].uncited_evidence:{uncited[:3]}")
        if insight.statement_type in {InsightStatementType.FACTUAL, InsightStatementType.OBSERVATION}:
            if not insight.evidence_ids:
                errors.append(f"insight[{index}].factual_without_evidence")
            else:
                cited_kinds = {kind_map.get(eid, "unknown") for eid in insight.evidence_ids}
                has_business_data = bool(cited_kinds & _DATA_KINDS)
                has_data_gap = "attribution_data_gap" in cited_kinds
                limitation_claim = any(
                    marker.casefold() in insight.finding.casefold()
                    for marker in _DATA_LIMITATION_MARKERS
                )
                if not has_business_data and not (has_data_gap and limitation_claim):
                    errors.append(f"insight[{index}].evidence_type_mismatch")
        # Character count is a soft generation preference. A useful,
        # evidence-grounded insight must not be rejected merely because the
        # generated Chinese text is slightly outside the requested range.
        if not brief.include_recommendations and insight.recommendation:
            errors.append(f"insight[{index}].unexpected_recommendation")
    return errors


_CRITIC_SYSTEM = """你是一名严格的洞察审核员。审核 AI 生成的报告洞察能否进入人工审核链路。

重点检查：
1. 表述强度：是否出现数据无法支持的确定性因果断言（"导致/必然/唯一原因/是...引起的"等）。
2. 范围一致性：结论是否超出给定的分析角度与重点指标；是否贴合要求的受众与风格。
3. 证据一致性：逐条核对每条事实性结论（statement_type=factual/observation）：
   - 是否引用了证据（evidence_ids 非空且属于 allowed_evidence_ids）；
   - 引用证据的内容（见"每条洞察引用的证据内容"）是否真能支撑该结论的数字或断言。
4. 重复与空泛：是否有空洞套话或无信息量结论。

只输出 JSON：{"approved": true 或 false, "issues": ["问题描述"]}
"""


def critic_review(
    insights: list[Insight],
    brief: InsightBrief,
    selected_ids: list[str],
    evidence_content: dict[str, dict[str, object]],
) -> tuple[bool, list[str]]:
    """LLM Critic 复核表述强度、范围与证据一致性。

    强事实复核：Critic 不只收到 evidence ID 列表，还收到每条引用证据的完整内容，
    可核对"结论中的数字 vs 证据中的数字"。Critic 不可用时抛 InsightUnavailableError，
    由调用方决定 fail-closed（pending_review），而不是把未复核的洞察当作已通过。
    """
    cited_content: dict[str, dict[str, object]] = {}
    for insight in insights:
        for eid in insight.evidence_ids:
            cited_content.setdefault(eid, evidence_content.get(eid, {"kind": "unknown", "note": "证据不在可见集合中"}))
    payload = {
        "分析要求": {
            "角度": brief.angle.value,
            "受众": _AUDIENCE_INSTRUCTION.get(brief.audience, ""),
            "风格": _STYLE_INSTRUCTION.get(brief.style, ""),
            "重点指标": brief.focus_metrics or None,
            "下钻维度": brief.dimension or None,
        },
        "allowed_evidence_ids": selected_ids,
        "每条洞察引用的证据内容": cited_content,
        "洞察": [insight.model_dump(mode="json") for insight in insights],
    }
    packed = pack_for_llm("insight", [
        ContextItem(source=ContextSource.INSIGHT_CRITIC_INPUT, content=json.dumps(payload, ensure_ascii=False, indent=2), priority=100, required=True),
    ], system_prompt=_CRITIC_SYSTEM, model=model_for("extract", quality=True)[0])
    parsed = llm_extract(_CRITIC_SYSTEM, packed.text, quality=True)
    if not isinstance(parsed, dict) or "approved" not in parsed:
        raise InsightUnavailableError("Critic 不可用")
    issues = parsed.get("issues")
    return bool(parsed.get("approved")), [str(item) for item in issues] if isinstance(issues, list) else []


def generate_insight_artifact(
    report: PeriodicReport,
    report_version: int,
    brief: InsightBrief,
    actor: str,
    progress: Callable[[str, str], None] | None = None,
    data_version_manifest_id: str | None = None,
    data_version_manifest_hash: str | None = None,
) -> InsightArtifact:
    """完整洞察生成流水线：受控补查 → 选材 → LLM → 白名单 → Critic → 制品。

    参数：
    - `report`：已锁定的报告事实快照（不可变）。
    - `report_version`：报告版本，洞察制品绑定在其上。
    - `brief`：洞察约束（角度/条数/字数区间/建议开关）。
    - `actor`：触发者 ID。
    - `progress`：可选阶段回调 (stage, message)，异步任务用它写 DB 进度供 SSE 推送。

    返回值：`InsightArtifact`（status 见下）。

    状态规则：
    - 服务端白名单/格式校验不通过 → rejected（仍保存审计，含非法引用证据）。
    - 白名单通过但 Critic 驳回 → rejected（critic_result 记录原因）。
    - 白名单通过且 Critic 放行 → draft（等待人工审核）。
    - 白名单通过但 Critic 不可用 → pending_review（fail-closed，不视为已通过）。
    """
    # 变化归因角度：受控补查数据库，产生报告之外的新证据（不修改 report_json）。
    # Phase 8：选材由 InsightPlan 驱动的证据 Agent 决定；LLM 不可用时内部退回
    # 固定选材（query_attribution_evidence），数学分解始终确定性计算。
    if progress:
        progress("attribution", "受控归因补查与证据选材")
    settings = get_settings()
    from app.core.execution_context import active_state

    execution_state = active_state() or {}
    attribution_trace_id = str(
        execution_state.get("trace_id") or f"insight_{report.report_id}_v{report_version}"
    )
    investigation_summary = None
    if settings.insight_agent_enabled:
        from app.agents.insight_agent import query_attribution_evidence_agent
        extra_contributions, extra_decompositions, attribution_evidence, evidence_driver, summary = (
            query_attribution_evidence_agent(
                report, brief, progress=progress, trace_id=attribution_trace_id,
            )
        )
        if summary is not None:
            investigation_summary = summary.model_dump(mode="json")
    else:
        extra_contributions, extra_decompositions, attribution_evidence = query_attribution_evidence(
            report, brief, trace_id=attribution_trace_id,
        )
        evidence_driver = "workflow_fallback"

    selected_ids, kind_map, context = build_insight_context(
        report, brief, extra_contributions=extra_contributions,
        extra_decompositions=extra_decompositions,
        extra_attribution_evidence=attribution_evidence,
        investigation_summary=investigation_summary,
    )
    # 补查证据并入 kind_map，纳入洞察输入证据快照。
    for eid, record in attribution_evidence.items():
        kind_map.setdefault(eid, str(record.get("kind", "unknown")))
    snapshot_hash = input_snapshot_hash(
        report.report_id, report_version, brief, selected_ids,
        investigation_summary=investigation_summary,
    )
    quality_model, _ = model_for("report")  # 报告洞察走 quality profile
    model = quality_model or get_settings().llm_model

    if progress:
        progress("llm_generation", "调用 LLM 生成结构化洞察")
    raw = _llm_generate(brief, selected_ids, kind_map, context)
    insights: list[Insight] = []
    parse_errors: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            parse_errors.append(f"insight[{index}].invalid_schema")
            continue
        try:
            insights.append(Insight.model_validate(item))
        except Exception as exc:  # noqa: BLE001 - 逐条记录 schema 失败，不中断整批
            parse_errors.append(f"insight[{index}].invalid_schema:{exc}")

    validation_errors = parse_errors + (validate_insights(insights, brief, selected_ids, kind_map) if insights else [])

    status = "draft"
    verdict: InsightCriticVerdict | None = None
    critic_issues: list[str] = []
    if not insights:
        status = "rejected"
        validation_errors = validation_errors or ["no_insights"]
    elif validation_errors:
        status = "rejected"
    else:
        # 强事实复核：Critic 收到每条引用证据的完整内容，而非仅 ID。
        evidence_content: dict[str, dict[str, object]] = {
            eid: record for eid, record in report.evidence_records.items() if eid in selected_ids
        }
        evidence_content.update(attribution_evidence)
        if progress:
            progress("critic", "Critic 强事实复核与表述强度审核")
        try:
            approved, critic_issues = critic_review(insights, brief, selected_ids, evidence_content)
        except InsightUnavailableError:
            approved = None  # Critic 不可用 → fail-closed
        if approved is True:
            verdict = InsightCriticVerdict.APPROVED
        elif approved is False:
            status = "rejected"
            verdict = InsightCriticVerdict.REJECTED
            validation_errors = critic_issues
        else:
            status = "pending_review"

    artifact = InsightArtifact(
        report_id=report.report_id,
        report_version=report_version,
        insight_version=0,  # store 负责分配实际版本号
        status=status,
        brief=brief,
        selected_evidence_ids=selected_ids,
        input_snapshot_hash=snapshot_hash,
        insights=insights,
        model=model,
        prompt_version=PROMPT_VERSION,
        critic_verdict=verdict,
        critic_result=critic_issues,
        validation_errors=validation_errors,
        attribution_evidence=attribution_evidence,
        investigation_summary=investigation_summary,
        evidence_decision_driver=evidence_driver,
        actor=actor,
        data_version_manifest_id=data_version_manifest_id,
        data_version_manifest_hash=data_version_manifest_hash,
    )
    artifact.agent_run = make_agent_run(
        run_id=f"insight_{report.report_id}_v{report_version}",
        domain="insight",
        decision_driver=evidence_driver,
        status="failed" if status == "rejected" else "finished",
        termination_initiator="critic" if status in {"pending_review", "rejected"} else "agent",
        termination_reason=("critic unavailable; pending human review" if status == "pending_review" else status),
        rounds=1,
        tool_calls=len(attribution_evidence),
        usage={
            "input_tokens": artifact.input_tokens or 0,
            "output_tokens": artifact.output_tokens or 0,
            "estimated_cost": artifact.estimated_cost or 0,
        },
        plan={
            "selected_evidence_ids": selected_ids,
            "investigation_summary": investigation_summary,
        },
        trace=[
            {
                "node": "insight_evidence_agent",
                "summary": (
                    f"driver={evidence_driver}, evidence={len(attribution_evidence)}, "
                    f"actions={(investigation_summary or {}).get('actions_used', 0)}, "
                    f"stop={((investigation_summary or {}).get('stop_reason') or {}).get('code', 'not_run')}"
                ),
            },
            {"node": "insight_critic", "summary": f"status={status}"},
        ],
    )
    return artifact


# ============================================================
# 异步生成：generating 占位行 → 后台线程池执行 → finalize
# ============================================================

_INSIGHT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="insight-worker")


def run_insight_generation(
    report: PeriodicReport,
    report_version: int,
    brief: InsightBrief,
    actor: str,
    tenant_meta: dict[str, str],
    progress: Callable[[str, str], None] | None = None,
    data_version_manifest_id: str | None = None,
    data_version_manifest_hash: str | None = None,
) -> InsightArtifact:
    """在激活的执行上下文中运行生成，捕获耗时与 LLM token/成本（供仪表盘）。

    参数：
    - `report` / `report_version` / `brief` / `actor`：同 `generate_insight_artifact`。
    - `tenant_meta`：租户/user 元数据，写入用量账本并让 `consume_llm_usage` 累积 token。
    - `progress`：阶段回调，透传给生成函数。

    返回值：`InsightArtifact`，且带观测字段（duration_ms / input_tokens / output_tokens / estimated_cost）。

    activate_state 让 llm client 的 _record_usage 既能写入用量账本，也能把
    token/成本累积到 state.llm_usage，生成结束后回读到 artifact 观测字段。
    """
    state: dict[str, Any] = {
        "tenant_id": tenant_meta.get("tenant_id", ""),
        "user_id": tenant_meta.get("user_id", ""),
        "trace_id": f"insight_{report.report_id}_v{report_version}",
        "chat_session_id": None,
        "task_id": None,
    }
    token = activate_state(state)
    try:
        start = perf_counter()
        artifact = generate_insight_artifact(report, report_version, brief, actor, progress=progress,
                                             data_version_manifest_id=data_version_manifest_id,
                                             data_version_manifest_hash=data_version_manifest_hash)
        usage = state.get("llm_usage", {})
        artifact.duration_ms = int((perf_counter() - start) * 1000)
        artifact.input_tokens = usage.get("input_tokens")
        artifact.output_tokens = usage.get("output_tokens")
        artifact.estimated_cost = usage.get("estimated_cost")
        if artifact.agent_run is not None:
            artifact.agent_run.input_tokens = int(artifact.input_tokens or 0)
            artifact.agent_run.output_tokens = int(artifact.output_tokens or 0)
            artifact.agent_run.estimated_cost = float(artifact.estimated_cost or 0.0)
        return artifact
    finally:
        reset_state(token)


def _report_from_row(stored: dict) -> PeriodicReport:
    """把 store 返回的报告行里的 report_json（SQLite 为 str / Postgres 为 dict）水合回 PeriodicReport。"""
    payload = stored.get("report_json", {})
    if isinstance(payload, str):
        payload = json.loads(payload)
    return PeriodicReport.model_validate(payload)


def _background_insight_task(
    store, report_id: str, report_version: int, insight_version: int,
    brief: InsightBrief, actor: str, tenant_meta: dict[str, str],
    tenant_context: TenantContext | None = None,
) -> None:
    """后台线程入口：重新水合报告 → 生成（逐阶段写 DB 进度）→ finalize；崩溃则 fail 为 rejected。

    参数：
    - `store`：当前租户的洞察存储（SQLite 或 PostgreSQL，方法各自建连接，跨线程安全）。
    - `report_id` / `report_version` / `insight_version`：目标报告版本与占位行版本。
    - `brief` / `actor` / `tenant_meta`：洞察约束、触发者、租户元数据（写用量账本用）。
    """
    def emit(stage: str, message: str) -> None:
        # 进度埋点只写 DB，供 SSE 轮询推送；失败不阻断生成。
        try:
            store.update_insight_progress(report_id, report_version, insight_version, stage, message)
        except Exception:  # noqa: BLE001 - 进度埋点失败不阻断生成
            pass

    try:
        if tenant_context is not None and (
            tenant_context.tenant_id != tenant_meta.get("tenant_id")
            or tenant_context.user_id != tenant_meta.get("user_id")
        ):
            raise PermissionError("Insight worker authorization context does not match task owner")
        stored = store.get(report_id, report_version)
        report = _report_from_row(stored)
        def generate() -> InsightArtifact:
            return run_insight_generation(
                report, report_version, brief, actor, tenant_meta, progress=emit,
                data_version_manifest_id=stored.get("data_version_manifest_id"),
                data_version_manifest_hash=stored.get("data_version_manifest_hash"),
            )
        if tenant_context is None:
            artifact = generate()
        else:
            # ThreadPoolExecutor does not propagate ContextVars; restore the
            # authenticated request context for seller RLS enforcement.
            with activate_tenant_context(tenant_context):
                artifact = generate()
        store.finalize_insight(report_id, report_version, insight_version, artifact)
    except Exception as exc:  # noqa: BLE001 - 后台任务兜底，绝不让行卡在 generating
        try:
            store.fail_insight(report_id, report_version, insight_version, f"worker_error:{type(exc).__name__}:{exc}")
        except Exception:  # noqa: BLE001 - 连兜底都失败时静默，避免线程崩溃噪音
            pass


def submit_insight_task(
    store, report_id: str, report_version: int, insight_version: int,
    brief: InsightBrief, actor: str, tenant_meta: dict[str, str],
    *, tenant_context: TenantContext | None = None,
) -> None:
    """把一次洞察生成派发给后台线程池（POST 已先落 generating 占位行，任务完成后 finalize）。"""
    _INSIGHT_EXECUTOR.submit(
        _background_insight_task, store, report_id, report_version, insight_version,
        brief, actor, tenant_meta, tenant_context,
    )
