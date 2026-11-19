"""Validate an anomaly question's directional premise against trusted data."""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any

from langgraph.types import interrupt

from app.agents.data_query_agent import build_metric_query_plan
from app.core.time_expression import data_time_bounds, extract_iso_dates, resolve_expression
from app.graph.state import AgentState
from app.mcp.schemas import MetricQueryPlan
from app.schemas.analysis_presentation import AnalysisPresentation, ChangeCard, EvidenceRef

_DOWN_WORDS = ("\u4e0b\u964d", "\u4e0b\u6ed1", "\u4e0b\u8dcc", "\u964d\u4f4e", "\u51cf\u5c11", "\u53d8\u5dee", "\u8dcc", "\u4f4e\u4e8e", "decrease", "decreased", "down", "fall", "fell", "dropped", "lower")
_UP_WORDS = ("\u4e0a\u5347", "\u4e0a\u6da8", "\u589e\u52a0", "\u589e\u957f", "\u53d8\u597d", "\u6da8", "increase", "increased", "up", "rise", "rose", "grew", "higher")
def _metric_display_name(metric_name: str) -> str:
    """Use the Metric Registry for premise-check user-facing text."""

    try:
        from app.metrics.registry import get_registry

        registry = get_registry()
        return registry.get(registry.resolve(metric_name)).display_name
    except Exception:
        return str(metric_name)


def _claimed_direction(question: str) -> str | None:
    lowered = question.casefold()
    if any(word in lowered for word in _DOWN_WORDS):
        return "down"
    if any(word in lowered for word in _UP_WORDS):
        return "up"
    return None


def _metric_for_directional_claim(question: str, direction: str) -> str | None:
    """Bind a directional premise to the nearest exact metric mention.

    Root-cause questions can mention a distractor metric in the same sentence
    (for example, GMV fell while delivery delay rose). A semantic query planner
    may prefer the latter metric, so the premise check uses exact registry
    mention positions around the claimed direction word.
    """
    from app.metrics.authorization import server_exact_mentions
    from app.metrics.registry import get_registry

    registry = get_registry()
    catalog = registry.planning_catalog(scope="platform", surface="chat")
    mentions = server_exact_mentions(question, catalog=catalog)
    if not mentions:
        return None
    direction_words = _DOWN_WORDS if direction == "down" else _UP_WORDS
    lowered = question.casefold()
    direction_positions = [
        (lowered.find(word.casefold()), len(word))
        for word in direction_words
        if lowered.find(word.casefold()) >= 0
    ]
    if not direction_positions:
        return None
    ranked: list[tuple[int, int, str]] = []
    for direction_start, direction_length in direction_positions:
        for mention in mentions:
            start, end = int(mention["start"]), int(mention["end"])
            if end <= direction_start:
                ranked.append((direction_start - end, 0, str(mention["metric_id"])))
            elif start >= direction_start + direction_length:
                ranked.append((start - (direction_start + direction_length), 1, str(mention["metric_id"])))
    if not ranked:
        return None
    return min(ranked)[2]


def has_directional_premise(question: str) -> bool:
    return _claimed_direction(question) is not None


def _find_trend(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        direct = payload.get("trend_result")
        if isinstance(direct, dict):
            return direct
        found = _find_trend(payload.get("result"))
        if found:
            return found
        for item in payload.get("batch_results", []) or []:
            found = _find_trend(item)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_trend(item)
            if found:
                return found
    return None


def _find_period_comparison(payload: Any) -> tuple[dict[str, Any] | None, dict[str, Any]] | None:
    """Find a registered period comparison and its validated query plan."""
    if isinstance(payload, dict):
        direct = payload.get("compare_result")
        if isinstance(direct, dict):
            parsed = payload.get("parsed")
            return (parsed if isinstance(parsed, dict) else None, direct)
        nested = _find_period_comparison(payload.get("result"))
        if nested:
            parsed = payload.get("parsed")
            return (parsed if isinstance(parsed, dict) else nested[0], nested[1])
        for item in payload.get("batch_results", []) or []:
            nested = _find_period_comparison(item)
            if nested:
                parsed = payload.get("parsed")
                return (parsed if isinstance(parsed, dict) else nested[0], nested[1])
    elif isinstance(payload, list):
        for item in payload:
            nested = _find_period_comparison(item)
            if nested:
                return nested
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _period_comparison_values(
    parsed: dict[str, Any] | None,
    comparison: dict[str, Any],
    question: str,
) -> dict[str, Any] | None:
    """Validate current/baseline comparison values against the server plan."""
    if not isinstance(parsed, dict) or parsed.get("query_type") != "metric_compare":
        return None
    if comparison.get("status") != "ok":
        return None
    metric = str(parsed.get("metric_name") or "").strip().casefold()
    result_metric = str(comparison.get("metric_name") or "").strip().casefold()
    authorized = parsed.get("authorized_metric")
    authorized_metric = str(authorized.get("metric_id") or "").strip().casefold() if isinstance(authorized, dict) else ""
    direction = _claimed_direction(question)
    claimed_metric = _metric_for_directional_claim(question, direction) if direction else None
    if not metric or result_metric != metric or authorized_metric != metric:
        return None
    if claimed_metric and str(claimed_metric).casefold() != metric:
        return None

    expected_current = (parsed.get("current_start"), parsed.get("current_end"))
    expected_baseline = (parsed.get("baseline_start"), parsed.get("baseline_end"))
    if not all(isinstance(value, str) and value for value in (*expected_current, *expected_baseline)):
        return None
    current_period = comparison.get("current_period")
    baseline_period = comparison.get("baseline_period")
    current_period = current_period if isinstance(current_period, dict) else {}
    baseline_period = baseline_period if isinstance(baseline_period, dict) else {}
    current_start = str(
        current_period.get("start_date")
        or comparison.get("current_date")
        or comparison.get("left_date")
        or ""
    )
    current_end = str(
        current_period.get("end_date")
        or comparison.get("left_end_date")
        or comparison.get("current_end_date")
        or current_start
    )
    baseline_start = str(
        baseline_period.get("start_date")
        or comparison.get("baseline_date")
        or comparison.get("right_date")
        or ""
    )
    baseline_end = str(
        baseline_period.get("end_date")
        or comparison.get("right_end_date")
        or comparison.get("baseline_end_date")
        or baseline_start
    )
    if (current_start, current_end) != expected_current or (baseline_start, baseline_end) != expected_baseline:
        return None
    try:
        if date.fromisoformat(current_start) > date.fromisoformat(current_end):
            return None
        if date.fromisoformat(baseline_start) > date.fromisoformat(baseline_end):
            return None
    except ValueError:
        return None

    current_value = _finite_number(comparison.get("current_value"))
    if current_value is None:
        current_value = _finite_number(comparison.get("left_value"))
    baseline_value = _finite_number(comparison.get("baseline_value"))
    if baseline_value is None:
        baseline_value = _finite_number(comparison.get("right_value"))
    if current_value is None or baseline_value is None:
        return None
    return {
        "metric_name": metric,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "baseline_period": (baseline_start if baseline_start == baseline_end
                            else f"{baseline_start} 至 {baseline_end}"),
        "current_period": (current_start if current_start == current_end
                           else f"{current_start} 至 {current_end}"),
        "unit": str(comparison.get("unit") or ""),
    }


def _trusted_period_comparison(state: AgentState, question: str) -> tuple[str, dict[str, Any]] | None:
    """Use a comparison only when its exact result is present in trusted evidence."""
    planned = _find_period_comparison(state.get("tool_result"))
    if planned is None:
        planned = _find_period_comparison(state.get("query_results"))
    if planned is None:
        return None
    parsed, planned_result = planned
    expected = _period_comparison_values(parsed, planned_result, question)
    if expected is None:
        return None
    planned_request_id = str((state.get("tool_result") or {}).get("request_id") or "")
    for bundle in state.get("evidence_bundles") or []:
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
            continue
        evidence_id = str(bundle.get("evidence_id") or "")
        content = bundle.get("content")
        if not evidence_id or not isinstance(content, dict):
            continue
        evidence_request_id = str(content.get("request_id") or "")
        if (
            planned_request_id and evidence_request_id
            and evidence_request_id != planned_request_id
            and not evidence_request_id.startswith(f"{planned_request_id}:")
        ):
            continue
        observed = _find_period_comparison(content)
        if observed is None:
            continue
        observed_values = _period_comparison_values(parsed, observed[1], question)
        if observed_values is None:
            continue
        if all(
            observed_values[key] == expected[key]
            for key in ("metric_name", "baseline_period", "current_period")
        ) and math.isclose(observed_values["baseline_value"], expected["baseline_value"], rel_tol=0.0, abs_tol=1e-9) and math.isclose(
            observed_values["current_value"], expected["current_value"], rel_tol=0.0, abs_tol=1e-9,
        ):
            return evidence_id, observed_values
    return None


def _premise_plan(question: str) -> MetricQueryPlan | None:
    explicit_dates = extract_iso_dates(question)
    try:
        if len(explicit_dates) >= 2:
            # Use the shared user-boundary role parser, then query the
            # chronological range required by a trend tool.  premise_check
            # maps the resulting buckets back to explicit baseline/current
            # roles when the user labeled them.
            from app.context.compiler import _comparison_date_roles

            role_dates = _comparison_date_roles(question) or tuple(explicit_dates[:2])
            ordered = sorted(date.fromisoformat(value) for value in role_dates)
            bounds = data_time_bounds()
            resolved = resolve_expression(
                f"{ordered[0].isoformat()} 至 {ordered[-1].isoformat()}",
                data_bounds=bounds,
            )
        elif explicit_dates:
            # A single date in a directional premise has the established
            # previous-day/current-day comparison window.  Calendar bounds
            # and clipping still come from the shared resolver.
            point = resolve_expression(question)
            if point is None:
                return None
            current = point.end_date
            previous = max(current - timedelta(days=1), point.data_min_date)
            resolved = resolve_expression(
                f"{previous.isoformat()} 至 {current.isoformat()}",
                data_bounds=(point.data_min_date, point.data_max_date),
            )
        else:
            # Relative windows are resolved only by the shared calendar
            # contract; no date is inferred from arbitrary numeric text.
            resolved = resolve_expression(question)
            if resolved is None and any(
                word in question.casefold()
                for word in ("最近", "recent", "latest", "昨天", "yesterday")
            ):
                first, latest = data_time_bounds()
                previous = max(latest - timedelta(days=1), first)
                resolved = resolve_expression(
                    f"{previous.isoformat()} 至 {latest.isoformat()}",
                    data_bounds=(first, latest),
                )
    except (TypeError, ValueError):
        return None
    if resolved is None:
        return None
    direction = _claimed_direction(question)
    metric = (
        _metric_for_directional_claim(question, direction)
        if direction else None
    ) or build_metric_query_plan(question).metric_name
    if not metric:
        return None
    return MetricQueryPlan(query_type="metric_trend", metric_name=metric,
                           start_date=resolved.start_date.isoformat(),
                           end_date=resolved.end_date.isoformat(),
                           granularity="day", tool_name="query_metric_trend")


def premise_query_hint(question: str) -> str | None:
    """Return a deterministic trend query for the first anomaly request."""
    if not _claimed_direction(question):
        return None
    plan = _premise_plan(question)
    if not plan:
        return None
    return f"查询 {plan.start_date} 到 {plan.end_date} 的 {plan.metric_name} 趋势"


def _ensure_trend(state: AgentState) -> dict[str, Any] | None:
    existing = _find_trend(state.get("tool_result")) or _find_trend(state.get("query_results"))
    # premise_check consumes the result of the server planner in data_query.
    # It must not issue an ad-hoc unsigned retry when that plan did not produce
    # a trend; only data_query can create the AuthorizedMetric for execution.
    return existing


def _trend_values(trend: dict[str, Any]) -> tuple[float, float, str, str] | None:
    points = [p for p in trend.get("points", []) if isinstance(p, dict) and isinstance(p.get("value"), (int, float))]
    if len(points) < 2:
        return None
    return (float(points[0]["value"]), float(points[-1]["value"]),
            str(points[0].get("bucket", "")), str(points[-1].get("bucket", "")))


def _trend_values_for_question(
    trend: dict[str, Any], question: str,
) -> tuple[float, float, str, str] | None:
    """Use user-boundary baseline/current roles when the question defines them."""

    from app.context.compiler import _comparison_date_roles

    roles = _comparison_date_roles(question)
    if not roles:
        return _trend_values(trend)
    points = {
        str(item.get("bucket")): item
        for item in trend.get("points", [])
        if isinstance(item, dict) and isinstance(item.get("value"), (int, float))
    }
    baseline_date, current_date = roles
    baseline = points.get(baseline_date)
    current = points.get(current_date)
    if baseline is None or current is None:
        return None
    return (
        float(baseline["value"]), float(current["value"]),
        baseline_date, current_date,
    )


def _trusted_premise_evidence(
    state: AgentState, premise: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Find current trusted trend evidence that exactly backs a premise check."""
    metric = str(premise.get("metric_name") or "").strip().casefold()
    baseline_date = str(premise.get("baseline_date") or "")
    current_date = str(premise.get("current_date") or "")
    try:
        baseline_value = float(premise.get("baseline_value"))
        current_value = float(premise.get("current_value"))
    except (TypeError, ValueError):
        return None
    if (
        not metric or not baseline_date or not current_date
        or not math.isfinite(baseline_value) or not math.isfinite(current_value)
    ):
        return None

    question = str(state.get("user_message") or "")
    for bundle in state.get("evidence_bundles") or []:
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
            continue
        evidence_id = str(bundle.get("evidence_id") or "")
        content = bundle.get("content")
        if not evidence_id or not isinstance(content, dict):
            continue
        trend = _find_trend(content)
        if not isinstance(trend, dict) or trend.get("status") != "ok":
            continue
        if str(trend.get("metric_name") or "").strip().casefold() != metric:
            continue
        values = _trend_values_for_question(trend, question)
        if values is None:
            continue
        observed_baseline, observed_current, observed_start, observed_end = values
        if (
            observed_start == baseline_date
            and observed_end == current_date
            and math.isclose(observed_baseline, baseline_value, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(observed_current, current_value, rel_tol=0.0, abs_tol=1e-9)
        ):
            return evidence_id, trend

    comparison = _trusted_period_comparison(state, question)
    if comparison is not None:
        evidence_id, values = comparison
        if (
            str(values.get("metric_name") or "").strip().casefold() == metric
            and str(values.get("baseline_period") or "") == baseline_date
            and str(values.get("current_period") or "") == current_date
            and math.isclose(float(values["baseline_value"]), baseline_value, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(float(values["current_value"]), current_value, rel_tol=0.0, abs_tol=1e-9)
        ):
            return evidence_id, values
    return None


def _complete_confirmed_premise_correction(state: AgentState) -> None:
    """Close a disproven premise with its verified trend, without inventing a cause.

    A confirmation acknowledges the corrected direction. The original cause
    question was bound to the opposite direction, so it must not launch a new
    multi-action root-cause search implicitly. A separate explicit question can
    request an investigation of the observed direction.
    """
    premise = state.get("premise_check") or {}
    verified = _trusted_premise_evidence(state, premise) if isinstance(premise, dict) else None
    evidence_ids = [verified[0]] if verified else []
    metric_name = str(premise.get("metric_name") or "") if isinstance(premise, dict) else ""
    display_name = str(premise.get("metric_display_name") or metric_name or "该指标") if isinstance(premise, dict) else "该指标"
    limitations = ["现有证据只确认了变化方向，尚不足以解释其业务原因；本次不作原因归因。"]

    if verified:
        evidence_id, trend = verified
        baseline_value = float(premise["baseline_value"])
        current_value = float(premise["current_value"])
        delta = current_value - baseline_value
        actual_direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
        unit = str(trend.get("unit") or "")
        baseline_date = str(premise["baseline_date"])
        current_date = str(premise["current_date"])
        ratio = delta / baseline_value * 100 if baseline_value else None
        ratio_text = f"（{ratio:+.2f}%）" if ratio is not None else ""
        unit_text = f" {unit}" if unit else ""
        delta_text = f"{delta:+,.2f}{unit_text}"
        answer = (
            f"方向已按已验证数据更正：{display_name}从 {baseline_date} 的 "
            f"{baseline_value:,.2f}{unit_text} 到 {current_date} 的 "
            f"{current_value:,.2f}{unit_text}，实际{actual_direction} {delta_text}{ratio_text}。"
            f"原先“{display_name}{'下降' if premise.get('claimed_direction') == 'down' else '上升'}”前提不成立。"
            "现有证据不足以解释该变化的业务原因，本次不作原因归因。"
        )
        state["analysis_result"] = {
            "summary": answer,
            "task_type": "anomaly_analysis",
            "abstain": True,
            "stop_reason": "insufficient_evidence",
            "evidence_ids": evidence_ids,
            "limitations": limitations,
        }
        state["analysis_presentation"] = AnalysisPresentation(
            status="partial",
            changes=[ChangeCard(
                metric_name=metric_name,
                display_name=display_name,
                current_value=current_value,
                baseline_value=baseline_value,
                delta_value=delta,
                unit=unit or None,
                change_unit=unit or None,
                delta_ratio=ratio,
                delta_ratio_unit="%" if ratio is not None else None,
                current_period=current_date,
                baseline_period=baseline_date,
                evidence_ids=evidence_ids,
            )],
            limitations=limitations,
            evidence=[EvidenceRef(
                evidence_id=evidence_id,
                source_type=str(next(
                    (row.get("source_type") for row in state.get("evidence_bundles") or []
                     if isinstance(row, dict) and row.get("evidence_id") == evidence_id),
                    "mcp",
                ) or "mcp"),
                summary=f"{display_name}已验证对比数据",
            )],
            trace_id=str(state.get("trace_id") or ""),
        ).model_dump(mode="json")
    else:
        # A checkpoint without its trusted supporting trend cannot be used to
        # repeat the numeric claim. End safely with an explicit evidence error.
        answer = "无法从当前可信证据重新核验变化方向，因此本次不继续作方向判断或原因归因。请重新发起查询。"
        state.setdefault("errors", []).append({
            "code": "premise_confirmation_evidence_missing",
            "message": "The trusted trend evidence backing the suspended premise is unavailable.",
            "public_message": answer,
            "agent_name": "premise_confirmation",
        })
        state["analysis_result"] = {
            "summary": answer,
            "task_type": "anomaly_analysis",
            "abstain": True,
            "stop_reason": "insufficient_evidence",
            "evidence_ids": [],
            "limitations": ["暂停前提对应的可信趋势证据不可用。"],
        }
        state["analysis_presentation"] = AnalysisPresentation(
            status="failed",
            limitations=["暂停前提对应的可信趋势证据不可用。"],
            trace_id=str(state.get("trace_id") or ""),
        ).model_dump(mode="json")

    state["answer"] = answer
    state["final_answer"] = answer
    state["awaiting_user_confirmation"] = False
    state["premise_check"]["resolution"] = "direction_corrected_without_cause_attribution"
    state["diagnostic_action"] = {
        "action_type": "ABSTAIN",
        "reason": "insufficient_evidence",
        "supporting_evidence_ids": evidence_ids,
    }
    state["finish_decision"] = state["diagnostic_action"]
    state["termination_initiator"] = "policy"
    state["premise_correction_finalized"] = True


def _evidence_refs_for_query(state: AgentState) -> list[EvidenceRef]:
    bundles = {
        str(item.get("evidence_id")): item
        for item in state.get("evidence_bundles", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    referenced: list[str] = []
    for result in [state.get("tool_result"), *(state.get("query_results") or [])]:
        if not isinstance(result, dict):
            continue
        evidence_id = str(result.get("evidence_id") or "")
        if evidence_id and evidence_id in bundles and evidence_id not in referenced:
            referenced.append(evidence_id)
    refs: list[EvidenceRef] = []
    for evidence_id in referenced:
        bundle = bundles[evidence_id]
        content = bundle.get("content") or {}
        summary = str(content.get("question") or content.get("summary") or content.get("metric_name") or "")
        refs.append(EvidenceRef(
            evidence_id=evidence_id,
            source_type=str(bundle.get("source_type") or ""),
            summary=summary[:80],
        ))
    return refs


def premise_check(state: AgentState) -> AgentState:
    """Validate direction before diagnostic/root-cause analysis."""
    claim = _claimed_direction(state.get("user_message", ""))
    check: dict[str, Any] = {"status": "not_applicable", "claimed_direction": claim}
    if state.get("task_type") != "anomaly_analysis" or not claim:
        state["premise_check"] = check
        state.setdefault("node_history", []).append({"node": "premise_check", "summary": "No directional premise to validate."})
        return state
    trend = _ensure_trend(state)
    values = _trend_values_for_question(trend, state.get("user_message", "")) if trend else None
    comparison: dict[str, Any] | None = None
    comparison_evidence_id: str | None = None
    if not values:
        verified_comparison = _trusted_period_comparison(state, str(state.get("user_message") or ""))
        if verified_comparison is not None:
            comparison_evidence_id, comparison = verified_comparison
    if not values:
        points = [
            item for item in (trend or {}).get("points", [])
            if isinstance(item, dict) and isinstance(item.get("value"), (int, float))
        ]
        if comparison is None:
            reason = "无法取得至少两个有效时间点或匹配的可信周期对比，不能验证变化方向。"
            check.update({
                "status": "unavailable", "reason": reason,
                "required_valid_points": 2, "observed_valid_points": len(points),
                "evidence_ids": [item.evidence_id for item in _evidence_refs_for_query(state)],
            })
            state["premise_check"] = check
            state["terminal_code"] = "DATA_INSUFFICIENT"
            state["termination_initiator"] = "policy"
            state["answer"] = "我暂时无法取得足够的真实趋势数据或匹配的可信周期对比，因此不会继续根据“上升/下降”的前提做原因归因。请补充时间范围或稍后重试。"
            state["final_answer"] = state["answer"]
            state["analysis_presentation"] = AnalysisPresentation(
                status="partial",
                limitations=[reason],
                evidence=_evidence_refs_for_query(state),
                trace_id=str(state.get("trace_id") or ""),
            ).model_dump(mode="json")
            state.setdefault("node_history", []).append({
                "node": "premise_check",
                "summary": f"Trend unavailable ({len(points)}/2 valid points) and no trusted period comparison; stopped with DATA_INSUFFICIENT.",
            })
            return state

        baseline = float(comparison["baseline_value"])
        current = float(comparison["current_value"])
        baseline_date = str(comparison["baseline_period"])
        current_date = str(comparison["current_period"])
        metric = comparison.get("metric_name") or "metric"
        verified_evidence_ids = [comparison_evidence_id] if comparison_evidence_id else []
    else:
        baseline, current, baseline_date, current_date = values
        metric = trend.get("metric_name") or "metric"
        verified_evidence_ids = []
    delta = current - baseline
    actual = "up" if delta > 0 else "down" if delta < 0 else "flat"
    metric_display = _metric_display_name(str(metric))
    check.update({"status": "mismatch" if actual != claim else "confirmed",
                  "metric_name": metric, "metric_display_name": metric_display, "actual_direction": actual,
                  "baseline_date": baseline_date, "current_date": current_date,
                  "baseline_value": baseline, "current_value": current,
                  "evidence_ids": verified_evidence_ids,
                  "delta": delta, "delta_ratio": (delta / baseline * 100) if baseline else None})
    state["premise_check"] = check
    if check["status"] == "mismatch":
        direction_text = {"up": "上升", "down": "下降", "flat": "持平"}[actual]
        claim_text = {"up": "上升", "down": "下降"}.get(claim, claim)
        ratio = check["delta_ratio"]
        ratio_text = f"（变化率 {ratio:.1f}%）" if isinstance(ratio, (int, float)) else ""
        state["answer"] = (f"我先核对了真实数据：{metric_display} 从 {baseline_date} 的 {baseline:g} "
                            f"变为 {current_date} 的 {current:g}，实际是{direction_text}{ratio_text}。\n\n"
                            f"因此，“{metric_display}{claim_text}”的前提与当前数据不一致，我先不继续做{claim_text}原因归因。"
                            f"需要我按实际{direction_text}方向分析原因吗？")
        state["final_answer"] = state["answer"]
        state["awaiting_user_confirmation"] = True
    state.setdefault("node_history", []).append({"node": "premise_check",
        "summary": f"Premise={check['status']}; claimed={claim}; actual={actual}; source={'period_comparison' if comparison else 'trend'}."})
    return state


def premise_confirmation(state: AgentState) -> AgentState:
    """Durably pause after a premise mismatch and resume on confirmation."""
    try:
        decision = interrupt({
            "type": "premise_mismatch",
            "question": state.get("user_message", ""),
            "premise_check": state.get("premise_check", {}),
            "answer": state.get("answer", ""),
        })
    except RuntimeError:
        return state
    action = str((decision or {}).get("action", "")).casefold()
    if action in {"confirm", "approve", "continue", "yes"}:
        state["premise_check"]["status"] = "confirmed_by_user"
        _complete_confirmed_premise_correction(state)
        state.setdefault("node_history", []).append({"node": "premise_confirmation", "summary": "User confirmed analysis using the observed direction."})
        return state
    state["awaiting_user_confirmation"] = False
    state["premise_check"]["status"] = "cancelled"
    state["answer"] = "已取消本次方向调整分析。"
    state["final_answer"] = state["answer"]
    state.setdefault("node_history", []).append({"node": "premise_confirmation", "summary": "User declined continuation after premise mismatch."})
    return state
