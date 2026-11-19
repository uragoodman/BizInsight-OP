"""Deterministic typed policy used by the Phase 5 offline Agent arm.

Two decision surfaces reach the LLM in the governed loop: the plan (call 1)
and the terminal judgment (after the server executed the required evidence
skeleton).  The terminal decision below is deliberately the *rule path*: read
the deterministic findings summary, pick the largest-|change| significant
factor, ABSTAIN when nothing is significant or the data is incomplete.

That makes the offline arm an honest baseline for the "rule path" — long-tail
cases where the rule fails (e.g. a correlated-but-not-causal metric moving
more than the true factor) are exactly what the real-LLM arm is measured
against, and the two are reported separately.
"""

from __future__ import annotations

import re
from typing import Any


_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_EVIDENCE_RE = re.compile(r'"evidence_ids"\s*:\s*\[(.*?)\]')
_METRIC_LINE = re.compile(r"- (\w+)(（目标指标）)?: \S+ → \S+（单日 ([+-][\d.]+)%）→ (显著|不显著)")
_DIMENSION_LINE = re.compile(r"- 维度贡献\((\w+)\): 最大单项 \w+ 占总变化 ([\d.]+)% → (显著|不显著)")


def _scenario(user_message: str) -> str:
    match = _DATE_RE.search(user_message)
    return match.group(1) if match else "2018-08-20"


def _evidence_ids(prompt: str) -> list[str]:
    ids: list[str] = []
    for match in _EVIDENCE_RE.findall(prompt):
        ids.extend(re.findall(r'"([^"]+)"', match))
    return [item for item in dict.fromkeys(ids) if item.startswith("ev_")]


def _select_candidate(
    candidates: list[tuple[float, str, bool]], *, target_metric: str | None,
) -> tuple[float, str, bool] | None:
    """Apply the direct-factor preference only to this offline GMV rule."""
    if not candidates:
        return None
    aligned = [item for item in candidates if item[2]]
    # This is an explicit GMV-only offline heuristic, not a registry lookup or
    # a general direct-factor policy for arbitrary target metrics.
    direct_gmv_factors = {"order_count", "avg_order_value"}
    aligned_direct = (
        [item for item in aligned if item[1] in direct_gmv_factors]
        if target_metric == "gmv" else []
    )
    return max(aligned_direct or aligned or candidates, key=lambda item: item[0])


def _rule_decision(prompt: str) -> dict[str, Any]:
    """Apply the threshold rule to the deterministic findings summary.

    Only the summary section (between the findings and raw-observations
    markers) is parsed — both the instruction text and the raw observation
    JSON can legitimately contain the same strings.
    """
    findings = prompt.split("[确定性证据摘要")[-1].split("[计划与原始观察]")[0]
    if "数据不完整" in findings:
        return {"action_type": "ABSTAIN", "reason": "insufficient_evidence",
                "supporting_evidence_ids": _evidence_ids(prompt)[:2]}
    target_row = next((
        (metric, float(pct))
        for metric, is_target, pct, _flag in _METRIC_LINE.findall(findings)
        if is_target
    ), None)
    target_metric, target_change = target_row if target_row else (None, None)
    candidates: list[tuple[float, str, bool]] = []
    for metric, is_target, pct, flag in _METRIC_LINE.findall(findings):
        # Keep related metrics as hypotheses, but when a signal moves against
        # the target, do not let its larger absolute change override a direct
        # factor that explains the observed direction.
        if flag == chr(0x663e) + chr(0x8457) and not is_target:
            change = float(pct)
            candidates.append((abs(change), metric, target_change is None or change * target_change > 0))
    for dimension, ratio, flag in _DIMENSION_LINE.findall(findings):
        if flag == chr(0x663e) + chr(0x8457):
            candidates.append((float(ratio), dimension, True))
    ids = _evidence_ids(prompt)[:2]
    if not candidates:
        return {"action_type": "ABSTAIN", "reason": "insufficient_evidence", "supporting_evidence_ids": ids}
    best = _select_candidate(candidates, target_metric=target_metric)
    assert best is not None
    return {
        "action_type": "FINISH", "found_factor": best[1],
        "conclusion": f"规则路径：{best[1]} 为变动最大的显著因子，判定为根因。",
        "evidence_ids": ids,
    }


def scripted_agent_action(system_prompt: str, user_message: str, *, _calls: dict[str, int] | None = None) -> dict[str, Any]:
    """Return valid plan/action JSON without making a provider request.

    Call 1 is the plan; every later call is the terminal judgment, decided by
    the threshold rule over the deterministic findings summary.
    """
    calls = _calls if _calls is not None else {}
    scenario = _scenario(user_message)
    calls[scenario] = calls.get(scenario, 0) + 1
    if calls[scenario] == 1:
        return {
            "plan": {
                "goal": "Explain the GMV anomaly with governed evidence.",
                "hypotheses": ["A dimension or related operating metric explains the GMV movement."],
                "steps": [
                    {"step_id": "dimension", "objective": "Check the priority dimension contribution.", "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"], "completion_condition": "A dimension result is observed."},
                    {"step_id": "cross_metric", "objective": "Check the related operating metric signals.", "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"], "completion_condition": "Related metric results are observed."},
                ],
                "stop_conditions": ["Evidence is sufficient.", "No supported cause remains."],
                "budget": {"max_rounds": 6, "max_tool_calls": 6, "max_tokens": 4000, "max_cost": 1.0},
            }
        }
    return {"action": _rule_decision(system_prompt)}
