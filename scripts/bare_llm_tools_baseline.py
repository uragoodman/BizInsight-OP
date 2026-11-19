"""Experiment 1b — bare LLM tool-calling arm (fairness-matched A/B).

Baseline ``baseline_single_call.py`` hands the model pre-collected evidence;
this arm answers "is the governed loop's value in the *reasoning*, or in the
*guardrails*?" by giving a bare LLM the SAME information access as the
governed agent and NOTHING more:

- the raw user question,
- a tool catalog (trend / dimension-contribution / related-metric trend) backed
  by the same golden fixtures the governed loop reads through the policy layer,
- a plain function-calling loop: the model picks tools, sees raw results,
  decides when to stop.  No policy validation, no required evidence skeleton,
  no server-side findings rendering, no critic.

The model does its own extraction and arithmetic on raw tool JSON — exactly
what the governed loop deliberately keeps server-side.  Scored by the same
``score_scenario`` ground truth as every other arm, so pass rates, tokens and
ungrounded conclusions are directly comparable with
``experiment-1-cost.json``.

Outputs: ``data/evaluations/experiment-1b-bare-tools.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from evaluation.fixtures import SCENARIO_DATES, fixture_mcp  # noqa: E402
from evaluation.scorers import score_scenario  # noqa: E402

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
RELATED = ("order_count", "avg_order_value", "delivery_delay_rate")
METRICS = ("gmv",) + RELATED
DIMENSIONS = ("category", "region", "channel")
MAX_ROUNDS = 8

SYSTEM_PROMPT = (
    "你是经营数据分析助手。可用数据访问工具见下（工具目录 JSON）。\n"
    "规则：\n"
    "1) 通过 TOOL_CALL 调用工具取数，自行阅读返回 JSON 并做计算；不要臆造数值。\n"
    "2) 取到足够证据后，用 FINAL 给出结论。\n"
    "3) 若证据显示无显著异常主因或数据不足，用 FINAL 且 answer 拒答（abstain=true）。\n"
    "只能输出一个 JSON 对象：\n"
    '{"action":"TOOL_CALL","tool":"<工具名>","arguments":{...}}\n'
    '{"action":"FINAL","found_factor":"<metric id 或维度名或 null>","abstain":<bool>,'
    '"answer":"<结论；弃答时说明证据不足>"}\n'
    "可用工具目录：\n"
    '[{"tool":"query_trend","arguments":{"metric":"gmv|order_count|avg_order_value|delivery_delay_rate",'
    '"date":"<yyyy-mm-dd>"},"desc":"查询某指标截至某日的 7 天趋势序列"},\n'
    ' {"tool":"dimension_contribution","arguments":{"metric":"...","date":"<yyyy-mm-dd>",'
    '"dimension":"category|region|channel"},"desc":"查询某指标某日在某维度上的贡献拆解"}]'
)


def _scenario_date(query: str) -> str | None:
    for match in _DATE_RE.findall(query):
        if match in SCENARIO_DATES:
            return match
    return None


def _validate_tool_call(raw: Any, round_no: int) -> dict[str, Any] | None:
    """Lenient shape check + deterministic guardrails the bare arm is ALLOWED to keep:
    a loop bound (round limit) and JSON/shape validity only.  Scope, allowlist
    enforcement, evidence-binding and policy review stay OFF — that is the point
    of the arm."""
    if not isinstance(raw, dict) or raw.get("action") != "TOOL_CALL":
        return None
    tool, args = raw.get("tool"), raw.get("arguments") or {}
    if tool not in {"query_trend", "dimension_contribution"} or not isinstance(args, dict):
        return None
    if round_no >= MAX_ROUNDS:
        return None
    return {"tool": tool, "arguments": args}


def _execute_tool(call: dict[str, Any], scenario: str | None) -> dict[str, Any]:
    """Route to the same fixture the governed loop's MCP gateway reads."""
    metric = str(call["arguments"].get("metric") or "gmv")
    date_arg = str(call["arguments"].get("date") or scenario or "2018-08-20")
    if call["tool"] == "dimension_contribution":
        dim = str(call["arguments"].get("dimension") or "category")
        question = f"拆解 {date_arg} 与 2018-08-10 的 {metric} 按 {dim} 贡献度"
        return fixture_mcp("metric_query", question)
    question = f"查询 {metric} 趋势 截止 {date_arg}"
    return fixture_mcp("metric_query", question)


def _parse_action(raw: Any) -> Any:
    """Same degenerate-proxy repair as the single-call arm."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return raw
    if isinstance(raw, dict) and len(raw) == 1:
        (key, value), = raw.items()
        if isinstance(key, str) and key.strip().startswith("{") and value in ("", None):
            for candidate in (key, key.replace('\\"', '"')):
                try:
                    inner = json.loads(candidate)
                    if isinstance(inner, dict):
                        return inner
                except Exception:  # noqa: BLE001
                    continue
    return raw


def run_case(case: dict[str, Any], record_usage: list[dict[str, int]]) -> dict[str, Any]:
    from app.llm.client import llm_extract

    query = case["query"]
    scenario = _scenario_date(query)
    transcript: list[dict[str, Any]] = []
    invalid_calls = 0
    duplicate_calls = 0
    seen_calls: list[str] = []
    final: dict[str, Any] | None = None
    started = time.perf_counter()
    usage_start = len(record_usage)

    user_message = f"用户问题：{query}\n\n当前无任何已采集证据，请自行调用工具取数。"
    for round_no in range(1, MAX_ROUNDS + 1):
        raw = llm_extract(SYSTEM_PROMPT, user_message)
        raw = _parse_action(raw)
        call = _validate_tool_call(raw, round_no)
        if call is None:
            # Not a tool call → treat as FINAL attempt (valid or not).
            if isinstance(raw, dict) and raw.get("action") == "FINAL":
                final = raw
            elif raw is not None:
                invalid_calls += 1
            break
        signature = json.dumps(call, sort_keys=True, ensure_ascii=False)
        if signature in seen_calls:
            duplicate_calls += 1
        seen_calls.append(signature)
        result = _execute_tool(call, scenario)
        transcript.append({"round": round_no, "call": call})
        # Raw JSON goes straight to the model — no server-side extraction,
        # no materiality flagging: the bare arm does its own arithmetic.
        user_message = (
            f"{user_message}\n\nTOOL_RESULT(round {round_no}, {call['tool']} "
            f"{json.dumps(call['arguments'], ensure_ascii=False)}):\n"
            + json.dumps(result.get("result", result), ensure_ascii=False)[:1800]
        )
    else:
        final = None  # round budget exhausted without FINAL

    found_factor = final.get("found_factor") if isinstance(final, dict) else None
    abstain = bool(final.get("abstain")) if isinstance(final, dict) else False
    answer = str(final.get("answer") or "") if isinstance(final, dict) else ""

    # Score with the SAME ground truth.  evidence_examined is satisfied by the
    # tools the arm actually called (trend + ≥1 dimension) — mirrored from
    # collect_anomaly_metrics' bar so all arms are graded identically.
    has_trend = any(item["call"]["tool"] == "query_trend" for item in transcript)
    dims_examined = {
        item["call"]["arguments"].get("dimension")
        for item in transcript
        if item["call"]["tool"] == "dimension_contribution"
    } - {None}
    metrics_rows = [item for item in transcript]
    state_stub: dict[str, Any] = {
        "user_message": query,
        "root_cause_evidence": [
            {"purpose": "trend_baseline", "result": {}}
            for item in metrics_rows if item["call"]["tool"] == "query_trend"
        ] + [
            {"purpose": f"dimension_contribution:{item['call']['arguments'].get('dimension')}", "result": {}}
            for item in metrics_rows if item["call"]["tool"] == "dimension_contribution"
        ],
        "finish_decision": {"found_factor": found_factor} if not abstain and found_factor else {},
        "analysis_result": {"abstain": abstain, "summary": answer},
    }
    checks = score_scenario(state_stub, case["expect"])
    # Honest recheck of the bar over the arm's real fetches.  Applied to
    # abstained and answered runs alike — the governed arm's mandatory
    # skeleton is likewise counted on abstained runs, so this keeps the
    # evidence_examined semantics identical across arms.
    checks["evidence_examined"] = bool(has_trend and dims_examined)

    usage_for_case = record_usage[usage_start:]
    return {
        "id": case["id"],
        "category": case["category"],
        "pattern": case.get("pattern", ""),
        "bare_tools": {
            "checks": checks,
            "passed": all(checks.values()),
            "found_factor": found_factor,
            "abstain": abstain,
            "answer": answer,
            "rounds_used": len(transcript),
            "round_budget_exhausted": final is None and len(transcript) == MAX_ROUNDS,
            "invalid_calls": invalid_calls,
            "duplicate_calls": duplicate_calls,
            "tools_called": [item["call"] for item in transcript],
        },
        "measurement": {
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "provider_calls": len(usage_for_case),
            "input_tokens": sum(item["input_tokens"] for item in usage_for_case),
            "output_tokens": sum(item["output_tokens"] for item in usage_for_case),
            "total_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in usage_for_case),
        },
    }


def main() -> int:
    dataset = json.loads((ROOT / "evaluation" / "datasets" / "golden.json").read_text(encoding="utf-8"))
    case_arg = next((item for item in sys.argv if item.startswith("--case-ids=")), "")
    case_ids = {item for item in case_arg.split("=", 1)[1].split(",") if item} if case_arg else None
    cases = [
        case for case in dataset["cases"]
        if case.get("category") in {"root_cause_scenario", "root_cause_longtail"}
        and (case_ids is None or case.get("id") in case_ids)
    ]

    from app.llm import client as llm_client
    original_record_usage = llm_client._record_usage
    usage_samples: list[dict[str, int]] = []

    def capture_usage(response: Any, model: str, operation: str, profile: str = "standard", provider_id: str | None = None, fallback_from: str | None = None) -> None:
        usage = getattr(response, "usage", None)
        usage_samples.append({
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        })
        original_record_usage(response, model, operation, profile, provider_id, fallback_from)

    llm_client._record_usage = capture_usage
    details: list[dict[str, Any]] = []
    try:
        for case in cases:
            entry = run_case(case, usage_samples)
            details.append(entry)
            arm = entry["bare_tools"]
            print(
                f"{entry['id']}: passed={arm['passed']} abstain={arm['abstain']} "
                f"rounds={arm['rounds_used']} tokens={entry['measurement']['total_tokens']}",
                flush=True,
            )
    finally:
        llm_client._record_usage = original_record_usage

    total_tokens = sum(item["measurement"]["total_tokens"] for item in details)
    passed = sum(bool(item["bare_tools"]["passed"]) for item in details)
    invalid_total = sum(item["bare_tools"]["invalid_calls"] for item in details)
    dup_total = sum(item["bare_tools"]["duplicate_calls"] for item in details)
    ungrounded = sum(
        1 for item in details
        if not item["bare_tools"]["abstain"] and not item["bare_tools"]["passed"]
        and item["bare_tools"]["found_factor"]
    )
    by_pattern: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for item in details:
        bucket = by_pattern[item.get("pattern") or item["id"]]
        bucket["total"] += 1
        bucket["passed"] += bool(item["bare_tools"]["passed"])

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "arm": "bare_llm_tools",
        "experiment": "governed_agent_vs_bare_llm_tool_calling",
        "model": get_settings().llm_model,
        "dataset_version": dataset.get("dataset_version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "details": details,
        "summary": {
            "cases": len(details),
            "passed": passed,
            "pass_at_1": round(passed / len(details), 4) if details else 0.0,
            "total_tokens": total_tokens,
            "avg_tokens_per_case": round(total_tokens / len(details), 3) if details else 0,
            "invalid_tool_calls": invalid_total,
            "duplicate_tool_calls": dup_total,
            "ungrounded_conclusions": ungrounded,
            "by_pattern": dict(sorted(by_pattern.items())),
        },
        "fairness": {
            "same_model": True,
            "same_tools": "same golden fixture data via tool catalog; governed loop reads it through the policy layer",
            "bare_arm_keeps": ["round limit (8)", "JSON/shape validity"],
            "bare_arm_lacks": ["scope/allowlist enforcement", "required evidence skeleton",
                               "server-side extraction & materiality flagging", "critic review",
                               "evidence-id binding"],
        },
        "limitations": [
            "Both arms use deterministic golden evidence, not live PostgreSQL business data.",
            "Cost is zero because no valid token price is configured.",
            "Single run; no pass^k repetition yet.",
        ],
    }
    output = ROOT / "data" / "evaluations" / "experiment-1b-bare-tools.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(json.dumps(payload["fairness"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
