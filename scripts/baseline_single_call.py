"""Single-call baseline: can one LLM call beat the governed loop?

For every root-cause golden case this script builds the SAME deterministic
evidence the agent loop's required skeleton would collect (target trend,
dimension contribution, related-metric trends — all through the offline
fixtures), renders the SAME findings summary the loop uses, and asks the
model exactly ONE question: FINISH with a factor, or ABSTAIN.

Three arms are then scored on identical ground truth:

- rule:        the threshold rule over the findings (the offline arm's policy)
- single_call: one LLM call on the findings (this script)
- agent:       the governed loop's provider numbers, imported from the latest
               ``agent-vs-workflow-provider.json`` when present

This is the "why do you need an agent at all?" comparison, answered with
data instead of narrative.
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

from app.agents.diagnostic_contracts import AbstainAction, FinishAction  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.graph.nodes.diagnostic import (  # noqa: E402
    TERMINAL_JUDGMENT_RULES,
    _findings_block,
    _first_dimension,
    _target_metric,
)
from evaluation.agent_policy import _rule_decision  # noqa: E402
from evaluation.fixtures import SCENARIO_DATES, fixture_mcp  # noqa: E402
from evaluation.scorers import collect_anomaly_metrics, score_scenario  # noqa: E402

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
RELATED = ("order_count", "avg_order_value", "delivery_delay_rate")


def _scenario_date(query: str) -> str | None:
    for match in _DATE_RE.findall(query):
        if match in SCENARIO_DATES:
            return match
    return None


def _evidence_state(case: dict[str, Any]) -> dict[str, Any]:
    """Deterministically collect the skeleton's evidence — no LLM involved."""
    query = case["query"]
    target = _target_metric({"user_message": query})
    scenario_date = _scenario_date(query)
    baseline = (scenario_date or "2018-08-20")
    start = baseline
    dimension = _first_dimension(target)
    entries: list[dict[str, Any]] = []

    def record(purpose: str, metric: str, question: str) -> None:
        envelope = fixture_mcp("metric_query", question)
        entries.append({"purpose": purpose, "metric_id": metric, "result": envelope})

    record("trend_baseline", target, f"查询 {target} 趋势 {query}")
    record(
        f"dimension_contribution:{dimension}", target,
        f"拆解 {scenario_date} 与 {baseline} 的 {target} 按 {dimension} 贡献度 {query}",
    )
    for metric in RELATED:
        if metric == target:
            continue
        record(f"cross_metric:{metric}", metric, f"查询 {metric} 趋势 {query}")
    return {
        "user_message": query,
        "diagnostic_target_metric": target,
        "root_cause_evidence": entries,
        "tool_calls": [],
        "claims": [],
    }


def _apply_action(state: dict[str, Any], action: Any, *, error: str | None = None) -> None:
    """Mirror ``diagnostic_finalize``'s state shape so the scorer reads both
    arms identically."""
    if action is None:
        state["finish_decision"] = {}
        state["analysis_result"] = {"summary": "", "abstain": True, "judgment_error": error}
        return
    dump = action.model_dump(mode="json") if hasattr(action, "model_dump") else dict(action)
    if dump.get("action_type") == "FINISH":
        state["finish_decision"] = dump
        state["analysis_result"] = {
            "summary": dump.get("conclusion", ""),
            "abstain": False,
            "evidence_ids": dump.get("evidence_ids", []),
        }
    else:
        state["finish_decision"] = dump
        state["analysis_result"] = {
            "summary": "证据不足，正确拒答。" if dump.get("reason") == "insufficient_evidence" else "",
            "abstain": True,
            "stop_reason": dump.get("reason"),
        }


def _unwrap(raw: Any) -> Any:
    """Repair the proxy's degenerate mode: the intended JSON arriving as an
    escaped-string key of an otherwise empty object (``{"{\"a\":1}": ""}``)."""
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


def _validate_action(raw: Any) -> Any:
    """Lenient on purpose: the single call has no evidence bundles, so the
    loop's ``evidence_ids`` contract does not apply here — the scorer only
    reads ``found_factor`` / abstain.  Everything else stays strict."""
    raw = _unwrap(raw)
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("action"), dict):
        raw = raw["action"]
    try:
        if raw.get("action_type") == "FINISH":
            if not raw.get("found_factor") or not raw.get("conclusion"):
                return None
            return raw
        if raw.get("action_type") == "ABSTAIN":
            return AbstainAction.model_validate(raw)
    except Exception:  # noqa: BLE001 - an invalid judgment counts as a miss
        return None
    return None


def main() -> int:
    dataset = json.loads((ROOT / "evaluation" / "datasets" / "golden.json").read_text(encoding="utf-8"))
    case_arg = next((item for item in sys.argv if item.startswith("--case-ids=")), "")
    case_ids = {item for item in case_arg.split("=", 1)[1].split(",") if item} if case_arg else None
    cases = [
        case for case in dataset["cases"]
        if case.get("category") in {"root_cause_scenario", "root_cause_longtail"}
        and (case_ids is None or case.get("id") in case_ids)
    ]
    use_llm = "--rule-only" not in sys.argv
    usage_samples: list[dict[str, int]] = []
    if use_llm:
        from app.llm import client as llm_client
        original_record_usage = llm_client._record_usage

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
            state = _evidence_state(case)
            findings = _findings_block(state)
            usage_start = len(usage_samples)
            started = time.perf_counter()

            # Arm 1 — rule path (the offline scripted policy's decision function).
            rule_state = dict(state)
            _apply_action(rule_state, _rule_decision(findings))
            rule_entry = {
                "checks": score_scenario(rule_state, case["expect"]),
                "found_factor": (rule_state.get("finish_decision") or {}).get("found_factor"),
                "abstain": bool((rule_state.get("analysis_result") or {}).get("abstain")),
            }
            rule_entry["passed"] = all(rule_entry["checks"].values())

            entry: dict[str, Any] = {
                "id": case["id"], "category": case["category"], "pattern": case.get("pattern", ""),
                "rule": rule_entry,
            }

        # Arm 2 — one LLM call over the same findings.  The governed loop's
        # terminal judgment gets 2 attempts on validation failure, so this arm
        # gets the same budget; the first-attempt failure is recorded as a
        # robustness finding.
            if use_llm:
                from app.llm.client import llm_extract

                action = None
                first_raw: str | None = None
                for attempt in range(2):
                    raw = llm_extract(
                        TERMINAL_JUDGMENT_RULES
                        + "输出前先逐行阅读上面的摘要，再判断哪个因子显著、哪个与目标指标机制相关。",
                        findings + f"\n\n用户问题：{state.get('user_message', '')}",
                    )
                    action = _validate_action(raw)
                    if attempt == 0:
                        first_raw = json.dumps(raw, ensure_ascii=False)[:300] if raw is not None else None
                        if action is not None:
                            first_raw = None
                    if action is not None:
                        break
                single_state = dict(state)
                _apply_action(single_state, action, error=first_raw)
                checks = score_scenario(single_state, case["expect"])
                entry["single_call"] = {
                    "checks": checks,
                    "passed": all(checks.values()),
                    "found_factor": (single_state.get("finish_decision") or {}).get("found_factor"),
                    "abstain": bool((single_state.get("analysis_result") or {}).get("abstain")),
                    "judgment_error": (single_state.get("analysis_result") or {}).get("judgment_error"),
                    "conclusion": (single_state.get("analysis_result") or {}).get("summary", "")[:160],
                    "answer": (single_state.get("analysis_result") or {}).get("summary", ""),
                }
            usage_for_case = usage_samples[usage_start:]
            entry["measurement"] = {
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "provider_calls": len(usage_for_case),
                "input_tokens": sum(item["input_tokens"] for item in usage_for_case),
                "output_tokens": sum(item["output_tokens"] for item in usage_for_case),
                "total_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in usage_for_case),
            }
            details.append(entry)
    finally:
        if use_llm:
            llm_client._record_usage = original_record_usage

    def summarize(arm: str) -> dict[str, Any]:
        rows = [item for item in details if arm in item]
        by_pattern: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
        for item in rows:
            bucket = by_pattern[item.get("pattern") or item["id"]]
            bucket["total"] += 1
            bucket["passed"] += bool(item[arm]["passed"])
        passed = sum(bool(item[arm]["passed"]) for item in rows)
        return {
            "cases": len(rows), "passed": passed,
            "completion_rate": round(passed / len(rows), 4) if rows else 0.0,
            "by_pattern": dict(sorted(by_pattern.items())),
        }

    payload: dict[str, Any] = {
        "arm": "single_call_baseline",
        "model": get_settings().llm_model if use_llm else None,
        "dataset_version": dataset.get("dataset_version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "details": details,
        "rule": summarize("rule"),
        "single_call": summarize("single_call") if use_llm else None,
        "measurement": {
            "total_input_tokens": sum(item["input_tokens"] for item in usage_samples),
            "total_output_tokens": sum(item["output_tokens"] for item in usage_samples),
            "total_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in usage_samples),
            "provider_calls": len(usage_samples),
            "avg_tokens_per_case": round(sum(item["input_tokens"] + item["output_tokens"] for item in usage_samples) / len(details), 3) if details else 0,
        },
    }
    provider_path = ROOT / "data" / "evaluations" / "agent-vs-workflow-provider.json"
    if provider_path.exists():
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        agent_rows = provider.get("details", []) + provider.get("longtail", {}).get("details", [])
        agent_by_id = {item["id"]: bool(item["passed"]) for item in agent_rows}
        matched = [(item["id"], item) for item in details if item["id"] in agent_by_id]
        agent_passed = sum(agent_by_id[item_id] for item_id, _ in matched)
        payload["agent_from_provider_artifact"] = {
            "cases": len(matched),
            "passed": agent_passed,
            "completion_rate": round(agent_passed / len(matched), 4) if matched else 0.0,
            "artifact": str(provider_path),
            "note": "governed loop numbers from the latest provider run (2 LLM calls per case)",
        }
    output_arg = next((item for item in sys.argv if item.startswith("--output=")), "")
    output = Path(output_arg.split("=", 1)[1]) if output_arg else ROOT / "data" / "evaluations" / "single-call-baseline.json"
    if not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rule": payload["rule"],
        "single_call": payload["single_call"],
        "agent": payload.get("agent_from_provider_artifact"),
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
