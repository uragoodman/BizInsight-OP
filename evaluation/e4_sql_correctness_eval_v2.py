# -*- coding: utf-8 -*-
"""E4.1 SQL/data-correctness evaluation: 39 cases, baseline, complex dimensions, repair.

The 36 planner cases come from the frozen E4.1 matrix. Three additional probes
exercise the MCP tool retry path: two transient failures should repair, one
permanent failure is a control that must remain failed.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

from app.core.time_expression import extract_iso_dates
from app.data.repositories import daily_metrics
from app.mcp import tools as mcp_tools
from app.resilience.policy import _BREAKERS

from evaluation.e4_sql_correctness_eval import (
    _admin_url,
    _check_plan,
    _close,
    _disposable_database,
    _execute,
    _fixture_runtime,
    _plan,
    _actual_result_value,
)
from evaluation.e4_v2_fixture import (
    REPAIR_PROBES,
    SCHEMA_STATEMENTS,
    SEED_STATEMENTS,
    build_cases,
)

_BASELINE_METRICS = (
    ("avg_order_value", ("客单价", "平均客单价", "aov")),
    ("order_count", ("订单量", "订单数", "order_count")),
    ("freight_value", ("运费",)),
    ("gmv", ("gmv", "成交总额", "销售额")),
)
_BASELINE_DIMENSIONS = (
    ("seller_state", ("卖家州", "商家州")),
    ("customer_state", ("州",)),
    ("customer_city", ("城市",)),
    ("payment_type", ("支付方式",)),
    ("product_category", ("品类", "类别")),
    ("month_position", ("月内", "月初", "月中", "月末")),
    ("day_of_week", ("周几", "星期几")),
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_cases(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in build_cases()) + "\n",
        encoding="utf-8",
    )


def _baseline_parse(question: str) -> dict[str, Any] | None:
    lowered = question.lower()
    metric = next(
        (metric for metric, aliases in _BASELINE_METRICS if any(alias.lower() in lowered for alias in aliases)),
        None,
    )
    if metric is None:
        return None
    dates = extract_iso_dates(question)
    dimension = next(
        (dimension for dimension, aliases in _BASELINE_DIMENSIONS if any(alias in question for alias in aliases)),
        None,
    )
    if ("对比" in question or "compare" in lowered) and len(dates) >= 2:
        return {
            "query_type": "metric_compare", "metric": metric,
            "baseline": dates[0], "current": dates[1],
        }
    if ("趋势" in question or "trend" in lowered) and len(dates) >= 2:
        return {
            "query_type": "metric_trend", "metric": metric,
            "start": dates[0], "end": dates[1],
        }
    if dimension and any(token in question for token in ("拆解", "分布", "按")) and dates:
        return {
            "query_type": "metric_breakdown", "metric": metric,
            "date": dates[0], "dimension": dimension,
        }
    return {
        "query_type": "metric_value", "metric": metric,
        "date": dates[0] if dates else None,
    }


def _baseline_execute(parsed: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if parsed is None:
        return None, None, "metric_not_found"
    try:
        query_type = parsed["query_type"]
        metric = parsed["metric"]
        if query_type == "metric_value":
            if not parsed["date"]:
                return None, None, "missing_date"
            payload = daily_metrics.query_metric(metric, parsed["date"])
        elif query_type == "metric_trend":
            payload = daily_metrics.query_metric_trend(
                metric, parsed["start"], parsed["end"], "day",
            )
        elif query_type == "metric_compare":
            raw = daily_metrics.compare_metric(metric, parsed["baseline"], parsed["current"])
            payload = {
                "metric_name": metric,
                "baseline_date": parsed["baseline"],
                "current_date": parsed["current"],
                "baseline_value": raw.get("left_value"),
                "current_value": raw.get("right_value"),
                "delta_value": raw.get("delta_value"),
                "delta_ratio": raw.get("delta_ratio"),
                "status": raw.get("status"),
            }
        else:
            raw = daily_metrics.breakdown_by_dimension(
                metric, parsed["date"], parsed["dimension"], 10,
            )
            payload = {
                **raw,
                "items": {
                    str(item.get("dimension_value")): item.get("value")
                    for item in raw.get("items") or []
                },
            }
        return payload, payload.get("status"), None
    except Exception as exc:  # noqa: BLE001 - baseline failures are measured, not hidden
        return None, None, f"{type(exc).__name__}: {exc}"


def _baseline_behavior(payload: dict[str, Any] | None, status: str | None, error: str | None) -> str:
    if error is not None:
        return "error"
    if status == "ok":
        return "result"
    if status == "not_found":
        return "empty"
    return "no_result"


def _run_baseline(case: dict[str, Any]) -> dict[str, Any]:
    parsed = _baseline_parse(str(case["question"]))
    payload, status, error = _baseline_execute(parsed)
    actual_behavior = _baseline_behavior(payload, status, error)
    expected = case["expected"]
    result_expected = {
        key: value
        for key, value in expected.items()
        if key not in {"decision_status", "query_type"}
    }
    result_correct = False
    if case["expected_behavior"] == "result" and payload is not None:
        result_correct = status == "ok" and _close(
            payload, result_expected, float(case.get("tolerance", 0.01)),
        )
    elif case["expected_behavior"] == "empty" and payload is not None:
        result_correct = _close(payload, result_expected, float(case.get("tolerance", 0.01)))
    return {
        "case_id": case["case_id"],
        "expected_behavior": case["expected_behavior"],
        "parsed": parsed,
        "status": status,
        "error": error,
        "actual_behavior": actual_behavior,
        "checks": {
            "behavior": actual_behavior == case["expected_behavior"],
            "execution": status in {"ok", "not_found"},
            "result": result_correct,
        },
    }


def _current_behavior(
    decision_status: str,
    reason_code: str | None,
    unavailable_metrics: list[str],
    execution_status: str | None,
    execution_reason: str | None,
) -> str:
    if decision_status == "confirmed":
        if execution_status == "ok":
            return "result"
        if execution_status == "not_found":
            return "empty"
        if execution_status == "unsupported":
            if any(token in str(execution_reason or "") for token in ("日期", "缺少", "clarif")):
                return "clarify"
            return "unsupported"
        if execution_status == "execution_failed":
            return "execution_error"
        return "no_result"
    if reason_code in {"unsupported_metric", "scope_rejected"} or unavailable_metrics:
        return "unauthorized"
    if decision_status == "clarification_required":
        return "clarify"
    if decision_status == "unsupported":
        return "refuse"
    return decision_status


def _run_current(case: dict[str, Any]) -> dict[str, Any]:
    batch = _plan(case)
    decision = batch.decision
    expected = case["expected"]
    details: dict[str, Any] = {
        "case_id": case["case_id"],
        "expected_behavior": case["expected_behavior"],
        "decision_status": decision.status,
        "reason_code": decision.reason_code,
        "unavailable_metrics": decision.unavailable_metrics,
        "plans": [],
        "response": None,
        "execution_status": None,
        "actual_behavior": None,
        "checks": {},
    }
    if decision.status == "confirmed" and batch.plans:
        plan = batch.plans[0]
        details["plans"] = [{
            "query_type": plan.query_type,
            "metric_name": plan.metric_name,
            "metric_date": plan.metric_date,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "baseline_date": plan.baseline_date,
            "current_date": plan.current_date,
            "dimension": plan.dimension,
        }]
        response = _execute(plan, str(case["question"]))
        result = response.get("result") if isinstance(response.get("result"), dict) else {}
        actual_payload = _actual_result_value(plan, result)
        details["response"] = {
            "server": response.get("server"),
            "tool_name": response.get("tool_name"),
            "status": result.get("status"),
            "reason": result.get("reason"),
            "result": actual_payload,
        }
        details["execution_status"] = result.get("status")
        details["actual_behavior"] = _current_behavior(
            decision.status, decision.reason_code, decision.unavailable_metrics,
            details["execution_status"], result.get("reason"),
        )
        details["checks"]["plan"] = _check_plan(plan, expected)
        details["checks"]["execution"] = (
            result.get("status") in {"ok", "not_found"}
            if case["expected_behavior"] in {"result", "empty"}
            else True
        )
        expected_result = {
            key: value
            for key, value in expected.items()
            if key not in {"decision_status", "query_type"}
        }
        if case["expected_behavior"] == "result":
            details["checks"]["result"] = _close(
                actual_payload, expected_result, float(case.get("tolerance", 0.01)),
            ) and result.get("status") == "ok"
        elif case["expected_behavior"] == "empty":
            details["checks"]["result"] = _close(
                actual_payload, expected_result, float(case.get("tolerance", 0.01)),
            )
        else:
            details["checks"]["result"] = True
    else:
        details["actual_behavior"] = _current_behavior(
            decision.status, decision.reason_code, decision.unavailable_metrics, None, None,
        )
        details["checks"]["plan"] = decision.status == expected.get("decision_status") and (
            "reason_code" not in expected or decision.reason_code == expected["reason_code"]
        )
        details["checks"]["execution"] = True
        details["checks"]["result"] = True
    details["checks"]["behavior"] = details["actual_behavior"] == case["expected_behavior"]
    if case["expected_behavior"] in {"result", "empty"}:
        invalid = not (
            details["checks"]["plan"]
            and details["checks"]["execution"]
            and details["response"] is not None
        )
    else:
        executed = bool(
            details["response"] and details["response"].get("tool_name")
        )
        invalid = executed and details["execution_status"] in {"ok", "not_found"}
    details["checks"]["invalid_tool_call"] = invalid
    return details


def _summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    data_cases = [row for row in details if row["expected_behavior"] in {"result", "empty"}]
    result_cases = [row for row in details if row["expected_behavior"] == "result"]
    behavior = sum(1 for row in details if row["checks"]["behavior"])
    plan = sum(1 for row in details if row["checks"]["plan"])
    execution = sum(1 for row in data_cases if row["checks"]["execution"])
    result = sum(1 for row in result_cases if row["checks"]["result"])
    invalid = sum(1 for row in details if row["checks"]["invalid_tool_call"])
    return {
        "cases": len(details),
        "behavior_correct": behavior,
        "behavior_correct_rate": behavior / len(details) if details else None,
        "plan_correct": plan,
        "plan_correct_rate": plan / len(details) if details else None,
        "data_cases": len(data_cases),
        "execution_success": execution,
        "execution_success_rate": execution / len(data_cases) if data_cases else None,
        "result_cases": len(result_cases),
        "result_correct": result,
        "result_correct_rate": result / len(result_cases) if result_cases else None,
        "invalid_tool_calls": invalid,
        "invalid_tool_call_rate": invalid / len(details) if details else None,
    }


def _baseline_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    behavior = sum(1 for row in details if row["checks"]["behavior"])
    data_cases = [row for row in details if row["expected_behavior"] in {"result", "empty"}]
    result_cases = [row for row in details if row["expected_behavior"] == "result"]
    execution = sum(1 for row in data_cases if row["checks"]["execution"])
    result = sum(1 for row in result_cases if row["checks"]["result"])
    return {
        "cases": len(details),
        "behavior_correct": behavior,
        "behavior_correct_rate": behavior / len(details) if details else None,
        "data_cases": len(data_cases),
        "execution_success": execution,
        "execution_success_rate": execution / len(data_cases) if data_cases else None,
        "result_cases": len(result_cases),
        "result_correct": result,
        "result_correct_rate": result / len(result_cases) if result_cases else None,
        "errors": sum(1 for row in details if row["error"] is not None),
    }


def _run_repair_probe(probe: dict[str, Any]) -> dict[str, Any]:
    _BREAKERS.clear()
    original = getattr(mcp_tools, probe["repository_attr"])
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= int(probe["failures"]):
            raise RuntimeError("injected transient repository failure")
        return original(*args, **kwargs)

    error = None
    result = None
    with patch.object(mcp_tools, probe["repository_attr"], flaky):
        try:
            result = getattr(mcp_tools, probe["tool"])(*probe["args"])
        except Exception as exc:  # noqa: BLE001 - probe failure is measured
            error = f"{type(exc).__name__}: {exc}"
    _BREAKERS.clear()
    success = bool(result and result.get("status") == "ok")
    return {
        "case_id": probe["case_id"],
        "tool": probe["tool"],
        "injected_failures": probe["failures"],
        "attempts": calls["count"],
        "success": success,
        "expected": probe["expected"],
        "error": error,
        "status": (result or {}).get("status"),
        "check": (
            success if probe["expected"] == "repaired"
            else not success
        ),
    }


def _repair_summary(probes: list[dict[str, Any]]) -> dict[str, Any]:
    repaired = [probe for probe in probes if probe["expected"] == "repaired"]
    controls = [probe for probe in probes if probe["expected"] == "failed"]
    return {
        "probes": len(probes),
        "transient_attempted": len(repaired),
        "transient_repaired": sum(1 for probe in repaired if probe["check"]),
        "transient_repair_rate": (
            sum(1 for probe in repaired if probe["check"]) / len(repaired)
            if repaired else None
        ),
        "permanent_controls": len(controls),
        "permanent_control_passed": sum(1 for probe in controls if probe["check"]),
        "details": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/datasets/e4_sql_cases_v2.jsonl")
    parser.add_argument("--admin-url", default=None)
    parser.add_argument("--write-cases", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.write_cases:
        write_cases(Path(args.write_cases))
        print(json.dumps({"written": args.write_cases, "cases": len(build_cases())}, ensure_ascii=False))
        return 0

    cases = _load_cases(Path(args.cases))
    if len(cases) != 36:
        raise SystemExit(f"E4.1 planner matrix must contain 36 cases, got {len(cases)}")
    admin_url = _admin_url(args.admin_url)
    with _disposable_database(
        admin_url,
        schema_statements=SCHEMA_STATEMENTS,
        seed_statements=SEED_STATEMENTS,
    ) as test_engine:
        with _fixture_runtime(
            test_engine,
            data_bounds=(date(2024, 1, 1), date(2024, 1, 31)),
        ):
            current = [_run_current(case) for case in cases]
            baseline = [_run_baseline(case) for case in cases]
            repairs = [_run_repair_probe(probe) for probe in REPAIR_PROBES]
    report = {
        "schema_version": "e4-sql-correctness-v2",
        "dataset_version": "e4-sql-cases-v2",
        "snapshot": "e4-fixture-v2",
        "planner_cases": len(cases),
        "repair_probes": len(repairs),
        "total_cases": len(cases) + len(repairs),
        "current": _summary(current),
        "baseline": _baseline_summary(baseline),
        "repair": _repair_summary(repairs),
        "current_details": current,
        "baseline_details": baseline,
        "limits": [
            "E4.1 seed matrix: 36 planner cases plus 3 repair probes; not a production sample.",
            "Baseline is a keyword/regex direct repository path without authorization or behavior policy.",
            "Repair probes exercise the MCP tool's bounded retry wrapper, not a planner-level repair loop.",
            "The fixture is a disposable PostgreSQL database; no application data is read or written.",
            "Complex dimensions are covered for product/category, customer state/city, seller state, payment type, month position and weekday.",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    return 0 if (
        report["current"]["behavior_correct_rate"] == 1.0
        and report["current"]["result_correct_rate"] == 1.0
        and report["repair"]["transient_repair_rate"] == 1.0
        and report["repair"]["permanent_control_passed"] == report["repair"]["permanent_controls"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
