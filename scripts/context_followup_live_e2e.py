"""Exercise real multi-turn metric follow-ups through the HTTP API and LLM.

Run from the API container so the JWT secret, database, provider credentials,
and configured real LLM are the same ones used by the running service:

    python -m scripts.context_followup_live_e2e

Each case completes a metric query, then asks an elliptical follow-up in the
same chat session and scope. The follow-up trace must show the compiler
resolving the prior metric, and its actual tool points must equal the trusted
Data Explorer preview.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from app.core.config import get_settings


BASE_URL = "http://127.0.0.1:8000"
START_DATE = "2017-09-01"
END_DATE = "2018-08-31"
GRAIN = "month"
TIMEOUT = (15, 600)
PLATFORM_ADMIN_USER_ID = "system"
CASES = [
    {
        "metric": "seller_items_per_order",
        "label": "店铺平均每单商品数",
        "seller_id": "6560211a19b47992c3666cc44a7e94c0",
    },
    {
        "metric": "seller_avg_late_delivery_days",
        "label": "店铺平均延迟送达天数",
        "seller_id": "4a3ca9315b744ce9f8e9374361493884",
        # Use the Registry's exact alias on the first turn; the follow-up
        # intentionally omits the metric name to exercise authorized reuse.
        "parent_question": "按月统计店铺平均延误送达天数，哪月最高？",
    },
    {
        "metric": "seller_avg_item_price",
        "label": "店铺平均商品售价",
        "seller_id": "da8622b14eb17ae2831f4ac5b9dab84a",
    },
    {
        "metric": "seller_new_customer_rate",
        "label": "店铺新客占比",
        "seller_id": "cc419e0650a3c5ba77189a1882b7556a",
    },
    {
        "metric": "seller_credit_card_payment_share",
        "label": "店铺信用卡支付金额占比",
        "seller_id": "1f50f920176fa81dab994f9023523100",
    },
    {
        "metric": "seller_active_product_count",
        "label": "店铺活跃商品数",
        "seller_id": "955fee9216a65b617aa5c0531780ce60",
    },
    {
        "metric": "seller_avg_payment_installments",
        "label": "店铺平均分期数",
        "seller_id": "7a67c85e85bb2ce8582c35f2203ad736",
    },
    {
        "metric": "seller_order_bad_review_rate",
        "label": "店铺订单差评率",
        "seller_id": "ea8482cd71df3c1969d7b9473ff13abc",
    },
    {
        "metric": "seller_historical_returning_customer_rate",
        "label": "店铺历史回流客户率",
        "seller_id": "4869f7a5dfa277a7dca6462dcf3b52b2",
    },
    {
        "metric": "seller_freight_to_gmv_rate",
        "label": "店铺运费率（占成交额）",
        "seller_id": "3d871de0142ce09b7081e2b9d1733cb1",
        "parent_question": "按月分析店铺运费占成交额的比例趋势。",
    },
    {
        "metric": "seller_new_customer_count",
        "label": "店铺新客数",
        "seller_id": "da8622b14eb17ae2831f4ac5b9dab84a",
    },
]


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwt(user_id: str) -> str:
    secret = get_settings().tenant_auth_secret
    if not secret:
        raise RuntimeError("TENANT_AUTH_SECRET is not configured")
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    claims = _b64url(json.dumps({
        "tenant_id": "platform", "sub": user_id, "scope_type": "platform",
        "role": "admin", "exp": int(time.time()) + 6 * 60 * 60,
    }, separators=(",", ":")).encode())
    signing_input = f"{header}.{claims}"
    signature = hmac.new(secret.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _json(response: requests.Response, label: str) -> dict:
    if response.status_code != 200:
        raise RuntimeError(f"{label}: HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}: expected JSON object")
    return payload


def _points(trace: dict) -> list[dict]:
    state = trace.get("state_snapshot") or {}
    result = state.get("tool_result") or {}
    nested = result.get("result") or {}
    trend = nested.get("trend_result") or {}
    return list(trend.get("points") or [])


def _same_points(expected: list[dict], actual: list[dict]) -> bool:
    if len(expected) != len(actual):
        return False
    for left, right in zip(expected, actual):
        left_bucket = str(left.get("bucket") or left.get("metric_date") or "")
        right_bucket = str(right.get("bucket") or right.get("metric_date") or "")
        try:
            same_value = math.isclose(float(left["value"]), float(right["value"]), rel_tol=1e-9, abs_tol=1e-8)
        except (KeyError, TypeError, ValueError):
            return False
        if left_bucket != right_bucket or not same_value:
            return False
    return True


def _answer_extrema_match(points: list[dict], answer: str) -> bool:
    """Check max/min claims against data across common Chinese phrasings."""
    if not points:
        return False
    normalized = []
    for point in points:
        bucket = str(point.get("bucket") or point.get("metric_date") or "")
        try:
            value = float(point["value"])
        except (KeyError, TypeError, ValueError):
            return False
        normalized.append((bucket, value))

    date_pattern = re.compile(r"20\d{2}(?:-\d{1,2}(?:-\d{1,2})?|[^\d]{0,4}\d{1,2}[^\d]{0,2})")
    number_pattern = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
    markers = {
        "max": re.compile("\u6700\u9ad8\u70b9|\u6700\u9ad8\u503c|\u6700\u9ad8|\u5cf0\u503c|\u5cf0\u9876|max", re.I),
        "min": re.compile("\u6700\u4f4e\u70b9|\u6700\u4f4e\u503c|\u6700\u4f4e|\u4f4e\u70b9|\u8c37\u503c|min", re.I),
    }
    checked_claim = False
    for kind, marker in markers.items():
        claims = list(marker.finditer(answer))
        if not claims:
            continue
        checked_claim = True
        target = (max if kind == "max" else min)(value for _, value in normalized)
        extreme_points = [
            (bucket, value) for bucket, value in normalized
            if math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-8)
        ]
        found = False
        for claim in claims:
            # Supports wording such as "2018-04 ???? 42" and
            # "??? 42?2018-04?".
            window = answer[max(0, claim.start() - 48):min(len(answer), claim.end() + 48)]
            dates = date_pattern.findall(window)
            numbers = number_pattern.findall(window)
            for raw_date in dates:
                parts = [int(part) for part in re.findall(r"\d+", raw_date)]
                claim_date = f"{parts[0]:04d}-{parts[1]:02d}-{(parts[2] if len(parts) > 2 else 1):02d}"
                for bucket, value in extreme_points:
                    normalized_bucket = bucket[:7] + "-01" if len(bucket) == 7 else bucket[:10]
                    if normalized_bucket == claim_date:
                        value_near_claim = any(
                            math.isclose(float(number.rstrip("%")), round(value, 2), abs_tol=0.011)
                            for number in numbers
                        )
                        # Some answers state the extremum month in the summary
                        # and give its exact value in the month-by-month list.
                        value_near_date = False
                        for date_match in date_pattern.finditer(answer):
                            date_parts = [int(part) for part in re.findall(r"\d+", date_match.group())]
                            date_key = f"{date_parts[0]:04d}-{date_parts[1]:02d}-{(date_parts[2] if len(date_parts) > 2 else 1):02d}"
                            if date_key != claim_date:
                                continue
                            date_window = answer[max(0, date_match.start() - 20):min(len(answer), date_match.end() + 24)]
                            value_near_date = any(
                                math.isclose(float(number.rstrip("%")), round(value, 2), abs_tol=0.011)
                                for number in number_pattern.findall(date_window)
                            )
                            if value_near_date:
                                break
                        if value_near_claim or value_near_date:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            return False
    return checked_claim

def _turn_id(session: requests.Session, session_id: str, trace_id: str) -> str:
    payload = _json(session.get(f"{BASE_URL}/chat/sessions/{session_id}/turns", timeout=30), "turn history")
    for turn in payload.get("items") or []:
        if turn.get("trace_id") == trace_id:
            if turn.get("status") != "completed":
                raise RuntimeError(f"parent turn status was {turn.get('status')!r}")
            return str(turn["id"])
    raise RuntimeError("completed parent turn is missing from persisted history")


def _run_case(session: requests.Session, index: int, case: dict) -> dict:
    session_id = f"context-followup-{uuid4().hex}"
    explorer = {
        "seller_id": case["seller_id"], "metric_name": case["metric"],
        "start_date": START_DATE, "end_date": END_DATE, "granularity": GRAIN,
    }
    preview = _json(session.post(f"{BASE_URL}/data-explorer/preview", json=explorer, timeout=60), "trusted preview")
    expected = list(preview.get("series") or [])
    if not expected or (preview.get("context") or {}).get("metric_name") != case["metric"]:
        raise RuntimeError("trusted preview is empty or resolved a different metric")

    parent_question = case.get("parent_question") or f"请按月分析{case['label']}的趋势。"
    parent = _json(session.post(f"{BASE_URL}/chat", json={
        "message": parent_question, "use_current_explorer_context": True,
        "explorer_context": explorer, "memory_enabled": False,
        "chat_session_id": session_id, "chat_turn_id": f"parent-{uuid4().hex}",
    }, timeout=TIMEOUT), "parent chat")
    if parent.get("error") or parent.get("clarification") or not parent.get("answer"):
        raise RuntimeError(f"parent query did not complete: {parent}")
    parent_turn_id = _turn_id(session, session_id, str(parent["trace_id"]))

    followup_question = "那再按月看看趋势。"
    followup = _json(session.post(f"{BASE_URL}/chat", json={
        "message": followup_question, "use_current_explorer_context": True,
        "explorer_context": explorer, "memory_enabled": False,
        "chat_session_id": session_id, "chat_turn_id": f"followup-{uuid4().hex}",
        "inherited_from_turn_id": parent_turn_id,
    }, timeout=TIMEOUT), "follow-up chat")
    expected_clarification = bool(case.get("expected_clarification"))
    followup_clarification = followup.get("clarification")
    if followup.get("error") or (expected_clarification and not followup_clarification):
        raise RuntimeError(f"follow-up did not produce the expected clarification: {followup}")
    if not expected_clarification and (followup_clarification or not followup.get("answer")):
        raise RuntimeError(f"follow-up query did not complete: {followup}")

    trace = _json(session.get(f"{BASE_URL}/traces/{followup['trace_id']}", timeout=30), "follow-up trace")
    state = trace.get("state_snapshot") or {}
    compilation = state.get("context_compilation") or {}
    decision = state.get("metric_planning_decision") or {}
    actual = _points(trace)
    expected_metric = case["metric"]
    resolved = compilation.get("resolved_references") or []
    metric_slot = (compilation.get("known_slots") or {}).get("metric") or {}
    checks = {
        "relation_continued": compilation.get("relation") == "continue",
        "reference_resolved": len(resolved) == 1 and resolved[0].get("value") == expected_metric,
        "standalone_question_names_metric": case["label"] in str(compilation.get("standalone_question") or ""),
        "compiler_did_not_request_clarification": not compilation.get("needs_clarification"),
        "metric_slot_from_history": metric_slot.get("source") == "previous_turn_context",
    }
    if expected_clarification:
        clarification_metrics = set((followup_clarification or {}).get("unavailable_metrics") or [])
        checks.update({
            "planner_requested_safe_clarification": decision.get("status") == "clarification_required",
            "no_unintended_data_query": not actual,
            "clarification_names_competing_metric": bool(clarification_metrics),
        })
    else:
        checks.update({
            "no_clarification": not followup_clarification,
            "metric_plan_confirmed": decision.get("status") == "confirmed",
            "actual_points_match_trusted_preview": _same_points(expected, actual),
            "answer_extrema_match_trusted_preview": _answer_extrema_match(actual, str(followup.get("answer") or "")),
            "answer_nonempty": bool(str(followup.get("answer") or "").strip()),
        })
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "case": index, "metric": expected_metric, "label": case["label"],
        "seller_id": case["seller_id"], "date_range": [START_DATE, END_DATE], "grain": GRAIN,
        "parent_question": parent_question, "followup_question": followup_question,
        "parent_trace_id": parent["trace_id"], "followup_trace_id": followup["trace_id"],
        "parent_turn_id": parent_turn_id, "followup_answer": followup["answer"],
        "followup_clarification": followup_clarification,
        "outcome": "safe_clarification" if expected_clarification else "executed_query",
        "context_compilation": compilation, "metric_planning_decision": decision,
        "trusted_preview_points": expected, "actual_tool_points": actual,
        "checks": checks, "status": "passed" if not failures else "failed", "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric", action="append", dest="metrics",
        help="Run only this metric id (repeat the option to run several).",
    )
    parser.add_argument(
        "--output", default="docs/testing/context-followup-live-results.json",
        help="Write the JSON evidence artifact to this path.",
    )
    args = parser.parse_args()
    selected_cases = [case for case in CASES if not args.metrics or case["metric"] in args.metrics]
    unknown_metrics = set(args.metrics or []) - {case["metric"] for case in selected_cases}
    if unknown_metrics:
        parser.error(f"unknown metrics: {', '.join(sorted(unknown_metrics))}")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {_jwt(PLATFORM_ADMIN_USER_ID)}",
        "Accept": "application/json",
    })
    health = _json(session.get(f"{BASE_URL}/health/ready", timeout=20), "readiness")
    results = []
    for index, case in enumerate(selected_cases, start=1):
        started = time.perf_counter()
        try:
            result = _run_case(session, index, case)
        except Exception as exc:  # preserve evidence from remaining metrics too
            result = {
                "case": index, "metric": case["metric"], "label": case["label"],
                "status": "failed", "failures": [f"{type(exc).__name__}: {exc}"],
            }
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        results.append(result)
        print(
            f"[{result['status'].upper()}] {index}/{len(selected_cases)} {case['metric']} "
            f"resolved={result.get('checks', {}).get('reference_resolved')} "
            f"points_match={result.get('checks', {}).get('actual_points_match_trusted_preview', 'clarification_expected')} "
            f"trace={result.get('followup_trace_id', '-')} {result.get('failures', [])}",
            flush=True,
        )

    artifact = {
        "schema_version": "1.0", "suite": "real_llm_metric_context_followups",
        "started_at": datetime.now(timezone.utc).isoformat(), "api_base_url": BASE_URL,
        "readiness": health, "case_count": len(results),
        "passed": sum(item.get("status") == "passed" for item in results),
        "failed": sum(item.get("status") != "passed" for item in results),
        "results": results,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: artifact[key] for key in ("suite", "started_at", "case_count", "passed", "failed")}, ensure_ascii=False), flush=True)
    return 0 if artifact["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
