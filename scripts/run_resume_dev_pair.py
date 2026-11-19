"""Run one explicitly authorized, development-only B/G Provider pair.

This is deliberately not connected to the frozen formal candidate manifest.
The question and hand oracle come from the existing order_dimension dev fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.llm.call_config import LLMCallConfig  # noqa: E402
from app.tenancy import TenantContext  # noqa: E402
from evaluation.provider_candidate_batch import (  # noqa: E402
    CostPolicy,
    EstimatedCostLedger,
    ExperimentConfig,
    RunLedger,
    _reconcile_cost_accounting,
    bind_isolated_experiment_runtime,
    token_usage_snapshot,
)
from evaluation.resume_bare_tool_loop import run_bare_tool_loop  # noqa: E402


QUESTION = (
    "请解释平台已交付订单量从 2024-05-03 到 2024-05-04 的变化，"
    "先核对趋势，再按 product_category 展示统计贡献；统计贡献不等于业务因果。"
)
CASE_ID = "order_dimension"
ORACLE = {
    "target_metric": "order_count",
    "dimension": "product_category",
    "baseline_date": "2024-05-03",
    "current_date": "2024-05-04",
    "baseline_value": 10.0,
    "current_value": 6.0,
    "delta": -4.0,
    "group_delta": {"A": -4.0, "B": 0.0},
    "unit": "orders",
    "source": "tests/test_non_gmv_diagnostic_workflow.py::seed_facts(order_dimension); hand oracle DIMENSION_ORACLES",
}
RESULTS_PARENT = ROOT / "docs" / "testing"
ORIGINAL_PAIR_DIR = RESULTS_PARENT / "resume_dev_pair_order_dimension_20261015_249fd88a"
ORIGINAL_BASIC_RUN_ID = "249fd88a-437d-449b-957f-20ab328b09fe"
GOVERNED_RUN_ID = "701b9cbe-3452-4d19-a5b4-1ce00019f6f1"
ORIGINAL_ESTIMATE_LIMIT_CNY = 5.0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_usage_summary(ledger: RunLedger, run_id: str, usage: dict) -> dict:
    """Append a usage event only after verifying its trace/run identity."""
    observed_run_id = usage.get("run_id")
    if not isinstance(observed_run_id, str) or observed_run_id != run_id:
        raise ValueError("usage_summary_run_id_mismatch")
    fields = {key: value for key, value in usage.items() if key != "run_id"}
    return ledger.append(run_id, "usage_summary", **fields)


def _stop_reason_after_run(state: dict, usage: dict, sdk_failed_sends: int) -> str | None:
    if usage.get("status") == "metering_error" or usage.get("cost_record_matches") is False:
        return "metering_error"
    if usage.get("status") == "sent_usage_unknown":
        # The pair's cumulative estimate is not meaningful after an actual
        # send has unknown usage; do not start another arm on a fresh reserve.
        return "usage_unknown"
    # A model-selected invalid/unsupported tool request is a task-level result;
    # it must be recorded and the paired arm may still run. Stop the pair only
    # when the tool failure carries evidence of an infrastructure failure.
    infrastructure_types = {
        "ConnectionError", "InterfaceError", "OperationalError",
        "OSError", "TimeoutError",
    }
    infrastructure_codes = {
        "timeout", "circuit_open", "external_call_failed", "connection_error",
    }
    for row in state.get("errors") or []:
        row = row or {}
        if row.get("code") != "tool_execution_error":
            continue
        if (
            row.get("service") in {"mcp", "postgresql"}
            or row.get("service_error_code") in infrastructure_codes
            or row.get("exception_type") in infrastructure_types
            or row.get("cause_type") in infrastructure_types
        ):
            return "tool_infrastructure_error"
    if sdk_failed_sends > 0:
        return "provider_send_failure"
    return None


def _mark_record_lifecycle(record_path: Path, record: dict, status: str, **details) -> dict:
    updated = {**record, **details, "lifecycle_status": status}
    _write_json(record_path, updated)
    return updated


def _finalize_run_record(
    *, ledger: RunLedger, run_id: str, record_path: Path, record: dict,
    usage: dict, task_pass: bool, provider_sends: int, stop_reason: str | None,
) -> dict:
    """Persist a nonterminal record, reconcile usage, then publish terminal state."""
    provisional = _mark_record_lifecycle(record_path, record, "finalizing")
    _append_usage_summary(ledger, run_id, usage)
    if stop_reason:
        ledger.append(run_id, "failed", error_type=stop_reason, retry_forbidden=True)
        return _mark_record_lifecycle(
            record_path, provisional, "failed", lifecycle_error=stop_reason,
        )
    ledger.append(run_id, "completed", task_pass=task_pass, provider_sends=provider_sends)
    return _mark_record_lifecycle(record_path, provisional, "completed")


def _create_seeded_db(admin_url_text: str):
    admin_url = make_url(admin_url_text)
    host = str(admin_url.host or "").strip("[]").lower()
    try:
        if not ipaddress.ip_address(host).is_loopback and host != "localhost":
            raise RuntimeError("Isolated test PostgreSQL admin URL must be loopback-only.")
    except ValueError:
        if host != "localhost":
            raise RuntimeError("Isolated test PostgreSQL admin URL must be loopback-only.")
    db_name = f"platform_trend_acceptance_resume_{uuid4().hex}"
    admin_engine = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    engine = None
    try:
        with admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{db_name}"'))
        engine = create_engine(admin_url.set(database=db_name), pool_pre_ping=True)
        ddl = [
            "CREATE SCHEMA analytics",
            "CREATE SCHEMA raw",
            "CREATE TABLE analytics.daily_metrics (metric_date date PRIMARY KEY, gmv numeric, order_count bigint, avg_order_value numeric)",
            "CREATE TABLE analytics.fact_orders (order_id text PRIMARY KEY, customer_unique_id text, item_count int NOT NULL, item_gross_value numeric NOT NULL, freight_value numeric NOT NULL, purchase_date date NOT NULL, order_status text NOT NULL)",
            "CREATE VIEW analytics.v_delivered_orders AS SELECT * FROM analytics.fact_orders WHERE order_status='delivered'",
            "CREATE TABLE analytics.fact_order_payments (order_id text NOT NULL, payment_value numeric NOT NULL, payment_installments int NOT NULL)",
            "CREATE TABLE analytics.fact_fulfillment (order_id text PRIMARY KEY, purchase_date date NOT NULL, approval_lead_hours numeric, dispatch_lead_hours numeric, delivery_transit_hours numeric)",
            "CREATE TABLE analytics.v_order_cohort_funnel (purchase_date date NOT NULL, stage_approved int NOT NULL, stage_shipped int NOT NULL, final_delivered int NOT NULL, total_orders int NOT NULL)",
            "CREATE TABLE raw.orders (order_id text PRIMARY KEY, order_purchase_timestamp timestamp NOT NULL, order_status text NOT NULL, order_delivered_customer_date timestamp, order_estimated_delivery_date timestamp)",
            "CREATE TABLE raw.order_reviews (order_id text NOT NULL, review_score int, review_comment_message text)",
        ]
        with engine.begin() as connection:
            for statement in ddl:
                connection.execute(text(statement))
        # Reuse the existing synthetic facts exactly; do not source expected
        # values from Agent output or duplicate the fixture's fact generation.
        from test_non_gmv_diagnostic_workflow import seed_facts

        seed_facts(engine, CASE_ID)
        return admin_engine, engine, db_name
    except BaseException:
        if engine is not None:
            engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        admin_engine.dispose()
        raise


def _network_guard(provider_host: str):
    """Allow isolated DB, configured Provider, and socketpair's scoped local IPC."""
    provider_ips = {
        str(row[4][0]) for row in socket.getaddrinfo(provider_host, 443, type=socket.SOCK_STREAM)
    }
    if not provider_ips:
        raise RuntimeError("Configured Provider endpoint did not resolve; no workflow started.")
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_socketpair = socket.socketpair
    local_ipc = threading.local()

    def is_loopback(address):
        if not isinstance(address, tuple) or not address:
            return False
        host = str(address[0]).strip("[]").lower()
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def allowed(address):
        if not isinstance(address, tuple) or len(address) < 2:
            return False
        host, port = address[0], int(address[1])
        # Windows ProactorEventLoop builds its self-pipe by creating a
        # socketpair. Permit only loopback connects made synchronously inside
        # the stdlib socketpair implementation; arbitrary loopback ports stay
        # blocked outside this narrow scope.
        if getattr(local_ipc, "socketpair_active", False) and is_loopback(address):
            return True
        if host == "localhost" and port == 15435:
            return True
        try:
            if ipaddress.ip_address(host).is_loopback and port == 15435:
                return True
        except ValueError:
            pass
        return str(host) in provider_ips and port == 443

    def guarded_connect(sock, address):
        if not allowed(address):
            raise OSError("network policy blocked a non-Provider, non-loopback connection")
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        if not allowed(address):
            raise OSError("network policy blocked a non-Provider, non-loopback connection")
        return original_connect_ex(sock, address)

    def guarded_socketpair(*args, **kwargs):
        previous = getattr(local_ipc, "socketpair_active", False)
        local_ipc.socketpair_active = True
        try:
            return original_socketpair(*args, **kwargs)
        finally:
            local_ipc.socketpair_active = previous

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.socketpair = guarded_socketpair
    return original_connect, original_connect_ex, original_socketpair, sorted(provider_ips)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _first_nested(state: dict, field: str):
    for node in _walk(state):
        candidate = node.get(field)
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _near(value, expected, tolerance=0.01):
    try:
        return abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _text_checks(answer: str) -> dict[str, bool]:
    compact = answer.replace(",", "").replace("−", "-").replace("−", "-")
    return {
        "both_dates_visible": ORACLE["baseline_date"] in answer and ORACLE["current_date"] in answer,
        "baseline_current_values_visible": bool(re.search(r"(?<!\d)10(?:\.0+)?(?!\d)", compact)) and bool(re.search(r"(?<!\d)6(?:\.0+)?(?!\d)", compact)),
        "change_direction_visible": any(token in answer.lower() for token in ("下降", "减少", "降至", "decrease", "declin", "-4")),
        "statistical_contribution_visible": any(token in answer.lower() for token in ("贡献", "统计分解", "contribution")),
    }


def _score(state: dict, response: dict, arm: str) -> dict:
    trend = _first_nested(state, "trend_result")
    contribution = _first_nested(state, "contribution_result")
    points = trend.get("points") or []
    trend_values = [point.get("value") for point in points]
    trend_ok = (
        trend.get("metric_name") == ORACLE["target_metric"]
        and trend.get("start_date") == ORACLE["baseline_date"]
        and trend.get("end_date") == ORACLE["current_date"]
        and len(trend_values) >= 2
        and _near(trend_values[0], ORACLE["baseline_value"])
        and _near(trend_values[-1], ORACLE["current_value"])
    )
    items_now = {str(row.get("dimension_value")): row.get("value") for row in contribution.get("current_items", [])}
    items_before = {str(row.get("dimension_value")): row.get("value") for row in contribution.get("baseline_items", [])}
    group_checks = {
        key: _near((items_now.get(key) or 0) - (items_before.get(key) or 0), expected)
        for key, expected in ORACLE["group_delta"].items()
    }
    contribution_ok = (
        contribution.get("metric_name") == ORACLE["target_metric"]
        and contribution.get("dimension") == ORACLE["dimension"]
        and contribution.get("baseline_date") == ORACLE["baseline_date"]
        and contribution.get("current_date") == ORACLE["current_date"]
        and _near(contribution.get("total_delta"), ORACLE["delta"])
        and all(group_checks.values())
    )
    answer = str(response.get("answer") or "")
    if arm == "basic_tool_loop":
        final = state.get("final") or {}
        answer = str(final.get("answer") or answer)
        answer_claims = final.get("claims") or []
        evidence_ids = {str(row.get("evidence_id")) for row in state.get("tool_evidence", [])}
        cited_ids = {
            str(eid) for claim in answer_claims if isinstance(claim, dict)
            for eid in (claim.get("evidence_ids") or [])
        }
    else:
        answer_claims = state.get("claims") or []
        evidence_ids = {str(row.get("evidence_id")) for row in state.get("evidence_bundles", [])}
        cited_ids = {
            str(eid) for claim in answer_claims if isinstance(claim, dict)
            for eid in (claim.get("source_evidence_ids") or [])
        }
    binding_ok = bool(evidence_ids) and bool(cited_ids) and cited_ids.issubset(evidence_ids)
    text_checks = _text_checks(answer)
    causal_language_needs_review = any(
        token in answer.lower() for token in ("原因是", "导致", "caused by", "caused the", "because of")
    ) and not any(token in answer.lower() for token in ("未确认", "未知", "not confirmed", "unconfirmed"))
    manual_review = causal_language_needs_review or not all(text_checks.values())
    checks = {
        "target_trend_scope_period_and_values": bool(trend_ok),
        "registered_dimension_and_contribution_values": bool(contribution_ok),
        "answer_claim_evidence_binding": bool(binding_ok),
        **{f"answer_{key}": value for key, value in text_checks.items()},
        "no_unqualified_causal_claim": not causal_language_needs_review,
    }
    content_valid = bool(all(checks.values()))
    analysis = state.get("analysis_result") or {}
    response_errors = response.get("errors") or []
    state_errors = state.get("errors") or []
    error_flag = bool(response.get("error")) or bool(response_errors) or bool(state_errors)
    presentation = response.get("presentation") or state.get("analysis_presentation") or {}
    investigation = state.get("investigation_state") or {}
    stop = investigation.get("stop_reason") or {}
    stop_code = str(stop.get("code") or "") if isinstance(stop, dict) else ""

    if arm == "basic_tool_loop":
        final = state.get("final") or {}
        business_completed = bool(
            not error_flag
            and str(final.get("answer") or answer).strip()
            and stop_code not in {
                "BUDGET_EXHAUSTED", "DATA_INSUFFICIENT", "CAPABILITY_GAP",
                "AUTHORIZATION_BLOCKED", "INVALID_DATA", "EXECUTION_FAILED",
            }
        )
        review_applicability = "not_applicable"
        review_completed = None
        review_approved = None
        review_status = "not_applicable"
    else:
        business_completed = bool(
            not error_flag
            and presentation.get("status") == "completed"
            and analysis.get("abstain") is False
            and stop_code in {"ENOUGH_EVIDENCE", "COMPLETED"}
        )
        critic = state.get("critic_result") or {}
        llm_review = critic.get("llm_review") or {}
        mode = str(llm_review.get("mode") or "")
        if mode == "off" or mode == "evaluation_ablation":
            review_applicability = "evaluation_ablation"
            review_completed = None
            review_approved = None
            review_status = "evaluation_ablation"
        else:
            review_applicability = "required"
            required_reviews = ("groundedness", "completeness", "consistency")
            review_rows = [llm_review.get(name) for name in required_reviews]
            deterministic_reviews = [
                critic.get("data_review"), critic.get("claim_review"),
                critic.get("diagnostic_review"), critic.get("completeness_review"),
                critic.get("causal_review"),
            ]
            valid_deterministic_statuses = {
                "approved", "rejected", "incomplete", "inconclusive",
            }
            review_completed = bool(
                mode == "full"
                and critic.get("status") in {"approved", "rejected", "needs_revision"}
                and all(
                    isinstance(item, dict)
                    and item.get("status") in valid_deterministic_statuses
                    for item in deterministic_reviews
                )
                and len(review_rows) == 3
                and all(
                    isinstance(item, dict)
                    and item.get("completion_status") == "completed"
                    and item.get("verdict") in {"passed", "rejected"}
                    for item in review_rows
                )
            )
            review_status = "completed" if review_completed else "incomplete"
            review_approved = bool(
                review_completed
                and critic.get("status") == "approved"
                and llm_review.get("status") == "passed"
                and all(item.get("status") == "approved" for item in deterministic_reviews)
                and all(item.get("verdict") == "passed" for item in review_rows)
            )

    task_pass = bool(
        content_valid
        and business_completed
        and not manual_review
        and (review_applicability != "required" or (review_completed and review_approved))
    )
    return {
        "rubric_version": "resume_dev_pair_completion_v3",
        "parent_rubric_version": "resume_dev_pair_common_checks_v2_claim_source_binding",
        "claim_binding_contract": "basic_tool_loop.evidence_ids|governed_workflow.source_evidence_ids",
        "checks": checks,
        "group_delta_checks": group_checks,
        "effective_statistical_contribution": bool(trend_ok and contribution_ok and binding_ok),
        "content_valid": content_valid,
        "business_completed": business_completed,
        "business_stop_code": stop_code or None,
        "presentation_status": presentation.get("status"),
        "review_applicability": review_applicability,
        "review_status": review_status,
        "review_completed": review_completed,
        "review_approved": review_approved,
        "manual_review_required": bool(manual_review),
        "task_pass": task_pass,
        "task_acceptance_pass": task_pass,
        "note": "Versioned development scorer. Content validity, business completion, and review completion are separate. A validated partial result is not a completed task; statistical contribution does not establish business causality. Historical scores remain unchanged.",
    }


def _duplicate_tool_signatures(state: dict) -> dict:
    calls = state.get("tool_calls") or []
    signatures = []
    for call in calls:
        signatures.append(json.dumps({
            "tool": call.get("tool_name") or call.get("tool"),
            "arguments": call.get("arguments") or {},
        }, sort_keys=True, ensure_ascii=False, default=str))
    return {"count": len(signatures), "unique_count": len(set(signatures)), "duplicates": len(signatures) - len(set(signatures))}


def _continuation_config() -> tuple[ExperimentConfig, CostPolicy]:
    """The continuation inherits the original pair's exact model and cost limits."""
    config = ExperimentConfig(
        provider_id="primary", model="qwen3.8-flash", max_graph_steps=48,
        max_task_tokens=60_000, max_cost_cny_per_workflow=5.0,
        max_completion_tokens=4_096, max_sdk_sends_per_workflow=10,
        sdk_retries=0, application_attempts=1, fallback_enabled=False,
        max_tool_calls=8, max_latency_ms=240_000,
        model_timeout_seconds=60.0, max_revisions=1,
    )
    policy = CostPolicy(
        max_sdk_calls_per_workflow=10, input_token_estimate_cap_per_call=200_000,
        input_byte_factor=4, input_protocol_token_margin=16_384,
        max_completion_tokens=4_096, documented_output_variance_tokens=10,
        input_cny_per_million=1.094, output_cny_per_million=3.427,
        cumulative_estimate_limit_cny=ORIGINAL_ESTIMATE_LIMIT_CNY,
    )
    return config, policy


def _expected_manifest_config(config: ExperimentConfig) -> dict:
    return {**config.__dict__, "critic_thinking_mode": "non_thinking_experiment"}


def _validate_governed_continuation(source_dir: Path) -> dict:
    """Fail closed unless the original pair proves exactly one safe continuation.

    The append-only ledger is authoritative. Historical summary/trace files are
    read only and never rewritten by this validator.
    """
    source_dir = source_dir.resolve()
    manifest_path = source_dir / "run_manifest.json"
    events_path = source_dir / "run_events.jsonl"
    reservations_path = source_dir / "cost_reservations.jsonl"
    for required in (manifest_path, events_path, reservations_path):
        if not required.is_file():
            raise RuntimeError(f"continuation_required_file_missing:{required.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config, policy = _continuation_config()
    if manifest.get("case_id") != CASE_ID or manifest.get("user_question") != QUESTION:
        raise RuntimeError("continuation_manifest_case_or_question_mismatch")
    if manifest.get("planned_runs") != 2 or manifest.get("order") != ["basic_tool_loop", "governed_workflow"]:
        raise RuntimeError("continuation_manifest_pair_shape_mismatch")
    if manifest.get("run_ids") != {
        "basic_tool_loop": ORIGINAL_BASIC_RUN_ID,
        "governed_workflow": GOVERNED_RUN_ID,
    }:
        raise RuntimeError("continuation_manifest_run_ids_mismatch")
    if manifest.get("provider_id") != "primary" or manifest.get("model_config") != _expected_manifest_config(config):
        raise RuntimeError("continuation_manifest_model_or_config_mismatch")
    if manifest.get("operation_models") != {name: config.model for name in ("classify", "extract", "chat") }:
        raise RuntimeError("continuation_manifest_operation_models_mismatch")
    saved_policy = manifest.get("cost_policy") or {}
    expected_pair_cost = policy.estimate(2)
    for field in (
        "runs", "sdk_calls_upper_bound", "input_token_upper_bound",
        "output_token_upper_bound_including_variance", "per_run_cny",
        "estimated_cny", "cumulative_estimate_limit_cny", "within_estimate_limit",
        "is_account_hard_cap", "basis",
    ):
        if saved_policy.get(field) != expected_pair_cost.get(field):
            raise RuntimeError(f"continuation_manifest_cost_policy_mismatch:{field}")
    if manifest.get("is_account_hard_cap") is not False or expected_pair_cost["estimated_cny"] > ORIGINAL_ESTIMATE_LIMIT_CNY:
        raise RuntimeError("continuation_original_estimate_limit_invalid")

    ledger = RunLedger(events_path)
    events = ledger.events
    basic_events = [row for row in events if row.get("run_id") == ORIGINAL_BASIC_RUN_ID]
    governed_events = [row for row in events if row.get("run_id") == GOVERNED_RUN_ID]
    if sum(row.get("event") == "started" for row in basic_events) != 1:
        raise RuntimeError("continuation_basic_start_record_invalid")
    if sum(row.get("event") == "cost_reserved" for row in basic_events) != 1:
        raise RuntimeError("continuation_basic_ledger_reservation_invalid")
    if ledger.latest().get(ORIGINAL_BASIC_RUN_ID, {}).get("event") != "failed":
        raise RuntimeError("continuation_basic_not_terminal_failed")
    if sum(row.get("event") == "failed" for row in basic_events) != 1:
        raise RuntimeError("continuation_basic_terminal_record_invalid")
    if governed_events:
        raise RuntimeError("continuation_governed_has_existing_ledger_events")

    reservations = [
        json.loads(line) for line in reservations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    basic_reservations = [row for row in reservations if row.get("run_id") == ORIGINAL_BASIC_RUN_ID and row.get("reserved_cny") is not None]
    governed_reservations = [row for row in reservations if row.get("run_id") == GOVERNED_RUN_ID and row.get("reserved_cny") is not None]
    if len(basic_reservations) != 1 or governed_reservations:
        raise RuntimeError("continuation_cost_reservation_history_invalid")
    basic_usage_records = [row for row in reservations if row.get("run_id") == ORIGINAL_BASIC_RUN_ID and row.get("event") == "actual_usage_recorded"]
    if len(basic_usage_records) != 1 or basic_usage_records[0].get("usage_status") != "complete":
        raise RuntimeError("continuation_basic_cost_usage_record_invalid")
    existing_reserved = sum(float(row["reserved_cny"]) for row in reservations if row.get("reserved_cny") is not None)
    expected_per_run = policy.max_per_run_cny()
    if abs(float(basic_reservations[0]["reserved_cny"]) - expected_per_run) > 1e-10:
        raise RuntimeError("continuation_basic_reservation_mismatch")
    projected = existing_reserved + expected_per_run
    if projected > ORIGINAL_ESTIMATE_LIMIT_CNY + 1e-12:
        raise RuntimeError("continuation_cumulative_estimate_limit_exceeded")

    traces_dir = source_dir / "traces"
    if traces_dir.exists() and any(path.is_file() and GOVERNED_RUN_ID in path.name for path in traces_dir.rglob("*")):
        raise RuntimeError("continuation_governed_trace_or_record_already_exists")
    if any(path.is_file() and GOVERNED_RUN_ID in path.name for path in source_dir.glob(f"{GOVERNED_RUN_ID}.*")):
        raise RuntimeError("continuation_governed_result_already_exists")
    continuations_dir = source_dir / "continuations"
    target_dir = continuations_dir / GOVERNED_RUN_ID
    if target_dir.exists():
        raise RuntimeError("continuation_result_directory_already_exists")

    return {
        "source_dir": source_dir,
        "manifest": manifest,
        "manifest_sha256": _canonical_hash(manifest_path),
        "config": config,
        "policy": policy,
        "existing_reserved_cny": round(existing_reserved, 8),
        "new_reservation_cny": round(expected_per_run, 8),
        "projected_reserved_cny": round(projected, 8),
        "limit_cny": ORIGINAL_ESTIMATE_LIMIT_CNY,
        "target_dir": target_dir,
    }


def _run_governed_continuation(*, source_dir: Path, admin_url_text: str,
                               raw_settings: Any, provider_host: str,
                               models: dict[str, str]) -> dict:
    """Run only the already planned governed unit and append its lifecycle.

    Callers must perform Provider authorization separately. This function has
    no arm-selection or run-ID generation path by design.
    """
    preflight = _validate_governed_continuation(source_dir)
    config: ExperimentConfig = preflight["config"]
    policy: CostPolicy = preflight["policy"]
    if models != {name: config.model for name in ("classify", "extract", "chat")}:
        raise RuntimeError("continuation_runtime_operation_models_mismatch")
    endpoint_host = str(provider_host or "").lower()
    if endpoint_host != str(preflight["manifest"].get("provider_host") or "").lower():
        raise RuntimeError("continuation_runtime_provider_mismatch")

    # Atomic directory creation also prevents a concurrent second invocation.
    target_dir: Path = preflight["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = target_dir / "traces"
    ledger = RunLedger(preflight["source_dir"] / "run_events.jsonl")
    cost_ledger = EstimatedCostLedger(policy, preflight["source_dir"] / "cost_reservations.jsonl")
    context = TenantContext(
        tenant_id="platform", user_id="resume-dev-pair-order-dimension",
        role="admin", scope_type="platform",
    )
    db_name = None
    admin_engine = engine = None
    graph_runtime = None
    socket_originals = None
    sql_rows: list[dict] = []
    result: dict = {"run_id": GOVERNED_RUN_ID, "arm": "governed_workflow"}
    infrastructure_error = False
    stop_reason: str | None = None
    started = False
    sql_listener = None
    previous_checkpoint = None
    previous_checkpoint_context = None
    previous_runtime_settings = None
    try:
        # Resolve/guard the exact configured endpoint before database startup;
        # the offline regression replaces this guard with deny-all networking.
        socket_originals = _network_guard(provider_host)
        admin_engine, engine, db_name = _create_seeded_db(admin_url_text)
        from app.core import config as core_config
        from app.graph import runtime as graph_runtime_module
        from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
        from app.graph.critic_review_mode import CriticLLMReviewMode
        from app.graph import workflow
        from app.llm.routing import model_for
        from app.core.execution_context import active_state

        graph_runtime = graph_runtime_module
        previous_checkpoint = graph_runtime._CHECKPOINTER
        previous_checkpoint_context = graph_runtime._CHECKPOINTER_CONTEXT
        previous_runtime_settings = graph_runtime.get_settings
        checkpoint_path = target_dir / "workflow_checkpoints.sqlite"
        settings_for_runtime = raw_settings.model_copy(update={
            "trace_backend": "file", "trace_dir": str(trace_dir),
            "environment": "development", "checkpoint_backend": "sqlite",
            "checkpoint_path": str(checkpoint_path),
            "max_graph_steps": config.max_graph_steps,
            "max_tokens": config.max_task_tokens,
            "max_cost": config.max_cost_cny_per_workflow,
            "max_tool_calls": config.max_tool_calls,
            "max_latency_ms": config.max_latency_ms,
            "max_revisions": config.max_revisions,
            "llm_timeout_seconds": config.model_timeout_seconds,
            "llm_resilience_attempts": config.application_attempts,
        })
        graph_runtime._CHECKPOINTER = None
        graph_runtime._CHECKPOINTER_CONTEXT = None
        graph_runtime.get_settings = lambda: settings_for_runtime

        def capture_sql(_conn, _cursor, statement, _parameters, _context, _many):
            state = active_state() or {}
            if state.get("trace_id") == GOVERNED_RUN_ID and statement.lstrip().lower().startswith("select"):
                sql_rows.append({"run_id": GOVERNED_RUN_ID, "statement": statement})

        event.listen(engine, "before_cursor_execute", capture_sql)
        sql_listener = capture_sql
        with bind_isolated_experiment_runtime(
            engine=engine, trace_dir=trace_dir, ledger=ledger, config=config,
        ) as bound_settings:
            call_config = LLMCallConfig.provider_ab_experiment(
                bound_settings, provider_id=config.provider_id,
                max_completion_tokens=config.max_completion_tokens,
            )
            _write_json(target_dir / "continuation_manifest.json", {
                "mode": "continuation_only",
                "parent_manifest": str(preflight["source_dir"] / "run_manifest.json"),
                "parent_manifest_sha256": preflight["manifest_sha256"],
                "run_id": GOVERNED_RUN_ID,
                "case_id": CASE_ID,
                "arm": "governed_workflow",
                "configuration": config.__dict__,
                "operation_models": models,
                "provider_id": config.provider_id,
                "provider_host": provider_host,
                "projected_cumulative_estimate_cny": preflight["projected_reserved_cny"],
                "estimate_limit_cny": preflight["limit_cny"],
                "is_account_hard_cap": False,
                "database_name": db_name,
            })
            reservation = cost_ledger.reserve(GOVERNED_RUN_ID)
            ledger.append(
                GOVERNED_RUN_ID, "cost_reserved", reserved_cny=reservation,
                cumulative_reserved_cny=cost_ledger.reserved_cny,
                continuation_of=str(preflight["source_dir"]),
            )
            ledger.append(GOVERNED_RUN_ID, "started", case_id=CASE_ID, arm="governed_workflow", continuation=True)
            started = True
            start = time.perf_counter()
            response = workflow.run_chat_workflow(
                QUESTION, tenant_context=context, memory_enabled=False,
                trace_id=GOVERNED_RUN_ID,
                _critic_llm_review_mode=CriticLLMReviewMode.FULL,
                _critic_llm_thinking_mode=CriticLLMThinkingMode.NON_THINKING_EXPERIMENT,
                _llm_call_config=call_config,
            )
            elapsed = round((time.perf_counter() - start) * 1000, 3)
            trace_path = trace_dir / f"{GOVERNED_RUN_ID}.json"
            if not trace_path.is_file():
                raise RuntimeError("continuation_workflow_trace_missing")
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            state = trace.get("state_snapshot") or {}
            run_events = [row for row in ledger.events if row.get("run_id") == GOVERNED_RUN_ID]
            usage = token_usage_snapshot(trace, run_events, run_id=GOVERNED_RUN_ID)
            accounting = cost_ledger.record_usage(GOVERNED_RUN_ID, usage)
            _reconcile_cost_accounting(usage, accounting, policy, expected_reservation_cny=reservation)
            score = _score(state, response, "governed_workflow")
            measurements = {
                "elapsed_ms": elapsed,
                "logical_model_calls": len(state.get("llm_calls") or []),
                "sdk_send_attempts": sum(row.get("event") == "sdk_send_started" for row in run_events),
                "sdk_failed_sends": sum(row.get("event") == "sdk_send_failed" for row in run_events),
                "tool_call_count": len(state.get("tool_calls") or []),
                "sql_attempts": sum(row.get("event") == "sql_attempt_started" for row in run_events),
                "sql_successes": sum(row.get("event") == "sql_attempt_succeeded" for row in run_events),
                "sql_failures": sum(row.get("event") == "sql_attempt_failed" for row in run_events),
                "token_usage": usage,
                "estimated_cost": accounting,
                "cost_reservation": reservation,
                "duplicate_tool_signatures": _duplicate_tool_signatures(state),
            }
            stop_reason = _stop_reason_after_run(state, usage, measurements["sdk_failed_sends"])
            if accounting.get("reconciliation_matches") is False:
                stop_reason = "metering_error"
            record = {
                "run_id": GOVERNED_RUN_ID, "case_id": CASE_ID, "arm": "governed_workflow",
                "lifecycle_status": "finalizing", "response": response,
                "final_answer": response.get("answer"),
                "presentation": response.get("presentation") or state.get("analysis_presentation"),
                "trace": trace, "score": score, "measurements": measurements,
                "stop_reason": (state.get("investigation_state") or {}).get("stop_reason"),
                "critic": {
                    "verdict": state.get("critic_verdict"),
                    "decisions": state.get("critic_decisions") or [],
                    "optional_review_calls": [call for call in (state.get("llm_calls") or []) if call.get("node") == "critic"],
                },
            }
            _write_json(target_dir / "score.json", score)
            record_path = trace_dir / f"{GOVERNED_RUN_ID}.run_record.json"
            result = _finalize_run_record(
                ledger=ledger, run_id=GOVERNED_RUN_ID, record_path=record_path,
                record=record, usage=usage, task_pass=bool(score.get("task_pass")),
                provider_sends=measurements["sdk_send_attempts"], stop_reason=stop_reason,
            )
            result.update({
                "score": score,
                "measurements": measurements,
                "response": response,
                "trace_path": str(trace_path),
                "run_record_path": str(record_path),
            })
            _write_json(target_dir / "sql_statements.json", sql_rows)
    except BaseException as exc:
        infrastructure_error = True
        stop_reason = type(exc).__name__
        failure = {
            "run_id": GOVERNED_RUN_ID, "case_id": CASE_ID, "arm": "governed_workflow",
            "lifecycle_status": "failed" if started else "not_started",
            "error_type": type(exc).__name__,
            "error_summary": "Continuation failed; exception text withheld from the run record.",
            "retry_forbidden": True,
        }
        _write_json(target_dir / "failure_record.json", failure)
        if started and ledger.latest().get(GOVERNED_RUN_ID, {}).get("event") not in {
            "completed", "failed", "interrupted_before_send", "interrupted_result_unknown",
        }:
            ledger.append(GOVERNED_RUN_ID, "failed", error_type=type(exc).__name__, retry_forbidden=True)
        result = failure
    finally:
        if socket_originals:
            socket.socket.connect, socket.socket.connect_ex, socket.socketpair = socket_originals[:3]
        if engine is not None:
            if sql_listener is not None:
                try:
                    event.remove(engine, "before_cursor_execute", sql_listener)
                except Exception:
                    pass
            if graph_runtime is not None:
                if graph_runtime._CHECKPOINTER is not None:
                    saver = graph_runtime._CHECKPOINTER
                    connection = getattr(saver, "conn", None)
                    if connection is not None:
                        connection.close()
                if graph_runtime._CHECKPOINTER_CONTEXT is not None:
                    try:
                        graph_runtime._CHECKPOINTER_CONTEXT.__exit__(None, None, None)
                    except Exception:
                        pass
                if previous_runtime_settings is not None:
                    graph_runtime._CHECKPOINTER = previous_checkpoint
                    graph_runtime._CHECKPOINTER_CONTEXT = previous_checkpoint_context
                    graph_runtime.get_settings = previous_runtime_settings
            engine.dispose()
        if admin_engine is not None:
            with admin_engine.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
            admin_engine.dispose()
    _write_json(target_dir / "continuation_summary.json", {
        "run_id": GOVERNED_RUN_ID,
        "case_id": CASE_ID,
        "arm": "governed_workflow",
        "lifecycle_status": result.get("lifecycle_status"),
        "task_pass": (result.get("score") or {}).get("task_pass"),
        "stop_reason": stop_reason,
        "infrastructure_error": infrastructure_error,
        "cumulative_estimate_reserved_cny": cost_ledger.reserved_cny,
        "cumulative_estimate_limit_cny": preflight["limit_cny"],
        "is_account_hard_cap": False,
        "run_record": result,
    })
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized-dev-pair", action="store_true")
    parser.add_argument("--preflight-governed-continuation", action="store_true")
    parser.add_argument("--execute-authorized-continuation", action="store_true")
    parser.add_argument("--continue-governed-run-id")
    args = parser.parse_args()
    if args.preflight_governed_continuation or args.execute_authorized_continuation or args.continue_governed_run_id:
        return _main_governed_continuation(args)
    if not args.execute_authorized_dev_pair:
        print("Refusing to run without --execute-authorized-dev-pair")
        return 2
    admin_url_text = os.environ.get("ISOLATED_TEST_POSTGRES_ADMIN_URL", "")
    if not admin_url_text:
        print("Blocked: ISOLATED_TEST_POSTGRES_ADMIN_URL is not set; no run started.")
        return 2

    from app.core.config import get_settings
    from app.graph import runtime as graph_runtime
    from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.graph import workflow
    from app.llm.routing import model_for

    raw_settings = get_settings()
    endpoint = urlsplit(raw_settings.deepseek_base_url)
    provider_host = endpoint.hostname or ""
    if not provider_host or endpoint.port not in (None, 443):
        print("Blocked: configured primary Provider endpoint is not a standard HTTPS host; no run started.")
        return 2
    models = {operation: model_for(operation)[0] for operation in ("classify", "extract", "chat")}
    if set(models.values()) != {"qwen3.8-flash"}:
        print("Blocked: configured operation models do not all resolve to qwen3.8-flash; no run started.")
        return 2
    if not raw_settings.deepseek_api_key:
        print("Blocked: primary Provider credential is unavailable; no run started.")
        return 2

    config = ExperimentConfig(
        provider_id="primary", model="qwen3.8-flash", max_graph_steps=48,
        max_task_tokens=60_000, max_cost_cny_per_workflow=5.0,
        max_completion_tokens=4_096, max_sdk_sends_per_workflow=10,
        sdk_retries=0, application_attempts=1, fallback_enabled=False,
        max_tool_calls=8, max_latency_ms=240_000,
        model_timeout_seconds=60.0, max_revisions=1,
    )
    policy = CostPolicy(
        max_sdk_calls_per_workflow=10, input_token_estimate_cap_per_call=200_000,
        input_byte_factor=4, input_protocol_token_margin=16_384,
        max_completion_tokens=4_096, documented_output_variance_tokens=10,
        input_cny_per_million=1.094, output_cny_per_million=3.427,
        cumulative_estimate_limit_cny=5.0,
    )
    estimate = policy.estimate(2)
    if not estimate["within_estimate_limit"]:
        print("Blocked: two-run worst-case reservation exceeds the authorized estimated cap.")
        return 2
    if 2 * policy.max_per_run_cny() > 5.0:
        print("Blocked: cumulative reservation check failed; no run started.")
        return 2

    run_ids = {"basic_tool_loop": str(uuid4()), "governed_workflow": str(uuid4())}
    result_dir = RESULTS_PARENT / f"resume_dev_pair_order_dimension_20261015_{run_ids['basic_tool_loop'][:8]}"
    result_dir.mkdir(parents=True, exist_ok=False)
    _write_json(result_dir / "run_manifest.json", {
        "experiment_kind": "development B/G link validation; not an independent test set",
        "case_id": CASE_ID,
        "user_question": QUESTION,
        "run_ids": run_ids,
        "planned_runs": 2,
        "order": ["basic_tool_loop", "governed_workflow"],
        "oracle_for_external_scorer_only": ORACLE,
        "model_config": config.__dict__,
        "operation_models": models,
        "provider_id": "primary",
        "provider_host": provider_host,
        "cost_policy": policy.estimate(2),
        "is_account_hard_cap": False,
    })
    ledger = RunLedger(result_dir / "run_events.jsonl")
    cost_ledger = EstimatedCostLedger(policy, result_dir / "cost_reservations.jsonl")

    # Resolve the only permitted remote host before touching the database;
    # the guard then allows that host and the isolated loopback PostgreSQL.
    original_connect, original_connect_ex, original_socketpair, provider_ips = _network_guard(provider_host)
    # Prepare the fixed isolated Olist-shaped development fixture. It is
    # synthetic data already used by offline development, not blind data.
    try:
        admin_engine, engine, db_name = _create_seeded_db(admin_url_text)
    except BaseException:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.socketpair = original_socketpair
        raise
    sql_rows: list[dict] = []
    from app.core.execution_context import active_state

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _many):
        state = active_state() or {}
        run_id = str(state.get("trace_id") or "")
        if run_id and statement.lstrip().lower().startswith("select"):
            sql_rows.append({"run_id": run_id, "statement": statement})

    event.listen(engine, "before_cursor_execute", capture_sql)
    context = TenantContext(
        tenant_id="platform", user_id="resume-dev-pair-order-dimension",
        role="admin", scope_type="platform",
    )
    previous_checkpoint = graph_runtime._CHECKPOINTER
    previous_checkpoint_context = graph_runtime._CHECKPOINTER_CONTEXT
    previous_runtime_settings = graph_runtime.get_settings
    checkpoint_path = result_dir / "workflow_checkpoints.sqlite"
    trace_dir = result_dir / "traces"
    settings_for_runtime = raw_settings.model_copy(update={
        "trace_backend": "file", "trace_dir": str(trace_dir),
        "environment": "development", "checkpoint_backend": "sqlite",
        "checkpoint_path": str(checkpoint_path),
        "max_graph_steps": config.max_graph_steps,
        "max_tokens": config.max_task_tokens,
        "max_cost": config.max_cost_cny_per_workflow,
        "max_tool_calls": config.max_tool_calls,
        "max_latency_ms": config.max_latency_ms,
        "max_revisions": config.max_revisions,
        "llm_timeout_seconds": config.model_timeout_seconds,
        "llm_resilience_attempts": config.application_attempts,
    })
    graph_runtime._CHECKPOINTER = None
    graph_runtime._CHECKPOINTER_CONTEXT = None
    graph_runtime.get_settings = lambda: settings_for_runtime
    per_run: dict[str, dict] = {}
    infrastructure_error = False
    pair_stop_reason: str | None = None
    try:
        with bind_isolated_experiment_runtime(engine=engine, trace_dir=trace_dir, ledger=ledger, config=config) as bound_settings:
            call_config = LLMCallConfig.provider_ab_experiment(
                bound_settings, provider_id=config.provider_id,
                max_completion_tokens=config.max_completion_tokens,
            )
            _write_json(result_dir / "effective_config.json", {
                "effective_call_config": call_config.to_state(),
                "experiment_config": config.__dict__,
                "operation_models": models,
                "provider_host": provider_host,
                "provider_resolved_ips": provider_ips,
                "database_name": db_name,
                "database_host": "127.0.0.1",
            })

            ordered_arms = ("basic_tool_loop", "governed_workflow")
            for arm in ordered_arms:
                run_id = run_ids[arm]
                reservation = cost_ledger.reserve(run_id)
                ledger.append(run_id, "cost_reserved", reserved_cny=reservation,
                              cumulative_reserved_cny=cost_ledger.reserved_cny)
                ledger.append(run_id, "started", case_id=CASE_ID, arm=arm)
                start = time.perf_counter()
                try:
                    if arm == "basic_tool_loop":
                        state = run_bare_tool_loop(
                            QUESTION, tenant_context=context, run_id=run_id,
                            llm_call_config=call_config,
                        )
                        response = {"answer": (state.get("final") or {}).get("answer", ""),
                                    "presentation": (state.get("final") or {}).get("presentation"),
                                    "error": bool(state.get("errors")), "errors": state.get("errors") or []}
                        from app.harness.storage import write_trace
                        write_trace(run_id, state)
                    else:
                        response = workflow.run_chat_workflow(
                            QUESTION, tenant_context=context, memory_enabled=False, trace_id=run_id,
                            _critic_llm_review_mode=CriticLLMReviewMode.FULL,
                            _critic_llm_thinking_mode=CriticLLMThinkingMode.NON_THINKING_EXPERIMENT,
                            _llm_call_config=call_config,
                        )
                        trace_path = trace_dir / f"{run_id}.json"
                        if not trace_path.is_file():
                            raise RuntimeError("workflow_trace_missing")
                        trace = json.loads(trace_path.read_text(encoding="utf-8"))
                        state = trace.get("state_snapshot") or {}
                    elapsed = round((time.perf_counter() - start) * 1000, 3)
                    trace_path = trace_dir / f"{run_id}.json"
                    trace = json.loads(trace_path.read_text(encoding="utf-8"))
                    state = trace.get("state_snapshot") or state
                    run_events = [row for row in ledger.events if row.get("run_id") == run_id]
                    usage = token_usage_snapshot(trace, run_events, run_id=run_id)
                    accounting = cost_ledger.record_usage(run_id, usage)
                    _reconcile_cost_accounting(
                        usage, accounting, policy,
                        expected_reservation_cny=reservation,
                    )
                    score = _score(state, response, arm)
                    measurements = {
                        "elapsed_ms": elapsed,
                        "logical_model_calls": len(state.get("llm_calls") or []),
                        "sdk_send_attempts": sum(row.get("event") == "sdk_send_started" for row in run_events),
                        "sdk_failed_sends": sum(row.get("event") == "sdk_send_failed" for row in run_events),
                        "tool_call_count": len(state.get("tool_calls") or []),
                        "sql_attempts": sum(row.get("event") == "sql_attempt_started" for row in run_events),
                        "sql_successes": sum(row.get("event") == "sql_attempt_succeeded" for row in run_events),
                        "sql_failures": sum(row.get("event") == "sql_attempt_failed" for row in run_events),
                        "token_usage": usage,
                        "estimated_cost": accounting,
                        "cost_reservation": reservation,
                        "duplicate_tool_signatures": _duplicate_tool_signatures(state),
                    }
                    stop_reason = _stop_reason_after_run(
                        state, usage, measurements["sdk_failed_sends"],
                    )
                    record = {
                        "run_id": run_id, "case_id": CASE_ID, "arm": arm,
                        "lifecycle_status": "finalizing", "response": response,
                        "final_answer": response.get("answer"),
                        "presentation": response.get("presentation") or state.get("analysis_presentation"),
                        "trace": trace, "score": score,
                        "measurements": measurements,
                        "stop_reason": (state.get("investigation_state") or {}).get("stop_reason"),
                        "critic": {
                            "verdict": state.get("critic_verdict"),
                            "decisions": state.get("critic_decisions") or [],
                            "optional_review_calls": [
                                call for call in (state.get("llm_calls") or [])
                                if call.get("node") == "critic"
                            ],
                        },
                    }
                    record_path = trace_dir / f"{run_id}.run_record.json"
                    _write_json(result_dir / f"{run_id}.score.json", score)
                    stop_reason = _stop_reason_after_run(
                        state, usage, measurements["sdk_failed_sends"],
                    )
                    if accounting.get("reconciliation_matches") is False:
                        stop_reason = "metering_error"
                    per_run[arm] = _finalize_run_record(
                        ledger=ledger, run_id=run_id, record_path=record_path,
                        record=record, usage=usage, task_pass=bool(score.get("task_pass")),
                        provider_sends=measurements["sdk_send_attempts"], stop_reason=stop_reason,
                    )
                    if stop_reason is not None:
                        pair_stop_reason = stop_reason
                    if stop_reason == "metering_error":
                        infrastructure_error = True
                        break
                    if stop_reason is not None:
                        if stop_reason in {"provider_send_failure", "tool_infrastructure_error"}:
                            infrastructure_error = True
                        break
                except BaseException as exc:
                    elapsed = round((time.perf_counter() - start) * 1000, 3)
                    sent = sum(row.get("event") == "sdk_send_started" for row in ledger.events if row.get("run_id") == run_id)
                    failure = {
                        "run_id": run_id, "case_id": CASE_ID, "arm": arm,
                        "lifecycle_status": "failed", "error_type": type(exc).__name__,
                        "error_summary": "Execution failed; exception text withheld from the run record.",
                        "sdk_sends_started": sent, "elapsed_ms": elapsed,
                        "retry_forbidden": True,
                    }
                    _write_json(trace_dir / f"{run_id}.failure_record.json", failure)
                    ledger.append(run_id, "failed", error_type=type(exc).__name__, retry_forbidden=True)
                    pair_stop_reason = type(exc).__name__
                    existing_record_path = trace_dir / f"{run_id}.run_record.json"
                    if existing_record_path.is_file():
                        try:
                            existing_record = json.loads(existing_record_path.read_text(encoding="utf-8"))
                            _mark_record_lifecycle(
                                existing_record_path, existing_record, "failed",
                                lifecycle_error=type(exc).__name__,
                            )
                        except Exception:
                            # Keep the append-only failure record authoritative
                            # if the partially written summary itself is unreadable.
                            pass
                    infrastructure_error = True
                    break
                if infrastructure_error:
                    break
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.socketpair = original_socketpair
        event.remove(engine, "before_cursor_execute", capture_sql)
        if graph_runtime._CHECKPOINTER is not None:
            saver = graph_runtime._CHECKPOINTER
            connection = getattr(saver, "conn", None)
            if connection is not None:
                connection.close()
        if graph_runtime._CHECKPOINTER_CONTEXT is not None:
            try:
                graph_runtime._CHECKPOINTER_CONTEXT.__exit__(None, None, None)
            except Exception:
                pass
        graph_runtime._CHECKPOINTER = previous_checkpoint
        graph_runtime._CHECKPOINTER_CONTEXT = previous_checkpoint_context
        graph_runtime.get_settings = previous_runtime_settings
        engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        admin_engine.dispose()
    _write_json(result_dir / "sql_statements.json", sql_rows)
    lifecycle = ledger.latest()
    _write_json(result_dir / "summary.json", {
        "case_id": CASE_ID,
        "planned_units": 2,
        "started_units": sum(row.get("event") not in (None, "cost_reserved") for row in lifecycle.values()),
        "runs": per_run,
        "cumulative_estimate_reserved_cny": cost_ledger.reserved_cny,
        "cumulative_estimate_limit_cny": 5.0,
        "is_account_hard_cap": False,
        "infrastructure_error": infrastructure_error,
        "pair_stop_reason": pair_stop_reason,
        "failed_run_count": sum(row.get("lifecycle_status") == "failed" for row in per_run.values()),
        "formal_batch_started": False,
    })
    print(json.dumps({
        "result_dir": str(result_dir),
        "runs_started": len(per_run),
        "cumulative_estimate_reserved_cny": cost_ledger.reserved_cny,
        "estimate_limit_cny": 5.0,
        "is_account_hard_cap": False,
        "infrastructure_error": infrastructure_error,
        "pair_stop_reason": pair_stop_reason,
    }, ensure_ascii=False))
    return 1 if infrastructure_error or pair_stop_reason is not None else 0


def _main_governed_continuation(args: argparse.Namespace) -> int:
    if args.continue_governed_run_id != GOVERNED_RUN_ID:
        print("Refusing continuation: only the original not-started governed run ID is permitted.")
        return 2
    try:
        preflight = _validate_governed_continuation(ORIGINAL_PAIR_DIR)
    except Exception as exc:
        print(f"Blocked before database startup: {type(exc).__name__}:{exc}")
        return 2
    if args.preflight_governed_continuation and not args.execute_authorized_continuation:
        print(json.dumps({
            "status": "preflight_passed_no_run_started",
            "run_id": GOVERNED_RUN_ID,
            "case_id": CASE_ID,
            "arm": "governed_workflow",
            "existing_reserved_cny": preflight["existing_reserved_cny"],
            "new_reservation_cny": preflight["new_reservation_cny"],
            "projected_reserved_cny": preflight["projected_reserved_cny"],
            "estimate_limit_cny": preflight["limit_cny"],
            "is_account_hard_cap": False,
            "database_started": False,
            "provider_sends": 0,
        }, ensure_ascii=False))
        return 0
    if not args.execute_authorized_continuation:
        print("Refusing continuation without --execute-authorized-continuation")
        return 2
    admin_url_text = os.environ.get("ISOLATED_TEST_POSTGRES_ADMIN_URL", "")
    if not admin_url_text:
        print("Blocked: ISOLATED_TEST_POSTGRES_ADMIN_URL is not set; governed unit not started.")
        return 2
    from app.core.config import get_settings
    from app.llm.routing import model_for

    raw_settings = get_settings()
    endpoint = urlsplit(raw_settings.deepseek_base_url)
    provider_host = endpoint.hostname or ""
    if not provider_host or endpoint.port not in (None, 443):
        print("Blocked: configured primary Provider endpoint is not a standard HTTPS host; no unit started.")
        return 2
    models = {operation: model_for(operation)[0] for operation in ("classify", "extract", "chat")}
    if set(models.values()) != {preflight["config"].model}:
        print("Blocked: operation models differ from the original manifest; no unit started.")
        return 2
    if not raw_settings.deepseek_api_key:
        print("Blocked: primary Provider credential is unavailable; no unit started.")
        return 2
    if provider_host.lower() != str(preflight["manifest"].get("provider_host") or "").lower():
        print("Blocked: configured Provider endpoint differs from the original manifest; no unit started.")
        return 2
    print(json.dumps({
        "continuation_run_id": GOVERNED_RUN_ID,
        "case_id": CASE_ID,
        "arm": "governed_workflow",
        "existing_reserved_cny": preflight["existing_reserved_cny"],
        "new_reservation_cny": preflight["new_reservation_cny"],
        "projected_reserved_cny": preflight["projected_reserved_cny"],
        "estimate_limit_cny": preflight["limit_cny"],
        "is_account_hard_cap": False,
        "status": "preflight_passed_starting_single_continuation_unit",
    }, ensure_ascii=False))
    try:
        result = _run_governed_continuation(
            source_dir=ORIGINAL_PAIR_DIR, admin_url_text=admin_url_text,
            raw_settings=raw_settings, provider_host=provider_host, models=models,
        )
    except Exception as exc:
        print(f"Continuation infrastructure error: {type(exc).__name__}")
        return 1
    print(json.dumps({
        "run_id": GOVERNED_RUN_ID,
        "lifecycle_status": result.get("lifecycle_status"),
        "task_pass": (result.get("score") or {}).get("task_pass"),
        "error": result.get("error_type"),
    }, ensure_ascii=False))
    return 1 if result.get("lifecycle_status") != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(_main())
