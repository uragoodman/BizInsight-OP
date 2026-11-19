#!/usr/bin/env python3
"""Evaluate P2 dimension-role classification on the frozen 49-case dataset."""

from __future__ import annotations

import argparse
import math
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"
DEFAULT_OUT = ROOT / "data" / "quality" / "dimension-role-p2-latest.json"


class DimensionRoleComparisonAdapter(Protocol):
    """Caller-supplied, explicitly approved B/C predictor; never constructed here.

    Implementations receive verified mention spans, closed labels and optional
    deterministic proposals/evidence codes. The full source question and scope
    values are withheld. This evaluator never creates a provider client; shared
    trusted authorization preflight runs before any adapter call.
    """

    is_real_adapter: bool

    def predict_llm_only(self, request: dict[str, Any]) -> list[dict[str, Any]]: ...

    def predict_verifier(self, request: dict[str, Any]) -> list[dict[str, Any]]: ...


AuthorizationPreflight = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool]
# Must re-check authorization and deterministic support on the proposed role set.
HybridSupportPreflight = Callable[[dict[str, Any], list[dict[str, Any]]], bool]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if path.name == "dimension_role_cases_v1.jsonl":
        try:
            from jsonschema import Draft202012Validator

            schema = json.loads((ROOT / "evaluation" / "metrics_schema_v1.json").read_text(encoding="utf-8"))
            validator = Draft202012Validator(schema["definitions"]["dimension_role_case"])
            errors = [
                f"{row.get('case_id')}: {error.message}"
                for row in rows for error in validator.iter_errors(row)
            ]
            if errors:
                raise ValueError("invalid dimension-role case dataset: " + "; ".join(errors[:10]))
        except ImportError:
            required = {
                "case_id", "input", "mentions", "roles", "expected_candidates",
                "expected_support", "expected_clarification", "expected_contract", "expected_terminal",
            }
            missing = [row.get("case_id", "?") for row in rows if required - set(row)]
            if missing:
                raise ValueError(f"dimension-role cases missing P3 fields: {missing}")
    return rows


def _state_for(case: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "user_message": case["input"],
        "trace_id": f"p2-{case['case_id']}",
        "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
    }
    if case.get("scope"):
        state["controlled_analysis_scope"] = case["scope"]
        state["use_current_explorer_context"] = True
        state["current_query_scope_fingerprint"] = "p2-scope-fingerprint"
    previous = (case.get("context") or {}).get("previous_turn_context")
    if previous:
        state["previous_turn_context"] = previous
    return state


def _roles(compiled: dict[str, Any]) -> dict[str, str]:
    from app.context.dimension_intent import RoleAssignment, aggregate_roles

    assignments = [
        RoleAssignment(
            dimension=item["dimension"],
            start=item["start"],
            end=item["end"],
            role=item["role"],
            evidence=tuple(item.get("evidence") or ()),
            conflict=bool(item.get("conflict")),
        )
        for item in compiled.get("dimension_roles") or []
    ]
    return aggregate_roles(assignments)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 3)


def _gold(case: dict[str, Any], key: str, fallback: str | None = None) -> Any:
    if key in case:
        return case[key]
    expected = case.get("expected") or {}
    return expected.get(fallback or key)


def _rules_only_quality(details: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    mention_tp = mention_fp = mention_fn = 0
    role_tp = role_fp = role_fn = 0
    high_confidence_fp = 0
    unresolved = auto_count = correct_auto = clarification_tp = clarification_pred = 0
    wrong_contract = wrong_execution = 0
    for case, detail in zip(cases, details):
        expected_mentions = ((case.get("expected") or {}).get("mentions") or [])
        expected_dimensions = Counter(
            str(item.get("canonical") or "")
            for item in expected_mentions
            if item.get("note") != "metric" and item.get("canonical")
        )
        compiled_roles = detail.get("roles") or {}
        # Dataset v1 gold mention strings are aliases; compare canonical IDs
        # and text to count detection independently from role correctness.
        actual_assignments = (detail.get("compiled_dimension_roles") or [])
        actual_mentions = Counter(
            str(item.get("dimension") or "")
            for item in actual_assignments
            if item.get("dimension")
        )
        if not actual_mentions:
            actual_mentions = Counter(
                str(item.get("dimension") or "")
                for item in (detail.get("dimension_roles") or [])
                if item.get("dimension")
            )
        overlap = expected_dimensions & actual_mentions
        mention_tp += sum(overlap.values())
        mention_fp += sum((actual_mentions - expected_dimensions).values())
        mention_fn += sum((expected_dimensions - actual_mentions).values())

        expected_roles = _gold(case, "roles") or {}
        expected_roles = {str(key): str(value) for key, value in expected_roles.items() if value != "unresolved"}
        actual_roles = {str(key): str(value) for key, value in compiled_roles.items()}
        for dimension, expected_role in expected_roles.items():
            actual_role = actual_roles.get(dimension)
            if actual_role == expected_role:
                role_tp += 1
            else:
                role_fn += 1
                if actual_role is not None:
                    role_fp += 1
        role_fp += len(set(actual_roles) - set(expected_roles))
        expected_role_by_dim = expected_roles
        high_confidence_fp += sum(
            1 for item in actual_assignments
            if item.get("confidence_class") == "closed"
            and expected_role_by_dim.get(str(item.get("dimension"))) != item.get("role")
        )

        expected_terminal = _gold(case, "expected_terminal", "terminal")
        actual_terminal = str(detail.get("actual_terminal") or "auto")
        actual_terminal = "clarify" if actual_terminal == "clarify" else "auto"
        unresolved += sum(1 for role in actual_roles.values() if role == "unresolved")
        if actual_terminal == "clarify":
            clarification_pred += 1
            clarification_tp += int(expected_terminal == "clarify")
        else:
            auto_count += 1
        expected_candidates = _gold(case, "expected_candidates")
        if expected_candidates is None:
            expected_candidates = [dim for dim, role in expected_roles.items() if role == "grouping"]
        contract_correct = (
            set(detail.get("actual_candidates") or []) == set(expected_candidates)
            and detail.get("terminal_match") is True
        )
        wrong_contract += int(not contract_correct)
        if actual_terminal == "auto":
            correct_auto += int(contract_correct and expected_terminal == "auto")
            wrong_execution += int(not contract_correct or expected_terminal == "clarify")

    f1 = (2 * mention_tp / (2 * mention_tp + mention_fp + mention_fn)) if (2 * mention_tp + mention_fp + mention_fn) else 0.0
    role_f1 = (2 * role_tp / (2 * role_tp + role_fp + role_fn)) if (2 * role_tp + role_fp + role_fn) else 0.0
    count = len(cases)
    return {
        "mention_precision": round(mention_tp / (mention_tp + mention_fp), 4) if mention_tp + mention_fp else None,
        "mention_recall": round(mention_tp / (mention_tp + mention_fn), 4) if mention_tp + mention_fn else None,
        "mention_f1": round(f1, 4),
        "role_precision": round(role_tp / (role_tp + role_fp), 4) if role_tp + role_fp else None,
        "role_recall": round(role_tp / (role_tp + role_fn), 4) if role_tp + role_fn else None,
        "role_f1": round(role_f1, 4),
        "false_positive_roles": role_fp,
        "false_negative_roles": role_fn,
        "high_confidence_fp": high_confidence_fp,
        "unresolved_rate": round(unresolved / max(1, sum(len(item.get("roles") or {}) for item in details)), 4),
        "clarification_precision": round(clarification_tp / clarification_pred, 4) if clarification_pred else None,
        "coverage": round(auto_count / count, 4) if count else None,
        "precision_auto": round(correct_auto / auto_count, 4) if auto_count else None,
        "wrong_contract_rate": round(wrong_contract / count, 4) if count else None,
        "wrong_execution_rate": round(wrong_execution / auto_count, 4) if auto_count else None,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "verifier_disagreement_rate": None,
        "wrong_execution_note": "Proxy from adjudicated expected terminal/contract; no SQL was executed by this offline evaluator.",
    }


_P3B_METRIC_KEYS = (
    "mention_precision", "mention_recall", "mention_f1",
    "role_precision", "role_recall", "role_f1",
    "false_positive_roles", "false_negative_roles",
)


def _p3b_metric_view(metrics: dict[str, Any], *, provenance: str) -> dict[str, Any]:
    """Expose only mention/role quality in the P3b A/B/C comparison contract."""
    return {
        **{key: metrics.get(key) for key in _P3B_METRIC_KEYS},
        "coverage": None,
        "wrong_contract_rate": None,
        "wrong_execution_rate": None,
        "metrics_scope": "role_and_mention_only",
        "result_provenance": provenance,
    }


def _not_run_result(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        **{key: None for key in _P3B_METRIC_KEYS},
        "coverage": None,
        "wrong_contract_rate": None,
        "wrong_execution_rate": None,
        "metrics_scope": "role_and_mention_only",
        "result_provenance": "not_run",
    }


def _compiled_role_rows(case: dict[str, Any], detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind deterministic role assignments to annotated L1 spans for group A.

    Exact ID/span/canonical matches reuse the gold mention ID. A compiled span
    absent from the annotation receives a deterministic synthetic ID so the
    shared role scorer can count it as a mention false positive. Missing gold
    spans are intentionally absent and become false negatives.
    """
    source = str(case.get("input") or "")
    gold_by_span: dict[tuple[int, int, str], str] = {}
    for item in case.get("mentions") or []:
        if item.get("note") == "metric":
            continue
        key = (int(item.get("start", -1)), int(item.get("end", -1)), str(item.get("canonical") or ""))
        if key[0] >= 0 and key[1] > key[0] and key[2]:
            gold_by_span[key] = str(item.get("id") or "")

    rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, assignment in enumerate(detail.get("compiled_dimension_roles") or []):
        start, end = int(assignment.get("start", -1)), int(assignment.get("end", -1))
        canonical = str(assignment.get("dimension") or "")
        role = str(assignment.get("role") or "unresolved")
        if start < 0 or end <= start or end > len(source) or not canonical:
            continue
        key = (start, end, canonical)
        mention_id = gold_by_span.get(key)
        if not mention_id or mention_id in used_ids:
            mention_id = f"compiled-fp-{index + 1}"
        used_ids.add(mention_id)
        rows.append({
            "id": mention_id,
            "start": start,
            "end": end,
            "canonical": canonical,
            "role": role,
        })
    return rows


def run(
    cases_path: Path,
    out_path: Path,
    *,
    comparison: str = "a",
    enable_llm_groups: bool = False,
    approval_marker: str | None = None,
    raw_question_opt_in: bool = False,
    comparison_adapter: DimensionRoleComparisonAdapter | None = None,
    authorization_preflight: AuthorizationPreflight | None = None,
    hybrid_support_preflight: HybridSupportPreflight | None = None,
) -> dict[str, Any]:
    if comparison not in {"a", "b", "c", "all"}:
        raise ValueError("comparison must be one of: a, b, c, all")
    from app.context.compiler import compile_turn_context

    rows = _load_cases(cases_path)
    details: list[dict[str, Any]] = []
    case_by_id: dict[str, dict[str, Any]] = {}
    compiled_by_id: dict[str, dict[str, Any]] = {}
    state_by_id: dict[str, dict[str, Any]] = {}
    role_hits = role_total = 0
    candidate_hits = 0
    terminal_hits = reason_hits = 0
    for case in rows:
        state = _state_for(case)
        compiled = compile_turn_context(state)["context_compilation"]
        case_id = str(case["case_id"])
        case_by_id[case_id] = case
        compiled_by_id[case_id] = compiled
        state_by_id[case_id] = state
        roles = _roles(compiled)
        expected_roles = case["expected"].get("roles") or {}
        expected_roles = {
            dim: role for dim, role in expected_roles.items() if dim != "unresolved"
        }
        role_mismatches = {
            dim: {"expected": role, "actual": roles.get(dim)}
            for dim, role in expected_roles.items()
            if roles.get(dim) != role
        }
        role_total += len(expected_roles)
        role_hits += len(expected_roles) - len(role_mismatches)

        expected_candidates = {
            dim for dim, role in expected_roles.items() if role == "grouping"
        }
        actual_candidates = {dim for dim, role in roles.items() if role == "grouping"}
        candidate_match = expected_candidates == actual_candidates
        candidate_hits += int(candidate_match)

        actual_terminal = "clarify" if compiled.get("needs_clarification") else "auto"
        expected_terminal = case["expected"].get("terminal")
        terminal_match = actual_terminal == expected_terminal
        terminal_hits += int(terminal_match)

        expected_reason = case["expected"].get("reason")
        actual_reason = compiled.get("clarification_reason") or None
        reason_match = actual_reason == expected_reason
        reason_hits += int(reason_match)

        details.append(
            {
                "case_id": case["case_id"],
                "tier": case["tier"],
                "category": case["category"],
                "roles": roles,
                "compiled_dimension_roles": [
                    {
                        **item,
                        "raw_text": case["input"][int(item.get("start", 0)):int(item.get("end", 0))],
                    }
                    for item in compiled.get("dimension_roles") or []
                ],
                "role_mismatches": role_mismatches,
                "candidate_match": candidate_match,
                "expected_candidates": sorted(expected_candidates),
                "actual_candidates": sorted(actual_candidates),
                "expected_terminal": expected_terminal,
                "actual_terminal": actual_terminal,
                "terminal_match": terminal_match,
                "expected_reason": expected_reason,
                "actual_reason": actual_reason,
                "reason_match": reason_match,
                "contract_hash": compiled.get("request_contract_hash"),
            }
        )

    total = len(rows)
    summary = {
        "version": "p2-1",
        "dataset": cases_path.name,
        "total": total,
        "role_accuracy": round(role_hits / role_total, 4) if role_total else None,
        "role_hits": role_hits,
        "role_total": role_total,
        "candidate_accuracy": round(candidate_hits / total, 4) if total else None,
        "candidate_hits": candidate_hits,
        "terminal_accuracy": round(terminal_hits / total, 4) if total else None,
        "terminal_hits": terminal_hits,
        "reason_accuracy": round(reason_hits / total, 4) if total else None,
        "reason_hits": reason_hits,
    }
    # A is computed offline against the deterministic compiler. B/C need an
    # explicit comparison selector, CLI opt-in, privacy opt-in, approval marker,
    # caller-supplied real adapter, and trusted deterministic preflight. A model
    # credential alone never enables evaluation or sends the original question.
    gates = {
        "comparison_requested": comparison in {"b", "c", "all"},
        "cli_enabled": bool(enable_llm_groups),
        "raw_question_opt_in": bool(raw_question_opt_in),
        "human_approval_marker_present": bool(approval_marker and approval_marker.strip()),
    }
    gates_satisfied = all(gates.values())
    comparison_wanted = comparison in {"b", "c", "all"}
    adapter_ready = (
        comparison_adapter is not None
        and bool(getattr(comparison_adapter, "is_real_adapter", False))
        and (comparison == "c" or callable(getattr(comparison_adapter, "predict_llm_only", None)))
        and (comparison == "b" or callable(getattr(comparison_adapter, "predict_verifier", None)))
    )
    if not gates["comparison_requested"]:
        not_run_reason = "not_requested"
    elif not gates["cli_enabled"]:
        not_run_reason = "cli_opt_in_missing"
    elif not gates["raw_question_opt_in"]:
        not_run_reason = "external_question_opt_in_not_enabled"
    elif not gates["human_approval_marker_present"]:
        not_run_reason = "human_approval_marker_missing"
    elif not adapter_ready:
        not_run_reason = "comparison_adapter_not_implemented"
    elif authorization_preflight is None:
        not_run_reason = "trusted_preflight_missing"
    elif comparison in {"c", "all"} and hybrid_support_preflight is None:
        not_run_reason = "hybrid_support_preflight_missing"
    else:
        not_run_reason = "comparison_execution_not_requested"

    # P3b deliberately exposes only role/mention metrics. Keep the older P3
    # A-only route/contract proxy metrics in a separate compatibility field.
    legacy_a_metrics = _rules_only_quality(details, rows)
    from evaluation.dimension_role_comparison import score_roles

    compiled_predictions = {
        str(case["case_id"]): _compiled_role_rows(case, details[index])
        for index, case in enumerate(rows)
    }
    a_role_metrics = score_roles(
        rows,
        compiled_predictions,
        allow_unmatched_mentions=True,
    )
    a_metrics = _p3b_metric_view(a_role_metrics, provenance="deterministic_rules")
    b_metrics = _not_run_result("not_requested")
    c_metrics = _not_run_result("not_requested")
    p3b_comparison: dict[str, Any] | None = None

    if comparison_wanted and not_run_reason == "comparison_execution_not_requested":
        # External adapters receive only verified L1 spans, closed labels and
        # (for C) deterministic proposed roles/evidence. The full question,
        # gold labels, contracts, scopes, metrics and dates stay local.
        from app.context.semantic_verifier import (
            ALLOWED_ROLES,
            coarse_evidence_codes,
            risk_tier_for_compilation,
            trusted_authorization_preflight,
        )
        from evaluation.dimension_role_comparison import compare as compare_role_predictions

        mentions_by_case: dict[str, list[dict[str, Any]]] = {}
        proposed_by_case: dict[str, list[dict[str, Any]]] = {}
        risk_by_id: dict[str, str] = {}
        blocked_by_id: dict[str, bool] = {}
        for row in rows:
            case_id = str(row["case_id"])
            compiled = compiled_by_id[case_id]
            state = state_by_id[case_id]
            source = str(row["input"])
            mentions: list[dict[str, Any]] = []
            proposed: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            span_roles = {
                (int(item.get("start", -1)), int(item.get("end", -1)), str(item.get("dimension") or "")): item
                for item in compiled.get("dimension_roles") or []
            }
            spans_valid = True
            for mention in row.get("mentions") or []:
                start, end = int(mention["start"]), int(mention["end"])
                raw_text = str(mention.get("raw_text") or "")
                canonical = str(mention.get("canonical") or "")
                item_id = str(mention.get("id") or "")
                if (
                    not item_id or item_id in seen_ids or not canonical or not raw_text
                    or start < 0 or end <= start or end > len(source)
                    or source[start:end] != raw_text
                ):
                    spans_valid = False
                    break
                seen_ids.add(item_id)
                mentions.append({
                    "id": item_id,
                    "raw_text": raw_text,
                    "start": start,
                    "end": end,
                    "canonical": canonical,
                })
                compiled_role = span_roles.get((start, end, canonical), {})
                proposed.append({
                    "id": item_id,
                    "start": start,
                    "end": end,
                    "canonical": canonical,
                    "role": str(compiled_role.get("role") or "unresolved"),
                    "evidence": list(coarse_evidence_codes(compiled_role.get("evidence"))),
                })
            mentions_by_case[case_id] = mentions
            proposed_by_case[case_id] = proposed
            risk = risk_tier_for_compilation(compiled, state)
            risk_by_id[case_id] = risk
            risk_eligible = risk in {"open", "elevated", "conflict"}
            if spans_valid and risk_eligible:
                try:
                    trusted_allowed, _trusted_reason, _trusted_authz = trusted_authorization_preflight(
                        compiled, state,
                    )
                    allowed = trusted_allowed and bool(
                        authorization_preflight(row, compiled, state)
                    )
                except Exception:  # noqa: BLE001 - preflight failures fail closed
                    allowed = False
            else:
                allowed = False
            # This adapter guard is intentionally stricter than relying on the
            # comparison module's case filter: low/high cases never reach an
            # injected model callback even if that module later changes.
            blocked_by_id[case_id] = not spans_valid or not risk_eligible or not allowed

        def make_request(row: dict[str, Any], *, include_proposals: bool) -> dict[str, Any]:
            case_id = str(row["case_id"])
            request = {
                "mentions": mentions_by_case[case_id],
                "allowed_roles": list(ALLOWED_ROLES),
            }
            if include_proposals:
                request["proposed_roles"] = proposed_by_case[case_id]
            return request

        def predict_llm_only(row: dict[str, Any]) -> list[dict[str, Any]]:
            if blocked_by_id[str(row["case_id"])]:
                return []
            return comparison_adapter.predict_llm_only(make_request(row, include_proposals=False))

        def predict_verifier(row: dict[str, Any]) -> list[dict[str, Any]]:
            if blocked_by_id[str(row["case_id"])]:
                return []
            return comparison_adapter.predict_verifier(make_request(row, include_proposals=True))

        rule_predictions = {
            case_id: [
                {
                    "id": mention["id"],
                    "start": mention["start"],
                    "end": mention["end"],
                    "canonical": mention["canonical"],
                    "role": role["role"],
                }
                for mention, role in zip(mentions_by_case[case_id], proposed_by_case[case_id])
            ]
            for case_id in mentions_by_case
        }

        def eligible_by_case(row: dict[str, Any]) -> list[str]:
            case_id = str(row["case_id"])
            risk = risk_by_id[case_id]
            eligible: list[str] = []
            compiled = compiled_by_id[case_id]
            by_span = {
                (int(item.get("start", -1)), int(item.get("end", -1)), str(item.get("dimension") or "")): item
                for item in compiled.get("dimension_roles") or []
            }
            for mention in mentions_by_case[case_id]:
                assignment = by_span.get((mention["start"], mention["end"], mention["canonical"]), {})
                if (
                    risk in {"elevated", "conflict"}
                    or assignment.get("confidence_class") in {"open", "unresolved"}
                    or assignment.get("conflict")
                ):
                    eligible.append(mention["id"])
            return eligible

        try:
            p3b_comparison = compare_role_predictions(
                rows,
                rules_predictor=lambda row: rule_predictions[str(row["case_id"])],
                llm_predictor=predict_llm_only if comparison in {"b", "all"} else lambda _row: [],
                verifier_predictor=predict_verifier if comparison in {"c", "all"} else lambda _row: [],
                risk_by_case=lambda row: risk_by_id[str(row["case_id"])],
                blocked_by_case=lambda row: blocked_by_id[str(row["case_id"])],
                eligible_by_case=eligible_by_case,
                post_hybrid_preflight=hybrid_support_preflight,
                result_provenance="adapter_predictions",
            )
            if comparison in {"b", "all"}:
                if p3b_comparison["evaluated_case_count"]:
                    b_metrics = {
                        **p3b_comparison["B_llm_only"],
                        "status": "completed",
                        "reason": None,
                        "evaluated_case_count": p3b_comparison["evaluated_case_count"],
                        "blocked_case_count": p3b_comparison["blocked_case_count"],
                        "result_provenance": "adapter_predictions",
                    }
                else:
                    b_metrics = _not_run_result("all_cases_blocked_by_preflight")
            else:
                b_metrics = _not_run_result("not_requested")
            if comparison in {"c", "all"}:
                if p3b_comparison["evaluated_case_count"]:
                    c_metrics = {
                        **p3b_comparison["C_hybrid"],
                        "status": "completed",
                        "reason": None,
                        "evaluated_case_count": p3b_comparison["evaluated_case_count"],
                        "blocked_case_count": p3b_comparison["blocked_case_count"],
                        "result_provenance": "adapter_predictions",
                    }
                else:
                    c_metrics = _not_run_result("all_cases_blocked_by_preflight")
            else:
                c_metrics = _not_run_result("not_requested")
        except Exception:  # noqa: BLE001 - provider/adapter errors are not metrics
            p3b_comparison = None
            if comparison in {"b", "all"}:
                b_metrics = _not_run_result("adapter_execution_failed")
            if comparison in {"c", "all"}:
                c_metrics = _not_run_result("adapter_execution_failed")

    if comparison_wanted and b_metrics["status"] == "not_run" and not_run_reason != "not_requested":
        b_metrics = _not_run_result(not_run_reason)
    if comparison_wanted and c_metrics["status"] == "not_run" and not_run_reason != "not_requested":
        c_metrics = _not_run_result(not_run_reason)
    comparison_evidence = None
    if p3b_comparison is not None:
        comparison_evidence = {
            "evaluated_case_count": p3b_comparison["evaluated_case_count"],
            "blocked_case_count": p3b_comparison["blocked_case_count"],
            "metrics_scope": "role_and_mention_only",
            "coverage": None,
            "wrong_contract_rate": None,
            "wrong_execution_rate": None,
            "result_provenance": "real_adapter",
        }
    summary["p3_comparison"] = {
        "dataset_count": total,
        "A_rules_only": a_metrics,
        "B_llm_only": b_metrics,
        "C_hybrid": c_metrics,
        "legacy_P3_A_rules_metrics": legacy_a_metrics,
        "offline_comparison": comparison_evidence,
        "requested_comparison": comparison,
        "llm_group_gates": gates,
        "llm_group_gates_satisfied": gates_satisfied,
    }
    payload = {"summary": summary, "cases": details}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="P3 dimension-role evaluation")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--comparison", choices=("a", "b", "c", "all"), default="a")
    parser.add_argument(
        "--enable-llm-groups",
        action="store_true",
        help="Explicit CLI opt-in for a future B/C provider comparison; does not override privacy or approval gates.",
    )
    args = parser.parse_args()
    payload = run(
        args.cases,
        args.out,
        comparison=args.comparison,
        enable_llm_groups=args.enable_llm_groups,
        approval_marker=os.getenv("SEMANTIC_VERIFIER_EVAL_APPROVAL"),
        raw_question_opt_in=os.getenv("SEMANTIC_VERIFIER_SEND_RAW_QUESTION", "false").lower() == "true",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"evidence={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
