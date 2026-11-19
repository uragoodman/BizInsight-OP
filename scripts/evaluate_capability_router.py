"""Offline evaluation of saved Semantic Interpreter contracts through the router."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.complex_analysis.capability_router import route_semantic_contract  # noqa: E402
from app.context.route_shadow_audit import _allowed_reason  # noqa: E402
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, SemanticCatalog  # noqa: E402
from app.context.semantic_contract_v2 import SemanticContractV21  # noqa: E402

DEFAULT_EVAL = Path("data/evaluations/semantic_interpreter_v3_7_request_scope_postspan_full70_20261114.json")
DEFAULT_OUTPUT = Path("data/evaluations/capability_router_v2_20261114.json")
DEFAULT_REVIEW = Path("docs/design/capability-router-v2-review-20261114.md")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_CONTRACT_STATUSES = {"CONFIRMED", "AMBIGUOUS", "UNSUPPORTED", "MISSING", "CONFLICT"}
_OP_KINDS = {
    "SEGMENT", "ANALYZE", "FILTER", "SORT", "AGGREGATE", "QUERY", "EXPLAIN",
    "EXPLAIN_CAUSE", "COMPARE", "TREND", "CONTRIBUTION", "BREAKDOWN", "ENUMERATE",
}
_POLARITIES = {"POSITIVE", "NEGATIVE"}
_MODES = {"EXECUTE", "EXPLAIN", "ASK"}
_ARGUMENT_PATHS = {
    path for operation in DEFAULT_SEMANTIC_CATALOG.operation_arguments
    for path in operation.allowed_paths
}
_ARGUMENT_STANCES = {"ADOPTED", "MENTIONED_ONLY", "EXCLUDED"}


def _safe_label(value: Any, *, purpose: str) -> str:
    text = str(value or "")
    if _SAFE_LABEL.fullmatch(text):
        return text
    digest = hashlib.sha256(f"capability-router-eval-v1:{purpose}:{text}".encode("utf-8")).hexdigest()
    return f"{purpose}-{digest[:20]}"


def _safe_concept(value: Any, catalog: SemanticCatalog) -> str:
    return str(value) if isinstance(value, str) and catalog.concept(value) is not None else "unregistered_concept"


def _argument_projection(raw_arguments: Any, catalog: SemanticCatalog) -> dict[str, list[dict[str, str]]]:
    if not isinstance(raw_arguments, Mapping):
        return {}
    projected: dict[str, list[dict[str, str]]] = {}
    for raw_path, raw_values in raw_arguments.items():
        path = str(raw_path)
        if path not in _ARGUMENT_PATHS or not isinstance(raw_values, (list, tuple)):
            continue
        values: list[dict[str, str]] = []
        for value in raw_values:
            if hasattr(value, "model_dump"):
                value = value.model_dump(mode="json")
            if isinstance(value, str):
                values.append({"concept_id": _safe_concept(value, catalog), "stance": "ADOPTED"})
            elif isinstance(value, Mapping):
                concept_id = value.get("concept_id")
                if not isinstance(concept_id, str):
                    continue
                stance = value.get("stance")
                values.append({
                    "concept_id": _safe_concept(concept_id, catalog),
                    "stance": stance if stance in _ARGUMENT_STANCES else "unknown",
                })
        if values:
            projected[path] = sorted(values, key=lambda item: (item["concept_id"], item["stance"]))
    return dict(sorted(projected.items()))


def _expected_operations(result: Mapping[str, Any], catalog: SemanticCatalog) -> list[dict[str, Any]] | None:
    expected = result.get("expected")
    contract = expected.get("contract") if isinstance(expected, Mapping) else None
    raw_operations = contract.get("operations") if isinstance(contract, Mapping) else None
    if not isinstance(raw_operations, list):
        return None
    projected: list[dict[str, Any]] = []
    for operation in raw_operations:
        if not isinstance(operation, Mapping):
            continue
        candidate = operation.get("candidate")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        validator = operation.get("validator")
        validator = validator if isinstance(validator, Mapping) else {}
        raw_kind = operation.get("kind")
        if raw_kind not in _OP_KINDS:
            continue
        raw_targets = operation.get("target_concept_ids")
        targets = sorted({
            _safe_concept(value, catalog)
            for value in raw_targets if isinstance(value, str)
        }) if isinstance(raw_targets, (tuple, list)) else []
        polarity = candidate.get("polarity", operation.get("polarity", "POSITIVE"))
        mode = candidate.get("mode", operation.get("mode", "EXECUTE"))
        item: dict[str, Any] = {
            "kind": raw_kind,
            "target_concept_ids": targets,
        }
        if polarity in _POLARITIES:
            item["polarity"] = polarity
        if mode in _MODES:
            item["mode"] = mode
        raw_args = validator.get("effective_arguments")
        if isinstance(raw_args, Mapping):
            item["arguments"] = _argument_projection(raw_args, catalog)
        mentioned_args = candidate.get("mentioned_arguments", operation.get("mentioned_arguments"))
        if isinstance(mentioned_args, Mapping):
            item["mentioned_arguments"] = _argument_projection(mentioned_args, catalog)
        projected.append(item)
    return projected


def _actual_operations(contract: SemanticContractV21, catalog: SemanticCatalog) -> list[dict[str, Any]]:
    return [
        {
            "operation_id": _opaque_id(operation.id, kind="operation"),
            "clause_id": _opaque_id(operation.clause_id, kind="clause"),
            "kind": operation.kind,
            "target_concept_ids": sorted(_safe_concept(value, catalog) for value in operation.target_concept_ids),
            "polarity": operation.polarity,
            "mode": operation.mode,
            "arguments": _argument_projection(operation.effective_arguments, catalog),
            "mentioned_arguments": _argument_projection(operation.mentioned_arguments, catalog),
        }
        for operation in contract.operations
    ]


def _draft_match(
    expected: list[dict[str, Any]] | None,
    actual: list[dict[str, Any]] | None,
) -> str:
    """Compare structured operation shapes against draft labels, never as gold accuracy."""
    if expected is None or actual is None:
        return "unavailable"
    remaining = [dict(item) for item in actual]
    for wanted in expected:
        match_index = None
        for index, observed in enumerate(remaining):
            if wanted.get("kind") != observed.get("kind"):
                continue
            if wanted.get("target_concept_ids") != observed.get("target_concept_ids"):
                continue
            if "polarity" in wanted and wanted["polarity"] != observed.get("polarity"):
                continue
            if "mode" in wanted and wanted["mode"] != observed.get("mode"):
                continue
            if "arguments" in wanted and wanted["arguments"] != observed.get("arguments"):
                continue
            if "mentioned_arguments" in wanted and wanted["mentioned_arguments"] != observed.get("mentioned_arguments"):
                continue
            match_index = index
            break
        if match_index is None:
            return "mismatch"
        remaining.pop(match_index)
    return "match" if not remaining else "mismatch"


def _catalog_bounded(outcome: Any, catalog: SemanticCatalog) -> bool:
    route_by_id = {item.route_id: item for item in catalog.routes}
    capability_by_id = {item.capability_id: item for item in catalog.capabilities}
    if outcome.route_id is not None and outcome.route_id not in route_by_id:
        return False
    capability_ids = {
        item.capability_id
        for item in (
            *outcome.capability_ids,
            *outcome.route_capability_ids,
            *outcome.decision_basis.capability_ids,
        )
    }
    if not capability_ids <= set(capability_by_id):
        return False
    if not set(outcome.decision_basis.rejected_candidates) <= set(capability_by_id):
        return False
    if outcome.status == "SELECTED":
        route = route_by_id.get(outcome.route_id)
        route_capabilities = {item.capability_id for item in route.capability_ids} if route else set()
        if not {item.capability_id for item in outcome.capability_ids} <= route_capabilities:
            return False
    return "template_id" not in outcome.model_dump(mode="json")


def _opaque_id(value: str, *, kind: str) -> str:
    digest = hashlib.sha256(f"capability-router-eval-id-v1:{kind}:{value}".encode("utf-8")).hexdigest()
    return f"{kind}_{digest}"


def _safe_reasons(outcome: Any, catalog: SemanticCatalog) -> list[str]:
    return sorted({
        reason if _allowed_reason(reason, catalog) else "unrecognized_rejection_reason"
        for reason in outcome.decision_basis.rejection_reasons
    })


def _evaluate_result(result: Mapping[str, Any], *, repeat: int, catalog: SemanticCatalog) -> dict[str, Any]:
    case_id = _safe_label(result.get("case_id"), purpose="case")
    category = _safe_label(result.get("category"), purpose="category")
    observation = result.get("observation")
    observation = observation if isinstance(observation, Mapping) else {}
    repair = observation.get("repair")
    repair = repair if isinstance(repair, Mapping) else {}
    contract_payload = observation.get("contract")
    expected = _expected_operations(result, catalog)
    row: dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "contract_present": isinstance(contract_payload, Mapping),
        "interpreter_status": str(observation.get("status") or "unknown")[:100],
        "repair_attempted": repair.get("attempted") is True,
        "repair_recovered": repair.get("recovered") is True,
        "repair_initial_status": str(repair.get("initial_status") or "")[:100] or None,
        "contract_status": None,
        "route_status": "no_contract",
        "route_id": None,
        "capability_ids": [],
        "route_capability_ids": [],
        "operation_ids": [],
        "clause_ids": [],
        "rejection_reasons": [],
        "route_outcome": None,
        "decision_basis": None,
        "domain": None,
        "request_mode": None,
        "modality": None,
        "has_positive_execute_operation": None,
        "operation_summaries": [],
        "expected_draft_operations": expected,
        "draft_match": "unavailable",
        "deterministic": None,
        "catalog_bounded": None,
    }
    if not isinstance(contract_payload, Mapping):
        return row

    try:
        contract = SemanticContractV21.model_validate_json(
            json.dumps(contract_payload, ensure_ascii=False, separators=(",", ":"))
        )
    except Exception:  # noqa: BLE001 - private model details must not enter evaluation JSON
        row["route_status"] = "interpreter_input_invalid"
        return row

    row["contract_status"] = contract.semantic_status
    row["domain"] = contract.domain
    row["request_mode"] = contract.request_frame.request_mode
    row["modality"] = contract.request_frame.modality
    operation_summaries = _actual_operations(contract, catalog)
    row["operation_summaries"] = operation_summaries
    row["has_positive_execute_operation"] = any(
        operation.polarity == "POSITIVE" and operation.mode == "EXECUTE"
        for operation in contract.operations
    )
    try:
        outcomes = [route_semantic_contract(contract, catalog) for _ in range(repeat)]
    except Exception:  # noqa: BLE001 - do not serialize provider or user material
        row["route_status"] = "router_error"
        row["deterministic"] = False
        row["catalog_bounded"] = False
        return row

    serialized = [
        json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for outcome in outcomes
    ]
    outcome = outcomes[0]
    bounded = _catalog_bounded(outcome, catalog)
    row.update({
        "route_status": outcome.status if bounded else "catalog_reference_violation",
        "route_id": outcome.route_id if bounded else None,
        "capability_ids": sorted({
            item.capability_id
            for item in (*outcome.capability_ids, *outcome.decision_basis.capability_ids)
            if item.capability_id in {capability.capability_id for capability in catalog.capabilities}
        }) if bounded else [],
        "route_capability_ids": sorted(item.capability_id for item in outcome.route_capability_ids) if bounded else [],
        "operation_ids": sorted(_opaque_id(value, kind="operation") for value in outcome.decision_basis.operation_ids),
        "clause_ids": sorted(_opaque_id(value, kind="clause") for value in outcome.decision_basis.clause_ids),
        "rejection_reasons": _safe_reasons(outcome, catalog),
        "draft_match": _draft_match(expected, operation_summaries),
        "route_outcome": {
            "status": outcome.status,
            "domain": outcome.domain,
            "route_id": outcome.route_id if bounded else None,
            "capability_ids": sorted(item.capability_id for item in outcome.capability_ids) if bounded else [],
            "route_capability_ids": sorted(item.capability_id for item in outcome.route_capability_ids) if bounded else [],
            "clarification_required": outcome.clarification_required,
        } if bounded else None,
        "decision_basis": {
            "contract_version": outcome.decision_basis.contract_version,
            "catalog_snapshot_hash": outcome.decision_basis.catalog_snapshot_hash,
            "operation_ids": sorted(_opaque_id(value, kind="operation") for value in outcome.decision_basis.operation_ids),
            "clause_ids": sorted(_opaque_id(value, kind="clause") for value in outcome.decision_basis.clause_ids),
            "capability_ids": sorted(item.capability_id for item in outcome.decision_basis.capability_ids),
            "effective_arguments_digest": outcome.decision_basis.effective_arguments_digest,
            "rejected_candidates": sorted(outcome.decision_basis.rejected_candidates),
            "rejection_reasons": _safe_reasons(outcome, catalog),
        } if bounded else None,
        "deterministic": len(set(serialized)) == 1,
        "catalog_bounded": bounded,
    })
    if outcome.status == "SELECTED" and outcome.route_id in {item.route_id for item in catalog.routes}:
        row["capability_ids"] = sorted(item.capability_id for item in outcome.capability_ids)
    return row


def evaluate_payload(payload: Mapping[str, Any], *, repeat: int = 2) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat must be positive")
    results = payload.get("results")
    if not isinstance(results, list) or any(not isinstance(item, Mapping) for item in results):
        raise ValueError("evaluation must contain a results array of objects")
    rows = [
        _evaluate_result(item, repeat=repeat, catalog=DEFAULT_SEMANTIC_CATALOG)
        for item in results
    ]
    route_status_counts = Counter(row["route_status"] for row in rows)
    by_category: dict[str, dict[str, int]] = defaultdict(dict)
    selected_capability_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    for row in rows:
        by_category[row["category"]][row["route_status"]] = (
            by_category[row["category"]].get(row["route_status"], 0) + 1
        )
        for capability_id in row["capability_ids"]:
            capability_counts[capability_id] += 1
            if row["route_status"] == "SELECTED":
                selected_capability_counts[capability_id] += 1
    source_digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "source_sha256": source_digest,
        "result_count": len(rows),
        "contract_count": sum(row["contract_present"] for row in rows),
        "no_contract_count": route_status_counts.get("no_contract", 0),
        "repair_attempt_count": sum(row["repair_attempted"] for row in rows),
        "repair_recovered_count": sum(row["repair_recovered"] for row in rows),
        "repeat_count": repeat,
        "determinism_passed": all(row["deterministic"] is not False for row in rows),
        "catalog_bounded": all(row["catalog_bounded"] is not False for row in rows),
        "by_route_status": dict(sorted(route_status_counts.items())),
        "by_category": {key: dict(sorted(value.items())) for key, value in sorted(by_category.items())},
        "by_capability": dict(sorted(capability_counts.items())),
        "selected_capability_counts": dict(sorted(selected_capability_counts.items())),
        "draft_match_counts": dict(sorted(Counter(row["draft_match"] for row in rows).items())),
    }
    output = {
        "schema_version": "capability-router-eval-v1",
        "catalog_snapshot_hash": DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
        "summary": summary,
        "results": rows,
    }
    _assert_no_private_fields(output)
    return output


def _assert_no_private_fields(value: Any) -> None:
    forbidden_keys = {"text", "user_message", "raw_text", "surface", "span", "template_id"}
    if isinstance(value, Mapping):
        leaked = forbidden_keys.intersection(value)
        if leaked:
            raise ValueError(f"evaluation output includes forbidden field(s): {sorted(leaked)}")
        for child in value.values():
            _assert_no_private_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_private_fields(child)


def _markdown_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_review_sheet(payload: Mapping[str, Any], output: Mapping[str, Any]) -> str:
    source_rows = payload["results"]
    evaluated_by_case = {row["case_id"]: row for row in output["results"]}
    lines = [
        "# Capability Router v2 Offline Review",
        "",
        "These are draft comparisons, not gold labels. Review status is pending; no reviewed accuracy is claimed.",
        "",
        "| Case ID | Category | Text | Contract status | Route status | Selected capability | Rejection reasons | Expected draft operations | Draft match | Review status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for source in source_rows:
        case_id = _safe_label(source.get("case_id"), purpose="case")
        row = evaluated_by_case[case_id]
        selected = row["capability_ids"] if row["route_status"] == "SELECTED" else []
        draft_ops = row["expected_draft_operations"]
        cells = (
            case_id,
            row["category"],
            source.get("text", ""),
            row["contract_status"] or "no_contract",
            row["route_status"],
            ", ".join(selected),
            ", ".join(row["rejection_reasons"]),
            json.dumps(draft_ops, ensure_ascii=False, separators=(",", ":")) if draft_ops is not None else "unavailable",
            row["draft_match"],
            "pending",
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", dest="eval_path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-sheet", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    payload = json.loads(args.eval_path.read_text(encoding="utf-8"))
    result = evaluate_payload(payload, repeat=args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.review_sheet.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.review_sheet.write_text(render_review_sheet(payload, result), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "review_sheet": str(args.review_sheet),
        "summary": result["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
