"""Score clarification predictions against reviewed Semantic Interpreter decisions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

PREDICTED_CLARIFICATION_STATUSES = {"AMBIGUOUS", "MISSING", "UNSUPPORTED", "CONFLICT"}


def _read_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("evaluation input must contain a results array")
    case_ids: set[str] = set()
    for result in payload["results"]:
        if not isinstance(result, dict):
            raise ValueError("every evaluation result must be an object")
        case_id = result.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError("evaluation results must have unique non-empty case_id values")
        case_ids.add(case_id)
    return payload


def _read_decisions(path: Path, known_case_ids: set[str]) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decision = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid decision JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(decision, dict):
            raise ValueError(f"decision at line {line_number} must be an object")
        case_id = decision.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"decision at line {line_number} has an invalid case_id")
        if case_id not in known_case_ids:
            raise ValueError(f"decision references unknown case_id {case_id!r}")
        if case_id in decisions:
            raise ValueError(f"duplicate decision for case_id {case_id!r}")
        clarification_required = decision.get("reviewed_clarification_required")
        if type(clarification_required) is not bool:
            raise ValueError(
                f"decision for {case_id!r} requires a boolean reviewed_clarification_required"
            )
        decisions[case_id] = clarification_required
    if not decisions:
        raise ValueError("review decisions contain no clarification labels")
    return decisions


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = sum(row["gold"] and row["predicted"] for row in rows)
    false_positive = sum(not row["gold"] and row["predicted"] for row in rows)
    false_negative = sum(row["gold"] and not row["predicted"] for row in rows)
    true_negative = sum(not row["gold"] and not row["predicted"] for row in rows)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {
        "case_count": len(rows),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _predict(result: dict[str, Any]) -> tuple[bool, str | None, int]:
    observation = result.get("observation") or {}
    contract = observation.get("contract") or {}
    semantic_status = contract.get("semantic_status")
    ambiguities = contract.get("ambiguities")
    ambiguity_count = len(ambiguities) if isinstance(ambiguities, (list, tuple, dict)) else 0
    predicted = semantic_status in PREDICTED_CLARIFICATION_STATUSES or ambiguity_count > 0
    return predicted, semantic_status if isinstance(semantic_status, str) else None, ambiguity_count


def evaluate_clarification(payload: dict[str, Any], decisions: dict[str, bool]) -> dict[str, Any]:
    """Evaluate reviewed decisions; draft cases are deliberately not scored."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("evaluation input must contain a results array")
    by_case_id = {item.get("case_id"): item for item in payload["results"] if isinstance(item, dict)}
    if len(by_case_id) != len(payload["results"]):
        raise ValueError("evaluation results must have unique non-empty case_id values")
    unknown = set(decisions) - set(by_case_id)
    if unknown:
        raise ValueError("decisions reference unknown case_id values: " + ", ".join(sorted(unknown)))

    rows = []
    for case_id, gold in decisions.items():
        result = by_case_id[case_id]
        if result.get("label_status") != "reviewed" or not isinstance(result.get("review"), dict):
            raise ValueError(f"case {case_id!r} is not present as reviewed in the merged evaluation")
        merged_gold = result["review"].get("reviewed_clarification_required")
        if merged_gold is not gold:
            raise ValueError(f"merged clarification label disagrees with decision for {case_id!r}")
        predicted, semantic_status, ambiguity_count = _predict(result)
        rows.append({
            "case_id": case_id,
            "category": str(result.get("category") or "unspecified"),
            "gold": gold,
            "predicted": predicted,
            "semantic_status": semantic_status,
            "ambiguity_count": ambiguity_count,
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_rule": (
            "semantic_status in AMBIGUOUS/MISSING/UNSUPPORTED/CONFLICT or contract.ambiguities is non-empty"
        ),
        "gold_field": "reviewed_clarification_required",
        "reviewed_case_count": len(rows),
        "overall": _binary_metrics(rows),
        "by_category": {
            category: _binary_metrics(items)
            for category, items in sorted(grouped.items())
        },
        "cases": rows,
    }


def _render_report(metrics: dict[str, Any], *, eval_path: Path, decisions_path: Path) -> str:
    overall = metrics["overall"]
    lines = [
        "# Semantic Interpreter v3.6 澄清指标",
        "",
        f"- 生成时间（UTC）：`{metrics['created_at_utc']}`",
        f"- Merged evaluation：`{eval_path.as_posix()}`",
        f"- Reviewed decisions：`{decisions_path.as_posix()}`",
        f"- Gold：`{metrics['gold_field']}`；预测规则：`{metrics['prediction_rule']}`。",
        f"- 样本数：`{metrics['reviewed_case_count']}` 条 reviewed；draft 不参与计算。",
        "- 本指标是 Phase 2 前置观察项，不纳入 semantic interpreter gate。",
        "",
        "| 范围 | n | TP | FP | FN | TN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("总体", overall),
    ]
    for category, summary in metrics["by_category"].items():
        lines.append(_metric_row(category, summary))
    lines.extend(["", "## Reviewed case predictions", "", "| Case | Category | Gold | Predicted | Status | Ambiguities |", "|---|---|---:|---:|---|---:|"])
    for item in metrics["cases"]:
        lines.append(
            f"| `{item['case_id']}` | `{item['category']}` | {item['gold']} | {item['predicted']} | "
            f"`{item['semantic_status'] or 'unavailable'}` | {item['ambiguity_count']} |"
        )
    lines.extend([
        "",
        "Precision/recall 在分母为 0 时记为 `null`；F1 在 TP/FP/FN 均为 0 时记为 `null`。",
        "",
    ])
    return "\n".join(lines)


def _metric_row(label: str, summary: dict[str, Any]) -> str:
    def value(field: str) -> str:
        metric = summary[field]
        return "null" if metric is None else str(metric)

    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        label, summary["case_count"], summary["true_positive"], summary["false_positive"],
        summary["false_negative"], summary["true_negative"], value("precision"),
        value("recall"), value("f1"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, help="Merged full evaluation JSON")
    parser.add_argument("--decisions", required=True, help="Reviewed decisions JSONL")
    parser.add_argument("--output", required=True, help="Output evidence JSON path")
    parser.add_argument("--report", help="Optional Markdown report path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    eval_path = Path(args.eval).resolve()
    decisions_path = Path(args.decisions).resolve()
    output_path = Path(args.output).resolve()
    try:
        payload = _read_evaluation(eval_path)
        known_case_ids = {item["case_id"] for item in payload["results"]}
        decisions = _read_decisions(decisions_path, known_case_ids)
        metrics = evaluate_clarification(payload, decisions)
        metrics["evaluation_path"] = str(eval_path.relative_to(ROOT)) if eval_path.is_relative_to(ROOT) else str(eval_path)
        metrics["decisions_path"] = str(decisions_path.relative_to(ROOT)) if decisions_path.is_relative_to(ROOT) else str(decisions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.report:
            report_path = Path(args.report).resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                _render_report(metrics, eval_path=eval_path, decisions_path=decisions_path),
                encoding="utf-8",
            )
        print(json.dumps({"reviewed_case_count": metrics["reviewed_case_count"], "overall": metrics["overall"], "output": str(output_path)}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
