# -*- coding: utf-8 -*-
"""Aggregate repeated E3-real provider runs with mean and sample variance.

Each input result file is one repeat of the same prompt and case set. The
script reuses the deterministic E3 fact/evidence scorer, then reports
per-metric mean, sample standard deviation, min and max across repeats.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from evaluation.e3_fact_score import _available_evidence, _load_cases, _score

_METRICS = (
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "fact_covered",
    "fact_total",
    "fact_recall",
    "evidence_retained",
    "required_evidence",
    "evidence_recall",
    "valid_citations",
    "cited_evidence",
    "evidence_precision",
    "unsupported_citations",
    "answer_nonempty",
)


def _stat(values: list[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "per_repeat": [],
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "per_repeat": clean,
        "mean": statistics.mean(clean),
        "variance": statistics.variance(clean) if len(clean) > 1 else 0.0,
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def _load_result(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_id"]): row for row in payload.get("results") or []}


def _mode_aggregate(rows: list[dict[str, Any]], mode: str, repeat: int) -> dict[str, Any]:
    values = [row["repeats"][repeat][mode] for row in rows]
    input_total = sum(value["input_tokens"] or 0 for value in values)
    output_total = sum(value["output_tokens"] or 0 for value in values)
    fact_total = sum(value["fact_total"] for value in values)
    fact_covered = sum(value["fact_covered"] for value in values)
    required = sum(value["required_evidence"] for value in values)
    retained = sum(value["evidence_retained"] for value in values)
    cited = sum(value["cited_evidence"] for value in values)
    valid = sum(value["valid_citations"] for value in values)
    return {
        "input_tokens": input_total,
        "output_tokens": output_total,
        "latency_ms": sum(value["latency_ms"] or 0 for value in values),
        "fact_total": fact_total,
        "fact_covered": fact_covered,
        "fact_recall": fact_covered / fact_total if fact_total else None,
        "required_evidence": required,
        "evidence_retained": retained,
        "evidence_recall": retained / required if required else None,
        "cited_evidence": cited,
        "valid_citations": valid,
        "evidence_precision": valid / cited if cited else None,
        "unsupported_citations": sum(value["unsupported_citations"] for value in values),
        "answer_nonempty": sum(value["answer_nonempty"] for value in values),
    }


def _summary(rows: list[dict[str, Any]], repeat_count: int) -> dict[str, Any]:
    per_repeat = {
        mode: [_mode_aggregate(rows, mode, repeat) for repeat in range(repeat_count)]
        for mode in ("raw", "packed")
    }
    reductions = [
        1 - packed["input_tokens"] / raw["input_tokens"]
        if raw["input_tokens"] else None
        for raw, packed in zip(per_repeat["raw"], per_repeat["packed"])
    ]
    return {
        "cases": len(rows),
        "raw": {
            metric: _stat([item[metric] for item in per_repeat["raw"]])
            for metric in _METRICS
        },
        "packed": {
            metric: _stat([item[metric] for item in per_repeat["packed"]])
            for metric in _METRICS
        },
        "input_token_reduction": _stat(reductions),
    }


def _score_case(
    case: dict[str, Any],
    result: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    available = _available_evidence(case)
    row: dict[str, Any] = {"repeat": repeat}
    for mode in ("raw", "packed"):
        call = result[mode]
        score = _score(case, call["answer"], available[mode])
        row[mode] = {
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "latency_ms": call.get("latency_ms"),
            "fact_total": score["fact_total"],
            "fact_covered": score["fact_covered"],
            "fact_recall": score["fact_recall"],
            "required_evidence": len(score["required_evidence_ids"]),
            "evidence_retained": len(
                set(score["required_evidence_ids"]) & set(score["found_evidence_ids"])
            ),
            "evidence_recall": score["evidence_recall"],
            "cited_evidence": len(score["found_evidence_ids"]),
            "valid_citations": len(score["valid_citation_ids"]),
            "evidence_precision": score["evidence_precision"],
            "unsupported_citations": len(score["unsupported_evidence_ids"]),
            "answer_nonempty": int(score["answer_nonempty"]),
        }
    row["input_token_reduction"] = result.get("input_token_reduction")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--cases", default="evaluation/datasets/e3_context_cases_v1.jsonl")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    paths = [Path(path) for path in args.results]
    if len(paths) < 2:
        parser.error("--results needs at least two repeat files")
    cases = _load_cases(Path(args.cases))
    repeats = [_load_result(path) for path in paths]
    case_ids = sorted(repeats[0])
    for path, result in zip(paths, repeats):
        if sorted(result) != case_ids:
            parser.error(f"{path} has a different case set")
        for case_id in case_ids:
            if case_id not in cases:
                parser.error(f"{path} contains unknown case {case_id}")

    details = []
    for case_id in case_ids:
        runs = [
            _score_case(cases[case_id], result[case_id], repeat)
            for repeat, result in enumerate(repeats)
        ]
        details.append({
            "case_id": case_id,
            "difficulty": repeats[0][case_id]["difficulty"],
            "question": repeats[0][case_id]["question"],
            "repeats": runs,
        })

    report = {
        "schema_version": "e3-real-repeat-v1",
        "repeat_count": len(paths),
        "result_files": [str(path) for path in paths],
        "std_kind": "sample standard deviation (ddof=1)",
        "overall": _summary(details, len(paths)),
        "by_difficulty": {
            difficulty: _summary(
                [row for row in details if row["difficulty"] == difficulty],
                len(paths),
            )
            for difficulty in ("easy", "medium", "hard")
        },
        "details": details,
        "limits": [
            f"{len(paths)} repeats of the same prompt; this is a variance check, not a production sample.",
            "Input token counts are deterministic for fixed context, so token variance can be zero.",
            "Fact/evidence scoring is deterministic; answer wording may still vary across repeats.",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    return 0 if all(
        report["overall"][mode]["answer_nonempty"]["mean"]
        for mode in ("raw", "packed")
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
