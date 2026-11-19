# -*- coding: utf-8 -*-
"""Real-provider E3 evaluation: raw context vs Packer context.

Opt-in: requires a configured OpenAI-compatible LLM provider. It calls the real
``llm_chat`` client for the selected E3 cases and records actual provider
usage, latency and whether required evidence IDs appear in the answer.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from app.context.packer import ContextBudget, ContextItem, ContextSource, pack_context
from app.core.execution_context import activate_state, reset_state
from app.llm.client import llm_chat


def _items(case: dict[str, Any]) -> list[ContextItem]:
    return [
        ContextItem(
            item_id=str(row["item_id"]),
            source=ContextSource(str(row["source"])),
            content=str(row["content"]),
            priority=int(row.get("priority", 50)),
            required=bool(row.get("required")),
            evidence_ids=tuple(str(v) for v in (row.get("evidence_ids") or [])),
        )
        for row in case["items"]
    ]


def _contexts(case: dict[str, Any]) -> tuple[str, str]:
    items = _items(case)
    raw_budget = ContextBudget(
        agent=str(case["agent"]), model=str(case.get("model") or "test-model"),
        **{**case["budget"], "available_context_tokens": 1_000_000},
    )
    compact_budget = ContextBudget(
        agent=str(case["agent"]), model=str(case.get("model") or "test-model"),
        **case["budget"],
    )
    return (
        pack_context(items, raw_budget).text,
        pack_context(items, compact_budget).text,
    )


def _extract_evidence(answer: str) -> list[str]:
    match = re.search(r"EVIDENCE_IDS\s*:\s*(.+)", answer, re.IGNORECASE)
    if not match:
        return []
    return [
        token.strip().strip(".。")
        for token in re.split(r"[,，\s]+", match.group(1))
        if token.strip().strip(".。")
    ]


def _call(context: str, question: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "trace_id": "e3-real",
        "chat_session_id": "e3-real",
        "llm_calls": [],
    }
    token = activate_state(state)
    started = time.perf_counter()
    try:
        answer = llm_chat(
            "You are a data analysis assistant. Use only the supplied context. "
            "Answer in at most 120 words, without tables. Preserve exact numbers "
            "(percentages, percentage points, amounts) from the context when they "
            "are relevant. The final line must be: "
            "EVIDENCE_IDS: comma-separated evidence ids used.",
            f"Context:\n{context}\n\nQuestion: {question}",
            temperature=0.0,
            max_tokens=512,
        )
    finally:
        reset_state(token)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    attempts = [a for call in state.get("llm_calls") or [] for a in call.get("attempts") or []]
    inputs = [a.get("input_tokens") for a in attempts if a.get("input_tokens") is not None]
    outputs = [a.get("output_tokens") for a in attempts if a.get("output_tokens") is not None]
    return {
        "answer": answer,
        "latency_ms": elapsed,
        "input_tokens": sum(inputs) if inputs else None,
        "output_tokens": sum(outputs) if outputs else None,
        "usage_complete": len(inputs) == len(attempts) and len(outputs) == len(attempts),
    }


def _report(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "e3-real-provider-v1",
        "cases": len(results),
        "model": "configured llm_chat model",
        "results": results,
        "limits": [
            f"{len(results)}-case real-provider run; fixed E3 case set, not a production sample.",
            "Answer quality is checked only by non-empty response and evidence-ID line.",
            "No provider key or request body is written to the result.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/datasets/e3_context_cases_v1.jsonl")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--per-difficulty", type=int, default=2)
    parser.add_argument(
        "--all-cases", action="store_true",
        help="Run every case in the dataset instead of the first N per difficulty.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed raw/packed answers from --json-out and retry failed cases.",
    )
    args = parser.parse_args()
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.all_cases:
        selected = list(cases)
    else:
        selected = []
        for difficulty in ("easy", "medium", "hard"):
            selected.extend([case for case in cases if case["difficulty"] == difficulty][: args.per_difficulty])

    target = Path(args.json_out) if args.json_out else None
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and target and target.is_file():
        for row in json.loads(target.read_text(encoding="utf-8")).get("results") or []:
            if row.get("case_id"):
                existing[str(row["case_id"])] = row

    results: list[dict[str, Any]] = []
    for case in selected:
        previous = existing.get(case["case_id"])
        if (
            previous
            and previous.get("raw", {}).get("answer_nonempty")
            and previous.get("packed", {}).get("answer_nonempty")
        ):
            results.append(previous)
            continue
        raw, compact = _contexts(case)
        try:
            raw_call = _call(raw, case["items"][0]["content"])
        except Exception as exc:  # noqa: BLE001 - record and keep the batch resumable
            raw_call = {
                "answer": "", "latency_ms": None, "input_tokens": None,
                "output_tokens": None, "usage_complete": False,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        try:
            compact_call = _call(compact, case["items"][0]["content"])
        except Exception as exc:  # noqa: BLE001 - record and keep the batch resumable
            compact_call = {
                "answer": "", "latency_ms": None, "input_tokens": None,
                "output_tokens": None, "usage_complete": False,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        expected = {
            str(value)
            for item in case.get("items") or []
            if item.get("required")
            for value in item.get("evidence_ids") or []
        }
        for mode, call in (("raw", raw_call), ("packed", compact_call)):
            found = set(_extract_evidence(call["answer"]))
            call["evidence_ids"] = sorted(found)
            call["evidence_recall"] = (
                len(found & expected) / len(expected) if expected else None
            )
            call["answer_nonempty"] = bool(call["answer"].strip())
        results.append({
            "case_id": case["case_id"],
            "difficulty": case["difficulty"],
            "question": case["items"][0]["content"],
            "expected_evidence_ids": sorted(expected),
            "raw": raw_call,
            "packed": compact_call,
            "input_token_reduction": (
                1 - compact_call["input_tokens"] / raw_call["input_tokens"]
                if raw_call["input_tokens"] and compact_call["input_tokens"] else None
            ),
        })
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(_report(results), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    report = _report(results)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    return 0 if any(row["raw"]["answer_nonempty"] for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
