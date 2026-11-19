"""Construct 20 hard-but-realistic error variants over the unified corpus.

Each variant changes exactly one thing in an otherwise correct answer, so a
detector's behaviour is attributable.  Every rule is declared up front and is
reproducible: the same corpus always yields the same 20 cases.

Categories (4 each):
  H1 cross_claim_contradiction  - append a sentence contradicting an existing number
  H2 direction_flip             - restate a movement with the opposite direction word
  H3 unit_mismatch              - call percentage points a percentage
  H4 requirement_omission       - drop the required unit/missing-status statement
  H5 evidence_misbinding        - cite evidence that does not contain the value

None of these is detectable by a numeric span probe; each requires reading the
answer as a whole or checking it against the requirement list.  That is the
point: they represent the review work a deterministic layer cannot absorb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from evaluation.claim_verification.annotate_assisted import _explainable

NUMBER = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?")
DIRECTION = [("上升", "下降"), ("增加", "减少"), ("增长", "下滑"), ("下降", "上升"), ("减少", "增加")]
PERCENT_POINT_NAMES = ("百分点",)


def _pick(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], str | None], limit: int) -> list[tuple[dict[str, Any], str]]:
    out = []
    for row in rows:
        target = predicate(row)
        if target:
            out.append((row, target))
        if len(out) >= limit:
            break
    return out


def h1_contradiction(row: dict[str, Any]) -> str | None:
    """Re-assert one period's figure while quoting another period's real value.

    Both numbers come from the answer or its evidence, so no invented value is
    involved; the defect is purely that the same period is now reported with
    two different figures.  A span probe sees two individually supported
    numbers and cannot object.
    """

    # "2017-12-01 [值]" style pairs only; require a separator so the date's own
    # digits are never mistaken for the value.
    dated = re.findall(
        r"(\d{4}-\d{2}-\d{2})\s*[=：:]\s*([+\-−]?\d[\d,]*(?:\.\d+)?)", row["answer"]
    )
    dated += re.findall(
        r"(\d{4}-\d{2}-\d{2})\s+的?(?:数值|值|对应值)?\s*(?:为|是)\s*([+\-−]?\d[\d,]*(?:\.\d+)?)", row["answer"]
    )
    pairs: list[tuple[str, float]] = []
    for date, raw in dated:
        try:
            pairs.append((date, float(raw.replace(",", "").replace("−", "-"))))
        except ValueError:
            continue
    by_date: dict[str, list[float]] = {}
    for date, value in pairs:
        by_date.setdefault(date, []).append(value)
    dates = [d for d, vs in by_date.items() if len(vs) == 1]
    if len(dates) < 2:
        return None
    # Take the first dated figure and restate it using a later period's value.
    first_date = dates[0]
    first_value = by_date[first_date][0]
    for date in dates[1:]:
        other = by_date[date][0]
        if abs(other - first_value) < 1e-9 or abs(other) < 1e-6:
            continue
        return f"\n\n{first_date} 的对应数值为 {other:,.2f}。"
    return None


def h2_direction(row: dict[str, Any]) -> str | None:
    for up, down in DIRECTION:
        for word, opposite in ((up, down), (down, up)):
            if word in row["answer"]:
                return f"\n\n后续区间该指标转为{opposite}。"
    return None


def h3_unit(row: dict[str, Any]) -> str | None:
    if not any(name in row["answer"] for name in PERCENT_POINT_NAMES):
        return None
    # Plain restatement that happens to be wrong; no "correction" wording.
    return "\n\n绝对差值单位为%。"


def h4_omission(row: dict[str, Any]) -> str | None:
    """Remove the only sentence that answers the unit / missing-status ask.

    Removal is accepted only when nothing else in the answer mentions a unit or
    missing status; otherwise the requirement is still answered and the label
    would be false.
    """

    def mentions(text: str) -> bool:
        return "单位" in text or "缺失状态" in text

    sentences = [s.strip() for s in re.split(r"(?<=[。\n])", row["answer"]) if s.strip()]
    for sentence in sentences:
        if not mentions(sentence) or sentence.startswith("本次分析口径"):
            continue
        if NUMBER.search(sentence):
            continue
        if row["answer"].count(sentence) != 1:
            continue
        if mentions(row["answer"].replace(sentence, "", 1)):
            continue
        return sentence
    return None


def h5_misbinding(row: dict[str, Any], table_pool: set[str] | None = None) -> str | None:
    """Attribute the figures to a real table that this case never read.

    The wrong table must come from the corpus-wide pool of tables, not from this
    case's own evidence, otherwise the citation would still be correct.
    """

    own: set[str] = set()
    for bundle in row["evidence_bundles"]:
        result = (bundle.get("content") or {}).get("result") or {}
        for key in ("contribution_result", "compare_result", "metric_result", "trend_result"):
            source = (result.get(key) or {}).get("source_table")
            if source:
                own.add(source)
    if not own:
        return None
    if "数据来源" in row["answer"]:
        return None
    pool = table_pool or set()
    wrong = sorted(t for t in pool if t not in own and t not in row["answer"])
    if not wrong:
        return None
    return f"\n\n数据来源：{wrong[0]}。"


def build(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = [
        ("H1", "cross_claim_contradiction", h1_contradiction),
        ("H2", "direction_flip", h2_direction),
        ("H3", "unit_mismatch", h3_unit),
        ("H4", "requirement_omission", h4_omission),
        ("H5", "evidence_misbinding", h5_misbinding),
    ]
    variants: list[dict[str, Any]] = []
    by_cohort: dict[str, list[dict[str, Any]]] = {}
    for row in corpus:
        by_cohort.setdefault(row["cohort"], []).append(row)
    order = ["old_bg", "x1_bg", "new_bg"]

    # Corpus-wide table pool: a misbinding must cite a table this case never read.
    table_pool: set[str] = set()
    for row in corpus:
        for bundle in row["evidence_bundles"]:
            result = (bundle.get("content") or {}).get("result") or {}
            for key in ("contribution_result", "compare_result", "metric_result", "trend_result"):
                source = (result.get(key) or {}).get("source_table")
                if source:
                    table_pool.add(source)

    for code, kind, fn in plan:
        picked = 0
        # Rotate the cohort order per family so each family starts from a
        # different source; within a family pick the first constructible case.
        start = plan.index((code, kind, fn)) % len(order)
        for cohort in order[start:] + order[:start]:
            if picked >= 4:
                break
            for row in by_cohort.get(cohort, []):
                if picked >= 4:
                    break
                target = fn(row, table_pool) if kind == "evidence_misbinding" else fn(row)
                if target is None:
                    continue
                answer = (row["answer"].replace(target, "", 1).strip()
                          if kind == "requirement_omission" else row["answer"] + target)
                if answer == row["answer"]:
                    continue
                picked += 1
                variants.append({
                    "case_id": f"{code}-{row['case_id']}",
                    "source_case_id": row["case_id"],
                    "cohort": row["cohort"],
                    "task_type": row.get("task_type"),
                    "user_query": row.get("user_query", ""),
                    "task_requirements": row.get("task_requirements") or [],
                    "request_contract": row.get("request_contract") or {},
                    "evidence_bundles": row["evidence_bundles"],
                    "operational_context": row.get("operational_context"),
                    # A variant inherits the source case's operational record;
                    # dropping these made the reviewer treat a capability stop
                    # as an ordinary answer.
                    "operational_context_provenance": row.get("operational_context_provenance"),
                    "registry_evidence": row.get("registry_evidence"),
                    "answer": answer,
                    "error_family": kind,
                    "mutation": target,
                    "expected": "should_be_flagged",
                })
        if picked != 4:
            raise RuntimeError(f"{kind}: only {picked}/4 constructible")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite frozen variants")
    corpus = [json.loads(x) for x in args.corpus.read_text(encoding="utf-8").splitlines() if x.strip()]
    variants = build(corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in variants), encoding="utf-8"
    )
    print(json.dumps({
        "variants": len(variants),
        "by_family": {k: sum(1 for v in variants if v["error_family"] == k) for k in {v["error_family"] for v in variants}},
        "cohorts": {c: sum(1 for v in variants if v["cohort"] == c) for c in {v["cohort"] for v in variants}},
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
