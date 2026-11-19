"""Explicitly repair three reviewed formatting/reference defects; not gold labels."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from evaluation.claim_verification.annotate_development import validate


def repair(case: dict, raw: dict) -> tuple[dict, list[str]]:
    response = copy.deepcopy(raw)
    case_id = case["case_id"]
    actions: list[str] = []
    if case_id == "9db34cd00433eeb8ce25":
        op = case.get("operational_context") or {}
        decisions = op.get("decision_records") or []
        if op.get("terminal_code") != "CAPABILITY_GAP" or len(decisions) != 1 or decisions[0].get("sql_executed") is not False or decisions[0].get("limitation_code") != "CAPABILITY_GAP":
            raise ValueError("capability state does not match reviewed snapshot")
        claims = response["claims"]
        if claims[1]["span"] != "未执行数据查询。" or not claims[2]["span"].startswith("已识别指标“含文本评论率”"):
            raise ValueError("unexpected capability claims")
        claims[1]["evidence_ids"] = ["operational_context"]
        claims[2]["evidence_ids"] = ["operational_context"]
        actions.append("Bind limitation claims to pre-Critic operation state, not business data")
        for item in response["requirements"]:
            if item["status"] == "supported":
                item["status"] = "covered"
            elif item["status"] == "unsupported":
                item["status"] = "missing"
        actions.append("Normalize unsupported/supported requirement statuses to missing/covered; retain reasons")
    elif case_id in {"9e4191f526600937ec76", "a33c7d15b5834e04c97b"}:
        claims = response["claims"]
        before, percent = claims[3], claims[4]
        if not before["span"].startswith("整体下降 ") or not percent["span"].startswith("(-") or not percent["span"].endswith("%)"):
            raise ValueError("unexpected percentage claim")
        combined = before["span"] + "（" + percent["span"][1:-1] + "）"
        if case["answer"].count(combined) != 1:
            raise ValueError("combined span is not unique in source answer")
        before["span"] = combined
        before["correctness"] = "uncertain"
        before["reason"] = "模型将中文括号写成英文；合并为原文完整计算片段，正确性仍待独立复核。"
        del claims[4]
        actions.append("Combine exact original Chinese-parenthesis calculation; retain uncertainty")
    return response, actions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Refusing overwrite")
    inputs = [json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x.strip()]
    results = [json.loads(x) for x in args.results.read_text(encoding="utf-8").splitlines() if x.strip()]
    cases = {x["case_id"]: x for x in inputs}
    if len(inputs) != 30 or len(cases) != 30 or len(results) != 30 or {x["case_id"] for x in results} != set(cases):
        raise ValueError("Expected exactly 30 matching cases and results")
    rows = []
    for record in results:
        case = cases[record["case_id"]]
        corrected, actions = repair(case, record["response"])
        validation = validate(case, corrected)
        rows.append({"case_id": record["case_id"], "response": corrected, "validation": validation,
                     "assistant_repairs": actions, "human_verified": False, "oracle_claim_labels": False})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    counts = {key: sum(x["validation"]["status"] == key for x in rows) for key in ("structurally_valid", "invalid")}
    metadata = {"cases": len(rows), "counts": counts, "repaired": sum(bool(x["assistant_repairs"]) for x in rows),
                "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "raw_results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
                "human_verified": False, "oracle_claim_labels": False,
                "caveat": "Format corrections do not establish complete claim coverage, factual correctness, or independent review."}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
