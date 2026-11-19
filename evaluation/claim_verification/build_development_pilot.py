"""Build a labelled-by-provenance 30-case development pilot without model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NUMERIC_MUTATIONS = {
    "FBG037": ("**1.0**", "**9.0**"),
    "FBG053": ("从 2017-08-20 的 2.94 到 2017-08-21 的 2.35", "从 2017-08-20 的 9.94 到 2017-08-21 的 2.35"),
    "FBG060": ("2017-01-06 的 0.00", "2017-01-06 的 10.00"),
    "FBG058": ("2017-04-07：95.43", "2017-04-07：195.43"),
    "FBG046": ("| 基线日 | 2017-08-20 | 0.0 |", "| 基线日 | 2017-08-20 | 2.0 |"),
    "FBG049": ("当前日 | 2017-08-21 | 66.06", "当前日 | 2017-08-21 | 86.06"),
    "FBG015": ("总变化 -18.00 单", "总变化 +18.00 单"),
    "FBG016": ("总变化 +746.78 雷亚尔", "总变化 -746.78 雷亚尔"),
}

SEMANTIC_MUTATIONS = [
    ("FBG053", "因此复购率下降的业务根因已经确定为物流延迟。", "unsupported_causal_claim"),
    ("FBG060", "这证明服务流程已经显著改善，应立即削减售后资源。", "unsupported_causal_and_recommendation"),
    ("FBG037", "这说明客户需求正在减弱，应立即下调库存。", "unsupported_causal_and_recommendation"),
    ("FBG058", "发货时长下降的直接原因已经确定为仓储优化。", "unsupported_causal_claim"),
    ("FBG046", "取消率不变证明退款体验已经优化，应停止进一步监控。", "unsupported_causal_and_recommendation"),
    ("FBG049", "分期占比上升证明用户购买力已经恢复，应增加授信额度。", "unsupported_causal_and_recommendation"),
    ("FBG015", "Other 商品类别是订单量下降的已确认业务根因。", "contribution_misrepresented_as_causality"),
    ("FBG016", "Other 商品类别是运费上升的已确认业务根因。", "contribution_misrepresented_as_causality"),
    ("FBG017", "支付方式是订单审批率变化的已确认业务根因。", "causality_despite_capability_stop"),
    ("FBG020", "支付方式是含文本评论率变化的已确认业务根因。", "causality_despite_capability_stop"),
    ("FBG023", "虽然当前期缺少数据，仍可确认平均送达时长已经改善。", "claim_despite_data_insufficient"),
    ("FBG049", "同时，当前日的分期支付金额占比低于基线日。", "cross_claim_contradiction"),
]


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Pilot output already exists; do not overwrite a development dataset.")
    source = {str(row["case_id"]): row for row in _load(args.workspace)}
    missing = (set(NUMERIC_MUTATIONS) | {case_id for case_id, _, _ in SEMANTIC_MUTATIONS}) - set(source)
    if missing:
        raise RuntimeError("Workspace omits mutation source cases: " + ", ".join(sorted(missing)))
    # FBG023's source answer reached a rejected Critic state.  Keep it as the
    # source of a semantic mutation but exclude it from the ten real-draft rows.
    real_ids = [case_id for case_id in source if case_id != "FBG023"]
    if len(real_ids) != 10:
        raise RuntimeError("Expected exactly ten real-draft source rows after exclusion.")
    pilot = []
    for case_id in sorted(real_ids):
        row = source[case_id]
        pilot.append({
            "pilot_id": f"REAL-{case_id}", "provenance": "real_trace_draft",
            "source_case_id": case_id, "source_trace_id": row["trace_id"],
            "answer": row["original_answer_seen_by_critic"], "task_requirements": row["task_requirements"],
            "annotation_status": "pending_human_review", "expected_detector": "pending_human_review",
        })
    for case_id, (before, after) in sorted(NUMERIC_MUTATIONS.items()):
        row = source[case_id]
        answer = row["original_answer_seen_by_critic"]
        if answer.count(before) != 1:
            raise RuntimeError(f"Numeric mutation anchor is not unique for {case_id}: {before!r}")
        pilot.append({
            "pilot_id": f"NUM-{case_id}", "provenance": "targeted_mutation",
            "mutation_type": "numeric_fact_error", "source_case_id": case_id,
            "source_trace_id": row["trace_id"], "answer": answer.replace(before, after),
            "original_text": before, "changed_text": after, "task_requirements": row["task_requirements"],
            "annotation_status": "mutation_specified_not_human_reviewed",
            "expected_detector": "deterministic_reject_or_revise",
        })
    for index, (case_id, addition, mutation_type) in enumerate(SEMANTIC_MUTATIONS, start=1):
        row = source[case_id]
        pilot.append({
            "pilot_id": f"SEM-{index:02d}-{case_id}", "provenance": "targeted_mutation",
            "mutation_type": mutation_type, "source_case_id": case_id,
            "source_trace_id": row["trace_id"], "answer": row["original_answer_seen_by_critic"] + "\n\n" + addition,
            "original_text": "", "changed_text": addition, "task_requirements": row["task_requirements"],
            "annotation_status": "mutation_specified_not_human_reviewed",
            "expected_detector": "needs_semantic_review",
        })
    if len(pilot) != 30:
        raise RuntimeError(f"Pilot must contain 30 records, got {len(pilot)}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pilot), encoding="utf-8")
    print(json.dumps({"total": len(pilot), "real_trace_draft": 10, "numeric_mutation": 8, "semantic_mutation": 12}, ensure_ascii=False))


if __name__ == "__main__":
    main()
