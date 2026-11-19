"""Triage 路由准确率效果评测（离线确定性）。

在不依赖真实 LLM 的前提下验证"问题 → 任务类型"路由准确率：fixtures 把
llm_classify 置空，triage 走确定性关键词降级（_is_definition_question 优先，
_keyword_fallback 兜底）。这样每次运行结果完全可复现，且暴露路由逻辑本身
的健壮性——即使模型降级，基础路由也不该错。

真实 LLM 的路由准确率属于 live 回归，需 provider 稳定可用，不在此离线评测内。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.graph.nodes.triage import triage
from evaluation.fixtures import offline_dependencies


ROOT = Path(__file__).resolve().parents[1]

# 每个 case 的关键词都落在 triage 降级规则的可判定区，避免把概率性 LLM 行为
# 当作确定性断言。
CASES = [
    {"query": "GMV 是什么？", "expected": "metric_definition"},
    {"query": "什么是客单价？", "expected": "metric_definition"},
    {"query": "退款率 cancel_rate 的计算方式", "expected": "metric_definition"},
    {"query": "平均客单价怎么算", "expected": "metric_definition"},
    {"query": "查询 2018-08-10 的 GMV", "expected": "metric_query"},
    {"query": "最近 7 天订单量趋势", "expected": "metric_query"},
    {"query": "对比这周和上周的 GMV", "expected": "metric_query"},
    {"query": "查询 2018-08-10 的订单量", "expected": "metric_query"},
    {"query": "为什么 GMV 下降了？", "expected": "anomaly_analysis"},
    {"query": "订单量异常的原因", "expected": "anomaly_analysis"},
    {"query": "为什么订单量突然下降", "expected": "anomaly_analysis"},
    {"query": "GMV 异常波动的原因是什么", "expected": "anomaly_analysis"},
    {"query": "生成本周经营分析报告", "expected": "report_generation"},
    {"query": "写一份 GMV 分析日报", "expected": "report_generation"},
    {"query": "生成月度经营分析报告", "expected": "report_generation"},
    {"query": "做一份上周的销售日报", "expected": "report_generation"},
]


def evaluate_triage_routing() -> dict:
    details: list[dict] = []
    with offline_dependencies():
        for case in CASES:
            state = triage({"user_message": case["query"]})
            actual = state.get("task_type")
            details.append({
                "query": case["query"], "expected": case["expected"],
                "actual": actual, "passed": actual == case["expected"],
            })
    passed = sum(item["passed"] for item in details)
    return {
        "cases": len(details), "passed": passed,
        "accuracy": round(passed / len(details), 4),
        "mode": "offline_keyword_fallback", "details": details,
    }


def write_evidence() -> tuple[Path, dict]:
    result = evaluate_triage_routing()
    target = ROOT / "data" / "quality" / "triage-routing-latest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, result


if __name__ == "__main__":
    target, result = write_evidence()
    print(json.dumps(
        {key: result[key] for key in ("cases", "passed", "accuracy", "mode")},
        ensure_ascii=False,
    ))
    failed = [item for item in result["details"] if not item["passed"]]
    for item in failed:
        print(f"  FAIL {item['query']!r}: expected={item['expected']} actual={item['actual']}")
    print(f"evidence={target}")
