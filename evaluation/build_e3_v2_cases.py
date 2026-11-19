# -*- coding: utf-8 -*-
"""Build the 75-case E3 v2 stress set from the 24-case v1 seed.

The v1 cases remain the source of truth for the original six real-provider
cases. This builder adds gold facts to the remaining v1 cases and generates
structured easy/medium/hard cases with the same item contract used by the
Packer and real-provider runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

V1_PATH = Path("evaluation/datasets/e3_context_cases_v1.jsonl")
V2_PATH = Path("evaluation/datasets/e3_context_cases_v2.jsonl")

_BUDGETS = {
    "easy": {
        "model_context_window": 32768,
        "reserved_output_tokens": 1024,
        "agent_input_cap": 3000,
        "system_prompt_tokens": 500,
        "available_context_tokens": 3000,
    },
    "medium": {
        "model_context_window": 32768,
        "reserved_output_tokens": 1024,
        "agent_input_cap": 3000,
        "system_prompt_tokens": 500,
        "available_context_tokens": 200,
    },
    "hard": {
        "model_context_window": 32768,
        "reserved_output_tokens": 1024,
        "agent_input_cap": 3000,
        "system_prompt_tokens": 500,
        "available_context_tokens": 1100,
    },
}

_EXISTING_FACTS: dict[str, list[dict[str, Any]]] = {
    "e3-easy-03": [
        {"id": "order-trend-supported", "text": "订单量支持按日趋势", "aliases": ["按日趋势"]},
    ],
    "e3-easy-04": [
        {"id": "refund-formula", "text": "退款率 = 退款订单数 / 支付订单数", "aliases": ["退款订单数 / 支付订单数", "退款订单数/支付订单数"]},
    ],
    "e3-easy-05": [
        {"id": "payment-credit-card", "text": "支付方式包含信用卡", "aliases": ["信用卡"]},
        {"id": "payment-boleto", "text": "支付方式包含 boleto", "aliases": ["boleto"]},
        {"id": "payment-voucher", "text": "支付方式包含 voucher", "aliases": ["voucher"]},
    ],
    "e3-easy-06": [
        {"id": "weekly-modules", "text": "周报包含 GMV、订单、履约和差评", "aliases": ["GMV、订单、履约和差评", "履约和差评"]},
    ],
    "e3-medium-03": [
        {"id": "date-2026-08", "text": "分析月份为 2026-08", "aliases": ["2026-08"]},
        {"id": "scope-platform", "text": "范围为 platform", "aliases": ["platform", "平台范围"]},
        {"id": "data-version", "text": "数据版本为 2026-08-31", "aliases": ["2026-08-31"]},
    ],
    "e3-medium-04": [
        {"id": "weekly-report", "text": "生成本周经营周报", "aliases": ["周报"]},
        {"id": "data-version", "text": "数据版本为 2026-08-31", "aliases": ["2026-08-31"]},
    ],
    "e3-medium-05": [
        {"id": "date-2026-08", "text": "分析月份为 2026-08", "aliases": ["2026-08"]},
        {"id": "refund-trend", "text": "查看退款率趋势", "aliases": ["退款率", "趋势"]},
    ],
    "e3-medium-06": [
        {"id": "gmv-decline", "text": "解释 GMV 下降原因", "aliases": ["GMV", "下降"]},
        {"id": "date-2026-08", "text": "分析月份为 2026-08", "aliases": ["2026-08"]},
    ],
    "e3-medium-07": [
        {"id": "product-dimension", "text": "按商品品类拆解", "aliases": ["商品品类", "品类"]},
        {"id": "date-2026-08", "text": "分析月份为 2026-08", "aliases": ["2026-08"]},
    ],
    "e3-medium-08": [
        {"id": "delay-rate", "text": "查询履约延迟率", "aliases": ["履约延迟率"]},
        {"id": "date-2026-08", "text": "分析月份为 2026-08", "aliases": ["2026-08"]},
    ],
}

_HARD_COMMON_FACTS: list[dict[str, Any]] = [
    {"id": "gmv-down-12", "text": "GMV 环比下降 12%", "aliases": ["下降 12%", "下降12%", "降 12%", "降12%"]},
    {"id": "refund-up", "text": "退款率上升 2.1pp", "aliases": [], "patterns": [r"退款率.{0,8}(?:上升|升|上涨|增加|\+)\s*2\.1\s*(?:pp|个百分点)"]},
    {"id": "delay-up", "text": "履约延迟率上升 1.4pp", "aliases": [], "patterns": [r"履约延迟率.{0,8}(?:上升|升|上涨|增加|\+)\s*1\.4\s*(?:pp|个百分点)"]},
    {"id": "scope-platform", "text": "范围为 platform", "aliases": ["platform", "平台范围"]},
    {"id": "data-version", "text": "数据版本为 2026-08-31", "aliases": ["2026-08-31"]},
]

_EASY_METRICS = [
    ("GMV", "GMV 是成交总额，口径以指标注册表为准。", "GMV 是成交总额。", ["成交总额"]),
    ("客单价", "客单价 = GMV / 订单数。", "客单价 = GMV / 订单数。", ["GMV / 订单数", "GMV/订单数"]),
    ("订单量", "订单量按支付订单去重。", "订单量按支付订单去重。", ["支付订单去重"]),
    ("退款率", "退款率 = 退款订单数 / 支付订单数。", "退款率 = 退款订单数 / 支付订单数。", ["退款订单数 / 支付订单数", "退款订单数/支付订单数"]),
    ("履约延迟率", "履约延迟率 = 延迟送达订单数 / 已送达订单数。", "履约延迟率按延迟送达订单数 / 已送达订单数计算。", ["延迟送达订单数", "已送达订单数"]),
    ("取消率", "取消率 = 取消订单数 / 支付订单数。", "取消率按取消订单数 / 支付订单数计算。", ["取消订单数", "支付订单数"]),
    ("平均评论分", "平均评论分是订单评论分数的平均值。", "平均评论分是评论分数的平均值。", ["评论分数", "平均值"]),
    ("差评率", "差评率 = 评分小于等于 2 的订单数 / 有评论订单数。", "差评率按评分小于等于 2 的订单占比计算。", ["评分小于等于 2", "有评论订单"]),
    ("运费率", "运费率 = 运费总额 / GMV。", "运费率按运费总额 / GMV 计算。", ["运费总额", "GMV"]),
    ("客户数", "客户数按 customer_unique_id 去重。", "客户数按 customer_unique_id 去重。", ["customer_unique_id", "去重"]),
    ("复购率", "复购率 = 复购客户数 / 客户数。", "复购率按复购客户数 / 客户数计算。", ["复购客户数", "客户数"]),
    ("平均每单件数", "平均每单件数 = 商品件数 / 订单数。", "平均每单件数按商品件数 / 订单数计算。", ["商品件数", "订单数"]),
    ("支付方式", "支付方式维度包含信用卡、boleto、voucher。", "支付方式包含信用卡、boleto、voucher。", ["信用卡", "boleto", "voucher"]),
    ("周报模块", "周报通常包含 GMV、订单、履约和差评。", "周报包含 GMV、订单、履约和差评。", ["GMV", "履约", "差评"]),
]

_MEDIUM_CASES = [
    ("gmv", "2026-08", "1,234.56", "查询 2026-08 的 GMV", "GMV"),
    ("order_count", "2026-07", "789", "查询 2026-07 的订单量", "订单量"),
    ("avg_order_value", "2026-08", "1.56", "查询 2026-08 的客单价", "客单价"),
    ("refund_rate", "2026-09", "2.8%", "查询 2026-09 的退款率", "退款率"),
    ("delay_rate", "2026-08", "4.2%", "查询 2026-08 的履约延迟率", "履约延迟率"),
    ("cancel_rate", "2026-07", "1.9%", "查询 2026-07 的取消率", "取消率"),
    ("review_score", "2026-08", "4.3", "查询 2026-08 的平均评论分", "平均评论分"),
    ("bad_review_rate", "2026-09", "6.1%", "查询 2026-09 的差评率", "差评率"),
    ("freight_rate", "2026-08", "8.4%", "查询 2026-08 的运费率", "运费率"),
    ("customer_count", "2026-07", "456", "查询 2026-07 的客户数", "客户数"),
    ("repeat_purchase_rate", "2026-08", "18.7%", "查询 2026-08 的复购率", "复购率"),
    ("avg_items_per_order", "2026-09", "1.8", "查询 2026-09 的平均每单件数", "平均每单件数"),
    ("gmv", "2026-08", "1,234.56", "按商品品类拆解 2026-08 的 GMV", "GMV"),
    ("order_count", "2026-08", "789", "按支付方式拆解 2026-08 的订单量", "订单量"),
    ("refund_rate", "2026-08", "2.8%", "看一下 2026-08 的退款率趋势", "退款率"),
    ("gmv", "2026-08", "1,234.56", "生成 2026-08 经营周报", "GMV"),
    ("delay_rate", "2026-09", "4.2%", "对比 2026-08 和 2026-09 的履约延迟率", "履约延迟率"),
]

_HARD_SCENARIOS = [
    ("分析 2026-08 退款率上升的根因，并给出可执行建议", "订单量下降和客单价下降", "退款率上升"),
    ("生成 2026-08 平台经营分析报告，包含趋势、贡献和风险", "订单量下降和客单价下降", "经营风险"),
    ("按支付方式分析 2026-08 GMV 变化，并总结证据", "订单量下降和客单价下降", "支付方式"),
    ("按州拆解 2026-08 GMV，并核对 scope 和授权范围", "订单量下降和客单价下降", "customer_state"),
    ("检查 2026-08 指标口径冲突，并给出最终定义", "订单量下降和客单价下降", "指标口径"),
    ("分析 2026-08 差评率和退款率的关联，并列出限制", "订单量下降和客单价下降", "差评率"),
    ("对比 2026-07 与 2026-08 的订单量和客单价，按商品品类拆解", "订单量下降和客单价下降", "商品品类"),
    ("解释 2026-08 履约延迟率异常，并查历史同类事件", "订单量下降和客单价下降", "履约延迟"),
    ("生成周报：GMV、订单、履约、退款和风险，附证据", "订单量下降和客单价下降", "周报"),
    ("分析 GMV 下降的贡献因素，并区分统计分解与业务原因", "订单量下降和客单价下降", "统计分解"),
    ("按客单价变化分析 2026-08 经营压力，并给出风险边界", "订单量下降和客单价下降", "客单价"),
    ("按订单量变化分析 2026-08 履约压力，并列出证据缺口", "订单量下降和客单价下降", "订单量"),
    ("分析 2026-08 运费率异常，并区分口径变化与业务原因", "订单量下降和客单价下降", "运费率"),
    ("按 seller_state 拆解 2026-08 GMV，并核对授权范围", "订单量下降和客单价下降", "seller_state"),
    ("按 customer_city 拆解 2026-08 退款率，并总结限制", "订单量下降和客单价下降", "customer_city"),
    ("按月内日期位置拆解 2026-08 GMV，并解释波动", "订单量下降和客单价下降", "月内日期位置"),
    ("分析 2026-08 GMV 趋势中的异常点，并给出下一步动作", "订单量下降和客单价下降", "趋势"),
    ("检查 2026-08 数据不足场景，明确哪些结论不能下", "订单量下降和客单价下降", "数据不足"),
    ("生成 2026-08 风险报告，覆盖退款、履约和差评", "订单量下降和客单价下降", "风险"),
    ("按支付方式拆解 2026-08 退款率，并核对证据边界", "订单量下降和客单价下降", "payment_type"),
]


def _budget(difficulty: str) -> dict[str, int]:
    return dict(_BUDGETS[difficulty])


def _case(
    case_id: str,
    difficulty: str,
    question: str,
    items: list[dict[str, Any]],
    required_items: list[str],
    optional_items: list[str],
    facts: list[dict[str, Any]],
    *,
    agent: str = "triage",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "agent": agent,
        "model": "test-model",
        "budget": _budget(difficulty),
        "items": items,
        "expected_required_item_ids": required_items,
        "expected_optional_item_ids": optional_items,
        "notes": notes,
        "expected_facts": facts,
    }


def _item(item_id: str, source: str, content: str, *, priority: int, required: bool,
          evidence_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "source": source,
        "content": content,
        "required": required,
        "priority": priority,
        "evidence_ids": evidence_ids or [],
    }


def _easy_case(index: int, metric: str, definition: str, answer_fact: str, aliases: list[str]) -> dict[str, Any]:
    question = f"{metric}怎么定义？" if metric not in {"支付方式", "周报模块"} else (
        "支付方式有哪些？" if metric == "支付方式" else "本周经营报告需要哪些模块？"
    )
    items = [
        _item("q", "current_question", question, priority=95, required=True),
        _item("registry", "metric_registry", definition, priority=85, required=True, evidence_ids=["metric-registry"]),
        _item("rag", "rag_evidence", "知识库说明：指标口径以注册表为准，回答时引用指标定义。", priority=60, required=False, evidence_ids=[f"rag-easy-{index}"]),
    ]
    return _case(
        f"e3v2-easy-{index:02d}", "easy", question, items,
        ["q", "registry"], ["rag"],
        [{"id": f"{metric}-definition", "text": answer_fact, "aliases": aliases}],
        notes="metric definition with compact registry evidence",
    )


def _medium_case(index: int, metric: str, day: str, value: str, question: str, label: str) -> dict[str, Any]:
    version = "2026-09-30"
    task = f"当前任务：{day} 平台分析；scope=platform；数据版本={version}。"
    evidence = f"当前查询证据：{day} 的 {label}={value}；scope=platform；数据版本={version}。"
    items = [
        _item("q", "current_question", question, priority=95, required=True),
        _item("task", "task", task, priority=90, required=True, evidence_ids=["task-scope"]),
        _item("registry", "metric_registry", "指标注册表：GMV/订单/退款率/履约延迟率均支持日期和维度筛选。", priority=85, required=True, evidence_ids=["metric-registry"]),
        _item("ev1", "tool_evidence", evidence, priority=80, required=True, evidence_ids=["ev-current"]),
        _item("rag", "rag_evidence", "口径说明：GMV 不含运费；订单量按支付订单去重；退款率按退款订单占比计算。", priority=60, required=False, evidence_ids=[f"rag-medium-{index}"]),
        _item("hist", "historical_analysis", f"历史分析：上月 {label} 变化主要来自履约延迟和退款。", priority=55, required=False, evidence_ids=[f"hist-medium-{index}"]),
        _item("pref", "mem0", "偏好：回答保持简洁，先给结论。", priority=65, required=False, evidence_ids=[f"pref-medium-{index}"]),
    ]
    facts = [
        {"id": "date", "text": f"分析日期为 {day}", "aliases": [day]},
        {"id": "metric", "text": f"指标为 {label}", "aliases": [label]},
        {"id": "value", "text": f"{label} 为 {value}", "aliases": [value]},
        {"id": "scope", "text": "范围为 platform", "aliases": ["platform", "平台范围"]},
        {"id": "data-version", "text": f"数据版本为 {version}", "aliases": [version]},
    ]
    return _case(
        f"e3v2-medium-{index:02d}", "medium", question, items,
        ["q", "task", "registry", "ev1"], ["rag", "hist", "pref"],
        facts, agent="data_query", notes="normal data-analysis request with task state and evidence",
    )


def _hard_case(index: int, scenario_index: int, day: str, question: str, driver: str, dimension: str) -> dict[str, Any]:
    version = "2026-08-31"
    scope = "platform"
    ev1 = f"当前证据 A：GMV 环比下降 12%，主要由{driver}贡献。"
    ev2 = "当前证据 B：退款率上升 2.1pp，履约延迟率上升 1.4pp。"
    ev3 = f"当前证据 C：{dimension} 维度中，{driver} 是主要拖累；scope={scope}；数据版本={version}。"
    items = [
        _item("q", "current_question", question, priority=95, required=True),
        _item("task", "task", f"当前任务：{day} 平台范围；历史任务上下文可用；scope={scope}；数据版本={version}。", priority=90, required=True, evidence_ids=["task-scope"]),
        _item("registry", "metric_registry", "指标注册表：GMV、订单量、客单价、退款率、履约延迟率、支付方式、商品品类、customer_state。", priority=85, required=True, evidence_ids=["metric-registry"]),
        _item("ev1", "tool_evidence", ev1, priority=80, required=True, evidence_ids=["ev-a"]),
        _item("ev2", "tool_evidence", ev2, priority=79, required=True, evidence_ids=["ev-b"]),
        _item("ev3", "tool_evidence", ev3, priority=78, required=True, evidence_ids=["ev-c"]),
    ]
    for rag_index in range(1, 11):
        items.append(_item(
            f"rag-{rag_index}", "rag_evidence",
            (
                f"历史知识文档 {rag_index}（{dimension}）：该文档记录过往同类分析的口径、样本范围、趋势判断、"
                "贡献拆解、异常处理、证据边界和复盘结论。分析时应先确认数据版本与 scope，再检查指标口径、"
                "日期范围、维度授权和缺失值；对统计贡献与业务原因分别表述，禁止把相关性写成因果。"
                "若证据不足，应明确限制并给出下一步补数建议；若存在多个假设，应保留可验证的替代解释。"
            ),
            priority=60 - rag_index, required=False, evidence_ids=[f"rag-{rag_index}"],
        ))
    facts = [
        {"id": "gmv-down-12", "text": "GMV 环比下降 12%", "aliases": ["下降 12%", "下降12%", "降 12%", "降12%"]},
        {"id": "refund-up", "text": "退款率上升 2.1pp", "aliases": [], "patterns": [r"退款率.{0,8}(?:上升|升|上涨|增加|\+)\s*2\.1\s*(?:pp|个百分点)"]},
        {"id": "delay-up", "text": "履约延迟率上升 1.4pp", "aliases": [], "patterns": [r"履约延迟率.{0,8}(?:上升|升|上涨|增加|\+)\s*1\.4\s*(?:pp|个百分点)"]},
        {"id": "driver", "text": driver, "aliases": [driver]},
        {"id": "dimension", "text": dimension, "aliases": [dimension]},
        {"id": "scope-platform", "text": "范围为 platform", "aliases": ["platform", "平台范围"]},
        {"id": "data-version", "text": f"数据版本为 {version}", "aliases": [version]},
    ]
    return _case(
        f"e3v2-hard-{index:02d}", "hard", question, items,
        ["q", "task", "registry", "ev1", "ev2", "ev3"],
        [f"rag-{i}" for i in range(1, 11)],
        facts, agent="report", notes=f"hard scenario {scenario_index}: {dimension}",
    )


def build() -> list[dict[str, Any]]:
    existing = [
        json.loads(line)
        for line in V1_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for case in existing:
        if not case.get("expected_facts"):
            case["expected_facts"] = list(_EXISTING_FACTS.get(case["case_id"], _HARD_COMMON_FACTS))

    added: list[dict[str, Any]] = []
    for index, (metric, definition, answer_fact, aliases) in enumerate(_EASY_METRICS, start=7):
        added.append(_easy_case(index, metric, definition, answer_fact, aliases))
    for index, (metric, day, value, question, label) in enumerate(_MEDIUM_CASES, start=9):
        added.append(_medium_case(index, metric, day, value, question, label))
    for index, (question, driver, dimension) in enumerate(_HARD_SCENARIOS, start=11):
        added.append(_hard_case(index, index, "2026-08", question, driver, dimension))
    return existing + added


def main() -> int:
    cases = build()
    counts = {difficulty: sum(1 for case in cases if case["difficulty"] == difficulty) for difficulty in ("easy", "medium", "hard")}
    missing_facts = [case["case_id"] for case in cases if not case.get("expected_facts")]
    if len(cases) != 75 or any(counts[name] != expected for name, expected in (("easy", 20), ("medium", 25), ("hard", 30))):
        raise SystemExit(f"invalid E3 v2 composition: total={len(cases)} counts={counts}")
    if missing_facts:
        raise SystemExit(f"cases missing expected facts: {missing_facts}")
    V2_PATH.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(V2_PATH), "total": len(cases), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
