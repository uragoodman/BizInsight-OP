#!/usr/bin/env python3
"""Build the P0 dimension-role evaluation dataset.

The dataset is deliberately declarative and offline: it describes the target
semantics for P1/P2 and the expected terminal behaviour.  P0 only uses it to
produce an audit baseline; it must not change runtime parsing.

Usage:
    python -m evaluation.build_dimension_role_cases
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"

SELLER_SCOPE = {
    "seller_id": "p0-seller-001",
    "default_start_date": "2018-08-01",
    "default_end_date": "2018-08-31",
    "metric_name": "gmv",
    "granularity": "day",
    "allowed_metrics": ["gmv", "order_count", "avg_order_value", "freight_value"],
    "allowed_dimensions": ["product_category", "payment_type", "month_position"],
}

PREV_GMV = {
    "authorized_metrics": [{"metric_id": "gmv"}],
    "context_slots": {"metric": {"value": "gmv", "source": "current_turn"}},
}
PREV_BREAKDOWN = {
    "authorized_metrics": [{"metric_id": "gmv"}],
    "context_slots": {
        "metric": {"value": "gmv", "source": "current_turn"},
        "dimension": {"value": "product_category", "source": "current_turn"},
    },
}

SUPPORTED = {"status": "supported", "reason_code": None}
SUPPORTED_SCOPED = {
    "status": "supported",
    "reason_code": None,
    "note": "scoped repository supports product_category; implementation metadata drift",
}
DEGENERATE = {
    "status": "degenerate",
    "reason_code": "dimension_degenerate_in_scope",
}
UNSUPPORTED = {
    "status": "unsupported",
    "reason_code": "dimension_not_supported_by_metric",
}
REGION = {
    "status": "unsupported",
    "reason_code": "region_filter_not_supported",
}
UNKNOWN = {
    "status": "unknown",
    "reason_code": "support_resolver_pending",
}
PLATFORM_SELLER = {
    "status": "supported",
    "reason_code": None,
    "note": "requires P2 registry addition for seller as a formal dimension",
}


def mention(text: str, canonical: str, role: str, *, note: str | None = None) -> dict:
    item = {"text": text, "canonical": canonical, "role": role}
    if note:
        item["note"] = note
    return item


def contract(
    metric_id: str | None,
    dimension: str | None,
    operation_intent: str = "ordinary_metric_query",
    comparison: dict | None = None,
) -> dict:
    return {
        "metric_id": metric_id,
        "dimension": dimension,
        "operation_intent": operation_intent,
        "comparison": comparison or {"kind": "none"},
    }


def case(
    case_id: str,
    tier: str,
    category: str,
    text: str,
    *,
    expected: dict,
    scope: dict | None = None,
    context: dict | None = None,
    notes: str | None = None,
) -> dict:
    item = {
        "case_id": case_id,
        "tier": tier,
        "category": category,
        "input": text,
        "expected": expected,
    }
    if scope is not None:
        item["scope"] = scope
    if context is not None:
        item["context"] = {"previous_turn_context": context}
    if notes:
        item["notes"] = notes
    return item


def expected(
    mentions: list[dict],
    roles: dict[str, str],
    support: dict[str, dict],
    contract_value: dict,
    terminal: str,
    *,
    reason: str | None = None,
) -> dict:
    value = {
        "mentions": mentions,
        "roles": roles,
        "support": support,
        "contract": contract_value,
        "terminal": terminal,
    }
    if reason:
        value["reason"] = reason
    return value


def build_cases() -> list[dict]:
    cases: list[dict] = []

    # ------------------------------------------------------------------
    # Core regression set (12)
    # ------------------------------------------------------------------
    cases.append(
        case(
            "core-01",
            "core",
            "original_failure",
            "请调查本会话当前授权店铺的订单量变化：基线期间为 2018 年 7 月，当前期间为 2018 年 8 月。"
            "先核对两期订单量趋势和总变化，再按商品品类展示统计贡献，指出贡献最大的正向和负向品类。",
            scope=SELLER_SCOPE,
            expected=expected(
                [
                    mention("店铺", "seller", "reference"),
                    mention("订单量", "order_count", "reference", note="metric"),
                    mention("商品品类", "product_category", "grouping"),
                ],
                {"seller": "reference", "product_category": "grouping"},
                {"product_category": SUPPORTED_SCOPED},
                contract(
                    "order_count",
                    "product_category",
                    "dimensional_statistical_contribution",
                    {
                        "kind": "period_pair",
                        "baseline": {
                            "start_date": "2018-07-01",
                            "end_date": "2018-07-31",
                        },
                        "current": {
                            "start_date": "2018-08-01",
                            "end_date": "2018-08-31",
                        },
                        "granularity": "month",
                    },
                ),
                "clarify",
                reason="comparison_outside_locked_scope",
            ),
            notes="Dimension is product_category only; baseline July is outside the August-only locked scope.",
        )
    )
    cases.append(
        case(
            "core-02",
            "core",
            "scope_reference",
            "店铺范围内的 GMV",
            scope=SELLER_SCOPE,
            expected=expected(
                [
                    mention("店铺", "seller", "reference"),
                    mention("GMV", "gmv", "reference", note="metric"),
                ],
                {"seller": "reference"},
                {},
                contract("gmv", None),
                "auto",
            ),
            notes="A scope phrase must not create a grouping dimension.",
        )
    )
    cases.append(
        case(
            "core-03",
            "core",
            "degenerate_dimension",
            "按店铺拆解 GMV",
            scope=SELLER_SCOPE,
            expected=expected(
                [
                    mention("店铺", "seller", "grouping"),
                    mention("GMV", "gmv", "reference", note="metric"),
                ],
                {"seller": "grouping"},
                {"seller": DEGENERATE},
                contract("gmv", "seller"),
                "clarify",
                reason="dimension_degenerate_in_scope",
            ),
            notes="Single-seller scope cannot form a meaningful seller comparison; explain, do not silently total.",
        )
    )
    cases.append(
        case(
            "core-04",
            "core",
            "ambiguous_dimension",
            "按店铺和商品品类拆解订单量",
            scope=SELLER_SCOPE,
            expected=expected(
                [
                    mention("店铺", "seller", "grouping"),
                    mention("商品品类", "product_category", "grouping"),
                    mention("订单量", "order_count", "reference", note="metric"),
                ],
                {"seller": "grouping", "product_category": "grouping"},
                {"seller": DEGENERATE, "product_category": SUPPORTED_SCOPED},
                contract("order_count", None),
                "clarify",
                reason="ambiguous_dimension",
            ),
            notes="Two genuine grouping dimensions still require clarification.",
        )
    )
    cases.append(
        case(
            "core-05",
            "core",
            "metric_noun_interference",
            "平均每单商品数的统计贡献",
            expected=expected(
                [
                    mention(
                        "商品", "product", "reference", note="metric_noun_component"
                    ),
                    mention(
                        "平均每单商品数",
                        "avg_items_per_order",
                        "reference",
                        note="metric",
                    ),
                ],
                {"product": "reference"},
                {},
                contract(
                    "avg_items_per_order", None, "dimensional_statistical_contribution"
                ),
                "clarify",
                reason="dimension_required",
            ),
            notes="'商品' is part of the metric phrase, not a grouping dimension.",
        )
    )
    cases.append(
        case(
            "core-06",
            "core",
            "definition_intent",
            "GMV 按地区的口径是什么",
            expected=expected(
                [
                    mention("GMV", "gmv", "reference", note="metric"),
                    mention("地区", "customer_state", "reference"),
                ],
                {"customer_state": "reference"},
                {},
                contract("gmv", None),
                "auto",
            ),
            notes="Definition question; '按地区' describes the caliber, not a grouping request.",
        )
    )
    cases.append(
        case(
            "core-07",
            "core",
            "standard_grouping",
            "按商品品类统计 GMV",
            expected=expected(
                [
                    mention("商品品类", "product_category", "grouping"),
                    mention("GMV", "gmv", "reference", note="metric"),
                ],
                {"product_category": "grouping"},
                {"product_category": SUPPORTED},
                contract("gmv", "product_category"),
                "auto",
            ),
        )
    )
    cases.append(
        case(
            "core-08",
            "core",
            "negation",
            "不要按店铺拆解，按商品品类",
            expected=expected(
                [
                    mention("店铺", "seller", "reference", note="negated"),
                    mention("商品品类", "product_category", "grouping"),
                ],
                {"seller": "reference", "product_category": "grouping"},
                {"product_category": SUPPORTED},
                contract(None, "product_category"),
                "auto",
            ),
            notes="Negated seller mention is excluded; product_category remains.",
        )
    )
    cases.append(
        case(
            "core-09",
            "core",
            "role_conflict_contribution",
            "本会话当前授权店铺和商品品类各贡献多少",
            scope=SELLER_SCOPE,
            expected=expected(
                [
                    mention("店铺", "seller", "reference"),
                    mention("商品品类", "product_category", "grouping"),
                ],
                {"seller": "reference", "product_category": "grouping"},
                {"product_category": SUPPORTED_SCOPED},
                contract(
                    None, "product_category", "dimensional_statistical_contribution"
                ),
                "clarify",
                reason="comparison_periods_incomplete",
            ),
            notes="Seller is a scope reference; contribution still needs comparison periods.",
        )
    )
    cases.append(
        case(
            "core-10",
            "core",
            "platform_two_dimensions",
            "按店铺和商品品类拆解订单量",
            expected=expected(
                [
                    mention("店铺", "seller", "grouping"),
                    mention("商品品类", "product_category", "grouping"),
                    mention("订单量", "order_count", "reference", note="metric"),
                ],
                {"seller": "grouping", "product_category": "grouping"},
                {"seller": PLATFORM_SELLER, "product_category": SUPPORTED},
                contract("order_count", None),
                "clarify",
                reason="ambiguous_dimension",
            ),
            notes="Platform scope permits seller as a conditional dimension, but two grouping dimensions still clarify.",
        )
    )
    cases.append(
        case(
            "core-11",
            "core",
            "platform_seller_grouping",
            "各店铺的 GMV",
            expected=expected(
                [
                    mention("店铺", "seller", "grouping"),
                    mention("GMV", "gmv", "reference", note="metric"),
                ],
                {"seller": "grouping"},
                {"seller": PLATFORM_SELLER},
                contract("gmv", "seller"),
                "auto",
            ),
            notes="Multi-seller/platform scope is where seller grouping is meaningful.",
        )
    )
    cases.append(
        case(
            "core-12",
            "core",
            "region_filter",
            "华东地区的 GMV",
            expected=expected(
                [
                    mention("地区", "customer_state", "filter"),
                    mention("GMV", "gmv", "reference", note="metric"),
                ],
                {"customer_state": "filter"},
                {"customer_state": REGION},
                contract("gmv", None),
                "clarify",
                reason="region_filter_not_supported",
            ),
            notes="Region is a filter, not a grouping dimension; current scope does not support it.",
        )
    )

    # ------------------------------------------------------------------
    # Extended set (37)
    # ------------------------------------------------------------------
    colloquial = [
        ("看看各个品类卖得咋样", "gmv", "product_category", SUPPORTED, "auto", None),
        ("最近店铺卖得怎么样", "gmv", None, {}, "auto", None),
        (
            "帮我瞅瞅最近哪些品类贡献最大",
            "gmv",
            "product_category",
            SUPPORTED,
            "clarify",
            "comparison_periods_incomplete",
        ),
        ("这个月各支付方式怎么样", "gmv", "payment_type", SUPPORTED, "auto", None),
        ("最近生意咋样，按州看看", "gmv", "customer_state", SUPPORTED, "auto", None),
        (
            "哪个品类拖后腿了",
            "gmv",
            "product_category",
            SUPPORTED,
            "clarify",
            "ambiguous_operation",
        ),
    ]
    for index, (text, metric, dimension, support, terminal, reason) in enumerate(
        colloquial, start=1
    ):
        mentions = []
        roles: dict[str, str] = {}
        support_map: dict[str, dict] = {}
        if dimension:
            alias = {
                "product_category": "品类",
                "payment_type": "支付方式",
                "customer_state": "州",
            }[dimension]
            mentions.append(mention(alias, dimension, "grouping"))
            roles[dimension] = "grouping"
            support_map[dimension] = support
        if "店铺" in text:
            mentions.append(mention("店铺", "seller", "reference"))
            roles["seller"] = "reference"
        mentions.append(mention(metric.upper(), metric, "reference", note="metric"))
        cases.append(
            case(
                f"ext-{index:02d}",
                "extended",
                "colloquial",
                text,
                context=PREV_GMV if metric == "gmv" else None,
                expected=expected(
                    mentions,
                    roles,
                    support_map,
                    contract(
                        metric,
                        dimension,
                        "dimensional_statistical_contribution"
                        if reason == "comparison_periods_incomplete"
                        or reason == "ambiguous_operation"
                        else "ordinary_metric_query",
                    ),
                    terminal,
                    reason=reason,
                ),
            )
        )

    synonym = [
        (
            "最近卖得怎么样，华东地区",
            "gmv",
            None,
            {"customer_state": REGION},
            "clarify",
            "region_filter_not_supported",
        ),
        (
            "华东那边最近生意如何",
            "gmv",
            None,
            {"customer_state": REGION},
            "clarify",
            "region_filter_not_supported",
        ),
        (
            "按类目拆分一下 GMV",
            "gmv",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
        (
            "不同品类的 GMV 是多少",
            "gmv",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
        (
            "支付方式维度看一下订单量",
            "order_count",
            "payment_type",
            {"payment_type": SUPPORTED},
            "auto",
            None,
        ),
        (
            "商品类别分布",
            "gmv",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
    ]
    for offset, (text, metric, dimension, support, terminal, reason) in enumerate(
        synonym, start=7
    ):
        mentions = []
        roles: dict[str, str] = {}
        if dimension:
            alias = {"product_category": "品类", "payment_type": "支付方式"}[dimension]
            mentions.append(mention(alias, dimension, "grouping"))
            roles[dimension] = "grouping"
        if "地区" in text or "华东" in text:
            mentions.append(mention("地区", "customer_state", "filter"))
            roles["customer_state"] = "filter"
        mentions.append(mention(metric.upper(), metric, "reference", note="metric"))
        cases.append(
            case(
                f"ext-{offset:02d}",
                "extended",
                "synonym_word_order",
                text,
                context=PREV_GMV if metric == "gmv" and "GMV" not in text else None,
                expected=expected(
                    mentions,
                    roles,
                    support,
                    contract(metric, dimension),
                    terminal,
                    reason=reason,
                ),
            )
        )

    multi_turn = [
        (
            "那华南呢？",
            PREV_GMV,
            "gmv",
            None,
            {"customer_state": REGION},
            "clarify",
            "region_filter_not_supported",
        ),
        (
            "那按品类呢？",
            PREV_GMV,
            "gmv",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
        (
            "再按支付方式看看",
            PREV_BREAKDOWN,
            "gmv",
            "payment_type",
            {"payment_type": SUPPORTED},
            "auto",
            None,
        ),
        ("还是那个店铺，换成订单量", PREV_GMV, "order_count", None, {}, "auto", None),
        (
            "它的统计贡献呢？",
            PREV_BREAKDOWN,
            "gmv",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
    ]
    for offset, (
        text,
        context,
        metric,
        dimension,
        support,
        terminal,
        reason,
    ) in enumerate(multi_turn, start=13):
        mentions = []
        roles: dict[str, str] = {}
        if dimension:
            alias = {"product_category": "品类", "payment_type": "支付方式"}[dimension]
            mentions.append(mention(alias, dimension, "grouping"))
            roles[dimension] = "grouping"
        if "华南" in text:
            mentions.append(mention("华南", "customer_state", "filter"))
            roles["customer_state"] = "filter"
        if "店铺" in text:
            mentions.append(mention("店铺", "seller", "reference"))
            roles["seller"] = "reference"
        mentions.append(mention(metric.upper(), metric, "reference", note="metric"))
        cases.append(
            case(
                f"ext-{offset:02d}",
                "extended",
                "multi_turn",
                text,
                context=context,
                expected=expected(
                    mentions,
                    roles,
                    support,
                    contract(
                        metric,
                        dimension,
                        "dimensional_statistical_contribution"
                        if "贡献" in text
                        else "ordinary_metric_query",
                    ),
                    terminal,
                    reason=reason,
                ),
                notes="Multi-turn context is preserved in the case context.",
            )
        )

    negation = [
        (
            "不要按店铺拆，按商品品类",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
        (
            "不看地区，按品类看",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
        ("先不要拆分，给我总数", None, {}, "auto", None),
        (
            "别按品类了，按支付方式",
            "payment_type",
            {"payment_type": SUPPORTED},
            "auto",
            None,
        ),
        (
            "不是按店铺，是按站点？",
            None,
            {"seller": UNKNOWN},
            "clarify",
            "dimension_required",
        ),
    ]
    for offset, (text, dimension, support, terminal, reason) in enumerate(
        negation, start=18
    ):
        mentions = []
        roles: dict[str, str] = {}
        if dimension:
            alias = {"product_category": "品类", "payment_type": "支付方式"}[dimension]
            mentions.append(mention(alias, dimension, "grouping"))
            roles[dimension] = "grouping"
        if "店铺" in text:
            mentions.append(mention("店铺", "seller", "reference", note="negated"))
            roles["seller"] = "reference"
        if "地区" in text:
            mentions.append(
                mention("地区", "customer_state", "reference", note="negated")
            )
            roles["customer_state"] = "reference"
        if "站点" in text:
            mentions.append(mention("站点", "unresolved", "unresolved"))
            roles["unresolved"] = "unresolved"
        cases.append(
            case(
                f"ext-{offset:02d}",
                "extended",
                "negation",
                text,
                expected=expected(
                    mentions,
                    roles,
                    support,
                    contract(None, dimension),
                    terminal,
                    reason=reason,
                ),
            )
        )

    conflicts = [
        (
            "按店铺范围的 GMV",
            "seller",
            "reference",
            "verify",
            "role_conflict",
            {"seller": DEGENERATE},
        ),
        (
            "当前授权店铺按店铺拆解",
            "seller",
            "grouping",
            "clarify",
            "dimension_degenerate_in_scope",
            {"seller": DEGENERATE},
        ),
        (
            "该店铺和品类的贡献",
            "product_category",
            "grouping",
            "clarify",
            "comparison_periods_incomplete",
            {"seller": SUPPORTED, "product_category": SUPPORTED},
        ),
        (
            "在已授权店铺范围内按品类拆解",
            "product_category",
            "grouping",
            "auto",
            None,
            {"product_category": SUPPORTED},
        ),
        (
            "本会话店铺按支付方式看",
            "payment_type",
            "grouping",
            "auto",
            None,
            {"payment_type": SUPPORTED},
        ),
    ]
    for offset, (text, dimension, role, terminal, reason, support) in enumerate(
        conflicts, start=23
    ):
        mentions = []
        roles: dict[str, str] = {}
        if "店铺" in text:
            mentions.append(mention("店铺", "seller", "reference"))
            roles["seller"] = "reference"
        if dimension == "seller":
            mentions.append(
                mention("店铺", "seller", role if role == "grouping" else "reference")
            )
            roles["seller"] = role
        elif dimension:
            alias = {"product_category": "品类", "payment_type": "支付方式"}[dimension]
            mentions.append(mention(alias, dimension, role))
            roles[dimension] = role
        cases.append(
            case(
                f"ext-{offset:02d}",
                "extended",
                "role_conflict",
                text,
                scope=SELLER_SCOPE,
                expected=expected(
                    mentions,
                    roles,
                    support,
                    contract(None, dimension),
                    terminal,
                    reason=reason,
                ),
            )
        )

    metric_nouns = [
        ("平均每单商品数", "avg_items_per_order", None, {}, "auto", None),
        ("平均商品售价的趋势", "seller_avg_item_price", None, {}, "auto", None),
        (
            "商品数量最多的品类",
            "gmv",
            "product_category",
            {"product_category": SUPPORTED},
            "auto",
            None,
        ),
        (
            "每个订单的商品数的贡献",
            "avg_items_per_order",
            None,
            {},
            "clarify",
            "dimension_required",
        ),
    ]
    for offset, (text, metric, dimension, support, terminal, reason) in enumerate(
        metric_nouns, start=28
    ):
        mentions = [
            mention("商品", "product", "reference", note="metric_noun_component")
        ]
        roles = {"product": "reference"}
        if dimension:
            mentions.append(mention("品类", dimension, "grouping"))
            roles[dimension] = "grouping"
        mentions.append(mention(metric, metric, "reference", note="metric"))
        cases.append(
            case(
                f"ext-{offset:02d}",
                "extended",
                "metric_noun",
                text,
                expected=expected(
                    mentions,
                    roles,
                    support,
                    contract(
                        metric,
                        dimension,
                        "dimensional_statistical_contribution"
                        if "贡献" in text
                        else "ordinary_metric_query",
                    ),
                    terminal,
                    reason=reason,
                ),
            )
        )

    unsupported = [
        (
            "按客户城市拆解 GMV",
            "customer_city",
            UNSUPPORTED,
            "clarify",
            "dimension_not_supported_by_metric",
        ),
        (
            "按卖家州拆解 GMV",
            "seller_state",
            UNSUPPORTED,
            "clarify",
            "dimension_not_supported_by_metric",
        ),
        (
            "按地区拆解 GMV",
            "customer_state",
            REGION,
            "clarify",
            "region_filter_not_supported",
        ),
        (
            "按店铺拆解",
            "seller",
            DEGENERATE,
            "clarify",
            "dimension_degenerate_in_scope",
        ),
    ]
    for offset, (text, dimension, support, terminal, reason) in enumerate(
        unsupported, start=32
    ):
        alias = {
            "customer_city": "客户城市",
            "seller_state": "卖家州",
            "customer_state": "地区",
            "seller": "店铺",
        }[dimension]
        cases.append(
            case(
                f"ext-{offset:02d}",
                "extended",
                "unsupported_dimension",
                text,
                scope=SELLER_SCOPE,
                expected=expected(
                    [
                        mention(alias, dimension, "grouping"),
                        mention("GMV", "gmv", "reference", note="metric"),
                    ],
                    {dimension: "grouping"},
                    {dimension: support},
                    contract("gmv", dimension),
                    terminal,
                    reason=reason,
                ),
            )
        )

    cases.append(
        case(
            "ext-36",
            "extended",
            "english",
            "by seller breakdown of GMV",
            scope=SELLER_SCOPE,
            expected=expected(
                [
                    mention("seller", "seller", "grouping"),
                    mention("GMV", "gmv", "reference", note="metric"),
                ],
                {"seller": "grouping"},
                {"seller": DEGENERATE},
                contract("gmv", "seller"),
                "clarify",
                reason="dimension_degenerate_in_scope",
            ),
        )
    )
    cases.append(
        case(
            "ext-37",
            "extended",
            "english",
            "current seller's GMV",
            scope=SELLER_SCOPE,
            expected=expected(
                [
                    mention("seller", "seller", "reference"),
                    mention("GMV", "gmv", "reference", note="metric"),
                ],
                {"seller": "reference"},
                {},
                contract("gmv", None),
                "auto",
            ),
        )
    )

    return cases


def main() -> None:
    cases = build_cases()
    ids = [item["case_id"] for item in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case_id")
    core = [item for item in cases if item["tier"] == "core"]
    extended = [item for item in cases if item["tier"] == "extended"]
    if len(core) != 12 or len(extended) != 37:
        raise SystemExit(
            f"expected 12 core + 37 extended, got {len(core)} + {len(extended)}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) for item in cases
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(cases)} cases to {OUTPUT} ({len(core)} core, {len(extended)} extended)"
    )


if __name__ == "__main__":
    main()
