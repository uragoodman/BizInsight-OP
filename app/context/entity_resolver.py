"""Deterministic entity mention detection, linking, and ambiguity reporting."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.context.semantic_contract import Entity, Mention, Span

EntityType = Literal[
    "metric", "dimension", "business_entity", "semantic_factor", "time", "scope"
]


class EntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected: Entity | None
    candidates: tuple[Entity, ...]
    ambiguous: bool = False
    errors: tuple[str, ...] = ()


_BUSINESS_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("customer", ("客户", "顾客", "customer", "customers")),
    ("order", ("订单", "order", "orders")),
    ("product", ("商品", "产品", "product", "products")),
)
_FACTOR_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("purchase_frequency", ("购买频次", "购买频率", "购买次数", "purchase frequency")),
    ("monetary", ("金额", "消费金额", "消费额", "monetary")),
    ("customer_value", ("客户价值", "顾客价值", "customer value")),
    ("rfm", ("RFM",)),
)
_TIME_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("previous_month", ("上个月", "上月", "last month")),
    ("current_month", ("本月", "这个月", "this month")),
    ("previous_week", ("上周", "上星期", "last week")),
    ("current_week", ("本周", "这周", "this week")),
    ("previous_year", ("去年", "last year")),
    ("today", ("今天", "当日", "today")),
    ("yesterday", ("昨天", "yesterday")),
)
_SCOPE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("seller_scope", ("店铺范围", "商家范围", "店铺内", "店铺", "seller scope")),
    ("platform_scope", ("平台范围", "全平台", "平台级", "platform scope")),
)
_SPECIFIC_METRIC_EXTRA: dict[str, tuple[str, ...]] = {
    "customer_count": ("客户数量", "顾客数量"),
    "order_count": ("订单数量",),
}
_GENERIC_METRIC_ALIASES = {
    "客户", "顾客", "customer", "customers", "订单", "order", "orders",
    "商品", "产品", "product", "products",
}


def _registry_metrics() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Read deterministic aliases from the existing metric registry."""

    try:
        from app.metrics.registry import get_registry

        rows = []
        for definition in get_registry().list():
            aliases = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in (
                        definition.display_name,
                        *definition.aliases,
                        *definition.planning.exact_aliases,
                        *_SPECIFIC_METRIC_EXTRA.get(definition.metric_id, ()),
                    )
                    if str(value or "").strip()
                )
            )
            rows.append((str(definition.metric_id), str(definition.display_name), aliases))
        return tuple(rows)
    except Exception:  # noqa: BLE001 - lexical fallback keeps this module pure and usable
        return (
            ("customer_count", "客户数", ("customer_count", "客户数", "成交客户数", "客户数量")),
            ("order_count", "订单量", ("order_count", "订单量", "订单数", "订单数量", "订单")),
            ("gmv", "成交总额", ("gmv", "成交总额", "交易额", "成交额", "成交金额", "销售额")),
        )


def _dimension_matches(question: str) -> tuple[tuple[str, str, int, int], ...]:
    """Reuse the compiler's established dimension aliases and registry filter."""

    try:
        from app.context.compiler import _dimension_alias_matches

        return tuple(_dimension_alias_matches(question, include_unregistered=True))
    except Exception:  # noqa: BLE001 - retain a small deterministic fallback
        aliases = {
            "product_category": ("商品品类", "商品类别", "品类", "类目"),
            "product": ("商品", "产品", "product"),
            "seller": ("店铺", "商家", "seller"),
            "customer_state": ("客户州", "省份", "地区", "区域"),
            "payment_type": ("支付方式",),
            "customer_city": ("客户城市", "顾客城市"),
        }
        matches: list[tuple[str, str, int, int]] = []
        for canonical, terms in aliases.items():
            for term in terms:
                for match in re.finditer(re.escape(term), question, flags=re.I):
                    matches.append((canonical, term, match.start(), match.end()))
        return tuple(matches)


def _add_alias(
    rows: dict[tuple[int, int], dict[str, Any]],
    question: str,
    alias: str,
    *,
    entity_id: str,
    entity_type: EntityType,
    alias_type: Literal["generic", "specific"] | None,
    confidence: float,
) -> None:
    if not alias:
        return
    for match in re.finditer(re.escape(alias), question, flags=re.IGNORECASE):
        key = (match.start(), match.end())
        row = rows.setdefault(key, {"text": question[match.start() : match.end()], "candidates": {}})
        entity = Entity(
            id=entity_id, type=entity_type, alias_type=alias_type, confidence=confidence
        )
        candidate_key = (entity.id, entity.type)
        previous = row["candidates"].get(candidate_key)
        if previous is None or float(previous.confidence or 0.0) < confidence:
            row["candidates"][candidate_key] = entity


def detect_entity_mentions(question: str) -> tuple[Mention, ...]:
    """Detect original-text mention spans with candidate entity categories."""

    if not question:
        return ()
    matches: dict[tuple[int, int], dict[str, Any]] = {}
    for metric_id, _label, aliases in _registry_metrics():
        for alias in aliases:
            generic = alias.casefold() in _GENERIC_METRIC_ALIASES
            _add_alias(
                matches,
                question,
                alias,
                entity_id=metric_id,
                entity_type="metric",
                alias_type="generic" if generic else "specific",
                confidence=0.36 if generic else 0.94,
            )

    for entity_id, aliases in _BUSINESS_ALIASES:
        for alias in aliases:
            _add_alias(
                matches,
                question,
                alias,
                entity_id=entity_id,
                entity_type="business_entity",
                alias_type="generic",
                confidence=0.78,
            )
    for entity_id, aliases in _FACTOR_ALIASES:
        for alias in aliases:
            _add_alias(
                matches,
                question,
                alias,
                entity_id=entity_id,
                entity_type="semantic_factor",
                alias_type="specific",
                confidence=0.88,
            )
    for entity_id, aliases in _TIME_ALIASES:
        for alias in aliases:
            _add_alias(
                matches,
                question,
                alias,
                entity_id=entity_id,
                entity_type="time",
                alias_type="specific",
                confidence=0.90,
            )
    for entity_id, aliases in _SCOPE_ALIASES:
        for alias in aliases:
            _add_alias(
                matches,
                question,
                alias,
                entity_id=entity_id,
                entity_type="scope",
                alias_type="generic",
                confidence=0.72,
            )

    for dimension_id, _alias, start, end in _dimension_matches(question):
        surface = question[start:end]
        row = matches.setdefault((start, end), {"text": surface, "candidates": {}})
        row["candidates"][(dimension_id, "dimension")] = Entity(
            id=dimension_id, type="dimension", alias_type="specific", confidence=0.82
        )

    # A common generic business noun is not an exact metric request even when
    # legacy YAML still contains it as a broad metric alias. Keep `metric` in
    # candidate_types for auditability, while entity resolution prefers the
    # business entity candidate below.
    spans = sorted(matches)
    selected_spans = [
        span
        for span in spans
        if not any(
            outer[0] <= span[0]
            and outer[1] >= span[1]
            and outer[1] - outer[0] > span[1] - span[0]
            for outer in spans
        )
    ]
    mentions: list[Mention] = []
    for start, end in selected_spans:
        row = matches[(start, end)]
        candidate_types = tuple(
            sorted({candidate.type for candidate in row["candidates"].values()})
        )
        mentions.append(
            Mention(
                text=row["text"],
                span=Span(text=row["text"], start=start, end=end),
                candidate_types=candidate_types,
            )
        )
    return tuple(mentions)


def resolve_entity(mention: Mention, *, question: str | None = None) -> EntityResolution:
    """Link a detected mention to registered entities and report ambiguity."""

    text = mention.text
    exact_matches: list[Entity] = []
    for metric_id, _label, aliases in _registry_metrics():
        if any(text.casefold() == alias.casefold() for alias in aliases):
            generic = text.casefold() in _GENERIC_METRIC_ALIASES
            exact_matches.append(
                Entity(
                    id=metric_id,
                    type="metric",
                    alias_type="generic" if generic else "specific",
                    confidence=0.36 if generic else 0.94,
                )
            )

    for entity_id, aliases in _BUSINESS_ALIASES:
        if any(text.casefold() == alias.casefold() for alias in aliases):
            exact_matches.append(
                Entity(id=entity_id, type="business_entity", alias_type="generic", confidence=0.78)
            )
    for entity_id, aliases in _FACTOR_ALIASES:
        if any(text.casefold() == alias.casefold() for alias in aliases):
            exact_matches.append(
                Entity(id=entity_id, type="semantic_factor", alias_type="specific", confidence=0.88)
            )
    for entity_id, aliases in _TIME_ALIASES:
        if any(text.casefold() == alias.casefold() for alias in aliases):
            exact_matches.append(
                Entity(id=entity_id, type="time", alias_type="specific", confidence=0.90)
            )
    for entity_id, aliases in _SCOPE_ALIASES:
        if any(text.casefold() == alias.casefold() for alias in aliases):
            exact_matches.append(
                Entity(id=entity_id, type="scope", alias_type="generic", confidence=0.72)
            )
    for dimension_id, alias, start, end in _dimension_matches(question or text):
        if (
            question is not None
            and start == mention.span.start
            and end == mention.span.end
            and question[start:end] == text
        ) or (question is None and alias.casefold() == text.casefold()):
            exact_matches.append(
                Entity(id=dimension_id, type="dimension", alias_type="specific", confidence=0.82)
            )

    unique: dict[tuple[str, str], Entity] = {}
    for candidate in exact_matches:
        key = (candidate.id, candidate.type)
        previous = unique.get(key)
        if previous is None or float(candidate.confidence or 0.0) > float(previous.confidence or 0.0):
            unique[key] = candidate
    candidates = tuple(
        sorted(unique.values(), key=lambda item: (-(item.confidence or 0.0), item.type, item.id))
    )
    # Generic metric aliases remain visible candidates but cannot defeat the
    # matching business noun (e.g. 客户 -> customer, never customer_count).
    ranked = [
        candidate
        for candidate in candidates
        if not (candidate.type == "metric" and candidate.alias_type == "generic")
    ]
    if not ranked:
        return EntityResolution(selected=None, candidates=candidates, errors=("unresolved_entity",))
    best_score = max(float(item.confidence or 0.0) for item in ranked)
    best = tuple(item for item in ranked if float(item.confidence or 0.0) == best_score)
    if len({(item.id, item.type) for item in best}) > 1:
        return EntityResolution(selected=None, candidates=candidates, ambiguous=True)
    return EntityResolution(selected=best[0], candidates=candidates)


def resolve_mentions(
    mentions: tuple[Mention, ...] | list[Mention], *, question: str
) -> tuple[EntityResolution, ...]:
    """Resolve each mention independently; do not pick among separate metrics."""

    return tuple(resolve_entity(mention, question=question) for mention in mentions)
