"""Shared, deterministic parsing for explicitly negated user instructions.

This is intentionally a bounded rule parser, not a general natural-language
understander.  It preserves source offsets so the resolver and compiler can
apply the same negation scope without masking unrelated text in the turn.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_NEGATION_RE = re.compile(
    r"不要|别|不需要|不必|无需|不用|不想|不再|"
    r"\b(?:do\s+not|don't|dont|never|avoid|no\s+longer\s+(?:use|follow|apply|adopt)|"
    r"without\s+(?:using|following|applying|adopting)|not\s+(?:use|follow|apply|adopt))\b",
    re.IGNORECASE,
)
_DOUBLE_NEGATION_RE = re.compile(
    r"(?:(?:不要|别|不需要|不必|无需|不用|不想)\s*(?:再)?不|"
    r"(?:不是|并非|并不是)\s*不)|"
    r"\b(?:do\s+not|don't|dont|never|avoid)\s+not\b|"
    r"\b(?:not\s+not|no\s+longer\s+not|without\s+not|not\s+(?:never|avoid|don't|do\s+not))\b",
    re.IGNORECASE,
)
_QUOTED_TEXT_RE = re.compile(r"“[^”]*”|‘[^’]*’|\"[^\"]*\"|'[^']*'|「[^」]*」|『[^』]*』")
# Bare conjunctions only split before a clear positive clause opener. This
# supports natural constructions such as “不要用表格而报告按习惯的格式”
# without splitting ordinary words that happen to contain “而” or “且”.
_POSITIVE_CLAUSE_OPENERS = (
    r"按|用|使用|采用|选择|换成|改成|保持|遵循|依照|报告|汇报|生成|给出|回答|"
    r"答复|回复|输出|呈现|制作|撰写|写|整理|总结|提供"
)
# The supported conjunctions start a new scope.  Repeated negations therefore
# remain negative on both sides (e.g. “不要分析并且不要总结”).
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[，,。.!?！？;；]+|\b(?:but|however|instead|and|then)\b|"
    r"但是|但|而是|不过|并且|并(?=" + _POSITIVE_CLAUSE_OPENERS + r")|同时|以及|然后|随后|而且|"
    # Bare conjunctions split only before a known positive clause opener.
    r"且(?=" + _POSITIVE_CLAUSE_OPENERS + r")|(?<!然)而(?=" + _POSITIVE_CLAUSE_OPENERS + r")",
    re.IGNORECASE,
)
_PREFERENCE_TARGET_RE = re.compile(
    r"我平时|我平常|我常用|我通常|我的习惯|我的方式|我的偏好|老规矩|"
    r"usual|preference|habit|as usual",
    re.IGNORECASE,
)
DEFAULT_QUERY_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "dimension": {
        "product_category": ("商品品类", "商品类别", "产品品类", "品类", "类目", "product_category", "product category", "category"),
        "product": ("商品", "产品", "product"),
        "seller": ("店铺", "商家", "seller"),
        "customer_state": ("客户州", "顾客州", "州", "省份", "地区", "区域", "region", "state", "customer_state", "customer state"),
        "payment_type": ("支付方式", "payment_type", "payment type"),
        "month_position": ("month_position", "month position", "月初", "月中", "月末"),
        "customer_city": ("客户城市", "顾客城市", "customer_city", "customer city"),
        "seller_state": ("卖家州", "商家州", "seller_state", "seller state"),
    },
    "operation": {
        "breakdown": ("拆开", "拆分", "拆解", "breakdown", "分布"),
        "trend": ("趋势", "走势", "trend"),
        "compare": ("对比", "compare", "同比", "环比", "比较一下", "比较"),
    },
    "date": {
        "last_month": ("上个月", "last month"),
        "last_30_days": ("最近一个月", "过去一个月", "last 30 days", "past month"),
        "last_three_months": ("最近三个月",),
        "this_month": ("本月", "this month"),
        "last_week": ("最近一周", "last week"),
        "this_year": ("今年", "this year"),
        "last_year": ("去年", "last year"),
    },
}


def default_query_aliases(
    *, metric_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return a fresh copy of shared query-parameter aliases."""
    aliases = {
        parameter: {canonical: tuple(values) for canonical, values in values_by_canonical.items()}
        for parameter, values_by_canonical in DEFAULT_QUERY_ALIASES.items()
    }
    if metric_aliases:
        aliases["metric"] = {
            str(canonical): tuple(str(alias) for alias in values if alias)
            for canonical, values in metric_aliases.items()
        }
    return aliases


def parse_negative_intents(
    text: str,
    *,
    query_aliases: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> dict[str, Any]:
    """Return clause-local negation and matching negative query-parameter spans.

    ``query_aliases`` maps a parameter name to canonical-value aliases. Aliases
    are only classified as negative query parameters when they occur after the
    negation and are not part of a preference-target phrase. This keeps
    “不要按我平时的分析习惯做同比” from treating its explicit 同比 as a
    negated query operation.
    """
    original = str(text or "")
    query_aliases = query_aliases or DEFAULT_QUERY_ALIASES
    unquoted = _QUOTED_TEXT_RE.sub(lambda match: " " * len(match.group(0)), original)
    clauses: list[dict[str, Any]] = []
    cursor = 0
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(unquoted):
        start, end = cursor, boundary.start()
        content = unquoted[start:end]
        if content.strip():
            left = len(content) - len(content.lstrip())
            right = len(content.rstrip())
            clause_start, clause_end = start + left, start + right
            clause_text = unquoted[clause_start:clause_end]
            negations = list(_NEGATION_RE.finditer(clause_text))
            clauses.append({
                "text": clause_text,
                "start": clause_start,
                "end": clause_end,
                "negative": bool(negations),
                "negation_spans": [
                    (clause_start + match.start(), clause_start + match.end())
                    for match in negations
                ],
                "unsupported_double_negation": bool(
                    _DOUBLE_NEGATION_RE.search(clause_text) or len(negations) > 1
                ),
            })
        cursor = boundary.end()
    content = unquoted[cursor:]
    if content.strip():
        left = len(content) - len(content.lstrip())
        right = len(content.rstrip())
        clause_start, clause_end = cursor + left, cursor + right
        clause_text = unquoted[clause_start:clause_end]
        negations = list(_NEGATION_RE.finditer(clause_text))
        clauses.append({
            "text": clause_text,
            "start": clause_start,
            "end": clause_end,
            "negative": bool(negations),
            "negation_spans": [
                (clause_start + match.start(), clause_start + match.end())
                for match in negations
            ],
            "unsupported_double_negation": bool(
                _DOUBLE_NEGATION_RE.search(clause_text) or len(negations) > 1
            ),
        })

    negative_values: dict[str, set[str]] = {}
    negative_spans: dict[str, list[tuple[int, int]]] = {}
    unsupported_spans: list[tuple[int, int]] = []
    for clause in clauses:
        if clause["unsupported_double_negation"]:
            clause_start = int(clause["start"])
            clause_text = str(clause["text"])
            unsupported_spans.append((clause_start, int(clause["end"])))
            for canonical_aliases in (query_aliases or {}).values():
                for canonical, aliases in canonical_aliases.items():
                    for alias in aliases:
                        if not alias:
                            continue
                        match = re.search(re.escape(str(alias)), clause_text, re.IGNORECASE)
                        if match:
                            negative_spans.setdefault("unsupported", []).append(
                                (clause_start + match.start(), clause_start + match.end())
                            )
            continue
        if not clause["negative"]:
            continue
        clause_start = int(clause["start"])
        clause_text = str(clause["text"])
        for negation_start, negation_end in clause["negation_spans"]:
            local_end = negation_end - clause_start
            tail = clause_text[local_end:]
            preference_target = _PREFERENCE_TARGET_RE.search(tail)
            preference_target_end = (
                local_end + preference_target.end() if preference_target else -1
            )
            for parameter, canonical_aliases in (query_aliases or {}).items():
                aliases_by_value = dict(canonical_aliases)
                if parameter == "date":
                    aliases_by_value.setdefault("explicit_date", ())
                for canonical, aliases in aliases_by_value.items():
                    if parameter == "date" and canonical == "explicit_date":
                        matches = list(re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", clause_text))
                        for match in matches:
                            found, found_end = match.span()
                            if found >= local_end and (preference_target_end < 0 or found < preference_target_end):
                                negative_values.setdefault(parameter, set()).add(match.group(0))
                                negative_spans.setdefault(parameter, []).append(
                                    (clause_start + found, clause_start + found_end)
                                )
                        continue
                    for alias in aliases:
                        if not alias:
                            continue
                        folded_clause = clause_text.casefold()
                        folded_alias = str(alias).casefold()
                        if folded_alias.isascii() and re.fullmatch(r"[a-z0-9_ ]+", folded_alias):
                            alias_matches = (
                                match.span() for match in re.finditer(
                                    rf"(?<![a-z0-9_]){re.escape(folded_alias)}(?![a-z0-9_])",
                                    folded_clause,
                                ) if match.start() >= local_end
                            )
                        else:
                            def chinese_or_symbol_matches():
                                search_at = local_end
                                while True:
                                    found = folded_clause.find(folded_alias, search_at)
                                    if found < 0:
                                        break
                                    yield (found, found + len(folded_alias))
                                    search_at = max(found + 1, found + len(folded_alias))
                            alias_matches = chinese_or_symbol_matches()
                        for found, found_end in alias_matches:
                            between_negation_and_alias = clause_text[local_end:found]
                            if parameter == "metric" and re.search(
                                r"按|拆|维度|同比|环比|对比|分析", between_negation_and_alias,
                                re.IGNORECASE,
                            ):
                                continue
                            absolute_start = clause_start + found
                            absolute_end = clause_start + found_end
                            if any(
                                int(existing_start) <= absolute_start
                                and int(existing_end) >= absolute_end
                                and int(existing_end) - int(existing_start) > found_end - found
                                for existing_start, existing_end in negative_spans.get(parameter, ())
                            ):
                                continue
                            # A negation of a saved preference category does not
                            # negate an explicit action later in the same clause.
                            if preference_target_end < 0 or found < preference_target_end:
                                negative_values.setdefault(parameter, set()).add(str(canonical))
                                negative_spans.setdefault(parameter, []).append(
                                    (absolute_start, absolute_end)
                                )

    return {
        "text": unquoted,
        "clauses": clauses,
        "positive_text": " ".join(
            str(clause["text"]) for clause in clauses if not clause["negative"]
        ),
        "negative_query_params": {
            key: sorted(values) for key, values in sorted(negative_values.items())
        },
        "negative_query_spans": {
            key: sorted(set(spans)) for key, spans in sorted(negative_spans.items())
        },
        "unsupported_spans": sorted(unsupported_spans),
        "query_text": mask_spans(unquoted, negative_spans),
        "unsupported_double_negation": any(
            bool(clause["unsupported_double_negation"]) for clause in clauses
        ),
    }


def span_is_negated(start: int, end: int, spans: Sequence[Sequence[int]]) -> bool:
    """Whether a source span overlaps one of the parser's negated spans."""
    return any(int(neg_start) < end and start < int(neg_end) for neg_start, neg_end in spans)


def mask_spans(text: str, spans: Mapping[str, Sequence[Sequence[int]]]) -> str:
    """Replace classified negative parameter mentions with spaces, preserving offsets."""
    chars = list(str(text or ""))
    all_spans = [span for values in spans.values() for span in values]
    for start, end in all_spans:
        for index in range(max(0, int(start)), min(len(chars), int(end))):
            chars[index] = " "
    return "".join(chars)
