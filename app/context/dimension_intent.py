"""Deterministic dimension-role classification for the Semantic Compiler.

P2 keeps the parser rule-based.  A mention is classified as one of:

- ``grouping``: an explicit group-by axis ("按品类拆解");
- ``filter``: a place/region restriction ("华东地区的 GMV");
- ``reference``: the analysis subject/scope ("当前授权店铺的 GMV");
- ``comparison``: a comparison operand ("比较品类和支付方式");
- ``conflict``: grouping and scope cues disagree ("按店铺范围的 GMV");
- ``unresolved``: no reliable cue.

Only ``grouping`` mentions may become dimension candidates.  The classifier is
intentionally local and evidence-based; it does not call an LLM and does not
decide whether the dimension is executable (that remains SupportResolver's job).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal[
    "grouping",
    "filter",
    "reference",
    "comparison",
    "conflict",
    "unresolved",
]
ConfidenceClass = Literal["closed", "open", "unresolved"]


@dataclass(frozen=True)
class RoleAssignment:
    dimension: str
    start: int
    end: int
    role: Role
    evidence: tuple[str, ...]
    confidence_class: ConfidenceClass = "open"
    conflict: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "start": self.start,
            "end": self.end,
            "role": self.role,
            "evidence": list(self.evidence),
            "confidence_class": self.confidence_class,
            "conflict": self.conflict,
        }


_GROUP_PREFIXES = (
    "按",
    "按照",
    "根据",
    "依据",
    "各",
    "各个",
    "每个",
    "每",
    "不同",
    "分别",
    "由",
    "从",
    "以",
    "基于",
    "哪个",
    "哪些",
    "哪种",
    "最多",
    "最少",
    "最高",
    "最低",
    "排名",
    "排行",
    "top",
    "by",
    "per",
    "each",
)
_GROUP_SUFFIXES = (
    "拆解",
    "拆分",
    "拆开",
    "分布",
    "贡献",
    "占比",
    "维度",
    "层面",
    "角度",
    "粒度",
    "排名",
    "排行",
    "明细",
    "breakdown",
    "distribution",
    "contribution",
    "share",
)
_REFERENCE_PREFIXES = (
    "当前",
    "本会话",
    "本次",
    "授权",
    "已授权",
    "所选",
    "已选",
    "该",
    "这个",
    "那个",
    "本店",
    "本店铺",
    "还是那个",
    "最近",
    "current",
    "this",
    "authorized",
    "selected",
    "given",
)
_FILTER_PREFIXES = (
    "筛选",
    "过滤",
    "只看",
    "仅看",
    "限定",
    "filter",
    "where",
    "only",
)
_NEGATION_CUES = (
    "不要",
    "不看",
    "不用",
    "无需",
    "不是",
    "并非",
    "别",
    "不需要",
    "不必",
    "不想",
    "don't",
    "do not",
    "not ",
)
_REFERENCE_SUFFIXES = ("范围", "范围内", "scope", "range", "'s")
_MEASURE_HINTS = (
    "的gmv",
    "的订单量",
    "的订单数",
    "的变化",
    "的趋势",
    "的数据",
    "的情况",
    "的表现",
    "的业绩",
    "的销售额",
    "的成交额",
    "的运费",
)
_METRIC_NOUN_RIGHT = ("数", "数量", "售价", "价格", "金额", "成本", "个数", "件数")
_COMPARISON_MARKERS = (
    "比较",
    "对比",
    "相比",
    "哪个更",
    "哪个卖得",
    "谁更",
    "vs",
    "versus",
)
_REGION_DIMENSIONS = {"customer_state", "customer_city", "seller_state"}


def classify_dimension_mentions(
    question: str,
    mentions: Sequence[tuple[str, str, int, int]],
    *,
    intent: str,
    metric_spans: Sequence[tuple[int, int]] = (),
    region_tokens: Sequence[str] = (),
    negated_spans: Sequence[Sequence[int]] = (),
) -> list[RoleAssignment]:
    """Classify every dimension mention in one deterministic pass."""

    assignments: list[RoleAssignment] = []
    for dimension, _alias, start, end in mentions:
        role, evidence, confidence, conflict = _classify_one(
            question,
            dimension=dimension,
            start=start,
            end=end,
            intent=intent,
            metric_spans=metric_spans,
            region_tokens=region_tokens,
            negated_spans=negated_spans,
        )
        assignments.append(
            RoleAssignment(
                dimension=dimension,
                start=start,
                end=end,
                role=role,
                evidence=evidence,
                confidence_class=confidence,
                conflict=conflict,
            )
        )
    if region_tokens and not any(
        item.dimension == "customer_state" for item in assignments
    ):
        folded_question = question.casefold()
        for token in region_tokens:
            folded = str(token).casefold()
            start = folded_question.find(folded)
            if start >= 0:
                assignments.append(
                    RoleAssignment(
                        dimension="customer_state",
                        start=start,
                        end=start + len(token),
                        role="filter",
                        evidence=("region_token",),
                        confidence_class="closed",
                    )
                )
    return _propagate_coordination(question, assignments)


def aggregate_roles(assignments: Sequence[RoleAssignment]) -> dict[str, Role]:
    """Collapse mentions of the same dimension with role precedence."""

    grouped: dict[str, list[RoleAssignment]] = {}
    for assignment in assignments:
        grouped.setdefault(assignment.dimension, []).append(assignment)
    result: dict[str, Role] = {}
    for dimension, items in grouped.items():
        roles = {item.role for item in items}
        if "grouping" in roles:
            result[dimension] = "grouping"
        elif "comparison" in roles:
            result[dimension] = "comparison"
        elif "filter" in roles:
            result[dimension] = "filter"
        elif "reference" in roles:
            result[dimension] = "reference"
        else:
            result[dimension] = "unresolved"
    return result


def grouping_dimensions(assignments: Sequence[RoleAssignment]) -> list[str]:
    return sorted(
        dimension
        for dimension, role in aggregate_roles(assignments).items()
        if role == "grouping"
    )


def conflict_dimensions(assignments: Sequence[RoleAssignment]) -> list[str]:
    return sorted({item.dimension for item in assignments if item.conflict})


def _classify_one(
    question: str,
    *,
    dimension: str,
    start: int,
    end: int,
    intent: str,
    metric_spans: Sequence[tuple[int, int]],
    region_tokens: Sequence[str],
    negated_spans: Sequence[Sequence[int]],
) -> tuple[Role, tuple[str, ...], ConfidenceClass, bool]:
    left = question[max(0, start - 10) : start].casefold()
    right = question[end : end + 16].casefold()
    right_after_de = right[1:] if right.startswith("的") else right
    prefix_left = _trim_after_coordination(left)
    negation_left = _trim_after_clause(left)

    if _span_is_negated(start, end, negated_spans) or any(
        cue.casefold() in negation_left for cue in _NEGATION_CUES
    ):
        return "reference", ("negated",), "closed", False

    if _inside_metric(start, end, metric_spans) or (
        dimension in {"product", "product_category"}
        and right.startswith(_METRIC_NOUN_RIGHT)
    ):
        return "reference", ("metric_noun",), "closed", False

    if intent == "retrieve_metric_definition":
        return "reference", ("intent:definition",), "closed", False

    group_prefix = _match_left(prefix_left, _GROUP_PREFIXES)
    reference_prefix = _match_left(prefix_left, _REFERENCE_PREFIXES)
    filter_prefix = _match_left(prefix_left, _FILTER_PREFIXES)
    reference_suffix = _match_right(right, _REFERENCE_SUFFIXES)
    possessive = right.startswith("的") and any(
        hint in right for hint in _MEASURE_HINTS
    )

    # "按店铺范围" says both "group by" and "within scope"; do not guess.
    if group_prefix and reference_suffix in {"范围", "范围内", "scope", "range"}:
        return (
            "reference",
            (f"group:{group_prefix}", f"reference:{reference_suffix}", "conflict"),
            "open",
            True,
        )

    if group_prefix:
        evidence = [f"group_prefix:{group_prefix}"]
        group_suffix = _match_right(right_after_de, _GROUP_SUFFIXES)
        if group_suffix:
            evidence.append(f"group_suffix:{group_suffix}")
        return "grouping", tuple(evidence), "closed", False

    if filter_prefix:
        return "filter", (f"filter_prefix:{filter_prefix}",), "closed", False

    if dimension in _REGION_DIMENSIONS and _region_context(
        left,
        right,
        region_tokens,
    ):
        return "filter", ("region_filter",), "closed", False

    if reference_prefix or reference_suffix or possessive:
        evidence = []
        if reference_prefix:
            evidence.append(f"reference_prefix:{reference_prefix}")
        if reference_suffix:
            evidence.append(f"reference_suffix:{reference_suffix}")
        if possessive:
            evidence.append("possessive")
        return "reference", tuple(evidence), "closed", False

    group_suffix = _match_right_any(right_after_de, _GROUP_SUFFIXES, window=10)
    if group_suffix:
        return "grouping", (f"group_suffix:{group_suffix}",), "closed", False

    if any(marker in question.casefold() for marker in _COMPARISON_MARKERS):
        return "comparison", ("comparison_context",), "open", False

    return "unresolved", (), "unresolved", False


_COORDINATION_RE = re.compile(r"^[\s和与及、]*$")


def _propagate_coordination(
    question: str,
    assignments: Sequence[RoleAssignment],
) -> list[RoleAssignment]:
    """Share grouping intent across coordinated mentions.

    "按客户州和商品品类" has one explicit grouping cue on the first mention;
    the second is coordinated and therefore also a grouping axis.  Reference
    and filter mentions are never overwritten.
    """

    items = sorted(assignments, key=lambda item: item.start)
    changed = True
    while changed:
        changed = False
        for index, left in enumerate(items):
            for right_index in range(index + 1, len(items)):
                right = items[right_index]
                if left.end > right.start:
                    continue
                between = question[left.end : right.start]
                if not _COORDINATION_RE.fullmatch(between):
                    continue
                if not any(
                    separator in between for separator in ("和", "与", "及", "、")
                ):
                    continue
                if left.role == "grouping" and right.role in {
                    "unresolved",
                    "comparison",
                }:
                    items[right_index] = _as_grouping(right)
                    changed = True
                elif right.role == "grouping" and left.role in {
                    "unresolved",
                    "comparison",
                }:
                    items[index] = _as_grouping(left)
                    changed = True
    return items


def _as_grouping(item: RoleAssignment) -> RoleAssignment:
    return RoleAssignment(
        dimension=item.dimension,
        start=item.start,
        end=item.end,
        role="grouping",
        evidence=item.evidence + ("coordination",),
        confidence_class="open",
        conflict=item.conflict,
    )


_COORDINATION_CUES = ("和", "与", "及", "、", ",", "，")
_CLAUSE_BOUNDARIES = ("，", ",", "。", ".", "!", "！", "?", "？", ";", "；", "\n")


def _trim_after_clause(left: str) -> str:
    trimmed = left
    for separator in _CLAUSE_BOUNDARIES:
        index = trimmed.rfind(separator)
        if index >= 0:
            trimmed = trimmed[index + 1 :]
    return trimmed


def _trim_after_coordination(left: str) -> str:
    trimmed = left
    for separator in _COORDINATION_CUES:
        index = trimmed.rfind(separator)
        if index >= 0:
            trimmed = trimmed[index + 1 :]
    return trimmed


def _match_right_any(
    right: str, cues: Sequence[str], *, window: int = 10
) -> str | None:
    for cue in cues:
        folded = cue.casefold()
        index = right.find(folded)
        if 0 <= index <= window:
            return cue
    return None


def _match_left(left: str, cues: Sequence[str]) -> str | None:
    for cue in cues:
        folded = cue.casefold()
        if len(folded) == 1:
            if left.endswith(folded):
                return cue
        elif folded in left:
            return cue
    return None


def _match_right(right: str, cues: Sequence[str]) -> str | None:
    for cue in cues:
        folded = cue.casefold()
        if right.startswith(folded):
            return cue
    return None


def _region_context(left: str, right: str, region_tokens: Sequence[str]) -> bool:
    return any(
        str(token).casefold() in left or str(token).casefold() in right
        for token in region_tokens
    )


def _inside_metric(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(left <= start and end <= right for left, right in spans)


def _span_is_negated(start: int, end: int, spans: Sequence[Sequence[int]]) -> bool:
    return any(
        int(neg_start) < end and start < int(neg_end) for neg_start, neg_end in spans
    )
