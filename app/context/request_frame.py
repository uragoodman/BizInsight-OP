"""Request mode, operation, polarity, dialogue-act and modality extraction.

The rules here are intentionally lexical and deterministic. They never look up
analysis templates or decide how an operation should be routed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.context.semantic_contract import Operation

RequestMode = Literal["EXECUTE", "EXPLAIN", "ASK"]
DialogueAct = Literal["NEW", "CONTINUE", "SWITCH", "AMBIGUOUS"]
Modality = Literal["FACTUAL", "HYPOTHETICAL"]


class RequestFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_mode: RequestMode
    operations: tuple[Operation, ...]
    dialogue_act_provisional: DialogueAct
    dialogue_act: DialogueAct | None
    modality: Modality


_EXPLAIN = re.compile(r"什么是|是什麽|是什么|含义|定义|怎么算|计算公式|计算方式|怎么计算|\bwhat is\b|\bdefinition\b", re.I)
_CAUSE = re.compile(r"为什么|为何|什么原因|原因是什么|为何会|\bwhy\b", re.I)
_HYPOTHETICAL = re.compile(r"如果|假如|假设|倘若|若是|\bwhat if\b|\bif\b", re.I)
_QUERY = re.compile(r"查询|查一下|查查|查|查看|看看|多少|是多少|有多少|\bquery\b|\bhow many\b", re.I)
_RFM = re.compile(r"(?<![A-Za-z0-9_])R\s*F\s*M(?![A-Za-z0-9_])", re.I)
_NEGATION = re.compile(r"(?:不要|不做|不进行|不需要|无需|别做|别进行|不想做|do not|don't|not)\s*(?:再|继续|去)?\s*(?:做|进行|执行|run)?\s*$", re.I)
_CLAUSE_BOUNDARY = re.compile(r"[，,。.!！?？；;\n]")

_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("compare", re.compile(r"对比|比较|相比|同比|环比|\bcompare\b", re.I)),
    ("filter", re.compile(r"筛选|过滤|只看|仅看|限定|\bfilter\b|\bwhere\b", re.I)),
    ("sort", re.compile(r"排序|排名|排行|最高|最低|\btop\b|\bsort\b", re.I)),
    ("trend", re.compile(r"趋势|走势|变化趋势|\btrend\b", re.I)),
    ("contribution", re.compile(r"贡献|归因|贡献度|\battribution\b", re.I)),
    ("breakdown", re.compile(r"拆解|拆分|拆开|分布|\bbreakdown\b", re.I)),
    ("aggregate", re.compile(r"汇总|聚合|合计|总计|\baggregate\b", re.I)),
    ("enumerate", re.compile(r"列出|有哪些|列举|枚举|\blist\b", re.I)),
    ("segment", re.compile(r"分层|分群|分层级|分段|\bsegment(?:ation)?\b", re.I)),
    ("stat", re.compile(r"统计|计数|均值|平均值|平均|\bstatistics?\b", re.I)),
    ("metric_query", _QUERY),
)


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 18) : start]
    prefix = re.split(r"[，,。.!！?？；;\n]", prefix)[-1]
    return bool(_NEGATION.search(prefix))


def _explanation_clauses(question: str) -> tuple[tuple[int, int, re.Match[str]], ...]:
    clauses: list[tuple[int, int, re.Match[str]]] = []
    for match in _EXPLAIN.finditer(question):
        previous_boundaries = list(_CLAUSE_BOUNDARY.finditer(question, 0, match.start()))
        clause_start = previous_boundaries[-1].end() if previous_boundaries else 0
        next_boundary = _CLAUSE_BOUNDARY.search(question, match.end())
        clause_end = next_boundary.start() if next_boundary else len(question)
        clauses.append((clause_start, clause_end, match))
    return tuple(clauses)


def _cause_clauses(question: str) -> tuple[tuple[int, int, re.Match[str]], ...]:
    clauses: list[tuple[int, int, re.Match[str]]] = []
    for match in _CAUSE.finditer(question):
        previous_boundaries = list(_CLAUSE_BOUNDARY.finditer(question, 0, match.start()))
        clause_start = previous_boundaries[-1].end() if previous_boundaries else 0
        next_boundary = _CLAUSE_BOUNDARY.search(question, match.end())
        clause_end = next_boundary.start() if next_boundary else len(question)
        clauses.append((clause_start, clause_end, match))
    return tuple(clauses)


def _inside_clause(position: int, clauses: Sequence[tuple[int, int, re.Match[str]]]) -> bool:
    return any(start <= position < end for start, end, _match in clauses)


def _candidate_operations(question: str) -> list[tuple[int, int, str, str, RequestMode, str]]:
    """Return offset, label, ID, mode, polarity candidates in text order."""

    if not question:
        return []
    explanation_clauses = _explanation_clauses(question)
    cause_clauses = _cause_clauses(question)
    operations = [
        (
            explanation.start(),
            explanation.end(),
            "definition",
            explanation.group(0),
            "EXPLAIN",
            "POSITIVE",
        )
        for _start, _end, explanation in explanation_clauses
    ]
    cause = next(
        (
            match
            for _start, _end, match in cause_clauses
            if not _inside_clause(match.start(), explanation_clauses)
        ),
        None,
    )
    if cause:
        operations.append(
            (cause.start(), cause.end(), "explain_cause", cause.group(0), "ASK", "POSITIVE")
        )

    hypothetical = bool(_HYPOTHETICAL.search(question))
    rfm_match = _RFM.search(question)
    if rfm_match and not _inside_clause(rfm_match.start(), explanation_clauses):
        negated = _is_negated(question, rfm_match.start())
        mode: RequestMode = "ASK" if hypothetical else ("EXECUTE" if negated or "做" in question else "EXECUTE")
        operations.append(
            (
                rfm_match.start(),
                rfm_match.end(),
                "rfm",
                rfm_match.group(0),
                mode,
                "NEGATIVE" if negated else "POSITIVE",
            )
        )

    occupied: list[tuple[int, int]] = [(start, end) for start, end, *_ in operations]
    for operation_id, pattern in _CUE_PATTERNS:
        for match in pattern.finditer(question):
            # Suppress execution cues only in the clause being explained.
            # Other clauses in a compound request still produce operations.
            if _inside_clause(match.start(), explanation_clauses):
                continue
            if any(start <= match.start() < end for start, end in occupied):
                continue
            if operation_id == "metric_query" and _inside_clause(match.start(), cause_clauses):
                continue
            if hypothetical:
                mode = "ASK"
            elif operation_id in {"metric_query", "stat"} or "?" in question or "？" in question:
                mode = "ASK"
            else:
                mode = "EXECUTE"
            polarity = "NEGATIVE" if _is_negated(question, match.start()) else "POSITIVE"
            operations.append(
                (match.start(), match.end(), operation_id, match.group(0), mode, polarity)
            )
            occupied.append((match.start(), match.end()))

    # One action may match broad and narrow cue patterns at the same location.
    # Prefer the operation with the longer exact cue, then keep source order.
    operations.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    deduplicated: list[tuple[int, int, str, str, RequestMode, str]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in operations:
        key = (item[2], item[0], item[1])
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return deduplicated


def infer_request_mode(question: str, operations: Sequence[Operation] = ()) -> RequestMode:
    positive_operations = [operation for operation in operations if operation.polarity == "POSITIVE"]
    if positive_operations:
        return positive_operations[0].mode
    if operations:
        # A request containing only negated operations has no positive
        # execution request; keep the request-level mode at ASK.
        return "ASK"
    if _EXPLAIN.search(question):
        return "EXPLAIN"
    if (
        _CAUSE.search(question)
        or _QUERY.search(question)
        or re.search(r"呢\s*$|吗\s*$|么\s*$", question)
        or "？" in question
        or "?" in question
    ):
        return "ASK"
    if any(operation.mode == "ASK" for operation in operations):
        return "ASK"
    return "EXECUTE"


def build_request_frame(
    question: str,
    *,
    provisional_dialogue_act: DialogueAct = "NEW",
    validated_dialogue_act: DialogueAct | None = None,
    inherited_operations: Sequence[Operation] = (),
) -> RequestFrame:
    """Parse a request frame using only the current turn and validated inputs."""

    found = _candidate_operations(question)
    operations = [
        Operation(id=operation_id, label=label, mode=mode, polarity=polarity)
        for _start, _end, operation_id, label, mode, polarity in found
    ]
    if not operations:
        operations.extend(inherited_operations)
    request_mode = infer_request_mode(question, operations)
    modality: Modality = "HYPOTHETICAL" if _HYPOTHETICAL.search(question) else "FACTUAL"
    return RequestFrame(
        request_mode=request_mode,
        operations=tuple(operations),
        dialogue_act_provisional=provisional_dialogue_act,
        dialogue_act=validated_dialogue_act,
        modality=modality,
    )


def operation_spans(question: str) -> tuple[tuple[str, int, int], ...]:
    """Expose deterministic action spans for Mention Detection."""

    return tuple((item[3], item[0], item[1]) for item in _candidate_operations(question))
