"""Deterministic wording checks for analysis claims that exceed their evidence."""

from __future__ import annotations

import re


_CAUSAL_ASSERTION_PATTERNS = (
    r"导致", r"造成", r"引起", r"促使", r"驱动", r"源于", r"归因于",
    r"(?:主要|直接)?原因(?:是|在于)", r"使得",
    r"\bcaus(?:e|es|ed|ing)\b", r"\bbecause\s+of\b", r"\bdue\s+to\b",
    r"\bresult(?:s|ed|ing)?\s+in\b", r"\bled\s+to\b",
    r"\bdriven\s+by\b", r"\battributable\s+to\b",
)
_CAUSAL_NEGATIONS = (
    "不能证明", "无法证明", "并不能证明", "未能证明", "不代表", "不等于",
    "不能说明", "无法说明", "并不能说明", "没有证据表明", "不是因果", "not necessarily",
    "does not prove", "doesn't prove", "cannot prove", "can't prove",
    "does not establish", "doesn't establish",
    "no evidence that", "not caused", "not due",
)
_SENTENCE_DELIMITERS = "。！？!?;；\n."
_CLAUSE_DELIMITERS = _SENTENCE_DELIMITERS + ",，:："
_CONTRAST_BOUNDARY = re.compile(r"\b(?:but|however|yet|whereas|while)\b", re.IGNORECASE)
_UNCONFIRMED_CAUSAL_PATTERNS = (
    r"\b(?:the\s+)?(?:business\s+)?root\s+cause\s+(?:remains|is|has\s+been)\s+"
    r"(?:unconfirmed|unknown|unclear|undetermined|not\s+(?:yet\s+)?confirmed)\b",
    r"\b(?:cause|causality|causal\s+(?:relationship|link))\s+(?:remains|is|has\s+been)\s+"
    r"(?:unconfirmed|unknown|unclear|undetermined|not\s+(?:yet\s+)?confirmed)\b",
)


def unsupported_causal_assertions(text: str) -> list[str]:
    """Return causal assertion markers not locally negated by a disclaimer.

    This is a fail-closed lexical guard for currently observational analysis
    paths. It is not a causal-language parser and cannot establish that any
    positive causal claim is valid.
    """
    assertions: list[str] = []
    uncertain_spans = [
        match.span()
        for pattern in _UNCONFIRMED_CAUSAL_PATTERNS
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    for pattern in _CAUSAL_ASSERTION_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            # An uncertainty statement suppresses only its own causal marker;
            # it cannot exempt a separate assertion later in the sentence.
            if any(start <= match.start() < end for start, end in uncertain_spans):
                continue
            clause_start = max(
                (text.rfind(delimiter, 0, match.start()) + 1 for delimiter in _CLAUSE_DELIMITERS),
                default=0,
            )
            contrast = list(_CONTRAST_BOUNDARY.finditer(text, clause_start, match.start()))
            if contrast:
                clause_start = contrast[-1].end()
            # Keep negation local to the immediately governing phrase. A
            # disclaimer elsewhere in a long clause must not mask a claim.
            context = text[max(clause_start, match.start() - 48):match.start()].casefold()
            if any(negation in context for negation in _CAUSAL_NEGATIONS):
                continue
            assertions.append(pattern)
            break
    return assertions
