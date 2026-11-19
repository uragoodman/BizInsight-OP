"""Deterministic checks that a sandbox result covers the confirmed question."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletenessResult:
    valid: bool
    missing: tuple[str, ...]


_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "gmv",
        "GMV",
        (r"\bgmv\b", r"gross[\s_-]+merchandise[\s_-]+value", "成交总额"),
    ),
    (
        "order_count",
        "订单数",
        ("订单数", "订单数量", r"order[_\s-]*count", r"\borders?\b"),
    ),
    (
        "monthly",
        "按月汇总",
        ("按月", "每月", "月度", r"\bmonthly\b", r"\bmonth\b"),
    ),
    (
        "period_change",
        "环比变化",
        ("环比", "月环比", r"month[_\s-]*over[_\s-]*month", r"\bmom\b"),
    ),
    (
        "max_month",
        "最高月份",
        (
            r"(?:最高|最大).{0,12}(?:月份|月)",
            r"(?:峰值|峰顶).{0,12}(?:月份|月)",
            r"max[_\s-]*month",
            r"peak[_\s-]*month",
            r"highest[\s_-]+month",
        ),
    ),
    (
        "min_month",
        "最低月份",
        (
            r"(?:最低|最小).{0,12}(?:月份|月)",
            r"(?:谷值|谷底).{0,12}(?:月份|月)",
            r"min[_\s-]*month",
            r"trough[_\s-]*month",
            r"lowest[\s_-]+month",
        ),
    ),
)


def _text_payload(result: dict[str, Any]) -> str:
    # Only inspect user-visible analytical content. A model must not satisfy
    # a requested metric by mentioning it only in limitations or recommendations.
    visible = {
        key: result.get(key)
        for key in ("conclusion", "method", "summary", "evidence", "charts")
        if key in result
    }
    return json.dumps(visible, ensure_ascii=False, default=str).lower()


def requested_requirements(
    question: str,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Return only requirements explicitly indicated by the user's question."""
    text = str(question or "").lower()
    return tuple(
        requirement
        for requirement in _REQUIREMENTS
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in requirement[2])
    )


def validate_result_completeness(
    question: str,
    result: dict[str, Any],
    expected_output: list[str] | tuple[str, ...] | None = None,
) -> CompletenessResult:
    """Check requested metric concepts and required output sections.

    This is deliberately a bounded presence check, not a claim that values are
    numerically correct. Numerical correctness remains the responsibility of
    sandbox execution and the trusted result contract.
    """
    payload = _text_payload(result)
    missing: list[str] = []
    for _key, label, patterns in requested_requirements(question):
        if not any(
            re.search(pattern, payload, flags=re.IGNORECASE)
            for pattern in patterns
        ):
            missing.append(label)
    for output in expected_output or ():
        # The worker adds this from server-side facts after execution.
        if output == "reproducibility":
            continue
        value = result.get(output)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(f"输出部分:{output}")
    return CompletenessResult(valid=not missing, missing=tuple(dict.fromkeys(missing)))
