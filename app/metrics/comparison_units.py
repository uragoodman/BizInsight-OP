"""Registry-backed display helpers for metric comparisons.

The metric level, absolute delta, and relative delta are distinct quantities.
This module is shared by governed and baseline output adapters so the two
arms use the same wording and unit contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_DISPLAY_UNITS = {
    "%": "%",
    "percentage_points": "百分点",
    "BRL": "雷亚尔",
    "BRL/order": "雷亚尔/单",
    "items/order": "件/单",
    "customers": "人",
    "orders": "单",
    "days": "天",
    "hours": "小时",
    "score": "分",
}


def comparison_unit_contract(metric_id: str) -> tuple[str | None, str | None, str | None]:
    """Return registry metric, absolute-delta, and relative-delta units.

    Unknown metrics or units remain unknown. The compare operation defines a
    relative change as a percentage, while an absolute change of a
    percentage-valued metric is expressed in percentage points.
    """
    try:
        from app.metrics.registry import get_registry

        registry = get_registry()
        definition = registry.get(registry.resolve(str(metric_id)))
    except Exception:
        return None, None, None
    metric_unit = str(definition.unit or "").strip() or None
    if metric_unit is None:
        return None, None, None
    change_unit = "percentage_points" if metric_unit == "%" else metric_unit
    return metric_unit, change_unit, "%"


def display_unit(unit: str | None) -> str:
    canonical = str(unit or "").strip()
    return _DISPLAY_UNITS.get(canonical, canonical)


def comparison_unit_note(metric_id: str) -> str:
    """Create a deterministic user-facing explanation from registered units."""
    metric_unit, change_unit, relative_unit = comparison_unit_contract(metric_id)
    if not metric_unit or not change_unit or not relative_unit:
        return ""
    return (
        f"单位说明：基线值与当前值单位为{display_unit(metric_unit)}；"
        f"绝对差值单位为{display_unit(change_unit)}；"
        f"相对变化率单位为{display_unit(relative_unit)}。"
    )


def ensure_comparison_unit_note(answer: str, metric_id: str) -> tuple[str, bool]:
    """Append the registered unit explanation unless this exact note exists."""
    note = comparison_unit_note(metric_id)
    text = str(answer or "")
    if not note or note in text:
        return text, False
    return (text.rstrip() + "\n\n" + note).strip(), True


def comparison_table_unit_projection(
    presentation: Mapping[str, Any], *, metric_id: str,
) -> dict[str, Any]:
    """Add explicit unit labels to an existing model-authored table.

    Numeric cells are not changed. Unknown presentation shapes are retained
    and receive a note only when the registry provides a complete contract.
    """
    metric_unit, change_unit, relative_unit = comparison_unit_contract(metric_id)
    note = comparison_unit_note(metric_id)
    result = dict(presentation)
    if not note:
        return result

    columns = result.get("columns")
    if isinstance(columns, list):
        updated: list[Any] = []
        for column in columns:
            label = str(column)
            if "变化率" in label:
                suffix = display_unit(relative_unit)
            elif "差值" in label or "绝对变化" in label:
                suffix = display_unit(change_unit)
            elif metric_id in label or "指标" in label or "占比" in label:
                suffix = display_unit(metric_unit)
            else:
                updated.append(column)
                continue
            marker = f"（{suffix}）"
            updated.append(column if marker in label else f"{label}{marker}")
        result["columns"] = updated

    notes = result.get("notes")
    notes_list = list(notes) if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)) else []
    if note not in notes_list:
        notes_list.append(note)
    result["notes"] = notes_list
    return result
