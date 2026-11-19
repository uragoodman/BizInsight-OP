"""Shared shaping for proportional order-count contributions.

An order can contain items from several product categories, so summing the
per-category distinct-order counts double counts that order.  The controlled
contribution path instead allocates ``1 / k`` of every order to each of the
``k`` categories it touches.  The allocated group values add up to the true
distinct-order total while remaining an order-equivalent measure, not an
independent per-category distinct count.

This module only shapes already-computed rows; SQL remains in the platform and
seller repositories because each scope owns its own FROM/RLS contract.
"""

from __future__ import annotations

from typing import Iterable

ALLOCATION_METHOD = "proportional_overlap_v1"
RESIDUAL_LABEL = "其他品类（分摊后合并）"
ALLOCATION_NOTE = (
    "每个订单按其在锁定范围内涉及的品类数均分（1/k）；"
    "各品类分摊值之和等于同口径总去重订单数。"
    "分摊值为订单等价量，不等于各品类独立去重订单数。"
)


def allocated_items(
    rows: Iterable[tuple[str, float]],
    *,
    total: float,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Return allocated groups ranked by value, folding the tail into residual."""

    ordered = sorted(
        ((str(name), float(value)) for name, value in rows),
        key=lambda item: (-item[1], item[0]),
    )
    shown = ordered
    residual = 0.0
    if limit is not None and limit > 0 and len(ordered) > limit:
        head_count = max(1, limit - 1)
        shown = ordered[:head_count]
        residual = sum(value for _, value in ordered[head_count:])
    items = [_item(name, value, total) for name, value in shown]
    if residual > 0:
        items.append(_item(RESIDUAL_LABEL, residual, total))
    return items


def paired_allocated_items(
    current_rows: Iterable[tuple[str, float]],
    baseline_rows: Iterable[tuple[str, float]],
    *,
    current_total: float,
    baseline_total: float,
    limit: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Rank the union of groups by absolute allocated delta and fold the tail."""

    current = {str(name): float(value) for name, value in current_rows}
    baseline = {str(name): float(value) for name, value in baseline_rows}
    ranked = sorted(
        current.keys() | baseline.keys(),
        key=lambda name: (-abs(current.get(name, 0.0) - baseline.get(name, 0.0)), name),
    )
    shown = ranked
    residual_current = residual_baseline = 0.0
    if limit is not None and limit > 0 and len(ranked) > limit:
        shown = ranked[:limit]
        residual_current = sum(current.get(name, 0.0) for name in ranked[limit:])
        residual_baseline = sum(baseline.get(name, 0.0) for name in ranked[limit:])

    current_items = [
        _item(name, current.get(name, 0.0), current_total) for name in shown
    ]
    baseline_items = [
        _item(name, baseline.get(name, 0.0), baseline_total) for name in shown
    ]
    if residual_current > 0 or residual_baseline > 0:
        current_items.append(_item(RESIDUAL_LABEL, residual_current, current_total))
        baseline_items.append(_item(RESIDUAL_LABEL, residual_baseline, baseline_total))
    return current_items, baseline_items


def require_reconciled(
    rows: Iterable[tuple[str, float]], total: float, *, label: str,
) -> None:
    allocated = sum(float(value) for _, value in rows)
    if abs(allocated - total) > 1e-6:
        raise ValueError(
            f"{label} order allocation does not reconcile with distinct order total"
        )


def _item(name: str, value: float, total: float) -> dict[str, object]:
    # Keep full precision: per-group rounding would break the aggregate
    # reconciliation when many categories each receive a fractional share.
    return {
        "dimension_value": name,
        "value": float(value),
        "share": round(float(value) * 100.0 / total, 2) if total else None,
    }
