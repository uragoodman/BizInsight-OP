"""Fail-closed readiness checks for future randomized causal analysis."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any


REQUIRED_EXPERIMENT_FIELDS = (
    "experiment_id",
    "assignment_unit",
    "treatment_group",
    "control_group",
    "outcome_metric",
    "observation_window",
    "randomization_validation",
)


def assess_causal_readiness(evidence_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Assess whether trusted evidence contains a validated experiment design.

    This function does not estimate treatment effects. It accepts only the
    server-owned typed-evidence payload under values.experiment_design;
    free-form claims or LLM-generated plan text cannot satisfy readiness.
    """
    designs: list[dict[str, Any]] = []
    for record in evidence_records:
        if not isinstance(record, dict):
            continue
        typed = record.get("investigation_evidence")
        values = typed.get("values") if isinstance(typed, dict) else None
        design = values.get("experiment_design") if isinstance(values, dict) else None
        if isinstance(design, dict):
            designs.append(design)

    if not designs:
        return {
            "status": "not_ready",
            "ready_for_estimation": False,
            "missing_fields": list(REQUIRED_EXPERIMENT_FIELDS),
            "reason": "当前受控数据证据中没有已注册且验证过的随机实验设计数据。",
            "causal_effect_estimated": False,
        }

    design = designs[-1]
    missing = [
        name for name in REQUIRED_EXPERIMENT_FIELDS
        if design.get(name) is None or design.get(name) == ""
    ]
    for field in ("experiment_id", "assignment_unit", "treatment_group", "control_group", "outcome_metric"):
        value = design.get(field)
        if isinstance(value, str) and not value.strip() and field not in missing:
            missing.append(field)
    if design.get("randomization_validation") is not True:
        missing.append("randomization_validation_passed")
    if design.get("treatment_group") == design.get("control_group"):
        missing.append("distinct_treatment_and_control_groups")
    window = design.get("observation_window")
    if isinstance(window, dict):
        start, end = window.get("start"), window.get("end")
        try:
            valid_window = date.fromisoformat(str(start)) <= date.fromisoformat(str(end))
        except (TypeError, ValueError):
            valid_window = False
        if not valid_window:
            missing.append("valid_observation_window")
    elif "observation_window" not in missing:
        missing.append("valid_observation_window")
    unique_missing = list(dict.fromkeys(missing))
    return {
        "status": "ready_for_estimator" if not unique_missing else "not_ready",
        "ready_for_estimation": not unique_missing,
        "missing_fields": unique_missing,
        "reason": (
            "实验设计字段齐全且随机化校验通过；当前版本仍不执行因果效应估计。"
            if not unique_missing
            else "随机实验设计信息不完整或随机化校验未通过。"
        ),
        "causal_effect_estimated": False,
    }
