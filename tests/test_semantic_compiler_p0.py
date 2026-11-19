"""P0 tests for the Semantic Compiler v2 audit layer.

These tests prove that the P0 instrumentation is additive: contract hashes do
not change, audit data is observe-only, raw questions are not persisted, and
shadow sampling is off by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.routes.traces import _redact_semantic_audit
from app.context.compiler import compile_turn_context
from app.context.orchestrator import ContextOrchestrator
from app.context.shadow_audit import shadow_enabled, should_sample
from app.observability.tracing import _MEASUREMENT_SPECS, record_measurement
from evaluation.dimension_role_audit import _load_cases, _state_for

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"
BASELINE = ROOT / "tests" / "fixtures" / "p0_contract_hashes.json"


@pytest.fixture(autouse=True)
def _silence_audit_io(monkeypatch):
    monkeypatch.setattr(
        "app.context.compiler.record_event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "app.context.compiler.record_measurement", lambda *_args, **_kwargs: None
    )


def _compile_case(case: dict) -> dict:
    return compile_turn_context(_state_for(case))["context_compilation"]


def _core_cases() -> list[dict]:
    return [case for case in _load_cases(DATASET) if case["tier"] == "core"]


P1_BEHAVIOR_CHANGE = {"core-03", "core-04", "core-05", "core-10", "core-11"}
P2_BEHAVIOR_CHANGE = {"core-01", "core-05", "core-06", "core-09", "core-12"}


def test_p0_baseline_preserved_except_p1_and_p2_changes() -> None:
    """P0 froze the pre-change hashes; P1/P2 intentionally change some cases.

    The current baselines live in ``p1_contract_hashes.json`` and
    ``p2_contract_hashes.json``.  This assertion documents exactly which core
    cases changed and ensures the remaining ones did not drift accidentally.
    """

    behavior_change = P1_BEHAVIOR_CHANGE | P2_BEHAVIOR_CHANGE
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = {item["case_id"]: item["contract_hash"] for item in baseline["cases"]}
    mismatches = []
    for case in _core_cases():
        compiled = _compile_case(case)
        actual = compiled.get("request_contract_hash")
        if case["case_id"] in behavior_change:
            if actual == expected.get(case["case_id"]):
                mismatches.append((case["case_id"], "expected_stage_drift_missing"))
        elif actual != expected.get(case["case_id"]):
            mismatches.append((case["case_id"], expected.get(case["case_id"]), actual))
        assert compiled.get("p0_audit"), case["case_id"]
    assert mismatches == []


def test_p0_audit_side_scans_unregistered_dimensions_without_changing_business_candidates() -> (
    None
):
    core = {case["case_id"]: case for case in _core_cases()}

    # P2 classifies seller as a scope reference; only product_category remains
    # a grouping candidate.  The audit still exposes both mentions.
    compiled = _compile_case(core["core-01"])
    audit = compiled["p0_audit"]
    detection = audit["dimension_detection"]
    assert "seller" in detection["candidates_all"]
    assert "product_category" in detection["candidates_all"]
    assert detection["registered"] == ["product_category"]
    assert detection["unregistered"] == ["seller"]
    assert detection["business_candidates"] == ["product_category"]
    assert (compiled["known_slots"].get("dimension") or {}).get(
        "value"
    ) == "product_category"

    # core-04 has no "贡献" but has an explicit "拆解". P1 promotes seller to a
    # business candidate (support validation then reports it as unregistered)
    # while product_category remains; the two candidates stay ambiguous.  Full
    # role classification is P2.
    compiled = _compile_case(core["core-04"])
    audit = compiled["p0_audit"]
    detection = audit["dimension_detection"]
    assert "seller" in detection["candidates_all"]
    assert detection["unregistered"] == ["seller"]
    assert detection["business_candidates"] == ["product_category", "seller"]
    assert set(compiled["dimension_candidates"]) == {
        "product_category", "seller",
    }
    assert compiled["clarification_reason"] == "ambiguous_dimension"


def test_p0_audit_does_not_store_raw_question_or_alias_text() -> None:
    marker = "P0_MARKER_7c1d"
    compiled = compile_turn_context(
        {
            "user_message": f"{marker} 的 GMV 按商品品类",
            "trace_id": "p0-privacy",
            "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
        }
    )["context_compilation"]
    audit = compiled["p0_audit"]
    serialized = json.dumps(audit, ensure_ascii=False)
    assert marker not in serialized
    for mention in audit["mentions"]:
        assert "alias" not in mention
        assert "raw_text" not in mention


def test_p0_semantic_events_are_emitted_without_raw_question(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.context.compiler.record_event",
        lambda name, attributes: captured.append((name, attributes)),
    )
    marker = "P0_EVENT_MARKER_5a2"
    compile_turn_context(
        {
            "user_message": f"{marker} 的 GMV 按商品品类",
            "trace_id": "p0-events",
            "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
        }
    )
    names = [name for name, _attributes in captured]
    assert "semantic.dimension.audit" in names
    assert "semantic.support.audit" in names
    assert marker not in json.dumps(captured, ensure_ascii=False)
    assert all("question" not in attributes for _name, attributes in captured)


def test_p0_observability_failure_does_not_break_compilation(monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("observability unavailable")

    monkeypatch.setattr("app.context.compiler.record_event", _boom)
    monkeypatch.setattr("app.context.compiler.record_measurement", _boom)
    compiled = compile_turn_context(
        {
            "user_message": "按商品品类统计 GMV",
            "trace_id": "p0-obs-failure",
            "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
        }
    )["context_compilation"]
    assert compiled["request_contract_hash"]
    assert compiled["p0_audit"]["version"] == "p0-1"


def test_p0_audit_is_not_exposed_to_context_bundle() -> None:
    state = _state_for(_core_cases()[0])
    state["task_type"] = "metric_query"
    compile_turn_context(state)
    bundle = ContextOrchestrator().build_specialist_bundle(
        state, "data_query", instruction="p0 bundle isolation",
    )
    serialized = json.dumps(
        {
            "resolved_request": bundle.resolved_request,
            "items": [item.content for item in bundle.items],
        },
        ensure_ascii=False,
    )
    assert "p0_audit" not in serialized


def test_trace_view_redacts_p0_audit() -> None:
    trace = {
        "state_snapshot": {
            "context_compilation": {
                "p0_audit": {"version": "p0-1"},
                "relation": "new",
            }
        }
    }
    redacted = _redact_semantic_audit(trace)
    assert "p0_audit" in trace["state_snapshot"]["context_compilation"]
    assert "p0_audit" not in redacted["state_snapshot"]["context_compilation"]
    assert redacted["state_snapshot"]["context_compilation"]["relation"] == "new"


def test_p0_measurement_specs_are_registered() -> None:
    expected = {
        "dimension_mentions_total": {"source": "alias"},
        "dimension_decisions_total": {"decision": "resolved"},
        "dimension_support_total": {"status": "not_observed"},
        "semantic_audit_latency_ms": {"stage": "audit"},
    }
    for name, labels in expected.items():
        assert name in _MEASUREMENT_SPECS
        record_measurement(name, 1, labels=labels)
    with pytest.raises(ValueError):
        record_measurement("unknown_p0_measurement", 1, labels={})


def test_p0_shadow_is_disabled_by_default(monkeypatch) -> None:
    off = SimpleNamespace(
        semantic_shadow_enabled=False,
        semantic_shadow_sample_rate=0.0,
        semantic_shadow_salt="",
    )
    monkeypatch.setattr("app.context.shadow_audit._settings", lambda: off)
    assert shadow_enabled(off) is False
    assert should_sample("trace-1", settings=off) is False

    no_salt = SimpleNamespace(
        semantic_shadow_enabled=True,
        semantic_shadow_sample_rate=5.0,
        semantic_shadow_salt="",
    )
    assert shadow_enabled(no_salt) is False
    assert should_sample("trace-1", settings=no_salt) is False

    on = SimpleNamespace(
        semantic_shadow_enabled=True,
        semantic_shadow_sample_rate=100.0,
        semantic_shadow_salt="test-salt",
    )
    assert shadow_enabled(on) is True
    assert should_sample("trace-1", settings=on) is False
    assert should_sample("trace-1", settings=on, eligible=True) is True
    assert should_sample("trace-1", settings=on, eligible=True) == should_sample(
        "trace-1", settings=on, eligible=True
    )

    zero_rate = SimpleNamespace(
        semantic_shadow_enabled=True,
        semantic_shadow_sample_rate=0.0,
        semantic_shadow_salt="test-salt",
    )
    assert should_sample("trace-1", settings=zero_rate) is False
