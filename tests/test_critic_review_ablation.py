"""Offline A/B checks for the internal optional Critic LLM-review ablation."""

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from test_non_gmv_diagnostic_workflow import seed_facts
from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401


def test_critic_review_mode_isolated_on_normal_workflow(monkeypatch, isolated_trend_db, tmp_path):
    from app.graph import workflow
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.graph.nodes.critic import critic as critic_impl
    from app.graph.nodes import triage
    from app.llm import client as llm
    from app.agents import critic_agent
    from app.mcp import client as mcp, tools
    from app.core import time_expression
    from app.data.repositories import daily_metrics
    from app.tenancy import TenantContext

    engine, statements, _, _ = isolated_trend_db
    seed_facts(engine, "aov_ratio")
    monkeypatch.setattr(daily_metrics, "engine", engine)
    monkeypatch.setattr(time_expression, "engine", engine)
    monkeypatch.setattr(triage, "llm_classify", lambda *a, **k: None)
    for name in ("llm_extract", "llm_chat", "llm_agent_action", "llm_diagnostic_plan"):
        if hasattr(llm, name):
            monkeypatch.setattr(llm, name, lambda *a, **k: None)

    async def local_transport(tool_name, arguments):
        allowed = {
            "compare_breakdown_by_dimension", "compare_breakdown_by_dimension_period",
            "compare_freight_rate_ratio_facts_by_dimension_period",
            "compare_avg_order_value_ratio_facts_by_dimension_period",
        }
        assert tool_name in allowed, f"Unexpected legacy tool {tool_name}"
        return getattr(tools, tool_name)(**arguments)

    monkeypatch.setattr(mcp, "_call_local_mcp_tool", local_transport)
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *a, **k: None)
    active = {"arm": ""}
    calls = {name: [] for name in ("full", "off", "full_again", "metric_full", "metric_off")}
    critic_results = {name: [] for name in calls}
    precritic = {}
    calls_before_critic = {name: [] for name in calls}
    call_counts_at_critic = {name: [] for name in calls}

    def offline_review(prompt, content, *args, **kwargs):
        arm = active["arm"]
        calls[arm].append({"prompt": str(prompt), "content": str(content)[:80]})
        # Exercise the normal A-arm revision path once; later LLM reviews pass.
        if arm in {"full", "full_again"} and len(calls[arm]) == 1:
            return {"verdict": "failed", "issues": ["offline stub: answer is not fully grounded"]}
        return {"verdict": "passed", "issues": []}

    monkeypatch.setattr(critic_agent, "llm_extract", offline_review)
    original_critic = workflow.critic

    def observe_critic(state, **kwargs):
        arm = active["arm"]
        precritic.setdefault(arm, copy.deepcopy(state))
        calls_before_critic[arm].append(len(calls[arm]))
        call_counts_at_critic[arm].append(len(calls[arm]))
        result = original_critic(state, **kwargs)
        call_counts_at_critic[arm][-1] = len(calls[arm])
        critic_results[arm].append(copy.deepcopy(result.get("critic_result")))
        return result

    monkeypatch.setattr(workflow, "critic", observe_critic)
    context = TenantContext(
        tenant_id="platform", user_id="critic-review-ablation", role="admin", scope_type="platform",
    )
    question = "为什么 2024-05-04 的客单价上升了？请检查证据，区分统计分解与业务原因。"
    runs = {}

    def run(arm, mode=None, message=question):
        active["arm"] = arm
        trace = {}
        monkeypatch.setattr(workflow, "write_trace", lambda _tid, state: trace.update(copy.deepcopy(state)))
        sql_start = len(statements)
        kwargs = {
            "tenant_context": context,
            "memory_enabled": False,
            "trace_id": f"critic-ablation-{arm}-{uuid4().hex}",
        }
        if mode is not None:
            kwargs["_critic_llm_review_mode"] = mode
        result = workflow.run_chat_workflow(message, **kwargs)
        runs[arm] = {
            "trace": trace,
            "result": result,
            "sql": list(statements[sql_start:]),
            "review_calls": list(calls[arm]),
            "critic_call_counts": list(call_counts_at_critic[arm]),
        }
        return runs[arm]

    # Interleave modes through the common business entry; the middle B run
    # must not change the default on either surrounding A run.
    full = run("full")
    off = run("off", "critic_llm_review_off")
    full_again = run("full_again", "full")

    assert full["trace"].get("critic_llm_review_mode") == "full"
    assert off["trace"].get("critic_llm_review_mode") == "critic_llm_review_off"
    assert full_again["trace"].get("critic_llm_review_mode") == "full"
    assert full["critic_call_counts"][0] == 3
    assert off["critic_call_counts"][0] == 0
    assert full_again["critic_call_counts"][0] == 3
    assert calls_before_critic["full"][0] == 0
    assert calls_before_critic["off"][0] == 0
    assert full["trace"]["critic_result"]["llm_review"]["status"] == "invoked"
    assert off["trace"]["critic_result"]["llm_review"]["status"] == "evaluation_ablation"
    assert off["trace"]["critic_result"]["llm_review"]["mode"] == "critic_llm_review_off"
    assert full["trace"]["critic_result"]["llm_review"]["mode"] == "full"
    first_full_review = critic_results["full"][0]["llm_review"]
    assert first_full_review["groundedness"]["verdict"] == "failed"
    assert first_full_review["completeness"]["verdict"] == "passed"
    assert first_full_review["consistency"]["verdict"] == "passed"
    assert [off["trace"]["critic_result"]["llm_review"][key]["verdict"]
            for key in ("groundedness", "completeness", "consistency")] == ["evaluation_ablation"] * 3
    assert full["trace"]["critic_decisions"][0]["verdict"] == "needs_revision"
    assert full["trace"]["critic_decisions"][0]["revision_target"] == "generate_answer"
    assert len([item for item in full["trace"].get("node_history", [])
                if item.get("node") == "diagnostic_plan"]) == 1
    assert off["trace"]["critic_decisions"][0]["verdict"] != "needs_revision"

    compared = (
        "task_type", "diagnostic_plan", "diagnostic_action", "tool_calls", "authorized_metrics",
        "evidence_bundles", "runtime_budget", "runtime_steps", "investigation_state", "query_results",
        "tenant_id", "user_id", "user_role", "scope_type", "controlled_analysis_scope",
    )

    dynamic_id_keys = {
        "request_id", "evidence_id", "evidence_ids", "source_action_id", "candidate_id",
        "action_id", "tool_call_id", "query_id", "provenance_id", "input_evidence_ids",
    }

    def normalize(value, trace_id, id_map):
        if isinstance(value, str):
            return value.replace(trace_id, "<TRACE>")
        if isinstance(value, list):
            return [normalize(item, trace_id, id_map) for item in value]
        if isinstance(value, dict):
            normalized = {}
            for key, item in value.items():
                if key in {"created_at", "recorded_at", "observed_at"}:
                    normalized[key] = "<RUN_TIMESTAMP>"
                    continue
                if key in dynamic_id_keys and item is not None:
                    values = item if isinstance(item, list) else [item]
                    mapped = []
                    for identifier in values:
                        token = id_map.setdefault(str(identifier), f"<ID{len(id_map)}>")
                        mapped.append(token)
                    normalized[key] = mapped if isinstance(item, list) else mapped[0]
                else:
                    normalized[key] = normalize(item, trace_id, id_map)
            return normalized
        return value

    def before_critic(arm):
        state = precritic[arm]
        return normalize(
            {key: state.get(key) for key in compared},
            str(state.get("trace_id") or ""),
            {},
        )

    assert before_critic("full") == before_critic("off")
    assert precritic["full"].get("tool_calls") and precritic["off"].get("tool_calls")
    assert precritic["full"].get("evidence_bundles") and precritic["off"].get("evidence_bundles")
    assert precritic["full"].get("runtime_budget") == precritic["off"].get("runtime_budget")

    # The deterministic claim/evidence check remains active in B.
    invalid_state = copy.deepcopy(precritic["off"])
    invalid_state["claims"] = [{"source_evidence_ids": ["missing-evidence-id"]}]
    invalid_review = critic_impl(
        invalid_state, llm_review_mode=CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF,
    )
    assert invalid_review["critic_result"]["claim_review"]["status"] == "rejected"
    assert invalid_review["critic_verdict"] == "needs_revision"
    assert invalid_review["critic_result"]["llm_review"]["status"] == "evaluation_ablation"

    # metric_query stays deterministic-only in both modes.
    value_question = "请查询 2024-05-04 的 GMV 指标值。"
    for arm, mode in (("metric_full", None), ("metric_off", "critic_llm_review_off")):
        run(arm, mode, value_question)
        trace = runs[arm]["trace"]
        assert trace.get("task_type") == "metric_query"
        assert trace.get("critic_llm_review_mode") == (mode or "full")
        assert calls[arm] == []
        assert trace["critic_result"]["llm_review"]["status"] == "not_applicable"
        assert all(trace["critic_result"]["llm_review"][key]["verdict"] == "skipped"
                   for key in ("groundedness", "completeness", "consistency"))
        assert runs[arm]["result"].get("error") is False
    metric_results = [
        (((runs[arm]["trace"].get("tool_result") or {}).get("result") or {}).get("metric_result"))
        for arm in ("metric_full", "metric_off")
    ]
    assert metric_results[0] == metric_results[1]

    with pytest.raises(ValueError, match="Invalid internal Critic review mode"):
        workflow.build_chat_graph(critic_llm_review_mode="unknown")
    with pytest.raises(ValueError, match="Invalid internal Critic review mode"):
        workflow.run_chat_workflow(
            value_question,
            tenant_context=context,
            memory_enabled=False,
            _critic_llm_review_mode="unknown",
        )

    artifact = {
        "case": "existing development fixture aov_ratio",
        "question": question,
        "arm_order": ["full (default)", "critic_llm_review_off", "full"],
        "arms": {
            arm: {
                "mode": runs[arm]["trace"].get("critic_llm_review_mode"),
                "pre_first_critic": {key: precritic[arm].get(key) for key in compared} if arm in precritic else None,
                "critic_results": critic_results[arm],
                "final_critic_result": runs[arm]["trace"].get("critic_result"),
                "critic_decisions": runs[arm]["trace"].get("critic_decisions"),
                "critic_review_call_count": len(runs[arm]["review_calls"]),
                "review_calls_before_first_critic": calls_before_critic[arm][0],
                "review_calls_after_first_critic": runs[arm]["critic_call_counts"][0],
                "sql_statements": runs[arm]["sql"],
                "answer": runs[arm]["result"].get("answer"),
                "task_type": runs[arm]["trace"].get("task_type"),
            }
            for arm in ("full", "off")
        },
        "metric_query_controls": {
            arm: {
                "mode": runs[arm]["trace"].get("critic_llm_review_mode"),
                "review_status": runs[arm]["trace"]["critic_result"]["llm_review"]["status"],
                "review_call_count": len(runs[arm]["review_calls"]),
            }
            for arm in ("metric_full", "metric_off")
        },
    }
    output_dir = tmp_path / "critic_review_ablation_pair"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_trace.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
