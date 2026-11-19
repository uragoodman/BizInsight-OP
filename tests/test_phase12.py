from app.llm.routing import model_for

def test_model_profiles_fall_back_to_default(monkeypatch):
    monkeypatch.setattr("app.llm.routing.get_settings", lambda: type("S",(),{"llm_model":"default","llm_fast_model":"fast","llm_standard_model":"","llm_quality_model":"quality"})())
    assert model_for("classify") == ("fast", "fast")
    assert model_for("chat") == ("default", "standard")
    assert model_for("report") == ("quality", "quality")

def test_task_relation_requires_human_choice_for_unrelated_turn(monkeypatch):
    from app.tasks.service import relation
    task = {
        "task_id": "ctk_1",
        "scope_data": {"seller_id": "a", "start_date": "2018-01-01", "end_date": "2018-01-31"},
        "task_state": {
            "scope_fingerprint": "scope-a",
            "data_version_manifest_hash": "version-a",
            "state_version": 2,
            "context_slots": {
                "metric": {
                    "value": "gmv", "source": "task_state", "confidence": 0.9,
                    "source_turn_id": "turn-a", "source_task_state_version": 2,
                },
            },
        },
    }
    monkeypatch.setattr("app.tasks.service.task_context", lambda *_: {"current_task": task})
    # 本测试针对确定性关键词回退路径：显式关闭 LLM 解析（Phase 7），不触真实 provider。
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_a, **_k: None)
    assert relation(
        object(), "s", "继续刚才的分析",
        {"seller_id": "a", "start_date": "2018-01-01", "end_date": "2018-01-31"},
        scope_fingerprint="scope-a", expected_data_version="version-a",
    )["action"] == "continue"
    assert relation(
        object(), "s", "分析另一个问题",
        {"seller_id": "b", "start_date": "2018-02-01", "end_date": "2018-02-28"},
    )["action"] == "clarify"


def test_contribution_intent_preserves_two_explicit_dates() -> None:
    from app.agents.data_query_agent import build_metric_query_plan

    plan = build_metric_query_plan("拆解 2018-08-10 与 2018-08-09 的 GMV 按客户城市贡献度")
    assert plan.query_type == "metric_contribution"
    assert plan.current_date == "2018-08-10"
    assert plan.baseline_date == "2018-08-09"


def test_legacy_mcp_tool_name_binds_fact_to_evidence() -> None:
    from app.graph.nodes.answer import _make_facts_from_tool

    tool_result = {"tool_name": "query_gmv", "result": {"metric_result": {"metric_name": "gmv", "metric_date": "2018-08-10", "value": 100}}}
    facts, evidence_ids = _make_facts_from_tool(tool_result, [{"evidence_id": "ev_gmv", "source_type": "mcp", "source_ref": "query_gmv"}])
    assert facts[0]["evidence_ids"] == ["ev_gmv"]
    assert evidence_ids == ["ev_gmv"]
