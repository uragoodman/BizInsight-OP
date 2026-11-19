from types import SimpleNamespace

from app.graph.workflow import _chat_context_scope_fingerprint
from app.graph.nodes import metric_definition as metric_definition_module
from app.graph.nodes.answer import _grounded_narrative
from app.rag.schemas import RagCitation, RagSearchHit, RagSearchResult
from app.tenancy import SellerGrant, TenantContext


def test_conversation_scope_fingerprint_changes_with_trusted_identity_and_grants():
    first = TenantContext(tenant_id="t1", user_id="u1", role="analyst", scope_type="seller", seller_grants=(SellerGrant("s1"),))
    other_user = TenantContext(tenant_id="t1", user_id="u2", role="analyst", scope_type="seller", seller_grants=(SellerGrant("s1"),))
    other_grant = TenantContext(tenant_id="t1", user_id="u1", role="analyst", scope_type="seller", seller_grants=(SellerGrant("s2"),))

    assert _chat_context_scope_fingerprint(first) != _chat_context_scope_fingerprint(other_user)
    assert _chat_context_scope_fingerprint(first) != _chat_context_scope_fingerprint(other_grant)


def _rag_result(query, *ids):
    hits = [RagSearchHit(
        chunk_id=chunk_id, text=f"definition {chunk_id}", score=1.0 / index,
        citation=RagCitation(source_path=f"{chunk_id}.md", title=chunk_id, doc_type="metric", chunk_id=chunk_id),
    ) for index, chunk_id in enumerate(ids, 1)]
    return RagSearchResult(query=query, backend="local", top_k=5, hits=hits, context="")


def _state_with_answer_evidence(user_message, evidence):
    evidence_id = "ev-grounded-test"
    stored_evidence = evidence.get("query_result", evidence) if isinstance(evidence, dict) else evidence
    return {
        "user_message": user_message,
        "task_type": "metric_query",
        "evidence_bundles": [{
            "evidence_id": evidence_id,
            "source_type": "mcp",
            "source_ref": "test_query",
            "content": stored_evidence,
            "trusted": True,
        }],
        "evidence_refresh": {"available": True, "evidence_ids": [evidence_id]},
    }


def test_metric_definition_retrieves_original_and_rewrite_then_fuses(monkeypatch):
    seen = []
    original = "\u90a3\u5b83\u7684\u9650\u5236\uff1f"
    rewritten = "GMV \u6307\u6807\u7684\u8ba1\u7b97\u53e3\u5f84\u6709\u54ea\u4e9b\u9650\u5236\uff1f"
    results = {original: _rag_result(original, "exact", "shared"),
               rewritten: _rag_result(rewritten, "shared", "semantic")}
    monkeypatch.setattr(metric_definition_module, "get_settings", lambda: SimpleNamespace(
        rag_release_controller_enabled=False, rag_timeout_seconds=5, rag_reranker_enabled=False, rag_reranker_strategy="rule"))
    monkeypatch.setattr(metric_definition_module, "retrieve_knowledge", lambda query, **kwargs: (seen.append(query), results[query])[1])
    monkeypatch.setattr(metric_definition_module, "execute_with_resilience", lambda fn, *a, **k: fn())
    monkeypatch.setattr(metric_definition_module, "_apply_rerank_strategy", lambda result, *_: result)
    monkeypatch.setattr(metric_definition_module, "_resolve_registry_metric", lambda _q: None)
    monkeypatch.setattr(metric_definition_module, "published_rag_version", lambda: "v1")

    state = {"user_message": original, "task_type": "metric_definition",
             "context_compilation": {"standalone_question": rewritten},
             "trace_id": "trace-1", "evidence_bundles": []}
    output = metric_definition_module.metric_definition(state)

    assert seen == [original, rewritten]
    assert {hit["chunk_id"] for hit in output["rag_result"]["hits"]} == {"exact", "shared", "semantic"}
    assert output["rag_query_trace"]["fusion"] == "rrf"


def test_grounded_answer_rejects_numbers_not_in_evidence(monkeypatch):
    monkeypatch.setattr("app.llm.client.llm_chat", lambda **_kwargs: "GMV ? 999 ??")
    evidence = {"value": 100}
    answer, status = _grounded_narrative(
        _state_with_answer_evidence("GMV ???", evidence), evidence, "GMV ? 100 ??",
    )

    assert answer == "GMV ? 100 ??"
    assert status == "template_fallback_numeric_guard"


def test_grounded_answer_accepts_supported_numbers(monkeypatch):
    monkeypatch.setattr("app.llm.client.llm_chat", lambda **_kwargs: "GMV ? 100.00 ??")
    evidence = {"value": 100}
    answer, status = _grounded_narrative(
        _state_with_answer_evidence("GMV ???", evidence), evidence, "fallback",
    )

    assert answer == "GMV ? 100.00 ??"
    assert status == "llm_grounded"


def test_grounded_answer_falls_back_when_llm_omits_breakdown_rows(monkeypatch):
    monkeypatch.setattr("app.llm.client.llm_chat", lambda **_kwargs: "a=1")
    evidence = {"query_result": {"result": {"breakdown_result": {"items": [
        {"dimension_value": "a", "value": 1}, {"dimension_value": "b", "value": 2},
    ]}}}}

    answer, status = _grounded_narrative(
        _state_with_answer_evidence("breakdown", evidence), evidence, "a=1, b=2",
    )

    assert answer == "a=1, b=2"
    assert status == "template_fallback_incomplete_breakdown"
