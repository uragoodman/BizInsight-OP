from fastapi.testclient import TestClient

from app.main import app
from app.rag.config import get_rag_settings
from app.rag.indexing import discover_knowledge_files, load_knowledge_chunks, load_knowledge_documents
from app.rag.retriever import retrieve_knowledge


client = TestClient(app)


def test_knowledge_documents_are_loaded_with_metadata() -> None:
    files = discover_knowledge_files()
    documents = load_knowledge_documents()

    assert files
    assert documents
    assert any(document.metric_name == "gmv" for document in documents)


def test_knowledge_chunks_record_index_version() -> None:
    chunks = load_knowledge_chunks()

    assert chunks
    assert all(chunk.index_version == get_rag_settings().index_version for chunk in chunks)
    assert all(chunk.document_hash for chunk in chunks)


def test_retrieve_gmv_definition_returns_citation() -> None:
    result = retrieve_knowledge("GMV 是什么？", doc_types={"metric"})

    assert result.hits
    assert result.strategy in {"local_hybrid", "qdrant_vector", "qdrant_hybrid"}
    assert result.index_version == get_rag_settings().index_version
    assert result.hits[0].citation.source_path.endswith("docs/knowledge/metrics/gmv.md")
    assert result.hits[0].citation.index_version == get_rag_settings().index_version
    assert result.hits[0].signals.rerank_score > 0
    assert "GMV" in result.context


def test_retrieve_uses_metadata_filters_for_templates() -> None:
    result = retrieve_knowledge("生成周报需要哪些模块？", doc_types={"template"})

    assert result.hits
    assert all(hit.citation.doc_type == "template" for hit in result.hits)
    assert result.hits[0].citation.source_path.endswith("docs/knowledge/templates/weekly_report.md")


def test_chat_trace_records_rag_citations() -> None:
    response = client.post("/chat", json={"message": "GMV 是什么？"})
    body = response.json()

    assert response.status_code == 200
    trace_response = client.get(f"/traces/{body['trace_id']}")
    trace = trace_response.json()

    assert trace_response.status_code == 200
    sn = trace["state_snapshot"]
    assert sn["rag_result"]["strategy"] in {"local_hybrid", "qdrant_vector", "qdrant_hybrid"}
    assert sn["rag_result"]["index_version"] == get_rag_settings().index_version
    assert sn["rag_result"]["hits"]
    assert "signals" in sn["rag_result"]["hits"][0]
    assert "vector_score" in sn["rag_result"]["hits"][0]["signals"]
    assert sn["rag_citations"]
    assert sn["rag_citations"][0]["source_path"].endswith("docs/knowledge/metrics/gmv.md")
    assert sn["critic_result"]["rag_review"]["status"] == "approved"


def test_eval_run_reports_rag_quality(monkeypatch) -> None:
    # This endpoint runs multiple graph cases. Keep the unit test deterministic
    # and offline; provider behavior is covered by dedicated LLM integration
    # tests, not by this RAG retrieval-quality assertion.
    monkeypatch.setattr("app.graph.nodes.triage.llm_classify", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("app.llm.client.llm_agent_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.agents.critic_agent.llm_extract", lambda *_args, **_kwargs: None)

    response = client.post("/eval/run")
    body = response.json()

    assert response.status_code == 200
    assert body["rag"]["hit_rate"] >= 0.66
