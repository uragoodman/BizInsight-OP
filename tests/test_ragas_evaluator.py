from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.rag.ragas_evaluator import (
    _assert_requested_backend,
    _result_value,
    _score_cases,
    load_cases,
)


class FakeMetric:
    def __init__(self, value: float):
        self.value = value
        self.calls: list[dict] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)

    async def single_turn_ascore(self, sample):
        self.calls.append(sample)
        return SimpleNamespace(value=self.value)


def _retrieval_result():
    hit = SimpleNamespace(
        text="GMV 是商品交易总额。",
        citation=SimpleNamespace(source_path="docs/knowledge/metrics/gmv.md"),
    )
    return SimpleNamespace(
        hits=[hit],
        citations=[],
        context=hit.text,
        backend="local",
        strategy="local_hybrid",
        filters={},
        diagnostics=SimpleNamespace(degraded=False, degraded_reason=None),
    )


def test_score_cases_runs_four_metrics_and_uses_document_ids_for_recall():
    metrics = {
        "faithfulness": FakeMetric(0.9),
        "answer_relevancy": FakeMetric(0.8),
        "context_precision": FakeMetric(0.7),
        "context_recall": FakeMetric(1.0),
    }
    result = asyncio.run(_score_cases(
        [{
            "query_id": "gmv",
            "query": "GMV是什么",
            "expected_doc_ids": ["docs/knowledge/metrics/gmv.md"],
        }],
        retrieve=lambda _: _retrieval_result(),
        generate=lambda question, _: (f"回答：{question}", "llm_grounded"),
        metrics=metrics,
    ))

    assert result["summary"]["cases"] == 1
    assert result["summary"]["faithfulness"] == 0.9
    assert result["details"][0]["scores"]["context_recall"] == 1.0
    recall_sample = metrics["context_recall"].calls[0]
    assert recall_sample.retrieved_context_ids == ["docs/knowledge/metrics/gmv.md"]
    assert recall_sample.reference_context_ids == ["docs/knowledge/metrics/gmv.md"]


def test_score_cases_omits_id_recall_for_unanswerable_case():
    metrics = {
        "faithfulness": FakeMetric(0.4),
        "answer_relevancy": FakeMetric(0.5),
        "context_precision": FakeMetric(0.2),
        "context_recall": FakeMetric(1.0),
    }
    result = asyncio.run(_score_cases(
        [{"query_id": "none", "query": "未覆盖的问题", "expected_doc_ids": []}],
        retrieve=lambda _: _retrieval_result(),
        generate=lambda question, _: ("无法从知识库确认。", "llm_grounded"),
        metrics=metrics,
    ))

    assert result["details"][0]["scores"]["context_recall"] is None
    assert not metrics["context_recall"].calls
    assert result["summary"]["context_recall_cases"] == 0


def test_load_cases_reads_existing_retrieval_dataset():
    cases = load_cases(limit=2)

    assert len(cases) == 2
    assert all(case["query"] and isinstance(case["expected_doc_ids"], list) for case in cases)


def test_metric_score_must_be_normalized():
    assert _result_value(SimpleNamespace(value=0.75)) == 0.75

    try:
        _result_value(SimpleNamespace(value=1.2))
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("out-of-range metric scores must fail")


def test_qdrant_evaluation_rejects_fallback_backend():
    degraded = _retrieval_result()
    degraded.diagnostics = SimpleNamespace(degraded=True, degraded_reason="qdrant_query_failed")

    try:
        _assert_requested_backend(degraded, "qdrant_vector")
    except RuntimeError as exc:
        assert "refusing to score fallback results" in str(exc)
    else:
        raise AssertionError("Qdrant fallback must not be reported as a Qdrant evaluation")
