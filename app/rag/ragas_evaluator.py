"""Reference-light Ragas evaluation for the project's real retrieval path.

The evaluator runs the selected RAG release (or the current configured
retriever), generates an answer with the same grounded narrative helper used by
the chat graph, and scores the resulting question/answer/contexts.  Context
recall uses the retrieval dataset's stable expected document IDs, so no gold
answer needs to be authored.  Ragas evidence is supplementary; deterministic
release contracts remain the release-blocking gate.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import text

import app.rag.ragas_compat  # noqa: F401  # Install compatibility stubs before importing Ragas.
from app.data.database import engine
from app.rag.release_controller import get_release, record_evaluation
from app.quality.evaluations import add_evidence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "evaluation" / "retrieval" / "queries.jsonl"
OUTPUT_PATH = ROOT / "data" / "evaluations" / "ragas-latest.json"
DATASET_VERSION = "ragas-retrieval-queries-v1"
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


def load_cases(dataset_path: str | Path = DEFAULT_DATASET, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load and validate the existing retrieval golden set."""
    path = Path(dataset_path)
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [case for case in cases if case.get("query") and isinstance(case.get("expected_doc_ids"), list)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"No valid Ragas evaluation cases found in {path}")
    return cases


def _result_value(result: Any) -> float:
    value = getattr(result, "value", result)
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Ragas metric score must be between 0 and 1, got {score}")
    return score


def _assert_requested_backend(result: Any, requested_backend: str | None) -> None:
    """Do not let a Qdrant outage masquerade as a Qdrant evaluation."""
    if requested_backend not in {"qdrant", "qdrant_vector", "qdrant_hybrid"}:
        return
    diagnostics = getattr(result, "diagnostics", None)
    actual_backend = str(getattr(result, "backend", ""))
    if actual_backend not in {"qdrant", "qdrant_vector", "qdrant_hybrid"} or (
        diagnostics and getattr(diagnostics, "degraded", False)
    ):
        reason = getattr(diagnostics, "degraded_reason", None) or "backend_mismatch"
        raise RuntimeError(
            f"Requested {requested_backend} Ragas evaluation, but retrieval used "
            f"{actual_backend or 'unknown'} ({reason}); refusing to score fallback results"
        )


def _build_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
    """Create current Ragas 0.4 metrics using the project's configured models."""
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    # Ragas 0.4.3 still exposes this deterministic metric only through the
    # legacy namespace; its collections package has not migrated it yet.
    from ragas.metrics import IDBasedContextRecall
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        Faithfulness,
    )

    from app.core.config import get_settings
    from app.llm.routing import model_for

    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("Ragas evaluation requires DEEPSEEK_API_KEY or an explicitly configured judge provider")
    client = AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url or None,
        timeout=settings.llm_timeout_seconds,
    )
    judge_model, _ = model_for("long_answer")
    judge = llm_factory(judge_model, client=client)
    embeddings = HuggingFaceEmbeddings(model=settings.rag_embedding_model)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecisionWithoutReference(llm=judge),
        "context_recall": IDBasedContextRecall(),
    }
    evaluator = {
        "package": "ragas",
        "version": __import__("ragas").__version__,
        "judge_model": judge_model,
        "embedding_model": settings.rag_embedding_model,
    }
    return metrics, evaluator


def _generate_grounded_answer(question: str, retrieval_result: Any) -> tuple[str, str]:
    """Use the same evidence-bound answer helper as the production graph."""
    from app.graph.nodes.answer import _grounded_narrative

    citations = [item.model_dump(mode="json") for item in retrieval_result.citations]
    hits = [item.model_dump(mode="json") for item in retrieval_result.hits]
    evidence = {
        "authoritative_registry_definition": None,
        "retrieved_knowledge": hits,
        "citations": citations,
    }
    fallback = (
        f"知识库检索结果：\n{retrieval_result.context}"
        if retrieval_result.context
        else "当前知识库没有检索到可用于回答该问题的内容。"
    )
    answer, grounding_status = _grounded_narrative(
        {"user_message": question}, evidence, fallback,
    )
    return answer, grounding_status


async def _score_cases(
    cases: list[dict[str, Any]],
    *,
    retrieve: Callable[[str], Any],
    generate: Callable[[str, Any], tuple[str, str]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Run the actual per-case retrieval, generation, and Ragas metric calls."""
    from ragas.dataset_schema import SingleTurnSample

    details: list[dict[str, Any]] = []
    for case in cases:
        question = str(case["query"])
        case_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retrieved = retrieve(question)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        generation_started = time.perf_counter()
        answer, grounding_status = generate(question, retrieved)
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000
        contexts = [str(hit.text) for hit in retrieved.hits]
        retrieved_doc_ids = sorted({str(hit.citation.source_path) for hit in retrieved.hits})
        expected_doc_ids = sorted({str(item) for item in case["expected_doc_ids"]})

        scoring_started = time.perf_counter()
        faithfulness, answer_relevancy, context_precision = await asyncio.gather(
            metrics["faithfulness"].ascore(
                user_input=question, response=answer, retrieved_contexts=contexts,
            ),
            metrics["answer_relevancy"].ascore(
                user_input=question, response=answer,
            ),
            metrics["context_precision"].ascore(
                user_input=question, response=answer, retrieved_contexts=contexts,
            ),
        )
        context_recall = None
        if expected_doc_ids:
            context_recall = await metrics["context_recall"].single_turn_ascore(
                SingleTurnSample(
                    retrieved_context_ids=retrieved_doc_ids,
                    reference_context_ids=expected_doc_ids,
                )
            )

        diagnostics = getattr(retrieved, "diagnostics", None)
        details.append({
            "query_id": str(case.get("query_id") or ""),
            "question": question,
            "scores": {
                "faithfulness": _result_value(faithfulness),
                "answer_relevancy": _result_value(answer_relevancy),
                "context_precision": _result_value(context_precision),
                "context_recall": _result_value(context_recall) if context_recall is not None else None,
            },
            "retrieval": {
                "backend": str(retrieved.backend),
                "strategy": str(retrieved.strategy),
                "release_id": retrieved.filters.get("release_id"),
                "hit_count": len(retrieved.hits),
                "retrieved_doc_ids": retrieved_doc_ids,
                "expected_doc_ids": expected_doc_ids,
                "degraded": bool(getattr(diagnostics, "degraded", False)),
                "degraded_reason": getattr(diagnostics, "degraded_reason", None),
            },
            "grounding_status": grounding_status,
            "timing_ms": {
                "retrieval": round(retrieval_latency_ms, 1),
                "answer_generation": round(generation_latency_ms, 1),
                "judge_scoring": round((time.perf_counter() - scoring_started) * 1000, 1),
                "total_evaluation": round((time.perf_counter() - case_started) * 1000, 1),
            },
        })

    metric_names = tuple(DEFAULT_THRESHOLDS)
    summary: dict[str, Any] = {"cases": len(details)}
    for name in metric_names:
        values = [item["scores"][name] for item in details if item["scores"][name] is not None]
        summary[name] = round(statistics.mean(values), 4) if values else None
        summary[f"{name}_cases"] = len(values)
    summary["p95_total_evaluation_ms"] = round(
        sorted(item["timing_ms"]["total_evaluation"] for item in details)[max(0, int(len(details) * 0.95) - 1)], 1,
    )
    return {"summary": summary, "details": details}


def _dataset_digest(cases: list[dict[str, Any]]) -> str:
    payload = json.dumps(cases, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_output(payload: dict[str, Any], output_path: str | Path = OUTPUT_PATH) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def evaluate_current(
    *, dataset_path: str | Path = DEFAULT_DATASET, limit: int | None = None,
    backend_override: str | None = None, output_path: str | Path = OUTPUT_PATH,
) -> dict[str, Any]:
    """Evaluate the currently configured retriever without requiring release DB state."""
    from app.rag.retriever import retrieve_knowledge
    from app.rag.config import get_rag_settings

    cases = load_cases(dataset_path, limit=limit)
    metrics, evaluator = _build_metrics()
    expected_backend = backend_override or get_rag_settings().backend

    def retrieve(question: str):
        result = retrieve_knowledge(question, backend_override=backend_override)
        _assert_requested_backend(result, expected_backend)
        return result

    scored = asyncio.run(_score_cases(
        cases, retrieve=retrieve, generate=_generate_grounded_answer, metrics=metrics,
    ))
    payload = {
        "kind": "ragas",
        "status": "completed",
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": _dataset_digest(cases),
        "backend_override": backend_override,
        "requested_backend": expected_backend,
        "evaluator": evaluator,
        **scored,
    }
    _write_output(payload, output_path)
    return payload


def _record(release_id: str, actor: str, release: dict, **kwargs) -> dict:
    result = record_evaluation(
        release_id, actor, "ragas", baseline_release_id=release.get("parent_release_id"),
        dataset_version=DATASET_VERSION, **kwargs,
    )
    from app.quality.evaluations import list_runs
    run = next((item for item in list_runs() if item["subject_type"] == "rag_release" and item["subject_id"] == release_id), None)
    if run:
        add_evidence(
            run["run_id"], "rag_release_evaluation", result["evaluation_id"], "online",
            {"kind": "ragas", "status": result["status"]},
        )
    return result


def evaluate_release(
    release_id: str, actor: str = "rag-eval-worker", *,
    dataset_path: str | Path = DEFAULT_DATASET, limit: int | None = None,
) -> dict:
    """Evaluate and persist Ragas evidence for one immutable release artifact."""
    from app.rag.retriever import retrieve_for_release

    release = get_release(release_id)
    if not release:
        raise ValueError("Release not found")
    cases = load_cases(dataset_path, limit=limit)
    expected_backend = str((release.get("spec") or {}).get("backend") or "")

    def retrieve(question: str):
        result = retrieve_for_release(question, release)
        _assert_requested_backend(result, expected_backend)
        return result

    try:
        metrics, evaluator = _build_metrics()
        scored = asyncio.run(_score_cases(
            cases,
            retrieve=retrieve,
            generate=_generate_grounded_answer,
            metrics=metrics,
        ))
        summary = scored["summary"]
        thresholds = dict(DEFAULT_THRESHOLDS)
        measured = {name: summary.get(name) for name in thresholds}
        passed = all(value is not None and value >= thresholds[name] for name, value in measured.items())
        result = _record(
            release_id, actor, release,
            metrics={"status": "completed", **summary, "details": scored["details"]},
            passed=passed,
            thresholds=thresholds,
            evaluator={**evaluator, "dataset_sha256": _dataset_digest(cases), "metric_mode": "reference_light"},
        )
        return {**result, "metrics": summary, "details": scored["details"]}
    except ImportError as exc:
        return _record(
            release_id, actor, release,
            metrics={"status": "unavailable"}, passed=None,
            evaluator={"package": "ragas", "reason": "dependency_not_installed"},
            failure_reason=f"Ragas evaluation dependency unavailable: {type(exc).__name__}",
        )
    except RuntimeError as exc:
        if "DEEPSEEK_API_KEY" not in str(exc):
            raise
        return _record(
            release_id, actor, release,
            metrics={"status": "unavailable"}, passed=None,
            evaluator={"package": "ragas", "reason": "judge_not_configured"},
            failure_reason="Configure an evaluation LLM before running Ragas",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the project's end-to-end Ragas evaluation")
    parser.add_argument("release_id", nargs="?", help="Evaluate and persist one immutable RAG release")
    parser.add_argument("--current", action="store_true", help="Evaluate the currently configured retrieval backend")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("local_hybrid", "qdrant_vector", "qdrant_hybrid"))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.current:
        result = evaluate_current(
            dataset_path=args.dataset, limit=args.limit,
            backend_override=args.backend, output_path=args.output,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Ragas report written to {args.output}")
        return
    if args.release_id:
        print(json.dumps(
            evaluate_release(args.release_id, dataset_path=args.dataset, limit=args.limit),
            ensure_ascii=False, indent=2,
        ))
        return
    if not args.watch:
        parser.error("provide release_id, --current, or --watch")
    while True:
        with engine.begin() as conn:
            releases = conn.execute(text("""SELECT release_id FROM governance.rag_releases r
              WHERE r.status IN ('evaluated','shadowing','canary_5','canary_25','canary_100')
                AND NOT EXISTS (SELECT 1 FROM governance.rag_release_evaluations e
                  WHERE e.release_id=r.release_id AND e.kind='ragas')""")).scalars().all()
        for release_id in releases:
            evaluate_release(release_id)
        time.sleep(30)


if __name__ == "__main__":
    main()
