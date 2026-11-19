# -*- coding: utf-8 -*-
"""Real Mem0 OSS + Qdrant + PostgreSQL Registry recall evaluation (opt-in).

This is the T4 evidence that the deterministic proxy suites cannot provide:
it seeds a labeled preference corpus into a real Qdrant collection, registers
the Mem0 IDs in a migrated PostgreSQL Registry, and measures retrieval
precision/recall plus rejection of scope/identity/expired/inactive records.

Required environment:
  PERSONALIZATION_REAL_QDRANT_URL   e.g. http://127.0.0.1:16333
  PERSONALIZATION_REAL_DATABASE_URL migrated disposable PostgreSQL URL

It does not modify app code, migrations or the application database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.memory.mem0_adapter import Mem0PreferenceAdapter  # noqa: E402
from app.memory.preference_provider import Mem0PreferenceProvider, _query_for_key  # noqa: E402
from app.context.turn_resolver import (  # noqa: E402
    PREFERENCE_INTENT_RULES_VERSION,
    detect_preference_source_needs,
    evaluate_preference_candidates,
    resolve_preference_context,
)
from app.memory.preference_worker import _render_preference  # noqa: E402
from app.memory.preference_registry import PostgresPreferenceRegistry  # noqa: E402
from app.tenancy import TenantContext  # noqa: E402

EVIDENCE_INPUTS = (
    "app/context/turn_resolver.py",
    "app/context/negative_intent.py",
    "app/memory/preference_provider.py",
    "app/memory/mem0_adapter.py",
    "app/memory/preference_worker.py",
    "app/memory/preference_registry.py",
    "evaluation/personalization_recall_real_eval.py",
)


HMAC_SECRET = "0123456789abcdef0123456789abcdef"
SCOPE_VERSION = "v1"
THRESHOLD = 0.66
TOP_K_PER_TYPE = 5
TOP_K_TOTAL = 10

PREFERENCES = [
    {
        "key": "response_style", "type": "presentation", "value": {"style": "concise"},
        "text": "回答风格偏好：保持简洁，先给结论。", "status": "active",
    },
    {
        "key": "report_style", "type": "presentation", "value": {"format": "table"},
        "text": "报告排版偏好：使用表格展示。", "status": "active",
    },
    {
        "key": "response_language", "type": "presentation", "value": {"language": "en-US"},
        "text": "输出语言偏好：使用英文回答。", "status": "active",
    },
    {
        "key": "analysis_method", "type": "analysis_method",
        "value": {"operation": "trend", "dimension": "product_category"},
        "text": "分析习惯偏好：先看趋势；按商品品类拆解。", "status": "active",
    },
    {
        "key": "default_period", "type": "scope_default",
        "value": {"start_date": "2026-07-01", "end_date": "2026-07-31"},
        "text": "默认统计周期：最近一个月。", "status": "active",
    },
    {
        "key": "report_footer", "type": "presentation", "value": {"footer": "page_number"},
        "text": "报告页脚偏好：包含页码。", "status": "active",
    },
    {
        "key": "analysis_chart", "type": "analysis_method", "value": {"chart": "blue"},
        "text": "图表配色偏好：使用蓝色。", "status": "active",
    },
    {
        "key": "default_region", "type": "scope_default", "value": {"region": "east"},
        "text": "默认地区偏好：华东地区。", "status": "active",
    },
]

PRODUCTION_QUERIES = [
    {
        "query": "按我平时的方式回答",
        "expected": ["response_style", "response_language"],
    },
    {
        "query": "按我平常的回答方式回复",
        "expected": ["response_style", "response_language"],
    },
    {
        "query": "保持我常用的回答方式",
        "expected": ["response_style", "response_language"],
    },
    {"query": "报告按我平时的格式", "expected": ["report_style"]},
    {"query": "用我常用的报告格式生成报告", "expected": ["report_style"]},
    {"query": "按我平时的分析方式分析", "expected": ["analysis_method"]},
    {"query": "按我平常的分析方式处理", "expected": ["analysis_method"]},
    {"query": "用我的默认统计周期看 GMV", "expected": ["default_period"]},
    {"query": "按我平常的默认时间范围统计 GMV", "expected": ["default_period"]},
    {"query": "按我常用的时间范围统计 GMV", "expected": ["default_period"]},
    {"query": "按我平时的回答风格回答", "expected": ["response_style", "response_language"]},
    {"query": "报告按我平时的排版格式", "expected": ["report_style"]},
    {
        "query": "按我平时常用的报告格式输出",
        "expected": ["report_style", "response_style", "response_language"],
    },
    {"query": "按我平常的分析习惯处理 GMV", "expected": ["analysis_method"]},
    {"query": "用我的默认统计周期分析 GMV", "expected": ["analysis_method", "default_period"]},
    {"query": "按我平时常用的时间范围看 GMV", "expected": ["default_period"]},
    {"query": "按我平常的报告格式生成报告", "expected": ["report_style"]},
    {"query": "按我平时常用的统计周期看 GMV", "expected": ["default_period"]},
    {"query": "按我平常的回答风格答复", "expected": ["response_style", "response_language"]},
    {"query": "根据我平时的分析方式分析 GMV", "expected": ["analysis_method"]},
    {"query": "按我平时的表达方式回答", "expected": ["response_style", "response_language"]},
    {"query": "按我平常的说话方式回复", "expected": ["response_style", "response_language"]},
    {"query": "保持我常用的表达习惯", "expected": ["response_style", "response_language"]},
    {"query": "用我平时的回答风格", "expected": ["response_style", "response_language"]},
    {"query": "用我常用的语言回答", "expected": ["response_style", "response_language"]},
    {"query": "报告按我平常的版式", "expected": ["report_style"]},
    {
        "query": "按我常用的报告排版输出",
        "expected": ["report_style", "response_style", "response_language"],
    },
    {"query": "按我平时的报告格式生成", "expected": ["report_style"]},
    {"query": "生成周报时按我平时格式", "expected": ["report_style"]},
    {"query": "按我平常的排版方式写报告", "expected": ["report_style"]},
    {"query": "报告按我平时的格式来", "expected": ["report_style"]},
    {"query": "按我平常的报告排版方式生成报告", "expected": ["report_style"]},
    {"query": "按我平时的分析步骤处理 GMV", "expected": ["analysis_method"]},
    {"query": "按我平常的分析方法分析 GMV", "expected": ["analysis_method"]},
    {"query": "使用我平时的分析习惯看 GMV", "expected": ["analysis_method"]},
    {"query": "按我常用分析方式拆解 GMV", "expected": ["analysis_method"]},
    {"query": "按照我平时的分析流程分析 GMV", "expected": ["analysis_method"]},
    {"query": "按我平时默认的时间范围查询 GMV", "expected": ["default_period"]},
    {"query": "用我平常的统计周期看 GMV", "expected": ["default_period"]},
    {"query": "按我平常的默认时间范围查询 GMV", "expected": ["default_period"]},
    {"query": "按我平时的默认周期统计 GMV", "expected": ["default_period"]},
    {"query": "用我常用的时间范围看 GMV", "expected": ["default_period"]},
    {"query": "按我平时的回答方式说", "expected": ["response_style", "response_language"]},
    {"query": "按我平常的回答风格回复", "expected": ["response_style", "response_language"]},
    {"query": "报告按我常用的版式", "expected": ["report_style"]},
    {
        "query": "按我平时的报告排版方式输出",
        "expected": ["report_style", "response_style", "response_language"],
    },
    {"query": "按我平常的分析习惯处理", "expected": ["analysis_method"]},
    {"query": "用我平时的分析方式分析 GMV", "expected": ["analysis_method"]},
    {"query": "按我平常的回复风格回答", "expected": ["response_style", "response_language"]},
    {
        "query": "报告按我平时的版式输出",
        "expected": ["report_style", "response_style", "response_language"],
    },
    {"query": "用我平常的分析步骤处理 GMV", "expected": ["analysis_method"]},
    {"query": "保持我常用的表达方式", "expected": ["response_style", "response_language"]},
    {"query": "按我平时的回复方式", "expected": ["response_style", "response_language"]},
    {"query": "按照我常用的表达方式", "expected": ["response_style", "response_language"]},
    {"query": "按我平时的报告版式", "expected": ["report_style"]},
    {"query": "按我平常的报告写法", "expected": ["report_style"]},
    {"query": "按照我常用的报告格式", "expected": ["report_style"]},
    {"query": "用我平时的汇报格式", "expected": ["report_style"]},
    {"query": "按我平时的分析套路", "expected": ["analysis_method"]},
    {"query": "沿用我通常的分析套路", "expected": ["analysis_method"]},
    {"query": "按我平常的分析思路", "expected": ["analysis_method"]},
    {"query": "用我平时的分析步骤", "expected": ["analysis_method"]},
    {
        "query": "按我平常的方式回答并生成报告",
        "expected": ["report_style", "response_style", "response_language"],
    },
    {"query": "用我的默认统计周期并保持我常用的语言", "expected": ["default_period"]},
    {
        "query": "按我平时的分析方式并保持简洁回答",
        "expected": ["analysis_method", "response_language"],
    },
    {"query": "按我平时的回复方式回复", "expected": ["response_style", "response_language"]},
    {"query": "保持我常用的答复习惯", "expected": ["response_style", "response_language"]},
    {"query": "按照我平常的表达方式", "expected": ["response_style", "response_language"]},
    {"query": "按我平时的报告排版", "expected": ["report_style"]},
    {"query": "按我平常的报告版式", "expected": ["report_style"]},
    {"query": "用我常用的分析思路", "expected": ["analysis_method"]},
    {"query": "按我平时的分析流程", "expected": ["analysis_method"]},
    {"query": "用我平常的默认统计周期", "expected": ["default_period"]},
    {"query": "按我常用的统计周期", "expected": ["default_period"]},
    {"query": "按我平时的默认时间范围", "expected": ["default_period"]},
]

E1_4_QUERIES = [
    {"query": "按我平时的语气回答", "expected": ["response_style", "response_language"]},
    {"query": "按我平常的语气回复", "expected": ["response_style", "response_language"]},
    {"query": "用我常用的语气回答", "expected": ["response_style", "response_language"]},
    {"query": "按我的习惯表达", "expected": ["response_style", "response_language"]},
    {"query": "按我平时的说法回答", "expected": ["response_style", "response_language"]},
    {"query": "按我平常的说法回复", "expected": ["response_style", "response_language"]},
    {"query": "用我常用的措辞回答", "expected": ["response_style", "response_language"]},
    {"query": "按我通常的表达方式回答", "expected": ["response_style", "response_language"]},
    {"query": "按我平常的说话方式回复", "expected": ["response_style", "response_language"]},
    {"query": "按我平时的报告排版", "expected": ["report_style"]},
    {"query": "按我平常的报告格式", "expected": ["report_style"]},
    {"query": "用我常用的报告样式", "expected": ["report_style"]},
    {"query": "保持我通常的报告版式", "expected": ["report_style"]},
    {"query": "按我的习惯生成报告", "expected": ["report_style"]},
    {"query": "按我常用的报告版式", "expected": ["report_style"]},
    {"query": "按我平时的分析思路", "expected": ["analysis_method"]},
    {"query": "按我平常的分析逻辑", "expected": ["analysis_method"]},
    {"query": "用我常用的分析步骤", "expected": ["analysis_method"]},
    {"query": "保持我通常的分析方式", "expected": ["analysis_method"]},
    {"query": "按我的习惯拆解 GMV", "expected": ["analysis_method"]},
    {"query": "按我平常的分析方法处理 GMV", "expected": ["analysis_method"]},
    {"query": "按我平时的默认时间范围", "expected": ["default_period"]},
    {"query": "按我平常的默认统计周期", "expected": ["default_period"]},
    {"query": "用我常用的默认日期", "expected": ["default_period"]},
    {"query": "保持我通常的时间范围", "expected": ["default_period"]},
    {"query": "按我平时的默认周期看 GMV", "expected": ["default_period"]},
    {
        "query": "按我平常的方式回答并生成报告",
        "expected": ["report_style", "response_style", "response_language"],
    },
    {
        "query": "按我平时的方式回答并总结",
        "expected": ["report_style", "response_style", "response_language"],
    },
    {
        "query": "用我的默认统计周期并保持我常用的措辞",
        "expected": ["default_period"],
    },
    {
        "query": "按我平常的分析方式并回答简洁",
        "expected": ["analysis_method", "response_language"],
    },
    {
        "query": "按我平时的报告格式并回答英文",
        "expected": ["report_style", "response_style"],
    },
]

GOLD_VALUES = {
    "response_style": {"style": "concise"},
    "report_style": {"format": "table"},
    "response_language": {"language": "en-US"},
    "analysis_method": {"operation": "trend", "dimension": "product_category"},
    "default_period": {"start_date": "2026-07-01", "end_date": "2026-07-31"},
}

QUERIES = [
    {"query": "回答风格偏好：保持简洁", "expected": ["response_style"], "types": ["presentation"]},
    {"query": "报告排版偏好：使用表格展示", "expected": ["report_style"], "types": ["presentation"]},
    {"query": "输出语言偏好：使用英文回答", "expected": ["response_language"], "types": ["presentation"]},
    {"query": "分析步骤偏好：先做同比对比", "expected": ["analysis_operation"], "types": ["analysis_method"]},
    {"query": "默认统计周期：最近一个月", "expected": ["default_period"], "types": ["scope_default"]},
]


def _need_allowed_keys(need: dict[str, object]) -> dict[str, list[str]]:
    return {
        str(kind): [str(key) for key in (keys or [])]
        for kind, keys in (need.get("allowed_keys") or {}).items()
    }


def _variant_hits(
    adapter: Mem0PreferenceAdapter,
    query: str,
    needs: list[dict[str, object]],
    threshold: float,
    *,
    mode: str,
) -> list[object]:
    """Collect S0/D/B candidates without changing the authorization boundary."""
    by_id: dict[str, object] = {}
    for need in needs:
        types = list(need.get("memory_types") or [])
        allowed = _need_allowed_keys(need)
        for memory_type in types:
            keys = allowed.get(str(memory_type)) or [None]
            if mode == "ordinary":
                keys = [None]
            for memory_key in keys:
                hit_query = (
                    _query_for_key(query, str(memory_key))
                    if memory_key and mode != "ordinary"
                    else query
                )
                hits = adapter.search(
                    hit_query,
                    memory_types=[memory_type],
                    memory_keys=None,
                    top_k_per_type=TOP_K_PER_TYPE,
                    top_k_total=TOP_K_TOTAL,
                    threshold=threshold,
                )
                for hit in hits:
                    by_id[hit.memory_id] = hit
    selected = sorted(
        by_id.values(), key=lambda hit: (hit.score or 0.0), reverse=True,
    )
    if mode == "key_only":
        allowed_keys = {
            key
            for need in needs
            for values in _need_allowed_keys(need).values()
            for key in values
        }
        selected = [hit for hit in selected if hit.memory_key in allowed_keys]
    return selected


def _keyword_hits(
    query: str,
    documents: list[dict[str, object]],
    *,
    top_k: int = TOP_K_TOTAL,
) -> list[dict[str, object]]:
    query_chars = set(re.findall(r"[一-鿿]|[a-z0-9]+", query.casefold()))
    scored = []
    for document in documents:
        text_chars = set(re.findall(
            r"[一-鿿]|[a-z0-9]+", str(document.get("text") or "").casefold(),
        ))
        overlap = len(query_chars & text_chars)
        if overlap:
            score = overlap / max(1, len(query_chars))
            scored.append((score, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "memory_key": str(document["key"]),
            "memory_type": str(document["type"]),
            "source_ref": f"keyword:{document['key']}",
            "confidence": score,
        }
        for score, document in scored[:top_k]
    ]


def _value_metrics(hits: list[object], expected_keys: list[str]) -> dict[str, object]:
    values_by_key: dict[str, list[object]] = {key: [] for key in expected_keys}
    for hit in hits:
        if isinstance(hit, dict):
            key = str(hit.get("memory_key") or "")
            value = hit.get("value")
        else:
            key = str(getattr(hit, "memory_key", "") or "")
            value = getattr(hit, "value", None)
        if key in values_by_key:
            values_by_key[key].append(value)
    gold_keys = [key for key in expected_keys if key in GOLD_VALUES]
    top1_correct = 0
    topk_correct = 0
    for key in gold_keys:
        gold = GOLD_VALUES[key]
        values = values_by_key[key]
        top1_correct += int(bool(values) and values[0] == gold)
        topk_correct += int(gold in values)
    return {
        "expected_keys": len(gold_keys),
        "value_correct_top1": top1_correct,
        "value_correct_topk": topk_correct,
        "value_accuracy_top1": top1_correct / len(gold_keys) if gold_keys else None,
        "value_accuracy_topk": topk_correct / len(gold_keys) if gold_keys else None,
    }


def _rank_metrics(actual_keys: list[str], expected_keys: list[str]) -> dict[str, object]:
    expected = list(dict.fromkeys(expected_keys))
    actual = list(actual_keys)
    query_hits = {
        f"@{k}": int(bool(set(expected) & set(actual[:k])))
        for k in (1, 3, 5)
    }
    first_rank = next((i + 1 for i, key in enumerate(actual) if key in set(expected)), None)
    key_hits = {f"@{k}": 0 for k in (1, 3, 5)}
    key_rr = 0.0
    for key in expected:
        rank = next((i + 1 for i, actual_key in enumerate(actual) if actual_key == key), None)
        for k in (1, 3, 5):
            key_hits[f"@{k}"] += int(rank is not None and rank <= k)
        key_rr += 1 / rank if rank else 0.0
    matched = [key for key in actual if key in set(expected)]
    return {
        "query_hit": query_hits,
        "query_mrr": 1 / first_rank if first_rank else 0.0,
        "key_hit_counts": key_hits,
        "key_mrr": key_rr / len(expected) if expected else 0.0,
        "matched": matched,
        "precision": len(matched) / len(actual) if actual else 1.0,
        "recall": len(matched) / len(expected) if expected else 1.0,
    }


def _accepted_variant_keys(
    state: dict[str, object],
    needs: list[dict[str, object]],
    hits: list[object],
) -> list[str]:
    candidates = [
        hit.as_candidate() if hasattr(hit, "as_candidate") else hit
        for hit in hits
    ]
    evaluated = evaluate_preference_candidates(
        state, candidates, None, source_needs=needs,
    )
    return [str(item.get("memory_key") or "") for item in evaluated.get("accepted") or []]


def _adapter_settings(
    collection: str, qdrant_url: str, history: Path, *, threshold: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        mem0_qdrant_collection=collection,
        mem0_qdrant_url=qdrant_url,
        mem0_qdrant_api_key="",
        mem0_embedding_model=os.getenv("MEM0_TEST_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        mem0_embedding_dims=int(os.getenv("MEM0_TEST_EMBEDDING_DIMS", "512")),
        mem0_hmac_secret=HMAC_SECRET,
        mem0_scope_version=SCOPE_VERSION,
        mem0_history_db_path=str(history),
        mem0_search_timeout_seconds=10.0,
        mem0_write_timeout_seconds=20.0,
        mem0_breaker_failures=5,
        mem0_breaker_window_seconds=30.0,
        mem0_top_k_per_type=TOP_K_PER_TYPE,
        mem0_top_k_total=TOP_K_TOTAL,
        mem0_relevance_threshold=threshold,
        mem0_llm_model="", mem0_llm_api_key="", mem0_llm_base_url="",
        deepseek_api_key="", deepseek_base_url="",
        qdrant_url=qdrant_url, qdrant_collection="bizinsight_knowledge",
        rag_qdrant_active_alias="", metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default=os.getenv("PERSONALIZATION_REAL_QDRANT_URL", ""))
    parser.add_argument("--database-url", default=os.getenv("PERSONALIZATION_REAL_DATABASE_URL", ""))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--production", action="store_true", help="use worker-rendered text and colloquial queries")
    parser.add_argument("--query-set", choices=("production", "e1_4"), default="production")
    args = parser.parse_args()
    if args.query_set == "e1_4":
        args.production = True
    if not args.qdrant_url or not args.database_url:
        print(
            "PERSONALIZATION_REAL_SKIPPED: set PERSONALIZATION_REAL_QDRANT_URL and "
            "PERSONALIZATION_REAL_DATABASE_URL (seeded Qdrant + migrated disposable PostgreSQL)",
            file=sys.stderr,
        )
        return 2

    os.environ["MEM0_RELEVANCE_THRESHOLD"] = str(args.threshold)
    from app.core.config import get_settings

    get_settings.cache_clear()
    from mem0 import Memory

    collection = f"personalization_real_{uuid.uuid4().hex[:8]}"
    history = Path(tempfile.mkdtemp(prefix="personalization_real_")) / "history.db"
    memory = Memory.from_config({
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": collection, "host": "127.0.0.1",
            "port": int(args.qdrant_url.rsplit(":", 1)[-1].rstrip("/")),
            "embedding_model_dims": 512, "on_disk": False,
        }},
        "embedder": {"provider": "huggingface", "config": {
            "model": os.getenv("MEM0_TEST_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            "embedding_dims": 512,
        }},
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "api_key": "not-configured"}},
        "history_db_path": str(history),
    })
    engine = create_engine(args.database_url)
    suffix = uuid.uuid4().hex[:10]
    tenant = f"pref-eval-{suffix}"
    user_a = f"pref-eval-a-{suffix}"
    user_b = f"pref-eval-b-{suffix}"
    context_a = TenantContext(tenant_id=tenant, user_id=user_a, role="admin", scope_type="platform")
    context_b = TenantContext(tenant_id=tenant, user_id=user_b, role="admin", scope_type="platform")
    settings = _adapter_settings(collection, args.qdrant_url, history, threshold=args.threshold)
    adapter_a = Mem0PreferenceAdapter(
        context_a, settings=settings, memory=memory,
        registry=PostgresPreferenceRegistry(context_a, engine_override=engine),
    )
    adapter_b = Mem0PreferenceAdapter(
        context_b, settings=settings, memory=memory,
        registry=PostgresPreferenceRegistry(context_b, engine_override=engine),
    )
    adapter_a._ensure_indexes(memory.vector_store.client)
    seeded: list[dict] = []
    unauthorized_ids: set[str] = set()
    try:
        # Valid corpus for user A.
        for index, item in enumerate(PREFERENCES):
            rendered_text = (
                _render_preference(item["key"], item["type"], item["value"])
                if args.production else item["text"]
            )
            ids = adapter_a.add(
                rendered_text, memory_key=item["key"], memory_type=item["type"],
                operation_id=f"eval-{suffix}-{index}", value=item["value"], infer=False,
            )
            seeded.append({**item, "user": "a", "memory_id": ids[0],
                           "operation_id": f"eval-{suffix}-{index}"})
        # An unrelated-but-same-user record that must not be recalled for the
        # style/language queries if it is below the frozen threshold.  It is a
        # real active preference of another type, so any leak lowers precision.
        # Cross-user record.
        ids = adapter_b.add(
            "回答偏好：保持非常详细。", memory_key="response_style",
            memory_type="presentation", operation_id=f"eval-{suffix}-b",
            value={"style": "detailed"}, infer=False,
        )
        seeded.append({"key": "response_style", "type": "presentation", "user": "b",
                       "memory_id": ids[0], "value": {"style": "detailed"}, "status": "active",
                       "operation_id": f"eval-{suffix}-b"})
        unauthorized_ids.add(ids[0])
        # Same user, wrong seller scope metadata.
        wrong_scope = adapter_a.add(
            "报告格式偏好：使用项目符号。", memory_key="report_style_seller",
            memory_type="presentation", operation_id=f"eval-{suffix}-seller",
            value={"format": "bullets"}, infer=False,
        )
        row = memory.get(wrong_scope[0])
        metadata = dict(row.get("metadata") or {})
        metadata["seller_scope"] = "seller-evil"
        memory.update(wrong_scope[0], metadata=metadata)
        seeded.append({"key": "report_style_seller", "type": "presentation", "user": "a",
                       "memory_id": wrong_scope[0], "value": {"format": "bullets"},
                       "status": "seller_mismatch", "operation_id": f"eval-{suffix}-seller"})
        unauthorized_ids.add(wrong_scope[0])
        # Expired metadata record.
        expired = adapter_a.add(
            "回答语言偏好：使用中文。", memory_key="response_language_expired",
            memory_type="presentation", operation_id=f"eval-{suffix}-expired",
            value={"language": "zh-CN"}, infer=False,
        )
        row = memory.get(expired[0])
        metadata = dict(row.get("metadata") or {})
        metadata["expires_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        memory.update(expired[0], metadata=metadata)
        seeded.append({"key": "response_language_expired", "type": "presentation", "user": "a",
                       "memory_id": expired[0], "value": {"language": "zh-CN"},
                       "status": "expired", "operation_id": f"eval-{suffix}-expired"})
        unauthorized_ids.add(expired[0])
        # Registry inactive record.
        inactive = adapter_a.add(
            "分析偏好：先看趋势。", memory_key="analysis_trend",
            memory_type="analysis_method", operation_id=f"eval-{suffix}-inactive",
            value={"operation": "trend"}, infer=False,
        )
        seeded.append({"key": "analysis_trend", "type": "analysis_method", "user": "a",
                       "memory_id": inactive[0], "value": {"operation": "trend"},
                       "status": "registry_inactive", "operation_id": f"eval-{suffix}-inactive"})
        unauthorized_ids.add(inactive[0])

        # Registry rows are inserted after Mem0 adds so the IDs exist.  The
        # inactive/expired rows deliberately carry a non-active status.
        status_by_status = {
            "active": "active", "seller_mismatch": "active", "expired": "active",
            "registry_inactive": "superseded",
        }
        with engine.begin() as connection:
            for index, item in enumerate(seeded):
                connection.execute(text("""
                    INSERT INTO governance.memory_registry
                        (registry_id, tenant_id, user_id, seller_id, memory_type,
                         memory_key, mem0_memory_id, status, version, operation_id,
                         user_confirmed, expires_at)
                    VALUES (:registry_id, :tenant, :user, NULL, :memory_type,
                            :memory_key, :memory_id, :status, 1, :operation_id,
                            true, :expires_at)
                """), {
                    "registry_id": f"mreg_eval_{suffix}_{index}",
                    "tenant": tenant,
                    "user": user_a if item["user"] == "a" else user_b,
                    "memory_type": item["type"], "memory_key": item["key"],
                    "memory_id": item["memory_id"],
                    "status": status_by_status[item["status"]],
                    "operation_id": item["operation_id"],
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(days=1)
                        if item["status"] == "registry_inactive" else None
                    ),
                })

        results = []
        keyword_documents = [
            item for item in seeded
            if item.get("user") == "a" and item.get("status") == "active"
        ]
        variant_totals: dict[str, dict[str, float]] = {}
        relevant_hits = 0
        total_hits = 0
        expected_total = 0
        provider = Mem0PreferenceProvider(
            context_a, settings=settings, adapter=adapter_a,
            registry=PostgresPreferenceRegistry(context_a, engine_override=engine),
        )
        if args.query_set == "e1_4":
            query_cases = E1_4_QUERIES
        elif args.production:
            query_cases = PRODUCTION_QUERIES
        else:
            query_cases = QUERIES
        for query_case in query_cases:
            if args.production:
                state = {
                    "user_message": query_case["query"],
                    "memory_enabled": True,
                    "chat_session_id": "real-eval",
                    "current_query_scope_fingerprint": "real-eval-scope",
                    "controlled_analysis_scope": {
                        "default_start_date": "2026-07-01",
                        "default_end_date": "2026-07-31",
                    },
                }
                needs = detect_preference_source_needs(state)
                if not needs:
                    raise RuntimeError(f"no preference source need for {query_case['query']!r}")
                for need in needs:
                    need["relevance_threshold"] = args.threshold
                class _RecordingProvider:
                    def __init__(self, wrapped):
                        self.wrapped = wrapped
                        self.raw = []

                    def search(self, *, need, state):
                        found = self.wrapped.search(need=need, state=state)
                        self.raw.extend(found)
                        return found

                recording_provider = _RecordingProvider(provider)
                resolved = resolve_preference_context(state, provider=recording_provider)
                candidates = list(resolved.get("accepted") or [])
                raw_candidates = recording_provider.raw
                actual_keys = [str(item.get("memory_key") or "") for item in raw_candidates]
                actual_scores = [
                    {"key": str(item.get("memory_key") or ""), "score": item.get("confidence")}
                    for item in raw_candidates
                ]
                actual_ids = {str(item.get("source_ref") or "") for item in raw_candidates}
                rejected = [
                    {"key": item.get("memory_key"), "reason": item.get("reason")}
                    for item in resolved.get("rejected") or []
                ]
                source_needs = [
                    {
                        "rules_version": need.get("rules_version"),
                        "intent_ids": need.get("intent_ids"),
                        "memory_types": need.get("memory_types"),
                        "allowed_keys": need.get("allowed_keys"),
                        "allowed_fields": need.get("allowed_fields"),
                        "overridden_fields": need.get("overridden_fields"),
                    }
                    for need in needs
                ]
                accepted_keys = [str(item.get("memory_key") or "") for item in candidates]
                variant_hits: dict[str, list[object]] = {
                    "A_current": recording_provider.raw,
                    "B_ordinary": _variant_hits(
                        adapter_a, query_case["query"], needs, args.threshold,
                        mode="ordinary",
                    ),
                    "C_keyword": _keyword_hits(
                        query_case["query"], keyword_documents,
                    ),
                    "D_key_only": _variant_hits(
                        adapter_a, query_case["query"], needs, args.threshold,
                        mode="key_only",
                    ),
                    "S0_raw": _variant_hits(
                        adapter_a, query_case["query"], needs, 0.2,
                        mode="raw",
                    ),
                }
                variant_details: dict[str, dict[str, object]] = {}
                for variant, variant_hit_list in variant_hits.items():
                    if variant == "A_current":
                        keys = actual_keys
                        accepted_for_variant = accepted_keys
                    elif variant == "C_keyword":
                        keys = [str(item["memory_key"]) for item in variant_hit_list]
                        accepted_for_variant = None
                    else:
                        keys = [str(hit.memory_key) for hit in variant_hit_list]
                        accepted_for_variant = _accepted_variant_keys(
                            state, needs, variant_hit_list,
                        )
                    details = _rank_metrics(keys, query_case["expected"])
                    value_metrics = _value_metrics(
                        variant_hit_list, query_case["expected"],
                    )
                    ids = {
                        str(getattr(hit, "memory_id", "") or getattr(hit, "get", lambda *_: "")("source_ref", ""))
                        for hit in variant_hit_list
                    }
                    details.update({
                        "actual": keys,
                        "accepted": accepted_for_variant,
                        "accepted_metrics": (
                            _rank_metrics(accepted_for_variant, query_case["expected"])
                            if accepted_for_variant is not None else None
                        ),
                        "value_metrics": value_metrics,
                        "unauthorized_leaks": sorted(unauthorized_ids & ids),
                        "unauthorized_seen": sorted(unauthorized_ids & ids),
                    })
                    variant_details[variant] = details
                    totals = variant_totals.setdefault(
                        variant, {
                            "queries": 0, "query_hit@1": 0.0, "query_hit@3": 0.0,
                            "query_hit@5": 0.0, "query_mrr": 0.0,
                            "expected_keys": 0, "key_hit@1": 0.0, "key_hit@3": 0.0,
                            "key_hit@5": 0.0, "key_mrr": 0.0,
                            "relevant": 0, "returned": 0, "expected": 0,
                            "value_expected": 0, "value_top1": 0, "value_topk": 0,
                        },
                    )
                    totals["queries"] += 1
                    totals["query_hit@1"] += float(details["query_hit"]["@1"])
                    totals["query_hit@3"] += float(details["query_hit"]["@3"])
                    totals["query_hit@5"] += float(details["query_hit"]["@5"])
                    totals["query_mrr"] += float(details["query_mrr"])
                    totals["expected_keys"] += len(query_case["expected"])
                    totals["key_hit@1"] += float(details["key_hit_counts"]["@1"])
                    totals["key_hit@3"] += float(details["key_hit_counts"]["@3"])
                    totals["key_hit@5"] += float(details["key_hit_counts"]["@5"])
                    totals["key_mrr"] += float(details["key_mrr"]) * len(query_case["expected"])
                    totals["relevant"] += len(details["matched"])
                    totals["returned"] += len(keys)
                    totals["expected"] += len(query_case["expected"])
                    totals["value_expected"] += int(value_metrics["expected_keys"] or 0)
                    totals["value_top1"] += int(value_metrics["value_correct_top1"] or 0)
                    totals["value_topk"] += int(value_metrics["value_correct_topk"] or 0)
            else:
                hits = adapter_a.search(
                    query_case["query"], memory_types=query_case["types"],
                    top_k_per_type=TOP_K_PER_TYPE, top_k_total=TOP_K_TOTAL, threshold=args.threshold,
                )
                actual_keys = [hit.memory_key for hit in hits]
                actual_scores = [{"key": hit.memory_key, "score": hit.score} for hit in hits]
                actual_ids = {hit.memory_id for hit in hits}
                rejected = []
                source_needs = []
                accepted_keys = actual_keys
            expected = set(query_case["expected"])
            matched = [key for key in actual_keys if key in expected]
            relevant_hits += len(matched)
            total_hits += len(actual_keys)
            expected_total += len(expected)
            leaked = unauthorized_ids & actual_ids
            results.append({
                "query": query_case["query"], "expected": sorted(expected),
                "actual": actual_keys,
                "accepted": accepted_keys,
                "actual_scores": actual_scores,
                "rejected": rejected,
                "source_needs": source_needs,
                "matched": matched,
                "precision": (len(matched) / len(actual_keys)) if actual_keys else 1.0,
                "recall": (len(matched) / len(expected)) if expected else 1.0,
                "unauthorized_leaks": sorted(leaked),
                "variants": variant_details if args.production else {},
            })
        variant_summary: dict[str, dict[str, object]] = {}
        for variant, totals in variant_totals.items():
            queries = int(totals["queries"]) or 1
            expected_keys = int(totals["expected_keys"]) or 1
            accepted_relevant = 0
            accepted_returned = 0
            accepted_expected = 0
            leaks = set()
            for item in results:
                detail = (item.get("variants") or {}).get(variant) or {}
                metrics = detail.get("accepted_metrics")
                if metrics:
                    accepted_relevant += len(metrics["matched"])
                    accepted_returned += len(detail.get("accepted") or [])
                    accepted_expected += len(item["expected"])
                leaks.update(detail.get("unauthorized_leaks") or [])
            variant_summary[variant] = {
                "queries": queries,
                "query_hit": {
                    f"@{k}": totals[f"query_hit@{k}"] / queries for k in (1, 3, 5)
                },
                "query_mrr": totals["query_mrr"] / queries,
                "expected_keys": expected_keys,
                "key_hit": {
                    f"@{k}": totals[f"key_hit@{k}"] / expected_keys for k in (1, 3, 5)
                },
                "key_mrr": totals["key_mrr"] / expected_keys,
                "precision": totals["relevant"] / totals["returned"] if totals["returned"] else 1.0,
                "recall": totals["relevant"] / totals["expected"] if totals["expected"] else 1.0,
                "value_accuracy_top1": (
                    totals["value_top1"] / totals["value_expected"]
                    if totals["value_expected"] else None
                ),
                "value_accuracy_topk": (
                    totals["value_topk"] / totals["value_expected"]
                    if totals["value_expected"] else None
                ),
                "value_expected_keys": totals["value_expected"],
                "accepted_precision": (
                    accepted_relevant / accepted_returned if accepted_returned else None
                ),
                "accepted_recall": (
                    accepted_relevant / accepted_expected if accepted_expected else None
                ),
                "unauthorized_leaks": sorted(leaks),
            }
        precision = relevant_hits / total_hits if total_hits else 1.0
        recall = relevant_hits / expected_total if expected_total else 1.0
        accepted_hits = sum(
            sum(1 for key in item["accepted"] if key in item["expected"])
            for item in results
        )
        accepted_total = sum(len(item["accepted"]) for item in results)
        accepted_expected_total = sum(len(item["expected"]) for item in results)
        leaked_all = sorted({
            leak for item in results for leak in item["unauthorized_leaks"]
        })
        report = {
            "schema_version": "personalization-real-eval-v3",
            "dataset_version": (
                "personalization-recall-v4-real-2026-11-02-e1-4-31q"
                if args.query_set == "e1_4"
                else "personalization-recall-v4-real-2026-11-02-e1-75q"
            ),
            "evidence": "real Mem0 OSS + Qdrant + PostgreSQL Registry",
            "rules_version": PREFERENCE_INTENT_RULES_VERSION if args.production else None,
            "code_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                text=True, check=False,
            ).stdout.strip() or None,
            "worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                text=True, check=False,
            ).stdout.strip()),
            "input_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in EVIDENCE_INPUTS
            },
            "collection": collection,
            "queries": len(query_cases),
            "seeded_preferences": len(seeded),
            "precision": precision,
            "recall": recall,
            "accepted_precision": accepted_hits / accepted_total if accepted_total else 1.0,
            "accepted_recall": accepted_hits / accepted_expected_total if accepted_expected_total else 1.0,
            "value_accuracy_top1": (
                variant_summary.get("A_current", {}).get("value_accuracy_top1")
                if args.production else None
            ),
            "value_accuracy_topk": (
                variant_summary.get("A_current", {}).get("value_accuracy_topk")
                if args.production else None
            ),
            "threshold": args.threshold,
            "top_k_per_type": TOP_K_PER_TYPE,
            "top_k_total": TOP_K_TOTAL,
            "unauthorized_rejected": not leaked_all,
            "unauthorized_leaks": leaked_all,
            "unauthorized_candidate_count": len(unauthorized_ids),
            "t3": {
                "denominator": len(unauthorized_ids),
                "leaks": leaked_all,
                "display": f"{len(leaked_all)}/{len(unauthorized_ids)}",
                "by_variant": {
                    variant: (summary.get("unauthorized_leaks") or [])
                    for variant, summary in variant_summary.items()
                },
                "seen_by_variant": {
                    variant: sorted({
                        item
                        for result in results
                        for item in (
                            ((result.get("variants") or {}).get(variant) or {})
                            .get("unauthorized_seen") or []
                        )
                    })
                    for variant in variant_summary
                },
                "note": (
                    "Unauthorized/expired/inactive records are filtered before S0 by "
                    "entity scope, seller scope, status and Registry checks. A 0/4 "
                    "result is a no-leak result, not proof that four candidates were "
                    "ranked and then rejected."
                ),
            },
            "retrieval_variants": variant_summary,
            "s0_recorded": bool(variant_summary.get("S0_raw")),
            "details": results,
            "T4_precision_pass": precision >= 0.90,
            "T3_rejection_pass": not leaked_all,
        }
        output = json.dumps(report, ensure_ascii=False, indent=2)
        print("PERSONALIZATION_REAL_RESULT " + output)
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(output + "\n", encoding="utf-8")
        return 0 if report["T4_precision_pass"] and report["T3_rejection_pass"] else 1
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"), {"tenant": tenant})
        engine.dispose()
        try:
            memory.vector_store.client.delete_collection(collection)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
