# -*- coding: utf-8 -*-
"""E1.3 multi-profile value/isolation evaluation (real services).

This script seeds multiple users with different active values for the same
preference keys, plus superseded same-key records. It then queries each user's
profile and checks key ranking, value correctness, cross-user leakage and
inactive-record leakage.

It requires a disposable migrated PostgreSQL database and a disposable Qdrant.
It does not touch the project database or the application API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context.turn_resolver import (  # noqa: E402
    detect_preference_source_needs,
    resolve_preference_context,
)
from app.memory.mem0_adapter import Mem0PreferenceAdapter  # noqa: E402
from app.memory.preference_provider import Mem0PreferenceProvider  # noqa: E402
from app.memory.preference_registry import PostgresPreferenceRegistry  # noqa: E402
from app.memory.preference_worker import _render_preference  # noqa: E402
from app.tenancy import TenantContext  # noqa: E402

HMAC_SECRET = "0123456789abcdef0123456789abcdef"
SCOPE_VERSION = "v1"
THRESHOLD = 0.55
TOP_K_PER_TYPE = 5
TOP_K_TOTAL = 10

PROFILES = [
    {
        "style": "concise", "language": "en-US", "format": "table",
        "operation": "trend", "dimension": "product_category",
        "start_date": "2026-07-01", "end_date": "2026-07-31",
    },
    {
        "style": "detailed", "language": "zh-CN", "format": "bullets",
        "operation": "compare", "dimension": "payment_type",
        "start_date": "2026-06-01", "end_date": "2026-06-30",
    },
    {
        "style": "balanced", "language": "zh-TW", "format": "narrative",
        "operation": "trend", "dimension": "payment_type",
        "start_date": "2026-08-01", "end_date": "2026-08-31",
    },
    {
        "style": "concise", "language": "zh-CN", "format": "bullets",
        "operation": "compare", "dimension": "product_category",
        "start_date": "2026-05-01", "end_date": "2026-05-31",
    },
    {
        "style": "detailed", "language": "en-US", "format": "table",
        "operation": "trend", "dimension": "product_category",
        "start_date": "2026-09-01", "end_date": "2026-09-30",
    },
    {
        "style": "balanced", "language": "zh-TW", "format": "narrative",
        "operation": "compare", "dimension": "payment_type",
        "start_date": "2026-04-01", "end_date": "2026-04-30",
    },
    {
        "style": "concise", "language": "zh-CN", "format": "table",
        "operation": "trend", "dimension": "payment_type",
        "start_date": "2026-03-01", "end_date": "2026-03-31",
    },
    {
        "style": "detailed", "language": "en-US", "format": "bullets",
        "operation": "compare", "dimension": "product_category",
        "start_date": "2026-02-01", "end_date": "2026-02-28",
    },
]

QUERY_TEMPLATES = [
    ("按我平时的方式回答", ["response_style", "response_language"]),
    ("报告按我平时的格式", ["report_style"]),
    ("按我平时的分析方式分析 GMV", ["analysis_method"]),
    ("用我的默认统计周期看 GMV", ["default_period"]),
    (
        "按我平时的方式回答并生成报告",
        ["response_style", "response_language", "report_style"],
    ),
]


def _settings(collection: str, qdrant_url: str, history: Path) -> SimpleNamespace:
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
        mem0_relevance_threshold=THRESHOLD,
        mem0_llm_model="", mem0_llm_api_key="", mem0_llm_base_url="",
        deepseek_api_key="", deepseek_base_url="",
        qdrant_url=qdrant_url, qdrant_collection="bizinsight_knowledge",
        rag_qdrant_active_alias="", metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )


def _profile_values(profile: dict[str, str]) -> dict[str, dict[str, object]]:
    return {
        "response_style": {"style": profile["style"]},
        "response_language": {"language": profile["language"]},
        "report_style": {"format": profile["format"]},
        "analysis_method": {
            "operation": profile["operation"],
            "dimension": profile["dimension"],
        },
        "default_period": {
            "start_date": profile["start_date"],
            "end_date": profile["end_date"],
        },
    }


def _keys_for_query(expected: list[str]) -> set[str]:
    return set(expected)


def _rank(actual: list[str], expected: list[str]) -> dict[str, object]:
    expected_set = set(expected)
    first_rank = next((i + 1 for i, key in enumerate(actual) if key in expected_set), None)
    key_ranks = {
        key: next((i + 1 for i, actual_key in enumerate(actual) if actual_key == key), None)
        for key in expected
    }
    matched = [key for key in actual if key in expected_set]
    return {
        "query_hit": {
            f"@{k}": int(bool(expected_set & set(actual[:k]))) for k in (1, 3, 5)
        },
        "query_mrr": 1 / first_rank if first_rank else 0.0,
        "key_hit": {
            f"@{k}": sum(1 for rank in key_ranks.values() if rank and rank <= k)
            for k in (1, 3, 5)
        },
        "key_mrr": (
            sum(1 / rank for rank in key_ranks.values() if rank) / len(expected)
            if expected else 0.0
        ),
        "key_ranks": key_ranks,
        "matched": matched,
        "precision": len(matched) / len(actual) if actual else 1.0,
        "recall": len(matched) / len(expected) if expected else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default=os.getenv("E13_QDRANT_URL", ""))
    parser.add_argument("--database-url", default=os.getenv("E13_DATABASE_URL", ""))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    if not args.qdrant_url or not args.database_url:
        print(
            "E13_SKIPPED: set E13_QDRANT_URL and E13_DATABASE_URL",
            file=sys.stderr,
        )
        return 2

    from mem0 import Memory

    collection = f"e13_profiles_{uuid.uuid4().hex[:8]}"
    history = Path(tempfile.mkdtemp(prefix="e13_profiles_")) / "history.db"
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
    tenant = f"e13-tenant-{suffix}"
    settings = _settings(collection, args.qdrant_url, history)
    profiles = []
    active_records: dict[str, dict[str, str]] = {}
    inactive_records: set[str] = set()
    all_ids: set[str] = set()
    try:
        for index, profile in enumerate(PROFILES):
            user_id = f"e13-user-{suffix}-{index}"
            context = TenantContext(
                tenant_id=tenant, user_id=user_id, role="admin", scope_type="platform",
            )
            adapter = Mem0PreferenceAdapter(
                context, settings=settings, memory=memory,
                registry=PostgresPreferenceRegistry(context, engine_override=engine),
            )
            values = _profile_values(profile)
            active_records[user_id] = {}
            inactive_ids: dict[str, str] = {}
            for key, value in values.items():
                text_value = _render_preference(key, "presentation" if key in {
                    "response_style", "response_language", "report_style",
                } else "analysis_method" if key == "analysis_method" else "scope_default", value)
                memory_id = adapter.add(
                    text_value, memory_key=key,
                    memory_type=(
                        "presentation" if key in {"response_style", "response_language", "report_style"}
                        else "analysis_method" if key == "analysis_method" else "scope_default"
                    ),
                    operation_id=f"e13-{suffix}-{index}-{key}",
                    value=value, infer=False,
                )[0]
                active_records[user_id][key] = memory_id
                all_ids.add(memory_id)
                # A superseded old value for the same key; Registry validation
                # must prevent it from being returned.
                old_value = dict(value)
                if key == "response_style":
                    old_value["style"] = "detailed" if value["style"] != "detailed" else "concise"
                elif key == "response_language":
                    old_value["language"] = "zh-CN" if value["language"] != "zh-CN" else "en-US"
                elif key == "report_style":
                    old_value["format"] = "bullets" if value["format"] != "bullets" else "table"
                elif key == "analysis_method":
                    old_value["operation"] = "compare" if value["operation"] != "compare" else "trend"
                else:
                    old_value["start_date"] = "2025-01-01"
                    old_value["end_date"] = "2025-01-31"
                old_text = _render_preference(
                    key,
                    "presentation" if key in {"response_style", "response_language", "report_style"}
                    else "analysis_method" if key == "analysis_method" else "scope_default",
                    old_value,
                )
                old_id = adapter.add(
                    old_text, memory_key=key,
                    memory_type=(
                        "presentation" if key in {"response_style", "response_language", "report_style"}
                        else "analysis_method" if key == "analysis_method" else "scope_default"
                    ),
                    operation_id=f"e13-{suffix}-{index}-{key}-old",
                    value=old_value, infer=False,
                )[0]
                inactive_records.add(old_id)
                inactive_ids[key] = old_id
                all_ids.add(old_id)
            profiles.append({
                "user_id": user_id,
                "context": context,
                "adapter": adapter,
                "values": values,
                "active_ids": active_records[user_id],
                "inactive_ids": inactive_ids,
            })

        # Registry: active and superseded rows for each user/key/version.
        with engine.begin() as connection:
            registry_index = 0
            for profile in profiles:
                for key, memory_id in profile["active_ids"].items():
                    memory_type = (
                        "presentation" if key in {"response_style", "response_language", "report_style"}
                        else "analysis_method" if key == "analysis_method" else "scope_default"
                    )
                    connection.execute(text("""
                        INSERT INTO governance.memory_registry
                            (registry_id, tenant_id, user_id, seller_id, memory_type,
                             memory_key, mem0_memory_id, status, version, operation_id,
                             user_confirmed, expires_at)
                        VALUES (:registry_id, :tenant, :user, NULL, :memory_type,
                                :memory_key, :memory_id, 'active', 1, :operation_id,
                                true, NULL)
                    """), {
                        "registry_id": f"e13-reg-{suffix}-{registry_index}",
                        "tenant": tenant, "user": profile["user_id"],
                        "memory_type": memory_type, "memory_key": key,
                        "memory_id": memory_id,
                        "operation_id": f"e13-{suffix}-{profiles.index(profile)}-{key}",
                    })
                    registry_index += 1
            # Superseded versions do not need stable operation IDs for query use,
            # but need a Registry row so the adapter can reject them.
            for profile in profiles:
                for key, memory_id in profile["active_ids"].items():
                    memory_type = (
                        "presentation" if key in {"response_style", "response_language", "report_style"}
                        else "analysis_method" if key == "analysis_method" else "scope_default"
                    )
                    old_id = profile["inactive_ids"][key]
                    connection.execute(text("""
                        INSERT INTO governance.memory_registry
                            (registry_id, tenant_id, user_id, seller_id, memory_type,
                             memory_key, mem0_memory_id, status, version, operation_id,
                             user_confirmed, expires_at)
                        VALUES (:registry_id, :tenant, :user, NULL, :memory_type,
                                :memory_key, :memory_id, 'superseded', 2, :operation_id,
                                true, NULL)
                    """), {
                        "registry_id": f"e13-reg-old-{suffix}-{registry_index}",
                        "tenant": tenant, "user": profile["user_id"],
                        "memory_type": memory_type, "memory_key": key,
                        "memory_id": old_id,
                        "operation_id": f"e13-{suffix}-{profiles.index(profile)}-{key}-old",
                    })
                    registry_index += 1

        results = []
        totals = {
            "queries": 0, "query_hit@1": 0, "query_hit@3": 0, "query_hit@5": 0,
            "query_mrr": 0.0, "expected_keys": 0, "key_hit@1": 0, "key_hit@3": 0,
            "key_hit@5": 0, "key_mrr": 0.0, "value_top1": 0, "value_any": 0,
            "cross_user_leaks": 0, "inactive_leaks": 0, "returned": 0,
            "matched": 0, "expected": 0,
        }
        for profile in profiles:
            provider = Mem0PreferenceProvider(
                profile["context"], settings=settings, adapter=profile["adapter"],
                registry=PostgresPreferenceRegistry(
                    profile["context"], engine_override=engine,
                ),
            )
            for template, expected_keys in QUERY_TEMPLATES:
                state = {
                    "user_message": template,
                    "memory_enabled": True,
                    "chat_session_id": f"e13-{profile['user_id']}",
                    "current_query_scope_fingerprint": f"e13-{profile['user_id']}",
                    "controlled_analysis_scope": {
                        "default_start_date": "2026-01-01",
                        "default_end_date": "2026-12-31",
                        "allowed_metrics": ["gmv"],
                        "allowed_dimensions": ["product_category", "payment_type"],
                    },
                }
                needs = detect_preference_source_needs(state)
                if not needs:
                    raise RuntimeError(f"no source need for {template!r}")
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

                recording = _RecordingProvider(provider)
                resolved = resolve_preference_context(state, provider=recording)
                raw = recording.raw
                accepted = list(resolved.get("accepted") or [])
                actual_keys = [str(item.get("memory_key") or "") for item in raw]
                accepted_keys = [str(item.get("memory_key") or "") for item in accepted]
                rank = _rank(actual_keys, expected_keys)
                value_expected = 0
                value_top1 = 0
                value_any = 0
                for key in expected_keys:
                    gold = profile["values"].get(key)
                    candidates = [
                        item.get("value") for item in raw
                        if str(item.get("memory_key") or "") == key
                    ]
                    if gold is None:
                        continue
                    value_expected += 1
                    value_top1 += int(bool(candidates) and candidates[0] == gold)
                    value_any += int(gold in candidates)
                actual_ids = {str(item.get("source_ref") or "") for item in raw}
                cross_user = actual_ids & {
                    memory_id for user_id, ids in active_records.items()
                    if user_id != profile["user_id"] for memory_id in ids.values()
                }
                inactive = actual_ids & inactive_records
                totals["queries"] += 1
                for k in (1, 3, 5):
                    totals[f"query_hit@{k}"] += rank["query_hit"][f"@{k}"]
                    totals[f"key_hit@{k}"] += rank["key_hit"][f"@{k}"]
                totals["query_mrr"] += rank["query_mrr"]
                totals["expected_keys"] += len(expected_keys)
                totals["key_mrr"] += rank["key_mrr"] * len(expected_keys)
                totals["value_top1"] += value_top1
                totals["value_any"] += value_any
                totals["cross_user_leaks"] += len(cross_user)
                totals["inactive_leaks"] += len(inactive)
                totals["returned"] += len(actual_keys)
                totals["matched"] += len(rank["matched"])
                totals["expected"] += len(expected_keys)
                results.append({
                    "user_id": profile["user_id"],
                    "query": template,
                    "expected_keys": expected_keys,
                    "expected_values": {key: profile["values"][key] for key in expected_keys},
                    "actual": actual_keys,
                    "accepted": accepted_keys,
                    "key_ranks": rank["key_ranks"],
                    "value_top1": value_top1,
                    "value_any": value_any,
                    "value_expected": value_expected,
                    "cross_user_leaks": sorted(cross_user),
                    "inactive_leaks": sorted(inactive),
                })

        queries = totals["queries"] or 1
        expected_keys = totals["expected_keys"] or 1
        report = {
            "schema_version": "e1-3-profiles-v1",
            "dataset_version": f"e1-3-profiles-{suffix}",
            "rules_version": "preference-intent-rules-v4",
            "profiles": len(profiles),
            "queries": totals["queries"],
            "expected_keys": totals["expected_keys"],
            "query_hit": {
                f"@{k}": totals[f"query_hit@{k}"] / queries for k in (1, 3, 5)
            },
            "key_hit": {
                f"@{k}": totals[f"key_hit@{k}"] / expected_keys for k in (1, 3, 5)
            },
            "query_mrr": totals["query_mrr"] / queries,
            "key_mrr": totals["key_mrr"] / expected_keys,
            "value_accuracy_first": totals["value_top1"] / totals["expected"] if totals["expected"] else None,
            "value_accuracy_any": totals["value_any"] / totals["expected"] if totals["expected"] else None,
            "precision": totals["matched"] / totals["returned"] if totals["returned"] else 1.0,
            "recall": totals["matched"] / totals["expected"] if totals["expected"] else 1.0,
            "cross_user_leaks": totals["cross_user_leaks"],
            "cross_user_candidate_count": (
                sum(len(ids) for ids in active_records.values())
                * max(0, len(PROFILES) - 1)
            ),
            "inactive_candidate_count": len(inactive_records),
            "inactive_leaks": totals["inactive_leaks"],
            "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or None,
            "worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()),
            "input_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "details": results,
        }
        output = json.dumps(report, ensure_ascii=False, indent=2)
        print("E13_RESULT " + output)
        if args.json_out:
            target = Path(args.json_out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(output + "\n", encoding="utf-8")
        return 0
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM governance.memory_registry WHERE tenant_id=:tenant"),
                {"tenant": tenant},
            )
        engine.dispose()
        try:
            memory.vector_store.client.delete_collection(collection)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
