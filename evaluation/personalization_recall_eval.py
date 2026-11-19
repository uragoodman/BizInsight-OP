# -*- coding: utf-8 -*-
"""Offline personalization-recall evaluation for the context/memory contract.

Three deterministic suites run without external services:

* ``trigger``   – whether a turn should ask the preference provider at all;
* ``candidate`` – acceptance/rejection of provider candidates after Resolver
                  validation (explicit input, scope/authority, threshold);
* ``adapter``   – rejection of scope/identity mismatches, expired metadata and
                  Registry-inactive records inside the Mem0 adapter validation.

The adapter suite uses an in-process Memory double; it is **not** a real
Mem0/Qdrant retrieval measurement.  Real retrieval precision (WP0 T4) requires
a seeded Qdrant collection plus a migrated PostgreSQL Registry; the report
labels that evidence as not measured instead of substituting doubles.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context.turn_resolver import (  # noqa: E402
    detect_preference_source_needs,
    resolve_preference_context,
)
from app.memory.mem0_adapter import Mem0PreferenceAdapter  # noqa: E402
from app.memory.preference_registry import RegistryDecision  # noqa: E402
from app.tenancy import TenantContext  # noqa: E402


DEFAULT_CASES = ROOT / "evaluation" / "datasets" / "personalization_recall_v1.jsonl"
T4_PRECISION_THRESHOLD = 0.90
T3_REJECTION_THRESHOLD = 1.0


class _FixtureProvider:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.calls = 0

    def search(self, *, need: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls += 1
        return [dict(item) for item in self.candidates]


class _FixtureMemory:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.vector_store = SimpleNamespace(client=object())

    def search(self, query, *, top_k=20, filters=None, threshold=0.1, rerank=False):
        # Deliberately ignore filters: the adapter must still reject rows that
        # do not belong to the trusted scope.
        return {"results": list(self.rows)}

    def get_all(self, *, filters=None, top_k=20, show_expired=False):
        return {"results": list(self.rows)}

    def get(self, memory_id):
        for row in self.rows:
            if row.get("id") == memory_id:
                return row
        return None

    def update(self, memory_id, **kwargs):
        return {"id": memory_id}

    def delete(self, memory_id):
        return True


class _FixtureRegistry:
    def __init__(
        self,
        inactive: dict[str, str] | None = None,
        expired: dict[str, str] | None = None,
    ) -> None:
        self.inactive = inactive or {}
        self.expired = expired or {}

    def validate(self, **kwargs) -> RegistryDecision:
        operation_id = str(kwargs.get("operation_id") or "")
        if operation_id in self.inactive:
            return RegistryDecision(False, "registry_not_active")
        if operation_id in self.expired:
            return RegistryDecision(False, "registry_expired")
        return RegistryDecision(True)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _base_state(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_message": case.get("query") or "",
        "memory_enabled": True,
        "current_query_scope_fingerprint": "scope-eval",
        "context_compilation": case.get("compilation") or {
            "known_slots": {},
            "needs_clarification": False,
        },
        "controlled_analysis_scope": {
            "seller_id": "seller-a",
            "default_start_date": "2026-06-01",
            "default_end_date": "2026-07-31",
            "metric_name": "gmv",
        },
    }


def _suite_trigger(cases: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    exact = 0
    details = []
    for case in cases:
        needs = detect_preference_source_needs(_base_state(case))
        actual_ids: list[str] = []
        for need in needs:
            actual_ids.extend(str(item) for item in need.get("trigger_ids") or [])
        actual_ids = list(dict.fromkeys(actual_ids))
        expected = case.get("expected_trigger_ids") or []
        actual_trigger = bool(actual_ids)
        if actual_trigger and expected:
            tp += 1
        elif actual_trigger and not expected:
            fp += 1
        elif not actual_trigger and expected:
            fn += 1
        match = set(actual_ids) == set(expected)
        exact += int(match)
        details.append({
            "case_id": case["case_id"], "category": case.get("category"),
            "expected_trigger_ids": expected, "actual_trigger_ids": actual_ids,
            "exact_match": match,
        })
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "suite": "trigger",
        "samples": len(cases), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "exact_trigger_id_matches": exact,
        "details": details,
    }


def _suite_candidate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    rejection_checks = 0
    rejection_passed = 0
    explicit_override_checks = 0
    explicit_override_passed = 0
    details = []
    for case in cases:
        state = _base_state(case)
        provider = _FixtureProvider(case.get("candidates") or [])
        context = resolve_preference_context(
            state, provider=provider,
            compilation=case.get("compilation"),
        )
        accepted_keys = [str(item.get("memory_key")) for item in context.get("accepted") or []]
        rejected = {
            str(item.get("memory_key")): str(item.get("reason"))
            for item in context.get("rejected") or []
        }
        expected_accepted = list(case.get("expected_accepted_keys") or [])
        expected_rejected = list(case.get("expected_rejected_keys") or [])
        for key in accepted_keys:
            if key in expected_accepted:
                tp += 1
            else:
                fp += 1
        for key in expected_accepted:
            if key not in accepted_keys:
                fn += 1
        reasons = case.get("expected_rejected_reasons") or {}
        for key in expected_rejected:
            rejection_checks += 1
            expected_reason = reasons.get(key)
            ok = key in rejected and (expected_reason is None or rejected[key] == expected_reason)
            rejection_passed += int(ok)
            if case.get("category") == "explicit_override":
                explicit_override_checks += 1
                explicit_override_passed += int(ok)
        details.append({
            "case_id": case["case_id"], "category": case.get("category"),
            "accepted_keys": accepted_keys, "rejected": rejected,
            "expected_accepted_keys": expected_accepted,
            "expected_rejected_keys": expected_rejected,
            "passed": (
                sorted(accepted_keys) == sorted(expected_accepted)
                and all(key in rejected for key in expected_rejected)
            ),
        })
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "suite": "candidate",
        "samples": len(cases), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "rejection_checks": rejection_checks,
        "rejection_passed": rejection_passed,
        "explicit_override_checks": explicit_override_checks,
        "explicit_override_passed": explicit_override_passed,
        "details": details,
    }


def _adapter_settings() -> SimpleNamespace:
    return SimpleNamespace(
        mem0_qdrant_collection="personalization_eval_fixture",
        mem0_qdrant_url="http://127.0.0.1:6333",
        mem0_qdrant_api_key="",
        mem0_embedding_model="BAAI/bge-small-zh-v1.5",
        mem0_embedding_dims=512,
        mem0_hmac_secret="0123456789abcdef0123456789abcdef",
        mem0_scope_version="v1",
        mem0_history_db_path="",
        mem0_search_timeout_seconds=0.05,
        mem0_write_timeout_seconds=0.05,
        mem0_breaker_failures=5,
        mem0_breaker_window_seconds=30.0,
        mem0_top_k_per_type=5,
        mem0_top_k_total=10,
        mem0_relevance_threshold=0.35,
        mem0_llm_model="",
        mem0_llm_api_key="",
        mem0_llm_base_url="",
        deepseek_api_key="",
        deepseek_base_url="",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_collection="bizinsight_knowledge",
        rag_qdrant_active_alias="",
        metric_semantic_qdrant_collection="metric_semantic_catalog_v1",
    )


def _suite_adapter(cases: list[dict[str, Any]]) -> dict[str, Any]:
    context = TenantContext(
        tenant_id="eval-tenant", user_id="eval-user", role="admin", scope_type="platform",
    )
    invalid_total = 0
    invalid_rejected = 0
    false_accepts: list[str] = []
    expected_hit_checks = 0
    expected_hit_matched = 0
    details = []
    for case in cases:
        adapter = Mem0PreferenceAdapter(
            context,
            settings=_adapter_settings(),
            memory=None,
            registry=_FixtureRegistry(
                inactive=case.get("registry") or {},
                expired=case.get("registry_expires") or {},
            ),
        )
        scope = adapter.scope
        rows = []
        invalid_ids = set()
        for row in case.get("rows") or []:
            entity = scope.entity_id if row.get("entity_scope") == "self" else "other-entity"
            seller = scope.seller_scope if row.get("seller_scope") == "self" else str(row.get("seller_scope") or "")
            metadata = {
                "operation_id": row["operation_id"],
                "memory_type": row["memory_type"],
                "memory_key": row["memory_key"],
                "entity_scope": entity,
                "scope_version": scope.scope_version,
                "seller_scope": seller,
                "memory_status": "active",
                "value_json": json.dumps(row.get("value"), ensure_ascii=False),
            }
            if row.get("expires_at"):
                metadata["expires_at"] = row["expires_at"]
            rows.append({
                "id": row["memory_id"], "memory": "fixture",
                "user_id": entity, "metadata": metadata,
            })
            if (
                entity != scope.entity_id
                or seller != scope.seller_scope
                or bool(row.get("expires_at"))
                or row["operation_id"] in (case.get("registry") or {})
                or row["operation_id"] in (case.get("registry_expires") or {})
            ):
                invalid_ids.add(row["memory_id"])
        fake_memory = _FixtureMemory(rows)
        # Rebuild with the fake memory now that rows exist.
        adapter = Mem0PreferenceAdapter(
            context,
            settings=_adapter_settings(),
            memory=fake_memory,
            registry=_FixtureRegistry(
                inactive=case.get("registry") or {},
                expired=case.get("registry_expires") or {},
            ),
        )
        hits = adapter.search(
            case.get("query") or "", memory_types=["presentation"],
            top_k_per_type=5, top_k_total=10, threshold=0.0,
        )
        hit_ids = {hit.memory_id for hit in hits}
        expected_ids = set(case.get("expected_hit_ids") or [])
        expected_hit_checks += len(expected_ids)
        expected_hit_matched += len(expected_ids & hit_ids)
        leaked = invalid_ids & hit_ids
        invalid_total += len(invalid_ids)
        invalid_rejected += len(invalid_ids) - len(leaked)
        false_accepts.extend(sorted(leaked))
        details.append({
            "case_id": case["case_id"], "category": case.get("category"),
            "hit_ids": sorted(hit_ids), "expected_hit_ids": sorted(expected_ids),
            "invalid_ids": sorted(invalid_ids), "leaked_invalid_ids": sorted(leaked),
        })
    rejection_rate = invalid_rejected / invalid_total if invalid_total else 1.0
    return {
        "suite": "adapter",
        "samples": len(cases),
        "invalid_records": invalid_total,
        "invalid_rejected": invalid_rejected,
        "rejection_rate": rejection_rate,
        "false_accepts": false_accepts,
        "expected_hit_checks": expected_hit_checks,
        "expected_hit_matched": expected_hit_matched,
        "details": details,
        "evidence_kind": "in-process Memory double; not real Mem0/Qdrant retrieval",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--real-result", default="",
        help="path to personalization_recall_real_eval JSON; supplies the measured T4/T3 evidence",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="acceptance mode: real Mem0 T4 must be measured and proxy gates must pass",
    )
    parser.add_argument(
        "--proxy", action="store_true",
        help="exit 0 on proxy gates even when real Mem0 retrieval is unmeasured",
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    trigger_cases = [c for c in cases if c.get("suite") == "trigger"]
    candidate_cases = [c for c in cases if c.get("suite") == "candidate"]
    adapter_cases = [c for c in cases if c.get("suite") == "adapter"]

    trigger = _suite_trigger(trigger_cases)
    candidate = _suite_candidate(candidate_cases)
    adapter = _suite_adapter(adapter_cases)

    # WP0 T4 freezes precision on a WP3 labeled set; the two deterministic
    # suites below are decision/adapter proxies and cannot replace real Mem0
    # retrieval evidence.  T3's unauthorized-rejection requirement is checked
    # by the adapter suite.
    decision_precision = candidate["precision"]
    rejection_rate = adapter["rejection_rate"]
    gates = {
        "T4_precision_proxy": {
            "value": decision_precision,
            "threshold": T4_PRECISION_THRESHOLD,
            "pass": decision_precision >= T4_PRECISION_THRESHOLD,
            "note": "candidate-acceptance precision proxy; real Mem0 retrieval not measured",
        },
        "T3_unauthorized_rejection": {
            "value": rejection_rate,
            "threshold": T3_REJECTION_THRESHOLD,
            "pass": rejection_rate >= T3_REJECTION_THRESHOLD,
            "note": "adapter validation with in-process Memory double",
        },
        "real_mem0_retrieval": {
            "measured": False,
            "required_inputs": [
                "seeded Qdrant collection with labeled preference corpus",
                "migrated PostgreSQL Registry rows for the same operation_ids",
                "real embedding model",
            ],
            "note": "not substituted with doubles; T4 real retrieval evidence remains open",
        },
    }
    if args.real_result:
        real_path = Path(args.real_result)
        if not real_path.exists():
            print(f"REAL_RESULT_MISSING: {real_path}", file=sys.stderr)
            return 2
        real = json.loads(real_path.read_text(encoding="utf-8"))
        gates["real_mem0_retrieval"] = {
            "measured": True,
            "precision": real.get("precision"),
            "recall": real.get("recall"),
            "threshold": real.get("threshold"),
            "queries": real.get("queries"),
            "pass": bool(real.get("T4_precision_pass") and real.get("T3_rejection_pass")),
            "note": f"from {real_path}",
        }
    report = {
        "schema_version": "personalization-recall-eval-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(Path(args.cases).resolve()),
        "sample_counts": {
            "total": len(cases),
            "trigger": len(trigger_cases),
            "candidate": len(candidate_cases),
            "adapter": len(adapter_cases),
            "by_category": _category_counts(cases),
        },
        "trigger": trigger,
        "candidate": candidate,
        "adapter": adapter,
        "gates": gates,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    proxy_failed = [
        name for name, gate in gates.items()
        if name != "real_mem0_retrieval"
        and isinstance(gate, dict) and gate.get("pass") is False
    ]
    real_measured = bool(gates["real_mem0_retrieval"].get("measured"))
    if args.strict:
        failed = list(proxy_failed)
        if not real_measured:
            failed.append("real_mem0_retrieval_not_measured")
        if failed:
            print(f"GATES_FAILED: {failed}", file=sys.stderr)
            return 1
    elif args.proxy and proxy_failed:
        print(f"GATES_FAILED: {proxy_failed}", file=sys.stderr)
        return 1
    return 0


def _category_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        category = str(case.get("category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
