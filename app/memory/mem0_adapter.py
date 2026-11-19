"""Internal Mem0 OSS adapter for WP3 preference memory.

Business code calls this module; it must not import or call the Mem0 SDK.  The
adapter owns the isolated Qdrant collection, opaque HMAC entity scope, SDK
timeouts/circuit-breaking, metadata validation and the read-only Registry
boundary.  Mem0 is never an authorization source and never stores KPI values,
analysis claims, Task State or authorization facts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from app.core.config import get_settings
from app.observability import record_event
from app.tenancy import TenantContext
from app.memory.preference_registry import (
    NullPreferenceRegistry,
    PreferenceRegistry,
    RegistryUnavailable,
)


ALLOWED_MEMORY_TYPES = frozenset({"presentation", "analysis_method", "scope_default"})
_ACTIVE_STATUS = "active"


class Mem0Error(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Mem0ConfigurationError(Mem0Error):
    pass


class Mem0Unavailable(Mem0Error):
    pass


class Mem0Timeout(Mem0Unavailable):
    pass


class Mem0CircuitOpen(Mem0Unavailable):
    pass


class Mem0ScopeRejected(Mem0Error):
    pass


@dataclass(frozen=True)
class Mem0Scope:
    entity_id: str
    seller_scope: str
    scope_version: str


@dataclass(frozen=True)
class PreferenceHit:
    memory_id: str
    text: str
    memory_type: str
    memory_key: str
    value: Any
    operation_id: str | None
    source_turn_id: str | None
    source_trace_id: str | None
    score: float | None
    metadata: dict[str, Any]

    def as_candidate(self) -> dict[str, Any]:
        return {
            "memory_key": self.memory_key,
            "memory_type": self.memory_type,
            "value": self.value,
            "source_ref": self.memory_id,
            "confidence": self.score if self.score is not None else 0.0,
            "operation_id": self.operation_id,
            "source_turn_id": self.source_turn_id,
            "source_trace_id": self.source_trace_id,
        }


def build_opaque_scope(
    context: TenantContext,
    *,
    hmac_secret: str,
    scope_version: str,
) -> Mem0Scope:
    """Derive the only entity ID sent to Mem0.

    The caller never supplies this ID.  Rotating the secret or scope version
    intentionally partitions old preferences from new ones.
    """
    if not isinstance(context, TenantContext):
        raise Mem0ConfigurationError("trusted_tenant_context_required")
    if not hmac_secret or len(hmac_secret) < 16:
        raise Mem0ConfigurationError("mem0_hmac_secret_missing_or_too_short")
    seller_scope = str(context.seller_id or "")
    material = f"{context.tenant_id}|{context.user_id}|{seller_scope}|{scope_version}"
    digest = hmac.new(
        hmac_secret.encode("utf-8"), material.encode("utf-8"), hashlib.sha256,
    ).digest()
    token = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return Mem0Scope(
        entity_id=f"mem0_{token[:32]}",
        seller_scope=seller_scope,
        scope_version=str(scope_version),
    )


class _CircuitBreaker:
    def __init__(self, failures: int, window_seconds: float) -> None:
        self.failures = max(1, int(failures))
        self.window = max(1.0, float(window_seconds))
        self._events: list[float] = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._events = [item for item in self._events if now - item <= self.window]
            return len(self._events) < self.failures

    def failure(self) -> None:
        with self._lock:
            self._events.append(time.monotonic())

    def success(self) -> None:
        with self._lock:
            self._events.clear()


class Mem0PreferenceAdapter:
    """Synchronous Mem0 OSS adapter with fail-closed validation."""

    _runtime_lock = threading.Lock()
    _runtimes: dict[tuple[Any, ...], tuple[Any, Any]] = {}
    _breakers: dict[str, _CircuitBreaker] = {}

    def __init__(
        self,
        context: TenantContext,
        *,
        trace_id: str | None = None,
        settings: Any | None = None,
        memory: Any | None = None,
        qdrant_client: Any | None = None,
        registry: PreferenceRegistry | None = None,
    ) -> None:
        self.context = context
        self.trace_id = str(trace_id or "")
        self.settings = settings or get_settings()
        self.scope = build_opaque_scope(
            context,
            hmac_secret=str(getattr(self.settings, "mem0_hmac_secret", "")),
            scope_version=str(getattr(self.settings, "mem0_scope_version", "v1")),
        )
        self.collection = str(getattr(self.settings, "mem0_qdrant_collection", ""))
        self._validate_collection()
        self._memory = memory
        self._client = qdrant_client
        self.registry = registry or NullPreferenceRegistry()
        self._timeout_error: str | None = None
        breaker_key = f"{self.collection}:{self.scope.entity_id}"
        self._breaker = self._breakers.setdefault(
            breaker_key,
            _CircuitBreaker(
                int(getattr(self.settings, "mem0_breaker_failures", 5)),
                float(getattr(self.settings, "mem0_breaker_window_seconds", 30)),
            ),
        )

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------
    @property
    def scope_id(self) -> str:
        return self.scope.entity_id

    def health(self) -> dict[str, Any]:
        try:
            memory, client = self._ensure_runtime()
            exists = bool(client.collection_exists(self.collection))
            info = client.get_collection(self.collection) if exists else None
            payload_schema = getattr(info, "payload_schema", {}) or {}
            payload_types = {
                str(name): str(getattr(value, "data_type", value)).lower()
                for name, value in payload_schema.items()
            }
            return {
                "available": True,
                "collection": self.collection,
                "collection_exists": exists,
                "payload_indexes": sorted(payload_types),
                "payload_index_types": payload_types,
                "sdk_version": _mem0_version(),
                "qdrant_client_version": _qdrant_client_version(),
            }
        except Exception as exc:  # noqa: BLE001 - health must not raise
            return {
                "available": False,
                "collection": self.collection,
                "reason": getattr(exc, "reason", type(exc).__name__),
            }

    def search(
        self,
        query: str,
        *,
        memory_types: Iterable[str] | None = None,
        memory_keys: Iterable[str] | None = None,
        top_k_per_type: int | None = None,
        top_k_total: int | None = None,
        threshold: float | None = None,
    ) -> list[PreferenceHit]:
        types = _normalise_types(memory_types)
        keys = sorted({str(item).strip() for item in (memory_keys or []) if str(item).strip()})
        per_type = _bounded_int(
            top_k_per_type if top_k_per_type is not None else getattr(self.settings, "mem0_top_k_per_type", 5),
            1, 5,
        )
        total = _bounded_int(
            top_k_total if top_k_total is not None else getattr(self.settings, "mem0_top_k_total", 10),
            1, 10,
        )
        threshold = _bounded_float(
            threshold if threshold is not None else getattr(self.settings, "mem0_relevance_threshold", 0.35),
            0.2, 1.0,
        )
        if not self._breaker.allow():
            raise Mem0CircuitOpen("circuit_open")
        memory, _ = self._ensure_runtime()
        hits: list[PreferenceHit] = []
        remaining = total
        try:
            for memory_type in types:
                if remaining <= 0:
                    break
                type_remaining = per_type
                for memory_key in keys or [None]:
                    if remaining <= 0 or type_remaining <= 0:
                        break
                    filters = {
                        "user_id": self.scope.entity_id,
                        "memory_type": memory_type,
                        "memory_status": _ACTIVE_STATUS,
                        "scope_version": self.scope.scope_version,
                    }
                    if memory_key is not None:
                        filters["memory_key"] = memory_key
                    response = self._run(
                        lambda f=filters, limit=min(type_remaining, remaining): memory.search(
                            query, top_k=limit, filters=f, threshold=threshold, rerank=False,
                        ),
                        timeout=float(getattr(self.settings, "mem0_search_timeout_seconds", 1.5)),
                        operation="search",
                    )
                    rows = _results(response)
                    for row in rows:
                        hit = self._validated_hit(
                            row, expected_types={memory_type},
                            expected_keys={memory_key} if memory_key is not None else None,
                        )
                        if hit is None:
                            continue
                        if hit.memory_id in {existing.memory_id for existing in hits}:
                            continue
                        hits.append(hit)
                        remaining -= 1
                        type_remaining -= 1
                        if remaining <= 0 or type_remaining <= 0:
                            break
        except Exception:
            self._breaker.failure()
            raise
        self._breaker.success()
        self._record("search", hit_count=len(hits), memory_types=sorted(types), threshold=threshold)
        return hits

    def add(
        self,
        text: str,
        *,
        memory_key: str,
        memory_type: str,
        operation_id: str,
        value: Any = None,
        source_turn_id: str | None = None,
        source_trace_id: str | None = None,
        infer: bool = False,
    ) -> list[str]:
        self._validate_type_and_key(memory_type, memory_key)
        self._validate_operation(operation_id)
        if not str(text or "").strip():
            raise Mem0ConfigurationError("empty_preference_text")
        memory, _ = self._ensure_runtime()
        metadata = {
            "operation_id": str(operation_id),
            "memory_key": str(memory_key),
            "memory_type": str(memory_type),
            "memory_status": _ACTIVE_STATUS,
            "entity_scope": self.scope.entity_id,
            "scope_version": self.scope.scope_version,
            "seller_scope": self.scope.seller_scope,
            "value_json": json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None,
            "source_turn_id": str(source_turn_id) if source_turn_id else None,
            "source_trace_id": str(source_trace_id) if source_trace_id else None,
        }
        response = self._run(
            lambda: memory.add(
                text,
                user_id=self.scope.entity_id,
                metadata=metadata,
                infer=infer,
            ),
            timeout=float(getattr(self.settings, "mem0_write_timeout_seconds", 5.0)),
            operation="add",
        )
        ids = [
            str(row.get("id")) for row in _results(response)
            if isinstance(row, dict) and row.get("id")
        ]
        self._record("add", result_count=len(ids), infer=bool(infer))
        return ids

    def update(
        self,
        memory_id: str,
        *,
        operation_id: str | None = None,
        text: str | None = None,
        memory_status: str | None = None,
        value: Any = None,
        source_turn_id: str | None = None,
    ) -> bool:
        memory, _ = self._ensure_runtime()
        if operation_id is not None:
            self._validate_operation(operation_id)
        row = self._owned_row(memory_id)
        if row is None:
            return False
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "operation_id": str(operation_id or metadata.get("operation_id") or ""),
            "memory_status": str(memory_status or metadata.get("memory_status") or _ACTIVE_STATUS),
            "entity_scope": self.scope.entity_id,
            "scope_version": self.scope.scope_version,
            "seller_scope": self.scope.seller_scope,
        })
        if source_turn_id:
            metadata["source_turn_id"] = str(source_turn_id)
        if value is not None:
            metadata["value_json"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        self._run(
            lambda: memory.update(memory_id, text=text, metadata=metadata)
            if text is not None else memory.update(memory_id, metadata=metadata),
            timeout=float(getattr(self.settings, "mem0_write_timeout_seconds", 5.0)),
            operation="update",
        )
        self._record("update")
        return True

    def delete(self, memory_id: str) -> bool:
        memory, _ = self._ensure_runtime()
        if self._owned_row(memory_id) is None:
            return False
        self._run(
            lambda: memory.delete(memory_id),
            timeout=float(getattr(self.settings, "mem0_write_timeout_seconds", 5.0)),
            operation="delete",
        )
        self._record("delete")
        return True

    def get_validated(
        self, memory_id: str, *, operation_id: str, memory_type: str,
        memory_key: str, include_expired: bool = False,
    ) -> PreferenceHit | None:
        """Fetch an owned preference; management may inspect expired scope defaults."""
        row = self._owned_row(memory_id)
        if row is None:
            return None
        metadata = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(metadata, dict):
            return None
        if str(metadata.get("operation_id") or "") != operation_id:
            return None
        if str(metadata.get("memory_type") or "") != memory_type:
            return None
        if str(metadata.get("memory_key") or "") != memory_key:
            return None
        if include_expired and str(metadata.get("memory_status") or "") != _ACTIVE_STATUS:
            return None
        return self._validated_hit(
            row, expected_types={memory_type}, expect_active=not include_expired,
        )

    def reconcile(self, operation_id: str) -> list[PreferenceHit]:
        """Return only records that also pass the authoritative Registry."""
        return self._reconcile(operation_id, require_registry=True)

    def reconcile_pending(self, operation_id: str) -> list[PreferenceHit]:
        """Find a just-written record before its Registry commit.

        This method is for the trusted Outbox worker only. It still constrains
        the Mem0 query and validates entity/seller scope, operation ID, type,
        and scope version; it skips only the Registry status/reference check,
        which cannot succeed until this very operation is committed.
        """
        return self._reconcile(operation_id, require_registry=False)

    def _reconcile(
        self, operation_id: str, *, require_registry: bool,
    ) -> list[PreferenceHit]:
        self._validate_operation(operation_id)
        memory, _ = self._ensure_runtime()
        response = self._run(
            lambda: memory.get_all(
                filters={
                    "user_id": self.scope.entity_id,
                    "operation_id": str(operation_id),
                },
                top_k=50,
            ),
            timeout=float(getattr(self.settings, "mem0_search_timeout_seconds", 1.5)),
            operation="reconcile",
        )
        hits: list[PreferenceHit] = []
        for row in _results(response):
            metadata = row.get("metadata") if isinstance(row, dict) else None
            if not isinstance(metadata, dict) or str(metadata.get("operation_id") or "") != operation_id:
                continue
            hit = self._validated_hit(
                row, expected_types=None, expect_active=False,
                validate_registry=require_registry,
            )
            if hit is not None:
                hits.append(hit)
        self._record("reconcile", result_count=len(hits))
        return hits

    def delete_operation(
        self, operation_id: str, *, keep_memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete duplicate operation rows one by one; never delete_all(metadata)."""
        hits = self.reconcile(operation_id)
        deleted = 0
        kept = keep_memory_id or (hits[0].memory_id if hits else None)
        for hit in hits:
            if hit.memory_id == kept:
                continue
            if self.delete(hit.memory_id):
                deleted += 1
        remaining = [hit.memory_id for hit in self.reconcile(operation_id)]
        self._record("delete_operation", deleted_count=deleted, remaining_count=len(remaining))
        return {
            "operation_id": operation_id,
            "kept_memory_id": kept,
            "deleted_count": deleted,
            "remaining_memory_ids": remaining,
        }

    # ------------------------------------------------------------------
    # Runtime and validation
    # ------------------------------------------------------------------
    def _ensure_runtime(self) -> tuple[Any, Any]:
        if self._memory is not None:
            if self._client is None:
                self._client = self._memory.vector_store.client
            return self._memory, self._client
        key = self._runtime_key()
        with self._runtime_lock:
            runtime = self._runtimes.get(key)
            if runtime is None:
                memory = self._build_memory()
                client = memory.vector_store.client
                self._ensure_indexes(client)
                runtime = (memory, client)
                self._runtimes[key] = runtime
            self._memory, self._client = runtime
        return self._memory, self._client

    def _runtime_key(self) -> tuple[Any, ...]:
        return (
            self.collection,
            str(getattr(self.settings, "mem0_qdrant_url", "") or getattr(self.settings, "qdrant_url", "")),
            str(getattr(self.settings, "mem0_embedding_model", "")),
            int(getattr(self.settings, "mem0_embedding_dims", 512)),
        )

    def _build_memory(self) -> Any:
        try:
            os.environ.setdefault("MEM0_TELEMETRY", "False")
            from mem0 import Memory
        except Exception as exc:  # noqa: BLE001 - optional runtime package
            raise Mem0Unavailable("mem0_sdk_not_installed") from exc
        url = str(getattr(self.settings, "mem0_qdrant_url", "") or getattr(self.settings, "qdrant_url", ""))
        parsed = urlsplit(url)
        if not parsed.hostname:
            raise Mem0ConfigurationError("mem0_qdrant_url_invalid")
        dims = int(getattr(self.settings, "mem0_embedding_dims", 512))
        collection = self.collection
        qdrant_config: dict[str, Any] = {
            "collection_name": collection,
            "embedding_model_dims": dims,
            "host": parsed.hostname,
            "port": parsed.port or (443 if parsed.scheme == "https" else 6333),
            "https": parsed.scheme == "https",
            "on_disk": False,
        }
        api_key = str(getattr(self.settings, "mem0_qdrant_api_key", "") or "")
        if api_key:
            qdrant_config["api_key"] = api_key
        llm_model = str(getattr(self.settings, "mem0_llm_model", "") or "")
        llm_key = str(getattr(self.settings, "mem0_llm_api_key", "") or getattr(self.settings, "deepseek_api_key", "") or "not-configured")
        llm_base = str(getattr(self.settings, "mem0_llm_base_url", "") or getattr(self.settings, "deepseek_base_url", "") or "")
        config = {
            "vector_store": {"provider": "qdrant", "config": qdrant_config},
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": str(getattr(self.settings, "mem0_embedding_model", "BAAI/bge-small-zh-v1.5")),
                    "embedding_dims": dims,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model or "gpt-4o-mini",
                    "api_key": llm_key,
                    **({"openai_base_url": llm_base} if llm_base else {}),
                },
            },
            "history_db_path": str(
                getattr(self.settings, "mem0_history_db_path", "")
                or os.path.join("data", "mem0", "history.db")
            ),
        }
        history_parent = os.path.dirname(config["history_db_path"])
        if history_parent:
            os.makedirs(history_parent, exist_ok=True)
        try:
            return Memory.from_config(config)
        except Exception as exc:  # noqa: BLE001 - provider construction is optional
            raise Mem0Unavailable("mem0_provider_init_failed") from exc

    def _ensure_indexes(self, client: Any) -> None:
        try:
            from qdrant_client import models
        except Exception as exc:  # noqa: BLE001
            raise Mem0Unavailable("qdrant_client_not_installed") from exc
        for field in ("operation_id", "memory_type", "memory_key", "memory_status", "scope_version"):
            try:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001 - existing index is fine, other failures degrade
                if "already" not in str(exc).lower() and "exist" not in str(exc).lower():
                    raise Mem0Unavailable("payload_index_creation_failed") from exc

    def _validated_hit(
        self,
        row: dict[str, Any],
        *,
        expected_types: set[str] | None,
        expected_keys: set[str] | None = None,
        expect_active: bool = True,
        validate_registry: bool = True,
    ) -> PreferenceHit | None:
        if not isinstance(row, dict) or not row.get("id"):
            return self._reject_hit("invalid_record")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            return self._reject_hit("metadata_missing")
        memory_type = str(metadata.get("memory_type") or "")
        memory_key = str(metadata.get("memory_key") or "")
        if memory_type not in ALLOWED_MEMORY_TYPES:
            return self._reject_hit("memory_type_not_allowed")
        if expected_types is not None and memory_type not in expected_types:
            return self._reject_hit("memory_type_mismatch")
        if expected_keys is not None and memory_key not in expected_keys:
            return self._reject_hit("memory_key_mismatch")
        if not memory_key or str(metadata.get("entity_scope") or "") != self.scope.entity_id:
            return self._reject_hit("entity_scope_mismatch")
        if str(metadata.get("scope_version") or "") != self.scope.scope_version:
            return self._reject_hit("scope_version_mismatch")
        if str(metadata.get("seller_scope") or "") != self.scope.seller_scope:
            return self._reject_hit("seller_scope_mismatch")
        if str(row.get("user_id") or "") != self.scope.entity_id:
            return self._reject_hit("entity_id_mismatch")
        if expect_active and str(metadata.get("memory_status") or "") != _ACTIVE_STATUS:
            return self._reject_hit("memory_not_active")
        if expect_active and _expired(metadata.get("expires_at")):
            return self._reject_hit("memory_expired")
        if validate_registry:
            try:
                decision = self.registry.validate(
                    context=self.context,
                    entity_scope=self.scope.entity_id,
                    memory_id=str(row["id"]),
                    operation_id=str(metadata.get("operation_id") or "") or None,
                    memory_type=memory_type,
                    memory_key=memory_key,
                    expect_active=expect_active,
                )
            except RegistryUnavailable as exc:
                raise Mem0Unavailable(getattr(exc, "args", ["registry_unavailable"])[0]) from exc
            if not decision.active:
                reason = str(decision.reason or "registry_not_active")
                return self._reject_hit(
                    reason if reason in {"registry_not_active", "registry_expired", "registry_identity_mismatch"}
                    else "registry_rejected"
                )
        value = _decode_value(metadata.get("value_json"), row.get("memory"))
        return PreferenceHit(
            memory_id=str(row["id"]),
            text=str(row.get("memory") or ""),
            memory_type=memory_type,
            memory_key=memory_key,
            value=value,
            operation_id=str(metadata.get("operation_id") or "") or None,
            source_turn_id=str(metadata.get("source_turn_id") or "") or None,
            source_trace_id=str(metadata.get("source_trace_id") or "") or None,
            score=_score(row.get("score")),
            metadata=metadata,
        )

    def _reject_hit(self, reason: str) -> None:
        """Audit a rejected Mem0 candidate without logging its body or identifiers."""
        try:
            record_event("mem0.preference_hit_rejected", {
                "trace_id": self.trace_id or None,
                "mem0.rejection_reason": reason,
                "mem0.collection": self.collection,
            })
        except Exception:  # noqa: BLE001 - audit instrumentation must not break fail-closed behavior
            pass
        return None

    def _owned_row(self, memory_id: str) -> dict[str, Any] | None:
        memory, _ = self._ensure_runtime()
        try:
            row = self._run(
                lambda: memory.get(memory_id),
                timeout=float(getattr(self.settings, "mem0_search_timeout_seconds", 1.5)),
                operation="get",
            )
        except Exception:
            return None
        if not isinstance(row, dict) or str(row.get("user_id") or "") != self.scope.entity_id:
            return None
        return row

    def _run(self, fn: Callable[[], Any], *, timeout: float, operation: str) -> Any:
        if timeout <= 0:
            raise Mem0Timeout("invalid_timeout")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0")
        try:
            future = executor.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FutureTimeout as exc:
                raise Mem0Timeout(f"{operation}_timeout") from exc
        except Mem0Error:
            raise
        except Exception as exc:  # noqa: BLE001 - provider failures are degradation signals
            raise Mem0Unavailable(f"{operation}_failed") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_collection(self) -> None:
        if not self.collection:
            raise Mem0ConfigurationError("mem0_collection_missing")
        reserved = {
            str(getattr(self.settings, "qdrant_collection", "")),
            str(getattr(self.settings, "rag_qdrant_active_alias", "")),
            str(getattr(self.settings, "metric_semantic_qdrant_collection", "")),
        }
        if self.collection in {item for item in reserved if item}:
            raise Mem0ConfigurationError("mem0_collection_must_be_isolated")

    @staticmethod
    def _validate_type_and_key(memory_type: str, memory_key: str) -> None:
        if str(memory_type) not in ALLOWED_MEMORY_TYPES:
            raise Mem0ConfigurationError("unsupported_memory_type")
        if not str(memory_key or "").strip():
            raise Mem0ConfigurationError("memory_key_required")

    @staticmethod
    def _validate_operation(operation_id: str) -> None:
        if not str(operation_id or "").strip():
            raise Mem0ConfigurationError("operation_id_required")

    def _record(self, operation: str, **fields: Any) -> None:
        try:
            record_event("mem0.preference", {
                "trace_id": getattr(self, "trace_id", None),
                "mem0.operation": operation,
                "mem0.collection": self.collection,
                **fields,
            })
        except Exception:  # noqa: BLE001 - observability must not break the request
            pass


def _normalise_types(values: Iterable[str] | None) -> list[str]:
    # None means "use the contract default"; an explicitly empty/invalid list
    # must not silently widen to all preference types.
    if values is None:
        return sorted(ALLOWED_MEMORY_TYPES)
    return [
        str(item) for item in values
        if str(item) in ALLOWED_MEMORY_TYPES
    ]


def _bounded_int(value: Any, lower: int, upper: int) -> int:
    try:
        return min(upper, max(lower, int(value)))
    except (TypeError, ValueError):
        return lower


def _bounded_float(value: Any, lower: float, upper: float) -> float:
    try:
        return min(upper, max(lower, float(value)))
    except (TypeError, ValueError):
        return lower


def _results(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        rows = response.get("results") or response.get("memories") or []
        return [item for item in rows if isinstance(item, dict)]
    return []


def _decode_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _expired(value: Any) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed < datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def _score(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _mem0_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("mem0ai")
    except Exception:  # noqa: BLE001
        return None


def _qdrant_client_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("qdrant-client")
    except Exception:  # noqa: BLE001
        return None
