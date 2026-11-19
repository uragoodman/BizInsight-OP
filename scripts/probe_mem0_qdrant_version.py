"""Probe Mem0 2.2.1 against a real Qdrant server for a specific client pin.

Usage:
  python scripts/probe_mem0_qdrant_version.py --url http://127.0.0.1:6333 \
      --collection wp3_version_probe

Run once with qdrant-client==1.16.2 and once with 1.20.0.  The script prints
one JSON line with the client/server versions, warnings and each operation
result.  It uses a disposable collection and removes it on exit.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
import warnings
from pathlib import Path

from mem0 import Memory
from qdrant_client import QdrantClient, models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default=f"wp3_version_{uuid.uuid4().hex[:8]}")
    args = parser.parse_args()
    client = QdrantClient(url=args.url, timeout=30)
    history = Path(tempfile.mkdtemp(prefix="wp3_version_")) / "history.db"
    collection = args.collection
    if client.collection_exists(collection):
        client.delete_collection(collection)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            memory = Memory.from_config({
                "vector_store": {"provider": "qdrant", "config": {
                    "collection_name": collection, "host": _host(args.url),
                    "port": _port(args.url), "embedding_model_dims": 512, "on_disk": False,
                }},
                "embedder": {"provider": "huggingface", "config": {
                    "model": "BAAI/bge-small-zh-v1.5", "embedding_dims": 512,
                }},
                "llm": {"provider": "openai", "config": {
                    "model": "gpt-4o-mini", "api_key": "not-configured",
                }},
                "history_db_path": str(history),
            })
            memory.vector_store.client.create_payload_index(
                collection, field_name="operation_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            a = memory.add("偏好简洁", user_id="uA", metadata={
                "operation_id": "op1", "memory_type": "presentation",
                "memory_status": "active",
            }, infer=False)
            b = memory.add("偏好表格", user_id="uB", metadata={
                "operation_id": "op1", "memory_type": "presentation",
                "memory_status": "active",
            }, infer=False)
            search = memory.search(
                "简洁", filters={"user_id": "uA", "operation_id": "op1"},
                top_k=5, threshold=0.0,
            )
            for index in range(10):
                memory.add(f"填充 {index}", user_id="uA", metadata={
                    "operation_id": f"f{index}", "memory_type": "presentation",
                    "memory_status": "active",
                }, infer=False)
            limited = memory.get_all(
                filters={"user_id": "uA", "operation_id": "op1"}, top_k=1,
            )
            memory.update(a["results"][0]["id"], text="偏好非常简洁")
            after = memory.get_all(filters={"user_id": "uA", "operation_id": "op1"}, top_k=5)
            memory.delete(b["results"][0]["id"])
            delete_count = len(memory.get_all(filters={"user_id": "uB"}, top_k=5).get("results") or [])
            first = memory.add("重复", user_id="uA", metadata={
                "operation_id": "dup", "memory_type": "presentation",
                "memory_status": "active",
            }, infer=False)
            second = memory.add("重复", user_id="uA", metadata={
                "operation_id": "dup", "memory_type": "presentation",
                "memory_status": "active",
            }, infer=False)
            duplicates = memory.get_all(filters={"user_id": "uA", "operation_id": "dup"}, top_k=10)
            memory.delete(second["results"][0]["id"])
            duplicate_after = memory.get_all(filters={"user_id": "uA", "operation_id": "dup"}, top_k=10)
            info = client.get_collection(collection)
            print(json.dumps({
                "client": _version("qdrant-client"),
                "server": _server_version(args.url),
                "mem0": _version("mem0ai"),
                "warnings": [str(item.message) for item in caught],
                "add_a": bool(a.get("results")), "add_b": bool(b.get("results")),
                "search_count": len(search.get("results") or []),
                "top_k_filter_count": len(limited.get("results") or []),
                "update_ok": after.get("results") is not None,
                "delete_b_count": delete_count,
                "duplicate_before": len(duplicates.get("results") or []),
                "duplicate_after": len(duplicate_after.get("results") or []),
                "operation_id_index": "operation_id" in info.payload_schema,
            }, ensure_ascii=False))
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        if client.collection_exists("mem0migrations"):
            client.delete_collection("mem0migrations")
        client.close()
    return 0


def _version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


def _server_version(url: str) -> str:
    try:
        import httpx

        return str(httpx.get(url.rstrip("/") + "/", timeout=5).json().get("version") or "unknown")
    except Exception:
        return "unknown"


def _host(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).hostname or "127.0.0.1"


def _port(url: str) -> int:
    from urllib.parse import urlsplit

    return urlsplit(url).port or 6333


if __name__ == "__main__":
    raise SystemExit(main())
