"""Persistent document sync commands for the RAG knowledge corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.rag.document_sync import (
    enqueue_reconciliation,
    process_one_reconciliation,
    run_worker,
    watch_knowledge_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile Markdown knowledge changes into Qdrant")
    parser.add_argument("command", choices=("reconcile", "worker", "watch"))
    parser.add_argument("--root", default=None, help="knowledge root; defaults to RAG_KNOWLEDGE_ROOT")
    parser.add_argument("--collection", default=None, help="Qdrant collection; defaults to QDRANT_COLLECTION")
    parser.add_argument("--once", action="store_true", help="worker: process at most one queued snapshot")
    parser.add_argument("--retry-failed", action="store_true", help="reconcile: requeue a terminally failed identical snapshot")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--reconcile-seconds", type=float, default=60.0)
    parser.add_argument("--debounce-seconds", type=float, default=1.0)
    parser.add_argument("--with-worker", action="store_true", help="watch: also run a worker in this process")
    args = parser.parse_args()
    settings = get_settings()
    root = args.root or settings.rag_knowledge_root
    collection = args.collection or settings.rag_qdrant_active_alias or settings.qdrant_collection

    if args.command == "reconcile":
        result = enqueue_reconciliation(
            root=root, collection_name=collection, retry_failed=args.retry_failed,
        )
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 0
    if args.command == "worker":
        if args.once:
            results = [process_one_reconciliation(root=root, collection_name=collection)]
            results = [item for item in results if item is not None]
        else:
            results = run_worker(
                root=root,
                collection_name=collection,
                poll_seconds=args.poll_seconds,
                reconcile_seconds=args.reconcile_seconds,
                on_result=lambda item: print(
                    json.dumps(item, ensure_ascii=False, default=str), flush=True
                ),
            )
        for result in results:
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return 0

    watch_knowledge_files(
        root=root,
        collection_name=collection,
        debounce_seconds=args.debounce_seconds,
        with_worker=args.with_worker,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
