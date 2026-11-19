"""Prefetch the configured Cross-Encoder into the project's HF cache."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.rag_reranker_enabled:
        print("Reranker is disabled; skipping model download.")
        return

    model_kwargs: dict[str, str] = {
        "repo_id": settings.rag_reranker_model,
        "cache_dir": settings.rag_cache_dir,
    }
    if settings.rag_reranker_model_revision:
        model_kwargs["revision"] = settings.rag_reranker_model_revision
    snapshot_download(
        **model_kwargs,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "modules.json",
            "sentence_bert_config.json",
        ],
    )
    print(
        f"Cached {settings.rag_reranker_model}"
        f"@{settings.rag_reranker_model_revision or 'default'}"
        f" in {settings.rag_cache_dir}"
    )


if __name__ == "__main__":
    main()
