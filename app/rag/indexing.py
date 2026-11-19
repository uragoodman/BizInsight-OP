from __future__ import annotations
"""RAG 知识库加载与 Qdrant 索引构建。

本文件负责把 `docs/knowledge` 下的 markdown 知识文档转换成结构化文档和 chunk，
并在需要时写入 Qdrant 向量数据库。

结构与职责：
- 文档发现：`discover_knowledge_files` 扫描知识库 markdown 文件。
- 元数据抽取：`infer_doc_type`、`_extract_title` 根据路径和标题抽取 doc_type/title。
- 文档切分：`load_knowledge_documents` 生成整篇文档对象，
  `load_knowledge_chunks` 按 markdown section 生成可检索 chunk。
- Qdrant 入库：`build_qdrant_index` 使用 LlamaIndex + HuggingFace embedding
  将 chunk 写入 Qdrant collection。

使用者：
- `scripts/ingest_knowledge.py` 会调用 `build_qdrant_index` 执行离线入库。
- `app.rag.retriever` 会调用 `load_knowledge_chunks` 作为本地降级检索的数据源。
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import uuid
from typing import Any

from app.core.config import get_settings
from app.rag.config import configure_rag_runtime, get_rag_settings
from app.rag.fingerprint import (
    SPARSE_FIELD_COMPOSITION,
    build_index_fingerprint,
    corpus_digest,
    sparse_model_digest,
)
from app.rag.schemas import KnowledgeChunk, KnowledgeDocument, KnowledgeDocType
from app.rag.pipeline import build_document_chunks, split_markdown_sections
from app.rag import sparse as sparse_features


MANIFEST_SCHEMA_VERSION = 2
CHUNK_PIPELINE_VERSION = "section-v1"
DENSE_CHUNK_SIZE = 512
DENSE_CHUNK_OVERLAP = 80
INDEXING_CODE_VERSION = "indexing-v2"
_CORPUS_DIGEST_CACHE: tuple[tuple[tuple[str, int, int], ...], str] | None = None


class IndexSafetyError(RuntimeError):
    """Raised when an index build would mutate an existing collection unsafely."""


def _require_collection_name(collection_name: str | None) -> str:
    """Require an explicit target so an ingest cannot silently reuse production data."""

    normalized = (collection_name or "").strip()
    if not normalized:
        raise IndexSafetyError(
            "collection_name is required; choose a new versioned collection explicitly"
        )
    return normalized


def _inspect_collection(client, collection_name: str) -> tuple[bool, int]:
    """Read collection state without creating, deleting, or upserting anything."""

    try:
        exists = bool(client.collection_exists(collection_name))
        count = int(client.count(collection_name).count) if exists else 0
        return exists, count
    except Exception as exc:  # pragma: no cover - depends on Qdrant service
        raise IndexSafetyError(
            f"cannot inspect target collection {collection_name!r}: {type(exc).__name__}"
        ) from exc


def _collection_preview(client, collection_name: str, expected_points: int) -> dict[str, object]:
    """Return the zero-write part of the ingest safety decision."""

    exists, current_points = _inspect_collection(client, collection_name)
    return {
        "collection": collection_name,
        "exists": exists,
        "current_points": current_points,
        "expected_points": expected_points,
        "would_delete": exists,
        "write_performed": False,
    }


def _guard_collection_target(
    client,
    collection_name: str,
    *,
    allow_destructive_rebuild: bool,
) -> tuple[bool, int]:
    """Enforce the safe default for both dense and native builders.

    An existing collection is never implicitly reused or deleted.  Callers that
    intentionally replace one must opt in explicitly and still perform the
    deletion at the narrowest point immediately before creating the new schema.
    """

    exists, current_points = _inspect_collection(client, collection_name)
    if exists and not allow_destructive_rebuild:
        raise IndexSafetyError(
            f"target collection {collection_name!r} already exists; "
            "use a new versioned collection or explicitly enable destructive rebuild"
        )
    return exists, current_points


def _validate_destructive_evidence(
    *,
    allow_destructive_rebuild: bool,
    snapshot_ref: str | None,
    release_manifest_path: str | Path | None,
) -> None:
    """Require an auditable backup/release pair before destructive rebuilds."""

    if not allow_destructive_rebuild:
        return
    if not snapshot_ref or not str(snapshot_ref).strip():
        raise IndexSafetyError("destructive rebuild requires snapshot_ref")
    if not release_manifest_path:
        raise IndexSafetyError("destructive rebuild requires release_manifest_path")
    manifest_path = Path(release_manifest_path)
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        raise IndexSafetyError(
            f"release manifest must exist and be non-empty: {manifest_path}"
        )
    try:
        release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexSafetyError("release manifest must be valid JSON") from exc
    if not isinstance(release_manifest, dict):
        raise IndexSafetyError("release manifest must contain a JSON object")
    declared_snapshot = (
        release_manifest.get("snapshot_ref")
        or release_manifest.get("qdrant_snapshot_ref")
    )
    if declared_snapshot and str(declared_snapshot) != str(snapshot_ref):
        raise IndexSafetyError("snapshot_ref does not match the release manifest")


def _hash_text(text: str) -> str:
    """计算文档内容 hash。

    参数：
    - `text`：文档原文。

    返回值：
    - `str`：SHA256 十六进制摘要。

    使用场景：
    - 增量索引时判断 markdown 内容是否发生变化。

    为什么需要：
    - 如果没有 hash，每次执行入库脚本都只能全量重建。
    - 有了 hash 后，未变化文档可以跳过，减少 embedding 计算和 Qdrant 写入。
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_path() -> Path:
    """返回索引 manifest 文件路径。

    参数：无。
    返回值：`Path`，来自 RAG 配置的 `RAG_MANIFEST_PATH`。

    使用场景：
    - `build_qdrant_index` 读取和更新增量索引状态。
    """

    return Path(get_rag_settings().manifest_path)


@lru_cache(maxsize=4)
def _load_dense_embedding(model_name: str):
    """Load one HuggingFace embedding instance per model and reuse it.

    Loading a transformer can take seconds and allocate hundreds of MB.  The
    previous implementation constructed the model inside the returned
    function, so every chunk/query paid that startup cost again.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=model_name)


@lru_cache(maxsize=4)
def _load_embedding_tokenizer(model_name: str, cache_dir: str, local_files_only: bool):
    """加载与 embedding 模型相同的 HuggingFace tokenizer。

    这个 tokenizer 只用于 SentenceSplitter 的 token 长度计算，不负责切分
    句子或生成 embedding。这样切片长度与实际 BGE 输入使用同一套 token 规则。
    """

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def _make_splitter_tokenizer(tokenizer):
    """把 HuggingFace tokenizer 适配成 LlamaIndex 所需的 token-id 函数。"""

    return lambda text: tokenizer.encode(
        text,
        add_special_tokens=False,
        verbose=False,
    )


def make_dense_embed_fn():
    """返回使用 HuggingFaceEmbedding 的稠密编码函数（模型延迟加载）。

    返回值：
    - 闭包：`Callable[[str], list[float]]`，把文本编码为 dense 向量。

    使用场景：
    - `build_qdrant_index_native` 在未注入 `embed_fn` 时生成默认稠密编码。
    - `retriever._retrieve_from_qdrant_native` 查询 dense 向量时使用同一实现。
    """

    rag_settings = get_rag_settings()

    def _embed(text: str) -> list[float]:
        embedding = _load_dense_embedding(rag_settings.embedding_model)
        return list(embedding.get_text_embedding(text))

    return _embed


def _artifact_path(base: Path, collection_name: str | None) -> Path:
    """Return a collection-scoped artifact path without changing legacy default."""

    if not collection_name or collection_name == get_rag_settings().collection_name:
        return base
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", collection_name)
    return base.with_name(f"{base.stem}.{safe_name}{base.suffix}")


def _sparse_model_path(collection_name: str | None = None) -> Path:
    """返回 sparse 词表/IDF 模型文件路径。

    参数：无。
    返回值：`Path`，位于 manifest 同目录下的 `rag-sparse-model.json`。

    使用场景：
    - `build_qdrant_index_native` 写入查询阶段必须加载的稀疏模型。
    - `retriever._retrieve_from_qdrant_native` 读取同一份模型编码 query。
    """

    base = Path(get_rag_settings().manifest_path).parent / "rag-sparse-model.json"
    return _artifact_path(base, collection_name)


def _load_manifest(collection_name: str | None = None) -> dict[str, Any]:
    """读取增量索引 manifest。

    参数：无。
    返回值：
    - `dict[str, Any]`：manifest 内容。文件不存在或损坏时返回空结构。

    manifest 记录：
    - schema_version：manifest 结构版本。
    - index_version：当前知识库索引版本。
    - embedding_model：本次索引使用的 embedding 模型。
    - documents：每篇文档的 hash、chunk 数、source_path。

    为什么需要：
    - 解决“知识库稍微改动就全量重建”的问题。
    - 支持后续 trace 回放时知道命中证据来自哪个索引版本。
    """

    path = _artifact_path(_manifest_path(), collection_name)
    if not path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "documents": {},
            "compatibility": "legacy_unverified",
        }
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            manifest["compatibility"] = "legacy_unverified"
        return manifest
    except json.JSONDecodeError:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "documents": {},
            "compatibility": "legacy_unverified",
        }


def _write_manifest(manifest: dict[str, Any], collection_name: str | None = None) -> None:
    """写入增量索引 manifest。

    参数：
    - `manifest`：要持久化的 manifest 字典。

    返回值：`None`。

    使用场景：
    - Qdrant 索引构建成功后，记录每篇文档的最新 hash 和 chunk 数。
    """

    path = _artifact_path(_manifest_path(), collection_name or manifest.get("collection"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sync_state_path(collection_name: str) -> Path:
    return _artifact_path(_manifest_path(), collection_name).with_suffix(".sync-state.json")


def _load_sync_state(collection_name: str) -> dict[str, Any]:
    path = _sync_state_path(collection_name)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise IndexSafetyError(f"cannot read interrupted-sync recovery state: {path}")
    if not isinstance(value, dict) or value.get("collection") != collection_name:
        raise IndexSafetyError(f"invalid interrupted-sync recovery state: {path}")
    return value


def _write_sync_state(collection_name: str, state: dict[str, Any]) -> None:
    path = _sync_state_path(collection_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _clear_sync_state(collection_name: str) -> None:
    _sync_state_path(collection_name).unlink(missing_ok=True)


def discover_knowledge_files(root: str | None = None) -> list[str]:
    """发现知识库 markdown 文件。

    参数：
    - `root`：知识库根目录。为空时使用 `.env` 中的 `RAG_KNOWLEDGE_ROOT`，
      默认是 `docs/knowledge`。

    返回值：
    - `list[str]`：按路径排序后的 markdown 文件路径列表。

    使用场景：
    - 入库脚本打印当前知识库文件数量。
    - `load_knowledge_documents` 遍历这些文件并构造文档对象。
    """

    knowledge_root = Path(root or get_rag_settings().knowledge_root)
    if not knowledge_root.exists():
        return []
    return [str(path) for path in sorted(knowledge_root.rglob("*.md"))]


def infer_doc_type(path: Path) -> KnowledgeDocType:
    """根据文件路径推断知识类型。

    参数：
    - `path`：知识文档路径。

    返回值：
    - `KnowledgeDocType`：`metric`、`sop`、`template` 或 `unknown`。

    具体规则：
    - 路径中包含 `metrics` 视为指标定义文档。
    - 路径中包含 `sop` 视为分析流程文档。
    - 路径中包含 `templates` 视为报告模板文档。

    使用场景：
    - 写入 Qdrant metadata。
    - 检索时通过 metadata filter 按任务类型定向召回。
    """

    parts = {part.lower() for part in path.parts}
    if "metrics" in parts:
        return "metric"
    if "sop" in parts:
        return "sop"
    if "templates" in parts:
        return "template"
    return "unknown"


def _extract_title(text: str, path: Path) -> str:
    """从 markdown 文本中抽取文档标题。

    参数：
    - `text`：markdown 文档内容。
    - `path`：文档路径，用于标题缺失时回退到文件名。

    返回值：
    - `str`：文档标题。优先取第一个 `#` 标题，否则返回 `path.stem`。

    使用场景：
    - 构造 `KnowledgeDocument.title`。
    - 构造 citation 和 trace 中展示的文档名称。
    """

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem


def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """按 markdown 二级标题切分文档。

    参数：
    - `text`：markdown 文档完整文本。

    返回值：
    - `list[tuple[str | None, str]]`：每个元素包含 section 标题和该 section 文本。
      如果正文出现在第一个 `##` 前，section 标题为 `None`。

    使用场景：
    - `load_knowledge_chunks` 将整篇文档切成更细粒度的检索证据。
    - Qdrant 入库时每个 section 作为一个向量点写入，提升 citation 精度。
    """

    return split_markdown_sections(text)


def load_knowledge_documents(root: str | None = None) -> list[KnowledgeDocument]:
    """加载知识库原始文档。

    参数：
    - `root`：知识库根目录。为空时使用 RAG 配置中的默认路径。

    返回值：
    - `list[KnowledgeDocument]`：包含文档原文和元数据的列表。

    具体操作：
    1. 调用 `discover_knowledge_files` 找到所有 markdown。
    2. 读取文件内容。
    3. 推断 doc_type、抽取标题、设置 metric_name 和 tags。

    使用场景：
    - `load_knowledge_chunks` 基于整篇文档继续切分。
    - 测试中用于验证知识库文件是否被正确加载。
    """

    knowledge_root = Path(root or get_rag_settings().knowledge_root)
    rag_settings = get_rag_settings()
    documents: list[KnowledgeDocument] = []
    for file_name in discover_knowledge_files(str(knowledge_root)):
        path = Path(file_name)
        text = path.read_text(encoding="utf-8")
        doc_type = infer_doc_type(path)
        relative_path = path.as_posix()
        documents.append(
            KnowledgeDocument(
                doc_id=relative_path,
                title=_extract_title(text, path),
                text=text,
                source_path=relative_path,
                doc_type=doc_type,
                metric_name=path.stem if doc_type == "metric" else None,
                tags=[doc_type, path.stem],
                document_hash=_hash_text(text),
                index_version=rag_settings.index_version,
            )
        )
    return documents


def load_knowledge_chunks(root: str | None = None) -> list[KnowledgeChunk]:
    """加载并切分知识库 chunk。

    参数：
    - `root`：知识库根目录。为空时使用 RAG 配置中的默认路径。

    返回值：
    - `list[KnowledgeChunk]`：按 markdown section 切分后的 chunk 列表。

    具体操作：
    1. 调用 `load_knowledge_documents` 得到整篇文档。
    2. 对每篇文档调用 `_split_markdown_sections`。
    3. 为每个 section 生成稳定的 `chunk_id` 和 citation metadata。

    使用场景：
    - Qdrant 入库时作为向量化单元。
    - Qdrant 不可用时，本地 `local_hybrid` 检索直接使用这些 chunk。
    """

    rag_settings = get_rag_settings()
    tokenizer = None
    if rag_settings.chunk_pipeline_version == "sentence-v2":
        tokenizer = _make_splitter_tokenizer(
            _load_embedding_tokenizer(
                rag_settings.embedding_model,
                rag_settings.cache_dir,
                rag_settings.offline_embeddings and Path(rag_settings.cache_dir).exists(),
            )
        )
    chunks: list[KnowledgeChunk] = []
    for document in load_knowledge_documents(root):
        chunks.extend(
            build_document_chunks(
                document,
                pipeline_version=rag_settings.chunk_pipeline_version,
                chunk_size=rag_settings.chunk_size,
                chunk_overlap=rag_settings.chunk_overlap,
                tokenizer=tokenizer,
            )
        )
    return chunks


def _select_changed_chunks(chunks: list[KnowledgeChunk], manifest: dict[str, Any]) -> list[KnowledgeChunk]:
    """根据 manifest 选择需要重新写入 Qdrant 的 chunk。

    参数：
    - `chunks`：当前知识库切分出的全部 chunk。
    - `manifest`：上一次成功索引后保存的 manifest。

    返回值：
    - `list[KnowledgeChunk]`：需要重新 embedding 并写入 Qdrant 的 chunk。

    判定规则：
    - 文档不存在于 manifest：需要写入。
    - 文档 hash 改变：需要写入。
    - 索引版本改变：需要写入。
    - embedding 模型改变：需要写入。

    改进点：
    - 原来每次执行入库都会对全部文档重新 embedding。
    - 现在只处理变更文档，降低模型计算和 Qdrant 写入成本。
    """

    rag_settings = get_rag_settings()
    documents = manifest.get("documents", {})
    changed_doc_ids = set()
    for chunk in chunks:
        previous = documents.get(chunk.doc_id)
        if previous is None:
            changed_doc_ids.add(chunk.doc_id)
            continue
        if previous.get("document_hash") != chunk.document_hash:
            changed_doc_ids.add(chunk.doc_id)
            continue
        if previous.get("index_version") != rag_settings.index_version:
            changed_doc_ids.add(chunk.doc_id)
            continue
        if previous.get("embedding_model") != rag_settings.embedding_model:
            changed_doc_ids.add(chunk.doc_id)
            continue
        if previous.get("embedding_model_revision", "") != rag_settings.embedding_model_revision:
            changed_doc_ids.add(chunk.doc_id)
            continue
        # Missing metadata means an old manifest, not a proven match.  Force
        # the document through the new build once so the manifest records the
        # pipeline contract explicitly.
        if previous.get("pipeline_version", "") != rag_settings.chunk_pipeline_version:
            changed_doc_ids.add(chunk.doc_id)

    return [chunk for chunk in chunks if chunk.doc_id in changed_doc_ids]


def _incremental_contract_differences(
    manifest: dict[str, Any], candidate_manifest: dict[str, Any]
) -> list[str]:
    """Compare the immutable embedding/chunk contract, excluding corpus content."""
    current = manifest.get("index_fingerprint")
    candidate = candidate_manifest.get("index_fingerprint")
    if isinstance(current, dict) and isinstance(candidate, dict):
        current_payload = dict(current.get("payload") or {})
        candidate_payload = dict(candidate.get("payload") or {})
        fields = set(current_payload) | set(candidate_payload)
        fields.discard("corpus_digest")
        return sorted(
            field for field in fields
            if current_payload.get(field) != candidate_payload.get(field)
        )

    # A legacy manifest can be upgraded in place only if its known contract
    # still matches. The live vector dimension is checked separately.
    differences = []
    for field in ("collection", "index_version", "embedding_model", "embedding_model_revision", "pipeline_version"):
        current_value = manifest.get(field)
        candidate_value = candidate_manifest.get(field)
        if current_value is not None and current_value != candidate_value:
            differences.append(field)
    return differences


def _build_manifest(
    chunks: list[KnowledgeChunk],
    *,
    collection_name: str | None = None,
    index_mode: str = "dense",
    dense_dimension: int | None = None,
    sparse_vocab_hash: str | None = None,
) -> dict[str, Any]:
    """根据当前知识库 chunk 生成新的 manifest。

    参数：
    - `chunks`：当前知识库全部 chunk。

    返回值：
    - `dict[str, Any]`：新的 manifest 内容。

    使用场景：
    - Qdrant 索引写入成功后，用它覆盖旧 manifest。
    """

    rag_settings = get_rag_settings()
    documents: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        item = documents.setdefault(
            chunk.doc_id,
            {
                "source_path": chunk.source_path,
                "document_hash": chunk.document_hash,
                "doc_type": chunk.doc_type,
                "metric_name": chunk.metric_name,
                "index_version": rag_settings.index_version,
                "embedding_model": rag_settings.embedding_model,
                "embedding_model_revision": rag_settings.embedding_model_revision,
                "pipeline_version": rag_settings.chunk_pipeline_version,
                "content_hashes": [],
                "chunks": [],
                "chunk_count": 0,
            },
        )
        item["chunk_count"] += 1
        item["content_hashes"].append(chunk.content_hash)
        item["chunks"].append({
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
            "section": chunk.section,
            "source_start": chunk.source_start,
            "source_end": chunk.source_end,
        })

    target_collection = collection_name or rag_settings.collection_name
    fingerprint = build_index_fingerprint(
        index_mode=index_mode,
        collection=target_collection,
        vector_names=["dense", "sparse"] if index_mode == "dense_sparse" else ["default"],
        distance="Cosine",
        dimension=dense_dimension,
        embedding_model=rag_settings.embedding_model,
        embedding_model_revision=rag_settings.embedding_model_revision,
        indexing_code_version=INDEXING_CODE_VERSION,
        chunk_pipeline_version=rag_settings.chunk_pipeline_version,
        chunk_parameters={
            "chunk_size": rag_settings.chunk_size,
            "chunk_overlap": rag_settings.chunk_overlap,
        },
        tokenizer_id=f"huggingface:{rag_settings.embedding_model}",
        sparse_vocab_hash=sparse_vocab_hash,
        corpus_hash=corpus_digest(chunks),
        index_version=rag_settings.index_version,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "index_version": rag_settings.index_version,
        "embedding_model": rag_settings.embedding_model,
        "embedding_model_revision": rag_settings.embedding_model_revision,
        "collection": target_collection,
        "index_mode": index_mode,
        "pipeline_version": rag_settings.chunk_pipeline_version,
        "corpus_digest": fingerprint["payload"]["corpus_digest"],
        "index_fingerprint": fingerprint,
        "documents": documents,
    }


def _fingerprint_comparison(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, object]:
    """Compare a proposed fingerprint without changing the manifest."""

    current = manifest.get("index_fingerprint")
    if not current:
        return {"status": "legacy_unverified", "different_fields": []}
    current_payload = dict(current.get("payload") or {})
    candidate_payload = dict(candidate.get("payload") or {})
    fields = sorted({*current_payload, *candidate_payload})
    different = [
        field for field in fields
        if current_payload.get(field) != candidate_payload.get(field)
    ]
    return {
        "status": "match" if not different else "mismatch",
        "different_fields": different,
        "current_digest": current.get("digest"),
        "candidate_digest": candidate.get("digest"),
    }


def _collection_schema_signature(collection_info: Any) -> dict[str, Any]:
    """Normalize Qdrant's dense/sparse schema into fingerprint fields.

    Qdrant returns a single ``VectorParams`` object for unnamed dense vectors
    and a mapping for named vectors.  The index manifest uses the stable name
    ``default`` for the former so both representations can be compared.
    """

    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    sparse = getattr(params, "sparse_vectors", None)

    def value_attr(value: Any, name: str) -> Any:
        item = getattr(value, name, None)
        return getattr(item, "value", item)

    if isinstance(vectors, dict):
        vector_names = sorted(str(name) for name in vectors)
        vector_values = list(vectors.values())
    else:
        vector_names = ["default"] if vectors is not None else []
        vector_values = [vectors] if vectors is not None else []
    first_vector = vector_values[0] if vector_values else None
    sparse_names = sorted(str(name) for name in sparse) if isinstance(sparse, dict) else []
    return {
        "vector_names": vector_names,
        "distance": value_attr(first_vector, "distance") if first_vector is not None else None,
        "dimension": value_attr(first_vector, "size") if first_vector is not None else None,
        "sparse_vector_names": sparse_names,
    }


def _current_corpus_digest(root: str | None = None) -> str | None:
    """Read document hashes cheaply enough for a cached compatibility check."""

    global _CORPUS_DIGEST_CACHE
    files = [Path(path) for path in discover_knowledge_files(root)]
    signature: tuple[tuple[str, int, int], ...] = tuple(
        (path.as_posix(), path.stat().st_mtime_ns, path.stat().st_size)
        for path in files
    )
    if _CORPUS_DIGEST_CACHE and _CORPUS_DIGEST_CACHE[0] == signature:
        return _CORPUS_DIGEST_CACHE[1]
    documents = [
        SimpleNamespace(
            doc_id=path.as_posix(),
            document_hash=_hash_text(path.read_text(encoding="utf-8")),
        )
        for path in files
    ]
    digest = corpus_digest(documents)
    _CORPUS_DIGEST_CACHE = (signature, digest)
    return digest


def resolve_collection_alias(client, collection_name: str) -> str:
    """Resolve a Qdrant alias to its immutable backing collection, if present."""
    try:
        aliases = client.get_aliases().aliases
    except Exception:
        return collection_name
    for alias in aliases:
        if getattr(alias, "alias_name", None) == collection_name:
            return str(alias.collection_name)
    return collection_name


def switch_collection_alias(client, alias_name: str, collection_name: str) -> dict[str, object]:
    """Atomically publish a validated dense index collection through an alias."""
    alias_name = alias_name.strip()
    collection_name = _require_collection_name(collection_name)
    if not alias_name or alias_name == collection_name:
        raise IndexSafetyError("alias_name must be non-empty and differ from collection_name")
    if not client.collection_exists(collection_name):
        raise IndexSafetyError(f"candidate collection does not exist: {collection_name}")
    manifest = _load_manifest(collection_name)
    if manifest.get("collection") != collection_name or not manifest.get("index_fingerprint"):
        raise IndexSafetyError("candidate collection is missing a collection-scoped fingerprint manifest")
    index_mode = manifest.get("index_mode")
    if index_mode not in {"dense", "dense_sparse"}:
        raise IndexSafetyError(f"unsupported index mode for alias publishing: {index_mode}")
    expected_points = sum(
        int(document.get("chunk_count", 0))
        for document in (manifest.get("documents") or {}).values()
    )
    actual_points = int(client.count(collection_name, exact=True).count)
    if actual_points != expected_points:
        raise IndexSafetyError(
            f"candidate point count mismatch: expected {expected_points}, got {actual_points}"
        )
    compatibility = check_collection_fingerprint(
        client, collection_name, index_mode=index_mode
    )
    if compatibility["status"] != "match":
        raise IndexSafetyError(
            f"candidate fingerprint validation failed: {compatibility['different_fields']}"
        )

    aliases = client.get_aliases().aliases
    current_target = next(
        (str(item.collection_name) for item in aliases
         if getattr(item, "alias_name", None) == alias_name),
        None,
    )
    if current_target == collection_name:
        return {
            "alias": alias_name,
            "previous_collection": current_target,
            "collection": collection_name,
            "points": actual_points,
            "index_mode": index_mode,
            "fingerprint_digest": manifest["index_fingerprint"]["digest"],
            "status": "already_active",
        }
    if client.collection_exists(alias_name) and current_target is None:
        raise IndexSafetyError(f"alias name conflicts with an existing collection: {alias_name}")
    actions = []
    from qdrant_client import models
    if current_target:
        actions.append(models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=alias_name)
        ))
    actions.append(models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection_name, alias_name=alias_name)
    ))
    client.update_collection_aliases(change_aliases_operations=actions)
    return {
        "alias": alias_name,
        "previous_collection": current_target,
        "collection": collection_name,
        "points": actual_points,
        "index_mode": index_mode,
        "fingerprint_digest": manifest["index_fingerprint"]["digest"],
    }


def check_collection_fingerprint(
    client,
    collection_name: str,
    *,
    index_mode: str,
) -> dict[str, Any]:
    """Check a readable collection against its immutable local manifest.

    The result has three states:

    * ``match`` — schema and runtime contract agree;
    * ``legacy_unverified`` — the old collection has no fingerprint and may
      be used only as an explicitly observable compatibility path;
    * ``mismatch`` — the caller must refuse the collection and use a safe
      fallback rather than querying an incompatible vector space.

    This function is read-only.  It deliberately does not rebuild a manifest
    or infer a new fingerprint from a live collection.
    """

    collection_name = resolve_collection_alias(client, collection_name)
    manifest = _load_manifest(collection_name)
    fingerprint = manifest.get("index_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint.get("digest"):
        return {"status": "legacy_unverified", "digest": None, "different_fields": []}

    payload = dict(fingerprint.get("payload") or {})
    differences: list[str] = []
    if payload.get("collection") != collection_name:
        differences.append("collection")
    if payload.get("index_mode") != index_mode:
        differences.append("index_mode")

    settings = get_rag_settings()
    expected_runtime = {
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "chunk_pipeline_version": settings.chunk_pipeline_version,
        "chunk_parameters": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "tokenizer_id": f"huggingface:{settings.embedding_model}",
        "index_version": settings.index_version,
        "sparse_field_composition": list(SPARSE_FIELD_COMPOSITION),
    }
    for field, expected in expected_runtime.items():
        if field in payload and payload.get(field) != expected:
            differences.append(field)

    try:
        schema = _collection_schema_signature(client.get_collection(collection_name))
    except Exception as exc:  # pragma: no cover - service-specific failure
        return {
            "status": "mismatch",
            "digest": fingerprint.get("digest"),
            "different_fields": ["collection_schema_unavailable"],
            "exception_type": type(exc).__name__,
        }
    expected_dense_names = sorted(
        name for name in (payload.get("vector_names") or []) if name != "sparse"
    )
    if expected_dense_names and expected_dense_names != schema["vector_names"]:
        differences.append("vector_names")
    if payload.get("distance") and payload.get("distance") != schema["distance"]:
        differences.append("distance")
    if payload.get("dimension") is not None and payload.get("dimension") != schema["dimension"]:
        differences.append("dimension")
    expected_sparse = sorted(
        name for name in (payload.get("vector_names") or []) if name == "sparse"
    )
    if index_mode == "dense_sparse" and expected_sparse != schema["sparse_vector_names"]:
        differences.append("sparse_vector_names")
    if index_mode == "dense_sparse" and not _sparse_model_path(collection_name).exists():
        differences.append("sparse_model_artifact")
    if payload.get("corpus_digest"):
        try:
            current_corpus = _current_corpus_digest()
        except OSError:
            current_corpus = None
        if current_corpus is None or current_corpus != payload.get("corpus_digest"):
            differences.append("corpus_digest")

    return {
        "status": "match" if not differences else "mismatch",
        "digest": fingerprint.get("digest"),
        "different_fields": sorted(set(differences)),
    }


def build_qdrant_index(
    root: str | None = None,
    *,
    collection_name: str | None = None,
    allow_destructive_rebuild: bool = False,
    dry_run: bool = False,
    snapshot_ref: str | None = None,
    release_manifest_path: str | Path | None = None,
    client=None,
) -> dict[str, object]:
    """构建 Qdrant 向量索引。

    参数：
    - `root`：知识库根目录。为空时使用 RAG 配置中的默认路径。
    - `collection_name`：必须显式指定的目标 collection。新版本应使用版本化名称。
    - `allow_destructive_rebuild`：是否显式允许删除已存在 collection，默认关闭。
    - `dry_run`：只读取 collection 状态并输出计划，不创建、删除或写入任何资源。
    - `snapshot_ref`：破坏性重建前已完成的 Qdrant snapshot 标识。
    - `release_manifest_path`：与 snapshot 绑定的 release manifest 路径。
    - `client`：可选 Qdrant client，测试可注入 in-memory client。

    返回值：
    - `dict[str, object]`：索引构建结果，包括：
      - `backend`：固定为 `qdrant`。
      - `status`：`ok`、`empty`、`dependency_error` 或 `index_error`。
      - `collection`：写入的 Qdrant collection 名。
      - `documents`：实际写入的 chunk 数量。
      - `error`：失败时的错误信息。

    具体操作：
    1. 调用 `configure_rag_runtime` 设置缓存与离线 embedding 环境。
    2. 调用 `load_knowledge_chunks` 加载并切分知识库。
    3. 将每个 chunk 转换为 LlamaIndex `Document`，并写入 metadata。
    4. 使用 `HuggingFaceEmbedding` 生成向量。
    5. 通过 `QdrantVectorStore` 写入 Qdrant collection。

    使用场景：
    - 手动执行 `python scripts/ingest_knowledge.py` 时调用。
    - 后续 CI/CD 或知识库更新任务也可以复用这个函数。

    说明：
    - LlamaIndex/Qdrant 相关 import 放在函数内部，是为了让应用在缺少
      向量数据库服务或 embedding 依赖时仍可使用本地降级检索。
    """

    rag_settings = configure_rag_runtime()
    try:
        target_collection = _require_collection_name(collection_name)
        _validate_destructive_evidence(
            allow_destructive_rebuild=allow_destructive_rebuild,
            snapshot_ref=snapshot_ref,
            release_manifest_path=release_manifest_path,
        )
    except IndexSafetyError as exc:
        return {"backend": "qdrant", "status": "safety_error", "error": str(exc)}
    app_settings = get_settings()
    knowledge_chunks = load_knowledge_chunks(root)

    # 增量索引核心：
    # 先读取上次成功索引的 manifest，再根据文档 hash / index_version / embedding_model
    # 判断哪些文档真的发生了变化。没有变化时直接跳过 Qdrant 写入，避免重复 embedding。
    manifest = _load_manifest(target_collection)
    changed_chunks = _select_changed_chunks(knowledge_chunks, manifest)

    try:
        from llama_index.core import Settings
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client import models
    except ImportError as exc:
        return {
            "backend": "qdrant",
            "status": "dependency_error",
            "documents": len(changed_chunks),
            "error": str(exc),
        }

    try:
        owned_client = client is None
        client = client or QdrantClient(url=app_settings.qdrant_url)
        has_collection = bool(client.collection_exists(target_collection))
        if knowledge_chunks:
            Settings.embed_model = HuggingFaceEmbedding(model_name=rag_settings.embedding_model)
            if has_collection:
                dense_dimension = _collection_schema_signature(
                    client.get_collection(target_collection)
                ).get("dimension")
                if not dense_dimension:
                    raise IndexSafetyError("cannot determine the existing collection vector dimension")
            else:
                dense_dimension = len(Settings.embed_model.get_text_embedding(knowledge_chunks[0].text))
        elif has_collection:
            dense_dimension = _collection_schema_signature(
                client.get_collection(target_collection)
            ).get("dimension")
            if not dense_dimension:
                raise IndexSafetyError("cannot determine the existing collection vector dimension")
        else:
            if owned_client:
                client.close()
            return {
                "backend": "qdrant", "status": "empty", "collection": target_collection,
                "documents": 0,
            }

        preview = _collection_preview(client, target_collection, len(knowledge_chunks))
        if dry_run:
            candidate_manifest = _build_manifest(
                knowledge_chunks,
                collection_name=target_collection,
                index_mode="dense",
                dense_dimension=dense_dimension,
            )
            preview.update({
                "backend": "qdrant",
                "status": "dry_run",
                "index_version": rag_settings.index_version,
                "embedding_model": rag_settings.embedding_model,
                "pipeline_version": rag_settings.chunk_pipeline_version,
                "fingerprint": candidate_manifest["index_fingerprint"],
                "fingerprint_comparison": _fingerprint_comparison(
                    manifest, candidate_manifest["index_fingerprint"],
                ),
            })
            if owned_client:
                client.close()
            return preview

        exists = bool(client.collection_exists(target_collection))
        if exists and allow_destructive_rebuild:
            client.delete_collection(target_collection)
            exists = False
            changed_chunks = knowledge_chunks

        candidate_manifest = _build_manifest(
            knowledge_chunks,
            collection_name=target_collection,
            index_mode="dense",
            dense_dimension=dense_dimension,
        )
        pending_state = _load_sync_state(target_collection)
        pending_doc_ids = set(pending_state.get("changed_doc_ids", []))
        pending_deleted_ids = set(pending_state.get("deleted_doc_ids", []))
        if pending_doc_ids or pending_deleted_ids:
            force_ids = pending_doc_ids | pending_deleted_ids
            changed_ids = {chunk.doc_id for chunk in changed_chunks} | force_ids
            changed_chunks = [chunk for chunk in knowledge_chunks if chunk.doc_id in changed_ids]
        if exists:
            differences = _incremental_contract_differences(manifest, candidate_manifest)
            if differences:
                raise IndexSafetyError(
                    "existing collection uses a different index contract; build a new "
                    f"versioned collection (fields: {', '.join(differences)})"
                )
            current_points = int(client.count(target_collection, exact=True).count)
            expected_points = sum(
                int(document.get("chunk_count", 0))
                for document in (manifest.get("documents") or {}).values()
            )
            if current_points != expected_points and not pending_state:
                raise IndexSafetyError(
                    f"collection has {current_points} points but manifest accounts for "
                    f"{expected_points}; reconcile by building a new versioned collection"
                )
        else:
            changed_chunks = knowledge_chunks

        manifest_doc_ids = set((manifest.get("documents") or {}).keys()) if exists else set()
        current_doc_ids = {chunk.doc_id for chunk in knowledge_chunks}
        deleted_doc_ids = sorted((manifest_doc_ids - current_doc_ids) | (pending_deleted_ids - current_doc_ids))
        if exists and not changed_chunks and not deleted_doc_ids and not pending_state:
            if owned_client:
                client.close()
            return {
                "backend": "qdrant",
                "status": "skipped",
                "collection": target_collection,
                "documents": len(knowledge_chunks),
                "changed_documents": 0,
                "deleted_documents": 0,
                "index_version": rag_settings.index_version,
            }

        nodes = []
        for chunk in changed_chunks:
            nodes.append(TextNode(
                id_=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                text=chunk.text,
                embedding=Settings.embed_model.get_text_embedding(chunk.text),
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_id": chunk.source_path,
                    "source_revision": chunk.document_hash,
                    "title": chunk.title,
                    "source_path": chunk.source_path,
                    "doc_type": chunk.doc_type,
                    "metric_name": chunk.metric_name,
                    "tags": chunk.tags,
                    "section": chunk.section,
                    "document_hash": chunk.document_hash,
                    "index_version": chunk.index_version,
                    "pipeline_version": chunk.pipeline_version,
                    "content_hash": chunk.content_hash,
                    "source_start": chunk.source_start,
                    "source_end": chunk.source_end,
                    "visibility": chunk.visibility,
                    "tenant_id": chunk.tenant_id,
                    "classification": chunk.classification,
                    "allowed_roles": chunk.allowed_roles,
                },
            ))
        vector_store = QdrantVectorStore(client=client, collection_name=target_collection)
        # Use the already-final chunks and deterministic point IDs. Going
        # through VectorStoreIndex.from_documents creates fresh node UUIDs on
        # each run, which makes retries duplicate vectors even when the source
        # chunk is identical.
        changed_doc_ids = sorted({chunk.doc_id for chunk in changed_chunks})
        _write_sync_state(target_collection, {
            "schema_version": 1,
            "collection": target_collection,
            "corpus_digest": candidate_manifest.get("corpus_digest"),
            "changed_doc_ids": changed_doc_ids,
            "deleted_doc_ids": deleted_doc_ids,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        if nodes:
            vector_store.add(nodes)

        # Publish new/changed chunks first. Then remove only stale points for
        # that source; retrying after a partial upsert is safe because point IDs
        # are deterministic. Deletion failures fail the job and preserve the
        # old manifest so a retry/reconciliation can finish cleanup.
        for doc_id in changed_doc_ids:
            current_hash = next(
                chunk.document_hash for chunk in changed_chunks if chunk.doc_id == doc_id
            )
            client.delete(
                collection_name=target_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="source_path",
                            match=models.MatchValue(value=next(
                                chunk.source_path for chunk in changed_chunks
                                if chunk.doc_id == doc_id
                            )),
                        )],
                        must_not=[models.FieldCondition(
                            key="document_hash", match=models.MatchValue(value=current_hash)
                        )],
                    )
                ),
                wait=True,
            )
        for doc_id in deleted_doc_ids:
            client.delete(
                collection_name=target_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(
                        key="source_path",
                        match=models.MatchValue(value=(manifest.get("documents", {}).get(doc_id, {}) or {}).get("source_path", doc_id)),
                    )])
                ),
                wait=True,
            )
    except Exception as exc:  # pragma: no cover - depends on local services/models
        if 'owned_client' in locals() and owned_client:
            client.close()
        if isinstance(exc, IndexSafetyError):
            return {
                "backend": "qdrant",
                "status": "safety_error",
                "collection": target_collection,
                "documents": len(changed_chunks),
                "error": str(exc),
            }
        return {
            "backend": "qdrant",
            "status": "index_error",
            "collection": target_collection,
            "documents": len(changed_chunks),
            "error": str(exc),
        }

    # 只有在 Qdrant 写入成功后才更新 manifest。
    # 如果入库中途失败，manifest 保持旧状态，下一次执行仍会尝试补齐。
    manifest = _build_manifest(
        knowledge_chunks,
        collection_name=target_collection,
        index_mode="dense",
        dense_dimension=dense_dimension,
    )
    _write_manifest(manifest, target_collection)
    _clear_sync_state(target_collection)

    if owned_client:
        client.close()

    return {
        "backend": "qdrant",
        "status": "ok",
        "collection": target_collection,
        "documents": len(changed_chunks),
        "total_documents": len(knowledge_chunks),
        "changed_documents": len({chunk.doc_id for chunk in changed_chunks}),
        "deleted_documents": len(deleted_doc_ids),
        "index_version": rag_settings.index_version,
        "fingerprint_digest": manifest["index_fingerprint"]["digest"],
        "manifest_path": str(_artifact_path(_manifest_path(), target_collection)),
    }


def build_qdrant_index_native(
    root: str | None = None,
    *,
    collection_name: str | None = None,
    allow_destructive_rebuild: bool = False,
    dry_run: bool = False,
    snapshot_ref: str | None = None,
    release_manifest_path: str | Path | None = None,
    client=None,
    embed_fn=None,
    sparse_model_path: Path | str | None = None,
    chunks: list[KnowledgeChunk] | None = None,
) -> dict[str, object]:
    """构建 Qdrant native dense+sparse collection。

    参数：
    - `root`：知识库根目录。为空时使用 RAG 配置中的默认路径。
    - `collection_name`：必须显式指定的目标 collection。新版本应使用版本化名称。
    - `allow_destructive_rebuild`：是否显式允许删除已存在 collection，默认关闭。
    - `dry_run`：只读取 collection 状态并输出计划，不创建、删除或写入任何资源。
    - `snapshot_ref`：破坏性重建前已完成的 Qdrant snapshot 标识。
    - `release_manifest_path`：与 snapshot 绑定的 release manifest 路径。
    - `client`：可选的 `QdrantClient`。测试可传 `QdrantClient(":memory:")`；
      为空时按配置连接真实 Qdrant 服务。
    - `embed_fn`：可选的稠密编码函数 ``Callable[[str], list[float]]``。
      为空时使用 HuggingFaceEmbedding 生成向量；测试可注入确定性函数避免下载模型。
    - `sparse_model_path`：可选稀疏词表/IDF 落盘路径；为空时使用 manifest 同目录默认值。
    - `chunks`：可选已加载的 chunk 列表；为空时按 `root` 从磁盘加载。

    返回值：
    - `dict[str, object]`：索引构建结果，包含 `mode="dense_sparse"`、
      dense 维度、sparse 词表大小和写入 chunk 数。

    与 `build_qdrant_index` 的区别：
    - 该函数创建同一 collection 下的 dense + sparse 两个命名空间（native multimodal
      collection），并在查询时用 Qdrant 原生融合检索，而不是「dense 召回 + Python 侧
      关键词重排」。sparse 特征由 `app.rag.sparse` 用 BM25 风格权重生成，离线可复现，
      无需外部稀疏模型。
    - 该函数采用全量重建（recreate collection），保证词表与 collection 状态一致；
      增量优化留给部署侧索引任务。
    """
    rag_settings = configure_rag_runtime()
    try:
        target_collection = _require_collection_name(collection_name)
        _validate_destructive_evidence(
            allow_destructive_rebuild=allow_destructive_rebuild,
            snapshot_ref=snapshot_ref,
            release_manifest_path=release_manifest_path,
        )
    except IndexSafetyError as exc:
        return {
            "backend": "qdrant", "mode": "dense_sparse", "status": "safety_error",
            "error": str(exc),
        }
    knowledge_chunks = chunks if chunks is not None else load_knowledge_chunks(root)
    if not knowledge_chunks:
        return {
            "backend": "qdrant", "mode": "dense_sparse", "status": "empty",
            "collection": target_collection, "documents": 0,
        }

    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        return {
            "backend": "qdrant", "mode": "dense_sparse", "status": "dependency_error",
            "documents": 0, "error": str(exc),
        }

    owned_client = client is None
    if client is None:
        client = QdrantClient(url=get_settings().qdrant_url)

    try:
        preview = _collection_preview(client, target_collection, len(knowledge_chunks))
        if dry_run:
            preview_vocab, preview_idf = sparse_features.build_vocab(knowledge_chunks)
            candidate_manifest = _build_manifest(
                knowledge_chunks,
                collection_name=target_collection,
                index_mode="dense_sparse",
                dense_dimension=None,
                sparse_vocab_hash=sparse_model_digest(preview_vocab, preview_idf),
            )
            preview.update({
                "backend": "qdrant",
                "mode": "dense_sparse",
                "status": "dry_run",
                "index_version": rag_settings.index_version,
                "embedding_model": rag_settings.embedding_model,
                "pipeline_version": rag_settings.chunk_pipeline_version,
                "fingerprint": candidate_manifest["index_fingerprint"],
                "fingerprint_comparison": _fingerprint_comparison(
                    _load_manifest(target_collection), candidate_manifest["index_fingerprint"],
                ),
            })
            if owned_client:
                client.close()
            return preview
        _guard_collection_target(
            client,
            target_collection,
            allow_destructive_rebuild=allow_destructive_rebuild,
        )
    except IndexSafetyError as exc:
        if owned_client:
            client.close()
        return {
            "backend": "qdrant", "mode": "dense_sparse", "status": "safety_error",
            "collection": target_collection, "error": str(exc),
        }

    if embed_fn is None:
        embed_fn = make_dense_embed_fn()

    try:
        sample_vector = embed_fn(knowledge_chunks[0].text)
    except Exception as exc:  # pragma: no cover - depends on local embedding models
        if owned_client:
            client.close()
        return {
            "backend": "qdrant", "mode": "dense_sparse", "status": "index_error",
            "documents": 0, "error": f"embedding failed: {exc}",
        }
    dense_dimension = len(sample_vector)

    vocab, idf = sparse_features.build_vocab(knowledge_chunks)
    sparse_hash = sparse_model_digest(vocab, idf)
    try:
        if client.collection_exists(target_collection):
            client.delete_collection(target_collection)
        client.create_collection(
            collection_name=target_collection,
            vectors_config={"dense": models.VectorParams(size=dense_dimension, distance="Cosine")},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        points = []
        for chunk in knowledge_chunks:
            # Qdrant 要求 point id 为 UUID 或整数；用基于 chunk_id 的稳定 UUID，
            # 真实服务和 in-memory 模式都可用，且 chunk_id 仍保留在 payload 供 citation。
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embed_fn(chunk.text),
                        "sparse": sparse_features.sparse_embed(
                            sparse_features.chunk_search_text(chunk), vocab, idf,
                        ),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "source_path": chunk.source_path,
                        "doc_type": chunk.doc_type,
                        "metric_name": chunk.metric_name,
                        "tags": chunk.tags,
                        "section": chunk.section,
                        "document_hash": chunk.document_hash,
                        "index_version": chunk.index_version,
                        "pipeline_version": chunk.pipeline_version,
                        "content_hash": chunk.content_hash,
                        "source_start": chunk.source_start,
                        "source_end": chunk.source_end,
                        "visibility": chunk.visibility,
                        "tenant_id": chunk.tenant_id,
                        "classification": chunk.classification,
                        "allowed_roles": chunk.allowed_roles,
                    },
                )
            )
        for offset in range(0, len(points), 64):
            client.upsert(collection_name=target_collection, points=points[offset : offset + 64])
    except Exception as exc:  # pragma: no cover - depends on Qdrant service
        if owned_client:
            client.close()
        return {
            "backend": "qdrant", "mode": "dense_sparse", "status": "index_error",
            "documents": len(knowledge_chunks), "error": str(exc),
        }

    manifest = _build_manifest(
        knowledge_chunks,
        collection_name=target_collection,
        index_mode="dense_sparse",
        dense_dimension=dense_dimension,
        sparse_vocab_hash=sparse_hash,
    )
    _write_manifest(manifest, target_collection)
    model_target = Path(sparse_model_path) if sparse_model_path else _sparse_model_path(target_collection)
    sparse_features.save_sparse_model(vocab, idf, model_target)
    if owned_client:
        client.close()

    return {
        "backend": "qdrant",
        "mode": "dense_sparse",
        "status": "ok",
        "collection": target_collection,
        "documents": len(knowledge_chunks),
        "dense_dimension": dense_dimension,
        "sparse_vocab_size": len(vocab),
        "sparse_model_path": str(model_target),
        "index_version": rag_settings.index_version,
        "fingerprint_digest": manifest["index_fingerprint"]["digest"],
        "manifest_path": str(_artifact_path(_manifest_path(), target_collection)),
    }
